"""P4 readiness and composition: no model weights or image inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from eden_vision.main import create_app
from eden_vision.production import create_production_app
from eden_vision.schemas.readiness import ReadinessResponse
from eden_vision.vision.runtime.lifecycle import RankingRuntimeSettings


@pytest.fixture
def settings(monkeypatch,tmp_path):
    for key in ('HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTORCH_ENABLE_MPS_FALLBACK'):
        monkeypatch.delenv(key,raising=False)
    return RankingRuntimeSettings(hf_home=tmp_path/'hf')


class FakeRuntime:
    def __init__(self,settings):
        self.loads=self.closes=0
        self.metadata={'ready':False,'model_id':settings.model_id,'model_revision':settings.revision,
                       'taxonomy_version':'eden-v2','candidate_count':78,'device':'mps',
                       'hf_home':'/private/secret/cache','environment':{'SECRET':'hidden'}}
        self.rank=MagicMock(side_effect=AssertionError('no ranking in readiness tests'))
    def load(self):self.loads+=1;self.metadata['ready']=True
    def close(self):self.closes+=1;self.metadata['ready']=False


def test_legacy_liveness_and_readiness():
    app=create_app()
    assert not app.state.app_ready
    with TestClient(app) as client:
        assert client.get('/health').json()=={'status':'ok','service':'eden-vision-ai'}
        response=client.get('/ready')
        assert response.status_code==200
        assert response.json()=={'status':'ready','mode':'legacy_classification',
                                'ranking_runtime_enabled':False,'ranking_runtime_ready':False}
        assert 'x-request-id' not in response.headers
    assert not app.state.app_ready


def test_production_safe_metadata_and_shutdown(settings):
    runtime=FakeRuntime(settings);factory=MagicMock(return_value=runtime)
    app=create_production_app(ranking_runtime_factory=factory,ranking_runtime_settings=settings)
    with TestClient(app) as client:
        for _ in range(2):
            response=client.get('/ready');assert response.status_code==200
            assert response.json()=={'status':'ready','mode':'clip_ranking',
                'ranking_runtime_enabled':True,'ranking_runtime_ready':True,
                'model_id':settings.model_id,'model_revision':settings.revision,'taxonomy_version':'eden-v2','device':'mps'}
            assert all(s not in response.text for s in ('secret','hf_home','environment','hidden'))
        assert client.get('/health').json()=={'status':'ok','service':'eden-vision-ai'}
        assert app.state.ranking_runtime is runtime
    assert runtime.loads==runtime.closes==1
    runtime.rank.assert_not_called();factory.assert_called_once()
    assert not app.state.app_ready and not app.state.ranking_runtime_ready


@pytest.mark.parametrize('reason',['flag','metadata','missing','failure','app','unsafe','exception'])
def test_not_ready_returns_sanitized_503(settings,reason):
    runtime=FakeRuntime(settings)
    app=create_production_app(ranking_runtime_factory=lambda *args:runtime,ranking_runtime_settings=settings)
    with TestClient(app) as client:
        if reason=='flag':app.state.ranking_runtime_ready=False
        elif reason=='metadata':runtime.metadata['ready']=False
        elif reason=='missing':app.state.ranking_runtime=None
        elif reason=='failure':app.state.ranking_runtime_startup_failed=True
        elif reason=='app':app.state.app_ready=False
        elif reason=='unsafe':runtime.metadata['model_id']='/Users/private/cache'
        else:
            class BrokenMetadata:
                @property
                def metadata(self):raise OSError('/private/secret/cache')
            app.state.ranking_runtime=BrokenMetadata()
        response=client.get('/ready')
        assert response.status_code==503
        assert response.json()=={'status':'not_ready','mode':'clip_ranking',
                                'ranking_runtime_enabled':True,'ranking_runtime_ready':False}
        assert 'private' not in response.text and 'cache' not in response.text
        assert client.get('/health').status_code==200
    runtime.rank.assert_not_called()


def test_no_lifespan_is_not_ready():
    # TestClient without its context manager does not start lifespan.
    client=TestClient(create_app())
    assert client.get('/ready').status_code==503
    assert client.get('/health').status_code==200
    client.close()


@pytest.mark.parametrize('changes',[{'extra':'secret'},{'mode':'invalid'},
    {'ranking_runtime_enabled':True},{'model_id':'/Users/private'}, {'ranking_runtime_ready':True}])
def test_readiness_schema_invariants(changes):
    with pytest.raises(ValidationError):
        ReadinessResponse.model_validate(dict(status='ready',mode='legacy_classification',
            ranking_runtime_enabled=False,ranking_runtime_ready=False)|changes)


def test_production_import_is_lightweight():
    code='''
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','transformers','huggingface_hub'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
from eden_vision.main import app as legacy
from eden_vision.production import app as production
assert production is not legacy
assert production.state.ranking_runtime_enabled and production.state.ranking_runtime is None
assert not production.state.ranking_runtime_ready
assert not legacy.state.ranking_runtime_enabled
assert 'eden_vision.vision.runtime.clip' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_production_cannot_disable_ranking():
    with pytest.raises(ValueError):create_production_app(ranking_runtime_factory=None)


def test_launcher_location_env_and_one_worker(tmp_path):
    # Fake project Python prints the launch command/environment, never runs uvicorn.
    repo=tmp_path/'project';(repo/'scripts').mkdir(parents=True);(repo/'.venv/bin').mkdir(parents=True)
    launcher=repo/'scripts/run_clip_api.sh'
    launcher.write_text((Path(__file__).resolve().parents[1]/'scripts/run_clip_api.sh').read_text())
    python=repo/'.venv/bin/python'
    python.write_text('#!/bin/sh\npwd\nprintf "%s\\n" "$HF_HOME" "$HF_HUB_CACHE" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$@"\n')
    python.chmod(0o755)
    env={k:v for k,v in os.environ.items() if k not in ('HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTORCH_ENABLE_MPS_FALLBACK')}
    env['HOME']=str(tmp_path/'user')
    result=subprocess.run(['bash',str(launcher)],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.splitlines()[0]==str(repo)
    assert str(tmp_path/'user/Library/Caches/eden-vision-ai/huggingface') in result.stdout
    assert 'eden_vision.production:app' in result.stdout and 'workers=1' in result.stdout
    for key,value in [('PYTORCH_ENABLE_MPS_FALLBACK','1'),('HF_HUB_OFFLINE','0'),('TRANSFORMERS_OFFLINE','0'),('HF_HOME','')]:
        bad=subprocess.run(['bash',str(launcher)],env=env|{key:value},capture_output=True,text=True)
        assert bad.returncode!=0 and 'Error:' in bad.stderr
