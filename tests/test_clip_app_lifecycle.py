"""No weights or image inference: P2 app ownership with fake runtimes."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from eden_vision.main import create_app
from eden_vision.vision.runtime import lifecycle as life


@pytest.fixture
def settings(monkeypatch, tmp_path):
    for key in ('HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTORCH_ENABLE_MPS_FALLBACK'):
        monkeypatch.delenv(key, raising=False)
    return life.RankingRuntimeSettings(hf_home=tmp_path/'hf')


class FakeRuntime:
    def __init__(self, error=None, ready=True, close_error=False):
        self.loads = self.closes = 0
        self.error, self.ready, self.close_error = error, ready, close_error
        self.metadata = {'ready': False, 'taxonomy_version': 'eden-v2', 'candidate_count': 78}
    def load(self):
        self.loads += 1
        if self.error: raise self.error
        self.metadata['ready'] = self.ready
    def close(self):
        self.closes += 1
        self.metadata['ready'] = False
        if self.close_error: raise RuntimeError('cleanup failure')


def test_opt_in_lifecycle_reuses_instance_and_keeps_dummy(settings):
    runtime = FakeRuntime()
    def build(taxonomy, config):
        assert os.environ['HF_HOME'] == str(settings.hf_home)
        assert os.environ['HF_HUB_CACHE'] == str(settings.hf_hub_cache)
        assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
        assert taxonomy.taxonomy_version == 'eden-v2' and len(taxonomy.concepts) == 78
        assert config == settings
        return runtime
    factory = MagicMock(side_effect=build)
    app = create_app(ranking_runtime_factory=factory, ranking_runtime_settings=settings)
    assert app.state.ranking_runtime_enabled and not app.state.ranking_runtime_ready
    with TestClient(app) as client:
        for _ in range(3):
            assert client.get('/health').status_code == 200
            assert app.state.ranking_runtime is runtime
        assert runtime.loads == 1 and runtime.closes == 0
        assert app.state.ranking_runtime_ready
        assert app.state.classification._model.info.model_id == 'dummy-injected-candidates'
        assert app.state.classification._policy.hierarchy.taxonomy_version == 'eden-v1'
    factory.assert_called_once()
    assert runtime.closes == 1 and not app.state.ranking_runtime_ready
    assert app.state.ranking_runtime is None
    assert not app.state.ranking_runtime_startup_failed


@pytest.mark.parametrize('close_error', [False, True])
def test_failed_load_preserves_original_and_cleans(settings, close_error):
    error = OSError('local cache unavailable')
    runtime = FakeRuntime(error=error, close_error=close_error)
    app = create_app(ranking_runtime_factory=lambda *args: runtime, ranking_runtime_settings=settings)
    with pytest.raises(OSError) as caught:
        with TestClient(app): pytest.fail('startup must fail')
    assert caught.value is error
    assert runtime.loads == runtime.closes == 1
    assert app.state.ranking_runtime_startup_failed
    assert not app.state.ranking_runtime_ready and app.state.ranking_runtime is None


def test_not_ready_is_startup_failure(settings):
    runtime = FakeRuntime(ready=False)
    app = create_app(ranking_runtime_factory=lambda *args: runtime, ranking_runtime_settings=settings)
    with pytest.raises(RuntimeError, match='did not become ready'):
        with TestClient(app): pass
    assert runtime.closes == 1 and app.state.ranking_runtime_startup_failed


@pytest.mark.parametrize('stage', ['taxonomy', 'factory'])
def test_preload_failures(settings, monkeypatch, stage):
    error = ValueError(stage+' failure')
    factory = MagicMock(side_effect=error)
    if stage == 'taxonomy': monkeypatch.setattr(life, 'load_ranking_taxonomy', MagicMock(side_effect=error))
    app = create_app(ranking_runtime_factory=factory, ranking_runtime_settings=settings)
    with pytest.raises(ValueError) as caught:
        with TestClient(app): pass
    assert caught.value is error
    assert app.state.ranking_runtime_startup_failed and app.state.ranking_runtime is None
    if stage == 'taxonomy': factory.assert_not_called()


@pytest.mark.parametrize('key,value', [
    ('HF_HOME','/conflict'),('HF_HUB_CACHE','/conflict'),('HF_HUB_OFFLINE','0'),
    ('TRANSFORMERS_OFFLINE','false'),('PYTORCH_ENABLE_MPS_FALLBACK','1'),
])
def test_env_conflicts_fail_before_factory(settings, monkeypatch, key, value):
    monkeypatch.setenv(key,value)
    before = dict(os.environ)
    factory = MagicMock()
    app = create_app(ranking_runtime_factory=factory, ranking_runtime_settings=settings)
    with pytest.raises(life.RuntimeEnvironmentError):
        with TestClient(app): pass
    assert dict(os.environ) == before
    factory.assert_not_called()
    assert app.state.ranking_runtime_startup_failed


@pytest.mark.parametrize('field,value', [
    ('model_id','other'),('revision','main'),('device','cpu'),('offline',False),
    ('taxonomy_version','eden-v1'),('prompt_template','other'),('hf_home',''),('hf_hub_cache',' '),
])
def test_invalid_config(settings, field, value):
    with pytest.raises(ValidationError):
        life.RankingRuntimeSettings(**(settings.model_dump() | {field:value}))


def test_resolution_precedence_and_inconsistent_paths(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path/'user'))
    default = life.RankingRuntimeSettings()
    assert default.hf_home == tmp_path/'user/Library/Caches/eden-vision-ai/huggingface'
    monkeypatch.setenv('HF_HOME', str(tmp_path/'env'))
    assert life.RankingRuntimeSettings().hf_home == tmp_path/'env'
    explicit = life.RankingRuntimeSettings(hf_home=tmp_path/'explicit')
    with pytest.raises(life.RuntimeEnvironmentError): life.prepare_runtime_environment(explicit)
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path/'wrong'))
    with pytest.raises(ValidationError): life.RankingRuntimeSettings()


@pytest.mark.parametrize('mismatch', ['home','offline','transformers'])
def test_already_imported_constants_fail_without_repair(settings, monkeypatch, mismatch):
    constants = SimpleNamespace(HF_HOME=str(settings.hf_home), HF_HUB_CACHE=str(settings.hf_hub_cache), HF_HUB_OFFLINE=True)
    if mismatch == 'home': constants.HF_HOME='/old/cache'
    if mismatch == 'offline': constants.HF_HUB_OFFLINE=False
    monkeypatch.setitem(sys.modules, 'huggingface_hub.constants', constants)
    if mismatch == 'transformers': monkeypatch.setitem(sys.modules, 'transformers.utils.hub', SimpleNamespace(is_offline_mode=lambda:False))
    before=dict(os.environ)
    with pytest.raises(life.RuntimeEnvironmentError, match='already-imported'):
        life.prepare_runtime_environment(settings)
    assert dict(os.environ)==before


def test_matching_preimported_constants_allowed(settings, monkeypatch):
    constants=SimpleNamespace(HF_HOME=str(settings.hf_home),HF_HUB_CACHE=str(settings.hf_hub_cache),HF_HUB_OFFLINE=True)
    monkeypatch.setitem(sys.modules,'huggingface_hub.constants',constants)
    life.prepare_runtime_environment(settings)
    assert sys.modules['huggingface_hub.constants'] is constants


def test_default_isolation_even_with_runtime_env(settings, monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','0')
    monkeypatch.setattr(life,'prepare_runtime_environment',MagicMock(side_effect=AssertionError('not opt-in')))
    monkeypatch.setattr(life,'load_ranking_taxonomy',MagicMock(side_effect=AssertionError('not opt-in')))
    app=create_app()
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        assert not app.state.ranking_runtime_enabled and app.state.ranking_runtime is None
    with pytest.raises(ValueError,match='explicit'):
        create_app(ranking_runtime_settings=settings)


def test_separate_apps_have_separate_instances_and_locks(settings):
    runtimes=[]
    def factory(*args):
        runtime=FakeRuntime();runtimes.append(runtime);return runtime
    a=create_app(ranking_runtime_factory=factory,ranking_runtime_settings=settings)
    b=create_app(ranking_runtime_factory=factory,ranking_runtime_settings=settings)
    with TestClient(a), TestClient(b):
        assert a.state.ranking_runtime is not b.state.ranking_runtime
        assert a.state.classification_lock is not b.state.classification_lock
    assert [r.closes for r in runtimes]==[1,1]


def test_default_subprocess_does_not_import_ml():
    script='''
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','transformers','huggingface_hub'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
from eden_vision.main import create_app,app
from fastapi.testclient import TestClient
with TestClient(create_app()) as client:
 assert client.get('/health').status_code==200
assert 'eden_vision.vision.runtime.clip' not in sys.modules
assert app.state.ranking_runtime_enabled is False
'''
    result=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
