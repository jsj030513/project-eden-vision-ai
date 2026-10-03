"""Fake ranking API integration; real ML runs only in the opt-in smoke script."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import threading
import time
from unittest.mock import MagicMock
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.config import Settings
from eden_vision.main import create_app
from eden_vision.schemas.classification import RankingResponse
from eden_vision.services.image_input import ImageInputService
from eden_vision.vision.runtime.lifecycle import RankingRuntimeSettings
from eden_vision.vision.types import ModelInfo, RankingItem, RankingResult
from test_mpo_input import mpo_bytes


@pytest.fixture
def app_runtime(monkeypatch,tmp_path):
    for key in ('HF_HOME','HF_HUB_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTORCH_ENABLE_MPS_FALLBACK'):
        monkeypatch.delenv(key,raising=False)
    class FakeRuntime:
        def __init__(self,taxonomy):
            self.loads=self.closes=self.ranks=self.active=self.max_active=0
            self.images=[];self.error=None;self.delay=0
            self.metadata={'ready':False,'taxonomy_version':'eden-v2','candidate_count':78}
            resolver=HierarchyResolver(taxonomy)
            self.result=RankingResult(ranking=tuple(RankingItem(i,cid,resolver.concept(cid).display_name,
                resolver.broad_category(cid),resolver.ancestor_path(cid),score)
                for i,(cid,score) in enumerate(zip(('cat','animal','dog','tree','plant'),(27.1,25.2,24.8,1.2,-3.)),1)),
                model=ModelInfo('openai/clip-vit-base-patch32','3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'),taxonomy_version='eden-v2')
        def load(self):self.loads+=1;self.metadata['ready']=True
        def close(self):self.closes+=1;self.metadata['ready']=False
        def rank(self,image):
            with pytest.raises(RuntimeError):asyncio.get_running_loop()
            self.ranks+=1;self.images.append(image);self.active+=1
            self.max_active=max(self.active,self.max_active)
            try:
                time.sleep(self.delay)
                if self.error:raise self.error
                return self.result
            finally:self.active-=1
    made=[]
    def factory(taxonomy,settings):
        runtime=FakeRuntime(taxonomy);made.append(runtime);return runtime
    app=create_app(Settings(_env_file=None),ranking_runtime_factory=factory,
                   ranking_runtime_settings=RankingRuntimeSettings(hf_home=tmp_path/'hf'))
    with TestClient(app) as client:
        app.state.classification.classify=MagicMock(side_effect=AssertionError('legacy policy must not run'))
        yield app,client,made[0]
        app.state.classification.classify.assert_not_called()
    assert len(made)==1 and made[0].loads==made[0].closes==1


def encoded(fmt='PNG',mode='RGB',orientation=None):
    b=BytesIO();options={}
    if orientation:
        exif=Image.Exif();exif[274]=orientation;options['exif']=exif
    with Image.new(mode,(8,6),(255,0,0,0) if mode=='RGBA' else (20,40,60)) as image:
        image.save(b,format=fmt,**options)
    return b.getvalue()


def post(client,data,mime='image/png'):
    return client.post('/v1/vision/classify',files={'image':('synthetic.bin',data,mime)})


def assert_request_id(response):
    assert response.headers['X-Request-ID']==response.json()['request_id']
    assert UUID(response.json()['request_id']).version==4


def test_ranking_http_reuses_runtime_and_preserves_scores(app_runtime):
    app,client,runtime=app_runtime
    for _ in range(3):
        response=post(client,encoded());assert response.status_code==200
        assert_request_id(response)
        data=response.json()
        assert data['schema_version']=='1.1' and data['status']=='ranked'
        assert data['predictions']==[] and data['unknown_decision']=='not_applied'
        assert RankingResponse.model_validate(data).to_domain()==runtime.result
        assert [r['rank'] for r in data['ranking']]==[1,2,3,4,5]
        assert [r['raw_score'] for r in data['ranking']]==[27.1,25.2,24.8,1.2,-3.]
        assert all(not {'confidence','probability','unknown'} & r.keys() for r in data['ranking'])
        assert app.state.ranking_runtime is runtime
    assert runtime.loads==1 and runtime.ranks==3


@pytest.mark.parametrize('fmt,mime', [('JPEG','image/jpeg'),('PNG','image/png'),('WEBP','image/webp'),('MPO','image/jpeg'),('ALPHA','image/png'),('EXIF','image/jpeg'),('HEIF','image/heic')])
def test_shared_decoder_uploads(app_runtime,fmt,mime):
    _,client,runtime=app_runtime
    data=mpo_bytes() if fmt=='MPO' else encoded('PNG',mode='RGBA') if fmt=='ALPHA' else encoded('JPEG',orientation=6) if fmt=='EXIF' else encoded(fmt)
    expected=ImageInputService(max_upload_bytes=10*1024*1024,max_image_pixels=25_000_000).decode(data,mime)
    response=post(client,data,mime)
    assert response.status_code==200 and runtime.ranks==1
    assert runtime.images[0]==expected
    assert response.json()['image']=={'width':expected.width,'height':expected.height}
    if fmt=='ALPHA':assert expected.pixels==bytes([128,128,128])*48
    if fmt=='EXIF':assert (expected.width,expected.height)==(6,8)


@pytest.mark.parametrize('state', ['app_not_ready','runtime_not_ready','missing'])
def test_unavailable_never_falls_back(app_runtime,state):
    app,client,runtime=app_runtime
    if state=='app_not_ready':app.state.ranking_runtime_ready=False
    elif state=='runtime_not_ready':runtime.metadata['ready']=False
    else:app.state.ranking_runtime=None
    response=post(client,encoded())
    assert response.status_code==500 and response.json()['error']['code']=='INFERENCE_FAILED'
    assert runtime.ranks==0;assert_request_id(response)


def test_rank_exception_is_sanitized_and_logged(app_runtime,caplog):
    _,client,runtime=app_runtime
    runtime.error=OSError('/private/cache/secret model failure')
    response=post(client,encoded())
    assert response.status_code==500
    assert response.json()['error']=={'code':'INFERENCE_FAILED','message':'Classification could not be completed','retryable':False}
    assert 'private/cache' not in response.text
    assert 'private/cache/secret' in caplog.text
    assert_request_id(response)


@pytest.mark.parametrize('kind,status,code', [('missing',422,'INVALID_IMAGE'),('corrupt',422,'INVALID_IMAGE'),('mime',415,'UNSUPPORTED_IMAGE_TYPE'),('body',413,'IMAGE_TOO_LARGE')])
def test_upload_errors_before_ranking(app_runtime,kind,status,code):
    _,client,runtime=app_runtime
    if kind=='missing':response=client.post('/v1/vision/classify')
    elif kind=='corrupt':response=post(client,b'broken')
    elif kind=='mime':response=post(client,encoded(),'image/jpeg')
    else:response=client.post('/v1/vision/classify',content=b'x'*(10*1024*1024+65537))
    assert response.status_code==status and response.json()['error']['code']==code
    assert runtime.ranks==0;assert_request_id(response)


def test_same_app_serializes_decode_and_rank(app_runtime):
    app,client,runtime=app_runtime
    runtime.delay=.015
    data=encoded()
    with ThreadPoolExecutor(max_workers=3) as pool:
        responses=list(pool.map(lambda _:post(client,data),range(3)))
    assert all(r.status_code==200 for r in responses)
    assert runtime.ranks==3 and runtime.max_active==1 and runtime.loads==1


def test_default_remains_legacy():
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response=post(client,encoded())
    assert response.status_code==200
    data=response.json()
    assert data['schema_version']=='1.0' and data['predictions'][0]['label']=='plant'
    assert data['meta']['model_id']=='dummy-injected-candidates' and data['meta']['taxonomy_version']=='eden-v1'
