"""Opt-in production lifespan smoke: health/ready only, no images or ranking.

Set the validated HF cache and offline environment in the shell before --load.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys

import httpx
from eden_vision.production import create_production_app
from eden_vision.vision.runtime.lifecycle import RankingRuntimeSettings, build_clip_ranking_runtime


def fingerprint(root):
    return {str(p):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


async def smoke():
    settings=RankingRuntimeSettings()
    assert os.environ.get('HF_HOME')==str(settings.hf_home)
    assert os.environ.get('HF_HUB_CACHE')==str(settings.hf_hub_cache)
    assert os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1'
    snapshot=settings.hf_hub_cache/('models--openai--clip-vit-base-patch32/snapshots/'+settings.revision)
    assert snapshot.is_dir()
    before=fingerprint(settings.hf_home)
    counts=dict(load=0,close=0,text_precompute=0,image_inference=0,network=0)
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo'):
            counts['network']+=1;raise RuntimeError('network forbidden')
    sys.addaudithook(audit)
    def observe(frame,event,arg):
        if event=='call':
            name=frame.f_code.co_name
            if name=='get_text_features':counts['text_precompute']+=1
            if name=='get_image_features':
                counts['image_inference']+=1;raise AssertionError('image inference forbidden')
            if name in ('rank','predict') and frame.f_code.co_filename.endswith('/runtime/clip.py'):
                raise AssertionError('ranking forbidden')
    def factory(taxonomy,config):
        runtime=build_clip_ranking_runtime(taxonomy,config)
        load,close=runtime.load,runtime.close
        def counted_load():counts['load']+=1;load()
        def counted_close():counts['close']+=1;close()
        runtime.load,runtime.close=counted_load,counted_close
        return runtime
    app=create_production_app(ranking_runtime_factory=factory,ranking_runtime_settings=settings)
    sys.setprofile(observe)
    try:
        async with app.router.lifespan_context(app):
            runtime=app.state.ranking_runtime
            assert app.state.app_ready and app.state.ranking_runtime_ready
            assert runtime.metadata['candidate_count']==78
            assert str(next(runtime._model.parameters()).device)=='mps:0'
            assert list(runtime._text_features.shape)==[78,512]
            print('RUNTIME',json.dumps(runtime.metadata),flush=True)
            print('TEXT_CACHE',list(runtime._text_features.shape),str(runtime._text_features.dtype),str(runtime._text_features.device),flush=True)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://smoke.local') as client:
                health=await client.get('/health')
                ready=await client.get('/ready')
            assert health.status_code==200 and health.json()=={'status':'ok','service':'eden-vision-ai'}
            assert ready.status_code==200
            assert ready.json()==dict(status='ready',mode='clip_ranking',ranking_runtime_enabled=True,
                ranking_runtime_ready=True,model_id=settings.model_id,model_revision=settings.revision,
                taxonomy_version='eden-v2',device='mps')
            assert 'x-request-id' not in ready.headers
            print('HEALTH',health.status_code,health.text,flush=True)
            print('READY',ready.status_code,ready.text,flush=True)
        assert not app.state.app_ready and not app.state.ranking_runtime_ready
        assert runtime._model is runtime._processor is runtime._text_features is runtime.cache_identity is None
        assert runtime.metadata['load_state']=='closed'
        assert counts==dict(load=1,close=1,text_precompute=1,image_inference=0,network=0)
        print('COUNTS',json.dumps(counts),flush=True)
        print('CLOSED',json.dumps(runtime.metadata),flush=True)
    finally:
        sys.setprofile(None)
        assert before==fingerprint(settings.hf_home),'HF cache changed'
        print('HF_CACHE_UNCHANGED',len(before),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load',action='store_true')
    if not parser.parse_args().load:parser.error('--load required; no model loaded')
    asyncio.run(smoke())


if __name__=='__main__':main()
