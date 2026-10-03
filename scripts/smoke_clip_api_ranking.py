"""Opt-in one-request MPS API smoke with an in-memory deterministic PNG.

Set HF_HOME/HF_HUB_CACHE and both offline flags in the shell; unset CPU fallback.
Run with --run-once. No datasets, accuracy assertions or performance benchmarks.
"""
import argparse
import asyncio
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import threading

import httpx
from PIL import Image, ImageDraw

from eden_vision.main import create_app
from eden_vision.schemas.classification import RankingResponse
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
    counts=dict(load=0,close=0,text_precompute=0,image_preprocess=0,image_inference=0,rank=0,http=0,network=0)
    observations={};stage='startup';runtime=None
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo'):
            counts['network']+=1
            raise RuntimeError('external network forbidden')
    sys.addaudithook(audit)
    def observe(frame,event,arg):
        name=frame.f_code.co_name
        if event=='call' and name=='get_text_features':counts['text_precompute']+=1
        if event=='call' and name=='get_image_features':
            counts['image_inference']+=1
            pixel_values=frame.f_locals['pixel_values']
            observations['input_device']=str(pixel_values.device)
            assert observations['input_device']=='mps:0'
        if event=='return' and name=='get_image_features' and arg is not None:
            features=arg.pooler_output
            import torch
            observations['image_feature_device']=str(features.device)
            observations['image_feature_finite']=bool(torch.isfinite(features).all().item())
            assert observations['image_feature_device']=='mps:0' and observations['image_feature_finite']
    def factory(taxonomy,config):
        nonlocal runtime
        runtime=build_clip_ranking_runtime(taxonomy,config)
        original_load,original_close,original_rank,original_logits=runtime.load,runtime.close,runtime.rank,runtime._matching_logits
        def load():
            counts['load']+=1;original_load()
            processor=runtime._processor
            class ObservedProcessor:
                def __call__(self,**kwargs):
                    if 'images' in kwargs:counts['image_preprocess']+=1
                    return processor(**kwargs)
            runtime._processor=ObservedProcessor()
        def close():counts['close']+=1;original_close()
        def rank(*args,**kwargs):counts['rank']+=1;return original_rank(*args,**kwargs)
        def logits(features):
            import torch
            observations['normalized_feature_finite']=bool(torch.isfinite(features).all().item())
            observations['normalized_feature_max_norm_error']=float((features.norm(p=2,dim=-1)-1).abs().max().item())
            assert observations['normalized_feature_finite'] and observations['normalized_feature_max_norm_error']<=1e-4
            scores=original_logits(features)
            observations['logits_shape']=list(scores.shape)
            observations['logits_finite']=bool(torch.isfinite(scores).all().item())
            assert observations['logits_shape']==[1,78] and observations['logits_finite']
            return scores
        runtime.load,runtime.close,runtime.rank,runtime._matching_logits=load,close,rank,logits
        return runtime
    app=create_app(ranking_runtime_factory=factory,ranking_runtime_settings=settings)
    # No randomness, external image, or disk output.
    with Image.new('RGB',(224,224),(128,128,128)) as image:
        draw=ImageDraw.Draw(image)
        draw.rectangle((24,24,103,199),fill=(200,40,40))
        draw.ellipse((120,56,207,143),fill=(30,70,210))
        stream=BytesIO();image.save(stream,format='PNG');png=stream.getvalue()
    sys.setprofile(observe);threading.setprofile(observe)
    try:
        async with app.router.lifespan_context(app):
            assert app.state.ranking_runtime_ready and runtime.metadata['ready']
            print('READY',json.dumps(runtime.metadata),flush=True)
            stage='HTTP image ranking'
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://smoke.local') as client:
                counts['http']+=1
                response=await client.post('/v1/vision/classify',files={'image':('synthetic.png',png,'image/png')})
            print('HTTP_STATUS',response.status_code,flush=True)
            assert response.status_code==200,response.text
            result=RankingResponse.model_validate(response.json())
            assert result.schema_version=='1.1' and result.status=='ranked'
            assert result.predictions==() and result.unknown_decision=='not_applied'
            assert len(result.ranking)==5 and [r.rank for r in result.ranking]==[1,2,3,4,5]
            scores=[r.raw_score for r in result.ranking]
            assert all(math.isfinite(s) for s in scores) and scores==sorted(scores,reverse=True)
            assert result.meta.model_id==settings.model_id and result.meta.model_revision==settings.revision
            assert result.meta.taxonomy_version=='eden-v2' and result.meta.policy_version=='not_applied'
            assert result.meta.calibration_version is None
            assert response.headers['X-Request-ID']==result.request_id
            print('RESPONSE',result.model_dump_json(),flush=True)
            print('TENSORS',json.dumps(observations),flush=True)
            stage='shutdown'
        assert not app.state.ranking_runtime_ready and app.state.ranking_runtime is None
        assert runtime._model is runtime._processor is runtime._text_features is runtime.cache_identity is None
        assert runtime.metadata['load_state']=='closed' and not runtime.metadata['ready']
        assert counts==dict(load=1,close=1,text_precompute=1,image_preprocess=1,image_inference=1,rank=1,http=1,network=0)
        print('COUNTS',json.dumps(counts),flush=True)
        print('CLOSED',json.dumps(runtime.metadata),flush=True)
    except Exception as exc:
        # Stage/type/tensor state are sufficient here; raw exception paths are not printed.
        print('FAILURE',json.dumps({'stage':stage,'exception_type':type(exc).__name__,
              'message':'one-request smoke failed; no retry performed','tensor_state':observations,
              'runtime_state':runtime.metadata if runtime else None,'counts':counts}),flush=True)
        raise SystemExit(1) from None
    finally:
        sys.setprofile(None);threading.setprofile(None)
        assert before==fingerprint(settings.hf_home),'HF cache changed'
        print('HF_CACHE_UNCHANGED',len(before),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-once',action='store_true')
    if not parser.parse_args().run_once:parser.error('--run-once required')
    asyncio.run(smoke())


if __name__=='__main__':main()
