"""Explicit real P2 lifespan smoke: local load/text-cache/close, no HTTP or images.

Set HF_HOME, HF_HUB_CACHE, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 and
unset PYTORCH_ENABLE_MPS_FALLBACK in the shell before running with --load.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys

from eden_vision.main import create_app
from eden_vision.vision.runtime.lifecycle import RankingRuntimeSettings, build_clip_ranking_runtime


def fingerprint(root):
    return {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


async def smoke():
    settings = RankingRuntimeSettings()
    assert os.environ.get('HF_HOME') == str(settings.hf_home)
    assert os.environ.get('HF_HUB_CACHE') == str(settings.hf_hub_cache)
    assert os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1'
    snapshot = settings.hf_hub_cache / ('models--openai--clip-vit-base-patch32/snapshots/' + settings.revision)
    assert snapshot.is_dir(), 'existing local snapshot required'
    before = fingerprint(settings.hf_home)
    counts = dict(factory=0, load=0, close=0, text_precompute=0, image_inference=0, network=0)
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            counts['network'] += 1
            raise RuntimeError('network forbidden in lifecycle smoke')
    sys.addaudithook(audit)
    def observe(frame, event, arg):
        if event == 'call':
            name = frame.f_code.co_name
            if name == 'get_text_features': counts['text_precompute'] += 1
            if name == 'get_image_features':
                counts['image_inference'] += 1
                raise AssertionError('image inference forbidden')
            if name in ('rank', 'predict') and frame.f_code.co_filename.endswith('/runtime/clip.py'):
                raise AssertionError('ranking forbidden')
    def factory(taxonomy, config):
        counts['factory'] += 1
        runtime = build_clip_ranking_runtime(taxonomy, config)
        load, close = runtime.load, runtime.close
        def counted_load():
            counts['load'] += 1
            load()
        def counted_close():
            counts['close'] += 1
            close()
        runtime.load, runtime.close = counted_load, counted_close
        return runtime
    app = create_app(ranking_runtime_factory=factory, ranking_runtime_settings=settings)
    sys.setprofile(observe)
    try:
        async with app.router.lifespan_context(app):
            runtime = app.state.ranking_runtime
            assert app.state.ranking_runtime_ready and runtime.metadata['ready']
            assert runtime.metadata['taxonomy_version'] == 'eden-v2'
            assert runtime.metadata['candidate_count'] == 78
            assert str(next(runtime._model.parameters()).device) in ('mps', 'mps:0')
            assert list(runtime._text_features.shape) == [78, 512]
            print('READY', json.dumps(runtime.metadata), flush=True)
            print('TEXT_CACHE', list(runtime._text_features.shape), str(runtime._text_features.dtype), str(runtime._text_features.device), flush=True)
            await asyncio.sleep(0)
            assert app.state.ranking_runtime is runtime and counts['load'] == 1
        assert not app.state.ranking_runtime_ready and app.state.ranking_runtime is None
        assert runtime._model is runtime._processor is runtime._text_features is runtime.cache_identity is None
        assert not runtime.metadata['ready'] and runtime.metadata['load_state'] == 'closed'
        assert counts == dict(factory=1, load=1, close=1, text_precompute=1, image_inference=0, network=0)
        print('CLOSED', json.dumps(runtime.metadata), flush=True)
        print('COUNTS', json.dumps(counts), flush=True)
    finally:
        sys.setprofile(None)
        assert before == fingerprint(settings.hf_home), 'HF cache changed'
        print('HF_CACHE_UNCHANGED', len(before), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load', action='store_true')
    if not parser.parse_args().load:
        parser.error('--load required; no weights loaded')
    asyncio.run(smoke())


if __name__ == '__main__':
    main()
