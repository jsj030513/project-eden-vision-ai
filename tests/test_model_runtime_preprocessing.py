"""Processor contracts without AI dependencies; optional cached-tokenizer check."""

import ast
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from experiments.model_runtime import cache_identity, create_runtime, load_evaluation_taxonomy


@pytest.mark.parametrize("model,kwargs", [
    ("clip", {"padding": True}),
    ("siglip2", {"padding": "max_length", "truncation": True}),
])
def test_taxonomy_encoding_calls_model_processor_and_preserves_normalization(model, kwargs):
    runtime = create_runtime(model)
    token_ids = MagicMock()
    runtime.processor = MagicMock(return_value={"input_ids": token_ids})
    raw_features = MagicMock()
    features = raw_features.__truediv__.return_value
    features.shape = (78, 4)
    features.dtype = "float32"
    features.numel.return_value = 78 * 4
    features.element_size.return_value = 4
    runtime.model = MagicMock()
    runtime.model.get_text_features.return_value = SimpleNamespace(pooler_output=raw_features)
    runtime._torch = SimpleNamespace(mps=MagicMock(), inference_mode=nullcontext)
    taxonomy = load_evaluation_taxonomy("eden-v2")
    identity = cache_identity(runtime.spec, taxonomy)

    info = runtime.prepare_taxonomy(taxonomy)

    runtime.processor.assert_called_once_with(text=list(identity.prompts), return_tensors="pt", **kwargs)
    token_ids.to.assert_called_once_with("mps")
    runtime.model.get_text_features.assert_called_once_with(input_ids=token_ids.to.return_value)
    raw_features.norm.assert_called_once_with(p=2, dim=-1, keepdim=True)
    raw_features.__truediv__.assert_called_once_with(raw_features.norm.return_value)
    assert runtime._features is features
    assert runtime.identity == identity
    assert info["candidate_count"] == 78 and info["shape"] == [78, 4]
    assert runtime.prepare_taxonomy(taxonomy) is info
    assert runtime.processor.call_count == runtime.model.get_text_features.call_count == 1


@pytest.mark.parametrize("model,expected", [("clip", 1.0), ("siglip2", -1.0)])
def test_matching_formula_keeps_scale_and_siglip_bias(model, expected):
    runtime = create_runtime(model)
    runtime.model = SimpleNamespace(logit_scale=MagicMock(), logit_bias=-2.0)
    runtime.model.logit_scale.exp.return_value = 4.0
    runtime._features = SimpleNamespace(T=object())
    image_features = MagicMock()
    image_features.__matmul__.return_value = 0.25

    assert runtime._matching_logits(image_features) == expected
    image_features.__matmul__.assert_called_once_with(runtime._features.T)
    runtime.model.logit_scale.exp.assert_called_once_with()


def test_siglip_baseline_processor_uses_fixed_length_without_loading_models():
    # Extract only the processor call: importing/running this historical tool
    # would load AI dependencies and its main() can download/write reports.
    path = Path(__file__).resolve().parents[1] / "experiments/siglip2_baseline.py"
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "processor"]
    assert len(calls) == 1
    processor = MagicMock()
    image = object()
    prompts = ["a photo of a cat"]
    eval(compile(ast.Expression(calls[0]), str(path), "eval"),
         {"processor": processor, "prompts": prompts, "image": image})
    processor.assert_called_once_with(text=prompts, images=image, return_tensors="pt",
                                     padding="max_length", truncation=True)


def test_cached_processors_actual_shapes_and_clip_token_regression():
    """Opt in with EDEN_PROCESSOR_TEST_PYTHON pointing to the model-eval Python.

    Uses pinned cached processors only, no model weights or dataset/report I/O.
    The lightweight service venv need not install torch or Transformers.
    """
    python = os.environ.get("EDEN_PROCESSOR_TEST_PYTHON")
    if not python:
        pytest.skip("Set EDEN_PROCESSOR_TEST_PYTHON to verify locally cached AutoProcessors")
    script = '''
from pathlib import Path
import os
os.environ.setdefault('HF_HOME', str(Path.home() / 'Library/Caches/eden-vision-ai/huggingface'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoProcessor
from experiments.model_runtime import cache_identity, create_runtime, load_evaluation_taxonomy

taxonomy = load_evaluation_taxonomy('eden-v2')
for name in ('clip', 'siglip2'):
    runtime = create_runtime(name)
    processor = AutoProcessor.from_pretrained(runtime.spec.model_id,
        revision=runtime.spec.revision, local_files_only=True)
    prompts = list(cache_identity(runtime.spec, taxonomy).prompts)
    current = processor(text=prompts, return_tensors='pt', **runtime._text_processor_kwargs())
    old = processor(text=prompts, return_tensors='pt', padding=True)
    if name == 'siglip2':
        assert tuple(current['input_ids'].shape) == (78, 64), current['input_ids'].shape
        assert tuple(old['input_ids'].shape) == (78, 9), old['input_ids'].shape
    else:
        assert current.keys() == old.keys()
        for key in old:
            assert current[key].shape == old[key].shape
            assert current[key].equal(old[key]), key
    print(name, 'old:', tuple(old['input_ids'].shape), 'current:', tuple(current['input_ids'].shape))
'''
    result = subprocess.run([python, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end="")
