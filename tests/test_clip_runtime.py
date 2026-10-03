"""P1 tests use only fake ML boundaries; never load model weights."""
from contextlib import nullcontext
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eden_vision.classification.taxonomy import parse_taxonomy
from eden_vision.vision.runtime import clip
from eden_vision.vision.types import ImageInput, UnknownDecision
from experiments.model_runtime import ClipEvaluationRuntime, MODELS, PROMPT_TEMPLATE, cache_identity


@pytest.fixture
def taxonomy():
    return parse_taxonomy((Path(__file__).resolve().parents[1] / 'src/eden_vision/resources/taxonomy_v2.json').read_text())


@pytest.fixture
def fake(monkeypatch, taxonomy):
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK', raising=False)
    monkeypatch.setattr(clip.platform, 'machine', lambda: 'arm64')
    torch = SimpleNamespace(__version__='fake-torch', backends=SimpleNamespace(mps=SimpleNamespace(
        is_built=lambda: True, is_available=lambda: True)), mps=MagicMock(),
        inference_mode=MagicMock(side_effect=nullcontext))
    text, image = MagicMock(), MagicMock()
    text.__truediv__.return_value.shape = (78, 512)
    scores = [float(i - 40) for i in range(78)]
    image.__truediv__.return_value.__matmul__.return_value.__mul__.return_value.__getitem__.return_value.float.return_value.cpu.return_value.tolist.return_value = scores
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([SimpleNamespace(device='mps:0')])
    model.get_text_features.return_value = SimpleNamespace(pooler_output=text)
    model.get_image_features.return_value = SimpleNamespace(pooler_output=image)
    token, pixels = MagicMock(), MagicMock()
    processor = MagicMock(side_effect=lambda **kw: {'input_ids': token} if 'text' in kw else {'pixel_values': pixels})
    progress = SimpleNamespace(enabled=True)
    progress.is_progress_bar_enabled = MagicMock(side_effect=lambda: progress.enabled)
    progress.disable_progress_bar = MagicMock(side_effect=lambda: setattr(progress, 'enabled', False))
    progress.enable_progress_bar = MagicMock(side_effect=lambda: setattr(progress, 'enabled', True))
    transformers = SimpleNamespace(__version__='fake-transformers', AutoProcessor=MagicMock(), CLIPModel=MagicMock(),
                                   utils=SimpleNamespace(logging=progress))
    transformers.AutoProcessor.from_pretrained.return_value = processor
    transformers.CLIPModel.from_pretrained.return_value = model
    monkeypatch.setattr(clip, '_dependencies', lambda: (torch, transformers))
    return SimpleNamespace(torch=torch, transformers=transformers, text=text, image=image,
                           model=model, processor=processor, token=token, pixels=pixels, scores=scores,
                           progress=progress)


def rgb():
    return ImageInput(2, 1, bytes([10, 20, 30, 40, 50, 60]))


def test_pinned_loading_cache_and_repeated_rank(fake, taxonomy):
    runtime = clip.ClipRankingRuntime(taxonomy)
    assert runtime.metadata['ready'] is False
    with pytest.raises(clip.RuntimeStateError):
        runtime.rank(rgb())
    runtime.load()
    for loader in [fake.transformers.AutoProcessor.from_pretrained, fake.transformers.CLIPModel.from_pretrained]:
        loader.assert_called_once_with(clip.SPEC.model_id, revision=clip.SPEC.revision, local_files_only=True)
    assert os.environ['HF_HUB_OFFLINE'] == '1'
    fake.model.to.assert_called_once_with('mps')
    fake.model.eval.assert_called_once_with()
    fake.processor.assert_called_once_with(text=[f'a photo of a {c.display_name.lower()}' for c in taxonomy.concepts], padding=True, return_tensors='pt')
    fake.token.to.assert_called_once_with('mps')
    fake.text.norm.assert_called_once_with(p=2, dim=-1, keepdim=True)
    fake.text.__truediv__.assert_called_once_with(fake.text.norm.return_value)
    result = runtime.rank(rgb(), top_k=78)
    assert runtime.predict(rgb(), top_k=78) == result
    assert min(r.raw_score for r in result.ranking) < 0 < 1 < max(r.raw_score for r in result.ranking)
    assert len(runtime.rank(rgb()).ranking) == clip.DEFAULT_TOP_K
    assert fake.model.get_text_features.call_count == 1
    assert fake.model.get_image_features.call_count == 3
    assert fake.torch.inference_mode.call_count == 4
    fake.image.norm.assert_called_with(p=2, dim=-1, keepdim=True)
    fake.image.__truediv__.assert_called_with(fake.image.norm.return_value)
    fake.model.logit_scale.exp.assert_called_with()
    fake.pixels.to.assert_called_with('mps')
    for call in fake.processor.call_args_list[1:]:
        assert set(call.kwargs) == {'images', 'return_tensors'}
        assert call.kwargs['images'].size == (2, 1)
        assert call.kwargs['return_tensors'] == 'pt'
    assert result.unknown_decision == UnknownDecision.NOT_APPLIED and result.predictions == ()
    assert result.policy_version == 'not_applied' and result.calibration_version is None
    byid = {c.concept_id: c for c in taxonomy.concepts}
    for item in result.ranking:
        assert item.display_name == byid[item.concept_id].display_name
        assert item.broad_category == byid[item.concept_id].broad_category_id
        assert item.hierarchy[-1] == item.concept_id
        assert item.score_kind == 'raw_image_text_logit'
    assert runtime.metadata['candidate_count'] == 78 and runtime.metadata['ready'] is True
    with pytest.raises(clip.RuntimeStateError):
        runtime.load()
    identity = runtime.cache_identity
    runtime.close()
    assert runtime.cache_identity is None and runtime._model is None and runtime._text_features is None
    with pytest.raises(clip.RuntimeStateError):
        runtime.predict(rgb())
    runtime.close()
    fake.torch.mps.empty_cache.assert_called_once()
    runtime.load()
    assert runtime.cache_identity == identity and fake.model.get_text_features.call_count == 2


@pytest.mark.parametrize('case', ['arch', 'built', 'available', 'fallback', 'wrong_device'])
def test_mps_requirements(fake, monkeypatch, taxonomy, case):
    if case == 'arch': monkeypatch.setattr(clip.platform, 'machine', lambda: 'x86_64')
    elif case == 'fallback': monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    elif case == 'wrong_device': fake.model.parameters.side_effect = lambda: iter([SimpleNamespace(device='cpu')])
    else: setattr(fake.torch.backends.mps, 'is_'+case, lambda: False)
    runtime = clip.ClipRankingRuntime(taxonomy)
    with pytest.raises(clip.MPSRequirementError): runtime.load()
    assert runtime.metadata['load_state'] == 'failed' and not runtime.metadata['ready']
    assert runtime._model is runtime._torch is runtime.cache_identity is None
    if case != 'wrong_device': fake.transformers.CLIPModel.from_pretrained.assert_not_called()


@pytest.mark.parametrize('initially_enabled', [True, False])
def test_load_restores_progress_state(fake, taxonomy, initially_enabled):
    fake.progress.enabled = initially_enabled
    stages = []

    def during_load(stage, result):
        assert fake.progress.enabled is False
        stages.append(stage)
        return result

    fake.transformers.AutoProcessor.from_pretrained.side_effect = lambda *a, **kw: during_load('processor', fake.processor)
    fake.transformers.CLIPModel.from_pretrained.side_effect = lambda *a, **kw: during_load('model', fake.model)
    fake.model.get_text_features.side_effect = lambda **kw: during_load('text', SimpleNamespace(pooler_output=fake.text))
    runtime = clip.ClipRankingRuntime(taxonomy)
    runtime.load()
    assert stages == ['processor', 'model', 'text']
    fake.progress.is_progress_bar_enabled.assert_called_once_with()
    fake.progress.disable_progress_bar.assert_called_once_with()
    assert fake.progress.enable_progress_bar.call_count == int(initially_enabled)
    assert fake.progress.enabled is initially_enabled
    assert runtime.metadata['ready'] is True
    runtime.close()


@pytest.mark.parametrize('initially_enabled', [True, False])
@pytest.mark.parametrize('stage', ['processor', 'model', 'text', 'cleanup'])
def test_load_failure_preserves_original_error_and_cleans(fake, taxonomy, stage, initially_enabled):
    fake.progress.enabled = initially_enabled
    error = OSError('local cached checkpoint absent or unreadable')
    if stage == 'processor': fake.transformers.AutoProcessor.from_pretrained.side_effect = error
    elif stage == 'model': fake.transformers.CLIPModel.from_pretrained.side_effect = error
    else: fake.model.get_text_features.side_effect = error
    if stage == 'cleanup': fake.torch.mps.synchronize.side_effect = RuntimeError('cleanup failed')
    runtime = clip.ClipRankingRuntime(taxonomy)
    with pytest.raises(OSError) as caught: runtime.load()
    assert caught.value is error
    assert runtime._model is runtime._processor is runtime._text_features is runtime.cache_identity is None
    assert runtime.metadata['load_state'] == 'failed'
    assert runtime.metadata['ready'] is False
    fake.progress.disable_progress_bar.assert_called_once_with()
    assert fake.progress.enable_progress_bar.call_count == int(initially_enabled)
    assert fake.progress.enabled is initially_enabled


@pytest.mark.parametrize('api', ['is_progress_bar_enabled', 'disable_progress_bar', 'enable_progress_bar'])
def test_progress_api_errors_propagate_and_clean_partial_runtime(fake, taxonomy, api):
    error = RuntimeError('progress API failed')
    getattr(fake.progress, api).side_effect = error
    runtime = clip.ClipRankingRuntime(taxonomy)
    with pytest.raises(RuntimeError) as caught:
        runtime.load()
    assert caught.value is error
    assert runtime.metadata['ready'] is False and runtime.metadata['load_state'] == 'failed'
    assert runtime._model is runtime._processor is runtime._text_features is runtime._torch is runtime.cache_identity is None


@pytest.mark.parametrize('top_k', [0, -1, 79, True, 1.5, '5', None])
def test_invalid_top_k(fake, taxonomy, top_k):
    runtime = clip.ClipRankingRuntime(taxonomy); runtime.load()
    with pytest.raises(ValueError): runtime.rank(rgb(), top_k=top_k)
    fake.model.get_image_features.assert_not_called()


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_scores_rejected_even_outside_topk(fake, taxonomy, score):
    fake.scores[0] = score
    runtime = clip.ClipRankingRuntime(taxonomy); runtime.load()
    with pytest.raises(ValueError, match='finite'): runtime.rank(rgb())


def test_ties_and_numerical_formula_parity(fake, taxonomy):
    fake.scores[:] = [2.] * 78
    runtime = clip.ClipRankingRuntime(taxonomy); runtime.load()
    expected = sorted(c.concept_id for c in taxonomy.concepts)[:5]
    assert [r.concept_id for r in runtime.rank(rgb()).ranking] == expected
    features = MagicMock(); features.__matmul__.return_value = -0.25
    runtime._model.logit_scale.exp.return_value = 12.
    reference = ClipEvaluationRuntime()
    reference.model, reference._features = runtime._model, runtime._text_features
    assert runtime._matching_logits(features) == reference._matching_logits(features) == -3.
    features.__matmul__.assert_called_with(runtime._text_features.T)
    assert clip.SPEC.model_id == MODELS['clip'].model_id
    assert clip.SPEC.revision == MODELS['clip'].revision
    assert clip.SPEC.prompt_template == PROMPT_TEMPLATE
    ref = cache_identity(MODELS['clip'], taxonomy)
    assert runtime.cache_identity.candidate_ids == ref.candidate_ids
    assert runtime.cache_identity.prompts == ref.prompts
    # Actual experimental text normalization path with the same fake boundary.
    reference._torch, reference.processor = fake.torch, fake.processor
    reference._encode_taxonomy(ref.prompts)
    fake.text.norm.assert_called_with(p=2, dim=-1, keepdim=True)
    fake.text.__truediv__.assert_called_with(fake.text.norm.return_value)
    source = Path(clip.__file__).read_text()
    assert 'softmax(' not in source and 'sigmoid(' not in source
    assert 'from experiments' not in source and 'import experiments' not in source


def test_taxonomy_errors_and_cache_identity(fake, taxonomy):
    with pytest.raises(ValueError): clip.ClipRankingRuntime(replace(taxonomy, concepts=()))
    with pytest.raises(ValueError): clip.ClipRankingRuntime(replace(taxonomy, concepts=taxonomy.concepts + (taxonomy.concepts[0],)))
    dup = replace(taxonomy.concepts[1], display_name=taxonomy.concepts[0].display_name.upper())
    with pytest.raises(ValueError, match='duplicate taxonomy prompts'):
        clip.ClipRankingRuntime(replace(taxonomy, concepts=(taxonomy.concepts[0], dup, *taxonomy.concepts[2:])))
    a = clip.ClipRankingRuntime(taxonomy)
    b = clip.ClipRankingRuntime(replace(taxonomy, concepts=tuple(reversed(taxonomy.concepts))))
    c = clip.ClipRankingRuntime(replace(taxonomy, taxonomy_version='new-version'))
    assert a._expected_identity != b._expected_identity != c._expected_identity
    a.load(); a.close()
    b.load()
    assert b.cache_identity != a.cache_identity and fake.model.get_text_features.call_count == 2


def test_import_isolation_and_dependency_errors(monkeypatch):
    script = '''
import sys
class RejectML:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise AssertionError('unexpected ML import')
sys.meta_path.insert(0, RejectML())
from eden_vision.vision.runtime.clip import ClipRankingRuntime
from eden_vision.main import create_app
from eden_vision.vision.adapters.dummy import DummyVisionModel
create_app()
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    def missing(name): raise ModuleNotFoundError('missing', name=name)
    monkeypatch.setattr(clip.importlib, 'import_module', missing)
    with pytest.raises(ModuleNotFoundError, match='model-runtime.txt'): clip._dependencies()
    error = ModuleNotFoundError('transitive bug', name='internal_dependency')
    def transitive(name): raise error
    monkeypatch.setattr(clip.importlib, 'import_module', transitive)
    with pytest.raises(ModuleNotFoundError) as caught: clip._dependencies()
    assert caught.value is error
