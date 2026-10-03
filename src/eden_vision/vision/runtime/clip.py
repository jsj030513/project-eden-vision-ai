"""Pinned, offline MPS CLIP ranking. Caller owns lifecycle and serializes use.

Independent of experiments and ConfidencePolicy. Taxonomy is immutable for an
instance; changing it requires a new instance/cache. Ties use concept_id ascending.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
import platform
from pathlib import Path

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.classification.taxonomy import Taxonomy, parse_taxonomy
from eden_vision.vision.types import ImageInput, ModelInfo, RankingItem, RankingResult


@dataclass(frozen=True, slots=True)
class ClipSpec:
    model_id: str = 'openai/clip-vit-base-patch32'
    revision: str = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
    prompt_template: str = 'a photo of a {display_name}'
    score_kind: str = 'raw_image_text_logit'
    device: str = 'mps'


SPEC = ClipSpec()
DEFAULT_TOP_K = 5


class RuntimeStateError(RuntimeError):
    """Operation incompatible with runtime lifecycle."""


class MPSRequirementError(RuntimeError):
    """Native MPS or no-fallback requirement not satisfied."""


@dataclass(frozen=True, slots=True)
class TextCacheIdentity:
    model_id: str
    revision: str
    taxonomy_version: str
    taxonomy_sha256: str
    candidate_ids: tuple[str, ...]
    prompts: tuple[str, ...]
    prompt_template: str


def _dependencies():
    # Only missing top-level dependencies get an actionable message. Transitive
    # import errors retain their original exception/traceback.
    modules = []
    for name in ('torch', 'transformers'):
        try:
            modules.append(importlib.import_module(name))
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            raise ModuleNotFoundError(
                f'{name} is required; install requirements/model-runtime.txt', name=name
            ) from exc
    return tuple(modules)


class ClipRankingRuntime:
    """load -> rank/predict -> close; no API wiring or accepted predictions.

    This ranking interface deliberately does not implement VisionModel.predict,
    whose return type feeds the legacy cosine decision policy.
    """

    def __init__(self, taxonomy: Taxonomy):
        if not isinstance(taxonomy, Taxonomy):
            raise TypeError('taxonomy must be Taxonomy')
        # Validate/copy even hand-built domain values and freeze candidate order.
        encoded = json.dumps(asdict(taxonomy), sort_keys=True, separators=(',', ':'))
        self._taxonomy = parse_taxonomy(encoded)
        self._hierarchy = HierarchyResolver(self._taxonomy)
        prompts = tuple(SPEC.prompt_template.format(display_name=c.display_name.lower())
                        for c in self._taxonomy.concepts)
        if len(set(prompts)) != len(prompts):
            raise ValueError('duplicate taxonomy prompts after lowercase normalization')
        self._expected_identity = TextCacheIdentity(
            SPEC.model_id, SPEC.revision, self._taxonomy.taxonomy_version,
            hashlib.sha256(encoded.encode()).hexdigest(),
            tuple(c.concept_id for c in self._taxonomy.concepts), prompts, SPEC.prompt_template)
        self._model = self._processor = self._torch = self._text_features = None
        self._identity = None
        self._state = 'not_loaded'
        self._versions = {}

    @property
    def info(self):
        return ModelInfo(SPEC.model_id, SPEC.revision)

    @property
    def cache_identity(self):
        return self._identity

    @property
    def metadata(self):
        return {'model_id': SPEC.model_id, 'model_revision': SPEC.revision,
                'device': SPEC.device, 'score_kind': SPEC.score_kind,
                'taxonomy_version': self._taxonomy.taxonomy_version,
                'candidate_count': len(self._taxonomy.concepts),
                'prompt_template': SPEC.prompt_template,
                'ready': self._state == 'ready', 'load_state': self._state,
                **self._versions}

    def load(self):
        if self._state in ('ready', 'loading'):
            raise RuntimeStateError('runtime already loaded or loading')
        self._state = 'loading'
        try:
            os.environ.setdefault('HF_HOME', str(Path.home() / 'Library/Caches/eden-vision-ai/huggingface'))
            os.environ['HF_HUB_OFFLINE'] = '1'
            if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
                raise MPSRequirementError('CPU fallback must be disabled')
            torch, transformers = _dependencies()
            if (platform.machine() != 'arm64' or not torch.backends.mps.is_built()
                    or not torch.backends.mps.is_available()):
                raise MPSRequirementError('native arm64 with available MPS is required')
            self._torch = torch
            self._versions = {'torch_version': torch.__version__, 'transformers_version': transformers.__version__}
            # Stay inside the lazy dependency boundary: HF environment is ready.
            progress = transformers.utils.logging
            previous_progress_state = progress.is_progress_bar_enabled()
            try:
                progress.disable_progress_bar()
                self._processor = transformers.AutoProcessor.from_pretrained(
                    SPEC.model_id, revision=SPEC.revision, local_files_only=True)
                self._model = transformers.CLIPModel.from_pretrained(
                    SPEC.model_id, revision=SPEC.revision, local_files_only=True)
                self._model.to(SPEC.device)
                self._model.eval()
                if str(next(self._model.parameters()).device) not in ('mps', 'mps:0'):
                    raise MPSRequirementError('model did not move to MPS')
                inputs = {k: v.to(SPEC.device) for k, v in self._processor(
                    text=list(self._expected_identity.prompts), padding=True, return_tensors='pt').items()}
                with torch.inference_mode():
                    features = self._model.get_text_features(**inputs).pooler_output
                    self._text_features = features / features.norm(p=2, dim=-1, keepdim=True)
                if self._text_features.shape[0] != len(self._taxonomy.concepts):
                    raise ValueError('text embedding count does not match taxonomy')
                torch.mps.synchronize()
                self._identity = self._expected_identity
            finally:
                if previous_progress_state:
                    progress.enable_progress_bar()
            self._state = 'ready'
        except BaseException:
            # Do not mask the original dependency/cache/model failure with cleanup.
            try:
                self.close()
            except Exception:
                pass
            self._state = 'failed'
            raise

    def rank(self, image: ImageInput, *, top_k: int = DEFAULT_TOP_K) -> RankingResult:
        if self._state != 'ready' or self._identity != self._expected_identity:
            raise RuntimeStateError('load must complete before ranking')
        if type(top_k) is not int or not 1 <= top_k <= len(self._taxonomy.concepts):
            raise ValueError('top_k must be an integer in 1..candidate_count')
        if not isinstance(image, ImageInput):
            raise TypeError('image must be decoded RGB8 ImageInput')
        from PIL import Image
        with Image.frombytes('RGB', (image.width, image.height), image.pixels) as pil_image:
            inputs = {k: v.to(SPEC.device) for k, v in self._processor(
                images=pil_image, return_tensors='pt').items()}
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs).pooler_output
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            scores = self._matching_logits(features)[0].float().cpu().tolist()
        if len(scores) != len(self._taxonomy.concepts) or any(not math.isfinite(s) for s in scores):
            raise ValueError('model scores must be finite and match candidate_count')
        ordered = sorted(zip(self._taxonomy.concepts, scores, strict=True),
                         key=lambda pair: (-pair[1], pair[0].concept_id))
        return RankingResult(
            ranking=tuple(RankingItem(
                rank=i, concept_id=c.concept_id, display_name=c.display_name,
                broad_category=self._hierarchy.broad_category(c.concept_id),
                hierarchy=self._hierarchy.ancestor_path(c.concept_id),
                raw_score=float(score), score_kind=SPEC.score_kind,
            ) for i, (c, score) in enumerate(ordered[:top_k], 1)),
            model=self.info, taxonomy_version=self._taxonomy.taxonomy_version)

    def predict(self, image: ImageInput, *, top_k: int = DEFAULT_TOP_K) -> RankingResult:
        return self.rank(image, top_k=top_k)

    def _matching_logits(self, image_features):
        return (image_features @ self._text_features.T) * self._model.logit_scale.exp()

    def close(self):
        torch = self._torch
        self._state = 'closed'
        self._identity = None
        self._model = self._processor = self._text_features = self._torch = None
        self._versions = {}
        # MPS allocator is process-wide; trim only at explicit lifecycle teardown,
        # never per request. Caller must not race close against an active rank.
        if torch is not None:
            torch.mps.synchronize()
            torch.mps.empty_cache()
