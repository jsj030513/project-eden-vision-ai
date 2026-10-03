"""Evaluation-only cached matching runtimes; AI imports happen in load()."""

from __future__ import annotations

import gc
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from eden_vision.classification.taxonomy import Taxonomy, parse_taxonomy
from eden_vision.vision.types import ImageInput

PROMPT_TEMPLATE = "a photo of a {display_name}"
TAXONOMIES = {"eden-v1": "taxonomy_v1.json", "eden-v2": "taxonomy_v2.json"}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str


MODELS = {
    "clip": ModelSpec("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
    "siglip2": ModelSpec("google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"),
}


def load_evaluation_taxonomy(version: str) -> Taxonomy:
    if version not in TAXONOMIES:
        raise ValueError(f"unsupported taxonomy: {version}")
    value = parse_taxonomy(files("eden_vision.resources").joinpath(TAXONOMIES[version]).read_text(encoding="utf-8"))
    if value.taxonomy_version != version:
        raise ValueError("taxonomy resource version mismatch")
    return value


@dataclass(frozen=True)
class CacheIdentity:
    model_id: str
    model_revision: str
    taxonomy_version: str
    prompt_template: str
    candidate_ids: tuple[str, ...]
    prompts: tuple[str, ...]


def cache_identity(spec: ModelSpec, taxonomy: Taxonomy) -> CacheIdentity:
    return CacheIdentity(
        spec.model_id, spec.revision, taxonomy.taxonomy_version, PROMPT_TEMPLATE,
        tuple(c.concept_id for c in taxonomy.concepts),
        tuple(PROMPT_TEMPLATE.format(display_name=c.display_name.lower()) for c in taxonomy.concepts),
    )


@dataclass
class RankedImage:
    candidates: list[dict[str, Any]]
    timings_ms: dict[str, float]


class EvaluationModelRuntime:
    """One loaded model and one active in-memory taxonomy cache.

    Lifecycle: load -> prepare_taxonomy -> rank* -> close. Re-preparing the
    same identity reuses the cache; changed identities replace it. There is
    no disk embedding cache, production policy, threshold, or confidence.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.model = self.processor = self._torch = None
        self._features = None
        self.identity: CacheIdentity | None = None
        self.cache_info: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}

    def load(self) -> None:
        if self.model is not None:
            raise RuntimeError("runtime already loaded")
        os.environ.setdefault("HF_HOME", str(Path.home() / "Library/Caches/eden-vision-ai/huggingface"))
        # Evaluation consumes the existing pinned snapshots, without network.
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        import transformers

        if (platform.machine() != "arm64" or not torch.backends.mps.is_built()
                or not torch.backends.mps.is_available()):
            raise RuntimeError("native Apple Silicon MPS is required; use the model-eval venv")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("CPU fallback must be disabled")
        self._torch = torch
        before = self.memory()
        started = time.perf_counter()
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(
                self.spec.model_id, revision=self.spec.revision, local_files_only=True)
            self.model = self._load_checkpoint(transformers)
            self.model.to("mps")
            self.model.eval()
            torch.mps.synchronize()
            device = str(next(self.model.parameters()).device)
            if device not in {"mps", "mps:0"}:
                raise RuntimeError(f"unexpected model device: {device}")
            self.metadata = {
                "model_id": self.spec.model_id, "model_revision": self.spec.revision,
                "device": device, "torch_version": torch.__version__,
                "transformers_version": transformers.__version__, "python_version": platform.python_version(),
                "model_class": type(self.model).__name__, "load_seconds": time.perf_counter() - started,
                "memory_before_load": before, "memory_after_load": self.memory(),
            }
        except BaseException:
            self.close()
            raise

    def _load_checkpoint(self, transformers: Any) -> Any:
        raise NotImplementedError

    def _require_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("load() must precede prepare_taxonomy()/rank()")

    def prepare_taxonomy(self, taxonomy: Taxonomy) -> dict[str, Any]:
        self._require_loaded()
        identity = cache_identity(self.spec, taxonomy)
        if identity == self.identity:
            return self.cache_info
        # Invalidate before encoding: failures must not leave an old active cache.
        self._features = None
        self.identity = None
        self.cache_info = {}
        features, details = self._encode_taxonomy(identity.prompts)
        self._features, self.identity = features, identity
        self.cache_info = {"identity": asdict(identity), "candidate_count": len(identity.candidate_ids), **details}
        return self.cache_info

    def _text_processor_kwargs(self) -> dict[str, Any]:
        return {"padding": True}

    def _encode_taxonomy(self, prompts: tuple[str, ...]) -> tuple[Any, dict[str, Any]]:
        torch = self._torch
        torch.mps.synchronize()
        start = time.perf_counter()
        inputs = {k: v.to("mps") for k, v in self.processor(
            text=list(prompts), return_tensors="pt", **self._text_processor_kwargs()).items()}
        torch.mps.synchronize()
        processed = time.perf_counter()
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs).pooler_output
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        torch.mps.synchronize()
        end = time.perf_counter()
        return features, {
            "text_processor_ms": (processed - start) * 1000,
            "text_precompute_ms": (end - processed) * 1000,
            "shape": list(features.shape), "dtype": str(features.dtype),
            "bytes": features.numel() * features.element_size(),
        }

    def rank(self, image: ImageInput) -> RankedImage:
        self._require_loaded()
        if self.identity is None:
            raise RuntimeError("prepare_taxonomy() must precede rank()")
        from PIL import Image

        torch = self._torch
        torch.mps.synchronize()
        start = time.perf_counter()
        with Image.frombytes("RGB", (image.width, image.height), image.pixels) as pil_image:
            inputs = {k: v.to("mps") for k, v in self.processor(images=pil_image, return_tensors="pt").items()}
        torch.mps.synchronize()
        processed = time.perf_counter()
        with torch.inference_mode():
            features = self.model.get_image_features(**inputs).pooler_output
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            torch.mps.synchronize()
            forwarded = time.perf_counter()
            scores = self._matching_logits(features)[0].float().cpu().tolist()
        candidates = sorted(
            ({"concept_id": cid, "raw_score": float(score), "score_kind": "raw_image_text_logit"}
             for cid, score in zip(self.identity.candidate_ids, scores, strict=True)),
            key=lambda row: (-row["raw_score"], row["concept_id"]),
        )
        torch.mps.synchronize()
        ended = time.perf_counter()
        return RankedImage(candidates, {
            "processor_ms": (processed - start) * 1000,
            "image_forward_ms": (forwarded - processed) * 1000,
            "similarity_ranking_ms": (ended - forwarded) * 1000,
            "total_ms": (ended - start) * 1000,
        })

    def _matching_logits(self, image_features: Any) -> Any:
        raise NotImplementedError

    def memory(self) -> dict[str, float]:
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             text=True, capture_output=True, check=True)
        return {"rss_mib": int(rss.stdout.strip()) / 1024,
                "mps_allocated_mib": self._torch.mps.current_allocated_memory() / 2**20,
                "mps_driver_mib": self._torch.mps.driver_allocated_memory() / 2**20}

    def close(self) -> None:
        self._features = self.model = self.processor = None
        self.identity = None
        self.cache_info = {}
        gc.collect()
        if self._torch is not None:
            self._torch.mps.synchronize()
            self._torch.mps.empty_cache()
        self._torch = None


class ClipEvaluationRuntime(EvaluationModelRuntime):
    def __init__(self) -> None:
        super().__init__(MODELS["clip"])

    def _load_checkpoint(self, transformers: Any) -> Any:
        return transformers.CLIPModel.from_pretrained(
            self.spec.model_id, revision=self.spec.revision, local_files_only=True)

    def _matching_logits(self, image_features: Any) -> Any:
        return (image_features @ self._features.T) * self.model.logit_scale.exp()


class Siglip2EvaluationRuntime(EvaluationModelRuntime):
    def __init__(self) -> None:
        super().__init__(MODELS["siglip2"])

    def _text_processor_kwargs(self) -> dict[str, Any]:
        # SigLIP2 needs fixed-length text, not padding to the longest prompt in
        # this batch. Let AutoProcessor select the checkpoint's maximum length.
        return {"padding": "max_length", "truncation": True}

    def _load_checkpoint(self, transformers: Any) -> Any:
        # This pinned FixRes checkpoint declares model_type=siglip.
        return transformers.AutoModel.from_pretrained(
            self.spec.model_id, revision=self.spec.revision, local_files_only=True)

    def _matching_logits(self, image_features: Any) -> Any:
        return (image_features @ self._features.T) * self.model.logit_scale.exp() + self.model.logit_bias


def create_runtime(model: str) -> EvaluationModelRuntime:
    factories = {"clip": ClipEvaluationRuntime, "siglip2": Siglip2EvaluationRuntime}
    if model not in factories:
        raise ValueError(f"unsupported model: {model}")
    return factories[model]()
