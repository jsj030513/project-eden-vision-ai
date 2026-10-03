"""Native MPS benchmark for fixed taxonomy text-feature caching.

This is an offline performance experiment, not an accuracy test or a
production adapter. Both checkpoints must already be in the external HF cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

# Hugging Face reads the cache setting during import. Never write weights here.
os.environ.setdefault("HF_HOME", str(Path.home() / "Library/Caches/eden-vision-ai/huggingface"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch
import transformers
from transformers import AutoModel, AutoProcessor, CLIPModel

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.classification.taxonomy import Taxonomy, load_taxonomy, parse_taxonomy
from eden_vision.services.image_input import ImageInputService
from experiments.clip_baseline import make_fixture

MODELS = {
    "clip": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
    "siglip2": ("google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"),
}
PROMPT_TEMPLATE = "a photo of a {display_name}"
DEFAULT_IMAGE = Path("/tmp/eden-vision-ai-clip-flower.png")
DEFAULT_OUTPUT = Path("/tmp/eden-taxonomy-scaling-benchmark.json")


def synchronize() -> None:
    torch.mps.synchronize()


def elapsed_ms(action: Callable[[], Any], *, gpu: bool = False) -> tuple[float, Any]:
    if gpu:
        synchronize()
    start = time.perf_counter()
    value = action()
    if gpu:
        synchronize()
    return (time.perf_counter() - start) * 1000, value


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = (len(ordered) - 1) * 0.95
    lower = int(p95_index)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (p95_index - lower)
    return {
        "count": len(values), "min_ms": min(values), "median_ms": statistics.median(values),
        "p95_ms": p95, "max_ms": max(values),
    }


def memory() -> dict[str, float]:
    rss = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True)
    return {
        "rss_mib": int(rss.stdout.strip()) / 1024,
        "mps_allocated_mib": torch.mps.current_allocated_memory() / 2**20,
        "mps_driver_mib": torch.mps.driver_allocated_memory() / 2**20,
    }


def ranked(logits: torch.Tensor, taxonomy: Taxonomy) -> list[dict[str, Any]]:
    values = logits[0].detach().float().cpu().tolist()
    rows = sorted(
        ({"concept_id": concept.concept_id, "raw_logit": float(score)}
         for concept, score in zip(taxonomy.concepts, values, strict=True)),
        key=lambda row: (-row["raw_logit"], row["concept_id"]),
    )
    return [{"rank": index, **row} for index, row in enumerate(rows, 1)]


def coherence(top: list[dict[str, Any]], resolver: HierarchyResolver) -> dict[str, float | int | None]:
    ids = [row["concept_id"] for row in top]
    pairs = [(left, right) for index, left in enumerate(ids) for right in ids[index + 1:]]
    return {
        "top_k": len(ids),
        "related_pair_fraction": (
            sum(resolver.common_ancestor(left, right) is not None for left, right in pairs) / len(pairs)
            if pairs else None
        ),
        "unique_broad_categories": len({resolver.broad_category(item) for item in ids}),
        "diagnostic_only": True,
    }


def to_device(inputs: Any) -> dict[str, torch.Tensor]:
    return {key: value.to("mps") for key, value in inputs.items()}


def projected_features(model: Any, inputs: dict[str, torch.Tensor], *, kind: str) -> torch.Tensor:
    # Transformers 5.17 returns BaseModelOutputWithPooling. pooler_output is
    # the official projected matching feature for both pinned checkpoints.
    output = model.get_text_features(**inputs) if kind == "text" else model.get_image_features(**inputs)
    features = output.pooler_output
    return features / features.norm(p=2, dim=-1, keepdim=True)


def cached_logits(model: Any, image_features: torch.Tensor, text_features: torch.Tensor, model_kind: str) -> torch.Tensor:
    scores = image_features @ text_features.T
    scores = scores * model.logit_scale.exp()
    if model_kind == "siglip2":
        scores = scores + model.logit_bias
    return scores


def process_full(processor: Any, prompts: list[str], image: Image.Image) -> dict[str, torch.Tensor]:
    return to_device(processor(text=prompts, images=image, return_tensors="pt", padding=True))


def process_text(processor: Any, prompts: list[str]) -> dict[str, torch.Tensor]:
    return to_device(processor(text=prompts, return_tensors="pt", padding=True))


def process_image(processor: Any, image: Image.Image) -> dict[str, torch.Tensor]:
    return to_device(processor(images=image, return_tensors="pt"))


def uncached_request(model: Any, processor: Any, prompts: list[str], image: Image.Image, taxonomy: Taxonomy) -> dict[str, Any]:
    started = time.perf_counter()
    processor_ms, inputs = elapsed_ms(lambda: process_full(processor, prompts, image))
    forward_ms, logits = elapsed_ms(lambda: model(**inputs).logits_per_image, gpu=True)
    rank_ms, ranking = elapsed_ms(lambda: ranked(logits, taxonomy), gpu=True)
    return {"processor_ms": processor_ms, "forward_ms": forward_ms, "ranking_ms": rank_ms,
            "total_ms": (time.perf_counter() - started) * 1000, "logits": logits, "ranking": ranking}


def cached_request(model: Any, processor: Any, text_features: torch.Tensor, image: Image.Image,
                   taxonomy: Taxonomy, model_kind: str) -> dict[str, Any]:
    started = time.perf_counter()
    processor_ms, image_inputs = elapsed_ms(lambda: process_image(processor, image))
    image_forward_ms, image_features = elapsed_ms(
        lambda: projected_features(model, image_inputs, kind="image"), gpu=True)
    similarity_ms, result = elapsed_ms(
        lambda: _score_and_rank(model, image_features, text_features, taxonomy, model_kind), gpu=True)
    logits, ranking = result
    return {"processor_ms": processor_ms, "image_forward_ms": image_forward_ms,
            "similarity_ranking_ms": similarity_ms, "total_ms": (time.perf_counter() - started) * 1000,
            "logits": logits, "ranking": ranking}


def _score_and_rank(model: Any, image_features: torch.Tensor, text_features: torch.Tensor,
                    taxonomy: Taxonomy, model_kind: str) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    logits = cached_logits(model, image_features, text_features, model_kind)
    return logits, ranked(logits, taxonomy)


def _timings(runs: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    return {key: summarize([run[key] for run in runs]) for key in keys}


def _embedding_metadata(features: torch.Tensor) -> dict[str, Any]:
    size = features.numel() * features.element_size()
    return {"shape": list(features.shape), "dtype": str(features.dtype), "numel": features.numel(),
            "bytes": size, "mib": size / 2**20, "device": str(features.device)}


def benchmark_taxonomy(model: Any, processor: Any, model_kind: str, taxonomy: Taxonomy,
                       image: Image.Image, warm_runs: int, memory_after_load: dict[str, float]) -> dict[str, Any]:
    prompts = [PROMPT_TEMPLATE.format(display_name=concept.display_name.lower()) for concept in taxonomy.concepts]
    resolver = HierarchyResolver(taxonomy)
    with torch.inference_mode():
        cold_uncached = uncached_request(model, processor, prompts, image, taxonomy)
        warm_uncached = [uncached_request(model, processor, prompts, image, taxonomy) for _ in range(warm_runs)]
        before_cache = memory()
        text_processor_ms, text_inputs = elapsed_ms(lambda: process_text(processor, prompts))
        text_precompute_ms, text_features = elapsed_ms(
            lambda: projected_features(model, text_inputs, kind="text"), gpu=True)
        del text_inputs
        after_cache = memory()
        cold_cached = cached_request(model, processor, text_features, image, taxonomy, model_kind)
        warm_cached = [cached_request(model, processor, text_features, image, taxonomy, model_kind)
                       for _ in range(warm_runs)]
        after_warm = memory()

        full_rank = [row["concept_id"] for row in warm_uncached[-1]["ranking"]]
        cached_rank = [row["concept_id"] for row in warm_cached[-1]["ranking"]]
        max_logit_delta = (warm_uncached[-1]["logits"] - warm_cached[-1]["logits"]).abs().max().item()
        equivalence = {
            "top_1_equal": full_rank[:1] == cached_rank[:1],
            "top_5_equal": full_rank[:5] == cached_rank[:5],
            "full_ranking_equal": full_rank == cached_rank,
            "max_absolute_logit_delta": max_logit_delta,
            "score_atol": 1e-3,
            "scores_close": bool(torch.allclose(warm_uncached[-1]["logits"], warm_cached[-1]["logits"], atol=1e-3, rtol=1e-4)),
        }
        if not equivalence["full_ranking_equal"] or not equivalence["scores_close"]:
            raise RuntimeError(f"cached and uncached mismatch for {model_kind}/{taxonomy.taxonomy_version}: {equivalence}")
        top_10 = warm_cached[-1]["ranking"][:10]
        result = {
            "taxonomy_version": taxonomy.taxonomy_version, "candidate_count": len(taxonomy.concepts),
            "prompt_template": PROMPT_TEMPLATE,
            "cache_key": {"model_id": MODELS[model_kind][0], "model_revision": MODELS[model_kind][1],
                          "taxonomy_version": taxonomy.taxonomy_version, "prompt_template": PROMPT_TEMPLATE,
                          "candidate_count": len(taxonomy.concepts)},
            "text_precompute_ms": text_precompute_ms, "text_processor_ms": text_processor_ms,
            "cached_embedding": _embedding_metadata(text_features),
            "uncached": {"cold_ms": cold_uncached["total_ms"], "warm_runs": warm_runs,
                         "segments": _timings(warm_uncached, ("processor_ms", "forward_ms", "ranking_ms", "total_ms"))},
            "cached": {"cold_ms": cold_cached["total_ms"], "warm_runs": warm_runs,
                       "segments": _timings(warm_cached, ("processor_ms", "image_forward_ms", "similarity_ranking_ms", "total_ms"))},
            "ranking_equivalence": equivalence,
            "top_10": top_10,
            "hierarchy_coherence_top_10": coherence(top_10, resolver),
            "memory": {"after_model_load": memory_after_load, "before_cache": before_cache,
                       "after_cache": after_cache, "after_warm": after_warm,
                       "cache_allocated_delta_mib": after_cache["mps_allocated_mib"] - before_cache["mps_allocated_mib"]},
        }
        del text_features, warm_uncached, warm_cached, cold_uncached, cold_cached
    gc.collect()
    torch.mps.empty_cache()
    synchronize()
    return result


def benchmark_model(model_kind: str, taxonomies: tuple[Taxonomy, ...], image: Image.Image,
                    warm_runs: int) -> dict[str, Any]:
    model_id, revision = MODELS[model_kind]
    before_load = memory()
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, local_files_only=True)
    if model_kind == "clip":
        model = CLIPModel.from_pretrained(model_id, revision=revision, local_files_only=True)
    else:
        # The pinned repository currently declares model_type=siglip; AutoModel
        # selects its compatible implementation in Transformers 5.17.
        model = AutoModel.from_pretrained(model_id, revision=revision, local_files_only=True)
    model.to("mps")
    model.eval()
    synchronize()
    load_seconds = time.perf_counter() - start
    parameter_device = str(next(model.parameters()).device)
    if parameter_device not in {"mps", "mps:0"}:
        raise RuntimeError(f"{model_kind} parameters are on {parameter_device}, not MPS")
    after_load = memory()
    conditions = {}
    for taxonomy in taxonomies:
        conditions[taxonomy.taxonomy_version] = benchmark_taxonomy(
            model, processor, model_kind, taxonomy, image, warm_runs, after_load)
    result = {"model_id": model_id, "model_revision": revision, "parameter_device": parameter_device,
              "load_seconds": load_seconds, "memory_before_load": before_load,
              "memory_after_load": after_load, "taxonomies": conditions}
    del model, processor
    gc.collect()
    torch.mps.empty_cache()
    synchronize()
    return result


def scaling(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, float]:
    keys = ("uncached", "cached")
    result = {"candidate_count_ratio": v2["candidate_count"] / v1["candidate_count"],
              "text_precompute_ratio": v2["text_precompute_ms"] / v1["text_precompute_ms"],
              "embedding_bytes_delta": v2["cached_embedding"]["bytes"] - v1["cached_embedding"]["bytes"]}
    for key in keys:
        first = v1[key]["segments"]["total_ms"]["median_ms"]
        second = v2[key]["segments"]["total_ms"]["median_ms"]
        result[f"{key}_total_median_ratio"] = second / first
        result[f"{key}_total_median_delta_ms"] = second - first
    first = v1["cached"]["segments"]["similarity_ranking_ms"]["median_ms"]
    second = v2["cached"]["segments"]["similarity_ranking_ms"]["median_ms"]
    result["similarity_ranking_median_ratio"] = second / first
    result["cache_allocated_delta_between_taxonomies_mib"] = (
        v2["memory"]["cache_allocated_delta_mib"] - v1["memory"]["cache_allocated_delta_mib"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warm-runs", type=int, default=20)
    args = parser.parse_args()
    if args.warm_runs < 20:
        parser.error("--warm-runs must be at least 20")
    if platform.machine() != "arm64" or not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("native Apple Silicon MPS is required")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must not be enabled")
    if not hasattr(torch.mps, "synchronize"):
        raise RuntimeError("torch.mps.synchronize is required")
    if args.image == DEFAULT_IMAGE and not args.image.exists():
        make_fixture(args.image)
    decoded = ImageInputService(max_upload_bytes=10 * 1024 * 1024, max_image_pixels=25_000_000).decode(
        args.image.read_bytes(), None)
    v1 = load_taxonomy()
    v2 = parse_taxonomy(files("eden_vision.resources").joinpath("taxonomy_v2.json").read_text(encoding="utf-8"))
    assert (v1.taxonomy_version, len(v1.concepts)) == ("eden-v1", 25)
    assert (v2.taxonomy_version, len(v2.concepts)) == ("eden-v2", 78)
    with Image.frombytes("RGB", (decoded.width, decoded.height), decoded.pixels) as image:
        models = {}
        for model_kind in ("clip", "siglip2"):
            models[model_kind] = benchmark_model(model_kind, (v1, v2), image, args.warm_runs)
            models[model_kind]["scaling_v1_to_v2"] = scaling(
                models[model_kind]["taxonomies"]["eden-v1"], models[model_kind]["taxonomies"]["eden-v2"])
            print(f"completed {model_kind}: v1/v2 uncached and cached", flush=True)
    result = {
        "environment": {"python": platform.python_version(), "architecture": platform.machine(),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
                        "hf_home": os.environ["HF_HOME"], "cpu_fallback_enabled": False},
        "image": {"path": str(args.image), "source_format": decoded.source_format,
                  "width": decoded.width, "height": decoded.height, "purpose": "synthetic performance fixture only"},
        "measurement": {"warm_runs_per_condition": args.warm_runs,
                        "preprocessing_included_in_request_total": True,
                        "mps_synchronized": True,
                        "score_note": "raw logits only; not calibrated probabilities; do not compare magnitude across models"},
        "models": models,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
