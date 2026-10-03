"""SigLIP2/MPS baseline using the exact Phase 4A candidate and prompt setup."""

import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / "Library/Caches/eden-vision-ai/huggingface"))

from PIL import Image
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModel, AutoProcessor

from eden_vision.classification.taxonomy import load_taxonomy
from eden_vision.services.image_input import ImageInputService

MODEL_ID = "google/siglip2-base-patch16-224"
PROMPT_TEMPLATE = "a photo of a {display_name}"
IMAGE_PATH = Path("/tmp/eden-vision-ai-clip-flower.png")
OUTPUT_PATH = Path("/tmp/eden-vision-ai-siglip2-baseline.json")
CLIP_PATH = Path("/tmp/eden-vision-ai-clip-baseline.json")


def rss_mib() -> float:
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True,
                            capture_output=True, check=True)
    return int(result.stdout.strip()) / 1024


def memory_snapshot() -> dict[str, float]:
    return {"rss_mib": round(rss_mib(), 2),
            "mps_allocated_mib": round(torch.mps.current_allocated_memory() / 2**20, 2),
            "mps_driver_allocated_mib": round(torch.mps.driver_allocated_memory() / 2**20, 2)}


def timed_forward(model, inputs):
    torch.mps.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs)
        logits = output.logits_per_image
    torch.mps.synchronize()
    return (time.perf_counter() - start) * 1000, logits


def main() -> None:
    if platform.machine() != "arm64" or not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Native Apple Silicon MPS is required; CPU fallback is disabled")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK for native MPS measurement")
    image_input = ImageInputService(max_upload_bytes=10 * 1024 * 1024,
                                    max_image_pixels=25_000_000).decode(IMAGE_PATH.read_bytes(), None)
    taxonomy = load_taxonomy()
    prompts = [PROMPT_TEMPLATE.format(display_name=c.display_name.lower()) for c in taxonomy.concepts]
    revision = HfApi().model_info(MODEL_ID).sha
    if not revision or len(revision) != 40:
        raise RuntimeError("Could not resolve full model revision SHA")
    cache_before = subprocess.check_output(["du", "-sk", os.environ["HF_HOME"]], text=True).split()[0]
    download_start = time.perf_counter()
    snapshot_download(repo_id=MODEL_ID, revision=revision,
                      allow_patterns=["*.json", "*.txt", "*.safetensors"], max_workers=2)
    download_s = time.perf_counter() - download_start
    cache_after = subprocess.check_output(["du", "-sk", os.environ["HF_HOME"]], text=True).split()[0]
    before_load = memory_snapshot()
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision, local_files_only=True)
    # The Hub checkpoint currently declares model_type=siglip; AutoModel selects
    # the compatible Siglip implementation. Direct Siglip2Model construction
    # causes patch/position embedding shape mismatches in this Transformers release.
    model = AutoModel.from_pretrained(MODEL_ID, revision=revision, local_files_only=True)
    model.to(torch.device("mps"))
    model.eval()
    torch.mps.synchronize()
    load_s = time.perf_counter() - load_start
    parameter_device = str(next(model.parameters()).device)
    if parameter_device not in ("mps", "mps:0"):
        raise RuntimeError(f"Model parameters are on {parameter_device}, not MPS")
    after_load = memory_snapshot()
    with Image.frombytes("RGB", (image_input.width, image_input.height), image_input.pixels) as image:
        # Match the evaluation runtime: batch-longest padding can collapse
        # SigLIP2 rankings. AutoProcessor supplies the checkpoint's max length.
        processed = processor(text=prompts, images=image, return_tensors="pt",
                              padding="max_length", truncation=True)
    inputs = {name: value.to("mps") for name, value in processed.items()}
    cold_ms, logits = timed_forward(model, inputs)
    after_cold = memory_snapshot()
    warm = []
    for _ in range(10):
        latency, logits = timed_forward(model, inputs)
        warm.append(latency)
    after_warm = memory_snapshot()
    scores = logits[0].float().cpu().tolist()
    ranked = sorted(({"rank": 0, "concept_id": c.concept_id, "display_name": c.display_name,
                      "prompt": p, "raw_logit": float(s)}
                     for c, p, s in zip(taxonomy.concepts, prompts, scores, strict=True)),
                    key=lambda x: (-x["raw_logit"], x["concept_id"]))
    for index, item in enumerate(ranked, 1):
        item["rank"] = index
    result = {"model_id": MODEL_ID, "model_revision": revision,
              "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
              "python_version": platform.python_version(), "architecture": platform.machine(),
              "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
              "device": parameter_device, "score_kind": "logit",
              "score_note": "Raw SigLIP2 logits_per_image; within-model ranking only, not calibrated probabilities",
              "prompt_template": PROMPT_TEMPLATE, "taxonomy_version": taxonomy.taxonomy_version,
              "candidate_count": len(taxonomy.concepts),
              "image": {"path": str(IMAGE_PATH), "source_format": image_input.source_format,
                        "width": image_input.width, "height": image_input.height,
                        "kind": "same deterministic synthetic illustration as Phase 4A"},
              "timing": {"download_s": round(download_s, 3), "model_load_s": round(load_s, 3),
                         "cold_forward_ms": round(cold_ms, 3), "warm_runs": 10,
                         "warm_forward_min_ms": round(min(warm), 3),
                         "warm_forward_median_ms": round(statistics.median(warm), 3),
                         "warm_forward_max_ms": round(max(warm), 3),
                         "warm_forward_all_ms": [round(v, 3) for v in warm],
                         "timing_scope": "synchronized model forward only; excludes processor and image decode"},
              "memory": {"unit": "MiB", "rss_method": "ps RSS / 1024 KiB",
                         "before_load": before_load, "after_load": after_load,
                         "after_cold": after_cold, "after_warm": after_warm},
              "cache": {"hf_home": os.environ["HF_HOME"], "before_kib": int(cache_before),
                        "after_kib": int(cache_after), "increase_kib": int(cache_after) - int(cache_before)},
              "ranked_candidates": ranked}
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    comparison = {"comparison_note": "Do not compare raw logit magnitudes across models",
                  "clip": None, "siglip2": {"model_id": MODEL_ID, "revision": revision,
                  "load_seconds": result["timing"]["model_load_s"], "cold_ms": result["timing"]["cold_forward_ms"],
                  "warm_min_ms": result["timing"]["warm_forward_min_ms"], "warm_median_ms": result["timing"]["warm_forward_median_ms"],
                  "warm_max_ms": result["timing"]["warm_forward_max_ms"], "rss_mib": result["memory"]["after_warm"]["rss_mib"],
                  "mps_allocated_mib": result["memory"]["after_warm"]["mps_allocated_mib"],
                  "mps_driver_allocated_mib": result["memory"]["after_warm"]["mps_driver_allocated_mib"],
                  "top_5_labels": [x["concept_id"] for x in ranked[:5]]}}
    if CLIP_PATH.exists():
        clip = json.loads(CLIP_PATH.read_text())
        comparison["clip"] = {"model_id": clip["model_id"], "revision": clip["model_revision"],
                               "load_seconds": clip["timing"]["model_load_s"], "cold_ms": clip["timing"]["cold_forward_ms"],
                               "warm_min_ms": clip["timing"]["warm_forward_min_ms"], "warm_median_ms": clip["timing"]["warm_forward_median_ms"],
                               "warm_max_ms": clip["timing"]["warm_forward_max_ms"], "rss_mib": clip["memory"]["after_warm"]["rss_mib"],
                               "mps_allocated_mib": clip["memory"]["after_warm"]["mps_allocated_mib"],
                               "mps_driver_allocated_mib": clip["memory"]["after_warm"]["mps_driver_allocated_mib"],
                               "top_5_labels": [x["concept_id"] for x in clip["ranked_candidates"][:5]]}
    Path("/tmp/eden-vision-ai-model-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({"model_id": MODEL_ID, "revision": revision, "device": parameter_device,
                      "timing": result["timing"], "memory": result["memory"], "cache": result["cache"],
                      "top5": ranked[:5], "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
