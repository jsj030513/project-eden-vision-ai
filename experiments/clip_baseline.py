"""One-model CLIP/MPS smoke baseline; raw taxonomy logits only.

Run with the separate model-eval venv. Nothing here changes the API model or
uses the deterministic test thresholds. Network access downloads only the
requested CLIP checkpoint into HF_HOME outside the repository.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

# Set before importing Hugging Face libraries, which read cache settings early.
os.environ.setdefault("HF_HOME", str(Path.home() / "Library/Caches/eden-vision-ai/huggingface"))

from PIL import Image, ImageDraw
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoProcessor, CLIPModel

from eden_vision.classification.taxonomy import load_taxonomy
from eden_vision.services.image_input import ImageInputService

MODEL_ID = "openai/clip-vit-base-patch32"
PROMPT_TEMPLATE = "a photo of a {display_name}"
DEFAULT_IMAGE = Path("/tmp/eden-vision-ai-clip-flower.png")
DEFAULT_OUTPUT = Path("/tmp/eden-vision-ai-clip-baseline.json")


def make_fixture(path: Path) -> None:
    """A deterministic illustration for smoke testing, not an accuracy sample."""
    with Image.new("RGB", (256, 256), (194, 220, 175)) as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 205, 255, 255), fill=(107, 160, 82))
        draw.line((128, 110, 128, 225), fill=(34, 117, 42), width=14)
        draw.ellipse((65, 160, 125, 188), fill=(53, 139, 52))
        draw.ellipse((128, 146, 188, 172), fill=(53, 139, 52))
        for box in ((98, 45, 158, 110), (98, 110, 158, 175),
                    (63, 78, 126, 142), (130, 78, 193, 142)):
            draw.ellipse(box, fill=(205, 61, 80))
        draw.ellipse((111, 92, 145, 126), fill=(239, 190, 54))
        image.save(path, format="PNG")


def rss_mib() -> float:
    # ps reports current resident set size in KiB on macOS.
    completed = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                               text=True, capture_output=True, check=True)
    return int(completed.stdout.strip()) / 1024


def memory_snapshot() -> dict[str, float]:
    return {
        "rss_mib": round(rss_mib(), 2),
        "mps_allocated_mib": round(torch.mps.current_allocated_memory() / 2**20, 2),
        "mps_driver_allocated_mib": round(torch.mps.driver_allocated_memory() / 2**20, 2),
    }


def timed_forward(model: CLIPModel, inputs: dict[str, torch.Tensor]) -> tuple[float, torch.Tensor]:
    torch.mps.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits_per_image
    torch.mps.synchronize()
    return (time.perf_counter() - start) * 1000, logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warm-runs", type=int, default=10)
    args = parser.parse_args()
    if args.warm_runs < 3:
        parser.error("--warm-runs must be at least 3")
    if platform.machine() != "arm64" or not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Native Apple Silicon MPS is required; CPU fallback is not enabled")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK for native MPS measurement")
    if not hasattr(torch.mps, "synchronize"):
        raise RuntimeError("torch.mps.synchronize is required for accurate timing")

    if args.image == DEFAULT_IMAGE and not args.image.exists():
        make_fixture(args.image)
    image_bytes = args.image.read_bytes()
    image_input = ImageInputService(max_upload_bytes=10 * 1024 * 1024,
                                    max_image_pixels=25_000_000).decode(image_bytes, None)
    taxonomy = load_taxonomy()
    concepts = taxonomy.concepts
    prompts = [PROMPT_TEMPLATE.format(display_name=c.display_name.lower()) for c in concepts]

    # Resolve and pin a concrete commit before downloading or loading.
    revision = HfApi().model_info(MODEL_ID).sha
    if not revision or len(revision) != 40:
        raise RuntimeError("Could not resolve a full model revision SHA")
    download_start = time.perf_counter()
    snapshot_download(repo_id=MODEL_ID, revision=revision,
                      allow_patterns=["*.json", "*.txt", "pytorch_model.bin"],
                      max_workers=2)
    download_s = time.perf_counter() - download_start

    memory_before_load = memory_snapshot()
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision, local_files_only=True)
    model = CLIPModel.from_pretrained(MODEL_ID, revision=revision, local_files_only=True)
    model.to(torch.device("mps"))
    model.eval()
    torch.mps.synchronize()
    load_s = time.perf_counter() - load_start
    parameter_device = str(next(model.parameters()).device)
    if parameter_device != "mps:0" and parameter_device != "mps":
        raise RuntimeError(f"Model parameters are on {parameter_device}, not MPS")
    memory_after_load = memory_snapshot()

    # Phase 3 RGB8 domain boundary -> model-specific PIL conversion stays here.
    with Image.frombytes("RGB", (image_input.width, image_input.height), image_input.pixels) as image:
        processed = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {name: value.to("mps") for name, value in processed.items()}
    cold_ms, logits = timed_forward(model, inputs)
    memory_after_cold = memory_snapshot()
    warm_ms = []
    for _ in range(args.warm_runs):
        latency_ms, logits = timed_forward(model, inputs)
        warm_ms.append(latency_ms)
    memory_after_warm = memory_snapshot()

    values = logits[0].float().cpu().tolist()
    ranked = sorted(({"rank": 0, "concept_id": concept.concept_id,
                      "display_name": concept.display_name, "prompt": prompt,
                      "raw_logit": float(score)}
                     for concept, prompt, score in zip(concepts, prompts, values, strict=True)),
                    key=lambda row: (-row["raw_logit"], row["concept_id"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    result = {
        "model_id": MODEL_ID,
        "model_revision": revision,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "device": parameter_device,
        "score_kind": "logit",
        "score_note": "Raw CLIP logits_per_image; ranking scores, not calibrated probabilities",
        "prompt_template": PROMPT_TEMPLATE,
        "taxonomy_version": taxonomy.taxonomy_version,
        "candidate_count": len(concepts),
        "image": {"path": str(args.image), "source_format": image_input.source_format,
                  "width": image_input.width, "height": image_input.height,
                  "kind": "deterministic synthetic illustration" if args.image == DEFAULT_IMAGE else "user-supplied"},
        "timing": {"download_s": round(download_s, 3), "model_load_s": round(load_s, 3),
                   "cold_forward_ms": round(cold_ms, 3),
                   "warm_runs": args.warm_runs,
                   "warm_forward_min_ms": round(min(warm_ms), 3),
                   "warm_forward_median_ms": round(statistics.median(warm_ms), 3),
                   "warm_forward_max_ms": round(max(warm_ms), 3),
                   "warm_forward_all_ms": [round(v, 3) for v in warm_ms],
                   "timing_scope": "synchronized model forward only; excludes processor and image decode"},
        "memory": {"unit": "MiB", "rss_method": "ps RSS / 1024 KiB",
                   "before_load": memory_before_load, "after_load": memory_after_load,
                   "after_cold": memory_after_cold, "after_warm": memory_after_warm},
        "ranked_candidates": ranked,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**{k: result[k] for k in ("model_id", "model_revision", "device", "candidate_count")},
                      "timing": result["timing"], "memory": result["memory"],
                      "top_candidates": ranked[:5], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
