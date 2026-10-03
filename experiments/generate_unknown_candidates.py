"""Generate a separate, human-reviewed low-information candidate set.

This tool only creates image variants and provenance metadata. It does not
assign Ground Truth and never writes into an evaluation dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()

DEFAULT_DATASET = Path("/Users/dev/Datasets/eden-vision-open-set-eval")
DEFAULT_OUTPUT = Path("/Users/dev/Datasets/eden-vision-open-set-candidates")
DEFAULT_SEED = 20260924
OUTPUT_SIZE = (640, 640)

# The selected canonical rows cover distinct, human-visible source categories.
# Their ordering also fixes the leakage-safe split: each source belongs wholly
# to calibration or holdout, and its two variants stay together.
SOURCE_IMAGE_IDS = (
    "eval-024",  # person
    "eval-014",  # cat
    "eval-010",  # coast / landscape
    "eval-009",  # buildings
    "eval-019",  # flowers
    "eval-003",  # tiger, represented as animal
    "eval-026",  # fruit
    "eval-033",  # sign
)

VARIANTS: dict[str, tuple[str, ...]] = {
    "eval-024": ("extreme_crop", "heavy_motion_blur"),
    "eval-014": ("heavy_defocus_blur", "severe_underexposure"),
    "eval-010": ("severe_overexposure", "heavy_defocus_blur"),
    "eval-009": ("extreme_crop", "heavy_motion_blur"),
    "eval-019": ("heavy_defocus_blur", "severe_underexposure"),
    "eval-003": ("severe_overexposure", "texture_only_crop"),
    "eval-026": ("extreme_crop", "texture_only_crop"),
    "eval-033": ("heavy_motion_blur", "severe_overexposure"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seed_for(global_seed: int, candidate_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{candidate_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big")


def _fit_base(image: Image.Image) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), OUTPUT_SIZE, method=Image.Resampling.LANCZOS)


def _motion_blur(image: Image.Image, angle_degrees: float, length: int) -> Image.Image:
    radians = math.radians(angle_degrees)
    dx, dy = math.cos(radians), math.sin(radians)
    samples = max(19, length // 3)
    result: Image.Image | None = None
    for index in range(samples):
        distance = -length / 2 + length * index / (samples - 1)
        shifted = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, -distance * dx, 0, 1, -distance * dy),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        result = shifted if result is None else Image.blend(result, shifted, 1 / (index + 1))
    assert result is not None
    return result


def transform_image(
    image: Image.Image, transformation: str, seed: int
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply one named transform with all randomness derived from its seed."""
    rng = random.Random(seed)
    base = _fit_base(image)
    if transformation == "extreme_crop":
        area_fraction = rng.choice((0.035, 0.05, 0.07))
        center_x = rng.random()
        center_y = rng.random()
        source = image.convert("RGB")
        width, height = source.size
        aspect = width / height
        crop_width = max(1, min(width, round(math.sqrt(area_fraction * width * height * aspect))))
        crop_height = max(1, min(height, round(math.sqrt(area_fraction * width * height / aspect))))
        left = round(center_x * max(0, width - crop_width))
        top = round(center_y * max(0, height - crop_height))
        crop_blur_radius = 48.0
        result = ImageOps.fit(
            source.crop((left, top, left + crop_width, top + crop_height)),
            OUTPUT_SIZE,
            method=Image.Resampling.LANCZOS,
        ).filter(ImageFilter.GaussianBlur(radius=crop_blur_radius))
        return result, {
            "area_fraction": area_fraction,
            "center_x_fraction": round(center_x, 6),
            "center_y_fraction": round(center_y, 6),
            "post_crop_blur_radius_pixels": crop_blur_radius,
            "output_size": list(OUTPUT_SIZE),
        }
    if transformation == "heavy_motion_blur":
        angle = rng.choice((17, 31, 63, 104, 143))
        length = rng.choice((420, 460, 500))
        return _motion_blur(base, angle, length), {
            "angle_degrees": angle,
            "length_pixels": length,
            "output_size": list(OUTPUT_SIZE),
        }
    if transformation == "heavy_defocus_blur":
        radius = rng.choice((48.0, 56.0, 64.0))
        return base.filter(ImageFilter.GaussianBlur(radius=radius)), {
            "gaussian_radius_pixels": radius,
            "output_size": list(OUTPUT_SIZE),
        }
    if transformation == "severe_underexposure":
        factor = rng.choice((0.018, 0.025, 0.035))
        return ImageOps.autocontrast(base).point(lambda value: round(value * factor)), {
            "brightness_factor": factor,
            "output_size": list(OUTPUT_SIZE),
        }
    if transformation == "severe_overexposure":
        factor = rng.choice((8.0, 9.0, 10.0))
        washout = rng.choice((0.96, 0.97, 0.98))
        bright = base.point(lambda value: min(255, round(value * factor)))
        white = Image.new("RGB", OUTPUT_SIZE, "white")
        return Image.blend(bright, white, washout), {
            "brightness_factor": factor,
            "white_wash_fraction": washout,
            "output_size": list(OUTPUT_SIZE),
        }
    if transformation == "texture_only_crop":
        area_fraction = rng.choice((0.06, 0.08, 0.10))
        center_x = rng.random()
        center_y = rng.random()
        source = image.convert("RGB")
        width, height = source.size
        aspect = width / height
        crop_width = max(1, min(width, round(math.sqrt(area_fraction * width * height * aspect))))
        crop_height = max(1, min(height, round(math.sqrt(area_fraction * width * height / aspect))))
        left = round(center_x * max(0, width - crop_width))
        top = round(center_y * max(0, height - crop_height))
        crop_blur_radius = 42.0
        result = ImageOps.fit(
            source.crop((left, top, left + crop_width, top + crop_height)),
            OUTPUT_SIZE,
            method=Image.Resampling.LANCZOS,
        ).filter(ImageFilter.GaussianBlur(radius=crop_blur_radius))
        return result, {
            "area_fraction": area_fraction,
            "center_x_fraction": round(center_x, 6),
            "center_y_fraction": round(center_y, 6),
            "post_crop_blur_radius_pixels": crop_blur_radius,
            "output_size": list(OUTPUT_SIZE),
        }
    raise ValueError(f"unsupported transformation: {transformation}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _draw_contact_sheet(rows: list[dict[str, Any]], generated: Path, output: Path) -> None:
    columns = 4
    cell_width, image_height, caption_height = 400, 300, 76
    rows_count = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows_count * (image_height + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % columns) * cell_width
        y = (index // columns) * (image_height + caption_height)
        with Image.open(generated / f"{row['candidate_id']}.jpg") as image:
            preview = ImageOps.contain(image.convert("RGB"), (cell_width - 16, image_height - 12))
        sheet.paste(preview, (x + (cell_width - preview.width) // 2, y + 4))
        caption = (
            row["candidate_id"],
            f"source: {row['source_image_id']}",
            row["transformation"],
        )
        for line_number, line in enumerate(caption):
            draw.text((x + 8, y + image_height + 4 + line_number * 21), line, fill="black")
    sheet.save(output, format="JPEG", quality=92, optimize=True)


def generate_candidates(
    dataset: Path = DEFAULT_DATASET,
    output: Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    dataset = Path(dataset).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    manifest_path = dataset / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"canonical manifest missing: {manifest_path}")
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")

    canonical = _read_jsonl(manifest_path)
    by_id = {row["id"]: row for row in canonical}
    missing = set(SOURCE_IMAGE_IDS) - by_id.keys()
    if missing:
        raise ValueError(f"canonical source IDs missing: {sorted(missing)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        source_dir, generated_dir, contact_dir = (
            staging / "source",
            staging / "generated",
            staging / "contact",
        )
        source_dir.mkdir()
        generated_dir.mkdir()
        contact_dir.mkdir()

        # Split assignment is completed before any transform is generated.
        source_split = {
            image_id: ("calibration" if index < len(SOURCE_IMAGE_IDS) // 2 else "holdout")
            for index, image_id in enumerate(SOURCE_IMAGE_IDS)
        }
        metadata: list[dict[str, Any]] = []
        for image_id in SOURCE_IMAGE_IDS:
            source_row = by_id[image_id]
            source_path = (dataset / source_row["file"]).resolve()
            if dataset not in source_path.parents or not source_path.is_file():
                raise ValueError(f"invalid canonical image path for {image_id}")
            source_bytes = source_path.read_bytes()
            source_hash = sha256_bytes(source_bytes)
            if source_hash != source_row["sha256"]:
                raise ValueError(f"source image hash mismatch for {image_id}")
            with Image.open(source_path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            for transformation in VARIANTS[image_id]:
                candidate_id = f"unknown-candidate-{len(metadata) + 1:03d}"
                candidate_seed = _seed_for(seed, candidate_id)
                transformed, parameters = transform_image(image, transformation, candidate_seed)
                candidate_path = generated_dir / f"{candidate_id}.jpg"
                transformed.save(candidate_path, format="JPEG", quality=94, optimize=True)
                generated_hash = sha256_bytes(candidate_path.read_bytes())
                metadata.append({
                    "candidate_id": candidate_id,
                    "source_image_id": image_id,
                    "source_sha256": source_hash,
                    "transformation": transformation,
                    "parameters": parameters,
                    "seed": candidate_seed,
                    "generated_sha256": generated_hash,
                    "intended_split": source_split[image_id],
                    "review_status": "pending",
                })
            if sha256_bytes(source_path.read_bytes()) != source_hash:
                raise RuntimeError(f"source image changed during processing: {image_id}")

        if len(metadata) != 16:
            raise RuntimeError(f"expected 16 variants, generated {len(metadata)}")
        with (staging / "metadata.jsonl").open("x", encoding="utf-8") as stream:
            for row in metadata:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        for split in ("calibration", "holdout"):
            _draw_contact_sheet(
                [row for row in metadata if row["intended_split"] == split],
                generated_dir,
                contact_dir / f"{split}.jpg",
            )
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    rows = generate_candidates(args.dataset, args.output, args.seed)
    print(json.dumps({
        "output": str(args.output),
        "candidates": len(rows),
        "splits": dict(Counter(row["intended_split"] for row in rows)),
        "transformations": dict(Counter(row["transformation"] for row in rows)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
