import hashlib
import json
from pathlib import Path

from PIL import Image

from experiments.generate_unknown_candidates import (
    SOURCE_IMAGE_IDS,
    generate_candidates,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_dataset(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "dataset.json").write_text(
        json.dumps({"dataset_id": "fixture", "taxonomy_version": "eden-v2", "image_count": 8}),
        encoding="utf-8",
    )
    rows = []
    for index, image_id in enumerate(SOURCE_IMAGE_IDS):
        image_path = root / "images" / f"{image_id}.jpg"
        image = Image.new("RGB", (320, 240))
        pixels = image.load()
        for y in range(image.height):
            for x in range(image.width):
                pixels[x, y] = ((x + index * 17) % 256, (y + index * 29) % 256, (x + y + index * 11) % 256)
        image.save(image_path, format="JPEG", quality=90)
        rows.append({
            "id": image_id,
            "file": f"images/{image_path.name}",
            "sha256": _sha(image_path),
        })
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "manifest.draft.jsonl").write_text("", encoding="utf-8")
    (root / "reports" / "historical.json").write_text('{"images": 8}\n', encoding="utf-8")


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generation_is_deterministic_and_leakage_safe(tmp_path):
    dataset = tmp_path / "source-dataset"
    _fixture_dataset(dataset)
    source_before = _tree_snapshot(dataset)

    first = generate_candidates(dataset, tmp_path / "first", seed=20260924)
    second = generate_candidates(dataset, tmp_path / "second", seed=20260924)

    assert first == second
    assert _tree_snapshot(dataset) == source_before
    assert len(first) == 16
    assert [row["candidate_id"] for row in first] == [f"unknown-candidate-{i:03d}" for i in range(1, 17)]
    assert {row["intended_split"] for row in first} == {"calibration", "holdout"}
    assert sum(row["intended_split"] == "calibration" for row in first) == 8
    assert sum(row["intended_split"] == "holdout" for row in first) == 8
    calibration_sources = {row["source_image_id"] for row in first if row["intended_split"] == "calibration"}
    holdout_sources = {row["source_image_id"] for row in first if row["intended_split"] == "holdout"}
    assert calibration_sources.isdisjoint(holdout_sources)
    assert len(calibration_sources) == len(holdout_sources) == 4

    for output in (tmp_path / "first", tmp_path / "second"):
        metadata = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]
        assert metadata == first
        for row in metadata:
            image_path = output / "generated" / f"{row['candidate_id']}.jpg"
            assert _sha(image_path) == row["generated_sha256"]
            assert row["review_status"] == "pending"
            assert not ({"unknown", "expected", "tags"} & row.keys())
        assert (output / "contact" / "calibration.jpg").is_file()
        assert (output / "contact" / "holdout.jpg").is_file()
        with Image.open(output / "contact" / "calibration.jpg") as image:
            assert image.width >= 4 * 380
            assert image.height >= 2 * 350
        with Image.open(output / "contact" / "holdout.jpg") as image:
            assert image.width >= 4 * 380
            assert image.height >= 2 * 350


def test_existing_output_is_never_overwritten(tmp_path):
    dataset = tmp_path / "source-dataset"
    _fixture_dataset(dataset)
    output = tmp_path / "already-there"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    try:
        generate_candidates(dataset, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected existing output to be refused")
    assert marker.read_text(encoding="utf-8") == "preserve"
