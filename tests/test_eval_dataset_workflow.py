import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from eden_vision.classification.taxonomy import load_taxonomy
from eden_vision.services.image_input import ImageInputError
from experiments.eval_dataset import import_inbox, promote_draft, set_dataset_taxonomy, update_dataset_count
from experiments.evaluation_core import ManifestValidationError, load_dataset
from experiments.model_runtime import load_evaluation_taxonomy


def _dataset(tmp_path):
    (tmp_path / "inbox").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "dataset.json").write_text(json.dumps({"dataset_id": "test", "taxonomy_version": "eden-v1", "image_count": 0}), encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
    return tmp_path


def _png(path, color=(255, 0, 0)):
    with Image.new("RGB", (4, 3), color) as image:
            image.save(path, format="PNG")


def _heif(path, color=(255, 0, 0)):
    with Image.new("RGB", (8, 6), color) as image:
        image.save(path, format="HEIF")


def test_valid_import_creates_safe_draft_and_preserves_inbox(tmp_path):
    dataset = _dataset(tmp_path)
    source = dataset / "inbox" / "카카오톡 사진 2026.png"
    _png(source)
    result = import_inbox(dataset)
    assert result == {"scanned": 1, "imported": 1, "duplicates": 0, "rejected": 0}
    assert source.exists()
    draft = json.loads((dataset / "manifest.draft.jsonl").read_text().strip())
    assert draft["id"] == "eval-001"
    assert draft["file"] == "images/eval-001.png"
    assert draft["expected"] is None and draft["unknown"] is None
    assert draft["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_duplicate_hash_is_not_registered_twice(tmp_path):
    dataset = _dataset(tmp_path)
    _png(dataset / "inbox" / "a.png")
    _png(dataset / "inbox" / "b.png")
    first = import_inbox(dataset)
    second = import_inbox(dataset)
    assert first["imported"] == 1
    assert second["imported"] == 0 and second["duplicates"] == 2
    assert len((dataset / "manifest.draft.jsonl").read_text().splitlines()) == 1


def test_invalid_and_unsupported_files_are_rejected(tmp_path):
    dataset = _dataset(tmp_path)
    (dataset / "inbox" / "bad.jpg").write_bytes(b"not an image")
    with Image.new("RGB", (2, 2)) as image:
        image.save(dataset / "inbox" / "animated.gif", format="GIF")
    result = import_inbox(dataset)
    assert result["imported"] == 0 and result["rejected"] == 2
    assert not (dataset / "images" / "eval-001.jpg").exists()


def test_id_sequence_does_not_reuse_existing_ids(tmp_path):
    dataset = _dataset(tmp_path)
    (dataset / "manifest.draft.jsonl").write_text(json.dumps({"id": "eval-001", "file": "images/eval-001.png"}) + "\n", encoding="utf-8")
    _png(dataset / "inbox" / "new.png", (0, 255, 0))
    assert import_inbox(dataset)["imported"] == 1
    assert "eval-002" in (dataset / "manifest.draft.jsonl").read_text()


def test_promotion_requires_human_labels_and_updates_count(tmp_path):
    dataset = _dataset(tmp_path)
    _png(dataset / "inbox" / "flower.png")
    import_inbox(dataset)
    first = promote_draft(dataset, load_taxonomy())
    assert first["promoted"] == 0 and first["skipped_incomplete"] == 1
    draft = json.loads((dataset / "manifest.draft.jsonl").read_text().strip())
    draft.update({"expected": [{"concept_id": "flower", "acceptable_ancestors": ["plant", "living_thing"]}], "tags": ["broad_only"], "unknown": False, "notes": "human label"})
    (dataset / "manifest.draft.jsonl").write_text(json.dumps(draft) + "\n", encoding="utf-8")
    assert promote_draft(dataset, load_taxonomy())["promoted"] == 1
    assert (dataset / "manifest.draft.jsonl").read_text() == ""
    assert json.loads((dataset / "dataset.json").read_text())["image_count"] == 1
    loaded = load_dataset(dataset, load_taxonomy())
    assert loaded.entries[0].expected[0].concept_id == "flower"


def test_unknown_promotion_and_duplicate_canonical_are_rejected(tmp_path):
    dataset = _dataset(tmp_path)
    _png(dataset / "inbox" / "unknown.png")
    import_inbox(dataset)
    draft = json.loads((dataset / "manifest.draft.jsonl").read_text().strip())
    draft.update({"expected": [], "tags": ["unknown"], "unknown": True, "notes": None})
    (dataset / "manifest.draft.jsonl").write_text(json.dumps(draft) + "\n", encoding="utf-8")
    assert promote_draft(dataset, load_taxonomy())["promoted"] == 1
    assert promote_draft(dataset, load_taxonomy())["promoted"] == 0


def test_update_dataset_count_uses_canonical_only(tmp_path):
    dataset = _dataset(tmp_path)
    (dataset / "manifest.jsonl").write_text("{\"id\":\"one\"}\n\n", encoding="utf-8")
    assert update_dataset_count(dataset) == 1
    assert json.loads((dataset / "dataset.json").read_text())["image_count"] == 1


def test_partial_promotion_preserves_incomplete_and_is_repeatable(tmp_path):
    dataset = _dataset(tmp_path)
    for name, color in (("a.png", (1, 2, 3)), ("b.png", (4, 5, 6)), ("c.png", (7, 8, 9))):
        _png(dataset / "inbox" / name, color)
    import_inbox(dataset)
    rows = [json.loads(line) for line in (dataset / "manifest.draft.jsonl").read_text().splitlines()]
    for row in rows[:2]:
        row.update({"expected": [{"concept_id": "flower"}], "tags": ["broad_only"], "unknown": False, "notes": None})
    (dataset / "manifest.draft.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = promote_draft(dataset, load_taxonomy())
    assert summary["promoted"] == 2 and summary["remaining_drafts"] == 1
    assert len((dataset / "manifest.draft.jsonl").read_text().splitlines()) == 1
    assert promote_draft(dataset, load_taxonomy())["promoted"] == 0


def test_v2_taxonomy_and_mismatch_are_explicit(tmp_path):
    dataset = _dataset(tmp_path)
    set_dataset_taxonomy(dataset, load_evaluation_taxonomy("eden-v2"))
    assert json.loads((dataset / "dataset.json").read_text())["taxonomy_version"] == "eden-v2"
    _png(dataset / "inbox" / "new.png")
    import_inbox(dataset)
    row = json.loads((dataset / "manifest.draft.jsonl").read_text().strip())
    row.update({"expected": [{"concept_id": "shrub"}], "tags": ["single_object"], "unknown": False, "notes": None})
    (dataset / "manifest.draft.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert promote_draft(dataset, load_evaluation_taxonomy("eden-v2"))["promoted"] == 1
    with pytest.raises(ManifestValidationError, match="taxonomy_version"):
        promote_draft(dataset, load_taxonomy())


def test_empty_tags_are_incomplete_and_not_canonical(tmp_path):
    dataset = _dataset(tmp_path)
    _png(dataset / "inbox" / "tagless.png")
    import_inbox(dataset)
    row = json.loads((dataset / "manifest.draft.jsonl").read_text().strip())
    row.update({"expected": [{"concept_id": "flower"}], "unknown": False, "notes": None})
    (dataset / "manifest.draft.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = promote_draft(dataset, load_taxonomy())
    assert result["promoted"] == 0 and result["skipped_incomplete"] == 1


def test_oversized_file_is_rejected_from_stat_before_read(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
    source = dataset / "inbox" / "large.png"
    source.write_bytes(b"0123456789")
    import experiments.eval_dataset as module
    original_open = Path.open

    def forbidden_open(self, *args, **kwargs):
        if self == source:
            raise AssertionError("oversized source was opened")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbidden_open)
    result = module.import_inbox(dataset, max_upload_bytes=1)
    assert result["rejected"] == 1 and result["imported"] == 0


@pytest.mark.parametrize("suffix", [".HEIC", ".heif"])
def test_heif_import_preserves_original_bytes_and_format(tmp_path, suffix):
    dataset = _dataset(tmp_path)
    source = dataset / "inbox" / ("IMG_1234" + suffix)
    _heif(source)
    result = import_inbox(dataset)
    assert result["imported"] == 1
    draft = json.loads((dataset / "manifest.draft.jsonl").read_text().strip())
    target = dataset / draft["file"]
    assert target.suffix == ".heif"
    assert target.read_bytes() == source.read_bytes()
    assert draft["format"] == "HEIF"
