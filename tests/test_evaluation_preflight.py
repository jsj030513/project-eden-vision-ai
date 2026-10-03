import json
import os
from pathlib import Path

import pytest

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.classification.taxonomy import load_taxonomy
from experiments.evaluation_core import ExpectedLabel, ManifestEntry, ManifestValidationError, aggregate_metrics, evaluate_image, load_dataset
from experiments.preflight_evaluation import run


def _base_dataset(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "dataset.json").write_text(json.dumps({"dataset_id": "invalid-test", "taxonomy_version": "eden-v1", "image_count": 1}), encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text("", encoding="utf-8")
    return tmp_path


def _row(file="images/a.png", **overrides):
    value = {"id": "a", "file": file, "sha256": "a" * 64, "expected": [{"concept_id": "rose"}], "tags": ["fine_grained"], "unknown": False, "notes": None}
    value.update(overrides)
    return value


def test_preflight_generates_two_reports_without_winner(tmp_path):
    result = run(tmp_path / "preflight")
    assert result["import"]["imported"] == 10
    assert result["promoted"]["promoted"] == 10
    comparison = json.loads((tmp_path / "preflight/reports/comparison-report.json").read_text())
    assert set(comparison["models"]) == {"fake-model-a", "fake-model-b"}
    assert comparison["winner"] is None
    assert comparison["models"]["fake-model-a"]["metrics"]["hierarchy"]["counts"]["ancestor_distance_1"] == 1
    assert comparison["models"]["fake-model-a"]["metrics"]["multi_label"]["top_5"]["macro_recall"] == pytest.approx(5 / 6)


def test_unknown_only_metrics_use_null_denominators():
    resolver = HierarchyResolver(load_taxonomy())
    entry = ManifestEntry("unknown", "images/u.png", (), ("unknown",), True)
    row = evaluate_image(entry, ["phone", "car"], 1.0, resolver)
    metrics = aggregate_metrics([row], resolver)
    assert metrics["top_k_exact"]["top_1"]["rate"] is None
    assert metrics["multi_label"]["top_3"]["macro_recall"] is None


def test_manifest_rejects_duplicate_hash_and_traversal(tmp_path):
    dataset = _base_dataset(tmp_path)
    (dataset / "images/a.png").write_bytes(b"x")
    rows = [_row(), _row(id="b", file="images/a2.png")]
    (dataset / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    (dataset / "dataset.json").write_text(json.dumps({"dataset_id": "invalid-test", "taxonomy_version": "eden-v1", "image_count": 2}), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="duplicate sha256"):
        load_dataset(dataset, load_taxonomy())
    rows[1]["sha256"] = "b" * 64
    rows[1]["file"] = "../../outside.png"
    (dataset / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="relative file path"):
        load_dataset(dataset, load_taxonomy())


def test_manifest_rejects_symlink_outside_dataset(tmp_path):
    dataset = _base_dataset(tmp_path)
    outside = tmp_path.parent / "outside-preflight.png"
    outside.write_bytes(b"outside")
    link = dataset / "images/link.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    row = _row(file="images/link.png")
    (dataset / "manifest.jsonl").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="does not exist"):
        load_dataset(dataset, load_taxonomy())


def test_aggregate_is_order_independent():
    resolver = HierarchyResolver(load_taxonomy())
    entries = [
        ManifestEntry("one", "images/one.png", (ExpectedLabel("rose"),), ("fine_grained",), False),
        ManifestEntry("two", "images/two.png", (ExpectedLabel("car"),), ("single_object",), False),
    ]
    rows = [evaluate_image(entries[0], ["rose", "flower"], 2.0, resolver), evaluate_image(entries[1], ["phone", "car"], 1.0, resolver)]
    assert aggregate_metrics(rows, resolver) == aggregate_metrics(list(reversed(rows)), resolver)
