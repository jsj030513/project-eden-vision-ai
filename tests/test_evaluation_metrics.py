import json

import pytest

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.classification.taxonomy import load_taxonomy
from experiments.evaluation_core import (
    ExpectedLabel,
    ManifestValidationError,
    ManifestEntry,
    aggregate_metrics,
    evaluate_image,
    hierarchy_relation,
    load_dataset,
)


@pytest.fixture()
def resolver():
    return HierarchyResolver(load_taxonomy())


def ranked(*ids):
    return [{"concept_id": concept_id, "raw_score": float(len(ids) - index), "score_kind": "raw_image_text_logit"} for index, concept_id in enumerate(ids)]


def test_hierarchy_relations(resolver):
    assert hierarchy_relation("rose", "rose", resolver) == "exact"
    assert hierarchy_relation("rose", "flower", resolver) == "ancestor_distance_1"
    assert hierarchy_relation("rose", "plant", resolver) == "ancestor_distance_2plus"
    assert hierarchy_relation("rose", "car", resolver) == "unrelated"
    assert hierarchy_relation("flower", "rose", resolver) == "over_specific"


def test_exact_parent_and_over_specific_metrics(resolver):
    exact = ManifestEntry("exact", "images/exact.jpg", (ExpectedLabel("rose"),), ("fine_grained",), False)
    result = evaluate_image(exact, ranked("rose", "flower", "plant"), 10.0, resolver)
    assert result["exact"]["top_1"] is True
    assert result["hierarchy_relations"]["rose"] == "exact"
    parent = ManifestEntry("parent", "images/parent.jpg", (ExpectedLabel("rose"),), ("fine_grained",), False)
    parent_result = evaluate_image(parent, ranked("flower", "plant"), 11.0, resolver)
    assert parent_result["hierarchy_relations"]["rose"] == "ancestor_distance_1"
    broad = ManifestEntry("broad", "images/broad.jpg", (ExpectedLabel("flower"),), ("broad_only",), False)
    broad_result = evaluate_image(broad, ranked("rose"), 12.0, resolver)
    assert broad_result["hierarchy_relations"]["flower"] == "over_specific"


def test_multilabel_metrics_and_unknown(resolver):
    entry = ManifestEntry("multi", "images/multi.jpg", (ExpectedLabel("person"), ExpectedLabel("dog"), ExpectedLabel("tree")), ("multi_object",), False)
    result = evaluate_image(entry, ranked("person", "dog", "phone", "tree", "flower"), 8.0, resolver)
    assert result["multi_label"]["top_3"]["precision"] == pytest.approx(2 / 3)
    assert result["multi_label"]["top_5"]["recall"] == pytest.approx(1.0)
    unknown = ManifestEntry("unknown", "images/unknown.jpg", (), ("unknown",), True)
    unknown_result = evaluate_image(unknown, ranked("phone", "car"), 9.0, resolver)
    assert unknown_result["unknown_ground_truth"] is True
    assert unknown_result["exact"]["top_1"] is False


def test_aggregate_metrics_are_deterministic(resolver):
    entry = ManifestEntry("one", "images/one.jpg", (ExpectedLabel("rose"),), ("fine_grained",), False)
    rows = [evaluate_image(entry, ranked("rose", "flower"), 10.0, resolver) for _ in range(2)]
    first, second = aggregate_metrics(rows, resolver), aggregate_metrics(rows, resolver)
    assert first == second
    assert first["top_k_exact"]["top_1"]["rate"] == 1.0


def _write_dataset(tmp_path, lines, *, image_count=None, taxonomy_version="eden-v1"):
    (tmp_path / "images").mkdir(exist_ok=True)
    (tmp_path / "images" / "one.jpg").write_bytes(b"placeholder")
    (tmp_path / "dataset.json").write_text(json.dumps({"dataset_id": "test", "taxonomy_version": taxonomy_version, "image_count": len(lines) if image_count is None else image_count}), encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")


def _valid_line(**overrides):
    value = {"id": "one", "file": "images/one.jpg", "expected": [{"concept_id": "rose"}], "tags": ["fine_grained"], "unknown": False, "notes": None}
    value.update(overrides)
    return value


def test_manifest_validator_accepts_valid_and_unknown(tmp_path):
    taxonomy = load_taxonomy()
    _write_dataset(tmp_path, [_valid_line()])
    assert len(load_dataset(tmp_path, taxonomy).entries) == 1
    _write_dataset(tmp_path, [_valid_line(expected=[], tags=["unknown"], unknown=True)])
    assert load_dataset(tmp_path, taxonomy).entries[0].unknown is True


@pytest.mark.parametrize("lines", [
    [_valid_line(id="one"), _valid_line(id="one")],
    [_valid_line(expected=[{"concept_id": "not_in_taxonomy"}])],
    [_valid_line(file="images/missing.jpg")],
    [_valid_line(expected=[], unknown=False, tags=["fine_grained"])],
    [_valid_line(expected=[], unknown=True, tags=["fine_grained"])],
])
def test_manifest_validator_rejects_invalid_inputs(tmp_path, lines):
    _write_dataset(tmp_path, lines)
    with pytest.raises(ManifestValidationError):
        load_dataset(tmp_path, load_taxonomy())


def test_manifest_validator_rejects_taxonomy_mismatch(tmp_path):
    _write_dataset(tmp_path, [_valid_line()], taxonomy_version="other-v1")
    with pytest.raises(ManifestValidationError, match="taxonomy_version"):
        load_dataset(tmp_path, load_taxonomy())
