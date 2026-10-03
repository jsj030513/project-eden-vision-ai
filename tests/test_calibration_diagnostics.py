import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from eden_vision.classification.hierarchy import HierarchyResolver
from experiments.calibration_diagnostics import (
    CALIBRATION_IDS,
    CalibrationValidationError,
    build_combined_diagnostic,
    calibration_inputs,
    collect_model_signals,
    cross_model_diagnostic,
    run_diagnostics,
    taxonomy_relation,
)
from experiments.model_runtime import load_evaluation_taxonomy


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture_roots(root: Path) -> tuple[Path, Path]:
    candidate = root / "candidate"
    generated = candidate / "generated"
    generated.mkdir(parents=True)
    canonical = root / "canonical"
    reports = canonical / "reports"
    reports.mkdir(parents=True)

    metadata = []
    reviews = []
    for number in range(1, 17):
        candidate_id = f"unknown-candidate-{number:03d}"
        split = "calibration" if number <= 8 else "holdout"
        source_id = f"eval-{number:03d}"
        image_hash = f"{number:064x}"
        row = {
            "candidate_id": candidate_id,
            "source_image_id": source_id,
            "source_sha256": f"{number + 100:064x}",
            "transformation": "fixture_transform",
            "parameters": {"fixture": True},
            "seed": number,
            "generated_sha256": image_hash,
            "intended_split": split,
            "review_status": "pending",
        }
        metadata.append(row)
        if split == "calibration":
            image_path = generated / f"{candidate_id}.jpg"
            Image.new("RGB", (48, 32), (number * 11, number * 7, number * 5)).save(image_path, "JPEG")
            data = image_path.read_bytes()
            row["generated_sha256"] = _sha(data)
            expected = [] if number <= 6 else [{
                "concept_id": "natural_scene" if number == 7 else "outdoor_scene",
                "acceptable_ancestors": ["scene"],
            }]
            unknown = number <= 6
            reviews.append({
                **row,
                "expected": expected,
                "unknown": unknown,
                "tags": ["unknown", "ambiguous"] if unknown else ["ambiguous", "broad_only"],
                "notes": "fixture human review",
                "review_status": "reviewed",
            })
    (candidate / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )
    (candidate / "calibration-review.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reviews), encoding="utf-8"
    )

    def canonical_report(model: str) -> dict:
        images = []
        for number in range(1, 34):
            image_id = f"eval-{number:03d}"
            unknown = number >= 31
            top_ids = ["cat", "dog", "animal", "plant", "flower"]
            scores = [5.0, 4.0, 3.0, 2.0, 1.0]
            images.append({
                "provenance": {"image_id": image_id},
                "ground_truth": {
                    "expected": [] if unknown else [{"concept_id": "cat", "acceptable_ancestors": ["living_thing", "animal"]}],
                    "unknown": unknown,
                    "tags": ["unknown"] if unknown else ["single_object"],
                    "notes": "canonical fixture",
                },
                "ranking": {
                    "top_1": {"concept_id": top_ids[0], "raw_score": scores[0]},
                    "top_2": {"concept_id": top_ids[1], "raw_score": scores[1]},
                    "top1_top2_raw_margin": 1.0,
                    "top_5": [
                        {"concept_id": concept_id, "raw_score": score}
                        for concept_id, score in zip(top_ids, scores, strict=True)
                    ],
                },
                "diagnostics": {
                    "top1_top5_raw_spread": 4.0,
                    "top_k_hierarchy_coherence": {"k": 5, "related_pair_fraction": 0.4, "unique_broad_categories": 2},
                },
            })
        return {"model": model, "taxonomy_version": "eden-v2", "images": images}

    for model, name in (
        ("clip", "clip-eden-v2-signals-33.json"),
        ("siglip2", "siglip2-eden-v2-signals-33.json"),
    ):
        (reports / name).write_text(json.dumps(canonical_report(model)), encoding="utf-8")
    return candidate, canonical


class _FakeRuntime:
    def __init__(self, model: str, calls: list[str]):
        self.model = model
        self.calls = calls
        self.metadata = {"model": model, "device": "mps-test-double"}

    def load(self):
        return None

    def prepare_taxonomy(self, taxonomy):
        return {"candidate_count": len(taxonomy.concepts)}

    def rank(self, image):
        index = len(self.calls)
        self.calls.append(self.model)
        ids = ["cat", "dog", "animal", "plant", "flower"]
        if self.model == "siglip2":
            ids = ["person", "cat", "dog", "animal", "flower"]
        return SimpleNamespace(
            candidates=[{"concept_id": concept_id, "raw_score": 5.0 - rank} for rank, concept_id in enumerate(ids)],
            timings_ms={"total_ms": 1.0},
        )

    def close(self):
        return None


def _fake_factory(calls: list[str]):
    return lambda model: _FakeRuntime(model, calls)


def test_calibration_filter_and_holdout_exclusion(tmp_path):
    candidate, _ = _fixture_roots(tmp_path)
    taxonomy = load_evaluation_taxonomy("eden-v2")
    resolver = HierarchyResolver(taxonomy)
    items = calibration_inputs(candidate, taxonomy, resolver)
    assert tuple(item["candidate_id"] for item in items) == CALIBRATION_IDS
    assert all(item["intended_split"] == "calibration" for item in items)
    assert not any("holdout" in str(item["generated_path"]) for item in items)

    calls: list[str] = []
    report = collect_model_signals(items, "clip", taxonomy, resolver, _fake_factory(calls))
    assert len(report["images"]) == 8
    # The fixture contains no holdout image files. Successful collection proves
    # inference receives only calibration paths.
    assert len(calls) == 8 and set(calls) == {"clip"}


def test_calibration_generated_sha_is_validated(tmp_path):
    candidate, _ = _fixture_roots(tmp_path)
    image = candidate / "generated" / "unknown-candidate-001.jpg"
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(CalibrationValidationError, match="SHA-256 mismatch"):
        calibration_inputs(candidate, load_evaluation_taxonomy("eden-v2"),
                           HierarchyResolver(load_evaluation_taxonomy("eden-v2")))


def test_taxonomy_relations_use_meaningful_shared_ancestor():
    resolver = HierarchyResolver(load_evaluation_taxonomy("eden-v2"))
    assert taxonomy_relation("cat", "cat", resolver)["relation"] == "exact"
    ancestor = taxonomy_relation("cat", "animal", resolver)
    assert ancestor["relation"] == "ancestor_descendant"
    assert ancestor["nearest_common_ancestor"] == "animal"
    shared = taxonomy_relation("cat", "dog", resolver)
    assert shared["relation"] == "shared_ancestor"
    assert shared["meaningful_common_ancestor"] == "animal"
    unrelated = taxonomy_relation("person", "cat", resolver)
    assert unrelated["relation"] == "unrelated"
    assert unrelated["nearest_common_ancestor"] == "living_thing"
    assert unrelated["meaningful_common_ancestor"] is None


def test_cross_model_id_alignment_is_required(tmp_path):
    candidate, _ = _fixture_roots(tmp_path)
    taxonomy = load_evaluation_taxonomy("eden-v2")
    resolver = HierarchyResolver(taxonomy)
    items = calibration_inputs(candidate, taxonomy, resolver)
    calls: list[str] = []
    clip = collect_model_signals(items, "clip", taxonomy, resolver, _fake_factory(calls))
    siglip = collect_model_signals(items, "siglip2", taxonomy, resolver, _fake_factory(calls))
    siglip["images"][0]["candidate_id"] = "unknown-candidate-016"
    with pytest.raises(CalibrationValidationError, match="exactly the eight calibration IDs"):
        cross_model_diagnostic(clip, siglip, resolver)


def test_combined_population_is_41_and_original_reports_remain_unchanged(tmp_path):
    candidate, canonical = _fixture_roots(tmp_path)
    report_files = [canonical / "reports" / name for name in (
        "clip-eden-v2-signals-33.json", "siglip2-eden-v2-signals-33.json"
    )]
    report_bytes = [path.read_bytes() for path in report_files]
    taxonomy = load_evaluation_taxonomy("eden-v2")
    resolver = HierarchyResolver(taxonomy)
    calls: list[str] = []
    result = run_diagnostics(candidate, canonical, runtime_factory=_fake_factory(calls))
    combined = result["combined"]
    assert combined["population"] == {
        "total": 41,
        "known": 32,
        "unknown": 9,
        "origins": {"canonical": 33, "calibration_generated": 8},
    }
    assert Counter(row["origin"] for row in combined["images"]) == {
        "canonical": 33,
        "calibration_generated": 8,
    }
    assert sum(row["truth_unknown"] for row in combined["images"]) == 9
    assert [path.read_bytes() for path in report_files] == report_bytes
    assert len(result["clip"]["images"]) == len(result["siglip2"]["images"]) == 8
    assert set(calls) == {"clip", "siglip2"} and len(calls) == 16
    for path in result["paths"].values():
        assert path.is_file()


def test_calibration_cross_report_uses_all_eight_ids(tmp_path):
    candidate, _ = _fixture_roots(tmp_path)
    taxonomy = load_evaluation_taxonomy("eden-v2")
    resolver = HierarchyResolver(taxonomy)
    items = calibration_inputs(candidate, taxonomy, resolver)
    calls: list[str] = []
    clip = collect_model_signals(items, "clip", taxonomy, resolver, _fake_factory(calls))
    siglip = collect_model_signals(items, "siglip2", taxonomy, resolver, _fake_factory(calls))
    cross = cross_model_diagnostic(clip, siglip, resolver)
    assert {row["candidate_id"] for row in cross["images"]} == set(CALIBRATION_IDS)
    assert all(row["cross_model_relation"]["relation"] == "unrelated" for row in cross["images"])
