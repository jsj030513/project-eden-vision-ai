import json

import pytest

from experiments import sweep_open_set_rules as sweep


def _fixture_files(tmp_path):
    canonical_ids = [f"eval-{number:03d}" for number in range(1, 34)]
    calibration_ids = [f"unknown-candidate-{number:03d}" for number in range(1, 9)]
    images = []
    for index, candidate_id in enumerate(canonical_ids + calibration_ids):
        is_unknown = index < 3 or 33 <= index < 39
        images.append({
            "candidate_id": candidate_id,
            "origin": "canonical" if index < 33 else "calibration_generated",
            "truth_unknown": is_unknown,
            "truth_expected": [] if is_unknown else [{"concept_id": "cat"}],
            "clip": {"top1": "cat", "margin": index / 10, "score": 25 + index / 10, "spread": index / 20},
            "siglip2": {"top1": "dog", "margin": index / 12, "score": -6 + index / 100, "spread": index / 15},
            "cross_model_relation": {"relation": ("unrelated" if index % 3 == 0 else "exact")},
        })
    combined = tmp_path / "combined.json"
    combined.write_text(json.dumps({
        "taxonomy_version": "eden-v2",
        "population": {"total": 41, "known": 32, "unknown": 9,
                       "origins": {"canonical": 33, "calibration_generated": 8}},
        "images": images,
    }))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"images": [
        {"candidate_id": candidate_id, "source_image_id": f"eval-{number:03d}"}
        for number, candidate_id in enumerate(calibration_ids, 24)
    ]}))
    return combined, calibration


def test_population_counts_holdout_exclusion_and_source_grouping(tmp_path):
    combined, calibration = _fixture_files(tmp_path)
    rows, sources = sweep.load_tuning_population(combined, calibration)
    assert len(rows) == 41
    assert sum(row["truth_unknown"] for row in rows) == 9
    assert sum(row["origin"] == "calibration_generated" for row in rows) == 8
    assert sources["unknown-candidate-001"] == "eval-024"
    assert next(row for row in rows if row["candidate_id"] == "unknown-candidate-001")["source_group"] == "eval-024"
    assert sum(row["source_group"] == "eval-024" for row in rows) == 2

    data = json.loads(combined.read_text())
    data["images"].append({"candidate_id": "unknown-candidate-009", "origin": "holdout"})
    combined.write_text(json.dumps(data))
    with pytest.raises(sweep.SweepValidationError):
        sweep.load_tuning_population(combined, calibration)


def test_relation_gate_metrics_and_zero_precision_denominator():
    rows = [
        {"candidate_id": "u1", "truth_unknown": True, "cross_model_relation": {"relation": "unrelated"}},
        {"candidate_id": "u2", "truth_unknown": True, "cross_model_relation": {"relation": "exact"}},
        {"candidate_id": "k1", "truth_unknown": False, "cross_model_relation": {"relation": "unrelated"}},
        {"candidate_id": "k2", "truth_unknown": False, "cross_model_relation": {"relation": "shared_ancestor"}},
    ]
    predictions = sweep.r0_predictions(rows)
    assert predictions == {"u1", "k1"}
    assert not sweep.rule_predictions(
        {"family": "R0", "parameters": {}},
        {"candidate_id": "ancestor", "cross_model_relation": {"relation": "ancestor_descendant"}},
    )
    metrics = sweep.confusion_metrics(rows, predictions)
    assert (metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"]) == (1, 1, 1, 1)
    assert metrics["unknown_recall"] == 0.5
    assert metrics["known_fpr"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert sweep.confusion_metrics(rows, set())["unknown_precision"] is None


def test_fixed_grid_is_deterministic_and_coarse():
    assert sweep.FIXED_GRID == dict(sweep.FIXED_GRID)
    assert sweep.enumerate_rules() == sweep.enumerate_rules()
    assert len(sweep.enumerate_rules()) == 1 + 100 + 100 + 36 + 3600 + 4900


def test_pareto_filtering():
    points = [
        {"rule_id": "a", "metrics": {"unknown_recall": 0.5, "known_fpr": 0.1}},
        {"rule_id": "b", "metrics": {"unknown_recall": 0.8, "known_fpr": 0.1}},
        {"rule_id": "c", "metrics": {"unknown_recall": 0.7, "known_fpr": 0.2}},
        {"rule_id": "d", "metrics": {"unknown_recall": 0.6, "known_fpr": 0.05}},
    ]
    assert {row["rule_id"] for row in sweep.pareto_frontier(points)} == {"b", "d"}


def test_leave_one_source_group_out_uses_fixed_rule_without_retuning():
    rows = [
        {"candidate_id": "u", "truth_unknown": True, "source_group": "g1",
         "cross_model_relation": {"relation": "unrelated"}},
        {"candidate_id": "k", "truth_unknown": False, "source_group": "g2",
         "cross_model_relation": {"relation": "unrelated"}},
    ]
    summary = sweep.leave_one_source_group_out({"family": "R0", "parameters": {}}, rows)
    assert summary["group_count"] == 2
    assert summary["fold_count"] == 2
    assert summary["recall_na_fold_count"] == 1
    assert summary["min_recall"] == 1.0
    assert summary["max_fpr"] == 1.0


def test_report_is_deterministic_and_inputs_are_not_modified(tmp_path):
    combined, calibration = _fixture_files(tmp_path)
    before = (combined.read_bytes(), calibration.read_bytes())
    rows, _ = sweep.load_tuning_population(combined, calibration)
    report1 = sweep.build_sweep_report(rows)
    report2 = sweep.build_sweep_report(rows)
    assert report1 == report2
    assert sweep.render_text(report1) == sweep.render_text(report2)
    assert (combined.read_bytes(), calibration.read_bytes()) == before
