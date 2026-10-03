import hashlib
import json
from pathlib import Path

import pytest


PREREGISTRATION = Path(
    "/Users/dev/Datasets/eden-vision-open-set-candidates/reports/open-set-rule-preregistration.json"
)
FINGERPRINT = PREREGISTRATION.with_suffix(".sha256")


@pytest.fixture(scope="module")
def preregistration():
    if not PREREGISTRATION.is_file() or not FINGERPRINT.is_file():
        pytest.skip("the external open-set preregistration artifact is not present")
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


def test_primary_rule_expression_and_thresholds_are_frozen(preregistration):
    rule = preregistration["primary_rule"]
    assert rule["name"] == "A-0014-primary"
    assert rule["expression"] == (
        'relation == "unrelated" AND (clip_margin <= 0.10 OR siglip2_margin <= 0.50)'
    )
    assert rule["boolean_expression"] == {
        "and": [
            {"eq": ["relation", "unrelated"]},
            {"or": [
                {"lte": ["clip_margin", 0.10]},
                {"lte": ["siglip2_margin", 0.50]},
            ]},
        ],
    }
    assert rule["thresholds"] == {"clip_margin_lte": 0.10, "siglip2_margin_lte": 0.50}
    assert rule["tuning_metrics"] == {
        "tp": 8, "fn": 1, "fp": 1, "tn": 31,
        "unknown_recall": 8 / 9, "known_fpr": 1 / 32,
        "population": 41, "known": 32, "unknown": 9,
    }


def test_challenger_rule_expression_and_thresholds_are_frozen(preregistration):
    rule = preregistration["challenger_rule"]
    assert rule["name"] == "E-4086-high-recall-challenger"
    assert rule["expression"] == (
        'relation == "unrelated" AND ((clip_margin <= 3.00 AND '
        'clip_top1_top5_spread <= 1.00) OR (siglip2_margin <= 0.50 AND '
        'siglip2_top1_top5_spread <= 4.00))'
    )
    assert rule["boolean_expression"] == {
        "and": [
            {"eq": ["relation", "unrelated"]},
            {"or": [
                {"and": [
                    {"lte": ["clip_margin", 3.00]},
                    {"lte": ["clip_top1_top5_spread", 1.00]},
                ]},
                {"and": [
                    {"lte": ["siglip2_margin", 0.50]},
                    {"lte": ["siglip2_top1_top5_spread", 4.00]},
                ]},
            ]},
        ],
    }
    assert rule["thresholds"] == {
        "clip_margin_lte": 3.00,
        "clip_top1_top5_spread_lte": 1.00,
        "siglip2_margin_lte": 0.50,
        "siglip2_top1_top5_spread_lte": 4.00,
    }
    assert rule["tuning_metrics"] == {
        "tp": 9, "fn": 0, "fp": 2, "tn": 30,
        "unknown_recall": 1.0, "known_fpr": 2 / 32,
        "population": 41, "known": 32, "unknown": 9,
    }


def test_freeze_state_population_and_relation_semantics(preregistration):
    assert preregistration["status"] == "frozen_before_holdout"
    assert preregistration["rules_frozen"] is True
    assert preregistration["holdout_seen"] is False
    assert preregistration["holdout_inference_run"] is False
    assert preregistration["taxonomy_version"] == "eden-v2"
    assert (preregistration["tuning_population_count"], preregistration["tuning_known_count"],
            preregistration["tuning_unknown_count"]) == (41, 32, 9)
    assert preregistration["relation_semantics"]["root_only_common_ancestor_is_unrelated"] is True
    assert preregistration["relation_semantics"]["frozen_through_holdout_evaluation"] is True
    assert "MUST NOT be changed after holdout inspection" in preregistration["freeze_notice"]


def test_source_reports_and_code_hashes_are_present_and_match(preregistration):
    sources = preregistration["source_tuning_reports"]
    assert {
        "calibration-clip-signals.json", "calibration-siglip2-signals.json",
        "calibration-cross-model.json", "calibration-combined-diagnostic.json",
        "open-set-rule-sweep.json", "open-set-rule-sweep.txt",
        "clip-eden-v2-signals-33.json", "siglip2-eden-v2-signals-33.json",
    } <= set(sources)
    for source in sources.values():
        path = Path(source["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    for relative, digest in preregistration["source_code_sha256"].items():
        path = Path("/Users/dev/Developer/eden-vision-ai/experiments") / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_preregistration_fingerprint_is_canonical_and_reproducible():
    raw = PREREGISTRATION.read_bytes()
    data = json.loads(raw)
    canonical = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2,
                            allow_nan=False) + "\n").encode("utf-8")
    assert raw == canonical
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == FINGERPRINT.read_text(encoding="ascii").strip()
    assert hashlib.sha256(canonical).hexdigest() == digest
