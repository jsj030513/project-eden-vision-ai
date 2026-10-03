"""Compare a small, fixed family of interpretable UNKNOWN rules on tuning data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from experiments.evaluation_core import _percentile


DEFAULT_CANDIDATE_ROOT = Path("/Users/dev/Datasets/eden-vision-open-set-candidates")
DEFAULT_COMBINED = DEFAULT_CANDIDATE_ROOT / "reports/calibration-combined-diagnostic.json"
DEFAULT_CALIBRATION = DEFAULT_CANDIDATE_ROOT / "reports/calibration-clip-signals.json"
DEFAULT_REPORT_DIR = DEFAULT_CANDIDATE_ROOT / "reports"
OUTPUT_JSON = "open-set-rule-sweep.json"
OUTPUT_TEXT = "open-set-rule-sweep.txt"
KNOWN_CASE_IDS = ("eval-002", "eval-018", "eval-026", "eval-027", "unknown-candidate-008")
RELATIONS = ("exact", "ancestor_descendant", "shared_ancestor", "unrelated")

# Fixed before rule evaluation. These coarse grids cover the observed tuning
# ranges while avoiding thresholds placed on individual sample values.
FIXED_GRID = {
    "clip_margin": (0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00),
    "siglip2_margin": (0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00),
    "clip_raw_score": (22.0, 24.0, 26.0, 28.0, 30.0, 32.0),
    "siglip2_raw_score": (-8.0, -7.0, -6.0, -5.0, -4.0, -3.0),
    "clip_spread": (0.25, 0.50, 1.0, 2.0, 3.0, 4.0, 5.5),
    "siglip2_spread": (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0),
}


class SweepValidationError(ValueError):
    """The tuning diagnostic is malformed or contains out-of-scope rows."""


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise SweepValidationError(f"expected JSON object: {path}")
    return result


def load_tuning_population(combined_path: Path, calibration_signals_path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load only the 41-row tuning report and source mapping for its 8 calibration rows."""
    combined = _json(combined_path)
    calibration = _json(calibration_signals_path)
    if combined.get("taxonomy_version") != "eden-v2":
        raise SweepValidationError("tuning report taxonomy must be eden-v2")
    rows = combined.get("images")
    if not isinstance(rows, list) or len(rows) != 41:
        raise SweepValidationError("tuning population must contain exactly 41 rows")
    ids = [row.get("candidate_id") for row in rows]
    if len(set(ids)) != 41:
        raise SweepValidationError("tuning population contains missing or duplicate IDs")
    canonical_ids = {f"eval-{number:03d}" for number in range(1, 34)}
    calibration_ids = {f"unknown-candidate-{number:03d}" for number in range(1, 9)}
    if set(ids) != canonical_ids | calibration_ids:
        raise SweepValidationError("tuning input must be exactly canonical 33 plus reviewed calibration 8")

    origins = Counter(row.get("origin") for row in rows)
    truths = Counter("unknown" if row.get("truth_unknown") is True else "known"
                     for row in rows if type(row.get("truth_unknown")) is bool)
    if origins != {"canonical": 33, "calibration_generated": 8}:
        raise SweepValidationError(f"invalid tuning origins: {origins}")
    if truths != {"known": 32, "unknown": 9}:
        raise SweepValidationError(f"invalid tuning truth counts: {truths}")
    if combined.get("population") != {
        "total": 41,
        "known": 32,
        "unknown": 9,
        "origins": {"canonical": 33, "calibration_generated": 8},
    }:
        raise SweepValidationError("tuning report population summary does not match its rows")

    calibration_images = calibration.get("images")
    if not isinstance(calibration_images, list) or len(calibration_images) != 8:
        raise SweepValidationError("calibration source report must contain exactly eight images")
    source_map = {row.get("candidate_id"): row.get("source_image_id") for row in calibration_images}
    if set(source_map) != calibration_ids or any(not isinstance(value, str) for value in source_map.values()):
        raise SweepValidationError("calibration source map must contain exactly the eight calibration IDs")

    checked = []
    for row in rows:
        if row["origin"] not in {"canonical", "calibration_generated"}:
            raise SweepValidationError("holdout or unsupported origin in tuning population")
        if type(row["truth_unknown"]) is not bool:
            raise SweepValidationError(f"invalid truth label: {row['candidate_id']}")
        relation = row.get("cross_model_relation", {}).get("relation")
        if relation not in RELATIONS:
            raise SweepValidationError(f"invalid taxonomy relation: {row['candidate_id']}")
        for model in ("clip", "siglip2"):
            features = row.get(model)
            if not isinstance(features, dict):
                raise SweepValidationError(f"missing {model} features: {row['candidate_id']}")
            for key in ("margin", "score", "spread"):
                value = features.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise SweepValidationError(f"invalid {model}.{key}: {row['candidate_id']}")
            if not isinstance(features.get("top1"), str):
                raise SweepValidationError(f"missing {model} top1: {row['candidate_id']}")
        row_copy = dict(row)
        if row["origin"] == "calibration_generated":
            row_copy["source_group"] = source_map[row["candidate_id"]]
        else:
            row_copy["source_group"] = row["candidate_id"]
        checked.append(row_copy)
    return checked, source_map


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def confusion_metrics(rows: list[dict[str, Any]], predictions: set[str]) -> dict[str, Any]:
    tp = sum(row["truth_unknown"] and row["candidate_id"] in predictions for row in rows)
    fn = sum(row["truth_unknown"] and row["candidate_id"] not in predictions for row in rows)
    fp = sum((not row["truth_unknown"]) and row["candidate_id"] in predictions for row in rows)
    tn = sum((not row["truth_unknown"]) and row["candidate_id"] not in predictions for row in rows)
    recall = _divide(tp, tp + fn)
    precision = _divide(tp, tp + fp)
    fpr = _divide(fp, fp + tn)
    specificity = _divide(tn, tn + fp)
    balanced = None if recall is None or specificity is None else (recall + specificity) / 2
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "unknown_recall": recall,
        "unknown_precision": precision,
        "known_fpr": fpr,
        "specificity": specificity,
        "balanced_accuracy": balanced,
    }


def r0_predictions(rows: list[dict[str, Any]]) -> set[str]:
    return {
        row["candidate_id"] for row in rows
        if row["cross_model_relation"]["relation"] == "unrelated"
    }


def rule_predictions(rule: dict[str, Any], row: dict[str, Any]) -> bool:
    if row["cross_model_relation"]["relation"] != "unrelated":
        return False
    p = rule["parameters"]
    family = rule["family"]
    if family == "R0":
        return True
    if family == "A_margin_or":
        return row["clip"]["margin"] <= p["clip_margin"] or row["siglip2"]["margin"] <= p["siglip2_margin"]
    if family == "B_margin_and":
        return row["clip"]["margin"] <= p["clip_margin"] and row["siglip2"]["margin"] <= p["siglip2_margin"]
    if family == "C_score_or":
        return row["clip"]["score"] <= p["clip_raw_score"] or row["siglip2"]["score"] <= p["siglip2_raw_score"]
    if family == "D_margin_score_confirm":
        return (
            row["clip"]["margin"] <= p["clip_margin"] and row["clip"]["score"] <= p["clip_raw_score"]
        ) or (
            row["siglip2"]["margin"] <= p["siglip2_margin"] and row["siglip2"]["score"] <= p["siglip2_raw_score"]
        )
    if family == "E_margin_spread_confirm":
        return (
            row["clip"]["margin"] <= p["clip_margin"] and row["clip"]["spread"] <= p["clip_spread"]
        ) or (
            row["siglip2"]["margin"] <= p["siglip2_margin"] and row["siglip2"]["spread"] <= p["siglip2_spread"]
        )
    raise ValueError(f"unknown rule family: {family}")


def _parameter_grid(family: str):
    if family in {"A_margin_or", "B_margin_and"}:
        keys = ("clip_margin", "siglip2_margin")
    elif family == "C_score_or":
        keys = ("clip_raw_score", "siglip2_raw_score")
    elif family == "D_margin_score_confirm":
        keys = ("clip_margin", "clip_raw_score", "siglip2_margin", "siglip2_raw_score")
    elif family == "E_margin_spread_confirm":
        keys = ("clip_margin", "clip_spread", "siglip2_margin", "siglip2_spread")
    else:
        raise ValueError(f"unknown family: {family}")
    for values in itertools.product(*(FIXED_GRID[key] for key in keys)):
        yield dict(zip(keys, values, strict=True))


def _rule_priority(rule: dict[str, Any]) -> tuple[Any, ...]:
    params = rule["parameters"]
    edge_count = 0
    center_distance = 0.0
    for name, value in params.items():
        grid = FIXED_GRID[name]
        edge_count += int(value == grid[0] or value == grid[-1])
        center = (grid[0] + grid[-1]) / 2
        radius = max(abs(grid[0] - center), abs(grid[-1] - center)) or 1.0
        center_distance += abs(value - center) / radius
    return (rule["complexity"], edge_count, center_distance, rule["rule_id"])


def enumerate_rules() -> list[dict[str, Any]]:
    rules = [{"rule_id": "R0", "family": "R0", "parameters": {}, "complexity": 1}]
    families = (
        "A_margin_or", "B_margin_and", "C_score_or",
        "D_margin_score_confirm", "E_margin_spread_confirm",
    )
    for family in families:
        complexity = 2 if family in {"A_margin_or", "B_margin_and", "C_score_or"} else 4
        for index, parameters in enumerate(_parameter_grid(family), 1):
            rules.append({
                "rule_id": f"{family[0]}-{index:04d}",
                "family": family,
                "parameters": parameters,
                "complexity": complexity,
            })
    return rules


def evaluate_rule(rule: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = {row["candidate_id"] for row in rows if rule_predictions(rule, row)}
    return confusion_metrics(rows, predictions)


def pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nondominated rules for high UNKNOWN recall and low known FPR."""
    valid = [row for row in candidates if row["metrics"]["unknown_recall"] is not None and row["metrics"]["known_fpr"] is not None]
    frontier = []
    for row in valid:
        recall = row["metrics"]["unknown_recall"]
        fpr = row["metrics"]["known_fpr"]
        dominated = any(
            other is not row
            and other["metrics"]["unknown_recall"] >= recall
            and other["metrics"]["known_fpr"] <= fpr
            and (other["metrics"]["unknown_recall"] > recall or other["metrics"]["known_fpr"] < fpr)
            for other in valid
        )
        if not dominated:
            frontier.append(row)
    return frontier


def _representative_by_metric(frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[float, float], dict[str, Any]] = {}
    for rule in frontier:
        point = (rule["metrics"]["unknown_recall"], rule["metrics"]["known_fpr"])
        if point not in best or _rule_priority(rule) < _rule_priority(best[point]):
            best[point] = rule
    return sorted(best.values(), key=lambda rule: (
        rule["metrics"]["known_fpr"], -rule["metrics"]["unknown_recall"], _rule_priority(rule)
    ))


def _select(candidates: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool],
            objective: Callable[[dict[str, Any]], tuple[Any, ...]]) -> dict[str, Any] | None:
    eligible = [row for row in candidates if predicate(row)]
    return min(eligible, key=lambda row: (*objective(row), *_rule_priority(row))) if eligible else None


def _with_decisions(rule: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    predicted = {row["candidate_id"] for row in rows if rule_predictions(rule, row)}
    false_positives = [row for row in rows if not row["truth_unknown"] and row["candidate_id"] in predicted]
    false_negatives = [row for row in rows if row["truth_unknown"] and row["candidate_id"] not in predicted]
    return predicted, false_positives, false_negatives


def _decision_detail(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["candidate_id"],
        "origin": row["origin"],
        "truth_expected": row.get("truth_expected", []),
        "clip_top1": row["clip"]["top1"],
        "clip_margin": row["clip"]["margin"],
        "clip_raw_score": row["clip"]["score"],
        "clip_spread": row["clip"]["spread"],
        "siglip2_top1": row["siglip2"]["top1"],
        "siglip2_margin": row["siglip2"]["margin"],
        "siglip2_raw_score": row["siglip2"]["score"],
        "siglip2_spread": row["siglip2"]["spread"],
        "relation": row["cross_model_relation"]["relation"],
    }


def leave_one_source_group_out(rule: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({row["source_group"] for row in rows})
    folds = []
    for group in groups:
        subset = [row for row in rows if row["source_group"] != group]
        metrics = evaluate_rule(rule, subset)
        folds.append({"excluded_source_group": group, "metrics": metrics})
    recall_values = [fold["metrics"]["unknown_recall"] for fold in folds if fold["metrics"]["unknown_recall"] is not None]
    fpr_values = [fold["metrics"]["known_fpr"] for fold in folds if fold["metrics"]["known_fpr"] is not None]
    return {
        "group_count": len(groups),
        "fold_count": len(folds),
        "recall_na_fold_count": len(folds) - len(recall_values),
        "min_recall": min(recall_values, default=None),
        "median_recall": _percentile(recall_values, 0.5),
        "max_fpr": max(fpr_values, default=None),
        "median_fpr": _percentile(fpr_values, 0.5),
        "folds": folds,
    }


def _rule_public(rule: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": rule["rule_id"],
        "family": rule["family"],
        "parameters": rule["parameters"],
        "complexity": rule["complexity"],
        "metrics": metrics,
    }


def build_sweep_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 41:
        raise SweepValidationError("rule sweep requires 41 tuning rows")
    truth_counts = Counter("unknown" if row["truth_unknown"] else "known" for row in rows)
    origin_counts = Counter(row["origin"] for row in rows)
    if truth_counts != {"known": 32, "unknown": 9} or origin_counts != {"canonical": 33, "calibration_generated": 8}:
        raise SweepValidationError("rule sweep population must be canonical 33 + calibration 8, known 32 + UNKNOWN 9")

    relations = {
        relation: {
            "known": sum(not row["truth_unknown"] and row["cross_model_relation"]["relation"] == relation for row in rows),
            "unknown": sum(row["truth_unknown"] and row["cross_model_relation"]["relation"] == relation for row in rows),
        }
        for relation in RELATIONS
    }
    feature_ranges = {
        key: {"min": min(float(row[model][field]) for row in rows),
              "max": max(float(row[model][field]) for row in rows)}
        for key, model, field in (
            ("clip_margin", "clip", "margin"),
            ("siglip2_margin", "siglip2", "margin"),
            ("clip_raw_score", "clip", "score"),
            ("siglip2_raw_score", "siglip2", "score"),
            ("clip_spread", "clip", "spread"),
            ("siglip2_spread", "siglip2", "spread"),
        )
    }

    rules = enumerate_rules()
    rule_results = [_rule_public(rule, evaluate_rule(rule, rows)) for rule in rules]

    def internal_rule(rule_id: str) -> dict[str, Any]:
        return next(rule for rule in rules if rule["rule_id"] == rule_id)

    by_id = {row["rule_id"]: row for row in rule_results}
    r0 = by_id["R0"]

    frontiers = {}
    family_representatives = {}
    for family in ("A_margin_or", "B_margin_and", "C_score_or", "D_margin_score_confirm", "E_margin_spread_confirm"):
        family_frontier = _representative_by_metric(pareto_frontier([r for r in rule_results if r["family"] == family]))
        frontiers[family] = family_frontier
        family_representatives[family] = _select(
            family_frontier, lambda _: True,
            lambda row: (-row["metrics"]["unknown_recall"], row["metrics"]["known_fpr"]),
        )

    global_frontier = _representative_by_metric(pareto_frontier(rule_results))
    best_fpr_5 = _select(
        rule_results,
        lambda row: row["metrics"]["known_fpr"] <= 0.05,
        lambda row: (-row["metrics"]["unknown_recall"], row["metrics"]["known_fpr"]),
    )
    best_fpr_10 = _select(
        rule_results,
        lambda row: row["metrics"]["known_fpr"] <= 0.10,
        lambda row: (-row["metrics"]["unknown_recall"], row["metrics"]["known_fpr"]),
    )
    best_recall_80 = _select(
        rule_results,
        lambda row: row["metrics"]["unknown_recall"] >= 0.80,
        lambda row: (row["metrics"]["known_fpr"], -row["metrics"]["unknown_recall"]),
    )
    best_recall_100 = _select(
        rule_results,
        lambda row: row["metrics"]["unknown_recall"] == 1.0,
        lambda row: (row["metrics"]["known_fpr"], -row["metrics"]["unknown_recall"]),
    )

    shortlist = global_frontier[:10]
    detailed = []
    for public_rule in shortlist:
        rule = internal_rule(public_rule["rule_id"])
        predicted, fp, fn = _with_decisions(rule, rows)
        detailed.append({
            **public_rule,
            "false_positives": [_decision_detail(row) for row in fp],
            "false_negatives": [_decision_detail(row) for row in fn],
            "known_case_rejections": {candidate_id: candidate_id in predicted for candidate_id in KNOWN_CASE_IDS},
            "source_group_robustness": leave_one_source_group_out(rule, rows),
        })

    def selected(rule: dict[str, Any] | None) -> dict[str, Any] | None:
        if rule is None:
            return None
        inner = internal_rule(rule["rule_id"])
        predicted, fp, fn = _with_decisions(inner, rows)
        return {
            **rule,
            "false_positives": [_decision_detail(row) for row in fp],
            "false_negatives": [_decision_detail(row) for row in fn],
            "known_case_rejections": {candidate_id: candidate_id in predicted for candidate_id in KNOWN_CASE_IDS},
            "source_group_robustness": leave_one_source_group_out(inner, rows),
        }

    return {
        "schema_version": "1.0",
        "kind": "interpretable_open_set_rule_sweep",
        "threshold_selection": "none; comparison only",
        "tuning_population": {"total": 41, "known": 32, "unknown": 9, "origins": dict(origin_counts)},
        "holdout_used": False,
        "model_raw_scores_comparable": False,
        "feature_ranges": feature_ranges,
        "fixed_threshold_grid": {key: list(values) for key, values in FIXED_GRID.items()},
        "threshold_grid_note": "Coarse fixed grids chosen to span the observed tuning feature ranges; no sample-value-specific thresholds.",
        "r0_baseline": r0,
        "relation_group_counts": relations,
        "rule_family_count": len(rules),
        "rules": rule_results,
        "family_frontiers": frontiers,
        "family_representatives": {family: selected(rule) for family, rule in family_representatives.items()},
        "pareto_frontier": global_frontier,
        "shortlist_max_10": detailed,
        "selection_summaries": {
            "best_recall_at_fpr_le_5_percent": selected(best_fpr_5),
            "best_recall_at_fpr_le_10_percent": selected(best_fpr_10),
            "lowest_fpr_at_recall_ge_80_percent": selected(best_recall_80),
            "lowest_fpr_at_recall_100_percent": selected(best_recall_100),
        },
        "interpretation_note": "No candidate is designated as a final production rule. Thresholds require human review and separate holdout evaluation later.",
    }


def _format_rule_line(rule: dict[str, Any]) -> str:
    metrics = rule["metrics"]
    return (
        f"{rule['rule_id']} {rule['family']} params={rule['parameters']} "
        f"TP/FN/FP/TN={metrics['tp']}/{metrics['fn']}/{metrics['fp']}/{metrics['tn']} "
        f"recall={metrics['unknown_recall']:.3f} precision={metrics['unknown_precision']} "
        f"FPR={metrics['known_fpr']:.3f} specificity={metrics['specificity']:.3f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.3f}"
    )


def _format_error(row: dict[str, Any]) -> str:
    return (
        f"{row['id']} ({row['origin']}) expected={row['truth_expected']} "
        f"CLIP={row['clip_top1']}/m{row['clip_margin']:.3f}/s{row['clip_raw_score']:.3f}/sp{row['clip_spread']:.3f} "
        f"SigLIP2={row['siglip2_top1']}/m{row['siglip2_margin']:.3f}/s{row['siglip2_raw_score']:.3f}/sp{row['siglip2_spread']:.3f} "
        f"relation={row['relation']}"
    )


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "Interpretable UNKNOWN rule sweep (tuning data only)",
        "Population: total=41, known=32, UNKNOWN=9; holdout_used=false",
        "No threshold is selected as a final rule.",
        "",
        "R0 baseline: unrelated => UNKNOWN",
        _format_rule_line(report["r0_baseline"]),
        "Relation group counts (known / UNKNOWN):",
    ]
    for relation, counts in report["relation_group_counts"].items():
        lines.append(f"  {relation}: {counts['known']} / {counts['unknown']}")
    lines.append("Fixed coarse threshold grid:")
    for name, values in report["fixed_threshold_grid"].items():
        lines.append(f"  {name}: {values}")
    lines.append("Feature ranges:")
    for name, values in report["feature_ranges"].items():
        lines.append(f"  {name}: {values['min']:.6g} .. {values['max']:.6g}")
    lines.extend(["", "Family Pareto representatives (comparison candidates only):"])
    for family, rule in report["family_representatives"].items():
        lines.append(f"  {family}: " + ("N/A" if rule is None else _format_rule_line(rule)))
    lines.extend(["", "Requested operating-region candidates:"])
    for name, rule in report["selection_summaries"].items():
        lines.append(f"  {name}: " + ("N/A" if rule is None else _format_rule_line(rule)))
    lines.extend(["", "Top Pareto shortlist source-group robustness and errors (max 10):"])
    for rule in report["shortlist_max_10"]:
        robust = rule["source_group_robustness"]
        lines.append(
            f"  {rule['rule_id']} ({rule['family']}): recall {rule['metrics']['unknown_recall']:.3f}, "
            f"FPR {rule['metrics']['known_fpr']:.3f}; LOGO recall min/median={robust['min_recall']}/{robust['median_recall']}, "
            f"FPR max/median={robust['max_fpr']}/{robust['median_fpr']}"
        )
        lines.append("    FP:")
        lines.extend("      " + _format_error(row) for row in rule["false_positives"])
        lines.append("    FN:")
        lines.extend("      " + _format_error(row) for row in rule["false_negatives"])
        lines.append("    Named known cases rejected: " + ", ".join(
            f"{candidate_id}={value}" for candidate_id, value in rule["known_case_rejections"].items()
        ))
    return "\n".join(lines) + "\n"


def publish(report: dict[str, Any], report_dir: Path) -> tuple[Path, Path]:
    report_dir = Path(report_dir).expanduser().absolute()
    json_path, text_path = report_dir / OUTPUT_JSON, report_dir / OUTPUT_TEXT
    if any(path.exists() or path.is_symlink() for path in (json_path, text_path)):
        raise FileExistsError("refusing to overwrite an existing rule sweep report")
    report_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rule-sweep.", dir=report_dir))
    created = []
    try:
        (staging / OUTPUT_JSON).write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / OUTPUT_TEXT).write_text(render_text(report), encoding="utf-8")
        for source, target in ((staging / OUTPUT_JSON, json_path), (staging / OUTPUT_TEXT, text_path)):
            os.link(source, target)
            created.append(target)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return json_path, text_path


def run_sweep(combined_path: Path = DEFAULT_COMBINED,
              calibration_signals_path: Path = DEFAULT_CALIBRATION,
              report_dir: Path = DEFAULT_REPORT_DIR) -> dict[str, Any]:
    rows, _source_map = load_tuning_population(combined_path, calibration_signals_path)
    report = build_sweep_report(rows)
    paths = publish(report, report_dir)
    return {"report": report, "paths": paths}


def _brief(report: dict[str, Any]) -> str:
    lines = ["R0 baseline:", _format_rule_line(report["r0_baseline"]), "Family Pareto representatives:"]
    for family, row in report["family_representatives"].items():
        lines.append(f"  {family}: " + ("N/A" if row is None else _format_rule_line(row)))
    lines.append("Operating-region candidates:")
    for name, row in report["selection_summaries"].items():
        lines.append(f"  {name}: " + ("N/A" if row is None else _format_rule_line(row)))
    lines.append("Top Pareto source-group robustness and FP/FN:")
    for row in report["shortlist_max_10"]:
        robust = row["source_group_robustness"]
        lines.append(
            f"  {row['rule_id']} ({row['family']}): recall={row['metrics']['unknown_recall']:.3f}, "
            f"FPR={row['metrics']['known_fpr']:.3f}, LOGO recall min/median={robust['min_recall']}/{robust['median_recall']}, "
            f"FPR max/median={robust['max_fpr']}/{robust['median_fpr']}"
        )
        lines.append("    FP: " + ("; ".join(_format_error(error) for error in row["false_positives"]) or "none"))
        lines.append("    FN: " + ("; ".join(_format_error(error) for error in row["false_negatives"]) or "none"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--calibration-signals", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args(argv)
    try:
        result = run_sweep(args.combined, args.calibration_signals, args.report_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"rule sweep failed: {exc}\n")
    print(f"Wrote {result['paths'][0]} and {result['paths'][1]}")
    print(_brief(result["report"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
