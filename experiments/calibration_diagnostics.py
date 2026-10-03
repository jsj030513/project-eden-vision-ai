"""Collect raw CLIP/SigLIP2 diagnostics for reviewed calibration candidates only."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eden_vision.classification.hierarchy import HierarchyResolver
from experiments.evaluation_core import ALLOWED_TAGS, _coherence, _percentile
from experiments.model_runtime import (
    MODELS,
    PROMPT_TEMPLATE,
    EvaluationModelRuntime,
    create_runtime,
    load_evaluation_taxonomy,
)
from eden_vision.services.image_input import ImageInputService


CANDIDATE_DEFAULT = Path("/Users/dev/Datasets/eden-vision-open-set-candidates")
CANONICAL_DEFAULT = Path("/Users/dev/Datasets/eden-vision-open-set-eval")
CALIBRATION_IDS = tuple(f"unknown-candidate-{index:03d}" for index in range(1, 9))
CANONICAL_REPORT_NAMES = {
    "clip": "clip-eden-v2-signals-33.json",
    "siglip2": "siglip2-eden-v2-signals-33.json",
}
OUTPUT_NAMES = {
    "clip": "calibration-clip-signals.json",
    "siglip2": "calibration-siglip2-signals.json",
    "cross": "calibration-cross-model.json",
    "combined": "calibration-combined-diagnostic.json",
}
SCORE_KIND = "raw_image_text_logit"


class CalibrationValidationError(ValueError):
    """Calibration inputs or reports do not satisfy the diagnostic contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CalibrationValidationError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationValidationError(f"expected JSON object: {path}")
    return value


def calibration_inputs(candidate_root: Path, taxonomy: Any,
                       resolver: HierarchyResolver) -> list[dict[str, Any]]:
    """Validate all metadata, but return/resolve generated paths for calibration rows only."""
    candidate_root = Path(candidate_root).expanduser().resolve()
    metadata_rows = _read_jsonl(candidate_root / "metadata.jsonl")
    review_rows = _read_jsonl(candidate_root / "calibration-review.jsonl")
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in metadata_by_id:
            raise CalibrationValidationError("candidate metadata has missing or duplicate IDs")
        if row.get("intended_split") not in {"calibration", "holdout"}:
            raise CalibrationValidationError(f"invalid split for {candidate_id}")
        metadata_by_id[candidate_id] = row

    calibration_meta = {
        row["candidate_id"]: row
        for row in metadata_rows
        if row["intended_split"] == "calibration"
    }
    holdout_ids = {
        row["candidate_id"]
        for row in metadata_rows
        if row["intended_split"] == "holdout"
    }
    if len(calibration_meta) != 8 or set(calibration_meta) != set(CALIBRATION_IDS):
        raise CalibrationValidationError("expected exactly the eight calibration candidates")
    if len(holdout_ids) != 8 or holdout_ids & set(CALIBRATION_IDS):
        raise CalibrationValidationError("candidate metadata split is invalid")

    review_by_id: dict[str, dict[str, Any]] = {}
    for row in review_rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in review_by_id:
            raise CalibrationValidationError("calibration review has missing or duplicate IDs")
        review_by_id[candidate_id] = row
    if len(review_by_id) != 8 or set(review_by_id) != set(calibration_meta):
        raise CalibrationValidationError("calibration review IDs must equal calibration metadata IDs")
    if set(review_by_id) & holdout_ids:
        raise CalibrationValidationError("holdout candidate found in review inputs")

    concept_ids = {concept.concept_id for concept in taxonomy.concepts}
    generated_root = candidate_root / "generated"
    checked: list[dict[str, Any]] = []
    for candidate_id in CALIBRATION_IDS:
        source = calibration_meta[candidate_id]
        review = review_by_id[candidate_id]
        for key in ("source_image_id", "source_sha256", "transformation", "generated_sha256", "seed"):
            if source.get(key) != review.get(key):
                raise CalibrationValidationError(f"review provenance mismatch for {candidate_id}: {key}")
        if review.get("intended_split") != "calibration" or review.get("review_status") != "reviewed":
            raise CalibrationValidationError(f"unreviewed or non-calibration item: {candidate_id}")

        image_path = generated_root / f"{candidate_id}.jpg"
        if image_path.parent != generated_root or not image_path.is_file():
            raise CalibrationValidationError(f"calibration image missing: {candidate_id}")
        image_bytes = image_path.read_bytes()
        if sha256_bytes(image_bytes) != source.get("generated_sha256"):
            raise CalibrationValidationError(f"generated SHA-256 mismatch: {candidate_id}")

        expected = review.get("expected")
        tags = review.get("tags")
        unknown = review.get("unknown")
        if not isinstance(expected, list) or not isinstance(tags, list) or not isinstance(unknown, bool):
            raise CalibrationValidationError(f"invalid Ground Truth fields for {candidate_id}")
        if not tags or len(tags) != len(set(tags)) or any(tag not in ALLOWED_TAGS for tag in tags):
            raise CalibrationValidationError(f"invalid tags for {candidate_id}")
        if unknown != ("unknown" in tags) or unknown != (expected == []):
            raise CalibrationValidationError(f"unknown/expected/tag consistency error for {candidate_id}")
        for label in expected:
            if not isinstance(label, dict) or set(label) != {"concept_id", "acceptable_ancestors"}:
                raise CalibrationValidationError(f"invalid expected label for {candidate_id}")
            concept_id = label["concept_id"]
            if concept_id not in concept_ids:
                raise CalibrationValidationError(f"unknown eden-v2 concept: {concept_id}")
            ancestors = list(resolver.ancestor_path(concept_id)[:-1])
            if label["acceptable_ancestors"] != ancestors:
                raise CalibrationValidationError(f"acceptable_ancestors mismatch for {candidate_id}")
        checked.append({
            "candidate_id": candidate_id,
            "source_image_id": source["source_image_id"],
            "source_sha256": source["source_sha256"],
            "generated_sha256": source["generated_sha256"],
            "transformation": source["transformation"],
            "intended_split": "calibration",
            "generated_path": image_path,
            "ground_truth": {
                "expected": expected,
                "unknown": unknown,
                "tags": tags,
                "notes": review.get("notes"),
            },
        })
    truth_counts = Counter("unknown" if row["ground_truth"]["unknown"] else "known" for row in checked)
    if truth_counts != {"unknown": 6, "known": 2}:
        raise CalibrationValidationError(f"expected six UNKNOWN and two known calibration items; got {truth_counts}")
    return checked


def _score_distribution(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "count": len(values),
        "min": min(values, default=None),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": max(values, default=None),
    }


def _validate_ranking(candidates: list[dict[str, Any]], concept_ids: set[str]) -> list[dict[str, Any]]:
    if len(candidates) < 5:
        raise CalibrationValidationError("runtime returned fewer than five concepts")
    seen: set[str] = set()
    ranked = []
    for item in candidates:
        concept_id = item.get("concept_id")
        score = item.get("raw_score")
        if concept_id not in concept_ids or concept_id in seen:
            raise CalibrationValidationError("runtime returned an invalid or duplicate concept")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise CalibrationValidationError("runtime returned a non-finite raw score")
        seen.add(concept_id)
        ranked.append({"concept_id": concept_id, "raw_score": float(score)})
    if [row["raw_score"] for row in ranked] != sorted((row["raw_score"] for row in ranked), reverse=True):
        raise CalibrationValidationError("runtime candidates are not ranked by descending raw score")
    return ranked


def collect_model_signals(
    items: list[dict[str, Any]], model: str, taxonomy: Any,
    resolver: HierarchyResolver,
    runtime_factory: Callable[[str], EvaluationModelRuntime] | None = None,
) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError(f"unsupported model: {model}")
    runtime = (runtime_factory or create_runtime)(model)
    concept_ids = {concept.concept_id for concept in taxonomy.concepts}
    image_service = ImageInputService(max_upload_bytes=10 * 1024 * 1024, max_image_pixels=25_000_000)
    results = []
    try:
        runtime.load()
        cache = runtime.prepare_taxonomy(taxonomy)
        for item in items:
            if item["candidate_id"] not in CALIBRATION_IDS:
                raise CalibrationValidationError("non-calibration item passed to inference")
            data = item["generated_path"].read_bytes()
            if sha256_bytes(data) != item["generated_sha256"]:
                raise CalibrationValidationError(f"calibration image changed before inference: {item['candidate_id']}")
            decoded = image_service.decode(data, "image/jpeg")
            ranked = _validate_ranking(runtime.rank(decoded).candidates, concept_ids)
            top5 = ranked[:5]
            top1, top2 = top5[:2]
            results.append({
                "candidate_id": item["candidate_id"],
                "source_image_id": item["source_image_id"],
                "transformation": item["transformation"],
                "generated_sha256": item["generated_sha256"],
                "ground_truth": item["ground_truth"],
                "top1": top1,
                "top2": top2,
                "top1_top2_raw_margin": top1["raw_score"] - top2["raw_score"],
                "top5": top5,
                "top1_top5_raw_spread": top1["raw_score"] - top5[-1]["raw_score"],
                "hierarchy_coherence": _coherence([row["concept_id"] for row in top5], resolver),
            })
    finally:
        runtime.close()
    spec = MODELS[model]
    truth_counts = Counter("unknown" if row["ground_truth"]["unknown"] else "known" for row in results)
    groups = {
        key: [row for row in results if row["ground_truth"]["unknown"] is unknown]
        for key, unknown in (("known", False), ("unknown", True))
    }
    return {
        "schema_version": "1.0",
        "kind": "calibration_raw_ranking_diagnostics",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "taxonomy_version": taxonomy.taxonomy_version,
        "candidate_count": len(taxonomy.concepts),
        "prompt_template": PROMPT_TEMPLATE,
        "score_kind": SCORE_KIND,
        "score_note": "Raw image-text logits; these values are not probabilities or confidence and are not comparable across models.",
        "thresholds_applied": False,
        "runtime_metadata": runtime.metadata,
        "cache": cache,
        "summary": {
            "candidate_count": len(results),
            "known_count": truth_counts["known"],
            "unknown_count": truth_counts["unknown"],
            "groups": {
                key: {
                    "top1_raw_score": _score_distribution(
                        [{"score": row["top1"]["raw_score"]} for row in rows], "score"
                    ),
                    "top1_top2_raw_margin": _score_distribution(rows, "top1_top2_raw_margin"),
                }
                for key, rows in groups.items()
            },
        },
        "images": results,
    }


def taxonomy_relation(left_id: str, right_id: str, resolver: HierarchyResolver) -> dict[str, Any]:
    """Classify the closest taxonomy relation; a shared root alone is unrelated."""
    left_path = resolver.ancestor_path(left_id)
    right_path = resolver.ancestor_path(right_id)
    common = resolver.common_ancestor(left_id, right_id)
    if left_id == right_id:
        relation = "exact"
    elif left_id in right_path or right_id in left_path:
        relation = "ancestor_descendant"
    elif common is not None and resolver.parent(common) is not None:
        relation = "shared_ancestor"
    else:
        relation = "unrelated"
    return {
        "relation": relation,
        "nearest_common_ancestor": common,
        "meaningful_common_ancestor": common if common is not None and resolver.parent(common) is not None else None,
    }


def _cross_row(candidate_id: str, truth: dict[str, Any], clip: dict[str, Any],
               siglip2: dict[str, Any], resolver: HierarchyResolver) -> dict[str, Any]:
    if clip["candidate_id"] != candidate_id or siglip2["candidate_id"] != candidate_id:
        raise CalibrationValidationError(f"cross-model candidate ID mismatch: {candidate_id}")
    clip_gt = clip["ground_truth"]
    siglip_gt = siglip2["ground_truth"]
    if clip_gt != siglip_gt or clip_gt["unknown"] != truth["unknown"] or clip_gt["expected"] != truth["expected"]:
        raise CalibrationValidationError(f"cross-model Ground Truth mismatch: {candidate_id}")
    relation = taxonomy_relation(clip["top1"]["concept_id"], siglip2["top1"]["concept_id"], resolver)
    return {
        "candidate_id": candidate_id,
        "source_image_id": truth.get("source_image_id"),
        "truth_unknown": truth["unknown"],
        "truth_expected": truth["expected"],
        "clip_top1": clip["top1"]["concept_id"],
        "clip_score": clip["top1"]["raw_score"],
        "clip_margin": clip["top1_top2_raw_margin"],
        "clip_spread": clip["top1_top5_raw_spread"],
        "siglip2_top1": siglip2["top1"]["concept_id"],
        "siglip2_score": siglip2["top1"]["raw_score"],
        "siglip2_margin": siglip2["top1_top2_raw_margin"],
        "siglip2_spread": siglip2["top1_top5_raw_spread"],
        "cross_model_relation": relation,
    }


def cross_model_diagnostic(clip_report: dict[str, Any], siglip_report: dict[str, Any],
                          resolver: HierarchyResolver) -> dict[str, Any]:
    clip_by_id = {row["candidate_id"]: row for row in clip_report["images"]}
    siglip_by_id = {row["candidate_id"]: row for row in siglip_report["images"]}
    expected = set(CALIBRATION_IDS)
    if set(clip_by_id) != expected or set(siglip_by_id) != expected:
        raise CalibrationValidationError("CLIP and SigLIP2 must each contain exactly the eight calibration IDs")
    results = []
    for candidate_id in CALIBRATION_IDS:
        clip, siglip2 = clip_by_id[candidate_id], siglip_by_id[candidate_id]
        truth = clip["ground_truth"] | {"source_image_id": clip["source_image_id"]}
        results.append(_cross_row(candidate_id, truth, clip, siglip2, resolver))
    return {
        "schema_version": "1.0",
        "kind": "calibration_cross_model_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy_version": resolver.taxonomy_version,
        "score_note": "Scores and margins are raw image-text logits, not probabilities or confidence; do not compare score magnitudes across models.",
        "thresholds_applied": False,
        "images": results,
    }


def _canonical_rows(report: dict[str, Any], expected_model: str) -> dict[str, dict[str, Any]]:
    if report.get("model") != expected_model or report.get("taxonomy_version") != "eden-v2":
        raise CalibrationValidationError(f"unexpected canonical signals report for {expected_model}")
    result = {}
    for row in report.get("images", []):
        provenance = row["provenance"]
        candidate_id = provenance["image_id"]
        if candidate_id in result:
            raise CalibrationValidationError(f"duplicate canonical image ID: {candidate_id}")
        result[candidate_id] = row
    expected_ids = {f"eval-{index:03d}" for index in range(1, 34)}
    if set(result) != expected_ids:
        raise CalibrationValidationError("canonical reports must contain exactly eval-001 through eval-033")
    return result


def _signal_parts(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ranking = row["ranking"]
    diagnostics = row["diagnostics"]
    top1 = ranking["top_1"]
    margin = ranking["top1_top2_raw_margin"]
    spread = diagnostics["top1_top5_raw_spread"]
    return ({
        "top1": top1["concept_id"],
        "score": float(top1["raw_score"]),
        "margin": float(margin),
        "spread": float(spread),
    }, {
        "top5": ranking["top_5"],
        "hierarchy_coherence": diagnostics["top_k_hierarchy_coherence"],
    })


def _calibration_parts(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "top1": row["top1"]["concept_id"],
        "score": float(row["top1"]["raw_score"]),
        "margin": float(row["top1_top2_raw_margin"]),
        "spread": float(row["top1_top5_raw_spread"]),
        "top5": row["top5"],
        "hierarchy_coherence": row["hierarchy_coherence"],
    }


def build_combined_diagnostic(
    canonical_clip: dict[str, Any], canonical_siglip: dict[str, Any],
    calibration_clip: dict[str, Any], calibration_siglip: dict[str, Any],
    resolver: HierarchyResolver,
) -> dict[str, Any]:
    clip_canonical = _canonical_rows(canonical_clip, "clip")
    siglip_canonical = _canonical_rows(canonical_siglip, "siglip2")
    if set(clip_canonical) != set(siglip_canonical):
        raise CalibrationValidationError("canonical CLIP and SigLIP2 IDs do not align")
    calibration_cross = cross_model_diagnostic(calibration_clip, calibration_siglip, resolver)
    clip_cal = {row["candidate_id"]: row for row in calibration_clip["images"]}
    siglip_cal = {row["candidate_id"]: row for row in calibration_siglip["images"]}
    combined = []

    for candidate_id in sorted(clip_canonical):
        clip_row, siglip_row = clip_canonical[candidate_id], siglip_canonical[candidate_id]
        gt = clip_row["ground_truth"]
        if gt != siglip_row["ground_truth"]:
            raise CalibrationValidationError(f"canonical model Ground Truth mismatch: {candidate_id}")
        clip_signal, clip_extra = _signal_parts(clip_row)
        siglip_signal, siglip_extra = _signal_parts(siglip_row)
        relation = taxonomy_relation(clip_signal["top1"], siglip_signal["top1"], resolver)
        combined.append({
            "candidate_id": candidate_id,
            "origin": "canonical",
            "truth_unknown": gt["unknown"],
            "truth_expected": gt["expected"],
            "tags": gt["tags"],
            "clip": {**clip_signal, **clip_extra},
            "siglip2": {**siglip_signal, **siglip_extra},
            "cross_model_relation": relation,
        })

    for cross in calibration_cross["images"]:
        candidate_id = cross["candidate_id"]
        clip_row, siglip_row = clip_cal[candidate_id], siglip_cal[candidate_id]
        combined.append({
            "candidate_id": candidate_id,
            "origin": "calibration_generated",
            "truth_unknown": cross["truth_unknown"],
            "truth_expected": cross["truth_expected"],
            "tags": clip_row["ground_truth"]["tags"],
            "clip": _calibration_parts(clip_row),
            "siglip2": _calibration_parts(siglip_row),
            "cross_model_relation": cross["cross_model_relation"],
        })

    if len(combined) != 41:
        raise CalibrationValidationError(f"expected combined population of 41, found {len(combined)}")
    population = Counter(
        "unknown" if row["truth_unknown"] else "known"
        for row in combined
    )
    origin_counts = Counter(row["origin"] for row in combined)
    if population != {"known": 32, "unknown": 9} or origin_counts != {"canonical": 33, "calibration_generated": 8}:
        raise CalibrationValidationError(f"combined population mismatch: {population}, {origin_counts}")

    relation_counts = {
        relation: {
            truth: sum(
                row["truth_unknown"] is (truth == "unknown")
                and row["cross_model_relation"]["relation"] == relation
                for row in combined
            )
            for truth in ("known", "unknown")
        }
        for relation in ("exact", "ancestor_descendant", "shared_ancestor", "unrelated")
    }
    known_rows = [row for row in combined if not row["truth_unknown"]]
    unknown_rows = [row for row in combined if row["truth_unknown"]]
    margin_summary = {
        model: {
            truth: _score_distribution(
                [{"margin": row[model]["margin"]} for row in rows], "margin"
            )
            for truth, rows in (("known", known_rows), ("unknown", unknown_rows))
        }
        for model in ("clip", "siglip2")
    }
    low_margin_known = [
        row for row in known_rows
        if row["clip"]["margin"] <= 1.0 or row["siglip2"]["margin"] <= 1.0
    ]
    return {
        "schema_version": "1.0",
        "kind": "combined_open_set_calibration_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy_version": "eden-v2",
        "thresholds_applied": False,
        "score_note": "Raw image-text logits and margins are not probabilities or confidence; compare margins within each model only.",
        "population": {
            "total": len(combined),
            "known": population["known"],
            "unknown": population["unknown"],
            "origins": dict(origin_counts),
        },
        "margin_summary": margin_summary,
        "cross_model_relation_counts": relation_counts,
        "low_margin_known_ids": [row["candidate_id"] for row in low_margin_known],
        "images": combined,
    }


def _publish_reports(report_dir: Path, reports: dict[str, dict[str, Any]]) -> dict[str, Path]:
    report_dir = Path(report_dir).expanduser().absolute()
    paths = {key: report_dir / OUTPUT_NAMES[key] for key in reports}
    existing = [path for path in paths.values() if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(f"refusing to overwrite reports: {existing}")
    report_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".calibration-diagnostics.", dir=report_dir))
    published: list[Path] = []
    try:
        for key, report in reports.items():
            path = staging / OUTPUT_NAMES[key]
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for key, destination in paths.items():
            os.link(staging / OUTPUT_NAMES[key], destination)
            published.append(destination)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return paths


def run_diagnostics(
    candidate_root: Path = CANDIDATE_DEFAULT,
    canonical_root: Path = CANONICAL_DEFAULT,
    report_dir: Path | None = None,
    runtime_factory: Callable[[str], EvaluationModelRuntime] | None = None,
) -> dict[str, Any]:
    candidate_root = Path(candidate_root).expanduser().resolve()
    canonical_root = Path(canonical_root).expanduser().resolve()
    taxonomy = load_evaluation_taxonomy("eden-v2")
    resolver = HierarchyResolver(taxonomy)
    items = calibration_inputs(candidate_root, taxonomy, resolver)

    report_dir = Path(report_dir) if report_dir is not None else candidate_root / "reports"
    targets = [Path(report_dir) / OUTPUT_NAMES[key] for key in OUTPUT_NAMES]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("one or more calibration report outputs already exist")

    canonical_clip = _read_json(canonical_root / "reports" / CANONICAL_REPORT_NAMES["clip"])
    canonical_siglip = _read_json(canonical_root / "reports" / CANONICAL_REPORT_NAMES["siglip2"])
    clip = collect_model_signals(items, "clip", taxonomy, resolver, runtime_factory)
    siglip2 = collect_model_signals(items, "siglip2", taxonomy, resolver, runtime_factory)
    cross = cross_model_diagnostic(clip, siglip2, resolver)
    combined = build_combined_diagnostic(canonical_clip, canonical_siglip, clip, siglip2, resolver)
    paths = _publish_reports(report_dir, {"clip": clip, "siglip2": siglip2, "cross": cross, "combined": combined})
    return {"paths": paths, "clip": clip, "siglip2": siglip2, "cross": cross, "combined": combined}


def diagnostic_tables(combined: dict[str, Any]) -> str:
    rows = combined["images"]
    lines = [
        "UNKNOWN rows (candidate/id | origin | CLIP top1 / margin | SigLIP2 top1 / margin | relation)",
    ]
    for row in rows:
        if not row["truth_unknown"]:
            continue
        relation = row["cross_model_relation"]["relation"]
        lines.append(
            f"{row['candidate_id']} | {row['origin']} | {row['clip']['top1']} / {row['clip']['margin']:.6f} | "
            f"{row['siglip2']['top1']} / {row['siglip2']['margin']:.6f} | {relation}"
        )
    lines.append("Known rows with either model margin <= 1.0 (same columns)")
    for row in rows:
        if row["truth_unknown"] or not (row["clip"]["margin"] <= 1.0 or row["siglip2"]["margin"] <= 1.0):
            continue
        lines.append(
            f"{row['candidate_id']} | {row['origin']} | {row['clip']['top1']} / {row['clip']['margin']:.6f} | "
            f"{row['siglip2']['top1']} / {row['siglip2']['margin']:.6f} | {row['cross_model_relation']['relation']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=CANDIDATE_DEFAULT)
    parser.add_argument("--canonical", type=Path, default=CANONICAL_DEFAULT)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_diagnostics(args.candidates, args.canonical, args.report_dir)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"calibration diagnostics failed: {exc}\n")
    print(json.dumps({"reports": {key: str(value) for key, value in result["paths"].items()},
                      "population": result["combined"]["population"],
                      "margin_summary": result["combined"]["margin_summary"],
                      "cross_model_relation_counts": result["combined"]["cross_model_relation_counts"]}, indent=2))
    print(diagnostic_tables(result["combined"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
