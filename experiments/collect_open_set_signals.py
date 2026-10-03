"""Collect model-specific raw ranking diagnostics; no thresholds or decisions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_models import evaluate_condition, make_plan, sha256
from experiments.evaluation_core import ManifestValidationError, _percentile
from experiments.model_runtime import MODELS, PROMPT_TEMPLATE, EvaluationModelRuntime, create_runtime


def distribution(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "min": min(values, default=None),
            "median": _percentile(values, 0.5), "p95": _percentile(values, 0.95),
            "max": max(values, default=None)}


def summarize(images: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "known": [r for r in images if not r["ground_truth"]["unknown"]],
        "broad_only": [r for r in images if "broad_only" in r["ground_truth"]["tags"]],
        "ambiguous": [r for r in images if "ambiguous" in r["ground_truth"]["tags"]],
        "unknown": [r for r in images if r["ground_truth"]["unknown"]],
    }
    return {
        "image_count": len(images), **{f"{name}_count": len(rows) for name, rows in groups.items()},
        "unknown_status": "available" if groups["unknown"] else "no_unknown_samples",
        "groups_overlap": True,
        "groups": {name: {
            "status": "available" if rows else "no_samples",
            "top1_raw_score": distribution([r["ranking"]["top_1"]["raw_score"] for r in rows]),
            "top1_top2_raw_margin": distribution([r["ranking"]["top1_top2_raw_margin"] for r in rows]),
        } for name, rows in groups.items()},
    }


def _publish_new(path: Path, result: dict[str, Any]) -> None:
    """Atomic publication with no overwrite, even if another writer wins a race."""
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike replace(), link() atomically refuses an existing destination.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_signals(dataset: Path, model: str, taxonomy: str, output: Path, *,
                    runtime_factory: Callable[[str], EvaluationModelRuntime] | None = None) -> dict[str, Any]:
    if model not in MODELS or taxonomy != "eden-v2":
        raise ValueError("choose clip or siglip2 with taxonomy eden-v2")
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists; choose a new filename: {output}")
    plan = make_plan(Path(dataset), model, taxonomy)
    if plan.dataset.metadata.taxonomy_version != "eden-v2":
        raise ManifestValidationError("open-set annotation taxonomy must be eden-v2")
    selected = plan.taxonomies[taxonomy]
    spec = MODELS[model]
    common = {"dataset_id": plan.dataset.metadata.dataset_id, "model": model,
              "model_id": spec.model_id, "model_revision": spec.revision,
              "taxonomy_version": taxonomy, "candidate_count": len(selected.concepts)}
    images = []
    runtime_metadata, cache = {}, None
    if plan.dataset.entries:
        runtime = (runtime_factory or create_runtime)(model)
        try:
            runtime.load()
            # Reuse existing decode/hash checks, cached runtime.rank(), and
            # evaluation_core's top-5 hierarchy coherence. No report is written here.
            evaluated = evaluate_condition(plan, runtime, model, selected)
            runtime_metadata = dict(runtime.metadata)
            cache = evaluated["cache"]
            valid_ids = {c.concept_id for c in selected.concepts}
            for row in evaluated["images"]:
                candidates = row["ranked_candidates"]
                ids = [c["concept_id"] for c in candidates]
                if len(ids) != len(valid_ids) or set(ids) != valid_ids:
                    raise ValueError("ranking must contain every taxonomy candidate exactly once")
                scores = [float(c["raw_score"]) for c in candidates]
                if not all(math.isfinite(s) for s in scores) or scores != sorted(scores, reverse=True):
                    raise ValueError("ranking scores must be finite and descending")
                top = [{"concept_id": c["concept_id"], "raw_score": s}
                       for c, s in zip(candidates[:5], scores[:5], strict=True)]
                margin, spread = scores[0] - scores[1], scores[0] - scores[4]
                if not math.isfinite(margin) or not math.isfinite(spread):
                    raise ValueError("non-finite score difference")
                images.append({
                    "provenance": {**common, "image_id": row["image_id"], "file": row["file"],
                                   "image_sha256": row["image_sha256"]},
                    "ground_truth": {"expected": row["expected_annotation"],
                                     "unknown": row["unknown_ground_truth"],
                                     "tags": row["tags"], "notes": row["notes"]},
                    "ranking": {"top_1": top[0], "top_2": top[1],
                                "top1_top2_raw_margin": margin, "top_5": top},
                    "diagnostics": {"top1_top5_raw_spread": spread,
                                    "top_k_hierarchy_coherence": {"k": 5, **row["hierarchy_coherence"]}},
                })
        finally:
            runtime.close()
    # Do not publish diagnostics whose labels/images changed during collection.
    if (sha256(plan.root / "dataset.json") != plan.provenance["dataset_metadata_sha256"]
            or sha256(plan.root / "manifest.jsonl") != plan.provenance["manifest_sha256"]
            or any(sha256(plan.root / entry.file) != plan.provenance["image_sha256"][entry.id]
                   for entry in plan.dataset.entries)):
        raise ManifestValidationError("dataset changed during signal collection")
    result = {
        "schema_version": "1.0", "kind": "open_set_ranking_diagnostics",
        "created_at": datetime.now(timezone.utc).isoformat(), **common,
        "annotation_taxonomy_version": plan.dataset.metadata.taxonomy_version,
        "status": "collected" if images else "empty_dataset",
        "model_loaded": bool(images), "thresholds_applied": False,
        "score_kind": "raw_image_text_logit",
        "score_note": "Uncalibrated raw logits, not probabilities. Interpret separately for each model; do not compare magnitudes across models.",
        "summary_note": "known includes all unknown=false entries; tag groups may overlap. No UNKNOWN decision is inferred.",
        "percentile_method": "linear interpolation at (n - 1) * q",
        "prompt_template": PROMPT_TEMPLATE, "runtime_metadata": runtime_metadata, "cache": cache,
        "dataset_provenance": plan.provenance,
        "summary": summarize(images), "images": images,
    }
    _publish_new(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--taxonomy", choices=["eden-v2"], default="eden-v2")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; existing output is always refused")
    args = parser.parse_args(argv)
    try:
        result = collect_signals(args.dataset, args.model, args.taxonomy, args.output)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"signal collection failed: {exc}\n")
    print(f"{result['status']}: {result['summary']['image_count']} images; model={args.model}; output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
