"""Cached sequential evaluation. Dry-run/validation/empty datasets load no AI.

Reports are only replaced with explicit --overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.classification.taxonomy import Taxonomy
from experiments.evaluation_core import (
    EvaluationDataset, ManifestValidationError, aggregate_metrics, evaluate_image, load_dataset,
)
from experiments.model_runtime import (
    MODELS, TAXONOMIES, PROMPT_TEMPLATE, EvaluationModelRuntime, create_runtime, load_evaluation_taxonomy,
)


def selections(value: str, available: dict) -> tuple[str, ...]:
    if value in {"all", "both"}:
        return tuple(available)
    if value not in available:
        raise ValueError(f"unsupported selection: {value}")
    return (value,)


def evaluation_matrix(model: str, taxonomy: str) -> tuple[tuple[str, str], ...]:
    return tuple((m, t) for m in selections(model, MODELS) for t in selections(taxonomy, TAXONOMIES))


def report_filename(model: str, taxonomy: str) -> str:
    if model not in MODELS or taxonomy not in TAXONOMIES:
        raise ValueError("unsupported report selection")
    return f"{model}-{taxonomy}-report.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class EvaluationPlan:
    root: Path
    output: Path
    dataset: EvaluationDataset
    taxonomies: dict[str, Taxonomy]
    reference: Taxonomy
    matrix: tuple[tuple[str, str], ...]
    provenance: dict[str, Any]


def make_plan(root: Path, model: str, taxonomy: str, output: Path | None = None) -> EvaluationPlan:
    root = root.expanduser().resolve()
    matrix = evaluation_matrix(model, taxonomy)
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    annotation = load_evaluation_taxonomy(metadata["taxonomy_version"])
    # Retain the canonical validator's strict annotation version check.
    dataset = load_dataset(root, annotation)
    taxonomies = {version: load_evaluation_taxonomy(version) for _, version in matrix}
    reference_version = "eden-v2" if "eden-v2" in (*taxonomies, annotation.taxonomy_version) else "eden-v1"
    reference = load_evaluation_taxonomy(reference_version)
    reference_ids = {c.concept_id: c for c in reference.concepts}
    # A common superset resolver permits v2 truths with v1 candidates, without
    # mapping truths to easier labels or dropping rows from a denominator.
    for source in (annotation, *taxonomies.values()):
        for concept in source.concepts:
            target = reference_ids.get(concept.concept_id)
            if target is None or (target.parent_id, target.broad_category_id) != (concept.parent_id, concept.broad_category_id):
                raise ManifestValidationError(f"incompatible taxonomy relationship: {concept.concept_id}")
    image_hashes = {entry.id: sha256(root / entry.file) for entry in dataset.entries}
    return EvaluationPlan(root, (output or root / "reports").expanduser().resolve(), dataset,
                          taxonomies, reference, matrix, {
                              "dataset_metadata_sha256": sha256(root / "dataset.json"),
                              "manifest_sha256": sha256(root / "manifest.jsonl"),
                              "image_sha256": image_hashes,
                          })


def _write(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(encoded)


def evaluate_condition(plan: EvaluationPlan, runtime: EvaluationModelRuntime, model: str,
                       taxonomy: Taxonomy) -> dict[str, Any]:
    from eden_vision.services.image_input import ImageInputService

    cache = runtime.prepare_taxonomy(taxonomy)
    after_cache = runtime.memory()
    resolver = HierarchyResolver(plan.reference)
    decoder = ImageInputService(max_upload_bytes=10 * 1024 * 1024, max_image_pixels=25_000_000)
    images = []
    for index, entry in enumerate(sorted(plan.dataset.entries, key=lambda entry: entry.id)):
        started = time.perf_counter()
        path = (plan.root / entry.file).resolve()
        if not path.is_relative_to(plan.root):
            raise ManifestValidationError(f"image path escaped dataset: {entry.id}")
        with path.open("rb") as stream:
            data = stream.read(decoder.max_upload_bytes + 1)
        if hashlib.sha256(data).hexdigest() != plan.provenance["image_sha256"][entry.id]:
            raise ManifestValidationError(f"image changed after validation or exceeds byte limit: {entry.id}")
        decoded = decoder.decode(data, None)
        decode_ms = (time.perf_counter() - started) * 1000
        prediction = runtime.rank(decoded)
        row = evaluate_image(entry, prediction.candidates, decode_ms + prediction.timings_ms["total_ms"], resolver)
        row.update({"expected_annotation": [asdict(label) for label in entry.expected],
                    "tags": list(entry.tags), "notes": entry.notes,
                    "first_request_after_prepare": index == 0,
                    "timings_ms": {"read_hash_decode_ms": decode_ms, **prediction.timings_ms},
                    "image_sha256": plan.provenance["image_sha256"][entry.id]})
        images.append(row)
    candidate_ids = {c.concept_id for c in taxonomy.concepts}
    return {
        "report_schema_version": "1.0", "dataset_id": plan.dataset.metadata.dataset_id,
        "annotation_taxonomy_version": plan.dataset.metadata.taxonomy_version,
        "taxonomy_version": taxonomy.taxonomy_version,
        "relationship_taxonomy_version": plan.reference.taxonomy_version,
        "candidate_count": len(taxonomy.concepts), "image_count": len(images),
        "model": model, **runtime.metadata,
        "prompt_template": PROMPT_TEMPLATE, "cache_mode": "precomputed_text",
        "cache": cache, "dataset_provenance": plan.provenance,
        "truth_concepts_outside_candidates": sorted({label.concept_id for entry in plan.dataset.entries
                                                    for label in entry.expected} - candidate_ids),
        "unknown_detection": {"evaluated": False, "reason": "no calibrated threshold; annotation status only"},
        "score_note": "Raw logits are not confidence; compare metrics, never raw magnitudes across models.",
        "latency_note": "One request per image; includes read/hash/decode, processor, transfers, image encoder, matching and ranking; excludes load/text preparation. First request is flagged; no warm repetition.",
        "metrics": aggregate_metrics(images, resolver), "images": images,
        "memory": {"unit": "MiB", "rss_method": "macOS ps RSS KiB / 1024",
                   "after_cache": after_cache, "after_evaluation": runtime.memory()},
    }


def run_evaluations(plan: EvaluationPlan, *, overwrite: bool = False,
                    runtime_factory: Callable[[str], EvaluationModelRuntime] = create_runtime) -> list[dict[str, Any]]:
    if not plan.dataset.entries:
        return []
    comparison_path = plan.output / "model-taxonomy-comparison.json"
    outputs = [plan.output / report_filename(*condition) for condition in plan.matrix] + [comparison_path]
    if not overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(f"report already exists; choose --output-dir or explicitly --overwrite: {path}")
    plan.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for model in dict.fromkeys(model for model, _ in plan.matrix):
        runtime = runtime_factory(model)
        try:
            runtime.load()
            for selected_model, version in plan.matrix:
                if selected_model != model:
                    continue
                report = evaluate_condition(plan, runtime, model, plan.taxonomies[version])
                _write(plan.output / report_filename(model, version), report, overwrite)
                reports.append(report)
        finally:
            runtime.close()
    comparison = {
        "dataset_id": plan.dataset.metadata.dataset_id,
        "dataset_provenance": plan.provenance,
        "annotation_taxonomy_version": plan.dataset.metadata.taxonomy_version,
        "relationship_taxonomy_version": plan.reference.taxonomy_version,
        "unknown_detection_evaluated": False,
        "comparison_note": "Identical GT and images. No winner or cross-model raw-score comparison.",
        "conditions": [{key: report[key] for key in (
            "model", "model_id", "model_revision", "taxonomy_version", "candidate_count", "image_count",
            "metrics", "memory", "truth_concepts_outside_candidates",
        )} for report in reports],
    }
    _write(comparison_path, comparison, overwrite)
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/Users/dev/Datasets/eden-vision-eval"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", choices=(*MODELS, "all", "both"), default="all", help="both is a compatibility alias for all")
    parser.add_argument("--taxonomy", choices=(*TAXONOMIES, "all"), default="eden-v1")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace selected reports")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args.dataset, args.model, args.taxonomy, args.output_dir)
        print(f"dataset validated: {plan.dataset.metadata.dataset_id}; entries={len(plan.dataset.entries)}; annotation={plan.dataset.metadata.taxonomy_version}")
        print("planned evaluations:")
        for model, taxonomy in plan.matrix:
            print(f"- {model} / {taxonomy} -> {plan.output / report_filename(model, taxonomy)}")
        if args.validate_only or args.dry_run or not plan.dataset.entries:
            print("model not loaded; no reports written" + ("; dataset empty" if not plan.dataset.entries else ""))
            return 0
        run_evaluations(plan, overwrite=args.overwrite)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"evaluation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
