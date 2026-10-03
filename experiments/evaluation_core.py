"""Model-independent dataset validation and ranking metrics.

This module deliberately imports no torch, transformers, or image package. It
is usable from the light service environment and is the shared contract for
the CLIP and SigLIP2 runners.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.classification.taxonomy import Taxonomy

ALLOWED_TAGS = frozenset(
    {"single_object", "multi_object", "fine_grained", "broad_only", "ambiguous", "unknown", "scene"}
)
RELATIONS = ("exact", "ancestor_distance_1", "ancestor_distance_2plus", "unrelated", "over_specific")


class ManifestValidationError(ValueError):
    """The evaluation dataset is not safe or internally consistent."""


@dataclass(frozen=True, slots=True)
class ExpectedLabel:
    concept_id: str
    acceptable_ancestors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    id: str
    file: str
    expected: tuple[ExpectedLabel, ...]
    tags: tuple[str, ...]
    unknown: bool
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    dataset_id: str
    taxonomy_version: str
    image_count: int


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    metadata: DatasetMetadata
    entries: tuple[ManifestEntry, ...]


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestValidationError(f"expected JSON object: {path}")
    return value


HASH_CHUNK_BYTES = 64 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(dataset_dir: Path, taxonomy: Taxonomy) -> EvaluationDataset:
    """Read canonical JSONL and use the same validator as draft promotion."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    raw_meta = _json_object(dataset_dir / "dataset.json")
    rows = []
    try:
        with (dataset_dir / "manifest.jsonl").open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ManifestValidationError(f"manifest line {line_no} is invalid JSON: {exc}") from exc
    except OSError as exc:
        raise ManifestValidationError(f"cannot read canonical manifest: {exc}") from exc
    return validate_dataset_rows(dataset_dir, raw_meta, rows, taxonomy)


def validate_dataset_rows(dataset_dir: Path, raw_meta: dict[str, Any],
                          rows: Iterable[dict[str, Any]], taxonomy: Taxonomy) -> EvaluationDataset:
    """Validate proposed or persisted canonical data before it is used."""
    required = {"dataset_id", "taxonomy_version", "image_count"}
    if set(raw_meta) != required:
        raise ManifestValidationError("dataset.json has unexpected or missing fields")
    if not isinstance(raw_meta["dataset_id"], str) or not raw_meta["dataset_id"].strip():
        raise ManifestValidationError("dataset_id must be non-empty text")
    if raw_meta["taxonomy_version"] != taxonomy.taxonomy_version:
        raise ManifestValidationError("dataset taxonomy_version does not match evaluation taxonomy")
    if not isinstance(raw_meta["image_count"], int) or raw_meta["image_count"] < 0:
        raise ManifestValidationError("image_count must be a non-negative integer")

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    rows = list(rows)
    known = {concept.concept_id for concept in taxonomy.concepts}
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    seen_hashes: set[str] = set()
    seen_files: set[Path] = set()
    for line_no, raw in enumerate(rows, 1):
        if not isinstance(raw, dict):
            raise ManifestValidationError(f"manifest line {line_no} must be an object")
        fields = {"id", "file", "expected", "tags", "unknown", "notes"}
        if set(raw) not in (fields, fields | {"sha256"}):
            raise ManifestValidationError(f"manifest line {line_no} has unexpected or missing fields")
        if "sha256" in raw and (not isinstance(raw["sha256"], str) or len(raw["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in raw["sha256"].lower())):
            raise ManifestValidationError(f"invalid sha256 at line {line_no}")
        if "sha256" in raw and raw["sha256"] in seen_hashes:
            raise ManifestValidationError(f"duplicate sha256 at line {line_no}")
        item_id, rel_file = raw["id"], raw["file"]
        if not isinstance(item_id, str) or not item_id.strip() or item_id in seen:
            raise ManifestValidationError(f"duplicate or invalid image id at line {line_no}")
        if not isinstance(rel_file, str) or not rel_file or Path(rel_file).is_absolute() or ".." in Path(rel_file).parts:
            raise ManifestValidationError(f"invalid relative file path at line {line_no}")
        file_path = (dataset_dir / rel_file).resolve()
        if dataset_dir not in file_path.parents or not file_path.is_file():
            raise ManifestValidationError(f"image file does not exist: {rel_file}")
        if file_path in seen_files:
            raise ManifestValidationError(f"duplicate file at line {line_no}")
        seen_files.add(file_path)
        if not isinstance(raw["expected"], list) or not isinstance(raw["tags"], list):
            raise ManifestValidationError(f"expected and tags must be lists at line {line_no}")
        if not isinstance(raw["unknown"], bool):
            raise ManifestValidationError(f"unknown must be boolean at line {line_no}")
        if raw["unknown"] and raw["expected"]:
            raise ManifestValidationError(f"unknown image cannot have expected labels at line {line_no}")
        if not raw["unknown"] and not raw["expected"]:
            raise ManifestValidationError(f"non-unknown image needs expected labels at line {line_no}")
        tags = tuple(raw["tags"])
        if not tags or any(not isinstance(tag, str) or tag not in ALLOWED_TAGS for tag in tags):
            raise ManifestValidationError(f"invalid tag at line {line_no}")
        if raw["unknown"] != ("unknown" in tags):
            raise ManifestValidationError(f"unknown tag must match unknown field at line {line_no}")
        expected: list[ExpectedLabel] = []
        for label in raw["expected"]:
            if not isinstance(label, dict) or set(label) not in ({"concept_id"}, {"concept_id", "acceptable_ancestors"}):
                raise ManifestValidationError(f"invalid expected label at line {line_no}")
            concept_id = label.get("concept_id")
            ancestors = label.get("acceptable_ancestors", [])
            if concept_id not in known:
                raise ManifestValidationError(f"unknown expected concept: {concept_id}")
            if not isinstance(ancestors, list) or any(value not in known for value in ancestors):
                raise ManifestValidationError(f"invalid acceptable ancestor at line {line_no}")
            expected.append(ExpectedLabel(concept_id, tuple(ancestors)))
        notes = raw["notes"]
        if notes is not None and not isinstance(notes, str):
            raise ManifestValidationError(f"notes must be text or null at line {line_no}")
        entries.append(ManifestEntry(item_id, rel_file, tuple(expected), tags, raw["unknown"], notes))
        seen.add(item_id)
        if "sha256" in raw:
            seen_hashes.add(raw["sha256"])
    if raw_meta["image_count"] != len(entries):
        raise ManifestValidationError("dataset.json image_count does not match manifest")
    for raw in rows:
        if "sha256" in raw and file_sha256(dataset_dir / raw["file"]) != raw["sha256"].lower():
            raise ManifestValidationError(f"SHA-256 does not match image: {raw['id']}")
    return EvaluationDataset(DatasetMetadata(raw_meta["dataset_id"], raw_meta["taxonomy_version"], len(entries)), tuple(entries))


def hierarchy_relation(truth: str, predicted: str, resolver: HierarchyResolver) -> str:
    """Classify a prediction relative to a ground-truth concept."""
    truth_path, predicted_path = resolver.ancestor_path(truth), resolver.ancestor_path(predicted)
    if truth == predicted:
        return "exact"
    if predicted in truth_path:
        distance = len(truth_path) - truth_path.index(predicted) - 1
        return "ancestor_distance_1" if distance == 1 else "ancestor_distance_2plus"
    if truth in predicted_path:
        return "over_specific"
    return "unrelated"


def _best_relation(expected: ExpectedLabel, predicted: Sequence[str], resolver: HierarchyResolver) -> str:
    relations = []
    for pred in predicted:
        relation = hierarchy_relation(expected.concept_id, pred, resolver)
        if relation == "ancestor_distance_1" and expected.acceptable_ancestors and pred not in expected.acceptable_ancestors:
            relation = "unrelated"
        elif relation == "ancestor_distance_2plus" and expected.acceptable_ancestors and pred not in expected.acceptable_ancestors:
            relation = "unrelated"
        relations.append(relation)
    rank = {name: index for index, name in enumerate(("exact", "ancestor_distance_1", "ancestor_distance_2plus", "over_specific", "unrelated"))}
    return min(relations, key=lambda relation: rank[relation]) if relations else "unrelated"


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _top_ids(ranked: Sequence[dict[str, Any] | str], k: int) -> list[str]:
    return [item if isinstance(item, str) else str(item["concept_id"]) for item in ranked[:k]]


def _coherence(top_ids: Sequence[str], resolver: HierarchyResolver) -> dict[str, Any]:
    if len(top_ids) < 2:
        return {"related_pair_fraction": None, "unique_broad_categories": len(top_ids)}
    pairs = [(left, right) for index, left in enumerate(top_ids) for right in top_ids[index + 1 :]]
    related = sum(resolver.common_ancestor(left, right) is not None for left, right in pairs)
    return {
        "related_pair_fraction": related / len(pairs),
        "unique_broad_categories": len({resolver.broad_category(item) for item in top_ids}),
    }


def evaluate_image(entry: ManifestEntry, ranked: Sequence[dict[str, Any] | str], latency_ms: float, resolver: HierarchyResolver) -> dict[str, Any]:
    ranked_ids = _top_ids(ranked, len(ranked))
    expected_ids = [label.concept_id for label in entry.expected]
    single = len(expected_ids) == 1 and not entry.unknown
    top = {k: _top_ids(ranked, k) for k in (1, 3, 5)}
    exact = {f"top_{k}": bool(single and expected_ids[0] in top[k]) for k in (1, 3, 5)}
    relations = {label.concept_id: _best_relation(label, top[1], resolver) for label in entry.expected}
    broad_hits = {f"top_{k}": [resolver.broad_category(label.concept_id) in {resolver.broad_category(item) for item in top[k]} for label in entry.expected] for k in (1, 5)}
    multi = {}
    if len(expected_ids) > 1 and not entry.unknown:
        for k in (3, 5):
            predicted = set(top[k])
            hits = len(predicted & set(expected_ids))
            multi[f"top_{k}"] = {"hits": hits, "precision": hits / k, "recall": hits / len(expected_ids)}
    return {
        "image_id": entry.id,
        "file": entry.file,
        "expected": expected_ids,
        "unknown_ground_truth": entry.unknown,
        "top_1": ranked[0] if ranked else None,
        "top_3": list(ranked[:3]),
        "top_5": list(ranked[:5]),
        "ranked_candidates": list(ranked),
        "exact": exact,
        "hierarchy_relations": relations,
        "broad_category_hits": broad_hits,
        "multi_label": multi,
        "hierarchy_coherence": _coherence(ranked_ids[:5], resolver),
        "latency_ms": latency_ms,
    }


def aggregate_metrics(images: Iterable[dict[str, Any]], resolver: HierarchyResolver) -> dict[str, Any]:
    rows = list(images)
    singles = [row for row in rows if len(row["expected"]) == 1 and not row["unknown_ground_truth"]]
    exact = {f"top_{k}": {"count": sum(row["exact"][f"top_{k}"] for row in singles), "total": len(singles), "rate": _rate(sum(row["exact"][f"top_{k}"] for row in singles), len(singles))} for k in (1, 3, 5)}
    relation_counts = {relation: 0 for relation in RELATIONS}
    for row in singles:
        relation_counts[row["hierarchy_relations"][row["expected"][0]]] += 1
    multi_rows = [row for row in rows if len(row["expected"]) > 1 and not row["unknown_ground_truth"]]
    multi = {}
    for k in (3, 5):
        values = [row["multi_label"][f"top_{k}"] for row in multi_rows]
        multi[f"top_{k}"] = {"images": len(values), "macro_precision": _rate(sum(value["precision"] for value in values), len(values)), "macro_recall": _rate(sum(value["recall"] for value in values), len(values))}
    broad = {}
    for k in (1, 5):
        values = [hit for row in rows if not row["unknown_ground_truth"] for hit in row["broad_category_hits"][f"top_{k}"]]
        broad[f"top_{k}"] = {"count": sum(values), "total": len(values), "rate": _rate(sum(values), len(values))}
    coherence = [row["hierarchy_coherence"]["related_pair_fraction"] for row in rows if row["hierarchy_coherence"]["related_pair_fraction"] is not None]
    return {
        "image_count": len(rows),
        "single_label_count": len(singles),
        "multi_label_count": len(multi_rows),
        "unknown_count": sum(row["unknown_ground_truth"] for row in rows),
        "top_k_exact": exact,
        "hierarchy": {"counts": relation_counts, "total_single_label": len(singles), "rates": {key: _rate(value, len(singles)) for key, value in relation_counts.items()}},
        "broad_category_hit": broad,
        "multi_label": multi,
        "hierarchy_coherence": {"mean_related_pair_fraction": _rate(sum(coherence), len(coherence))},
        "latency_ms": {"count": len(rows), "min": min((row["latency_ms"] for row in rows), default=None), "median": _median([row["latency_ms"] for row in rows]), "p95": _percentile([row["latency_ms"] for row in rows], 0.95), "max": max((row["latency_ms"] for row in rows), default=None)},
    }


def _median(values: Sequence[float]) -> float | None:
    return _percentile(values, 0.5)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
