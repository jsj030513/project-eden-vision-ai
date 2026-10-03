"""Interactive human labeling helper; it never inspects image pixels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eden_vision.classification.hierarchy import HierarchyResolver
from experiments.eval_dataset import _read_jsonl, _write_jsonl, dataset_taxonomy
from experiments.model_runtime import load_evaluation_taxonomy


def _show_concepts(query: str | None = None, taxonomy_version: str = "eden-v1") -> None:
    taxonomy = load_evaluation_taxonomy(taxonomy_version)
    resolver = HierarchyResolver(taxonomy)
    for concept in taxonomy.concepts:
        if query and query.lower() not in (concept.concept_id + " " + concept.display_name).lower():
            continue
        print(f"{concept.concept_id}: {' > '.join(resolver.ancestor_path(concept.concept_id))} ({concept.display_name})")


def _label(dataset: Path, taxonomy_version: str) -> int:
    taxonomy = load_evaluation_taxonomy(taxonomy_version)
    dataset_taxonomy(dataset, taxonomy)
    resolver = HierarchyResolver(taxonomy)
    known = {concept.concept_id for concept in taxonomy.concepts}
    path = dataset / "manifest.draft.jsonl"
    rows = _read_jsonl(path)
    changed = 0
    for row in rows:
        if row.get("expected") is not None and row.get("unknown") is not None:
            continue
        row["taxonomy_version"] = taxonomy_version
        print(f"\nGround Truth taxonomy: {taxonomy_version} ({len(taxonomy.concepts)} concepts)")
        print(f"Image: {row['file']}  ({row['width']}x{row['height']}, {row['format']})")
        print("Enter concept IDs separated by commas, or leave blank only for unknown.")
        print("Use --concepts or the taxonomy paths above to browse concepts.")
        raw_expected = input("Expected concept IDs: ").strip()
        raw_unknown = input("Unknown? [y/N]: ").strip().lower()
        unknown = raw_unknown in {"y", "yes"}
        expected_ids = [value.strip() for value in raw_expected.split(",") if value.strip()]
        if any(value not in known for value in expected_ids):
            raise ValueError("unknown concept ID; run --concepts or --search")
        if unknown and expected_ids:
            raise ValueError("unknown entries cannot have expected concepts")
        tags_text = input("Tags (comma separated): ").strip()
        tags = [value.strip() for value in tags_text.split(",") if value.strip()]
        if not tags:
            raise ValueError("at least one tag is required")
        if unknown and "unknown" not in tags:
            tags.append("unknown")
        notes = input("Notes (optional): ").strip() or None
        row["expected"] = [{"concept_id": value, "acceptable_ancestors": list(resolver.ancestor_path(value)[:-1])} for value in expected_ids]
        row["unknown"] = unknown
        row["tags"] = tags
        row["notes"] = notes
        changed += 1
        # Persist every completed annotation so an interruption does not
        # discard earlier human Ground Truth work.
        _write_jsonl(path, rows)
    _write_jsonl(path, rows)
    print(f"updated draft entries: {changed}")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/Users/dev/Datasets/eden-vision-eval"))
    parser.add_argument("--taxonomy", choices=("eden-v1", "eden-v2"), default="eden-v1")
    parser.add_argument("--concepts", action="store_true", help="list taxonomy concepts and hierarchy paths")
    parser.add_argument("--search", type=str, help="search taxonomy concept IDs/display names")
    parser.add_argument("--label", action="store_true", help="label incomplete draft entries interactively")
    args = parser.parse_args()
    if args.concepts or args.search:
        _show_concepts(args.search, args.taxonomy)
        return 0
    if args.label:
        _label(args.dataset, args.taxonomy)
        return 0
    parser.error("choose label, --concepts, or --search")


if __name__ == "__main__":
    raise SystemExit(main())
