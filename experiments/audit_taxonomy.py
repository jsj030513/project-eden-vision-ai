"""Static audit and v1/v2 compatibility report for Eden taxonomies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eden_vision.classification.taxonomy import Taxonomy, load_taxonomy, parse_taxonomy
from eden_vision.classification.hierarchy import HierarchyResolver

V1_PATH = Path(__file__).resolve().parents[1] / "src/eden_vision/resources/taxonomy_v1.json"
V2_PATH = Path(__file__).resolve().parents[1] / "src/eden_vision/resources/taxonomy_v2.json"

DOMAIN_QUERIES = {
    "person": "person", "pet": "animal", "wild animal": "animal", "plant": "plant", "flower": "flower",
    "food": "food", "drink": "drink", "vehicle": "vehicle", "computer/device": "electronic_device",
    "furniture": "furniture", "household object": "household_item", "clothing": "clothing",
    "tool": "tool", "building/structure": "building", "nature": "natural_feature",
    "indoor scene": "indoor_scene", "outdoor scene": "outdoor_scene", "document": "document",
    "sign/screen": "sign", "artwork": "artwork",
}


def load_path(path: Path) -> Taxonomy:
    return parse_taxonomy(path.read_text(encoding="utf-8"))


def audit(taxonomy: Taxonomy) -> dict[str, Any]:
    concepts = taxonomy.concepts
    ids = [concept.concept_id for concept in concepts]
    display_names = [concept.display_name.casefold() for concept in concepts]
    roots = [concept.concept_id for concept in concepts if concept.parent_id is None]
    resolver = HierarchyResolver(taxonomy)
    depths = {concept.concept_id: len(resolver.ancestor_path(concept.concept_id)) for concept in concepts}
    orphan = [concept.concept_id for concept in concepts if concept.parent_id is not None and concept.parent_id not in ids]
    empty = [concept.concept_id for concept in concepts if not concept.display_name.strip() or any(not item.strip() for item in concept.aliases + concept.descriptions)]
    return {
        "taxonomy_version": taxonomy.taxonomy_version,
        "concept_count": len(concepts),
        "root_concepts": roots,
        "maximum_hierarchy_depth": max(depths.values()),
        "leaf_concept_count": sum(not any(candidate.parent_id == concept.concept_id for candidate in concepts) for concept in concepts),
        "duplicate_ids": sorted({value for value in ids if ids.count(value) > 1}),
        "duplicate_display_names": sorted({value for value in display_names if display_names.count(value) > 1}),
        "orphan_concepts": orphan,
        "empty_labels": empty,
        "broad_category_invalid": [concept.concept_id for concept in concepts if concept.broad_category_id not in resolver.ancestor_path(concept.concept_id)],
        "unknown_concept_present": any(concept.concept_id.casefold() in {"unknown", "other"} for concept in concepts),
    }


def compatibility(v1: Taxonomy, v2: Taxonomy) -> dict[str, Any]:
    old = {concept.concept_id: concept for concept in v1.concepts}
    new = {concept.concept_id: concept for concept in v2.concepts}
    preserved = sorted(set(old) & set(new))
    return {
        "v1_version": v1.taxonomy_version,
        "v2_version": v2.taxonomy_version,
        "v1_concept_count": len(old),
        "v2_concept_count": len(new),
        "preserved_ids": preserved,
        "added_ids": sorted(set(new) - set(old)),
        "removed_ids": sorted(set(old) - set(new)),
        "parent_changes": [{"concept_id": concept_id, "v1_parent_id": old[concept_id].parent_id, "v2_parent_id": new[concept_id].parent_id} for concept_id in preserved if old[concept_id].parent_id != new[concept_id].parent_id],
    }


def domain_coverage(taxonomy: Taxonomy) -> dict[str, dict[str, Any]]:
    ids = {concept.concept_id for concept in taxonomy.concepts}
    result = {}
    for domain, concept_id in DOMAIN_QUERIES.items():
        result[domain] = {"status": "supported" if concept_id in ids else "not_represented", "concept_id": concept_id}
    return result


def build_report(v1: Taxonomy, v2: Taxonomy) -> dict[str, Any]:
    return {"v1_audit": audit(v1), "v2_audit": audit(v2), "compatibility": compatibility(v1, v2), "domain_coverage": domain_coverage(v2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=V1_PATH)
    parser.add_argument("--v2", type=Path, default=V2_PATH)
    parser.add_argument("--output", type=Path, default=Path("/tmp/eden-taxonomy-v1-v2-report.json"))
    args = parser.parse_args()
    report = build_report(load_path(args.v1), load_path(args.v2))
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
