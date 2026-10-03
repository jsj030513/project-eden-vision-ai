import hashlib
import json
from importlib.resources import files
from pathlib import Path

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.classification.taxonomy import parse_taxonomy
from experiments.audit_taxonomy import audit, build_report, load_path


V1_PATH = Path("src/eden_vision/resources/taxonomy_v1.json")
V2_PATH = Path("src/eden_vision/resources/taxonomy_v2.json")


def test_v2_parses_and_has_expected_static_shape():
    taxonomy = load_path(V2_PATH)
    report = audit(taxonomy)
    assert taxonomy.taxonomy_version == "eden-v2"
    assert report["concept_count"] == 78
    assert report["maximum_hierarchy_depth"] == 4
    assert report["duplicate_ids"] == []
    assert report["duplicate_display_names"] == []
    assert report["orphan_concepts"] == []
    assert report["broad_category_invalid"] == []
    assert report["unknown_concept_present"] is False


def test_v1_is_unchanged_and_all_ids_are_preserved():
    digest = hashlib.sha256(V1_PATH.read_bytes()).hexdigest()
    assert digest == "0480e225e96e30e5c9ad26250377dad3eeb13f0fa9de78e654aa0256a2026a99"
    v1, v2 = load_path(V1_PATH), load_path(V2_PATH)
    comparison = build_report(v1, v2)["compatibility"]
    assert comparison["v1_concept_count"] == 25
    assert len(comparison["preserved_ids"]) == 25
    assert comparison["removed_ids"] == []
    assert comparison["parent_changes"] == []


def test_v2_hierarchy_resolver_uses_existing_and_new_paths():
    resolver = HierarchyResolver(load_path(V2_PATH))
    assert resolver.ancestor_path("dog") == ("living_thing", "animal", "dog")
    assert resolver.ancestor_path("rose") == ("living_thing", "plant", "flower", "rose")
    assert resolver.ancestor_path("car") == ("artifact", "vehicle", "car")
    assert resolver.ancestor_path("sofa") == ("artifact", "furniture", "sofa")
    assert resolver.ancestor_path("mountain") == ("natural_feature", "mountain")


def test_v2_has_no_attribute_only_concepts():
    taxonomy = load_path(V2_PATH)
    ids = {concept.concept_id for concept in taxonomy.concepts}
    assert not ids & {"red", "blue", "large", "small", "wooden", "metal", "old", "new", "unknown", "other"}
