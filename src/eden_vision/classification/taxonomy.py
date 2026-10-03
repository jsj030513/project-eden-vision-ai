"""Minimal seed reader and structural validation, not a hierarchy resolver.

Reads the package resource directly; distribution of JSON in built wheels must
be configured in the packaging phase (pyproject.toml is unchanged in Phase 1).
"""

import json
import re
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True, slots=True)
class Concept:
    concept_id: str
    parent_id: str | None
    display_name: str
    broad_category_id: str
    aliases: tuple[str, ...]
    descriptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Taxonomy:
    taxonomy_version: str
    concepts: tuple[Concept, ...]


def _identifier(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]*", value) is not None


def parse_taxonomy(text: str) -> Taxonomy:
    """Validate JSON shape, references, cycles and broad-category ancestry."""
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"taxonomy_version", "concepts"}:
        raise ValueError("taxonomy requires taxonomy_version and concepts")
    version = data["taxonomy_version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("taxonomy_version must be non-empty text")
    if not isinstance(data["concepts"], list) or not data["concepts"]:
        raise ValueError("concepts must be a non-empty list")
    concepts = []
    keys = {"concept_id", "parent_id", "display_name", "broad_category_id", "aliases", "descriptions"}
    for item in data["concepts"]:
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError("concept has missing or unexpected fields")
        if not _identifier(item["concept_id"]) or not _identifier(item["broad_category_id"]):
            raise ValueError("invalid concept or broad category ID")
        if item["parent_id"] is not None and not _identifier(item["parent_id"]):
            raise ValueError("invalid parent ID")
        if not isinstance(item["display_name"], str) or not item["display_name"].strip():
            raise ValueError("display_name must be non-empty text")
        for key in ("aliases", "descriptions"):
            if (not isinstance(item[key], list)
                    or any(not isinstance(v, str) or not v.strip() for v in item[key])):
                raise ValueError(f"{key} must be a list of non-empty text")
        concepts.append(Concept(**{**item, "aliases": tuple(item["aliases"]),
                                  "descriptions": tuple(item["descriptions"])}))
    by_id = {c.concept_id: c for c in concepts}
    if len(by_id) != len(concepts):
        raise ValueError("duplicate concept_id")
    for concept in concepts:
        if concept.parent_id is not None and concept.parent_id not in by_id:
            raise ValueError(f"missing parent: {concept.parent_id}")
        if concept.broad_category_id not in by_id:
            raise ValueError(f"missing broad category: {concept.broad_category_id}")
    for concept in concepts:
        visited: set[str] = set()
        current: str | None = concept.concept_id
        while current is not None:
            if current in visited:
                raise ValueError(f"cycle at {current}")
            visited.add(current)
            current = by_id[current].parent_id
        if concept.broad_category_id not in visited:
            raise ValueError("broad category must be self or ancestor")
    return Taxonomy(version, tuple(concepts))


def load_taxonomy() -> Taxonomy:
    """Load the bundled eden-v1 seed without network or model access."""
    resource = files("eden_vision.resources").joinpath("taxonomy_v1.json")
    return parse_taxonomy(resource.read_text(encoding="utf-8"))
