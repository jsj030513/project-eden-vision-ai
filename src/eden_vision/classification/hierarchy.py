"""Taxonomy relationships only; no evidence or fallback decisions."""

from dataclasses import asdict
import json

from eden_vision.classification.taxonomy import Concept, Taxonomy, parse_taxonomy


class UnknownConceptError(ValueError):
    """Adapter/configuration referenced a concept outside the taxonomy."""


class HierarchyResolver:
    def __init__(self, taxonomy: Taxonomy) -> None:
        # Also validate Taxonomy instances constructed without the JSON loader.
        self._taxonomy = parse_taxonomy(json.dumps(asdict(taxonomy)))
        self._concepts = {c.concept_id: c for c in self._taxonomy.concepts}

    @property
    def taxonomy_version(self) -> str:
        return self._taxonomy.taxonomy_version

    def concept(self, concept_id: str) -> Concept:
        try:
            return self._concepts[concept_id]
        except KeyError:
            raise UnknownConceptError(f"unknown taxonomy concept: {concept_id}") from None

    def parent(self, concept_id: str) -> str | None:
        return self.concept(concept_id).parent_id

    def ancestor_path(self, concept_id: str) -> tuple[str, ...]:
        """Root-to-concept path, including the concept itself."""
        path = []
        current: str | None = concept_id
        while current is not None:
            path.append(current)
            current = self.parent(current)
        return tuple(reversed(path))

    def common_ancestor(self, *concept_ids: str) -> str | None:
        """Deepest shared node, including self; None for separate roots."""
        if not concept_ids:
            raise ValueError("common_ancestor requires at least one concept")
        paths = [self.ancestor_path(c) for c in concept_ids]
        common = None
        for level in zip(*paths):
            if len(set(level)) != 1:
                break
            common = level[0]
        return common

    def broad_category(self, concept_id: str) -> str:
        return self.concept(concept_id).broad_category_id
