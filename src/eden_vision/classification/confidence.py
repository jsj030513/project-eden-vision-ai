"""Uncalibrated deterministic test policy, NOT validated model thresholds.

Only directly scored concepts may be accepted. Taxonomy ancestors in output
are relationships, not independently inferred probabilities. Siblinghood alone
does not imply exclusivity: the small configured conflict groups below are
explicit test assumptions, scoped to one image or one region. In particular,
person/dog/tree and unconfigured siblings remain independent labels.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from eden_vision.classification.hierarchy import HierarchyResolver
from eden_vision.vision.types import (
    CandidatePrediction, ClassificationPrediction, PredictionScope, ScoreKind,
)


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Test defaults only; score kinds are never silently mixed or converted.

    max_predictions caps accepted labels per scope/region, after deduplication.
    Empty/rejected groups still produce UNKNOWN. Changing these settings should
    be accompanied by a distinct version for reproducible result metadata.
    """

    version: str = "deterministic-test-v1"
    score_kind: ScoreKind = ScoreKind.COSINE_SIMILARITY
    specific_threshold: float = 0.80
    broad_threshold: float = 0.70
    sibling_margin: float = 0.05
    max_predictions: int = 10
    exclusive_sibling_groups: tuple[tuple[str, ...], ...] = (("rose", "tulip"),)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be non-empty")
        if not isinstance(self.score_kind, ScoreKind):
            raise ValueError("score_kind must be ScoreKind")
        for value in (self.specific_threshold, self.broad_threshold, self.sibling_margin):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError("thresholds and margin must be finite numbers")
        if self.broad_threshold > self.specific_threshold or self.sibling_margin < 0:
            raise ValueError("require broad <= specific and non-negative margin")
        if type(self.max_predictions) is not int or self.max_predictions < 1:
            raise ValueError("max_predictions must be a positive integer")
        if not isinstance(self.exclusive_sibling_groups, tuple):
            raise ValueError("exclusive_sibling_groups must be an immutable tuple")
        seen: set[str] = set()
        for group in self.exclusive_sibling_groups:
            if (not isinstance(group, tuple) or len(group) < 2
                    or any(not isinstance(c, str) or not c for c in group)
                    or len(set(group)) != len(group) or seen.intersection(group)):
                raise ValueError("conflict groups must be disjoint tuples of distinct concept IDs")
            seen.update(group)


class ConfidencePolicy:
    def __init__(self, hierarchy: HierarchyResolver, config: PolicyConfig | None = None) -> None:
        self.hierarchy = hierarchy
        self.config = config if config is not None else PolicyConfig()
        for group in self.config.exclusive_sibling_groups:
            parents = {hierarchy.parent(c) for c in group}
            if len(parents) != 1 or None in parents:
                raise ValueError("conflict groups must contain siblings with a shared parent")

    def evaluate(self, candidates: Iterable[CandidatePrediction]) -> tuple[ClassificationPrediction, ...]:
        groups: dict[tuple[PredictionScope, str | None], list[CandidatePrediction]] = {}
        for candidate in candidates:
            if not isinstance(candidate, CandidatePrediction):
                raise TypeError("expected CandidatePrediction")
            self.hierarchy.concept(candidate.concept_id)  # Fail explicitly, even for low scores.
            if candidate.score_kind != self.config.score_kind:
                raise ValueError("candidate score_kind does not match policy configuration")
            key = (candidate.scope, candidate.region_id)
            groups.setdefault(key, []).append(candidate)
        if not groups:
            return (self._unknown(),)
        output = []
        for key in sorted(groups, key=lambda k: (k[0].value, k[1] or "")):
            output.extend(self._evaluate_group(groups[key]))
        return tuple(output)

    def _evaluate_group(self, candidates: list[CandidatePrediction]) -> list[ClassificationPrediction]:
        reference = candidates[0]
        if any(c.bbox != reference.bbox for c in candidates):
            raise ValueError("one region_id must refer to one consistent bbox")
        # Repeated evidence is not added; retain the maximum score per concept.
        scores: dict[str, float] = {}
        for candidate in candidates:
            scores[candidate.concept_id] = max(scores.get(candidate.concept_id, float("-inf")),
                                              candidate.model_score)
        paths = {c: self.hierarchy.ancestor_path(c) for c in scores}
        blocked: set[str] = set()
        for group in self.config.exclusive_sibling_groups:
            present = sorted((c for c in group if c in scores), key=lambda c: (-scores[c], c))
            if len(present) < 2:
                continue
            gap = scores[present[0]] - scores[present[1]]
            # Exact ties are ambiguous even when a caller configures margin=0.
            losers = present if gap == 0 or gap < self.config.sibling_margin else present[1:]
            for concept in scores:
                if any(loser in paths[concept] for loser in losers):
                    blocked.add(concept)
        accepted = {
            c for c, score in scores.items() if c not in blocked and score >= (
                self.config.broad_threshold if self.hierarchy.broad_category(c) == c
                else self.config.specific_threshold
            )
        }
        # Retain the deepest accepted node on each path, within this scope only.
        selected = {c for c in accepted if not any(c in paths[other][:-1] for other in accepted)}
        ranked = sorted(selected, key=lambda c: (-scores[c], c))[:self.config.max_predictions]
        if not ranked:
            return [self._unknown(reference)]
        output = []
        for concept in ranked:
            broad = self.hierarchy.broad_category(concept)
            fallback = any(concept in paths[other][:-1] and other not in accepted for other in scores)
            output.append(ClassificationPrediction(
                label=concept, hierarchy=paths[concept], broad_category=broad,
                detailed_category=None if concept == broad else concept,
                fallback=fallback,
                fallback_reason="insufficient_specific_evidence" if fallback else None,
                scope=reference.scope, region_id=reference.region_id, bbox=reference.bbox,
            ))
        return output

    @staticmethod
    def _unknown(reference: CandidatePrediction | None = None) -> ClassificationPrediction:
        return ClassificationPrediction(
            label="UNKNOWN", hierarchy=(), broad_category=None, detailed_category=None,
            unknown=True,
            scope=reference.scope if reference else PredictionScope.IMAGE,
            region_id=reference.region_id if reference else None,
            bbox=reference.bbox if reference else None,
        )
