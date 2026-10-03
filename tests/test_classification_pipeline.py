"""Phase 2 policy tests use injected raw scores, not image recognition."""

from dataclasses import replace
from itertools import permutations

import pytest

from eden_vision.classification.confidence import ConfidencePolicy, PolicyConfig
from eden_vision.classification.hierarchy import HierarchyResolver, UnknownConceptError
from eden_vision.classification.taxonomy import load_taxonomy
from eden_vision.schemas.classification import ClassificationResponse, ImageInfo
from eden_vision.services.classification import ClassificationService
from eden_vision.vision.adapters.dummy import DummyVisionModel
from eden_vision.vision.contracts import VisionModel
from eden_vision.vision.types import (
    BoundingBox, CandidatePrediction, ClassificationStatus, ConfidenceKind,
    ImageInput, PredictionScope, ScoreKind,
)


@pytest.fixture
def hierarchy():
    return HierarchyResolver(load_taxonomy())


@pytest.fixture
def policy(hierarchy):
    return ConfidencePolicy(hierarchy)


@pytest.fixture
def image():
    return ImageInput(1, 1, b"\x00\x00\x00")


def candidate(concept, score, **kwargs):
    return CandidatePrediction(concept, score, kwargs.pop("score_kind", ScoreKind.COSINE_SIMILARITY), **kwargs)


def classify(policy, image, candidates):
    model: VisionModel = DummyVisionModel(candidates)
    model.load()
    try:
        return ClassificationService(model, policy).classify(image)
    finally:
        model.close()


@pytest.mark.parametrize(("scores", "label", "fallback"), [
    ({"rose": .97, "flower": .92, "plant": .95}, "rose", False),
    ({"rose": .4, "flower": .82, "plant": .91}, "flower", True),
    ({"rose": .58, "tulip": .55, "flower": .87, "plant": .93}, "flower", True),
    ({"rose": .81, "tulip": .80, "flower": .92}, "flower", True),
    ({"rose": .4, "flower": .5, "plant": .91}, "plant", True),
    ({"flower": .88}, "flower", False),
    ({"rose": .97}, "rose", False),
])
def test_exact_and_evidence_based_fallback(policy, image, scores, label, fallback):
    value = classify(policy, image, [candidate(c, s) for c, s in scores.items()])
    assert value.status == ClassificationStatus.CLASSIFIED
    assert len(value.predictions) == 1
    prediction = value.predictions[0]
    assert prediction.label == label
    assert prediction.hierarchy == policy.hierarchy.ancestor_path(label)
    assert prediction.broad_category == "plant"
    assert prediction.fallback is fallback
    assert prediction.fallback_reason == ("insufficient_specific_evidence" if fallback else None)


@pytest.mark.parametrize("scores", [
    {"rose": .4}, {"rose": .4, "flower": .5, "plant": .2}, {},
    {"rose": .81, "tulip": .80}, {"rose": .9, "tulip": .9},
])
def test_unknown_never_synthesizes_parent_evidence(policy, image, scores):
    value = classify(policy, image, [candidate(c, s) for c, s in scores.items()])
    assert value.status == ClassificationStatus.UNKNOWN
    prediction, = value.predictions
    assert prediction.label == "UNKNOWN"
    assert prediction.hierarchy == ()
    assert prediction.unknown is True
    assert prediction.fallback is False
    assert prediction.confidence is None


def test_multi_label_and_unconfigured_siblings_are_independent(policy, image):
    value = classify(policy, image, [candidate(c, .9) for c in ("person", "dog", "tree", "cat")])
    assert {p.label for p in value.predictions} == {"person", "dog", "tree", "cat"}


@pytest.mark.parametrize(("concepts", "expected"), [
    (("rose", "tulip"), "flower"), (("dog", "cat"), "animal"),
    (("rose", "flower"), "flower"), (("rose", "chair"), None),
    (("rose",), "rose"),
])
def test_common_ancestor(hierarchy, concepts, expected):
    assert hierarchy.common_ancestor(*concepts) == expected


def test_paths_parents_and_broad_category(hierarchy):
    assert hierarchy.ancestor_path("rose") == ("living_thing", "plant", "flower", "rose")
    assert hierarchy.parent("rose") == "flower"
    assert hierarchy.parent("flower") == "plant"
    assert hierarchy.parent("plant") == "living_thing"
    assert hierarchy.parent("living_thing") is None
    assert hierarchy.broad_category("chair") == "furniture"
    with pytest.raises(ValueError):
        hierarchy.common_ancestor()


def test_broad_category_comes_from_taxonomy():
    taxonomy = load_taxonomy()
    alternate = replace(taxonomy, concepts=tuple(
        replace(c, broad_category_id="living_thing") if c.concept_id == "rose" else c
        for c in taxonomy.concepts
    ))
    assert HierarchyResolver(alternate).broad_category("rose") == "living_thing"


def test_invalid_concept_is_an_error_even_for_low_score(policy, image):
    with pytest.raises(UnknownConceptError, match="imaginary"):
        classify(policy, image, [candidate("imaginary", -5)])
    with pytest.raises(UnknownConceptError):
        policy.hierarchy.parent("imaginary")


def test_deterministic_under_candidate_permutations_and_duplicates(policy, image):
    candidates = [candidate("rose", .81), candidate("tulip", .8),
                  candidate("flower", .92), candidate("dog", .92)]
    expected = classify(policy, image, candidates)
    for ordering in permutations(candidates):
        assert classify(policy, image, ordering) == expected
    assert classify(policy, image, candidates + [candidate("flower", .2)]) == expected


def test_raw_score_is_never_copied_to_response_confidence(policy, image):
    result = classify(policy, image, [candidate("rose", .97)])
    response = ClassificationResponse.from_result(result, request_id="r1", image=ImageInfo(width=1, height=1))
    assert response.predictions[0].confidence is None
    assert response.predictions[0].confidence_kind == ConfidenceKind.UNCALIBRATED
    assert response.meta.calibration_version is None
    assert response.meta.policy_version == "deterministic-test-v1"
    assert response.meta.model_id == "dummy-injected-candidates"
    assert response.meta.taxonomy_version == "eden-v1"


def test_dummy_lifecycle_and_snapshot(image):
    candidates = [candidate("rose", .3)]
    model = DummyVisionModel(candidates)
    candidates.clear()
    model.close()
    with pytest.raises(RuntimeError):
        model.predict(image)
    model.load()
    model.load()
    assert model.predict(image) == (candidate("rose", .3),)
    assert model.predict(image) == model.predict(image)
    model.close()
    model.close()
    with pytest.raises(RuntimeError):
        model.predict(image)
    model.load()
    assert model.predict(image) == (candidate("rose", .3),)


def test_service_does_not_hide_lifecycle_errors(policy, image):
    service = ClassificationService(DummyVisionModel([]), policy)
    with pytest.raises(RuntimeError):
        service.classify(image)


def test_regions_do_not_compete_or_deduplicate_across_regions(policy, image):
    box = BoundingBox(0, 0, .5, .5)
    candidates = [candidate(c, .9, scope=PredictionScope.REGION, region_id=r, bbox=box)
                  for c, r in (("rose", "r1"), ("tulip", "r2"), ("rose", "r3"))]
    value = classify(policy, image, candidates)
    assert [(p.region_id, p.label) for p in value.predictions] == [("r1", "rose"), ("r2", "tulip"), ("r3", "rose")]


def test_partial_preserves_unresolved_region(policy, image):
    box = BoundingBox(0, 0, 1, 1)
    value = classify(policy, image, [candidate("dog", .9), candidate("rose", .2,
        scope=PredictionScope.REGION, region_id="r1", bbox=box)])
    assert value.status == ClassificationStatus.PARTIAL
    assert value.predictions[1].unknown
    assert value.predictions[1].bbox == box
    assert value.predictions[1].region_id == "r1"


def test_inconsistent_region_geometry_rejected(policy):
    with pytest.raises(ValueError, match="bbox"):
        policy.evaluate([candidate("rose", .9, scope=PredictionScope.REGION, region_id="r1", bbox=box)
                         for box in (BoundingBox(0, 0, 1, 1), BoundingBox(0, 0, .5, .5))])


def test_mixed_score_kinds_are_rejected(policy):
    with pytest.raises(ValueError, match="score_kind"):
        policy.evaluate([candidate("rose", .9), candidate("flower", 10, score_kind=ScoreKind.LOGIT)])


def test_explicit_logit_config_does_not_assume_probability_range(hierarchy):
    policy = ConfidencePolicy(hierarchy, PolicyConfig(version="test-logit-v1", score_kind=ScoreKind.LOGIT,
        specific_threshold=5, broad_threshold=3, sibling_margin=1))
    assert policy.evaluate([candidate("rose", 12, score_kind=ScoreKind.LOGIT)])[0].label == "rose"


def test_thresholds_and_output_limit_are_configurable(hierarchy):
    policy = ConfidencePolicy(hierarchy, PolicyConfig(specific_threshold=.95, broad_threshold=.9,
                                                    max_predictions=1))
    result = policy.evaluate([candidate("rose", .91), candidate("plant", .93), candidate("dog", .99)])
    assert [p.label for p in result] == ["dog"]


def test_threshold_boundary_is_inclusive(hierarchy):
    policy = ConfidencePolicy(hierarchy)
    assert policy.evaluate([candidate("rose", policy.config.specific_threshold)])[0].label == "rose"
    assert policy.evaluate([candidate("plant", policy.config.broad_threshold)])[0].label == "plant"


def test_clear_sibling_winner_and_disabled_conflicts(hierarchy):
    candidates = [candidate("rose", .95), candidate("tulip", .81), candidate("flower", .99)]
    assert [p.label for p in ConfidencePolicy(hierarchy).evaluate(candidates)] == ["rose"]
    independent = ConfidencePolicy(hierarchy, PolicyConfig(exclusive_sibling_groups=()))
    assert {p.label for p in independent.evaluate(candidates)} == {"rose", "tulip"}


def test_zero_margin_does_not_arbitrarily_resolve_exact_ties(hierarchy):
    policy = ConfidencePolicy(hierarchy, PolicyConfig(sibling_margin=0))
    assert policy.evaluate([candidate("rose", .9), candidate("tulip", .9)])[0].unknown


@pytest.mark.parametrize("settings", [
    {"specific_threshold": float("nan")}, {"broad_threshold": 1.1},
    {"sibling_margin": -.1}, {"max_predictions": 0}, {"max_predictions": True},
    {"score_kind": "logit"}, {"version": ""},
    {"exclusive_sibling_groups": (("rose", "rose"),)},
    {"exclusive_sibling_groups": (("rose", "tulip"), ("tulip", "dog"))},
])
def test_invalid_configuration(settings):
    with pytest.raises(ValueError):
        PolicyConfig(**settings)


@pytest.mark.parametrize("group", [("rose", "dog"), ("rose", "missing"), ("food", "document")])
def test_conflicts_require_real_siblings(hierarchy, group):
    with pytest.raises(ValueError):
        ConfidencePolicy(hierarchy, PolicyConfig(exclusive_sibling_groups=(group,)))


def test_custom_conflict_blocks_descendants_of_ambiguous_branch(hierarchy):
    policy = ConfidencePolicy(hierarchy, PolicyConfig(exclusive_sibling_groups=(("tree", "flower"),)))
    predictions = policy.evaluate([candidate("tree", .91), candidate("flower", .92),
                                   candidate("rose", .98), candidate("plant", .9)])
    assert [p.label for p in predictions] == ["plant"]
    assert predictions[0].fallback
