"""Contract examples only: no adapter or decision algorithm is exercised."""

from dataclasses import FrozenInstanceError, replace
import json

import pytest
from pydantic import ValidationError

from eden_vision.schemas.classification import ClassificationResponse, ImageInfo, PredictionResponse
from eden_vision.vision.contracts import VisionModel
from eden_vision.vision.types import (
    BoundingBox, CandidatePrediction, ClassificationPrediction, ClassificationResult,
    ClassificationStatus, ConfidenceKind, ImageInput, ModelInfo, PredictionScope, ScoreKind,
)


def flower(**changes):
    values = dict(label="flower", hierarchy=("living_thing", "plant", "flower"),
                  broad_category="plant", detailed_category="flower")
    return ClassificationPrediction(**(values | changes))


def unknown(**changes):
    values = dict(label="UNKNOWN", hierarchy=(), broad_category=None,
                  detailed_category=None, unknown=True)
    return ClassificationPrediction(**(values | changes))


def result(*predictions, status=ClassificationStatus.CLASSIFIED, **changes):
    values = dict(status=status, predictions=predictions, model=ModelInfo("test-model", "r1"),
                  taxonomy_version="eden-v1", policy_version="policy-v1")
    return ClassificationResult(**(values | changes))


def response(value):
    return ClassificationResponse.from_result(value, request_id="test-request",
                                              image=ImageInfo(width=1600, height=1200))


def test_normal_classification_json_round_trip():
    value = response(result(flower()))
    data = json.loads(value.model_dump_json())
    assert data["predictions"][0]["hierarchy"] == ["living_thing", "plant", "flower"]
    assert data["predictions"][0]["confidence"] is None
    assert data["predictions"][0]["confidence_kind"] == "uncalibrated"
    assert data["meta"]["coverage"] == "not_exhaustive"
    assert ClassificationResponse.model_validate_json(value.model_dump_json()) == value


def test_fallback_contract_does_not_include_rejected_rose():
    prediction = flower(fallback=True, fallback_reason="insufficient_specific_evidence")
    data = response(result(prediction)).model_dump(mode="json")["predictions"][0]
    assert data["label"] == "flower"
    assert data["fallback"] is True
    assert data["fallback_reason"]
    assert "rose" not in data["hierarchy"]


def test_unknown_is_a_successful_result_with_no_invented_categories():
    value = response(result(unknown(), status=ClassificationStatus.UNKNOWN))
    data = value.model_dump(mode="json")
    assert data["status"] == "unknown"
    assert data["predictions"][0]["label"] == "UNKNOWN"
    assert data["predictions"][0]["hierarchy"] == []
    assert data["predictions"][0]["broad_category"] is None
    assert data["predictions"][0]["unknown"] is True


def test_multi_label_person_dog_tree():
    predictions = (
        ClassificationPrediction("person", ("living_thing", "person"), "person", None),
        ClassificationPrediction("dog", ("living_thing", "animal", "dog"), "animal", "dog"),
        ClassificationPrediction("tree", ("living_thing", "plant", "tree"), "plant", "tree"),
    )
    value = response(result(*predictions))
    assert [p.label for p in value.predictions] == ["person", "dog", "tree"]
    assert [p.id for p in value.predictions] == ["p1", "p2", "p3"]


def test_region_and_partial_future_compatibility():
    box = BoundingBox(0.1, 0.2, 0.8, 0.9)
    known = flower(scope=PredictionScope.REGION, region_id="r1", bbox=box)
    unresolved = unknown(scope=PredictionScope.REGION, region_id="r2", bbox=box)
    value = response(result(known, unresolved, status=ClassificationStatus.PARTIAL))
    data = value.model_dump(mode="json")
    assert data["status"] == "partial"
    assert data["predictions"][0]["bbox"] == [0.1, 0.2, 0.8, 0.9]
    assert value.predictions[0].to_domain() == known


def test_calibrated_confidence_requires_version():
    prediction = flower(confidence=0.88, confidence_kind=ConfidenceKind.CALIBRATED_PROBABILITY)
    with pytest.raises(ValueError, match="calibration_version"):
        result(prediction)
    assert response(result(prediction, calibration_version="c1")).predictions[0].confidence == 0.88


@pytest.mark.parametrize("score", [-8.0, 0.0, 12.5])
def test_raw_scores_are_not_restricted_to_probability_range(score):
    value = CandidatePrediction("flower", score, ScoreKind.LOGIT)
    assert value.model_score == score
    assert not hasattr(value, "confidence")


@pytest.mark.parametrize("changes", [
    {"confidence": -0.1, "confidence_kind": ConfidenceKind.CALIBRATED_PROBABILITY},
    {"confidence": 1.1, "confidence_kind": ConfidenceKind.CALIBRATED_PROBABILITY},
    {"confidence": float("nan"), "confidence_kind": ConfidenceKind.CALIBRATED_PROBABILITY},
    {"confidence": float("inf"), "confidence_kind": ConfidenceKind.CALIBRATED_PROBABILITY},
    {"confidence": 0.7},
    {"confidence_kind": ConfidenceKind.CALIBRATED_PROBABILITY},
    {"confidence_kind": "invalid"},
    {"scope": "invalid"},
    {"hierarchy": ()},
    {"hierarchy": ("living_thing", "flower", "plant")},
    {"broad_category": "animal"},
    {"detailed_category": "rose"},
    {"fallback": True},
    {"fallback_reason": "unexplained"},
    {"unknown": True},
    {"region_id": "r1"},
    {"scope": PredictionScope.REGION},
])
def test_invalid_domain_predictions(changes):
    with pytest.raises(ValueError):
        flower(**changes)


@pytest.mark.parametrize("box", [(-0.1, 0, 1, 1), (0, 0, 1.1, 1), (0.8, 0, 0.2, 1),
                                 (0, 0.9, 1, 0.2), (float("nan"), 0, 1, 1)])
def test_invalid_boxes(box):
    with pytest.raises(ValueError):
        BoundingBox(*box)


@pytest.mark.parametrize("changes", [
    {"confidence": -0.1, "confidence_kind": "calibrated_probability"},
    {"confidence": 1.1, "confidence_kind": "calibrated_probability"},
    {"scope": "not-a-scope"},
    {"confidence_kind": "probability"},
    {"scope": "region", "region_id": "r1", "bbox": [0.8, 0, 0.2, 1]},
    {"scope": "region", "region_id": "r1", "bbox": [0, 0, 2, 1]},
    {"scope": "region", "region_id": "r1", "bbox": [0, 0, 1]},
    {"label": "UNKNOWN", "unknown": True},
    {"fallback": True},
    {"unexpected": "field"},
])
def test_api_rejects_invalid_prediction_payloads(changes):
    data = response(result(flower())).predictions[0].model_dump(mode="json")
    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(data | changes)


@pytest.mark.parametrize("status", [ClassificationStatus.UNKNOWN, ClassificationStatus.PARTIAL])
def test_status_cannot_disagree_with_predictions(status):
    with pytest.raises(ValueError, match="status"):
        result(flower(), status=status)


@pytest.mark.parametrize("mutation", ["status", "empty", "duplicate", "calibration", "enum"])
def test_response_level_validation(mutation):
    data = response(result(flower())).model_dump(mode="json")
    if mutation == "status":
        data["status"] = "unknown"
    elif mutation == "empty":
        data["predictions"] = []
    elif mutation == "duplicate":
        data["predictions"] *= 2
    elif mutation == "calibration":
        data["predictions"][0].update(confidence=0.8, confidence_kind="calibrated_probability")
    else:
        data["status"] = "error"
    with pytest.raises(ValidationError):
        ClassificationResponse.model_validate(data)


def test_domain_values_are_frozen():
    with pytest.raises(FrozenInstanceError):
        flower().label = "rose"


def test_image_input_contract_without_decoder():
    image = ImageInput(width=1, height=1, pixels=b"\xff\x00\x00")
    assert image.pixels == b"\xff\x00\x00"
    with pytest.raises(ValueError):
        replace(image, pixels=b"short")
    with pytest.raises(ValueError):
        replace(image, width=0)


def test_vision_model_is_only_a_contract():
    with pytest.raises(TypeError):
        VisionModel()


@pytest.mark.parametrize("changes", [{"score_kind": "bad"}, {"model_score": float("nan")},
                                     {"concept_id": "UNKNOWN"}])
def test_invalid_raw_candidate(changes):
    with pytest.raises(ValueError):
        CandidatePrediction(**(dict(concept_id="flower", model_score=2.0, score_kind=ScoreKind.LOGIT) | changes))
