"""Pydantic v2 wire contracts and an explicit domain-to-response boundary."""

from dataclasses import asdict
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eden_vision.vision.types import (
    BoundingBox,
    ClassificationPrediction,
    ClassificationResult,
    ClassificationStatus,
    ConfidenceKind,
    Coverage,
    ModelInfo,
    PredictionScope,
    RankingItem,
    RankingResult,
    UnknownDecision,
)

type NonEmptyText = Annotated[str, Field(min_length=1, pattern=r"\S")]
type Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
type BoxCoordinates = tuple[Coordinate, Coordinate, Coordinate, Coordinate]


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageInfo(_Schema):
    """Dimensions after orientation correction, before model resizing."""

    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)


class PredictionResponse(_Schema):
    id: NonEmptyText
    scope: PredictionScope = PredictionScope.IMAGE
    region_id: NonEmptyText | None = None
    bbox: BoxCoordinates | None = None
    label: NonEmptyText
    broad_category: NonEmptyText | None
    detailed_category: NonEmptyText | None
    hierarchy: tuple[NonEmptyText, ...]
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False, strict=True)
    confidence_kind: ConfidenceKind = ConfidenceKind.UNCALIBRATED
    fallback: bool = Field(default=False, strict=True)
    fallback_reason: NonEmptyText | None = None
    unknown: bool = Field(default=False, strict=True)

    def to_domain(self) -> ClassificationPrediction:
        data = self.model_dump(exclude={"id", "bbox"})
        return ClassificationPrediction(**data, bbox=BoundingBox(*self.bbox) if self.bbox is not None else None)

    @model_validator(mode="after")
    def consistent_prediction(self) -> Self:
        self.to_domain()
        return self


class ClassificationMeta(_Schema):
    model_id: NonEmptyText
    model_revision: NonEmptyText
    taxonomy_version: NonEmptyText
    policy_version: NonEmptyText
    calibration_version: NonEmptyText | None = None
    coverage: Coverage = Coverage.NOT_EXHAUSTIVE


class ClassificationResponse(_Schema):
    request_id: NonEmptyText
    schema_version: Literal["1.0"] = "1.0"
    status: ClassificationStatus
    image: ImageInfo
    predictions: tuple[PredictionResponse, ...] = Field(min_length=1)
    meta: ClassificationMeta

    @model_validator(mode="after")
    def consistent_result(self) -> Self:
        if len({p.id for p in self.predictions}) != len(self.predictions):
            raise ValueError("prediction IDs must be unique within the response")
        ClassificationResult(
            status=self.status,
            predictions=tuple(p.to_domain() for p in self.predictions),
            model=ModelInfo(self.meta.model_id, self.meta.model_revision),
            taxonomy_version=self.meta.taxonomy_version,
            policy_version=self.meta.policy_version,
            calibration_version=self.meta.calibration_version,
            coverage=self.meta.coverage,
        )
        return self

    @classmethod
    def from_result(
        cls, result: ClassificationResult, *, request_id: str, image: ImageInfo,
    ) -> Self:
        predictions = []
        for index, prediction in enumerate(result.predictions, start=1):
            data = asdict(prediction)
            box = prediction.bbox
            data["bbox"] = None if box is None else (box.x_min, box.y_min, box.x_max, box.y_max)
            predictions.append(PredictionResponse(id=f"p{index}", **data))
        return cls(
            request_id=request_id,
            status=result.status,
            image=image,
            predictions=tuple(predictions),
            meta=ClassificationMeta(
                model_id=result.model.model_id,
                model_revision=result.model.model_revision,
                taxonomy_version=result.taxonomy_version,
                policy_version=result.policy_version,
                calibration_version=result.calibration_version,
                coverage=result.coverage,
            ),
        )


class RankingItemResponse(_Schema):
    """Raw model ranking; deliberately has no confidence or unknown fields."""

    rank: int = Field(ge=1, strict=True)
    concept_id: NonEmptyText
    display_name: NonEmptyText
    broad_category: NonEmptyText | None
    hierarchy: tuple[NonEmptyText, ...] = Field(min_length=1)
    raw_score: float = Field(allow_inf_nan=False, strict=True)
    score_kind: Literal["raw_image_text_logit"] = "raw_image_text_logit"

    def to_domain(self) -> RankingItem:
        return RankingItem(**self.model_dump())

    @model_validator(mode="after")
    def consistent_item(self) -> Self:
        self.to_domain()
        return self


class RankingMeta(ClassificationMeta):
    policy_version: Literal["not_applied"] = "not_applied"
    calibration_version: None = None


class RankingResponse(_Schema):
    """1.1 adds ranking-only output; it does not relax the 1.0 parser."""

    request_id: NonEmptyText
    schema_version: Literal["1.1"] = "1.1"
    status: Literal["ranked"] = "ranked"
    image: ImageInfo
    predictions: tuple[PredictionResponse, ...] = Field(default=(), max_length=0)
    ranking: tuple[RankingItemResponse, ...] = Field(min_length=1)
    unknown_decision: Literal[UnknownDecision.NOT_APPLIED] = UnknownDecision.NOT_APPLIED
    meta: RankingMeta

    def to_domain(self) -> RankingResult:
        return RankingResult(
            ranking=tuple(r.to_domain() for r in self.ranking),
            model=ModelInfo(self.meta.model_id, self.meta.model_revision),
            taxonomy_version=self.meta.taxonomy_version,
            status=self.status,
            predictions=tuple(p.to_domain() for p in self.predictions),
            unknown_decision=UnknownDecision(self.unknown_decision),
            policy_version=self.meta.policy_version,
            calibration_version=self.meta.calibration_version,
            coverage=self.meta.coverage,
        )

    @model_validator(mode="after")
    def consistent_result(self) -> Self:
        self.to_domain()
        return self

    @classmethod
    def from_result(cls, result: RankingResult, *, request_id: str, image: ImageInfo) -> Self:
        return cls(
            request_id=request_id, image=image,
            ranking=tuple(RankingItemResponse(**asdict(r)) for r in result.ranking),
            meta=RankingMeta(model_id=result.model.model_id, model_revision=result.model.model_revision,
                             taxonomy_version=result.taxonomy_version, policy_version=result.policy_version,
                             calibration_version=result.calibration_version, coverage=result.coverage),
        )


type VisionResponse = Annotated[ClassificationResponse | RankingResponse, Field(discriminator="schema_version")]


def response_from_result(
    result: ClassificationResult | RankingResult, *, request_id: str, image: ImageInfo,
) -> ClassificationResponse | RankingResponse:
    """Choose wire version from the domain result, never from a request option."""
    if isinstance(result, RankingResult):
        return RankingResponse.from_result(result, request_id=request_id, image=image)
    if isinstance(result, ClassificationResult):
        return ClassificationResponse.from_result(result, request_id=request_id, image=image)
    raise TypeError("unsupported classification result type")
