"""Domain values only: no HTTP, validation framework, or inference runtime.

Hierarchy paths describe accepted concepts, not rejected candidate labels.
These contracts validate consistency; they do not evaluate model evidence.
"""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Literal


class PredictionScope(StrEnum):
    IMAGE = "image"
    REGION = "region"


class ScoreKind(StrEnum):
    COSINE_SIMILARITY = "cosine_similarity"
    LOGIT = "logit"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


class ConfidenceKind(StrEnum):
    UNCALIBRATED = "uncalibrated"
    CALIBRATED_PROBABILITY = "calibrated_probability"


class ClassificationStatus(StrEnum):
    CLASSIFIED = "classified"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class Coverage(StrEnum):
    # No claim of exhaustive object detection is currently supported.
    NOT_EXHAUSTIVE = "not_exhaustive"


class UnknownDecision(StrEnum):
    NOT_APPLIED = "not_applied"
    KNOWN = "known"
    UNKNOWN = "unknown"


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized xyxy coordinates in the oriented source image."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for value in (self.x_min, self.y_min, self.x_max, self.y_max):
            _finite(value, "bbox coordinate")
            if not 0 <= value <= 1:
                raise ValueError("bbox coordinates must be between 0 and 1")
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("bbox minimum must not exceed maximum")


def _region(scope: PredictionScope, region_id: str | None, bbox: BoundingBox | None) -> None:
    if not isinstance(scope, PredictionScope):
        raise ValueError("scope must be a PredictionScope")
    if scope == PredictionScope.IMAGE:
        if region_id is not None or bbox is not None:
            raise ValueError("image predictions cannot contain region information")
    else:
        _text(region_id, "region_id")
        if not isinstance(bbox, BoundingBox):
            raise ValueError("region predictions require a BoundingBox")


@dataclass(frozen=True, slots=True)
class ImageInput:
    """Interleaved RGB8 bytes, row-major, already oriented; no decoder here.

    Adapter-specific resizing and normalization remain the adapter's concern.
    """

    width: int
    height: int
    pixels: bytes
    source_format: Literal["JPEG", "PNG", "WEBP", "HEIF"] | None = None

    def __post_init__(self) -> None:
        if self.source_format not in (None, "JPEG", "PNG", "WEBP", "HEIF"):
            raise ValueError("unsupported source_format")
        if any(type(size) is not int or size <= 0 for size in (self.width, self.height)):
            raise ValueError("image dimensions must be positive integers")
        if not isinstance(self.pixels, bytes) or len(self.pixels) != self.width * self.height * 3:
            raise ValueError("RGB8 pixels must contain width * height * 3 bytes")


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    """Raw adapter output. model_score is never a calibrated confidence."""

    concept_id: str
    model_score: float
    score_kind: ScoreKind
    scope: PredictionScope = PredictionScope.IMAGE
    region_id: str | None = None
    bbox: BoundingBox | None = None

    def __post_init__(self) -> None:
        _text(self.concept_id, "concept_id")
        if self.concept_id == "UNKNOWN":
            raise ValueError("UNKNOWN is a policy outcome, not a model candidate")
        _finite(self.model_score, "model_score")
        if not isinstance(self.score_kind, ScoreKind):
            raise ValueError("score_kind must be a ScoreKind")
        _region(self.scope, self.region_id, self.bbox)


@dataclass(frozen=True, slots=True)
class ClassificationPrediction:
    label: str
    hierarchy: tuple[str, ...]
    broad_category: str | None
    detailed_category: str | None
    confidence: float | None = None
    confidence_kind: ConfidenceKind = ConfidenceKind.UNCALIBRATED
    fallback: bool = False
    fallback_reason: str | None = None
    unknown: bool = False
    scope: PredictionScope = PredictionScope.IMAGE
    region_id: str | None = None
    bbox: BoundingBox | None = None

    def __post_init__(self) -> None:
        _text(self.label, "label")
        _region(self.scope, self.region_id, self.bbox)
        if not isinstance(self.hierarchy, tuple):
            raise ValueError("hierarchy must be an immutable tuple")
        for concept in self.hierarchy:
            _text(concept, "hierarchy concept")
        if type(self.fallback) is not bool or type(self.unknown) is not bool:
            raise ValueError("fallback and unknown must be booleans")
        if not isinstance(self.confidence_kind, ConfidenceKind):
            raise ValueError("confidence_kind must be a ConfidenceKind")
        if self.confidence_kind == ConfidenceKind.UNCALIBRATED:
            if self.confidence is not None:
                raise ValueError("uncalibrated confidence must be null")
        else:
            _finite(self.confidence, "confidence")
            if not 0 <= self.confidence <= 1:
                raise ValueError("confidence must be between 0 and 1")
        if self.fallback:
            _text(self.fallback_reason, "fallback_reason")
        elif self.fallback_reason is not None:
            raise ValueError("fallback_reason requires fallback=true")
        if self.unknown:
            if (self.label != "UNKNOWN" or self.hierarchy or self.broad_category is not None
                    or self.detailed_category is not None or self.confidence is not None
                    or self.fallback):
                raise ValueError("UNKNOWN must not invent categories, confidence, or fallback")
        elif (self.label == "UNKNOWN" or not self.hierarchy
                or "UNKNOWN" in self.hierarchy or self.hierarchy[-1] != self.label
                or len(set(self.hierarchy)) != len(self.hierarchy)
                or self.broad_category not in self.hierarchy):
            raise ValueError("known prediction requires a unique path ending at label and a broad category")
        elif self.detailed_category != (None if self.label == self.broad_category else self.label):
            raise ValueError("detailed_category must be the accepted label below broad_category")


@dataclass(frozen=True, slots=True)
class ModelInfo:
    model_id: str
    model_revision: str

    def __post_init__(self) -> None:
        _text(self.model_id, "model_id")
        _text(self.model_revision, "model_revision")


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """partial means a mix of known and unresolved predictions, not an error."""

    status: ClassificationStatus
    predictions: tuple[ClassificationPrediction, ...]
    model: ModelInfo
    taxonomy_version: str
    policy_version: str
    calibration_version: str | None = None
    coverage: Coverage = Coverage.NOT_EXHAUSTIVE

    def __post_init__(self) -> None:
        if not isinstance(self.status, ClassificationStatus) or not isinstance(self.coverage, Coverage):
            raise ValueError("invalid result enum")
        if not isinstance(self.model, ModelInfo):
            raise ValueError("model must be ModelInfo")
        if (not isinstance(self.predictions, tuple) or not self.predictions
                or any(not isinstance(p, ClassificationPrediction) for p in self.predictions)):
            raise ValueError("predictions must be a non-empty tuple of ClassificationPrediction")
        for name in ("taxonomy_version", "policy_version"):
            _text(getattr(self, name), name)
        if self.calibration_version is not None:
            _text(self.calibration_version, "calibration_version")
        if any(p.confidence_kind == ConfidenceKind.CALIBRATED_PROBABILITY for p in self.predictions):
            if self.calibration_version is None:
                raise ValueError("calibrated predictions require calibration_version")
        unknown_count = sum(p.unknown for p in self.predictions)
        expected = (ClassificationStatus.UNKNOWN if unknown_count == len(self.predictions)
                    else ClassificationStatus.PARTIAL if unknown_count
                    else ClassificationStatus.CLASSIFIED)
        if self.status != expected:
            raise ValueError("status must agree with known/unknown predictions")


@dataclass(frozen=True, slots=True)
class RankingItem:
    """Image-level candidate evidence, never an accepted label or probability."""

    rank: int
    concept_id: str
    display_name: str
    broad_category: str | None
    hierarchy: tuple[str, ...]
    raw_score: float
    score_kind: Literal["raw_image_text_logit"] = "raw_image_text_logit"

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        _text(self.concept_id, "concept_id")
        _text(self.display_name, "display_name")
        _finite(self.raw_score, "raw_score")
        if self.score_kind != "raw_image_text_logit":
            raise ValueError("ranking requires raw_image_text_logit")
        if not isinstance(self.hierarchy, tuple) or not self.hierarchy:
            raise ValueError("ranking requires an immutable non-empty hierarchy")
        for concept in self.hierarchy:
            _text(concept, "hierarchy concept")
        if ("UNKNOWN" in self.hierarchy or self.hierarchy[-1] != self.concept_id
                or len(set(self.hierarchy)) != len(self.hierarchy)):
            raise ValueError("ranking hierarchy must be unique and end at concept_id")
        if self.broad_category is not None:
            _text(self.broad_category, "broad_category")
            if self.broad_category not in self.hierarchy:
                raise ValueError("broad_category must belong to hierarchy")


@dataclass(frozen=True, slots=True)
class RankingResult:
    """Schema 1.1 ranking-only result: neither acceptance nor UNKNOWN is applied.

    Legacy ClassificationResult retains its non-empty decision semantics.
    Equal logits are allowed; their supplied order is preserved, not re-scored.
    """

    ranking: tuple[RankingItem, ...]
    model: ModelInfo
    taxonomy_version: str
    status: Literal["ranked"] = "ranked"
    predictions: tuple[ClassificationPrediction, ...] = ()
    unknown_decision: UnknownDecision = UnknownDecision.NOT_APPLIED
    policy_version: Literal["not_applied"] = "not_applied"
    calibration_version: None = None
    coverage: Coverage = Coverage.NOT_EXHAUSTIVE

    def __post_init__(self) -> None:
        if self.status != "ranked" or self.predictions != ():
            raise ValueError("ranked results cannot contain accepted predictions")
        if (not isinstance(self.unknown_decision, UnknownDecision)
                or self.unknown_decision != UnknownDecision.NOT_APPLIED):
            raise ValueError("ranking-only UNKNOWN decision must be not_applied")
        if self.policy_version != "not_applied" or self.calibration_version is not None:
            raise ValueError("ranking-only results cannot claim policy or calibration")
        if not isinstance(self.model, ModelInfo) or not isinstance(self.coverage, Coverage):
            raise ValueError("invalid model or coverage")
        _text(self.taxonomy_version, "taxonomy_version")
        if (not isinstance(self.ranking, tuple) or not self.ranking
                or any(not isinstance(r, RankingItem) for r in self.ranking)):
            raise ValueError("ranking must be a non-empty tuple of RankingItem")
        if [r.rank for r in self.ranking] != list(range(1, len(self.ranking) + 1)):
            raise ValueError("ranks must be consecutive 1..N in order")
        if len({r.concept_id for r in self.ranking}) != len(self.ranking):
            raise ValueError("ranking concept IDs must be unique")
        if any(a.raw_score < b.raw_score for a, b in zip(self.ranking, self.ranking[1:])):
            raise ValueError("ranking raw scores must be descending")
