"""Synchronous adapter boundary; scheduling belongs to the future service."""

from typing import Protocol

from eden_vision.vision.types import CandidatePrediction, ClassificationResult, ImageInput, ModelInfo, RankingResult


class ClassificationHandler(Protocol):
    """HTTP-facing result boundary; a handler returns decisions OR raw ranking.

    This does not change VisionModel.predict or add a runtime implementation.
    """

    def classify(self, image: ImageInput) -> ClassificationResult | RankingResult: ...


class VisionModel(Protocol):
    """A loaded model produces raw candidates, never policy decisions.

    load prepares explicitly provisioned local resources; it must not implicitly
    download weights. predict requires successful load. close releases resources
    and must be safe after partial loading or repeated calls. Adapter failures
    raise exceptions: UNKNOWN is a classification outcome, not a runtime error.
    Implementations need not be thread-safe; callers serialize use by default.
    Candidate vocabularies and model-specific prompts belong to adapter setup.
    """

    @property
    def info(self) -> ModelInfo: ...

    def load(self) -> None: ...

    def predict(self, image: ImageInput) -> tuple[CandidatePrediction, ...]: ...

    def close(self) -> None: ...
