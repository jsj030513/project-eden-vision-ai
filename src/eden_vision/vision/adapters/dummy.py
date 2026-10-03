"""Deterministic injected candidates, with no classification policy."""

from collections.abc import Iterable

from eden_vision.vision.types import CandidatePrediction, ImageInput, ModelInfo


class DummyVisionModel:
    def __init__(self, predictions: Iterable[CandidatePrediction]) -> None:
        self._predictions = tuple(predictions)
        if any(not isinstance(p, CandidatePrediction) for p in self._predictions):
            raise ValueError("predictions must contain CandidatePrediction values")
        self._loaded = False

    @property
    def info(self) -> ModelInfo:
        return ModelInfo("dummy-injected-candidates", "v1")

    def load(self) -> None:
        self._loaded = True

    def predict(self, image: ImageInput) -> tuple[CandidatePrediction, ...]:
        if not self._loaded:
            raise RuntimeError("DummyVisionModel must be loaded before predict")
        if not isinstance(image, ImageInput):
            raise TypeError("image must be ImageInput")
        return self._predictions

    def close(self) -> None:
        self._loaded = False
