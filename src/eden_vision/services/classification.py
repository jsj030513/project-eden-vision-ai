"""HTTP-independent orchestration. Caller owns model load/close lifecycle."""

from eden_vision.classification.confidence import ConfidencePolicy
from eden_vision.vision.contracts import VisionModel
from eden_vision.vision.types import ClassificationResult, ClassificationStatus, ImageInput


class ClassificationService:
    def __init__(self, model: VisionModel, policy: ConfidencePolicy) -> None:
        self._model = model
        self._policy = policy

    def classify(self, image: ImageInput) -> ClassificationResult:
        # Runtime/model errors propagate; they must not masquerade as UNKNOWN.
        predictions = self._policy.evaluate(self._model.predict(image))
        unknown_count = sum(p.unknown for p in predictions)
        status = (ClassificationStatus.UNKNOWN if unknown_count == len(predictions)
                  else ClassificationStatus.PARTIAL if unknown_count
                  else ClassificationStatus.CLASSIFIED)
        return ClassificationResult(
            status=status, predictions=predictions, model=self._model.info,
            taxonomy_version=self._policy.hierarchy.taxonomy_version,
            policy_version=self._policy.config.version,
        )
