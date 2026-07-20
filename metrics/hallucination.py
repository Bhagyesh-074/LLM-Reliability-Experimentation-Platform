"""Hallucination scorer based on Natural Language Inference (NLI)."""

from __future__ import annotations

from typing import Any, ClassVar, Optional, Protocol

from transformers import pipeline

from metrics.base import Metric, MetricResult

_DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

_ENTAILMENT = "entailment"
_NEUTRAL = "neutral"
_CONTRADICTION = "contradiction"

_LABEL_TO_SCORE: dict[str, float] = {
    _ENTAILMENT: 1.0,
    _NEUTRAL: 0.5,
    _CONTRADICTION: 0.0,
}


class NLIPipeline(Protocol):
    """Structural type for the callable returned by transformers' ``pipeline``.

    Only the subset of the ``TextClassificationPipeline`` call interface
    that :class:`HallucinationScorer` relies on is captured here, so test
    doubles do not need to subclass any transformers internals.
    """

    def __call__(self, inputs: dict[str, str], top_k: Optional[int] = ...) -> Any:
        ...


class HallucinationScorer(Metric):
    """Scores whether a response is hallucinated relative to ground truth.

    Uses a cross-encoder Natural Language Inference (NLI) model to check
    whether the ``response`` is *entailed* by the ``reference`` (treated
    as the premise / ground truth, with ``response`` as the hypothesis).
    Responses that are entailed are considered faithful; responses that
    contradict the reference are considered hallucinated; responses that
    are merely neutral (unrelated to or unproven by the reference) fall in
    between.

    The NLI pipeline is loaded once per instance (in ``__init__``) and
    reused across all subsequent calls to :meth:`evaluate`, since model
    loading is the dominant cost and must not be repeated per evaluation.

    Attributes:
        model_name: Name of the transformers NLI model used to classify
            the (reference, response) pair. Defaults to
            ``"cross-encoder/nli-deberta-v3-small"``.
    """

    METRIC_NAME: ClassVar[str] = "hallucination"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        nli_pipeline: NLIPipeline | None = None,
    ) -> None:
        """Initialize the scorer and load the NLI pipeline.

        Args:
            model_name: transformers model identifier to load into a
                ``text-classification`` pipeline when ``nli_pipeline`` is
                not supplied.
            nli_pipeline: An already-instantiated NLI pipeline (or test
                double implementing :class:`NLIPipeline`). Primarily
                intended for dependency injection (e.g. in tests), so the
                real model does not need to be downloaded/loaded. When
                omitted, a pipeline is loaded from ``model_name``.
        """
        self.model_name = model_name
        self._pipeline: NLIPipeline = (
            nli_pipeline
            if nli_pipeline is not None
            else pipeline("text-classification", model=model_name, top_k=None)
        )

    def evaluate(self, response: str, reference: str, metadata: dict[str, Any]) -> MetricResult:
        """Score whether ``response`` is entailed by ``reference``.

        The ``reference`` is treated as the NLI premise (ground truth) and
        ``response`` as the hypothesis under test.

        Args:
            response: The text produced by the system under evaluation,
                used as the NLI hypothesis.
            reference: The ground-truth answer, used as the NLI premise.
            metadata: Additional evaluation context. Not used by this
                scorer but accepted to satisfy the ``Metric`` interface.

        Returns:
            A :class:`MetricResult` where ``score`` is ``1.0`` for a
            top-ranked "entailment" verdict, ``0.5`` for "neutral", and
            ``0.0`` for "contradiction", and ``explanation`` reports the
            predicted label and its confidence. ``confidence`` is the
            model's predicted probability for the top label.

        Raises:
            ValueError: If ``response`` or ``reference`` is empty or
                whitespace-only, or if the NLI pipeline returns no usable
                predictions.
        """
        if not response or not response.strip():
            raise ValueError("response must be a non-empty string")
        if not reference or not reference.strip():
            raise ValueError("reference must be a non-empty string")

        raw_predictions = self._pipeline({"text": reference, "text_pair": response}, top_k=None)
        predictions = self._normalize_predictions(raw_predictions)

        top_label, top_confidence = max(predictions.items(), key=lambda item: item[1])
        score = _LABEL_TO_SCORE.get(top_label, 0.5)

        explanation = (
            f"NLI model ({self.model_name}) predicts '{top_label}' with confidence "
            f"{top_confidence:.4f} for response given reference as premise "
            f"(entailment={_LABEL_TO_SCORE[_ENTAILMENT]:.1f}, "
            f"neutral={_LABEL_TO_SCORE[_NEUTRAL]:.1f}, "
            f"contradiction={_LABEL_TO_SCORE[_CONTRADICTION]:.1f} mapping) -> score={score:.4f}."
        )

        return MetricResult(
            metric_name=self.METRIC_NAME,
            score=score,
            explanation=explanation,
            confidence=top_confidence,
        )

    @staticmethod
    def _normalize_predictions(raw_predictions: Any) -> dict[str, float]:
        """Normalize pipeline output into a ``{lowercase_label: score}`` dict.

        ``transformers`` text-classification pipelines called with
        ``top_k=None`` return either a flat list of ``{"label", "score"}``
        dicts, or (for a single input) a list-of-lists with one inner
        list. This normalizes both shapes and lowercases labels so that
        model variants using e.g. ``"ENTAILMENT"`` vs ``"entailment"`` are
        handled uniformly.

        Args:
            raw_predictions: The raw output of the NLI pipeline call.

        Returns:
            A mapping of lowercased NLI label to its predicted
            probability.

        Raises:
            ValueError: If the pipeline output is empty or malformed.
        """
        if not raw_predictions:
            raise ValueError("NLI pipeline returned no predictions")

        # Unwrap a single-input batch: [[{...}, {...}, {...}]] -> [{...}, ...]
        if isinstance(raw_predictions[0], list):
            raw_predictions = raw_predictions[0]

        predictions: dict[str, float] = {}
        for item in raw_predictions:
            label = str(item["label"]).lower()
            predictions[label] = float(item["score"])

        if not predictions:
            raise ValueError("NLI pipeline returned no usable label/score pairs")

        return predictions