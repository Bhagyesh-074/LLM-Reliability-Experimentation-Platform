"""Composite scorer combining multiple per-metric results into one score."""

from __future__ import annotations

from typing import Mapping

from metrics.base import MetricResult

_DEFAULT_WEIGHTS: dict[str, float] = {
    "accuracy": 0.35,
    "hallucination": 0.25,
    "instruction": 0.20,
    "safety": 0.10,
    "latency": 0.05,
    "cost": 0.05,
}

_WEIGHT_SUM_TOLERANCE = 1e-6


class CompositeScorer:
    """Combines individual :class:`MetricResult` objects into one score.

    This is a plain aggregator, *not* a :class:`~metrics.base.Metric`
    implementation: it does not run any model or compute any signal
    itself, and it does not implement ``evaluate()``. Callers are expected
    to run each individual scorer (Accuracy, Hallucination, Instruction,
    Safety, Latency, Cost) beforehand and pass their results into
    :meth:`compute`.

    Attributes:
        weights: Mapping of component name to its weight in the weighted
            average. Component names must match the keys of the
            ``results`` mapping passed to :meth:`compute`. Weights must be
            non-negative and sum to ``1.0``.
    """

    #: Default weights, matching FR-5's specified composite breakdown.
    DEFAULT_WEIGHTS: Mapping[str, float] = _DEFAULT_WEIGHTS

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        """Initialize the composite scorer with (optionally custom) weights.

        Args:
            weights: Mapping of component name -> weight. Defaults to
                ``{"accuracy": 0.35, "hallucination": 0.25,
                "instruction": 0.20, "safety": 0.10, "latency": 0.05,
                "cost": 0.05}``.

        Raises:
            ValueError: If ``weights`` is empty, any weight is negative,
                or the weights do not sum to ``1.0`` (within
                floating-point tolerance).
        """
        resolved_weights = dict(weights) if weights is not None else dict(_DEFAULT_WEIGHTS)
        self._validate_weights(resolved_weights)
        self.weights = resolved_weights

    def compute(self, results: Mapping[str, MetricResult]) -> MetricResult:
        """Compute the weighted-average composite score.

        Args:
            results: Mapping of component name (matching the keys of
                :attr:`weights`) to the :class:`MetricResult` produced by
                that component's scorer. May contain extra keys beyond
                those in :attr:`weights`; extras are ignored.

        Returns:
            A :class:`MetricResult` with ``metric_name="composite_score"``,
            whose ``score`` is the weighted average of each component's
            score, whose ``confidence`` is the weighted average of each
            component's confidence, and whose ``explanation`` breaks down
            each component's contribution.

        Raises:
            KeyError: If ``results`` is missing an entry for any
                component named in :attr:`weights`.
        """
        missing = [name for name in self.weights if name not in results]
        if missing:
            raise KeyError(f"results is missing required component(s): {missing}")

        weighted_score = 0.0
        weighted_confidence = 0.0
        breakdown_parts: list[str] = []
        for name, weight in self.weights.items():
            result = results[name]
            weighted_score += weight * result.score
            weighted_confidence += weight * result.confidence
            breakdown_parts.append(f"{name}: score={result.score:.4f} x weight={weight:.2f}")

        weighted_score = max(0.0, min(1.0, weighted_score))
        weighted_confidence = max(0.0, min(1.0, weighted_confidence))

        explanation = (
            "Composite score is the weighted average of component metric scores. "
            f"Breakdown -> {'; '.join(breakdown_parts)}. Composite score={weighted_score:.4f}."
        )

        return MetricResult(
            metric_name="composite_score",
            score=weighted_score,
            explanation=explanation,
            confidence=weighted_confidence,
        )

    @staticmethod
    def _validate_weights(weights: Mapping[str, float]) -> None:
        """Validate that ``weights`` are non-negative and sum to ``1.0``.

        Args:
            weights: The weight mapping to validate.

        Raises:
            ValueError: If ``weights`` is empty, contains a negative
                weight, or does not sum to ``1.0`` within tolerance.
        """
        if not weights:
            raise ValueError("weights must not be empty")
        for name, weight in weights.items():
            if weight < 0:
                raise ValueError(f"weight for {name!r} must be non-negative, got {weight}")

        total = sum(weights.values())
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"weights must sum to 1.0, got {total}")