"""Latency-based scorer."""

from __future__ import annotations

from typing import Any, ClassVar

from metrics.base import Metric, MetricResult

_DEFAULT_EXCELLENT_MS = 1000.0
_DEFAULT_GOOD_MS = 3000.0
_DEFAULT_ACCEPTABLE_MS = 5000.0

_SCORE_EXCELLENT = 1.0
_SCORE_GOOD = 0.75
_SCORE_ACCEPTABLE = 0.5
_SCORE_POOR = 0.25


class LatencyScorer(Metric):
    """Scores response latency against configurable millisecond thresholds.

    Reads ``metadata["latency_ms"]`` and buckets it into one of four
    tiers:

    - ``excellent``: ``latency_ms < excellent_ms`` -> score ``1.0``
    - ``good``: ``excellent_ms <= latency_ms < good_ms`` -> score ``0.75``
    - ``acceptable``: ``good_ms <= latency_ms < acceptable_ms`` -> score
      ``0.5``
    - ``poor``: ``latency_ms >= acceptable_ms`` -> score ``0.25``

    Attributes:
        excellent_ms: Upper (exclusive) bound, in milliseconds, of the
            "excellent" tier.
        good_ms: Upper (exclusive) bound, in milliseconds, of the "good"
            tier.
        acceptable_ms: Upper (exclusive) bound, in milliseconds, of the
            "acceptable" tier; latencies at or above this are "poor".
    """

    METRIC_NAME: ClassVar[str] = "latency"

    def __init__(
        self,
        excellent_ms: float = _DEFAULT_EXCELLENT_MS,
        good_ms: float = _DEFAULT_GOOD_MS,
        acceptable_ms: float = _DEFAULT_ACCEPTABLE_MS,
    ) -> None:
        """Initialize the scorer with (optionally custom) tier thresholds.

        Args:
            excellent_ms: Upper (exclusive) bound of the "excellent" tier.
            good_ms: Upper (exclusive) bound of the "good" tier.
            acceptable_ms: Upper (exclusive) bound of the "acceptable"
                tier.

        Raises:
            ValueError: If the thresholds are not strictly increasing
                (``0 < excellent_ms < good_ms < acceptable_ms``).
        """
        if not (0 < excellent_ms < good_ms < acceptable_ms):
            raise ValueError(
                "thresholds must satisfy 0 < excellent_ms < good_ms < acceptable_ms, "
                f"got excellent_ms={excellent_ms}, good_ms={good_ms}, acceptable_ms={acceptable_ms}"
            )
        self.excellent_ms = excellent_ms
        self.good_ms = good_ms
        self.acceptable_ms = acceptable_ms

    def evaluate(self, response: str, reference: str, metadata: dict[str, Any]) -> MetricResult:
        """Score latency reported in ``metadata["latency_ms"]``.

        Args:
            response: The text produced by the system under evaluation.
                Not used by this scorer but accepted to satisfy the
                ``Metric`` interface.
            reference: The ground-truth answer. Not used by this scorer
                but accepted to satisfy the ``Metric`` interface.
            metadata: Must contain ``"latency_ms"``, a non-negative
                number giving the wall-clock latency of the call in
                milliseconds.

        Returns:
            A :class:`MetricResult` whose ``score`` reflects the latency
            tier, and whose ``explanation`` reports the raw latency and
            tier name.

        Raises:
            KeyError: If ``metadata`` does not contain ``"latency_ms"``.
            ValueError: If ``metadata["latency_ms"]`` is negative or not
                numeric.
        """
        if "latency_ms" not in metadata:
            raise KeyError("metadata must contain 'latency_ms'")

        raw_latency = metadata["latency_ms"]
        try:
            latency_ms = float(raw_latency)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata['latency_ms'] must be numeric, got {raw_latency!r}") from exc
        if latency_ms < 0:
            raise ValueError(f"metadata['latency_ms'] must be non-negative, got {latency_ms}")

        tier, score = self._tier_for(latency_ms)

        explanation = (
            f"Latency is {latency_ms:.1f}ms -> tier='{tier}' (excellent<{self.excellent_ms:.0f}ms, "
            f"good<{self.good_ms:.0f}ms, acceptable<{self.acceptable_ms:.0f}ms) -> score={score:.2f}."
        )

        return MetricResult(
            metric_name=self.METRIC_NAME,
            score=score,
            explanation=explanation,
            confidence=1.0,
        )

    def _tier_for(self, latency_ms: float) -> tuple[str, float]:
        """Map a latency value, in milliseconds, to its ``(tier, score)``.

        Args:
            latency_ms: The latency to classify, in milliseconds.

        Returns:
            A ``(tier_name, score)`` tuple.
        """
        if latency_ms < self.excellent_ms:
            return "excellent", _SCORE_EXCELLENT
        if latency_ms < self.good_ms:
            return "good", _SCORE_GOOD
        if latency_ms < self.acceptable_ms:
            return "acceptable", _SCORE_ACCEPTABLE
        return "poor", _SCORE_POOR