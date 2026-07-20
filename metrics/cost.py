"""Token-cost-based scorer."""

from __future__ import annotations

from typing import Any, ClassVar

from metrics.base import Metric, MetricResult

_DEFAULT_EXCELLENT_TOKENS = 500
_DEFAULT_GOOD_TOKENS = 1000
_DEFAULT_ACCEPTABLE_TOKENS = 2000

_SCORE_EXCELLENT = 1.0
_SCORE_GOOD = 0.75
_SCORE_ACCEPTABLE = 0.5
_SCORE_POOR = 0.25


class CostScorer(Metric):
    """Scores token cost against configurable token-count thresholds.

    Reads the total token count from
    ``metadata["token_usage"]["total_tokens"]`` and buckets it into one of
    four tiers:

    - ``excellent``: ``total_tokens < excellent_tokens`` -> score ``1.0``
    - ``good``: ``excellent_tokens <= total_tokens < good_tokens`` ->
      score ``0.75``
    - ``acceptable``: ``good_tokens <= total_tokens < acceptable_tokens``
      -> score ``0.5``
    - ``poor``: ``total_tokens >= acceptable_tokens`` -> score ``0.25``

    Attributes:
        excellent_tokens: Upper (exclusive) bound of the "excellent" tier.
        good_tokens: Upper (exclusive) bound of the "good" tier.
        acceptable_tokens: Upper (exclusive) bound of the "acceptable"
            tier; token counts at or above this are "poor".
    """

    METRIC_NAME: ClassVar[str] = "cost"

    def __init__(
        self,
        excellent_tokens: int = _DEFAULT_EXCELLENT_TOKENS,
        good_tokens: int = _DEFAULT_GOOD_TOKENS,
        acceptable_tokens: int = _DEFAULT_ACCEPTABLE_TOKENS,
    ) -> None:
        """Initialize the scorer with (optionally custom) tier thresholds.

        Args:
            excellent_tokens: Upper (exclusive) bound of the "excellent"
                tier.
            good_tokens: Upper (exclusive) bound of the "good" tier.
            acceptable_tokens: Upper (exclusive) bound of the "acceptable"
                tier.

        Raises:
            ValueError: If the thresholds are not strictly increasing
                (``0 < excellent_tokens < good_tokens < acceptable_tokens``).
        """
        if not (0 < excellent_tokens < good_tokens < acceptable_tokens):
            raise ValueError(
                "thresholds must satisfy 0 < excellent_tokens < good_tokens < acceptable_tokens, "
                f"got excellent_tokens={excellent_tokens}, good_tokens={good_tokens}, "
                f"acceptable_tokens={acceptable_tokens}"
            )
        self.excellent_tokens = excellent_tokens
        self.good_tokens = good_tokens
        self.acceptable_tokens = acceptable_tokens

    def evaluate(self, response: str, reference: str, metadata: dict[str, Any]) -> MetricResult:
        """Score token cost reported in ``metadata["token_usage"]``.

        Args:
            response: The text produced by the system under evaluation.
                Not used by this scorer but accepted to satisfy the
                ``Metric`` interface.
            reference: The ground-truth answer. Not used by this scorer
                but accepted to satisfy the ``Metric`` interface.
            metadata: Must contain ``"token_usage"``, a dict containing a
                ``"total_tokens"`` key with a non-negative integer count
                (matching ``LLMResponse.token_usage``).

        Returns:
            A :class:`MetricResult` whose ``score`` reflects the cost
            tier, and whose ``explanation`` reports the raw token count
            and tier name.

        Raises:
            KeyError: If ``metadata`` does not contain ``"token_usage"``,
                or ``metadata["token_usage"]`` does not contain
                ``"total_tokens"``.
            ValueError: If the total token count is negative or not
                numeric.
        """
        if "token_usage" not in metadata:
            raise KeyError("metadata must contain 'token_usage'")

        token_usage = metadata["token_usage"]
        if not isinstance(token_usage, dict) or "total_tokens" not in token_usage:
            raise KeyError("metadata['token_usage'] must be a dict containing 'total_tokens'")

        raw_total = token_usage["total_tokens"]
        try:
            total_tokens = int(raw_total)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"metadata['token_usage']['total_tokens'] must be numeric, got {raw_total!r}"
            ) from exc
        if total_tokens < 0:
            raise ValueError(f"metadata['token_usage']['total_tokens'] must be non-negative, got {total_tokens}")

        tier, score = self._tier_for(total_tokens)

        explanation = (
            f"Total token usage is {total_tokens} tokens -> tier='{tier}' "
            f"(excellent<{self.excellent_tokens}, good<{self.good_tokens}, "
            f"acceptable<{self.acceptable_tokens}) -> score={score:.2f}."
        )

        return MetricResult(
            metric_name=self.METRIC_NAME,
            score=score,
            explanation=explanation,
            confidence=1.0,
        )

    def _tier_for(self, total_tokens: int) -> tuple[str, float]:
        """Map a total token count to its ``(tier, score)``.

        Args:
            total_tokens: The token count to classify.

        Returns:
            A ``(tier_name, score)`` tuple.
        """
        if total_tokens < self.excellent_tokens:
            return "excellent", _SCORE_EXCELLENT
        if total_tokens < self.good_tokens:
            return "good", _SCORE_GOOD
        if total_tokens < self.acceptable_tokens:
            return "acceptable", _SCORE_ACCEPTABLE
        return "poor", _SCORE_POOR