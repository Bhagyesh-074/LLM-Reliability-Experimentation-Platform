"""Pairwise statistical comparison between two models' score distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from scipy import stats


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of a pairwise comparison between two independent samples.

    Attributes:
        mean_diff: ``mean(model_a) - mean(model_b)``, or ``None`` if the
            comparison could not be computed.
        p_value: Two-sided p-value from an independent-samples t-test, or
            ``None`` if the comparison could not be computed.
        significant: ``True`` if ``p_value < 0.05``.
        winner: ``"model_a"`` or ``"model_b"`` if the difference is
            significant, ``"tie"`` if not significant, or
            ``"insufficient_data"`` if either sample had fewer than two
            observations.
        n_a: Number of observations in ``model_a_scores``.
        n_b: Number of observations in ``model_b_scores``.
    """

    mean_diff: Optional[float]
    p_value: Optional[float]
    significant: bool
    winner: str
    n_a: int
    n_b: int


class PairwiseComparator:
    """Compares two models' score distributions using an independent t-test."""

    SIGNIFICANCE_LEVEL: float = 0.05

    def compare(
        self, model_a_scores: List[float], model_b_scores: List[float]
    ) -> ComparisonResult:
        """Compare two independent samples of scores.

        Runs Welch's-adjustable independent t-test (``scipy.stats.ttest_ind``)
        against the two samples and determines which model, if either,
        performs significantly better.

        Args:
            model_a_scores: Scores for model A.
            model_b_scores: Scores for model B.

        Returns:
            A ``ComparisonResult``. If either sample has fewer than two
            observations, a t-test cannot be computed reliably and a
            result with ``winner="insufficient_data"`` (and ``mean_diff``
            / ``p_value`` set to ``None``) is returned instead.
        """
        n_a, n_b = len(model_a_scores), len(model_b_scores)
        if n_a < 2 or n_b < 2:
            return ComparisonResult(
                mean_diff=None,
                p_value=None,
                significant=False,
                winner="insufficient_data",
                n_a=n_a,
                n_b=n_b,
            )

        mean_a = sum(model_a_scores) / n_a
        mean_b = sum(model_b_scores) / n_b
        mean_diff = mean_a - mean_b

        _, p_value = stats.ttest_ind(model_a_scores, model_b_scores)
        p_value = float(p_value)
        significant = p_value < self.SIGNIFICANCE_LEVEL

        if significant and mean_diff > 0:
            winner = "model_a"
        elif significant and mean_diff < 0:
            winner = "model_b"
        else:
            winner = "tie"

        return ComparisonResult(
            mean_diff=mean_diff,
            p_value=p_value,
            significant=significant,
            winner=winner,
            n_a=n_a,
            n_b=n_b,
        )