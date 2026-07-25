"""Confidence interval calculations for evaluation scores."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import stats


class ConfidenceIntervalCalculator:
    """Computes confidence intervals for the mean of a sample of scores.

    Uses the Student's t-distribution rather than the normal distribution,
    since evaluation runs typically produce small sample sizes where the
    t-distribution's heavier tails give a more honest (wider) interval.
    """

    def compute_ci(
        self, scores: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute a two-sided confidence interval for the mean of ``scores``.

        Args:
            scores: Raw numeric scores.
            confidence: Confidence level in the open interval (0, 1),
                e.g. ``0.95`` for a 95% confidence interval.

        Returns:
            A ``(lower, upper)`` tuple giving the bounds of the interval.
            For a single-element sample, returns ``(score, score)`` since
            no variance can be estimated. When the sample has zero
            variance (all scores identical), also returns
            ``(mean, mean)``.

        Raises:
            ValueError: If ``scores`` is empty, or if ``confidence`` is
                not strictly between 0 and 1.
        """
        if not scores:
            raise ValueError("Cannot compute a confidence interval for an empty sample.")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be strictly between 0 and 1.")

        n = len(scores)
        if n == 1:
            single = float(scores[0])
            return single, single

        arr = np.asarray(scores, dtype=float)
        mean = float(np.mean(arr))
        sample_std = float(np.std(arr, ddof=1))
        sem = sample_std / np.sqrt(n)

        if sem == 0.0:
            return mean, mean

        degrees_of_freedom = n - 1
        t_critical = float(stats.t.ppf((1.0 + confidence) / 2.0, degrees_of_freedom))
        margin = t_critical * sem
        return mean - margin, mean + margin