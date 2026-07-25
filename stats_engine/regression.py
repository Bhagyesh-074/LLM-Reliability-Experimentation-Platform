"""Regression detection between a current score and a historical baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from database.repositories.evaluation_repository import EvaluationRepository


@dataclass(frozen=True)
class RegressionResult:
    """Outcome of comparing a current score against a baseline score.

    Attributes:
        current_score: The score being evaluated.
        baseline_score: The reference score it was compared against.
        pct_change: Signed percent change from baseline to current
            (negative means the score dropped).
        is_regression: ``True`` if ``severity`` is not ``"none"``.
        severity: ``"none"``, ``"minor"``, or ``"major"``.
    """

    current_score: float
    baseline_score: float
    pct_change: float
    is_regression: bool
    severity: str


class RegressionDetector:
    """Flags meaningful score drops relative to a baseline run.

    Formalizes the inline regression check previously embedded in the
    dashboard: a run is considered a regression when its score drops by
    more than ``threshold_pct`` percent relative to the baseline. Drops
    between half the threshold and the full threshold are flagged as
    ``"minor"``; drops beyond the full threshold are ``"major"``.
    """

    def detect(
        self,
        current_score: float,
        baseline_score: float,
        threshold_pct: float = 10.0,
    ) -> RegressionResult:
        """Compare ``current_score`` against ``baseline_score``.

        Args:
            current_score: The score of the run being evaluated.
            baseline_score: The score of the reference/previous run.
            threshold_pct: The percentage drop that constitutes a
                "major" regression. Half of this value is used as the
                "minor" threshold. Must be positive.

        Returns:
            A ``RegressionResult`` describing the percent change and
            severity of any regression.

        Raises:
            ValueError: If ``threshold_pct`` is not positive.
        """
        if threshold_pct <= 0:
            raise ValueError("threshold_pct must be positive.")

        if baseline_score == 0:
            # Percent change is undefined against a zero baseline. Only
            # treat it as a (major) regression if the score moved into
            # negative territory; otherwise there is nothing to compare
            # against, so report no regression.
            if current_score < 0:
                return RegressionResult(
                    current_score=current_score,
                    baseline_score=baseline_score,
                    pct_change=float("-inf"),
                    is_regression=True,
                    severity="major",
                )
            return RegressionResult(
                current_score=current_score,
                baseline_score=baseline_score,
                pct_change=0.0,
                is_regression=False,
                severity="none",
            )

        pct_change = ((current_score - baseline_score) / abs(baseline_score)) * 100.0
        drop_pct = -pct_change  # positive when the score dropped
        minor_threshold = threshold_pct / 2.0

        if drop_pct > threshold_pct:
            severity = "major"
        elif drop_pct > minor_threshold:
            severity = "minor"
        else:
            severity = "none"

        return RegressionResult(
            current_score=current_score,
            baseline_score=baseline_score,
            pct_change=pct_change,
            is_regression=severity != "none",
            severity=severity,
        )

    def detect_for_runs(
        self,
        repository: EvaluationRepository,
        current_run_id: str,
        baseline_run_id: str,
        threshold_pct: float = 10.0,
    ) -> Optional[RegressionResult]:
        """Fetch two runs via ``repository`` and detect regression between them.

        Convenience wrapper around :meth:`detect` for callers that only
        have run IDs (e.g. the dashboard), sparing them from fetching and
        unpacking ``EvaluationRun`` rows themselves.

        Args:
            repository: Repository used to look up each run's composite
                score.
            current_run_id: Primary key of the run being evaluated.
            baseline_run_id: Primary key of the reference run.
            threshold_pct: See :meth:`detect`.

        Returns:
            A ``RegressionResult``, or ``None`` if either run cannot be
            found, or either is missing a ``composite_score``.
        """
        current_run = repository.get_run_with_relations(current_run_id)
        baseline_run = repository.get_run_with_relations(baseline_run_id)

        if current_run is None or baseline_run is None:
            return None
        if current_run.composite_score is None or baseline_run.composite_score is None:
            return None

        return self.detect(
            current_score=current_run.composite_score,
            baseline_score=baseline_run.composite_score,
            threshold_pct=threshold_pct,
        )