"""Unit tests for the statistics engine: aggregation, confidence intervals,
pairwise comparison, and regression detection."""

from __future__ import annotations

import math
from typing import Optional

import pytest

from stats_engine.aggregation import AggregationService
from stats_engine.confidence import ConfidenceIntervalCalculator
from stats_engine.pairwise import ComparisonResult, PairwiseComparator
from stats_engine.regression import RegressionDetector, RegressionResult


# ---------------------------------------------------------------------------
# AggregationService
# ---------------------------------------------------------------------------


class TestAggregationService:
    def setup_method(self) -> None:
        self.service = AggregationService()

    def test_compute_mean_std_empty_list(self) -> None:
        mean, std = self.service.compute_mean_std([])
        assert mean == 0.0
        assert std == 0.0

    def test_compute_mean_std_single_value(self) -> None:
        mean, std = self.service.compute_mean_std([0.75])
        assert mean == pytest.approx(0.75)
        assert std == 0.0

    def test_compute_mean_std_multiple_values(self) -> None:
        scores = [1.0, 2.0, 3.0, 4.0]
        mean, std = self.service.compute_mean_std(scores)
        assert mean == pytest.approx(2.5)
        # population std of [1,2,3,4] is sqrt(1.25)
        assert std == pytest.approx(math.sqrt(1.25))

    def test_compute_mean_std_identical_values(self) -> None:
        mean, std = self.service.compute_mean_std([0.5, 0.5, 0.5])
        assert mean == pytest.approx(0.5)
        assert std == pytest.approx(0.0)

    def test_aggregate_by_model_groups_and_computes_stats(self) -> None:
        results = [
            {"model_name": "gpt", "score": 0.8},
            {"model_name": "gpt", "score": 0.9},
            {"model_name": "claude", "score": 0.95},
            {"model_name": "claude", "score": 0.85},
        ]
        aggregated = self.service.aggregate_by_model(results)

        assert set(aggregated.keys()) == {"gpt", "claude"}
        assert aggregated["gpt"]["count"] == 2
        assert aggregated["gpt"]["mean"] == pytest.approx(0.85)
        assert aggregated["gpt"]["scores"] == [0.8, 0.9]
        assert aggregated["claude"]["count"] == 2
        assert aggregated["claude"]["mean"] == pytest.approx(0.9)

    def test_aggregate_by_model_skips_incomplete_rows(self) -> None:
        results = [
            {"model_name": "gpt", "score": 0.8},
            {"model_name": "gpt"},  # missing score
            {"score": 0.5},  # missing model_name
        ]
        aggregated = self.service.aggregate_by_model(results)

        assert set(aggregated.keys()) == {"gpt"}
        assert aggregated["gpt"]["count"] == 1

    def test_aggregate_by_model_empty_results(self) -> None:
        assert self.service.aggregate_by_model([]) == {}


# ---------------------------------------------------------------------------
# ConfidenceIntervalCalculator
# ---------------------------------------------------------------------------


class TestConfidenceIntervalCalculator:
    def setup_method(self) -> None:
        self.calculator = ConfidenceIntervalCalculator()

    def test_compute_ci_empty_list_raises(self) -> None:
        with pytest.raises(ValueError):
            self.calculator.compute_ci([])

    def test_compute_ci_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError):
            self.calculator.compute_ci([1.0, 2.0, 3.0], confidence=1.0)
        with pytest.raises(ValueError):
            self.calculator.compute_ci([1.0, 2.0, 3.0], confidence=0.0)

    def test_compute_ci_single_data_point(self) -> None:
        lower, upper = self.calculator.compute_ci([0.42])
        assert lower == pytest.approx(0.42)
        assert upper == pytest.approx(0.42)

    def test_compute_ci_zero_variance(self) -> None:
        lower, upper = self.calculator.compute_ci([0.6, 0.6, 0.6, 0.6])
        assert lower == pytest.approx(0.6)
        assert upper == pytest.approx(0.6)

    def test_compute_ci_bounds_are_symmetric_around_mean(self) -> None:
        scores = [0.7, 0.75, 0.8, 0.85, 0.9]
        lower, upper = self.calculator.compute_ci(scores, confidence=0.95)
        mean = sum(scores) / len(scores)
        assert lower < mean < upper
        assert (mean - lower) == pytest.approx(upper - mean)

    def test_compute_ci_wider_for_higher_confidence(self) -> None:
        scores = [0.7, 0.75, 0.8, 0.85, 0.9]
        lower_90, upper_90 = self.calculator.compute_ci(scores, confidence=0.90)
        lower_99, upper_99 = self.calculator.compute_ci(scores, confidence=0.99)
        assert (upper_99 - lower_99) > (upper_90 - lower_90)


# ---------------------------------------------------------------------------
# PairwiseComparator
# ---------------------------------------------------------------------------


class TestPairwiseComparator:
    def setup_method(self) -> None:
        self.comparator = PairwiseComparator()

    def test_compare_insufficient_data_model_a(self) -> None:
        result = self.comparator.compare([0.5], [0.5, 0.6, 0.7])
        assert isinstance(result, ComparisonResult)
        assert result.winner == "insufficient_data"
        assert result.mean_diff is None
        assert result.p_value is None
        assert result.significant is False

    def test_compare_insufficient_data_model_b(self) -> None:
        result = self.comparator.compare([0.5, 0.6, 0.7], [])
        assert result.winner == "insufficient_data"

    def test_compare_identical_distributions_is_tie(self) -> None:
        scores = [0.8, 0.82, 0.79, 0.81, 0.80]
        result = self.comparator.compare(scores, list(scores))
        assert result.mean_diff == pytest.approx(0.0)
        assert result.significant is False
        assert result.winner == "tie"

    def test_compare_significant_difference_model_a_wins(self) -> None:
        model_a_scores = [0.9, 0.92, 0.91, 0.93, 0.89, 0.90, 0.94]
        model_b_scores = [0.5, 0.52, 0.48, 0.51, 0.49, 0.50, 0.47]
        result = self.comparator.compare(model_a_scores, model_b_scores)

        assert result.mean_diff is not None and result.mean_diff > 0
        assert result.p_value is not None and result.p_value < 0.05
        assert result.significant is True
        assert result.winner == "model_a"
        assert result.n_a == 7
        assert result.n_b == 7

    def test_compare_significant_difference_model_b_wins(self) -> None:
        model_a_scores = [0.5, 0.52, 0.48, 0.51, 0.49, 0.50, 0.47]
        model_b_scores = [0.9, 0.92, 0.91, 0.93, 0.89, 0.90, 0.94]
        result = self.comparator.compare(model_a_scores, model_b_scores)

        assert result.mean_diff is not None and result.mean_diff < 0
        assert result.significant is True
        assert result.winner == "model_b"

    def test_compare_no_significant_difference_close_scores(self) -> None:
        model_a_scores = [0.80, 0.81, 0.79, 0.82, 0.78]
        model_b_scores = [0.80, 0.79, 0.81, 0.80, 0.79]
        result = self.comparator.compare(model_a_scores, model_b_scores)

        assert result.significant is False
        assert result.winner == "tie"


# ---------------------------------------------------------------------------
# RegressionDetector
# ---------------------------------------------------------------------------


class _StubRun:
    def __init__(self, composite_score: Optional[float]) -> None:
        self.composite_score = composite_score


class _StubRepository:
    """Minimal stand-in for EvaluationRepository, keyed by run_id."""

    def __init__(self, runs: dict) -> None:
        self._runs = runs

    def get_run_with_relations(self, run_id: str):
        return self._runs.get(run_id)


class TestRegressionDetector:
    def setup_method(self) -> None:
        self.detector = RegressionDetector()

    def test_detect_no_regression_on_improvement(self) -> None:
        result = self.detector.detect(current_score=0.95, baseline_score=0.90)
        assert isinstance(result, RegressionResult)
        assert result.is_regression is False
        assert result.severity == "none"
        assert result.pct_change > 0

    def test_detect_no_regression_small_drop(self) -> None:
        # 2% drop, well under the default 5% minor threshold (10% / 2)
        result = self.detector.detect(current_score=0.882, baseline_score=0.90)
        assert result.severity == "none"
        assert result.is_regression is False

    def test_detect_minor_regression(self) -> None:
        # ~7% drop: above minor threshold (5%), below major threshold (10%)
        result = self.detector.detect(current_score=0.837, baseline_score=0.90)
        assert result.severity == "minor"
        assert result.is_regression is True

    def test_detect_major_regression(self) -> None:
        # 20% drop: above the 10% major threshold
        result = self.detector.detect(current_score=0.72, baseline_score=0.90)
        assert result.severity == "major"
        assert result.is_regression is True
        assert result.pct_change == pytest.approx(-20.0)

    def test_detect_custom_threshold(self) -> None:
        # 12% drop is "none" against a 30% threshold, "major" against a 10% one
        lenient = self.detector.detect(
            current_score=0.792, baseline_score=0.90, threshold_pct=30.0
        )
        strict = self.detector.detect(
            current_score=0.792, baseline_score=0.90, threshold_pct=10.0
        )
        assert lenient.severity == "none"
        assert strict.severity == "major"

    def test_detect_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            self.detector.detect(current_score=0.5, baseline_score=0.6, threshold_pct=0)
        with pytest.raises(ValueError):
            self.detector.detect(current_score=0.5, baseline_score=0.6, threshold_pct=-5)

    def test_detect_zero_baseline_non_negative_current(self) -> None:
        result = self.detector.detect(current_score=0.5, baseline_score=0.0)
        assert result.severity == "none"
        assert result.is_regression is False

    def test_detect_zero_baseline_negative_current(self) -> None:
        result = self.detector.detect(current_score=-0.1, baseline_score=0.0)
        assert result.severity == "major"
        assert result.is_regression is True
        assert result.pct_change == float("-inf")

    def test_detect_for_runs_computes_regression(self) -> None:
        repository = _StubRepository(
            {
                "run-current": _StubRun(composite_score=0.72),
                "run-baseline": _StubRun(composite_score=0.90),
            }
        )
        result = self.detector.detect_for_runs(
            repository, current_run_id="run-current", baseline_run_id="run-baseline"
        )
        assert result is not None
        assert result.severity == "major"

    def test_detect_for_runs_missing_run_returns_none(self) -> None:
        repository = _StubRepository({"run-baseline": _StubRun(composite_score=0.90)})
        result = self.detector.detect_for_runs(
            repository, current_run_id="does-not-exist", baseline_run_id="run-baseline"
        )
        assert result is None

    def test_detect_for_runs_missing_composite_score_returns_none(self) -> None:
        repository = _StubRepository(
            {
                "run-current": _StubRun(composite_score=None),
                "run-baseline": _StubRun(composite_score=0.90),
            }
        )
        result = self.detector.detect_for_runs(
            repository, current_run_id="run-current", baseline_run_id="run-baseline"
        )
        assert result is None