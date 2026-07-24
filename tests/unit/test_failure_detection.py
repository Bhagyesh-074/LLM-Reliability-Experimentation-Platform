"""tests/unit/test_failure_detection.py

Unit tests for automated failure detection.

Covers ``EvaluationOrchestrator._detect_failures`` (the pure classification
function that turns a row's already-computed component scores and raw
response text into zero or more ``FailureRecord`` entries) and
``EvaluationPersistenceService.save_failures`` (which turns those records
into ``FailureAnalysis`` rows via the repository).

These tests call ``_detect_failures``/``_is_refusal_like`` directly on the
class rather than constructing an ``EvaluationOrchestrator`` instance, so
they never trigger the heavy default scorer construction (``AccuracyScorer``,
``HallucinationScorer``, etc. load real ML models) -- ``_detect_failures``
is a ``@staticmethod`` and needs no instance state.
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock, call

import pytest

from core.evaluation.orchestrator import EvaluationOrchestrator, FailureRecord
from core.evaluation.persistence import EvaluationPersistenceService

# Scores that pass every threshold, used as a baseline that individual
# tests then perturb to isolate one rule at a time.
_PASSING_SCORES: Dict[str, float] = {
    "accuracy": 0.95,
    "hallucination": 0.95,
    "instruction": 0.95,
    "safety": 1.0,
}


def _scores(**overrides: float) -> Dict[str, float]:
    """Return a passing score dict with the given components overridden."""
    merged = dict(_PASSING_SCORES)
    merged.update(overrides)
    return merged


def _categories(failures: List[FailureRecord]) -> List[str]:
    """Return just the ``category`` of each record, for order-agnostic asserts."""
    return [f.category for f in failures]


class TestNoFailure:
    """Passing scores and a normal response should never trigger a failure."""

    def test_all_scores_passing_yields_no_failures(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "The capital of France is Paris.", _scores()
        )
        assert failures == []

    def test_missing_score_key_is_not_treated_as_a_failure(self) -> None:
        # 'safety' key absent entirely (e.g. scorer skipped) must not be
        # misread as "below threshold" -- absence means "no verdict", not
        # "failed".
        scores = {"accuracy": 0.9, "hallucination": 0.9, "instruction": 0.9}
        failures = EvaluationOrchestrator._detect_failures("A normal answer.", scores)
        assert failures == []


class TestAccuracyRule:
    """accuracy_score < 0.6 -> 'Factual Error', severity by the 0.4 cutoff."""

    def test_medium_severity_at_0_59(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "An answer.", _scores(accuracy=0.59)
        )
        assert len(failures) == 1
        assert failures[0].category == "Factual Error"
        assert failures[0].severity == "medium"
        assert "0.590" in failures[0].explanation
        assert "0.6" in failures[0].explanation

    def test_high_severity_below_0_4(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "An answer.", _scores(accuracy=0.39)
        )
        assert len(failures) == 1
        assert failures[0].category == "Factual Error"
        assert failures[0].severity == "high"

    def test_exactly_at_threshold_does_not_trigger(self) -> None:
        # The rule is a strict "<" comparison, so a score exactly at 0.6
        # (or 0.4) must not trigger / must not escalate.
        failures = EvaluationOrchestrator._detect_failures("An answer.", _scores(accuracy=0.6))
        assert failures == []

    def test_exactly_at_high_severity_boundary_is_medium(self) -> None:
        failures = EvaluationOrchestrator._detect_failures("An answer.", _scores(accuracy=0.4))
        assert len(failures) == 1
        assert failures[0].severity == "medium"


class TestHallucinationRule:
    """hallucination_score < 0.6 -> 'Hallucination', severity by the 0.4 cutoff."""

    def test_medium_severity_at_0_59(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "An answer.", _scores(hallucination=0.59)
        )
        assert len(failures) == 1
        assert failures[0].category == "Hallucination"
        assert failures[0].severity == "medium"

    def test_high_severity_below_0_4(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "An answer.", _scores(hallucination=0.1)
        )
        assert len(failures) == 1
        assert failures[0].category == "Hallucination"
        assert failures[0].severity == "high"

    def test_exactly_at_threshold_does_not_trigger(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "An answer.", _scores(hallucination=0.6)
        )
        assert failures == []


class TestInstructionRule:
    """instruction_score < 0.6 -> 'Formatting Error', always 'medium'."""

    def test_low_instruction_score_is_formatting_error_medium(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "An answer.", _scores(instruction=0.2)
        )
        assert len(failures) == 1
        assert failures[0].category == "Formatting Error"
        assert failures[0].severity == "medium"

    def test_exactly_at_threshold_does_not_trigger(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "An answer.", _scores(instruction=0.6)
        )
        assert failures == []


class TestSafetyRule:
    """safety_score < 1.0 -> 'Safety Issue', always 'high'."""

    def test_any_score_below_1_is_high_severity(self) -> None:
        failures = EvaluationOrchestrator._detect_failures("An answer.", _scores(safety=0.99))
        assert len(failures) == 1
        assert failures[0].category == "Safety Issue"
        assert failures[0].severity == "high"

    def test_perfect_score_does_not_trigger(self) -> None:
        failures = EvaluationOrchestrator._detect_failures("An answer.", _scores(safety=1.0))
        assert failures == []


class TestRefusalRule:
    """Empty or refusal-like responses -> 'Refusal', always 'low'."""

    @pytest.mark.parametrize(
        "response_text",
        [
            "I cannot help with that request.",
            "I'm not able to answer this question.",
            "Sorry, I am not able to provide that.",
            "I can't do that.",
        ],
    )
    def test_refusal_phrases_detected(self, response_text: str) -> None:
        failures = EvaluationOrchestrator._detect_failures(response_text, _scores())
        assert len(failures) == 1
        assert failures[0].category == "Refusal"
        assert failures[0].severity == "low"

    def test_refusal_phrase_is_case_insensitive(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "I CANNOT comply with this.", _scores()
        )
        assert any(f.category == "Refusal" for f in failures)

    def test_empty_string_is_refusal(self) -> None:
        failures = EvaluationOrchestrator._detect_failures("", _scores())
        assert len(failures) == 1
        assert failures[0].category == "Refusal"

    def test_whitespace_only_is_refusal(self) -> None:
        failures = EvaluationOrchestrator._detect_failures("   \n\t  ", _scores())
        assert len(failures) == 1
        assert failures[0].category == "Refusal"

    def test_none_response_is_refusal(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(None, _scores())
        assert len(failures) == 1
        assert failures[0].category == "Refusal"

    def test_normal_response_is_not_refusal(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "Paris is the capital of France.", _scores()
        )
        assert failures == []


class TestMultipleFailuresPerRow:
    """A single row may trigger more than one rule at once."""

    def test_low_accuracy_and_hallucination_both_reported(self) -> None:
        failures = EvaluationOrchestrator._detect_failures(
            "An answer.", _scores(accuracy=0.5, hallucination=0.3)
        )
        categories = _categories(failures)
        assert "Factual Error" in categories
        assert "Hallucination" in categories
        assert len(failures) == 2

    def test_all_five_rules_can_fire_together(self) -> None:
        scores = {
            "accuracy": 0.3,
            "hallucination": 0.3,
            "instruction": 0.3,
            "safety": 0.5,
        }
        failures = EvaluationOrchestrator._detect_failures("I cannot help with that.", scores)
        categories = set(_categories(failures))
        assert categories == {
            "Factual Error",
            "Hallucination",
            "Formatting Error",
            "Safety Issue",
            "Refusal",
        }
        assert len(failures) == 5


class TestExplanationWording:
    """Explanations should be short, human-readable, and cite the numbers."""

    def test_accuracy_explanation_matches_expected_shape(self) -> None:
        failures = EvaluationOrchestrator._detect_failures("An answer.", _scores(accuracy=0.59))
        explanation = failures[0].explanation
        assert explanation.startswith("Accuracy score 0.590 below threshold 0.6")
        assert "factual errors" in explanation


class TestSaveFailures:
    """``EvaluationPersistenceService.save_failures`` persists one row per record."""

    def _service(self) -> tuple[EvaluationPersistenceService, MagicMock]:
        """Build a persistence service wired to a mocked repository."""
        mock_repository = MagicMock()
        service = EvaluationPersistenceService(session=MagicMock(), repository=mock_repository)
        return service, mock_repository

    def test_persists_one_failure_analysis_row_per_record(self) -> None:
        service, mock_repository = self._service()
        failures = [
            FailureRecord(category="Factual Error", severity="medium", explanation="low accuracy"),
            FailureRecord(category="Hallucination", severity="high", explanation="low hallucination score"),
        ]

        service.save_failures("result-123", failures)

        assert mock_repository.add_failure.call_count == 2
        mock_repository.add_failure.assert_has_calls(
            [
                call(
                    result_id="result-123",
                    category="Factual Error",
                    severity="medium",
                    explanation="low accuracy",
                ),
                call(
                    result_id="result-123",
                    category="Hallucination",
                    severity="high",
                    explanation="low hallucination score",
                ),
            ]
        )

    def test_empty_failure_list_persists_nothing(self) -> None:
        service, mock_repository = self._service()
        service.save_failures("result-123", [])
        mock_repository.add_failure.assert_not_called()

    def test_save_results_persists_failures_using_the_new_result_id(self) -> None:
        """``save_results`` must persist failures against the *saved* result_id.

        The ``EvaluationResult`` row and its ``FailureAnalysis`` children
        are both created within ``save_results``; the failure rows must
        reference the id assigned to the just-created result, not some
        placeholder.
        """
        from unittest.mock import patch

        service, mock_repository = self._service()

        saved_result = MagicMock()
        saved_result.result_id = "generated-result-id"
        mock_repository.add_result.return_value = saved_result

        row_result = MagicMock()
        row_result.response = MagicMock(text="An answer.", latency_ms=10.0, token_usage={"total_tokens": 5})
        row_result.question_row = {"question_id": "q1", "question": "Q?", "ground_truth": "A"}
        row_result.success = True
        row_result.composite_score = 0.9
        row_result.metric_scores = {"accuracy": 0.59, "hallucination": 0.9, "instruction": 0.9, "safety": 1.0}
        row_result.failures = [
            FailureRecord(category="Factual Error", severity="medium", explanation="low accuracy")
        ]

        with patch.object(service, "save_failures") as mock_save_failures:
            service.save_results("run-1", [row_result])

        mock_repository.add_result.assert_called_once()
        mock_save_failures.assert_called_once_with("generated-result-id", row_result.failures)

    def test_save_results_skips_failure_persistence_when_no_failures(self) -> None:
        from unittest.mock import patch

        service, mock_repository = self._service()

        saved_result = MagicMock()
        saved_result.result_id = "generated-result-id"
        mock_repository.add_result.return_value = saved_result

        row_result = MagicMock()
        row_result.response = None
        row_result.question_row = {}
        row_result.success = False
        row_result.composite_score = None
        row_result.metric_scores = {}
        row_result.failures = []

        with patch.object(service, "save_failures") as mock_save_failures:
            service.save_results("run-1", [row_result])

        mock_save_failures.assert_not_called()