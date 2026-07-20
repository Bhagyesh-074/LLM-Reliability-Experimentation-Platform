"""Unit tests for EvaluationOrchestrator.

Every external dependency is mocked: the provider, all six scorers, the
composite scorer, the registry services, and the persistence layer. No
DB, no network, no ML model loading. These tests exercise the
orchestrator's control flow: happy path, partial failure, majority
failure, progress emission, and per-row error isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from core.evaluation.config import EvaluationConfig
from core.evaluation.orchestrator import (
    COMPONENT_KEYS,
    EvaluationOrchestrator,
    EvaluationRowResult,
)
from metrics.base import MetricResult
from providers.base import LLMRequest, LLMResponse, ProviderError


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #


def _config() -> EvaluationConfig:
    return EvaluationConfig(
        provider_name="openai",
        model_name="gpt-4o",
        prompt_version_id="pv-1",
        dataset_version_id="dv-1",
        temperature=0.3,
        max_tokens=256,
    )


def _rows(n: int) -> List[Dict[str, Any]]:
    return [
        {
            "question_id": f"q{i}",
            "question": f"Question {i}?",
            "ground_truth": f"Answer {i}.",
            "domain": "medical",
            "difficulty": "easy",
        }
        for i in range(n)
    ]


def _response(text: str = "an answer", latency_ms: float = 500.0) -> LLMResponse:
    return LLMResponse(
        text=text,
        latency_ms=latency_ms,
        token_usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason="stop",
    )


def _metric_result(name: str, score: float) -> MetricResult:
    return MetricResult(metric_name=name, score=score, explanation="mock", confidence=1.0)


def _scorer(score: float) -> MagicMock:
    scorer = MagicMock()
    scorer.evaluate.return_value = _metric_result("mock", score)
    return scorer


def _all_scorers(score: float = 0.9) -> Dict[str, MagicMock]:
    return {key: _scorer(score) for key in COMPONENT_KEYS}


def _prompt_version() -> MagicMock:
    return MagicMock(name="PromptVersionResponse")


def _build_orchestrator(
    *,
    rows: List[Dict[str, Any]],
    provider: MagicMock,
    scorers: Dict[str, Any],
    composite_score: float = 0.88,
    persistence: MagicMock | None = None,
    progress_callback=None,
) -> EvaluationOrchestrator:
    prompt_service = MagicMock()
    prompt_service.get_version_by_id.return_value = _prompt_version()

    benchmark_service = MagicMock()
    benchmark_service.load_dataset_rows.return_value = rows

    composite_scorer = MagicMock()
    composite_scorer.compute.return_value = _metric_result("composite_score", composite_score)

    if persistence is None:
        persistence = MagicMock()
        persistence.save_run.return_value = "persisted-run-id"

    orch = EvaluationOrchestrator(
        config=_config(),
        session=MagicMock(),
        prompt_service=prompt_service,
        benchmark_service=benchmark_service,
        provider=provider,
        scorers=scorers,
        composite_scorer=composite_scorer,
        persistence=persistence,
        progress_callback=progress_callback,
    )
    # RequestBuilder is exercised indirectly; stub it so we don't depend on
    # template internals in these control-flow tests.
    orch.request_builder = MagicMock()
    orch.request_builder.build_request.return_value = LLMRequest(user_prompt="hello")
    return orch


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_happy_path_all_rows_succeed() -> None:
    provider = MagicMock()
    provider.generate.return_value = _response()
    scorers = _all_scorers(score=0.9)
    persistence = MagicMock()
    persistence.save_run.return_value = "persisted-run-id"

    orch = _build_orchestrator(
        rows=_rows(3), provider=provider, scorers=scorers,
        composite_score=0.88, persistence=persistence,
    )

    run = await orch.run()

    assert run.status == "completed"
    assert run.run_id == "persisted-run-id"
    assert len(run.results) == 3
    assert all(r.success for r in run.results)
    # Composite is the mean across successful rows (all 0.88).
    assert run.composite_score == pytest.approx(0.88)
    # Averages present for every component, each equal to the scorer score.
    assert set(run.metric_averages) == set(COMPONENT_KEYS)
    for key in COMPONENT_KEYS:
        assert run.metric_averages[key] == pytest.approx(0.9)

    # Provider called once per row; every scorer called once per row.
    assert provider.generate.call_count == 3
    for scorer in scorers.values():
        assert scorer.evaluate.call_count == 3

    # Persistence invoked in order with the persisted id.
    persistence.save_run.assert_called_once()
    persistence.save_results.assert_called_once()
    persistence.save_metrics.assert_called_once()
    assert persistence.save_results.call_args.args[0] == "persisted-run-id"


@pytest.mark.asyncio
async def test_metadata_includes_expected_format_only_when_present() -> None:
    provider = MagicMock()
    provider.generate.return_value = _response()
    scorers = _all_scorers()

    rows = _rows(2)
    rows[0]["expected_format"] = "json"  # present on row 0 only

    orch = _build_orchestrator(rows=rows, provider=provider, scorers=scorers)
    await orch.run()

    # Inspect the metadata passed to any scorer for each row.
    instruction = scorers["instruction"]
    md_row0 = instruction.evaluate.call_args_list[0].args[2]
    md_row1 = instruction.evaluate.call_args_list[1].args[2]

    assert md_row0["expected_format"] == "json"
    assert "expected_format" not in md_row1
    # Latency + token usage always present for the cost/latency scorers.
    assert md_row0["latency_ms"] == 500.0
    assert md_row0["token_usage"]["total_tokens"] == 30
    assert md_row0["domain"] == "medical"


# --------------------------------------------------------------------------- #
# Partial failure (<= 50%)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_partial_failure_marks_completed_with_errors() -> None:
    # 4 rows, 1 provider failure = 25% failure -> completed_with_errors.
    provider = MagicMock()
    provider.generate.side_effect = [
        _response(),
        ProviderError("upstream 500"),
        _response(),
        _response(),
    ]
    scorers = _all_scorers(score=0.8)
    persistence = MagicMock()
    persistence.save_run.return_value = "run-x"

    orch = _build_orchestrator(
        rows=_rows(4), provider=provider, scorers=scorers,
        composite_score=0.7, persistence=persistence,
    )

    run = await orch.run()

    assert run.status == "completed_with_errors"
    failed = [r for r in run.results if not r.success]
    assert len(failed) == 1
    assert failed[0].row_index == 1
    assert "upstream 500" in (failed[0].error or "")
    # Composite averages only successful rows (3 of them, all 0.7).
    assert run.composite_score == pytest.approx(0.7)
    # Partial results still persisted.
    persistence.save_run.assert_called_once()
    persistence.save_results.assert_called_once()
    persistence.save_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_scorer_error_isolated_to_row() -> None:
    provider = MagicMock()
    provider.generate.return_value = _response()
    scorers = _all_scorers(score=0.9)
    # Safety scorer blows up only on the 2nd call.
    scorers["safety"].evaluate.side_effect = [
        _metric_result("safety", 1.0),
        RuntimeError("scorer boom"),
        _metric_result("safety", 1.0),
    ]

    orch = _build_orchestrator(rows=_rows(3), provider=provider, scorers=scorers)
    run = await orch.run()

    assert run.status == "completed_with_errors"
    assert run.results[1].success is False
    assert "Scoring error" in (run.results[1].error or "")
    # The failed row still recorded its response (scoring failed after generation).
    assert run.results[1].response is not None


# --------------------------------------------------------------------------- #
# Majority failure (> 50%)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_majority_failure_marks_run_failed_but_persists() -> None:
    # 3 rows, 2 provider failures = 66% -> failed.
    provider = MagicMock()
    provider.generate.side_effect = [
        ProviderError("boom 1"),
        _response(),
        ProviderError("boom 2"),
    ]
    scorers = _all_scorers()
    persistence = MagicMock()
    persistence.save_run.return_value = "failed-run"

    orch = _build_orchestrator(
        rows=_rows(3), provider=provider, scorers=scorers, persistence=persistence
    )

    run = await orch.run()

    assert run.status == "failed"
    assert sum(1 for r in run.results if not r.success) == 2
    # Partial results are still persisted even for a failed run.
    persistence.save_run.assert_called_once()
    persistence.save_results.assert_called_once()
    persistence.save_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_all_rows_fail_gives_no_composite() -> None:
    provider = MagicMock()
    provider.generate.side_effect = ProviderError("always down")
    scorers = _all_scorers()

    orch = _build_orchestrator(rows=_rows(2), provider=provider, scorers=scorers)
    run = await orch.run()

    assert run.status == "failed"
    assert run.composite_score is None
    assert run.metric_averages == {}


# --------------------------------------------------------------------------- #
# Progress tracking
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_progress_updates_emitted() -> None:
    provider = MagicMock()
    provider.generate.return_value = _response()
    scorers = _all_scorers()

    updates: List[Dict[str, Any]] = []

    orch = _build_orchestrator(
        rows=_rows(2), provider=provider, scorers=scorers,
        progress_callback=updates.append,
    )
    run = await orch.run()

    # started + one per row + final = 4 updates for 2 rows.
    assert len(updates) == 4
    assert updates[0] == {"current": 0, "total": 2, "status": "started"}
    assert updates[1] == {"current": 1, "total": 2, "status": "ok"}
    assert updates[2] == {"current": 2, "total": 2, "status": "ok"}
    assert updates[-1] == {"current": 2, "total": 2, "status": run.status}


@pytest.mark.asyncio
async def test_progress_marks_failed_rows() -> None:
    provider = MagicMock()
    provider.generate.side_effect = [ProviderError("x"), _response()]
    scorers = _all_scorers()

    updates: List[Dict[str, Any]] = []
    orch = _build_orchestrator(
        rows=_rows(2), provider=provider, scorers=scorers,
        progress_callback=updates.append,
    )
    await orch.run()

    per_row = [u for u in updates if u["current"] in (1, 2) and u["status"] in ("ok", "row_failed")]
    assert per_row[0]["status"] == "row_failed"
    assert per_row[1]["status"] == "ok"


@pytest.mark.asyncio
async def test_faulty_progress_callback_does_not_abort_run() -> None:
    provider = MagicMock()
    provider.generate.return_value = _response()
    scorers = _all_scorers()

    def bad_callback(_: Dict[str, Any]) -> None:
        raise RuntimeError("UI exploded")

    orch = _build_orchestrator(
        rows=_rows(2), provider=provider, scorers=scorers, progress_callback=bad_callback
    )
    run = await orch.run()  # must not raise

    assert run.status == "completed"
    assert len(run.results) == 2


# --------------------------------------------------------------------------- #
# Missing template variable (request never built)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_template_variable_fails_row_without_request() -> None:
    from core.evaluation.request_builder import MissingTemplateVariableError

    provider = MagicMock()
    provider.generate.return_value = _response()
    scorers = _all_scorers()

    orch = _build_orchestrator(rows=_rows(2), provider=provider, scorers=scorers)
    orch.request_builder.build_request.side_effect = [
        MissingTemplateVariableError("name", ["question", "ground_truth"]),
        LLMRequest(user_prompt="ok"),
    ]

    run = await orch.run()

    assert run.results[0].success is False
    assert run.results[0].request is None
    assert run.results[0].response is None
    # Provider only called for the row whose request built successfully.
    assert provider.generate.call_count == 1



# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #


def test_determine_status_thresholds() -> None:
    def row(success: bool) -> EvaluationRowResult:
        return EvaluationRowResult(row_index=0, question_row={}, success=success)

    assert EvaluationOrchestrator._determine_status([]) == "completed"
    assert EvaluationOrchestrator._determine_status([row(True), row(True)]) == "completed"
    # Exactly 50% failure is NOT > 50%, so completed_with_errors.
    assert (
        EvaluationOrchestrator._determine_status([row(True), row(False)])
        == "completed_with_errors"
    )
    # 2/3 > 50% -> failed.
    assert (
        EvaluationOrchestrator._determine_status([row(True), row(False), row(False)])
        == "failed"
    )


def test_aggregate_metric_averages_excludes_failed_rows() -> None:
    good = EvaluationRowResult(
        row_index=0, question_row={}, success=True,
        metric_scores={k: 1.0 for k in COMPONENT_KEYS},
    )
    also_good = EvaluationRowResult(
        row_index=1, question_row={}, success=True,
        metric_scores={k: 0.0 for k in COMPONENT_KEYS},
    )
    bad = EvaluationRowResult(row_index=2, question_row={}, success=False)

    averages = EvaluationOrchestrator._aggregate_metric_averages([good, also_good, bad])
    for key in COMPONENT_KEYS:
        assert averages[key] == pytest.approx(0.5)
