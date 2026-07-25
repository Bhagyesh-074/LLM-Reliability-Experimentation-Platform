"""tests/unit/test_mlflow_tracker.py

Unit tests for ``core.evaluation.mlflow_tracker``.

The real ``mlflow`` module is patched out everywhere it's referenced
(``core.evaluation.mlflow_tracker.mlflow``), so nothing here touches disk,
network, or an actual ``mlruns/`` directory -- every assertion is against
call arguments made to the mock.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from core.evaluation.mlflow_tracker import (
    DEFAULT_TRACKING_URI,
    MLflowTracker,
    _is_local_file_uri,
    _load_tracking_uri,
    build_experiment_name,
    resolve_benchmark_name,
)


class _FakeConfig:
    """Minimal stand-in for ``EvaluationConfig`` -- only the attributes
    ``MLflowTracker`` actually reads."""

    def __init__(
        self,
        provider_name: str = "ollama",
        model_name: str = "llama3",
        temperature: Optional[float] = 0.2,
        max_tokens: int = 512,
        prompt_version_id: str = "pv-1",
        dataset_version_id: str = "ds-1",
        benchmark_name: Optional[str] = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_version_id = prompt_version_id
        self.dataset_version_id = dataset_version_id
        self.benchmark_name = benchmark_name


def _make_run(run_id: str = "mlflow-run-abc") -> SimpleNamespace:
    """Build a fake object matching MLflow's ``ActiveRun.info.run_id`` shape."""
    return SimpleNamespace(info=SimpleNamespace(run_id=run_id))


@pytest.fixture()
def mlflow_mock():
    """Patch the ``mlflow`` module referenced inside ``mlflow_tracker``."""
    with patch("core.evaluation.mlflow_tracker.mlflow") as mocked:
        mocked.start_run.return_value = _make_run()
        yield mocked


@pytest.fixture()
def tracker(mlflow_mock: MagicMock) -> MLflowTracker:
    """A tracker pointed at an explicit URI so no YAML config file is read."""
    return MLflowTracker(tracking_uri="sqlite:///:memory:")


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------


def test_resolve_benchmark_name_prefers_explicit_field() -> None:
    config = _FakeConfig(benchmark_name="reasoning-suite", dataset_version_id="ds-42")
    assert resolve_benchmark_name(config) == "reasoning-suite"


def test_resolve_benchmark_name_falls_back_to_dataset_version_id() -> None:
    config = _FakeConfig(benchmark_name=None, dataset_version_id="ds-42")
    assert resolve_benchmark_name(config) == "ds-42"


def test_build_experiment_name() -> None:
    config = _FakeConfig(benchmark_name="reasoning-suite")
    assert build_experiment_name(config) == "llm-eval-reasoning-suite"


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("mlruns/", True),
        ("./mlruns", True),
        ("/abs/path/mlruns", True),
        ("file:///abs/path/mlruns", True),
        ("sqlite:///./mlflow.db", False),
        ("http://localhost:5000", False),
        ("https://mlflow.example.com", False),
        ("databricks", False),
    ],
)
def test_is_local_file_uri(uri: str, expected: bool) -> None:
    assert _is_local_file_uri(uri) is expected


def test_load_tracking_uri_reads_yaml(tmp_path) -> None:
    config_file = tmp_path / "default.yaml"
    config_file.write_text("mlflow:\n  tracking_uri: './custom-mlruns'\n")
    assert _load_tracking_uri(str(config_file)) == "./custom-mlruns"


def test_load_tracking_uri_missing_file_falls_back(tmp_path) -> None:
    missing = tmp_path / "does-not-exist.yaml"
    assert _load_tracking_uri(str(missing)) == DEFAULT_TRACKING_URI


def test_load_tracking_uri_missing_key_falls_back(tmp_path) -> None:
    config_file = tmp_path / "default.yaml"
    config_file.write_text("app:\n  name: 'x'\n")
    assert _load_tracking_uri(str(config_file)) == DEFAULT_TRACKING_URI


def test_load_tracking_uri_malformed_yaml_falls_back(tmp_path) -> None:
    config_file = tmp_path / "default.yaml"
    config_file.write_text(":::not: valid: yaml:::")
    assert _load_tracking_uri(str(config_file)) == DEFAULT_TRACKING_URI


# --------------------------------------------------------------------------
# __init__
# --------------------------------------------------------------------------


def test_init_sets_tracking_uri(mlflow_mock: MagicMock) -> None:
    MLflowTracker(tracking_uri="sqlite:///./mlflow.db")
    mlflow_mock.set_tracking_uri.assert_called_once_with("sqlite:///./mlflow.db")


def test_init_reads_uri_from_config_when_not_given(mlflow_mock: MagicMock, tmp_path) -> None:
    config_file = tmp_path / "default.yaml"
    config_file.write_text("mlflow:\n  tracking_uri: './from-yaml'\n")

    tracker = MLflowTracker(config_path=str(config_file))

    assert tracker.tracking_uri == "./from-yaml"
    mlflow_mock.set_tracking_uri.assert_called_once_with("./from-yaml")


def test_init_sets_file_store_opt_out_env_var_for_local_uri(mlflow_mock: MagicMock, monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)

    MLflowTracker(tracking_uri="./mlruns")

    assert os.environ.get("MLFLOW_ALLOW_FILE_STORE") == "true"


def test_init_does_not_set_env_var_for_remote_uri(mlflow_mock: MagicMock, monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)

    MLflowTracker(tracking_uri="sqlite:///./mlflow.db")

    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


def test_init_swallows_set_tracking_uri_failure(mlflow_mock: MagicMock) -> None:
    mlflow_mock.set_tracking_uri.side_effect = RuntimeError("boom")
    # Must not raise.
    MLflowTracker(tracking_uri="sqlite:///./mlflow.db")


# --------------------------------------------------------------------------
# start_run
# --------------------------------------------------------------------------


def test_start_run_returns_mlflow_run_id(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    mlflow_mock.start_run.return_value = _make_run("run-123")

    run_id = tracker.start_run("llm-eval-bench", "my-run")

    assert run_id == "run-123"
    mlflow_mock.set_experiment.assert_called_once_with("llm-eval-bench")
    mlflow_mock.start_run.assert_called_once_with(run_name="my-run")


def test_start_run_returns_none_on_failure(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    mlflow_mock.set_experiment.side_effect = RuntimeError("backend unreachable")

    run_id = tracker.start_run("llm-eval-bench", "my-run")

    assert run_id is None


def test_start_run_failure_makes_later_calls_no_ops(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    mlflow_mock.start_run.side_effect = RuntimeError("backend unreachable")

    assert tracker.start_run("llm-eval-bench", "my-run") is None

    tracker.log_params(_FakeConfig())
    tracker.log_metrics({"accuracy": 0.9})
    tracker.log_artifact("results.csv")
    tracker.end_run("completed")

    mlflow_mock.log_params.assert_not_called()
    mlflow_mock.log_metrics.assert_not_called()
    mlflow_mock.log_artifact.assert_not_called()
    mlflow_mock.end_run.assert_not_called()


# --------------------------------------------------------------------------
# log_params
# --------------------------------------------------------------------------


def test_log_params_logs_all_six_fields(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")
    config = _FakeConfig(
        provider_name="anthropic",
        model_name="claude-sonnet-5",
        temperature=0.3,
        max_tokens=1024,
        prompt_version_id="pv-7",
        dataset_version_id="ds-9",
        benchmark_name="reasoning-suite",
    )

    tracker.log_params(config)

    logged: Dict[str, Any] = mlflow_mock.log_params.call_args.args[0]
    assert logged == {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "temperature": 0.3,
        "max_tokens": 1024,
        "prompt_version": "pv-7",
        "benchmark": "reasoning-suite",
    }


def test_log_params_stringifies_none_values(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")
    config = _FakeConfig(temperature=None)

    tracker.log_params(config)

    logged: Dict[str, Any] = mlflow_mock.log_params.call_args.args[0]
    assert logged["temperature"] == "none"


def test_log_params_is_noop_without_active_run(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.log_params(_FakeConfig())
    mlflow_mock.log_params.assert_not_called()


def test_log_params_swallows_mlflow_error(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")
    mlflow_mock.log_params.side_effect = RuntimeError("boom")

    # Must not raise.
    tracker.log_params(_FakeConfig())


# --------------------------------------------------------------------------
# log_metrics
# --------------------------------------------------------------------------


def test_log_metrics_logs_component_averages_and_composite(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")

    tracker.log_metrics(
        {"accuracy": 0.9, "hallucination": 0.8, "instruction": 0.95, "safety": 1.0},
        composite_score=0.91,
    )

    logged: Dict[str, float] = mlflow_mock.log_metrics.call_args.args[0]
    assert logged == {
        "accuracy": 0.9,
        "hallucination": 0.8,
        "instruction": 0.95,
        "safety": 1.0,
        "composite_score": 0.91,
    }


def test_log_metrics_without_composite_score(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")

    tracker.log_metrics({"accuracy": 0.9})

    logged: Dict[str, float] = mlflow_mock.log_metrics.call_args.args[0]
    assert logged == {"accuracy": 0.9}


def test_log_metrics_empty_dict_and_no_composite_is_noop(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")

    tracker.log_metrics({})

    mlflow_mock.log_metrics.assert_not_called()


def test_log_metrics_drops_none_values(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")

    tracker.log_metrics({"accuracy": 0.9, "cost": None})

    logged: Dict[str, float] = mlflow_mock.log_metrics.call_args.args[0]
    assert logged == {"accuracy": 0.9}


def test_log_metrics_is_noop_without_active_run(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.log_metrics({"accuracy": 0.9})
    mlflow_mock.log_metrics.assert_not_called()


def test_log_metrics_swallows_mlflow_error(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")
    mlflow_mock.log_metrics.side_effect = RuntimeError("boom")

    # Must not raise.
    tracker.log_metrics({"accuracy": 0.9})


# --------------------------------------------------------------------------
# log_artifact
# --------------------------------------------------------------------------


def test_log_artifact_delegates_to_mlflow(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")

    tracker.log_artifact("results.csv")

    mlflow_mock.log_artifact.assert_called_once_with("results.csv")


def test_log_artifact_is_noop_without_active_run(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.log_artifact("results.csv")
    mlflow_mock.log_artifact.assert_not_called()


def test_log_artifact_swallows_mlflow_error(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")
    mlflow_mock.log_artifact.side_effect = RuntimeError("boom")

    # Must not raise.
    tracker.log_artifact("results.csv")


# --------------------------------------------------------------------------
# end_run
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected_mlflow_status",
    [
        ("completed", "FINISHED"),
        ("completed_with_errors", "FINISHED"),
        ("failed", "FAILED"),
        ("FINISHED", "FINISHED"),
        ("FAILED", "FAILED"),
        ("KILLED", "KILLED"),
        ("some_unknown_status", "FINISHED"),
    ],
)
def test_end_run_maps_status(
    tracker: MLflowTracker, mlflow_mock: MagicMock, status: str, expected_mlflow_status: str
) -> None:
    tracker.start_run("llm-eval-bench", "my-run")

    tracker.end_run(status)

    mlflow_mock.end_run.assert_called_once_with(status=expected_mlflow_status)


def test_end_run_is_noop_without_active_run(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.end_run("completed")
    mlflow_mock.end_run.assert_not_called()


def test_end_run_deactivates_run_even_on_failure(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")
    mlflow_mock.end_run.side_effect = RuntimeError("boom")

    tracker.end_run("completed")  # must not raise

    # A second call is a no-op: the run is considered inactive even
    # though mlflow.end_run() itself raised.
    tracker.end_run("completed")
    mlflow_mock.end_run.assert_called_once()


def test_end_run_twice_only_calls_mlflow_once(tracker: MLflowTracker, mlflow_mock: MagicMock) -> None:
    tracker.start_run("llm-eval-bench", "my-run")

    tracker.end_run("completed")
    tracker.end_run("completed")

    mlflow_mock.end_run.assert_called_once()


# --------------------------------------------------------------------------
# No real mlruns/ directory is ever created by these tests
# --------------------------------------------------------------------------


def test_no_real_mlruns_directory_created(
    mlflow_mock: MagicMock, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tracker = MLflowTracker(tracking_uri="sqlite:///:memory:")

    tracker.start_run("llm-eval-bench", "my-run")
    tracker.log_params(_FakeConfig())
    tracker.log_metrics({"accuracy": 0.9})
    tracker.end_run("completed")

    assert list(tmp_path.iterdir()) == []