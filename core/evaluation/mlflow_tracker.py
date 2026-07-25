"""core/evaluation/mlflow_tracker.py

MLflow experiment-tracking service for the evaluation pipeline.

``MLflowTracker`` wraps the (global, thread-local) MLflow client behind a
small instance API that mirrors one evaluation run's lifecycle: start a
run, log its config as params, log its aggregate metrics once scoring
finishes, optionally attach artifacts (e.g. a results CSV), then end the
run.

MLflow here is observability, not the critical path: every public method
catches and logs its own failures rather than raising, so a broken or
unreachable tracking backend never aborts an evaluation run (mirrors the
orchestrator's own robustness contract, where a single row's failure
never aborts the run -- here, tracking's failure never aborts it either).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import mlflow
import yaml

from core.evaluation.config import EvaluationConfig

logger = logging.getLogger(__name__)

#: Local, file-based MLflow store -- no tracking server needed for the MVP.
#: Matches ``mlflow.tracking_uri`` in ``configs/default.yaml``.
DEFAULT_TRACKING_URI = "mlruns/"

#: Where ``_load_tracking_uri`` looks for ``mlflow.tracking_uri`` when the
#: caller doesn't pass one explicitly.
DEFAULT_CONFIG_PATH = "configs/default.yaml"

#: Maps ``EvaluationRun.status`` (``RunStatus`` in
#: ``core.evaluation.orchestrator``) onto MLflow's own run-status
#: vocabulary. ``"completed_with_errors"`` still finalizes as
#: ``"FINISHED"`` -- the run itself completed; row-level failures are
#: tracked separately via ``FailureAnalysis``, not the MLflow run status.
_STATUS_MAP: Dict[str, str] = {
    "completed": "FINISHED",
    "completed_with_errors": "FINISHED",
    "failed": "FAILED",
}

#: MLflow's own native statuses, passed through unchanged if a caller
#: already hands one of these to ``end_run`` instead of an
#: ``EvaluationRun.status`` value.
_MLFLOW_NATIVE_STATUSES: tuple[str, ...] = ("FINISHED", "FAILED", "KILLED")


def _is_local_file_uri(tracking_uri: str) -> bool:
    """Return whether ``tracking_uri`` points at a local filesystem path.

    Covers a bare relative/absolute path (``"mlruns/"``, ``"./mlruns"``)
    and an explicit ``file://`` URI; excludes database URIs
    (``sqlite:///...``), remote/server URIs (``http(s)://``), and MLflow's
    special-cased ``"databricks"`` magic string, none of which need the
    opt-out below.
    """
    if tracking_uri.lower().startswith("databricks"):
        return False
    return tracking_uri.startswith("file://") or "://" not in tracking_uri


def _load_tracking_uri(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Read ``mlflow.tracking_uri`` out of the YAML app config.

    Falls back to ``DEFAULT_TRACKING_URI`` if the config file is missing,
    unreadable, malformed, or simply doesn't define the key -- tracking
    configuration is best-effort, not a hard dependency for the app to
    boot.

    Args:
        config_path: Path to the YAML app config (e.g.
            ``configs/default.yaml``).

    Returns:
        The configured tracking URI, or ``DEFAULT_TRACKING_URI``.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "Could not read MLflow tracking_uri from %r (%s); using default %r",
            config_path,
            exc,
            DEFAULT_TRACKING_URI,
        )
        return DEFAULT_TRACKING_URI
    mlflow_section = data.get("mlflow") if isinstance(data, dict) else None
    tracking_uri = mlflow_section.get("tracking_uri") if isinstance(mlflow_section, dict) else None
    return str(tracking_uri) if tracking_uri else DEFAULT_TRACKING_URI


def resolve_benchmark_name(config: EvaluationConfig) -> str:
    """Best-effort benchmark identifier for experiment naming and params.

    Prefers an explicit ``benchmark_name`` attribute on ``config`` if the
    caller's ``EvaluationConfig`` carries one; otherwise falls back to
    ``dataset_version_id``, since a run's benchmark is, in the absence of
    a friendlier name, identified by the dataset version it's scored
    against.

    Args:
        config: The run's ``EvaluationConfig``.
    """
    benchmark_name = getattr(config, "benchmark_name", None)
    return str(benchmark_name) if benchmark_name else str(config.dataset_version_id)


def build_experiment_name(config: EvaluationConfig) -> str:
    """Build the ``"llm-eval-{benchmark_name}"`` experiment name for a run.

    Args:
        config: The run's ``EvaluationConfig``.
    """
    return f"llm-eval-{resolve_benchmark_name(config)}"


class MLflowTracker:
    """Tracks one evaluation run's lifecycle as one MLflow run.

    Usage is exactly one full cycle per instance, matching
    ``EvaluationOrchestrator.run()``::

        tracker = MLflowTracker()
        mlflow_run_id = tracker.start_run(experiment_name, run_name)
        tracker.log_params(config)
        ...  # rows execute
        tracker.log_metrics(metric_averages, composite_score=run_composite)
        tracker.end_run(status)

    If ``start_run`` fails (backend unreachable, bad tracking URI, ...),
    every subsequent call becomes a no-op rather than raising, so the
    evaluation run this tracker is attached to proceeds untracked instead
    of crashing.
    """

    def __init__(self, tracking_uri: Optional[str] = None, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        """Point the MLflow client at a tracking backend.

        Args:
            tracking_uri: Explicit tracking URI. When omitted, it's read
                from ``mlflow.tracking_uri`` in ``config_path``, defaulting
                to the local ``"mlruns/"`` directory (file-based, no
                server required) if that key is absent too.
            config_path: YAML app config consulted for the tracking URI
                when ``tracking_uri`` isn't given directly. Ignored if
                ``tracking_uri`` is passed.
        """
        self.tracking_uri = tracking_uri or _load_tracking_uri(config_path)
        self._run_active = False
        if _is_local_file_uri(self.tracking_uri):
            # Recent MLflow versions put the local filesystem store into
            # "maintenance mode" and refuse to use it unless this is set.
            # The MVP spec calls for a server-less local "mlruns/" default,
            # so opt in automatically rather than making every caller know
            # about this flag. Respect an explicit user override (e.g. a
            # deployment that deliberately sets this to "false").
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
        except Exception:  # noqa: BLE001 - tracking setup must never crash the caller
            logger.warning(
                "Failed to set MLflow tracking URI %r; tracking disabled for this run",
                self.tracking_uri,
                exc_info=True,
            )

    def start_run(self, experiment_name: str, run_name: str) -> Optional[str]:
        """Start a new MLflow run under ``experiment_name``.

        Creates the experiment first if it doesn't exist yet (MLflow's
        default behavior for ``set_experiment``).

        Args:
            experiment_name: Groups runs logically, e.g.
                ``"llm-eval-{benchmark_name}"`` (see
                ``build_experiment_name``).
            run_name: Human-readable label for this run within the
                experiment, shown in the MLflow UI's run list.

        Returns:
            The MLflow-assigned run id, or ``None`` if starting the run
            failed. A ``None`` return means every other method on this
            tracker becomes a no-op for the rest of its lifecycle -- the
            caller should treat the evaluation run as untracked and
            continue regardless.
        """
        try:
            mlflow.set_experiment(experiment_name)
            run = mlflow.start_run(run_name=run_name)
        except Exception:  # noqa: BLE001 - starting the run must never crash the caller
            logger.warning(
                "Failed to start MLflow run in experiment %r; continuing without tracking",
                experiment_name,
                exc_info=True,
            )
            self._run_active = False
            return None

        self._run_active = True
        mlflow_run_id = run.info.run_id
        logger.info("Started MLflow run %s in experiment %r", mlflow_run_id, experiment_name)
        return mlflow_run_id

    def log_params(self, config: EvaluationConfig) -> None:
        """Log the run's configuration as MLflow params.

        Logs six params: ``provider``, ``model``, ``temperature``,
        ``max_tokens``, ``prompt_version``, and ``benchmark`` (see
        ``resolve_benchmark_name``). A ``None``-valued field is logged as
        the string ``"none"`` -- MLflow params must be stringifiable, and
        silently dropping the key would make its absence indistinguishable
        from a logging failure.

        A no-op (with a debug log) if no run is active, e.g. because
        ``start_run`` failed.

        Args:
            config: The run's ``EvaluationConfig``.
        """
        if not self._run_active:
            logger.debug("log_params called with no active MLflow run; skipping")
            return
        params = {
            "provider": config.provider_name,
            "model": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "prompt_version": config.prompt_version_id,
            "benchmark": resolve_benchmark_name(config),
        }
        try:
            mlflow.log_params({key: ("none" if value is None else value) for key, value in params.items()})
        except Exception:  # noqa: BLE001 - logging failure must never crash the caller
            logger.warning("Failed to log MLflow params", exc_info=True)

    def log_metrics(self, metric_averages: Dict[str, float], composite_score: Optional[float] = None) -> None:
        """Log the run's aggregate metrics.

        Args:
            metric_averages: Mean of each component score
                (``accuracy``, ``hallucination``, ``instruction``,
                ``safety``, and, if present, ``latency``/``cost``) across
                successful rows, keyed by component name -- the same dict
                ``EvaluationPersistenceService.save_metrics`` writes to
                ``RunMetrics``. Only present keys are logged, so this is
                safe to call with a partial or empty dict (e.g. when no
                rows succeeded).
            composite_score: The run's mean composite score across
                successful rows. Logged under the key ``composite_score``
                when not ``None``.
        """
        if not self._run_active:
            logger.debug("log_metrics called with no active MLflow run; skipping")
            return
        metrics = {key: value for key, value in metric_averages.items() if value is not None}
        if composite_score is not None:
            metrics["composite_score"] = composite_score
        if not metrics:
            logger.debug("No metrics to log for this MLflow run (nothing succeeded)")
            return
        try:
            mlflow.log_metrics(metrics)
        except Exception:  # noqa: BLE001 - logging failure must never crash the caller
            logger.warning("Failed to log MLflow metrics", exc_info=True)

    def log_artifact(self, file_path: str) -> None:
        """Attach a local file (e.g. a results CSV) to the active run.

        Optional: not called by the orchestrator's default wiring, but
        available for callers that export per-row results to disk and
        want them attached to the MLflow run for later download from the
        UI.

        Args:
            file_path: Path to the local file to upload as an artifact.
        """
        if not self._run_active:
            logger.debug("log_artifact called with no active MLflow run; skipping")
            return
        try:
            mlflow.log_artifact(file_path)
        except Exception:  # noqa: BLE001 - logging failure must never crash the caller
            logger.warning("Failed to log MLflow artifact %r", file_path, exc_info=True)

    def end_run(self, status: str) -> None:
        """Finalize the active MLflow run.

        A no-op if no run is active (e.g. ``start_run`` failed, or
        ``end_run`` was already called once for this tracker).

        Args:
            status: Either an ``EvaluationRun.status`` value
                (``"completed"``, ``"completed_with_errors"``, or
                ``"failed"``), mapped onto MLflow's own status vocabulary,
                or an already-native MLflow status (``"FINISHED"``,
                ``"FAILED"``, ``"KILLED"``), passed through unchanged.
                Anything else defaults to ``"FINISHED"``.
        """
        if not self._run_active:
            logger.debug("end_run called with no active MLflow run; skipping")
            return
        if status in _MLFLOW_NATIVE_STATUSES:
            mlflow_status = status
        else:
            mlflow_status = _STATUS_MAP.get(status, "FINISHED")
        try:
            mlflow.end_run(status=mlflow_status)
        except Exception:  # noqa: BLE001 - ending the run must never crash the caller
            logger.warning("Failed to end MLflow run cleanly", exc_info=True)
        finally:
            self._run_active = False