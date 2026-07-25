"""core/evaluation/persistence.py

Persistence service for completed evaluation runs.

``EvaluationPersistenceService`` is the seam between the in-memory
``EvaluationRun`` DTO produced by the orchestrator and the SQLAlchemy ORM
rows in ``database.models``. It translates the DTO into an
``EvaluationRun`` row, its child ``EvaluationResult`` rows, and its 1:1
``RunMetrics`` row, delegating the actual writes to
``EvaluationRepository``. It also links the run to its external MLflow
tracking run via a ``MlflowRun`` row, when ``MLflowTracker.start_run``
succeeded (see ``save_mlflow_run``).

The orchestrator is deliberately kept ignorant of ORM column names and
FK wiring: it hands over a DTO and per-row results and gets back a
persisted ``run_id``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import EvaluationRun as EvaluationRunModel
from database.models import MlflowRun, Provider
from database.repositories.evaluation_repository import EvaluationRepository

if TYPE_CHECKING:  # avoid a runtime import cycle with the orchestrator
    from core.evaluation.orchestrator import EvaluationRowResult, EvaluationRun, FailureRecord

logger = logging.getLogger(__name__)

#: Component keys whose averages map 1:1 onto ``RunMetrics`` float columns.
_METRIC_COLUMNS: tuple[str, ...] = (
    "accuracy",
    "hallucination",
    "instruction",
    "safety",
    "latency",
    "cost",
)

#: Per-row ``metric_scores`` keys that have a matching per-question column on
#: ``EvaluationResult`` (``"<key>_score"``). ``latency`` and ``cost`` are
#: intentionally excluded: they aren't persisted per row on
#: ``EvaluationResult`` (latency/token usage are already captured via
#: ``latency_ms`` / ``token_usage`` from the raw response), only as run-level
#: averages on ``RunMetrics``.
_ROW_SCORE_KEYS: tuple[str, ...] = ("accuracy", "hallucination", "instruction", "safety")


class EvaluationPersistenceService:
    """Persists an ``EvaluationRun`` DTO and its children to the database.

    All three ``save_*`` methods flush through ``EvaluationRepository`` but
    do not commit; committing is the caller's (or an outer unit-of-work's)
    responsibility, so an entire run can be persisted atomically.
    """

    def __init__(self, session: Session, repository: Optional[EvaluationRepository] = None) -> None:
        """Bind the service to a session and (optionally) a repository.

        Args:
            session: Active SQLAlchemy session.
            repository: Optional pre-built ``EvaluationRepository`` (for
                tests). Defaults to a new instance bound to ``session``.
        """
        self.session = session
        self.repository = repository or EvaluationRepository(session)

    def save_run(self, run: "EvaluationRun") -> str:
        """Persist the top-level ``EvaluationRun`` row and return its id.

        Resolves ``config.provider_name`` to a ``providers.provider_id``
        FK (required by the schema), then writes the run's parameters,
        status, composite score, and timestamps.

        Args:
            run: The in-memory run DTO to persist. Its ``composite_score``
                is the mean composite across successful rows (may be
                ``None`` if none succeeded).

        Returns:
            The persisted ``run_id`` (DB primary key).

        Raises:
            ValueError: If ``run.config.provider_name`` does not resolve to
                a stored ``providers`` row, since ``provider_id`` is a
                non-nullable FK.
        """
        provider_id = self._resolve_provider_id(run.config.provider_name)
        model = self.repository.create(
            provider_id=provider_id,
            model_name=run.config.model_name,
            prompt_version_id=run.config.prompt_version_id,
            dataset_version_id=run.config.dataset_version_id,
            temperature=run.config.temperature,
            max_tokens=run.config.max_tokens,
            composite_score=run.composite_score,
            status=run.status,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )
        logger.info("Persisted evaluation run %s (status=%s)", model.run_id, run.status)
        return model.run_id

    def save_results(self, run_id: str, results: List["EvaluationRowResult"]) -> None:
        """Persist every per-row result belonging to a run.

        Both successful and failed rows are persisted so the run's history
        is complete. For failed rows, ``response`` / latency / tokens / the
        per-question scores are absent and stored as ``None``, and
        ``status`` is recorded as ``"failed"``.

        Per-question scores (``accuracy_score``, ``hallucination_score``,
        ``instruction_score``, ``safety_score``, ``composite_score``) are
        read directly off each row's ``metric_scores`` / ``composite_score``
        -- the same values the orchestrator already computed via
        ``_score_row`` -- so the Results dashboard can render per-question
        breakdowns without recomputing anything.

        Any failures the orchestrator detected for a row (via
        ``_detect_failures``) are persisted right after that row, via
        ``save_failures``, using the just-created ``result_id`` -- so each
        ``FailureAnalysis`` row is correctly linked to its
        ``EvaluationResult`` even though both are created in the same pass.

        Args:
            run_id: The persisted parent run id.
            results: The orchestrator's per-row results, in row order.
        """
        for result in results:
            response = result.response
            saved_result = self.repository.add_result(
                run_id=run_id,
                question_id=self._row_str(result.question_row, "question_id"),
                question=self._row_str(result.question_row, "question"),
                ground_truth=self._row_str(result.question_row, "ground_truth"),
                response=response.text if response is not None else None,
                latency_ms=self._latency_int(response),
                token_usage=self._total_tokens(response),
                accuracy_score=self._row_score(result, "accuracy"),
                hallucination_score=self._row_score(result, "hallucination"),
                instruction_score=self._row_score(result, "instruction"),
                safety_score=self._row_score(result, "safety"),
                composite_score=result.composite_score if result.success else None,
                status=self._row_status(result),
            )
            if result.failures:
                self.save_failures(saved_result.result_id, result.failures)
        logger.info("Persisted %d result row(s) for run %s", len(results), run_id)

    def save_failures(self, result_id: str, failures: List["FailureRecord"]) -> None:
        """Persist one ``FailureAnalysis`` row per detected failure record.

        A single evaluation row can trigger more than one failure rule at
        once (e.g. both low accuracy and low hallucination-resistance), so
        each entry in ``failures`` becomes its own ``FailureAnalysis`` row,
        all pointing at the same ``result_id``. Called from
        ``save_results`` right after the owning ``EvaluationResult`` row is
        created; it can also be called directly (e.g. by tests, or by a
        future re-analysis job) given any existing ``result_id``.

        Args:
            result_id: The persisted ``EvaluationResult.result_id`` these
                failure classifications belong to.
            failures: The failure records detected for this row, e.g. by
                ``EvaluationOrchestrator._detect_failures``.
        """
        for failure in failures:
            self.repository.add_failure(
                result_id=result_id,
                category=failure.category,
                severity=failure.severity,
                explanation=failure.explanation,
            )
        if failures:
            logger.info("Persisted %d failure record(s) for result %s", len(failures), result_id)

    def save_metrics(self, run_id: str, metric_averages: Dict[str, float]) -> None:
        """Persist the 1:1 aggregate ``RunMetrics`` row for a run.

        Only the six component averages are written; ``consistency`` is
        left ``NULL`` as there is no consistency scorer in scope. If
        ``metric_averages`` is empty (no rows succeeded), no metrics row is
        written.

        Args:
            run_id: The persisted parent run id.
            metric_averages: Mean of each component score across
                successful rows, keyed by the component names.
        """
        columns = {key: metric_averages[key] for key in _METRIC_COLUMNS if key in metric_averages}
        if not columns:
            logger.info("No metric averages to persist for run %s (no successful rows)", run_id)
            return
        self.repository.set_metrics(run_id, **columns)
        logger.info("Persisted aggregate metrics for run %s: %s", run_id, columns)

    def save_mlflow_run(
        self,
        run_id: str,
        mlflow_run_id: str,
        experiment_name: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Persist the 1:many link between a run and its MLflow tracking run.

        Written directly against ``self.session`` (rather than through
        ``EvaluationRepository``) for the same reason
        ``_resolve_provider_id`` is: this is a small, self-contained
        write with no per-row fan-out, so a dedicated repository method
        would just forward the same three arguments.

        Called by the orchestrator only when ``MLflowTracker.start_run``
        actually returned an id -- MLflow tracking is best-effort, so a
        run with no ``mlflow_run_id`` (tracking failed or was disabled)
        simply has no row here rather than one with a placeholder id.

        Args:
            run_id: The persisted parent ``EvaluationRun.run_id``.
            mlflow_run_id: MLflow's own run identifier, as returned by
                ``MLflowTracker.start_run``. This is the row's primary
                key, so it must be unique across all runs.
            experiment_name: The MLflow experiment this run was logged
                under (e.g. ``"llm-eval-{benchmark_name}"``), if known.
            artifact_path: Where this run's artifacts were logged, if any
                were logged via ``MLflowTracker.log_artifact``.
        """
        mlflow_run = MlflowRun(
            mlflow_run_id=mlflow_run_id,
            run_id=run_id,
            experiment_name=experiment_name,
            artifact_path=artifact_path,
        )
        self.session.add(mlflow_run)
        self.session.flush()
        logger.info("Linked MLflow run %s to evaluation run %s", mlflow_run_id, run_id)

    def _resolve_provider_id(self, provider_name: str) -> str:
        """Resolve a provider name to its ``providers.provider_id`` FK,
        creating the ``providers`` row if it doesn't exist yet.

        ``EvaluationConfig`` only carries a provider *name* (e.g.
        ``"ollama"``), not a DB row, so the first run against a given
        provider name registers it automatically rather than failing.

        Args:
            provider_name: The provider key from the run config.

        Returns:
            The existing or newly created ``provider_id``.
        """
        stmt = select(Provider).where(Provider.name == provider_name)
        provider = self.session.execute(stmt).scalars().first()
        if provider is not None:
            return provider.provider_id

        provider_type = "local" if provider_name.strip().lower() == "ollama" else "api"
        provider = Provider(name=provider_name, type=provider_type)
        self.session.add(provider)
        self.session.flush()
        logger.info("Registered new provider %r (id=%s, type=%s)", provider_name, provider.provider_id, provider_type)
        return provider.provider_id

    @staticmethod
    def _row_score(result: "EvaluationRowResult", key: str) -> Optional[float]:
        """Read one component score off a row's ``metric_scores``, or ``None``.

        Returns ``None`` for failed rows (``metric_scores`` is empty on
        failure, per ``EvaluationRowResult``) and for successful rows where
        that particular scorer key is absent for any reason.

        Args:
            result: The orchestrator's per-row result.
            key: One of ``_ROW_SCORE_KEYS`` (e.g. ``"accuracy"``).
        """
        if not result.success:
            return None
        return result.metric_scores.get(key)

    @staticmethod
    def _row_status(result: "EvaluationRowResult") -> str:
        """Derive the persisted per-row ``status`` from the row's outcome.

        Mirrors ``EvaluationRowResult.success``: ``"passed"`` for a row that
        completed generation and scoring without error, ``"failed"``
        otherwise (missing template variable, provider error, or scoring
        error -- all of which set ``success=False``).
        """
        return "passed" if result.success else "failed"

    @staticmethod
    def _row_str(row: Dict[str, Any], key: str) -> Optional[str]:
        """Return ``row[key]`` coerced to ``str``, or ``None`` if absent/empty."""
        value = row.get(key)
        return None if value is None else str(value)

    @staticmethod
    def _latency_int(response: Any) -> Optional[int]:
        """Round a response's ``latency_ms`` to an int, or ``None``.

        ``EvaluationResult.latency_ms`` is an integer column, while
        ``LLMResponse.latency_ms`` is a float.
        """
        if response is None:
            return None
        return int(round(response.latency_ms))

    @staticmethod
    def _total_tokens(response: Any) -> Optional[int]:
        """Extract ``total_tokens`` from a response's token usage, or ``None``.

        ``EvaluationResult.token_usage`` is a single integer column, so the
        response's ``token_usage`` dict is reduced to its total.
        """
        if response is None:
            return None
        return response.token_usage.get("total_tokens")