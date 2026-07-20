"""core/evaluation/persistence.py

Persistence service for completed evaluation runs.

``EvaluationPersistenceService`` is the seam between the in-memory
``EvaluationRun`` DTO produced by the orchestrator and the SQLAlchemy ORM
rows in ``database.models``. It translates the DTO into an
``EvaluationRun`` row, its child ``EvaluationResult`` rows, and its 1:1
``RunMetrics`` row, delegating the actual writes to
``EvaluationRepository``.

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
from database.models import Provider
from database.repositories.evaluation_repository import EvaluationRepository

if TYPE_CHECKING:  # avoid a runtime import cycle with the orchestrator
    from core.evaluation.orchestrator import EvaluationRowResult, EvaluationRun

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
        is complete. For failed rows, ``response`` / latency / tokens may
        be absent and are stored as ``None``.

        Args:
            run_id: The persisted parent run id.
            results: The orchestrator's per-row results, in row order.
        """
        for result in results:
            response = result.response
            self.repository.add_result(
                run_id=run_id,
                question_id=self._row_str(result.question_row, "question_id"),
                question=self._row_str(result.question_row, "question"),
                ground_truth=self._row_str(result.question_row, "ground_truth"),
                response=response.text if response is not None else None,
                latency_ms=self._latency_int(response),
                token_usage=self._total_tokens(response),
            )
        logger.info("Persisted %d result row(s) for run %s", len(results), run_id)

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