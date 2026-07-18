"""Repository for EvaluationRun and its related results, metrics, and failures."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import EvaluationResult, EvaluationRun, FailureAnalysis, RunMetrics
from database.repositories.base import BaseRepository


class EvaluationRepository(BaseRepository[EvaluationRun]):
    """CRUD and query helpers for evaluation runs and their child records."""

    def __init__(self, session: Session) -> None:
        """Bind this repository to a session, targeting the ``EvaluationRun`` model."""
        super().__init__(EvaluationRun, session)

    def add_result(self, run_id: str, **kwargs: Any) -> EvaluationResult:
        """Attach a single question/response result row to a run."""
        result = EvaluationResult(run_id=run_id, **kwargs)
        self.session.add(result)
        self.session.flush()
        return result

    def get_results(self, run_id: str) -> Sequence[EvaluationResult]:
        """Return all evaluation results belonging to a run."""
        stmt = select(EvaluationResult).where(EvaluationResult.run_id == run_id)
        return self.session.execute(stmt).scalars().all()

    def set_metrics(self, run_id: str, **kwargs: Any) -> RunMetrics:
        """Create or replace the 1:1 ``RunMetrics`` row for a run.

        If a metrics row already exists for ``run_id`` its fields are
        updated in place (preserving the 1:1 relationship); otherwise a
        new row is created.
        """
        existing = self.get_metrics(run_id)
        if existing is not None:
            for field, value in kwargs.items():
                setattr(existing, field, value)
            self.session.flush()
            return existing
        metrics = RunMetrics(run_id=run_id, **kwargs)
        self.session.add(metrics)
        self.session.flush()
        return metrics

    def get_metrics(self, run_id: str) -> Optional[RunMetrics]:
        """Return the ``RunMetrics`` row for a run, if it exists."""
        stmt = select(RunMetrics).where(RunMetrics.run_id == run_id)
        return self.session.execute(stmt).scalars().first()

    def add_failure(self, result_id: str, **kwargs: Any) -> FailureAnalysis:
        """Attach a failure classification to an evaluation result."""
        failure = FailureAnalysis(result_id=result_id, **kwargs)
        self.session.add(failure)
        self.session.flush()
        return failure

    def list_by_model(self, model_name: str) -> Sequence[EvaluationRun]:
        """Return all runs for a given model name (indexed lookup)."""
        stmt = select(EvaluationRun).where(EvaluationRun.model_name == model_name)
        return self.session.execute(stmt).scalars().all()

    def list_by_status(self, status: str) -> Sequence[EvaluationRun]:
        """Return all runs with a given status (indexed lookup)."""
        stmt = select(EvaluationRun).where(EvaluationRun.status == status)
        return self.session.execute(stmt).scalars().all()

    def top_models_by_score(self, limit: int = 10) -> Sequence[Tuple[Optional[str], float]]:
        """Return ``(model_name, avg composite_score)`` pairs, highest score first.

        Mirrors the "Top models" query example in DATABASE_SCHEMA.md.
        """
        stmt = (
            select(EvaluationRun.model_name, func.avg(EvaluationRun.composite_score))
            .group_by(EvaluationRun.model_name)
            .order_by(func.avg(EvaluationRun.composite_score).desc())
            .limit(limit)
        )
        return [(row[0], row[1]) for row in self.session.execute(stmt).all()]