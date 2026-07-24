"""
database/models.py

SQLAlchemy ORM models for the LLM Reliability & Experimentation
Platform, implementing every table defined in DATABASE_SCHEMA.md.

Design notes
------------
- Primary keys are UUIDs, stored as `String(36)` (`str(uuid.uuid4())`).
  This keeps the schema portable between SQLite (no native UUID type)
  and PostgreSQL (where the column can later be swapped to the native
  `UUID` type without touching application code).
- All foreign keys are declared explicitly and enforced. SQLite does
  not enforce FKs by default, so `PRAGMA foreign_keys=ON` is set in
  `database/session.py`.
- `prompt_versions` and `dataset_versions` are immutable by
  convention (enforced at the repository/service layer, not the DB
  layer, since SQL has no native "immutable row" constraint).
- `evaluation_runs.composite_score` is derived (computed by the
  scoring service) and must not be written to directly by clients;
  this is a service-layer contract, not a DB-level constraint.
- `run_metrics` is a 1:1 child of `evaluation_runs`, enforced via a
  UNIQUE constraint on `run_metrics.run_id`.
- `mlflow_runs.mlflow_run_id` is a TEXT primary key (MLflow's own run
  identifier), not a platform-generated UUID.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy import JSON as SAJSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime, Float, String

from database.base import Base


def _uuid() -> str:
    """Generate a new UUID4 string for use as a primary key default."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return the current UTC timestamp for `created_at` defaults."""
    return datetime.now(timezone.utc)


class Provider(Base):
    """An LLM provider (local runtime or hosted API) available for evaluation."""

    __tablename__ = "providers"
    __table_args__ = (
        CheckConstraint("type IN ('local', 'api')", name="ck_providers_type"),
    )

    provider_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False, doc="local/api")
    sdk_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    evaluation_runs: Mapped[List["EvaluationRun"]] = relationship(
        back_populates="provider", cascade="save-update, merge"
    )


class Prompt(Base):
    """A logical, named prompt. Immutable content lives in `PromptVersion`."""

    __tablename__ = "prompts"

    prompt_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    author: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    versions: Mapped[List["PromptVersion"]] = relationship(
        back_populates="prompt",
        cascade="all, delete-orphan",
        order_by="PromptVersion.version",
    )


class PromptVersion(Base):
    """
    An immutable, versioned snapshot of a prompt's content.

    A prompt may have many versions; once created, a version's content
    must never be mutated (enforced at the service/repository layer).
    """

    __tablename__ = "prompt_versions"
    __table_args__ = (
        UniqueConstraint("prompt_id", "version", name="uq_prompt_versions_prompt_version"),
        Index("ix_prompt_versions_prompt_id", "prompt_id"),
    )

    version_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    prompt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("prompts.prompt_id"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[Optional[dict]] = mapped_column(SAJSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    prompt: Mapped["Prompt"] = relationship(back_populates="versions")
    evaluation_runs: Mapped[List["EvaluationRun"]] = relationship(
        back_populates="prompt_version"
    )


class Benchmark(Base):
    """A named benchmark/domain suite, e.g. 'MMLU', 'HellaSwag'."""

    __tablename__ = "benchmarks"

    benchmark_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    dataset_versions: Mapped[List["DatasetVersion"]] = relationship(
        back_populates="benchmark", cascade="all, delete-orphan"
    )


class DatasetVersion(Base):
    """
    An immutable, versioned snapshot of a benchmark's dataset.

    Once created, the underlying question set must not change; a new
    `DatasetVersion` row should be created instead.
    """

    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint(
            "benchmark_id", "version", name="uq_dataset_versions_benchmark_version"
        ),
        Index("ix_dataset_versions_benchmark_id", "benchmark_id"),
    )

    dataset_version_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    benchmark_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("benchmarks.benchmark_id"), nullable=False
    )
    version: Mapped[str] = mapped_column(Text, nullable=False)
    question_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    benchmark: Mapped["Benchmark"] = relationship(back_populates="dataset_versions")
    evaluation_runs: Mapped[List["EvaluationRun"]] = relationship(
        back_populates="dataset_version"
    )


class EvaluationRun(Base):
    """
    A single evaluation execution: one provider/model, one prompt
    version, one dataset version, with its own generation parameters.
    """

    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("ix_evaluation_runs_model_name", "model_name"),
        Index("ix_evaluation_runs_status", "status"),
        Index("ix_evaluation_runs_started_at", "started_at"),
        Index("ix_evaluation_runs_provider_id", "provider_id"),
        Index("ix_evaluation_runs_prompt_version_id", "prompt_version_id"),
        Index("ix_evaluation_runs_dataset_version_id", "dataset_version_id"),
    )

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    provider_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("providers.provider_id"), nullable=False
    )
    model_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("prompt_versions.version_id"), nullable=False
    )
    dataset_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dataset_versions.dataset_version_id"), nullable=False
    )
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    composite_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, doc="Derived value; do not set directly from client input."
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    provider: Mapped["Provider"] = relationship(back_populates="evaluation_runs")
    prompt_version: Mapped["PromptVersion"] = relationship(
        back_populates="evaluation_runs"
    )
    dataset_version: Mapped["DatasetVersion"] = relationship(
        back_populates="evaluation_runs"
    )
    results: Mapped[List["EvaluationResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    metrics: Mapped[Optional["RunMetrics"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False
    )
    mlflow_runs: Mapped[List["MlflowRun"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvaluationResult(Base):
    """One evaluated question/response pair belonging to an evaluation run."""

    __tablename__ = "evaluation_results"
    __table_args__ = (
        Index("ix_evaluation_results_run_id", "run_id"),
        Index("ix_evaluation_results_status", "status"),
    )

    result_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("evaluation_runs.run_id"), nullable=False
    )
    question_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ground_truth: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    token_usage: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    accuracy_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hallucination_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    instruction_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    safety_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    composite_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="'passed' or 'failed', derived from row success at persist time."
    )

    run: Mapped["EvaluationRun"] = relationship(back_populates="results")
    failures: Mapped[List["FailureAnalysis"]] = relationship(
        back_populates="result", cascade="all, delete-orphan"
    )


class RunMetrics(Base):
    """
    Aggregate metrics for a single evaluation run.

    1:1 with `EvaluationRun`, enforced via a UNIQUE constraint on
    `run_id`.
    """

    __tablename__ = "run_metrics"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_run_metrics_run_id"),
    )

    metric_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("evaluation_runs.run_id"), nullable=False
    )
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hallucination: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    instruction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    safety: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    consistency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    run: Mapped["EvaluationRun"] = relationship(back_populates="metrics")


class FailureAnalysis(Base):
    """A categorized failure analysis attached to one evaluation result."""

    __tablename__ = "failure_analysis"
    __table_args__ = (
        CheckConstraint(
            "category IN ("
            "'Hallucination', 'Factual Error', 'Reasoning Error', "
            "'Formatting Error', 'Refusal', 'Safety Issue'"
            ")",
            name="ck_failure_analysis_category",
        ),
        Index("ix_failure_analysis_category", "category"),
        Index("ix_failure_analysis_result_id", "result_id"),
    )

    failure_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    result_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("evaluation_results.result_id"), nullable=False
    )
    category: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    severity: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    result: Mapped["EvaluationResult"] = relationship(back_populates="failures")


class MlflowRun(Base):
    """Link between a platform `EvaluationRun` and an external MLflow run."""

    __tablename__ = "mlflow_runs"

    mlflow_run_id: Mapped[str] = mapped_column(
        Text, primary_key=True, doc="MLflow's own run identifier (not platform UUID)."
    )
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("evaluation_runs.run_id"), nullable=False
    )
    artifact_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    experiment_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    run: Mapped["EvaluationRun"] = relationship(back_populates="mlflow_runs")


__all__ = [
    "Provider",
    "Prompt",
    "PromptVersion",
    "Benchmark",
    "DatasetVersion",
    "EvaluationRun",
    "EvaluationResult",
    "RunMetrics",
    "FailureAnalysis",
    "MlflowRun",
]