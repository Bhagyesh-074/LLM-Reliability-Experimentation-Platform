"""Unit tests for the database layer, backed by an in-memory SQLite database.

Each test gets a fresh in-memory database (via the ``session`` fixture)
so tests are fully isolated and require no external services.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from database import models
from database.base import Base
from database.repositories.benchmark_repository import BenchmarkRepository
from database.repositories.evaluation_repository import EvaluationRepository
from database.repositories.prompt_repository import PromptRepository


@pytest.fixture()
def session() -> Session:
    """Yield a Session backed by a fresh in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = factory()
    try:
        yield db
    finally:
        db.close()


def _make_provider(session: Session) -> models.Provider:
    provider = models.Provider(name="openai", type="api", sdk_version="1.0.0")
    session.add(provider)
    session.flush()
    return provider


class TestProviderCRUD:
    """Tests for the base Provider table (exercised directly, no repository)."""

    def test_create_and_get(self, session: Session) -> None:
        provider = _make_provider(session)
        fetched = session.get(models.Provider, provider.provider_id)
        assert fetched is not None
        assert fetched.name == "openai"
        assert fetched.type == "api"
        assert fetched.provider_id  # UUID auto-generated

    def test_name_unique_constraint(self, session: Session) -> None:
        _make_provider(session)
        session.add(models.Provider(name="openai", type="local"))
        with pytest.raises(IntegrityError):
            session.flush()


class TestPromptRepository:
    """Tests for PromptRepository, including version immutability behavior."""

    def test_create_prompt(self, session: Session) -> None:
        repo = PromptRepository(session)
        prompt = repo.create(name="summarizer", author="alice", status="active")
        assert prompt.prompt_id
        assert repo.get(prompt.prompt_id) is not None

    def test_add_version_auto_increments(self, session: Session) -> None:
        repo = PromptRepository(session)
        prompt = repo.create(name="summarizer", author="alice", status="active")

        v1 = repo.add_version(prompt.prompt_id, content="Summarize: {text}")
        v2 = repo.add_version(prompt.prompt_id, content="Summarize concisely: {text}")

        assert v1.version == 1
        assert v2.version == 2
        assert v1.version_id != v2.version_id
        assert v1.content_hash != v2.content_hash

    def test_list_versions_ordered(self, session: Session) -> None:
        repo = PromptRepository(session)
        prompt = repo.create(name="p", author="a", status="active")
        repo.add_version(prompt.prompt_id, content="v1")
        repo.add_version(prompt.prompt_id, content="v2")
        repo.add_version(prompt.prompt_id, content="v3")

        versions = repo.list_versions(prompt.prompt_id)
        assert [v.version for v in versions] == [1, 2, 3]

    def test_get_by_name(self, session: Session) -> None:
        repo = PromptRepository(session)
        repo.create(name="classifier", author="bob", status="draft")
        found = repo.get_by_name("classifier")
        assert found is not None
        assert found.author == "bob"

    def test_versions_are_immutable_rows(self, session: Session) -> None:
        """Each call to add_version creates a distinct row, never edits a prior one."""
        repo = PromptRepository(session)
        prompt = repo.create(name="p", author="a", status="active")
        v1 = repo.add_version(prompt.prompt_id, content="original text")
        repo.add_version(prompt.prompt_id, content="new text")

        # The original version's content is untouched by later additions.
        reloaded_v1 = repo.get_version(prompt.prompt_id, 1)
        assert reloaded_v1 is not None
        assert reloaded_v1.content == "original text"
        assert reloaded_v1.version_id == v1.version_id


class TestBenchmarkRepository:
    """Tests for BenchmarkRepository, including dataset version immutability."""

    def test_add_dataset_version(self, session: Session) -> None:
        repo = BenchmarkRepository(session)
        benchmark = repo.create(name="mmlu", domain="general", description="Massive multitask")
        dv = repo.add_dataset_version(benchmark.benchmark_id, version="1.0", question_count=100)

        assert dv.version == "1.0"
        assert dv.question_count == 100

        fetched = repo.get_dataset_version_by_version(benchmark.benchmark_id, "1.0")
        assert fetched is not None
        assert fetched.dataset_version_id == dv.dataset_version_id

    def test_list_dataset_versions(self, session: Session) -> None:
        repo = BenchmarkRepository(session)
        benchmark = repo.create(name="hellaswag", domain="reasoning")
        repo.add_dataset_version(benchmark.benchmark_id, version="1.0")
        repo.add_dataset_version(benchmark.benchmark_id, version="2.0")

        versions = repo.list_dataset_versions(benchmark.benchmark_id)
        assert len(versions) == 2

    def test_duplicate_version_label_rejected(self, session: Session) -> None:
        repo = BenchmarkRepository(session)
        benchmark = repo.create(name="hellaswag", domain="reasoning")
        repo.add_dataset_version(benchmark.benchmark_id, version="1.0")
        with pytest.raises(IntegrityError):
            repo.add_dataset_version(benchmark.benchmark_id, version="1.0")


class TestEvaluationRepository:
    """Tests for EvaluationRepository and its related child records."""

    def _setup_run(self, session: Session) -> models.EvaluationRun:
        provider = _make_provider(session)

        prompt_repo = PromptRepository(session)
        prompt = prompt_repo.create(name="p", author="a", status="active")
        prompt_version = prompt_repo.add_version(prompt.prompt_id, content="hello")

        bench_repo = BenchmarkRepository(session)
        benchmark = bench_repo.create(name="b", domain="d")
        dataset_version = bench_repo.add_dataset_version(benchmark.benchmark_id, version="1.0")

        eval_repo = EvaluationRepository(session)
        return eval_repo.create(
            provider_id=provider.provider_id,
            model_name="gpt-test",
            prompt_version_id=prompt_version.version_id,
            dataset_version_id=dataset_version.dataset_version_id,
            temperature=0.0,
            composite_score=0.87,
            status="completed",
            started_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )

    def test_create_run(self, session: Session) -> None:
        run = self._setup_run(session)
        assert run.run_id
        assert run.model_name == "gpt-test"
        assert run.status == "completed"

    def test_add_result_and_failure(self, session: Session) -> None:
        run = self._setup_run(session)
        eval_repo = EvaluationRepository(session)

        result = eval_repo.add_result(
            run.run_id,
            question_id="q1",
            question="What is 2+2?",
            ground_truth="4",
            response="5",
            latency_ms=120,
            token_usage=42,
        )
        failure = eval_repo.add_failure(
            result.result_id,
            category="Reasoning Error",
            explanation="Model miscalculated a simple sum.",
            severity="medium",
        )

        results = eval_repo.get_results(run.run_id)
        assert len(results) == 1
        assert results[0].result_id == result.result_id
        assert failure.category == "Reasoning Error"

    def test_invalid_failure_category_rejected(self, session: Session) -> None:
        run = self._setup_run(session)
        eval_repo = EvaluationRepository(session)
        result = eval_repo.add_result(run.run_id, question_id="q1")
        with pytest.raises(IntegrityError):
            eval_repo.add_failure(result.result_id, category="Not A Real Category")

    def test_set_metrics_creates_then_updates_same_row(self, session: Session) -> None:
        run = self._setup_run(session)
        eval_repo = EvaluationRepository(session)

        eval_repo.set_metrics(run.run_id, accuracy=0.9, hallucination=0.1)
        metrics = eval_repo.get_metrics(run.run_id)
        assert metrics is not None
        assert metrics.accuracy == 0.9

        eval_repo.set_metrics(run.run_id, accuracy=0.95)
        updated = eval_repo.get_metrics(run.run_id)
        assert updated is not None
        assert updated.accuracy == 0.95
        assert updated.metric_id == metrics.metric_id  # same row, not duplicated

    def test_list_by_model_and_status(self, session: Session) -> None:
        self._setup_run(session)
        eval_repo = EvaluationRepository(session)

        assert len(eval_repo.list_by_model("gpt-test")) == 1
        assert len(eval_repo.list_by_status("completed")) == 1
        assert len(eval_repo.list_by_status("failed")) == 0

    def test_top_models_by_score(self, session: Session) -> None:
        self._setup_run(session)
        eval_repo = EvaluationRepository(session)
        top = eval_repo.top_models_by_score()
        assert top[0][0] == "gpt-test"
        assert top[0][1] == pytest.approx(0.87)


class TestBaseRepositoryCRUD:
    """Tests for the generic update/delete behavior shared by all repositories."""

    def test_update_and_delete(self, session: Session) -> None:
        repo = PromptRepository(session)
        prompt = repo.create(name="temp", author="carol", status="draft")

        updated = repo.update(prompt.prompt_id, status="active")
        assert updated is not None
        assert updated.status == "active"

        deleted = repo.delete(prompt.prompt_id)
        assert deleted is True
        assert repo.get(prompt.prompt_id) is None

    def test_delete_missing_returns_false(self, session: Session) -> None:
        repo = PromptRepository(session)
        assert repo.delete("nonexistent-id") is False

    def test_update_missing_raises(self, session: Session) -> None:
        repo = PromptRepository(session)
        with pytest.raises(LookupError):
            repo.update("nonexistent-id", status="active")