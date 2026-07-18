"""
tests/unit/test_benchmark_service.py

Unit tests for ``registry.benchmark_service.BenchmarkService``, backed by
an in-memory SQLite database so tests are fast and fully isolated from
any real ``llm_reliability_platform.db`` file.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.base import Base
from database import models  # noqa: F401  (registers tables on Base.metadata)
from registry.benchmark_service import BenchmarkService, DatasetUploadError

VALID_CSV = (
    "question,ground_truth,domain,category,difficulty\n"
    "What is 2+2?,4,Math,arithmetic,easy\n"
    "What is the capital of France?,Paris,Geography,facts,medium\n"
)

MISSING_COLUMN_CSV = (
    "question,domain,category,difficulty\n"
    "What is 2+2?,Math,arithmetic,easy\n"
)

EMPTY_CELL_CSV = (
    "question,ground_truth,domain,category,difficulty\n"
    ",4,Math,arithmetic,easy\n"
    "What is the capital of France?,,Geography,facts,medium\n"
)

BAD_DIFFICULTY_CSV = (
    "question,ground_truth,domain,category,difficulty\n"
    "What is 2+2?,4,Math,arithmetic,extreme\n"
)


def _csv_file(content: str) -> io.BytesIO:
    """Wrap CSV text as an in-memory binary file, as Streamlit's uploader provides."""
    return io.BytesIO(content.encode("utf-8"))


@pytest.fixture()
def session() -> Iterator[Session]:
    """A fresh in-memory SQLite session with all tables created, per test."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db_session = session_factory()
    try:
        yield db_session
    finally:
        db_session.close()


@pytest.fixture()
def service(session: Session, tmp_path: Path) -> BenchmarkService:
    """A BenchmarkService writing dataset files under a per-test tmp directory."""
    return BenchmarkService(session, storage_dir=tmp_path / "datasets")


class TestCreateBenchmark:
    def test_creates_benchmark_with_expected_fields(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(
            name="MedQA-Core", domain="Medical", description="Medical Q&A suite."
        )

        assert benchmark.benchmark_id
        assert benchmark.name == "MedQA-Core"
        assert benchmark.domain == "Medical"
        assert benchmark.description == "Medical Q&A suite."


class TestUploadDatasetValid:
    def test_valid_csv_creates_dataset_version(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")

        dataset_version, result = service.upload_dataset(
            benchmark_id=benchmark.benchmark_id,
            file=_csv_file(VALID_CSV),
            version_label="v1.0",
        )

        assert result.is_valid
        assert dataset_version.dataset_version_id
        assert dataset_version.version == "v1.0"
        assert dataset_version.question_count == 2
        assert dataset_version.checksum == service.compute_checksum(
            VALID_CSV.encode("utf-8")
        )

    def test_valid_csv_is_written_to_disk_and_previewable(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")
        dataset_version, _ = service.upload_dataset(
            benchmark_id=benchmark.benchmark_id,
            file=_csv_file(VALID_CSV),
            version_label="v1.0",
        )

        preview = service.preview_dataset(dataset_version.dataset_version_id, rows=1)

        assert len(preview) == 1
        assert preview.iloc[0]["question"] == "What is 2+2?"

    def test_upload_to_unknown_benchmark_raises_lookup_error(self, service: BenchmarkService) -> None:
        with pytest.raises(LookupError):
            service.upload_dataset(
                benchmark_id="does-not-exist",
                file=_csv_file(VALID_CSV),
                version_label="v1.0",
            )


class TestUploadDatasetInvalid:
    def test_missing_required_column_is_rejected(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")

        with pytest.raises(DatasetUploadError) as exc_info:
            service.upload_dataset(
                benchmark_id=benchmark.benchmark_id,
                file=_csv_file(MISSING_COLUMN_CSV),
                version_label="v1.0",
            )

        result = exc_info.value.validation_result
        assert not result.is_valid
        assert "ground_truth" in result.missing_columns
        assert "Missing column: ground_truth" in result.error_summary

    def test_empty_question_and_ground_truth_cells_are_rejected(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")

        with pytest.raises(DatasetUploadError) as exc_info:
            service.upload_dataset(
                benchmark_id=benchmark.benchmark_id,
                file=_csv_file(EMPTY_CELL_CSV),
                version_label="v1.0",
            )

        messages = exc_info.value.validation_result.error_summary
        assert any("Empty question cell" in m for m in messages)
        assert any("Empty ground_truth cell" in m for m in messages)

    def test_invalid_difficulty_value_is_rejected(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")

        with pytest.raises(DatasetUploadError) as exc_info:
            service.upload_dataset(
                benchmark_id=benchmark.benchmark_id,
                file=_csv_file(BAD_DIFFICULTY_CSV),
                version_label="v1.0",
            )

        messages = exc_info.value.validation_result.error_summary
        assert any("Invalid difficulty" in m for m in messages)

    def test_rejected_upload_does_not_create_dataset_version(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")

        with pytest.raises(DatasetUploadError):
            service.upload_dataset(
                benchmark_id=benchmark.benchmark_id,
                file=_csv_file(MISSING_COLUMN_CSV),
                version_label="v1.0",
            )

        assert service.get_benchmark(benchmark.benchmark_id).dataset_versions == []

    def test_rejected_upload_does_not_write_file_to_disk(
        self, service: BenchmarkService, tmp_path: Path
    ) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")

        with pytest.raises(DatasetUploadError):
            service.upload_dataset(
                benchmark_id=benchmark.benchmark_id,
                file=_csv_file(BAD_DIFFICULTY_CSV),
                version_label="v1.0",
            )

        assert list((tmp_path / "datasets").glob("*.csv")) == []


class TestListAndGetBenchmarks:
    def test_list_benchmarks_reflects_latest_version(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")
        service.upload_dataset(
            benchmark_id=benchmark.benchmark_id,
            file=_csv_file(VALID_CSV),
            version_label="v1.0",
        )

        summaries = service.list_benchmarks()

        assert len(summaries) == 1
        assert summaries[0].name == "MathBench"
        assert summaries[0].latest_version == "v1.0"
        assert summaries[0].latest_question_count == 2
        assert summaries[0].version_count == 1

    def test_list_benchmarks_filters_by_domain(self, service: BenchmarkService) -> None:
        service.create_benchmark(name="MathBench", domain="Math")
        service.create_benchmark(name="MedQA-Core", domain="Medical")

        math_only = service.list_benchmarks(domain="Math")

        assert [b.name for b in math_only] == ["MathBench"]

    def test_get_benchmark_returns_all_versions(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")
        service.upload_dataset(
            benchmark_id=benchmark.benchmark_id,
            file=_csv_file(VALID_CSV),
            version_label="v1.0",
        )
        service.upload_dataset(
            benchmark_id=benchmark.benchmark_id,
            file=_csv_file(VALID_CSV),
            version_label="v1.1",
        )

        detail = service.get_benchmark(benchmark.benchmark_id)

        assert detail.name == "MathBench"
        assert {v.version for v in detail.dataset_versions} == {"v1.0", "v1.1"}

    def test_get_unknown_benchmark_raises_lookup_error(self, service: BenchmarkService) -> None:
        with pytest.raises(LookupError):
            service.get_benchmark("does-not-exist")


class TestComputeChecksum:
    def test_checksum_is_deterministic_sha256(self, service: BenchmarkService) -> None:
        content = b"hello world"
        import hashlib

        assert service.compute_checksum(content) == hashlib.sha256(content).hexdigest()

    def test_checksum_differs_for_different_content(self, service: BenchmarkService) -> None:
        assert service.compute_checksum(b"a") != service.compute_checksum(b"b")


class TestDeleteBenchmark:
    def test_delete_existing_benchmark_returns_true(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")

        assert service.delete_benchmark(benchmark.benchmark_id) is True
        with pytest.raises(LookupError):
            service.get_benchmark(benchmark.benchmark_id)

    def test_delete_unknown_benchmark_returns_false(self, service: BenchmarkService) -> None:
        assert service.delete_benchmark("does-not-exist") is False

    def test_delete_cascades_to_dataset_versions(self, service: BenchmarkService) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")
        service.upload_dataset(
            benchmark_id=benchmark.benchmark_id,
            file=_csv_file(VALID_CSV),
            version_label="v1.0",
        )

        service.delete_benchmark(benchmark.benchmark_id)

        assert service.list_benchmarks() == []

    def test_delete_removes_stored_dataset_files(
        self, service: BenchmarkService, tmp_path: Path
    ) -> None:
        benchmark = service.create_benchmark(name="MathBench", domain="Math")
        dataset_version, _ = service.upload_dataset(
            benchmark_id=benchmark.benchmark_id,
            file=_csv_file(VALID_CSV),
            version_label="v1.0",
        )
        stored_path = tmp_path / "datasets" / f"{dataset_version.dataset_version_id}.csv"
        assert stored_path.exists()

        service.delete_benchmark(benchmark.benchmark_id)

        assert not stored_path.exists()


class TestPreviewDataset:
    def test_preview_unknown_dataset_version_raises_file_not_found(
        self, service: BenchmarkService
    ) -> None:
        with pytest.raises(FileNotFoundError):
            service.preview_dataset("does-not-exist")