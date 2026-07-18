"""
registry/benchmark_service.py

Service layer for the Benchmark Registry.

Wraps ``BenchmarkRepository`` with the higher-level operations the
dashboard (and, later, other callers) need: creating benchmarks,
validating and persisting new immutable dataset versions, and reading
back benchmarks/versions in shapes convenient for display.

Dataset versions are immutable by contract (see DATABASE_SCHEMA.md and
``BenchmarkRepository``'s module docstring) — there is intentionally no
"update dataset version" method here.

Raw file storage
-----------------
``DatasetVersion`` stores only a ``checksum`` and a ``question_count``;
the schema has no column for the underlying rows. So that
``preview_dataset`` can re-read a dataset's content after the request
that uploaded it has ended, each uploaded file's raw bytes are also
written to local disk under ``storage_dir``, named
``<dataset_version_id>.csv``. The DB checksum remains the source of
truth for integrity verification; the file on disk is the source of
truth for content.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.models import Benchmark, DatasetVersion
from database.repositories.benchmark_repository import BenchmarkRepository
from registry.dataset_validator import ValidationResult, validate

DATASET_STORAGE_DIR: Path = Path("data/datasets")


class DatasetUploadError(Exception):
    """Raised when an uploaded dataset fails schema validation.

    No ``DatasetVersion`` row is created and no file is written to disk
    when this is raised — the caller can inspect ``validation_result``
    for the specific, per-column and per-row failures.
    """

    def __init__(self, validation_result: ValidationResult) -> None:
        self.validation_result = validation_result
        super().__init__("Dataset failed validation: " + "; ".join(validation_result.error_summary))


class DatasetVersionInfo(BaseModel):
    """A single dataset version, for detail views."""

    dataset_version_id: str
    version: str
    question_count: Optional[int] = None
    checksum: Optional[str] = None
    created_at: Optional[str] = None

    model_config = {"from_attributes": True}


class BenchmarkDetail(BaseModel):
    """A benchmark with all of its dataset versions, for detail views."""

    benchmark_id: str
    name: str
    domain: Optional[str] = None
    description: Optional[str] = None
    dataset_versions: List[DatasetVersionInfo] = Field(default_factory=list)


class BenchmarkSummary(BaseModel):
    """A benchmark plus a summary of its latest dataset version, for list views."""

    benchmark_id: str
    name: str
    domain: Optional[str] = None
    description: Optional[str] = None
    latest_version: Optional[str] = None
    latest_question_count: Optional[int] = None
    version_count: int = 0


class BenchmarkService:
    """Application-level operations for benchmarks and their dataset versions."""

    def __init__(self, session: Session, storage_dir: Path = DATASET_STORAGE_DIR) -> None:
        """
        Args:
            session: An active SQLAlchemy session. The service flushes
                but does not commit; callers own the transaction
                boundary (e.g. via ``database.session.session_scope``).
            storage_dir: Directory raw dataset files are written to,
                keyed by ``dataset_version_id``. Created if it does not
                already exist.
        """
        self.session = session
        self.repository = BenchmarkRepository(session)
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def create_benchmark(
        self,
        name: str,
        domain: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Benchmark:
        """Create a new benchmark record.

        Args:
            name: Display name, e.g. "MedQA-Core".
            domain: One of the platform's benchmark domains (e.g. "Medical").
            description: Free-text description.

        Returns:
            The newly created, flushed ``Benchmark`` row.
        """
        benchmark = self.repository.create(name=name, domain=domain, description=description)
        logger.info("Created benchmark {!r} (id={})", name, benchmark.benchmark_id)
        return benchmark

    def upload_dataset(
        self,
        benchmark_id: str,
        file: Any,
        version_label: str,
    ) -> Tuple[DatasetVersion, ValidationResult]:
        """
        Validate an uploaded dataset file and, if valid, persist it as a
        new immutable ``DatasetVersion``.

        Args:
            benchmark_id: The parent benchmark's primary key.
            file: A file-like object containing CSV content, exposing
                ``.read()`` (and, ideally, ``.seek()``) — e.g. a
                Streamlit ``UploadedFile`` or ``io.BytesIO``.
            version_label: Human-readable version tag, e.g. "v1.1".

        Returns:
            A tuple of ``(dataset_version, validation_result)``.

        Raises:
            LookupError: If ``benchmark_id`` does not exist.
            DatasetUploadError: If the file fails schema validation.
                Nothing is written to the database or disk in this case.
        """
        self.repository.get_or_raise(benchmark_id)

        raw_bytes = self._read_all_bytes(file)
        df = pd.read_csv(io.BytesIO(raw_bytes))

        validation_result = validate(df)
        if not validation_result.is_valid:
            logger.warning(
                "Dataset upload for benchmark {} rejected: {}",
                benchmark_id,
                validation_result.error_summary,
            )
            raise DatasetUploadError(validation_result)

        checksum = self.compute_checksum(raw_bytes)
        dataset_version = self.repository.add_dataset_version(
            benchmark_id=benchmark_id,
            version=version_label,
            question_count=len(df),
            checksum=checksum,
        )
        self._write_dataset_file(dataset_version.dataset_version_id, raw_bytes)

        logger.info(
            "Uploaded dataset version {} for benchmark {} ({} rows, checksum={})",
            version_label,
            benchmark_id,
            len(df),
            checksum,
        )
        return dataset_version, validation_result

    def delete_benchmark(self, benchmark_id: str) -> bool:
        """
        Permanently delete a benchmark and all of its dataset versions.

        Dataset version *content* is immutable once created (per
        DATABASE_SCHEMA.md), but that constraint governs editing, not
        the benchmark's lifecycle — deleting a benchmark outright,
        including its full version history, is a supported
        administrative action. The DB-level cascade
        (``Benchmark.dataset_versions`` uses
        ``cascade="all, delete-orphan"``) removes the ``DatasetVersion``
        rows; this method additionally removes each version's stored
        CSV file from disk, since that isn't tracked by the ORM.

        Args:
            benchmark_id: The benchmark's primary key.

        Returns:
            True if a benchmark was deleted, False if it did not exist.
        """
        versions = self.repository.list_dataset_versions(benchmark_id)
        deleted = self.repository.delete(benchmark_id)
        if deleted:
            for version in versions:
                self._delete_dataset_file(version.dataset_version_id)
            logger.info(
                "Deleted benchmark {} and its {} dataset version(s)",
                benchmark_id,
                len(versions),
            )
        return deleted

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def list_benchmarks(self, domain: Optional[str] = None) -> List[BenchmarkSummary]:
        """
        List benchmarks with a summary of their latest dataset version.

        Args:
            domain: If given, only benchmarks with this exact domain are
                returned. If omitted, all benchmarks are returned.

        Returns:
            One ``BenchmarkSummary`` per benchmark. ``latest_version`` /
            ``latest_question_count`` are ``None`` for benchmarks with no
            dataset versions yet.
        """
        if domain is not None:
            benchmarks = self.repository.list(domain=domain)
        else:
            benchmarks = self.repository.all()

        summaries: List[BenchmarkSummary] = []
        for benchmark in benchmarks:
            versions = self.repository.list_dataset_versions(benchmark.benchmark_id)
            latest = versions[0] if versions else None
            summaries.append(
                BenchmarkSummary(
                    benchmark_id=benchmark.benchmark_id,
                    name=benchmark.name,
                    domain=benchmark.domain,
                    description=benchmark.description,
                    latest_version=latest.version if latest else None,
                    latest_question_count=latest.question_count if latest else None,
                    version_count=len(versions),
                )
            )
        return summaries

    def get_benchmark(self, benchmark_id: str) -> BenchmarkDetail:
        """
        Fetch a single benchmark along with all of its dataset versions.

        Args:
            benchmark_id: The benchmark's primary key.

        Returns:
            A ``BenchmarkDetail`` with ``dataset_versions`` ordered
            newest-created first.

        Raises:
            LookupError: If ``benchmark_id`` does not exist.
        """
        benchmark = self.repository.get_or_raise(benchmark_id)
        versions = self.repository.list_dataset_versions(benchmark_id)
        return BenchmarkDetail(
            benchmark_id=benchmark.benchmark_id,
            name=benchmark.name,
            domain=benchmark.domain,
            description=benchmark.description,
            dataset_versions=[
                DatasetVersionInfo(
                    dataset_version_id=v.dataset_version_id,
                    version=v.version,
                    question_count=v.question_count,
                    checksum=v.checksum,
                    created_at=v.created_at.isoformat() if v.created_at else None,
                )
                for v in versions
            ],
        )

    def preview_dataset(self, dataset_version_id: str, rows: int = 5) -> pd.DataFrame:
        """
        Return the first ``rows`` rows of a previously uploaded dataset
        version, read back from local disk.

        Args:
            dataset_version_id: The dataset version's primary key.
            rows: Number of rows to return.

        Returns:
            A DataFrame with at most ``rows`` rows.

        Raises:
            FileNotFoundError: If no file is stored for this
                ``dataset_version_id`` (e.g. storage was cleared, or the
                version predates this service).
        """
        path = self._dataset_path(dataset_version_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No stored file found for dataset_version_id={dataset_version_id!r} at {path}"
            )
        return pd.read_csv(path, nrows=rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def compute_checksum(content: bytes) -> str:
        """Return the SHA-256 hex digest of raw dataset file content."""
        return hashlib.sha256(content).hexdigest()

    def _dataset_path(self, dataset_version_id: str) -> Path:
        """Return the on-disk path for a given dataset version's stored file."""
        return self.storage_dir / f"{dataset_version_id}.csv"

    def _write_dataset_file(self, dataset_version_id: str, raw_bytes: bytes) -> None:
        """Persist raw dataset bytes to disk, keyed by dataset_version_id."""
        self._dataset_path(dataset_version_id).write_bytes(raw_bytes)

    def _delete_dataset_file(self, dataset_version_id: str) -> None:
        """Remove a stored dataset file from disk, if present. No-op if already missing."""
        path = self._dataset_path(dataset_version_id)
        if path.exists():
            path.unlink()

    @staticmethod
    def _read_all_bytes(file: Any) -> bytes:
        """Read a file-like object's full content as bytes, rewinding first if possible."""
        if hasattr(file, "seek"):
            try:
                file.seek(0)
            except (OSError, ValueError):
                pass
        content = file.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        return content


__all__ = [
    "BenchmarkService",
    "BenchmarkSummary",
    "BenchmarkDetail",
    "DatasetVersionInfo",
    "DatasetUploadError",
    "DATASET_STORAGE_DIR",
]