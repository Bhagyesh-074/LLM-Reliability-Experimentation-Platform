"""Repository for Benchmark and DatasetVersion access, enforcing version immutability.

Per DATABASE_SCHEMA.md, dataset versions are immutable once created:
there is intentionally no method to edit an existing ``DatasetVersion``
row. Changing the underlying question set always means creating a new
version via :meth:`BenchmarkRepository.add_dataset_version`.
"""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Benchmark, DatasetVersion
from database.repositories.base import BaseRepository


class BenchmarkRepository(BaseRepository[Benchmark]):
    """CRUD and lookup helpers for benchmarks and their immutable dataset versions."""

    def __init__(self, session: Session) -> None:
        """Bind this repository to a session, targeting the ``Benchmark`` model."""
        super().__init__(Benchmark, session)

    def get_by_name(self, name: str) -> Optional[Benchmark]:
        """Look up a benchmark by its display name (returns the first match)."""
        stmt = select(Benchmark).where(Benchmark.name == name)
        return self.session.execute(stmt).scalars().first()

    def add_dataset_version(
        self,
        benchmark_id: str,
        version: str,
        question_count: Optional[int] = None,
        checksum: Optional[str] = None,
    ) -> DatasetVersion:
        """Create a new immutable ``DatasetVersion`` for a benchmark."""
        dataset_version = DatasetVersion(
            benchmark_id=benchmark_id,
            version=version,
            question_count=question_count,
            checksum=checksum,
        )
        self.session.add(dataset_version)
        self.session.flush()
        return dataset_version

    def list_dataset_versions(self, benchmark_id: str) -> Sequence[DatasetVersion]:
        """Return all dataset versions for a benchmark, newest first."""
        stmt = (
            select(DatasetVersion)
            .where(DatasetVersion.benchmark_id == benchmark_id)
            .order_by(DatasetVersion.created_at.desc())
        )
        return self.session.execute(stmt).scalars().all()

    def get_dataset_version_by_version(
        self, benchmark_id: str, version: str
    ) -> Optional[DatasetVersion]:
        """Look up a specific dataset version by its version label."""
        stmt = select(DatasetVersion).where(
            DatasetVersion.benchmark_id == benchmark_id,
            DatasetVersion.version == version,
        )
        return self.session.execute(stmt).scalars().first()