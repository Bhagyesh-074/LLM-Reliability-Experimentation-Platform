"""
database/repositories/base.py

Generic base repository implementing typed CRUD operations shared by
every entity in the platform (providers, prompts, prompt_versions,
benchmarks, dataset_versions, evaluation_runs, evaluation_results,
run_metrics, failure_analysis, mlflow_runs).

Concrete repositories should subclass `BaseRepository[ModelType]` and
add entity-specific query methods on top of the generic CRUD surface,
e.g.:

    class ProviderRepository(BaseRepository[Provider]):
        def __init__(self, session: Session) -> None:
            super().__init__(Provider, session)

        def get_by_name(self, name: str) -> Optional[Provider]:
            return self.session.execute(
                select(Provider).where(Provider.name == name)
            ).scalar_one_or_none()
"""

from __future__ import annotations

from typing import Any, Generic, List, Optional, Sequence, Type, TypeVar

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Generic repository providing Create / Read / Update / Delete
    operations for a single SQLAlchemy model.

    A repository does not manage transactions itself: callers are
    expected to commit (e.g. via `database.session.session_scope`) or
    to rely on the owning service layer to do so. This keeps
    repositories composable within a single unit of work spanning
    multiple entities.
    """

    def __init__(self, model: Type[ModelType], session: Session) -> None:
        """
        Args:
            model: The SQLAlchemy declarative model class this
                repository operates on.
            session: An active SQLAlchemy `Session`.
        """
        self.model: Type[ModelType] = model
        self.session: Session = session

    def create(self, **kwargs: Any) -> ModelType:
        """
        Instantiate and persist a new row.

        Flushes (but does not commit) so that any server-side or
        default-generated primary key is immediately available on the
        returned instance.
        """
        instance = self.model(**kwargs)
        self.session.add(instance)
        self.session.flush()
        logger.debug("Created {} ({!r})", self.model.__name__, kwargs)
        return instance

    def get(self, pk: Any) -> Optional[ModelType]:
        """Fetch a single row by primary key, or `None` if not found."""
        return self.session.get(self.model, pk)

    def get_or_raise(self, pk: Any) -> ModelType:
        """Fetch a single row by primary key, raising `LookupError` if absent."""
        instance = self.get(pk)
        if instance is None:
            raise LookupError(f"{self.model.__name__} with primary key {pk!r} not found")
        return instance

    def list(
        self,
        *,
        offset: int = 0,
        limit: Optional[int] = None,
        **filters: Any,
    ) -> Sequence[ModelType]:
        """
        Fetch multiple rows, optionally filtered by exact-match column
        values, with pagination.

        Args:
            offset: Number of rows to skip.
            limit: Maximum number of rows to return (`None` = no limit).
            **filters: Column-name/value pairs matched with equality.
        """
        stmt = select(self.model)
        for column_name, value in filters.items():
            column = getattr(self.model, column_name)
            stmt = stmt.where(column == value)
        stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.execute(stmt).scalars().all()

    def update(self, pk: Any, **kwargs: Any) -> ModelType:
        """
        Update an existing row's attributes by primary key.

        Raises:
            LookupError: If no row exists for `pk`.
        """
        instance = self.get_or_raise(pk)
        for field, value in kwargs.items():
            setattr(instance, field, value)
        self.session.flush()
        logger.debug("Updated {} pk={!r} ({!r})", self.model.__name__, pk, kwargs)
        return instance

    def delete(self, pk: Any) -> bool:
        """
        Delete a row by primary key.

        Returns:
            `True` if a row was deleted, `False` if it did not exist.
        """
        instance = self.get(pk)
        if instance is None:
            return False
        self.session.delete(instance)
        self.session.flush()
        logger.debug("Deleted {} pk={!r}", self.model.__name__, pk)
        return True

    def count(self, **filters: Any) -> int:
        """Return the number of rows matching the given equality filters."""
        stmt = select(self.model)
        for column_name, value in filters.items():
            column = getattr(self.model, column_name)
            stmt = stmt.where(column == value)
        return len(self.session.execute(stmt).scalars().all())

    def all(self) -> List[ModelType]:
        """Return every row for this model. Use with care on large tables."""
        return list(self.session.execute(select(self.model)).scalars().all())
        