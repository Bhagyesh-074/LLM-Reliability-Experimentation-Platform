"""
database/session.py

Engine and session factory for the platform database.

Defaults to a local SQLite file for the MVP, but is written entirely
against the SQLAlchemy engine abstraction so swapping in PostgreSQL or
MySQL later only requires changing the connection URL (e.g. via the
`DATABASE_URL` environment variable) and adding the corresponding
driver dependency.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator

from loguru import logger
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from database.base import Base

# All models must be imported so their tables are registered on
# `Base.metadata` before `create_all()` is called.
from database import models  # noqa: F401  (import for side effects)

DEFAULT_SQLITE_URL = "sqlite:///./data/llm_reliability_platform.db"

DATABASE_URL: str = os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL)

_is_sqlite = DATABASE_URL.startswith("sqlite")


def _ensure_sqlite_dir_exists(database_url: str) -> None:
    """
    Create the parent directory of a SQLite database file if it does
    not already exist.

    SQLite will not create missing intermediate directories itself, so
    if the configured path (e.g. ``./data/platform.db``, as mounted by
    the Docker Compose ``./data:/app/data`` volume) doesn't exist yet
    inside the container, connecting fails with
    ``sqlite3.OperationalError: unable to open database file``. This
    is a no-op for in-memory SQLite URLs and for non-SQLite backends.
    """
    if not database_url.startswith("sqlite"):
        return
    # Strip the "sqlite:///" (or "sqlite://") prefix to get the raw path.
    raw_path = database_url.split("///", 1)[-1] if "///" in database_url else ""
    if not raw_path or raw_path == ":memory:":
        return
    db_path = Path(raw_path)
    parent_dir = db_path.parent
    if parent_dir and str(parent_dir) not in ("", "."):
        parent_dir.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_dir_exists(DATABASE_URL)

_engine_kwargs: dict = {"echo": False, "future": True}
if _is_sqlite:
    # Required for SQLite when the same connection may be used across
    # threads (e.g. in web frameworks that reuse a session per request
    # but not per thread).
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine: Engine = create_engine(DATABASE_URL, **_engine_kwargs)


if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:  # type: ignore[no-untyped-def]
        """
        Enable foreign-key constraint enforcement on every new SQLite
        connection.

        SQLite ships with foreign keys disabled by default; without
        this pragma, the FK columns declared in `database.models` are
        documentation only and are never actually enforced.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
)


def init_db() -> None:
    """
    Create all tables registered on `Base.metadata` if they do not
    already exist.

    Safe to call on every application startup: `create_all` is a
    no-op for tables that already exist. Not a substitute for Alembic
    migrations once the schema needs to evolve on existing data.
    """
    logger.info("Initializing database schema at {}", DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ready.")


def get_session() -> Session:
    """Return a new `Session` bound to the configured engine."""
    return SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Provide a transactional scope around a series of operations.

    Commits on success, rolls back on exception, and always closes the
    session. Intended for use as:

        with session_scope() as session:
            session.add(obj)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Session rolled back due to an unhandled exception.")
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI-style dependency generator that yields a `Session` and
    guarantees it is closed afterwards.

    Usage (e.g. with FastAPI):

        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()