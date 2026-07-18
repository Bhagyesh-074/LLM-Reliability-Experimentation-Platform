"""Alembic migration environment.

Wires Alembic's autogenerate support to the platform's SQLAlchemy
metadata (``database.base.Base.metadata``) so ``alembic revision
--autogenerate`` can diff the live database against the ORM models in
``database/models.py``.

The target database URL can be overridden via the ``DATABASE_URL``
environment variable (falling back to ``alembic.ini``'s
``sqlalchemy.url``), so the same migration scripts run unmodified
against SQLite (MVP) today and PostgreSQL/MySQL later.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ensure the project root (containing the `database` package) is importable
# regardless of the working directory `alembic` is invoked from.
sys.path.insert(0, os.getcwd())

from database.base import Base  # noqa: E402
from database import models  # noqa: E402,F401  (registers all models on Base.metadata)

# Alembic Config object, providing access to values in alembic.ini.
config = context.config

# Set up Python logging per alembic.ini, if a config file was supplied.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Allow overriding the configured URL via an environment variable, so the
# same alembic.ini works across environments (dev/staging/prod, or a
# future switch from SQLite to PostgreSQL/MySQL).
_database_url = os.environ.get("DATABASE_URL")
if _database_url:
    config.set_main_option("sqlalchemy.url", _database_url)

# Single source of truth for autogenerate: the platform's declarative
# metadata, which already contains every table defined in database/models.py.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and emits SQL to stdout
    rather than executing it against a live connection. Useful for
    generating migration SQL to hand off to a DBA.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode, against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()