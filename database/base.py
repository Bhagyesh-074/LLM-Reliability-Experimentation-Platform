"""
database/base.py

Declarative base for all SQLAlchemy ORM models in the platform.

A shared metadata naming convention is applied so that Alembic
autogenerates deterministic, human-readable constraint names
(important for reliable migrations on SQLite, which otherwise assigns
anonymous names to constraints).
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Naming convention recommended by the SQLAlchemy docs for Alembic
# compatibility. Applies consistently across SQLite / PostgreSQL / MySQL.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """
    Shared declarative base class for all ORM models.

    All model classes in `database.models` must inherit from this base
    so that they share a single `MetaData` instance, which is required
    for `Base.metadata.create_all()` and for Alembic autogeneration to
    see every table.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)