"""Repository for Prompt and PromptVersion access, enforcing version immutability.

Per DATABASE_SCHEMA.md, prompt versions are immutable once created: there
is intentionally no method to edit an existing ``PromptVersion`` row.
Changing a prompt's content always means creating a new version via
:meth:`PromptRepository.add_version`.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Prompt, PromptVersion
from database.repositories.base import BaseRepository


class PromptRepository(BaseRepository[Prompt]):
    """CRUD and lookup helpers for prompts and their immutable versions."""

    def __init__(self, session: Session) -> None:
        """Bind this repository to a session, targeting the ``Prompt`` model."""
        super().__init__(Prompt, session)

    def get_by_name(self, name: str) -> Optional[Prompt]:
        """Look up a prompt by its display name (returns the first match)."""
        stmt = select(Prompt).where(Prompt.name == name)
        return self.session.execute(stmt).scalars().first()

    def get_latest_version(self, prompt_id: str) -> Optional[PromptVersion]:
        """Return the highest-numbered version for a prompt, if any exist."""
        stmt = (
            select(PromptVersion)
            .where(PromptVersion.prompt_id == prompt_id)
            .order_by(PromptVersion.version.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalars().first()

    def add_version(
        self,
        prompt_id: str,
        content: str,
        tags: Optional[Any] = None,
    ) -> PromptVersion:
        """Create the next immutable ``PromptVersion`` for a prompt.

        The version number auto-increments from the highest existing
        version for the prompt (starting at 1). The content hash is
        computed automatically so downstream consumers can detect
        duplicate or drifted content without re-hashing themselves.
        """
        latest = self.get_latest_version(prompt_id)
        next_version = 1 if latest is None else latest.version + 1
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        version = PromptVersion(
            prompt_id=prompt_id,
            version=next_version,
            content=content,
            content_hash=content_hash,
            tags=tags,
        )
        self.session.add(version)
        self.session.flush()
        return version

    def list_versions(self, prompt_id: str) -> Sequence[PromptVersion]:
        """Return all versions of a prompt, ordered oldest to newest."""
        stmt = (
            select(PromptVersion)
            .where(PromptVersion.prompt_id == prompt_id)
            .order_by(PromptVersion.version.asc())
        )
        return self.session.execute(stmt).scalars().all()

    def get_version(self, prompt_id: str, version: int) -> Optional[PromptVersion]:
        """Look up a specific version number for a prompt."""
        stmt = select(PromptVersion).where(
            PromptVersion.prompt_id == prompt_id,
            PromptVersion.version == version,
        )
        return self.session.execute(stmt).scalars().first()