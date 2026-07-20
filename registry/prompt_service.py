"""Business logic for the Prompt Registry (FR-2).

``PromptService`` sits between the UI / API layer and
``PromptRepository``. It owns:

- turning DTOs (``registry.schemas``) into ORM writes and back,
- prompt-template syntax validation,
- content hashing,
- the "no-op version" guard (refusing to create a new immutable version
  whose content is byte-identical to the current one).

Per DATABASE_SCHEMA.md, ``prompts`` rows are mutable (name, description,
author, status) but ``prompt_versions`` rows are immutable — there is no
"edit" path for an existing version, only ``create_version``.

Depends on ``database.repositories.base.BaseRepository`` (generic
``get``/``get_or_raise``/``create``/``list``/``update``/``delete``) and
the ``Prompt``-specific ``PromptRepository`` (``get_by_name``,
``get_latest_version``, ``add_version``, ``list_versions``,
``get_version``).
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Prompt, PromptVersion
from database.repositories.prompt_repository import PromptRepository
from registry.schemas import (
    PromptCreate,
    PromptResponse,
    PromptValidationResult,
    PromptVersionCreate,
    PromptVersionResponse,
)

_VARIABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_ESCAPED_OPEN_SENTINEL = "\x00__ESCAPED_OPEN__\x00"
_ESCAPED_CLOSE_SENTINEL = "\x00__ESCAPED_CLOSE__\x00"
_PLACEHOLDER_RE = re.compile(r"\{([^{}]*)\}")


class PromptNotFoundError(LookupError):
    """Raised when a prompt_id doesn't correspond to any stored prompt."""


class DuplicatePromptNameError(ValueError):
    """Raised when creating a prompt whose name is already taken."""


class InvalidPromptSyntaxError(ValueError):
    """Raised when prompt template content fails syntax validation."""


class NoOpVersionError(ValueError):
    """Raised when a new version's content is identical to the current one."""


class PromptService:
    """Prompt Registry service: create/version prompts, list, validate."""

    def __init__(self, session: Session) -> None:
        """Bind the service to a session, reusing PromptRepository for storage."""
        self.session = session
        self.repo = PromptRepository(session)

    # ------------------------------------------------------------------
    # Hashing / validation (pure functions, no I/O)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Return the sha256 hex digest of a prompt version's content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def validate_prompt_syntax(content: str) -> PromptValidationResult:
        """Validate ``{variable}`` placeholder syntax in a prompt template.

        Rules:
        - ``{{`` and ``}}`` are treated as escaped literal braces (as in
          Python's ``str.format``) and are not variables.
        - Every remaining ``{`` must be paired with a ``}`` on the same
          "level" — placeholders do not nest.
        - A placeholder's body must be a non-empty identifier, optionally
          dotted (e.g. ``{user.name}``); it cannot start with a digit.

        Returns a :class:`PromptValidationResult` with ``is_valid=False``
        and one or more human-readable ``errors`` if the template is
        malformed, otherwise ``is_valid=True`` with the distinct variable
        names found in ``variables``.
        """
        errors: list[str] = []

        working = content.replace("{{", _ESCAPED_OPEN_SENTINEL).replace(
            "}}", _ESCAPED_CLOSE_SENTINEL
        )

        open_count = working.count("{")
        close_count = working.count("}")
        if open_count != close_count:
            errors.append(
                f"Unbalanced braces: {open_count} '{{' vs {close_count} '}}'."
            )

        variables: list[str] = []
        matched_spans: list[tuple[int, int]] = []
        for match in _PLACEHOLDER_RE.finditer(working):
            matched_spans.append(match.span())
            raw_name = match.group(1).strip()
            if not raw_name:
                errors.append("Empty placeholder '{}' is not allowed.")
                continue
            if not _VARIABLE_NAME_RE.match(raw_name):
                errors.append(f"Invalid variable name in placeholder: '{raw_name}'.")
                continue
            if raw_name not in variables:
                variables.append(raw_name)

        # Anything left over after stripping matched, well-formed
        # placeholders indicates nesting or a stray unmatched brace, e.g.
        # "{{a}" or "{a{b}}".
        residual = _PLACEHOLDER_RE.sub("", working)
        if "{" in residual or "}" in residual:
            errors.append(
                "Malformed or nested placeholder braces detected outside "
                "simple '{variable}' patterns."
            )

        return PromptValidationResult(
            is_valid=len(errors) == 0, errors=errors, variables=variables
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def create_prompt(self, payload: PromptCreate) -> PromptResponse:
        """Create a new ``Prompt`` plus its immutable first version (v1).

        Raises:
            InvalidPromptSyntaxError: if ``payload.content`` fails
                placeholder validation.
            DuplicatePromptNameError: if a prompt with the same name
                already exists.
        """
        validation = self.validate_prompt_syntax(payload.content)
        if not validation.is_valid:
            raise InvalidPromptSyntaxError("; ".join(validation.errors))

        if self.repo.get_by_name(payload.name) is not None:
            raise DuplicatePromptNameError(
                f"A prompt named '{payload.name}' already exists."
            )

        prompt = Prompt(
            name=payload.name,
            description=payload.description,
            author=payload.author,
            status=payload.status,
        )
        self.session.add(prompt)
        self.session.flush()  # populate prompt.prompt_id

        first_version = self.repo.add_version(
            prompt_id=prompt.prompt_id, content=payload.content, tags=payload.tags
        )
        self.session.commit()

        return self._to_prompt_response(prompt, latest=first_version)

    def create_version(
        self, prompt_id: str, payload: PromptVersionCreate
    ) -> PromptVersionResponse:
        """Create the next immutable ``PromptVersion`` for an existing prompt.

        Raises:
            InvalidPromptSyntaxError: if ``payload.content`` fails
                placeholder validation.
            PromptNotFoundError: if ``prompt_id`` doesn't exist.
            NoOpVersionError: if ``payload.content`` hashes identically to
                the current latest version (immutable versions shouldn't
                be created just to duplicate existing content).
        """
        validation = self.validate_prompt_syntax(payload.content)
        if not validation.is_valid:
            raise InvalidPromptSyntaxError("; ".join(validation.errors))

        self._get_prompt_or_raise(prompt_id)

        latest = self.repo.get_latest_version(prompt_id)
        if latest is not None:
            new_hash = self.compute_content_hash(payload.content)
            if latest.content_hash == new_hash:
                raise NoOpVersionError(
                    "Content is identical to the current latest version; "
                    "no new version was created."
                )

        version = self.repo.add_version(
            prompt_id=prompt_id, content=payload.content, tags=payload.tags
        )
        self.session.commit()
        return PromptVersionResponse.model_validate(version)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def list_prompts(
        self,
        tag: Optional[str] = None,
        author: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[PromptResponse]:
        """List prompts, each enriched with its latest version's number/tags.

        Args:
            tag: if given, only include prompts whose *latest* version
                carries this tag.
            author: if given, only include prompts by this author.
            status: if given, only include prompts with this status
                (``draft`` / ``active`` / ``deprecated``).
        """
        stmt = select(Prompt)
        if author is not None:
            stmt = stmt.where(Prompt.author == author)
        if status is not None:
            stmt = stmt.where(Prompt.status == status)

        prompts = self.session.execute(stmt).scalars().all()

        results: list[PromptResponse] = []
        for prompt in prompts:
            latest = self.repo.get_latest_version(prompt.prompt_id)
            latest_tags = list(latest.tags or []) if latest else []
            if tag is not None and tag not in latest_tags:
                continue
            results.append(self._to_prompt_response(prompt, latest))
        return results

    def get_prompt(self, prompt_id: str) -> PromptResponse:
        """Return a prompt with its full immutable version history attached.

        Raises:
            PromptNotFoundError: if ``prompt_id`` doesn't exist.
        """
        prompt = self._get_prompt_or_raise(prompt_id)
        versions = self.repo.list_versions(prompt_id)
        latest = versions[-1] if versions else None

        response = self._to_prompt_response(prompt, latest)
        response.versions = [
            PromptVersionResponse.model_validate(v) for v in versions
        ]
        return response

    def get_version_history(self, prompt_id: str) -> list[PromptVersionResponse]:
        """Return all versions of a prompt, newest first.

        Raises:
            PromptNotFoundError: if ``prompt_id`` doesn't exist.
        """
        self._get_prompt_or_raise(prompt_id)
        versions = self.repo.list_versions(prompt_id)
        return [
            PromptVersionResponse.model_validate(v) for v in reversed(versions)
        ]

    def get_version_by_id(self, version_id: str) -> PromptVersionResponse:
        """Look up a single immutable prompt version directly by its primary key.

        Added for the Evaluation Orchestrator, which only stores a bare
        ``PromptVersion.version_id`` in ``EvaluationConfig`` (not the
        parent ``prompt_id``), so ``get_prompt``/``get_version_history``
        can't be used to resolve it.

        Raises:
            PromptNotFoundError: if no version exists with this id.
        """
        version = self.repo.get_version_by_id(version_id)
        if version is None:
            raise PromptNotFoundError(
                f"No prompt version found with id '{version_id}'."
            )
        return PromptVersionResponse.model_validate(version)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_prompt_or_raise(self, prompt_id: str) -> Prompt:
        try:
            return self.repo.get_or_raise(prompt_id)
        except LookupError as exc:
            raise PromptNotFoundError(
                f"No prompt found with id '{prompt_id}'."
            ) from exc

    @staticmethod
    def _to_prompt_response(
        prompt: Prompt, latest: Optional[PromptVersion]
    ) -> PromptResponse:
        return PromptResponse(
            prompt_id=prompt.prompt_id,
            name=prompt.name,
            description=prompt.description,
            author=prompt.author,
            status=prompt.status,
            created_at=prompt.created_at,
            current_version=latest.version if latest else 0,
            tags=list(latest.tags or []) if latest else [],
        )