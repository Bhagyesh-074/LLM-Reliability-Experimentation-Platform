"""Unit tests for PromptService, run against an in-memory SQLite database."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.base import Base
from database import models  # noqa: F401  (register all mapped classes on Base)
from registry.prompt_service import (
    DuplicatePromptNameError,
    InvalidPromptSyntaxError,
    NoOpVersionError,
    PromptNotFoundError,
    PromptService,
)
from registry.schemas import PromptCreate, PromptVersionCreate


@pytest.fixture()
def session():
    """A fresh in-memory SQLite session with all tables created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        yield sess


@pytest.fixture()
def service(session: Session) -> PromptService:
    return PromptService(session)


def make_prompt(**overrides) -> PromptCreate:
    defaults = dict(
        name="customer-support-triage",
        description="Triages inbound support tickets.",
        author="bhagyesh",
        content="You are a helpful assistant. Context: {context}",
        tags=["support", "triage"],
        status="draft",
    )
    defaults.update(overrides)
    return PromptCreate(**defaults)


# ----------------------------------------------------------------------
# validate_prompt_syntax
# ----------------------------------------------------------------------


class TestValidatePromptSyntax:
    def test_valid_single_placeholder(self):
        result = PromptService.validate_prompt_syntax("Hello {name}!")
        assert result.is_valid
        assert result.variables == ["name"]
        assert result.errors == []

    def test_valid_multiple_and_dotted_placeholders(self):
        result = PromptService.validate_prompt_syntax(
            "Hi {user.name}, your order {order_id} is ready. Bye {user.name}."
        )
        assert result.is_valid
        # dedup, preserves first-seen order
        assert result.variables == ["user.name", "order_id"]

    def test_no_placeholders_is_valid(self):
        result = PromptService.validate_prompt_syntax("Just plain text.")
        assert result.is_valid
        assert result.variables == []

    def test_escaped_double_braces_are_literal(self):
        result = PromptService.validate_prompt_syntax("Literal braces: {{not_a_var}}")
        assert result.is_valid
        assert result.variables == []

    def test_unbalanced_open_brace_is_invalid(self):
        result = PromptService.validate_prompt_syntax("Missing close: {name")
        assert not result.is_valid
        assert any("Unbalanced braces" in e for e in result.errors)

    def test_unbalanced_close_brace_is_invalid(self):
        result = PromptService.validate_prompt_syntax("Stray close: name}")
        assert not result.is_valid
        assert any("Unbalanced braces" in e for e in result.errors)

    def test_empty_placeholder_is_invalid(self):
        result = PromptService.validate_prompt_syntax("Empty: {}")
        assert not result.is_valid
        assert any("Empty placeholder" in e for e in result.errors)

    def test_invalid_variable_name_is_invalid(self):
        result = PromptService.validate_prompt_syntax("Bad: {1abc}")
        assert not result.is_valid
        assert any("Invalid variable name" in e for e in result.errors)

    def test_nested_placeholder_is_invalid(self):
        result = PromptService.validate_prompt_syntax("Nested: {a{b}}")
        assert not result.is_valid


# ----------------------------------------------------------------------
# compute_content_hash
# ----------------------------------------------------------------------


class TestComputeContentHash:
    def test_deterministic(self):
        h1 = PromptService.compute_content_hash("hello world")
        h2 = PromptService.compute_content_hash("hello world")
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex digest length

    def test_differs_for_different_content(self):
        h1 = PromptService.compute_content_hash("hello world")
        h2 = PromptService.compute_content_hash("hello there")
        assert h1 != h2


# ----------------------------------------------------------------------
# create_prompt
# ----------------------------------------------------------------------


class TestCreatePrompt:
    def test_creates_prompt_and_first_version(self, service: PromptService):
        response = service.create_prompt(make_prompt())

        assert response.name == "customer-support-triage"
        assert response.current_version == 1
        assert response.tags == ["support", "triage"]
        assert response.prompt_id

    def test_rejects_duplicate_name(self, service: PromptService):
        service.create_prompt(make_prompt())
        with pytest.raises(DuplicatePromptNameError):
            service.create_prompt(make_prompt())

    def test_rejects_invalid_syntax(self, service: PromptService):
        with pytest.raises(InvalidPromptSyntaxError):
            service.create_prompt(make_prompt(content="Unbalanced {oops"))

    def test_first_version_hash_matches_content(self, service: PromptService):
        payload = make_prompt(content="Fixed content, no vars.")
        response = service.create_prompt(payload)
        full = service.get_prompt(response.prompt_id)
        assert full.versions is not None
        assert full.versions[0].content_hash == PromptService.compute_content_hash(
            "Fixed content, no vars."
        )


# ----------------------------------------------------------------------
# create_version
# ----------------------------------------------------------------------


class TestCreateVersion:
    def test_increments_version_number(self, service: PromptService):
        created = service.create_prompt(make_prompt())
        v2 = service.create_version(
            created.prompt_id,
            PromptVersionCreate(content="Updated content v2 {context}", tags=["v2"]),
        )
        assert v2.version == 2

        history = service.get_version_history(created.prompt_id)
        assert [v.version for v in history] == [2, 1]  # newest first

    def test_rejects_invalid_syntax(self, service: PromptService):
        created = service.create_prompt(make_prompt())
        with pytest.raises(InvalidPromptSyntaxError):
            service.create_version(
                created.prompt_id, PromptVersionCreate(content="{bad")
            )

    def test_rejects_unknown_prompt(self, service: PromptService):
        with pytest.raises(PromptNotFoundError):
            service.create_version(
                "does-not-exist", PromptVersionCreate(content="hello")
            )

    def test_rejects_identical_content_noop(self, service: PromptService):
        payload = make_prompt(content="Same content every time.")
        created = service.create_prompt(payload)
        with pytest.raises(NoOpVersionError):
            service.create_version(
                created.prompt_id,
                PromptVersionCreate(content="Same content every time."),
            )

    def test_versions_are_immutable_rows(self, service: PromptService):
        """There is no update/edit path — only new versions are appended."""
        created = service.create_prompt(make_prompt())
        service.create_version(
            created.prompt_id, PromptVersionCreate(content="v2 body {context}")
        )
        history = service.get_version_history(created.prompt_id)
        assert len(history) == 2
        assert history[1].content == make_prompt().content  # v1 untouched


# ----------------------------------------------------------------------
# list_prompts
# ----------------------------------------------------------------------


class TestListPrompts:
    def test_lists_all_by_default(self, service: PromptService):
        service.create_prompt(make_prompt(name="p1"))
        service.create_prompt(make_prompt(name="p2", author="someone_else"))

        results = service.list_prompts()
        assert {r.name for r in results} == {"p1", "p2"}

    def test_filters_by_author(self, service: PromptService):
        service.create_prompt(make_prompt(name="p1", author="alice"))
        service.create_prompt(make_prompt(name="p2", author="bob"))

        results = service.list_prompts(author="alice")
        assert [r.name for r in results] == ["p1"]

    def test_filters_by_status(self, service: PromptService):
        service.create_prompt(make_prompt(name="p1", status="active"))
        service.create_prompt(make_prompt(name="p2", status="draft"))

        results = service.list_prompts(status="active")
        assert [r.name for r in results] == ["p1"]

    def test_filters_by_tag_on_latest_version(self, service: PromptService):
        created = service.create_prompt(make_prompt(name="p1", tags=["v1-tag"]))
        service.create_prompt(make_prompt(name="p2", tags=["other"]))

        # Retag p1 by creating a new version with a different tag set.
        service.create_version(
            created.prompt_id,
            PromptVersionCreate(content="new body {context}", tags=["fresh-tag"]),
        )

        results = service.list_prompts(tag="fresh-tag")
        assert [r.name for r in results] == ["p1"]

        # Old tag no longer matches, since only the latest version's tags count.
        assert service.list_prompts(tag="v1-tag") == []


# ----------------------------------------------------------------------
# get_prompt / get_version_history
# ----------------------------------------------------------------------


class TestGetPrompt:
    def test_returns_full_history(self, service: PromptService):
        created = service.create_prompt(make_prompt())
        service.create_version(
            created.prompt_id, PromptVersionCreate(content="v2 {context}")
        )
        service.create_version(
            created.prompt_id, PromptVersionCreate(content="v3 {context}")
        )

        full = service.get_prompt(created.prompt_id)
        assert full.current_version == 3
        assert full.versions is not None
        assert [v.version for v in full.versions] == [1, 2, 3]

    def test_raises_for_unknown_prompt(self, service: PromptService):
        with pytest.raises(PromptNotFoundError):
            service.get_prompt("does-not-exist")


class TestGetVersionHistory:
    def test_newest_first(self, service: PromptService):
        created = service.create_prompt(make_prompt())
        service.create_version(
            created.prompt_id, PromptVersionCreate(content="v2 {context}")
        )

        history = service.get_version_history(created.prompt_id)
        assert [v.version for v in history] == [2, 1]

    def test_raises_for_unknown_prompt(self, service: PromptService):
        with pytest.raises(PromptNotFoundError):
            service.get_version_history("does-not-exist")