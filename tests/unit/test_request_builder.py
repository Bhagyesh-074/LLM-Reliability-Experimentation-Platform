"""Unit tests for core.evaluation.request_builder.RequestBuilder."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.evaluation.config import EvaluationConfig
from core.evaluation.request_builder import MissingTemplateVariableError, RequestBuilder
from registry.schemas import PromptVersionResponse


def _make_prompt_version(content: str) -> PromptVersionResponse:
    """Build a PromptVersionResponse with the given template content."""
    return PromptVersionResponse(
        version_id="pv-1",
        prompt_id="p-1",
        version=1,
        content=content,
        content_hash="deadbeef",
        tags=[],
        created_at=datetime.now(timezone.utc),
    )


def _make_config(**overrides: object) -> EvaluationConfig:
    """Build a valid EvaluationConfig, with optional field overrides."""
    kwargs = dict(
        provider_name="openai",
        model_name="gpt-4o",
        prompt_version_id="pv-1",
        dataset_version_id="dv-1",
        temperature=0.5,
        max_tokens=256,
    )
    kwargs.update(overrides)
    return EvaluationConfig(**kwargs)


class TestBuildRequestSuccess:
    """Happy-path template rendering."""

    def test_fills_single_variable(self) -> None:
        builder = RequestBuilder(_make_config())
        prompt_version = _make_prompt_version("Answer this: {question}")

        request = builder.build_request(prompt_version, {"question": "What is 2+2?"})

        assert request.user_prompt == "Answer this: What is 2+2?"

    def test_fills_multiple_variables(self) -> None:
        builder = RequestBuilder(_make_config())
        prompt_version = _make_prompt_version("Context: {context}\nQuestion: {question}")
        row = {
            "context": "Paris is the capital of France.",
            "question": "What is the capital of France?",
        }

        request = builder.build_request(prompt_version, row)

        assert request.user_prompt == (
            "Context: Paris is the capital of France.\n"
            "Question: What is the capital of France?"
        )

    def test_ignores_extra_row_columns(self) -> None:
        builder = RequestBuilder(_make_config())
        prompt_version = _make_prompt_version("Q: {question}")
        row = {"question": "Why?", "domain": "science", "difficulty": "easy"}

        request = builder.build_request(prompt_version, row)

        assert request.user_prompt == "Q: Why?"

    def test_preserves_escaped_literal_braces(self) -> None:
        builder = RequestBuilder(_make_config())
        prompt_version = _make_prompt_version("Use format {{like this}} for {question}")

        request = builder.build_request(prompt_version, {"question": "this"})

        assert request.user_prompt == "Use format {like this} for this"

    def test_template_with_no_variables_passes_through(self) -> None:
        builder = RequestBuilder(_make_config())
        prompt_version = _make_prompt_version("A static prompt with no placeholders.")

        request = builder.build_request(prompt_version, {"question": "unused"})

        assert request.user_prompt == "A static prompt with no placeholders."

    def test_applies_config_temperature_and_max_tokens(self) -> None:
        builder = RequestBuilder(_make_config(temperature=1.3, max_tokens=999))
        prompt_version = _make_prompt_version("Q: {question}")

        request = builder.build_request(prompt_version, {"question": "hi"})

        assert request.temperature == 1.3
        assert request.max_tokens == 999

    def test_system_prompt_is_unset(self) -> None:
        builder = RequestBuilder(_make_config())
        prompt_version = _make_prompt_version("Q: {question}")

        request = builder.build_request(prompt_version, {"question": "hi"})

        assert request.system_prompt is None


class TestBuildRequestMissingVariable:
    """A template variable absent from the row must raise a clear error."""

    def test_missing_variable_raises_with_variable_name(self) -> None:
        builder = RequestBuilder(_make_config())
        prompt_version = _make_prompt_version("Context: {context}\nQuestion: {question}")

        with pytest.raises(MissingTemplateVariableError) as exc_info:
            builder.build_request(prompt_version, {"question": "Only question, no context"})

        assert exc_info.value.variable_name == "context"
        assert exc_info.value.available_columns == ["question"]

    def test_error_message_lists_available_columns(self) -> None:
        builder = RequestBuilder(_make_config())
        prompt_version = _make_prompt_version("{missing_field}")
        row = {"question": "q", "ground_truth": "gt"}

        with pytest.raises(MissingTemplateVariableError) as exc_info:
            builder.build_request(prompt_version, row)

        message = str(exc_info.value)
        assert "missing_field" in message
        assert "question" in message
        assert "ground_truth" in message

    def test_does_not_raise_for_partially_present_variables_when_all_present(self) -> None:
        # Sanity check: a template needing two variables succeeds when
        # both -- not just one -- are present.
        builder = RequestBuilder(_make_config())
        prompt_version = _make_prompt_version("{a} and {b}")

        request = builder.build_request(prompt_version, {"a": "1", "b": "2"})

        assert request.user_prompt == "1 and 2"