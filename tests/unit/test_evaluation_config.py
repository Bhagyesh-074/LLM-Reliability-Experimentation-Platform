"""Unit tests for core.evaluation.config.EvaluationConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.evaluation.config import EvaluationConfig


def _base_kwargs(**overrides: object) -> dict:
    """Build a valid EvaluationConfig kwargs dict, with optional field overrides."""
    kwargs = dict(
        provider_name="openai",
        model_name="gpt-4o",
        prompt_version_id="pv-123",
        dataset_version_id="dv-456",
        temperature=0.7,
        max_tokens=512,
    )
    kwargs.update(overrides)
    return kwargs


class TestEvaluationConfigConstruction:
    """Happy-path construction and field wiring."""

    def test_valid_config_constructs(self) -> None:
        config = EvaluationConfig(**_base_kwargs())
        assert config.provider_name == "openai"
        assert config.model_name == "gpt-4o"
        assert config.prompt_version_id == "pv-123"
        assert config.dataset_version_id == "dv-456"
        assert config.temperature == 0.7
        assert config.max_tokens == 512

    @pytest.mark.parametrize(
        "field",
        [
            "provider_name",
            "model_name",
            "prompt_version_id",
            "dataset_version_id",
            "temperature",
            "max_tokens",
        ],
    )
    def test_missing_required_field_raises(self, field: str) -> None:
        kwargs = _base_kwargs()
        del kwargs[field]
        with pytest.raises(ValidationError):
            EvaluationConfig(**kwargs)

    @pytest.mark.parametrize(
        "empty_field",
        ["provider_name", "model_name", "prompt_version_id", "dataset_version_id"],
    )
    def test_empty_string_field_raises(self, empty_field: str) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(**_base_kwargs(**{empty_field: ""}))


class TestTemperatureBounds:
    """temperature must be constrained to [0.0, 2.0]."""

    @pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
    def test_boundary_values_accepted(self, temperature: float) -> None:
        config = EvaluationConfig(**_base_kwargs(temperature=temperature))
        assert config.temperature == temperature

    @pytest.mark.parametrize("temperature", [-0.01, 2.01, -5.0, 10.0])
    def test_out_of_range_raises(self, temperature: float) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(**_base_kwargs(temperature=temperature))


class TestMaxTokens:
    """max_tokens must be a positive integer."""

    def test_minimum_positive_value_accepted(self) -> None:
        config = EvaluationConfig(**_base_kwargs(max_tokens=1))
        assert config.max_tokens == 1

    @pytest.mark.parametrize("max_tokens", [0, -1, -100])
    def test_non_positive_raises(self, max_tokens: int) -> None:
        with pytest.raises(ValidationError):
            EvaluationConfig(**_base_kwargs(max_tokens=max_tokens))


class TestImmutability:
    """EvaluationConfig is frozen: a run's config should not mutate mid-flight."""

    def test_assigning_a_field_raises(self) -> None:
        config = EvaluationConfig(**_base_kwargs())
        with pytest.raises(ValidationError):
            config.temperature = 1.5  # type: ignore[misc]