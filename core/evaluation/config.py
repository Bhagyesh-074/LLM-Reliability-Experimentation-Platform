"""core/evaluation/config.py

Configuration for a single evaluation run.

``EvaluationConfig`` is the single input the Evaluation Orchestrator
needs to execute a run: which provider/model to call, which immutable
prompt version and dataset version to use, and the generation
parameters to apply uniformly across every row in the dataset.

This is intentionally a thin, validated DTO -- it carries no behavior
and performs no I/O. Resolving the referenced prompt version, dataset
version, and provider instance is the orchestrator's job (see
``core/evaluation/orchestrator.py``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvaluationConfig(BaseModel):
    """Declarative configuration for a single evaluation run.

    Attributes:
        provider_name: Provider key understood by ``ProviderFactory``
            (e.g. "openai", "anthropic", "gemini", "ollama").
        model_name: Model identifier passed through to the provider
            (e.g. "gpt-4o", "claude-sonnet-4-6").
        prompt_version_id: Primary key of the immutable
            ``PromptVersion`` row to evaluate
            (``database.models.PromptVersion.version_id``).
        dataset_version_id: Primary key of the immutable
            ``DatasetVersion`` row to evaluate against
            (``database.models.DatasetVersion.dataset_version_id``).
        temperature: Sampling temperature applied to every generation
            call in the run. Constrained to the range every supported
            provider accepts, [0.0, 2.0].
        max_tokens: Maximum number of tokens to generate per row. Must
            be a positive integer.
    """

    model_config = ConfigDict(frozen=True)

    provider_name: str = Field(
        ..., min_length=1, description="Provider key understood by ProviderFactory, e.g. 'openai'."
    )
    model_name: str = Field(
        ..., min_length=1, description="Model identifier passed through to the provider."
    )
    prompt_version_id: str = Field(
        ..., min_length=1, description="PromptVersion.version_id of the prompt to evaluate."
    )
    dataset_version_id: str = Field(
        ...,
        min_length=1,
        description="DatasetVersion.dataset_version_id of the dataset to evaluate against.",
    )
    temperature: float = Field(
        ..., ge=0.0, le=2.0, description="Sampling temperature applied to every row, [0.0, 2.0]."
    )
    max_tokens: int = Field(..., gt=0, description="Maximum tokens to generate per row. Must be positive.")