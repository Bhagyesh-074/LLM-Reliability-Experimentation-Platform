"""Pydantic schemas for the Prompt Registry service.

These are the DTOs that cross the boundary between ``PromptService`` and
its callers (the Streamlit UI, tests, future API routes). They are
intentionally decoupled from the SQLAlchemy models in ``database.models``
so the ORM layer can change shape without breaking callers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

PromptStatus = Literal["draft", "active", "deprecated"]


class PromptCreate(BaseModel):
    """Input payload for creating a new prompt and its first version."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    author: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    status: PromptStatus = "draft"


class PromptVersionCreate(BaseModel):
    """Input payload for adding a new immutable version to a prompt."""

    content: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)


class PromptVersionResponse(BaseModel):
    """A single immutable version of a prompt."""

    model_config = ConfigDict(from_attributes=True)

    version_id: str
    prompt_id: str
    version: int
    content: str
    content_hash: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime


class PromptResponse(BaseModel):
    """A prompt, optionally enriched with latest-version info and full history."""

    model_config = ConfigDict(from_attributes=True)

    prompt_id: str
    name: str
    description: str
    author: str
    status: PromptStatus
    created_at: datetime

    current_version: int = Field(
        default=0, description="Highest version number that exists for this prompt."
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags carried on the latest prompt version."
    )
    versions: Optional[list[PromptVersionResponse]] = Field(
        default=None, description="Full version history, populated by get_prompt()."
    )


class PromptValidationResult(BaseModel):
    """Result of validating a prompt template's ``{variable}`` syntax."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    variables: list[str] = Field(
        default_factory=list, description="Distinct variable names found in the template."
    )