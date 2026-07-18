"""Provider abstraction layer for the LLM Reliability & Experimentation Platform.

This module defines the common contract that every LLM provider
implementation must honor: a single ``generate`` method that accepts an
``LLMRequest`` and returns a normalized ``LLMResponse``, regardless of the
underlying vendor SDK.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ProviderError(Exception):
    """Base exception for all provider-related failures.

    Attributes:
        code: A stable error code, matching the error codes documented in
            API_SPEC.md (e.g. "PROVIDER_ERROR", "MODEL_NOT_FOUND").
        message: Human readable error message.
        details: Optional extra context about the failure.
    """

    code = "PROVIDER_ERROR"

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not available for a provider."""

    code = "MODEL_NOT_FOUND"


class RateLimitedError(ProviderError):
    """Raised when the upstream provider rate-limits the request."""

    code = "RATE_LIMITED"


class TransientProviderError(ProviderError):
    """Raised for retryable, transient provider failures (timeouts, 5xx, etc.)."""

    code = "PROVIDER_ERROR"


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


class LLMRequest(BaseModel):
    """Normalized request sent to any provider.

    Attributes:
        system_prompt: Optional system-level instruction/persona.
        user_prompt: The user-facing content to send to the model.
        temperature: Sampling temperature, constrained to [0, 2].
        max_tokens: Maximum number of tokens to generate. Must be positive.
    """

    system_prompt: Optional[str] = None
    user_prompt: str
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)

    @field_validator("user_prompt")
    @classmethod
    def _non_empty_prompt(cls, value: str) -> str:
        """Reject empty or whitespace-only prompts."""
        if not value or not value.strip():
            raise ValueError("user_prompt must not be empty")
        return value


class LLMResponse(BaseModel):
    """Normalized response returned by any provider.

    Attributes:
        text: The generated text content.
        latency_ms: Wall-clock latency of the call, in milliseconds.
        token_usage: Dict with keys such as ``prompt_tokens``,
            ``completion_tokens``, ``total_tokens``. Providers that cannot
            report a given field simply omit it or report 0.
        finish_reason: Normalized finish reason (e.g. "stop", "length").
        raw_metadata: The untouched, provider-specific response payload,
            retained for debugging and auditing purposes.
    """

    text: str
    latency_ms: float
    token_usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str = "unknown"
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #


def retry_with_backoff(
    exceptions: Tuple[Type[Exception], ...],
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.1,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator implementing manual exponential backoff with jitter.

    Retries the wrapped callable whenever it raises one of ``exceptions``.
    The last exception is re-raised once ``max_attempts`` is exhausted.

    Args:
        exceptions: Exception types considered transient/retryable.
        max_attempts: Total number of attempts, including the first.
        base_delay: Initial delay, in seconds, before the first retry.
        max_delay: Upper bound applied to the computed backoff delay.
        jitter: Fractional random jitter applied to each delay (0.1 == ±10%).

    Returns:
        A decorator that wraps a callable with retry behavior.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            delay = base_delay
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= max_attempts:
                        logger.error(
                            "%s failed after %d attempt(s): %s",
                            func.__qualname__,
                            attempt,
                            exc,
                        )
                        raise
                    sleep_for = min(delay, max_delay) * (1 + random.uniform(-jitter, jitter))
                    sleep_for = max(sleep_for, 0.0)
                    logger.warning(
                        "%s attempt %d/%d failed (%s); retrying in %.2fs",
                        func.__qualname__,
                        attempt,
                        max_attempts,
                        exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    delay *= 2

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Base provider
# --------------------------------------------------------------------------- #


class BaseLLMProvider(ABC):
    """Abstract base class every LLM provider implementation must extend.

    Subclasses must accept identical ``LLMRequest`` objects and return
    identical ``LLMResponse`` objects, normalizing away any provider-specific
    metadata into ``raw_metadata``.
    """

    #: Name used for registry/factory lookups (e.g. "openai", "anthropic").
    name: str = "base"

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Initialize the provider.

        Args:
            model: The model identifier to use for generation.
            **kwargs: Provider-specific configuration (e.g. api_key, host).

        Raises:
            ValueError: If ``model`` is empty or whitespace-only.
        """
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")
        self.model = model
        self.config = kwargs

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for the given request.

        Args:
            request: The normalized request to send to the provider.

        Returns:
            A normalized ``LLMResponse``.

        Raises:
            ProviderError: On any unrecoverable provider failure.
        """
        raise NotImplementedError

    @staticmethod
    def _time_call(func: Callable[[], Any]) -> Tuple[Any, float]:
        """Execute ``func``, returning its result and elapsed time in ms.

        Args:
            func: A zero-argument callable to invoke and time.

        Returns:
            A tuple of (result, elapsed_milliseconds).
        """
        start = time.perf_counter()
        result = func()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(model={self.model!r})"