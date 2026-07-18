"""OpenAI provider implementation backed by the Chat Completions API."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from providers.base import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ModelNotFoundError,
    ProviderError,
    RateLimitedError,
    TransientProviderError,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when package is missing
    openai = None  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]


class OpenAIProvider(BaseLLMProvider):
    """Provider backed by the OpenAI Chat Completions API."""

    name = "openai"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize the OpenAI provider.

        Args:
            model: The OpenAI model identifier (e.g. "gpt-4o").
            api_key: Explicit API key override. When omitted, the key is
                read from the ``OPENAI_API_KEY`` environment variable.
            **kwargs: Forwarded to ``BaseLLMProvider``.

        Raises:
            ImportError: If the ``openai`` package is not installed.
            ProviderError: If no API key can be resolved.
        """
        super().__init__(model, **kwargs)
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required to use OpenAIProvider. "
                "Install it with `pip install openai`."
            )
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ProviderError(
                "Missing OpenAI API key. Set the OPENAI_API_KEY environment variable."
            )
        self._client = OpenAI(api_key=resolved_key)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the OpenAI Chat Completions API.

        Args:
            request: The normalized request.

        Returns:
            A normalized ``LLMResponse``.

        Raises:
            ModelNotFoundError: If the model does not exist or is inaccessible.
            RateLimitedError: If OpenAI returns a rate-limit error.
            TransientProviderError: For retryable connection/server errors.
            ProviderError: For any other non-retryable API error.
        """
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        @retry_with_backoff((TransientProviderError, RateLimitedError), max_attempts=3)
        def _call() -> Any:
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            except openai.NotFoundError as exc:
                raise ModelNotFoundError(
                    f"Model '{self.model}' not found or inaccessible.",
                    details={"model": self.model},
                ) from exc
            except openai.RateLimitError as exc:
                raise RateLimitedError(str(exc)) from exc
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as exc:
                raise TransientProviderError(str(exc)) from exc
            except openai.APIStatusError as exc:
                raise ProviderError(
                    str(exc), details={"status_code": getattr(exc, "status_code", None)}
                ) from exc

        result, elapsed_ms = self._time_call(_call)

        choice = result.choices[0]
        usage = result.usage
        token_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

        raw_metadata = result.model_dump() if hasattr(result, "model_dump") else {"raw": str(result)}

        return LLMResponse(
            text=choice.message.content or "",
            latency_ms=elapsed_ms,
            token_usage=token_usage,
            finish_reason=choice.finish_reason or "unknown",
            raw_metadata=raw_metadata,
        )