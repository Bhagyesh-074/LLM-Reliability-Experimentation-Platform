"""Anthropic provider implementation backed by the Messages API."""

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
    import anthropic
except ImportError:  # pragma: no cover - exercised only when package is missing
    anthropic = None  # type: ignore[assignment]


class AnthropicProvider(BaseLLMProvider):
    """Provider backed by the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize the Anthropic provider.

        Args:
            model: The Anthropic model identifier (e.g. "claude-sonnet-4-6").
            api_key: Explicit API key override. When omitted, the key is
                read from the ``ANTHROPIC_API_KEY`` environment variable.
            **kwargs: Forwarded to ``BaseLLMProvider``.

        Raises:
            ImportError: If the ``anthropic`` package is not installed.
            ProviderError: If no API key can be resolved.
        """
        super().__init__(model, **kwargs)
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required to use AnthropicProvider. "
                "Install it with `pip install anthropic`."
            )
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ProviderError(
                "Missing Anthropic API key. Set the ANTHROPIC_API_KEY environment variable."
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the Anthropic Messages API.

        Args:
            request: The normalized request.

        Returns:
            A normalized ``LLMResponse``.

        Raises:
            ModelNotFoundError: If the model does not exist or is inaccessible.
            RateLimitedError: If Anthropic returns a rate-limit error.
            TransientProviderError: For retryable connection/server errors.
            ProviderError: For any other non-retryable API error.
        """

        @retry_with_backoff((TransientProviderError, RateLimitedError), max_attempts=3)
        def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": request.user_prompt}],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            if request.system_prompt:
                kwargs["system"] = request.system_prompt
            try:
                return self._client.messages.create(**kwargs)
            except anthropic.NotFoundError as exc:
                raise ModelNotFoundError(
                    f"Model '{self.model}' not found or inaccessible.",
                    details={"model": self.model},
                ) from exc
            except anthropic.RateLimitError as exc:
                raise RateLimitedError(str(exc)) from exc
            except (
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
            ) as exc:
                raise TransientProviderError(str(exc)) from exc
            except anthropic.APIStatusError as exc:
                raise ProviderError(
                    str(exc), details={"status_code": getattr(exc, "status_code", None)}
                ) from exc

        result, elapsed_ms = self._time_call(_call)

        text = "".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )
        usage = result.usage
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0

        raw_metadata = result.model_dump() if hasattr(result, "model_dump") else {"raw": str(result)}

        return LLMResponse(
            text=text,
            latency_ms=elapsed_ms,
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason=result.stop_reason or "unknown",
            raw_metadata=raw_metadata,
        )