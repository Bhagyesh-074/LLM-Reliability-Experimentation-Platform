"""Google Gemini provider implementation backed by google-genai."""

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
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised only when package is missing
    genai = None  # type: ignore[assignment]
    genai_errors = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

# HTTP status codes that should be treated as retryable/transient.
_TRANSIENT_STATUS_CODES = {500, 502, 503, 504}


class GeminiProvider(BaseLLMProvider):
    """Provider backed by the Google Gen AI (Gemini) API."""

    name = "gemini"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize the Gemini provider.

        Args:
            model: The Gemini model identifier (e.g. "gemini-2.5-pro").
            api_key: Explicit API key override. When omitted, the key is
                read from the ``GOOGLE_API_KEY`` environment variable.
            **kwargs: Forwarded to ``BaseLLMProvider``.

        Raises:
            ImportError: If the ``google-genai`` package is not installed.
            ProviderError: If no API key can be resolved.
        """
        super().__init__(model, **kwargs)
        if genai is None:
            raise ImportError(
                "The 'google-genai' package is required to use GeminiProvider. "
                "Install it with `pip install google-genai`."
            )
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ProviderError(
                "Missing Google API key. Set the GOOGLE_API_KEY environment variable."
            )
        self._client = genai.Client(api_key=resolved_key)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the Gemini API.

        Args:
            request: The normalized request.

        Returns:
            A normalized ``LLMResponse``.

        Raises:
            ModelNotFoundError: If the model does not exist or is inaccessible.
            RateLimitedError: If Gemini returns a resource-exhausted (429) error.
            TransientProviderError: For retryable connection/server errors.
            ProviderError: For any other non-retryable API error.
        """
        config = genai_types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            system_instruction=request.system_prompt or None,
        )

        @retry_with_backoff((TransientProviderError, RateLimitedError), max_attempts=3)
        def _call() -> Any:
            try:
                return self._client.models.generate_content(
                    model=self.model,
                    contents=request.user_prompt,
                    config=config,
                )
            except genai_errors.APIError as exc:
                status_code = getattr(exc, "code", None)
                if status_code == 404:
                    raise ModelNotFoundError(
                        f"Model '{self.model}' not found or inaccessible.",
                        details={"model": self.model},
                    ) from exc
                if status_code == 429:
                    raise RateLimitedError(str(exc)) from exc
                if status_code in _TRANSIENT_STATUS_CODES:
                    raise TransientProviderError(str(exc)) from exc
                raise ProviderError(str(exc)) from exc

        result, elapsed_ms = self._time_call(_call)

        text = result.text if hasattr(result, "text") else ""
        usage_meta = getattr(result, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        finish_reason = "unknown"
        candidates = getattr(result, "candidates", None)
        if candidates:
            finish_reason = str(candidates[0].finish_reason)

        raw_metadata = (
            result.model_dump() if hasattr(result, "model_dump") else {"raw": str(result)}
        )

        return LLMResponse(
            text=text,
            latency_ms=elapsed_ms,
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason=finish_reason,
            raw_metadata=raw_metadata,
        )