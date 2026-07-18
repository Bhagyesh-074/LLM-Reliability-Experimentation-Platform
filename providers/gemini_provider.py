"""Google Gemini provider implementation backed by google-generativeai."""

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
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ImportError:  # pragma: no cover - exercised only when package is missing
    genai = None  # type: ignore[assignment]
    google_exceptions = None  # type: ignore[assignment]


class GeminiProvider(BaseLLMProvider):
    """Provider backed by the Google Generative AI (Gemini) API."""

    name = "gemini"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize the Gemini provider.

        Args:
            model: The Gemini model identifier (e.g. "gemini-1.5-pro").
            api_key: Explicit API key override. When omitted, the key is
                read from the ``GOOGLE_API_KEY`` environment variable.
            **kwargs: Forwarded to ``BaseLLMProvider``.

        Raises:
            ImportError: If the ``google-generativeai`` package is not installed.
            ProviderError: If no API key can be resolved.
        """
        super().__init__(model, **kwargs)
        if genai is None:
            raise ImportError(
                "The 'google-generativeai' package is required to use GeminiProvider. "
                "Install it with `pip install google-generativeai`."
            )
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ProviderError(
                "Missing Google API key. Set the GOOGLE_API_KEY environment variable."
            )
        genai.configure(api_key=resolved_key)
        self._default_model = genai.GenerativeModel(model_name=self.model)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the Gemini API.

        Args:
            request: The normalized request.

        Returns:
            A normalized ``LLMResponse``.

        Raises:
            ModelNotFoundError: If the model does not exist or is inaccessible.
            RateLimitedError: If Gemini returns a resource-exhausted error.
            TransientProviderError: For retryable connection/server errors.
            ProviderError: For any other non-retryable API error.
        """
        # Gemini binds the system instruction at model-construction time, so a
        # fresh model handle is built when a system prompt is supplied.
        model = (
            genai.GenerativeModel(model_name=self.model, system_instruction=request.system_prompt)
            if request.system_prompt
            else self._default_model
        )

        generation_config = genai.types.GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )

        @retry_with_backoff((TransientProviderError, RateLimitedError), max_attempts=3)
        def _call() -> Any:
            try:
                return model.generate_content(
                    request.user_prompt,
                    generation_config=generation_config,
                )
            except google_exceptions.NotFound as exc:
                raise ModelNotFoundError(
                    f"Model '{self.model}' not found or inaccessible.",
                    details={"model": self.model},
                ) from exc
            except google_exceptions.ResourceExhausted as exc:
                raise RateLimitedError(str(exc)) from exc
            except (google_exceptions.DeadlineExceeded, google_exceptions.ServiceUnavailable) as exc:
                raise TransientProviderError(str(exc)) from exc
            except google_exceptions.GoogleAPIError as exc:
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

        raw_metadata = result.to_dict() if hasattr(result, "to_dict") else {"raw": str(result)}

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