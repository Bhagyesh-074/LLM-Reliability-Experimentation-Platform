"""Ollama provider implementation (local inference, no API key required)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from providers.base import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ModelNotFoundError,
    TransientProviderError,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:  # pragma: no cover - exercised only when package is missing
    ollama = None  # type: ignore[assignment]


class OllamaProvider(BaseLLMProvider):
    """Provider for locally running Ollama models.

    Requires the ``ollama`` Python package and a running Ollama daemon.
    No API key is required since Ollama runs entirely locally.
    """

    name = "ollama"

    def __init__(self, model: str, host: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize the Ollama provider.

        Args:
            model: The local model tag (e.g. "llama3.1", "mistral").
            host: Optional Ollama daemon host URL. When omitted, the
                client falls back to the SDK default
                (typically http://localhost:11434).

        Raises:
            ImportError: If the ``ollama`` package is not installed.
        """
        super().__init__(model, host=host, **kwargs)
        if ollama is None:
            raise ImportError(
                "The 'ollama' package is required to use OllamaProvider. "
                "Install it with `pip install ollama`."
            )
        self._client = ollama.Client(host=host) if host else ollama

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the local Ollama daemon.

        Args:
            request: The normalized request.

        Returns:
            A normalized ``LLMResponse``.

        Raises:
            ModelNotFoundError: If the model has not been pulled locally.
            TransientProviderError: For connection/timeout failures.
        """
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        @retry_with_backoff((TransientProviderError,), max_attempts=3)
        def _call() -> Any:
            try:
                return self._client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                )
            except (ModelNotFoundError, TransientProviderError):
                raise
            except Exception as exc:  # noqa: BLE001 - normalize any SDK error
                message = str(exc).lower()
                if "not found" in message or "no such model" in message:
                    raise ModelNotFoundError(
                        f"Model '{self.model}' not found locally. Pull it with "
                        f"`ollama pull {self.model}`.",
                        details={"model": self.model},
                    ) from exc
                raise TransientProviderError(str(exc)) from exc

        result, elapsed_ms = self._time_call(_call)

        message = self._field(result, "message", {})
        text = self._field(message, "content", "") or ""

        prompt_tokens = self._field(result, "prompt_eval_count", 0) or 0
        completion_tokens = self._field(result, "eval_count", 0) or 0
        done = self._field(result, "done", True)

        return LLMResponse(
            text=text,
            latency_ms=elapsed_ms,
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="stop" if done else "length",
            raw_metadata=self._to_raw_metadata(result),
        )

    @staticmethod
    def _field(obj: Any, key: str, default: Any = None) -> Any:
        """Read ``key`` off ``obj``, whether it's a plain dict or a typed SDK object.

        Newer versions of the ``ollama`` package return typed response
        objects (e.g. pydantic models) rather than plain dicts, so a bare
        ``.get()`` call silently short-circuited to ``default`` for every
        field on those objects. This checks both shapes explicitly instead
        of assuming one.

        Args:
            obj: A dict-like or attribute-bearing object (or ``None``).
            key: The field name to read.
            default: Value to return if ``obj`` is ``None`` or lacks ``key``.

        Returns:
            The field's value, or ``default``.
        """
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _to_raw_metadata(result: Any) -> dict[str, Any]:
        """Normalize the raw Ollama result into a plain dict for ``raw_metadata``.

        Args:
            result: The raw object returned by the Ollama client, either a
                dict (older client versions) or a typed SDK object (newer
                versions, which usually expose ``model_dump()``).

        Returns:
            A plain ``dict`` snapshot of the result, falling back to a
            string representation if no structured form is available.
        """
        if isinstance(result, dict):
            return dict(result)
        model_dump = getattr(result, "model_dump", None)
        if callable(model_dump):
            try:
                return model_dump()
            except Exception:  # noqa: BLE001 - metadata is best-effort only
                pass
        return {"raw": str(result)}