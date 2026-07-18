"""Provider registry: lists available providers and detects local Ollama models."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import ollama
except ImportError:  # pragma: no cover - exercised only when package is missing
    ollama = None  # type: ignore[assignment]

# Static provider metadata: name -> (requires_api_key, env_var).
_STATIC_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "ollama": {"requires_api_key": False, "env_var": None},
    "openai": {"requires_api_key": True, "env_var": "OPENAI_API_KEY"},
    "anthropic": {"requires_api_key": True, "env_var": "ANTHROPIC_API_KEY"},
    "gemini": {"requires_api_key": True, "env_var": "GEMINI_API_KEY"},
}


class ProviderRegistry:
    """Reports which providers are known/configured at runtime.

    Backs the ``GET /api/v1/providers`` and ``GET /api/v1/providers/models``
    endpoints described in API_SPEC.md.
    """

    @staticmethod
    def list_providers() -> List[Dict[str, Any]]:
        """Return metadata for every provider the platform knows about.

        Returns:
            A list of dicts, one per provider, each with keys:
            ``name``, ``requires_api_key``, ``env_var``, and ``configured``
            (True when the required environment variable is set, or always
            True for providers that need no API key).
        """
        providers: List[Dict[str, Any]] = []
        for name, meta in _STATIC_PROVIDERS.items():
            env_var = meta["env_var"]
            configured = True if not meta["requires_api_key"] else bool(os.environ.get(env_var, ""))
            providers.append(
                {
                    "name": name,
                    "requires_api_key": meta["requires_api_key"],
                    "env_var": env_var,
                    "configured": configured,
                }
            )
        return providers

    @staticmethod
    def detect_ollama_models() -> List[str]:
        """Detect locally installed Ollama models via ``ollama.list()``.

        Returns:
            A list of model tags currently pulled/available on the local
            Ollama daemon. Returns an empty list if the ``ollama`` package
            is not installed or the daemon cannot be reached.
        """
        if ollama is None:
            logger.warning("The 'ollama' package is not installed; cannot detect models.")
            return []
        try:
            response = ollama.list()
        except Exception as exc:  # noqa: BLE001 - daemon may be down, degrade gracefully
            logger.warning("Failed to reach local Ollama daemon: %s", exc)
            return []

        if isinstance(response, dict):
            raw_models = response.get("models", [])
        else:  # pragma: no cover - defensive, newer SDKs may return objects
            raw_models = getattr(response, "models", [])

        models: List[str] = []
        for item in raw_models:
            if isinstance(item, dict):
                tag = item.get("name") or item.get("model")
            else:
                tag = getattr(item, "name", None) or getattr(item, "model", None)
            if tag:
                models.append(tag)
        return models