"""Factory for constructing provider instances from configuration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from providers.anthropic_provider import AnthropicProvider
from providers.base import BaseLLMProvider, ProviderError
from providers.gemini_provider import GeminiProvider
from providers.ollama_provider import OllamaProvider
from providers.openai_provider import OpenAIProvider

_PROVIDER_BUILDERS: Dict[str, Callable[..., BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


class ProviderFactory:
    """Creates provider instances from a configuration string and model name.

    This is the single entry point the Evaluation Orchestrator uses to turn
    a config-driven ``provider`` string (see SYSTEM_DESIGN.md's Provider
    Router) into a concrete, ready-to-use ``BaseLLMProvider`` instance.
    """

    @staticmethod
    def create(provider: str, model: str, **kwargs: Any) -> BaseLLMProvider:
        """Instantiate the provider matching the given name.

        Args:
            provider: One of "ollama", "openai", "anthropic", "gemini"
                (case-insensitive, surrounding whitespace tolerated).
            model: The model identifier to pass to the provider.
            **kwargs: Additional provider-specific keyword arguments
                (e.g. ``api_key``, ``host``).

        Returns:
            An initialized ``BaseLLMProvider`` instance.

        Raises:
            ProviderError: If ``provider`` does not match a known provider.
        """
        key = provider.strip().lower()
        builder = _PROVIDER_BUILDERS.get(key)
        if builder is None:
            raise ProviderError(
                f"Unknown provider '{provider}'. Supported providers: "
                f"{sorted(_PROVIDER_BUILDERS)}.",
                details={"provider": provider},
            )
        return builder(model=model, **kwargs)

    @staticmethod
    def supported_providers() -> List[str]:
        """Return the sorted list of provider names the factory can build."""
        return sorted(_PROVIDER_BUILDERS)