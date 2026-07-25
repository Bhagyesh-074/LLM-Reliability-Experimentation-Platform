"""Provider Configuration page.

Lets the user inspect the four supported LLM providers, see their
connection status, enter (masked) API keys, and pick a default model per
provider. Provider/status/model data comes from ``ProviderRegistry``
(live env-var checks and, for Ollama, live daemon detection); no network
calls are made from this page itself.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st

from providers.registry import ProviderRegistry

# Static catalog of selectable models per provider. The registry only
# tells us whether a provider is configured (and, for Ollama, which
# models are actually installed) -- it doesn't expose a "list all
# models" call for the hosted providers, so those options are curated
# here for the UI's model picker.
_MODEL_CATALOG: Dict[str, List[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini"],
    "anthropic": [
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
    ],
    "gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
}

# Fallback options shown for Ollama when the local daemon is unreachable
# or has no models pulled, so the selectbox never receives an empty list.
_FALLBACK_OLLAMA_MODELS: List[str] = ["llama3.1", "llama3.2", "mistral"]

_DISPLAY_NAMES: Dict[str, str] = {
    "ollama": "Ollama",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
}


@dataclass
class Provider:
    """View-model for a single provider card on the Providers page."""

    name: str
    status: str
    masked_api_key: str
    models: List[str]
    default_model: str


def _mask_key(env_var: Optional[str]) -> str:
    """Build a masked display string for an API key held in the environment.

    Args:
        env_var: Name of the environment variable holding the key, or
            ``None`` for providers (like Ollama) that don't need one.

    Returns:
        A masked string such as ``"sk-...ab12"`` when a key is present,
        or a human-readable placeholder when it is absent/not required.
    """
    if env_var is None:
        return "Not required"
    value = os.environ.get(env_var, "")
    if not value:
        return "Not configured"
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:3]}{'•' * 6}{value[-4:]}"


def get_providers() -> List[Provider]:
    """Build the provider list for the page from live ``ProviderRegistry`` data.

    Connection status rules:
        * Ollama: "connected" when at least one local model is detected
          via ``ProviderRegistry.detect_ollama_models()``; otherwise
          "disconnected" (including when the daemon can't be reached).
        * OpenAI / Anthropic / Gemini: "connected" when the provider's
          required API key environment variable is set (as reported by
          ``ProviderRegistry.list_providers()``'s ``configured`` flag),
          otherwise "disconnected".

    Any failure talking to the registry (e.g. Ollama daemon down) is
    caught and degrades to a disconnected/empty result rather than
    raising, so the page always renders.

    Returns:
        A list of :class:`Provider` view-models, one per known provider.
    """
    try:
        registry_providers = ProviderRegistry.list_providers()
    except Exception:  # noqa: BLE001 - degrade gracefully, page must still render
        registry_providers = []

    providers: List[Provider] = []
    for entry in registry_providers:
        name = entry["name"]
        display_name = _DISPLAY_NAMES.get(name, name.title())

        if name == "ollama":
            try:
                models = ProviderRegistry.detect_ollama_models()
            except Exception:  # noqa: BLE001 - daemon may be down
                models = []
            status = "connected" if models else "disconnected"
            if not models:
                models = _FALLBACK_OLLAMA_MODELS
        else:
            models = _MODEL_CATALOG.get(name, [])
            status = "connected" if entry.get("configured") else "disconnected"

        if not models:
            models = ["(no models available)"]

        providers.append(
            Provider(
                name=display_name,
                status=status,
                masked_api_key=_mask_key(entry.get("env_var")),
                models=models,
                default_model=models[0],
            )
        )
    return providers


def render_status_badge(status: str) -> str:
    """Return an HTML badge string for a provider's connection status."""

    if status == "connected":
        return (
            "<span style='background-color:#123524;color:#4ade80;padding:3px 10px;"
            "border-radius:12px;font-size:12px;font-weight:600;'>● Connected</span>"
        )
    return (
        "<span style='background-color:#3a1f1f;color:#f87171;padding:3px 10px;"
        "border-radius:12px;font-size:12px;font-weight:600;'>● Disconnected</span>"
    )


def render_provider_card(provider: Provider) -> None:
    """Render a single provider as a card with key input and model picker."""

    with st.container(border=True):
        header_col, badge_col = st.columns([3, 1])
        with header_col:
            st.subheader(provider.name)
        with badge_col:
            st.markdown(render_status_badge(provider.status), unsafe_allow_html=True)

        col_key, col_model = st.columns(2)
        with col_key:
            st.text_input(
                "API Key",
                value=provider.masked_api_key,
                type="password",
                key=f"api_key_{provider.name}",
                disabled=provider.name == "Ollama",
                help="Keys are masked and never displayed in full.",
            )
        with col_model:
            st.selectbox(
                "Default model",
                options=provider.models,
                index=provider.models.index(provider.default_model),
                key=f"default_model_{provider.name}",
            )

        with st.expander(f"Available models ({len(provider.models)})"):
            for model in provider.models:
                st.markdown(f"- `{model}`")

        action_col1, action_col2, _ = st.columns([1, 1, 3])
        with action_col1:
            st.button(
                "Test connection",
                key=f"test_{provider.name}",
                use_container_width=True,
            )
        with action_col2:
            st.button(
                "Save",
                key=f"save_{provider.name}",
                type="primary",
                use_container_width=True,
            )


def main() -> None:
    st.title("Provider Configuration")
    st.caption("Manage connections for every LLM provider supported by the platform.")

    providers = get_providers()
    connected_count = sum(1 for p in providers if p.status == "connected")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total providers", len(providers))
    m2.metric("Connected", connected_count)
    m3.metric("Disconnected", len(providers) - connected_count)

    st.divider()

    left, right = st.columns(2, gap="large")
    columns = [left, right]
    for i, provider in enumerate(providers):
        with columns[i % 2]:
            render_provider_card(provider)
            st.write("")


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()