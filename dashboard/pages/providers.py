"""Provider Configuration page.

Lets the user inspect the four supported LLM providers, see their
connection status, enter (masked) API keys, and pick a default model per
provider. All data is mock; no network calls are made.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.mock.data_new import Provider, get_providers




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
            st.button("Test connection", key=f"test_{provider.name}", use_container_width=True)
        with action_col2:
            st.button("Save", key=f"save_{provider.name}", type="primary", use_container_width=True)


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
