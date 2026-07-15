"""Prompt Registry page.

Displays all versioned prompts, supports creating a new prompt via a form
(mock, not persisted), and lets the user drill into the full version
history of any prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.mock.data_new import Prompt, get_prompts




def prompts_to_dataframe(prompts: list[Prompt]) -> pd.DataFrame:
    """Convert prompt records into a flat DataFrame for table display."""

    return pd.DataFrame(
        [
            {
                "Name": p.name,
                "Version": p.current_version,
                "Author": p.author,
                "Status": p.status,
                "Tags": ", ".join(p.tags),
                "Description": p.description,
            }
            for p in prompts
        ]
    )


def render_status_pill(status: str) -> str:
    colors = {
        "active": ("#123524", "#4ade80"),
        "draft": ("#3a2f12", "#facc15"),
        "deprecated": ("#3a1f1f", "#f87171"),
    }
    bg, fg = colors.get(status, ("#333", "#ccc"))
    return (
        f"<span style='background-color:{bg};color:{fg};padding:2px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:600;'>{status}</span>"
    )


def render_create_prompt_form() -> None:
    with st.expander("➕ Create new prompt", expanded=False):
        with st.form("create_prompt_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Prompt name", placeholder="e.g. customer-support-triage")
                author = st.text_input("Author", placeholder="your.username")
            with col2:
                status = st.selectbox("Status", options=["draft", "active", "deprecated"])
                tags = st.text_input("Tags (comma separated)", placeholder="support, triage")

            description = st.text_area("Description", placeholder="What is this prompt for?")
            template = st.text_area(
                "Prompt template",
                placeholder="You are a helpful assistant. {{context}} ...",
                height=140,
            )

            submitted = st.form_submit_button("Create prompt", type="primary")
            if submitted:
                if not name.strip():
                    st.error("Prompt name is required.")
                else:
                    st.success(
                        f"Mock: prompt '{name}' (v1) would be created by {author or 'unknown'}. "
                        "No data was persisted."
                    )


def render_version_history(prompt: Prompt) -> None:
    st.markdown(f"#### Version history — `{prompt.name}`")
    history_df = pd.DataFrame(
        [
            {
                "Version": v.version,
                "Created": v.created_at.strftime("%Y-%m-%d %H:%M"),
                "Author": v.author,
                "Hash": v.prompt_hash,
                "Change note": v.change_note,
            }
            for v in reversed(prompt.history)
        ]
    )
    st.dataframe(history_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Prompt Registry")
    st.caption("Every prompt is versioned and immutable once published.")

    prompts = get_prompts()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total prompts", len(prompts))
    m2.metric("Active", sum(1 for p in prompts if p.status == "active"))
    m3.metric("Draft / Deprecated", sum(1 for p in prompts if p.status != "active"))

    render_create_prompt_form()

    st.divider()
    st.markdown("### All prompts")

    status_filter = st.multiselect(
        "Filter by status",
        options=["active", "draft", "deprecated"],
        default=["active", "draft", "deprecated"],
    )
    filtered = [p for p in prompts if p.status in status_filter]

    df = prompts_to_dataframe(filtered)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status"),
        },
    )

    st.divider()
    st.markdown("### Inspect a prompt")
    selected_name = st.selectbox("Select a prompt", options=[p.name for p in filtered] or [p.name for p in prompts])
    selected_prompt = next((p for p in prompts if p.name == selected_name), None)

    if selected_prompt:
        info_col, badge_col = st.columns([4, 1])
        with info_col:
            st.write(selected_prompt.description)
            st.caption(f"Author: {selected_prompt.author} · Tags: {', '.join(selected_prompt.tags)}")
        with badge_col:
            st.markdown(render_status_pill(selected_prompt.status), unsafe_allow_html=True)

        render_version_history(selected_prompt)


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()