"""Benchmark Registry page.

Displays all versioned benchmark datasets and lets the user upload a new
CSV benchmark file, previewing the first five rows (mock ingestion — the
file is parsed in-memory only and not persisted anywhere).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.mock.data_new import Benchmark, get_benchmarks



_DIFFICULTY_ICON = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}


def benchmarks_to_dataframe(benchmarks: list[Benchmark]) -> pd.DataFrame:
    """Convert benchmark records into a flat DataFrame for table display."""

    return pd.DataFrame(
        [
            {
                "Name": b.name,
                "Domain": b.domain,
                "Version": b.version,
                "Questions": b.question_count,
                "Difficulty": f"{_DIFFICULTY_ICON.get(b.difficulty, '')} {b.difficulty}",
                "Creator": b.creator,
                "Tags": ", ".join(b.tags),
            }
            for b in benchmarks
        ]
    )


def render_upload_section() -> None:
    st.markdown("### Upload a new benchmark")
    st.caption("Upload a CSV of questions/answers. This is a local preview only — nothing is saved yet.")

    col_file, col_meta = st.columns([2, 1])
    with col_file:
        uploaded = st.file_uploader("Benchmark CSV", type=["csv"])
    with col_meta:
        domain = st.selectbox(
            "Domain",
            options=[
                "Medical",
                "Legal",
                "Finance",
                "Coding",
                "Math",
                "Safety",
                "Prompt Injection",
                "Long Context",
                "Summarization",
            ],
        )
        version = st.text_input("Version tag", value="v1.0")

    if uploaded is not None:
        try:
            preview_df = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
            st.error(f"Could not parse CSV: {exc}")
            return

        st.success(
            f"Parsed {len(preview_df)} rows for domain **{domain}**, version **{version}**. "
            "Showing first 5 rows below."
        )
        st.dataframe(preview_df.head(5), use_container_width=True, hide_index=True)

        register_col, _ = st.columns([1, 3])
        with register_col:
            if st.button("Register benchmark", type="primary", use_container_width=True):
                st.success(
                    f"Mock: benchmark with {len(preview_df)} questions would be registered as "
                    f"{domain} {version}. No data was persisted."
                )


def main() -> None:
    st.title("Benchmark Registry")
    st.caption("Standardized, versioned datasets used to evaluate model performance across domains.")

    benchmarks = get_benchmarks()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total benchmarks", len(benchmarks))
    m2.metric("Total questions", sum(b.question_count for b in benchmarks))
    m3.metric("Domains covered", len({b.domain for b in benchmarks}))

    st.divider()
    st.markdown("### All benchmarks")

    domain_filter = st.multiselect(
        "Filter by domain",
        options=sorted({b.domain for b in benchmarks}),
        default=sorted({b.domain for b in benchmarks}),
    )
    filtered = [b for b in benchmarks if b.domain in domain_filter]

    st.dataframe(
        benchmarks_to_dataframe(filtered),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    render_upload_section()


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()