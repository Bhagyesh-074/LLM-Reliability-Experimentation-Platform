"""Failure Analysis page.

Provides a filterable browser of individual failure cases, classified by
category (Hallucination, Factual Error, Reasoning Error, Formatting,
Refusal), with an expandable view of the question, model answer, and
ground truth for each case.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.mock.data_new import FailureCase, get_failures

_CATEGORY_ICON = {
    "Hallucination": "👻",
    "Factual Error": "❌",
    "Reasoning Error": "🧮",
    "Formatting": "🔤",
    "Refusal": "🚫",
}


def render_filters(failures: list[FailureCase]) -> tuple[list[str], list[str], list[str]]:
    models = sorted({f.model for f in failures})
    domains = sorted({f.domain for f in failures})
    categories = sorted({f.category for f in failures})

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_models = st.multiselect("Model", options=models, default=models)
    with col2:
        selected_domains = st.multiselect("Domain", options=domains, default=domains)
    with col3:
        selected_categories = st.multiselect("Failure category", options=categories, default=categories)

    return selected_models, selected_domains, selected_categories


def render_summary_table(failures: list[FailureCase]) -> None:
    df = pd.DataFrame(
        [
            {
                "Category": f"{_CATEGORY_ICON.get(f.category, '')} {f.category}",
                "Domain": f.domain,
                "Provider": f.provider,
                "Model": f.model,
                "Run": f.run_id,
                "Confidence": f.confidence,
            }
            for f in failures
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_failure_expanders(failures: list[FailureCase]) -> None:
    for failure in failures:
        icon = _CATEGORY_ICON.get(failure.category, "")
        title = f"{icon} [{failure.category}] {failure.domain} · {failure.model} · {failure.run_id}"
        with st.expander(title):
            st.markdown(f"**Question**\n\n{failure.question}")
            col_answer, col_truth = st.columns(2)
            with col_answer:
                st.markdown("**Model answer**")
                st.error(failure.model_answer)
            with col_truth:
                st.markdown("**Ground truth**")
                st.success(failure.ground_truth)
            st.caption(f"Confidence: {failure.confidence:.2f} · Provider: {failure.provider}")


def main() -> None:
    st.title("Failure Analysis")
    st.caption("Browse and classify individual evaluation failures across models, domains, and categories.")

    failures = get_failures()

    selected_models, selected_domains, selected_categories = render_filters(failures)
    filtered = [
        f
        for f in failures
        if f.model in selected_models and f.domain in selected_domains and f.category in selected_categories
    ]

    m1, m2, m3 = st.columns(3)
    m1.metric("Failures shown", len(filtered))
    m2.metric("Total failures", len(failures))
    m3.metric(
        "Most common category",
        max({f.category for f in filtered}, key=lambda c: sum(1 for f in filtered if f.category == c))
        if filtered
        else "—",
    )

    st.divider()
    st.markdown("### Summary")
    if filtered:
        render_summary_table(filtered)
    else:
        st.info("No failures match the current filters.")

    st.divider()
    st.markdown("### Case details")
    if filtered:
        render_failure_expanders(filtered)


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()