"""Failure Analysis page.

Provides a filterable browser of individual failure cases sourced from the
platform database (``EvaluationResult`` joined with ``FailureAnalysis`` and
``EvaluationRun``), classified by category (Hallucination, Factual Error,
Reasoning Error, Formatting Error, Refusal, Safety Issue), with an
expandable view of the question, model answer, ground truth, and per-row
scores for each case.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import func, select

from database.models import EvaluationResult, EvaluationRun, FailureAnalysis
from database.session import session_scope

_CATEGORY_ICON = {
    "Hallucination": "👻",
    "Factual Error": "❌",
    "Reasoning Error": "🧮",
    "Formatting Error": "🔤",
    "Refusal": "🚫",
    "Safety Issue": "⚠️",
}

_SEVERITY_COLOR = {
    "high": "#ff4b4b",
    "medium": "#ffd43b",
    "low": "#adb5bd",
}

_ALL_RUNS_LABEL = "All runs"


@dataclass(frozen=True)
class FailureRecord:
    """A single failure case, flattened from its joined DB rows.

    Combines one ``EvaluationResult`` row, its associated
    ``FailureAnalysis`` classification, and identifying fields from its
    parent ``EvaluationRun`` -- everything the page needs to render a case
    without holding onto ORM instances past the session's lifetime.
    """

    result_id: str
    run_id: str
    model_name: str
    question: Optional[str]
    response: Optional[str]
    ground_truth: Optional[str]
    category: Optional[str]
    severity: Optional[str]
    explanation: Optional[str]
    accuracy_score: Optional[float]
    hallucination_score: Optional[float]
    instruction_score: Optional[float]
    safety_score: Optional[float]
    composite_score: Optional[float]


@dataclass(frozen=True)
class RunOption:
    """One entry in the run selector dropdown."""

    run_id: str
    label: str


def _truncate(text: Optional[str], length: int = 100) -> str:
    """Truncate ``text`` to ``length`` characters, appending an ellipsis.

    Returns an empty string for ``None`` so downstream table rendering
    never has to special-case missing text.
    """
    if not text:
        return ""
    return text if len(text) <= length else f"{text[: length - 1]}…"


def get_run_options() -> list[RunOption]:
    """Return every evaluation run as a dropdown option, most recent first.

    Runs with a ``NULL`` ``started_at`` sort last. The label combines the
    model name, a short run id, and the start timestamp so runs from the
    same model remain distinguishable in the dropdown.
    """
    with session_scope() as session:
        stmt = select(
            EvaluationRun.run_id,
            EvaluationRun.model_name,
            EvaluationRun.started_at,
        ).order_by(EvaluationRun.started_at.desc().nulls_last())
        rows = session.execute(stmt).all()

        options: list[RunOption] = []
        for run_id, model_name, started_at in rows:
            model_label = model_name or "Unknown model"
            started_label = started_at.strftime("%Y-%m-%d %H:%M") if started_at else "unscheduled"
            options.append(
                RunOption(
                    run_id=run_id,
                    label=f"{model_label} · {run_id[:8]} · {started_label}",
                )
            )
        return options


def load_failures(run_id: Optional[str]) -> list[FailureRecord]:
    """Load failure cases from the database, optionally scoped to one run.

    Joins ``EvaluationResult`` to its ``FailureAnalysis`` classification(s)
    and to the parent ``EvaluationRun`` (for the model name), and returns
    plain, detached ``FailureRecord`` values so they can be used freely
    after the session closes.

    Args:
        run_id: If given, only failures belonging to this run are
            returned. If ``None``, failures across all runs are returned.
    """
    with session_scope() as session:
        stmt = (
            select(
                EvaluationResult.result_id,
                EvaluationResult.run_id,
                EvaluationRun.model_name,
                EvaluationResult.question,
                EvaluationResult.response,
                EvaluationResult.ground_truth,
                FailureAnalysis.category,
                FailureAnalysis.severity,
                FailureAnalysis.explanation,
                EvaluationResult.accuracy_score,
                EvaluationResult.hallucination_score,
                EvaluationResult.instruction_score,
                EvaluationResult.safety_score,
                EvaluationResult.composite_score,
            )
            .join(FailureAnalysis, FailureAnalysis.result_id == EvaluationResult.result_id)
            .join(EvaluationRun, EvaluationRun.run_id == EvaluationResult.run_id)
        )
        if run_id is not None:
            stmt = stmt.where(EvaluationResult.run_id == run_id)

        rows = session.execute(stmt).all()
        return [
            FailureRecord(
                result_id=r.result_id,
                run_id=r.run_id,
                model_name=r.model_name or "Unknown model",
                question=r.question,
                response=r.response,
                ground_truth=r.ground_truth,
                category=r.category,
                severity=r.severity,
                explanation=r.explanation,
                accuracy_score=r.accuracy_score,
                hallucination_score=r.hallucination_score,
                instruction_score=r.instruction_score,
                safety_score=r.safety_score,
                composite_score=r.composite_score,
            )
            for r in rows
        ]


def count_total_results(run_id: Optional[str]) -> int:
    """Count total evaluated questions in scope, for the failure-rate card.

    Args:
        run_id: If given, counts only ``EvaluationResult`` rows for this
            run. If ``None``, counts across all runs.
    """
    with session_scope() as session:
        stmt = select(func.count(EvaluationResult.result_id))
        if run_id is not None:
            stmt = stmt.where(EvaluationResult.run_id == run_id)
        return session.execute(stmt).scalar_one()


def render_run_selector(run_options: list[RunOption]) -> Optional[str]:
    """Render the run selector dropdown and return the selected ``run_id``.

    Returns ``None`` when "All runs" is selected.
    """
    labels = [_ALL_RUNS_LABEL] + [option.label for option in run_options]
    selected_label = st.selectbox("Run", options=labels, index=0)
    if selected_label == _ALL_RUNS_LABEL:
        return None
    return next(option.run_id for option in run_options if option.label == selected_label)


def render_filters(failures: list[FailureRecord]) -> tuple[list[str], list[str], list[str]]:
    """Render the Model / Category / Severity multiselect filters."""
    models = sorted({f.model_name for f in failures})
    categories = sorted({f.category for f in failures if f.category})
    severities = sorted({f.severity for f in failures if f.severity})

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_models = st.multiselect("Model", options=models, default=models)
    with col2:
        selected_categories = st.multiselect("Failure category", options=categories, default=categories)
    with col3:
        selected_severities = st.multiselect("Severity", options=severities, default=severities)

    return selected_models, selected_categories, selected_severities


def render_summary_cards(filtered: list[FailureRecord], total_results: int) -> None:
    """Render the four summary metric cards."""
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total failures", len(filtered))

    if filtered:
        categories = [f.category for f in filtered if f.category]
        most_common_category = max(set(categories), key=categories.count) if categories else "—"
        models = [f.model_name for f in filtered]
        most_affected_model = max(set(models), key=models.count) if models else "—"
    else:
        most_common_category = "—"
        most_affected_model = "—"

    m2.metric("Most common category", most_common_category)
    m3.metric("Most affected model", most_affected_model)

    failure_rate = (len(filtered) / total_results * 100) if total_results else 0.0
    m4.metric("Failure rate", f"{failure_rate:.1f}%")


def _style_severity(value: str) -> str:
    """Return a CSS background-color declaration for a severity value."""
    color = _SEVERITY_COLOR.get(value, "")
    return f"background-color: {color}" if color else ""


def render_summary_table(filtered: list[FailureRecord]) -> None:
    """Render the failure table with severity color-coded via a Styler."""
    df = pd.DataFrame(
        [
            {
                "Question": _truncate(f.question, 80),
                "Model": f.model_name,
                "Category": f"{_CATEGORY_ICON.get(f.category or '', '')} {f.category or ''}",
                "Severity": f.severity or "",
                "Explanation": _truncate(f.explanation, 100),
            }
            for f in filtered
        ]
    )
    styler = df.style.applymap(_style_severity, subset=["Severity"])
    st.dataframe(styler, width="stretch", hide_index=True)


def render_row_detail(filtered: list[FailureRecord]) -> None:
    """Render a selectbox + expander showing full detail for one case."""
    if not filtered:
        return

    options = {
        f"{_CATEGORY_ICON.get(f.category or '', '')} {_truncate(f.question, 60)} · {f.model_name}": f
        for f in filtered
    }
    selected_label = st.selectbox("Select a case to inspect", options=list(options.keys()))
    record = options[selected_label]

    with st.expander("Case detail", expanded=True):
        st.markdown(f"**Question**\n\n{record.question or '—'}")
        col_answer, col_truth = st.columns(2)
        with col_answer:
            st.markdown("**Model answer**")
            st.error(record.response or "—")
        with col_truth:
            st.markdown("**Ground truth**")
            st.success(record.ground_truth or "—")

        st.markdown(f"**Explanation**\n\n{record.explanation or '—'}")

        st.caption(
            f"Category: {record.category or '—'} · Severity: {record.severity or '—'} · "
            f"Run: {record.run_id}"
        )

        score_cols = st.columns(5)
        score_cols[0].metric("Accuracy", f"{record.accuracy_score:.2f}" if record.accuracy_score is not None else "—")
        score_cols[1].metric(
            "Hallucination", f"{record.hallucination_score:.2f}" if record.hallucination_score is not None else "—"
        )
        score_cols[2].metric(
            "Instruction", f"{record.instruction_score:.2f}" if record.instruction_score is not None else "—"
        )
        score_cols[3].metric("Safety", f"{record.safety_score:.2f}" if record.safety_score is not None else "—")
        score_cols[4].metric(
            "Composite", f"{record.composite_score:.2f}" if record.composite_score is not None else "—"
        )


def render_category_chart(filtered: list[FailureRecord]) -> None:
    """Render a Plotly bar chart of failure count per category."""
    if not filtered:
        return
    df = pd.DataFrame({"Category": [f.category or "Unknown" for f in filtered]})
    counts = df["Category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    fig = px.bar(counts, x="Category", y="Count", color="Category", title="Failures by category")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    """Render the Failure Analysis page."""
    st.title("Failure Analysis")
    st.caption("Browse and classify individual evaluation failures across models, domains, and categories.")

    run_options = get_run_options()
    selected_run_id = render_run_selector(run_options)

    failures = load_failures(selected_run_id)
    total_results = count_total_results(selected_run_id)

    selected_models, selected_categories, selected_severities = render_filters(failures)
    filtered = [
        f
        for f in failures
        if f.model_name in selected_models
        and f.category in selected_categories
        and f.severity in selected_severities
    ]

    st.divider()
    render_summary_cards(filtered, total_results)

    st.divider()
    st.markdown("### Summary")
    if filtered:
        render_summary_table(filtered)
    else:
        st.info("No failures match the current filters.")

    st.divider()
    st.markdown("### Case details")
    if filtered:
        render_row_detail(filtered)

    st.divider()
    st.markdown("### Category breakdown")
    if filtered:
        render_category_chart(filtered)


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()