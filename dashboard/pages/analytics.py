"""Analytics page.

Visualizes cross-run trends: composite score over time, cost per
provider, latency comparison, and the relationship between temperature
and accuracy. All figures are built with Plotly against live data
pulled from the database via ``session_scope()``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import EvaluationResult, EvaluationRun, Provider, RunMetrics
from database.session import session_scope

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=50, b=60),
    title=dict(x=0.01, xanchor="left", y=0.98, yanchor="top", font=dict(size=16)),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.18,
        xanchor="left",
        x=0,
    ),
)

# Minimum number of evaluation runs required before trend charts are
# meaningful. Below this, we show an info message instead of sparse
# or empty-looking charts.
_MIN_RUNS_FOR_TRENDS = 2


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------


def _count_runs(session: Session) -> int:
    """Return the total number of evaluation runs recorded in the database."""
    stmt = select(func.count()).select_from(EvaluationRun)
    return session.execute(stmt).scalar_one()


def _fetch_trend_data(session: Session) -> pd.DataFrame:
    """Return ``started_at`` / ``composite_score`` / ``model_name`` rows.

    Only runs that have both a ``started_at`` timestamp and a
    ``composite_score`` are included, ordered chronologically so the
    resulting line chart reads left-to-right in time.
    """
    stmt = (
        select(
            EvaluationRun.started_at,
            EvaluationRun.composite_score,
            EvaluationRun.model_name,
        )
        .where(
            EvaluationRun.started_at.is_not(None),
            EvaluationRun.composite_score.is_not(None),
        )
        .order_by(EvaluationRun.started_at.asc())
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=["started_at", "composite_score", "model_name"])


def _fetch_cost_by_provider(session: Session) -> Tuple[pd.DataFrame, str]:
    """Return an average cost-like metric per provider.

    Prefers ``RunMetrics.cost`` averaged per provider. If no run in the
    database has a recorded cost value, falls back to average
    ``EvaluationResult.token_usage`` per provider so the chart still
    conveys a meaningful relative comparison between providers.

    Returns:
        A tuple of ``(dataframe, metric_name)`` where the dataframe has
        columns ``provider`` and ``value``, and ``metric_name`` is either
        ``"cost_usd_per_1k"`` or ``"avg_token_usage"`` depending on which
        source was used.
    """
    cost_stmt = (
        select(Provider.name, func.avg(RunMetrics.cost))
        .join(EvaluationRun, EvaluationRun.provider_id == Provider.provider_id)
        .join(RunMetrics, RunMetrics.run_id == EvaluationRun.run_id)
        .where(RunMetrics.cost.is_not(None))
        .group_by(Provider.name)
    )
    rows = session.execute(cost_stmt).all()
    if rows:
        return pd.DataFrame(rows, columns=["provider", "value"]), "cost_usd_per_1k"

    token_stmt = (
        select(Provider.name, func.avg(EvaluationResult.token_usage))
        .join(EvaluationRun, EvaluationRun.provider_id == Provider.provider_id)
        .join(EvaluationResult, EvaluationResult.run_id == EvaluationRun.run_id)
        .where(EvaluationResult.token_usage.is_not(None))
        .group_by(Provider.name)
    )
    rows = session.execute(token_stmt).all()
    return pd.DataFrame(rows, columns=["provider", "value"]), "avg_token_usage"


def _fetch_latency_by_model(session: Session) -> pd.DataFrame:
    """Return average ``EvaluationResult.latency_ms`` grouped by model_name."""
    stmt = (
        select(
            EvaluationRun.model_name,
            func.avg(EvaluationResult.latency_ms),
        )
        .join(EvaluationResult, EvaluationResult.run_id == EvaluationRun.run_id)
        .where(EvaluationResult.latency_ms.is_not(None))
        .group_by(EvaluationRun.model_name)
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=["model_name", "avg_latency_ms"])


def _fetch_temperature_vs_accuracy(session: Session) -> pd.DataFrame:
    """Return ``temperature`` / ``accuracy`` / ``model_name`` rows for the scatter plot."""
    stmt = (
        select(
            EvaluationRun.temperature,
            RunMetrics.accuracy,
            EvaluationRun.model_name,
        )
        .join(RunMetrics, RunMetrics.run_id == EvaluationRun.run_id)
        .where(
            EvaluationRun.temperature.is_not(None),
            RunMetrics.accuracy.is_not(None),
        )
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=["temperature", "accuracy", "model_name"])


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_accuracy_trend_chart(df: pd.DataFrame) -> None:
    """Render the composite-score-over-time line chart.

    One line per ``model_name`` when more than one model is present in
    ``df``; color grouping is skipped when only a single model was
    tested, per requirements.
    """
    if df.empty:
        st.info("Run more evaluations to see trends")
        return

    color_col: Optional[str] = "model_name" if df["model_name"].nunique() > 1 else None
    fig = px.line(
        df,
        x="started_at",
        y="composite_score",
        color=color_col,
        markers=True,
        title="Composite score over time",
        labels={
            "started_at": "Date",
            "composite_score": "Composite Score",
            "model_name": "Model",
        },
    )
    fig.update_layout(**_DARK_LAYOUT)
    st.plotly_chart(fig, width="stretch")


def render_cost_chart(df: pd.DataFrame, metric_name: str) -> None:
    """Render the average cost (or token usage fallback) per provider bar chart."""
    if df.empty:
        st.info("Run more evaluations to see trends")
        return

    if metric_name == "cost_usd_per_1k":
        title = "Cost per provider ($ / 1k calls)"
        y_label = "$ / 1k calls"
    else:
        title = "Token usage per provider (avg tokens/call)"
        y_label = "Avg tokens/call"

    fig = px.bar(
        df,
        x="provider",
        y="value",
        color="provider",
        title=title,
        labels={"provider": "Provider", "value": y_label},
        text_auto=".2f",
    )
    fig.update_layout(**_DARK_LAYOUT, showlegend=False)
    st.plotly_chart(fig, width="stretch")


def render_latency_chart(df: pd.DataFrame) -> None:
    """Render the average result latency per model bar chart."""
    if df.empty:
        st.info("Run more evaluations to see trends")
        return

    fig = px.bar(
        df,
        x="model_name",
        y="avg_latency_ms",
        color="model_name",
        title="Latency comparison (avg ms per model)",
        labels={"model_name": "Model", "avg_latency_ms": "Avg latency (ms)"},
        text_auto=".0f",
    )
    fig.update_layout(**_DARK_LAYOUT, showlegend=False)
    st.plotly_chart(fig, width="stretch")


def render_temperature_scatter(df: pd.DataFrame) -> None:
    """Render the temperature-vs-accuracy scatter plot.

    Color grouping by ``model_name`` is skipped when only a single
    model was tested, per requirements.
    """
    if df.empty:
        st.info("Run more evaluations to see trends")
        return

    color_col: Optional[str] = "model_name" if df["model_name"].nunique() > 1 else None
    fig = px.scatter(
        df,
        x="temperature",
        y="accuracy",
        color=color_col,
        title="Temperature vs. accuracy",
        labels={"temperature": "Temperature", "accuracy": "Accuracy (%)", "model_name": "Model"},
    )
    fig.update_layout(**_DARK_LAYOUT)
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.title("Analytics")
    st.caption("Cross-run trends and comparisons across providers, models, and configurations.")

    with session_scope() as session:
        run_count = _count_runs(session)
        if run_count < _MIN_RUNS_FOR_TRENDS:
            st.info("Run more evaluations to see trends")
            return

        trend_df = _fetch_trend_data(session)
        cost_df, cost_metric = _fetch_cost_by_provider(session)
        latency_df = _fetch_latency_by_model(session)
        temperature_df = _fetch_temperature_vs_accuracy(session)

    render_accuracy_trend_chart(trend_df)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        render_cost_chart(cost_df, cost_metric)
    with col2:
        render_latency_chart(latency_df)

    st.divider()
    render_temperature_scatter(temperature_df)


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()