"""Analytics page.

Visualizes cross-run trends: accuracy over time, cost per provider,
latency comparison, and the relationship between temperature and
accuracy. All figures are built with Plotly against mock data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.mock.data_new import (
    get_accuracy_trend,
    get_cost_by_provider,
    get_latency_by_provider,
    get_temperature_vs_accuracy,
)

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


def render_accuracy_trend_chart() -> None:
    df = pd.DataFrame(get_accuracy_trend())
    fig = px.line(
        df,
        x="date",
        y="accuracy",
        color="model",
        markers=True,
        title="Accuracy over time (last 30 days)",
        labels={"date": "Date", "accuracy": "Accuracy (%)", "model": "Model"},
    )
    fig.update_layout(**_DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def render_cost_chart() -> None:
    df = pd.DataFrame(get_cost_by_provider())
    fig = px.bar(
        df,
        x="provider",
        y="cost_usd_per_1k",
        color="provider",
        title="Cost per provider ($ / 1k calls)",
        labels={"provider": "Provider", "cost_usd_per_1k": "$ / 1k calls"},
        text_auto=".2f",
    )
    fig.update_layout(**_DARK_LAYOUT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_latency_chart() -> None:
    df = pd.DataFrame(get_latency_by_provider())
    fig = go.Figure()
    fig.add_bar(name="p50", x=df["provider"], y=df["p50_ms"])
    fig.add_bar(name="p95", x=df["provider"], y=df["p95_ms"])
    fig.update_layout(
        barmode="group",
        xaxis_title="Provider",
        yaxis_title="Latency (ms)",
        **_DARK_LAYOUT,
    )
    fig.update_layout(title_text="Latency comparison (p50 vs p95, ms)")
    st.plotly_chart(fig, use_container_width=True)


def render_temperature_scatter() -> None:
    df = pd.DataFrame(get_temperature_vs_accuracy())
    fig = px.scatter(
        df,
        x="temperature",
        y="accuracy",
        color="model",
        title="Temperature vs. accuracy",
        labels={"temperature": "Temperature", "accuracy": "Accuracy (%)", "model": "Model"},
    )
    fig.update_layout(**_DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Analytics")
    st.caption("Cross-run trends and comparisons across providers, models, and configurations.")

    render_accuracy_trend_chart()

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        render_cost_chart()
    with col2:
        render_latency_chart()

    st.divider()
    render_temperature_scatter()


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()