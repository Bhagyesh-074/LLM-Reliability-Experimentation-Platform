"""Main dashboard page for the LLM Reliability & Experimentation Platform.

Displays top-level KPI cards, the model leaderboard, a radar comparison
chart, recent evaluation runs, and a regression alert banner. All data is
sourced exclusively from `dashboard.mock.data` — there are no backend or
network calls on this page.
"""
from __future__ import annotations

import html
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from dashboard.components.charts import leaderboard_bar_chart, radar_chart
from dashboard.mock.data import (
    RegressionAlert,
    SummaryMetrics,
    get_leaderboard,
    get_radar_data,
    get_recent_runs,
    get_regression_alert,
    get_summary_metrics,
)

STATUS_COLORS: Dict[str, Tuple[str, str]] = {
    "passed": ("#22c55e", "rgba(34,197,94,0.12)"),
    "failed": ("#ef4444", "rgba(239,68,68,0.12)"),
    "running": ("#3b82f6", "rgba(59,130,246,0.12)"),
    "flagged": ("#f59e0b", "rgba(245,158,11,0.12)"),
}

CUSTOM_CSS = """
<style>
.stApp { background-color: #0f172a; }
[data-testid="stSidebar"] { background-color: #0b1120; }
h1, h2, h3 { color: #e2e8f0 !important; }
p, span, label { color: #cbd5e1; }

.section-title {
    color: #e2e8f0;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 1.75rem 0 0.75rem 0;
    letter-spacing: 0.01em;
}

.metric-card {
    background: linear-gradient(180deg, #161f36 0%, #121a2e 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.metric-value { color: #f8fafc; font-size: 1.7rem; font-weight: 700; }
.metric-sub { color: #818cf8; font-size: 0.78rem; margin-top: 4px; }

.alert-banner {
    background: linear-gradient(90deg, rgba(239,68,68,0.16) 0%, rgba(239,68,68,0.05) 100%);
    border: 1px solid rgba(239,68,68,0.45);
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.alert-icon { font-size: 1.3rem; line-height: 1.4; }
.alert-title { color: #fecaca; font-weight: 700; font-size: 0.95rem; }
.alert-message { color: #fca5a5; font-size: 0.86rem; margin-top: 2px; }

.panel {
    background: #121a2e;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 6px 20px 4px 20px;
}

table.custom-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
table.custom-table th {
    text-align: left;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    padding: 10px 12px;
    border-bottom: 1px solid #1e293b;
}
table.custom-table td { padding: 10px 12px; border-bottom: 1px solid #1a2338; color: #e2e8f0; }
table.custom-table tr:last-child td { border-bottom: none; }
table.custom-table tr:hover td { background: #16203a; }

.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 6px;
    background: #1e293b;
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 8px;
}
.rank-badge.top { background: #6366f1; color: #ffffff; }

.score-bar-wrap { display: flex; align-items: center; gap: 8px; min-width: 130px; }
.score-bar-bg {
    flex: 1;
    height: 6px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
    max-width: 90px;
}
.score-bar-fill { height: 100%; background: #6366f1; border-radius: 4px; }

.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: capitalize;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_regression_alert(alert: RegressionAlert) -> None:
    """Render a red banner if a regression has been detected."""
    if not alert.detected:
        return
    model = html.escape(alert.model_name or "Unknown model")
    metric = html.escape(alert.metric or "Unknown metric")
    message = html.escape(alert.message or "A regression was detected.")
    st.markdown(
        f"""
        <div class="alert-banner">
            <div class="alert-icon">🚨</div>
            <div>
                <div class="alert-title">Regression Alert — {model} · {metric}</div>
                <div class="alert-message">{message}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: SummaryMetrics) -> None:
    """Render the 4 top-level KPI cards."""
    card_defs = [
        ("Total Runs", f"{metrics.total_runs:,}", "Last 30 days"),
        ("Avg Accuracy", f"{metrics.avg_accuracy}%", "Across active models"),
        ("Best Model", metrics.best_model, "By composite score"),
        ("Active Providers", str(metrics.active_providers), "Currently monitored"),
    ]
    cols = st.columns(4)
    for col, (label, value, sub) in zip(cols, card_defs):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{html.escape(label)}</div>
                    <div class="metric-value">{html.escape(str(value))}</div>
                    <div class="metric-sub">{html.escape(sub)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_leaderboard(df: pd.DataFrame) -> None:
    """Render the model leaderboard as a styled HTML table."""
    rows_html = ""
    for i, row in df.iterrows():
        rank = int(i) + 1
        badge_cls = "rank-badge top" if rank == 1 else "rank-badge"
        composite = float(row["composite"])
        rows_html += f"""
        <tr>
            <td><span class="{badge_cls}">{rank}</span></td>
            <td>{html.escape(str(row['model_name']))}</td>
            <td>{html.escape(str(row['provider']))}</td>
            <td>{row['accuracy']:.1f}%</td>
            <td>{row['hallucination']:.1f}%</td>
            <td>{row['instruction']:.1f}%</td>
            <td>{row['safety']:.1f}%</td>
            <td>
                <div class="score-bar-wrap">
                    <span>{composite:.1f}</span>
                    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{composite:.0f}%"></div></div>
                </div>
            </td>
        </tr>
        """
    st.markdown(
        f"""
        <div class="panel">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>#</th><th>Model</th><th>Provider</th><th>Accuracy</th>
                        <th>Hallucination Resistance</th><th>Instruction</th><th>Safety</th><th>Composite</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recent_runs(df: pd.DataFrame) -> None:
    """Render the recent evaluation runs as a styled HTML table."""
    rows_html = ""
    for _, row in df.iterrows():
        status = str(row["status"])
        color, bg = STATUS_COLORS.get(status, ("#94a3b8", "rgba(148,163,184,0.12)"))
        started = row["started_at"]
        started_str = started.strftime("%b %d, %H:%M") if hasattr(started, "strftime") else str(started)
        rows_html += f"""
        <tr>
            <td>{html.escape(str(row['run_id']))}</td>
            <td>{html.escape(str(row['model_name']))}</td>
            <td>{html.escape(str(row['suite']))}</td>
            <td><span class="status-badge" style="color:{color}; background:{bg};">{html.escape(status)}</span></td>
            <td>{started_str}</td>
            <td>{int(row['duration_sec'])}s</td>
            <td>{int(row['samples'])}</td>
        </tr>
        """
    st.markdown(
        f"""
        <div class="panel">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Run ID</th><th>Model</th><th>Suite</th><th>Status</th>
                        <th>Started</th><th>Duration</th><th>Samples</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the full dashboard page."""
    st.title("LLM Reliability Dashboard")
    st.caption("Live overview of model evaluation performance, regressions, and run history.")

    render_regression_alert(get_regression_alert())

    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    render_metric_cards(get_summary_metrics())

    leaderboard_df = get_leaderboard()

    st.markdown('<div class="section-title">Model Leaderboard</div>', unsafe_allow_html=True)
    render_leaderboard(leaderboard_df)

    col_radar, col_bar = st.columns(2)
    with col_radar:
        st.markdown('<div class="section-title">Dimension Comparison</div>', unsafe_allow_html=True)
        fig_radar = radar_chart(get_radar_data(), title="Top 3 Models — 5 Dimensions")
        st.plotly_chart(fig_radar, width="stretch")
    with col_bar:
        st.markdown('<div class="section-title">Composite Ranking</div>', unsafe_allow_html=True)
        fig_bar = leaderboard_bar_chart(leaderboard_df, title="Composite Score by Model")
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown('<div class="section-title">Recent Evaluation Runs</div>', unsafe_allow_html=True)
    render_recent_runs(get_recent_runs())


def render() -> None:
    """Render the dashboard page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    main()


if __name__ == "__main__":
    render()