"""Results page.

Lets the user pick a past evaluation run and inspect its metric
breakdown as both a styled table and a set of KPI cards, then download
the results as CSV (mock data). Visual language matches dashboard.py.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.mock.data_new import EvaluationRun, get_evaluation_runs

STATUS_COLORS = {
    "completed": ("#22c55e", "rgba(34,197,94,0.12)"),
    "running": ("#3b82f6", "rgba(59,130,246,0.12)"),
    "failed": ("#ef4444", "rgba(239,68,68,0.12)"),
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

.panel {
    background: #121a2e;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 6px 20px 4px 20px;
}
.panel-header {
    background: #121a2e;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 1rem;
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

.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: capitalize;
}

.meta-line { color: #cbd5e1; font-size: 0.9rem; }
.meta-line code {
    background: #1e293b;
    color: #a5b4fc;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.85rem;
}
.meta-sub { color: #94a3b8; font-size: 0.8rem; margin-top: 4px; }
</style>
"""


def render_status_badge(status: str) -> str:
    color, bg = STATUS_COLORS.get(status, ("#94a3b8", "rgba(148,163,184,0.12)"))
    return (
        f'<span class="status-badge" style="color:{color}; background:{bg};">'
        f"{html.escape(status)}</span>"
    )


def metrics_to_dataframe(run: EvaluationRun) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": m.metric_name,
                "Mean": m.mean,
                "Median": m.median,
                "Std Dev": m.std_dev,
                "Unit": m.unit,
            }
            for m in run.metrics
        ]
    )


def render_run_header(run: EvaluationRun) -> None:
    """Panel with run metadata + status badge, styled like dashboard panels."""
    st.markdown(
        f"""
        <div class="panel-header">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div class="meta-line">
                        <b>Provider:</b> {html.escape(run.provider)} &nbsp;·&nbsp;
                        <b>Model:</b> <code>{html.escape(run.model)}</code> &nbsp;·&nbsp;
                        <b>Prompt:</b> {html.escape(run.prompt_name)} ({html.escape(run.prompt_version)}) &nbsp;·&nbsp;
                        <b>Benchmark:</b> {html.escape(run.benchmark_name)} ({html.escape(run.benchmark_version)})
                    </div>
                    <div class="meta-sub">
                        Started {run.started_at:%Y-%m-%d %H:%M} · Duration {run.duration_seconds:.1f}s
                    </div>
                </div>
                <div>{render_status_badge(run.status)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_cards(run: EvaluationRun) -> None:
    """KPI cards matching dashboard.py's metric-card style."""
    cards_per_row = 4
    metrics = run.metrics
    for row_start in range(0, len(metrics), cards_per_row):
        row_metrics = metrics[row_start : row_start + cards_per_row]
        cols = st.columns(cards_per_row)
        for col, metric in zip(cols, row_metrics):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{html.escape(metric.metric_name)}</div>
                        <div class="metric-value">{metric.mean} {html.escape(metric.unit)}</div>
                        <div class="metric-sub">median {metric.median} · std {metric.std_dev}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_metrics_table(df: pd.DataFrame) -> None:
    """Metrics breakdown as a styled HTML table matching dashboard's custom-table."""
    rows_html = ""
    for _, row in df.iterrows():
        rows_html += f"""
        <tr>
            <td>{html.escape(str(row['Metric']))}</td>
            <td>{row['Mean']}</td>
            <td>{row['Median']}</td>
            <td>{row['Std Dev']}</td>
            <td>{html.escape(str(row['Unit']))}</td>
        </tr>
        """
    st.markdown(
        f"""
        <div class="panel">
            <table class="custom-table">
                <thead>
                    <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Unit</th></tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Results")
    st.caption("Inspect metric breakdowns for any completed evaluation run.")

    runs = get_evaluation_runs()
    run_labels = {
        f"{r.run_id} — {r.model} on {r.benchmark_name} ({r.started_at:%Y-%m-%d %H:%M})": r.run_id
        for r in runs
    }

    selected_label = st.selectbox("Select an evaluation run", options=list(run_labels.keys()))
    selected_run_id = run_labels[selected_label]
    run = next(r for r in runs if r.run_id == selected_run_id)

    render_run_header(run)

    st.markdown('<div class="section-title">Score Cards</div>', unsafe_allow_html=True)
    render_score_cards(run)

    st.markdown('<div class="section-title">Metrics Breakdown</div>', unsafe_allow_html=True)
    metrics_df = metrics_to_dataframe(run)
    render_metrics_table(metrics_df)

    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download results (CSV)",
        data=csv_bytes,
        file_name=f"{run.run_id}_metrics.csv",
        mime="text/csv",
        type="primary",
    )


def render() -> None:
    """Render the results page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    main()


if __name__ == "__main__":
    render()