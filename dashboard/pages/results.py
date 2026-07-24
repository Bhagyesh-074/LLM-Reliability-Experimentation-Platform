"""Results page.

Lets the user pick a past evaluation run and inspect its metric
breakdown as both a styled table and a set of KPI cards, its per-question
results, and export the full results as CSV. Data is read live from the
platform database via ``EvaluationRepository``. Visual language matches
dashboard.py.
"""

from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from database.models import EvaluationResult, EvaluationRun, RunMetrics
from database.repositories.evaluation_repository import EvaluationRepository
from database.session import session_scope

STATUS_COLORS = {
    "completed": ("#22c55e", "rgba(34,197,94,0.12)"),
    "completed_with_errors": ("#f59e0b", "rgba(245,158,11,0.12)"),
    "running": ("#3b82f6", "rgba(59,130,246,0.12)"),
    "failed": ("#ef4444", "rgba(239,68,68,0.12)"),
    "passed": ("#22c55e", "rgba(34,197,94,0.12)"),
}

#: Score -> color thresholds shared by the metrics table and per-question
#: table (green > 0.8, yellow > 0.6, red <= 0.6).
SCORE_COLOR_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (0.8, "#22c55e"),
    (0.6, "#eab308"),
)
SCORE_COLOR_DEFAULT = "#ef4444"

#: Score -> qualitative grade thresholds for the metrics breakdown table.
GRADE_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (0.85, "Excellent"),
    (0.70, "Good"),
    (0.50, "Acceptable"),
)
GRADE_DEFAULT = "Poor"

GRADE_COLORS = {
    "Excellent": ("#22c55e", "rgba(34,197,94,0.12)"),
    "Good": ("#3b82f6", "rgba(59,130,246,0.12)"),
    "Acceptable": ("#eab308", "rgba(234,179,8,0.12)"),
    "Poor": ("#ef4444", "rgba(239,68,68,0.12)"),
}

#: (RunMetrics column, display label), in display order.
RUN_METRIC_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("accuracy", "Accuracy"),
    ("hallucination", "Hallucination"),
    ("instruction", "Instruction"),
    ("safety", "Safety"),
    ("latency", "Latency"),
    ("cost", "Cost"),
    ("consistency", "Consistency"),
)

QUESTION_TRUNCATE_LEN = 80

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


def render_status_badge(status: Optional[str], colors: Dict[str, Tuple[str, str]] = STATUS_COLORS) -> str:
    """Render a small pill badge for a status/grade string, HTML-escaped."""
    label = status or "unknown"
    color, bg = colors.get(label, ("#94a3b8", "rgba(148,163,184,0.12)"))
    return (
        f'<span class="status-badge" style="color:{color}; background:{bg};">'
        f"{html.escape(label)}</span>"
    )


def score_color(score: Optional[float]) -> str:
    """Map a 0-1 score to a hex color per ``SCORE_COLOR_THRESHOLDS``."""
    if score is None:
        return "#94a3b8"
    for threshold, color in SCORE_COLOR_THRESHOLDS:
        if score > threshold:
            return color
    return SCORE_COLOR_DEFAULT


def score_grade(score: Optional[float]) -> str:
    """Map a 0-1 score to a qualitative grade per ``GRADE_THRESHOLDS``."""
    if score is None:
        return GRADE_DEFAULT
    for threshold, grade in GRADE_THRESHOLDS:
        if score > threshold:
            return grade
    return GRADE_DEFAULT


def truncate(text: Optional[str], length: int = QUESTION_TRUNCATE_LEN) -> str:
    """Truncate ``text`` to ``length`` characters, appending an ellipsis if cut."""
    if not text:
        return ""
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"


def format_run_label(run: EvaluationRun) -> str:
    """Build the run-selector dropdown label: id (short) — model — date — score."""
    short_id = run.run_id[:8]
    model = run.model_name or "unknown model"
    date = f"{run.started_at:%Y-%m-%d %H:%M}" if run.started_at else "unknown date"
    score = f"{run.composite_score:.3f}" if run.composite_score is not None else "n/a"
    return f"{short_id} — {model} — {date} — {score}"


def render_run_header(run: EvaluationRun) -> None:
    """Panel with run metadata + status badge, styled like dashboard panels.

    Requires ``run`` to have been fetched with ``provider``,
    ``prompt_version`` (+ ``prompt``), and ``dataset_version`` (+
    ``benchmark``) eagerly loaded, e.g. via
    ``EvaluationRepository.get_run_with_relations``.
    """
    provider_name = run.provider.name if run.provider else "unknown"
    prompt_name = run.prompt_version.prompt.name if run.prompt_version and run.prompt_version.prompt else "unknown"
    prompt_version_label = f"v{run.prompt_version.version}" if run.prompt_version else "n/a"
    benchmark_name = (
        run.dataset_version.benchmark.name if run.dataset_version and run.dataset_version.benchmark else "unknown"
    )
    benchmark_version = run.dataset_version.version if run.dataset_version else "n/a"

    started_label = f"{run.started_at:%Y-%m-%d %H:%M}" if run.started_at else "n/a"
    duration = compute_duration_seconds(run)
    duration_label = f"{duration:.1f}s" if duration is not None else "n/a"

    st.markdown(
        f"""
        <div class="panel-header">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div class="meta-line">
                        <b>Provider:</b> {html.escape(provider_name)} &nbsp;·&nbsp;
                        <b>Model:</b> <code>{html.escape(run.model_name or "unknown")}</code> &nbsp;·&nbsp;
                        <b>Prompt:</b> {html.escape(prompt_name)} ({html.escape(prompt_version_label)}) &nbsp;·&nbsp;
                        <b>Benchmark:</b> {html.escape(benchmark_name)} ({html.escape(str(benchmark_version))})
                    </div>
                    <div class="meta-sub">
                        Started {html.escape(started_label)} · Duration {html.escape(duration_label)}
                    </div>
                </div>
                <div>{render_status_badge(run.status)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_duration_seconds(run: EvaluationRun) -> Optional[float]:
    """Return ``completed_at - started_at`` in seconds, or ``None`` if either is missing."""
    if run.started_at is None or run.completed_at is None:
        return None
    return (run.completed_at - run.started_at).total_seconds()


def render_summary_cards(run: EvaluationRun, total_questions: int, failed_count: int) -> None:
    """Four KPI cards: composite score, total questions, failed rows, duration."""
    duration = compute_duration_seconds(run)
    cards = [
        (
            "Composite Score",
            f"{run.composite_score:.3f}" if run.composite_score is not None else "n/a",
            score_grade(run.composite_score) if run.composite_score is not None else "",
        ),
        ("Total Questions", str(total_questions), ""),
        ("Failed Rows", str(failed_count), ""),
        ("Duration", f"{duration:.1f}s" if duration is not None else "n/a", ""),
    ]
    cols = st.columns(len(cards))
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{html.escape(label)}</div>
                    <div class="metric-value">{html.escape(value)}</div>
                    <div class="metric-sub">{html.escape(sub)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def metrics_to_dataframe(metrics: Optional[RunMetrics]) -> pd.DataFrame:
    """Build the metrics-breakdown dataframe (Metric, Score, Grade) from ``RunMetrics``.

    Columns with a ``NULL`` value (e.g. an unscored component, or
    ``consistency`` which currently has no scorer) are omitted rather than
    shown as a fabricated zero.
    """
    if metrics is None:
        return pd.DataFrame(columns=["Metric", "Score", "Grade"])
    rows = []
    for column, label in RUN_METRIC_COLUMNS:
        score = getattr(metrics, column)
        if score is None:
            continue
        rows.append({"Metric": label, "Score": score, "Grade": score_grade(score)})
    return pd.DataFrame(rows, columns=["Metric", "Score", "Grade"])


def render_metrics_table(df: pd.DataFrame) -> None:
    """Metrics breakdown as a styled HTML table, with color-coded Score and Grade."""
    if df.empty:
        st.info("No aggregate metrics recorded for this run yet.")
        return
    rows_html = ""
    for _, row in df.iterrows():
        color = score_color(row["Score"])
        grade_badge = render_status_badge(row["Grade"], GRADE_COLORS)
        rows_html += f"""
        <tr>
            <td>{html.escape(str(row['Metric']))}</td>
            <td style="color:{color}; font-weight:600;">{row['Score']:.3f}</td>
            <td>{grade_badge}</td>
        </tr>
        """
    st.markdown(
        f"""
        <div class="panel">
            <table class="custom-table">
                <thead>
                    <tr><th>Metric</th><th>Score</th><th>Grade</th></tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def results_to_dataframe(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    """Build the full (untruncated) per-question dataframe used for display and CSV export."""
    rows = []
    for idx, result in enumerate(results, start=1):
        rows.append(
            {
                "#": idx,
                "result_id": result.result_id,
                "Question": result.question or "",
                "Response": result.response or "",
                "Accuracy": result.accuracy_score,
                "Hallucination": result.hallucination_score,
                "Instruction": result.instruction_score,
                "Safety": result.safety_score,
                "Composite": result.composite_score,
                "Status": result.status or "unknown",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "#",
            "result_id",
            "Question",
            "Response",
            "Accuracy",
            "Hallucination",
            "Instruction",
            "Safety",
            "Composite",
            "Status",
        ],
    )


#: Composite-column color thresholds for the per-question results table
#: (green > 0.85, yellow > 0.70, red <= 0.50; anything in between is left
#: uncolored since it falls in neither band).
COMPOSITE_COLOR_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (0.85, "#22c55e"),
    (0.70, "#eab308"),
)
COMPOSITE_COLOR_LOW = "#ef4444"
COMPOSITE_COLOR_LOW_MAX = 0.50


def composite_cell_style(value: Optional[float]) -> str:
    """Pandas Styler callback: color + weight for one Composite cell."""
    if value is None or pd.isna(value):
        return ""
    for threshold, color in COMPOSITE_COLOR_THRESHOLDS:
        if value > threshold:
            return f"color: {color}; font-weight: 600;"
    if value <= COMPOSITE_COLOR_LOW_MAX:
        return f"color: {COMPOSITE_COLOR_LOW}; font-weight: 600;"
    return ""


def _format_score(value: Optional[float]) -> str:
    """Format a nullable 0-1 score for display, or an em dash if missing."""
    return f"{value:.3f}" if pd.notna(value) else "—"


def render_results_table(df: pd.DataFrame) -> None:
    """Per-question results as an st.dataframe with truncated text and a color-coded Composite column."""
    if df.empty:
        st.info("No results match the current filters.")
        return

    display_df = pd.DataFrame(
        {
            "#": df["#"],
            "Question": df["Question"].apply(truncate),
            "Response": df["Response"].apply(truncate),
            "Accuracy": df["Accuracy"],
            "Hallucination": df["Hallucination"],
            "Composite": df["Composite"],
            "Status": df["Status"],
        }
    ).set_index("#")

    styler = display_df.style.format(
        {
            "Accuracy": _format_score,
            "Hallucination": _format_score,
            "Composite": _format_score,
        }
    )
    style_method = getattr(styler, "map", None) or styler.applymap
    styler = style_method(composite_cell_style, subset=["Composite"])

    st.dataframe(styler, use_container_width=True)


def render_result_detail(row: pd.Series) -> None:
    """Expanded detail for one selected result: full text + every metric score."""
    with st.expander(f"Result #{row['#']} details", expanded=True):
        st.markdown("**Question**")
        st.write(row["Question"] or "_(none recorded)_")
        st.markdown("**Response**")
        st.write(row["Response"] or "_(none recorded)_")

        score_cols = st.columns(5)
        score_fields = [
            ("Accuracy", row["Accuracy"]),
            ("Hallucination", row["Hallucination"]),
            ("Instruction", row["Instruction"]),
            ("Safety", row["Safety"]),
            ("Composite", row["Composite"]),
        ]
        for col, (label, value) in zip(score_cols, score_fields):
            with col:
                value_label = f"{value:.3f}" if value is not None else "—"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{html.escape(label)}</div>
                        <div class="metric-value" style="font-size:1.2rem; color:{score_color(value)};">
                            {value_label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def apply_status_filter(df: pd.DataFrame, status_filter: str) -> pd.DataFrame:
    """Filter the results dataframe by status ('All' / 'Passed' / 'Failed')."""
    if status_filter == "All":
        return df
    target = "passed" if status_filter == "Passed" else "failed"
    return df[df["Status"] == target]


def render_export_section(run: EvaluationRun, full_df: pd.DataFrame) -> None:
    """Download button exporting the complete (untruncated, unfiltered) results as CSV."""
    export_df = full_df.drop(columns=["result_id"])
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    short_id = run.run_id[:8]
    date_part = run.started_at.strftime("%Y%m%d") if run.started_at else "unknown"
    st.download_button(
        "⬇ Download results (CSV)",
        data=csv_bytes,
        file_name=f"run_{short_id}_{date_part}.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:
    """Render the Results page end to end within a single DB session."""
    st.title("Results")
    st.caption("Inspect metric breakdowns for any completed evaluation run.")

    with session_scope() as session:
        repo = EvaluationRepository(session)
        runs: Sequence[EvaluationRun] = repo.list_runs()

        if not runs:
            st.warning("No evaluation runs found. Run an evaluation first.")
            return

        run_labels = {format_run_label(r): r.run_id for r in runs}
        selected_label = st.selectbox("Select an evaluation run", options=list(run_labels.keys()))
        selected_run_id = run_labels[selected_label]

        run = repo.get_run_with_relations(selected_run_id)
        if run is None:
            st.warning("No evaluation runs found. Run an evaluation first.")
            return

        render_run_header(run)

        results: Sequence[EvaluationResult] = repo.get_results(selected_run_id)
        full_df = results_to_dataframe(results)
        failed_count = int((full_df["Status"] == "failed").sum()) if not full_df.empty else 0

        st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
        render_summary_cards(run, total_questions=len(results), failed_count=failed_count)

        st.markdown('<div class="section-title">Metrics Breakdown</div>', unsafe_allow_html=True)
        metrics = repo.get_metrics(selected_run_id)
        render_metrics_table(metrics_to_dataframe(metrics))

        st.markdown('<div class="section-title">Per-Question Results</div>', unsafe_allow_html=True)
        status_filter = st.selectbox("Filter by status", options=["All", "Passed", "Failed"])
        filtered_df = apply_status_filter(full_df, status_filter)
        render_results_table(filtered_df)

        if not filtered_df.empty:
            detail_options = {
                f"#{r['#']} — {truncate(r['Question'], 60)}": r["#"] for _, r in filtered_df.iterrows()
            }
            selected_detail_label = st.selectbox(
                "Inspect a result", options=list(detail_options.keys()), key="result_detail_select"
            )
            selected_row_number = detail_options[selected_detail_label]
            selected_row = filtered_df[filtered_df["#"] == selected_row_number].iloc[0]
            render_result_detail(selected_row)

        st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
        render_export_section(run, full_df)


def render() -> None:
    """Render the results page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    main()


if __name__ == "__main__":
    render()