"""Main dashboard page for the LLM Reliability & Experimentation Platform.

Displays top-level KPI cards, the model leaderboard, a radar comparison
chart, recent evaluation runs, and a regression alert banner. All data is
sourced from the application database via `session_scope()` — there are
no mock data calls on this page.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import EvaluationRun, Provider, RunMetrics
from database.repositories.evaluation_repository import EvaluationRepository
from database.session import session_scope
from dashboard.components.charts import leaderboard_bar_chart, radar_chart
from stats_engine.regression import RegressionDetector

# The regression threshold (in percent) is sourced from application
# settings when a `config.settings` module exposing
# `regression_threshold_pct` is available, so ops can tune sensitivity
# without a code change. Falls back to 10.0 if settings aren't wired up.
try:
    from config.settings import settings as _settings

    DEFAULT_REGRESSION_THRESHOLD_PCT: float = float(
        getattr(_settings, "regression_threshold_pct", 10.0)
    )
except ImportError:
    DEFAULT_REGRESSION_THRESHOLD_PCT = 10.0

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

.alert-banner-warning {
    background: linear-gradient(90deg, rgba(245,158,11,0.16) 0%, rgba(245,158,11,0.05) 100%);
    border: 1px solid rgba(245,158,11,0.45);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.alert-banner-warning .alert-title { color: #fde68a; font-weight: 700; font-size: 0.95rem; }
.alert-banner-warning .alert-message { color: #fcd34d; font-size: 0.86rem; margin-top: 2px; }

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


# --------------------------------------------------------------------------
# Data transfer objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryMetrics:
    """Top-level KPI values shown in the Overview cards."""

    total_runs: int
    avg_accuracy: float
    best_model: str
    active_providers: int


@dataclass(frozen=True)
class RegressionAlert:
    """A single model's regression finding, as reported by RegressionDetector.

    One instance is produced per model whose latest run regressed against
    its immediately preceding run. Multiple instances may coexist when
    more than one model regressed in the same evaluation cycle.
    """

    model_name: str
    severity: str
    pct_change: float
    baseline_score: float
    current_score: float
    message: str


# --------------------------------------------------------------------------
# Data access — all queries go through session_scope()
# --------------------------------------------------------------------------


def _count_total_runs(session: Session) -> int:
    """Return the total number of evaluation runs."""
    return session.execute(select(func.count(EvaluationRun.run_id))).scalar_one()


def _avg_accuracy(session: Session) -> Optional[float]:
    """Return the mean ``RunMetrics.accuracy`` across all runs, or ``None`` if unset."""
    return session.execute(select(func.avg(RunMetrics.accuracy))).scalar_one()


def _best_model_by_avg_composite(session: Session) -> Optional[str]:
    """Return the model_name with the highest average composite_score."""
    stmt = (
        select(EvaluationRun.model_name, func.avg(EvaluationRun.composite_score).label("avg_score"))
        .where(EvaluationRun.model_name.is_not(None))
        .where(EvaluationRun.composite_score.is_not(None))
        .group_by(EvaluationRun.model_name)
        .order_by(func.avg(EvaluationRun.composite_score).desc())
        .limit(1)
    )
    row = session.execute(stmt).first()
    return row[0] if row is not None else None


def _count_active_providers(session: Session) -> int:
    """Return the number of distinct providers that have been used in a run."""
    return session.execute(
        select(func.count(func.distinct(EvaluationRun.provider_id)))
    ).scalar_one()


def get_summary_metrics() -> SummaryMetrics:
    """Fetch the four top-level KPI values for the Overview cards."""
    with session_scope() as session:
        total_runs = _count_total_runs(session)
        avg_accuracy = _avg_accuracy(session)
        best_model = _best_model_by_avg_composite(session)
        active_providers = _count_active_providers(session)

    return SummaryMetrics(
        total_runs=total_runs,
        avg_accuracy=round(float(avg_accuracy), 1) if avg_accuracy is not None else 0.0,
        best_model=best_model or "—",
        active_providers=active_providers,
    )


def get_leaderboard() -> pd.DataFrame:
    """Fetch one aggregated row per model for the leaderboard and bar chart.

    Each model is represented once, averaging its composite score and
    per-dimension metrics across every run it has, and counting how many
    runs contributed. The model's provider is taken from its most recent
    run. Columns: ``model_name``, ``provider``, ``accuracy``,
    ``hallucination``, ``instruction``, ``safety``, ``composite``,
    ``runs_count``. Sorted by ``composite`` descending.
    """
    with session_scope() as session:
        agg_stmt = (
            select(
                EvaluationRun.model_name,
                func.avg(EvaluationRun.composite_score).label("composite"),
                func.count(EvaluationRun.run_id).label("runs_count"),
            )
            .where(EvaluationRun.model_name.is_not(None))
            .group_by(EvaluationRun.model_name)
        )
        agg_rows = session.execute(agg_stmt).all()

        latest_run_subq = (
            select(
                EvaluationRun.model_name.label("model_name"),
                func.max(EvaluationRun.started_at).label("max_started"),
            )
            .where(EvaluationRun.model_name.is_not(None))
            .group_by(EvaluationRun.model_name)
            .subquery()
        )
        latest_provider_stmt = (
            select(EvaluationRun.model_name, Provider.name)
            .join(Provider, Provider.provider_id == EvaluationRun.provider_id)
            .join(
                latest_run_subq,
                (EvaluationRun.model_name == latest_run_subq.c.model_name)
                & (EvaluationRun.started_at == latest_run_subq.c.max_started),
            )
        )
        provider_by_model: Dict[str, str] = dict(session.execute(latest_provider_stmt).all())

        dim_stmt = (
            select(
                EvaluationRun.model_name,
                func.avg(RunMetrics.accuracy).label("accuracy"),
                func.avg(RunMetrics.hallucination).label("hallucination"),
                func.avg(RunMetrics.instruction).label("instruction"),
                func.avg(RunMetrics.safety).label("safety"),
            )
            .join(RunMetrics, RunMetrics.run_id == EvaluationRun.run_id)
            .where(EvaluationRun.model_name.is_not(None))
            .group_by(EvaluationRun.model_name)
        )
        dim_by_model = {row.model_name: row for row in session.execute(dim_stmt).all()}

    records = []
    for row in agg_rows:
        model_name = row.model_name
        dims = dim_by_model.get(model_name)
        records.append(
            {
                "model_name": model_name,
                "provider": provider_by_model.get(model_name, "Unknown"),
                "accuracy": float(dims.accuracy) if dims and dims.accuracy is not None else 0.0,
                "hallucination": float(dims.hallucination) if dims and dims.hallucination is not None else 0.0,
                "instruction": float(dims.instruction) if dims and dims.instruction is not None else 0.0,
                "safety": float(dims.safety) if dims and dims.safety is not None else 0.0,
                "composite": float(row.composite) if row.composite is not None else 0.0,
                "runs_count": int(row.runs_count),
            }
        )

    df = pd.DataFrame.from_records(
        records,
        columns=["model_name", "provider", "accuracy", "hallucination", "instruction", "safety", "composite", "runs_count"],
    )
    if df.empty:
        return df
    return df.sort_values("composite", ascending=False).reset_index(drop=True)


def get_radar_data() -> pd.DataFrame:
    """Fetch per-dimension scores for the top 3 models by average composite score.

    ROOT CAUSE OF THE "only Hallucination shows" BUG (now fixed here): the
    previous implementation joined each model to a single specific run
    (the one with the highest ``composite_score``) and read that ONE
    run's ``RunMetrics`` row. ``RunMetrics.accuracy``, ``.instruction``,
    and ``.safety`` are independently nullable, so whenever the run with
    the top composite score happened to have those columns still NULL
    (e.g. a judging pass hadn't populated them yet), the ``None -> 0.0``
    defaulting silently zeroed 3 of the 4 radar axes for that model, even
    though other runs for the same model had real values for those
    dimensions.

    Fix: aggregate with SQL ``AVG()`` across *all* of a model's runs, per
    dimension, independently (the same pattern already used successfully
    in ``get_leaderboard()``'s ``dim_stmt``). SQL's ``AVG()`` ignores NULL
    rows per column, so a model's Accuracy average is computed only from
    the runs that actually recorded accuracy, its Instruction average
    only from runs that recorded instruction, etc. A dimension now only
    shows 0 if truly no run for that model ever recorded it.

    Returns a long-format DataFrame with columns ``["model", "dimension",
    "value"]`` (one row per model+dimension pair), matching what
    ``dashboard.components.charts.radar_chart`` expects.
    """
    with session_scope() as session:
        # Top 3 models by average composite_score — same ranking basis as
        # the leaderboard — used to scope which models the radar covers.
        top_models_stmt = (
            select(EvaluationRun.model_name)
            .where(EvaluationRun.model_name.is_not(None))
            .where(EvaluationRun.composite_score.is_not(None))
            .group_by(EvaluationRun.model_name)
            .order_by(func.avg(EvaluationRun.composite_score).desc())
            .limit(3)
        )
        top_models = [row[0] for row in session.execute(top_models_stmt).all()]

        if not top_models:
            return pd.DataFrame(columns=["model", "dimension", "value"])

        dim_stmt = (
            select(
                EvaluationRun.model_name,
                func.avg(RunMetrics.accuracy).label("accuracy"),
                func.avg(RunMetrics.hallucination).label("hallucination"),
                func.avg(RunMetrics.instruction).label("instruction"),
                func.avg(RunMetrics.safety).label("safety"),
            )
            .join(RunMetrics, RunMetrics.run_id == EvaluationRun.run_id)
            .where(EvaluationRun.model_name.in_(top_models))
            .group_by(EvaluationRun.model_name)
        )
        dim_rows = session.execute(dim_stmt).all()

    records = []
    for row in dim_rows:
        records.append(
            {
                "model": row.model_name,
                # Keys below are RunMetrics' actual attribute names
                # (accuracy, hallucination, instruction, safety) — this is
                # the single source of truth for DIMENSION_COLUMNS below,
                # so the melt can never silently drop or miss a column.
                "accuracy": (float(row.accuracy) if row.accuracy is not None else 0.0) * 100.0,
                "hallucination": 100.0 - (float(row.hallucination) if row.hallucination is not None else 0.0) * 100.0,
                "instruction": (float(row.instruction) if row.instruction is not None else 0.0) * 100.0,
                "safety": (float(row.safety) if row.safety is not None else 0.0) * 100.0,
            }
        )

    DIMENSION_COLUMNS = ["accuracy", "hallucination", "instruction", "safety"]
    wide_df = pd.DataFrame.from_records(records, columns=["model", *DIMENSION_COLUMNS])
    if wide_df.empty:
        return pd.DataFrame(columns=["model", "dimension", "value"])

    # radar_chart() expects one row per (model, dimension) pair, so melt
    # from wide (one column per dimension) to long format. value_vars is
    # passed explicitly (rather than relying on "melt everything not in
    # id_vars") so the set of dimensions plotted always matches
    # DIMENSION_COLUMNS exactly, even if wide_df ever gains extra columns.
    long_df = wide_df.melt(
        id_vars="model", value_vars=DIMENSION_COLUMNS, var_name="dimension", value_name="value"
    )
    long_df["dimension"] = long_df["dimension"].str.capitalize()

    # Preserve the top_models ranking order for a stable legend/plot order.
    long_df["model"] = pd.Categorical(long_df["model"], categories=top_models, ordered=True)
    return long_df.sort_values(["model", "dimension"]).reset_index(drop=True)


def get_recent_runs() -> pd.DataFrame:
    """Fetch the 5 most recent evaluation runs for the Recent Runs table."""
    with session_scope() as session:
        repo = EvaluationRepository(session)
        runs = repo.list_runs(limit=5)
        records = [
            {
                "run_id": run.run_id,
                "model_name": run.model_name or "Unknown",
                "started_at": run.started_at,
                "composite_score": run.composite_score,
                "status": run.status or "unknown",
            }
            for run in runs
        ]

    return pd.DataFrame.from_records(
        records, columns=["run_id", "model_name", "started_at", "composite_score", "status"]
    )


def get_regression_alerts() -> List[RegressionAlert]:
    """Detect composite-score regressions by comparing each model's two most recent runs.

    For every model with at least two runs, the current run's
    ``composite_score`` is compared against the immediately preceding
    (baseline) run's ``composite_score`` via
    :meth:`RegressionDetector.detect`, using
    :data:`DEFAULT_REGRESSION_THRESHOLD_PCT` as the threshold. Any model
    whose result is flagged as a regression (``major`` or ``minor``
    severity) contributes one :class:`RegressionAlert` to the returned
    list, so multiple simultaneous regressions are all reported.

    Returns:
        A list of alerts, one per regressed model. Empty if no model
        regressed or if fewer than two runs exist for every model.
    """
    with session_scope() as session:
        stmt = (
            select(EvaluationRun.model_name, EvaluationRun.composite_score, EvaluationRun.started_at)
            .where(EvaluationRun.model_name.is_not(None))
            .where(EvaluationRun.composite_score.is_not(None))
            .order_by(EvaluationRun.model_name, EvaluationRun.started_at.desc().nulls_last())
        )
        rows = session.execute(stmt).all()

    runs_by_model: Dict[str, List[float]] = {}
    for model_name, composite_score, _started_at in rows:
        runs_by_model.setdefault(model_name, []).append(float(composite_score))

    detector = RegressionDetector()
    alerts: List[RegressionAlert] = []

    for model_name, scores in runs_by_model.items():
        if len(scores) < 2:
            # Need a current run and a prior baseline run to compare.
            continue
        current_score, baseline_score = scores[0], scores[1]

        result = detector.detect(
            current_score=current_score,
            baseline_score=baseline_score,
            threshold_pct=DEFAULT_REGRESSION_THRESHOLD_PCT,
        )

        if not result.is_regression or result.severity not in ("major", "minor"):
            continue

        alerts.append(
            RegressionAlert(
                model_name=model_name,
                severity=result.severity,
                pct_change=result.pct_change,
                baseline_score=baseline_score,
                current_score=current_score,
                message=(
                    f"Regression detected: {model_name} dropped {result.pct_change:.1f}% "
                    f"(from {baseline_score:.1f} to {current_score:.1f})"
                ),
            )
        )

    return alerts


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_regression_alerts(alerts: List[RegressionAlert]) -> None:
    """Render one banner per regressed model, or a green note if none regressed.

    Major-severity regressions render in the existing red ``alert-banner``
    style; minor-severity regressions render in the ``alert-banner-warning``
    (yellow) style. Multiple banners are rendered when multiple models
    have regressed simultaneously.
    """
    if not alerts:
        st.caption("✅ No regressions detected across the latest evaluation runs.")
        return

    for alert in alerts:
        model = html.escape(alert.model_name)
        message = html.escape(alert.message)
        if alert.severity == "major":
            banner_class, icon, title = "alert-banner", "🚨", f"Regression Alert — {model} · composite_score"
        else:
            banner_class, icon, title = "alert-banner-warning", "⚠️", f"Minor Regression — {model} · composite_score"
        st.markdown(
            f"""
            <div class="{banner_class}">
                <div class="alert-icon">{icon}</div>
                <div>
                    <div class="alert-title">{title}</div>
                    <div class="alert-message">{message}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metric_cards(metrics: SummaryMetrics) -> None:
    """Render the 4 top-level KPI cards."""
    card_defs = [
        ("Total Runs", f"{metrics.total_runs:,}", "All time"),
        ("Avg Accuracy", f"{metrics.avg_accuracy}%", "Across active models"),
        ("Best Model", metrics.best_model, "By avg composite score"),
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


def _composite_score_style(value: float) -> str:
    """Return a CSS background/foreground style for a composite score.

    Composite scores are on a 0-1 scale. Thresholds: green above 0.85,
    yellow above 0.70, red at or below 0.50, neutral otherwise.
    """
    if pd.isna(value):
        return ""
    if value > 0.85:
        bg, fg = "rgba(34,197,94,0.35)", "#f0fdf4"
    elif value > 0.70:
        bg, fg = "rgba(245,158,11,0.35)", "#fffbeb"
    elif value <= 0.50:
        bg, fg = "rgba(239,68,68,0.35)", "#fef2f2"
    else:
        bg, fg = "rgba(148,163,184,0.20)", "#e2e8f0"
    return f"background-color: {bg}; color: {fg}; font-weight: 600;"


def render_leaderboard(df: pd.DataFrame) -> None:
    """Render the model leaderboard via st.dataframe with a color-coded score column.

    Columns: Model, Provider, Avg Composite Score, Runs Count. Sorted by
    Avg Composite Score descending (the sort is already applied by
    ``get_leaderboard``).
    """
    if df.empty:
        st.info("No evaluation runs yet.")
        return

    display_df = df[["model_name", "provider", "composite", "runs_count"]].rename(
        columns={
            "model_name": "Model",
            "provider": "Provider",
            "composite": "Avg Composite Score",
            "runs_count": "Runs Count",
        }
    )
    styled = display_df.style.map(
        _composite_score_style, subset=["Avg Composite Score"]
    ).format({"Avg Composite Score": "{:.2f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)



def render_recent_runs(df: pd.DataFrame) -> None:
    """Render the 5 most recent evaluation runs via st.dataframe.

    Columns: Run ID (short), Model, Started, Composite Score, Status. The
    Status column is color-coded to match the platform's status palette.
    """
    if df.empty:
        st.info("No evaluation runs yet.")
        return

    display_df = pd.DataFrame(
        {
            "Run ID": df["run_id"].astype(str).str.slice(0, 8),
            "Model": df["model_name"],
            "Started": df["started_at"].apply(
                lambda ts: ts.strftime("%b %d, %H:%M") if hasattr(ts, "strftime") else "—"
            ),
            "Composite Score": df["composite_score"].apply(
                lambda v: f"{v:.1f}" if pd.notna(v) else "—"
            ),
            "Status": df["status"].astype(str).str.capitalize(),
        }
    )

    def _status_style(value: str) -> str:
        color, bg = STATUS_COLORS.get(str(value).lower(), ("#94a3b8", "rgba(148,163,184,0.12)"))
        return f"color: {color}; background-color: {bg}; font-weight: 600;"

    styled = display_df.style.map(_status_style, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def main() -> None:
    """Render the full dashboard page."""
    st.title("LLM Reliability Dashboard")
    st.caption("Live overview of model evaluation performance, regressions, and run history.")

    metrics = get_summary_metrics()
    if metrics.total_runs == 0:
        st.info("No evaluations yet. Go to the Evaluation page to run your first benchmark.")
        return

    render_regression_alerts(get_regression_alerts())

    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    render_metric_cards(metrics)

    leaderboard_df = get_leaderboard()

    st.markdown('<div class="section-title">Model Leaderboard</div>', unsafe_allow_html=True)
    render_leaderboard(leaderboard_df)

    col_radar, col_bar = st.columns(2)
    with col_radar:
        st.markdown('<div class="section-title">Dimension Comparison</div>', unsafe_allow_html=True)
        radar_df = get_radar_data()
        if not radar_df.empty:
            fig_radar = radar_chart(radar_df, title="Top 3 Models — 4 Dimensions")
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Not enough data yet for a dimension comparison.")
    with col_bar:
        st.markdown('<div class="section-title">Composite Ranking</div>', unsafe_allow_html=True)
        fig_bar = leaderboard_bar_chart(leaderboard_df, title="Composite Score by Model")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-title">Recent Evaluation Runs</div>', unsafe_allow_html=True)
    render_recent_runs(get_recent_runs())


def render() -> None:
    """Render the dashboard page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    main()


if __name__ == "__main__":
    render()