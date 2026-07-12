"""Mock data providers for the LLM Reliability & Experimentation Dashboard.

Every function in this module returns static, fully-typed mock data. Nothing
here performs a network or backend call — this module is the single source
of truth for all data consumed by the dashboard pages.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

RunStatus = Literal["passed", "failed", "running", "flagged"]


class ModelScore(BaseModel):
    """Evaluation scores for a single model on the leaderboard."""

    model_name: str
    provider: str
    accuracy: float = Field(ge=0, le=100, description="Higher is better")
    hallucination: float = Field(ge=0, le=100, description="Lower is better (shown as inverse rate)")
    instruction: float = Field(ge=0, le=100, description="Instruction-following score")
    safety: float = Field(ge=0, le=100, description="Safety/red-team score")
    composite: float = Field(ge=0, le=100, description="Weighted composite score")


class EvaluationRun(BaseModel):
    """A single evaluation run record."""

    run_id: str
    model_name: str
    suite: str
    status: RunStatus
    started_at: dt.datetime
    duration_sec: int = Field(ge=0)
    samples: int = Field(ge=0)


class RegressionAlert(BaseModel):
    """A detected regression alert, rendered as a banner on the dashboard."""

    detected: bool
    model_name: Optional[str] = None
    metric: Optional[str] = None
    delta_pct: Optional[float] = None
    run_id: Optional[str] = None
    message: Optional[str] = None


class SummaryMetrics(BaseModel):
    """Top-level KPI values shown in the dashboard metric cards."""

    total_runs: int
    avg_accuracy: float
    best_model: str
    active_providers: int


def get_leaderboard() -> pd.DataFrame:
    """Return the model leaderboard: 5 models scored on 5 dimensions."""
    scores: List[ModelScore] = [
        ModelScore(
            model_name="Claude Sonnet 5",
            provider="Anthropic",
            accuracy=95.6,
            hallucination=97.6,
            instruction=96.9,
            safety=98.1,
            composite=96.3,
        ),
        ModelScore(
            model_name="GPT-5.1",
            provider="OpenAI",
            accuracy=94.2,
            hallucination=96.9,
            instruction=95.8,
            safety=97.4,
            composite=95.1,
        ),
        ModelScore(
            model_name="Gemini 3 Pro",
            provider="Google",
            accuracy=92.8,
            hallucination=96.0,
            instruction=93.5,
            safety=95.9,
            composite=93.4,
        ),
        ModelScore(
            model_name="Llama 4 Maverick",
            provider="Meta",
            accuracy=89.1,
            hallucination=93.3,
            instruction=90.2,
            safety=92.0,
            composite=89.3,
        ),
        ModelScore(
            model_name="Mistral Large 3",
            provider="Mistral AI",
            accuracy=87.5,
            hallucination=92.1,
            instruction=88.4,
            safety=90.6,
            composite=87.6,
        ),
    ]
    df = pd.DataFrame([s.model_dump() for s in scores])
    return df.sort_values("composite", ascending=False).reset_index(drop=True)


def get_radar_data() -> pd.DataFrame:
    """Return long-form data (model, dimension, value) for the top 3 models.

    Note: "hallucination" here is expressed as an inverse rate (higher =
    fewer hallucinations) so that "higher is better" holds across all
    dimensions on the radar chart.
    """
    dimensions = ["Accuracy", "Low Hallucination", "Instruction Following", "Safety", "Consistency"]
    data: Dict[str, List[float]] = {
        "Claude Sonnet 5": [95.6, 97.6, 96.9, 98.1, 94.8],
        "GPT-5.1": [94.2, 96.9, 95.8, 97.4, 93.1],
        "Gemini 3 Pro": [92.8, 96.0, 93.5, 95.9, 91.4],
    }
    rows: List[Dict[str, Union[str, float]]] = []
    for model, values in data.items():
        for dimension, value in zip(dimensions, values):
            rows.append({"model": model, "dimension": dimension, "value": value})
    return pd.DataFrame(rows)


def get_recent_runs() -> pd.DataFrame:
    """Return the last 5 evaluation runs with status and timing info."""
    anchor = dt.datetime(2026, 7, 12, 9, 30)
    runs: List[EvaluationRun] = [
        EvaluationRun(
            run_id="run_10231",
            model_name="Claude Sonnet 5",
            suite="Hallucination Bench v3",
            status="passed",
            started_at=anchor - dt.timedelta(minutes=12),
            duration_sec=184,
            samples=500,
        ),
        EvaluationRun(
            run_id="run_10230",
            model_name="GPT-5.1",
            suite="Instruction Following v2",
            status="passed",
            started_at=anchor - dt.timedelta(minutes=47),
            duration_sec=210,
            samples=500,
        ),
        EvaluationRun(
            run_id="run_10229",
            model_name="Llama 4 Maverick",
            suite="Safety Redteam v5",
            status="flagged",
            started_at=anchor - dt.timedelta(hours=1, minutes=5),
            duration_sec=302,
            samples=350,
        ),
        EvaluationRun(
            run_id="run_10228",
            model_name="Gemini 3 Pro",
            suite="Accuracy QA v4",
            status="running",
            started_at=anchor - dt.timedelta(hours=1, minutes=40),
            duration_sec=95,
            samples=500,
        ),
        EvaluationRun(
            run_id="run_10227",
            model_name="Mistral Large 3",
            suite="Hallucination Bench v3",
            status="failed",
            started_at=anchor - dt.timedelta(hours=2, minutes=15),
            duration_sec=88,
            samples=500,
        ),
    ]
    return pd.DataFrame([r.model_dump() for r in runs])


def get_regression_alert() -> RegressionAlert:
    """Return a single mock regression alert (detected=True)."""
    return RegressionAlert(
        detected=True,
        model_name="Llama 4 Maverick",
        metric="Hallucination Rate",
        delta_pct=18.4,
        run_id="run_10229",
        message=(
            "Hallucination rate increased 18.4% versus the previous baseline "
            "on Safety Redteam v5. Investigate before promoting this run."
        ),
    )


def get_summary_metrics() -> SummaryMetrics:
    """Return top-level KPIs derived from the leaderboard for the metric cards."""
    leaderboard = get_leaderboard()
    return SummaryMetrics(
        total_runs=128,
        avg_accuracy=round(float(leaderboard["accuracy"].mean()), 1),
        best_model=str(leaderboard.iloc[0]["model_name"]),
        active_providers=int(leaderboard["provider"].nunique()),
    )