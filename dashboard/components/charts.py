"""Reusable Plotly chart components, styled for the dark dashboard theme.

All charts use `plotly_dark` as a base template plus explicit color overrides
so they render consistently against a #0f172a background with a #6366f1
accent, regardless of the caller's Streamlit theme settings.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

BACKGROUND = "#0f172a"
SURFACE = "#111827"
ACCENT = "#6366f1"
GRID = "#1e293b"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"

PALETTE: List[str] = ["#6366f1", "#22d3ee", "#f472b6", "#facc15", "#34d399"]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a '#rrggbb' hex color to an 'rgba(r,g,b,a)' string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _apply_dark_layout(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    """Apply shared dark-theme layout settings to a figure in place."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT, family="Inter, -apple-system, Segoe UI, sans-serif", size=13),
        title=dict(text=title, font=dict(size=15, color=TEXT)) if title else None,
        margin=dict(l=40, r=30, t=50 if title else 20, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11), orientation="h", y=-0.15),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(color=TEXT), bordercolor=GRID),
    )
    return fig


def radar_chart(df: pd.DataFrame, title: str = "Model Comparison") -> go.Figure:
    """Build a radar chart comparing models across evaluation dimensions.

    Args:
        df: Long-form DataFrame with columns ["model", "dimension", "value"].
            Values are expected on a 0-100 scale, higher is always better.
        title: Chart title.

    Returns:
        A dark-themed Plotly Figure ready for st.plotly_chart.
    """
    fig = go.Figure()
    dimensions = list(df["dimension"].unique())
    models = list(df["model"].unique())

    for i, model in enumerate(models):
        model_df = df[df["model"] == model].set_index("dimension").reindex(dimensions).reset_index()
        color = PALETTE[i % len(PALETTE)]
        values = model_df["value"].tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=dimensions + [dimensions[0]],
                fill="toself",
                name=model,
                line=dict(color=color, width=2),
                fillcolor=_hex_to_rgba(color, 0.18),
                marker=dict(size=5, color=color),
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor=SURFACE,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=GRID,
                linecolor=GRID,
                tickfont=dict(color=MUTED, size=10),
            ),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=TEXT, size=12)),
        ),
        showlegend=True,
    )
    return _apply_dark_layout(fig, title)


def leaderboard_bar_chart(
    df: pd.DataFrame, metric: str = "composite", title: str = "Composite Score by Model"
) -> go.Figure:
    """Build a horizontal bar chart ranking models by a given metric.

    Args:
        df: DataFrame with a "model_name" column and the given metric column.
        metric: Column name to rank/plot, expected on a 0-100 scale.
        title: Chart title.

    Returns:
        A dark-themed Plotly Figure ready for st.plotly_chart.
    """
    sorted_df = df.sort_values(metric, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=sorted_df[metric],
            y=sorted_df["model_name"],
            orientation="h",
            marker=dict(color=ACCENT, line=dict(width=0)),
            text=sorted_df[metric].round(1),
            textposition="outside",
            textfont=dict(color=TEXT, size=12),
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 105], gridcolor=GRID, zerolinecolor=GRID, title=None)
    fig.update_yaxes(gridcolor=GRID, title=None)
    return _apply_dark_layout(fig, title)