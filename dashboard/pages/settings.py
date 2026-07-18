"""Settings page.

Lets the user configure platform-wide defaults: the MLflow tracking URI,
the default sampling temperature for new evaluation runs, and the
regression alert threshold used by the dashboard. Settings are held in
Streamlit session state (mock persistence) and are not written to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.mock.data_new import Settings, get_default_settings




def get_current_settings() -> Settings:
    """Return settings from session state, seeding from defaults on first load."""

    if "settings" not in st.session_state:
        st.session_state["settings"] = get_default_settings()
    return st.session_state["settings"]


def main() -> None:
    st.title("Settings")
    st.caption("Platform-wide defaults for experiment tracking and evaluation runs.")

    current = get_current_settings()

    if st.session_state.pop("_reset_pending", False):
        defaults = get_default_settings()
        st.session_state["settings"] = defaults
        st.session_state["mlflow_uri"] = defaults.mlflow_tracking_uri
        st.session_state["default_temperature"] = defaults.default_temperature
        st.session_state["regression_threshold"] = defaults.regression_alert_threshold
        current = defaults

    with st.container(border=True):
        st.markdown("#### Experiment tracking")
        mlflow_uri = st.text_input(
            "MLflow tracking URI",
            value=current.mlflow_tracking_uri,
            help="Where evaluation runs, params, and artifacts are logged.",
            key="mlflow_uri",
        )

    with st.container(border=True):
        st.markdown("#### Evaluation defaults")
        default_temperature = st.slider(
            "Default temperature",
            min_value=0.0,
            max_value=1.0,
            value=current.default_temperature,
            step=0.05,
            help="Applied to new evaluation runs unless explicitly overridden.",
            key="default_temperature",
        )

    with st.container(border=True):
        st.markdown("#### Regression detection")
        regression_threshold = st.slider(
            "Regression alert threshold (% drop)",
            min_value=0.5,
            max_value=25.0,
            value=current.regression_alert_threshold,
            step=0.5,
            help="A metric drop larger than this percentage triggers a regression alert.",
            key="regression_threshold",
        )

    st.write("")
    save_col, reset_col, _ = st.columns([1, 1, 3])
    with save_col:
        if st.button("💾 Save settings", type="primary", width="stretch"):
            st.session_state["settings"] = Settings(
                mlflow_tracking_uri=mlflow_uri,
                default_temperature=default_temperature,
                regression_alert_threshold=regression_threshold,
            )
            st.success("Settings saved for this session.")
    with reset_col:
        if st.button("↺ Reset to defaults", width="stretch"):
            st.session_state["_reset_pending"] = True
            st.rerun()

    st.divider()
    st.markdown("#### Current effective settings")
    saved = get_current_settings()
    st.json(
        {
            "mlflow_tracking_uri": saved.mlflow_tracking_uri,
            "default_temperature": saved.default_temperature,
            "regression_alert_threshold": saved.regression_alert_threshold,
        }
    )


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()