"""Evaluation page.

Lets the user configure and launch an evaluation run: provider, model,
prompt, benchmark, temperature, and max tokens. Running the evaluation is
fully mocked with a simulated progress bar — no provider is actually
called.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.mock.data_new import get_benchmarks, get_prompts, get_providers




def render_config_form() -> dict[str, object]:
    """Render the evaluation configuration controls and return the selections."""

    providers = get_providers()
    prompts = get_prompts()
    benchmarks = get_benchmarks()

    provider_names = [p.name for p in providers]

    col1, col2 = st.columns(2)
    with col1:
        provider_name = st.selectbox("Provider", options=provider_names)
        provider = next(p for p in providers if p.name == provider_name)
        model = st.selectbox("Model", options=provider.models)
        prompt_name = st.selectbox("Prompt", options=[p.name for p in prompts])
        prompt = next(p for p in prompts if p.name == prompt_name)
        st.caption(f"Selected prompt version: `{prompt.current_version}`")

    with col2:
        benchmark_name = st.selectbox("Benchmark", options=[b.name for b in benchmarks])
        benchmark = next(b for b in benchmarks if b.name == benchmark_name)
        st.caption(f"Domain: {benchmark.domain} · {benchmark.question_count} questions · v{benchmark.version}")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        max_tokens = st.number_input("Max tokens", min_value=16, max_value=32000, value=1024, step=16)

    return {
        "provider": provider_name,
        "model": model,
        "prompt": prompt_name,
        "prompt_version": prompt.current_version,
        "benchmark": benchmark_name,
        "benchmark_version": benchmark.version,
        "question_count": benchmark.question_count,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def run_mock_evaluation(config: dict[str, object]) -> None:
    """Simulate an evaluation run with a progress bar and status updates."""

    total_questions = int(config["question_count"])
    progress_bar = st.progress(0, text="Starting evaluation...")
    status_placeholder = st.empty()

    steps = min(total_questions, 40)  # cap animation steps for a snappy UI
    for step in range(1, steps + 1):
        pct = step / steps
        questions_done = int(pct * total_questions)
        progress_bar.progress(
            pct,
            text=f"Evaluating {questions_done}/{total_questions} questions...",
        )
        status_placeholder.caption(
            f"Provider: {config['provider']} · Model: {config['model']} · "
            f"Temp: {config['temperature']} · Max tokens: {config['max_tokens']}"
        )
        time.sleep(0.04)

    progress_bar.progress(1.0, text="Evaluation complete.")
    st.success(
        f"Mock run complete: {total_questions} questions evaluated on "
        f"{config['model']} using prompt `{config['prompt']}` "
        f"({config['prompt_version']}) against `{config['benchmark']}` "
        f"({config['benchmark_version']}). No real provider was called."
    )
    st.info("View results for this run on the **Results** page (mock data).")


def main() -> None:
    st.title("Run Evaluation")
    st.caption("Configure a dataset → prompt → model evaluation pipeline and launch a run.")

    with st.container(border=True):
        st.markdown("#### Configuration")
        config = render_config_form()

    st.write("")
    run_col, _ = st.columns([1, 4])
    with run_col:
        run_clicked = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    if run_clicked:
        with st.container(border=True):
            st.markdown("#### Progress")
            run_mock_evaluation(config)


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()