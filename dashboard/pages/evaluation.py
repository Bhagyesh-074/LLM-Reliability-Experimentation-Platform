"""Evaluation page.

Lets the user configure and launch a real evaluation run: provider,
model, prompt version, benchmark dataset version, temperature, and max
tokens. Launching the run drives the real ``EvaluationOrchestrator``
end to end and renders a live progress bar fed by the orchestrator's
``progress_callback``.

Data sources are all real backend services:
    * providers/models: ``ProviderRegistry``
    * prompts:          ``PromptService``
    * benchmarks:       ``BenchmarkService``

The run itself is executed synchronously on Streamlit's script thread
via ``asyncio.run(orchestrator.run())``; the progress callback fires on
that same thread, so updating the captured progress widgets from inside
it is safe.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

import asyncio

import streamlit as st

from core.evaluation.config import EvaluationConfig
from core.evaluation.orchestrator import EvaluationOrchestrator, EvaluationRun
from database.session import session_scope
from metrics.accuracy import AccuracyScorer
from metrics.base import Metric
from metrics.composite import CompositeScorer
from metrics.cost import CostScorer
from metrics.hallucination import HallucinationScorer
from metrics.instruction import InstructionScorer
from metrics.latency import LatencyScorer
from metrics.safety import SafetyScorer
from providers.base import ProviderError
from providers.registry import ProviderRegistry
from registry.benchmark_service import BenchmarkService
from registry.prompt_service import PromptService

#: Session-state key guarding against concurrent/re-entrant runs (used to
#: disable the Run button while an evaluation is in flight).
_RUNNING_KEY = "evaluation_running"

#: Session-state key holding the most recent progress payload emitted by
#: the orchestrator's callback.
_PROGRESS_KEY = "evaluation_progress"


@st.cache_resource(show_spinner=False)
def get_cached_scorers() -> Dict[str, Metric]:
    """Build and cache the six metric scorers for the lifetime of the process.

    ``AccuracyScorer`` and ``HallucinationScorer`` each load a real ML model
    (a sentence-transformers embedder and a transformers NLI pipeline,
    respectively) in their constructors. Without caching, every evaluation
    run would reload both models from scratch. ``@st.cache_resource`` keeps
    a single instance of each scorer alive across reruns and across
    evaluation runs within the same Streamlit process, so the models are
    loaded exactly once per session.

    Returns:
        A mapping of ``orchestrator.COMPONENT_KEYS`` -> ``Metric`` instance
        (``"accuracy"``, ``"hallucination"``, ``"instruction"``,
        ``"safety"``, ``"latency"``, ``"cost"``), ready to hand to
        ``EvaluationOrchestrator(scorers=...)``.
    """
    return {
        "accuracy": AccuracyScorer(),
        "hallucination": HallucinationScorer(),
        "instruction": InstructionScorer(),
        "safety": SafetyScorer(),
        "latency": LatencyScorer(),
        "cost": CostScorer(),
    }


@st.cache_resource(show_spinner=False)
def get_cached_composite_scorer() -> CompositeScorer:
    """Build and cache the ``CompositeScorer`` for the lifetime of the process.

    ``CompositeScorer`` is cheap to construct (no ML model, just a weights
    dict), but it is cached alongside the six component scorers so the
    dashboard has one single, consistent source for every scorer instance
    ``EvaluationOrchestrator`` needs.

    Returns:
        A ``CompositeScorer`` instance with default component weights.
    """
    return CompositeScorer()


def _load_prompt_options() -> List[Dict[str, Any]]:
    """Load selectable prompts, each resolved to its latest version id.

    ``PromptService.list_prompts`` returns summaries without the immutable
    ``version_id`` primary key the orchestrator needs, so each prompt's
    latest version is resolved via ``get_prompt`` here.

    Returns:
        One dict per prompt with keys ``label`` (display string),
        ``prompt_id``, ``version_id`` (latest version PK), and ``version``
        (latest version number). Prompts with no versions are skipped.
    """
    options: List[Dict[str, Any]] = []
    with session_scope() as session:
        service = PromptService(session)
        for summary in service.list_prompts():
            detail = service.get_prompt(summary.prompt_id)
            if not detail.versions:
                continue
            latest = detail.versions[-1]
            options.append(
                {
                    "label": f"{detail.name} (v{latest.version})",
                    "prompt_id": detail.prompt_id,
                    "version_id": latest.version_id,
                    "version": latest.version,
                }
            )
    return options


def _load_benchmark_options() -> List[Dict[str, Any]]:
    """Load selectable benchmarks, each resolved to its latest dataset version.

    ``BenchmarkService.list_benchmarks`` returns summaries without the
    ``dataset_version_id`` primary key, so each benchmark's newest dataset
    version is resolved via ``get_benchmark`` here.

    Returns:
        One dict per benchmark with keys ``label`` (display string),
        ``benchmark_id``, ``dataset_version_id`` (latest version PK),
        ``version``, and ``question_count``. Benchmarks with no dataset
        versions are skipped.
    """
    options: List[Dict[str, Any]] = []
    with session_scope() as session:
        service = BenchmarkService(session)
        for summary in service.list_benchmarks():
            detail = service.get_benchmark(summary.benchmark_id)
            if not detail.dataset_versions:
                continue
            latest = detail.dataset_versions[0]  # newest-created first
            domain = detail.domain or "—"
            options.append(
                {
                    "label": f"{detail.name} · {domain}",
                    "benchmark_id": detail.benchmark_id,
                    "dataset_version_id": latest.dataset_version_id,
                    "version": latest.version,
                    "question_count": latest.question_count,
                }
            )
    return options


def render_config_form() -> Optional[Dict[str, object]]:
    """Render the evaluation configuration controls and return the selections.

    Returns:
        A dict of the resolved selections needed to build an
        ``EvaluationConfig`` (provider/model names, prompt and dataset
        version ids, generation params, plus display metadata), or
        ``None`` if the form cannot be completed because prerequisite
        data (providers, prompts, or benchmarks) is missing. In the
        ``None`` case an inline warning has already been rendered.
    """
    providers = ProviderRegistry.list_providers()
    provider_names = [p["name"] for p in providers]

    prompt_options = _load_prompt_options()
    benchmark_options = _load_benchmark_options()

    if not prompt_options:
        st.warning(
            "No prompts with a saved version were found. Create one on the "
            "**Prompts** page before running an evaluation."
        )
        return None
    if not benchmark_options:
        st.warning(
            "No benchmarks with an uploaded dataset were found. Create one on "
            "the **Benchmarks** page before running an evaluation."
        )
        return None

    col1, col2 = st.columns(2)
    with col1:
        provider_name = st.selectbox("Provider", options=provider_names)
        provider_meta = next(p for p in providers if p["name"] == provider_name)

        models = ProviderRegistry.get_models(provider_name)
        if not models:
            st.warning(
                f"No models available for provider `{provider_name}`. "
                "For Ollama, ensure the daemon is running and models are pulled."
            )
            model = None
        else:
            model = st.selectbox("Model", options=models)

        if provider_meta["requires_api_key"] and not provider_meta["configured"]:
            st.caption(
                f"⚠️ `{provider_name}` needs `{provider_meta['env_var']}` to be set."
            )

        prompt_labels = [opt["label"] for opt in prompt_options]
        selected_prompt_label = st.selectbox("Prompt", options=prompt_labels)
        prompt = next(opt for opt in prompt_options if opt["label"] == selected_prompt_label)
        st.caption(f"Selected prompt version: `v{prompt['version']}`")

    with col2:
        benchmark_labels = [opt["label"] for opt in benchmark_options]
        selected_benchmark_label = st.selectbox("Benchmark", options=benchmark_labels)
        benchmark = next(
            opt for opt in benchmark_options if opt["label"] == selected_benchmark_label
        )
        st.caption(
            f"{benchmark['question_count']} questions · v{benchmark['version']}"
        )
        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.05
        )
        max_tokens = st.number_input(
            "Max tokens", min_value=16, max_value=32000, value=512, step=16
        )

    if model is None:
        return None

    return {
        "provider": provider_name,
        "model": model,
        "prompt_label": selected_prompt_label,
        "prompt_version_id": prompt["version_id"],
        "prompt_version": prompt["version"],
        "benchmark_label": selected_benchmark_label,
        "dataset_version_id": benchmark["dataset_version_id"],
        "benchmark_version": benchmark["version"],
        "question_count": benchmark["question_count"],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }


def _make_progress_callback(
    progress_bar: "st._DeltaGenerator",
    status_placeholder: "st._DeltaGenerator",
) -> Callable[[Dict[str, Any]], None]:
    """Build a sync progress callback that updates the captured widgets.

    The returned callable is invoked by the orchestrator once before the
    first row, once after each row, and once when the run finishes, with a
    ``{"current", "total", "status"}`` dict. It records the latest payload
    in session state and re-renders the progress bar and status line.

    The orchestrator swallows exceptions raised by the callback, so a
    Streamlit rendering hiccup here can never abort a run.

    Args:
        progress_bar: The ``st.progress`` widget to advance.
        status_placeholder: An ``st.empty`` placeholder for the status line.

    Returns:
        A synchronous ``progress_callback`` suitable for the orchestrator.
    """

    def _callback(payload: Dict[str, Any]) -> None:
        st.session_state[_PROGRESS_KEY] = payload
        current = int(payload.get("current", 0))
        total = int(payload.get("total", 0)) or 1
        status = str(payload.get("status", ""))

        pct = min(max(current / total, 0.0), 1.0)
        progress_bar.progress(
            pct, text=f"Evaluating question {current} of {total}..."
        )
        status_placeholder.caption(f"Question {current}/{total} — status: {status}")

    return _callback


def run_real_evaluation(config_selections: Dict[str, object]) -> None:
    """Execute a real evaluation run and render its progress and outcome.

    Builds an ``EvaluationConfig`` from the resolved form selections,
    drives ``EvaluationOrchestrator.run`` via ``asyncio.run`` with a live
    progress callback, and renders a success or failure summary. All work
    happens inside a single ``session_scope`` so the run, its per-row
    results, and its aggregate metrics are committed atomically.

    Args:
        config_selections: The dict returned by ``render_config_form``.
    """
    progress_bar = st.progress(0, text="Starting evaluation...")
    status_placeholder = st.empty()
    callback = _make_progress_callback(progress_bar, status_placeholder)

    try:
        eval_config = EvaluationConfig(
            provider_name=str(config_selections["provider"]),
            model_name=str(config_selections["model"]),
            prompt_version_id=str(config_selections["prompt_version_id"]),
            dataset_version_id=str(config_selections["dataset_version_id"]),
            temperature=float(config_selections["temperature"]),
            max_tokens=int(config_selections["max_tokens"]),
        )
    except Exception as exc:  # noqa: BLE001 - surface validation errors to the user
        st.error(f"Invalid configuration: {exc}")
        return

    with st.spinner("Loading models (first run only)..."):
        scorers = get_cached_scorers()
        composite_scorer = get_cached_composite_scorer()

    try:
        with session_scope() as session:
            orchestrator = EvaluationOrchestrator(
                config=eval_config,
                session=session,
                scorers=scorers,
                composite_scorer=composite_scorer,
                progress_callback=callback,
            )
            run: EvaluationRun = asyncio.run(orchestrator.run())
    except ProviderError as exc:
        progress_bar.empty()
        st.error(
            f"Provider `{config_selections['provider']}` is not available: {exc}"
        )
        return
    except FileNotFoundError as exc:
        progress_bar.empty()
        st.error(f"Dataset file could not be read: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
        progress_bar.empty()
        st.error(f"Evaluation failed to run: {exc}")
        return

    _render_run_outcome(run, config_selections)


def _render_run_outcome(
    run: EvaluationRun, config_selections: Dict[str, object]
) -> None:
    """Render the success/error summary for a completed orchestrator run.

    Distinguishes the three orchestrator statuses: a fully clean run, a
    run that completed with some (<=50%) failed rows, and a failed run
    (>50% of rows failed). Failed rows are listed with their index and
    error, per the error-handling contract.

    Args:
        run: The ``EvaluationRun`` DTO returned by the orchestrator.
        config_selections: The original form selections, for the summary.
    """
    composite = run.composite_score
    composite_str = f"{composite:.4f}" if composite is not None else "n/a"

    failed_rows = [r for r in run.results if not r.success]

    if run.status == "completed":
        st.success(
            f"Run complete — composite score **{composite_str}** "
            f"across {len(run.results)} question(s).\n\n"
            f"Run ID: `{run.run_id}`"
        )
    elif run.status == "completed_with_errors":
        st.warning(
            f"Run completed with errors — composite score **{composite_str}** "
            f"across {len(run.results) - len(failed_rows)} successful "
            f"question(s) ({len(failed_rows)} failed).\n\n"
            f"Run ID: `{run.run_id}`"
        )
    else:  # "failed"
        st.error(
            f"Run failed — more than half of the {len(run.results)} question(s) "
            f"failed ({len(failed_rows)} failed). Partial results were still "
            f"saved.\n\n"
            f"Run ID: `{run.run_id}`"
        )

    if failed_rows:
        with st.expander(f"Failed rows ({len(failed_rows)})", expanded=run.status == "failed"):
            for row in failed_rows:
                st.markdown(f"- **Row {row.row_index}** — {row.error or 'unknown error'}")

    st.info("View full results for this run on the **Results** page.")


def main() -> None:
    """Render the Evaluation page and handle run launches."""
    st.title("Run Evaluation")
    st.caption(
        "Configure a dataset → prompt → model evaluation pipeline and launch a run."
    )

    if _RUNNING_KEY not in st.session_state:
        st.session_state[_RUNNING_KEY] = False

    with st.container(border=True):
        st.markdown("#### Configuration")
        config_selections = render_config_form()

    st.write("")
    run_col, _ = st.columns([1, 4])
    with run_col:
        run_clicked = st.button(
            "▶ Run Evaluation",
            type="primary",
            use_container_width=True,
            disabled=config_selections is None or st.session_state[_RUNNING_KEY],
        )

    if run_clicked and config_selections is not None:
        st.session_state[_RUNNING_KEY] = True
        try:
            with st.container(border=True):
                st.markdown("#### Progress")
                run_real_evaluation(config_selections)
        finally:
            st.session_state[_RUNNING_KEY] = False


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()