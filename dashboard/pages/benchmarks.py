"""Benchmark Registry page.

Displays all versioned benchmark datasets from the platform database and
lets the user register a new benchmark (or add a new version to an
existing one) by uploading a CSV of questions/answers. Uploaded files
are schema-validated immediately; nothing is written to the database
unless validation passes and the user confirms.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from database.session import session_scope
from registry.benchmark_service import (
    BenchmarkService,
    BenchmarkSummary,
    DatasetUploadError,
)
from registry.dataset_validator import ValidationResult, validate

_NEW_BENCHMARK_SENTINEL = "+ Create new benchmark"

_DOMAIN_OPTIONS = [
    "Medical",
    "Legal",
    "Finance",
    "Coding",
    "Math",
    "Safety",
    "Prompt Injection",
    "Long Context",
    "Summarization",
]


def _load_benchmarks() -> list[BenchmarkSummary]:
    """Fetch all benchmarks with a summary of their latest dataset version."""
    with session_scope() as session:
        service = BenchmarkService(session)
        return service.list_benchmarks()


def benchmarks_to_dataframe(benchmarks: list[BenchmarkSummary]) -> pd.DataFrame:
    """Convert benchmark summaries into a flat DataFrame for table display.

    Only columns backed by real schema fields are shown (Name, Domain,
    Version, Questions) — the schema has no creator/difficulty/tags
    columns on Benchmark or DatasetVersion.
    """
    return pd.DataFrame(
        [
            {
                "Name": b.name,
                "Domain": b.domain or "—",
                "Version": b.latest_version or "—",
                "Questions": b.latest_question_count or 0,
            }
            for b in benchmarks
        ]
    )


def _render_validation_report(result: ValidationResult) -> None:
    """Render a pass/fail validation report with specific, per-item error messages."""
    if result.is_valid:
        st.success("Validation passed: all required columns present, no row errors.")
        return

    st.error("Validation failed — nothing was saved.")
    for message in result.error_summary:
        st.markdown(f"- {message}")


def _render_danger_zone(selected: BenchmarkSummary) -> None:
    """Render a confirm-gated control to permanently delete a benchmark."""
    with st.expander(f"⚠️ Danger zone — delete '{selected.name}'"):
        st.caption(
            f"Permanently deletes '{selected.name}' and all "
            f"{selected.version_count} dataset version(s), including their stored "
            "files. This cannot be undone."
        )
        confirm_key = f"confirm_delete_{selected.benchmark_id}"
        confirmed = st.checkbox("I understand this is permanent", key=confirm_key)
        if st.button(
            "Delete benchmark",
            type="secondary",
            disabled=not confirmed,
            key=f"delete_btn_{selected.benchmark_id}",
        ):
            with session_scope() as session:
                service = BenchmarkService(session)
                service.delete_benchmark(selected.benchmark_id)
            st.success(f"Deleted '{selected.name}'.")
            st.rerun()


def render_upload_section(benchmarks: list[BenchmarkSummary]) -> None:
    st.markdown("### Upload a new benchmark")
    st.caption(
        "Upload a CSV of questions/answers. It's validated immediately; "
        "nothing is saved until you click **Register benchmark**."
    )

    existing_names = [b.name for b in benchmarks]
    name_choice = st.selectbox(
        "Benchmark",
        options=[_NEW_BENCHMARK_SENTINEL, *existing_names],
    )
    is_new = name_choice == _NEW_BENCHMARK_SENTINEL

    col_file, col_meta = st.columns([2, 1])
    with col_meta:
        if is_new:
            new_name = st.text_input("New benchmark name")
            domain = st.selectbox("Domain", options=_DOMAIN_OPTIONS)
            description = st.text_area("Description", height=80)
        else:
            new_name = name_choice
            selected = next(b for b in benchmarks if b.name == name_choice)
            domain = selected.domain or ""
            description = selected.description or ""
            st.text_input("Domain", value=domain or "—", disabled=True)
        version = st.text_input("Version tag", value="v1.0")
    with col_file:
        uploaded = st.file_uploader("Benchmark CSV", type=["csv"])

    if not is_new:
        _render_danger_zone(selected)

    if uploaded is None:
        return

    try:
        preview_df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
        st.error(f"Could not parse CSV: {exc}")
        return

    validation_result = validate(preview_df)
    st.markdown("#### Validation report")
    _render_validation_report(validation_result)

    if validation_result.is_valid:
        st.caption(f"Parsed {len(preview_df)} rows. Showing first 5 below.")
        st.dataframe(preview_df.head(5), use_container_width=True, hide_index=True)

    register_col, _ = st.columns([1, 3])
    with register_col:
        register_disabled = not validation_result.is_valid or (is_new and not new_name.strip())
        if st.button(
            "Register benchmark",
            type="primary",
            use_container_width=True,
            disabled=register_disabled,
        ):
            uploaded.seek(0)
            try:
                with session_scope() as session:
                    service = BenchmarkService(session)
                    if is_new:
                        benchmark = service.create_benchmark(
                            name=new_name.strip(),
                            domain=domain,
                            description=description or None,
                        )
                        benchmark_id = benchmark.benchmark_id
                    else:
                        benchmark_id = next(
                            b.benchmark_id for b in benchmarks if b.name == name_choice
                        )
                    service.upload_dataset(
                        benchmark_id=benchmark_id,
                        file=uploaded,
                        version_label=version,
                    )
            except DatasetUploadError as exc:
                st.error(
                    "Registration failed: " + "; ".join(exc.validation_result.error_summary)
                )
                return
            except LookupError as exc:
                st.error(str(exc))
                return

            st.success(
                f"Registered {len(preview_df)} questions as "
                f"{new_name if is_new else name_choice} {version}."
            )
            st.rerun()


def main() -> None:
    st.title("Benchmark Registry")
    st.caption("Standardized, versioned datasets used to evaluate model performance across domains.")

    benchmarks = _load_benchmarks()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total benchmarks", len(benchmarks))
    m2.metric("Total questions", sum(b.latest_question_count or 0 for b in benchmarks))
    m3.metric("Domains covered", len({b.domain for b in benchmarks if b.domain}))

    st.divider()
    st.markdown("### All benchmarks")

    domain_options = sorted({b.domain for b in benchmarks if b.domain})
    domain_filter = st.multiselect(
        "Filter by domain",
        options=domain_options,
        default=domain_options,
    )
    filtered = [b for b in benchmarks if b.domain in domain_filter]

    st.dataframe(
        benchmarks_to_dataframe(filtered),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    render_upload_section(benchmarks)


def render() -> None:
    """Render the dashboard page."""
    main()


if __name__ == "__main__":
    render()