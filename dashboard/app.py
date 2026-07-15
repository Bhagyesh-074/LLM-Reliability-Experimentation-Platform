"""
dashboard/app.py
----------------
Streamlit entry point for the LLM Reliability & Experimentation Platform.

Run with:
    streamlit run dashboard/app.py

Responsibilities:
* Bootstrap logging and settings exactly once per process.
* Configure the Streamlit page (title, icon, layout).
* Render the sidebar navigation component.
* Dispatch to the correct page module based on the user's selection.

Page modules are expected to live under ``dashboard/pages/`` and expose a
single ``render() -> None`` function.  They are imported lazily so that
importing this entry-point module does not pull in heavy dependencies unless
the corresponding page is actually visited.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when the file is executed directly
# (e.g. `streamlit run dashboard/app.py` from the repo root).
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import settings  # noqa: E402  (after sys.path fix)
from core.logger import setup_logging  # noqa: E402
from dashboard.components.sidebar import render_sidebar  # noqa: E402

# Must be the very first Streamlit command — module-level guarantees this.
st.set_page_config(
    page_title=settings.dashboard.page_title,
    page_icon=settings.dashboard.page_icon,
    layout=settings.dashboard.layout,  # type: ignore[arg-type]
    initial_sidebar_state=settings.dashboard.initial_sidebar_state,  # type: ignore[arg-type]
)

# ---------------------------------------------------------------------------
# Page → module mapping
# ---------------------------------------------------------------------------

# Keys must match the ``NavItem.key`` values defined in sidebar.py.
# Values are dotted module paths relative to the project root.
# Each module must expose a ``render() -> None`` callable.
_PAGE_MODULES: dict[str, str] = {
    "dashboard":  "dashboard.pages.dashboard",
    "providers":  "dashboard.pages.providers",
    "prompts":    "dashboard.pages.prompts",
    "benchmarks": "dashboard.pages.benchmarks",
    "evaluation": "dashboard.pages.evaluation",
    "results":    "dashboard.pages.results",
    "failures":   "dashboard.pages.failures",
    "analytics":  "dashboard.pages.analytics",
    "settings":   "dashboard.pages.settings",
}


# ---------------------------------------------------------------------------
# Bootstrap (runs once per Streamlit process, not on every re-run)
# ---------------------------------------------------------------------------


@st.cache_resource
def _bootstrap() -> None:
    """Initialise logging and any other one-time platform setup.

    ``st.cache_resource`` ensures this runs only once for the lifetime of
    the Streamlit server process, not on every script re-run triggered by
    widget interaction.

    Returns:
        None
    """
    setup_logging()


# ---------------------------------------------------------------------------
# Page dispatch
# ---------------------------------------------------------------------------


def _load_page_renderer(module_path: str) -> Callable[[], None]:
    """Dynamically import a page module and return its ``render`` function.

    Args:
        module_path: Dotted Python module path, e.g.
            ``"dashboard.pages.providers"``.

    Returns:
        The ``render`` callable from the imported module.

    Raises:
        ModuleNotFoundError: If the module does not exist yet.
        AttributeError: If the module exists but has no ``render`` function.
    """
    module = importlib.import_module(module_path)
    return module.render  # type: ignore[attr-defined]


def _render_coming_soon(page_key: str) -> None:
    """Placeholder renderer shown while a page module is not yet implemented.

    Args:
        page_key: The navigation key for the missing page.

    Returns:
        None
    """
    st.title(page_key.capitalize())
    st.info(
        f"The **{page_key}** page module has not been created yet.  "
        f"Add it at `dashboard/pages/{page_key}.py` and expose a "
        "`render() -> None` function.",
        icon="🚧",
    )


def _dispatch(active_page: str) -> None:
    """Route to the appropriate page renderer for ``active_page``.

    Attempts a lazy import of the page module.  Falls back gracefully to
    ``_render_coming_soon`` if the module does not exist yet, so the
    application remains navigable during incremental development.

    Args:
        active_page: The currently selected page key from the sidebar.

    Returns:
        None
    """
    module_path = _PAGE_MODULES.get(active_page)

    if module_path is None:
        st.error(f"Unknown page key: `{active_page}`")
        return

    try:
        renderer = _load_page_renderer(module_path)
        renderer()
    except ModuleNotFoundError:
        _render_coming_soon(active_page)
    except AttributeError:
        st.error(
            f"Module `{module_path}` exists but does not expose a "
            "`render()` function."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Render navigation and dispatch routing to the active page.

    This is the sole entry point called by Streamlit.  It must remain free of
    any blocking I/O so the UI stays responsive while page modules load.

    Returns:
        None
    """
    # One-time initialisation (logging, etc.)
    _bootstrap()

    # Render sidebar and obtain the active page key.
    active_page: str = render_sidebar()

    # Route to the selected page.
    _dispatch(active_page)


if __name__ == "__main__":
    main()
