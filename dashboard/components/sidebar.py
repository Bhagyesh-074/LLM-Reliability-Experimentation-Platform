"""
dashboard/components/sidebar.py
--------------------------------
Sidebar navigation component for the LLM Reliability Platform.

Renders a styled navigation menu in the Streamlit sidebar and returns the
name of the page the user has selected.  All visual state lives in
``st.session_state`` so the selection survives widget re-runs.

Usage:
    from dashboard.components.sidebar import render_sidebar

    page = render_sidebar()
    # Route to the appropriate page module based on `page`.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Navigation definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NavItem:
    """A single entry in the sidebar navigation menu.

    Attributes:
        label: Human-readable page name shown in the UI.
        icon:  Emoji or Unicode glyph rendered beside the label.
        key:   Stable string key used in ``session_state`` and routing.
    """

    label: str
    icon: str
    key: str


NAV_ITEMS: tuple[NavItem, ...] = (
    NavItem("Dashboard",  "📊", "dashboard"),
    NavItem("Providers",  "🔌", "providers"),
    NavItem("Prompts",    "✏️",  "prompts"),
    NavItem("Benchmarks", "⏱️",  "benchmarks"),
    NavItem("Evaluation", "🧪", "evaluation"),
    NavItem("Results",    "📈", "results"),
    NavItem("Failures",   "🚨", "failures"),
    NavItem("Analytics",  "🔍", "analytics"),
    NavItem("Settings",   "⚙️",  "settings"),
)

_SESSION_KEY = "active_page"
_DEFAULT_PAGE = NAV_ITEMS[0].key


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Initialise session state keys used by the sidebar.

    Sets ``st.session_state[_SESSION_KEY]`` to the default page if it has
    not already been set by a previous render or a query-parameter redirect.

    Returns:
        None
    """
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = _DEFAULT_PAGE


def _nav_button(item: NavItem) -> bool:
    """Render a single navigation button and update session state on click.

    The button is styled as active (highlighted) when its key matches the
    current ``session_state`` selection.  Streamlit re-runs the script on
    click, so the caller only needs to read ``session_state`` after this
    function returns.

    Args:
        item: The ``NavItem`` to render.

    Returns:
        ``True`` if the button was clicked during this run, ``False``
        otherwise.
    """
    is_active = st.session_state.get(_SESSION_KEY) == item.key

    # Inject per-button styling via a unique CSS class written to the page.
    active_style = (
        "background-color: rgba(99,102,241,0.18); "
        "border-left: 3px solid #6366f1; "
        "font-weight: 600;"
    ) if is_active else (
        "background-color: transparent; "
        "border-left: 3px solid transparent;"
    )

    st.markdown(
        f"""
        <style>
        div[data-testid="stButton"] > button[kind="secondary"]#btn-{item.key} {{
            {active_style}
            width: 100%;
            text-align: left;
            padding: 0.45rem 0.75rem;
            border-radius: 6px;
            color: #f1f5f9;
            transition: background-color 0.15s ease;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    clicked: bool = st.button(
        f"{item.icon}  {item.label}",
        key=f"nav_{item.key}",
        use_container_width=True,
    )

    if clicked:
        st.session_state[_SESSION_KEY] = item.key
        logger.debug("Navigation: active_page={page}", page=item.key)

    return clicked


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    """Render the full sidebar and return the currently active page key.

    Draws:
    * Platform logo / title.
    * Divider.
    * Navigation buttons for every item in ``NAV_ITEMS``.
    * Footer with version string.

    The returned key can be used directly for routing in ``app.py``.

    Returns:
        The ``key`` attribute of the currently selected ``NavItem``,
        e.g. ``"dashboard"``, ``"providers"``, etc.
    """
    _init_session_state()

    with st.sidebar:
        # Brand header
        st.markdown(
            """
            <div style="padding: 1rem 0 0.5rem 0;">
                <span style="font-size:1.6rem; font-weight:700;
                             color:#6366f1; letter-spacing:-0.5px;">
                    ⚡ LLM Platform
                </span>
                <p style="font-size:0.7rem; color:#64748b;
                           margin:0.15rem 0 0 0; letter-spacing:0.05em;">
                    RELIABILITY &amp; EXPERIMENTATION
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Navigation items
        for item in NAV_ITEMS:
            _nav_button(item)

        st.divider()

        # Footer
        st.markdown(
            f"""
            <p style="font-size:0.65rem; color:#475569;
                       text-align:center; margin:0; padding:0.5rem 0;">
                v{settings.app.version} &nbsp;·&nbsp; {settings.app.environment}
            </p>
            """,
            unsafe_allow_html=True,
        )

    active: str = st.session_state.get(_SESSION_KEY, _DEFAULT_PAGE)
    return active