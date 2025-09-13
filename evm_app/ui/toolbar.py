from typing import Any, Dict, List, Optional

import streamlit as st

from ..config import PROVIDER, get_active_default_model


def render_results_toolbar(items: List[Dict[str, Any]] | None, totals: Dict[str, Any] | None):
    """Render a compact toolbar above results with quick context badges.

    Shows project count, As Of date (if any), provider and active model. Keeps a
    low visual profile to avoid cluttering the page.
    """
    n = len(items or [])
    as_of = (totals or {}).get("AsOf")

    st.markdown(
        """
        <style>
        .toolbar-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:6px 0 8px}
        .tb-badge{font-size:11px;padding:4px 8px;border:1px solid #2a2f3a;border-radius:999px;background:#0b1220}
        .tb-title{font-weight:700;margin-right:6px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    parts = [
        "<div class='toolbar-row'>",
        "<div class='tb-title'>Results</div>",
        f"<span class='tb-badge'>Projects: {n}</span>",
    ]
    if as_of:
        parts.append(f"<span class='tb-badge'>As Of: {as_of}</span>")
    parts.append(f"<span class='tb-badge'>Provider: {PROVIDER}</span>")
    parts.append(f"<span class='tb-badge'>Model: {get_active_default_model()}</span>")
    parts.append("</div>")

    st.markdown("".join(parts), unsafe_allow_html=True)


__all__ = ["render_results_toolbar"]

