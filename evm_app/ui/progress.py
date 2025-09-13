from typing import List, Dict

import streamlit as st


PROG_KEY = "__progress_items__"
_CSS_KEY = "__progress_css_injected__"
_PH_KEY = "__progress_placeholder__"
_BADGES_KEY = "__sidebar_badges__"


def _ensure():
    if PROG_KEY not in st.session_state:
        st.session_state[PROG_KEY] = []  # list of {label,status}


def reset_progress():
    st.session_state[PROG_KEY] = []


def add_step(label: str, status: str = "running") -> int:
    _ensure()
    st.session_state[PROG_KEY].append({"label": label, "status": status})
    return len(st.session_state[PROG_KEY]) - 1


def update_step(index: int, status: str | None = None, label: str | None = None):
    _ensure()
    if 0 <= index < len(st.session_state[PROG_KEY]):
        if status is not None:
            st.session_state[PROG_KEY][index]["status"] = status
        if label is not None:
            st.session_state[PROG_KEY][index]["label"] = label


def _normalize(text: str) -> str:
    # Normalize different dash characters and excess whitespace for robust matching
    if not isinstance(text, str):
        text = str(text)
    for ch in ("—", "–", "−"):
        text = text.replace(ch, "-")
    text = " ".join(text.split())
    return text.strip()


def remove_steps_by_prefix(prefix: str):
    """Remove all progress steps whose label starts with prefix.

    Matching is dash/whitespace tolerant to avoid Unicode/spacing mismatches.
    """
    _ensure()
    items = st.session_state.get(PROG_KEY, [])
    norm_prefix = _normalize(prefix)
    kept = []
    for it in items:
        lbl = _normalize(it.get("label", ""))
        if not lbl.startswith(norm_prefix):
            kept.append(it)
    st.session_state[PROG_KEY] = kept


def prune_completed(max_items: int = 4):
    """Keep the sidebar tidy by trimming older completed entries.

    - Always keep all running/active items.
    - Among completed items, keep the most recent ones so that
      total items <= max_items.
    """
    _ensure()
    items = st.session_state.get(PROG_KEY, [])
    running = [it for it in items if (it.get("status", "").lower() == "running")]
    completed = [it for it in items if (it.get("status", "").lower() == "done")]
    remaining_slots = max(0, max_items - len(running))
    completed_kept = completed[-remaining_slots:] if remaining_slots < len(completed) else completed
    st.session_state[PROG_KEY] = running + completed_kept


def set_sidebar_badges(provider: str, default_model: str, summary_model: str):
    """Set compact sidebar badges that render above the progress list.

    Keep data in session_state to avoid import cycles; call before render_sidebar().
    """
    st.session_state[_BADGES_KEY] = {
        "provider": provider,
        "default_model": default_model,
        "summary_model": summary_model,
    }


def render_sidebar():
    _ensure()
    items: List[Dict[str, str]] = st.session_state.get(PROG_KEY, [])
    if not items:
        ph = st.session_state.get(_PH_KEY)
        if ph is not None:
            try:
                ph.empty()
            except Exception:
                pass
        return
    ph = st.session_state.get(_PH_KEY)
    if ph is None:
        ph = st.sidebar.empty()
        st.session_state[_PH_KEY] = ph
    else:
        try:
            ph.empty()
        except Exception:
            pass
    with ph.container():
        # Top badges (provider + models)
        badges = st.session_state.get(_BADGES_KEY)
        if badges:
            prov = badges.get("provider", "-")
            dmdl = badges.get("default_model", "-")
            smdl = badges.get("summary_model", "-")
            st.markdown(
                (
                    "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px'>"
                    f"<span style='font-size:11px;padding:3px 6px;border:1px solid #2a2f3a;border-radius:999px;background:#0b1220'>Provider: {prov}</span>"
                    f"<span style='font-size:11px;padding:3px 6px;border:1px solid #2a2f3a;border-radius:999px;background:#0b1220'>Model: {dmdl}</span>"
                    f"<span style='font-size:11px;padding:3px 6px;border:1px solid #2a2f3a;border-radius:999px;background:#0b1220'>Summary: {smdl}</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        st.markdown("### Working")
        if not st.session_state.get(_CSS_KEY):
            st.markdown(
                """
                <style>
                .agent-row{display:flex;align-items:center;gap:8px;margin:2px 0;}
                .agent-label{flex:1;}
                .pulse-dot{width:8px;height:8px;border-radius:999px;background:var(--acc1,#0b6cf0);
                           box-shadow:0 0 0 0 rgba(11,108,240,.6);animation:pulse 1.5s infinite;}
                @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(11,108,240,.6);}70%{box-shadow:0 0 0 9px rgba(11,108,240,0);}100%{box-shadow:0 0 0 0 rgba(11,108,240,0);}}
                .done-dot{color:#16a34a;}
                .done-dot, .done-dot * {color:#16a34a !important}
                .fail-dot{color:#dc2626;}
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.session_state[_CSS_KEY] = True
        for it in items:
            raw_label = it.get("label", "...")
            label = raw_label or "..."
            status = (it.get("status") or "").lower()
            if status == "done":
                if not label.rstrip().endswith("Done"):
                    label = f"{label} &mdash; Done"
                # Use inline SVG for a reliable green check across themes
                html = (
                    "<div class='agent-row'>"
                    "<span aria-label='done'>"
                    "<svg width='12' height='12' viewBox='0 0 16 16' xmlns='http://www.w3.org/2000/svg'>"
                    "<path fill='#16a34a' d='M6.173 12.414L2.222 8.464l1.414-1.414 2.537 2.536 6.192-6.192 1.414 1.415z'/></svg>"
                    "</span>"
                    f"<span class='agent-label'>{label}</span>"
                    "</div>"
                )
            elif status == "running":
                html = (
                    "<div class='agent-row'>"
                    "<span class='pulse-dot' aria-label='running'></span>"
                    f"<span class='agent-label'>{label}</span>"
                    "</div>"
                )
            else:
                html = (
                    "<div class='agent-row'>"
                    "<span class='fail-dot' aria-label='failed'>&#10005;</span>"
                    f"<span class='agent-label'>{label}</span>"
                    "</div>"
                )
            st.markdown(html, unsafe_allow_html=True)


__all__ = [
    "reset_progress",
    "add_step",
    "update_step",
    "remove_steps_by_prefix",
    "prune_completed",
    "render_sidebar",
]
