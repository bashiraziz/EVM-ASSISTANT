from typing import List, Dict

import streamlit as st


PROG_KEY = "__progress_items__"


def _ensure():
    if PROG_KEY not in st.session_state:
        st.session_state[PROG_KEY] = []  # list of {label,status}


def reset_progress():
    st.session_state[PROG_KEY] = []


def add_step(label: str, status: str = "running") -> int:
    _ensure()
    st.session_state[PROG_KEY].append({"label": label, "status": status})
    return len(st.session_state[PROG_KEY]) - 1


def update_step(index: int, status: str):
    _ensure()
    if 0 <= index < len(st.session_state[PROG_KEY]):
        st.session_state[PROG_KEY][index]["status"] = status


def render_sidebar():
    _ensure()
    items: List[Dict[str, str]] = st.session_state.get(PROG_KEY, [])
    if not items:
        return
    with st.sidebar:
        st.markdown("### Working")
        for it in items:
            label = it.get("label", "…")
            status = (it.get("status") or "").lower()
            prefix = "●" if status == "done" else ("○" if status == "running" else "×")
            st.write(f"{prefix} {label}")


__all__ = [
    "reset_progress",
    "add_step",
    "update_step",
    "render_sidebar",
]

