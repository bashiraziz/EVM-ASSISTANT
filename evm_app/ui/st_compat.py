"""Streamlit compatibility helpers.

Allows other modules to check if code is running inside a real
Streamlit ScriptRunContext. In Gradio or plain Python, importing
Streamlit may work but there is no ScriptRunContext, which can
trigger warnings when touching session_state.
"""
from __future__ import annotations

from typing import Any

try:
    import streamlit as st  # type: ignore
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        IN_STREAMLIT: bool = get_script_run_ctx() is not None
    except Exception:
        IN_STREAMLIT = False
except Exception:
    st = None  # type: ignore
    IN_STREAMLIT = False


def in_streamlit() -> bool:
    """Return True if a real Streamlit ScriptRunContext exists."""
    return IN_STREAMLIT


__all__ = ["IN_STREAMLIT", "in_streamlit", "st"]

