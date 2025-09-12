from typing import Any, Dict, Optional

import streamlit as st

# Detect if running under a real Streamlit ScriptRunContext; in bare mode this is None
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
    _HAS_ST_CTX = get_script_run_ctx() is not None
except Exception:
    _HAS_ST_CTX = False


TRACE_KEY = "trace"
TRACE_PH_KEY = "__trace_ph__"


def _ensure_trace():
    if not _HAS_ST_CTX:
        return
    if TRACE_KEY not in st.session_state:
        st.session_state[TRACE_KEY] = []


def log_event(kind: str, title: str, details: Optional[Dict[str, Any]] = None):
    if not _HAS_ST_CTX:
        return
    _ensure_trace()
    st.session_state[TRACE_KEY].append({
        "kind": kind,
        "title": title,
        "details": details or {},
    })
    _render_trace_into_placeholder()


def clear_trace():
    if not _HAS_ST_CTX:
        return
    st.session_state[TRACE_KEY] = []
    _render_trace_into_placeholder()


def render_trace():
    if not _HAS_ST_CTX:
        return
    st.markdown("### Run Trace")
    if not st.session_state.get(TRACE_KEY):
        st.info("No trace available.")
        return

    for ev in st.session_state[TRACE_KEY]:
        kind = ev["kind"]
        title = ev["title"]
        details = ev["details"]

        if kind == "handoff":
            st.markdown(
                f"""
<div class='trace-card' style=\"padding:12px;border-radius:10px;\">
  <div style=\"font-weight:700;color:#fff;\">{title}</div>
  <div style=\"color:#cfe3ff;margin-top:6px;\">Reason: {details.get('reason','-')}</div>
  <div style=\"color:#cfe3ff;\">Context: {details.get('context','-')}</div>
  <div style=\"color:#9ad0ff;\">From: {details.get('from','-')} | To: {details.get('to','-')}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

        elif kind == "tool_call":
            by = details.get("by", "-") if isinstance(details, dict) else "-"
            st.write(f"Tool called: {title} (by: {by})")

        elif kind == "tool_done":
            by = details.get("by", "-") if isinstance(details, dict) else "-"
            st.write(f"Tool finished: {title} (by: {by})")

        elif kind == "agent_result":
            with st.expander(f"{title}", expanded=False):
                st.write(details.get("text", ""))


def _render_trace_into_placeholder():
    if not _HAS_ST_CTX:
        return
    ph = st.session_state.get(TRACE_PH_KEY)
    if not ph:
        return
    with ph:
        st.markdown(
            "<div style='position:sticky; top:0; background:transparent; padding-bottom:6px'><h3>Run Trace</h3></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='max-height:70vh; overflow-y:auto; padding-right:6px; font-size:0.9rem; line-height:1.2;'>",
            unsafe_allow_html=True,
        )
        events = st.session_state.get(TRACE_KEY, [])
        compact = True

        if compact:
            filtered = []
            last_sig = None
            for ev in events:
                if ev.get("kind") == "agent_result":
                    continue
                sig = (ev.get("kind"), ev.get("title"))
                if sig == last_sig:
                    continue
                filtered.append(ev)
                last_sig = sig
            events_to_render = filtered[-15:]
        else:
            events_to_render = events[-15:]

        if not events_to_render:
            st.info("No trace available.")
        else:
            # Render a compact bullet list for readability
            lines = []
            for ev in events_to_render:
                kind = ev["kind"]
                title = ev.get("title", "")
                details = ev.get("details", {}) or {}
                if kind == "handoff":
                    frm = details.get('from','-'); to = details.get('to','-'); reason = details.get('reason','-')
                    lines.append(f"- Handoff: {frm} → {to} — {reason}")
                elif kind == "tool_call":
                    by = details.get("by", "-") if isinstance(details, dict) else "-"
                    lines.append(f"- Tool: {title} (by {by})")
                elif kind == "tool_done":
                    by = details.get("by", "-") if isinstance(details, dict) else "-"
                    lines.append(f"- Done: {title} (by {by})")
                elif kind == "agent_result":
                    continue

            st.markdown("\n".join(lines))
        st.markdown("</div>", unsafe_allow_html=True)


__all__ = [
    "TRACE_KEY",
    "TRACE_PH_KEY",
    "log_event",
    "clear_trace",
    "render_trace",
]
