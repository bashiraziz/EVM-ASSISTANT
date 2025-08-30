from typing import Any, Dict, Optional

import streamlit as st


TRACE_KEY = "trace"
TRACE_PH_KEY = "__trace_ph__"


def _ensure_trace():
    if TRACE_KEY not in st.session_state:
        st.session_state[TRACE_KEY] = []


def log_event(kind: str, title: str, details: Optional[Dict[str, Any]] = None):
    _ensure_trace()
    st.session_state[TRACE_KEY].append({
        "kind": kind,
        "title": title,
        "details": details or {},
    })
    _render_trace_into_placeholder()


def clear_trace():
    st.session_state[TRACE_KEY] = []
    _render_trace_into_placeholder()


def render_trace():
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
    ph = st.session_state.get(TRACE_PH_KEY)
    if not ph:
        return
    with ph:
        st.markdown("<div style='position:sticky; top:0; background:transparent; padding-bottom:6px'><h3>Run Trace</h3></div>", unsafe_allow_html=True)

        st.markdown("<div style='max-height:70vh; overflow-y:auto; padding-right:6px'>", unsafe_allow_html=True)
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
            events_to_render = filtered[-20:]
        else:
            events_to_render = events[-20:]

        if not events_to_render:
            st.info("No trace available.")
        else:
            for ev in events_to_render:
                kind = ev["kind"]
                title = ev["title"]
                details = ev["details"]

                if kind == "handoff":
                    if compact:
                        frm = details.get('from','-'); to = details.get('to','-'); reason = details.get('reason','-')
                        st.write(f"Handoff: {frm} -> {to} - {reason}")
                    else:
                        st.markdown(
                            f"""
<div class=\"trace-card\" style=\"padding:12px;border-radius:10px;\">\n  <div style=\"font-weight:700;color:#fff;\">{title}</div>\n  <div style=\"color:#cfe3ff;margin-top:6px;\">Reason: {details.get('reason','-')}</div>\n  <div style=\"color:#cfe3ff;\">Context: {details.get('context', details.get('location','-'))}</div>\n  <div style=\"color:#9ad0ff;\">From: {details.get('from','-')} | To: {details.get('to','-')}</div>\n</div>
                            """,
                            unsafe_allow_html=True,
                        )
                elif kind == "tool_call":
                    by = details.get("by", "-") if isinstance(details, dict) else "-"
                    st.write(f"Tool: {title} - by {by}")
                elif kind == "tool_done":
                    by = details.get("by", "-") if isinstance(details, dict) else "-"
                    st.write(f"Done: {title} - by {by}")
                elif kind == "agent_result":
                    if not compact:
                        with st.expander(f"{title}", expanded=False):
                            st.write(details.get("text", ""))
        st.markdown("</div>", unsafe_allow_html=True)


__all__ = [
    "TRACE_KEY",
    "TRACE_PH_KEY",
    "log_event",
    "clear_trace",
    "render_trace",
]

