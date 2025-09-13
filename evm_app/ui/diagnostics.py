from typing import Optional, Tuple

import asyncio
import os
import streamlit as st

from agents import Agent
from agents.run import Runner, RunConfig

from ..config import (
    PROVIDER,
    get_active_default_model,
    get_active_summary_model,
)


def _redact(val: Optional[str], keep: int = 4) -> str:
    if not val:
        return "-"
    val = str(val)
    if len(val) <= keep:
        return "..."
    return val[:keep] + "..."


async def _ping_model(model_name: str) -> Tuple[bool, str]:
    test_agent = Agent(
        name="HealthCheck",
        instructions="Reply with exactly: OK",
        tool_use_behavior="run_llm_again",
    )
    try:
        import time as _t
        t0 = _t.perf_counter()
        res = await Runner.run(test_agent, "Say OK", run_config=RunConfig(model=model_name))
        dt = (_t.perf_counter() - t0) * 1000.0
        ok = (res.final_output or "").strip().upper() == "OK"
        return ok, (f"OK • {dt:.0f} ms" if ok else f"Unexpected output: {res.final_output!r}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _run_async_in_new_loop(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.close()
        except Exception:
            pass


def render_diagnostics_panel():
    with st.expander("Diagnostics", expanded=False):
        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.markdown("#### Environment")
            st.markdown(
                f"- Provider: `{PROVIDER}`\n"
                f"- Default Model: `{get_active_default_model()}`\n"
                f"- Summary Model: `{get_active_summary_model()}`\n",
            )
        with col_b:
            st.markdown("#### Secrets (redacted)")
            st.code(
                "\n".join(
                    [
                        f"OPENAI_API_KEY={_redact(os.getenv('OPENAI_API_KEY'))}",
                        f"OPENAI_PROJECT_ID={_redact(os.getenv('OPENAI_PROJECT_ID'))}",
                        f"OPENAI_ORG_ID={_redact(os.getenv('OPENAI_ORG_ID'))}",
                        f"GEMINI_API_KEY={_redact(os.getenv('GEMINI_API_KEY'))}",
                        f"LITELLM_LOG={os.getenv('LITELLM_LOG', '-')}",
                    ]
                ),
                language="bash",
            )

        st.divider()
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Run model health check"):
                with st.spinner("Pinging selected model…"):
                    ok, msg = _run_async_in_new_loop(_ping_model(get_active_default_model()))
                (st.success if ok else st.error)(msg)
        with col2:
            st.caption("Checks a minimal round-trip to the selected default model and reports latency.")


__all__ = ["render_diagnostics_panel"]
