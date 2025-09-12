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
        res = await Runner.run(test_agent, "Say OK", run_config=RunConfig(model=model_name))
        ok = (res.final_output or "").strip().upper() == "OK"
        return ok, ("OK" if ok else f"Unexpected output: {res.final_output!r}")
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
        st.write("Environment (redacted)")
        st.json(
            {
                "Provider": PROVIDER,
                "DefaultModel": get_active_default_model(),
                "SummaryModel": get_active_summary_model(),
                "OPENAI_API_KEY": _redact(os.getenv("OPENAI_API_KEY")),
                "OPENAI_PROJECT_ID": _redact(os.getenv("OPENAI_PROJECT_ID")),
                "OPENAI_ORG_ID": _redact(os.getenv("OPENAI_ORG_ID")),
                "GEMINI_API_KEY": _redact(os.getenv("GEMINI_API_KEY")),
                "LITELLM_LOG": os.getenv("LITELLM_LOG", "-"),
            }
        )

        if st.button("Run model health check"):
            with st.spinner("Pinging selected model..."):
                ok, msg = _run_async_in_new_loop(_ping_model(get_active_default_model()))
            (st.success if ok else st.error)(msg)


__all__ = ["render_diagnostics_panel"]
