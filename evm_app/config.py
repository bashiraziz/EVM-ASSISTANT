import os
from datetime import date
from typing import Optional

from dotenv import load_dotenv

try:
    # Optional import; only used for safe checks
    from evm_app.ui.st_compat import in_streamlit
except Exception:
    def in_streamlit() -> bool:  # type: ignore
        return False

# Load .env early
load_dotenv()


# =============================
# Model selection and provider
# =============================
def _resolve_default_model() -> str:
    env_val = os.getenv("AGENTS_DEFAULT_MODEL") or os.getenv("OPENAI_DEFAULT_MODEL")
    if env_val:
        val = env_val.strip()
        low = val.lower()
        if low.startswith("gemini") and not low.startswith("litellm/"):
            return f"litellm/gemini/{val}"
        return val
    return "gpt-4o-mini"


DEFAULT_MODEL = _resolve_default_model()
PROVIDER = "LiteLLM" if DEFAULT_MODEL.lower().startswith("litellm/") else "OpenAI"


def _resolve_model(name: str) -> str:
    low = name.strip().lower()
    if low.startswith("gemini") and not low.startswith("litellm/"):
        return f"litellm/gemini/{name.strip()}"
    return name.strip()


SUMMARY_MODEL = _resolve_model(os.getenv("AGENTS_SUMMARY_MODEL", "gemini-1.5-flash"))


def _base_model_name(name: str) -> str:
    if not name:
        return name
    low = name.strip().lower()
    if low.startswith("litellm/gemini/"):
        return name.split("/", 2)[-1]
    return name


def get_active_default_model() -> str:
    ui_val: Optional[str] = None
    try:
        import streamlit as st  # local import to avoid CLI import cost
        if in_streamlit():
            ui_val = st.session_state.get("__model_default__")
    except Exception:
        pass
    if ui_val:
        return _resolve_model(ui_val)
    return DEFAULT_MODEL


def get_active_summary_model() -> str:
    ui_val: Optional[str] = None
    try:
        import streamlit as st
        if in_streamlit():
            ui_val = st.session_state.get("__model_summary__")
    except Exception:
        pass
    if ui_val:
        return _resolve_model(ui_val)
    return SUMMARY_MODEL


# =============================
# LiteLLM worker reset (Streamlit reruns)
# =============================
def reset_litellm_logging_worker() -> None:
    if PROVIDER != "LiteLLM":
        return
    try:
        import asyncio
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        async def _stop():
            try:
                await GLOBAL_LOGGING_WORKER.stop()  # cancel background task if any
            except Exception:
                pass

        loop = asyncio.get_event_loop()
        try:
            if loop.is_running():
                # schedule cancellation on this loop; don't await
                loop.create_task(_stop())
            else:
                loop.run_until_complete(_stop())
        except Exception:
            pass

        # Hard reset internal handles so next start() binds cleanly
        try:
            GLOBAL_LOGGING_WORKER._queue = None  # type: ignore[attr-defined]
            GLOBAL_LOGGING_WORKER._worker_task = None  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


# Keep LiteLLM logs quieter by default
if PROVIDER == "LiteLLM" and not os.getenv("LITELLM_LOG"):
    os.environ["LITELLM_LOG"] = "INFO"


# =============================
# URL param helpers
# =============================
def load_url_params_into_state():
    try:
        import streamlit as st
        if not in_streamlit():
            return
        try:
            qp = dict(st.query_params)  # Streamlit >=1.31
        except Exception:
            qp = st.experimental_get_query_params() or {}

        if "md" in qp and qp["md"]:
            st.session_state.setdefault(
                "__model_default__", (qp["md"][0] if isinstance(qp["md"], list) else qp["md"])
            )
        if "ms" in qp and qp["ms"]:
            st.session_state.setdefault(
                "__model_summary__", (qp["ms"][0] if isinstance(qp["ms"], list) else qp["ms"])
            )
        # thresholds handled by UI
    except Exception:
        pass


def set_url_params_safe(**params: str):
    try:
        import streamlit as st
        if not in_streamlit():
            return
        clean = {k: v for k, v in params.items() if v is not None and v != ""}
        if clean:
            try:
                for k, v in clean.items():
                    st.query_params[k] = v
            except Exception:
                st.experimental_set_query_params(**clean)
    except Exception:
        pass


# =============================
# Thresholds (shared mutable state)
# =============================
RISK_CPI_THRESHOLD: float = 0.9
RISK_SPI_THRESHOLD: float = 0.9


__all__ = [
    "DEFAULT_MODEL",
    "SUMMARY_MODEL",
    "PROVIDER",
    "_base_model_name",
    "get_active_default_model",
    "get_active_summary_model",
    "reset_litellm_logging_worker",
    "load_url_params_into_state",
    "set_url_params_safe",
    "RISK_CPI_THRESHOLD",
    "RISK_SPI_THRESHOLD",
]
