"""
EVM Assistant (Agents-as-Tools, Streamlit UI)

What it does
- Accepts a CSV upload with headers similar to `project_template.csv`.
- Uses a swarm of specialist agents (Ingestion, EVM Calculator, Risk Analyst) as tools.
- Computes EVM metrics per project: PV, EV, AC, %Complete, CPI, SPI, CV, SV, EAC, ETC.
- Orchestrator agent decides which tools to call and merges results into a single report.

LLM Provider
- This app is built with the OpenAI Agents SDK. To use Gemini, configure your environment
  to point the SDK at Gemini (e.g., via .env). For example (adjust to your SDK version):
    GOOGLE_API_KEY=...   # Gemini key
    AGENTS_DEFAULT_MODEL=gemini-1.5-pro   # or gemini-1.5-flash
  If your SDK uses a different env naming, set accordingly.
"""

from agents import Agent, Runner, function_tool
from agents.run import RunConfig
import asyncio
import streamlit as st
from dotenv import load_dotenv
import csv
import io
import json
from datetime import datetime, date
import os
from typing import Any, Dict, List, Optional, Tuple

load_dotenv()


# =============================
# Model selection
# =============================
def _resolve_default_model() -> str:
    """Choose the default model, supporting Gemini via LiteLLM.

    Priority:
    - AGENTS_DEFAULT_MODEL (app-specific)
    - OPENAI_DEFAULT_MODEL (fallback)
    - gpt-4o-mini (final fallback)

    If the chosen value looks like a Gemini model (e.g., "gemini-1.5-pro" or "gemini-1.5-flash"),
    route through LiteLLM by prefixing with "litellm/gemini/".
    """
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

# Optional: allow a different model for specific agents (e.g., Summary)
def _resolve_model(name: str) -> str:
    low = name.strip().lower()
    if low.startswith("gemini") and not low.startswith("litellm/"):
        return f"litellm/gemini/{name.strip()}"
    return name.strip()

# Secondary/agent-specific model (defaults to a faster Gemini)
SUMMARY_MODEL = _resolve_model(os.getenv("AGENTS_SUMMARY_MODEL", "gemini-1.5-flash"))


def _base_model_name(name: str) -> str:
    """Return provider-agnostic base name (e.g., 'gemini-1.5-pro')."""
    if not name:
        return name
    low = name.strip().lower()
    if low.startswith("litellm/gemini/"):
        return name.split("/", 2)[-1]
    return name


def get_active_default_model() -> str:
    """Resolved active model for orchestrator/agents (UI override > env)."""
    ui_val = None
    try:
        import streamlit as _st  # local import to avoid issues in non-UI contexts
        ui_val = _st.session_state.get("__model_default__")
    except Exception:
        pass
    if ui_val:
        return _resolve_model(ui_val)
    return DEFAULT_MODEL


def get_active_summary_model() -> str:
    """Resolved active model for summary (UI override > env)."""
    ui_val = None
    try:
        import streamlit as _st  # local import to avoid issues in non-UI contexts
        ui_val = _st.session_state.get("__model_summary__")
    except Exception:
        pass
    if ui_val:
        return _resolve_model(ui_val)
    return SUMMARY_MODEL

# Keep LiteLLM logs quieter and suppress benign shutdown noise
if PROVIDER == "LiteLLM":
    if not os.getenv("LITELLM_LOG"):
        os.environ["LITELLM_LOG"] = "INFO"
    try:
        import logging

        _lite_logger = logging.getLogger("LiteLLM")

        class _IgnoreLiteCancelled(logging.Filter):
            def filter(self, record):
                return "LoggingWorker cancelled" not in record.getMessage()

        _lite_logger.addFilter(_IgnoreLiteCancelled())
    except Exception:
        # If logging isn't available yet, skip filtering.
        pass


def _reset_litellm_logging_worker() -> None:
    """Best-effort reset of LiteLLM's background logging worker.

    Streamlit frequently recreates event loops between reruns; LiteLLM's global
    LoggingWorker binds its queue to the loop at creation time. This ensures the
    worker is reset so it rebinds cleanly to the current loop when first used.
    """
    if PROVIDER != "LiteLLM":
        return
    try:
        import asyncio
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(GLOBAL_LOGGING_WORKER.stop())
        except Exception:
            # If we can't synchronously stop, continue to hard reset below
            pass

        # Hard reset internal state so the next start() creates a fresh queue/task on this loop
        try:
            GLOBAL_LOGGING_WORKER._queue = None  # type: ignore[attr-defined]
            GLOBAL_LOGGING_WORKER._worker_task = None  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


# =============================
# Diagnostics helpers
# =============================
def _redact(val: Optional[str], keep: int = 4) -> str:
    if not val:
        return "-"
    val = str(val)
    if len(val) <= keep:
        return "..."
    return val[:keep] + "..."


async def _ping_model(model_name: str) -> Tuple[bool, str]:
    """Run a tiny health-check through the selected model.

    Returns (ok, message) where ok indicates success and message provides detail.
    """
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
    """Run an async coroutine in a fresh event loop to avoid conflicts."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.close()
        except Exception:
            pass


# =============================
# URL param helpers (shareable links)
# =============================
def _load_url_params_into_state():
    try:
        import streamlit as _st
        # Prefer modern API; fall back to experimental on older Streamlit
        try:
            qp = dict(_st.query_params)
        except Exception:
            qp = _st.experimental_get_query_params() or {}
        # Models
        if "md" in qp and qp["md"]:
            _st.session_state.setdefault("__model_default__", (qp["md"][0] if isinstance(qp["md"], list) else qp["md"]))
        if "ms" in qp and qp["ms"]:
            _st.session_state.setdefault("__model_summary__", (qp["ms"][0] if isinstance(qp["ms"], list) else qp["ms"]))
        # Thresholds
        global RISK_CPI_THRESHOLD, RISK_SPI_THRESHOLD
        if "cpi" in qp and qp["cpi"]:
            try:
                RISK_CPI_THRESHOLD = float(qp["cpi"][0] if isinstance(qp["cpi"], list) else qp["cpi"])
            except Exception:
                pass
        if "spi" in qp and qp["spi"]:
            try:
                RISK_SPI_THRESHOLD = float(qp["spi"][0] if isinstance(qp["spi"], list) else qp["spi"])
            except Exception:
                pass
        # Filters are applied later when available; stash raw values
        for key, st_key in (("pm", "__flt_pms__"), ("dept", "__flt_depts__"), ("status", "__flt_status__")):
            if key in qp and qp[key]:
                raw = qp[key]
                if isinstance(raw, list):
                    raw = raw[0]
                _st.session_state.setdefault(st_key, [x for x in str(raw).split(",") if x])
    except Exception:
        pass


def _set_url_params_safe(**params: str):
    try:
        import streamlit as _st
        clean = {k: v for k, v in params.items() if v is not None and v != ""}
        if clean:
            try:
                # Modern API: mutate st.query_params (doesn't require full replace)
                for k, v in clean.items():
                    _st.query_params[k] = v
            except Exception:
                _st.experimental_set_query_params(**clean)
    except Exception:
        pass


# =============================
# Trace utilities
# =============================
TRACE_KEY = "trace"
TRACE_PH_KEY = "__trace_ph__"

# =============================
# Minimal theme (Edge-inspired blue/green)
# =============================
def inject_theme():
    st.markdown(
        """
        <style>
        :root{
          --acc1:#0b6cf0;      /* azure-ish blue */
          --acc2:#18c29c;      /* teal/green */
          --panel:#0f1b2e;     /* panel navy */
        }
        .trace-card{background:linear-gradient(135deg,rgba(11,108,240,.18),rgba(24,194,156,.18));border:1px solid rgba(255,255,255,.08);}
        .table-head th{background:linear-gradient(135deg,rgba(11,108,240,.35),rgba(24,194,156,.35));color:#fff;border-bottom:1px solid rgba(255,255,255,.08)!important}
        .totals-banner{background:linear-gradient(135deg,var(--acc1),var(--acc2));color:#fff}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        compact = True  # force compact mode for cleaner UX

        # Compact mode: remove agent_result events and collapse duplicates; limit count
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


# =============================
# Utilities (CSV parsing, EVM math)
# =============================
EXPECTED_HEADERS = [
    "ProjectID",
    "ProjectName",
    "Customer",
    "CustomerContact",
    "PM",
    "Dept",
    "PlannedCost",
    "ActualCost",
    "StartDate",
    "EndDate",
    "Status",
    "ProgressPercent_Manual",
    "ContractType",
]

RISK_CPI_THRESHOLD: float = 0.9
RISK_SPI_THRESHOLD: float = 0.9


# =============================
# UI helpers (sample file, colored table)
# =============================
SAMPLE_REL_PATH = os.path.join("samples", "evms_sample.csv")


def get_sample_csv_text() -> str:
    header = ",".join(EXPECTED_HEADERS)
    default_sample = """{}\nP1001,New Website,Acme Corp,alice@acme.com,Sam,IT,100000,45000,2025-01-01,2025-12-31,Active,40,Fixed Price\nP1002,Mobile App,Globex,bob@globex.com,Kim,Product,200000,130000,2025-02-01,2025-11-30,Active,55,Time & Materials\nP1003,Data Migration,Initech,carol@initech.com,Raj,Ops,150000,90000,2025-03-15,2025-09-15,Active,35,Fixed Price\n""".format(header)
    try:
        if os.path.exists(SAMPLE_REL_PATH):
            with open(SAMPLE_REL_PATH, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return default_sample


def _compute_portfolio_from_csv_for_ui(csv_text: str, as_of_date: Optional[str]):
    f = io.StringIO(csv_text or "")
    reader = csv.DictReader(f)
    items = []
    row_errors: List[str] = []
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date() if as_of_date else date.today()
    for idx, r in enumerate(reader, start=2):  # start=2 to account for header line
        raw = {k: (v or "").strip() for k, v in r.items()}
        try:
            items.append(_compute_evms_row(raw, as_of))
        except Exception as e:
            row_errors.append(f"Row {idx}: {e}")

    def sumf(key: str) -> float:
        return round(sum((x.get(key) or 0.0) for x in items), 2)

    totals = {
        "BAC": sumf("BAC"),
        "PV": sumf("PV"),
        "EV": sumf("EV"),
        "AC": sumf("AC"),
        "CPI": round((sumf("EV") / sumf("AC")), 3) if sumf("AC") > 0 else None,
        "SPI": round((sumf("EV") / sumf("PV")), 3) if sumf("PV") > 0 else None,
        "CV": round(sumf("EV") - sumf("AC"), 2),
        "SV": round(sumf("EV") - sumf("PV"), 2),
        "AsOf": as_of.isoformat(),
    }
    return items, totals, row_errors


def _score_color(val: Optional[float]) -> str:
    if val is None:
        return "#444"
    if val >= 1.0:
        return "#1b5e20"  # green
    if val >= 0.95:
        return "#ff8f00"  # amber
    return "#b71c1c"      # red


def render_evms_colored_table(items: List[Dict[str, Any]], totals: Dict[str, Any]):
    cols = [
        "ProjectID", "ProjectName", "PM", "Dept", "Status", "StartDate", "OriginalEndDate", "ExpectedEndDate",
        "BAC", "PV", "EV", "AC", "CPI", "SPI", "CV", "SV", "EAC", "ETC", "Risk"
    ]
    tooltips = {
        "BAC": "Budget At Completion: total planned cost",
        "PV": "Planned Value (BCWS): BAC x planned % complete",
        "EV": "Earned Value (BCWP): BAC x actual % complete",
        "AC": "Actual Cost (ACWP): actual cost to date",
        "CPI": "Cost Performance Index = EV / AC (>= 1.0 is on/under cost)",
        "SPI": "Schedule Performance Index = EV / PV (>= 1.0 is on/ahead of schedule)",
        "CV": "Cost Variance = EV - AC (>0 favorable)",
        "SV": "Schedule Variance = EV - PV (>0 favorable)",
        "EAC": "Estimate At Completion ~ BAC / CPI",
        "ETC": "Estimate To Complete = EAC - AC",
        "Risk": "Derived from CPI/SPI and variances; thresholds configurable above",
        "OriginalEndDate": "Baseline planned end date",
        "ExpectedEndDate": "Current expected end date (if provided)",
        "StartDate": "Project start date",
        "ProjectID": "Identifier from your CSV",
        "ProjectName": "Name from your CSV",
        "PM": "Project Manager",
        "Dept": "Department or cost center",
        "Status": "Project status from your CSV",
    }

    def _cell_tooltip(row: Dict[str, Any], col: str, val: Any) -> str:
        """Return a concise tooltip for a cell value, including active thresholds where relevant."""
        if val is None or val == "":
            return ""
        try:
            v = float(val) if isinstance(val, (int, float, str)) else None
        except Exception:
            v = None

        if col == "CPI":
            tip = f"CPI = EV / AC. Threshold: < {RISK_CPI_THRESHOLD} flagged low."
            if v is not None:
                tip += f" Value: {v}."
            return tip
        if col == "SPI":
            tip = f"SPI = EV / PV. Threshold: < {RISK_SPI_THRESHOLD} flagged low."
            if v is not None:
                tip += f" Value: {v}."
            return tip
        if col == "CV":
            return "CV = EV - AC (>0 favorable)."
        if col == "SV":
            return "SV = EV - PV (>0 favorable)."
        if col == "EAC":
            return "EAC ~ BAC / CPI."
        if col == "ETC":
            return "ETC = EAC - AC."
        if col == "PV":
            return "PV = BAC x planned % complete."
        if col == "EV":
            return "EV = BAC x actual % complete."
        if col == "AC":
            return "Actual Cost to date."
        # For other columns, fall back to header tooltip if present
        return tooltips.get(col, "")
    def fmt(v):
        if v is None:
            return "-"
        return f"{v}"
    rows = []
    for it in items:
        risk_level, _ = _risk_level_and_reasons(it)
        row = {c: it.get(c) for c in cols}
        row["Risk"] = risk_level
        rows.append(row)

    html = [
        "<div style='margin-top:12px; overflow-x:auto; max-width:100%; max-height:60vh; overflow-y:auto'>",
        "<table style='border-collapse:collapse;width:100%'>",
        "<thead class='table-head'><tr>" + "".join(
            (lambda _c: f"<th style='text-align:left;padding:8px 10px;cursor:help' title=\"{tooltips.get(_c, '')}\">{_c}</th>") (c)
            for c in cols
        ) + "</tr></thead>",
        "<tbody>"
    ]
    for r in rows:
        html.append("<tr>")
        for c in cols:
            val = r.get(c)
            style = "padding:6px 8px;border-bottom:1px solid #333;"
            if c in ("CPI", "SPI"):
                style += f"background:{_score_color(val)};color:#fff;"
            if c == "Risk":
                color = {"high": "#b71c1c", "medium": "#ff8f00", "low": "#1b5e20"}.get(str(val), "#444")
                style += f"background:{color};color:#fff;font-weight:600;"
            title = _cell_tooltip(r, c, val)
            if title:
                style += "cursor:help;"
            html.append(f"<td style='{style}' title=\"{title}\">{fmt(val)}</td>")
        html.append("</tr>")
    html.append("</tbody>")
    html.append("</table>")

    # Totals summary
    tot = totals or {}
    html.append(
        "<div class='totals-banner' style='margin-top:10px;padding:10px;border-radius:10px;'>"
        f"<b>Portfolio Totals</b> &mdash; As Of: {fmt(tot.get('AsOf'))} | "
        f"BAC: {fmt(tot.get('BAC'))}, PV: {fmt(tot.get('PV'))}, EV: {fmt(tot.get('EV'))}, AC: {fmt(tot.get('AC'))}, "
        f"CPI: {fmt(tot.get('CPI'))}, SPI: {fmt(tot.get('SPI'))}, CV: {fmt(tot.get('CV'))}, SV: {fmt(tot.get('SV'))}"
        "</div>"
    )
    html.append("</div>")

    st.markdown("".join(html), unsafe_allow_html=True)


def _risk_level_and_reasons(it: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons = []
    cpi = it.get("CPI")
    spi = it.get("SPI")
    cv = it.get("CV")
    sv = it.get("SV")
    if cpi is not None and cpi < RISK_CPI_THRESHOLD:
        reasons.append(f"CPI low ({cpi})")
    if spi is not None and spi < RISK_SPI_THRESHOLD:
        reasons.append(f"SPI low ({spi})")
    if cv is not None and cv < 0:
        reasons.append(f"Over cost (CV={cv})")
    if sv is not None and sv < 0:
        reasons.append(f"Behind schedule (SV={sv})")
    level = "low"
    if len(reasons) >= 3:
        level = "high"
    elif len(reasons) >= 1:
        level = "medium"
    return level, reasons


def render_cpi_spi_heatmap(items: List[Dict[str, Any]]):
    # Define bins based on thresholds used in colors
    def bin_val(v: Optional[float]) -> Optional[str]:
        if v is None:
            return None
        if v >= 1.0:
            return "high"
        if v >= 0.95:
            return "mid"
        return "low"

    bins = {("low", "low"): [], ("low", "mid"): [], ("low", "high"): [],
            ("mid", "low"): [], ("mid", "mid"): [], ("mid", "high"): [],
            ("high", "low"): [], ("high", "mid"): [], ("high", "high"): []}
    excluded = 0
    for it in items:
        cb = bin_val(it.get("CPI"))
        sb = bin_val(it.get("SPI"))
        if cb is None or sb is None:
            excluded += 1
            continue
        bins[(cb, sb)].append(it)

    label = {"low": "< 0.95", "mid": "0.95-<1.0", "high": ">= 1.0"}
    color = {"low": "#b71c1c", "mid": "#ff8f00", "high": "#1b5e20"}

    html = [
        "<div style='margin-top:12px; overflow-x:auto; max-width:100%'>",
        "<b>CPI x SPI Heatmap</b>",
        "<table style='margin-top:6px;border-collapse:collapse;width:100%'>",
        "<thead><tr><th></th>" + "".join(
            f"<th style='text-align:center;border-bottom:1px solid #444;padding:6px 8px'>SPI {label[c]}</th>" for c in ("low","mid","high")
        ) + "</tr></thead>",
        "<tbody>"
    ]
    for rbin in ("low", "mid", "high"):
        html.append("<tr>")
        html.append(f"<th style='text-align:right;border-right:1px solid #444;padding:6px 8px'>CPI {label[rbin]}</th>")
        for cbin in ("low", "mid", "high"):
            cell = bins[(rbin, cbin)]
            bg = color[rbin]  # row color tint
            # mix with column tint by adjusting opacity via gradient overlay
            style = f"padding:8px;border:1px solid #333;background:{bg};color:#fff;text-align:center" \
                    
            count = len(cell)
            top = ", ".join([str(x.get("ProjectID")) for x in cell[:3]])
            content = f"<div style='font-size:16px;font-weight:700'>{count}</div>"
            if top:
                content += f"<div style='font-size:12px;opacity:.9'>[{top}]</div>"
            html.append(f"<td style='{style}'>{content}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    if excluded:
        html.append(f"<div style='margin-top:6px;color:#94a3b8'>Excluded {excluded} project(s) with missing CPI/SPI.</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except Exception:
            continue
    return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(s: Any) -> Optional[float]:
    try:
        if s is None:
            return None
        return float(str(s).strip())
    except Exception:
        return None


def _compute_evms_row(row: Dict[str, Any], as_of: date) -> Dict[str, Any]:
    bac = _safe_float(row.get("PlannedCost")) or 0.0
    ac = _safe_float(row.get("ActualCost")) or 0.0
    start = _parse_date(row.get("StartDate", ""))
    end_original = _parse_date(row.get("EndDate", ""))
    end_expected = _parse_date(row.get("ExpectedEndDate", ""))
    # Effective end date for schedule calculations: expected if provided, else original
    end = end_expected or end_original

    # Planned percent complete based on schedule
    if start and end and end > start:
        total_days = (end - start).days
        elapsed_days = (min(as_of, end) - start).days
        planned_pct = _clamp(elapsed_days / total_days, 0.0, 1.0)
    else:
        planned_pct = 0.0

    # Manual progress percent
    manual_pct = (_safe_float(row.get("ProgressPercent_Manual")) or 0.0) / 100.0

    # PV, EV (using manual percent as EV driver), AC from data
    pv = bac * planned_pct
    ev = bac * manual_pct

    # Core indicators
    cpi = (ev / ac) if ac > 0 else None
    spi = (ev / pv) if pv > 0 else None
    cv = ev - ac
    sv = ev - pv
    eac = (bac / cpi) if cpi and cpi > 0 else None
    etc = (eac - ac) if eac is not None else None

    return {
        **row,
        "OriginalEndDate": row.get("EndDate", ""),
        "ExpectedEndDate": row.get("ExpectedEndDate", ""),
        "BAC": round(bac, 2),
        "PV": round(pv, 2),
        "EV": round(ev, 2),
        "AC": round(ac, 2),
        "%Complete_Manual": round(manual_pct * 100.0, 2),
        "%Complete_Schedule": round(planned_pct * 100.0, 2),
        "CPI": round(cpi, 3) if cpi is not None else None,
        "SPI": round(spi, 3) if spi is not None else None,
        "CV": round(cv, 2),
        "SV": round(sv, 2),
        "EAC": round(eac, 2) if eac is not None else None,
        "ETC": round(etc, 2) if etc is not None else None,
    }


# =============================
# Tools
# =============================
@function_tool
def parse_and_validate_csv(csv_text: str) -> Dict[str, Any]:
    """Validate the uploaded CSV against the expected headers and parse rows.

    Returns { ok: bool, errors: list[str], projects: list[dict] }
    """
    log_event("tool_call", "parse_and_validate_csv", {"by": "Ingestion Agent"})
    errors: List[str] = []
    projects: List[Dict[str, Any]] = []

    try:
        f = io.StringIO(csv_text)
        reader = csv.DictReader(f)

        headers = reader.fieldnames or []
        missing = [h for h in EXPECTED_HEADERS if h not in headers]
        extra = [h for h in headers if h not in EXPECTED_HEADERS]

        if missing:
            errors.append(f"Missing headers: {', '.join(missing)}")
        if not headers:
            errors.append("No headers found in CSV.")

        for r in reader:
            # Keep as raw strings; convert later during EVMS computation
            projects.append({k: (v or "").strip() for k, v in r.items()})

        ok = len(errors) == 0
        result = {"ok": ok, "errors": errors, "projects": projects, "extraHeaders": extra}
    except Exception as e:
        result = {"ok": False, "errors": [f"CSV parse error: {e}"], "projects": []}

    log_event("tool_done", "parse_and_validate_csv", {"by": "Ingestion Agent"})
    return result


@function_tool
def compute_evms(projects_json: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
    """Compute EVM metrics per project and portfolio aggregates.

    - projects_json: JSON string of list[project dict] following the expected template.
    - as_of_date: ISO date string (YYYY-MM-DD). Defaults to today.
    Returns { items: list[dict], totals: dict }
    """
    log_event("tool_call", "compute_evms", {"by": "EVM Calculator Agent"})

    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date() if as_of_date else date.today()
    computed: List[Dict[str, Any]] = []

    try:
        projects = json.loads(projects_json) if projects_json else []
        if not isinstance(projects, list):
            raise ValueError("projects_json must be a JSON array")
    except Exception as e:
        log_event("tool_done", "compute_evms")
        return {"items": [], "totals": {"error": f"Invalid projects_json: {e}"}}

    for row in projects:
        computed.append(_compute_evms_row(row, as_of))

    # Portfolio totals (sum BAC, PV, EV, AC; others derived)
    def sumf(key: str) -> float:
        return round(sum((x.get(key) or 0.0) for x in computed), 2)

    total_bac = sumf("BAC")
    total_pv = sumf("PV")
    total_ev = sumf("EV")
    total_ac = sumf("AC")
    portfolio_cpi = (total_ev / total_ac) if total_ac > 0 else None
    portfolio_spi = (total_ev / total_pv) if total_pv > 0 else None

    totals = {
        "BAC": total_bac,
        "PV": total_pv,
        "EV": total_ev,
        "AC": total_ac,
        "CPI": round(portfolio_cpi, 3) if portfolio_cpi is not None else None,
        "SPI": round(portfolio_spi, 3) if portfolio_spi is not None else None,
        "CV": round(total_ev - total_ac, 2),
        "SV": round(total_ev - total_pv, 2),
        "AsOf": as_of.isoformat(),
    }

    result = {"items": computed, "totals": totals}
    log_event("tool_done", "compute_evms", {"by": "EVM Calculator Agent"})
    return result


@function_tool
def compute_evms_from_csv(csv_text: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
    """Convenience tool: parse CSV and compute EVM in one call.

    - csv_text: raw CSV content (UTF-8)
    - as_of_date: ISO date string (YYYY-MM-DD). Defaults to today.
    Returns same structure as `compute_evms`.
    """
    log_event("tool_call", "compute_evms_from_csv", {"by": "EVM Calculator Agent"})
    parsed = parse_and_validate_csv(csv_text)
    if not parsed.get("ok"):
        log_event("tool_done", "compute_evms_from_csv")
        return {"items": [], "totals": {"error": "; ".join(parsed.get("errors", []))}}

    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date() if as_of_date else date.today()
    projects = parsed.get("projects", [])
    computed: List[Dict[str, Any]] = []
    for row in projects:
        computed.append(_compute_evms_row(row, as_of))

    def sumf(key: str) -> float:
        return round(sum((x.get(key) or 0.0) for x in computed), 2)

    total_bac = sumf("BAC")
    total_pv = sumf("PV")
    total_ev = sumf("EV")
    total_ac = sumf("AC")
    portfolio_cpi = (total_ev / total_ac) if total_ac > 0 else None
    portfolio_spi = (total_ev / total_pv) if total_pv > 0 else None

    totals = {
        "BAC": total_bac,
        "PV": total_pv,
        "EV": total_ev,
        "AC": total_ac,
        "CPI": round(portfolio_cpi, 3) if portfolio_cpi is not None else None,
        "SPI": round(portfolio_spi, 3) if portfolio_spi is not None else None,
        "CV": round(total_ev - total_ac, 2),
        "SV": round(total_ev - total_pv, 2),
        "AsOf": (as_of_date or date.today().isoformat()),
    }

    result = {"items": computed, "totals": totals}
    log_event("tool_done", "compute_evms_from_csv", {"by": "EVM Calculator Agent"})
    return result


@function_tool
def assess_risks(evms_result_json: str) -> Dict[str, Any]:
    """Assess risk signals per project based on CPI/SPI and schedule/cost variances.

    Returns { risks: list[ {ProjectID, ProjectName, level, reasons[]} ] }
    """
    log_event("tool_call", "assess_risks", {"by": "Risk Analyst Agent"})
    risks: List[Dict[str, Any]] = []

    try:
        evms_result = json.loads(evms_result_json) if evms_result_json else {}
    except Exception as e:
        log_event("tool_done", "assess_risks")
        return {"risks": [], "error": f"Invalid evms_result_json: {e}"}

    items = evms_result.get("items", [])
    for it in items:
        reasons = []
        cpi = it.get("CPI")
        spi = it.get("SPI")
        cv = it.get("CV")
        sv = it.get("SV")

        if cpi is not None and cpi < RISK_CPI_THRESHOLD:
            reasons.append(f"CPI low ({cpi})")
        if spi is not None and spi < RISK_SPI_THRESHOLD:
            reasons.append(f"SPI low ({spi})")
        if cv is not None and cv < 0:
            reasons.append(f"Over cost (CV={cv})")
        if sv is not None and sv < 0:
            reasons.append(f"Behind schedule (SV={sv})")

        level = "low"
        if len(reasons) >= 3:
            level = "high"
        elif len(reasons) >= 1:
            level = "medium"

        if reasons:
            risks.append({
                "ProjectID": it.get("ProjectID"),
                "ProjectName": it.get("ProjectName"),
                "level": level,
                "reasons": reasons,
            })

    result = {"risks": risks}
    log_event("tool_done", "assess_risks", {"by": "Risk Analyst Agent"})
    return result


# =============================
# Specialist Agents
# =============================
ingestion_agent = Agent(
    name="Ingestion Agent",
    instructions="""
    You parse and validate a CSV against the expected project template headers.
    Return a short, clear summary of any issues found, and proceed if valid.
    """,
    tools=[parse_and_validate_csv],
    tool_use_behavior="run_llm_again",
)


evms_calculator_agent = Agent(
    name="EVM Calculator Agent",
    instructions="""
    Compute EVM metrics for each project. Use:
    - PV = BAC * planned_percent_complete (time-based between StartDate and EndDate)
    - EV = BAC * ProgressPercent_Manual/100
    - AC from ActualCost
    Derive CPI, SPI, CV, SV, and EAC.
    Return concise tables and short interpretation.
    """,
    tools=[compute_evms_from_csv, compute_evms],
    tool_use_behavior="run_llm_again",
)


risk_analyst_agent = Agent(
    name="Risk Analyst Agent",
    instructions="""
    Analyze EVM results and flag projects at risk based on CPI, SPI, CV, and SV.
    Provide prioritized recommendations for corrective actions.
    """,
    tools=[assess_risks],
    tool_use_behavior="run_llm_again",
)

# A lightweight summarizer used in Fast mode to generate a single narrative.
summary_agent = Agent(
    name="Summary Agent",
    instructions="""
    You write a concise, executive-friendly summary of portfolio EVM results.
    Input will contain portfolio totals and a short list of at-risk projects.
    Keep it under 6 bullets. Prefer plain language and actionable advice.
    """,
    tool_use_behavior="run_llm_again",
)

# Q&A agent (answers only from provided data)
qa_agent = Agent(
    name="Project Q&A Agent",
    instructions="""
    You answer questions strictly using ONLY the JSON data provided in the prompt.
    Guardrails:
    - Do not use outside knowledge, do not browse, and do not infer beyond data.
    - If the answer cannot be derived from the data or the question is unrelated, return exactly the token: OUT_OF_SCOPE
    - When doing counts or filters (e.g., over budget, behind schedule), describe the rule you used.
    - Keep answers concise (<= 8 bullets or <= 8 short lines).
    """,
    tool_use_behavior="run_llm_again",
)


# =============================
# High-level tools for Orchestrator
# =============================
@function_tool
async def get_ingestion_summary(csv_text: str) -> str:
    log_event(
        "handoff",
        "Handing off to Ingestion Agent...",
        {"reason": "Parse and validate CSV", "context": "project_template.csv", "from": "EVM Orchestrator", "to": "Ingestion Agent"},
    )
    result = await Runner.run(
        ingestion_agent,
        input=(
            "Parse and validate this CSV content against the expected headers. "
            "If valid, briefly summarize row count and note any extra headers. "
            "If invalid, list missing headers.\n\nCSV:\n" + csv_text
        ),
        run_config=RunConfig(model=get_active_default_model()),
    )
    log_event("agent_result", "Ingestion Agent response", {"text": result.final_output})
    return result.final_output


@function_tool
async def get_evms_report(csv_text: str, as_of_date: Optional[str] = None) -> str:
    log_event(
        "handoff",
        "Handing off to EVM Calculator...",
        {"reason": "Compute EVM metrics", "context": as_of_date or "today", "from": "EVM Orchestrator", "to": "EVM Calculator Agent"},
    )
    result = await Runner.run(
        evms_calculator_agent,
        input=(
            "Compute EVM metrics for this CSV. "
            "Return a concise per-project summary and portfolio totals.\n\n"
            f"AsOf: {as_of_date or date.today().isoformat()}\n"
            f"CSV:\n{csv_text}"
        ),
        run_config=RunConfig(model=get_active_default_model()),
    )
    log_event("agent_result", "EVM Calculator response", {"text": result.final_output})
    return result.final_output


@function_tool
async def get_risk_assessment(evms_result_json: str) -> str:
    log_event(
        "handoff",
        "Handing off to Risk Analyst...",
        {"reason": "Identify and rank risks", "context": "CPI, SPI, CV, SV", "from": "EVM Orchestrator", "to": "Risk Analyst Agent"},
    )
    result = await Runner.run(
        risk_analyst_agent,
        input=(
            "Assess risk levels per project based on CPI, SPI, CV, and SV, "
            "and propose corrective actions.\n\n"
            f"EVM JSON: {evms_result_json}"
        ),
        run_config=RunConfig(model=get_active_default_model()),
    )
    log_event("agent_result", "Risk Analyst response", {"text": result.final_output})
    return result.final_output


# =============================
# Orchestrator
# =============================
orchestrator_agent = Agent(
    name="EVM Orchestrator",
    instructions="""
    You orchestrate three tools: `get_ingestion_summary`, `get_evms_report`, and `get_risk_assessment`.
    Flow:
    1) Always parse/validate the CSV (string) first via `parse_and_validate_csv` or the ingestion tool.
    2) If valid, compute EVM metrics for the projects with `get_evms_report`.
    3) Then produce a risk assessment with `get_risk_assessment`.
    4) Merge into one final answer: Overview, Portfolio Totals, Per-Project Highlights, Risks & Actions.
    Keep responses concise and practical for PMs.
    """,
    tools=[get_ingestion_summary, get_evms_report, get_risk_assessment],
    tool_use_behavior="run_llm_again",
)


async def run_agent(csv_text: str, as_of_date: Optional[str]) -> str:
    # Provide the orchestrator enough context to decide the flow
    prompt = (
        "You will receive raw CSV text. Parse and validate it using tools, "
        "compute EVM, assess risks, and provide a single concise report.\n\n"
        f"AsOf: {as_of_date or date.today().isoformat()}\n"
        f"CSV:\n{csv_text}"
    )
    result = await Runner.run(
        orchestrator_agent,
        prompt,
        run_config=RunConfig(model=get_active_default_model()),
    )
    return result.final_output


# =============================
# Streamlit UI
# =============================
def main():
    st.set_page_config(page_title="EVM Assistant", page_icon="EV", layout="centered")
    inject_theme()
    st.title("EVM Assistant")
    # Load shareable URL params (models, thresholds) before building widgets
    _load_url_params_into_state()

    # Live model switchers (Gemini variants) and risk thresholds
    model_options = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]
    col_m1, col_m2, col_thr = st.columns([2,2,3])
    with col_m1:
        st.selectbox(
            "Agents' Model",
            options=model_options,
            index=(model_options.index(_base_model_name(DEFAULT_MODEL)) if _base_model_name(DEFAULT_MODEL) in model_options else 0),
            key="__model_default__",
            help="Model used by the orchestrator and most agents",
        )
    with col_m2:
        st.selectbox(
            "Summary Model",
            options=model_options,
            index=(model_options.index(_base_model_name(SUMMARY_MODEL)) if _base_model_name(SUMMARY_MODEL) in model_options else 1),
            key="__model_summary__",
            help="Faster model recommended for summaries",
        )

    with col_thr:
        st.caption("Risk thresholds")
        cpi_thr = st.slider("CPI low if <", min_value=0.5, max_value=1.0, value=float(RISK_CPI_THRESHOLD), step=0.01)
        spi_thr = st.slider("SPI low if <", min_value=0.5, max_value=1.0, value=float(RISK_SPI_THRESHOLD), step=0.01)
        globals()["RISK_CPI_THRESHOLD"] = float(cpi_thr)
        globals()["RISK_SPI_THRESHOLD"] = float(spi_thr)

    st.caption(f"Model: {get_active_default_model()}  |  Summary Model: {get_active_summary_model()}  |  Provider: {PROVIDER}  |  Thresholds: CPI<{cpi_thr}, SPI<{spi_thr}")
    # Update URL so users can share current model/threshold settings
    _set_url_params_safe(md=_base_model_name(get_active_default_model()), ms=_base_model_name(get_active_summary_model()), cpi=str(cpi_thr), spi=str(spi_thr))
    _reset_litellm_logging_worker()

    # Diagnostics panel
    with st.expander("Diagnostics", expanded=False):
        st.write("Environment (redacted)")
        st.json({
            "Provider": PROVIDER,
            "DefaultModel": get_active_default_model(),
            "SummaryModel": get_active_summary_model(),
            "OPENAI_API_KEY": _redact(os.getenv("OPENAI_API_KEY")),
            "OPENAI_PROJECT_ID": _redact(os.getenv("OPENAI_PROJECT_ID")),
            "OPENAI_ORG_ID": _redact(os.getenv("OPENAI_ORG_ID")),
            "GEMINI_API_KEY": _redact(os.getenv("GEMINI_API_KEY")),
            "LITELLM_LOG": os.getenv("LITELLM_LOG", "-"),
        })

        if st.button("Run model health check"):
            with st.spinner("Pinging selected model..."):
                ok, msg = _run_async_in_new_loop(_ping_model(get_active_default_model()))
            (st.success if ok else st.error)(msg)

    # Initialize persistent UI flags
    if "__trace_hide__" not in st.session_state:
        st.session_state["__trace_hide__"] = True
    if "__is_running__" not in st.session_state:
        st.session_state["__is_running__"] = False

    # Handle deferred reset BEFORE any widgets are created
    if st.session_state.get("__reset_requested__"):
        keys = [
            "__pending_file__","__pending_asof__","__should_run__evms__","__has_results__",
            "__last_csv__","__last_asof__","__last_agent_response__","__last_items__","__last_totals__",
            TRACE_KEY, TRACE_PH_KEY, "__rows_to_show__", "__uploader__", "__paste__"
        ]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["__trace_hide__"] = True
        st.session_state["__reset_requested__"] = False
        st.rerun()

    st.caption(
        "Upload a CSV matching the template headers. The assistant will compute EVM metrics "
        "and summarize portfolio health, risks, and suggestions."
    )

    # Template download + inline preview
    sample_text = get_sample_csv_text()
    st.download_button("Get CSV Template", data=sample_text, file_name="evms_sample.csv", mime="text/csv")
    with st.expander("See template headers", expanded=False):
        st.code(sample_text.splitlines()[0] + "\n...", language="csv")

    as_of_date = st.date_input("As-of date (optional)")
    mode = st.radio("Mode", ["Fast (local compute + 1 summary)", "Agentic (multi-agent with handoffs)"], index=1, help="Fast mode avoids extra agent hops for speed.")

    input_mode = st.radio("Input method", ["Upload CSV", "Paste CSV"], horizontal=True, key="__input_mode__")
    pasted_csv_text: Optional[str] = None
    file = None
    if input_mode == "Upload CSV":
        file = st.file_uploader("Upload CSV (project_template.csv format)", type=["csv"], key="__uploader__")
    else:
        pasted_csv_text = st.text_area(
            "Paste CSV content",
            value=get_sample_csv_text(),
            height=180,
            help="Paste the full CSV text here if upload is blocked (e.g., 403).",
            key="__paste__",
        )

    # Global reset (easy to find before running)
    cols_reset = st.columns([6,1])
    with cols_reset[1]:
        if st.button("Reset App"):
            # Defer reset to the next rerun to avoid touching widget state
            st.session_state["__reset_requested__"] = True
            st.rerun()

    # Use two-step pattern for reliable trace rendering
    with st.form("evms_form", clear_on_submit=False):
        submitted = st.form_submit_button("Run EVM Analysis")

    if submitted:
        if input_mode == "Upload CSV":
            st.session_state["__pending_file__"] = file.read().decode("utf-8") if file else ""
        else:
            st.session_state["__pending_file__"] = pasted_csv_text or ""
        st.session_state["__pending_asof__"] = as_of_date.isoformat() if as_of_date else None
        st.session_state["__should_run__evms__"] = True
        st.rerun()

    if st.session_state.get("__should_run__evms__"):
        clear_trace()
        csv_text = st.session_state.get("__pending_file__", "")
        as_of_str = st.session_state.get("__pending_asof__")

        if not csv_text:
            st.error("Please upload a CSV file.")
            st.session_state["__should_run__evms__"] = False
            return

        # User controls for trace presentation (outside the trace container)
        ctrl_col1, ctrl_col2 = st.columns([1,6])
        with ctrl_col1:
            # Only compact mode is supported; show a simple toggle to reveal trace
            # Disable while the analyzer is running to avoid disruptive reruns.
            is_running = True
            st.session_state["__is_running__"] = True
            show_trace = st.checkbox(
                "Show trace",
                value=not st.session_state.get("__trace_hide__", True),
                disabled=is_running,
                key="__show_trace_chk__",
            )
            st.session_state["__trace_hide__"] = not show_trace

        # Two-column layout: left = live trace (narrow), right = results (wide)
        if st.session_state.get("__trace_hide__", False):
            # Full-width results
            right = st.container()
            left = None
        else:
            left, right = st.columns([1, 6], gap="large")

        # Initialize live trace placeholder BEFORE starting the run (left side)
        if left is not None:
            with left:
                st.session_state[TRACE_PH_KEY] = st.container()
                _render_trace_into_placeholder()

        with right:
            with st.status("Analyzing portfolio...", expanded=True) as status:
                try:
                    if mode.startswith("Fast"):
                        # Offer header mapping if needed (robust header detection)
                        try:
                            _hdr_reader = csv.DictReader(io.StringIO(csv_text or ""))
                            file_headers = _hdr_reader.fieldnames or []
                        except Exception:
                            file_headers = []
                        missing = [h for h in EXPECTED_HEADERS if h not in file_headers]
                        if missing:
                            st.warning("Headers differ from the template. Map columns to continue.")
                            MISSING_SENTINEL = "<not present>"
                            options = [MISSING_SENTINEL, *file_headers]
                            mapping = {}
                            with st.form("header_map_form"):
                                for h in EXPECTED_HEADERS:
                                    default_index = options.index(h) if h in options else 0
                                    mapping[h] = st.selectbox(f"Map to '{h}'", options=options, index=default_index, key=f"map_{h}")
                                apply = st.form_submit_button("Apply Mapping & Continue")
                            if apply:
                                fsrc = io.StringIO(csv_text)
                                reader = csv.DictReader(fsrc)
                                fout = io.StringIO()
                                writer = csv.DictWriter(fout, fieldnames=EXPECTED_HEADERS)
                                writer.writeheader()
                                for row in reader:
                                    new_row = {}
                                    for exp in EXPECTED_HEADERS:
                                        src = mapping.get(exp)
                                        new_row[exp] = row.get(src, "") if src and src != MISSING_SENTINEL else ""
                                    writer.writerow(new_row)
                                csv_text = fout.getvalue()
                            else:
                                st.info("Submit the mapping form to proceed.")
                                st.session_state["__is_running__"] = False
                                return

                        # Pure local compute for speed; 1 LLM call for narrative.
                        items, totals, row_errors = _compute_portfolio_from_csv_for_ui(csv_text, as_of_str)
                        if row_errors:
                            st.warning("Some rows had issues (bad date/number). Showing first 5:")
                            for e in row_errors[:5]:
                                st.text(e)
                        # Build a compact risk list
                        risks = []
                        for it in items:
                            lvl, reasons = _risk_level_and_reasons(it)
                            if lvl in ("medium", "high"):
                                risks.append({
                                    "ProjectID": it.get("ProjectID"),
                                    "ProjectName": it.get("ProjectName"),
                                    "level": lvl,
                                    "reasons": reasons[:3],
                                })
                        # One summary pass via the LLM
                        summary_prompt = (
                        "Create a short overview of portfolio EVM results.\n"
                            f"Totals: {json.dumps(totals)}\n"
                            f"Risks: {json.dumps(risks[:8])}\n"
                            "Return 4-6 bullets max, with concrete guidance."
                        )
                        summary = asyncio.run(
                            Runner.run(
                                summary_agent,
                                summary_prompt,
                                run_config=RunConfig(model=get_active_summary_model()),
                            )
                        )
                        agent_response = summary.final_output
                        st.session_state["__last_agent_response__"] = agent_response
                    else:
                        agent_response = asyncio.run(run_agent(csv_text, as_of_str))
                        st.session_state["__last_agent_response__"] = agent_response
                finally:
                    st.session_state["__is_running__"] = False
                status.update(label="Done", state="complete")

            # If trace is hidden, show a compact run summary (agents and tools)
            if st.session_state.get("__trace_hide__", False):
                if mode.startswith("Agentic"):
                    used_agents = [
                        {"name": "EVM Orchestrator", "tools": ["get_ingestion_summary", "get_evms_report", "get_risk_assessment"]},
                        {"name": "Ingestion Agent", "tools": ["parse_and_validate_csv"]},
                        {"name": "EVM Calculator Agent", "tools": ["compute_evms_from_csv", "compute_evms"]},
                        {"name": "Risk Analyst Agent", "tools": ["assess_risks"]},
                        {"name": "Project Q&A Agent", "tools": ["data-only answers"]},
                    ]
                else:
                    used_agents = [
                        {"name": "Summary Agent", "tools": ["LLM summary"]},
                    ]
                st.markdown("<div class='banner-gradient' style='margin-bottom:8px'><b>Run Summary</b></div>", unsafe_allow_html=True)
                st.write(f"Agents involved: {len(used_agents)}")
                for a in used_agents:
                    tools_txt = ", ".join(a["tools"]) if a.get("tools") else "-"
                    st.markdown(f"- <b>{a['name']}</b> - tools: {tools_txt}", unsafe_allow_html=True)

            st.markdown("### Final Report")
            st.write(agent_response)
            rep_col1, rep_col2 = st.columns([2,6])
            with rep_col1:
                st.download_button(
                    "Download report (Markdown)",
                    data=(st.session_state.get("__last_agent_response__") or agent_response or "").strip() or "Report unavailable.",
                    file_name="evm_report.md",
                    mime="text/markdown",
                    key="__dl_report_md_top__",
                )

            # Local colored table for quick visual scan
            try:
                if mode.startswith("Fast"):
                    # Already computed above
                    pass
                else:
                    items, totals, _ = _compute_portfolio_from_csv_for_ui(csv_text, as_of_str)

                # Heatmap ahead of filters for quick scan
                st.markdown("### Portfolio Heatmap")
                render_cpi_spi_heatmap(items)

                # Filters
                st.markdown("### Filters")
                pms = sorted({(it.get("PM") or "").strip() for it in items if (it.get("PM") or "").strip()})
                depts = sorted({(it.get("Dept") or "").strip() for it in items if (it.get("Dept") or "").strip()})
                statuses = sorted({(it.get("Status") or "").strip() for it in items if (it.get("Status") or "").strip()})

                col1, col2, col3 = st.columns(3)
                with col1:
                    default_pms = st.session_state.get("__flt_pms__", pms)
                    sel_pms = st.multiselect("PM", options=pms, default=default_pms)
                with col2:
                    default_depts = st.session_state.get("__flt_depts__", depts)
                    sel_depts = st.multiselect("Dept", options=depts, default=default_depts)
                with col3:
                    default_status = st.session_state.get("__flt_status__", statuses)
                    sel_status = st.multiselect("Status", options=statuses, default=default_status)

                def apply_filters(rows: List[Dict[str, Any]]):
                    flt = []
                    for r in rows:
                        if sel_pms and (r.get("PM") or "") not in sel_pms:
                            continue
                        if sel_depts and (r.get("Dept") or "") not in sel_depts:
                            continue
                        if sel_status and (r.get("Status") or "") not in sel_status:
                            continue
                        flt.append(r)
                    return flt

                filtered_items = apply_filters(items)
                # Recompute totals on filtered set
                def sumf_local(key: str) -> float:
                    return round(sum((x.get(key) or 0.0) for x in filtered_items), 2)

                filtered_totals = {
                    "BAC": sumf_local("BAC"),
                    "PV": sumf_local("PV"),
                    "EV": sumf_local("EV"),
                    "AC": sumf_local("AC"),
                    "CPI": round((sumf_local("EV") / sumf_local("AC")), 3) if sumf_local("AC") > 0 else None,
                    "SPI": round((sumf_local("EV") / sumf_local("PV")), 3) if sumf_local("PV") > 0 else None,
                    "CV": round(sumf_local("EV") - sumf_local("AC"), 2),
                    "SV": round(sumf_local("EV") - sumf_local("PV"), 2),
                    "AsOf": totals.get("AsOf"),
                }
                # Quick exports placed just under the caption for visibility
                exp_a, exp_b = st.columns(2)
                with exp_a:
                    st.download_button(
                        "Download report (Markdown)",
                        data=(st.session_state.get("__last_agent_response__") or ""),
                        file_name="evm_report.md",
                        mime="text/markdown",
                        key="__dl_report_md_caption__",
                        use_container_width=True,
                    )
                with exp_b:
                    import io as _io2, csv as _csv2
                    cols2 = [
                        "ProjectID","ProjectName","PM","Dept","Status","StartDate",
                        "OriginalEndDate","ExpectedEndDate","BAC","PV","EV","AC",
                        "CPI","SPI","CV","SV","EAC","ETC","Risk",
                    ]
                    def _to_rows2(rows):
                        out=[]
                        for it in rows:
                            rk,_ = _risk_level_and_reasons(it)
                            row={c: it.get(c) for c in cols2}
                            row["Risk"]=rk
                            out.append(row)
                        return out
                    buf2=_io2.StringIO()
                    w2=_csv2.DictWriter(buf2, fieldnames=cols2)
                    w2.writeheader()
                    for r in _to_rows2(filtered_items):
                        w2.writerow({k: ("" if v is None else v) for k,v in r.items()})
                    st.download_button(
                        "Download filtered table (CSV)",
                        data=buf2.getvalue(),
                        file_name="evm_filtered.csv",
                        mime="text/csv",
                        key="__dl_filtered_csv_caption__",
                        use_container_width=True,
                    )

                # Pagination / load more controls
                page_size_default = 50
                rows_to_show = st.session_state.get("__rows_to_show__", page_size_default)
                rows_to_show = max(page_size_default, rows_to_show)
                total_rows = len(filtered_items)
                st.caption(f"Showing {min(rows_to_show, total_rows)} of {total_rows} projects")
                col_a, col_b, col_c = st.columns([1,1,6])
                with col_a:
                    if rows_to_show < total_rows and st.button("Load more"):
                        st.session_state["__rows_to_show__"] = rows_to_show + page_size_default
                        st.rerun()
                with col_b:
                    if total_rows > page_size_default and st.button("Reset view"):
                        st.session_state["__rows_to_show__"] = page_size_default
                        st.rerun()

                st.markdown("### Computed Metrics")
                items_page = filtered_items[:rows_to_show]
                render_evms_colored_table(items_page, filtered_totals)
                # Update URL with filter state for shareable links
                _set_url_params_safe(
                    md=_base_model_name(get_active_default_model()),
                    ms=_base_model_name(get_active_summary_model()),
                    cpi=str(RISK_CPI_THRESHOLD),
                    spi=str(RISK_SPI_THRESHOLD),
                    pm=",".join(sel_pms),
                    dept=",".join(sel_depts),
                    status=",".join(sel_status),
                )

                # Exports: report markdown and filtered data CSV
                st.markdown("### Export")
                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    report_md = (st.session_state.get("__last_agent_response__") or "").strip()
                    st.download_button(
                        "Download report (Markdown)",
                        data=report_md if report_md else "Report unavailable.",
                        file_name="evm_report.md",
                        mime="text/markdown",
                    )
                with export_col2:
                    # Convert filtered rows to CSV with the same columns as the table
                    cols = [
                        "ProjectID", "ProjectName", "PM", "Dept", "Status", "StartDate",
                        "OriginalEndDate", "ExpectedEndDate",
                        "BAC", "PV", "EV", "AC", "CPI", "SPI", "CV", "SV", "EAC", "ETC", "Risk",
                    ]
                    def to_rows(rows: List[Dict[str, Any]]):
                        out = []
                        for it in rows:
                            risk_level, _ = _risk_level_and_reasons(it)
                            row = {c: it.get(c) for c in cols}
                            row["Risk"] = risk_level
                            out.append(row)
                        return out
                    import io as _io, csv as _csv
                    buf = _io.StringIO()
                    w = _csv.DictWriter(buf, fieldnames=cols)
                    w.writeheader()
                    for r in to_rows(filtered_items):
                        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
                    st.download_button(
                        "Download filtered table (CSV)",
                        data=buf.getvalue(),
                        file_name="evm_filtered.csv",
                        mime="text/csv",
                    )
                with st.expander("What's this?", expanded=False):
                    st.markdown(
                        "- CPI: Cost Performance Index = EV / AC. < 1.0 means over cost.\n"
                        "- SPI: Schedule Performance Index = EV / PV. < 1.0 means behind schedule.\n"
                        "- EAC: Estimate At Completion = BAC / CPI.\n"
                        "- ETC: Estimate To Complete = EAC - AC.\n"
                        f"- Risk thresholds in use: CPI < {RISK_CPI_THRESHOLD}, SPI < {RISK_SPI_THRESHOLD}."
                    )

                with st.expander("Formulas used", expanded=False):
                    st.markdown("""
                    - BAC: PlannedCost
                    - Planned %: days elapsed from StartDate to the As-of date, divided by total planned days (StartDate -> End); capped between 0% and 100%. End = ExpectedEndDate if provided, otherwise OriginalEndDate
                    - PV: BAC x Planned %
                    - EV: BAC x ProgressPercent_Manual/100
                    - AC: ActualCost
                    - CV: EV - AC
                    - SV: EV - PV
                    - CPI: EV / AC (if AC > 0)
                    - SPI: EV / PV (if PV > 0)
                    - EAC: BAC / CPI (if CPI > 0)
                    - ETC: EAC - AC
                    """)

                st.markdown("### Ask Rowshni a project related question")
                with st.form("qa_form", clear_on_submit=True):
                    question = st.text_input(
                        "Question about these projects (answers use only this data)",
                        placeholder="e.g., How many projects are over budget?",
                    )
                    st.caption("Examples: Which projects have SPI < 0.9? | Total AC by Dept | List projects with CV < 0")
                    ask = st.form_submit_button("Ask")
                if ask and question.strip():
                    qa_prompt = (
                        "Answer strictly using this data. If unknown, say so.\n\n"
                        f"Totals: {json.dumps(filtered_totals)}\n"
                        f"Items: {json.dumps(filtered_items)}\n"
                        f"Question: {question.strip()}"
                    )
                    qa_result = asyncio.run(
                        Runner.run(
                            qa_agent,
                            qa_prompt,
                            run_config=RunConfig(model=get_active_default_model()),
                        )
                    )
                    resp = (qa_result.final_output or "").strip()
                    if "OUT_OF_SCOPE" in resp:
                        st.info("Rowshni can only answer questions related to the projects data provided")
                    else:
                        st.write(resp)
            except Exception as e:
                st.warning(f"Could not render metrics table: {e}")

        if left is not None:
            with left:
                _render_trace_into_placeholder()

        # Persist results for re-rendering across reruns (e.g., toggling trace)
        st.session_state["__last_csv__"] = csv_text
        st.session_state["__last_asof__"] = as_of_str
        st.session_state["__last_agent_response__"] = agent_response
        try:
            st.session_state["__last_items__"] = items
            st.session_state["__last_totals__"] = totals
        except Exception:
            st.session_state["__last_items__"] = []
            st.session_state["__last_totals__"] = {}
        st.session_state["__has_results__"] = True
        st.session_state["__should_run__evms__"] = False
        # Refresh UI once run completes so controls (like Show trace) become enabled
        st.rerun()

    elif st.session_state.get("__has_results__"):
        # Re-render the last results without re-running the agent
        csv_text = st.session_state.get("__last_csv__", "")
        as_of_str = st.session_state.get("__last_asof__")
        agent_response = st.session_state.get("__last_agent_response__", "")
        items = st.session_state.get("__last_items__", [])
        totals = st.session_state.get("__last_totals__", {})

        ctrl_col1, _ = st.columns([1,6])
        with ctrl_col1:
            show_trace = st.checkbox(
                "Show trace",
                value=not st.session_state.get("__trace_hide__", True),
                disabled=st.session_state.get("__is_running__", False),
                key="__show_trace_chk__",
            )
            st.session_state["__trace_hide__"] = not show_trace

        if st.session_state.get("__trace_hide__", False):
            right = st.container()
            left = None
        else:
            left, right = st.columns([1, 6], gap="large")

        if left is not None:
            with left:
                st.session_state[TRACE_PH_KEY] = st.container()
                _render_trace_into_placeholder()

        with right:
            if st.session_state.get("__trace_hide__", False):
                if mode.startswith("Agentic"):
                    used_agents = [
                        {"name": "EVM Orchestrator", "tools": ["get_ingestion_summary", "get_evms_report", "get_risk_assessment"]},
                        {"name": "Ingestion Agent", "tools": ["parse_and_validate_csv"]},
                        {"name": "EVM Calculator Agent", "tools": ["compute_evms_from_csv", "compute_evms"]},
                        {"name": "Risk Analyst Agent", "tools": ["assess_risks"]},
                        {"name": "Project Q&A Agent", "tools": ["data-only answers"]},
                    ]
                else:
                    used_agents = [
                        {"name": "Summary Agent", "tools": ["LLM summary"]},
                    ]
                st.markdown("<div class='banner-gradient' style='margin-bottom:8px'><b>Run Summary</b></div>", unsafe_allow_html=True)
                st.write(f"Agents involved: {len(used_agents)}")
                for a in used_agents:
                    tools_txt = ", ".join(a["tools"]) if a.get("tools") else "-"
                    st.markdown(f"- <b>{a['name']}</b> - tools: {tools_txt}", unsafe_allow_html=True)

            st.markdown("### Final Report")
            st.write(agent_response)

            try:
                # Heatmap first, then filters
                st.markdown("### Portfolio Heatmap")
                render_cpi_spi_heatmap(items)

                st.markdown("### Filters")
                pms = sorted({(it.get("PM") or "").strip() for it in items if (it.get("PM") or "").strip()})
                depts = sorted({(it.get("Dept") or "").strip() for it in items if (it.get("Dept") or "").strip()})
                statuses = sorted({(it.get("Status") or "").strip() for it in items if (it.get("Status") or "").strip()})

                col1, col2, col3 = st.columns(3)
                with col1:
                    sel_pms = st.multiselect("PM", options=pms, default=pms)
                with col2:
                    sel_depts = st.multiselect("Dept", options=depts, default=depts)
                with col3:
                    sel_status = st.multiselect("Status", options=statuses, default=statuses)

                def apply_filters(rows: List[Dict[str, Any]]):
                    flt = []
                    for r in rows:
                        if sel_pms and (r.get("PM") or "") not in sel_pms:
                            continue
                        if sel_depts and (r.get("Dept") or "") not in sel_depts:
                            continue
                        if sel_status and (r.get("Status") or "") not in sel_status:
                            continue
                        flt.append(r)
                    return flt

                filtered_items = apply_filters(items)
                # Recompute totals on filtered set
                def sumf_local(key: str) -> float:
                    return round(sum((x.get(key) or 0.0) for x in filtered_items), 2)

                filtered_totals = {
                    "BAC": sumf_local("BAC"),
                    "PV": sumf_local("PV"),
                    "EV": sumf_local("EV"),
                    "AC": sumf_local("AC"),
                    "CPI": round((sumf_local("EV") / sumf_local("AC")), 3) if sumf_local("AC") > 0 else None,
                    "SPI": round((sumf_local("EV") / sumf_local("PV")), 3) if sumf_local("PV") > 0 else None,
                    "CV": round(sumf_local("EV") - sumf_local("AC"), 2),
                    "SV": round(sumf_local("EV") - sumf_local("PV"), 2),
                    "AsOf": totals.get("AsOf"),
                }

                # Pagination controls
                page_size_default = 50
                rows_to_show = st.session_state.get("__rows_to_show__", page_size_default)
                rows_to_show = max(page_size_default, rows_to_show)
                total_rows = len(filtered_items)
                st.caption(f"Showing {min(rows_to_show, total_rows)} of {total_rows} projects")
                col_a, col_b, col_c = st.columns([1,1,6])
                with col_a:
                    if rows_to_show < total_rows and st.button("Load more"):
                        st.session_state["__rows_to_show__"] = rows_to_show + page_size_default
                        st.rerun()
                with col_b:
                    if total_rows > page_size_default and st.button("Reset view"):
                        st.session_state["__rows_to_show__"] = page_size_default
                        st.rerun()

                st.markdown("### Computed Metrics")
                items_page = filtered_items[:rows_to_show]
                render_evms_colored_table(items_page, filtered_totals)

                with st.expander("Formulas used", expanded=False):
                    st.markdown("""
                    - BAC: PlannedCost
                    - Planned %: days elapsed from StartDate to the As-of date, divided by total planned days (StartDate -> End); capped between 0% and 100%. End = ExpectedEndDate if provided, otherwise OriginalEndDate
                    - PV: BAC x Planned %
                    - EV: BAC x ProgressPercent_Manual/100
                    - AC: ActualCost
                    - CV: EV - AC
                    - SV: EV - PV
                    - CPI: EV / AC (if AC > 0)
                    - SPI: EV / PV (if PV > 0)
                    - EAC: BAC / CPI (if CPI > 0)
                    - ETC: EAC - AC
                    """)

                st.markdown("### Ask Rowshni a project related question")
                with st.form("qa_form2", clear_on_submit=True):
                    question = st.text_input(
                        "Question about these projects (answers use only this data)",
                        placeholder="e.g., How many projects are over budget?",
                    )
                    st.caption("Examples: Which projects have SPI < 0.9? | Total AC by Dept | List projects with CV < 0")
                    ask = st.form_submit_button("Ask")
                if ask and question.strip():
                    qa_prompt = (
                        "Answer strictly using this data. If unknown, say so.\n\n"
                        f"Totals: {json.dumps(filtered_totals)}\n"
                        f"Items: {json.dumps(filtered_items)}\n"
                        f"Question: {question.strip()}"
                    )
                    qa_result = asyncio.run(
                        Runner.run(
                            qa_agent,
                            qa_prompt,
                            run_config=RunConfig(model=get_active_default_model()),
                        )
                    )
                    resp = (qa_result.final_output or "").strip()
                    if "OUT_OF_SCOPE" in resp:
                        st.info("Rowshni can only answer questions related to the projects data provided")
                    else:
                        st.write(resp)
            except Exception as e:
                st.warning(f"Could not render metrics table: {e}")

        if left is not None:
            with left:
                _render_trace_into_placeholder()


if __name__ == "__main__":
    main()




