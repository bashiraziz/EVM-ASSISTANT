"""
EVM Assistant (Streamlit UI)

This UI is now thin: all logic, agents, tools, and UI helpers live under `evm_app/`.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
from datetime import date
from typing import Any, Dict, List, Optional

import streamlit as st
from agents import Runner
from agents.run import RunConfig

import evm_app.config as app_config
from evm_app.agents.orchestrator import run_agent
from evm_app.agents.qa_agent import qa_agent
from evm_app.agents.summary_agent import summary_agent
from evm_app.config import (
    DEFAULT_MODEL,
    PROVIDER,
    SUMMARY_MODEL,
    _base_model_name,
    get_active_default_model,
    get_active_summary_model,
    load_url_params_into_state,
    reset_litellm_logging_worker,
    set_url_params_safe,
)
from evm_app.tools.csv_tools import EXPECTED_HEADERS
from evm_app.tools.evm_tools import compute_portfolio_for_ui, risk_level_and_reasons
from evm_app.ui.diagnostics import render_diagnostics_panel
from evm_app.ui.tables import render_cpi_spi_heatmap, render_evms_colored_table
from evm_app.ui.theme import inject_theme
from evm_app.ui.trace import (
    TRACE_KEY,
    TRACE_PH_KEY,
    _render_trace_into_placeholder as render_trace_placeholder,
    clear_trace,
)

# Disable/neutralize LiteLLM's background LoggingWorker to avoid event-loop issues/warnings in Streamlit
try:
    import litellm.litellm_core_utils.logging_worker as _lw
    import asyncio

    def _noop_start(self):
        return None

    def _consume_or_schedule(self, async_coroutine):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Fire-and-forget on the active loop to avoid 'never awaited' warnings
                loop.create_task(async_coroutine)
                return
        except Exception:
            pass
        # If we cannot schedule it, close the coroutine to silence warnings
        try:
            async_coroutine.close()
        except Exception:
            pass

    _lw.LoggingWorker.start = _noop_start  # type: ignore[assignment]
    _lw.LoggingWorker.ensure_initialized_and_enqueue = _consume_or_schedule  # type: ignore[assignment]
except Exception:
    pass

# =============================
# Helpers
# =============================
SAMPLE_REL_PATH = os.path.join("samples", "evms_sample.csv")


def get_sample_csv_text() -> str:
    header = ",".join(EXPECTED_HEADERS)
    default_sample = (
        f"{header}\n"
        "P1001,New Website,Acme Corp,alice@acme.com,Sam,IT,100000,45000,2025-01-01,2025-12-31,Active,40,Fixed Price\n"
        "P1002,Mobile App,Globex,bob@globex.com,Kim,Product,200000,130000,2025-02-01,2025-11-30,Active,55,Time & Materials\n"
        "P1003,Data Migration,Initech,carol@initech.com,Raj,Ops,150000,90000,2025-03-15,2025-09-15,Active,35,Fixed Price\n"
    )
    try:
        if os.path.exists(SAMPLE_REL_PATH):
            with open(SAMPLE_REL_PATH, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return default_sample


def _maybe_map_headers_ui(csv_text: str) -> str:
    """If headers don't match EXPECTED_HEADERS, provide a simple mapping UI.

    Returns a CSV text with headers mapped to EXPECTED_HEADERS when the user submits.
    If mapping isn't applied, returns original csv_text.
    """
    try:
        f = io.StringIO(csv_text)
        reader = csv.DictReader(f)
        options = list(reader.fieldnames or [])
        missing = [h for h in EXPECTED_HEADERS if h not in options]
        if not missing:
            return csv_text

        st.warning("CSV headers do not match the expected template. Map columns below.")
        MISSING = "<missing>"
        mapping: Dict[str, str] = {}
        with st.form("header_map_form"):
            for h in EXPECTED_HEADERS:
                default_idx = options.index(h) if h in options else None
                choices = options + ([MISSING] if h not in options else [])
                mapping[h] = st.selectbox(
                    f"Map to '{h}'", options=choices, index=(default_idx or 0), key=f"map_{h}"
                )
            apply = st.form_submit_button("Apply Mapping & Continue")
        if not apply:
            st.stop()  # wait for submit

        # Rewrite CSV with mapped headers
        fsrc = io.StringIO(csv_text)
        rdr = csv.DictReader(fsrc)
        fout = io.StringIO()
        w = csv.DictWriter(fout, fieldnames=EXPECTED_HEADERS)
        w.writeheader()
        for row in rdr:
            new_row = {}
            for exp in EXPECTED_HEADERS:
                src = mapping.get(exp)
                new_row[exp] = row.get(src, "") if src and src != MISSING else ""
            w.writerow(new_row)
        return fout.getvalue()
    except Exception:
        return csv_text


# =============================
# Streamlit UI
# =============================
def main():
    st.set_page_config(page_title="EVM Assistant", page_icon="EV", layout="centered")
    inject_theme()
    st.title("EVM Assistant")

    # Load shareable URL params (models, thresholds)
    load_url_params_into_state()

    # Model pickers and thresholds
    model_options = ["gemini-1.5-pro", "gemini-1.5-flash"]
    col_m1, col_m2, col_thr = st.columns([2, 2, 3])
    with col_m1:
        st.selectbox(
            "Agents' Model",
            options=model_options,
            index=(
                model_options.index(_base_model_name(DEFAULT_MODEL))
                if _base_model_name(DEFAULT_MODEL) in model_options
                else 0
            ),
            key="__model_default__",
            help="Model used by the orchestrator and most agents",
        )
    with col_m2:
        st.selectbox(
            "Summary Model",
            options=model_options,
            index=(
                model_options.index(_base_model_name(SUMMARY_MODEL))
                if _base_model_name(SUMMARY_MODEL) in model_options
                else 1
            ),
            key="__model_summary__",
            help="Faster model recommended for summaries",
        )
    with col_thr:
        st.caption("Risk thresholds")
        cpi_thr = st.slider(
            "CPI low if <", min_value=0.5, max_value=1.0, value=float(app_config.RISK_CPI_THRESHOLD), step=0.01
        )
        spi_thr = st.slider(
            "SPI low if <", min_value=0.5, max_value=1.0, value=float(app_config.RISK_SPI_THRESHOLD), step=0.01
        )
        app_config.RISK_CPI_THRESHOLD = float(cpi_thr)
        app_config.RISK_SPI_THRESHOLD = float(spi_thr)

    st.caption(
        f"Model: {get_active_default_model()}  |  Summary Model: {get_active_summary_model()}  |  Provider: {PROVIDER}  |  Thresholds: CPI<{cpi_thr}, SPI<{spi_thr}"
    )
    set_url_params_safe(
        md=_base_model_name(get_active_default_model()),
        ms=_base_model_name(get_active_summary_model()),
        cpi=str(cpi_thr),
        spi=str(spi_thr),
    )
    reset_litellm_logging_worker()

    # Diagnostics
    render_diagnostics_panel()

    st.caption(
        "Upload a CSV matching the template headers to compute EVM metrics and summarize portfolio health."
    )
    sample_text = get_sample_csv_text()
    st.download_button("Get CSV Template", data=sample_text, file_name="evms_sample.csv", mime="text/csv")
    with st.expander("See template headers", expanded=False):
        st.code(sample_text.splitlines()[0] + "\n...", language="csv")

    as_of_date = st.date_input("As-of date (optional)")
    mode = st.radio(
        "Mode",
        ["Fast (local compute + 1 summary)", "Agentic (multi-agent with handoffs)"],
        index=1,
        help="Fast mode avoids extra agent hops for speed.",
    )

    input_mode = st.radio("Input method", ["Upload CSV", "Paste CSV"], horizontal=True)
    pasted_csv_text: Optional[str] = None
    file = None
    if input_mode == "Upload CSV":
        file = st.file_uploader("Upload CSV (project_template.csv format)", type=["csv"])
    else:
        pasted_csv_text = st.text_area(
            "Paste CSV content",
            value=get_sample_csv_text(),
            height=180,
        )

    # Submit
    if st.button("Run EVM Analysis"):
        csv_text = (file.read().decode("utf-8") if file else pasted_csv_text) or ""
        if not csv_text:
            st.error("Please provide CSV input.")
            st.stop()

        # Optional mapping if headers differ
        csv_text = _maybe_map_headers_ui(csv_text)

        clear_trace()
        show_trace = st.checkbox("Show trace", value=False)
        left, right = (st.columns([1, 6], gap="large") if show_trace else (None, st.container()))
        if left is not None:
            with left:
                st.session_state[TRACE_PH_KEY] = st.container()
                render_trace_placeholder()

        with right:
            as_of_str = as_of_date.isoformat() if as_of_date else None

            if mode.startswith("Fast"):
                items, totals, row_errors = compute_portfolio_for_ui(csv_text, as_of_str)
                if row_errors:
                    st.warning("Some rows had issues. Showing first 5:")
                    for e in row_errors[:5]:
                        st.text(e)

                # One short LLM summary
                risks = []
                for it in items:
                    lvl, reasons = risk_level_and_reasons(it)
                    if lvl in ("medium", "high"):
                        risks.append(
                            {
                                "ProjectID": it.get("ProjectID"),
                                "ProjectName": it.get("ProjectName"),
                                "level": lvl,
                                "reasons": reasons[:3],
                            }
                        )
                prompt = (
                    "Create a short overview of portfolio EVM results.\n"
                    f"Totals: {json.dumps(totals)}\n"
                    f"Risks: {json.dumps(risks[:8])}\n"
                    "Return 4-6 concise bullets with actionable guidance."
                )
                summary = asyncio.run(
                    Runner.run(
                        summary_agent, prompt, run_config=RunConfig(model=get_active_summary_model())
                    )
                )
                agent_response = (summary.final_output or "").strip()
            else:
                agent_response = asyncio.run(run_agent(csv_text, as_of_str))

            st.markdown("### Final Report")
            st.write(agent_response or "(no response)")
            st.download_button(
                "Download report (Markdown)",
                data=(agent_response or "Report unavailable.").strip(),
                file_name="evm_report.md",
                mime="text/markdown",
            )

            # Visuals
            st.markdown("### Portfolio Heatmap")
            items, totals, _ = compute_portfolio_for_ui(csv_text, as_of_str)
            render_cpi_spi_heatmap(items)

            st.markdown("### Computed Metrics")
            render_evms_colored_table(items[:200], totals)

            # Simple Q&A (data-only)
            st.markdown("### Ask a question about these projects")
            q = st.text_input(
                "Question (answers use only this data)",
                placeholder="e.g., How many projects are over budget?",
            )
            if st.button("Ask") and q.strip():
                qa_prompt = (
                    "Answer strictly using this data. If unknown, say so.\n\n"
                    f"Totals: {json.dumps(totals)}\n"
                    f"Items: {json.dumps(items)}\n"
                    f"Question: {q.strip()}"
                )
                ans = asyncio.run(
                    Runner.run(
                        qa_agent, qa_prompt, run_config=RunConfig(model=get_active_default_model())
                    )
                )
                resp = (ans.final_output or "").strip()
                if "OUT_OF_SCOPE" in resp:
                    st.info("The assistant can only answer questions about the uploaded data.")
                else:
                    st.write(resp)


if __name__ == "__main__":
    main()
