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
import random
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
from evm_app.ui.tables import render_cpi_spi_heatmap, render_evms_colored_table, render_totals_chips
from evm_app.ui.theme import inject_theme
from evm_app.ui.progress import render_sidebar as render_progress_sidebar, reset_progress, add_step, update_step
# Trace UI removed per request; core logging remains internal

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


# History persistence removed per request


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
    # Anchors for in-page links
    st.markdown("<a name='top'></a>", unsafe_allow_html=True)
    st.title("EVM Assistant")
    # Subtle pill-style link buttons
    st.markdown(
        """
        <style>
        .link-btn {display:inline-block; padding:6px 12px; border:1px solid rgba(255,255,255,0.25);
                   border-radius:999px; text-decoration:none; color:inherit; font-size:0.9rem;}
        .link-btn:hover {background:rgba(255,255,255,0.06); text-decoration:none;}
        .link-row {display:flex; gap:10px; align-items:center;}
        .link-right {text-align:right; padding-top:6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load shareable URL params (models, thresholds)
    load_url_params_into_state()
    # Handle reset via URL param (?reset=1) so we can style it like a link
    try:
        qp = dict(st.query_params)
    except Exception:
        qp = st.experimental_get_query_params() or {}
    if (qp.get("reset") or [None])[0] in ("1", 1, True, "true"):
        for k in ("evms_items", "evms_totals", "evms_report", "evms_as_of"):
            st.session_state.pop(k, None)
        # Reapply URL params without 'reset'
        try:
            # Preserve current md/ms/cpi/spi
            md = st.session_state.get("__model_default__")
            ms = st.session_state.get("__model_summary__")
            cpi = qp.get("cpi") if isinstance(qp.get("cpi"), str) else None
            spi = qp.get("spi") if isinstance(qp.get("spi"), str) else None
            clean = {}
            if md:
                clean["md"] = md
            if ms:
                clean["ms"] = ms
            if cpi:
                clean["cpi"] = cpi
            if spi:
                clean["spi"] = spi
            if clean:
                try:
                    for k, v in clean.items():
                        st.query_params[k] = v
                    # Remove reset key if available
                    try:
                        del st.query_params["reset"]
                    except Exception:
                        pass
                except Exception:
                    st.experimental_set_query_params(**clean)
            else:
                # Clear all query params if nothing to preserve
                try:
                    st.query_params.clear()
                except Exception:
                    st.experimental_set_query_params()
        except Exception:
            pass

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
    # Always render progress (if any) in the sidebar
    render_progress_sidebar()

    # Sidebar Q&A will be rendered once at the end after results are available

    # History UI removed

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

    # Helpers: reset state + quick links
    col_links1, col_reset = st.columns([3, 1])
    with col_links1:
        st.markdown("<div class='link-row'><a class='link-btn' href='#results'>Scroll to results</a></div>", unsafe_allow_html=True)
    with col_reset:
        # Reset as a pill-style link that triggers a rerun with ?reset=1
        st.markdown("<div class='link-right'><a class='link-btn' href='?reset=1#top'>Reset</a></div>", unsafe_allow_html=True)

    # A dedicated results area that renders BELOW inputs and the button
    st.markdown("<a name='results'></a>", unsafe_allow_html=True)
    results_area = st.container()

    # Submit + Back to top link side-by-side
    btn_col, top_col = st.columns([1, 1])
    with btn_col:
        run_clicked = st.button("Run EVM Analysis")
    with top_col:
        st.markdown("<div class='link-right'><a class='link-btn' href='#top'>Back to top</a></div>", unsafe_allow_html=True)

    if run_clicked:
        csv_text = (file.read().decode("utf-8") if file else pasted_csv_text) or ""
        if not csv_text:
            st.error("Please provide CSV input.")
            st.stop()

        # Optional mapping if headers differ
        csv_text = _maybe_map_headers_ui(csv_text)
        # Proceed to main rendering

        with results_area:
            # Live progress panel to keep users engaged
            try:
                status = st.status("Working", expanded=True)
                status.write("Parsing CSV and validating headers…")
            except Exception:
                status = None
            as_of_str = as_of_date.isoformat() if as_of_date else None

            if mode.startswith("Fast"):
                if status:
                    status.update(label="Computing EVM metrics…", state="running")
                reset_progress()
                p1 = add_step("Compute metrics locally", "running")
                items, totals, row_errors = compute_portfolio_for_ui(csv_text, as_of_str)
                update_step(p1, "done")
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
                if status:
                    status.update(label="Generating summary…", state="running")
                summary = asyncio.run(
                    Runner.run(
                        summary_agent,
                        prompt,
                        run_config=RunConfig(model=get_active_summary_model(), tracing_disabled=True),
                    )
                )
                agent_response = (summary.final_output or "").strip()
                # Persist computed results for later (visuals + Q&A)
                st.session_state["evms_items"] = items
                st.session_state["evms_totals"] = totals
                st.session_state["evms_report"] = agent_response
                st.session_state["evms_as_of"] = as_of_str
            else:
                if status:
                    status.update(label="Running multi-agent workflow…", state="running")
                # Sidebar progress with friendly agent names
                reset_progress()
                s1 = add_step("Ingestion Agent — Parse & validate CSV", "running")
                # Inline live sidebar placeholders for immediate feedback
                try:
                    with st.sidebar:
                        st.markdown("### Working")
                        sb_p1 = st.empty()
                        sb_p2 = st.empty()
                        sb_p3 = st.empty()
                        sb_p1.write("○ Ingestion Agent — Parse & validate CSV")
                        sb_p2.write("• Pending: EVM Calculator — Compute portfolio metrics")
                        sb_p3.write("• Pending: Risk Analyst — Assess risks")
                except Exception:
                    sb_p1 = sb_p2 = sb_p3 = None
                from evm_app.agents.ingestion_agent import ingestion_agent
                ing_res = asyncio.run(
                    Runner.run(
                        ingestion_agent,
                        input=(
                            "Parse and validate this CSV content against the expected headers. "
                            "If valid, briefly summarize row count and note any extra headers. "
                            "If invalid, list missing headers.\n\nCSV:\n" + csv_text
                        ),
                        run_config=RunConfig(model=get_active_default_model(), tracing_disabled=True),
                    )
                )
                ing = (ing_res.final_output or "").strip()
                update_step(s1, "done")
                if sb_p1:
                    sb_p1.write("● Ingestion Agent — Done")

                s2 = add_step("EVM Calculator Agent — Compute portfolio metrics", "running")
                from evm_app.agents.evms_calculator_agent import evms_calculator_agent
                if sb_p2:
                    sb_p2.write("○ EVM Calculator Agent — Compute portfolio metrics")
                evm_res = asyncio.run(
                    Runner.run(
                        evms_calculator_agent,
                        input=(
                            "Compute EVM metrics for this CSV. "
                            "Return a concise per-project summary and portfolio totals.\n\n"
                            f"AsOf: {as_of_str or date.today().isoformat()}\n"
                            f"CSV:\n{csv_text}"
                        ),
                        run_config=RunConfig(model=get_active_default_model(), tracing_disabled=True),
                    )
                )
                evm = (evm_res.final_output or "").strip()
                update_step(s2, "done")
                if sb_p2:
                    sb_p2.write("● EVM Calculator Agent — Done")

                s3 = add_step("Risk Analyst Agent — Assess risks", "running")
                from evm_app.agents.risk_analyst_agent import risk_analyst_agent
                if sb_p3:
                    sb_p3.write("○ Risk Analyst Agent — Assess risks")
                risk_res = asyncio.run(
                    Runner.run(
                        risk_analyst_agent,
                        input=(
                            "Assess risk levels per project based on CPI, SPI, CV, and SV, and propose corrective actions.\n\n"
                            f"EVM JSON: {evm}"
                        ),
                        run_config=RunConfig(model=get_active_default_model(), tracing_disabled=True),
                    )
                )
                risk = (risk_res.final_output or "").strip()
                update_step(s3, "done")
                if sb_p3:
                    sb_p3.write("● Risk Analyst Agent — Done")

                agent_response = f"{ing}\n\n{evm}\n\n{risk}"

                # Also compute and persist items/totals for visuals & Q&A
                if status:
                    status.update(label="Computing visuals…", state="running")
                items, totals, _ = compute_portfolio_for_ui(csv_text, as_of_str)
                st.session_state["evms_items"] = items
                st.session_state["evms_totals"] = totals
                st.session_state["evms_report"] = agent_response
                st.session_state["evms_as_of"] = as_of_str
                # History persistence removed

            if status:
                status.update(label="Done", state="complete")
            st.success("Analysis complete. See results below.")

    # Render results outside submit branch so they persist across reruns
    with results_area:
        report_ss = st.session_state.get("evms_report")
        items = st.session_state.get("evms_items") or []
        totals = st.session_state.get("evms_totals") or {}
        if report_ss:
            st.markdown("### Final Report")
            st.write(report_ss or "(no response)")
            st.download_button(
                "Download report (Markdown)",
                data=(report_ss or "Report unavailable.").strip(),
                file_name="evm_report.md",
                mime="text/markdown",
            )

            st.markdown("### Portfolio Heatmap")
            render_cpi_spi_heatmap(items)

            # Compact portfolio totals (chips) + table
            render_totals_chips(totals)
            st.markdown("### Computed Metrics")
            render_evms_colored_table(items[:200], totals, show_totals_banner=False)

            # Suggest questions helper (appears above the Q&A section)
            def _suggest_questions(items_list, totals_dict):
                pool = [
                    "Which projects are behind schedule (SPI < 1.0)?",
                    "Which projects are over budget (CPI < 1.0)?",
                    "List top 3 projects by cost variance (CV).",
                    "Which projects have both CPI and SPI below thresholds?",
                    "What is the total BAC, PV, EV, and AC across the portfolio?",
                    "Which projects improved SPI compared to last period?",
                    "Which projects should be watched next month based on risk level?",
                    "What corrective actions are suggested for high‑risk projects?",
                    "How many projects are on track (CPI ≥ 1 and SPI ≥ 1)?",
                    "Which department has the most at‑risk projects?",
                ]
                try:
                    return random.sample(pool, k=5)
                except ValueError:
                    return pool[:5]

            if st.button("Suggest 5 EVM questions", key="suggest_main", help="Get five example prompts you can ask about this data"):
                st.session_state["__qa_suggestions_main__"] = _suggest_questions(items, totals)
            # If a suggestion was clicked or an answer exists, show quick jump link
            if st.session_state.get("__qa_trigger_main__") or st.session_state.get("__qa_last_q"):
                st.markdown("[View answer](#answer)")
            if st.session_state.get("__qa_suggestions_main__"):
                st.markdown("#### Suggested questions")
                cols = st.columns(1)
                for i, qtext in enumerate(st.session_state["__qa_suggestions_main__"]):
                    if st.button(qtext, key=f"suggest_pick_main_{i}"):
                        st.session_state["qa_question"] = qtext
                        st.session_state["__qa_trigger_main__"] = True
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()

    # Simple Q&A (data-only) — stays outside submit and follows results
    with results_area:
        items_ss = st.session_state.get("evms_items")
        totals_ss = st.session_state.get("evms_totals")
        if items_ss is not None and totals_ss is not None:
            st.markdown("### Ask Rowshni a question about these projects")

            # Clear pending field if requested (must happen BEFORE rendering the widget)
            if st.session_state.get("__qa_clear"):
                st.session_state["qa_question"] = ""
                del st.session_state["__qa_clear"]

            with st.form("qa_form"):
                q = st.text_input(
                    "Ask Rowshni",
                    placeholder="Ask Rowshni…",
                    key="qa_question",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("Ask Rowshni")
                st.caption("Press Enter or click ‘Ask Rowshni’")
            auto_trigger = st.session_state.get("__qa_trigger_main__", False)
            if submitted and q.strip() or (auto_trigger and (st.session_state.get("qa_question") or "").strip()):
                if auto_trigger:
                    q = st.session_state.get("qa_question", "")
                qa_prompt = (
                    "Answer strictly using this data. If unknown, say so.\n\n"
                    f"Totals: {json.dumps(totals_ss)}\n"
                    f"Items: {json.dumps(items_ss)}\n"
                    f"Question: {q.strip()}"
                )
                ans = asyncio.run(
                    Runner.run(
                        qa_agent,
                        qa_prompt,
                        run_config=RunConfig(model=get_active_default_model(), tracing_disabled=True),
                    )
                )
                resp = (ans.final_output or "").strip()
                # Save last Q&A, request clear, and rerun to safely reset the input
                st.session_state["__qa_last_q"] = q.strip()
                st.session_state["__qa_last_a"] = resp
                st.session_state["__qa_clear"] = True
                if auto_trigger:
                    st.session_state.pop("__qa_trigger_main__", None)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

            # Render last asked question and answer after rerun
            last_q = st.session_state.get("__qa_last_q")
            last_a = st.session_state.get("__qa_last_a")
            if last_q:
                st.markdown("<a name='answer'></a>", unsafe_allow_html=True)
                st.markdown(f"#### Q: {last_q}")
                if last_a and "OUT_OF_SCOPE" in last_a:
                    st.info("The assistant can only answer questions about the uploaded data.")
                elif last_a:
                    st.write(last_a)
        else:
            st.caption("Run EVM Analysis to enable Q&A.")

    # Sidebar Q&A rendered at end so it activates immediately after results are computed
    with st.sidebar:
        st.markdown("### Ask Rowshni")
        sb_items = st.session_state.get("evms_items")
        sb_totals = st.session_state.get("evms_totals")
        if sb_items is None or sb_totals is None:
            st.caption("Run EVM Analysis to enable Q&A.")
        else:
            if st.session_state.get("__qa_clear_sb"):
                st.session_state["qa_question_sb"] = ""
                del st.session_state["__qa_clear_sb"]
            # Sidebar suggestions button
            def _suggest_sidebar(items_list, totals_dict):
                base = [
                    "Which projects are behind schedule (SPI < 1.0)?",
                    "Which projects are over budget (CPI < 1.0)?",
                    "Show top 3 projects by negative cost variance.",
                    "Which projects have both CPI and SPI below thresholds?",
                    "Summarize portfolio totals (BAC, PV, EV, AC).",
                    "Which projects are medium/high risk and why?",
                ]
                try:
                    return random.sample(base, k=5)
                except ValueError:
                    return base[:5]

            if st.button("Suggest 5 EVM questions", key="suggest_sb", help="Get five example prompts you can ask about this data"):
                st.session_state["__qa_suggestions_sb__"] = _suggest_sidebar(sb_items, sb_totals)
            # Quick access to latest answer without scrolling
            if st.session_state.get("__qa_last_q"):
                with st.expander("View latest answer", expanded=False):
                    st.markdown(f"**Q:** {st.session_state.get('__qa_last_q')}")
                    st.write(st.session_state.get("__qa_last_a", ""))
            if st.session_state.get("__qa_suggestions_sb__"):
                for i, s in enumerate(st.session_state["__qa_suggestions_sb__"]):
                    if st.button(s, key=f"suggest_pick_sb_{i}"):
                        st.session_state["qa_question_sb"] = s
                        st.session_state["__qa_trigger_sb__"] = True
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()
            with st.form("qa_form_sidebar"):
                q_sb = st.text_input(
                    "Ask Rowshni",
                    placeholder="Ask Rowshni…",
                    key="qa_question_sb",
                    label_visibility="collapsed",
                )
                sb_submit = st.form_submit_button("Ask Rowshni")
                st.caption("Press Enter or click ‘Ask Rowshni’")
            auto_sb = st.session_state.get("__qa_trigger_sb__", False)
            if sb_submit and q_sb.strip() or (auto_sb and (st.session_state.get("qa_question_sb") or "").strip()):
                if auto_sb:
                    q_sb = st.session_state.get("qa_question_sb", "")
                qa_prompt_sb = (
                    "Answer strictly using this data. If unknown, say so.\n\n"
                    f"Totals: {json.dumps(sb_totals)}\n"
                    f"Items: {json.dumps(sb_items)}\n"
                    f"Question: {q_sb.strip()}"
                )
                ans_sb = asyncio.run(
                    Runner.run(
                        qa_agent,
                        qa_prompt_sb,
                        run_config=RunConfig(model=get_active_default_model(), tracing_disabled=True),
                    )
                )
                resp_sb = (ans_sb.final_output or "").strip()
                st.session_state["__qa_last_q"] = q_sb.strip()
                st.session_state["__qa_last_a"] = resp_sb
                st.session_state["__qa_clear_sb"] = True
                if auto_sb:
                    st.session_state.pop("__qa_trigger_sb__", None)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()


if __name__ == "__main__":
    main()
