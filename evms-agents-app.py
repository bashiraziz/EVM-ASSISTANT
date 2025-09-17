"""
EVM Assistant (Streamlit UI)

This UI is now thin: all logic, agents, tools, and UI helpers live under `evm_app/`.
"""

from __future__ import annotations

# --- Compatibility shim for third-party `agents` package importing TF1 APIs ---
# Some distributions of the `agents` package import TensorFlow 1.x symbols like
# `tf.contrib.distributions` at import time. We provide a minimal shim using
# `tensorflow_probability` so that importing `agents` does not crash on TF2.
try:
    import types
    import tensorflow as tf  # type: ignore
    if not hasattr(tf, "contrib"):
        try:
            from tensorflow_probability import distributions as _tfd  # type: ignore
            tf.contrib = types.SimpleNamespace(distributions=_tfd)  # type: ignore[attr-defined]
        except Exception:
            # Fallback to a dummy object to avoid AttributeError during import.
            dummy = types.SimpleNamespace()
            dummy.distributions = None
            tf.contrib = dummy  # type: ignore[attr-defined]
except Exception:
    # If TensorFlow is not present or anything else fails, continue. The
    # OpenAI Agents SDK path does not require TF at runtime.
    pass

import asyncio
import time
import csv
import io
import json
import os
from datetime import date
import random
from typing import Any, Dict, List, Optional

import streamlit as st
from agents.run import Runner, RunConfig

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
from evm_app.ui.about import render_about_panel
from evm_app.ui.progress import (
    render_sidebar as render_progress_sidebar,
    set_sidebar_badges,
    reset_progress,
    add_step,
    update_step,
    remove_steps_by_prefix,
    prune_completed,
)
from evm_app.ui.toolbar import render_results_toolbar
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
        import hashlib as _hashlib

        suffix_key = "__header_map_suffix__"
        signature_key = "__header_map_signature__"
        cache_key = "__header_map_cache__"
        existing_suffix = st.session_state.get(suffix_key, 0)
        signature = (tuple(options), _hashlib.sha256(csv_text.encode("utf-8")).hexdigest())

        cache = st.session_state.get(cache_key, {})
        cached = cache.get(signature)
        if cached is not None:
            return cached

        if st.session_state.get(signature_key) != signature:
            existing_suffix += 1
            st.session_state[suffix_key] = existing_suffix
            st.session_state[signature_key] = signature
        else:
            st.session_state.setdefault(suffix_key, existing_suffix)
        form_suffix = st.session_state.get(suffix_key, existing_suffix)

        mapping: Dict[str, str] = {}
        with st.form(f"header_map_form_{form_suffix}"):
            for h in EXPECTED_HEADERS:
                choices = options if h in options else options + [MISSING]
                default_idx = choices.index(h) if h in options else choices.index(MISSING)
                mapping[h] = st.selectbox(
                    f"Map to '{h}'", options=choices, index=default_idx, key=f"map_{form_suffix}_{h}"
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
        mapped_csv = fout.getvalue()
        cache = st.session_state.setdefault(cache_key, {})
        cache[signature] = mapped_csv
        st.success("Headers mapped to the expected template.")
        st.download_button(
            "Download mapped CSV",
            data=mapped_csv,
            mime="text/csv",
            file_name="mapped_evms.csv",
            key=f"mapped_csv_download_{form_suffix}"
        )
        preview = mapped_csv.splitlines()
        if preview:
            st.code("\n".join(preview[: min(6, len(preview))]), language="csv")
        else:
            st.code("(empty)", language="csv")
        return mapped_csv
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
        /* Make primary call-to-action buttons look like the uploader tile */
        .primary-cta .stButton>button {
            width: 100%;
            background: #2b2d33 !important; /* match uploader tile */
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
            border-radius: 10px !important;
            padding: 10px 16px !important;
            white-space: nowrap; /* keep on one line */
        }
        /* Wrapper used for prominent run buttons */
        .primary-cta { display:flex; align-items:center; justify-content:center; }
        /* Gentle vertical alignment for the run button next to uploader */
        .run-upload-wrap { padding-top: 22px; }
        /* Evenly spaced trio of links under uploader */
        .link-trio { display:flex; justify-content:space-around; align-items:center; gap:12px; width:100%; padding-top:8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # (Header links removed per request; links remain in footer)

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
            md = st.session_state.get("__mdl_default__") or st.session_state.get("__model_default__")
            ms = st.session_state.get("__mdl_summary__") or st.session_state.get("__model_summary__")
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
            key="__mdl_default__",
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
            key="__mdl_summary__",
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

    # Sidebar Q&A will be rendered once at the end after results are available

    # History UI removed

    st.caption(
        "Upload a CSV matching the template headers to compute EVM metrics and summarize portfolio health."
    )
    sample_text = get_sample_csv_text()
    col_dl, col_run_sample, col_headers = st.columns([1, 1, 1.4])
    with col_dl:
        st.download_button("Get CSV Template", data=sample_text, file_name="evms_sample.csv", mime="text/csv")
    with col_run_sample:
        run_sample_clicked = st.button(
            "Run Sample Analysis",
            help="Loads the built-in sample CSV and runs the analysis",
            key="run_sample_btn",
        )
    with col_headers:
        with st.expander("Template Headers", expanded=False):
            st.code(sample_text.splitlines()[0] + "\n...", language="csv")
    # Optional: show a hosted direct-download link if configured
    _sample_url = os.getenv("SAMPLE_CSV_URL")
    if _sample_url:
        st.markdown(f"Or download from: [Hosted CSV]({_sample_url})")

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
    run_clicked_upload = False
    if input_mode == "Upload CSV":
        up_col, run_col = st.columns([3, 1.2])
        with up_col:
            file = st.file_uploader("Upload CSV (project_template.csv format)", type=["csv"])
            # Validate file size against 20MB limit (also enforced via .streamlit/config.toml)
            MAX_MB = 20
            MAX_BYTES = MAX_MB * 1024 * 1024
            if file is not None and getattr(file, "size", 0) > MAX_BYTES:
                st.error(f"File is too large. Max allowed is {MAX_MB} MB.")
                st.session_state["__oversize_file__"] = True
            else:
                st.session_state.pop("__oversize_file__", None)
        with run_col:
            st.markdown("<div class='run-upload-wrap'>", unsafe_allow_html=True)
            st.markdown("<div class='primary-cta'>", unsafe_allow_html=True)
            run_clicked_upload = st.button("Run EVM Analysis", key="run_btn_upload")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        pasted_csv_text = st.text_area(
            "Paste CSV content",
            value=get_sample_csv_text(),
            height=180,
        )

    # Helpers: reset state + quick links
    # Place three links evenly spaced; under the Upload column when in Upload mode
    if input_mode == "Upload CSV":
        with up_col:
            st.markdown(
                "<div class='link-trio'>"
                "<a class='link-btn' href='#results'>Scroll to results</a>"
                "<a class='link-btn' href='#top'>Back to top</a>"
                "<a class='link-btn' href='?reset=1#top'>Reset</a>"
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        # In paste mode, render full-width trio centered by Streamlit's column
        st.markdown(
            "<div class='link-trio'>"
            "<a class='link-btn' href='#results'>Scroll to results</a>"
            "<a class='link-btn' href='#top'>Back to top</a>"
            "<a class='link-btn' href='?reset=1#top'>Reset</a>"
            "</div>",
            unsafe_allow_html=True,
        )

    # If oversize, suppress any run via upload even if clicked
    if st.session_state.get("__oversize_file__"):
        run_clicked_upload = False

    # A dedicated results area that renders BELOW inputs and the button
    st.markdown("<a name='results'></a>", unsafe_allow_html=True)
    results_area = st.container()

    # Submit button row (only for Paste mode)
    run_clicked = False
    if input_mode == "Paste CSV":
        btn_col = st.columns([1])[0]
        with btn_col:
            st.markdown("<div class='primary-cta'>", unsafe_allow_html=True)
            run_clicked = st.button("▶ Run EVM Analysis", key="run_btn_paste")
            st.markdown("</div>", unsafe_allow_html=True)
    # Back to top link moved next to Reset above

    # Any run trigger (primary button, upload-row button, or sample pill)
    run_clicked_any = (
        (('run_clicked' in locals() and run_clicked) or ('run_clicked_upload' in locals() and run_clicked_upload))
        or (run_sample_clicked if 'run_sample_clicked' in locals() else False)
    )

    # Keep sidebar badges in sync
    try:
        set_sidebar_badges(PROVIDER, get_active_default_model(), get_active_summary_model())
    except Exception:
        pass

    # When starting a new run, clear progress immediately so the sidebar starts fresh
    try:
        _run_now = run_clicked_any
    except NameError:
        _run_now = run_clicked
    if _run_now:
        reset_progress()
        render_progress_sidebar()
    else:
        # Idle render to show any existing statuses
        render_progress_sidebar()

    # Use the combined run flag if available
    try:
        _run_now = run_clicked_any
    except NameError:
        _run_now = run_clicked

    if _run_now:
        # If the sample pill was clicked, ignore other inputs and use sample text
        if 'run_sample_clicked' in locals() and run_sample_clicked:
            csv_text = sample_text
        else:
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
                # Keep all completed items visible in Fast mode as well
            else:
                if status:
                    status.update(label="Running multi-agent workflow…", state="running")
                # Sidebar progress with friendly agent names
                reset_progress()
                s1 = add_step("Ingestion Agent — Parse & validate CSV", "running")
                render_progress_sidebar()
                # Inline live sidebar placeholders for immediate feedback
                # Sidebar is rendered by progress module; avoid separate ad-hoc rows
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
                render_progress_sidebar()

                s2 = add_step("EVM Calculator Agent — Compute portfolio metrics", "running")
                render_progress_sidebar()
                from evm_app.agents.evms_calculator_agent import evms_calculator_agent
                # Progress sidebar handles live rendering
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
                render_progress_sidebar()

                s3 = add_step("Risk Analyst Agent — Assess risks", "running")
                render_progress_sidebar()
                from evm_app.agents.risk_analyst_agent import risk_analyst_agent
                # Progress sidebar handles live rendering
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
                render_progress_sidebar()

                agent_response = f"{ing}\n\n{evm}\n\n{risk}"

                # Keep all completed items visible until a new run

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
            with st.expander("Executive Summary & Narrative", expanded=False):
                st.markdown(report_ss or "(no response)")
            # Download options: MD, TXT, HTML, JSON
            dl_fmt = st.selectbox(
                "Download format",
                ["Markdown (.md)", "Text (.txt)", "HTML (.html)", "JSON (.json)"],
                index=0,
            )

            report_text = (report_ss or "Report unavailable.").strip()

            if dl_fmt.startswith("Markdown"):
                dl_data = report_text
                dl_name = "evm_report.md"
                dl_mime = "text/markdown"
            elif dl_fmt.startswith("Text"):
                dl_data = report_text
                dl_name = "evm_report.txt"
                dl_mime = "text/plain"
            elif dl_fmt.startswith("HTML"):
                # Simple HTML wrapper without extra dependencies
                # Note: this does not render Markdown; it wraps as <pre> for portability
                html_body = (
                    "<html><head><meta charset=\"utf-8\"><title>EVM Report</title>"
                    "<style>body{font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; line-height:1.5; padding:24px;} pre{white-space:pre-wrap;}</style>"
                    "</head><body><pre>" +
                    report_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") +
                    "</pre></body></html>"
                )
                dl_data = html_body
                dl_name = "evm_report.html"
                dl_mime = "text/html"
            else:  # JSON
                try:
                    dl_data = json.dumps({"report": report_text}, ensure_ascii=False, indent=2)
                except Exception:
                    dl_data = '{"report": "' + report_text.replace('"', '\\"') + '"}'
                dl_name = "evm_report.json"
                dl_mime = "application/json"

            st.download_button(
                "Download report",
                data=dl_data,
                file_name=dl_name,
                mime=dl_mime,
            )

            # Compact context toolbar above results
            render_results_toolbar(items, totals)

            st.markdown("### Portfolio Heatmap")
            render_cpi_spi_heatmap(items)

            # Compact portfolio totals (chips) + table
            render_totals_chips(totals)
            st.markdown("### Computed Metrics")
            render_evms_colored_table(items[:200], totals, show_totals_banner=False)

            # Suggest questions helper (appears above the Q&A section)
            def _suggest_questions(items_list, totals_dict):
                # Build suggestions that are answerable strictly from Items/Totals.
                # Avoid past/future comparisons or prescriptive guidance.
                available_keys = set()
                try:
                    if items_list:
                        available_keys = set(items_list[0].keys())
                except Exception:
                    available_keys = set()

                base_pool = [
                    "Which projects are behind schedule (SPI < 1.0)?",
                    "Which projects are over budget (CPI < 1.0)?",
                    "List top 3 projects by cost variance (CV).",
                    "Which projects have both CPI and SPI below thresholds?",
                    "What is the total BAC, PV, EV, and AC across the portfolio?",
                    "How many projects are on track (CPI >= 1 and SPI >= 1)?",
                    "Which projects have the worst schedule variance (SV)?",
                    "Which projects have the worst cost variance (CV)?",
                ]

                # Conditionally include department-related prompt only if such a column exists
                if any(k.lower() in {"department", "dept", "org", "organization"} for k in available_keys):
                    base_pool.append("Which department has the most at-risk projects?")

                # Sample up to 5 suggestions
                try:
                    return random.sample(base_pool, k=min(5, len(base_pool)))
                except ValueError:
                    return base_pool[:5]

            if st.button(
                "💡 Suggest 5 EVM Questions",
                key="suggest_main",
                help="Get five example prompts you can ask about this data",
            ):
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
                # Remove any prior Q&A steps and show a live sidebar placeholder
                # Keep prior Q&A history; add a new fetching entry without pruning
                p_qna = add_step("Project Q&A Agent - Fetching answer", "running")
                # Re-render progress so the new fetching state appears immediately
                render_progress_sidebar()
                try:
                    with st.sidebar:
                        qa_live = st.empty()
                        qa_live.write("○ Project Q&A Agent - Fetching answer")
                except Exception:
                    qa_live = None
                # Ensure the fetching state is visible even for quick responses
                time.sleep(1.0)
                ans = asyncio.run(
                    Runner.run(
                        qa_agent,
                        qa_prompt,
                        run_config=RunConfig(model=get_active_default_model(), tracing_disabled=True),
                    )
                )
                update_step(p_qna, "done", "Project Q&A Agent - Done")
                render_progress_sidebar()
                if qa_live:
                    qa_live.write("● Project Q&A Agent - Done")
                # Keep Q&A history visible; do not prune here
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

            if st.button(
                "💡 Suggest 5 EVM Questions",
                key="suggest_sb",
                help="Get five example prompts you can ask about this data",
                use_container_width=True,
            ):
                st.session_state["__qa_suggestions_sb__"] = _suggest_sidebar(sb_items, sb_totals)
            # Quick access to latest answer without scrolling
            if st.session_state.get("__qa_last_q"):
                st.markdown("<div class='hl-exp'>", unsafe_allow_html=True)
                with st.expander("View latest answer", expanded=False):
                    st.markdown(f"**Q:** {st.session_state.get('__qa_last_q')}")
                    st.write(st.session_state.get("__qa_last_a", ""))
                st.markdown("</div>", unsafe_allow_html=True)
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
                # Remove any prior Q&A steps and show a live sidebar placeholder
                # Keep prior Q&A history; add a new fetching entry without pruning
                p_qna_sb = add_step("Project Q&A Agent - Fetching answer", "running")
                render_progress_sidebar()
                try:
                    with st.sidebar:
                        qa_live_sb = st.empty()
                        qa_live_sb.write("○ Project Q&A Agent - Fetching answer")
                except Exception:
                    qa_live_sb = None
                time.sleep(1.0)
                ans_sb = asyncio.run(
                    Runner.run(
                        qa_agent,
                        qa_prompt_sb,
                        run_config=RunConfig(model=get_active_default_model(), tracing_disabled=True),
                    )
                )
                update_step(p_qna_sb, "done", "Project Q&A Agent - Done")
                render_progress_sidebar()
                if qa_live_sb:
                    qa_live_sb.write("● Project Q&A Agent - Done")
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

        # About panel (collapsed expander)
        render_about_panel()

    # Footer links on main page
    _repo_url = "https://github.com/bashiraziz/EVM-ASSISTANT"
    _readme_url = _repo_url + "/blob/main/README.md"
    _deploy_url = _repo_url + "/blob/main/DEPLOY.md"
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='link-row'>
          <a class='link-btn' href='{_repo_url}' target='_blank'>GitHub Repo</a>
          <a class='link-btn' href='{_readme_url}' target='_blank'>README</a>
          <a class='link-btn' href='{_deploy_url}' target='_blank'>DEPLOY</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
