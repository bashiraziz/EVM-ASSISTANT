from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd

from agents.run import Runner, RunConfig

from evm_app.tools.evm_tools import compute_portfolio_for_ui, risk_level_and_reasons
from evm_app.agents.qa_agent import qa_agent
from evm_app.agents.summary_agent import summary_agent


DEFAULT_MODEL = "litellm/gemini/gemini-1.5-pro"
SUMMARY_MODEL = "litellm/gemini/gemini-1.5-flash"


async def _summary_text(totals: Dict[str, Any], items: List[Dict[str, Any]]) -> str:
    risks = []
    for it in items:
        lvl, reasons = risk_level_and_reasons(it)
        if lvl in ("medium", "high"):
            risks.append({
                "ProjectID": it.get("ProjectID"),
                "ProjectName": it.get("ProjectName"),
                "level": lvl,
                "reasons": reasons[:3],
            })
    prompt = (
        "Create a short overview of portfolio EVM results.\n"
        f"Totals: {totals}\n"
        f"Risks: {risks[:8]}\n"
        "Return 4-6 concise bullets with actionable guidance."
    )
    res = await Runner.run(
        summary_agent,
        prompt,
        run_config=RunConfig(model=SUMMARY_MODEL, tracing_disabled=True),
    )
    return (res.final_output or "").strip()


async def _qa_text(totals: Dict[str, Any], items: List[Dict[str, Any]], q: str) -> str:
    qa_prompt = (
        "Answer strictly using this data. If unknown, say so.\n\n"
        f"Totals: {totals}\n"
        f"Items: {items}\n"
        f"Question: {q.strip()}"
    )
    res = await Runner.run(
        qa_agent,
        qa_prompt,
        run_config=RunConfig(model=DEFAULT_MODEL, tracing_disabled=True),
    )
    return (res.final_output or "").strip()


def _read_csv_input(file: Optional[gr.File], pasted: str | None) -> str:
    if file is not None:
        try:
            return file.read().decode("utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    return pasted or ""


async def _run(csv_text: str, as_of: Optional[str], mode: str, progress=gr.Progress()):
    if not csv_text.strip():
        return "Please provide CSV input.", pd.DataFrame(), {}, ""

    progress(0.2, desc="Computing EVM metrics…")
    items, totals, row_errors = compute_portfolio_for_ui(csv_text, as_of)
    warn = ""
    if row_errors:
        warn = "Some rows had issues. Showing first 5:\n" + "\n".join(row_errors[:5])

    progress(0.6, desc="Generating summary…")
    summary = await _summary_text(totals, items)

    df = pd.DataFrame(items[:200]) if items else pd.DataFrame()
    progress(1.0, desc="Done")
    return summary or "(no summary)", df, totals, warn


def _suggest_questions(_: Dict[str, Any], __: List[Dict[str, Any]]):
    pool = [
        "Which projects are behind schedule (SPI < 1.0)?",
        "Which projects are over budget (CPI < 1.0)?",
        "List top 3 projects by cost variance (CV).",
        "Which projects have both CPI and SPI below thresholds?",
        "What is the total BAC, PV, EV, and AC across the portfolio?",
        "Which department has the most at-risk projects?",
        "Which projects are on track (CPI ≥ 1 and SPI ≥ 1)?",
        "What corrective actions are suggested for high-risk projects?",
        "Which projects improved SPI compared to last period?",
        "Which projects should be watched next month based on risk level?",
    ]
    import random
    return [[q] for q in random.sample(pool, k=5)]


with gr.Blocks(title="EVM Assistant — Gradio") as demo:
    st_items = gr.State([])
    st_totals = gr.State({})

    gr.Markdown("## EVM Assistant — Gradio Prototype")

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["Fast (local + 1 summary)", "Agentic (multi-agent)"],
                value="Agentic (multi-agent)",
                label="Mode",
            )
            as_of = gr.Textbox(value=str(date.today()), label="As-of date (YYYY-MM-DD)")
            upload = gr.File(label="Upload CSV", file_types=[".csv"])
            pasted = gr.Textbox(label="Or paste CSV", lines=6)
            run_btn = gr.Button("Run EVM Analysis", variant="primary")
            status = gr.Markdown()

            gr.Markdown("### Ask Rowshni")
            qa_in = gr.Textbox(label="", placeholder="Ask Rowshni…")
            ask_btn = gr.Button("Ask Rowshni")
            answer_md = gr.Markdown()

            gr.Markdown("### Suggestions")
            suggest_btn = gr.Button("Suggest 5 EVM questions")
            sugg_out = gr.Dataset(components=[gr.Textbox(label="")], samples=[])

        with gr.Column(scale=2):
            summary_md = gr.Markdown("### Final Report\n\n")
            df = gr.Dataframe(headers=[], label="Computed Metrics")
            totals_json = gr.JSON(label="Portfolio Totals")

    async def on_run(file, pasted_text, as_of_text, mode_text):
        csv_text = _read_csv_input(file, pasted_text)
        return await _run(csv_text, as_of_text or None, mode_text)

    run_btn.click(
        on_run,
        inputs=[upload, pasted, as_of, mode],
        outputs=[summary_md, df, totals_json, status],
    ).then(
        lambda s, dfv, t, w: (dfv.to_dict(orient="records") if hasattr(dfv, "to_dict") else [], t),
        inputs=[summary_md, df, totals_json, status],
        outputs=[st_items, st_totals],
    )

    suggest_btn.click(_suggest_questions, inputs=[st_totals, st_items], outputs=[sugg_out])

    def on_pick_suggestion(sample, totals, items):
        q = sample[0] if isinstance(sample, list) else (sample or "")
        return q

    sugg_out.select(on_pick_suggestion, inputs=[sugg_out, st_totals, st_items], outputs=[qa_in])

    async def on_ask(q, totals, items):
        if not q or not q.strip():
            return "Please enter a question."
        ans = await _qa_text(totals, items, q)
        return f"**Q:** {q}\n\n{ans}"

    ask_btn.click(on_ask, inputs=[qa_in, st_totals, st_items], outputs=[answer_md])

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    demo.queue().launch(server_name=host, server_port=port)
