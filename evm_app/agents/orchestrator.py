from datetime import date
from typing import Optional

from agents import Agent, Runner, function_tool
from agents.run import RunConfig

from ..config import get_active_default_model
from ..ui.trace import log_event
from .ingestion_agent import ingestion_agent
from .evms_calculator_agent import evms_calculator_agent
from .risk_analyst_agent import risk_analyst_agent


@function_tool
async def get_ingestion_summary(csv_text: str) -> str:
    log_event(
        "handoff",
        "Handing off to Ingestion Agent...",
        {
            "reason": "Parse and validate CSV",
            "context": "project_template.csv",
            "from": "EVM Orchestrator",
            "to": "Ingestion Agent",
        },
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
        {
            "reason": "Compute EVM metrics",
            "context": as_of_date or "today",
            "from": "EVM Orchestrator",
            "to": "EVM Calculator Agent",
        },
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
        {
            "reason": "Identify and rank risks",
            "context": "CPI, SPI, CV, SV",
            "from": "EVM Orchestrator",
            "to": "Risk Analyst Agent",
        },
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


orchestrator_agent = Agent(
    name="EVM Orchestrator",
    instructions=(
        "You orchestrate three tools: `get_ingestion_summary`, `get_evms_report`, and `get_risk_assessment`.\n"
        "Flow:\n"
        "1) Always parse/validate the CSV (string) first via `parse_and_validate_csv` or the ingestion tool.\n"
        "2) If valid, compute EVM metrics for the projects with `get_evms_report`.\n"
        "3) Then produce a risk assessment with `get_risk_assessment`.\n"
        "4) Merge into one final answer: Overview, Portfolio Totals, Per-Project Highlights, Risks & Actions.\n"
        "Keep responses concise and practical for PMs."
    ),
    tools=[get_ingestion_summary, get_evms_report, get_risk_assessment],
    tool_use_behavior="run_llm_again",
)


async def run_agent(csv_text: str, as_of_date: Optional[str]) -> str:
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


__all__ = [
    "orchestrator_agent",
    "get_ingestion_summary",
    "get_evms_report",
    "get_risk_assessment",
    "run_agent",
]

