from agents import Agent

from ..tools.csv_tools import parse_and_validate_csv


ingestion_agent = Agent(
    name="Ingestion Agent",
    instructions=(
        "You parse and validate a CSV against the expected project template headers.\n"
        "Return a short, clear summary of any issues found, and proceed if valid."
    ),
    tools=[parse_and_validate_csv],
    tool_use_behavior="run_llm_again",
)


__all__ = ["ingestion_agent"]

