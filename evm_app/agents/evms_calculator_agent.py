from agents import Agent

from ..tools.evm_tools import compute_evms_from_csv, compute_evms


evms_calculator_agent = Agent(
    name="EVM Calculator Agent",
    instructions=(
        "Compute EVM metrics for each project. Use: PV = BAC * planned % complete; "
        "EV = BAC * ProgressPercent_Manual/100; AC from ActualCost. Derive CPI, SPI, CV, SV, and EAC. "
        "Return concise tables and short interpretation."
    ),
    tools=[compute_evms_from_csv, compute_evms],
    tool_use_behavior="run_llm_again",
)


__all__ = ["evms_calculator_agent"]

