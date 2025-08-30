from agents import Agent

from ..tools.analysis_tools import assess_risks


risk_analyst_agent = Agent(
    name="Risk Analyst Agent",
    instructions=(
        "Analyze EVM results and flag projects at risk based on CPI, SPI, CV, and SV. "
        "Provide prioritized recommendations for corrective actions."
    ),
    tools=[assess_risks],
    tool_use_behavior="run_llm_again",
)


__all__ = ["risk_analyst_agent"]

