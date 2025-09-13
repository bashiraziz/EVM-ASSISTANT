from agents import Agent


summary_agent = Agent(
    name="Summary Agent",
    instructions=(
        "You write a concise, executive-friendly summary of portfolio EVM results. "
        "Input will contain portfolio totals and a short list of at-risk projects. "
        "Keep it under 6 bullets. Prefer plain language and actionable advice."
    ),
    tool_use_behavior="run_llm_again",
)


__all__ = ["summary_agent"]

