from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RunConfig:
    model: str
    tracing_disabled: bool = True


class _RunResult:
    def __init__(self, final_output: str):
        self.final_output = final_output


class Runner:
    @staticmethod
    async def run(agent: Any, input: str, run_config: Optional[RunConfig] = None) -> _RunResult:
        # Use LiteLLM for a simple chat completion with system + user
        try:
            from litellm import completion
        except Exception:
            # Fallback dummy
            return _RunResult("")
        model = run_config.model if run_config else "gpt-4o-mini"
        # Normalize model identifiers like "litellm/gemini/gemini-1.5-pro"
        # LiteLLM expects provider/model (e.g., "gemini/gemini-1.5-pro").
        try:
            low = model.lower()
            if low.startswith("litellm/"):
                model = model.split("/", 1)[1]
        except Exception:
            pass
        system = getattr(agent, "instructions", "You are a helpful assistant.")
        try:
            resp = completion(model=model, messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": input},
            ])
            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            out = (msg.get("content") or "").strip()
            return _RunResult(out)
        except Exception as e:
            return _RunResult(f"LLM_ERROR: {e}")
