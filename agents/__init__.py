from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional


# Minimal, local shim of an Agents interface to avoid heavy external deps.

@dataclass
class Agent:
    name: str
    instructions: str
    tools: Optional[List[Callable[..., Any]]] = None
    tool_use_behavior: Optional[str] = None


def function_tool(*dargs, **dkwargs):
    """Compatible decorator that supports optional kwargs and direct use.

    Usage:
    - @function_tool
    - @function_tool()
    - @function_tool(strict_mode=False)
    """
    if dargs and callable(dargs[0]) and not dkwargs:
        # Used as @function_tool
        fn = dargs[0]
        return fn

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _wrap


__all__ = ["Agent", "function_tool"]
