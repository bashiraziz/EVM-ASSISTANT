from typing import Any, Dict, List, Optional

import json

from agents import function_tool

from ..config import RISK_CPI_THRESHOLD, RISK_SPI_THRESHOLD
from ..ui.trace import log_event


@function_tool
def assess_risks(evms_result_json: str) -> Dict[str, Any]:
    log_event("tool_call", "assess_risks", {"by": "Risk Analyst Agent"})
    try:
        data = json.loads(evms_result_json)
        items = data.get("items", []) if isinstance(data, dict) else []
        risks: List[Dict[str, Any]] = []
        for it in items:
            try:
                cpi = it.get("CPI")
                spi = it.get("SPI")
                cv = it.get("CV")
                sv = it.get("SV")
                reasons: List[str] = []
                if cpi is not None and cpi < RISK_CPI_THRESHOLD:
                    reasons.append(f"CPI low ({cpi})")
                if spi is not None and spi < RISK_SPI_THRESHOLD:
                    reasons.append(f"SPI low ({spi})")
                if cv is not None and cv < 0:
                    reasons.append(f"Over cost (CV={cv})")
                if sv is not None and sv < 0:
                    reasons.append(f"Behind schedule (SV={sv})")
                if reasons:
                    level = "high" if len(reasons) >= 3 else "medium"
                    risks.append(
                        {
                            "ProjectID": it.get("ProjectID"),
                            "ProjectName": it.get("ProjectName"),
                            "level": level,
                            "reasons": reasons,
                        }
                    )
            except Exception:
                continue
        return {"risks": risks}
    except Exception:
        return {"risks": []}
    finally:
        log_event("tool_done", "assess_risks", {"by": "Risk Analyst Agent"})


__all__ = ["assess_risks"]

