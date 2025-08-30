from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import csv
import io

from agents import function_tool

from ..config import RISK_CPI_THRESHOLD, RISK_SPI_THRESHOLD
from ..ui.trace import log_event


def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except Exception:
            continue
    return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(s: Any) -> Optional[float]:
    try:
        if s is None:
            return None
        return float(str(s).strip())
    except Exception:
        return None


def _compute_evms_row(row: Dict[str, Any], as_of: date) -> Dict[str, Any]:
    bac = _safe_float(row.get("PlannedCost")) or 0.0
    ac = _safe_float(row.get("ActualCost")) or 0.0
    start = _parse_date(row.get("StartDate", ""))
    end_original = _parse_date(row.get("EndDate", ""))
    end_expected = _parse_date(row.get("ExpectedEndDate", ""))
    end = end_expected or end_original

    if start and end and end > start:
        total_days = (end - start).days
        elapsed_days = (min(as_of, end) - start).days
        planned_pct = _clamp(elapsed_days / total_days, 0.0, 1.0)
    else:
        planned_pct = 0.0

    manual_pct = (_safe_float(row.get("ProgressPercent_Manual")) or 0.0) / 100.0

    pv = bac * planned_pct
    ev = bac * manual_pct

    cpi = (ev / ac) if ac > 0 else None
    spi = (ev / pv) if pv > 0 else None
    cv = ev - ac
    sv = ev - pv
    eac = (bac / cpi) if cpi and cpi > 0 else None
    etc = (eac - ac) if eac is not None else None

    return {
        **row,
        "OriginalEndDate": row.get("EndDate", ""),
        "ExpectedEndDate": row.get("ExpectedEndDate", ""),
        "BAC": round(bac, 2),
        "PV": round(pv, 2),
        "EV": round(ev, 2),
        "AC": round(ac, 2),
        "%Complete_Manual": round(manual_pct * 100.0, 2),
        "%Complete_Schedule": round(planned_pct * 100.0, 2),
        "CPI": round(cpi, 3) if cpi is not None else None,
        "SPI": round(spi, 3) if spi is not None else None,
        "CV": round(cv, 2),
        "SV": round(sv, 2),
        "EAC": round(eac, 2) if eac is not None else None,
        "ETC": round(etc, 2) if etc is not None else None,
    }


def compute_evms_fn(data: List[Dict[str, Any]], as_of_date: Optional[str] = None) -> Dict[str, Any]:
    """Compute EVM metrics given parsed projects list.

    Returns { items: list[dict], totals: dict, row_errors: list[str] }
    """
    log_event("tool_call", "compute_evms", {"by": "EVM Calculator Agent"})
    errors: List[str] = []
    items: List[Dict[str, Any]] = []
    as_of = _parse_date(as_of_date or "") or date.today()

    try:
        for row in data:
            try:
                items.append(_compute_evms_row(row, as_of))
            except Exception as e:  # per-row robustness
                errors.append(f"Row error for {row.get('ProjectID')}: {e}")

        def sumf(key: str) -> float:
            return round(sum((x.get(key) or 0.0) for x in items), 2)

        totals = {
            "BAC": sumf("BAC"),
            "PV": sumf("PV"),
            "EV": sumf("EV"),
            "AC": sumf("AC"),
            "CPI": round((sumf("EV") / sumf("AC")), 3) if sumf("AC") > 0 else None,
            "SPI": round((sumf("EV") / sumf("PV")), 3) if sumf("PV") > 0 else None,
            "CV": round(sumf("EV") - sumf("AC"), 2),
            "SV": round(sumf("EV") - sumf("PV"), 2),
            "AsOf": as_of.isoformat(),
        }
        return {"items": items, "totals": totals, "row_errors": errors}
    finally:
        log_event("tool_done", "compute_evms", {"by": "EVM Calculator Agent"})

# Expose as an Agents tool for use by LLM
compute_evms = function_tool(strict_mode=False)(compute_evms_fn)


def compute_evms_from_csv_fn(csv_text: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
    log_event("tool_call", "compute_evms_from_csv", {"by": "EVM Calculator Agent"})
    try:
        f = io.StringIO(csv_text)
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        return compute_evms_fn(rows, as_of_date)
    finally:
        log_event("tool_done", "compute_evms_from_csv", {"by": "EVM Calculator Agent"})

# Expose as an Agents tool for use by LLM
compute_evms_from_csv = function_tool(strict_mode=False)(compute_evms_from_csv_fn)


def compute_portfolio_for_ui(csv_text: str, as_of_date: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    data = compute_evms_from_csv_fn(csv_text, as_of_date or None)
    return data.get("items", []), data.get("totals", {}), data.get("row_errors", [])


def risk_level_and_reasons(it: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    cpi = it.get("CPI")
    spi = it.get("SPI")
    cv = it.get("CV")
    sv = it.get("SV")
    if cpi is not None and cpi < RISK_CPI_THRESHOLD:
        reasons.append(f"CPI low ({cpi})")
    if spi is not None and spi < RISK_SPI_THRESHOLD:
        reasons.append(f"SPI low ({spi})")
    if cv is not None and cv < 0:
        reasons.append(f"Over cost (CV={cv})")
    if sv is not None and sv < 0:
        reasons.append(f"Behind schedule (SV={sv})")
    level = "low"
    if len(reasons) >= 3:
        level = "high"
    elif len(reasons) >= 1:
        level = "medium"
    return level, reasons


__all__ = [
    "compute_evms",
    "compute_evms_from_csv",
    "compute_portfolio_for_ui",
    "risk_level_and_reasons",
]
