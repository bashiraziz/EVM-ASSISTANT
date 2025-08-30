from typing import Any, Dict, List, Optional

import streamlit as st

from ..config import RISK_CPI_THRESHOLD, RISK_SPI_THRESHOLD
from ..tools.evm_tools import risk_level_and_reasons


def _score_color(val: Optional[float]) -> str:
    if val is None:
        return "#444"
    if val >= 1.0:
        return "#1b5e20"
    if val >= 0.95:
        return "#ff8f00"
    return "#b71c1c"


def render_evms_colored_table(items: List[Dict[str, Any]], totals: Dict[str, Any]):
    cols = [
        "ProjectID",
        "ProjectName",
        "PM",
        "Dept",
        "Status",
        "StartDate",
        "OriginalEndDate",
        "ExpectedEndDate",
        "BAC",
        "PV",
        "EV",
        "AC",
        "CPI",
        "SPI",
        "CV",
        "SV",
        "EAC",
        "ETC",
        "Risk",
    ]
    tooltips = {
        "BAC": "Budget At Completion: total planned cost",
        "PV": "Planned Value (BCWS): BAC x planned % complete",
        "EV": "Earned Value (BCWP): BAC x actual % complete",
        "AC": "Actual Cost (ACWP): actual cost to date",
        "CPI": "Cost Performance Index = EV / AC (>= 1.0 is on/under cost)",
        "SPI": "Schedule Performance Index = EV / PV (>= 1.0 is on/ahead of schedule)",
        "CV": "Cost Variance = EV - AC (>0 favorable)",
        "SV": "Schedule Variance = EV - PV (>0 favorable)",
        "EAC": "Estimate At Completion ~ BAC / CPI",
        "ETC": "Estimate To Complete = EAC - AC",
        "Risk": "Derived from CPI/SPI and variances; thresholds configurable above",
        "OriginalEndDate": "Baseline planned end date",
        "ExpectedEndDate": "Current expected end date (if provided)",
        "StartDate": "Project start date",
        "ProjectID": "Identifier from your CSV",
        "ProjectName": "Name from your CSV",
        "PM": "Project Manager",
        "Dept": "Department or cost center",
        "Status": "Project status from your CSV",
    }

    def _cell_tooltip(row: Dict[str, Any], col: str, val: Any) -> str:
        if val is None or val == "":
            return ""
        try:
            v = float(val) if isinstance(val, (int, float, str)) else None
        except Exception:
            v = None

        if col == "CPI":
            tip = f"CPI = EV / AC. Threshold: < {RISK_CPI_THRESHOLD} flagged low."
            if v is not None:
                tip += f" Value: {v}."
            return tip
        if col == "SPI":
            tip = f"SPI = EV / PV. Threshold: < {RISK_SPI_THRESHOLD} flagged low."
            if v is not None:
                tip += f" Value: {v}."
            return tip
        if col == "CV":
            return "CV = EV - AC (>0 favorable)."
        if col == "SV":
            return "SV = EV - PV (>0 favorable)."
        if col == "EAC":
            return "EAC ~ BAC / CPI."
        if col == "ETC":
            return "ETC = EAC - AC."
        if col == "PV":
            return "PV = BAC x planned % complete."
        if col == "EV":
            return "EV = BAC x actual % complete."
        if col == "AC":
            return "Actual Cost to date."
        return tooltips.get(col, "")

    def fmt(v):
        if v is None:
            return "-"
        return f"{v}"

    rows = []
    for it in items:
        risk_level, _ = risk_level_and_reasons(it)
        row = {c: it.get(c) for c in cols}
        row["Risk"] = risk_level
        rows.append(row)

    html = [
        "<div style='margin-top:12px; overflow-x:auto; max-width:100%; max-height:60vh; overflow-y:auto'>",
        "<table style='border-collapse:collapse;width:100%'>",
        "<thead class='table-head'><tr>"
        + "".join(
            (lambda _c: f"<th style='text-align:left;padding:8px 10px;cursor:help' title=\"{tooltips.get(_c, '')}\">{_c}</th>")
            (c)
            for c in cols
        )
        + "</tr></thead>",
        "<tbody>",
    ]
    for r in rows:
        html.append("<tr>")
        for c in cols:
            val = r.get(c)
            style = "padding:6px 8px;border-bottom:1px solid #333;"
            if c in ("CPI", "SPI"):
                style += f"background:{_score_color(val)};color:#fff;"
            if c == "Risk":
                color = {"high": "#b71c1c", "medium": "#ff8f00", "low": "#1b5e20"}.get(
                    str(val), "#444"
                )
                style += f"background:{color};color:#fff;font-weight:600;"
            title = _cell_tooltip(r, c, val)
            if title:
                style += "cursor:help;"
            html.append(f"<td style='{style}' title=\"{title}\">{fmt(val)}</td>")
        html.append("</tr>")
    html.append("</tbody>")
    html.append("</table>")

    tot = totals or {}
    html.append(
        "<div class='totals-banner' style='margin-top:10px;padding:10px;border-radius:10px;'>"
        f"<b>Portfolio Totals</b> &mdash; As Of: {fmt(tot.get('AsOf'))} | "
        f"BAC: {fmt(tot.get('BAC'))}, PV: {fmt(tot.get('PV'))}, EV: {fmt(tot.get('EV'))}, AC: {fmt(tot.get('AC'))}, "
        f"CPI: {fmt(tot.get('CPI'))}, SPI: {fmt(tot.get('SPI'))}, CV: {fmt(tot.get('CV'))}, SV: {fmt(tot.get('SV'))}"
        "</div>"
    )
    html.append("</div>")

    st.markdown("".join(html), unsafe_allow_html=True)


def render_cpi_spi_heatmap(items: List[Dict[str, Any]]):
    def bin_val(v: Optional[float]) -> Optional[str]:
        if v is None:
            return None
        if v >= 1.0:
            return "high"
        if v >= 0.95:
            return "mid"
        return "low"

    bins = {
        ("low", "low"): [],
        ("low", "mid"): [],
        ("low", "high"): [],
        ("mid", "low"): [],
        ("mid", "mid"): [],
        ("mid", "high"): [],
        ("high", "low"): [],
        ("high", "mid"): [],
        ("high", "high"): [],
    }
    excluded = 0
    for it in items:
        cb = bin_val(it.get("CPI"))
        sb = bin_val(it.get("SPI"))
        if cb is None or sb is None:
            excluded += 1
            continue
        bins[(cb, sb)].append(it)

    label = {"low": "< 0.95", "mid": "0.95-<1.0", "high": ">= 1.0"}
    color = {"low": "#b71c1c", "mid": "#ff8f00", "high": "#1b5e20"}

    html = [
        "<div style='margin-top:12px; overflow-x:auto; max-width:100%'>",
        "<b>CPI x SPI Heatmap</b>",
        "<table style='margin-top:6px;border-collapse:collapse;width:100%'>",
        "<thead><tr><th></th>"
        + "".join(
            f"<th style='text-align:center;border-bottom:1px solid #444;padding:6px 8px'>SPI {label[c]}</th>"
            for c in ("low", "mid", "high")
        )
        + "</tr></thead>",
        "<tbody>",
    ]
    for rbin in ("low", "mid", "high"):
        html.append("<tr>")
        html.append(
            f"<th style='text-align:right;border-right:1px solid #444;padding:6px 8px'>CPI {label[rbin]}</th>"
        )
        for cbin in ("low", "mid", "high"):
            cell = bins[(rbin, cbin)]
            bg = color[rbin]
            style = (
                f"padding:8px;border:1px solid #333;background:{bg};color:#fff;text-align:center"
            )
            count = len(cell)
            top = ", ".join([str(x.get("ProjectID")) for x in cell[:3]])
            content = f"<div style='font-size:16px;font-weight:700'>{count}</div>"
            if top:
                content += f"<div style='font-size:12px;opacity:.9'>[{top}]</div>"
            html.append(f"<td style='{style}'>{content}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    if excluded:
        html.append(
            f"<div style='margin-top:6px;color:#94a3b8'>Excluded {excluded} project(s) with missing CPI/SPI.</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


__all__ = ["render_evms_colored_table", "render_cpi_spi_heatmap"]

