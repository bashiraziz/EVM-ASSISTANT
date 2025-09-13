from typing import Any, Dict, List, Optional

import streamlit as st

from ..config import RISK_CPI_THRESHOLD, RISK_SPI_THRESHOLD
from ..tools.evm_tools import risk_level_and_reasons


def _score_color(val: Optional[float]) -> str:
    if val is None:
        return "#444"
    # Consolidated, professional palette: green (good), slate (neutral), red (bad)
    if val >= 1.0:
        return "#166534"  # green-800
    if val >= 0.95:
        return "#334155"  # slate-700
    return "#7f1d1d"      # red-800


def render_evms_colored_table(items: List[Dict[str, Any]], totals: Dict[str, Any], show_totals_banner: bool = True):
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

    header_bg = "#0f172a"  # sober dark navy
    header_style = f"background:{header_bg};color:#e2e8f0;border-bottom:1px solid #1f2937;"
    html = [
        "<div style='margin-top:12px; overflow-x:auto; max-width:100%; max-height:60vh; overflow-y:auto'>",
        "<table style='border-collapse:collapse;width:100%'>",
        "<thead class='table-head' style='" + header_style + "'><tr>"
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
                color = {"high": "#7f1d1d", "medium": "#334155", "low": "#166534"}.get(
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

    if show_totals_banner:
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


def render_totals_chips(totals: Dict[str, Any]):
    """Compact portfolio totals as a responsive chip grid (2x4) with a clear header.

    Clarifies that values are portfolio-level aggregates/derived metrics. If an
    `AsOf` key is present in `totals`, it is displayed alongside the heading.
    """
    def fmt_money(v: Optional[float]) -> str:
        try:
            return f"${abs(float(v)):,}" if float(v) >= 0 else f"-${abs(float(v)):,}"
        except Exception:
            return "-"

    def fmt_num(v: Optional[float]) -> str:
        try:
            return f"{float(v):.2f}" if isinstance(v, (int, float, str)) else "-"
        except Exception:
            return "-"

    tot = totals or {}
    # raw values (for trend)
    raw = {
        "BAC": tot.get("BAC"),
        "PV": tot.get("PV"),
        "EV": tot.get("EV"),
        "AC": tot.get("AC"),
        "CPI": tot.get("CPI"),
        "SPI": tot.get("SPI"),
        "CV": tot.get("CV"),
        "SV": tot.get("SV"),
    }

    chips = [
        ("BAC", fmt_money(raw["BAC"]), "Budget At Completion", raw["BAC"]),
        ("PV", fmt_money(raw["PV"]), "Planned Value = BAC × planned % complete", raw["PV"]),
        ("EV", fmt_money(raw["EV"]), "Earned Value = BAC × actual % complete", raw["EV"]),
        ("AC", fmt_money(raw["AC"]), "Actual Cost to date", raw["AC"]),
        ("CPI", fmt_num(raw["CPI"]), "Cost Performance Index = EV / AC (≥ 1.0 is favorable)", raw["CPI"]),
        ("SPI", fmt_num(raw["SPI"]), "Schedule Performance Index = EV / PV (≥ 1.0 is favorable)", raw["SPI"]),
        ("CV", fmt_money(raw["CV"]), "Cost Variance = EV − AC (> 0 favorable)", raw["CV"]),
        ("SV", fmt_money(raw["SV"]), "Schedule Variance = EV − PV (> 0 favorable)", raw["SV"]),
    ]

    def chip_style(label: str, value: str) -> str:
        style = "padding:8px 10px;border:1px solid #2a2f3a;border-radius:10px;background:#0b1220;"
        if label in ("CPI", "SPI"):
            try:
                v = float(value)
                if v >= 1.0:
                    style = style.replace("#0b1220", "#166534")
                elif v >= 0.95:
                    style = style.replace("#0b1220", "#334155")
                else:
                    style = style.replace("#0b1220", "#7f1d1d")
                style += ";color:#fff"
            except Exception:
                pass
        return style

    as_of = tot.get("AsOf")
    base_note = "Sum/derived across all projects"
    subtitle_text = f"{base_note} • As Of: {as_of}" if as_of else base_note
    subtitle = f"<span style='opacity:.75;margin-left:8px'>{subtitle_text}</span>"
    html = [
        "<div style='margin:8px 0'>",
        "<div style='display:flex;align-items:baseline;gap:8px;margin-bottom:6px'>",
        "<div style='font-weight:700'>Portfolio Totals</div>",
        subtitle,
        "</div>",
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px'>",
    ]
    def trend_for(label: str, raw_value: Any) -> str:
        try:
            v = float(raw_value)
        except Exception:
            return ""
        # Favorable rules
        if label in ("CPI", "SPI"):
            if v >= 1.0:
                return "<span style='margin-left:6px;color:#86efac'>▲</span>"  # green-300
            elif v >= 0.95:
                return "<span style='margin-left:6px;color:#cbd5e1'>▲</span>"  # slate-300
            else:
                return "<span style='margin-left:6px;color:#fecaca'>▼</span>"  # red-200
        if label in ("CV", "SV"):
            if v > 0:
                return "<span style='margin-left:6px;color:#86efac'>▲</span>"
            elif v < 0:
                return "<span style='margin-left:6px;color:#fecaca'>▼</span>"
            else:
                return ""
        return ""

    for lab, val, tip, raw_val in chips:
        stl = chip_style(lab, str(val))
        glyph = trend_for(lab, raw_val)
        html.append(
            f"<div style='{stl}' title=\"{tip}\"><div style='font-size:12px;opacity:.85'>{lab}</div>"
            f"<div style='font-size:18px;font-weight:700;display:flex;align-items:center'>{val}{glyph}</div></div>"
        )
    html.append("</div></div>")
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


__all__ = ["render_evms_colored_table", "render_cpi_spi_heatmap", "render_totals_chips"]
