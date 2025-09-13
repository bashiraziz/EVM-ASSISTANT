import re
from typing import List, Tuple


def _chips_html(pairs: List[Tuple[str, str]]) -> str:
    items = []
    for k, v in pairs:
        k = (k or "").strip()
        v = (v or "-").strip()
        items.append(
            "<div style='padding:8px 10px;border:1px solid #2a2f3a;border-radius:10px;background:#0b1220'>"
            f"<div style='font-size:12px;opacity:.85'>{k}</div>"
            f"<div style='font-size:18px;font-weight:700'>{v}</div>"
            "</div>"
        )
    return (
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin:6px 0'>"
        + "".join(items)
        + "</div>"
    )


def prettify_report(md: str) -> str:
    """Compress common 2-column tables (e.g., Metric/Total) into a compact chip grid.

    Looks for Markdown tables whose headers include 'Metric' and 'Total'. Converts the
    rows into a responsive grid of chips, reducing vertical space.
    """
    if not md:
        return md

    # Regex to capture a markdown table block with two columns, starting with a header
    # row including 'Metric' and 'Total'. Non-greedy up to a blank line.
    pattern = re.compile(
        r"(^|\n)\|\s*Metric\s*\|\s*Total\s*\|\s*\n\|[^\n]*\n(?P<body>(?:\|[^\n]*\n)+)",
        re.IGNORECASE,
    )

    def repl(m: re.Match) -> str:
        body = m.group("body")
        pairs: List[Tuple[str, str]] = []
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) >= 2:
                pairs.append((cols[0], cols[1]))
        if not pairs:
            return m.group(0)
        html = _chips_html(pairs)
        # Preserve leading newline to not break surrounding markdown structure
        return ("\n" if m.group(1) else "") + html + "\n"

    try:
        return pattern.sub(repl, md)
    except Exception:
        return md


__all__ = ["prettify_report"]

