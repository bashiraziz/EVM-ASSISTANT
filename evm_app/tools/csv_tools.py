from typing import Any, Dict, List

import csv
import io

from agents import function_tool

from ..ui.trace import log_event


EXPECTED_HEADERS = [
    "ProjectID",
    "ProjectName",
    "Customer",
    "CustomerContact",
    "PM",
    "Dept",
    "PlannedCost",
    "ActualCost",
    "StartDate",
    "EndDate",
    "Status",
    "ProgressPercent_Manual",
    "ContractType",
]


@function_tool
def parse_and_validate_csv(csv_text: str) -> Dict[str, Any]:
    """Validate the uploaded CSV against the expected headers and parse rows.

    Returns { ok: bool, errors: list[str], projects: list[dict] }
    """
    log_event("tool_call", "parse_and_validate_csv", {"by": "Ingestion Agent"})
    errors: List[str] = []
    projects: List[Dict[str, Any]] = []

    try:
        f = io.StringIO(csv_text)
        reader = csv.DictReader(f)

        # Validate headers
        headers = [h.strip() for h in (reader.fieldnames or [])]
        missing = [h for h in EXPECTED_HEADERS if h not in headers]
        if missing:
            errors.append("Missing headers: " + ", ".join(missing))
            return {"ok": False, "errors": errors, "projects": []}

        # Parse rows into JSON list
        for row in reader:
            clean = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            projects.append(clean)

        return {"ok": True, "errors": [], "projects": projects}
    except Exception as e:
        errors.append(str(e))
        return {"ok": False, "errors": errors, "projects": []}
    finally:
        log_event("tool_done", "parse_and_validate_csv", {"by": "Ingestion Agent"})


__all__ = ["parse_and_validate_csv", "EXPECTED_HEADERS"]

