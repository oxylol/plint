"""The generated tables of docs/convergence.md from tools/convergence/results/<study>.json.

    python tools/convergence/report.py            (no simulator; deterministic)

docs/convergence.md holds the narrative; every block between ``<!-- convergence:<name>:begin -->`` and
``<!-- convergence:<name>:end -->`` is rewritten here: ``summary`` (one row per study and quantity) and one block per
study (levels with their values and cost; per quantity the reference, observed order, the shipped default's error,
the tolerance and the verdict).
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DOC = HERE.parents[1] / "docs" / "convergence.md"

sys.path.insert(0, str(HERE))
from plots import LABELS  # noqa: E402

ORDER = ["stride_slot", "stride_side", "lattice_side", "lattice_corner", "element_flat", "element_ball",
         "bins_runout", "bins_chatter", "macro", "lag", "thermal_cell", "thermal_every", "turn_theta", "turn_dz"]


def _num(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1e4 or a < 1e-3:
        return f"{v:.{digits - 1}e}"
    return f"{v:.{max(digits - 1 - int(math.floor(math.log10(a))), 0)}f}"


def _err(a: dict, e) -> str:
    if e is None:
        return "-"
    if a["relative"]:
        return f"{100.0 * e:.3g} %"
    return f"{e:.3g}"


def _tol(a: dict) -> str:
    return f"{100.0 * a['tol']:.3g} %" if a["relative"] else f"{a['tol']:g}"


def _unit(q: str) -> str:
    return LABELS.get(q, (q, ""))[1]


def _label(q: str) -> str:
    return LABELS.get(q, (q, ""))[0]


def verdict(a: dict) -> str:
    if "default_inside" not in a:
        return "-"
    if a["default_inside"]:
        return "inside" + ("" if a.get("conclusive", True) else " (reference uncertain)")
    h = a.get("coarsest_within_tol_h")
    return "outside" + (f": use {h:g}" if h is not None else ": no level within")


def study_block(doc: dict) -> str:
    levels = [lv for lv in doc.get("levels", [])]
    analysis = doc.get("analysis", {})
    qs = list(analysis)
    lines = [f"Case: {doc.get('note', '')}.", ""]
    head = ["level", doc.get("h_name", "h")] + [f"{_label(q)} ({_unit(q)})" for q in qs] + ["CPU s"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for lv in sorted(levels, key=lambda v: -float(v["h"])):
        mark = " (default)" if lv["label"] == doc.get("default_label") else ""
        row = [lv["label"] + mark, _num(float(lv["h"]))]
        for q in qs:
            row.append(_num(lv.get("quantities", {}).get(q)))
        cost = (lv.get("cost") or {}).get("cpu_total_s")
        row.append(_num(cost, 3) if cost else ("failed: " + lv["error"][:80] if lv.get("error") else "-"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("| quantity | reference | how | observed order | default error | tolerance | default |")
    lines.append("|---|---|---|---|---|---|---|")
    for q in qs:
        a = analysis[q]
        how = {"richardson": "Richardson (3 finest)", "flat": "all levels agree", "oscillating": "finest (oscillating)",
               "finest": "finest level"}.get(a.get("status"), a.get("status", "-"))
        order = a.get("order")
        lines.append(f"| {_label(q)} | {_num(a.get('reference'))} {_unit(q)} | {how} | "
                     f"{_num(order, 3) if order is not None else '-'} | {_err(a, a.get('default_error'))} | {_tol(a)} | "
                     f"{verdict(a)} |")
    return "\n".join(lines)


def summary_block(docs: list[dict]) -> str:
    lines = ["| study | setting | shipped default | quantities inside / measured | worst default error (tolerance) | "
             "verdict |", "|---|---|---|---|---|---|"]
    for doc in docs:
        analysis = doc.get("analysis", {})
        if not analysis:
            continue
        default = next((lv for lv in doc["levels"] if lv["label"] == doc.get("default_label")), None)
        inside = [q for q, a in analysis.items() if a.get("default_inside")]
        measured = [q for q, a in analysis.items() if "default_inside" in a]
        worst, worst_q = None, None
        for q, a in analysis.items():
            if "default_error" not in a:
                continue
            ratio = a["default_error"] / a["tol"] if a["tol"] else math.inf
            if worst is None or ratio > worst:
                worst, worst_q = ratio, q
        wa = analysis.get(worst_q) if worst_q else None
        w = f"{_label(worst_q)} {_err(wa, wa['default_error'])} ({_tol(wa)})" if wa else "-"
        ok = len(inside) == len(measured) and measured
        lines.append(f"| {doc.get('title', doc['study'])} | `{doc.get('setting', '')}` | "
                     f"{default['label'] if default else '-'} | {len(inside)} / {len(measured)} | {w} | "
                     f"{'inside the converged range' if ok else 'outside for some quantities'} |")
    return "\n".join(lines)


def replace_block(text: str, name: str, body: str) -> str:
    begin, end = f"<!-- convergence:{name}:begin -->", f"<!-- convergence:{name}:end -->"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        return text
    return pattern.sub(lambda _m: f"{begin}\n{body}\n{end}", text)


def main(argv) -> int:
    docs = []
    for name in ORDER:
        p = RESULTS / f"{name}.json"
        if p.exists():
            docs.append(json.loads(p.read_text()))
    text = DOC.read_text()
    text = replace_block(text, "summary", summary_block(docs))
    for doc in docs:
        if doc.get("analysis"):
            text = replace_block(text, doc["study"], study_block(doc))
    DOC.write_text(text)
    print(DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
