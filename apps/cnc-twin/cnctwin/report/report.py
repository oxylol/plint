"""The run report (report.html, report.pdf) and the setup sheet (setup_sheet.html, setup_sheet.pdf),
built from the run directory only (docs/report.md).

The HTML is one self-contained file (inline CSS, pictures embedded), the same bytes for the same run
directory and build; the PDF is that HTML printed by WeasyPrint (A4, same content).
"""

from __future__ import annotations

import json
from pathlib import Path

from .context import Context
from .html import Section, document, note
from .pdf import write_atomic, write_pdf
from .rundir import RunData
from .sections_part import (chatter_section, chips_section, conservation_section, coolant_section,
                            deflection_section, inspection_section, surface_section, thermal_section)
from .sections_production import cycle_section, energy_section, tools_section
from .sections_refs import approximations_section, estimates_section, pictures_section
from .sections_run import findings_section, identity_section, missing_section, verdict_header
from .sections_series import series_section
from .setup_sheet import setup_sheet_html

FORMATS = ("html", "pdf")
STUDY_FILE = Path("uncertainty") / "study.json"


def load_study(run: RunData, study=None) -> dict | None:
    """An uncertainty study for the report: a dict, a path to study.json, or ``run_dir/uncertainty/study.json``."""
    if isinstance(study, dict):
        return study
    path = Path(study) if study is not None else run.path / STUDY_FILE
    if not path.is_file():
        return None
    try:
        document_ = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        run.note(str(path), f"the uncertainty study {path.name} cannot be read ({error})")
        return None
    return document_ if isinstance(document_, dict) else None


def uncertainty_section(ctx: Context) -> Section:
    from ..uncertainty.html import html_section
    return Section("uncertainty", "Uncertainty", html_section(ctx.study, heading_level=3),
                   lead="How the results scatter over the project's stated tolerances.")


def report_html(run_dir, study=None) -> tuple[str, RunData]:
    """(the report as text, the RunData it was built from)."""
    run = run_dir if isinstance(run_dir, RunData) else RunData(run_dir)
    ctx = Context(run, study=load_study(run, study))
    header = verdict_header(ctx)
    sections = [findings_section(ctx), cycle_section(ctx), energy_section(ctx), tools_section(ctx),
                inspection_section(ctx), surface_section(ctx), chips_section(ctx), thermal_section(ctx),
                coolant_section(ctx), chatter_section(ctx), deflection_section(ctx), conservation_section(ctx),
                approximations_section(ctx), estimates_section(ctx), pictures_section(ctx), series_section(ctx)]
    if ctx.study is not None:
        sections.append(uncertainty_section(ctx))
    else:
        sections.append(Section("uncertainty", "Uncertainty", note(
            "No uncertainty study for this run (run python -m cnctwin.uncertainty, or POST "
            "/api/runs/{id}/uncertainty; its study.json in uncertainty/ is then shown here).")))
    sections.append(identity_section(ctx))
    sections.append(missing_section(ctx))
    verdict = "safe to run" if run.verdict.get("safe") else "NOT SAFE TO RUN"
    text = document(f"CNC Twin report: {run.name}", sections, header=header,
                    running_title=f"CNC Twin · {run.name} · {verdict}")
    return text, run


def build_report(run_dir, out_dir=None, *, formats=FORMATS, setup_sheet: bool = True, study=None) -> dict:
    """Write report.html/.pdf (and setup_sheet.html/.pdf) for one run directory.

    ``out_dir`` defaults to the run directory itself (architecture 16). Returns the files written, the
    page counts of the PDFs, the verdict and what the run directory lacked.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} is not a run directory")
    out = Path(out_dir) if out_dir is not None else run_dir
    formats = tuple(formats)
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"unknown report format(s) {', '.join(unknown)}: use html and/or pdf")
    text, run = report_html(run_dir, study)
    result = {"run_dir": str(run_dir), "files": [], "verdict": run.verdict, "missing": list(run.missing)}
    _write(out, "report", text, formats, result)
    if setup_sheet:
        sheet = setup_sheet_html(run)
        _write(out, "setup_sheet", sheet, formats, result)
    result["missing"] = list(run.missing)
    return result


def _write(out: Path, stem: str, text: str, formats, result: dict) -> None:
    data = text.encode("utf-8")
    if "html" in formats:
        path = write_atomic(out / f"{stem}.html", data)
        result["files"].append(str(path))
        result[f"{stem}_html"] = str(path)
    if "pdf" in formats:
        path = out / f"{stem}.pdf"
        pages = write_pdf(text, path)
        result["files"].append(str(path))
        result[f"{stem}_pdf"] = str(path)
        result[f"{stem}_pdf_pages"] = pages


__all__ = ["build_report", "report_html", "load_study", "FORMATS", "STUDY_FILE"]
