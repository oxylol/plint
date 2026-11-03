"""The run report and the setup sheet of the synthetic run: HTML that parses and is self-contained,
the verdict first, findings linked to program lines and cause chains, every section present, a PDF
with pages that fit A4, and the same bytes for the same run directory."""

from __future__ import annotations

import hashlib
import logging

import pytest
from htmlcheck import check

from cnctwin.report import build_report

SECTIONS = ("findings", "cause-chains", "program", "cycle", "energy", "tools", "inspection", "surface", "chips",
            "thermal", "coolant", "chatter", "deflection", "conservation", "approximations", "estimates", "pictures",
            "series", "uncertainty", "identity", "not-in-run")


@pytest.fixture(scope="module")
def built(runs, tmp_path_factory):
    out = tmp_path_factory.mktemp("report-out")
    result = build_report(runs["unsafe"], out / "a")
    twin = build_report(runs["twin"], out / "b", formats=("html",), setup_sheet=False)
    return {"result": result, "twin": twin, "html": (out / "a" / "report.html").read_text(encoding="utf-8"),
            "sheet": (out / "a" / "setup_sheet.html").read_text(encoding="utf-8"), "out": out}


def test_html_parses_and_is_self_contained(built):
    checker = check(built["html"])
    assert checker.tags.get("img", 0) >= 10            # matplotlib plots and the run's images, all embedded
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in built["html"]
    assert "@media (max-width:480px)" in built["html"] and "@page" in built["html"]
    for section in SECTIONS:
        assert f'id="{section}"' in built["html"], section


def test_verdict_comes_first_and_says_why(built):
    html = built["html"]
    verdict = html.index("NOT SAFE TO RUN")
    assert verdict < html.index('id="findings"')
    assert "pocket-depth 2.062 mm" in html[verdict:html.index('id="findings"')]
    assert built["result"]["verdict"]["safe"] is False


def test_findings_link_to_lines_and_chains(built):
    html = built["html"]
    for line in (10, 11, 13, 15, 17):                  # the lines the synthetic findings are tied to
        assert f'href="#L-OP10-op10_nc-{line}"' in html
        assert f'id="L-OP10-op10_nc-{line}"' in html
    for finding in range(5):
        assert f'id="F-{finding}"' in html and f'id="C-{finding}"' in html
    assert "thermal_growth" in html and "floor_position" in html     # the chain of the out-of-tolerance finding


def test_sections_carry_the_run_content(built):
    html = built["html"]
    assert "1620000" in html                           # cutting ticks of the cycle-time breakdown
    assert "compressed air" in html and "kWh" in html  # energy by consumer
    assert "sandvik-2p342-1000-pa" in html and "VB max / limit" in html
    for ident in ("pocket-depth", "floor-flatness", "pocket-width", "pocket-length", "wall-perpendicularity"):
        assert ident in html
    assert "Error budget" in html and "length_offset T1" in html
    assert "floor-finish" in html and "pocket wall X16" in html
    assert "Settled chip height" in html and "surface_temp_K" in html and "Part temperature at unload" in html
    assert "Jets" in html and "Below the curve the cut is stable" in html and "Chatter findings" in html
    assert "dynamics.tool_displacement_m" in html and "Contribution to each dimensional error" in html
    assert "mass" in html and "1.00e-9" in html        # conservation tolerance of the mass balance
    assert "flow_stress.m" in html                     # an estimated database value
    assert "images/part.png" in html and "From frames.bin" in html


def test_pdf_has_pages_and_fits_the_page(built):
    result = built["result"]
    pdf = (built["out"] / "a" / "report.pdf").read_bytes()
    assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
    assert result["report_pdf_pages"] >= 5 and result["setup_sheet_pdf_pages"] >= 1
    from weasyprint import HTML
    logging.getLogger("weasyprint").setLevel(logging.ERROR)
    document = HTML(string=built["html"]).render()
    assert len(document.pages) == result["report_pdf_pages"]
    wide = []
    for number, page in enumerate(document.pages, 1):
        stack = [page._page_box]
        while stack:
            box = stack.pop()
            stack.extend(getattr(box, "children", None) or [])
            if type(box).__name__ == "TableBox" and box.position_x + box.margin_width() > page.width + 0.5:
                wide.append(number)
    assert not wide, f"tables wider than the page on pages {wide}"


def test_same_run_directory_same_bytes(built, runs, tmp_path):
    again = build_report(runs["unsafe"], tmp_path, formats=("html", "pdf"), setup_sheet=True)
    for name in ("report.html", "report.pdf", "setup_sheet.html", "setup_sheet.pdf"):
        first = (built["out"] / "a" / name).read_bytes()
        second = (tmp_path / name).read_bytes()
        assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest(), name
    twin = (built["out"] / "b" / "report.html").read_bytes()      # another directory, same seed and name
    assert twin == (built["out"] / "a" / "report.html").read_bytes()
    assert again["missing"] == built["result"]["missing"]


def test_setup_sheet(built):
    sheet = built["sheet"]
    check(sheet)
    for text in ("Haas VF-2SS", "lang-makro-grip-77", "vise", "G54", "−381.0000", "−203.2000", "35.00",
                 "120.530", "op10.nc", "SYNTHETIC POCKET 16X12", "Operator notes", "chip_packing", "emulsion-8pct",
                 "40.00 × 30.00 × 15.00", "Aluminium 6061-T6", "From frames.bin"):
        assert text in sheet, text


def test_safe_run(runs, tmp_path):
    result = build_report(runs["safe"], tmp_path, formats=("html",), setup_sheet=False)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    check(html)
    assert result["verdict"]["safe"] is True
    assert "SAFE TO RUN" in html and "NOT SAFE TO RUN" not in html.split('id="findings"')[0]
    assert any(m["item"] == "images/" for m in result["missing"])      # made without media: said, not hidden


def test_cli(runs, tmp_path, capsys):
    from cnctwin.report import main
    assert main([str(runs["safe"]), "--out", str(tmp_path), "--format", "html", "--no-setup-sheet"]) == 0
    assert (tmp_path / "report.html").is_file() and not (tmp_path / "setup_sheet.html").exists()
    assert "safe to run" in capsys.readouterr().out
    assert main([str(tmp_path / "nothing-here"), "--format", "html"]) == 1
    assert main([str(runs["safe"]), "--out", str(tmp_path), "--format", "docx"]) == 1
