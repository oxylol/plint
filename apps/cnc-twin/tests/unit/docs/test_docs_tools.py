"""cnctwin.docs_tools: the Markdown reader, the statement and column rules, the sources of
docs/approximations.md, the model-doc gap check, determinism, and the real documentation tree."""

import json

import pytest

from cnctwin import docs_tools as D
from cnctwin.report.approximations import read_sections, relevant

MODEL_DOC = """# Foo: a model of foo

## 1. Equations

The force is `F = k (h / h_ref)^-m b` and the power `P = F v`.

The heat to the chip is Q = F v (1 - beta).

    t_ready(k) = max(t_ready(k-1), t_free(k)) + c(k)

```python
doc = study(project_path, out_dir)   # code, not an equation of the model
```

## 2. Parameters

| name | value | source |
|---|---|---|
| `k_N_mm2` | 800 | johnson-cook-1985 |

## 3. Assumptions and limitations (with consequences)

* Quasi-static chips (no inertia): forces +-30 % at high speed, as reported by Johnson & Cook 1985.
  The continuation line belongs to the bullet.
  * a nested bullet joins its parent
* **Lumped tool.** The body is one node. Consequence: transients faster than 5 ms are smoothed.
* The constant is `est` (assumed); error below 1e-9 relative (tested against the closed form).

## 4. Validity range and error bounds

| part | validity | error and how it was established | basis |
|---|---|---|---|
| drag | Re < 2.6e5 | ~5 % for spheres | test_drag.py against the closed form |

### 4.1 Below a matching heading

A paragraph of the subsection is collected with it, cutting_force validation included.

## 5. Elsewhere

Validity: only below 100 m/min (case force-orth-a) within 10 % of the validation data.

```
# not a heading inside a code block
```

## References

* Johnson, G.R., Cook, W.H. (1985).
"""

BARE_DOC = """# Bar

## 1. Use

Nothing but use.
"""

ARCHITECTURE = """# Architecture

## 1. Stack

**Scope decisions** (each with its consequence):
* Multi-path controls are in scope: several paths with waits. Single-path mills are the common case.
* Out-of-contact dynamics include unbalance.

Next paragraph, not a decision.
"""

PERFORMANCE = """# Performance

Budget per simulated second. The load moves single timings by +-15 % (removal more than the rest).

**Status.** Met: L1 (0.03 s/s). Missed: the hot loop (2.6 s/s against 0.45).

## 4. Whole runs

| run | CPU | s/s |
|---|---|---|
| slot, rigid | 5.05 s | 0.96 |

## 7. Memory

| run | peak RSS |
|---|---|
| slot | 0.66 GB |

## 9. Open problems

* **Removal's tooth pass** is the largest section. It costs 1.5 ms per pass.
"""

RESULTS = {
    "format": "cnctwin-validation-results", "version": 1, "quantity": "cutting_force",
    "harness": "cnctwin.validation.harnesses.cutforce.cutting_force",
    "budget": {"pct": None, "components": {"main_force_pct": 15.0, "thrust_force_pct": 30.0}},
    "independent": {"cases": 3, "passing": 1, "median_abs_rel": 0.31, "p90_abs_rel": 1.8, "compared": 40,
                    "class_values": 0, "class_matches": 0},
    "calibration_dependent": {"cases": 2},
    "cases": [{"model_chain": ["cnctwin.mechanics.materials (databases)",
                               "cnctwin.cutting.coefficients.MechanicsModel (rows)", "cnctwin.tooling.build_tool"]}],
    "verdict": "fail", "verdict_reason": "only 1 of 3 cases within budget",
}


@pytest.fixture()
def tree(tmp_path):
    (tmp_path / "docs" / "models").mkdir(parents=True)
    (tmp_path / "docs" / "models" / "foo.md").write_text(MODEL_DOC, encoding="utf-8")
    (tmp_path / "docs" / "models" / "bar.md").write_text(BARE_DOC, encoding="utf-8")
    (tmp_path / "docs" / "models" / "cutting.md").write_text("# Cutting\n\n## 2. Limitations\n\n* Forces are "
                                                             "quasi-static per element.\n", encoding="utf-8")
    (tmp_path / "docs" / "architecture.md").write_text(ARCHITECTURE, encoding="utf-8")
    (tmp_path / "docs" / "performance.md").write_text(PERFORMANCE, encoding="utf-8")
    (tmp_path / "validation" / "results").mkdir(parents=True)
    (tmp_path / "validation" / "results" / "cutting_force.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    (tmp_path / "validation" / "cases" / "cutting_force").mkdir(parents=True)
    (tmp_path / "validation" / "cases" / "cutting_force" / "force-orth-a.yaml").write_text("id: force-orth-a\n")
    (tmp_path / "db" / "bibliography").mkdir(parents=True)
    (tmp_path / "db" / "bibliography" / "x.yaml").write_text(
        "format: cnctwin-bibliography\nversion: 1\nentries:\n  johnson-cook-1985:\n    type: article\n"
        "    title: t\n    year: 1985\n", encoding="utf-8")
    return tmp_path


def test_headings_inside_code_blocks_are_not_sections():
    sections = D.parse_markdown(MODEL_DOC)
    titles = [s.title for s in sections]
    assert "not a heading inside a code block" not in " ".join(titles)
    nested = next(i for i, s in enumerate(sections) if s.title.startswith("4.1"))
    assert sections[nested].number == "4.1" and sections[nested].name == "Below a matching heading"
    assert D.path_label(sections, nested) == "§ 4. Validity range and error bounds > 4.1 Below a matching heading"


def test_statements_bullets_rows_and_paragraphs():
    sections = D.parse_markdown(MODEL_DOC)
    assumptions = next(s for s in sections if s.title.startswith("3."))
    statements = D.section_statements(assumptions)
    assert [s.kind for s in statements] == ["bullet", "bullet", "bullet"]
    assert "continuation line belongs to the bullet" in statements[0].text
    assert "; a nested bullet joins its parent" in statements[0].text
    validity = next(s for s in sections if s.title.startswith("4. "))
    rows = D.section_statements(validity)
    assert len(rows) == 1 and rows[0].kind == "row" and rows[0].cells["validity"] == "Re < 2.6e5"


def test_columns_of_bullets_and_table_rows(tree):
    vocabulary = D.read_vocabulary(tree)
    assert "johnson-cook-1985" in vocabulary.bibliography and "force-orth-a" in vocabulary.cases
    rows, headings, leads = D.doc_rows("foo", tree / "docs" / "models" / "foo.md", tree, vocabulary)
    by_start = {r.approximation[:20]: r for r in rows}
    quasi = by_start["Quasi-static chips ("]
    assert "+-30 % at high speed" in quasi.error_bound
    assert quasi.evidence.startswith("literature")
    assert quasi.consequence == "The continuation line belongs to the bullet; a nested bullet joins its parent"
    lumped = next(r for r in rows if "Lumped tool" in r.approximation)
    assert lumped.consequence == "transients faster than 5 ms are smoothed."
    constant = next(r for r in rows if "constant is" in r.approximation)
    assert "1e-9 relative" in constant.error_bound
    assert "convergence/test" in constant.evidence and "estimate" in constant.evidence
    drag = next(r for r in rows if r.approximation.startswith("drag"))
    assert drag.approximation == "drag; valid: Re < 2.6e5" and drag.error_bound == "~5 % for spheres"
    assert "convergence/test (test_drag.py)" in drag.evidence
    nested = next(r for r in rows if "paragraph of the subsection" in r.approximation)
    assert nested.evidence.startswith("validation case (cutting_force)")
    lead = next(r for r in rows if r.approximation.startswith("Validity:"))
    assert "within 10 %" in lead.error_bound and "force-orth-a" in lead.evidence
    assert "5. Elsewhere" in leads and not any(h.startswith("1.") or h.startswith("2.") for h in headings)
    assert headings[0].startswith("3. Assumptions")


def test_equation_lines_skip_tagged_code():
    sections = D.parse_markdown(MODEL_DOC)
    lines = D.equation_lines(sections)
    text = MODEL_DOC.splitlines()
    assert len(lines) == 3 and not any("study(" in text[n - 1] for n in lines)
    assert any("t_ready" in text[n - 1] for n in lines)


def test_error_cues_need_an_error_word_for_plain_numbers():
    assert D.states_error("Mass is exact")
    assert D.states_error("refill planes second order (<= r dphi^2 / 4)")
    assert D.states_error("within 10 % of the measured forces")
    assert not D.states_error("Grid tilt up to 54.7 deg")
    assert not D.states_error("give load_reference_W when the meter's 100 % is known")


def test_clip_keeps_code_spans_and_bold_balanced():
    text = "**A long bold lead** and `a code span that runs past the limit` " + "word " * 40
    clipped = D.clip(text, 40)
    assert clipped.endswith(" ...") and clipped.count("`") % 2 == 0 and clipped.count("**") % 2 == 0
    assert D.clip("short", 40) == "short"


def test_approximations_document(tree):
    document = D.build_approximations(tree)
    text = document.text
    assert document.unmatched == ["docs/models/bar.md"]
    assert "## Foo (docs/models/foo.md)" in text and "## Scope (docs/architecture.md)" in text
    assert "## Loop cost and performance (docs/performance.md)" in text
    assert "* docs/models/bar.md" in text.split("## Documents whose structure did not match")[1]
    scope = [r for r in document.rows if r.kind == "scope"]
    assert len(scope) == 2 and scope[0].consequence == "Single-path mills are the common case."
    validation = [r for r in document.rows if r.kind == "validation"]
    assert len(validation) == 1 and validation[0].domain == "cutting"
    assert "median 31.0 %, p90 180.0 % over 40 values" in validation[0].error_bound
    assert "cnctwin.cutting.coefficients.MechanicsModel" in validation[0].approximation
    performance = [r for r in document.rows if r.kind == "performance"]
    assert {r.approximation for r in performance} >= {"CPU per simulated second: slot, rigid", "peak memory: slot"}
    assert all("+-15 %" in r.error_bound for r in performance if r.approximation.startswith("CPU"))
    # every table row has the six cells: no '|' inside a cell
    for line in text.splitlines():
        if line.startswith("| "):
            assert line.count("|") == len(D.HEADER) + 1, line


def test_report_reader_picks_the_domain_sections(tree):
    path = tree / "approximations.md"
    path.write_text(D.build_approximations(tree).text, encoding="utf-8")
    sections = read_sections(path)
    picked, _ = relevant(sections, {"cutting"})
    headings = [s.heading for s in picked]
    assert "Cutting (docs/models/cutting.md)" in headings and "Scope (docs/architecture.md)" in headings


def test_generation_is_deterministic_and_check_mode(tree, capsys):
    first = D.build_approximations(tree).text
    assert D.build_approximations(tree).text == first
    out = tree / "docs" / "approximations.md"
    assert D.main(["--root", str(tree), "approximations"]) == 0 and out.read_text(encoding="utf-8") == first
    assert D.main(["--root", str(tree), "approximations", "--check"]) == 0
    out.write_text(first + "edited\n", encoding="utf-8")
    assert D.main(["--root", str(tree), "approximations", "--check"]) == 1
    assert out.read_text(encoding="utf-8").endswith("edited\n")


def test_gap_check(tree, capsys):
    report = D.check_docs(tree)
    by_doc = {d.doc: d for d in report.docs}
    foo = by_doc["docs/models/foo.md"]
    assert foo.missing == [] and foo.unstructured == []
    bar = by_doc["docs/models/bar.md"]
    assert bar.missing == list(D.REQUIRED_ITEMS)
    cutting = by_doc["docs/models/cutting.md"]
    assert "references" in cutting.missing and "assumptions" in cutting.missing
    assert "| docs/models/bar.md | **missing** |" in report.text
    assert D.main(["--root", str(tree), "check"]) == 0
    assert (tree / "docs" / "doc-gaps.md").read_text(encoding="utf-8") == report.text
    assert D.main(["--root", str(tree), "check", "--strict"]) == 1


def test_text_only_items_are_told_apart():
    doc = "# X\n\n## 1. Use\n\nThe model is valid for h > 0.1 mm. Coefficients: db/models/x.yaml.\n"
    sections = D.parse_markdown(doc)
    assert D._text_only(sections, "validity range").startswith("§ 1. Use")
    assert D._text_only(sections, "parameters and sources").startswith("§ 1. Use")
    assert D._text_only(sections, "references") == ""


def test_real_documentation_tree():
    """The repository's own docs: every model doc is either collected or listed as not matching, the gap
    report has one row per model doc, and the validation and performance sources are read."""
    root = D.ROOT
    document = D.build_approximations(root)
    docs = [D.rel(p, root) for _, p in D.model_docs(root)]
    assert sorted(set(document.matched) | set(document.unmatched)) == sorted(docs)
    results = list((root / "validation" / "results").glob("*.json"))
    assert any(r.kind == "validation" for r in document.rows) or not results
    assert any(r.kind == "performance" for r in document.rows)
    report = D.check_docs(root)
    assert [d.doc for d in report.docs] == docs
    assert all(line.count("|") == len(D.HEADER) + 1 for line in document.text.splitlines() if line.startswith("| "))
