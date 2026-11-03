"""The uncertainty study as HTML: a section for the run report and a page of its own.

Headline first (probability that the run is not safe, the least capable characteristic), then the
factors and how each was applied, every inspected dimension (distribution, limits, Cp/Cpk with their
intervals, probability out of tolerance, histogram), the probability of every finding by category
and by program location, the scalar results, and the sensitivity ranking (Spearman rank correlation,
partial rank correlation) with a tornado chart per dimension.
"""

from __future__ import annotations

from ..report import fmt, plots
from ..report.html import Raw, Section, document, esc, figure, kv, note, para, table, tiles

TOP_FACTORS = 8      # presentation: factors shown per tornado chart and per sensitivity table


def _prob(record: dict | None, decimals: int = 1) -> str:
    if not record or record.get("p") is None:
        return fmt.DASH
    return f"{fmt.pct(record['p'], decimals)} % [{fmt.pct(record['low'], decimals)}, {fmt.pct(record['high'], decimals)}]"


def _value(name: str, value) -> str:
    """A sampled or measured value in display units, by the SI unit its name ends with."""
    base = name.split("[")[0]
    number = fmt.as_float(value)
    if number is None:
        return fmt.DASH
    if base.endswith("_m") or base in ("stock_size", "length_offset", "radius_offset", "runout", "x_m"):
        return f"{fmt.um(number, 2)} \u00b5m" if abs(number) < 1e-3 else f"{fmt.mm(number, 4)} mm"
    if base.endswith("_J"):
        return f"{fmt.kj(number)} kJ"
    if base.endswith("_kg"):
        return f"{fmt.grams(number)} g"
    if base.endswith("_eur"):
        return f"{fmt.eur(number, 4)} EUR"
    if base.endswith("_s"):
        return f"{fmt.seconds(number)} s"
    return fmt.num(number, 5)


def _distribution(record: dict, unit: str) -> str:
    kind = record.get("kind")
    render = (lambda v: _value("x_m", v)) if unit == "m" else (lambda v: fmt.num(v, 5))
    if kind == "uniform":
        return f"uniform {render(record['low'])} … {render(record['high'])}"
    if kind == "truncated_normal":
        return (f"normal σ {fmt.num(record['sigma'], 3)}, truncated to {fmt.num(record['low'], 4)} … "
                f"{fmt.num(record['high'], 4)}")
    return str(kind)


def _sensitivity_of(study: dict, output: str) -> dict | None:
    for record in (study.get("results") or {}).get("sensitivity", []):
        if record["output"] == output:
            return record
    return None


def html_section(study: dict, heading_level: int = 2) -> Raw:
    """The study as the body of a report section (sub-headings at ``heading_level``)."""
    h = f"h{heading_level}"
    results = study.get("results") or {}
    settings = study.get("settings") or {}
    parts = [_headline(study)]
    parts.append(f"<{h}>Factors</{h}>" + _factors(study))
    parts.append(f"<{h}>Inspected dimensions</{h}>" + _dimensions(study))
    parts.append(f"<{h}>Findings</{h}>" + _findings(results))
    parts.append(f"<{h}>Other results</{h}>" + _scalars(results))
    parts.append(f"<{h}>Sensitivity</{h}>" + _sensitivity(study))
    failed = results.get("failed") or []
    if failed:
        parts.append(f"<{h}>Failed samples</{h}>" + table(["sample", "error", "run"],
                                                          [[f["index"], f["error"], f["run_dir"]] for f in failed]))
    parts.append(note(f"Plan: {settings.get('runs')} samples by {settings.get('method')}, seed {settings.get('seed')}"
                      f"{', per-sample seeds' if settings.get('vary_seed') else ', the project seed in every sample'}; "
                      f"intervals at {fmt.pct((study.get('parameters') or {}).get('confidence_level'), 0)} % "
                      "confidence (docs/models/uncertainty.md)."))
    return Raw("".join(parts))


def _headline(study: dict) -> str:
    results = study.get("results") or {}
    unsafe = (results.get("verdict") or {}).get("unsafe")
    dims = results.get("dimensions") or []
    worst = None
    for record in dims:
        cpk = (record.get("capability") or {}).get("Cpk")
        if cpk is not None and (worst is None or cpk < worst[1]):
            worst = (record["id"], cpk)
    items = [("not safe to run", _prob(unsafe), f"of {results.get('samples_ok', 0)} samples")]
    if worst:
        items.append(("least capable", f"{worst[0]}", f"Cpk {fmt.num(worst[1], 3)}"))
    out_probs = [(r["id"], (r.get("out_of_tolerance") or {}).get("p")) for r in dims]
    out_probs = [x for x in out_probs if x[1] is not None]
    if out_probs:
        name, p = max(out_probs, key=lambda x: (x[1], x[0]))
        items.append(("most often out of tolerance", name, f"{fmt.pct(p)} % of samples"))
    items.append(("factors varied", str(len(study.get("factors") or [])),
                  f"{len(study.get('options_only') or [])} through run options only"))
    return tiles(items)


def _factors(study: dict) -> str:
    rows = []
    for factor in study.get("factors") or []:
        rows.append([factor["name"], factor["label"], _distribution(factor["distribution"], factor["unit"]),
                     _value(factor["name"], factor["nominal"]) if factor["unit"] == "m" else fmt.num(factor["nominal"], 5),
                     factor["applies"], factor["target"], factor["stated_by"]])
    out = table(["factor", "what", "distribution", "nominal", "applied in", "where", "stated by"], rows,
                empty="The project states no tolerance to vary.")
    if study.get("options_only"):
        out += note("Applied through options['uncertainty']['factors'] only (no project-file key holds them): "
                    + ", ".join(study["options_only"]) + ". The simulation must read them for them to have an effect.")
    for item in study.get("not_varied") or []:
        out += note(f"Not varied: {item['key']}: {item['reason']}.")
    return out


def _dimensions(study: dict) -> str:
    results = study.get("results") or {}
    levels = [f"{v:.6g}" for v in (study.get("parameters") or {}).get("percentile_levels", [])]
    rows, charts = [], ""
    for record in results.get("dimensions") or []:
        measured = record.get("measured") or {}
        quantiles = measured.get("quantiles") or {}
        cap = record.get("capability") or {}
        show = cap.get("reported")
        cpk_ci = cap.get("Cpk_ci")
        rows.append([record["id"], record.get("kind") or fmt.DASH, measured.get("n"), fmt.mm(measured.get("mean"), 4),
                     fmt.um(measured.get("std"), 2), fmt.mm(quantiles.get(levels[0]) if levels else None, 4),
                     fmt.mm(quantiles.get(levels[-1]) if levels else None, 4),
                     f"{fmt.mm(record.get('lower_m'), 4)} … {fmt.mm(record.get('upper_m'), 4)}",
                     fmt.num(cap.get("Cp"), 3) if show else fmt.DASH,
                     (f"{fmt.num(cap.get('Cpk'), 3)} [{fmt.num(cpk_ci[0], 3)}, {fmt.num(cpk_ci[1], 3)}]"
                      if show and cpk_ci else fmt.num(cap.get("Cpk"), 3) if show else fmt.DASH),
                     _prob(record.get("out_of_tolerance")),
                     f"{fmt.pct(cap.get('expected_out_frac'), 3)} %" if cap.get("expected_out_frac") is not None else fmt.DASH])
        values = [v * fmt.MM_PER_M for v in record.get("samples") or [] if v is not None]
        charts += figure(plots.histogram(values, "mm", f"{record['id']}: {len(values)} samples",
                                         lower=_mm(record.get("lower_m")), upper=_mm(record.get("upper_m")),
                                         nominal=_mm(record.get("nominal_m")), reference=_mm(record.get("nominal_run_m"))),
                         f"{record['id']} distribution",
                         f"{record['id']}: the samples against the tolerance (LSL/USL); 'this run' is the run the "
                         "study was made around.")
    caption = (f"Cp, Cpk from mean and standard deviation (one-sided zones: Cpk = Cpu); Cpk with its interval; "
               f"P(out) observed with its Wilson interval, and expected for a normal distribution. "
               f"Quantiles {levels[0] if levels else ''} and {levels[-1] if levels else ''}.")
    out = table(["characteristic", "kind", "n", "mean (mm)", "std (µm)", "low quantile (mm)", "high quantile (mm)",
                 "limits (mm)", "Cp", "Cpk [interval]", "P(out) observed", "P(out) normal"], rows,
                numeric={2, 3, 4, 5, 6, 7, 8, 9}, caption=caption, empty="The samples report no inspected dimension.")
    target = (study.get("parameters") or {}).get("capability_target")
    if target is not None:
        out += note(f"Capable: Cpk at least {fmt.num(target, 3)}; Cp and Cpk are not reported below "
                    f"{(study.get('parameters') or {}).get('min_samples_capability')} samples.")
    return out + charts


def _mm(value):
    number = fmt.as_float(value)
    return None if number is None else number * fmt.MM_PER_M


def _findings(results: dict) -> str:
    stats = results.get("findings") or {}
    rows = [[r["category"], _prob(r["probability"]), fmt.num(r["mean_count"], 3), r["worst"]]
            for r in stats.get("by_category") or []]
    out = table(["category", "P(at least one) [interval]", "mean per run", "worst severity"], rows, numeric={2},
                empty="No sample raised a finding.")
    rows = [[r["category"], r["location"], _prob(r["probability"]), fmt.num(r["mean_count"], 3), r["worst"]]
            for r in stats.get("by_location") or []]
    if rows:
        out += "<h4>By program location</h4>" + table(["category", "where", "P(at least one) [interval]", "mean per run",
                                                       "worst severity"], rows, numeric={3})
    return out


def _scalars(results: dict) -> str:
    rows = []
    for record in results.get("scalars") or []:
        name = record["name"]
        rows.append([name, record.get("n"), _value(name, record.get("mean")), _value(name, record.get("std")),
                     _value(name, record.get("min")), _value(name, record.get("max"))])
    return table(["result", "n", "mean", "std", "min", "max"], rows, numeric={1, 2, 3, 4, 5},
                 empty="The samples report no scalar results.")


def _sensitivity(study: dict) -> str:
    results = study.get("results") or {}
    out = para("Spearman rank correlation of each factor with each result (sign: the result grows with the "
               "factor), its p-value, and the partial rank correlation that removes the other factors' effect; "
               "factors are ranked by |rank correlation|.")
    for record in results.get("sensitivity") or []:
        ranking = [r for r in record["ranking"] if r["rho"] is not None][:TOP_FACTORS]
        if not ranking:
            out += note(f"{record['output']}: constant over the samples (or no varied factor).")
            continue
        rows = [[r["factor"], fmt.signed(r["rho"], 3), fmt.num(r["p"], 3), "yes" if r["significant"] else "no",
                 fmt.signed(r["prcc"], 3) if r["prcc"] is not None else fmt.DASH] for r in ranking]
        out += f"<h4>{esc(record['output'])}</h4>" + table(["factor", "rank correlation", "p", "significant", "partial"],
                                                           rows, numeric={1, 2, 4})
        if record["kind"] == "dimension":
            out += figure(plots.tornado([r["factor"] for r in ranking], [r["rho"] for r in ranking],
                                        f"{record['output']}: rank correlation", "Spearman rank correlation"),
                          f"{record['output']} sensitivity", "Which tolerances move this dimension, and in which "
                                                             "direction (red: up, blue: down).")
    return out


def study_page(study: dict) -> str:
    project = study.get("project") or {}
    header = Raw(f"<h1>Uncertainty study: {esc(project.get('name') or '')}</h1>"
                 + kv([("project", project.get("path", fmt.DASH)), ("base run", study.get("base_run") or fmt.DASH),
                       ("samples", f"{(study.get('results') or {}).get('samples_ok', 0)} ok, "
                                   f"{(study.get('results') or {}).get('samples_failed', 0)} failed")]))
    section = Section("study", "Results", str(html_section(study, heading_level=3)))
    return document(f"Uncertainty study: {project.get('name') or ''}", [section], header=header,
                    running_title="CNC Twin uncertainty study", with_toc=False)


__all__ = ["html_section", "study_page"]
