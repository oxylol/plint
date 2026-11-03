"""Plots of every recorded time series, and the channel table (the plots' table view).

Every channel of channels.json is plotted: tick- and macro-tier channels from their series (each
column a line, the band its minimum and maximum within a plot bucket), fast channels from their
tick-tier envelopes (``.min/.mean/.max``) and, for the recorded fast windows, the window with the
highest peak. Panels are grouped by domain, at most :data:`PANELS_PER_FIGURE` per figure. Columns
found in the series but not declared in channels.json are plotted too and listed as undeclared.
"""

from __future__ import annotations

import numpy as np

from . import fmt, plots
from .context import Context
from .html import Section, figure, note, para, table
from .rundir import TierSummary, split_column

PANELS_PER_FIGURE = 6      # presentation: small multiples per image (one image stays readable on a phone)


def _column_key(column: str):
    name, index, stat = split_column(column)
    numbers = [int(n) for n in index.strip("[]").split("][") if n] if index else []
    return name, numbers, stat


def _columns_of(summary: TierSummary | None, channel: str, stat: str = "") -> list[str]:
    if summary is None:
        return []
    found = [c for c in summary.columns if split_column(c)[0] == channel and split_column(c)[2] == stat
             and summary.columns[c].finite]
    return sorted(found, key=_column_key)


def _fold(summary: TierSummary, columns: list[str]):
    """Across many columns: per bucket the lowest minimum, the highest maximum and the mean of means."""
    stats = [summary.columns[c] for c in columns]
    count = np.sum([s.b_cnt for s in stats], axis=0)
    used = count > 0
    lo = np.min([np.where(s.b_cnt > 0, s.b_min, np.inf) for s in stats], axis=0)[used]
    hi = np.max([np.where(s.b_cnt > 0, s.b_max, -np.inf) for s in stats], axis=0)[used]
    total = np.sum([s.b_sum for s in stats], axis=0)[used]
    t_s = 0.5 * (summary.b_tick_lo[used] + summary.b_tick_hi[used]).astype(float) * summary.tick_s
    return t_s, total / count[used], lo, hi


def _lines(summary: TierSummary, name: str, columns: list[str], shown, envelope: dict | None = None):
    if len(columns) > plots.MAX_SERIES:
        if envelope:
            t_s, mean, lo, hi = _fold(summary, envelope["mean"])
            if envelope["min"] and envelope["max"]:
                low, high = _fold(summary, envelope["min"])[2], _fold(summary, envelope["max"])[3]
                if len(low) == len(t_s) and len(high) == len(t_s):
                    lo, hi = low, high
        else:
            t_s, mean, lo, hi = _fold(summary, columns)
        return [plots.Line(f"all {len(columns)} columns", t_s, shown(mean), shown(lo), shown(hi))], \
            f"{len(columns)} columns folded: mean, band min to max"
    lines = []
    for column in columns:
        t_s, mean, lo, hi = summary.curve(column)
        label = column[len(name):].replace(".mean", "") or name
        if envelope:
            base = column[: -len(".mean")]
            low = summary.curve(base + ".min")
            high = summary.curve(base + ".max")
            lo = low[2] if low is not None and len(low[2]) == len(t_s) else lo
            hi = high[3] if high is not None and len(high[3]) == len(t_s) else hi
        lines.append(plots.Line(label, t_s, shown(mean), shown(lo), shown(hi)))
    return lines, ""


def _panel(ctx: Context, channel: dict) -> tuple[plots.Panel | None, dict]:
    """The panel of one channel and its statistics row (in display units)."""
    name, tier, unit = channel["name"], channel.get("tier", ""), channel.get("unit", "")
    envelope = None
    if tier == "fast":
        summary = ctx.tier("tick")
        columns = _columns_of(summary, name, "mean")
        envelope = {"mean": columns, "min": _columns_of(summary, name, "min"), "max": _columns_of(summary, name, "max")}
        source = "tick-tier envelope of the fast channel"
    else:
        summary = ctx.tier(tier) if tier in ("tick", "macro") else None
        columns = _columns_of(summary, name)
        source = f"{tier} tier"
    stats = [summary.columns[c] for c in columns] if summary is not None else []
    low_stats, high_stats = stats, stats
    if envelope and summary is not None:          # the range from the envelope's minima and maxima
        low_stats = [summary.columns[c] for c in envelope["min"]] or stats
        high_stats = [summary.columns[c] for c in envelope["max"]] or stats
    magnitude = max((s.max_abs for s in low_stats + high_stats if s.max_abs is not None), default=None)
    shown = fmt.display_unit(unit, magnitude)
    means = [s.mean for s in stats if s.mean is not None]
    row = {"name": name, "tier": tier, "unit": unit, "domain": channel.get("domain", ""), "doc": channel.get("doc", ""),
           "columns": len(columns), "declared": _declared_columns(channel), "display": shown.label,
           "min": min((s.min for s in low_stats if s.min is not None), default=None),
           "max": max((s.max for s in high_stats if s.max is not None), default=None),
           "mean": float(np.mean(means)) if means else None, "source": source}
    if summary is None or not columns:
        return None, row
    lines, folded = _lines(summary, name, columns, shown, envelope)
    return plots.Panel(name, shown.label, lines, note=folded), row


def _declared_columns(channel: dict) -> int:
    count = 1
    for size in channel.get("max_shape") or []:
        count *= int(size)
    return count


def series_section(ctx: Context) -> Section:
    run = ctx.run
    channels = list(run.channels)
    declared = {c["name"] for c in channels}
    for tier in ("tick", "macro"):
        summary = ctx.tier(tier)
        for column in sorted(summary.columns if summary else []):
            name, _, stat = split_column(column)
            if name not in declared and not stat:
                channels.append({"name": name, "tier": tier, "unit": "", "domain": name.split(".")[0],
                                 "doc": "not declared in channels.json", "undeclared": True})
                declared.add(name)
    rows, figures = [], []
    by_domain: dict[str, list[plots.Panel]] = {}
    for channel in sorted(channels, key=lambda c: (c.get("domain", ""), c["name"])):
        panel, row = _panel(ctx, channel)
        rows.append(row)
        if panel is not None:
            by_domain.setdefault(channel.get("domain", ""), []).append(panel)
    for domain, panels in sorted(by_domain.items()):
        for start in range(0, len(panels), PANELS_PER_FIGURE):
            chunk = panels[start:start + PANELS_PER_FIGURE]
            png = plots.series_panels(chunk)
            suffix = f" ({start // PANELS_PER_FIGURE + 1})" if len(panels) > PANELS_PER_FIGURE else ""
            figures.append(figure(png, f"{domain} time series", f"{domain}{suffix}: "
                                  + ", ".join(p.title for p in chunk)))
    body = para("Each line is a column of a channel, averaged per plot bucket; the shaded band spans the minimum "
                "and maximum inside the bucket, so short peaks stay visible.") if figures else \
        note("The run directory has no tick- or macro-tier series to plot.")
    body += "".join(figures)
    body += _fast_windows(ctx)
    body += "<h3>Channels</h3>" + _channel_table(rows)
    return Section("series", "Time series", body, lead="Every recorded channel over the run, grouped by domain.")


def _channel_table(rows: list[dict]) -> str:
    out = []
    for row in rows:
        shown = fmt.display_unit(row["unit"], max(abs(v) for v in (row["min"], row["max"]) if v is not None)
                                 if row["min"] is not None else None)
        out.append([row["name"] + (" (undeclared)" if row["doc"] == "not declared in channels.json" else ""),
                    row["tier"], row["unit"] or fmt.DASH, f"{row['columns']} / {row['declared']}",
                    fmt.num(shown(row["min"]) if row["min"] is not None else None),
                    fmt.num(shown(row["mean"]) if row["mean"] is not None else None),
                    fmt.num(shown(row["max"]) if row["max"] is not None else None), shown.label, row["doc"]])
    return table(["channel", "tier", "SI unit", "columns with data / declared", "min", "mean", "max", "shown in",
                  "what it is"], out, numeric={4, 5, 6},
                 caption="All channels of channels.json with their range over the run (the table view of the plots; "
                         "fast channels from their tick-tier envelopes).", empty="channels.json lists no channels.")


def _fast_windows(ctx: Context) -> str:
    run = ctx.run
    fast = [c for c in run.channels if c.get("tier") == "fast"]
    if not fast or not run.series_parts("fast"):
        return note("No fast-tier windows were recorded.") if fast else ""
    columns: dict[str, list[str]] = {}
    for channel in fast:
        count = _declared_columns(channel)
        columns[channel["name"]] = [channel["name"]] if not channel.get("max_shape") else \
            [f"{channel['name']}[{i}]" for i in range(count)]
    windows, best = run.fast_windows(columns)
    out = "<h3>Fast windows</h3>" + para(
        f"{len(windows)} windows recorded at L0 resolution, {sum(w.rows for w in windows)} samples, from "
        f"{fmt.seconds(min(w.tick0 for w in windows) * run.tick_s)} s to "
        f"{fmt.seconds(max(w.tick1 for w in windows) * run.tick_s)} s.") if windows else ""
    for channel in fast:
        window = best.get(channel["name"])
        if window is None or window.t_s is None:
            continue
        magnitude = window.peak if window.peak > 0 else None
        shown = fmt.display_unit(channel.get("unit", ""), magnitude)
        values = [(c[len(channel["name"]):] or channel["name"], shown(v)) for c, v in window.values.items()
                  if np.isfinite(v).any()]
        png = plots.window_plot(window.t_s - window.t_s[0], values[:plots.MAX_SERIES],
                                f"{channel['name']} from {fmt.seconds(window.t_s[0])} s", shown.label)
        out += figure(png, f"{channel['name']} fast window",
                      f"The recorded window with the highest |{channel['name']}| "
                      f"({fmt.num(shown(window.peak))} {shown.label}); time from the window start.")
    return out


__all__ = ["series_section", "PANELS_PER_FIGURE"]
