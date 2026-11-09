"""Convergence curves of every study (docs/img/convergence/<study>.svg) from tools/convergence/results/<study>.json.

    python tools/convergence/plots.py [study ...]        (no simulator: Matplotlib only; deterministic SVG)

One small multiple per quantity: the value at each level against the refined setting's step size (log axis, finer to
the left), the reference (Richardson estimate or finest level) as a hairline, the documented tolerance as a band
around it, the shipped default marked; a last panel holds the CPU time of each level. SVGs carry fixed metadata (no
date, a fixed hash salt), so the same results give the same bytes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE.parents[1] / "docs" / "img" / "convergence"

# reference palette of the data-viz method (light surface; validated: blue / orange pass every check)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
DEFAULT = "#eb6834"

LABELS = {
    "F_mean_N": ("mean force magnitude", "N"), "Fx_N": ("mean Fx (T0)", "N"), "Fy_N": ("mean Fy (T0)", "N"),
    "Fz_N": ("mean Fz (T0)", "N"), "F_peak_N": ("peak force", "N"),
    "F_peak_corner_N": ("peak force in the corner", "N"),
    "torque_Nm": ("mean torque", "N m"), "chip_volume_analytic_mm3": ("chip per tooth (analytic)", "mm3"),
    "chip_volume_field_mm3": ("chip per tooth (field)", "mm3"),
    "removed_mm3": ("removed volume", "mm3"),
    "engaged_deg": ("engaged arc", "deg"), "engaged_max_deg": ("largest engaged arc", "deg"),
    "wall_dev_um": ("wall position - nominal", "um"), "wall_lo_dev_um": ("slot wall (-y) - nominal", "um"),
    "wall_hi_dev_um": ("slot wall (+y) - nominal", "um"), "wall_bottom_dev_um": ("bottom wall - nominal", "um"),
    "wall_right_dev_um": ("right wall - nominal", "um"), "chip_tooth_spread": ("tooth-to-tooth chip spread", "-"),
    "vib_pp_um": ("vibration peak-to-peak", "um"), "onset_depth_mm": ("chatter onset depth (5 N/um)", "mm"),
    "Ra_um": ("Ra of the turned surface", "um"), "Rt_um": ("Rt of the turned surface", "um"),
    "tool_rise_K": ("tool body rise", "K"), "edge_rise_K": ("edge peak rise", "K"), "rake_rise_K": ("rake rise", "K"),
    "VB_growth_um": ("VB growth", "um"), "bulk_rise_K": ("workpiece bulk rise", "K"),
    "bulk_max_rise_K": ("workpiece hottest cell rise", "K"), "surface_rise_K": ("contact surface rise", "K"),
    "preheat_rise_K": ("preheat rise (material entering the cut)", "K"),
    "rake_rise_mean_K": ("rake rise, 0.5 s mean", "K"), "edge_rise_mean_K": ("edge peak rise, 0.5 s mean", "K"),
    "preheat_rise_mean_K": ("preheat rise, 0.5 s mean", "K"),
    "expansion_um": ("thermal expansion at the contact", "um"),
}


def _style():
    plt.rcParams.update({
        "svg.hashsalt": "cnctwin-convergence", "svg.fonttype": "none", "font.family": "DejaVu Sans",
        "font.size": 8.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlesize": 9, "axes.titlecolor": INK, "axes.titleweight": "normal", "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.grid": True, "axes.axisbelow": True, "lines.solid_capstyle": "round", "path.simplify": False,
    })


def _fmt_h(x, _pos=None):
    if x <= 0:
        return ""
    if x >= 1:
        return f"{x:g}"
    return f"{x:.3g}"


def plot_study(doc: dict, path: Path) -> Path:
    _style()
    analysis = doc.get("analysis", {})
    names = [q for q in analysis if analysis[q].get("values")]
    levels = [lv for lv in doc.get("levels", []) if lv.get("quantities")]
    n = len(names) + 1
    cols = 3 if n > 4 else n
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.55 * rows), squeeze=False)
    default_label = doc.get("default_label")
    for k, qname in enumerate(names):
        ax = axes[k // cols][k % cols]
        a = analysis[qname]
        hs, vs = a["h"], a["values"]
        ref, tol = a.get("reference"), a["tol"]
        label, unit = LABELS.get(qname, (qname, ""))
        if ref is not None:
            half = tol * abs(ref) if a["relative"] else tol
            ax.axhspan(ref - half, ref + half, color=SERIES, alpha=0.10, lw=0, zorder=0)
            ax.axhline(ref, color=MUTED, lw=1.0, zorder=1)
        ax.plot(hs, vs, color=SERIES, lw=2, zorder=2)
        ax.scatter(hs, vs, s=36, color=SERIES, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        dh = a.get("default_h")
        if "default_value" in a:
            ax.scatter([dh], [a["default_value"]], s=64, color=DEFAULT, edgecolors=SURFACE, linewidths=2, zorder=4)
            ax.annotate("default", (dh, a["default_value"]), xytext=(4, 6), textcoords="offset points", fontsize=7.5,
                        color=INK2)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=2.0))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_h))
        ax.set_title(label, loc="left")
        ax.set_ylabel(unit)
        status = a.get("status", "")
        order = a.get("order")
        how = {"richardson": f"Richardson p {order:.2f}" if order else "Richardson", "flat": "flat",
               "oscillating": "oscillating", "finest": "finest", "exact": "exact"}.get(status, status)
        band = f"\u00b1{100 * tol:g} %" if a["relative"] else f"\u00b1{tol:g} {unit}"
        ax.set_title(f"{how} \u00b7 {band}", loc="right", fontsize=6.8, color=MUTED)
        ax.margins(x=0.08, y=0.3)
    # cost panel
    ax = axes[(n - 1) // cols][(n - 1) % cols]
    pts = [(lv["h"], (lv.get("cost") or {}).get("cpu_total_s")) for lv in levels]
    pts = sorted((h, c) for h, c in pts if c)
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES, lw=2)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=36, color=SERIES, edgecolors=SURFACE, linewidths=1.5,
                   zorder=3)
        d = [lv for lv in levels if lv["label"] == default_label]
        if d and (d[0].get("cost") or {}).get("cpu_total_s"):
            ax.scatter([d[0]["h"]], [d[0]["cost"]["cpu_total_s"]], s=64, color=DEFAULT, edgecolors=SURFACE,
                       linewidths=2, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=2.0))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_h))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.set_title("CPU time of the run", loc="left")
        ax.set_ylabel("s")
        ax.margins(x=0.08, y=0.25)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    for r in range(rows):
        for c in range(cols):
            if r == rows - 1 or (r * cols + c + cols) >= n:
                axes[r][c].set_xlabel(doc.get("h_name", ""), color=INK2)
    fig.suptitle(doc.get("title", doc.get("study", "")), x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "cnctwin tools/convergence/plots.py"})
    plt.close(fig)
    return path


CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")     # reference order; validated (light)
HISTORY = (("rake_K", "rake temperature rise (K)", 293.15, 1.0), ("VB_m", "flank wear VB (um)", 0.0, 1e6),
           ("body_K", "tool body rise (K)", 293.15, 1.0), ("bulk_K", "workpiece bulk rise (K)", 293.15, 1.0))


def plot_history(doc: dict, path: Path) -> Path | None:
    """Time histories of the whole-loop levels (details.series): one line per level, the evaluation moment marked."""
    levels = sorted([lv for lv in doc.get("levels", []) if (lv.get("details") or {}).get("series")],
                    key=lambda lv: float(lv["h"]))
    if len(levels) < 3:
        return None
    _style()
    fig, axes = plt.subplots(1, len(HISTORY), figsize=(3.3 * len(HISTORY), 2.7), squeeze=False)
    t_eval = None
    for k, (key, title, offset, scale) in enumerate(HISTORY):
        ax = axes[0][k]
        for i, lv in enumerate(levels[:len(CATEGORICAL)]):
            ser = lv["details"]["series"]
            pts = [(t, (v - offset) * scale) for t, v, c in zip(ser["t"], ser.get(key, []), ser["in_cut"])
                   if v is not None and t is not None and math.isfinite(v) and t >= 3.5]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=CATEGORICAL[i], lw=2 if lv["label"] !=
                    doc.get("default_label") else 2.6, label=lv["label"])
            t_eval = t_eval or (lv.get("details") or {}).get("t_eval_s")
        if t_eval:
            ax.axvline(t_eval, color=MUTED, lw=1.0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("simulated time (s)")
        ax.margins(x=0.02)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=len(labels), labelcolor=INK2)
    fig.suptitle(doc.get("title", "") + ": time histories (vertical line: the comparison moment)", x=0.01, ha="left",
                 color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "cnctwin tools/convergence/plots.py"})
    plt.close(fig)
    return path


def main(argv) -> int:
    names = argv or sorted(p.stem for p in RESULTS.glob("*.json"))
    for name in names:
        p = RESULTS / f"{name}.json"
        if not p.exists():
            print(f"no results for {name}")
            continue
        doc = json.loads(p.read_text())
        if not doc.get("analysis"):
            print(f"{name}: no analysis yet")
            continue
        out = plot_study(doc, OUT / f"{name}.svg")
        print(out)
        hist = plot_history(doc, OUT / f"{name}_history.svg")
        if hist is not None:
            print(hist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
