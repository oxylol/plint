"""Plots for reports: Matplotlib (Agg, object API, no pyplot state) to PNG bytes, deterministically.

* Same data, same bytes: a fixed style (``matplotlib.rcdefaults`` then :data:`RC`, inside an
  ``rc_context``), DejaVu fonts, PNG written without the ``Software`` text chunk (so the bytes do
  not depend on the Matplotlib version string) and with no date (PNG output never has one).
* Palette: the categorical order of the report's design system (8 hues, fixed order, never
  cycled past 8: more series fold into a min/max band), one-hue blue ramp for magnitudes,
  blue/red with a grey midpoint for signed quantities, status colours only next to a text label.
  Text uses the ink tokens, never a series colour. One y-scale per axes (no twin axes).
"""

from __future__ import annotations

import io
import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL = "#f0efec"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQUENTIAL = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
DIVERGING_NEG = "#2a78d6"
DIVERGING_POS = "#e34948"
WIDTH_IN = 4.8       # narrow figures keep 8 pt text readable when a phone shows them 390 px wide
DPI = 150
MAX_SERIES = len(PALETTE)


def rc() -> dict:
    from cycler import cycler
    return {
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.0, "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.labelsize": 8.0, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "axes.prop_cycle": cycler(color=list(PALETTE)), "lines.linewidth": 1.3, "lines.markersize": 4.0,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "legend.frameon": False, "legend.fontsize": 7.0, "legend.labelcolor": INK2,
        "path.simplify": True, "path.simplify_threshold": 0.3, "agg.path.chunksize": 20000,
        "axes.formatter.useoffset": False, "axes.formatter.limits": (-4, 5), "figure.dpi": DPI,
        "savefig.dpi": DPI, "image.cmap": "viridis", "svg.hashsalt": "cnctwin",
    }


@contextmanager
def style():
    import matplotlib
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        matplotlib.rcParams.update(rc())
        yield


def new_figure(height_in: float, nrows: int = 1, ncols: int = 1, width_in: float = WIDTH_IN, sharex: bool = False,
               squeeze: bool = True):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(width_in, height_in), dpi=DPI, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(nrows, ncols, sharex=sharex, squeeze=squeeze)
    return figure, axes


def to_png(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=DPI, metadata={"Software": None})
    figure.clear()
    return buffer.getvalue()


def colormap(kind: str = "sequential"):
    from matplotlib.colors import LinearSegmentedColormap
    if kind == "diverging":
        return LinearSegmentedColormap.from_list("cnctwin-div", [DIVERGING_NEG, NEUTRAL, DIVERGING_POS])
    return LinearSegmentedColormap.from_list("cnctwin-seq", list(SEQUENTIAL))


def _finite(*arrays):
    mask = np.ones(len(arrays[0]), bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


# ----------------------------------------------------------------------------- time series

@dataclass
class Line:
    label: str
    t_s: np.ndarray
    y: np.ndarray
    lo: np.ndarray | None = None        # band (bucket minimum) or None
    hi: np.ndarray | None = None


@dataclass
class Panel:
    title: str
    unit: str                           # display unit label
    lines: list[Line] = field(default_factory=list)
    note: str = ""                      # shown at the right of the title (e.g. "max over 16 columns")
    step: bool = False


def series_panels(panels: list[Panel], height_per_panel_in: float = 1.55) -> bytes | None:
    """Small multiples stacked vertically, one y-scale each, a shared time axis (s)."""
    panels = [p for p in panels if any(np.isfinite(line.y).any() for line in p.lines)]
    if not panels:
        return None
    with style():
        figure, axes = new_figure(0.5 + height_per_panel_in * len(panels), nrows=len(panels), sharex=True,
                                  squeeze=False)
        for ax, panel in zip(axes[:, 0], panels):
            _draw_panel(ax, panel)
        axes[-1, 0].set_xlabel("time (s)")
        return to_png(figure)


def _draw_panel(ax, panel: Panel) -> None:
    for index, line in enumerate(panel.lines[:MAX_SERIES]):
        colour = PALETTE[index % MAX_SERIES]
        mask = _finite(line.t_s, line.y)
        if not mask.any():
            continue
        if line.lo is not None and line.hi is not None:
            band = mask & np.isfinite(line.lo) & np.isfinite(line.hi)
            if band.any() and np.any(line.hi[band] > line.lo[band]):
                ax.fill_between(line.t_s[band], line.lo[band], line.hi[band], color=colour, alpha=0.18, linewidth=0)
        drawstyle = "steps-post" if panel.step else "default"
        if mask.sum() == 1:
            ax.plot(line.t_s[mask], line.y[mask], "o", color=colour, label=line.label)
        else:
            ax.plot(line.t_s[mask], line.y[mask], color=colour, label=line.label, drawstyle=drawstyle)
    title = panel.title + (f" ({panel.unit})" if panel.unit and panel.unit != "-" else "")
    ax.set_title(title)
    if panel.note:
        ax.set_title(panel.note, loc="right", fontsize=7.0, color=INK2)
    if len([line for line in panel.lines[:MAX_SERIES] if np.isfinite(line.y).any()]) >= 2:
        ax.legend(loc="upper left", ncols=min(4, len(panel.lines)), handlelength=1.4)


def window_plot(t_s: np.ndarray, columns: list[tuple[str, np.ndarray]], title: str, unit: str) -> bytes | None:
    """One recorded fast window (raw samples)."""
    if not len(t_s):
        return None
    with style():
        figure, ax = new_figure(2.0)
        for index, (label, values) in enumerate(columns[:MAX_SERIES]):
            mask = _finite(t_s, values)
            if mask.any():
                ax.plot(t_s[mask] * 1e3, values[mask], color=PALETTE[index], label=label, linewidth=1.0)
        ax.set_title(title + (f" ({unit})" if unit and unit != "-" else ""))
        ax.set_xlabel("time (ms)")
        if len(columns) >= 2:
            ax.legend(loc="upper left", ncols=min(4, len(columns)))
        return to_png(figure)


# ----------------------------------------------------------------------------- bars

def hbar(labels: list[str], values: list[float], unit: str, title: str, *, value_labels: list[str] | None = None,
         colour: str = PALETTE[0]) -> bytes | None:
    """Horizontal bars of one series (one colour), largest first as given, values written at the bar end."""
    if not labels:
        return None
    with style():
        figure, ax = new_figure(0.7 + 0.24 * len(labels))
        y = np.arange(len(labels))[::-1]
        data = np.array([v if v is not None and math.isfinite(v) else 0.0 for v in values])
        ax.barh(y, data, color=colour, height=0.62, linewidth=0)
        ax.set_yticks(y, labels)
        ax.grid(axis="y", visible=False)
        ax.set_title(title + (f" ({unit})" if unit else ""))
        span = float(np.max(np.abs(data))) if len(data) else 0.0
        for yi, value, text in zip(y, data, value_labels or [""] * len(labels)):
            if text:
                ax.text(value + span * 0.01, yi, text, va="center", ha="left", fontsize=7.0, color=INK2)
        ax.set_xlim(min(0.0, float(data.min()) * 1.1), max(span * 1.18, 1e-30))
        return to_png(figure)


def signed_stack(rows: list[str], sources: list[str], values: np.ndarray, unit: str, title: str,
                 totals: np.ndarray | None = None) -> bytes | None:
    """Signed contributions per row stacked left (negative) and right (positive) of zero; one colour per
    source in fixed order (more than 8 sources fold into 'other'); the row total as a black tick."""
    if not rows or not sources:
        return None
    values = np.asarray(values, float)
    if len(sources) > MAX_SERIES:
        keep = MAX_SERIES - 1
        values = np.column_stack([values[:, :keep], values[:, keep:].sum(axis=1)])
        sources = list(sources[:keep]) + ["other"]
    with style():
        figure, ax = new_figure(0.9 + 0.3 * len(rows))
        y = np.arange(len(rows))[::-1]
        left_pos = np.zeros(len(rows))
        left_neg = np.zeros(len(rows))
        for j, source in enumerate(sources):
            column = np.nan_to_num(values[:, j])
            positive = np.where(column > 0, column, 0.0)
            negative = np.where(column < 0, column, 0.0)
            ax.barh(y, positive, left=left_pos, color=PALETTE[j], height=0.6, label=source, linewidth=0,
                    edgecolor=SURFACE)
            ax.barh(y, negative, left=left_neg, color=PALETTE[j], height=0.6, linewidth=0, edgecolor=SURFACE)
            left_pos += positive
            left_neg += negative
        if totals is not None:
            ax.plot(np.asarray(totals, float), y, "|", color=INK, markersize=12, markeredgewidth=1.6,
                    label="total deviation")
        ax.axvline(0.0, color=AXIS, linewidth=0.8)
        ax.set_yticks(y, rows)
        ax.grid(axis="y", visible=False)
        ax.set_title(title + (f" ({unit})" if unit else ""))
        ax.legend(loc="lower right", ncols=min(4, len(sources) + 1))
        return to_png(figure)


def tornado(labels: list[str], values: list[float], title: str, xlabel: str = "") -> bytes | None:
    """Signed bars from zero (negative blue, positive red), as given (sorted by the caller)."""
    if not labels:
        return None
    with style():
        figure, ax = new_figure(0.7 + 0.26 * len(labels))
        y = np.arange(len(labels))[::-1]
        data = np.array([v if v is not None and math.isfinite(v) else 0.0 for v in values])
        colours = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in data]
        ax.barh(y, data, color=colours, height=0.62, linewidth=0)
        ax.axvline(0.0, color=AXIS, linewidth=0.8)
        ax.set_yticks(y, labels)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-1.0, 1.0)
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        for yi, value in zip(y, data):
            ax.text(value + (0.02 if value >= 0 else -0.02), yi, f"{value:+.2f}", va="center",
                    ha="left" if value >= 0 else "right", fontsize=7.0, color=INK2)
        return to_png(figure)


def tooth_bars(values_um: list[float], limit_um: float | None, title: str) -> bytes | None:
    if not values_um:
        return None
    with style():
        figure, ax = new_figure(1.9)
        x = np.arange(1, len(values_um) + 1)
        ax.bar(x, values_um, color=PALETTE[0], width=0.6, linewidth=0)
        if limit_um is not None:
            ax.axhline(limit_um, color=STATUS["critical"], linewidth=1.0)
            ax.text(x[-1] + 0.45, limit_um, " VB limit", va="bottom", ha="right", fontsize=7.0, color=INK2)
        ax.set_xticks(x)
        ax.set_xlabel("tooth")
        ax.set_title(title + " (µm)")
        ax.grid(axis="x", visible=False)
        return to_png(figure)


# ----------------------------------------------------------------------------- maps and distributions

def heatmap(grid: np.ndarray, extent_mm: tuple[float, float, float, float], label: str, title: str, *,
            diverging: bool = False, vmin: float | None = None, vmax: float | None = None,
            overlay=None) -> bytes | None:
    """A 2-D field (rows = y, columns = x) over ``extent_mm`` (x0, x1, y0, y1), with a scale bar."""
    data = np.asarray(grid, float)
    if data.ndim != 2 or not np.isfinite(data).any():
        return None
    with style():
        aspect = (extent_mm[3] - extent_mm[2]) / max(extent_mm[1] - extent_mm[0], 1e-12)
        figure, ax = new_figure(min(4.8, max(2.2, 0.9 + 3.6 * aspect)))
        finite = data[np.isfinite(data)]
        low = float(np.min(finite)) if vmin is None else vmin
        high = float(np.max(finite)) if vmax is None else vmax
        if diverging:
            bound = max(abs(low), abs(high), 1e-30)
            low, high = -bound, bound
        if high <= low:
            high = low + max(abs(low) * 1e-6, 1e-12)
        cmap = colormap("diverging" if diverging else "sequential").with_extremes(bad=NEUTRAL)
        image = ax.imshow(np.ma.masked_invalid(data), origin="lower", extent=extent_mm, cmap=cmap, vmin=low,
                          vmax=high, interpolation="nearest", aspect="equal")
        if overlay is not None:
            overlay(ax)
        ax.grid(False)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(title)
        bar = figure.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
        bar.set_label(label, color=INK2)
        bar.outline.set_edgecolor(AXIS)
        return to_png(figure)


def histogram(values: np.ndarray, unit: str, title: str, *, lower=None, upper=None, nominal=None,
              reference=None) -> bytes | None:
    data = np.asarray(values, float)
    data = data[np.isfinite(data)]
    if not len(data):
        return None
    with style():
        figure, ax = new_figure(2.0)
        span = float(data.max() - data.min())
        if span > 0:
            edges = np.histogram_bin_edges(data, bins="auto")
        else:
            centre = float(data[0])
            half = max(abs(centre) * 1e-6, 1e-12)
            edges = np.array([centre - half, centre + half])
        ax.hist(data, bins=edges, color=PALETTE[0], linewidth=0.4, edgecolor=SURFACE)
        top = ax.get_ylim()[1]
        for value, name, colour in ((lower, "LSL", STATUS["critical"]), (upper, "USL", STATUS["critical"]),
                                    (nominal, "nominal", INK2), (reference, "this run", PALETTE[1])):
            if value is not None and math.isfinite(value):
                ax.axvline(value, color=colour, linewidth=1.1)
                ax.text(value, top * 0.98, f" {name}", va="top", ha="left", fontsize=7.0, color=INK2)
        ax.set_title(title + (f" ({unit})" if unit else ""))
        ax.set_ylabel("samples")
        ax.grid(axis="x", visible=False)
        return to_png(figure)


def stability_lobes(speed_rpm: np.ndarray, depth_mm: np.ndarray, points: list[dict], title: str) -> bytes | None:
    """Limiting depth of cut over spindle speed (stable below the curve) with the run's operating points."""
    if not len(speed_rpm):
        return None
    with style():
        figure, ax = new_figure(2.6)
        ax.fill_between(speed_rpm, 0.0, depth_mm, color=SEQUENTIAL[1], linewidth=0, label="stable")
        ax.plot(speed_rpm, depth_mm, color=PALETTE[0], label="stability limit")
        for point in points:
            stable = point.get("stable")
            marker = "o" if stable else "X"
            colour = STATUS["good"] if stable else STATUS["critical"]
            ax.plot([point["rpm"]], [point["ap_mm"]], marker, color=colour, markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=1.5)
            ax.annotate(point.get("label", ""), (point["rpm"], point["ap_mm"]), xytext=(6, 4),
                        textcoords="offset points", fontsize=7.0, color=INK2)
        ax.set_xlabel("spindle speed (rpm)")
        ax.set_ylabel("depth of cut (mm)")
        ax.set_ylim(bottom=0.0)
        ax.set_title(title)
        ax.legend(loc="upper right")
        return to_png(figure)


def toolpath(segments: list[tuple[np.ndarray, str]], stock_box_mm, title: str, zero_mm=None) -> bytes | None:
    """Top (XY) and side (XZ) views of the tool tip path in the stock frame: feed moves solid in slot 1,
    rapids thin in grey; the stock outline in the axis colour; the work zero as a cross."""
    if not segments:
        return None
    with style():
        figure, axes = new_figure(5.6, nrows=2)
        for ax, (i, j, name) in zip(axes, ((0, 1, "top view, X-Y"), (0, 2, "side view, X-Z"))):
            if stock_box_mm is not None:
                lo, hi = stock_box_mm
                ax.add_patch(_rect(lo[i], lo[j], hi[i] - lo[i], hi[j] - lo[j]))
            for points, kind in segments:
                if len(points) < 2:
                    continue
                if kind == "rapid":
                    ax.plot(points[:, i], points[:, j], color=MUTED, linewidth=0.7)
                else:
                    ax.plot(points[:, i], points[:, j], color=PALETTE[0], linewidth=1.2)
            if zero_mm is not None:
                ax.plot([zero_mm[i]], [zero_mm[j]], "+", color=INK, markersize=9, markeredgewidth=1.4)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("xyz"[i] + " (mm)")
            ax.set_ylabel("xyz"[j] + " (mm)")
            ax.set_title(name)
        figure.suptitle(title, x=0.01, ha="left", fontsize=9.0, color=INK)
        return to_png(figure)


def _rect(x, y, w, h):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fill=True, facecolor=NEUTRAL, edgecolor=AXIS, linewidth=1.0)


def scatter_xy(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> bytes | None:
    mask = _finite(np.asarray(x, float), np.asarray(y, float))
    if mask.sum() < 2:
        return None
    with style():
        figure, ax = new_figure(2.2, width_in=3.5)
        ax.plot(np.asarray(x)[mask], np.asarray(y)[mask], "o", color=PALETTE[0], markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return to_png(figure)


__all__ = ["PALETTE", "INK", "INK2", "MUTED", "GRID", "AXIS", "SURFACE", "STATUS", "SEQUENTIAL", "style",
           "new_figure", "to_png", "colormap", "Line", "Panel", "series_panels", "window_plot", "hbar",
           "signed_stack", "tornado", "tooth_bars", "heatmap", "histogram", "stability_lobes", "toolpath",
           "scatter_xy"]
