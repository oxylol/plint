"""Text labels and colour bars drawn onto finished pictures (Pillow with the DejaVu Sans font that ships with
Matplotlib, so every build draws the same glyphs; Pillow's own default font when that file is missing)."""

from __future__ import annotations

import functools
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import colour as C
from .scene import ColorBar, Label


@functools.lru_cache(maxsize=None)
def font(size_px: int):
    size_px = max(int(size_px), 6)
    try:
        import matplotlib

        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(path, size_px)
    except (ImportError, OSError):
        return ImageFont.load_default(size=size_px)


def _rgb255(srgb) -> tuple:
    return tuple(int(v) for v in C.to_uint8(srgb))


def draw_label(image: Image.Image, xy, label: Label, size_px: int) -> None:
    draw = ImageDraw.Draw(image)
    face = font(size_px)
    anchor = {"left": "lm", "centre": "mm", "center": "mm", "right": "rm"}.get(label.align, "lm")
    fill = _rgb255(label.srgb)
    halo = (255, 255, 255) if sum(fill) < 384 else (0, 0, 0)
    draw.text((float(xy[0]), float(xy[1])), label.text, font=face, fill=fill, anchor=anchor,
              stroke_width=max(1, size_px // 8), stroke_fill=halo)


def draw_colorbar(image: Image.Image, bar: ColorBar, x0: int, y0: int, width: int, height: int,
                  size_px: int) -> None:
    """Vertical colour bar with its title, unit and tick values; the top is ``vmax``."""
    rows = np.linspace(1.0, 0.0, max(height, 2))
    values = bar.vmin + rows * (bar.vmax - bar.vmin)
    rgb = C.to_uint8(C.apply_colormap(values, bar.vmin, bar.vmax, bar.cmap))
    strip = np.repeat(rgb[:, None, :], max(width, 1), axis=1)
    image.paste(Image.fromarray(strip, "RGB"), (int(x0), int(y0)))
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0 - 1, y0 - 1, x0 + width, y0 + height], outline=(40, 40, 40), width=1)
    face = font(size_px)
    ticks = bar.ticks if bar.ticks is not None else C.nice_ticks(bar.vmin, bar.vmax)
    span = bar.vmax - bar.vmin if bar.vmax > bar.vmin else 1.0
    for value in ticks:
        frac = (value - bar.vmin) / span
        if not -1e-9 <= frac <= 1.0 + 1e-9:
            continue
        y = y0 + (1.0 - frac) * (height - 1)
        draw.line([x0 + width, y, x0 + width + max(3, size_px // 3), y], fill=(40, 40, 40), width=1)
        draw.text((x0 + width + max(5, size_px // 2), y), _format_tick(value, ticks), font=face, fill=(25, 25, 25),
                  anchor="lm", stroke_width=1, stroke_fill=(255, 255, 255))
    title = f"{bar.title} ({bar.unit})" if bar.unit else bar.title
    draw.text((x0 + width, y0 - size_px), title, font=face, fill=(25, 25, 25), anchor="rs",
              stroke_width=1, stroke_fill=(255, 255, 255))


def _format_tick(value: float, ticks: list) -> str:
    step = min((abs(b - a) for a, b in zip(ticks, ticks[1:])), default=abs(value) or 1.0)
    if step >= 1.0 or step == 0.0:
        return f"{value:.0f}"
    decimals = min(6, max(0, int(np.ceil(-np.log10(step) - 1e-9))))
    return f"{value:.{decimals}f}"
