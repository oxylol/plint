"""Report sections that put the results in context: the model approximations relevant to the run,
the estimated database values it used, and the pictures (rendered images and the tool path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import approximations, fmt, plots
from .context import Context
from .html import Section, code, esc, figure, note, para, table


# ----------------------------------------------------------------------------- approximations

def active_domains(ctx: Context) -> set[str]:
    run = ctx.run
    domains = {str(c.get("domain", "")) for c in run.channels}
    domains |= {str(f.get("domain", "")) for f in run.findings}
    domains |= run.trace.domains()
    if run.production is not None:
        domains.add("production")
    return {d for d in domains if d}


def approximations_section(ctx: Context, path: Path | None = None) -> Section:
    listed = ctx.run.run.get("approximations")
    if isinstance(listed, list) and listed:
        rows = [[str(a.get("domain", fmt.DASH)), str(a.get("approximation", a.get("text", ""))),
                 str(a.get("error", fmt.DASH))] for a in listed if isinstance(a, dict)]
        body = table(["domain", "approximation", "error bound"], rows, caption="As recorded by the run (run.json).")
        return Section("approximations", "Model approximations", body)
    path = path or approximations.default_path()
    if not path.is_file():
        ctx.run.note("docs/approximations.md", "docs/approximations.md is not in this build: the model "
                                               "approximations cannot be listed")
        return Section("approximations", "Model approximations",
                       note("docs/approximations.md is not in this build, and run.json lists no approximations."))
    sections = approximations.read_sections(path)
    picked, rule = approximations.relevant(sections, active_domains(ctx))
    body = para(f"From docs/approximations.md: {len(picked)} of {len(sections)} sections ({rule}).")
    for section in picked:
        body += f"<h4>{esc(section.heading)}</h4>"
        body += approximations.render(section)
    return Section("approximations", "Model approximations", body,
                   lead="What the models leave out, and how large an error that can make, for this run.")


# ----------------------------------------------------------------------------- estimates

def estimates_section(ctx: Context) -> Section:
    run = ctx.run
    estimates = run.run.get("estimates", run.run.get("db_estimates"))
    source = "run.json"
    if estimates is None and run.has("estimates.json"):
        document = run.json("estimates.json")
        estimates = document.get("estimates") if isinstance(document, dict) else document
        source = "estimates.json"
    if not isinstance(estimates, list):
        run.note("estimates", "run.json has no estimates list (and there is no estimates.json): the estimated "
                              "database values used cannot be listed")
        return Section("estimates", "Estimated database values used",
                       note("The run directory does not list the estimated database values it used."))
    rows = []
    for item in sorted((e for e in estimates if isinstance(e, dict)),
                       key=lambda e: (str(e.get("obj", "")), str(e.get("file", "")), int(e.get("line") or 0))):
        rows.append([item.get("obj", fmt.DASH), f"{item.get('file', '')}:{item.get('line', '')}",
                     code(item.get("path", "")), str(item.get("value", "")), item.get("note") or "(no note)"])
    body = table(["object", "file:line", "key", "value as written", "how it was estimated"], rows,
                 caption=f"Values whose source is an estimate (src: est), from {source}.",
                 empty="The run used no estimated database value.")
    return Section("estimates", "Estimated database values used", body,
                   lead="Numbers without a published source: the first to check when a result looks wrong.")


# ----------------------------------------------------------------------------- pictures

def _json_file(path: Path):
    try:
        import json
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _render_manifest(run) -> dict:
    """render.json (cnctwin.render, written with the pictures): caption and time of each picture, the video."""
    return _json_file(run.path / "render.json") or {}


def _media_lines(run, manifest: dict) -> str:
    """One line each on the video and the sound of the run (files beside the report, not embedded)."""
    body = ""
    video = manifest.get("video") if isinstance(manifest.get("video"), dict) else None
    if video and (run.path / "video.mp4").is_file():
        body += para(f"Video: video.mp4, {video.get('frames')} frames at {fmt.num(video.get('fps'))} frames/s, "
                     f"{fmt.num(video.get('speed'))} simulated seconds per video second, from "
                     f"{fmt.num(video.get('t0_s'))} s to {fmt.num(video.get('t1_s'))} s of the run, the camera "
                     "following the tool (cnctwin.render).")
    audio = _json_file(run.path / "audio.json")
    if audio and (run.path / "audio.wav").is_file():
        levels = audio.get("levels") or {}
        listener = (audio.get("listener") or {}).get("kind", "operator")
        body += para(f"Sound: audio.wav, {fmt.num(audio.get('duration_s'))} s at {fmt.num(audio.get('sample_rate_Hz'))}"
                     f" Hz, heard by the {listener}: Leq {fmt.num(levels.get('Leq_dB'))} dB, LAeq "
                     f"{fmt.num(levels.get('LAeq_dB'))} dB(A) (cnctwin.sound; levels are estimates within about "
                     "10 dB, frequencies and timing are those of the run).")
    return body


def pictures_section(ctx: Context) -> Section:
    run = ctx.run
    body = ""
    drawn = toolpath_figure(ctx)
    if drawn:
        body += drawn
    images = run.images
    manifest = _render_manifest(run)
    described = {str(e.get("file")): e for e in manifest.get("images") or [] if isinstance(e, dict)}
    order = {name: i for i, name in enumerate(described)}
    for rel, data, mime in sorted(images, key=lambda item: (order.get(item[0], len(order)), item[0])):
        entry = described.get(rel)
        if entry is not None and entry.get("caption"):
            when = f", t = {fmt.num(entry.get('time_s'))} s" if entry.get("time_s") is not None else ""
            caption = f"{entry['caption']} ({rel}{when}, cnctwin.render)"
        else:
            caption = f"{rel} (cnctwin.render)"
        body += figure(data, Path(rel).stem.replace("_", " "), caption, mime=mime)
    if not images:
        body += note("images/ holds no rendered pictures: only the drawings made from the run's data are shown.")
    body += _media_lines(run, manifest)
    return Section("pictures", "Pictures", body)


def toolpath_figure(ctx: Context, title: str = "Tool tip path in the stock frame") -> str:
    frames = ctx.run.frames()
    if frames is None or "tip_S_m" not in frames:
        return ""
    ticks = frames["tick"]
    tip = frames["tip_S_m"] * fmt.MM_PER_M
    blocks = sorted(ctx.run.timeline, key=lambda b: int(b["tick0"]))
    if blocks:
        starts = np.array([int(b["tick0"]) for b in blocks], np.int64)
        kinds = np.array([1 if b.get("kind") == "rapid" else 0 for b in blocks], np.int8)
        rapid = kinds[np.clip(np.searchsorted(starts, ticks, side="right") - 1, 0, len(starts) - 1)]
    else:
        rapid = np.zeros(len(ticks), np.int8)
    segments = []
    start = 0
    for i in range(1, len(ticks) + 1):
        if i == len(ticks) or rapid[i] != rapid[start]:
            segment = tip[max(0, start - 1):i]
            segments.append((segment, "rapid" if rapid[start] else "feed"))
            start = i
    stock = (ctx.run.scene or {}).get("stock") or {}
    box = None
    size = stock.get("size_m")
    if stock.get("kind", "block") == "block" and isinstance(size, list) and len(size) == 3:
        box = (np.zeros(3), np.asarray(size, float) * fmt.MM_PER_M)
    zero = (ctx.run.scene or {}).get("work_zero_S_m")
    png = plots.toolpath(segments, box, title, None if zero is None else np.asarray(zero, float) * fmt.MM_PER_M)
    caption = (f"From frames.bin ({frames['count']} L1 ticks, {len(ticks)} drawn): feed moves blue, rapids grey, "
               "stock outline shaded, work zero +.")
    return figure(png, "tool path", caption)


__all__ = ["approximations_section", "estimates_section", "pictures_section", "toolpath_figure", "active_domains"]
