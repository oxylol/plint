"""The model approximations relevant to a run, from docs/approximations.md (when the build has it).

The document is read as Markdown sections (``#`` headings). A section is relevant to a run when its
heading or its text names a domain the run used (a domain of its channels, findings or trace
events), or its heading names one of the topics every run depends on (:data:`ALWAYS`); when no section names any domain the
document is not organised by domain and every section is listed. The text is rendered by a small
Markdown subset (paragraphs, ``*``/``-`` bullets, ``|`` tables, ``code``, ``**bold**``), escaped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .html import Raw, esc

DOMAINS = ("controller", "motion", "tooling", "removal", "cutting", "mechanics", "chips", "thermal", "coolant",
           "wear", "holding", "structure", "dynamics", "surface", "inspect", "production", "sanity", "sound",
           "render")
ALWAYS = ("general", "scope", "core", "loop", "clock", "frames", "conservation", "recording", "determinism",
          "numerics", "all runs")


@dataclass
class DocSection:
    heading: str
    level: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def names(self, words, heading_only: bool = False) -> set[str]:
        haystack = (self.heading if heading_only else self.heading + "\n" + self.text).lower()
        return {w for w in words if re.search(rf"\b{re.escape(w)}\b", haystack)}


def default_path() -> Path:
    from ..paths import DOCS
    return Path(DOCS) / "approximations.md"


def read_sections(path: Path) -> list[DocSection]:
    sections: list[DocSection] = []
    current: DocSection | None = None
    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            current = DocSection(match.group(2).strip(), len(match.group(1)))
            sections.append(current)
            continue
        if current is None:
            current = DocSection("(introduction)", 1)
            sections.append(current)
        current.lines.append(line)
    return sections


def relevant(sections: list[DocSection], domains: set[str]) -> tuple[list[DocSection], str]:
    """(sections relevant to a run that used ``domains``, how they were chosen)."""
    tagged = [s for s in sections if s.names(DOMAINS)]
    if not tagged:
        return [s for s in sections if s.text.strip()], "the document is not organised by domain: all sections"
    used = {d.lower() for d in domains if d}
    picked = [s for s in sections if s.text.strip() and (s.names(used) or s.names(ALWAYS, heading_only=True))]
    return picked, ("sections naming a domain this run used (" + ", ".join(sorted(d for d in domains if d)) +
                    ") or a topic of every run")


def _inline(text: str) -> str:
    out = esc(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    return out


def render(section: DocSection) -> Raw:
    """A small Markdown subset to HTML (escaped)."""
    html: list[str] = []
    bullets: list[str] = []
    rows: list[list[str]] = []
    paragraph: list[str] = []

    def flush():
        if paragraph:
            html.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()
        if bullets:
            html.append("<ul>" + "".join(f"<li>{_inline(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()
        if rows:
            body = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in r if c.strip())]
            if body:
                head, *rest = body
                html.append('<div class="scroll"><table><thead><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head)
                            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r)
                                                               + "</tr>" for r in rest) + "</tbody></table></div>")
            rows.clear()

    for line in section.lines:
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("|"):
            if paragraph or bullets:
                flush()
            rows.append([c.strip() for c in stripped.strip("|").split("|")])
        elif re.match(r"^[*-]\s+", stripped):
            if paragraph or rows:
                flush()
            bullets.append(re.sub(r"^[*-]\s+", "", stripped))
        elif bullets and line.startswith((" ", "\t")):
            bullets[-1] += " " + stripped
        else:
            if bullets or rows:
                flush()
            paragraph.append(stripped)
    flush()
    return Raw("".join(html))


__all__ = ["DocSection", "read_sections", "relevant", "render", "default_path", "DOMAINS", "ALWAYS"]
