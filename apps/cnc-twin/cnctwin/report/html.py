"""Self-contained HTML: escaping, tables, figures (images embedded as data URIs), sections, the page.

The page is one file with inline CSS, no scripts and no external resources: it reads on a 390 px
phone (tables scroll sideways inside their box, images scale to the width) and prints to A4 through
WeasyPrint with the same markup (``@media print`` and ``@page`` rules). Colours follow the report's
palette (cnctwin.report.plots): ink tokens for text, status colours only with a text label.
"""

from __future__ import annotations

import base64
import html as _html
import re
from dataclasses import dataclass, field


class Raw(str):
    """HTML that is already escaped (or markup we built)."""


def esc(value) -> str:
    return _html.escape("" if value is None else str(value), quote=True)


def cell(value) -> str:
    return value if isinstance(value, Raw) else esc(value)


def join(parts) -> Raw:
    return Raw("".join(cell(p) for p in parts if p is not None and p != ""))


def anchor(*parts) -> str:
    """A stable element id from parts (letters, digits, ``-`` and ``_`` only)."""
    text = "-".join(str(p) for p in parts if p is not None and str(p) != "")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "x"


def tag(name: str, body="", **attrs) -> Raw:
    rendered = "".join(f' {key.rstrip("_").replace("_", "-")}="{esc(value)}"' for key, value in attrs.items()
                       if value is not None and value is not False)
    return Raw(f"<{name}{rendered}>{cell(body)}</{name}>")


def para(text, cls: str | None = None) -> Raw:
    return tag("p", text, class_=cls)


def note(text) -> Raw:
    return tag("p", text, class_="note")


def code(text) -> Raw:
    return tag("code", text)


def link(href: str, text) -> Raw:
    return Raw(f'<a href="{esc(href)}">{cell(text)}</a>')


def details(summary, body, open_: bool = False) -> Raw:
    return Raw(f"<details{' open' if open_ else ''}><summary>{cell(summary)}</summary>{cell(body)}</details>")


def ul(items, cls: str | None = None) -> Raw:
    rows = "".join(f"<li>{cell(item)}</li>" for item in items)
    attr = f' class="{esc(cls)}"' if cls else ""
    return Raw(f"<ul{attr}>{rows}</ul>")


SEVERITY_LABEL = {"info": "INFO", "warning": "WARNING", "error": "ERROR", "fatal": "FATAL"}


def badge(text, kind: str) -> Raw:
    """A coloured label; the colour never carries meaning alone (the text says it)."""
    return Raw(f'<span class="badge b-{esc(kind)}">{esc(text)}</span>')


def severity_badge(severity: str) -> Raw:
    return badge(SEVERITY_LABEL.get(severity, str(severity).upper()), str(severity))


def pass_badge(passed) -> Raw:
    if passed is None:
        return badge("not judged", "info")
    return badge("PASS", "pass") if passed else badge("FAIL", "fail")


NUM = ' class="num"'
WIDE_COLUMNS = 5      # tables with more columns keep a minimum width and scroll sideways on a phone


def table(headers, rows, *, numeric=(), cls: str = "", caption=None, empty: str = "nothing to show",
          row_classes=None, row_ids=None) -> Raw:
    """A table in a horizontally scrolling box. ``numeric``: column indexes right-aligned."""
    if not rows:
        return note(empty)
    numeric = set(numeric)
    head = "".join(f'<th{NUM if i in numeric else ""} scope="col">{cell(h)}</th>' for i, h in enumerate(headers))
    body = []
    for r, row in enumerate(rows):
        attrs = ""
        if row_classes and row_classes[r]:
            attrs += f' class="{esc(row_classes[r])}"'
        if row_ids and row_ids[r]:
            attrs += f' id="{esc(row_ids[r])}"'
        cells = "".join(f'<td{NUM if i in numeric else ""}>{cell(v)}</td>' for i, v in enumerate(row))
        body.append(f"<tr{attrs}>{cells}</tr>")
    cap = f"<caption>{cell(caption)}</caption>" if caption is not None else ""
    classes = " ".join(c for c in (cls, "wide" if len(headers) > WIDE_COLUMNS else "") if c)
    return Raw(f'<div class="scroll"><table class="{esc(classes)}">{cap}<thead><tr>{head}</tr></thead>'
               f'<tbody>{"".join(body)}</tbody></table></div>')


def kv(pairs, cls: str = "kv") -> Raw:
    """Two-column key/value table (``pairs``: (label, value) with value text or Raw)."""
    rows = "".join(f'<tr><th scope="row">{cell(k)}</th><td>{cell(v)}</td></tr>' for k, v in pairs)
    return Raw(f'<div class="scroll"><table class="{esc(cls)}"><tbody>{rows}</tbody></table></div>')


def tiles(items) -> Raw:
    """Headline numbers: (label, value, sub) each."""
    out = []
    for label, value, sub in items:
        out.append(f'<div class="tile"><div class="tile-label">{cell(label)}</div>'
                   f'<div class="tile-value">{cell(value)}</div>'
                   f'<div class="tile-sub">{cell(sub)}</div></div>')
    return Raw(f'<div class="tiles">{"".join(out)}</div>')


def png_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def image_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def figure(png: bytes | None, alt: str, caption=None, mime: str = "image/png", cls: str = "") -> Raw:
    if png is None:
        return Raw("")
    cap = f"<figcaption>{cell(caption)}</figcaption>" if caption is not None else ""
    return Raw(f'<figure class="{esc(cls)}"><img src="{image_uri(png, mime)}" alt="{esc(alt)}">{cap}</figure>')


@dataclass
class Section:
    id: str
    title: str
    body: str
    level: int = 2
    children: list["Section"] = field(default_factory=list)
    lead: str = ""                  # one line under the title (what the section says)

    def render(self) -> str:
        heading = f'<h{self.level} id="{esc(self.id)}">{esc(self.title)}</h{self.level}>'
        lead = f'<p class="lead">{cell(self.lead)}</p>' if self.lead else ""
        inner = "".join(child.render() for child in self.children)
        return f'<section class="s{self.level}">{heading}{lead}{self.body}{inner}</section>'


def toc(sections: list[Section]) -> Raw:
    items = []
    for section in sections:
        sub = "".join(f'<li><a href="#{esc(c.id)}">{esc(c.title)}</a></li>' for c in section.children)
        items.append(f'<li><a href="#{esc(section.id)}">{esc(section.title)}</a>'
                     f'{f"<ol>{sub}</ol>" if sub else ""}</li>')
    return Raw(f'<nav class="toc"><details open><summary>Contents</summary><ol>{"".join(items)}</ol></details></nav>')


def document(title: str, sections: list[Section], *, header: str = "", running_title: str = "",
             with_toc: bool = True, extra_css: str = "") -> str:
    """The whole page as text (UTF-8, deterministic)."""
    running = esc(running_title or title)
    body = "".join(section.render() for section in sections)
    contents = toc(sections) if with_toc else ""
    return ("<!DOCTYPE html>\n"
            '<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<meta name="generator" content="cnctwin.report"><title>{esc(title)}</title>'
            f"<style>{CSS}{extra_css}</style></head>"
            f'<body><div class="running">{running}</div><main>{header}{contents}{body}</main></body></html>\n')


CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--surface:#fcfcfb;
--page:#ffffff;--good:#0ca30c;--good-ink:#006300;--warn:#fab219;--serious:#ec835a;--critical:#d03b3b;
--accent:#2a78d6}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,"DejaVu Sans",sans-serif}
main{max-width:980px;margin:0 auto;padding:12px 14px 40px}
.running{display:none}
h1{font-size:24px;line-height:1.2;margin:8px 0 4px}
h2{font-size:19px;margin:28px 0 8px;padding-top:6px;border-top:1px solid var(--grid)}
h3{font-size:16px;margin:18px 0 6px}
h4{font-size:14px;margin:14px 0 4px;color:var(--ink2)}
p{margin:6px 0}
.lead{color:var(--ink2);margin-top:0}
.note{color:var(--ink2);font-size:13px}
a{color:#1c5cab}
code,pre,.mono{font-family:"DejaVu Sans Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
code{white-space:pre-wrap;word-break:break-word}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:6px 0 10px;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:13px}
.scroll>table.wide{min-width:720px}
caption{text-align:left;color:var(--ink2);font-size:12.5px;padding:2px 0 4px;caption-side:top}
th,td{padding:4px 6px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
thead th{border-bottom:1px solid var(--axis);color:var(--ink2);font-weight:600;white-space:nowrap}
td.num,th.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
table.kv th{width:38%;color:var(--ink2);font-weight:600}
tr.hit-error td,tr.hit-fatal td{background:#fbeaea}
tr.hit-warning td{background:#fff6e0}
tr.fail td{background:#fbeaea}
tr.changed td{background:#fff6e0}
figure{margin:10px 0 14px}
figure img{display:block;max-width:100%;height:auto;border:1px solid var(--grid);background:var(--surface)}
figcaption{color:var(--ink2);font-size:12.5px;margin-top:3px}
.badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11.5px;font-weight:700;
letter-spacing:.02em;white-space:nowrap;border:1px solid transparent}
.b-info{background:#f0efec;color:var(--ink);border-color:var(--axis)}
.b-warning{background:var(--warn);color:var(--ink)}
.b-error{background:var(--serious);color:var(--ink)}
.b-fatal{background:var(--critical);color:#fff}
.b-pass{background:#e2f4e2;color:var(--good-ink);border-color:var(--good)}
.b-fail{background:var(--critical);color:#fff}
.b-better{background:#e2f4e2;color:var(--good-ink);border-color:var(--good)}
.b-worse{background:#fbeaea;color:#8f1d1d;border-color:var(--critical)}
.b-same{background:#f0efec;color:var(--ink2);border-color:var(--axis)}
.verdict{border:1px solid var(--grid);border-left:8px solid var(--critical);border-radius:6px;padding:10px 12px;
margin:10px 0;background:#fdf3f3}
.verdict.safe{border-left-color:var(--good);background:#f1f9f1}
.verdict-title{font-size:22px;font-weight:800;margin:0 0 4px}
.verdict-icon{font-weight:800;margin-right:6px}
.verdict.safe .verdict-icon{color:var(--good-ink)}
.verdict.unsafe .verdict-icon{color:var(--critical)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;margin:10px 0}
.tile{border:1px solid var(--grid);border-radius:6px;padding:8px;background:var(--surface)}
.tile-label{font-size:12px;color:var(--ink2)}
.tile-value{font-size:19px;font-weight:700}
.tile-sub{font-size:11.5px;color:var(--muted)}
.toc{margin:10px 0 0;font-size:14px}
.toc ol{margin:4px 0 4px 18px;padding:0}
.toc ol ol{font-size:13px}
details>summary{cursor:pointer;color:#1c5cab}
.chain{margin:4px 0 4px 6px;padding-left:10px;border-left:2px solid var(--grid);font-size:12.5px}
.listing{font-family:"DejaVu Sans Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;
border:1px solid var(--grid);background:var(--surface);padding:4px 0;margin:6px 0 12px;overflow-x:auto}
.listing .ln{display:block;white-space:pre;padding:0 8px}
.listing .ln .no{display:inline-block;min-width:4.5em;color:var(--muted);text-align:right;margin-right:10px}
.listing .ln.hit-warning{background:#fff6e0}
.listing .ln.hit-error,.listing .ln.hit-fatal{background:#fbeaea}
.listing .ln.hit-info{background:#eef4fc}
.listing .ln:target{outline:2px solid var(--accent)}
.listing .gap{display:block;color:var(--muted);padding:0 8px}
.muted{color:var(--muted)}
.small{font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:640px){.grid2{grid-template-columns:1fr}}
@media (max-width:480px){body{font-size:14px}main{padding:8px 8px 30px}h1{font-size:20px}h2{font-size:17px}
th,td{padding:3px 4px}table{font-size:12px}.verdict-title{font-size:19px}.tile-value{font-size:17px}}
@page{size:A4;margin:15mm 12mm 16mm 12mm;
@top-left{content:string(running);font-size:8pt;color:#52514e}
@bottom-right{content:"page " counter(page) " of " counter(pages);font-size:8pt;color:#52514e}}
@media print{body{font-size:9.5pt}main{max-width:none;padding:0}
.running{display:block;string-set:running content();height:0;overflow:hidden;font-size:0}
.scroll{overflow:visible}table{font-size:7.8pt}.scroll>table.wide{min-width:0;font-size:6.8pt}
th,td{padding:2px 3px}thead th{white-space:normal}
td,th{word-break:break-word}td.num,th.num{white-space:nowrap}
h2{break-after:avoid;font-size:13pt}h3,h4{break-after:avoid}
figure,tr,.tile,.verdict{break-inside:avoid}.listing{font-size:7.5pt;overflow:visible}
.listing .ln{white-space:pre-wrap}.toc details>summary{color:var(--ink)}a{color:var(--ink);text-decoration:none}
figure img{max-height:120mm;width:auto;margin:0 auto}}
"""


__all__ = ["Raw", "esc", "cell", "join", "anchor", "tag", "para", "note", "code", "link", "details", "ul", "badge",
           "severity_badge", "pass_badge", "table", "kv", "tiles", "png_uri", "image_uri", "figure", "Section",
           "toc", "document", "CSS"]
