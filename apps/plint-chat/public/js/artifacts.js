// Artifacts: the panel beside the chat (full screen on phones), cards in the
// thread and tiles in galleries. HTML/SVG/Mermaid run in a sandboxed iframe with
// no access to this origin; Markdown is rendered with the escaped renderer.
import { h, icon, toast, copyText, ago, narrow } from "./ui.js";
import { api, text } from "./api.js";
import { markdown, highlight } from "./md.js";

const KIND_ICON = { html: "globe", svg: "image", markdown: "fileText", mermaid: "chart", code: "code" };
const KIND_LABEL = { html: "Web page", svg: "Image", markdown: "Document", mermaid: "Diagram", code: "Code" };
const SANDBOX = "allow-scripts allow-forms allow-popups allow-modals allow-downloads allow-pointer-lock";
let current = null; // { id, chatId }

export const panelChat = () => current?.chatId || null;
export function closePanel() {
  const p = document.getElementById("panel");
  if (!p) return;
  p.hidden = true;
  p.replaceChildren();
  current = null;
}

function srcdoc(kind, body) {
  if (kind === "svg") return `<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body{margin:0;height:100%;display:grid;place-items:center;background:#fff}svg{max-width:100%;max-height:100vh;height:auto}</style>${body}`;
  if (kind === "mermaid") {
    const safe = body.replace(/</g, "&lt;");
    return `<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0;padding:24px;font-family:system-ui;background:#fff}</style><pre class="mermaid">${safe}</pre><script type="module">import m from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";m.initialize({startOnLoad:true,theme:matchMedia("(prefers-color-scheme: dark)").matches?"dark":"default"});</script>`;
  }
  return body;
}

export async function openArtifact(id, version = null, { quiet = false } = {}) {
  const p = document.getElementById("panel");
  let a;
  try { a = await api(`artifacts/${id}`); } catch (e) { return toast(e.message); }
  const v = version || a.versions.length;
  let body;
  try { body = await text(`artifacts/${id}/raw?v=${v}`); } catch (e) { return toast(e.message); }
  if (quiet && current && current.id !== id && !p.hidden) return; // don't steal the panel from another artifact
  current = { id, chatId: a.chatId };
  let tab = a.kind === "code" ? "code" : "preview";
  const view = h("div.panel-body");
  const seg = h("div.seg",
    h("button", { type: "button", on: { click: () => { tab = "preview"; draw(); } } }, "Preview"),
    h("button", { type: "button", on: { click: () => { tab = "code"; draw(); } } }, "Code"));
  const versions = a.versions.length > 1 ? h("select.version-sel", { "aria-label": "Version", on: { change: (e) => openArtifact(id, Number(e.target.value)) } },
    a.versions.map((x) => h("option", { value: x.n, selected: x.n === v }, `v${x.n}${x.n === a.versions.length ? " · latest" : ""}`)).reverse()) : null;
  function draw() {
    seg.children[0].classList.toggle("on", tab === "preview");
    seg.children[1].classList.toggle("on", tab === "code");
    if (tab === "code") {
      const lang = a.kind === "markdown" ? "md" : a.kind === "code" ? a.path.split(".").pop() : a.kind;
      view.replaceChildren(h("div.code-view", h("pre", h("code", { html: highlight(body, lang) }))));
    } else if (a.kind === "markdown") {
      view.replaceChildren(h("div.prose", { html: markdown(body) }));
    } else {
      view.replaceChildren(h("iframe", { sandbox: SANDBOX, title: a.title, srcdoc: srcdoc(a.kind, body), allow: "fullscreen; clipboard-write" }));
    }
  }
  p.replaceChildren(
    h("div.panel-head",
      h("span.t", { title: a.title }, a.title),
      versions,
      a.kind !== "code" ? seg : null,
      h("button.icon-btn", { title: "Copy", "aria-label": "Copy", on: { click: () => copyText(body) } }, icon("copy")),
      h("a.icon-btn", { title: "Download", "aria-label": "Download", href: `/api/artifacts/${id}/raw?v=${v}&download=1` }, icon("download")),
      h("button.icon-btn", { title: "Close", "aria-label": "Close", on: { click: closePanel } }, icon("x"))),
    view);
  p.hidden = false;
  draw();
}

export function artifactCard(ev) {
  return h("button.art-card", { type: "button", on: { click: () => openArtifact(ev.id, ev.version) } },
    h("span.ai", icon(KIND_ICON[ev.kind] || "file")),
    h("span.meta", h("div.t", ev.title), h("div.s", `${KIND_LABEL[ev.kind] || "File"}${ev.version > 1 ? ` · version ${ev.version}` : ""} · ${narrow() ? "Tap" : "Click"} to open`)));
}

export function artifactTile(a, { onMenu } = {}) {
  const thumb = h("div.thumb", icon(KIND_ICON[a.kind] || "file"));
  if (a.kind === "html" || a.kind === "svg") {
    // Static thumbnail: no scripts, rendered small.
    text(`artifacts/${a.id}/raw`).then((body) => {
      thumb.append(h("iframe", { sandbox: "", loading: "lazy", tabindex: "-1", "aria-hidden": "true", srcdoc: srcdoc(a.kind, body) }));
    }).catch(() => {});
  }
  const tile = h("a.art-tile", { href: `#/chat/${a.chatId}`, on: { click: (e) => { e.preventDefault(); openArtifact(a.id); } } },
    thumb, h("div.meta", h("div.t", a.title), h("div.s", `${KIND_LABEL[a.kind] || "File"} · ${ago(a.updated)}`)));
  if (onMenu) tile.append(h("button.icon-btn.sm", { style: { position: "absolute", top: "6px", right: "6px", background: "var(--popover)" }, "aria-label": "More", on: { click: (e) => { e.preventDefault(); e.stopPropagation(); onMenu(e.currentTarget, a); } } }, icon("more")));
  return tile;
}
