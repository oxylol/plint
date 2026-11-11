// Artifacts: the panel beside the chat (full screen on phones), cards in the
// thread and tiles in galleries. HTML/SVG/Mermaid run in a sandboxed iframe with
// no access to this origin; Markdown is rendered with the escaped renderer.
import { h, icon, menu, toast, copyText, ago, narrow } from "./ui.js";
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

// Drag the panel's left edge to resize it; the width is remembered.
function grip(p) {
  const g = h("div.panel-grip", { role: "separator", "aria-orientation": "vertical", "aria-label": "Resize side panel" });
  g.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    g.setPointerCapture(e.pointerId);
    document.body.classList.add("resizing");
    const right = p.getBoundingClientRect().right;
    const move = (ev) => { const w = Math.max(320, Math.min(right - ev.clientX, p.parentNode.clientWidth - 420)); p.style.flex = `0 0 ${w}px`; };
    g.addEventListener("pointermove", move);
    g.addEventListener("pointerup", () => { g.removeEventListener("pointermove", move); document.body.classList.remove("resizing"); }, { once: true });
  });
  return g;
}
export function showPanel(nodes, { slim = false } = {}) {
  const p = document.getElementById("panel");
  if (p.hidden) p.style.flex = "";
  p.classList.toggle("slim", slim);
  p.replaceChildren(grip(p), ...nodes);
  p.hidden = false;
  return p;
}
export const numbered = (html) => html.split("\n").map((l) => `<span class="ln">${l || " "}</span>`).join("");

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
  const seg = h("div.seg.icons", { role: "radiogroup", "aria-label": "File view mode" },
    h("button", { type: "button", role: "radio", title: "Preview", "aria-label": "Preview", on: { click: () => { tab = "preview"; draw(); } } }, icon("eye")),
    h("button", { type: "button", role: "radio", title: "Code", "aria-label": "Code", on: { click: () => { tab = "code"; draw(); } } }, icon("code")));
  const ext = (a.path?.split(".").pop() || a.kind).toUpperCase();
  function draw() {
    seg.children[0].classList.toggle("on", tab === "preview");
    seg.children[1].classList.toggle("on", tab === "code");
    if (tab === "code") {
      const lang = a.kind === "markdown" ? "md" : a.kind === "code" ? a.path.split(".").pop() : a.kind;
      view.replaceChildren(h("div.code-view", h("pre.numbered", h("code", { html: numbered(highlight(body, lang)) }))));
    } else if (a.kind === "markdown") {
      view.replaceChildren(h("div.doc", h("div.prose", { html: markdown(body) })));
    } else {
      view.replaceChildren(h("iframe", { sandbox: SANDBOX, title: a.title, srcdoc: srcdoc(a.kind, body), allow: "fullscreen; clipboard-write" }));
    }
  }
  const copy = h("button.btn.secondary", { type: "button", on: { click: async () => { await copyText(body, { quiet: true }); copy.textContent = "Copied"; setTimeout(() => (copy.textContent = "Copy"), 1600); } } }, "Copy");
  const more = h("button.icon-btn", { type: "button", "aria-label": "More options", "aria-haspopup": "menu", on: { click: (e) => menu(e.currentTarget, [
    { label: "Download", onClick: () => { location.href = `/api/artifacts/${id}/raw?v=${v}&download=1`; } },
    a.chatId ? { label: "Go to chat", onClick: () => { location.hash = `#/chat/${a.chatId}`; } } : null,
    a.versions.length > 1 ? "sep" : null,
    a.versions.length > 1 ? { label: "Version", right: `${v} of ${a.versions.length}`, children: a.versions.map((x) => ({ label: `Version ${x.n}${x.n === a.versions.length ? " (latest)" : ""}`, checked: x.n === v, onClick: () => openArtifact(id, x.n) })).reverse() } : null,
  ], { align: "end" }) } }, icon("downSmall", "s16"));
  showPanel([
    h("div.panel-head", { role: "toolbar", "aria-label": `${a.title} controls` },
      a.kind !== "code" ? seg : null,
      h("h2.t", { style: { margin: 0 } }, a.title, h("span.dotsep", " · "), h("span", ext)),
      copy, more,
      h("button.icon-btn", { type: "button", title: "Close", "aria-label": "Close", on: { click: closePanel } }, icon("x"))),
    view]);
  draw();
}

export function artifactCard(ev) {
  return h("button.art-card", { type: "button", on: { click: () => openArtifact(ev.id, ev.version) } },
    h("span.ai", icon(KIND_ICON[ev.kind] || "file")),
    h("span.meta", h("div.t", ev.title), h("div.s", `${KIND_LABEL[ev.kind] || "File"}${ev.version > 1 ? ` · Version ${ev.version}` : ""}`)));
}

const shortDate = (ts) => new Date(ts).toLocaleDateString("en-US", { month: "short", day: "numeric" });
export function artifactTile(a, { onMenu } = {}) {
  const live = a.kind === "html" || a.kind === "svg";
  const sheet = h("span.sheet", live ? null : h("span.lines", h("i"), h("i", { style: { width: "68%" } }), h("i", { style: { width: "59%" } }), h("i", { style: { width: "62%" } }), h("i", { style: { width: "44%" } })));
  if (live) {
    // Static thumbnail: no scripts, rendered small.
    text(`artifacts/${a.id}/raw`).then((body) => {
      sheet.append(h("iframe", { sandbox: "", loading: "lazy", tabindex: "-1", "aria-hidden": "true", srcdoc: srcdoc(a.kind, body) }));
    }).catch(() => {});
  }
  const tile = h(`a.art-tile${live ? ".live" : ""}`, { href: `#/chat/${a.chatId}`, title: a.title, on: { click: (e) => { e.preventDefault(); openArtifact(a.id); } } },
    h("div.thumb", sheet), h("div.meta", h("span.t", a.title), h("time.s", shortDate(a.updated))));
  if (onMenu) tile.append(h("button.icon-btn.sm.more", { type: "button", "aria-label": `More options for ${a.title}`, on: { click: (e) => { e.preventDefault(); e.stopPropagation(); onMenu(e.currentTarget, a); } } }, icon("more", "s16")));
  return tile;
}
