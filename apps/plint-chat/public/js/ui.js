// Small UI kit: element helper, icons, menus, modals, toasts, formatting.
// Let append/replaceChildren take optional children (null/false/undefined are skipped,
// arrays are spread) instead of printing "null".
for (const m of ["append", "replaceChildren"]) {
  const orig = Element.prototype[m];
  Element.prototype[m] = function (...kids) { return orig.apply(this, kids.flat(Infinity).filter((k) => k !== null && k !== undefined && k !== false)); };
}

export const $ = (s, root = document) => root.querySelector(s);
export const $$ = (s, root = document) => [...root.querySelectorAll(s)];
export const touch = matchMedia("(pointer: coarse)").matches;
export const narrow = () => matchMedia("(max-width: 860px)").matches;
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// h("div.cls#id", {attrs, on:{click}}, ...children)
export function h(tag, attrs, ...kids) {
  const [, name = "div", rest = ""] = tag.match(/^([a-z0-9-]*)(.*)$/i);
  const n = document.createElement(name || "div");
  for (const part of rest.match(/[.#][^.#]+/g) || []) part[0] === "." ? n.classList.add(part.slice(1)) : (n.id = part.slice(1));
  if (attrs && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) { kids.unshift(attrs); attrs = null; }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "on") for (const [ev, fn] of Object.entries(v)) n.addEventListener(ev, fn);
    else if (k === "html") n.innerHTML = v;
    else if (k === "class") n.className += ` ${v}`;
    else if (k === "style" && typeof v === "object") Object.assign(n.style, v);
    else if (k in n && typeof v !== "string") n[k] = v;
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat(Infinity)) if (k !== null && k !== undefined && k !== false) n.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return n;
}

// Line icons, drawn for this app on a 24px grid.
const P = {
  plus: "M12 5v14M5 12h14",
  search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-3.5-3.5",
  folder: "M3.5 7.5a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z",
  artifacts: "M4 5.5A1.5 1.5 0 0 1 5.5 4h4A1.5 1.5 0 0 1 11 5.5v4A1.5 1.5 0 0 1 9.5 11h-4A1.5 1.5 0 0 1 4 9.5zM13 5.5A1.5 1.5 0 0 1 14.5 4h4A1.5 1.5 0 0 1 20 5.5v4a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 13 9.5zM4 14.5A1.5 1.5 0 0 1 5.5 13h4a1.5 1.5 0 0 1 1.5 1.5v4A1.5 1.5 0 0 1 9.5 20h-4A1.5 1.5 0 0 1 4 18.5zM13 14.5a1.5 1.5 0 0 1 1.5-1.5h4a1.5 1.5 0 0 1 1.5 1.5v4a1.5 1.5 0 0 1-1.5 1.5h-4a1.5 1.5 0 0 1-1.5-1.5z",
  clock: "M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM12 7.5V12l3 2",
  sliders: "M4 7h9M17 7h3M4 17h3M11 17h9M15 5v4M9 15v4",
  sidebar: "M4 5.5A1.5 1.5 0 0 1 5.5 4h13A1.5 1.5 0 0 1 20 5.5v13a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18.5zM9.5 4v16",
  menu: "M4 7h16M4 12h16M4 17h11",
  edit: "M12 20h8M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z",
  more: "M6 12h.01M12 12h.01M18 12h.01",
  chev: "M9 6l6 6-6 6",
  down: "M7 10l5 5 5-5",
  up: "M12 19V5M5.5 11.5L12 5l6.5 6.5",
  arrowDown: "M12 5v14M6 13l6 6 6-6",
  stop: "M7 7h10v10H7z",
  x: "M6 6l12 12M18 6L6 18",
  check: "M5 12.5l4.5 4.5L19 7.5",
  copy: "M9 9.5A1.5 1.5 0 0 1 10.5 8h8A1.5 1.5 0 0 1 20 9.5v9a1.5 1.5 0 0 1-1.5 1.5h-8A1.5 1.5 0 0 1 9 18.5zM15 8V5.5A1.5 1.5 0 0 0 13.5 4h-8A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H9",
  retry: "M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5",
  pin: "M9 4h6l-1 6 4 3v2H6v-2l4-3zM12 15v5",
  trash: "M5 7h14M10 11v6M14 11v6M6 7l1 12a1.5 1.5 0 0 0 1.5 1.4h7a1.5 1.5 0 0 0 1.5-1.4L18 7M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2",
  paperclip: "M20 11.5l-7.8 7.8a5 5 0 0 1-7.1-7.1l8.5-8.5a3.3 3.3 0 0 1 4.7 4.7L9.8 16.9a1.7 1.7 0 0 1-2.4-2.4l7.5-7.5",
  image: "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5zM9 10.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM20 15l-4.5-4.5L7 19",
  mic: "M12 3.5a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0v-5a3 3 0 0 0-3-3zM6 11a6 6 0 0 0 12 0M12 17v3.5",
  ghost: "M6 20V11a6 6 0 0 1 12 0v9l-2-1.5-2 1.5-2-1.5-2 1.5-2-1.5zM10 11h.01M14 11h.01",
  hand: "M8 13V6.5a1.5 1.5 0 0 1 3 0V12M11 11.5V5a1.5 1.5 0 0 1 3 0v6.5M14 11.5V6.5a1.5 1.5 0 0 1 3 0v7a6.5 6.5 0 0 1-6.5 6.5H10a5 5 0 0 1-4-2l-2.4-3.4a1.4 1.4 0 0 1 2.2-1.7L8 15",
  bolt: "M13 3L5 13.5h6L10 21l8-10.5h-6z",
  pen: "M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4",
  terminal: "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5zM8 9.5l3 2.5-3 2.5M13 15h3",
  file: "M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8zM14 3.5V8h4.5",
  fileText: "M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8zM14 3.5V8h4.5M9 13h6M9 16.5h4",
  globe: "M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM3.5 12h17M12 3.5c2.3 2.4 3.5 5.2 3.5 8.5s-1.2 6.1-3.5 8.5c-2.3-2.4-3.5-5.2-3.5-8.5S9.7 5.9 12 3.5z",
  code: "M8.5 7L3.5 12l5 5M15.5 7l5 5-5 5M13.5 4.5l-3 15",
  chart: "M4 20V10M10 20V4M16 20v-7M21 20H3",
  eye: "M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12zM12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z",
  download: "M12 4v11M7 10.5l5 5 5-5M5 19.5h14",
  external: "M14 4h6v6M20 4l-9 9M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10",
  gear: "M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM19.4 13.5a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V19a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-2.7-1.1l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0-1.1-2.7H3.5a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.1-2.7l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 2.7-1.1V3.5a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 2.7 1.1l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0 1.1 2.7h.1a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z",
  logout: "M15 4h3.5A1.5 1.5 0 0 1 20 5.5v13a1.5 1.5 0 0 1-1.5 1.5H15M10 16.5L5.5 12 10 7.5M5.5 12H15",
  sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4L6 18M18 6l1.4-1.4",
  moon: "M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z",
  type: "M5 7V5h14v2M12 5v14M9 19h6",
  brain: "M9 4.5a2.5 2.5 0 0 0-2.5 2.5v.3A3 3 0 0 0 4.5 10a3 3 0 0 0 1 2.2A3 3 0 0 0 7 17.5a2.5 2.5 0 0 0 5 .5V7a2.5 2.5 0 0 0-3-2.5zM15 4.5A2.5 2.5 0 0 1 17.5 7v.3a3 3 0 0 1 2 2.7 3 3 0 0 1-1 2.2 3 3 0 0 1-1.5 5.3 2.5 2.5 0 0 1-5 .5",
  play: "M7 5l12 7-12 7z",
  calendar: "M4.5 7A2.5 2.5 0 0 1 7 4.5h10A2.5 2.5 0 0 1 19.5 7v10a2.5 2.5 0 0 1-2.5 2.5H7A2.5 2.5 0 0 1 4.5 17zM4.5 9.5h15M8.5 3v3M15.5 3v3",
  heart: "M12 20s-7.5-4.5-7.5-10A4.5 4.5 0 0 1 12 7a4.5 4.5 0 0 1 7.5 3c0 5.5-7.5 10-7.5 10z",
  game: "M7 8.5h10a4 4 0 0 1 3.9 4.8l-.8 3.8a2 2 0 0 1-3.5.9L15 16H9l-1.6 2a2 2 0 0 1-3.5-.9l-.8-3.8A4 4 0 0 1 7 8.5zM8 11v3M6.5 12.5h3M15.5 12h.01M17.5 13.5h.01",
  shield: "M12 3.5l7 3v5c0 4.5-3 8-7 9.5-4-1.5-7-5-7-9.5v-5z",
  box: "M12 3.5l8 4.5v8l-8 4.5-8-4.5V8zM4 8l8 4.5L20 8M12 12.5v8",
  bell: "M6 16.5V11a6 6 0 0 1 12 0v5.5l1.5 2h-15zM10 20.5a2 2 0 0 0 4 0",
  user: "M12 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM4.5 20a7.5 7.5 0 0 1 15 0",
  upload: "M12 16V4M7 8.5l5-4.5 5 4.5M5 15v3.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V15",
  filter: "M4 6h16M7 12h10M10 18h4",
  branch: "M6 4v10M6 14a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM18 4a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM18 10c0 4-4 5.5-12 6",
  sparkle: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z",
};
export function icon(name, cls = "") {
  const ns = "http://www.w3.org/2000/svg";
  const s = document.createElementNS(ns, "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("class", `i ${cls}`.trim());
  s.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", P[name] || P.file);
  s.append(p);
  return s;
}

// The Plint spark: eight rounded rays of varying length.
export function spark() {
  const n = document.createElement("span");
  n.className = "spark";
  const rays = [[0, 10], [45, 8.3], [90, 9.6], [135, 7.6], [180, 10], [225, 8.6], [270, 9.2], [315, 7.8]];
  const lines = rays.map(([a, len]) => {
    const r = (a * Math.PI) / 180;
    const x1 = 12 + Math.cos(r) * 2.2, y1 = 12 + Math.sin(r) * 2.2, x2 = 12 + Math.cos(r) * (2.2 + len), y2 = 12 + Math.sin(r) * (2.2 + len);
    return `<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}"/>`;
  }).join("");
  n.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" aria-hidden="true">${lines}</svg>`;
  return n;
}

// --- menus ------------------------------------------------------------------
// items: [{label, sub, icon, onClick, checked, danger, right, switch:bool, on:bool} | "sep" | {heading}]
let openMenu = null;
export function closeMenu() { openMenu?.remove(); openMenu = null; }
export function menu(anchor, items, { align = "start", side = "auto" } = {}) {
  if (openMenu?._anchor === anchor) { closeMenu(); return; }
  closeMenu();
  const m = h("div.menu", { role: "menu" });
  for (const it of items) {
    if (!it) continue;
    if (it === "sep") { m.append(h("div.sep")); continue; }
    if (it.heading) { m.append(h("div.label", it.heading)); continue; }
    const b = h(`button.mi${it.danger ? ".danger" : ""}${it.switch ? ".switch" : ""}`, { role: "menuitem", type: "button", disabled: it.disabled },
      it.icon ? icon(it.icon) : null,
      h("span.grow", it.label, it.sub ? h("span.sub", it.sub) : null),
      it.right ? h("span.right", it.right) : null,
      it.switch ? h(`span.sw${it.on ? ".on" : ""}`) : null,
      it.checked ? h("span.check", icon("check")) : null);
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      if (it.switch) { it.on = !it.on; b.querySelector(".sw").classList.toggle("on", it.on); it.onClick?.(it.on); return; }
      closeMenu();
      it.onClick?.();
    });
    m.append(b);
  }
  m._anchor = anchor;
  document.body.append(m);
  const r = anchor.getBoundingClientRect();
  const mw = m.offsetWidth, mh = m.offsetHeight, vw = innerWidth, vh = innerHeight;
  const below = side === "below" || (side === "auto" && r.bottom + mh + 8 < vh) || r.top < mh + 8;
  let left = align === "end" ? r.right - mw : r.left;
  left = Math.max(8, Math.min(left, vw - mw - 8));
  const top = below ? r.bottom + 6 : r.top - mh - 6;
  m.style.left = `${left}px`;
  m.style.top = `${Math.max(8, top)}px`;
  m.style.setProperty("--origin", `${align === "end" ? "right" : "left"} ${below ? "top" : "bottom"}`);
  openMenu = m;
  return m;
}
document.addEventListener("click", (e) => { if (openMenu && !openMenu.contains(e.target) && !openMenu._anchor.contains(e.target)) closeMenu(); });
addEventListener("resize", closeMenu);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeMenu(); closeModal(); } });

// --- modals -----------------------------------------------------------------
let openModal = null;
export function closeModal() { openModal?.remove(); openModal = null; }
export function modal(content, { cls = "" } = {}) {
  closeModal();
  const box = h(`div.modal${cls ? `.${cls}` : ""}`, { role: "dialog", "aria-modal": "true" }, content);
  const back = h("div.modal-back", box);
  back.addEventListener("mousedown", (e) => { if (e.target === back) closeModal(); });
  document.body.append(back);
  openModal = back;
  setTimeout(() => box.querySelector("[autofocus], input, textarea")?.focus(), 30);
  return box;
}
// A small form in a modal. fields: [{name, label, type, value, placeholder, hint, options:[[v,l]]}]
export function formModal({ title, fields, submit = "Save", danger, onSubmit, extra }) {
  return new Promise((resolve) => {
    const form = h("form", h("h2", title));
    for (const f of fields) {
      let input;
      if (f.type === "textarea") input = h("textarea", { name: f.name, placeholder: f.placeholder || "", rows: f.rows || 5 }, f.value || "");
      else if (f.type === "select") input = h("select", { name: f.name }, f.options.map(([v, l]) => h("option", { value: v, selected: String(v) === String(f.value) }, l)));
      else input = h("input", { name: f.name, type: f.type || "text", value: f.value ?? "", placeholder: f.placeholder || "", autocomplete: "off" });
      if (f.autofocus) input.setAttribute("autofocus", "");
      const field = h("label.field", { "data-field": f.name }, h("span", f.label), input, f.hint ? h("span.hint", f.hint) : null);
      if (f.show) field.hidden = !f.show(f.value);
      form.append(field);
    }
    if (extra) form.append(extra);
    const cancel = h("button.btn.ghost", { type: "button", on: { click: () => { closeModal(); resolve(null); } } }, "Cancel");
    form.append(h("div.actions", cancel, h(`button.btn.${danger ? "danger" : "primary"}`, { type: "submit" }, submit)));
    const sync = () => {
      const data = Object.fromEntries(new FormData(form));
      for (const f of fields) if (f.show) form.querySelector(`[data-field="${f.name}"]`).hidden = !f.show(data[f.name], data);
    };
    form.addEventListener("change", sync);
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const data = Object.fromEntries(new FormData(form));
      try { if (onSubmit) await onSubmit(data); closeModal(); resolve(data); }
      catch (err) { toast(err.message); }
    });
    modal(form);
    sync();
  });
}
export function confirmModal(title, text, ok = "Delete") {
  return new Promise((resolve) => {
    modal([
      h("h2", title), h("p.muted", text),
      h("div.actions",
        h("button.btn.ghost", { on: { click: () => { closeModal(); resolve(false); } } }, "Cancel"),
        h("button.btn.danger", { autofocus: true, on: { click: () => { closeModal(); resolve(true); } } }, ok)),
    ]);
  });
}

// --- toasts -----------------------------------------------------------------
export function toast(text, ms = 2600) {
  const t = h("div.toast", text);
  document.getElementById("toasts").append(t);
  setTimeout(() => t.remove(), ms);
}

// --- formatting -------------------------------------------------------------
export function ago(ts) {
  const s = (Date.now() - ts) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 7 * 86400) return `${Math.floor(s / 86400)}d ago`;
  return new Date(ts).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
export const when = (ts) => new Date(ts).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
export function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch {
    const ta = h("textarea", { style: { position: "fixed", opacity: "0" } }, text);
    document.body.append(ta); ta.select(); document.execCommand("copy"); ta.remove(); toast("Copied");
  }
}
