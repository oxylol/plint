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
export const narrow = () => matchMedia("(max-width: 700px)").matches; // the sidebar is an overlay
export const phone = () => matchMedia("(max-width: 639.98px)").matches; // …and a full-screen sheet; menus are bottom sheets
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

// Icons. Each has a glyph in the icon font (used when the look-alike fonts are installed in
// DATA_DIR/brand, see README) and a line drawing of our own on a 24px grid as the fallback.
const ICONS = {
  speaker: [0xe0e4, "M5 9.5h3l4.5-4v13L8 14.5H5zM16 9a4 4 0 0 1 0 6M18.5 6.5a7.5 7.5 0 0 1 0 11"],
  plus: [0xe001, "M12 5v14M5 12h14"],
  search: [0xe0d3, "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-3.5-3.5"],
  folder: [0xe072, "M3.5 7.5a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"],
  folderOpen: [0xe073, "M3.5 7.5a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"],
  projects: [0xe0c9, "M3.5 7.5a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"],
  artifacts: [0xe017, "M4 5.5A1.5 1.5 0 0 1 5.5 4h4A1.5 1.5 0 0 1 11 5.5v4A1.5 1.5 0 0 1 9.5 11h-4A1.5 1.5 0 0 1 4 9.5zM13 5.5A1.5 1.5 0 0 1 14.5 4h4A1.5 1.5 0 0 1 20 5.5v4a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 13 9.5zM4 14.5A1.5 1.5 0 0 1 5.5 13h4a1.5 1.5 0 0 1 1.5 1.5v4A1.5 1.5 0 0 1 9.5 20h-4A1.5 1.5 0 0 1 4 18.5zM13 14.5a1.5 1.5 0 0 1 1.5-1.5h4a1.5 1.5 0 0 1 1.5 1.5v4a1.5 1.5 0 0 1-1.5 1.5h-4a1.5 1.5 0 0 1-1.5-1.5z"],
  clock: [0xe043, "M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM12 7.5V12l3 2"],
  customize: [0xe100, "M4 7h9M17 7h3M4 17h3M11 17h9M15 5v4M9 15v4"],
  sliders: [0xf079, "M4 7h9M17 7h3M4 17h3M11 17h9M15 5v4M9 15v4"],
  sidebar: [0xe0dd, "M4 5.5A1.5 1.5 0 0 1 5.5 4h13A1.5 1.5 0 0 1 20 5.5v13a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18.5zM9.5 4v16"],
  menu: [0xe0aa, "M4 7h16M4 12h16M4 17h11"],
  edit: [0xe064, "M12 20h8M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"],
  more: [0xe061, "M6 12h.01M12 12h.01M18 12h.01"],
  chev: [0xe02a, "M9 6l6 6-6 6"],
  chevLeft: [0xe029, "M15 6l-6 6 6 6"],
  down: [0xe027, "M7 10l5 5 5-5"],
  downSmall: [0xf003, "M7 10l5 5 5-5"],
  upDown: [0xe02c, "M8 9.5l4-4 4 4M8 14.5l4 4 4-4"],
  up: [0xe013, "M12 19V5M5.5 11.5L12 5l6.5 6.5"],
  arrowDown: [0xe009, "M12 5v14M6 13l6 6 6-6"],
  arrowLeft: [0xe00c, "M19 12H5M11.5 5.5L5 12l6.5 6.5"],
  arrowRight: [0xe010, "M5 12h14M12.5 5.5L19 12l-6.5 6.5"],
  stop: [0xe0ec, "M7 7h10v10H7z"],
  x: [0xe10f, "M6 6l12 12M18 6L6 18"],
  check: [0xe03b, "M5 12.5l4.5 4.5L19 7.5"],
  copy: [0xe056, "M9 9.5A1.5 1.5 0 0 1 10.5 8h8A1.5 1.5 0 0 1 20 9.5v9a1.5 1.5 0 0 1-1.5 1.5h-8A1.5 1.5 0 0 1 9 18.5zM15 8V5.5A1.5 1.5 0 0 0 13.5 4h-8A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H9"],
  retry: [0xe11d, "M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"],
  rewind: [0xe11e, "M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"],
  pin: [0xe0bd, "M9 4h6l-1 6 4 3v2H6v-2l4-3zM12 15v5"],
  unpin: [0xe0bf, "M9 4h6l-1 6 4 3v2H6v-2l4-3zM12 15v5"],
  trash: [0xe101, "M5 7h14M10 11v6M14 11v6M6 7l1 12a1.5 1.5 0 0 0 1.5 1.4h7a1.5 1.5 0 0 0 1.5-1.4L18 7M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"],
  paperclip: [0xe019, "M20 11.5l-7.8 7.8a5 5 0 0 1-7.1-7.1l8.5-8.5a3.3 3.3 0 0 1 4.7 4.7L9.8 16.9a1.7 1.7 0 0 1-2.4-2.4l7.5-7.5"],
  image: [0xe08c, "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5zM9 10.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM20 15l-4.5-4.5L7 19"],
  camera: [0xe025, "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5zM9 10.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM20 15l-4.5-4.5L7 19"],
  mic: [0xe0ab, "M12 3.5a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0v-5a3 3 0 0 0-3-3zM6 11a6 6 0 0 0 12 0M12 17v3.5"],
  voice: [0xe108, "M12 3.5a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0v-5a3 3 0 0 0-3-3zM6 11a6 6 0 0 0 12 0M12 17v3.5"],
  ghost: [0xe075, "M6 20V11a6 6 0 0 1 12 0v9l-2-1.5-2 1.5-2-1.5-2 1.5-2-1.5zM10 11h.01M14 11h.01"],
  hand: [0xe085, "M8 13V6.5a1.5 1.5 0 0 1 3 0V12M11 11.5V5a1.5 1.5 0 0 1 3 0v6.5M14 11.5V6.5a1.5 1.5 0 0 1 3 0v7a6.5 6.5 0 0 1-6.5 6.5H10a5 5 0 0 1-4-2l-2.4-3.4a1.4 1.4 0 0 1 2.2-1.7L8 15"],
  bolt: [0xe098, "M13 3L5 13.5h6L10 21l8-10.5h-6z"],
  auto: [0xf0ab, "M13 3L5 13.5h6L10 21l8-10.5h-6z"],
  pen: [0xe064, "M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4"],
  terminal: [0xe04f, "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5zM8 9.5l3 2.5-3 2.5M13 15h3"],
  prompt: [0xe051, "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5zM8 9.5l3 2.5-3 2.5M13 15h3"],
  file: [0xe06c, "M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8zM14 3.5V8h4.5"],
  files: [0xe06f, "M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8zM14 3.5V8h4.5"],
  fileText: [0xe0b3, "M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8zM14 3.5V8h4.5M9 13h6M9 16.5h4"],
  globe: [0xe082, "M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM3.5 12h17M12 3.5c2.3 2.4 3.5 5.2 3.5 8.5s-1.2 6.1-3.5 8.5c-2.3-2.4-3.5-5.2-3.5-8.5S9.7 5.9 12 3.5z"],
  code: [0xe048, "M8.5 7L3.5 12l5 5M15.5 7l5 5-5 5M13.5 4.5l-3 15"],
  codeBlock: [0xe049, "M8.5 7L3.5 12l5 5M15.5 7l5 5-5 5M13.5 4.5l-3 15"],
  chart: [0xe02f, "M4 20V10M10 20V4M16 20v-7M21 20H3"],
  eye: [0xe069, "M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12zM12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z"],
  download: [0xe063, "M12 4v11M7 10.5l5 5 5-5M5 19.5h14"],
  external: [0xe015, "M14 4h6v6M20 4l-9 9M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10"],
  open: [0xe00e, "M14 4h6v6M20 4l-9 9M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10"],
  gear: [0xe0d6, "M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM19.4 13.5a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V19a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-2.7-1.1l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0-1.1-2.7H3.5a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.1-2.7l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 2.7-1.1V3.5a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 2.7 1.1l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0 1.1 2.7h.1a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"],
  logout: [0xe0a4, "M15 4h3.5A1.5 1.5 0 0 1 20 5.5v13a1.5 1.5 0 0 1-1.5 1.5H15M10 16.5L5.5 12 10 7.5M5.5 12H15"],
  sun: [0xe0ee, "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4L6 18M18 6l1.4-1.4"],
  moon: [0xe0b0, "M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"],
  type: [0xe0f5, "M5 7V5h14v2M12 5v14M9 19h6"],
  brain: [0xe068, "M9 4.5a2.5 2.5 0 0 0-2.5 2.5v.3A3 3 0 0 0 4.5 10a3 3 0 0 0 1 2.2A3 3 0 0 0 7 17.5a2.5 2.5 0 0 0 5 .5V7a2.5 2.5 0 0 0-3-2.5zM15 4.5A2.5 2.5 0 0 1 17.5 7v.3a3 3 0 0 1 2 2.7 3 3 0 0 1-1 2.2 3 3 0 0 1-1.5 5.3 2.5 2.5 0 0 1-5 .5"],
  play: [0xe0c1, "M7 5l12 7-12 7z"],
  pause: [0xe0bb, "M7 7h10v10H7z"],
  calendar: [0xe024, "M4.5 7A2.5 2.5 0 0 1 7 4.5h10A2.5 2.5 0 0 1 19.5 7v10a2.5 2.5 0 0 1-2.5 2.5H7A2.5 2.5 0 0 1 4.5 17zM4.5 9.5h15M8.5 3v3M15.5 3v3"],
  heart: [0xe127, "M12 20s-7.5-4.5-7.5-10A4.5 4.5 0 0 1 12 7a4.5 4.5 0 0 1 7.5 3c0 5.5-7.5 10-7.5 10z"],
  game: [0xe090, "M7 8.5h10a4 4 0 0 1 3.9 4.8l-.8 3.8a2 2 0 0 1-3.5.9L15 16H9l-1.6 2a2 2 0 0 1-3.5-.9l-.8-3.8A4 4 0 0 1 7 8.5zM8 11v3M6.5 12.5h3M15.5 12h.01M17.5 13.5h.01"],
  shield: [0xe0a3, "M12 3.5l7 3v5c0 4.5-3 8-7 9.5-4-1.5-7-5-7-9.5v-5z"],
  box: [0xe020, "M12 3.5l8 4.5v8l-8 4.5-8-4.5V8zM4 8l8 4.5L20 8M12 12.5v8"],
  bell: [0xe0b4, "M6 16.5V11a6 6 0 0 1 12 0v5.5l1.5 2h-15zM10 20.5a2 2 0 0 0 4 0"],
  user: [0xe104, "M12 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM4.5 20a7.5 7.5 0 0 1 15 0"],
  upload: [0xe103, "M12 16V4M7 8.5l5-4.5 5 4.5M5 15v3.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V15"],
  filter: [0xe070, "M4 6h16M7 12h10M10 18h4"],
  branch: [0xe078, "M6 4v10M6 14a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM18 4a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM18 10c0 4-4 5.5-12 6"],
  diff: [0xe07a, "M6 4v10M6 14a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM18 4a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM18 10c0 4-4 5.5-12 6"],
  sparkle: [0xe003, "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"],
  chats: [0xe039, "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"],
  chat: [0xe031, "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"],
  devices: [0xf031, "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5zM8 9.5l3 2.5-3 2.5M13 15h3"],
  computer: [0xe053, "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5zM8 9.5l3 2.5-3 2.5M13 15h3"],
  laptop: [0xe093, "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5zM8 9.5l3 2.5-3 2.5M13 15h3"],
  thumbUp: [0xe0fb, "M5 12.5l4.5 4.5L19 7.5"],
  thumbDown: [0xe0f9, "M6 6l12 12M18 6L6 18"],
  share: [0xe0da, "M12 16V4M7 8.5l5-4.5 5 4.5M5 15v3.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V15"],
  star: [0xe0e7, "M9 4h6l-1 6 4 3v2H6v-2l4-3zM12 15v5"],
  styles: [0xe0ed, "M5 7V5h14v2M12 5v14M9 19h6"],
  info: [0xe08f, "M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12zM12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z"],
  warning: [0xe109, "M12 3.5l7 3v5c0 4.5-3 8-7 9.5-4-1.5-7-5-7-9.5v-5z"],
  help: [0xe088, "M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12zM12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z"],
  link: [0xe099, "M20 11.5l-7.8 7.8a5 5 0 0 1-7.1-7.1l8.5-8.5a3.3 3.3 0 0 1 4.7 4.7L9.8 16.9a1.7 1.7 0 0 1-2.4-2.4l7.5-7.5"],
  lock: [0xe0a1, "M12 3.5l7 3v5c0 4.5-3 8-7 9.5-4-1.5-7-5-7-9.5v-5z"],
  bulb: [0xe097, "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"],
  research: [0xe0d0, "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-3.5-3.5"],
  tasks: [0xe0f1, "M5 12.5l4.5 4.5L19 7.5"],
  spinner: [0xe0e5, "M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"],
  expand: [0xe067, "M14 4h6v6M20 4l-9 9M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10"],
  collapse: [0xe04d, "M6 6l12 12M18 6L6 18"],
  paneRight: [0xf03f, "M4 5.5A1.5 1.5 0 0 1 5.5 4h13A1.5 1.5 0 0 1 20 5.5v13a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18.5zM9.5 4v16"],
  plugin: [0xe0c5, "M12 3.5l8 4.5v8l-8 4.5-8-4.5V8zM4 8l8 4.5L20 8M12 12.5v8"],
  connectors: [0xe055, "M12 3.5l8 4.5v8l-8 4.5-8-4.5V8zM4 8l8 4.5L20 8M12 12.5v8"],
  book: [0xe01d, "M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8zM14 3.5V8h4.5M9 13h6M9 16.5h4"],
  wrench: [0xe10d, "M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM19.4 13.5a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V19a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-2.7-1.1l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0-1.1-2.7H3.5a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.1-2.7l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 2.7-1.1V3.5a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 2.7 1.1l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0 1.1 2.7h.1a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"],
  checkCircle: [0xe03c, "M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM8 12.3l2.7 2.7L16 9.5"],
  xCircle: [0xe110, "M6 6l12 12M18 6L6 18"],
  circle: [0xe186, "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16z"],
  minus: [0xe0ae, "M5 12h14"],
  list: [0xe09c, "M4 7h16M4 12h16M4 17h11"],
  grid: [0xe084, "M4 5.5A1.5 1.5 0 0 1 5.5 4h4A1.5 1.5 0 0 1 11 5.5v4A1.5 1.5 0 0 1 9.5 11h-4A1.5 1.5 0 0 1 4 9.5zM13 5.5A1.5 1.5 0 0 1 14.5 4h4A1.5 1.5 0 0 1 20 5.5v4a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 13 9.5zM4 14.5A1.5 1.5 0 0 1 5.5 13h4a1.5 1.5 0 0 1 1.5 1.5v4A1.5 1.5 0 0 1 9.5 20h-4A1.5 1.5 0 0 1 4 18.5zM13 14.5a1.5 1.5 0 0 1 1.5-1.5h4a1.5 1.5 0 0 1 1.5 1.5v4a1.5 1.5 0 0 1-1.5 1.5h-4a1.5 1.5 0 0 1-1.5-1.5z"],
  agent: [0xe11b, "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z"],
  timer: [0xe0fd, "M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM12 7.5V12l3 2"],
  archive: [0xe008, "M12 3.5l8 4.5v8l-8 4.5-8-4.5V8zM4 8l8 4.5L20 8M12 12.5v8"],
  reply: [0xf087, "M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"],
  keyboard: [0xe092, "M5 7V5h14v2M12 5v14M9 19h6"],
  phone: [0xe0af, "M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v11a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5zM8 9.5l3 2.5-3 2.5M13 15h3"],
  cloud: [0xe044, "M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM3.5 12h17M12 3.5c2.3 2.4 3.5 5.2 3.5 8.5s-1.2 6.1-3.5 8.5c-2.3-2.4-3.5-5.2-3.5-8.5S9.7 5.9 12 3.5z"],
  send: [0xe0ba, "M12 19V5M5.5 11.5L12 5l6.5 6.5"],
  history: [0xe089, "M12 3.5a8.5 8.5 0 1 0 0 17 8.5 8.5 0 0 0 0-17zM12 7.5V12l3 2"],
  home: [0xe08a, "M12 3.5l8 4.5v8l-8 4.5-8-4.5V8zM4 8l8 4.5L20 8M12 12.5v8"],
  changes: [0xe02d, "M6 4v10M6 14a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM18 4a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM18 10c0 4-4 5.5-12 6"],
  folderPlus: [0xe074, "M3.5 7.5a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"],
  text: [0xf0af, "M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8zM14 3.5V8h4.5M9 13h6M9 16.5h4"],
  palette: [0xe0b8, "M5 7V5h14v2M12 5v14M9 19h6"],
  memory: [0xe0a8, "M9 4.5a2.5 2.5 0 0 0-2.5 2.5v.3A3 3 0 0 0 4.5 10a3 3 0 0 0 1 2.2A3 3 0 0 0 7 17.5a2.5 2.5 0 0 0 5 .5V7a2.5 2.5 0 0 0-3-2.5zM15 4.5A2.5 2.5 0 0 1 17.5 7v.3a3 3 0 0 1 2 2.7 3 3 0 0 1-1 2.2 3 3 0 0 1-1.5 5.3 2.5 2.5 0 0 1-5 .5"],
  library: [0xe096, "M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8zM14 3.5V8h4.5M9 13h6M9 16.5h4"],
};
let glyphs = false;
export function useGlyphs(on) { glyphs = !!on; document.documentElement.toggleAttribute("data-glyphs", glyphs); }
// icon("plus") is 20px; icon("down", "s12") / "s16" are the small sizes.
export function icon(name, cls = "") {
  const def = ICONS[name] || ICONS.file;
  if (glyphs) {
    const g = document.createElement("span");
    g.className = `i ${cls}`.trim();
    g.setAttribute("aria-hidden", "true");
    g.textContent = String.fromCodePoint(def[0]);
    return g;
  }
  const ns = "http://www.w3.org/2000/svg";
  const s = document.createElementNS(ns, "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("class", `i ${cls}`.trim());
  s.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", def[1]);
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
// items: [{label, sub, icon, right, checked, danger, disabled, onClick, switch:bool, on:bool, children:[…]} | "sep" | {heading}]
// A dropdown beside its anchor on a computer, a bottom sheet on a phone. Items with children open a submenu.
const stack = []; // open menus, root first
let sheetBack = null;
const sheetMode = () => touch && phone();
export function closeMenu(from = 0) {
  for (const m of stack.splice(from)) {
    m._anchor?.setAttribute?.("aria-expanded", "false");
    m._anchor?.closest?.(".row")?.classList.remove("menu-open");
    if (m.classList.contains("sheet")) { m.classList.add("out"); setTimeout(() => m.remove(), 220); } else m.remove();
  }
  if (stack.length) stack.at(-1).hidden = false;
  if (!stack.length && sheetBack) { const sb = sheetBack; sheetBack = null; sb.classList.add("out"); setTimeout(() => sb.remove(), 220); }
}
export function menu(anchor, items, { align = "start", side = "auto", parent = null, title = "", wide = false, gap = 6 } = {}) {
  if (parent) closeMenu(stack.indexOf(parent) + 1);
  else { const same = stack[0]?._anchor === anchor; closeMenu(); if (same) return null; }
  const sheet = sheetMode();
  const m = h(`div.menu${sheet ? ".sheet" : ""}${wide ? ".wide" : ""}`, { role: "menu", tabindex: "-1" });
  if (sheet) {
    m.append(h("div.grab"));
    if (parent) m.append(h("button.mi.back", { type: "button", on: { click: (e) => { e.stopPropagation(); closeMenu(stack.indexOf(m)); } } }, icon("chevLeft"), h("span.grow", title || "Back")));
  }
  for (const it of items) {
    if (!it) continue;
    if (it === "sep") { if (m.lastChild && !m.lastChild.classList?.contains("sep")) m.append(h("div.sep")); continue; }
    if (it.heading) { m.append(h(`div.label${it.wrap ? ".wrap" : ""}`, it.heading)); continue; }
    const b = h(`button.mi${it.danger ? ".danger" : ""}`, { "data-key": it.key || null, role: it.checked !== undefined ? "menuitemradio" : "menuitem", type: "button", disabled: it.disabled, "aria-checked": it.checked !== undefined ? String(!!it.checked) : null, "aria-haspopup": it.children ? "menu" : null },
      it.icon ? icon(it.icon) : null,
      h("span.grow", h("span.l", it.label), it.sub ? h("span.sub", it.sub) : null),
      it.right || it.key ? h("span.right", it.right || it.key) : null,
      it.switch ? h(`span.sw${it.on ? ".on" : ""}`) : null,
      it.checked ? icon("check", "tick") : null,
      it.children ? icon("chev", "s16 go") : null);
    const openChild = () => menu(b, typeof it.children === "function" ? it.children() : it.children, { parent: m, align: "start", title: it.label, wide });
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      if (it.children) { if (stack[stack.indexOf(m) + 1]?._anchor !== b) openChild(); return; }
      if (it.switch) { it.on = !it.on; b.querySelector(".sw").classList.toggle("on", it.on); it.onClick?.(it.on); return; }
      closeMenu();
      it.onClick?.();
    });
    if (!sheet) {
      let t = null;
      b.addEventListener("pointerenter", () => {
        clearTimeout(t);
        t = setTimeout(() => {
          const i = stack.indexOf(m);
          if (i < 0) return;
          if (it.children) { if (stack[i + 1]?._anchor !== b) openChild(); } else closeMenu(i + 1);
        }, it.children ? 120 : 200);
      });
      b.addEventListener("pointerleave", () => clearTimeout(t));
    }
    m.append(b);
  }
  if (m.lastChild?.classList?.contains("sep")) m.lastChild.remove();
  m._anchor = anchor;
  m.addEventListener("keydown", (e) => {
    const its = [...m.querySelectorAll(".mi:not(:disabled)")];
    const i = its.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); its[(i + 1) % its.length]?.focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); its[(i - 1 + its.length) % its.length]?.focus(); }
    else if (e.key === "Home") { e.preventDefault(); its[0]?.focus(); }
    else if (e.key === "End") { e.preventDefault(); its.at(-1)?.focus(); }
    else if (e.key === "ArrowRight" && document.activeElement?.getAttribute("aria-haspopup")) { e.preventDefault(); document.activeElement.click(); stack.at(-1)?.querySelector(".mi:not(:disabled)")?.focus(); }
    else if (e.key === "ArrowLeft" && parent) { e.preventDefault(); e.stopPropagation(); closeMenu(stack.indexOf(m)); anchor.focus(); }
    else if (e.key === "Tab") { e.preventDefault(); closeMenu(); }
    else if (e.key.length === 1 && !e.metaKey && !e.ctrlKey && !e.altKey) {
      const hit = its.find((b) => b.dataset.key === e.key.toUpperCase());
      if (!hit) return;
      e.preventDefault(); hit.click();
    }
    else return;
    e.stopPropagation();
  });
  if (sheet) {
    if (!sheetBack) { sheetBack = h("div.sheet-back", { on: { click: () => closeMenu() } }); document.body.append(sheetBack); }
    if (parent) parent.hidden = true;
    document.body.append(m);
  } else {
    document.body.append(m);
    const r = anchor.getBoundingClientRect();
    const mw = m.offsetWidth, mh = m.offsetHeight, vw = innerWidth, vh = innerHeight;
    let left, top, origin;
    if (parent) {
      const right = r.right + mw + 4 < vw || r.left - mw - 4 < 0;
      left = right ? r.right + 2 : r.left - mw - 2;
      top = Math.min(r.top - 6, vh - mh - 8);
      origin = `${right ? "left" : "right"} top`;
    } else {
      const below = side === "below" || (side === "auto" && r.bottom + mh + 8 < vh) || (side !== "above" && r.top < mh + 8);
      left = align === "end" ? r.right - mw : r.left;
      top = below ? r.bottom + gap : r.top - mh - gap;
      origin = `${align === "end" ? "right" : "left"} ${below ? "top" : "bottom"}`;
    }
    m.style.left = `${Math.max(8, Math.min(left, vw - mw - 8))}px`;
    m.style.top = `${Math.max(8, top)}px`;
    m.style.setProperty("--origin", origin);
  }
  if (!parent) { anchor.setAttribute?.("aria-expanded", "true"); anchor.closest?.(".row")?.classList.add("menu-open"); }
  stack.push(m);
  m.focus({ preventScroll: true });
  return m;
}
document.addEventListener("click", (e) => {
  if (!stack.length) return;
  if (stack.some((m) => m.contains(e.target)) || stack[0]._anchor?.contains?.(e.target)) return;
  closeMenu();
});
addEventListener("resize", () => closeMenu());
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (stack.length) { const a = stack[0]._anchor; closeMenu(); a?.focus?.(); e.stopPropagation(); return; }
  closeModal();
});

// --- tooltips ---------------------------------------------------------------
// Any element with a title gets the app's own tooltip instead of the browser's (not on touch screens).
let tip = null, tipTimer = null, tipFor = null, tipLast = 0;
function hideTip() { clearTimeout(tipTimer); tip?.remove(); if (tip) tipLast = Date.now(); tip = null; tipFor = null; }
function showTip(el) {
  const text = el.dataset.tip;
  if (!text || !el.isConnected || el.getAttribute("aria-expanded") === "true") return;
  tip = h("div.tip", { role: "tooltip" }, text);
  document.body.append(tip);
  const r = el.getBoundingClientRect(), w = tip.offsetWidth, ht = tip.offsetHeight;
  const above = el.dataset.tipSide !== "bottom" && r.top - ht - 8 > 0;
  tip.style.left = `${Math.max(6, Math.min(r.left + r.width / 2 - w / 2, innerWidth - w - 6))}px`;
  tip.style.top = `${above ? r.top - ht - 4 : r.bottom + 4}px`;
  tip.style.transformOrigin = above ? "bottom" : "top";
}
if (!touch) {
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest?.("[title], [data-tip]");
    if (el === tipFor) return;
    hideTip();
    if (!el) return;
    if (el.title) { el.dataset.tip = el.title; el.removeAttribute("title"); }
    if (el.dataset.tipIf === "overflow") { const l = el.querySelector(".label") || el; if (l.scrollWidth <= l.clientWidth) return; }
    tipFor = el;
    tipTimer = setTimeout(() => showTip(el), Date.now() - tipLast < 200 ? 0 : 300);
  });
  for (const ev of ["pointerdown", "keydown", "wheel"]) document.addEventListener(ev, hideTip, { capture: true, passive: true });
  document.addEventListener("pointerleave", hideTip);
}

// --- modals -----------------------------------------------------------------
let openModal = null, modalReturn = null;
export function closeModal() {
  if (!openModal) return;
  const back = openModal;
  openModal = null;
  back.classList.add("out");
  setTimeout(() => back.remove(), 150);
  modalReturn?.focus?.({ preventScroll: true });
  modalReturn = null;
}
export function modal(content, { cls = "" } = {}) {
  closeMenu();
  if (openModal) { openModal.remove(); openModal = null; } else modalReturn = document.activeElement;
  const box = h(`div.modal${cls ? `.${cls}` : ""}`, { role: "dialog", "aria-modal": "true", tabindex: "-1" }, content);
  const back = h("div.modal-back", box);
  back.addEventListener("mousedown", (e) => { if (e.target === back) closeModal(); });
  // Keep Tab inside the dialog.
  box.addEventListener("keydown", (e) => {
    if (e.key !== "Tab") return;
    const f = [...box.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])")].filter((x) => !x.disabled && x.offsetParent !== null);
    if (!f.length) return;
    if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f.at(-1).focus(); }
    else if (!e.shiftKey && document.activeElement === f.at(-1)) { e.preventDefault(); f[0].focus(); }
  });
  document.body.append(back);
  openModal = back;
  const first = box.querySelector("[autofocus]") || (touch ? null : box.querySelector("input, textarea, select")) || box;
  setTimeout(() => { first.focus({ preventScroll: true }); if (first.select && first.type !== "file") try { first.select(); } catch {} }, 30);
  return box;
}
// A small form in a dialog. fields: [{name, label, type, value, placeholder, hint, options:[[v,l]], show(value, data)}]
export function formModal({ title, text, fields, submit = "Save", danger, onSubmit, extra, cls = "" }) {
  return new Promise((resolve) => {
    const form = h("form", h("h2", title), text ? h("p.modal-text", text) : null);
    for (const f of fields) {
      let input;
      if (f.type === "textarea") input = h("textarea.input", { name: f.name, placeholder: f.placeholder || "", rows: f.rows || 5 }, f.value || "");
      else if (f.type === "select") input = h("select.input", { name: f.name }, f.options.map(([v, l]) => h("option", { value: v, selected: String(v) === String(f.value) }, l)));
      else input = h("input.input", { name: f.name, type: f.type || "text", value: f.value ?? "", placeholder: f.placeholder || "", autocomplete: "off" });
      if (f.autofocus) input.setAttribute("autofocus", "");
      const field = h("label.field", { "data-field": f.name }, f.label ? h("span", f.label) : null, input, f.hint ? h("span.hint", f.hint) : null);
      if (f.show) field.hidden = !f.show(f.value);
      form.append(field);
    }
    if (extra) form.append(extra);
    const cancel = h("button.btn.secondary", { type: "button", on: { click: () => { closeModal(); resolve(null); } } }, "Cancel");
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
    modal(form, { cls });
    sync();
  });
}
export function confirmModal(title, text, ok = "Delete", { danger = true } = {}) {
  return new Promise((resolve) => {
    modal([
      h("h2", title), h("p.modal-text", text),
      h("div.actions",
        h("button.btn.secondary", { type: "button", on: { click: () => { closeModal(); resolve(false); } } }, "Cancel"),
        h(`button.btn.${danger ? "danger" : "primary"}`, { type: "button", autofocus: true, on: { click: () => { closeModal(); resolve(true); } } }, ok)),
    ], { cls: "confirm" });
  });
}

// --- toasts -----------------------------------------------------------------
export function toast(text, ms = 6000) {
  const box = document.getElementById("toasts");
  const close = () => { if (!t.isConnected) return; t.classList.add("out"); setTimeout(() => t.remove(), 200); };
  const t = h("div.toast", { role: "status" }, icon("info"), h("div.grow", text),
    h("button.icon-btn.sm", { type: "button", "aria-label": "Dismiss", on: { click: close } }, icon("x", "s16")));
  box.append(t);
  setTimeout(close, ms);
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
// "just now", "2 minutes ago", "3 hours ago", then the date after a week.
export function agoLong(ts) {
  const s = (Date.now() - ts) / 1000;
  const n = (v, u) => `${v} ${u}${v === 1 ? "" : "s"} ago`;
  if (s < 60) return "just now";
  if (s < 3600) return n(Math.floor(s / 60), "minute");
  if (s < 86400) return n(Math.floor(s / 3600), "hour");
  if (s < 7 * 86400) return n(Math.floor(s / 86400), "day");
  return new Date(ts).toLocaleDateString(undefined, { day: "numeric", month: "short", year: new Date(ts).getFullYear() === new Date().getFullYear() ? undefined : "numeric" });
}
export const iconHTML = (name, cls) => icon(name, cls).outerHTML;
export const when = (ts) => new Date(ts).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
export function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
export async function copyText(text, { quiet = false } = {}) {
  try { await navigator.clipboard.writeText(text); }
  catch {
    const ta = h("textarea", { style: { position: "fixed", opacity: "0" } }, text);
    document.body.append(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (!quiet) toast("Copied to clipboard");
}
