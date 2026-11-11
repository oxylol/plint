// Markdown for model replies. Text is escaped before any markup is added, so a
// reply can never inject HTML; links are limited to http(s).
import { esc, iconHTML } from "./ui.js";

function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => `\u0000${codes.push(c) - 1}\u0000`);
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/(^|[^\w])_([^_\n]+)_(?!\w)/g, "$1<em>$2</em>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?'"])/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${esc(codes[i])}</code>`);
}

// A light highlighter: comments, strings, numbers, keywords. Good enough to read.
const KW = /\b(const|let|var|function|return|if|else|for|while|do|switch|case|break|continue|new|class|extends|import|from|export|default|async|await|try|catch|finally|throw|def|elif|lambda|with|as|pass|yield|in|of|not|and|or|is|None|True|False|null|true|false|undefined|this|self|fn|pub|struct|impl|use|mod|match|type|interface|enum|package|func|go|defer|echo|then|fi|done|esac|local|sudo|SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|JOIN|ORDER|BY|GROUP)\b/g;
export function highlight(code, lang) {
  if (code.length > 40000 || /^(text|txt|plain|output|log)$/i.test(lang || "")) return esc(code);
  const hashComments = /^(py|python|sh|bash|shell|zsh|yaml|yml|toml|ruby|rb|r|conf|ini|dockerfile|make|makefile|perl)$/i.test(lang || "");
  const re = hashComments
    ? /(#[^\n]*)|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d+)?\b)/g
    : /(\/\/[^\n]*|\/\*[\s\S]*?\*\/|<!--[\s\S]*?-->)|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`)|(\b\d+(?:\.\d+)?\b)/g;
  let out = "", last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index)).replace(KW, '<span class="tok-k">$1</span>');
    const cls = m[1] ? "tok-c" : m[2] ? "tok-s" : "tok-n";
    out += `<span class="${cls}">${esc(m[0])}</span>`;
    last = m.index + m[0].length;
  }
  return out + esc(code.slice(last)).replace(KW, '<span class="tok-k">$1</span>');
}

function list(lines, i) {
  // Parse a (possibly nested) list starting at lines[i]; returns [html, nextIndex].
  const first = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+/);
  const indent = first[1].length;
  const ordered = /\d/.test(first[2]);
  const start = ordered ? parseInt(first[2], 10) : 1;
  const items = [];
  while (i < lines.length) {
    const m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (!m || m[1].length < indent) break;
    if (m[1].length > indent) { // nested list belongs to the previous item
      const [sub, next] = list(lines, i);
      if (items.length) items[items.length - 1] += sub;
      i = next;
      continue;
    }
    let text = m[3];
    i++;
    while (i < lines.length && lines[i].trim() && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i]) && /^\s+/.test(lines[i])) text += ` ${lines[i++].trim()}`;
    const task = text.match(/^\[( |x)\]\s+(.*)/i);
    items.push(task ? `${task[1] === " " ? "☐" : "☑"} ${inline(task[2])}` : inline(text));
    while (i < lines.length && !lines[i].trim() && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i + 1] || "") && (lines[i + 1].match(/^\s*/)[0].length >= indent)) i++;
  }
  const tag = ordered ? "ol" : "ul";
  return [`<${tag}${ordered && start !== 1 ? ` start="${start}"` : ""}>${items.map((x) => `<li>${x}</li>`).join("")}</${tag}>`, i];
}

export function markdown(src) {
  const out = [];
  const lines = String(src || "").replace(/\r/g, "").split("\n");
  const para = [];
  const flush = () => { if (para.length) { out.push(`<p>${inline(para.join("\n")).replace(/\n/g, "<br>")}</p>`); para.length = 0; } };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    let m;
    if ((m = line.match(/^\s*(```+|~~~+)\s*([\w+#.-]*)/))) {
      flush();
      const fence = m[1];
      const body = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence)) body.push(lines[i++]);
      i++;
      const lang = m[2] || "";
      out.push(`<div class="code" role="group"><button class="icon-btn code-copy" data-copy type="button" aria-label="Copy to clipboard" title="Copy">${iconHTML("copy")}</button>${lang ? `<div class="code-lang">${esc(lang.toLowerCase())}</div>` : ""}<pre${lang ? "" : ' class="wrap"'}><code>${highlight(body.join("\n"), lang)}</code></pre></div>`);
      continue;
    }
    if ((m = line.match(/^(#{1,6})\s+(.*?)\s*#*$/))) { flush(); const n = Math.min(4, m[1].length); out.push(`<h${n}>${inline(m[2])}</h${n}>`); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push("<hr>"); i++; continue; }
    if (/^\s*>/.test(line)) {
      flush();
      const q = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ""));
      out.push(`<blockquote>${markdown(q.join("\n"))}</blockquote>`);
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1] || "")) {
      flush();
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map((c) => inline(c.trim()));
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }
    if (/^(\s*)([-*+]|\d+[.)])\s+/.test(line)) { flush(); const [html, next] = list(lines, i); out.push(html); i = next; continue; }
    if (!line.trim()) { flush(); i++; continue; }
    para.push(line);
    i++;
  }
  flush();
  return out.join("");
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-copy]");
  if (!b) return;
  const code = b.closest(".code").querySelector("code").textContent;
  try { await navigator.clipboard.writeText(code); } catch {}
  b.innerHTML = iconHTML("check");
  setTimeout(() => (b.innerHTML = iconHTML("copy")), 1600);
});

// While a reply streams, new words fade in. `shown` is how many characters were already on screen.
export function fadeFresh(root, shown) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let seen = 0;
  const fresh = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const len = n.data.length;
    if (seen + len > shown && n.data.trim() && !n.parentNode.closest("pre")) fresh.push([n, Math.max(0, shown - seen)]);
    seen += len;
  }
  for (const [n, at] of fresh) {
    const tail = at ? n.splitText(at) : n;
    const span = document.createElement("span");
    span.className = "fresh";
    tail.replaceWith(span);
    span.append(tail);
  }
  return seen;
}
