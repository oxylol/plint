// Claude agent loop (browser side) + chat rendering.
// The server streams NDJSON events; tools run here against the simulator.

const MAX_TOOL_TURNS = 8;

// Decode the (possibly incomplete) JSON string value of `key` from a partial JSON object.
export function partialJsonString(json, key) {
  const m = new RegExp(`"${key}"\\s*:\\s*"`).exec(json);
  if (!m) return null;
  let i = m.index + m[0].length;
  let out = '';
  while (i < json.length) {
    const ch = json[i];
    if (ch === '"') return { value: out, complete: true };
    if (ch === '\\') {
      const n = json[i + 1];
      if (n === undefined) break;
      const map = { n: '\n', t: '\t', r: '\r', b: '\b', f: '\f', '"': '"', '\\': '\\', '/': '/' };
      if (n === 'u') {
        const hex = json.slice(i + 2, i + 6);
        if (hex.length < 4) break;
        out += String.fromCharCode(parseInt(hex, 16));
        i += 6;
        continue;
      }
      out += map[n] ?? n;
      i += 2;
      continue;
    }
    out += ch;
    i++;
  }
  return { value: out, complete: false };
}

// After a mid-output refusal fallback, model-internal blocks before the last fallback marker must not be echoed back.
function sanitizeAssistant(content) {
  const lastFb = content.map((b) => b.type).lastIndexOf('fallback');
  if (lastFb < 0) return content;
  return content.filter((b, i) => i > lastFb || !['thinking', 'redacted_thinking', 'tool_use', 'server_tool_use'].includes(b.type));
}

export class ClaudeAgent {
  constructor({ getSettings, buildContext, tools, view }) {
    this.getSettings = getSettings;
    this.buildContext = buildContext;
    this.tools = tools; // { write_program(input, ui) -> {content, summary}, view_simulation(input) -> {content} }
    this.view = view;
    this.history = [];
    this.busy = false;
    this.abort = null;
    this.usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
  }

  reset() {
    if (this.busy) this.stop();
    this.history = [];
    this.view.clear();
  }

  stop() {
    this.stopped = true;
    this.abort?.abort();
  }

  // images: [{ mediaType, data }], documents: [{ name, data }] (PDF drawings, base64)
  async send(text, { images = [], documents = [] } = {}) {
    if (this.busy) return;
    this.busy = true;
    this.stopped = false;
    this.view.setBusy(true);
    const context = this.buildContext();
    const content = [];
    for (const d of documents) content.push({ type: 'document', source: { type: 'base64', media_type: 'application/pdf', data: d.data }, title: d.name });
    for (const img of images) content.push({ type: 'image', source: { type: 'base64', media_type: img.mediaType, data: img.data } });
    content.push({ type: 'text', text: `${context}\n\n<user_request>\n${text}\n</user_request>` });
    this.history.push({ role: 'user', content });
    const turnStart = this.history.length - 1;
    this.view.addUser(text, images, documents);
    try {
      for (let turn = 0; turn <= MAX_TOOL_TURNS; turn++) {
        const final = await this.streamTurn();
        if (!final) {
          // the request never got an answer: drop it so a retry starts clean
          if (this.history.length - 1 === turnStart) this.history.pop();
          break;
        }
        this.history.push({ role: 'assistant', content: sanitizeAssistant(final.content) });
        this.addUsage(final.usage);
        if (final.stop_reason === 'refusal') {
          this.view.notice('Claude declined this request' + (final.stop_details?.explanation ? `: ${final.stop_details.explanation}` : '.'), 'error');
          break;
        }
        if (final.stop_reason === 'max_tokens') {
          this.view.notice('The response hit the output limit before finishing.', 'warning');
          if (final.content.some((b) => b.type === 'tool_use')) break; // truncated tool input: never run it
        }
        const uses = final.content.filter((b) => b.type === 'tool_use');
        if (final.stop_reason !== 'tool_use' || !uses.length) break;
        if (turn === MAX_TOOL_TURNS) {
          this.view.notice(`Stopped after ${MAX_TOOL_TURNS} tool rounds.`, 'warning');
          // answer the pending calls so the history stays valid
          this.history.push({ role: 'user', content: uses.map((u) => ({ type: 'tool_result', tool_use_id: u.id, content: 'Tool round limit reached; stop and summarise.', is_error: true })) });
          break;
        }
        const results = [];
        for (const u of uses) results.push(await this.runTool(u));
        this.history.push({ role: 'user', content: results });
        if (this.stopped) break;
      }
    } catch (e) {
      if (this.history.length - 1 === turnStart) this.history.pop();
      if (e.name !== 'AbortError') this.view.notice(e.message || String(e), 'error');
    } finally {
      this.busy = false;
      this.abort = null;
      this.view.setBusy(false);
      this.view.setUsage(this.usage);
    }
  }

  addUsage(u) {
    if (!u) return;
    this.usage.input += u.input_tokens || 0;
    this.usage.output += u.output_tokens || 0;
    this.usage.cacheRead += u.cache_read_input_tokens || 0;
    this.usage.cacheWrite += u.cache_creation_input_tokens || 0;
  }

  async runTool(use) {
    const card = this.view.toolCard(use.id);
    let input = use.input;
    if (typeof input === 'string') { try { input = JSON.parse(input); } catch { input = null; } }
    const fn = this.tools[use.name];
    if (!fn) return { type: 'tool_result', tool_use_id: use.id, content: `Unknown tool ${use.name}`, is_error: true };
    const QUERIES = ['summary', 'view', 'section', 'probe', 'compare', 'target'];
    const bad = !input || typeof input !== 'object'
      ? 'Invalid input: expected a JSON object'
      : use.name === 'write_program' && typeof input.gcode !== 'string' ? 'Invalid input: expected {"gcode": "<complete program>"}'
      : use.name === 'inspect_part' && !QUERIES.includes(input.query) ? `Invalid input: query must be one of ${QUERIES.join(', ')}`
      : use.name === 'inspect_part' && input.query === 'section' && (!['x', 'y', 'z'].includes(String(input.axis).toLowerCase()) || !Number.isFinite(Number(input.at))) ? 'Invalid input: a section needs axis (x, y or z) and at (mm)'
      : use.name === 'inspect_part' && input.query === 'probe' && (!Array.isArray(input.points) || !input.points.every((p) => Array.isArray(p) && p.length === 3 && p.every((v) => Number.isFinite(Number(v))))) ? 'Invalid input: probe needs points as [[x, y, z], …]'
      : null;
    if (bad) {
      card?.done({ error: bad });
      return { type: 'tool_result', tool_use_id: use.id, content: bad, is_error: true };
    }
    try {
      const r = await fn(input, card);
      return { type: 'tool_result', tool_use_id: use.id, content: r.content, ...(r.isError ? { is_error: true } : {}) };
    } catch (e) {
      card?.done({ error: e.message });
      return { type: 'tool_result', tool_use_id: use.id, content: `Tool failed: ${e.message}`, is_error: true };
    }
  }

  async streamTurn() {
    const s = this.getSettings();
    this.abort = new AbortController();
    const headers = { 'Content-Type': 'application/json' };
    if (s.apiKey) headers['x-anthropic-key'] = s.apiKey;
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers,
      body: JSON.stringify({ model: s.model, effort: s.effort, messages: this.history }),
      signal: this.abort.signal,
    });
    if (!resp.ok || !resp.body) {
      let msg = `Server error ${resp.status}`;
      try { msg = (await resp.json()).error || msg; } catch { /* not json */ }
      throw new Error(msg);
    }
    const msgView = this.view.startAssistant();
    const blocks = new Map();
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let final = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        let ev;
        try { ev = JSON.parse(line); } catch { continue; }
        switch (ev.t) {
          case 'start': msgView.setModel(ev.model); break;
          case 'block_start': {
            const b = { type: ev.type, name: ev.name, id: ev.id, text: '', json: '' };
            blocks.set(ev.i, b);
            if (ev.type === 'thinking') b.ui = msgView.thinking();
            else if (ev.type === 'text') b.ui = msgView.text();
            else if (ev.type === 'tool_use') {
              b.ui = msgView.tool(ev.name, ev.id);
              if (ev.name === 'write_program') this.tools.beginWrite?.();
            } else if (ev.type === 'fallback') msgView.fallback(ev.from, ev.to);
            break;
          }
          case 'delta': {
            const b = blocks.get(ev.i);
            if (!b) break;
            if (ev.kind === 'input') {
              b.json += ev.v;
              if (b.name === 'write_program') {
                const g = partialJsonString(b.json, 'gcode');
                if (g && !b.prepared) {
                  // the program is starting: switch to (or create) the operation it is for first
                  b.prepared = true;
                  const op = partialJsonString(b.json, 'operation'), mc = partialJsonString(b.json, 'machine');
                  this.tools.prepareWrite?.({ operation: op?.complete ? op.value : undefined, machine: mc?.complete ? mc.value : undefined });
                }
                if (g) { this.tools.streamWrite?.(g.value); b.ui?.progress(g.value); }
                const t = partialJsonString(b.json, 'title');
                if (t) b.ui?.title(t.value);
              }
            } else {
              b.text += ev.v;
              b.ui?.append(ev.v);
            }
            break;
          }
          case 'block_stop': {
            const b = blocks.get(ev.i);
            b?.ui?.finish?.();
            if (b?.type === 'tool_use' && b.name === 'write_program') this.tools.endWrite?.();
            break;
          }
          case 'final': final = ev.message; break;
          case 'error':
            msgView.finish();
            msgView.discardIfEmpty();
            if (ev.kind === 'aborted') return null;
            this.view.notice(ev.message, 'error', ev.kind);
            if (ev.kind === 'auth') this.view.needKey?.();
            return null;
        }
      }
    }
    msgView.finish(final?.usage);
    if (!final) { if (!this.stopped) this.view.notice('The stream ended unexpectedly.', 'error'); return null; }
    return final;
  }
}

// ---------------------------------------------------------------------------
// Chat rendering
// ---------------------------------------------------------------------------

function esc(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function renderMarkdown(src) {
  const lines = src.replace(/\r/g, '').split('\n');
  let html = '';
  let inCode = false, code = [];
  let list = null;
  let para = [];
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\s][^*]*)\*(?=[\s).,!?:;]|$)/g, '$1<em>$2</em>');
  const flushPara = () => { if (para.length) { html += `<p>${inline(para.join(' '))}</p>`; para = []; } };
  const flushList = () => { if (list) { html += `<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join('')}</${list.tag}>`; list = null; } };
  for (const line of lines) {
    if (/^```/.test(line)) {
      if (inCode) { html += `<pre><code>${esc(code.join('\n'))}</code></pre>`; code = []; inCode = false; }
      else { flushPara(); flushList(); inCode = true; }
      continue;
    }
    if (inCode) { code.push(line); continue; }
    if (/^\s*\|.*\|\s*$/.test(line)) {
      flushPara(); flushList();
      const cells = line.trim().slice(1, -1).split('|').map((c) => c.trim());
      if (cells.every((c) => /^:?-+:?$/.test(c))) continue;
      html += `<div class="md-row">${cells.map((c) => `<span>${inline(c)}</span>`).join('')}</div>`;
      continue;
    }
    let m;
    if ((m = /^\s*[-*•]\s+(.*)$/.exec(line))) { flushPara(); if (!list || list.tag !== 'ul') { flushList(); list = { tag: 'ul', items: [] }; } list.items.push(m[1]); continue; }
    if ((m = /^\s*\d+[.)]\s+(.*)$/.exec(line))) { flushPara(); if (!list || list.tag !== 'ol') { flushList(); list = { tag: 'ol', items: [] }; } list.items.push(m[1]); continue; }
    if ((m = /^#{1,4}\s+(.*)$/.exec(line))) { flushPara(); flushList(); html += `<p class="md-h">${inline(m[1])}</p>`; continue; }
    if (!line.trim()) { flushPara(); flushList(); continue; }
    flushList();
    para.push(line.trim());
  }
  if (inCode) html += `<pre><code>${esc(code.join('\n'))}</code></pre>`;
  flushPara(); flushList();
  return html;
}

export class ChatView {
  constructor(root, { onRestoreVersion, onIssueClick, onImageClick }) {
    this.root = root;
    this.onRestoreVersion = onRestoreVersion;
    this.onIssueClick = onIssueClick;
    this.onImageClick = onImageClick;
    this.cards = new Map();
    this.clear();
  }

  clear() {
    this.root.innerHTML = '';
    this.cards.clear();
    this.empty = document.createElement('div');
    this.empty.className = 'chat-empty';
    this.empty.innerHTML = `<div class="chat-empty-title">Describe the toolpath you want</div>
      <div class="chat-empty-sub">Claude writes the G-code into the editor, runs the simulation, reads the report, looks at the result and fixes its own mistakes.</div>`;
    this.root.appendChild(this.empty);
  }

  scroll() {
    const el = this.root;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 160 || this.stick) el.scrollTop = el.scrollHeight;
  }

  setBusy(b) {
    this.root.classList.toggle('busy', b);
    this.onBusy?.(b);
    this.stick = b;
  }

  setUsage(u) { this.onUsage?.(u); }

  addUser(text, images, documents = []) {
    this.empty?.remove();
    const el = document.createElement('div');
    el.className = 'msg user';
    el.innerHTML = `<div class="bubble">${esc(text).replace(/\n/g, '<br>')}</div>`;
    for (const d of documents) {
      const c = document.createElement('span');
      c.className = 'msg-doc';
      c.textContent = `PDF · ${d.name}`;
      el.appendChild(c);
    }
    for (const img of images || []) {
      const i = document.createElement('img');
      i.src = `data:${img.mediaType};base64,${img.data}`;
      i.className = 'msg-thumb';
      el.appendChild(i);
    }
    this.root.appendChild(el);
    this.root.scrollTop = this.root.scrollHeight;
  }

  notice(text, level = 'info', kind) {
    const el = document.createElement('div');
    el.className = `msg notice ${level}`;
    el.innerHTML = esc(text);
    if (kind === 'auth') {
      const b = document.createElement('button');
      b.className = 'btn small';
      b.textContent = 'Open settings';
      b.onclick = () => this.onOpenSettings?.();
      el.appendChild(b);
    }
    this.root.appendChild(el);
    this.scroll();
  }

  startAssistant() {
    this.empty?.remove();
    const el = document.createElement('div');
    el.className = 'msg assistant streaming';
    const head = document.createElement('div');
    head.className = 'msg-head';
    head.innerHTML = '<span class="dot"></span><span class="who">Claude</span><span class="model"></span>';
    el.appendChild(head);
    const body = document.createElement('div');
    body.className = 'msg-body';
    el.appendChild(body);
    this.root.appendChild(el);
    this.scroll();
    const self = this;
    return {
      setModel(m) { head.querySelector('.model').textContent = m.replace(/^claude-/, ''); },
      thinking() {
        const d = document.createElement('details');
        d.className = 'thinking';
        d.innerHTML = '<summary><span class="spin"></span>Thinking</summary><div class="thinking-text"></div>';
        body.appendChild(d);
        const t = d.querySelector('.thinking-text');
        let raw = '';
        return {
          append(s) { raw += s; t.textContent = raw; self.scroll(); },
          finish() { d.querySelector('summary').innerHTML = raw.trim() ? 'Thought process' : 'Thought'; if (!raw.trim()) d.classList.add('empty'); },
        };
      },
      text() {
        const d = document.createElement('div');
        d.className = 'md';
        body.appendChild(d);
        let raw = '';
        let pending = false;
        return {
          append(s) {
            raw += s;
            if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; d.innerHTML = renderMarkdown(raw); self.scroll(); }); }
          },
          finish() { d.innerHTML = renderMarkdown(raw); if (!raw.trim()) d.remove(); self.scroll(); },
        };
      },
      tool(name, id) {
        const c = self.makeToolCard(name, id);
        body.appendChild(c.el);
        self.scroll();
        return c;
      },
      fallback(from, to) {
        const d = document.createElement('div');
        d.className = 'fallback-note';
        d.textContent = `Switched model${from ? ` from ${from}` : ''}${to ? ` to ${to}` : ''} (safety fallback).`;
        body.appendChild(d);
      },
      discardIfEmpty() { if (!body.children.length) el.remove(); },
      finish(usage) {
        el.classList.remove('streaming');
        if (usage) {
          const u = document.createElement('div');
          u.className = 'msg-usage';
          const cache = usage.cache_read_input_tokens ? ` · ${(usage.cache_read_input_tokens / 1000).toFixed(1)}k cached` : '';
          u.textContent = `${((usage.input_tokens || 0) / 1000).toFixed(1)}k in${cache} · ${((usage.output_tokens || 0) / 1000).toFixed(1)}k out`;
          head.appendChild(u);
        }
      },
    };
  }

  makeToolCard(name, id) {
    const el = document.createElement('div');
    el.className = `tool-card ${name}`;
    const label = name === 'write_program' ? 'Writing program' : name === 'view_simulation' ? 'Looking at the simulation' : name === 'inspect_part' ? 'Measuring the part' : name;
    el.innerHTML = `<div class="tool-top"><span class="spin"></span><span class="tool-label">${label}</span><span class="tool-title"></span><span class="tool-meta"></span></div><div class="tool-detail"></div>`;
    const meta = el.querySelector('.tool-meta');
    const detail = el.querySelector('.tool-detail');
    const card = {
      el,
      title(t) { el.querySelector('.tool-title').textContent = t ? `— ${t}` : ''; },
      progress(gcode) { meta.textContent = `${gcode.split('\n').length} lines`; },
      finish() { /* input complete; result comes from done() */ },
      done: (r) => {
        el.classList.add('done');
        el.querySelector('.spin')?.remove();
        if (r.error) { el.classList.add('error'); detail.textContent = r.error; return; }
        if (name === 'write_program') {
          el.querySelector('.tool-label').textContent = 'Wrote program';
          const s = r.summary;
          const badge = (n, cls, word) => (n ? `<span class="badge ${cls}">${n} ${word}${n > 1 ? 's' : ''}</span>` : '');
          detail.innerHTML = `<div class="tool-result">
            ${s.errors || s.warnings ? '' : '<span class="badge ok">✓ clean</span>'}
            ${badge(s.errors, 'error', 'error')}${badge(s.warnings, 'warning', 'warning')}
            <span class="muted">${s.lines} lines · ${s.time}</span>
            <button class="link restore">restore</button>
            <button class="link details">report</button></div>
            <pre class="report" hidden></pre>`;
          detail.querySelector('.report').textContent = s.report;
          detail.querySelector('.details').onclick = () => { const p = detail.querySelector('.report'); p.hidden = !p.hidden; };
          detail.querySelector('.restore').onclick = () => this.onRestoreVersion?.(s.versionId);
        } else if (name === 'inspect_part') {
          el.querySelector('.tool-label').textContent = 'Measured the part';
          el.querySelector('.tool-title').textContent = r.title ? `— ${r.title}` : '';
          detail.innerHTML = '<div class="tool-result"><button class="link details">show</button></div><pre class="report" hidden></pre>';
          detail.querySelector('.report').textContent = r.text || '';
          detail.querySelector('.details').onclick = () => { const p = detail.querySelector('.report'); p.hidden = !p.hidden; };
        } else if (name === 'view_simulation') {
          el.querySelector('.tool-label').textContent = 'Checked the simulation';
          detail.innerHTML = '';
          for (const img of r.images || []) {
            const i = document.createElement('img');
            i.src = img.dataUrl;
            i.title = img.view;
            i.onclick = () => this.onImageClick?.(img.dataUrl);
            detail.appendChild(i);
          }
        }
        this.scroll();
      },
    };
    this.cards.set(id, card);
    return card;
  }

  toolCard(id) { return this.cards.get(id); }
}
