// CodeMirror 5 wrapper with a G-code mode, issue gutter and execution-line highlight.
/* global CodeMirror */

let modeDefined = false;
function defineMode() {
  if (modeDefined) return;
  modeDefined = true;
  CodeMirror.defineSimpleMode('gcode', {
    start: [
      { regex: /\((?:[^()]|\([^()]*\))*\)?/, token: 'comment' },
      { regex: /;.*/, token: 'comment' },
      { regex: /%.*/, token: 'meta' },
      { regex: /^\s*\//, token: 'meta' },
      { regex: /[Nn]\d+/, token: 'blocknum' },
      { regex: /[Oo]\d+/, token: 'program' },
      { regex: /\b(?:IF|WHILE|DO|END|GOTO|THEN|EQ|NE|GT|GE|LT|LE|AND|OR|XOR|MOD|SIN|COS|TAN|ASIN|ACOS|ATAN|SQRT|ABS|ROUND|FIX|FUP|LN|EXP|POW)\b/i, token: 'macro' },
      { regex: /#\d+/, token: 'variable' },
      { regex: /[Gg]\s*0*(?:0|1)(?:\.\d)?(?!\d)/, token: 'gmotion' },
      { regex: /[Gg]\s*\d+(?:\.\d)?/, token: 'gcode' },
      { regex: /[Mm]\s*\d+/, token: 'mcode' },
      { regex: /[Tt]\s*\d+/, token: 'tool' },
      { regex: /,\s*[RCrc]\s*[-+]?\d*\.?\d+/, token: 'param' },
      { regex: /[XYZUVWABCxyzuvwabc]\s*[-+]?(?:\d*\.?\d+|(?=[#[]))/, token: 'axis' },
      { regex: /[FSfs]\s*[-+]?\d*\.?\d+/, token: 'feed' },
      { regex: /[IJKRPQDHLEijkrpqdhle]\s*[-+]?\d*\.?\d+/, token: 'param' },
      { regex: /[[\]=+*/-]/, token: 'operator' },
    ],
    meta: { lineComment: ';' },
  });
}

export class Editor {
  constructor(el, { onChange, onCursorLine, onLineClick }) {
    defineMode();
    this.cm = CodeMirror(el, {
      value: '',
      mode: 'gcode',
      lineNumbers: true,
      styleActiveLine: true,
      gutters: ['cm-issues', 'CodeMirror-linenumbers'],
      indentUnit: 2,
      tabSize: 2,
      lineWrapping: false,
      viewportMargin: 50,
      extraKeys: {
        'Ctrl-/': 'toggleComment',
        'Cmd-/': 'toggleComment',
      },
    });
    this.execLine = null;
    this.silent = false;
    this.cm.on('change', () => { if (!this.silent) onChange?.(this.value); });
    this.cm.on('cursorActivity', () => onCursorLine?.(this.cm.getCursor().line));
    this.cm.on('gutterClick', (cm, line) => onLineClick?.(line));
    this.cm.on('dblclick', () => onLineClick?.(this.cm.getCursor().line));
  }

  get value() { return this.cm.getValue(); }

  setValue(text, { silent = true, keepCursor = false } = {}) {
    const cur = this.cm.getCursor();
    const scroll = this.cm.getScrollInfo();
    this.silent = silent;
    this.cm.setValue(text);
    this.silent = false;
    if (keepCursor) { this.cm.setCursor(cur); this.cm.scrollTo(scroll.left, scroll.top); }
    this.cm.clearHistory();
  }

  // streaming: replace the whole document with `text` cheaply by appending the new suffix
  streamTo(text) {
    const doc = this.cm.getDoc();
    const cur = doc.getValue();
    this.silent = true;
    if (text.startsWith(cur)) {
      const add = text.slice(cur.length);
      if (add) doc.replaceRange(add, CodeMirror.Pos(doc.lastLine()));
    } else {
      doc.setValue(text);
    }
    this.silent = false;
    this.cm.scrollIntoView(CodeMirror.Pos(doc.lastLine()), 60);
  }

  setStreaming(on) {
    this.cm.setOption('readOnly', on ? 'nocursor' : false);
    this.cm.getWrapperElement().classList.toggle('streaming', on);
  }

  setIssues(messages) {
    this.cm.clearGutter('cm-issues');
    for (const h of this.issueLines || []) this.cm.removeLineClass(h, 'background', 'cm-issue-error');
    this.issueLines = [];
    const byLine = new Map();
    for (const m of messages) {
      if (m.callLine === null || m.callLine === undefined) continue;
      const prev = byLine.get(m.callLine);
      const rank = { error: 3, warning: 2, info: 1 }[m.level] || 0;
      if (!prev || rank > prev.rank) byLine.set(m.callLine, { rank, level: m.level, texts: [...(prev?.texts || []), m.text] });
      else prev.texts.push(m.text);
    }
    for (const [line, info] of byLine) {
      if (line >= this.cm.lineCount()) continue;
      const el = document.createElement('div');
      el.className = `cm-issue-marker ${info.level}`;
      el.title = info.texts.join('\n');
      el.textContent = info.level === 'error' ? '●' : info.level === 'warning' ? '▲' : '·';
      this.cm.setGutterMarker(line, 'cm-issues', el);
      if (info.level === 'error') {
        const h = this.cm.addLineClass(line, 'background', 'cm-issue-error');
        this.issueLines.push(h);
      }
    }
  }

  setExecLine(line, scroll = true) {
    if (this.execLine === line) return;
    if (this.execHandle) this.cm.removeLineClass(this.execHandle, 'background', 'cm-exec-line');
    this.execHandle = null;
    this.execLine = line;
    if (line === null || line === undefined || line < 0 || line >= this.cm.lineCount()) return;
    this.execHandle = this.cm.addLineClass(line, 'background', 'cm-exec-line');
    if (scroll) {
      const info = this.cm.getScrollInfo();
      const top = this.cm.heightAtLine(line, 'local');
      if (top < info.top + 20 || top > info.top + info.clientHeight - 40) this.cm.scrollTo(null, top - info.clientHeight / 2);
    }
  }

  goToLine(line) {
    this.cm.setCursor({ line, ch: 0 });
    this.cm.scrollIntoView({ line, ch: 0 }, 120);
    this.cm.focus();
  }

  selection() {
    const sel = this.cm.listSelections()[0];
    if (!sel) return null;
    const a = Math.min(sel.anchor.line, sel.head.line), b = Math.max(sel.anchor.line, sel.head.line);
    const text = this.cm.getSelection();
    if (!text.trim()) return null;
    return { from: a, to: b, text };
  }

  refresh() { this.cm.refresh(); }
}
