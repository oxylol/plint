/* The HTML layer over the 3D canvas: the layer's colour bar (or categories), the measuring toolbar and
 * readout, measurement labels at their 3-D points, the deflection exaggeration control, and a status
 * line with what the view could not draw and why. Plint's tokens (--ink, --bg, --edge, --accent,
 * --font, --mono) with the same fallbacks as web/css/tokens.css; 44 px targets on coarse pointers;
 * the overlay passes pointer events through except on its controls. */

const STYLE_ID = 'cnctwin-view3d-style';
const CSS = `
.v3d-root { position: absolute; inset: 0; pointer-events: none; font-family: var(--font, system-ui, sans-serif); color: var(--ink, #22302a); font-size: 12px; line-height: 1.35; }
.v3d-panel { position: absolute; background: var(--bg, #e4e9e5); background: color-mix(in srgb, var(--bg, #e4e9e5) 86%, transparent); border: 1px solid var(--edge, rgba(34,48,42,0.14)); border-radius: 10px; padding: 6px 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.12); max-width: calc(100% - 16px); }
.v3d-legend { left: 8px; top: 8px; min-width: 150px; }
.v3d-legend canvas { display: block; width: 160px; height: 10px; border-radius: 3px; margin: 4px 0 2px; }
.v3d-legend .v3d-ends { display: flex; justify-content: space-between; gap: 8px; font-family: var(--mono, ui-monospace, monospace); font-variant-numeric: tabular-nums; }
.v3d-legend .v3d-cat { display: flex; align-items: center; gap: 6px; margin-top: 3px; }
.v3d-legend .v3d-swatch { width: 12px; height: 12px; border-radius: 3px; border: 1px solid var(--edge, rgba(34,48,42,0.2)); }
.v3d-title { font-weight: 600; }
.v3d-muted { color: var(--ink2, #56655d); }
.v3d-measure { right: 8px; top: 8px; pointer-events: auto; display: flex; flex-direction: column; gap: 6px; align-items: flex-end; }
.v3d-buttons { display: flex; gap: 4px; flex-wrap: wrap; justify-content: flex-end; }
.v3d-btn { font: inherit; color: inherit; background: var(--bg, #e4e9e5); border: 1px solid var(--edge, rgba(34,48,42,0.2)); border-radius: 8px; padding: 4px 8px; cursor: pointer; min-height: 28px; }
.v3d-btn[aria-pressed="true"] { background: var(--accent, #3f6a53); color: var(--bg, #fff); border-color: transparent; }
.v3d-readout { font-family: var(--mono, ui-monospace, monospace); font-variant-numeric: tabular-nums; }
.v3d-defl { left: 8px; bottom: 8px; pointer-events: auto; display: flex; align-items: center; gap: 6px; }
.v3d-status { right: 8px; bottom: 8px; font-size: 11px; text-align: right; pointer-events: auto; max-width: min(360px, calc(100% - 16px)); }
.v3d-status summary { cursor: pointer; list-style: none; }
.v3d-status summary::-webkit-details-marker { display: none; }
.v3d-status ul { margin: 4px 0 0; padding-left: 16px; text-align: left; }
.v3d-label { position: absolute; transform: translate(-50%, -130%); background: var(--ink, #22302a); color: var(--bg, #e4e9e5); padding: 2px 6px; border-radius: 6px; font-family: var(--mono, ui-monospace, monospace); font-size: 12px; white-space: nowrap; }
@media (pointer: coarse) { .v3d-btn { min-height: 44px; min-width: 44px; padding: 6px 10px; } }
`;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export class Overlay {
  constructor(container, { modes, onMode, onClear, onExaggeration }) {
    if (!document.getElementById(STYLE_ID)) {
      const style = document.createElement('style');
      style.id = STYLE_ID;
      style.textContent = CSS;
      document.head.append(style);
    }
    this.root = el('div', 'v3d-root');
    container.append(this.root);
    this.labelLayer = el('div');
    this.root.append(this.labelLayer);
    this.legendBox = el('div', 'v3d-panel v3d-legend');
    this.legendBox.hidden = true;
    this.root.append(this.legendBox);
    this.measureBox = el('div', 'v3d-panel v3d-measure');
    this.measureBox.hidden = true;
    const buttons = el('div', 'v3d-buttons');
    this.modeButtons = new Map();
    for (const mode of modes) {
      const button = el('button', 'v3d-btn', mode.label);
      button.type = 'button';
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', () => onMode(mode.id));
      buttons.append(button);
      this.modeButtons.set(mode.id, button);
    }
    const clear = el('button', 'v3d-btn', 'Clear');
    clear.type = 'button';
    clear.addEventListener('click', () => onClear());
    buttons.append(clear);
    this.measureText = el('div', 'v3d-readout');
    this.measureBox.append(buttons, this.measureText);
    this.root.append(this.measureBox);
    this.deflBox = el('div', 'v3d-panel v3d-defl');
    this.deflBox.hidden = true;
    const less = el('button', 'v3d-btn', '−');
    less.type = 'button';
    less.setAttribute('aria-label', 'Less exaggeration');
    less.addEventListener('click', () => onExaggeration(-1));
    const more = el('button', 'v3d-btn', '+');
    more.type = 'button';
    more.setAttribute('aria-label', 'More exaggeration');
    more.addEventListener('click', () => onExaggeration(1));
    this.deflText = el('span', 'v3d-readout');
    this.deflBox.append(el('span', 'v3d-muted', 'Deflection'), less, this.deflText, more);
    this.root.append(this.deflBox);
    this.statusBox = el('details', 'v3d-panel v3d-status');
    this.statusSummary = el('summary', 'v3d-muted');
    this.statusList = el('ul', 'v3d-muted');
    this.statusBox.append(this.statusSummary, this.statusList);
    this.root.append(this.statusBox);
    this.labels = [];
  }

  /** Colour bar: {title, min, max, unit, table (RGB bytes x 256)} or {title, categories: [[css, label]]}, or null. */
  legend(spec) {
    this.legendBox.replaceChildren();
    if (!spec) { this.legendBox.hidden = true; return; }
    this.legendBox.hidden = false;
    this.legendBox.append(el('div', 'v3d-title', spec.title));
    if (spec.categories) {
      for (const [colour, label] of spec.categories) {
        const row = el('div', 'v3d-cat');
        const swatch = el('span', 'v3d-swatch');
        swatch.style.background = colour;
        row.append(swatch, el('span', '', label));
        this.legendBox.append(row);
      }
      return;
    }
    if (spec.pending) { this.legendBox.append(el('div', 'v3d-muted', spec.pending)); return; }
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 8;
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(256, 8);
    for (let x = 0; x < 256; x++) {
      for (let y = 0; y < 8; y++) {
        const o = (y * 256 + x) * 4;
        image.data[o] = spec.table[x * 3];
        image.data[o + 1] = spec.table[x * 3 + 1];
        image.data[o + 2] = spec.table[x * 3 + 2];
        image.data[o + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', `${spec.title} colour scale from ${spec.minText} to ${spec.maxText}`);
    const ends = el('div', 'v3d-ends');
    ends.append(el('span', '', spec.minText), el('span', '', spec.maxText));
    this.legendBox.append(canvas, ends);
    if (spec.note) this.legendBox.append(el('div', 'v3d-muted', spec.note));
  }

  measure(on, mode, text) {
    this.measureBox.hidden = !on;
    for (const [id, button] of this.modeButtons) button.setAttribute('aria-pressed', String(id === mode));
    this.measureText.textContent = text || '';
  }

  deflection(visible, factorText, valueText) {
    const text = valueText ? `×${factorText} · ${valueText}` : `×${factorText}`;
    if (this.deflBox.hidden === !visible && this.deflText.textContent === text) return;
    this.deflBox.hidden = !visible;
    this.deflText.textContent = text;
  }

  status(summary, notes) {
    const key = `${summary}\n${notes.join('\n')}`;
    if (key === this.statusKey) return;
    this.statusKey = key;
    this.statusSummary.textContent = notes.length ? `${summary} · notes (${notes.length})` : summary;
    this.statusList.replaceChildren(...notes.map((n) => el('li', '', n)));
    this.statusBox.hidden = !summary && !notes.length;
  }

  /** Labels at screen positions [{x, y, text}] (CSS pixels in the container). */
  place(labels) {
    while (this.labels.length < labels.length) {
      const node = el('div', 'v3d-label');
      this.labelLayer.append(node);
      this.labels.push(node);
    }
    for (let i = 0; i < this.labels.length; i++) {
      const node = this.labels[i];
      const label = labels[i];
      if (!label) { node.hidden = true; continue; }
      node.hidden = false;
      node.textContent = label.text;
      node.style.left = `${label.x}px`;
      node.style.top = `${label.y}px`;
    }
  }

  destroy() { this.root.remove(); }
}
