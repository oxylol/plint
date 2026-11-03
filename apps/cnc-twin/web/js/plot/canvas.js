/* Canvas helpers: device-pixel sizing, size observation, rounded bar ends, text. */

export function dpr() {
  return Math.max(1, Math.min(3, (typeof devicePixelRatio === 'number' && devicePixelRatio) || 1));
}

/** Size a canvas to CSS width x height at the device pixel ratio; returns a context scaled to CSS px. */
export function fitCanvas(canvas, width, height) {
  const ratio = dpr();
  const w = Math.max(1, Math.round(width));
  const h = Math.max(1, Math.round(height));
  if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
  }
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: w, height: h, ratio };
}

/** callback(width) whenever the element's content width changes (ResizeObserver, else window resize). */
export function observeWidth(element, callback) {
  let last = -1;
  const check = () => {
    const width = Math.floor(element.clientWidth);
    if (width > 0 && width !== last) { last = width; callback(width); }
  };
  if (typeof ResizeObserver === 'function') {
    const observer = new ResizeObserver(() => check());
    observer.observe(element);
    requestAnimationFrame(check);
    return () => observer.disconnect();
  }
  window.addEventListener('resize', check);
  requestAnimationFrame(check);
  return () => window.removeEventListener('resize', check);
}

export function setFont(ctx, size, weight = 500, family = 'system-ui, sans-serif') {
  ctx.font = `${weight} ${size}px ${family}`;
}

/** A bar with a rounded data end (radius r) and a square base; `toward` is the data end side. */
export function bar(ctx, x, y, w, h, r, toward = 'right') {
  const rr = Math.max(0, Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2));
  if (w < 0) { x += w; w = -w; }
  if (h < 0) { y += h; h = -h; }
  ctx.beginPath();
  if (toward === 'right') {
    ctx.moveTo(x, y); ctx.lineTo(x + w - rr, y); ctx.arcTo(x + w, y, x + w, y + rr, rr);
    ctx.lineTo(x + w, y + h - rr); ctx.arcTo(x + w, y + h, x + w - rr, y + h, rr); ctx.lineTo(x, y + h);
  } else if (toward === 'left') {
    ctx.moveTo(x + w, y); ctx.lineTo(x + rr, y); ctx.arcTo(x, y, x, y + rr, rr);
    ctx.lineTo(x, y + h - rr); ctx.arcTo(x, y + h, x + rr, y + h, rr); ctx.lineTo(x + w, y + h);
  } else {
    ctx.moveTo(x, y + h); ctx.lineTo(x, y + rr); ctx.arcTo(x, y, x + rr, y, rr);
    ctx.lineTo(x + w - rr, y); ctx.arcTo(x + w, y, x + w, y + rr, rr); ctx.lineTo(x + w, y + h);
  }
  ctx.closePath();
  ctx.fill();
}

export function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

/** Text that fits `max` px: shortened with an ellipsis. */
export function fitText(ctx, text, max) {
  if (ctx.measureText(text).width <= max) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ctx.measureText(`${text.slice(0, mid)}…`).width <= max) lo = mid; else hi = mid - 1;
  }
  return lo > 0 ? `${text.slice(0, lo)}…` : '';
}
