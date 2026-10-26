// Stock (blank) definitions.
//  Mill:  (STOCK X-50:50 Y-40:40 Z-20:0)   explicit min:max per axis
//         (STOCK X100 Y80 Z20)             size; X0 Y0 at the lower-left corner, top at Z0
//         legacy headers: "%1 (X160 Y100 Z14)", "(Part size X:100 Y:100 Z:50)"
//  Lathe: (STOCK D50 L80)                  Ø50, from Z0 to Z-80 (chuck face at Z-80)
//         (STOCK D50 L80 Z1 ID20)          1 mm face stock beyond Z0, Ø20 bore (tube)

export function parseStockFromComments(comments, machine) {
  for (const c of comments) {
    const s = parseStockComment(c.text, machine);
    if (s) return { ...s, source: 'program', line: c.line };
  }
  return null;
}

function rng(re, s) {
  const m = re.exec(s);
  if (!m) return null;
  return [parseFloat(m[1]), parseFloat(m[2])];
}
function val(re, s) {
  const m = re.exec(s);
  return m ? parseFloat(m[1]) : null;
}

export function parseStockComment(text, machine) {
  // CncSimulator directive: ($SetPartSize(200, 150, 20, 1) ) = X size, Y size, Z size, origin mode
  const sp = /\$SETPARTSIZE\s*\(\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)/i.exec(text);
  if (sp) {
    const [x, y, z] = sp.slice(1, 4).map(parseFloat);
    if (machine === 'lathe') return { kind: 'lathe', diameter: x, length: z || y, faceStock: 0, innerDiameter: 0 };
    return box(0, 0, -z, x, y, 0);
  }
  const t = text.toUpperCase().replace(/,/g, '.');
  const isStock = /^(STOCK|BLANK|RÄMNE|RAMNE|ÄMNE|AMNE|WORKPIECE|PART SIZE)\b/.test(t.trim());
  const num = '(-?\\d*\\.?\\d+)';
  if (machine === 'lathe') {
    if (!isStock) return null;
    const d = val(new RegExp(`(?:^|\\s)(?:D|DIA|Ø)\\s*=?\\s*${num}`), t);
    const l = val(new RegExp(`(?:^|\\s)(?:L|LEN)\\s*=?\\s*${num}`), t);
    if (!d || !l) return null;
    const face = val(new RegExp(`(?:^|\\s)Z\\s*=?\\s*${num}`), t) ?? 0;
    const id = val(new RegExp(`(?:^|\\s)ID\\s*=?\\s*${num}`), t) ?? 0;
    return { kind: 'lathe', diameter: Math.abs(d), length: Math.abs(l), faceStock: face, innerDiameter: Math.abs(id) };
  }
  // explicit ranges
  const xr = rng(new RegExp(`X\\s*${num}\\s*(?::|\\.\\.|TO)\\s*${num}`), t);
  const yr = rng(new RegExp(`Y\\s*${num}\\s*(?::|\\.\\.|TO)\\s*${num}`), t);
  const zr = rng(new RegExp(`Z\\s*${num}\\s*(?::|\\.\\.|TO)\\s*${num}`), t);
  if (xr && yr && zr) {
    return box(Math.min(...xr), Math.min(...yr), Math.min(...zr), Math.max(...xr), Math.max(...yr), Math.max(...zr));
  }
  // sizes: "X100 Y80 Z20" / "X:100 Y:100 Z:50"
  const sx = val(new RegExp(`X\\s*[:=]?\\s*${num}`), t);
  const sy = val(new RegExp(`Y\\s*[:=]?\\s*${num}`), t);
  const sz = val(new RegExp(`Z\\s*[:=]?\\s*${num}`), t);
  const onlySizes = /^[\sXYZ:=.\d-]+$/.test(t.replace(/^(STOCK|BLANK|PART SIZE|RÄMNE|RAMNE|ÄMNE|AMNE|WORKPIECE)\b/, ''));
  if (sx && sy && sz && (isStock || onlySizes)) {
    return box(0, 0, -Math.abs(sz), Math.abs(sx), Math.abs(sy), 0);
  }
  return null;
}

function box(x0, y0, z0, x1, y1, z1) {
  return { kind: 'box', min: [x0, y0, z0], max: [x1, y1, z1] };
}

// Fit a stock to the cutting moves when the program does not define one.
export function autoStock(moves, machine, toolsByNum) {
  if (machine === 'lathe') {
    let maxR = 0, minZ = 0, maxZ = 0, any = false;
    for (const m of moves) {
      if (m.kind !== 'feed') continue;
      for (const p of m.pts) {
        any = true;
        if (p[0] > maxR) maxR = p[0];
        if (p[2] < minZ) minZ = p[2];
        if (p[2] > maxZ) maxZ = p[2];
      }
    }
    if (!any) return { kind: 'lathe', diameter: 50, length: 80, faceStock: 0, innerDiameter: 0, source: 'auto' };
    const d = Math.max(10, Math.ceil((maxR * 2 - 0.5) / 2) * 2);
    return { kind: 'lathe', diameter: d, length: Math.ceil(-minZ + 10), faceStock: Math.min(2, Math.max(0, maxZ > 0.05 ? 0 : 0)), innerDiameter: 0, source: 'auto' };
  }
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, any = false;
  for (const m of moves) {
    if (m.kind !== 'feed') continue;
    const r = (toolsByNum[m.tool]?.diameter ?? 10) / 2;
    for (const p of m.pts) {
      if (p[2] > 0.5) continue;
      any = true;
      x0 = Math.min(x0, p[0] - r); x1 = Math.max(x1, p[0] + r);
      y0 = Math.min(y0, p[1] - r); y1 = Math.max(y1, p[1] + r);
      z0 = Math.min(z0, p[2]);
    }
  }
  if (!any) return { kind: 'box', min: [-50, -50, -20], max: [50, 50, 0], source: 'auto' };
  // Facing paths overhang the part; shrink a little so the block looks like a real blank.
  const pad = 0;
  const depth = Math.max(5, Math.ceil(-z0 + 5));
  return { kind: 'box', min: [round(x0 + pad), round(y0 + pad), -depth], max: [round(x1 - pad), round(y1 - pad), 0], source: 'auto' };
}

function round(v) { return Math.round(v * 10) / 10; }

export function describeStock(s) {
  if (!s) return 'none';
  if (s.kind === 'lathe') {
    return `Ø${s.diameter} × ${s.length} mm (Z${s.faceStock ? '+' + s.faceStock : '0'} → Z-${s.length})${s.innerDiameter ? `, bore Ø${s.innerDiameter}` : ''}`;
  }
  const [a, b, c] = s.min, [d, e, f] = s.max;
  return `X${a}..${d}  Y${b}..${e}  Z${c}..${f}  (${+(d - a).toFixed(3)} × ${+(e - b).toFixed(3)} × ${+(f - c).toFixed(3)} mm)`;
}

export function stockToComment(s) {
  if (s.kind === 'lathe') return `(STOCK D${s.diameter} L${s.length}${s.faceStock ? ' Z' + s.faceStock : ''}${s.innerDiameter ? ' ID' + s.innerDiameter : ''})`;
  return `(STOCK X${s.min[0]}:${s.max[0]} Y${s.min[1]}:${s.max[1]} Z${s.min[2]}:${s.max[2]})`;
}
