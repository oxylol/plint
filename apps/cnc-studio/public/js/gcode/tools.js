import { buildLatheGeometry, describeLatheGeometry } from './lathetools.js';

// Tool library + parsing tool definitions out of program comments.
// Understands CAM-style headers such as
//   (T6 D=10. CR=0. - FLAT END MILL)      (T13 D=23.5 TAPER=118DEG - DRILL)
//   (T1 CNMT. CR=0.8)                      (T2 OD GROOVING. CR=0.8)
// and the explicit form Claude is asked to write:
//   (TOOL T1 FLAT D10 L30 "10mm end mill")  (TOOL T3 GROOVE W3 R0.2)

export const TOOL_COLORS = [
  '#4cc9f0', '#f72585', '#7bd389', '#ffb703', '#b388ff', '#ff7b54',
  '#3fd0c9', '#ef476f', '#a3e635', '#f4a261', '#90caf9', '#e879f9',
];

export function toolColor(num) {
  return TOOL_COLORS[(Math.abs(num | 0) + TOOL_COLORS.length - 1) % TOOL_COLORS.length];
}

export const MILL_TYPES = ['flat', 'bull', 'ball', 'drill', 'spot', 'chamfer', 'tap', 'threadmill', 'face', 'engrave'];
export const LATHE_TYPES = ['turn', 'bore', 'groove', 'thread', 'drill'];

export function defaultLibrary(machine) {
  if (machine === 'lathe') {
    return {
      1: { num: 1, type: 'turn', noseRadius: 0.8, insertAngle: 80, name: 'CNMG 80° rough R0.8' },
      2: { num: 2, type: 'turn', noseRadius: 0.4, insertAngle: 35, name: 'VNMG 35° finish R0.4' },
      3: { num: 3, type: 'groove', width: 3, noseRadius: 0.2, name: 'Groove 3 mm' },
      4: { num: 4, type: 'thread', tipAngle: 60, noseRadius: 0.05, name: 'Thread 60°' },
      5: { num: 5, type: 'drill', diameter: 10, tipAngle: 118, name: 'Drill Ø10' },
      6: { num: 6, type: 'bore', noseRadius: 0.4, insertAngle: 55, name: 'Boring bar R0.4' },
    };
  }
  return {
    1: { num: 1, type: 'flat', diameter: 10, fluteLength: 30, name: 'End mill Ø10' },
    2: { num: 2, type: 'flat', diameter: 6, fluteLength: 20, name: 'End mill Ø6' },
    3: { num: 3, type: 'ball', diameter: 6, fluteLength: 20, name: 'Ball mill Ø6' },
    4: { num: 4, type: 'drill', diameter: 6.8, tipAngle: 118, fluteLength: 40, name: 'Drill Ø6.8' },
    5: { num: 5, type: 'spot', diameter: 10, tipAngle: 90, fluteLength: 10, name: 'Spot drill Ø10 90°' },
    6: { num: 6, type: 'chamfer', diameter: 10, tipAngle: 90, tipDiameter: 0.5, fluteLength: 6, name: 'Chamfer mill 90°' },
    7: { num: 7, type: 'tap', diameter: 8, pitch: 1.25, fluteLength: 25, name: 'Tap M8×1.25' },
    8: { num: 8, type: 'face', diameter: 50, fluteLength: 6, name: 'Face mill Ø50' },
    9: { num: 9, type: 'flat', diameter: 3, fluteLength: 10, name: 'End mill Ø3' },
  };
}

function num(re, s) {
  const m = re.exec(s);
  return m ? parseFloat(m[1]) : undefined;
}

// Parse one comment. Returns a partial tool or null.
export function parseToolComment(text, machine) {
  const t = text.toUpperCase().trim();
  let m = /^(?:TOOL\s+)?T\s*0*(\d{1,4})\b(.*)$/.exec(t);
  if (!m) return null;
  let n = parseInt(m[1], 10);
  if (machine === 'lathe' && m[1].length >= 3) n = Math.floor(n / 100) || n;
  const rest = m[2];
  // a plain "(T1)" comment is not a tool definition
  if (!rest.trim()) return null;
  const explicit = /^TOOL\s/.test(t);
  const nameMatch = /"([^"]*)"/.exec(text);
  const tool = { num: n, name: nameMatch ? nameMatch[1] : text.replace(/^\s*(TOOL\s+)?T\s*\d+\s*[-:]?\s*/i, '').trim() };

  const d = num(/(?:^|[\s,])(?:D|DIA|Ø)\s*=?\s*(\d*\.?\d+)/, rest);
  const cr = num(/\b(?:CR|R|RE|NR)\s*=?\s*(\d*\.?\d+)/, rest);
  const taper = num(/\b(?:TAPER|ANGLE|A|TA)\s*=?\s*(\d*\.?\d+)\s*(?:DEG|°)?/, rest);
  const len = num(/\b(?:L|FL|LEN)\s*=?\s*(\d*\.?\d+)/, rest);
  const w = num(/\b(?:W|WIDTH)\s*=?\s*(\d*\.?\d+)/, rest);
  const pitchM = /M(\d+(?:\.\d+)?)\s*[X×]\s*(\d*\.?\d+)/.exec(rest);

  if (machine === 'lathe') {
    let type = 'turn';
    if (/GROOV|SPÅR|SPAR|PART|CUT-?OFF|AVSTICK/.test(rest)) type = 'groove';
    else if (/THREAD|GÄNG|GANG/.test(rest)) type = 'thread';
    else if (/BORING|BORE\b|INNER|\bID\b|URSVARV|INVÄNDIG/.test(rest)) type = 'bore';
    else if (/DRILL|BORR\b|\bBORR|CENTER/.test(rest)) type = 'drill';
    for (const k of LATHE_TYPES) if (explicit && new RegExp(`\\b${k.toUpperCase()}\\b`).test(rest)) type = k;
    tool.type = type;
    if (cr !== undefined) tool.noseRadius = cr;
    if (w !== undefined) tool.width = w;
    if (d !== undefined && type === 'drill') tool.diameter = d;
    if (type === 'drill' && len !== undefined) tool.length = len;
    if (taper !== undefined) {
      if (type === 'turn' || type === 'bore') tool.insertAngle = taper;
      else tool.tipAngle = taper;
    }
    if (/\b(RIGHT|RH|HÖGER)\b/.test(rest)) tool.hand = 'right';
    const h = num(/\b(?:H|HEIGHT)\s*=?\s*(\d*\.?\d+)/, rest);
    if (h !== undefined && type === 'groove') tool.height = h;
    const ref = num(/\b(?:REF|ORIGIN)\s*=?\s*([1-8])\b/, rest);
    if (ref !== undefined) tool.origin = ref;
    const bar = num(/\b(?:BAR|SHANK)\s*=?\s*(\d*\.?\d+)/, rest);
    if (bar !== undefined && type === 'bore') tool.barDiameter = bar;
    if (tool.insertAngle === undefined) {
      if (/\bV[NB]M|\b35\b/.test(rest)) tool.insertAngle = 35;
      else if (/\bD[NC]M|\b55\b/.test(rest)) tool.insertAngle = 55;
      else if (/\bC[NC]M|\b80\b/.test(rest)) tool.insertAngle = 80;
      else if (/\bT[NC]M|\b60\b/.test(rest)) tool.insertAngle = 60;
      else if (/\bS[NC]M/.test(rest)) tool.insertAngle = 90;
    }
    return tool;
  }

  let type = 'flat';
  if (/BALL|KULF|SPHER/.test(rest)) type = 'ball';
  else if (/SPOT|CENTER|NC ?DRILL|PUNKT/.test(rest)) type = 'spot';
  else if (/CHAMFER|FAS|COUNTERSINK|VSINK/.test(rest)) type = 'chamfer';
  else if (/THREAD ?MILL|GÄNGFRÄS/.test(rest)) type = 'threadmill';
  else if (/\bTAP\b|TAP\b|GÄNGTAPP|GANGTAPP/.test(rest)) type = 'tap';
  else if (/DRILL|BORR|\bBOR\b/.test(rest)) type = 'drill';
  else if (/FACE|PLAN|SHELL/.test(rest)) type = 'face';
  else if (/ENGRAV|GRAVER/.test(rest)) type = 'engrave';
  else if (/BULL/.test(rest) || (cr !== undefined && cr > 0)) type = 'bull';
  for (const k of MILL_TYPES) if (explicit && new RegExp(`\\b${k.toUpperCase()}\\b`).test(rest)) type = k;
  tool.type = type;
  if (d !== undefined) tool.diameter = d;
  else if (pitchM) tool.diameter = parseFloat(pitchM[1]);
  if (pitchM) tool.pitch = parseFloat(pitchM[2]);
  if (cr !== undefined) tool.cornerRadius = cr;
  if (taper !== undefined) {
    // CAM posts give a chamfer mill's TAPER as the half angle (90° tool → TAPER=45DEG)
    tool.tipAngle = type === 'chamfer' && /TAPER/.test(rest) && taper <= 60 ? taper * 2 : taper;
  }
  if (len !== undefined) tool.fluteLength = len;
  const sl = num(/\b(?:SL|STICKOUT|OAL)\s*=?\s*(\d*\.?\d+)/, rest);
  if (sl !== undefined) tool.overallLength = sl;
  const hd = num(/\b(?:HD|HOLDER)\s*=?\s*(\d*\.?\d+)/, rest);
  if (hd !== undefined) tool.holderDiameter = hd;
  if (type === 'ball' && tool.diameter) tool.cornerRadius = tool.diameter / 2;
  return tool;
}

export function parseToolsFromComments(comments, machine) {
  const tools = {};
  for (const c of comments) {
    const t = parseToolComment(c.text, machine);
    if (t && (t.diameter || t.noseRadius !== undefined || t.width || t.type !== (machine === 'lathe' ? 'turn' : 'flat') || /TOOL|D=|CR=|DIA/i.test(c.text))) {
      tools[t.num] = { ...(tools[t.num] || {}), ...t, source: 'program', line: c.line };
    }
  }
  return tools;
}

// Fill in defaults so every consumer can rely on the fields it needs.
export function normalizeTool(t, machine) {
  const tool = { ...t };
  tool.color = tool.color || toolColor(tool.num);
  if (machine === 'lathe') {
    tool.type = LATHE_TYPES.includes(tool.type) ? tool.type : 'turn';
    if (tool.noseRadius === undefined) tool.noseRadius = tool.type === 'groove' ? 0.2 : tool.type === 'thread' ? 0.05 : 0.4;
    if (tool.type === 'groove' && !tool.width) tool.width = 3;
    if (tool.type === 'drill') { tool.diameter = tool.diameter || 10; tool.tipAngle = tool.tipAngle || 118; }
    if (tool.type === 'thread') tool.tipAngle = tool.tipAngle || 60;
    tool.insertAngle = tool.insertAngle || (tool.type === 'bore' ? 55 : 80);
    tool.name = tool.name || `T${tool.num}`;
    // a library edit that changes the type/size must regenerate built-in geometry
    if (tool.generated) { delete tool.cutter; delete tool.holder; delete tool.noseOffset; }
    buildLatheGeometry(tool);
    return tool;
  }
  tool.type = MILL_TYPES.includes(tool.type) ? tool.type : 'flat';
  tool.diameter = tool.diameter > 0 ? tool.diameter : 10;
  if (tool.type === 'ball') tool.cornerRadius = tool.diameter / 2;
  tool.cornerRadius = Math.min(tool.cornerRadius || 0, tool.diameter / 2);
  if (['drill', 'spot'].includes(tool.type)) tool.tipAngle = tool.tipAngle || (tool.type === 'spot' ? 90 : 118);
  if (tool.type === 'chamfer') { tool.tipAngle = tool.tipAngle || 90; tool.tipDiameter = tool.tipDiameter ?? 0; }
  if (tool.type === 'engrave') { tool.tipAngle = tool.tipAngle || 60; tool.tipDiameter = tool.tipDiameter ?? 0.2; }
  if (!tool.fluteLength) {
    const D = tool.diameter;
    tool.fluteLength = {
      drill: Math.max(25, D * 8), tap: Math.max(15, D * 3), threadmill: Math.max(15, D * 2),
      spot: Math.max(8, D), chamfer: Math.max(8, D), engrave: 10, face: 10,
    }[tool.type] || Math.max(12, D * 3);
  }
  // stick-out = tip to holder face; shank between flutes and holder
  tool.shankDiameter = tool.shankDiameter || (tool.type === 'face' ? Math.min(tool.diameter * 0.45, 32) : Math.max(3, Math.min(tool.diameter, 32)));
  tool.overallLength = Math.max(tool.overallLength || tool.fluteLength + Math.max(15, tool.diameter * 1.5), tool.fluteLength + 2);
  tool.holderDiameter = tool.holderDiameter || Math.max(32, tool.shankDiameter + 18);
  tool.name = tool.name || `T${tool.num}`;
  return tool;
}

export function describeTool(t, machine, { detail = false } = {}) {
  if (machine === 'lathe') {
    const parts = [`T${t.num}`, t.type];
    if (t.type === 'groove') parts.push(`W${t.width}`);
    if (t.type === 'drill') parts.push(`Ø${t.diameter}`);
    if (t.type !== 'drill') parts.push(`R${t.noseRadius}`);
    if (t.name) parts.push(`"${t.name}"`);
    if (detail) {
      const g = describeLatheGeometry(t);
      if (g) parts.push(`— ${g}`);
      if (t.feed || t.speed) parts.push(`(register cutting data: f${t.feed} Vc${t.speed})`);
    }
    return parts.join(' ');
  }
  const parts = [`T${t.num}`, t.type, `Ø${t.diameter}`];
  if (t.cornerRadius && t.type === 'bull') parts.push(`CR${t.cornerRadius}`);
  if (t.tipAngle && ['drill', 'spot', 'chamfer', 'engrave'].includes(t.type)) parts.push(`${t.tipAngle}°`);
  if (t.fluteLength) parts.push(`L${t.fluteLength}`);
  if (t.name) parts.push(`"${t.name}"`);
  if (detail) {
    parts.push(`— flutes ${t.fluteLength} mm, stick-out ${t.overallLength} mm (max depth before the Ø${t.holderDiameter} holder hits the stock)`);
    if (t.feed || t.speed) parts.push(`(register: F${t.feed ?? '–'} S${t.speed ?? '–'})`);
  }
  return parts.join(' ');
}

// Height of the tool's cutting surface above its tip at radial distance d (mill).
export function millProfile(tool) {
  const r = tool.diameter / 2;
  switch (tool.type) {
    case 'ball':
      return (d) => r - Math.sqrt(Math.max(0, r * r - d * d));
    case 'bull': {
      const cr = tool.cornerRadius;
      const flat = r - cr;
      return (d) => (d <= flat ? 0 : cr - Math.sqrt(Math.max(0, cr * cr - (d - flat) * (d - flat))));
    }
    case 'drill':
    case 'spot': {
      const k = 1 / Math.tan((tool.tipAngle * Math.PI / 180) / 2);
      return (d) => d * k;
    }
    case 'chamfer':
    case 'engrave': {
      const k = 1 / Math.tan((tool.tipAngle * Math.PI / 180) / 2);
      const t0 = (tool.tipDiameter || 0) / 2;
      return (d) => Math.max(0, d - t0) * k;
    }
    default:
      return () => 0;
  }
}
