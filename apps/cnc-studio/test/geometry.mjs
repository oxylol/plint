// Geometry checks with known answers. node test/geometry.mjs
import { interpret } from '../public/js/gcode/interpreter.js';

let fails = 0;
const ok = (cond, msg) => { if (!cond) { fails++; console.log('FAIL', msg); } else console.log('ok  ', msg); };
const near = (a, b, t = 1e-3) => Math.abs(a - b) <= t;
const feedPts = (res) => res.moves.filter((m) => m.kind === 'feed').flatMap((m) => m.pts);
const errs = (res) => res.messages.filter((m) => m.level === 'error').map((m) => `L${m.line + 1} ${m.text}`);

// 1) G41 around a 40x40 square, tool Ø10 -> centre path is a 50x50 square with R5 corners
{
  const src = `(TOOL T1 FLAT D10)
G90 G17 G21 G40 G80 G54
T1 M6
S3000 M3
G0 X-30 Y-30 Z5
G1 Z-2 F500
G41 D1 G1 X-20 Y-20
Y20
X20
Y-20
X-20
G40 G1 X-30 Y-30
G0 Z5
M30`;
  const res = interpret(src, { machine: 'mill' });
  ok(errs(res).length === 0, 'G41 square: no errors ' + errs(res).join('; '));
  // contour moves only (lines 8..11), not the lead-in / lead-out
  const pts = res.moves.filter((m) => m.kind === 'feed' && m.line >= 7 && m.line <= 10).flatMap((m) => m.pts);
  const maxX = Math.max(...pts.map((p) => p[0])), minX = Math.min(...pts.map((p) => p[0]));
  const maxY = Math.max(...pts.map((p) => p[1]));
  ok(near(maxX, 25, 0.01) && near(minX, -25, 0.01) && near(maxY, 25, 0.01), `G41 square: centre path spans ±25 (got X ${minX.toFixed(3)}..${maxX.toFixed(3)}, Ymax ${maxY.toFixed(3)})`);
  // every point on the straight sides must be exactly 5 from the 40x40 square (outside)
  let worst = 0;
  for (const p of pts) {
    const dx = Math.max(Math.abs(p[0]) - 20, 0), dy = Math.max(Math.abs(p[1]) - 20, 0);
    const d = Math.hypot(dx, dy);
    if (Math.abs(p[0]) <= 25.01 && Math.abs(p[1]) <= 25.01 && (Math.abs(p[0]) > 20 || Math.abs(p[1]) > 20)) worst = Math.max(worst, Math.abs(d - 5));
  }
  ok(worst < 0.02, `G41 square: offset distance 5.000 everywhere on the contour (max error ${worst.toFixed(4)})`);
}

// 2) G42 inside a pocket: inner corners must be trimmed (no arcs), tool stays 5 from walls
{
  const src = `(TOOL T1 FLAT D10)
T1 M6
S3000 M3
G0 X0 Y0 Z5
G1 Z-2 F500
G42 D1 G1 X-20 Y-20
Y20
X20
Y-20
X-20
G40 G1 X0 Y0
M30`;
  const res = interpret(src, { machine: 'mill' });
  const pts = feedPts(res).filter((p) => p[2] < -1);
  const maxX = Math.max(...pts.map((p) => p[0]));
  ok(near(maxX, 15, 0.01), `G42 pocket: inner offset at X15 (got ${maxX.toFixed(3)})`);
  const corner = pts.some((p) => near(p[0], 15, 0.01) && near(p[1], 15, 0.01));
  ok(corner, 'G42 pocket: inside corner resolved to the intersection (15,15)');
}

// 3) ,R fillet: L-corner with ,R5 -> arc tangent points at (15,0) and (20,5)
{
  const src = `T1 M6
S1000 M3
G0 X0 Y0 Z1
G1 Z-1 F100
G1 X20 ,R5
G1 Y20
M30`;
  const res = interpret(src, { machine: 'mill' });
  const pts = feedPts(res);
  ok(pts.some((p) => near(p[0], 15) && near(p[1], 0)) && pts.some((p) => near(p[0], 20) && near(p[1], 5)), ',R5 fillet tangent points (15,0) and (20,5)');
  const mid = pts.find((p) => p[0] > 15.1 && p[1] > 0.1 && p[1] < 4.9);
  ok(mid && near(Math.hypot(mid[0] - 15, mid[1] - 5), 5, 0.01), ',R5 fillet points lie on R5 about (15,5)');
}

// 4) G71 turning: Ø40 stock, profile Ø20 face to Ø36 shoulder; passes 2 mm (radius) deep
{
  const src = `(STOCK D40 L60)
T0101
G97 S1500 M3
G0 X42 Z2
G71 U2 R0.5
G71 P10 Q20 U0 W0 F0.2
N10 G0 X20
G1 Z-20
X36 Z-30
N20 Z-40
G0 X100 Z100
M30`;
  const res = interpret(src, { machine: 'lathe' });
  ok(errs(res).length === 0, 'G71: no errors ' + errs(res).join('; '));
  // roughing levels at radius 19,17,15,13,11 (start radius 21)
  const levels = new Set();
  for (const m of res.moves) if (m.kind === 'feed' && m.cycle && m.cycle.startsWith('G71') && Math.abs(m.pts[0][0] - m.pts[m.pts.length - 1][0]) < 1e-6) levels.add(+m.pts[0][0].toFixed(3));
  ok([19, 17, 15, 13, 11].every((l) => levels.has(l)), `G71: roughing levels r=19..11 (got ${[...levels].sort((a, b) => b - a).join(',')})`);
  // pass at r=15 must stop where the taper reaches r=15: z = -20 - (15-10)/(18-10)*10 = -26.25
  const p15 = res.moves.find((m) => m.kind === 'feed' && near(m.pts[0][0], 15) && near(m.pts[m.pts.length - 1][0], 15));
  ok(p15 && near(p15.pts[p15.pts.length - 1][2], -26.25, 0.01), `G71: r15 pass ends at Z-26.25 (got ${p15 && p15.pts[p15.pts.length - 1][2].toFixed(3)})`);
  // after G71 the program continues after N20 (no extra profile execution)
  const after = res.moves.filter((m) => m.line === 10);
  ok(after.length > 0, 'G71: execution continues after the Q block');
}

// 5) G83 peck: R2 Z-10 Q4 -> feeds to -2, -6, -10
{
  const src = `(TOOL T1 DRILL D5)
T1 M6
S2000 M3
G0 X10 Y10 Z20
G99 G83 Z-10 R2 Q4 F100
G80
M30`;
  const res = interpret(src, { machine: 'mill' });
  const ends = res.moves.filter((m) => m.kind === 'feed').map((m) => +m.pts[m.pts.length - 1][2].toFixed(3));
  ok(JSON.stringify(ends) === JSON.stringify([-2, -6, -10]), `G83: peck ends ${JSON.stringify(ends)}`);
  const last = res.moves[res.moves.length - 1];
  ok(near(last.pts[last.pts.length - 1][2], 2), 'G83 with G99 returns to R plane');
}

// 6) Macro B loop: 3 passes of 1 mm
{
  const src = `T1 M6
S1000 M3
G0 X0 Y0 Z5
#1=0
WHILE [#1 LT 3] DO1
#1=#1+1
G1 Z-#1 F100
G1 X10
G0 Z5
G0 X0
END1
M30`;
  const res = interpret(src, { machine: 'mill' });
  const plunges = res.moves.filter((m) => m.kind === 'feed' && m.pts[0][0] === 0 && m.pts[m.pts.length - 1][0] === 0);
  const zs = plunges.map((m) => +m.pts[m.pts.length - 1][2].toFixed(3));
  ok(JSON.stringify(zs) === JSON.stringify([-1, -2, -3]), `Macro WHILE loop depths ${JSON.stringify(zs)}`);
}

// 7) Helical arc with Z (thread milling style)
{
  const src = `T1 M6
S1000 M3
G0 X5 Y0 Z0
G1 F200
G3 X5 Y0 Z-2 I-5 J0
M30`;
  const res = interpret(src, { machine: 'mill' });
  const m = res.moves.find((mv) => mv.arc);
  const pts = m.pts;
  ok(near(pts[pts.length - 1][2], -2) && pts.every((p) => near(Math.hypot(p[0], p[1]), 5, 1e-6)), 'Helix: full circle R5 descending 2 mm');
}

console.log(fails ? `\n${fails} FAILED` : '\nall geometry checks passed');
process.exitCode = fails ? 1 : 0;
