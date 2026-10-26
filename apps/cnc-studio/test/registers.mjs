// Tool register import checks against the installed CncSimulator5 files. node test/registers.mjs
import { readFileSync, existsSync } from 'node:fs';
import { parseTurnToolsIni, parseMillReg, detectRegister, polygonBounds } from '../public/js/gcode/lathetools.js';
import { decodeLegacyText } from '../public/js/gcode/textdecode.js';
import { normalizeTool, describeTool } from '../public/js/gcode/tools.js';

const APP = 'C:/Users/Filip/AppData/Roaming/CncSimulator5UserFiles/Tools';
const PF = 'C:/Program Files (x86)/MicroTech StellaData/CncSimulator5/CncSimulator5UserFiles/Tools';
let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
const read = (p) => decodeLegacyText(readFileSync(p));

if (!existsSync(`${PF}/TurnTools.ini`)) { console.log('CncSimulator5 not installed — skipping'); process.exit(0); }

// factory lathe register
{
  const text = read(`${PF}/TurnTools.ini`);
  ok(detectRegister(text) === 'lathe', 'TurnTools.ini detected as a lathe register');
  const { tools, skipped } = parseTurnToolsIni(text);
  ok(Object.keys(tools).length === 28 && !skipped.length, `28 lathe tools (got ${Object.keys(tools).length}, skipped ${skipped.length})`);
  const t1 = tools[1];
  ok(t1.type === 'turn' && t1.hand === 'left' && t1.noseRadius === 0.7 && t1.cutter.length === 13, 'T1: left-hand turning, R0.7, 13-point insert');
  ok(JSON.stringify(t1.noseOffset) === '[0.7,0.7]', 'T1 nose centre (R,R) from the register');
  const b = polygonBounds(t1.cutter);
  ok(b.z0 >= -0.01 && b.x0 >= -0.01 && b.z1 > 9 && b.x1 > 17, 'T1 insert lies above/right of the tip (Z 0..10, X 0..17.7)');
  ok(tools[6].hand === 'right' && tools[6].noseOffset[0] < 0, 'T6: right-hand tool, nose offset −Z');
  ok(tools[9].type === 'bore' && tools[9].noseOffset[1] < 0, 'T9: boring bar (insert below the tip)');
  ok(tools[12].type === 'groove' && tools[12].width === 4 && tools[12].origin === 1, 'T12: groove W4, reference bottom-left');
  const g13 = polygonBounds(tools[13].cutter);
  ok(Math.abs(g13.z1) < 1e-9 && Math.abs(g13.z0 + 6) < 1e-9, 'T13 (ToolOrigin 3): blade spans Z −6..0 (reference = right corner)');
  const h12 = polygonBounds(tools[12].holder);
  ok(Math.abs((h12.z0 + h12.z1) / 2 - 2) < 1e-9 && h12.x0 > 14, 'T12 holder centred over the blade, above its top');
  const d21 = tools[21];
  ok(d21.type === 'drill' && d21.diameter === 35 && d21.length === 100 && d21.tipAngle === 120, 'T21: drill Ø35 × 100, 120°');
  ok(polygonBounds(d21.holder).z0 > 98, 'T21 holder starts at the drill\'s back end');
  ok(tools[20].hand === 'right' && polygonBounds(tools[20].cutter).z1 <= 0, 'T20: DrillRight points toward +Z');
  ok(tools[26].noseOffsetFixed && tools[26].noseOffset[1] > 0, 'T26: contradictory nose offset corrected from its polygon');
  console.log('     ' + describeTool(normalizeTool(tools[12], 'lathe'), 'lathe', { detail: true }));
  console.log('     ' + describeTool(normalizeTool(tools[9], 'lathe'), 'lathe', { detail: true }));
}

// the user's own register (AppData) has an edited T11
if (existsSync(`${APP}/TurnTools.ini`)) {
  const { tools, skipped } = parseTurnToolsIni(read(`${APP}/TurnTools.ini`));
  ok(tools[11].type === 'groove' && tools[11].width === 4, 'AppData register: T11 is the user\'s 4 mm groove tool');
  ok(skipped.some((s) => s.startsWith('T11')), 'duplicate T11 reported, first one kept (like CncSimulator)');
}

// mill register
{
  const text = read(`${PF}/MillTools.millreg`);
  ok(detectRegister(text) === 'mill', 'MillTools.millreg detected as a mill register');
  const { tools } = parseMillReg(text);
  ok(Object.keys(tools).length === 13, `13 mill tools (empty slots skipped) — got ${Object.keys(tools).length}`);
  const t1 = normalizeTool(tools[1], 'mill');
  ok(t1.type === 'flat' && t1.diameter === 10 && t1.fluteLength === 30 && t1.overallLength === 75 && t1.feed === 300, 'T1: Ø10 end mill, 30 flute, 75 stick-out, F300');
  ok(tools[4].type === 'drill' && tools[4].diameter === 4, 'T4 "BORR 4 MM" becomes a Ø4 drill');
  console.log('     ' + describeTool(t1, 'mill', { detail: true }));
}

// decoding: CP850 header
ok(read(`${PF}/MillTools.millreg`).startsWith('FRÄSVERKTYG'), 'CP850 header decodes to FRÄSVERKTYG');

process.exitCode = fails ? 1 : 0;
