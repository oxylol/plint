// Simulate the original's samples with the user's CncSimulator5 tool register. node test/register-sim.mjs
import { readFileSync } from 'node:fs';
import { interpret } from '../public/js/gcode/interpreter.js';
import { analyzeProgram } from '../public/js/sim/simrunner.js';
import { parseTurnToolsIni, parseMillReg } from '../public/js/gcode/lathetools.js';
import { decodeLegacyText } from '../public/js/gcode/textdecode.js';
import { describeTool } from '../public/js/gcode/tools.js';

const REG = 'C:/Users/Filip/AppData/Roaming/CncSimulator5UserFiles/Tools';
const S = 'C:/Users/Filip/Documents/CNC-Files/Samples';
const lathe = parseTurnToolsIni(decodeLegacyText(readFileSync(`${REG}/TurnTools.ini`))).tools;
const mill = parseMillReg(decodeLegacyText(readFileSync(`${REG}/MillTools.millreg`))).tools;
for (const [file, machine] of [
  [`${S}/Turn/ISO/TURN_STANDARD1.nc`, 'lathe'], [`${S}/Turn/ISO/TURN_STANDARD2.nc`, 'lathe'],
  [`${S}/Turn/Fanuc/FANUC-TURN_G71_G70_G74.nc`, 'lathe'], [`${S}/Turn/Fanuc/FANUC-TURN_G72_G73_G70.nc`, 'lathe'],
  ['C:/Users/Filip/Downloads/1007.nc', 'lathe'], [`${S}/Mill/ISO/SAMPLE10.NC`, 'mill'], [`${S}/Mill/ISO/_Sample6.nc`, 'mill'],
]) {
  const t0 = performance.now();
  const r = interpret(readFileSync(file, 'latin1'), { machine, toolLibrary: machine === 'lathe' ? lathe : mill, settings: { latheRadiusMode: /STANDARD2/.test(file) } });
  const a = analyzeProgram(r);
  console.log(`\n=== ${file.split('/').pop()} — ${(performance.now() - t0).toFixed(0)} ms, removed ${(a.removedVolume / 1000).toFixed(1)} cm³`);
  for (const n of r.usedTools) console.log('   ', describeTool(r.tools[n], machine, { detail: true }));
  for (const m of [...r.messages, ...a.issues]) console.log(`    ${m.level.toUpperCase()} L${m.callLine === null ? '-' : m.callLine + 1}: ${m.text}`);
}
