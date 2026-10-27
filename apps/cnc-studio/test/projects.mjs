// Projects kept on the server (DATA_DIR/projects): /api/projects. node test/projects.mjs
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DATA = mkdtempSync(join(tmpdir(), 'cncstudio-projects-'));
const start = (env, port) => {
  const srv = spawn(process.execPath, [join(ROOT, 'server.js')], { env: { ...process.env, PORT: String(port), MOCK_CLAUDE: '1', ...env }, stdio: ['ignore', 'pipe', 'pipe'] });
  srv.stderr.on('data', (d) => process.stderr.write('[server] ' + d));
  return new Promise((r) => srv.stdout.once('data', () => r(srv)));
};
const srv = await start({ DATA_DIR: DATA }, 5183);
const plain = await start({ DATA_DIR: '' }, 5184);
const url = (p) => `http://127.0.0.1:5183/api/projects${p}`;
const json = (method, p, body) => fetch(url(p), { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const list = async () => (await (await fetch(url(''))).json()).projects;

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
try {
  const off = await fetch('http://127.0.0.1:5184/api/projects');
  ok(off.status === 404 && (await off.json()).enabled === false, 'without DATA_DIR there are no projects');
  ok((await list()).length === 0, 'no projects to start with');

  const ops = [{ fileName: 'plate.nc', machine: 'mill', text: '%\nG0 X0\nM30\n%' }, { fileName: 'op20.nc', machine: 'lathe', text: '%\nG0 X40\nM30\n%' }];
  const tris = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0.5]);
  ok((await fetch(url('/p1/model'), { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' }, body: tris })).ok, 'the model can go up before project.json exists');
  const r1 = await json('PUT', '/p1', { name: '  Bracket  ', ops, subprograms: [{ name: 'O9001.nc', text: 'M99' }], model: { name: 'bracket.step', format: 'step', triCount: 1, placement: { up: '+Z' } } });
  const s1 = await r1.json();
  ok(r1.ok && s1.id === 'p1' && s1.savedAt && s1.createdAt === s1.savedAt, 'save a project');
  const l1 = await list();
  ok(l1.length === 1 && l1[0].name === 'Bracket' && l1[0].ops === 2 && l1[0].machines.join() === 'mill,lathe' && l1[0].model === 'bracket.step' && l1[0].subprograms === 1, 'the list has its name (trimmed), operations, machines, model and subprograms');
  const p1 = await (await fetch(url('/p1'))).json();
  ok(p1.name === 'Bracket' && p1.ops[1].text.includes('X40') && p1.subprograms[0].text === 'M99' && p1.model.placement.up === '+Z', 'a project comes back whole');
  const back = new Float32Array(await (await fetch(url('/p1/model'))).arrayBuffer());
  ok(back.length === 9 && back[8] === 0.5, 'and so does its model');

  await new Promise((r) => setTimeout(r, 5));
  const r2 = await json('PUT', '/p1', { name: 'Bracket', ops: [ops[0]], subprograms: [], model: null });
  const s2 = await r2.json();
  ok(r2.ok && s2.createdAt === s1.createdAt && s2.savedAt > s1.savedAt, 'saving again keeps createdAt and moves savedAt on');
  ok((await fetch(url('/p1/model'))).status === 404 && !existsSync(join(DATA, 'projects', 'p1', 'model.bin')), 'a project saved without a model drops the old model file');

  const r3 = await json('PATCH', '/p1', { name: 'Bracket v2' });
  const p1b = await (await fetch(url('/p1'))).json();
  ok(r3.ok && p1b.name === 'Bracket v2' && p1b.savedAt === s2.savedAt && p1b.ops.length === 1, 'rename changes the name only');

  await json('PUT', '/p2', { name: 'Shaft', ops: [ops[1]] });
  const l2 = await list();
  ok(l2.length === 2 && l2[0].id === 'p2' && l2[1].id === 'p1', 'the list is newest first');

  ok((await json('PUT', '/p3', { name: '', ops })).status === 400 && (await json('PUT', '/p3', { name: 'x', ops: [] })).status === 400, 'a project needs a name and an operation');
  ok((await json('PUT', '/..%2Fescape', { name: 'x', ops })).status === 400 && (await fetch(url('/nope'))).status === 404 && (await json('PATCH', '/nope', { name: 'x' })).status === 404, 'bad ids and missing projects');
  ok(!readFileSync(join(DATA, 'projects', 'p1', 'project.json'), 'utf8').includes('apiKey'), 'project.json holds the job only');

  ok((await fetch(url('/p1'), { method: 'DELETE' })).ok && (await list()).length === 1 && !existsSync(join(DATA, 'projects', 'p1')), 'delete removes the whole project folder');
} finally {
  srv.kill();
  plain.kill();
  rmSync(DATA, { recursive: true, force: true });
}
process.exitCode = fails ? 1 : 0;
