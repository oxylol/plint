// The work kept on the server (DATA_DIR): /api/store, /api/blob and /api/backup. node test/store.mjs
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DATA = mkdtempSync(join(tmpdir(), 'cncstudio-store-'));
const start = (env, port) => {
  const srv = spawn(process.execPath, [join(ROOT, 'server.js')], { env: { ...process.env, PORT: String(port), MOCK_CLAUDE: '1', ...env }, stdio: ['ignore', 'pipe', 'pipe'] });
  srv.stderr.on('data', (d) => process.stderr.write('[server] ' + d));
  return new Promise((r) => srv.stdout.once('data', () => r(srv)));
};
const srv = await start({ DATA_DIR: DATA }, 5181);
const plain = await start({ DATA_DIR: '' }, 5182);
const url = (p) => `http://127.0.0.1:5181${p}`;
const put = (p, body, type = 'application/json') => fetch(url(p), { method: 'PUT', headers: { 'Content-Type': type }, body });

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
try {
  const off = await fetch('http://127.0.0.1:5182/api/store');
  ok(off.ok && (await off.json()).enabled === false && (await fetch('http://127.0.0.1:5182/api/store/ops', { method: 'PUT', body: '1' })).status === 404, 'without DATA_DIR there is no server storage (the browser keeps the work)');
  const s0 = await (await fetch(url('/api/store'))).json();
  ok(s0.values && !Object.keys(s0.values).length, 'empty store to start with');

  const ops = [{ id: 'a1', fileName: 'plate.nc', machine: 'mill', text: '%\nG0 X0\nM30\n%' }];
  const r1 = await (await put('/api/store/ops', JSON.stringify(ops))).json();
  await put('/api/store/settings', JSON.stringify({ effort: 'max', apiKey: 'sk-ant-secret' }));
  const s1 = await (await fetch(url('/api/store'))).json();
  ok(JSON.stringify(s1.values.ops) === JSON.stringify(ops), 'a value comes back as written');
  ok(s1.values.settings.effort === 'max' && !('apiKey' in s1.values.settings) && !readFileSync(join(DATA, 'store', 'settings.json'), 'utf8').includes('secret'), 'an API key typed into Settings is not stored on the server');
  ok(s1.rev > s0.rev && s1.rev >= r1.rev, 'every write moves the revision on');
  ok((await (await fetch(url('/api/store/rev'))).json()).rev === s1.rev, '/api/store/rev reports it');

  await put('/api/store/ops', 'null');
  ok(!('ops' in (await (await fetch(url('/api/store'))).json()).values), 'null removes a value');
  ok((await put('/api/store/..%2Fescape', '1')).status === 400, 'keys cannot leave the data folder');

  const tris = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0.5]);
  ok((await put('/api/blob/model', tris, 'application/octet-stream')).ok, 'blob upload');
  const back = new Float32Array(await (await fetch(url('/api/blob/model'))).arrayBuffer());
  ok(back.length === 9 && back[8] === 0.5, 'blob comes back byte for byte');
  await fetch(url('/api/blob/model'), { method: 'DELETE' });
  ok((await fetch(url('/api/blob/model'))).status === 404, 'blob delete');

  await fetch(url('/api/backup'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: { ops } }) });
  const bk = readdirSync(join(DATA, 'backup'));
  ok(bk.length === 1 && JSON.parse(readFileSync(join(DATA, 'backup', bk[0]), 'utf8')).values.ops[0].fileName === 'plate.nc', 'a browser\'s older job is kept as a backup file');
} finally {
  srv.kill();
  plain.kill();
  rmSync(DATA, { recursive: true, force: true });
}
process.exitCode = fails ? 1 : 0;
