// CNC Studio server: static files + Claude API streaming proxy.
// Run: npm start  (reads ANTHROPIC_API_KEY, or a key entered in the app's settings)
// Environment: PORT (5178), HOST (127.0.0.1), ANTHROPIC_API_KEY, APP_PASSWORD / APP_USER (password-protect
// the whole site — use it whenever the site is reachable from other machines), CNCSIM_TOOLS_DIR (a folder
// with TurnTools.ini / MillTools.millreg to offer in "Import from CncSimulator5").
import http from 'node:http';
import { timingSafeEqual, createHash } from 'node:crypto';
import { createReadStream, readFileSync } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { dirname, extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import Anthropic from '@anthropic-ai/sdk';

const ROOT = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 5178;
const HOST = process.env.HOST || '127.0.0.1';
const MOCK = process.env.MOCK_CLAUDE === '1';
const APP_USER = process.env.APP_USER || 'cnc';
const APP_PASSWORD = process.env.APP_PASSWORD || '';

// HTTP basic auth for the whole site when APP_PASSWORD is set (the browser asks once and
// sends it with every request, including the model-reader worker)
const digest = (v) => createHash('sha256').update(v).digest();
function authorized(req) {
  if (!APP_PASSWORD) return true;
  const m = /^Basic\s+(.+)$/i.exec(req.headers.authorization || '');
  if (!m) return false;
  const given = Buffer.from(m[1], 'base64').toString('utf8');
  return timingSafeEqual(digest(given), digest(`${APP_USER}:${APP_PASSWORD}`));
}

export const MODELS = [
  { id: 'claude-opus-5-5', label: 'Claude Opus 5.5', fallbacks: true },
  { id: 'claude-sonnet-5-5', label: 'Claude Sonnet 5.5', fallbacks: true },
  { id: 'claude-fable-5-1', label: 'Claude Fable 5.1', fallbacks: true },
  { id: 'claude-haiku-5-5', label: 'Claude Haiku 5.5', fallbacks: false }, // no server-side fallback for Haiku
];
const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];

const TOOLS = [
  {
    name: 'write_program',
    description:
      'Write a complete NC program into an operation of the job (replacing its program) and run the simulation. ' +
      'Returns the simulation report: errors, warnings, collisions (rapids into material, shank, holder, chuck, vise), cycle time, ' +
      'removed volume, the cut envelope of each tool, the measured geometry of the result (features, sizes, positions in work ' +
      'coordinates) and, with a target model, what is still to remove or gouged. Always pass the whole program, never a fragment.',
    input_schema: {
      type: 'object',
      properties: {
        title: { type: 'string', description: 'Short program name shown in the version history, e.g. "Pocket 40x30 + 4 holes".' },
        operation: { type: 'string', description: 'Operation to write, e.g. "OP20". Default: the active operation. A name that does not exist yet adds a new operation at the end of the job.' },
        machine: { type: 'string', enum: ['mill', 'lathe'], description: 'Machine for this operation, when it differs from the current one (e.g. a mill operation after turning).' },
        gcode: { type: 'string', description: 'The complete program text, starting with % and ending with M30 and %.' },
      },
      required: ['gcode'],
    },
    eager_input_streaming: true,
  },
  {
    name: 'inspect_part',
    description:
      'Measure the simulated part of an operation — exact numbers instead of pictures, in that operation\'s work coordinates. ' +
      'query "summary": size, outline and all features (pockets, holes with diameter along the depth, slots, steps, islands; lathe: profile). ' +
      '"view": features seen from one side plus a character depth map. "section": the exact material outline in a plane. ' +
      '"probe": material or air at given points. "compare": deviations from the target model (material still to remove, gouges). ' +
      '"target": the target model\'s own features in this operation\'s coordinates.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', enum: ['summary', 'view', 'section', 'probe', 'compare', 'target'] },
        operation: { type: 'string', description: 'Operation whose result to measure, e.g. "OP10". Default: the active operation.' },
        view: { type: 'string', enum: ['top', 'bottom', 'front', 'back', 'left', 'right'], description: 'For "view": the side to look from (front = from −Y).' },
        axis: { type: 'string', enum: ['x', 'y', 'z'], description: 'For "section": the axis the cutting plane is perpendicular to.' },
        at: { type: 'number', description: 'For "section": position of the plane along that axis (mm).' },
        points: { type: 'array', items: { type: 'array', items: { type: 'number' } }, description: 'For "probe": points as [x, y, z] in work coordinates (max 200).' },
      },
      required: ['query'],
    },
    eager_input_streaming: true,
  },
  {
    name: 'view_simulation',
    description:
      'Render pictures of the fully machined result of an operation. Slow, and rarely needed: the reports and inspect_part ' +
      'already give exact measurements. Use it only for what numbers cannot show (engraved text, 3D surfaces). Grey is untouched ' +
      'stock; surfaces cut by each tool are tinted in that tool\'s colour; dashed red lines are rapids, solid lines feed moves.',
    input_schema: {
      type: 'object',
      properties: {
        operation: { type: 'string', description: 'Operation to show, e.g. "OP20". Default: the active operation.' },
        views: {
          type: 'array',
          items: { type: 'string', enum: ['iso', 'top', 'front', 'side'] },
          description: 'Camera views to render. Default: iso + top for the mill, iso + front for the lathe.',
        },
      },
    },
    eager_input_streaming: true,
  },
];

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png',
  '.ico': 'image/x-icon', '.wasm': 'application/wasm', '.nc': 'text/plain; charset=utf-8', '.txt': 'text/plain; charset=utf-8', '.woff2': 'font/woff2',
};

const STATIC_ROOTS = [
  { prefix: '/vendor/three/', dir: join(ROOT, 'node_modules', 'three') },
  { prefix: '/vendor/codemirror/', dir: join(ROOT, 'node_modules', 'codemirror') },
  { prefix: '/vendor/occt/', dir: join(ROOT, 'node_modules', 'occt-import-js', 'dist') },
  { prefix: '/', dir: join(ROOT, 'public') },
];

async function serveStatic(req, res, pathname) {
  for (const { prefix, dir } of STATIC_ROOTS) {
    if (!pathname.startsWith(prefix)) continue;
    let rel = decodeURIComponent(pathname.slice(prefix.length));
    if (!rel || rel.endsWith('/')) rel += 'index.html';
    const file = normalize(join(dir, rel));
    if (!file.startsWith(dir + sep)) break;
    try {
      const st = await stat(file);
      if (!st.isFile()) break;
      res.writeHead(200, { 'Content-Type': MIME[extname(file).toLowerCase()] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
      createReadStream(file).pipe(res);
      return;
    } catch {
      break;
    }
  }
  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not found');
}

function readJson(req, limit = 40 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) { reject(new Error('Request too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); } catch (e) { reject(e); }
    });
    req.on('error', reject);
  });
}

// cache breakpoint on the last block of the conversation (system prompt + tools carry the other one)
function withCacheBreakpoint(messages) {
  const out = structuredClone(messages);
  const last = out[out.length - 1];
  if (last && Array.isArray(last.content) && last.content.length) {
    const blk = last.content[last.content.length - 1];
    if (blk && typeof blk === 'object') blk.cache_control = { type: 'ephemeral' };
  }
  return out;
}

function describeError(e) {
  if (e instanceof Anthropic.APIUserAbortError) return { status: 499, kind: 'aborted', message: 'Request cancelled' };
  if (e instanceof Anthropic.AuthenticationError) return { status: 401, kind: 'auth', message: 'Authentication failed — the Anthropic API key is missing or invalid. Add one in Settings or set ANTHROPIC_API_KEY.' };
  if (e instanceof Anthropic.PermissionDeniedError) return { status: 403, kind: 'permission', message: `This API key cannot use that model or feature: ${e.message}` };
  if (e instanceof Anthropic.NotFoundError) return { status: 404, kind: 'not_found', message: `Model or endpoint not found: ${e.message}` };
  if (e instanceof Anthropic.RateLimitError) {
    const ra = e.headers?.get?.('retry-after');
    return { status: 429, kind: 'rate_limit', message: `Rate limited by the API${ra ? ` — retry in ${ra}s` : ''}.`, retryAfter: ra ? Number(ra) : null };
  }
  if (e instanceof Anthropic.BadRequestError) return { status: 400, kind: 'bad_request', message: e.message };
  if (e instanceof Anthropic.InternalServerError) return { status: e.status || 500, kind: 'server', message: 'Anthropic API error — try again in a moment.' };
  if (e instanceof Anthropic.APIConnectionError) return { status: 502, kind: 'network', message: 'Could not reach the Anthropic API (network error).' };
  if (e instanceof Anthropic.APIError) return { status: e.status || 500, kind: 'api', message: e.message };
  const msg = String(e?.message || e);
  if (/auth|api key|apikey|credential/i.test(msg)) return { status: 401, kind: 'auth', message: 'No Anthropic API key configured. Add one in Settings or set ANTHROPIC_API_KEY before `npm start`.' };
  return { status: 500, kind: 'internal', message: msg };
}

let defaultClient = null;
function clientFor(req) {
  const key = req.headers['x-anthropic-key'];
  if (typeof key === 'string' && key.trim()) return new Anthropic({ apiKey: key.trim() });
  defaultClient ||= new Anthropic();
  return defaultClient;
}

async function handleChat(req, res) {
  let body;
  try {
    body = await readJson(req);
  } catch (e) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: `Bad request: ${e.message}` }));
    return;
  }
  const model = MODELS.find((m) => m.id === body.model) || MODELS[0];
  const effort = EFFORTS.includes(body.effort) ? body.effort : 'high';
  if (!Array.isArray(body.messages) || !body.messages.length) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'messages must be a non-empty array' }));
    return;
  }
  res.writeHead(200, { 'Content-Type': 'application/x-ndjson; charset=utf-8', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no' });
  const send = (o) => { if (!res.writableEnded) res.write(JSON.stringify(o) + '\n'); };

  if (MOCK) { await mockStream(body, send); res.end(); return; }

  const system = readFileSync(join(ROOT, 'prompts', 'system.md'), 'utf8');
  const params = {
    model: model.id,
    max_tokens: 64000,
    system: [{ type: 'text', text: system, cache_control: { type: 'ephemeral' } }],
    tools: TOOLS,
    thinking: { type: 'adaptive', display: 'summarized' },
    output_config: { effort },
    messages: withCacheBreakpoint(body.messages),
  };
  if (model.fallbacks) {
    // retry a safety-classifier decline on Anthropic's recommended fallback model, inside the same call
    params.betas = ['server-side-fallback-2026-07-01'];
    params.fallbacks = 'default';
  }

  let stream = null;
  res.on('close', () => { if (!res.writableEnded && stream) stream.abort(); });
  try {
    stream = clientFor(req).beta.messages.stream(params);
    for await (const ev of stream) {
      switch (ev.type) {
        case 'message_start':
          send({ t: 'start', model: ev.message.model });
          break;
        case 'content_block_start': {
          const b = ev.content_block;
          send({ t: 'block_start', i: ev.index, type: b.type, name: b.name, id: b.id, from: b.from?.model ?? b.from_?.model, to: b.to?.model });
          break;
        }
        case 'content_block_delta': {
          const d = ev.delta;
          if (d.type === 'text_delta') send({ t: 'delta', i: ev.index, kind: 'text', v: d.text });
          else if (d.type === 'thinking_delta') send({ t: 'delta', i: ev.index, kind: 'thinking', v: d.thinking });
          else if (d.type === 'input_json_delta') send({ t: 'delta', i: ev.index, kind: 'input', v: d.partial_json });
          break;
        }
        case 'content_block_stop':
          send({ t: 'block_stop', i: ev.index });
          break;
        default:
          break;
      }
    }
    const final = await stream.finalMessage();
    send({ t: 'final', message: { id: final.id, model: final.model, role: final.role, content: final.content, stop_reason: final.stop_reason, stop_details: final.stop_details ?? null, usage: final.usage } });
  } catch (e) {
    const err = describeError(e);
    if (err.kind !== 'aborted') console.error(`[chat] ${err.status} ${err.message}`);
    send({ t: 'error', ...err });
  }
  res.end();
}

// --------------------------------------------------------------------------
// Mock mode (MOCK_CLAUDE=1): replays a scripted tool-use conversation so the
// whole agent loop can be exercised without an API key.
// --------------------------------------------------------------------------
const MOCK_PROGRAM = `%
O1001 (MOCK POCKET 40X30)
(STOCK X-40:40 Y-30:30 Z-15:0)
(TOOL T1 FLAT D8 L25 "END MILL D8")
G90 G17 G21 G40 G49 G80 G94
T1 M06
S6000 M03
G54
G00 X0. Y0.
G43 Z25. H01 M08
G00 Z2.
G01 Z0. F300
G03 X0. Y0. Z-2. I2. J0. F800
G03 X0. Y0. Z-4. I2. J0.
G01 X-14. Y-9.
X14.
Y9.
X-14.
Y-9.
G00 Z25.
M05
M09
G28 G91 Z0.
G90
M30
%`;

async function mockStream(body, send) {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const last = body.messages[body.messages.length - 1];
  const results = Array.isArray(last.content) ? last.content.filter((b) => b.type === 'tool_result') : [];
  const prevAssistant = [...body.messages].reverse().find((m) => m.role === 'assistant');
  const lastToolName = prevAssistant?.content?.find?.((b) => b.type === 'tool_use')?.name;
  send({ t: 'start', model: 'mock-claude' });
  const content = [];
  let idx = 0;
  const text = async (s, kind = 'text') => {
    send({ t: 'block_start', i: idx, type: kind === 'text' ? 'text' : 'thinking' });
    for (const chunk of s.match(/.{1,12}/gs)) { send({ t: 'delta', i: idx, kind, v: chunk }); await sleep(15); }
    send({ t: 'block_stop', i: idx });
    content.push(kind === 'text' ? { type: 'text', text: s } : { type: 'thinking', thinking: s, signature: 'mock' });
    idx++;
  };
  const tool = async (name, input) => {
    const id = `toolu_mock_${Date.now()}_${idx}`;
    send({ t: 'block_start', i: idx, type: 'tool_use', name, id });
    const json = JSON.stringify(input);
    for (const chunk of json.match(/.{1,24}/gs)) { send({ t: 'delta', i: idx, kind: 'input', v: chunk }); await sleep(8); }
    send({ t: 'block_stop', i: idx });
    content.push({ type: 'tool_use', id, name, input });
    idx++;
  };
  let stop = 'end_turn';
  if (!results.length) {
    await text('Planning a 40×30 pocket, 4 mm deep, helical entry with an Ø8 end mill in aluminium.', 'thinking');
    await text('I will write the program and simulate it.');
    await tool('write_program', { title: 'Mock pocket 40x30', gcode: MOCK_PROGRAM });
    stop = 'tool_use';
  } else if (lastToolName === 'write_program') {
    await text('Report looks clean — checking the shape visually.');
    await tool('view_simulation', { views: ['iso', 'top'] });
    stop = 'tool_use';
  } else {
    await text('Done. **Pocket 40×30×4** with a helical entry (Ø8 end mill, S6000 F800, aluminium assumed). Check the corner radius R4 is acceptable.');
  }
  send({ t: 'final', message: { id: 'msg_mock', model: 'mock-claude', role: 'assistant', content, stop_reason: stop, stop_details: null, usage: { input_tokens: 0, output_tokens: 0 } } });
}

// --------------------------------------------------------------------------
// Installed CncSimulator5 tool registers, in the simulator's own search order
// (%appdata%\CamModul6\Tools, %appdata%\CncSimulator5UserFiles\Tools, [ExeDir]\CncSimulator5UserFiles\Tools)
// --------------------------------------------------------------------------
function registerCandidates() {
  const appdata = process.env.APPDATA || join(process.env.USERPROFILE || '', 'AppData', 'Roaming');
  const pf86 = process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)';
  const exeDir = join(pf86, 'MicroTech StellaData', 'CncSimulator5', 'CncSimulator5UserFiles', 'Tools');
  const dirs = [
    ...(process.env.CNCSIM_TOOLS_DIR ? [{ dir: process.env.CNCSIM_TOOLS_DIR, where: 'tool registers on this server (CNCSIM_TOOLS_DIR)' }] : []),
    { dir: join(appdata, 'CamModul6', 'Tools'), where: 'CamModul6 (AppData)' },
    { dir: join(appdata, 'CncSimulator5UserFiles', 'Tools'), where: 'CncSimulator5 user files (AppData) — what the simulator uses' },
    { dir: exeDir, where: 'CncSimulator5 install folder (factory copy)' },
  ];
  const out = [];
  for (const { dir, where } of dirs) {
    out.push({ kind: 'lathe', path: join(dir, 'TurnTools.ini'), where });
    out.push({ kind: 'lathe', path: join(dir, 'TurnTools_25.ini'), where: `${where}, alternative register` });
    out.push({ kind: 'mill', path: join(dir, 'MillTools.millreg'), where });
  }
  return out;
}

async function findRegisters() {
  const found = [];
  for (const c of registerCandidates()) {
    try {
      const st = await stat(c.path);
      if (!st.isFile() || st.size > 2_000_000) continue;
      const bytes = await readFile(c.path);
      found.push({ ...c, mtime: st.mtime.toISOString(), size: st.size, b64: bytes.toString('base64') });
    } catch { /* not there */ }
  }
  return found;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  if (!authorized(req)) {
    res.writeHead(401, { 'WWW-Authenticate': 'Basic realm="CNC Studio", charset="UTF-8"', 'Content-Type': 'text/plain' });
    res.end('Password required');
    return;
  }
  try {
    if (url.pathname === '/api/cncsim/registers' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' });
      res.end(JSON.stringify({ registers: await findRegisters() }));
      return;
    }
    if (url.pathname === '/api/status' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' });
      res.end(JSON.stringify({
        ok: true,
        mock: MOCK,
        serverKey: Boolean(process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_AUTH_TOKEN),
        models: MODELS.map(({ id, label }) => ({ id, label })),
        efforts: EFFORTS,
      }));
      return;
    }
    if (url.pathname === '/api/chat' && req.method === 'POST') {
      await handleChat(req, res);
      return;
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405); res.end(); return;
    }
    await serveStatic(req, res, url.pathname);
  } catch (e) {
    console.error(e);
    if (!res.headersSent) res.writeHead(500);
    res.end();
  }
});

server.listen(PORT, HOST, () => {
  console.log(`CNC Studio running at http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}${MOCK ? '  (mock Claude)' : ''}${APP_PASSWORD ? `  (password protected, user "${APP_USER}")` : ''}`);
  if (!APP_PASSWORD && !['127.0.0.1', 'localhost', '::1'].includes(HOST)) {
    console.log(`WARNING: listening on ${HOST} without APP_PASSWORD — anyone who can reach this port can use the app${process.env.ANTHROPIC_API_KEY ? ' and your API key' : ''}.`);
  }
  if (!MOCK && !process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_AUTH_TOKEN) {
    console.log('No ANTHROPIC_API_KEY in the environment — enter a key in the app (Settings), or restart with the variable set.');
  }
});
