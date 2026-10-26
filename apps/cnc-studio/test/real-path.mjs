// End-to-end check of server.js's real Claude path against a local fake of the Messages API.
// Verifies the request the SDK sends and that SSE events reach the browser protocol. node test/real-path.mjs
import http from 'node:http';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const captured = [];

const sse = (res, events) => {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'request-id': 'req_fake' });
  for (const e of events) res.write(`event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`);
  res.end();
};
const msgStart = { type: 'message_start', message: { id: 'msg_fake', type: 'message', role: 'assistant', model: 'claude-opus-5-5', content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 1200, output_tokens: 1, cache_read_input_tokens: 900, cache_creation_input_tokens: 0 } } };

const fake = http.createServer((req, res) => {
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    const json = JSON.parse(body);
    captured.push({ url: req.url, headers: req.headers, body: json });
    const last = json.messages[json.messages.length - 1];
    const hasResult = Array.isArray(last.content) && last.content.some((b) => b.type === 'tool_result');
    if (!hasResult) {
      const gcode = '%\nO1 (FACE)\n(STOCK X-50:50 Y-40:40 Z-20:0)\n(TOOL T1 FACE D50)\nT1 M06\nS1900 M03\nG00 X-80. Y-20. Z2.\nG01 Z-1. F400\nX80. F900\nG00 Z25.\nM30\n%';
      const input = JSON.stringify({ title: 'Face 1 mm', gcode });
      const chunks = input.match(/.{1,30}/gs);
      sse(res, [
        msgStart,
        { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: '' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Face mill, two passes.' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'sig-abc' } },
        { type: 'content_block_stop', index: 0 },
        { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 'toolu_1', name: 'write_program', input: {} } },
        ...chunks.map((c) => ({ type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: c } })),
        { type: 'content_block_stop', index: 1 },
        { type: 'message_delta', delta: { stop_reason: 'tool_use', stop_sequence: null }, usage: { output_tokens: 140 } },
        { type: 'message_stop' },
      ]);
    } else {
      sse(res, [
        msgStart,
        { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Faced 1 mm in two passes.' } },
        { type: 'content_block_stop', index: 0 },
        { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 12 } },
        { type: 'message_stop' },
      ]);
    }
  });
});
await new Promise((r) => fake.listen(5199, '127.0.0.1', r));

const srv = spawn(process.execPath, [join(ROOT, 'server.js')], {
  env: { ...process.env, ANTHROPIC_BASE_URL: 'http://127.0.0.1:5199', ANTHROPIC_API_KEY: 'sk-ant-test', PORT: '5180', MOCK_CLAUDE: '' },
  stdio: ['ignore', 'pipe', 'pipe'],
});
srv.stderr.on('data', (d) => process.stderr.write('[server] ' + d));
await new Promise((r) => srv.stdout.once('data', r));

async function chat(messages) {
  const r = await fetch('http://127.0.0.1:5180/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: 'claude-opus-5-5', effort: 'high', messages }) });
  const lines = (await r.text()).trim().split('\n').map((l) => JSON.parse(l));
  return lines;
}

let fails = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fails++; };
try {
  const history = [{ role: 'user', content: [{ type: 'text', text: '<simulator_state>…</simulator_state>\n<user_request>Face 1 mm</user_request>' }] }];
  const ev1 = await chat(history);
  const kinds = ev1.map((e) => e.t + (e.kind ? ':' + e.kind : ''));
  ok(kinds[0] === 'start' && kinds.includes('delta:thinking') && kinds.includes('delta:input') && kinds.at(-1) === 'final', 'stream 1 relayed: ' + [...new Set(kinds)].join(', '));
  const input = ev1.filter((e) => e.kind === 'input').map((e) => e.v).join('');
  ok(JSON.parse(input).gcode.includes('G01 Z-1.'), 'tool input JSON reassembles from input deltas');
  const final1 = ev1.at(-1).message;
  ok(final1.stop_reason === 'tool_use' && final1.content[0].type === 'thinking' && final1.content[0].signature === 'sig-abc', 'final message keeps thinking block + signature');
  ok(final1.content[1].type === 'tool_use' && final1.content[1].input.title === 'Face 1 mm', 'final tool_use input parsed by the SDK');

  const req1 = captured[0];
  ok(req1.url.startsWith('/v1/messages'), `POST ${req1.url}`);
  // SDK 1.x sends the key as a bearer token (older versions used x-api-key)
  ok(req1.headers.authorization === 'Bearer sk-ant-test' || req1.headers['x-api-key'] === 'sk-ant-test', 'API key from env reaches the API');
  ok(String(req1.headers['anthropic-beta'] || '').includes('server-side-fallback-2026-07-01'), `anthropic-beta: ${req1.headers['anthropic-beta']}`);
  const b = req1.body;
  ok(b.model === 'claude-opus-5-5' && b.stream === true && b.max_tokens === 64000, 'model / stream / max_tokens');
  ok(b.fallbacks === 'default', 'fallbacks: "default"');
  ok(b.thinking?.type === 'adaptive' && b.thinking.display === 'summarized', 'adaptive thinking, summarized');
  ok(b.output_config?.effort === 'high', 'output_config.effort = high');
  ok(b.system?.[0]?.cache_control?.type === 'ephemeral' && b.system[0].text.length > 2000, 'system prompt with cache breakpoint');
  ok(b.messages.at(-1).content.at(-1).cache_control?.type === 'ephemeral', 'cache breakpoint on the last message block');
  ok(['write_program', 'inspect_part', 'view_simulation'].every((n) => b.tools.some((t) => t.name === n)) && b.tools.every((t) => t.eager_input_streaming === true), 'tools with eager_input_streaming');
  ok(!('betas' in b) && !('temperature' in b), 'no betas/temperature in the body');

  // second turn: echo assistant content + tool_result, as the browser does
  history.push({ role: 'assistant', content: final1.content });
  history.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'Simulation report — mill … Issues: none.' }] });
  const ev2 = await chat(history);
  ok(ev2.at(-1).t === 'final' && ev2.at(-1).message.stop_reason === 'end_turn', 'turn 2 ends the loop with end_turn');
  const sent2 = captured[1].body.messages;
  ok(sent2[1].content[0].type === 'thinking' && sent2[1].content[0].signature === 'sig-abc', 'thinking block passed back unchanged');
  ok(!sent2[0].content[0].cache_control, 'earlier turns carry no stale cache_control');

  // haiku: no fallbacks
  await fetch('http://127.0.0.1:5180/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: 'claude-haiku-5-5', effort: 'low', messages: history.slice(0, 1) }) }).then((r) => r.text());
  const req3 = captured[2];
  ok(!('fallbacks' in req3.body) && !String(req3.headers['anthropic-beta'] || '').includes('fallback'), 'Haiku request has no fallbacks');
} finally {
  srv.kill();
  fake.close();
}
process.exitCode = fails ? 1 : 0;
