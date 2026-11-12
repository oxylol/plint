// Ultron · stand-ins for the two VoiceStudios, so his voice logic can be tested without the real ones.
// :5940 is the slow local one, :5941 the fast one (the PC). Each answers /health, /profiles and the licence
// check, and "speaks" a short tone for /v1/audio/speech. Set how it behaves with /_set on its own port:
//
//   /_set?delay=150      every sentence takes this many ms
//   /_set?once=15000     the next sentence alone takes this much longer (a model that has to load)
//   /_set?dead=true      it stops answering at all       /_set?health=false   it answers, but says it is not well
//   /_log                what it was asked to say since the last look, how long each took, and whether it was sent
//
// What was checked with it on 2026-11-12 (a scratch server with VOICE_URL :5940 and VOICE_FAST_URL :5941):
// a cold model (once=15000) is waited for on the fast voice; a slow sentence (once=10000) is given up after
// 8 s and asked again there; dead=true sends the answer to the local voice; /api/wake warms a cold one.
'use strict';
const http = require('node:http');

function wav(sec) {
  const rate = 16000, n = Math.round(rate * sec), b = Buffer.alloc(44 + n * 2);
  b.write('RIFF', 0); b.writeUInt32LE(36 + n * 2, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(rate, 24); b.writeUInt32LE(rate * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.round(9000 * Math.sin(2 * Math.PI * 180 * i / rate)), 44 + i * 2);
  return b;
}

function make(port, name) {
  const s = { delay: 150, once: 0, dead: false, health: true };
  let log = [];
  http.createServer((req, res) => {
    const u = new URL(req.url, 'http://x');
    const j = (code, o) => { res.writeHead(code, { 'content-type': 'application/json' }); res.end(JSON.stringify(o)); };
    if (u.pathname === '/_set') { for (const [k, v] of u.searchParams) s[k] = v === 'true' ? true : v === 'false' ? false : Number(v); return j(200, s); }
    if (u.pathname === '/_log') { const l = log; log = []; return j(200, l); }
    if (s.dead) return req.socket.destroy();
    if (u.pathname === '/health') return s.health ? j(200, { status: 'ok', device: 'mock gpu', engine: 'omnivoice' }) : j(503, { status: 'no' });
    if (u.pathname === '/profiles') return j(200, [{ name: 'Ultron' }]);
    if (u.pathname === '/engines/tts') return j(200, { active_model: 'k2-fsa/OmniVoice' });
    if (u.pathname.startsWith('/models/licenses/acceptance/')) return j(200, { required: true, accepted: true });
    if (u.pathname === '/v1/audio/speech' && req.method === 'POST') {
      let body = '';
      req.on('data', (c) => { body += c; });
      req.on('end', () => {
        const input = JSON.parse(body).input, wait = s.delay + s.once, t0 = Date.now();
        s.once = 0;
        const timer = setTimeout(() => { log.push({ name, input, ms: Date.now() - t0, sent: true }); res.writeHead(200, { 'content-type': 'audio/wav' }); res.end(wav(Math.min(2.5, 0.25 + input.length / 60))); }, wait);
        res.on('close', () => { if (!res.writableEnded) { clearTimeout(timer); log.push({ name, input, ms: Date.now() - t0, sent: false }); } });
      });
      return;
    }
    j(404, { error: 'no' });
  }).listen(port, '127.0.0.1', () => console.log(`${name} voice on :${port}`));
}
make(5940, 'local');
make(5941, 'fast');
