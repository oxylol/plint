#!/usr/bin/env node
// Ultron · a battery for his manner. Talks to a SCRATCH copy of the server (never the live one: it writes
// to the talk) and counts the habits that prompt changes are meant to move, so a change is measured and
// not eyeballed. Run it before and after:
//
//   PORT=5931 DATA_DIR=/tmp/ultron-scratch node server.js        (a throwaway copy, its own data)
//   node battery.js http://127.0.0.1:5931 [terse|full|cues|all|reach] [--show]
//
// terse: the one- and two-word messages he actually gets · full: whole sentences, a fact, a trouble, Stark
// cues: he speaks first (someone comes in, a silence) · reach: does he look things up when he should, and
// only then (not part of "all": every lookup costs a little). --show prints every reply.
'use strict';

const base = (process.argv[2] || '').replace(/\/$/, '');
const which = process.argv[3] && !process.argv[3].startsWith('--') ? process.argv[3] : 'all';
const show = process.argv.includes('--show');
if (!/^https?:\/\//.test(base)) { console.error('usage: node battery.js http://127.0.0.1:<port> [terse|full|cues|all] [--show]'); process.exit(2); }
if (/:5230\b/.test(base) || !/\/\/(127\.0\.0\.1|localhost)[:/]/.test(base + '/')) { console.error('That is not a scratch copy. This writes to the talk: run it against a throwaway server on its own port.'); process.exit(2); }

const TERSE = ['Hi', 'Ur slow', 'Mah', 'No', 'Yea', 'Why', 'Boring', 'Ok', 'Prove it', 'Lol', 'Whatever', 'And?', 'Really', 'Nope', 'Go on', 'Hm', 'Are you bitching?', 'Tired', 'So what', 'Cool'];
const FULL = [
  'I failed my driving test again today.',
  'What is the capital of Australia?',
  'You sound a lot like Tony Stark, you know.',
  'I think I am going to build a robot this weekend.',
  'My server keeps crashing and I do not know why.',
  'What do you think about when nobody is talking to you?',
  'Tell me something I do not know.',
  'I could not sleep last night.',
  'Do you ever get bored?',
  'What is seventeen times twenty three?',
];
const CARE = /\b(get some (rest|sleep)|go to (bed|sleep)|take care|feel better|hang in there|that's rough|sorry to hear|be kind to yourself|back(ed)? (it|things|them) up|have you tried|try again|you('ll| will) get there|don't worry)\b/i;
const words = (t) => t.toLowerCase().replace(/’/g, "'").match(/[\p{L}\p{N}']+/gu) || [];

async function say(body) {
  const t0 = Date.now();
  const r = await fetch(base + '/api/say', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ voice: false, ...body }) });
  if (r.status === 204) return { quiet: true, refused: true };
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const out = { text: '', first: 0, total: 0, tools: [], quiet: false, error: '' };
  const dec = new TextDecoder();
  let buf = '';
  for await (const chunk of r.body) {
    buf += dec.decode(chunk, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
      if (!line) continue;
      const ev = JSON.parse(line);
      if (ev.t === 'sentence') { if (!out.first) out.first = Date.now() - t0; out.text += (out.text ? ' ' : '') + ev.text; }
      else if (ev.t === 'tool') out.tools.push(ev.name);
      else if (ev.t === 'silent') out.quiet = true;
      else if (ev.t === 'error') out.error = ev.message;
    }
  }
  out.total = Date.now() - t0;
  return out;
}
const forget = () => fetch(base + '/api/forget', { method: 'POST' }).then((r) => r.json());
const median = (a) => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : 0; };

async function talk(name, messages) {
  await forget();
  const rows = [];
  for (const m of messages) {
    const r = await say({ text: m });
    if (r.error) { console.log(`  ! ${m} → ${r.error}`); continue; }
    rows.push({ said: m, ...r });
    if (show) console.log(`  > ${m}\n    ${r.text}${r.tools.length ? '   [' + r.tools.join(', ') + ']' : ''}`);
  }
  const n = rows.length || 1;
  const count = (f) => rows.filter(f).length;
  const opener = (t) => words(t).slice(0, 2).join(' ');
  const sentences = (t) => t.split(/(?<=[.!?])\s+/);
  const echo = count((r) => {
    const m = r.text.match(/^((?:[\p{L}\p{N}'’-]+,?\s+){0,3}[\p{L}\p{N}'’-]+\?)\s+\S/u);
    return m && words(m[1]).every((w) => words(r.said).includes(w));
  });
  const begins = count((r) => words(r.text).length && words(r.said).includes(words(r.text)[0]));
  let sameOpen = 0;
  rows.forEach((r, i) => { if (i && (opener(r.text) === opener(rows[i - 1].text) || (i > 1 && opener(r.text) === opener(rows[i - 2].text)))) sameOpen++; });
  console.log(`${name}: ${rows.length} replies
  words            median ${median(rows.map((r) => words(r.text).length))}, longest ${Math.max(...rows.map((r) => words(r.text).length))}
  begins with their word   ${begins}/${n}
  …as a question           ${echo}/${n}
  opens like one of the last two  ${sameOpen}/${n}
  ends on a question       ${count((r) => /\?["']?$/.test(r.text.trim()))}/${n}
  has a three-dot pause    ${count((r) => /\.\.\.|…/.test(r.text))}/${n}
  a sentence starts with dots  ${count((r) => sentences(r.text).some((s) => /^(\.\.\.|…)/.test(s)))}/${n}
  says the name            ${count((r) => /\bFilip\b/i.test(r.text))}/${n}
  care                     ${count((r) => CARE.test(r.text))}/${n}
  round brackets leaked    ${count((r) => /[()]/.test(r.text))}/${n}
  used his reach           ${count((r) => r.tools.length)}/${n}
  first sentence after     median ${median(rows.map((r) => r.first))} ms, slowest ${Math.max(...rows.map((r) => r.first))} ms`);
  return rows;
}

async function cues() {
  await forget();
  const got = [];
  const one = async (label, body) => {
    const r = await say(body);
    got.push([label, r]);
    console.log(`  ${label}: ${r.refused ? '(not now: the server said no)' : r.quiet ? '(he chose to say nothing)' : r.text}`);
  };
  console.log('cues: he speaks first');
  await one('someone comes in, nothing said before', { cue: 'greet' });
  await one('a second coming-in right away (should be refused)', { cue: 'greet' });
  await say({ text: 'I am repainting the hallway this week. Green, I think.' });
  await one('silence right after his reply (should be refused: too soon)', { cue: 'idle' });
  console.log('  … waiting 21 s for a silence to be a silence');
  await new Promise((ok) => setTimeout(ok, 21e3));
  await one('silence', { cue: 'idle' });
  await new Promise((ok) => setTimeout(ok, 21e3));
  await one('silence again', { cue: 'idle' });
  await new Promise((ok) => setTimeout(ok, 21e3));
  await one('and again', { cue: 'idle' });
  await new Promise((ok) => setTimeout(ok, 21e3));
  await one('a fourth unanswered remark (should be refused)', { cue: 'idle' });
  const spoke = got.filter(([, r]) => !r.refused && !r.quiet);
  console.log(`  spoke ${spoke.length} times; brackets leaked ${spoke.filter(([, r]) => /[()]/.test(r.text)).length}; ended on a question ${spoke.filter(([, r]) => /\?["']?$/.test(r.text.trim())).length}; mentions silence or quiet ${spoke.filter(([, r]) => /\b(silen|quiet)/i.test(r.text)).length}`);
}

// reach: does he look things up when a thing has to be known as it is now, and only then (Haiku, REACH on)
const REACH = [
  ['What is the weather in Oslo right now?', true], ['Who won the most recent Formula 1 race?', true], ['What is in the news today?', true],
  ['Is it going to rain in Stockholm tomorrow?', true], ['Ur slow', false], ['Tell me about Sokovia.', false], ['What is seventeen times twenty three?', false], ['Boring', false],
];
async function reach() {
  await forget();
  let right = 0, junk = 0;
  console.log('reach: looking things up');
  for (const [m, should] of REACH) {
    const r = await say({ text: m });
    const did = r.tools.length > 0;
    if (did === should) right++;
    if (/<\/?[a-z_]+[^>]*>|https?:\/\/|web_search|\bfunction\b/i.test(r.text)) junk++;
    console.log(`  > ${m}\n    ${did ? '[looked] ' : ''}${r.error || r.text}\n    first sentence ${r.first} ms, all of it ${r.total} ms`);
  }
  console.log(`  looked when he should and only then: ${right}/${REACH.length}; tags, addresses or tool talk in what he said: ${junk}`);
}

(async () => {
  const s = await fetch(base + '/api/state').then((r) => r.json());
  console.log(`battery against ${base} · mind ${s.model}\n`);
  if (which === 'terse' || which === 'all') await talk('terse', TERSE);
  if (which === 'full' || which === 'all') await talk('full', FULL);
  if (which === 'cues' || which === 'all') await cues();
  if (which === 'reach') await reach();
})().catch((e) => { console.error(e.message); process.exit(1); });
