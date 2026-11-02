// Line diff (Myers' O((N+M)·D) algorithm after trimming the common head and tail) for the chat cards,
// the version history and the lines an edit changed. Pure: runs in Node for the tests.

// ops: [{ t: ' ' | '-' | '+', s: line }]
export function diffLines(a, b) {
  let head = 0;
  while (head < a.length && head < b.length && a[head] === b[head]) head++;
  let tail = 0;
  while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++;
  const A = a.slice(head, a.length - tail), B = b.slice(head, b.length - tail);
  const ops = [];
  for (let i = 0; i < head; i++) ops.push({ t: ' ', s: a[i] });
  ops.push(...(A.length + B.length > 6000 ? crude(A, B) : myers(A, B)));
  for (let i = a.length - tail; i < a.length; i++) ops.push({ t: ' ', s: a[i] });
  return ops;
}

function crude(A, B) {
  return [...A.map((s) => ({ t: '-', s })), ...B.map((s) => ({ t: '+', s }))];
}

function myers(A, B) {
  const N = A.length, M = B.length;
  if (!N) return B.map((s) => ({ t: '+', s }));
  if (!M) return A.map((s) => ({ t: '-', s }));
  const MAX = N + M, off = MAX;
  const V = new Int32Array(2 * MAX + 2);
  const trace = [];
  for (let d = 0; d <= MAX; d++) {
    trace.push(V.slice());
    for (let k = -d; k <= d; k += 2) {
      let x = k === -d || (k !== d && V[off + k - 1] < V[off + k + 1]) ? V[off + k + 1] : V[off + k - 1] + 1;
      let y = x - k;
      while (x < N && y < M && A[x] === B[y]) { x++; y++; }
      V[off + k] = x;
      if (x >= N && y >= M) return backtrack(trace, A, B, d, off);
    }
  }
  return crude(A, B);
}

function backtrack(trace, A, B, D, off) {
  const out = [];
  let x = A.length, y = B.length;
  for (let d = D; d > 0; d--) {
    const V = trace[d];
    const k = x - y;
    const down = k === -d || (k !== d && V[off + k - 1] < V[off + k + 1]);
    const kPrev = down ? k + 1 : k - 1;
    const xPrev = V[off + kPrev], yPrev = xPrev - kPrev;
    while (x > xPrev && y > yPrev) { out.push({ t: ' ', s: A[x - 1] }); x--; y--; }
    if (down) { out.push({ t: '+', s: B[y - 1] }); y--; } else { out.push({ t: '-', s: A[x - 1] }); x--; }
  }
  while (x > 0 && y > 0) { out.push({ t: ' ', s: A[x - 1] }); x--; y--; }
  while (x > 0) { out.push({ t: '-', s: A[x - 1] }); x--; }
  while (y > 0) { out.push({ t: '+', s: B[y - 1] }); y--; }
  return out.reverse();
}

export function diffCounts(ops) {
  let added = 0, removed = 0;
  for (const o of ops) { if (o.t === '+') added++; else if (o.t === '-') removed++; }
  return { added, removed };
}

// 1-based inclusive line ranges of the new text that were added or sit where lines were removed
export function changedRanges(ops) {
  const ranges = [];
  let line = 0, open = null;
  const mark = (ln) => {
    if (open && ln <= open[1] + 1) open[1] = Math.max(open[1], ln);
    else { open = [ln, ln]; ranges.push(open); }
  };
  let total = 0;
  for (const o of ops) if (o.t !== '-') total++;
  for (const o of ops) {
    if (o.t === ' ') { line++; continue; }
    if (o.t === '+') { line++; mark(line); continue; }
    mark(Math.max(1, Math.min(total, line + 1))); // removed: the line that now sits where it was
  }
  return ranges;
}

export function describeRanges(ranges) {
  return ranges.map(([a, b]) => (a === b ? `${a}` : `${a}–${b}`)).join(', ');
}

// unified diff text with `context` lines around each change
export function unifiedDiff(ops, context = 3) {
  const out = [];
  let i = 0;
  let oldLine = 1, newLine = 1;
  while (i < ops.length) {
    if (ops[i].t === ' ') { i++; oldLine++; newLine++; continue; }
    // hunk: from context before the change to context after the last change within reach
    let start = i;
    let ctxStart = Math.max(0, start - context);
    let end = i;
    let j = i;
    while (j < ops.length) {
      if (ops[j].t !== ' ') { end = j; j++; continue; }
      let run = 0;
      while (j + run < ops.length && ops[j + run].t === ' ') run++;
      if (run > 2 * context || j + run >= ops.length) break;
      j += run;
    }
    const ctxEnd = Math.min(ops.length, end + 1 + context);
    let o0 = oldLine - (start - ctxStart), n0 = newLine - (start - ctxStart);
    let oc = 0, nc = 0;
    const lines = [];
    for (let k = ctxStart; k < ctxEnd; k++) {
      const o = ops[k];
      lines.push(`${o.t}${o.s}`);
      if (o.t !== '+') oc++;
      if (o.t !== '-') nc++;
    }
    out.push(`@@ -${o0},${oc} +${n0},${nc} @@`, ...lines);
    for (let k = i; k < ctxEnd; k++) { if (ops[k].t !== '+') oldLine++; if (ops[k].t !== '-') newLine++; }
    i = ctxEnd;
  }
  return out.join('\n');
}
