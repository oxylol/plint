/**
 * The body rig's mesh (src/rig/body.rig.bin): one of MakeHuman's low-poly bodies, a mesh an artist
 * laid out by hand (loops round the shoulders, the chest, the joints), fitted to the adult male,
 * skinned to MakeHuman's default rig cut down to the bones that shape a body (twist bones kept),
 * and cut off at a loop round the neck for the head's skin to be joined to. Its own edges are
 * what the rig's view draws.
 *
 * MakeHuman's mesh, rig, weights, targets and proxies are CC0. The base mesh and rig come from
 * https://github.com/makehumancommunity/makehuman (LICENSE.ASSETS.md); the low-poly proxies from
 * the CC0 system assets pack ("explicitly released as CC0 in september 2020"):
 * https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip
 * The copies on the old download mirror still say AGPL: not those.
 *
 * A proxy's vertex is three of the base mesh's, weighted, plus an offset: so it takes the base
 * mesh's shape and, the same way, its skin weights. The rig keeps 38 of the 163 bones: the spine,
 * neck and head, and down each arm and leg every bone MakeHuman has, twist bones included
 * (upperarm01/02, lowerarm01/02, upperleg01/02, lowerleg01/02). Finger bones fold into the
 * wrist, the face's into the head, the breasts into spine02; the toes become one bone each side.
 *
 * It's subdivided once (Catmull-Clark: the same loops, twice as many), each new vertex then put
 * back on the base mesh's surface, so the finer mesh is the body's shape and not the coarse one's
 * rounded off.
 *
 * Reproducible: the base mesh, targets and rig come from commit-pinned URLs, and the proxy (whose
 * pack has no versioned URL) is checked against the hashes it was made from. Inputs are cached
 * in ~/.cache/chimera-rig.
 *
 * Layout (little endian): u32 header bytes, the header's JSON (padded to 4 bytes), then
 *   f32 positions[n][3] (cm), u8 skinIndex[n][8], u16 skinWeight[n][8] (sum 65535),
 *   u16 quads[q][4].
 * Body cm: x forward, y up, z out of the right side (as the CHIMERA head's).
 *
 * Usage: bun tools/rig/gen-body.ts [out.bin]   (default: src/rig/body.rig.bin)
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";

const MH_COMMIT = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482", MPFB_COMMIT = "9de0707fdc88438d7c53a1a2ac7937d9221e2431";
const SRC = `https://raw.githubusercontent.com/makehumancommunity/makehuman/${MH_COMMIT}/makehuman/data`;
/** downloads, by the commit they came from (a cache from another commit isn't read) */
const CACHE = `${process.env.HOME}/.cache/chimera-rig/makehuman-${MH_COMMIT.slice(0, 12)}`;
const MPFB_CACHE = `${process.env.HOME}/.cache/chimera-rig/mpfb2-${MPFB_COMMIT.slice(0, 12)}`;
const PACK = "https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip";
const PACK_DIR = `${process.env.HOME}/.cache/chimera-rig/mh-cc0`;
const OUT = process.argv[2] ?? new URL("../../src/rig/body.rig.bin", import.meta.url);
const LEVELS = 1;
const PROXY = "proxy741";
/** the proxy's files the mesh was made from (sha256): its fitting to the base mesh, and its quads */
const PROXY_SHA256: Record<string, string> = {
  proxy: "1dc925fe62f28b1e55e02528e3b7ad36fa7283f1128b326889f85b86efa5e05e",
  obj: "002a1dac4e912ca6c0a123a5bf4c43d4ce0db5152ea4b3452a0bead5e6ffad4b",
};
const proxyPath = `${PACK_DIR}/proxymeshes/${PROXY}/${PROXY}`;
if (!existsSync(`${proxyPath}.obj`)) {
  mkdirSync(PACK_DIR, { recursive: true });
  if (!existsSync(`${PACK_DIR}/pack.zip`)) {
    const r = await fetch(PACK);
    if (!r.ok) throw new Error(`${PACK}: ${r.status}`);
    writeFileSync(`${PACK_DIR}/pack.zip`, new Uint8Array(await r.arrayBuffer()));
  }
  execFileSync("unzip", ["-o", "-q", `${PACK_DIR}/pack.zip`, `proxymeshes/${PROXY}/*`, "-d", PACK_DIR]);
}
for (const [ext, want] of Object.entries(PROXY_SHA256)) {
  const got = createHash("sha256").update(readFileSync(`${proxyPath}.${ext}`)).digest("hex");
  if (got !== want) throw new Error(`${PROXY}.${ext} isn't the one the rig was made from (sha256 ${got}, want ${want}): the pack changed, or the cache is stale (${PACK_DIR})`);
}

const get = async (path: string) => {
  mkdirSync(CACHE, { recursive: true });
  const file = `${CACHE}/${path.replaceAll("/", "__")}`;
  if (existsSync(file)) return readFileSync(file, "utf8");
  const r = await fetch(`${SRC}/${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  const text = await r.text();
  writeFileSync(file, text);
  return text;
};

type V3 = [number, number, number];

// ------------------------------------------------------------------ the base mesh, as the adult male
const base: V3[] = [];
const baseQuads: number[][] = [];
/** the joint cubes MakeHuman places in each body (helper geometry): their vertices */
const cubes = new Map<string, Set<number>>();
let group = "";
for (const line of (await get("3dobjs/base.obj")).split("\n")) {
  const w = line.trim().split(/\s+/);
  if (w[0] === "v") base.push([+w[1], +w[2], +w[3]]);
  else if (w[0] === "g") group = w[1];
  else if (w[0] === "f") {
    const f = w.slice(1).map((s) => parseInt(s, 10) - 1);
    if (group === "body") baseQuads.push(f);
    else if (group.startsWith("joint-")) {
      if (!cubes.has(group)) cubes.set(group, new Set());
      for (const v of f) cubes.get(group)!.add(v);
    }
  }
}
const mh = base.map((p) => [...p] as V3);
for (const race of ["african", "asian", "caucasian"]) {
  for (const line of (await get(`targets/macrodetails/${race}-male-young.target`)).split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const [i, dx, dy, dz] = line.split(/\s+/).map(Number);
    mh[i][0] += dx / 3;
    mh[i][1] += dy / 3;
    mh[i][2] += dz / 3;
  }
}
/** MakeHuman decimetres, facing +z with its left on +x → body cm */
const cm = ([x, y, z]: V3): V3 => [z * 10, y * 10, -x * 10];
const male = mh.map(cm);

// ------------------------------------------------------------------ the rig
// MakeHuman's "Rigify human" (MPFB2): the one of its rigs whose weights were tuned for how the body
// deforms, with helper bones at the shoulder, elbow, knee and hip. Its weights are on the base
// mesh, so the proxy takes them as it takes the shape. CC0 (mpfb2 LICENSE.md, assets).
const MPFB = `https://raw.githubusercontent.com/makehumancommunity/mpfb2/${MPFB_COMMIT}/src/mpfb/data/rigs/rigify`;
const getMpfb = async (file: string) => {
  mkdirSync(MPFB_CACHE, { recursive: true });
  const path = `${MPFB_CACHE}/rigify__${file}`;
  if (existsSync(path)) return readFileSync(path, "utf8");
  const r = await fetch(`${MPFB}/${file}`);
  if (!r.ok) throw new Error(`${file}: ${r.status}`);
  const text = await r.text();
  writeFileSync(path, text);
  return text;
};
type End = { strategy: string; cube_name?: string; vertex_index?: number; vertex_indices?: number[] };
const rigify = JSON.parse(await getMpfb("rig.human_toes.json")) as { bones: Record<string, { head: End; tail: End; parent: string }> };
const weights = (JSON.parse(await getMpfb("weights.human_toes.json")) as { weights: Record<string, [number, number][]> }).weights;
const mean = (ids: Iterable<number>): V3 => {
  const c: V3 = [0, 0, 0];
  let n = 0;
  for (const i of ids) {
    for (let k = 0; k < 3; k++) c[k] += male[i][k];
    n++;
  }
  return [c[0] / n, c[1] / n, c[2] / n];
};
const end = (e: End): V3 => {
  if (e.strategy === "CUBE") return mean(cubes.get(e.cube_name!)!);
  if (e.strategy === "VERTEX") return [...male[e.vertex_index!]] as V3;
  if (e.strategy === "MEAN") return mean(e.vertex_indices!);
  throw new Error(`no place for ${JSON.stringify(e)}`);
};
const cube = (name: string) => mean(cubes.get(name)!);
const midpoint = (a: V3, b: V3): V3 => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
/**
 * The rig's bones, named for what they are, each from a Rigify bone: its ends, and the weights
 * that go to it. A twist bone (…2) is the second half of its limb bone (Rigify splits each limb
 * bone in two at its middle). Helpers are posed by the rig from their neighbours' turns.
 */
interface BoneSpec { name: string; parent: string | null; from: string; head?: V3; tail?: V3; helper?: boolean }
const SPECS: BoneSpec[] = [
  { name: "hips", parent: null, from: "spine" },
  { name: "spine1", parent: "hips", from: "spine.001" },
  { name: "spine2", parent: "spine1", from: "spine.002" },
  { name: "chest", parent: "spine2", from: "spine.003" },
  { name: "upperchest", parent: "chest", from: "spine.004" },
  { name: "neck", parent: "upperchest", from: "spine.005" },
  { name: "head", parent: "neck", from: "spine.006" },
];
for (const S of ["R", "L"]) {
  const r = (n: string) => rigify.bones[`${n}.${S}`];
  const half = (n: string): [V3, V3, V3] => {
    const h = end(r(n).head), t = end(r(n).tail);
    return [h, midpoint(h, t), t];
  };
  const [ua0, ua1, ua2] = half("upper_arm"), [fa0, fa1, fa2] = half("forearm"), [th0, th1, th2] = half("thigh"), [sh0, sh1, sh2] = half("shin");
  SPECS.push(
    { name: `clavicle.${S}`, parent: "upperchest", from: `shoulder.${S}` },
    { name: `upperarm.${S}`, parent: `clavicle.${S}`, from: `upper_arm.${S}`, head: ua0, tail: ua1 },
    { name: `upperarm2.${S}`, parent: `upperarm.${S}`, from: `DEF-upper_arm.${S}.001`, head: ua1, tail: ua2 },
    { name: `forearm.${S}`, parent: `upperarm2.${S}`, from: `forearm.${S}`, head: fa0, tail: fa1 },
    { name: `forearm2.${S}`, parent: `forearm.${S}`, from: `DEF-forearm.${S}.001`, head: fa1, tail: fa2 },
    { name: `hand.${S}`, parent: `forearm2.${S}`, from: `hand.${S}` },
    { name: `pelvis.${S}`, parent: "hips", from: `pelvis.${S}` },
    { name: `thigh.${S}`, parent: "hips", from: `thigh.${S}`, head: th0, tail: th1 },
    { name: `thigh2.${S}`, parent: `thigh.${S}`, from: `DEF-thigh.${S}.001`, head: th1, tail: th2 },
    { name: `shin.${S}`, parent: `thigh2.${S}`, from: `shin.${S}`, head: sh0, tail: sh1 },
    { name: `shin2.${S}`, parent: `shin.${S}`, from: `DEF-shin.${S}.001`, head: sh1, tail: sh2 },
    { name: `foot.${S}`, parent: `shin2.${S}`, from: `foot.${S}` },
    { name: `toes.${S}`, parent: `foot.${S}`, from: `toe.${S}` },
    { name: `shoulderHelper.${S}`, parent: `clavicle.${S}`, from: `DEF-shoulder-helper.${S}`, helper: true },
    { name: `elbowHelper.${S}`, parent: `upperarm2.${S}`, from: `DEF-elbow-helper.${S}`, helper: true },
    { name: `kneeHelper.${S}`, parent: `thigh2.${S}`, from: `DEF-knee-helper.${S}`, helper: true },
    { name: `hipHelper.${S}`, parent: "hips", from: `DEF-pelvis-helper.${S}`, helper: true },
    { name: `hipFront.${S}`, parent: "hips", from: `DEF-pelvis-helper.front.${S}`, helper: true },
  );
}
const KEEP = SPECS.map((s) => s.name);
const NB = KEEP.length;
const keepIndex = new Map(KEEP.map((n, i) => [n, i]));
const parentOf = (name: string): number => {
  const p = SPECS[keepIndex.get(name)!].parent;
  return p ? keepIndex.get(p)! : -1;
};
const ends = (name: string): { head: V3; tail: V3 } => {
  const s = SPECS[keepIndex.get(name)!];
  const rb = rigify.bones[s.from] ?? rigify.bones[s.from.replace(/^DEF-(.*)\.([LR])\.001$/, "$1.$2")];
  return { head: s.head ?? end(rb.head), tail: s.tail ?? end(rb.tail) };
};
/** which of the rig's bones a Rigify weight group goes to */
const BY_FROM = new Map(SPECS.map((s) => [s.from, s.name]));
const owner = (group: string): number | null => {
  if (group.startsWith("mhmask-")) return null;
  const direct = BY_FROM.get(group);
  if (direct) return keepIndex.get(direct)!;
  const side = /\.([LR])(\.\d+)?$/.exec(group)?.[1];
  if (side && /^(f_|thumb|palm)/.test(group)) return keepIndex.get(`hand.${side}`)!;
  if (side && /^toe/.test(group)) return keepIndex.get(`toes.${side}`)!;
  if (side && /^breast/.test(group)) return keepIndex.get("chest")!;
  // the face's bones, the eyes, the jaw, the tongue and teeth: the head's
  return keepIndex.get("head")!;
};
const folded = new Map<string, string>();
const baseW = new Float64Array(base.length * NB);
for (const [group, list] of Object.entries(weights)) {
  const b = owner(group);
  if (b === null) continue;
  folded.set(group, KEEP[b]);
  for (const [vi, w] of list) baseW[vi * NB + b] += w;
}
console.log("weights folded:", [...new Set(folded.values())].length, "bones from", folded.size, "groups; to the head:", [...folded].filter(([, b]) => b === "head").map(([g]) => g).length);
// the cut at the neck is where the lotus film's was: from MakeHuman's default rig's neck joint
const skel = JSON.parse(await get("rigs/default.mhskel")) as { joints: Record<string, number[]> };
const neck01 = mean(skel.joints["neck01____head"]);
const fingers = (side: string) => ({
  [`finger1-1.${side}`]: cube(`joint-${side.toLowerCase()}-finger-1-3`),
  [`finger3-1.${side}`]: cube(`joint-${side.toLowerCase()}-finger-3-1`),
  [`finger3-4.${side}`]: cube(`joint-${side.toLowerCase()}-finger-3-4`),
  [`finger5-1.${side}`]: cube(`joint-${side.toLowerCase()}-finger-5-1`),
});

// ------------------------------------------------------------------ the proxy
const scale: Record<string, number> = {};
const refs: { v: number[]; w: number[]; d: number[] }[] = [];
let inVerts = false;
for (const line of readFileSync(`${proxyPath}.proxy`, "utf8").split("\n")) {
  const w = line.trim().split(/\s+/);
  if (!w[0] || w[0].startsWith("#")) continue;
  if (/^[xyz]_scale$/.test(w[0])) {
    const axis = "xyz".indexOf(w[0][0]);
    scale[w[0][0]] = Math.abs(mh[+w[1]][axis] - mh[+w[2]][axis]) / +w[3];
  } else if (w[0] === "verts") inVerts = true;
  else if (inVerts && /^-?\d/.test(w[0])) {
    if (w.length === 1) refs.push({ v: [+w[0], +w[0], +w[0]], w: [1, 0, 0], d: [0, 0, 0] });
    else refs.push({ v: w.slice(0, 3).map(Number), w: w.slice(3, 6).map(Number), d: w.slice(6, 9).map(Number) });
  } else inVerts = false;
}
let quads: number[][] = [];
for (const line of readFileSync(`${proxyPath}.obj`, "utf8").split("\n")) {
  const w = line.trim().split(/\s+/);
  if (w[0] === "f") quads.push(w.slice(1).map((s) => parseInt(s, 10) - 1));
}
if (quads.some((q) => q.length !== 4)) throw new Error("the proxy is not all quads");
/** each vertex: position (3) then the kept bones' weights */
const D = 3 + NB;
let V = new Float64Array(refs.length * D);
refs.forEach((r, i) => {
  const p: V3 = [0, 0, 0];
  for (let j = 0; j < 3; j++) {
    for (let k = 0; k < 3; k++) p[k] += r.w[j] * mh[r.v[j]][k];
    for (let b = 0; b < NB; b++) V[i * D + 3 + b] += r.w[j] * baseW[r.v[j] * NB + b];
  }
  p[0] += r.d[0] * scale.x;
  p[1] += r.d[1] * scale.y;
  p[2] += r.d[2] * scale.z;
  V.set(cm(p), i * D);
});

// ------------------------------------------------------------------ the cut at the neck
// The head's neck ends in a level ring just under the chin (as the dancer's cut: the lowest point
// of the midline well forward of the throat, less a little). The mesh is cut at the loop of its
// own edges round the neck that lies nearest that level, so the rim is a row of whole quads' tops.
const BODY = 13380;
const neckY = neck01[1];
const mid = male.slice(0, BODY).filter((p) => Math.abs(p[2]) < 1);
const throat = Math.max(...mid.filter((p) => Math.abs(p[1] - neckY) < 0.5).map((p) => p[0]));
const CUT = Math.min(...mid.filter((p) => p[0] > throat + 3 && p[1] > neckY).map((p) => p[1])) - 0.7;
const neckMid: V3 = [0, 0, 0];
{
  const ring = male.slice(0, BODY).filter((p) => Math.abs(p[1] - (CUT - 2)) < 1);
  neckMid[0] = (Math.max(...ring.map((p) => p[0])) + Math.min(...ring.map((p) => p[0]))) / 2;
}
const edgeKey = (a: number, b: number) => (a < b ? a * 1e6 + b : b * 1e6 + a);
const topology = (qs: number[][], n: number) => {
  const edgeFaces = new Map<number, number[]>();
  const near: Set<number>[] = Array.from({ length: n }, () => new Set());
  qs.forEach((q, f) => {
    for (let j = 0; j < 4; j++) {
      const a = q[j], b = q[(j + 1) % 4], k = edgeKey(a, b);
      edgeFaces.set(k, [...(edgeFaces.get(k) ?? []), f]);
      near[a].add(b);
      near[b].add(a);
    }
  });
  return { edgeFaces, near };
};
{
  const { edgeFaces, near } = topology(quads, refs.length);
  const y = (i: number) => V[i * D + 1];
  /** the edge a loop carries on along through a vertex of four edges: the one sharing no quad with the one it came by */
  const onward = (from: number, at: number): number | null => {
    if (near[at].size !== 4) return null;
    const came = new Set(edgeFaces.get(edgeKey(from, at)));
    for (const to of near[at]) if (to !== from && !edgeFaces.get(edgeKey(at, to))!.some((f) => came.has(f))) return to;
    return null;
  };
  const loops = new Map<string, number[]>();
  for (let a = 0; a < refs.length; a++) {
    if (Math.abs(y(a) - CUT) > 6) continue;
    for (const b of near[a]) {
      const loop = [a, b];
      for (;;) {
        const to = onward(loop[loop.length - 2], loop[loop.length - 1]);
        if (to === null || loop.length > 200) break;
        if (to === a) {
          if (onward(loop[loop.length - 1], a) === b) loops.set([...loop].sort((p, q) => p - q).join(","), loop);
          break;
        }
        loop.push(to);
      }
    }
  }
  // the loops that go once round the neck's axis
  const round = [...loops.values()]
    .map((loop) => {
      let turn = 0;
      const ang = (i: number) => Math.atan2(V[i * D + 2], V[i * D] - neckMid[0]);
      loop.forEach((v, j) => {
        let d = ang(loop[(j + 1) % loop.length]) - ang(v);
        d -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
        turn += d;
      });
      const ys = loop.map(y);
      return { loop, turn: Math.round(turn / (2 * Math.PI)), mean: ys.reduce((s, v) => s + v, 0) / ys.length, lo: Math.min(...ys), hi: Math.max(...ys) };
    })
    .filter((l) => Math.abs(l.turn) === 1)
    .sort((p, q) => Math.abs(p.mean - CUT) - Math.abs(q.mean - CUT));
  console.log(`cut ${CUT.toFixed(2)} cm (neck01 ${neckY.toFixed(2)}); loops round the neck:`);
  for (const l of round.slice(0, 8)) console.log(`  ${l.loop.length} vertices, mean ${l.mean.toFixed(2)}, ${l.lo.toFixed(2)} to ${l.hi.toFixed(2)}`);
  const cut = round[0];
  if (!cut) throw new Error("no loop round the neck");
  // everything on the head's side of the loop goes: spread from the highest quad, not across the loop
  const barrier = new Set(cut.loop.map((v, j) => edgeKey(v, cut.loop[(j + 1) % cut.loop.length])));
  const gone = new Set<number>();
  let start = 0;
  quads.forEach((q, f) => {
    if (Math.max(...q.map(y)) > Math.max(...quads[start].map(y))) start = f;
  });
  const queue = [start];
  gone.add(start);
  while (queue.length) {
    const q = quads[queue.pop()!];
    for (let j = 0; j < 4; j++) {
      const k = edgeKey(q[j], q[(j + 1) % 4]);
      if (barrier.has(k)) continue;
      for (const g of edgeFaces.get(k)!) {
        if (!gone.has(g)) {
          gone.add(g);
          queue.push(g);
        }
      }
    }
  }
  // the head's own inside (eye sockets, the mouth's bag) and anything else not joined to the body
  quads = quads.filter((_, f) => !gone.has(f));
  const body = new Set<number>();
  const t2 = topology(quads, refs.length);
  let seed = 0;
  quads.forEach((q, f) => {
    if (Math.min(...q.map(y)) < Math.min(...quads[seed].map(y))) seed = f;
  });
  const stack = [seed];
  body.add(seed);
  while (stack.length) {
    const q = quads[stack.pop()!];
    for (let j = 0; j < 4; j++) {
      for (const g of t2.edgeFaces.get(edgeKey(q[j], q[(j + 1) % 4]))!) {
        if (!body.has(g)) {
          body.add(g);
          stack.push(g);
        }
      }
    }
  }
  const kept = quads.filter((_, f) => body.has(f));
  console.log(`${quads.length - kept.length} quads not joined to the body dropped`);
  quads = kept;
  // only the vertices still used
  const remap = new Map<number, number>();
  for (const q of quads) for (const v of q) if (!remap.has(v)) remap.set(v, remap.size);
  const W2 = new Float64Array(remap.size * D);
  for (const [from, to] of remap) W2.set(V.subarray(from * D, from * D + D), to * D);
  V = W2;
  quads = quads.map((q) => q.map((v) => remap.get(v)!));
}

// ------------------------------------------------------------------ finer: Catmull-Clark, put back on the body
const baseTris: number[] = [];
for (const q of baseQuads) for (let j = 1; j + 1 < q.length; j++) baseTris.push(q[0], q[j], q[j + 1]);
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
/** the nearest point of triangle abc to p (Ericson) */
const nearest = (p: V3, a: V3, b: V3, c: V3): V3 => {
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const bp = sub(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
  }
  const cp = sub(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    return [b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])];
  }
  const den = 1 / (va + vb + vc), v = vb * den, w = vc * den;
  return [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w];
};
const triMid: V3[] = [], triRad: number[] = [], triN: V3[] = [];
for (let t = 0; t < baseTris.length; t += 3) {
  const a = male[baseTris[t]], b = male[baseTris[t + 1]], c = male[baseTris[t + 2]];
  const m: V3 = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
  triMid.push(m);
  triRad.push(Math.max(Math.hypot(...sub(a, m)), Math.hypot(...sub(b, m)), Math.hypot(...sub(c, m))));
  const nn = cross(sub(b, a), sub(c, a)), l = Math.hypot(...nn) || 1;
  triN.push([nn[0] / l, nn[1] / l, nn[2] / l]);
}
/** each vertex's normal, from its quads */
const normals = (qs: number[][], n: number): V3[] => {
  const N: V3[] = Array.from({ length: n }, () => [0, 0, 0]);
  const p = (i: number): V3 => [V[i * D], V[i * D + 1], V[i * D + 2]];
  for (const q of qs) {
    const nn = cross(sub(p(q[2]), p(q[0])), sub(p(q[3]), p(q[1])));
    for (const v of q) for (let k = 0; k < 3; k++) N[v][k] += nn[k];
  }
  return N.map((v) => {
    const l = Math.hypot(...v) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  });
};
/** move every vertex (but `keep`) to the nearest point of the base mesh's skin that faces its way */
const project = (qs: number[][], n: number, keep: Set<number>) => {
  const N = normals(qs, n);
  let moved = 0, far = 0;
  for (let i = 0; i < n; i++) {
    if (keep.has(i)) continue;
    const p: V3 = [V[i * D], V[i * D + 1], V[i * D + 2]];
    let best = Infinity, at = p;
    for (let t = 0; t < triMid.length; t++) {
      const dm = Math.hypot(p[0] - triMid[t][0], p[1] - triMid[t][1], p[2] - triMid[t][2]) - triRad[t];
      if (dm > 0 && dm * dm >= best) continue;
      // only skin that faces the way this vertex does: not the next finger's, nor the other side of an armpit
      if (dot(triN[t], N[i]) < 0.2) continue;
      const c = nearest(p, male[baseTris[3 * t]], male[baseTris[3 * t + 1]], male[baseTris[3 * t + 2]]);
      const d = (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 + (c[2] - p[2]) ** 2;
      if (d < best) {
        best = d;
        at = c;
      }
    }
    if (best > 1) far++;
    moved = Math.max(moved, Math.sqrt(best));
    V.set(at, i * D);
  }
  console.log(`  put back on the body: furthest ${moved.toFixed(2)} cm, ${far} over 1 cm`);
};
const rimOf = (qs: number[][], n: number): number[] => {
  const { edgeFaces } = topology(qs, n);
  const next = new Map<number, number>();
  // an open edge runs the way its quad winds
  for (const q of qs) for (let j = 0; j < 4; j++) if (edgeFaces.get(edgeKey(q[j], q[(j + 1) % 4]))!.length === 1) next.set(q[j], q[(j + 1) % 4]);
  const rim: number[] = [];
  if (!next.size) return rim;
  let at = next.keys().next().value as number;
  while (rim.length < next.size) {
    rim.push(at);
    at = next.get(at)!;
  }
  return rim;
};
// ------------------------------------------------------------------ the rim made level
// The mesh's loops round the neck tilt, low at the throat and high at the nape, as the neck leans;
// the head's neck starts from a level ring. So the rim is moved to one height (where the dancer's
// body was cut), and the loops under it take up the difference, each less than the one above: the
// rows stand a little taller down the throat and closer at the nape, and nothing is bunched.
const LEVEL = CUT - 0.65, SPREAD = 3;
/** the edges that run up the neck, between the loops that were moved */
const neckEdges = new Set<number>();
{
  const n = V.length / D;
  const rim = rimOf(quads, n);
  const { near } = topology(quads, n);
  const hops = new Int32Array(n).fill(-1);
  let ring = rim.slice();
  for (const v of ring) hops[v] = 0;
  for (let d = 1; d <= SPREAD; d++) {
    const next: number[] = [];
    for (const v of ring) {
      for (const o of near[v]) {
        if (hops[o] < 0) {
          hops[o] = d;
          next.push(o);
        }
      }
    }
    ring = next;
  }
  const cx = (Math.max(...rim.map((i) => V[i * D])) + Math.min(...rim.map((i) => V[i * D]))) / 2;
  const ang = (i: number) => Math.atan2(V[i * D + 2], V[i * D] - cx);
  const shift = rim.map((i) => ({ a: ang(i), dy: LEVEL - V[i * D + 1] })).sort((p, q) => p.a - q.a);
  /** the rim's shift at a turn round the neck: between the rim vertices either side of it */
  const at = (a: number) => {
    let j = shift.findIndex((s) => s.a > a);
    if (j < 0) j = 0;
    const hi = shift[j], lo = shift[(j + shift.length - 1) % shift.length];
    let span = hi.a - lo.a, t = a - lo.a;
    if (span <= 0) span += 2 * Math.PI;
    if (t < 0) t += 2 * Math.PI;
    return lo.dy + (hi.dy - lo.dy) * (t / span);
  };
  for (let i = 0; i < n; i++) if (hops[i] >= 0) V[i * D + 1] += at(ang(i)) * (1 - hops[i] / (SPREAD + 1));
  for (let i = 0; i < n; i++) for (const o of near[i]) if (hops[i] >= 0 && hops[o] === hops[i] + 1) neckEdges.add(edgeKey(i, o));
  console.log(`rim levelled at ${LEVEL.toFixed(2)}: moved ${Math.min(...shift.map((s) => s.dy)).toFixed(2)} to ${Math.max(...shift.map((s) => s.dy)).toFixed(2)} cm`);
}

// ------------------------------------------------------------------ loops added where the quads are long
// The low-poly mesh has few loops round the arms and legs: quads there four times as long as wide,
// against near squares on the trunk. A strip of quads (each edge, the edge opposite it across the
// next quad, and so on: a band round a limb, or a run along one) whose crossing edges are much
// longer than the strip is wide gets loops put across it, as a modeller would add them, until its
// quads are near square. Fingers and toes are left (MIN): nothing there is big enough to show.
// The neck's strips are cut finer (the last two figures): the head's skin grows from the rim with
// rows about as far apart as the rim's vertices, and the neck under it should not be coarser.
const ASPECT = 1.25, MIN = 3.6, ASPECT_NECK = 0.8, MIN_NECK = 2.4;
{
  const n = V.length / D;
  const { edgeFaces } = topology(quads, n);
  const len = (a: number, b: number) => Math.hypot(V[a * D] - V[b * D], V[a * D + 1] - V[b * D + 1], V[a * D + 2] - V[b * D + 2]);
  const cut = new Map<number, number>();
  const seen = new Set<number>();
  let added = 0;
  for (const start of edgeFaces.keys()) {
    if (seen.has(start)) continue;
    seen.add(start);
    const edges = [start];
    let side = 0, steps = 0;
    for (const first of edgeFaces.get(start)!) {
      let e = start, f = first;
      for (;;) {
        const q = quads[f], j = [0, 1, 2, 3].find((c) => edgeKey(q[c], q[(c + 1) % 4]) === e)!;
        const o = edgeKey(q[(j + 2) % 4], q[(j + 3) % 4]);
        if (seen.has(o)) {
          if (o === start) {
            side += (len(q[(j + 1) % 4], q[(j + 2) % 4]) + len(q[(j + 3) % 4], q[j])) / 2;
            steps++;
          }
          break;
        }
        side += (len(q[(j + 1) % 4], q[(j + 2) % 4]) + len(q[(j + 3) % 4], q[j])) / 2;
        steps++;
        seen.add(o);
        edges.push(o);
        const nf = edgeFaces.get(o)!.find((g) => g !== f);
        if (nf === undefined) break;
        e = o;
        f = nf;
      }
      if (edges.length > 1 && seen.has(start) && steps >= edges.length) break;
    }
    const H = edges.reduce((sum, k) => sum + len(Math.floor(k / 1e6), k % 1e6), 0) / edges.length, wide = side / Math.max(1, steps);
    const neck = edges.some((k) => neckEdges.has(k));
    const pieces = Math.min(6, Math.max(1, Math.round(H / Math.max(wide * (neck ? ASPECT_NECK : ASPECT), neck ? MIN_NECK : MIN))));
    if (process.env.DBG && neck) console.log(`  neck strip of ${edges.length}: ${H.toFixed(1)} cm long (${Math.min(...edges.map((k) => len(Math.floor(k / 1e6), k % 1e6))).toFixed(1)} to ${Math.max(...edges.map((k) => len(Math.floor(k / 1e6), k % 1e6))).toFixed(1)}), ${wide.toFixed(1)} wide → ${pieces}`);
    if (pieces > 1) {
      for (const k of edges) cut.set(k, pieces - 1);
      added += pieces - 1;
      if (process.env.DBG) console.log(`  strip of ${edges.length}: ${H.toFixed(1)} cm long, ${wide.toFixed(1)} wide → ${pieces} pieces`);
    }
  }
  // each quad a grid: its edges' new points, and points between them
  const rows: Float64Array[] = [];
  const mix = (a: Float64Array, b: Float64Array, t: number) => a.map((x, k) => x + (b[k] - x) * t);
  const row = (i: number) => (i < n ? V.subarray(i * D, i * D + D) : rows[i - n]);
  const push = (r: Float64Array) => rows.push(r) + n - 1;
  const onEdge = new Map<number, number[]>();
  /** the new points along the edge from a to b, in that order */
  const along = (a: number, b: number): number[] => {
    const k = edgeKey(a, b), c = cut.get(k) ?? 0;
    if (!onEdge.has(k)) onEdge.set(k, Array.from({ length: c }, (_, i) => push(mix(row(Math.min(a, b)), row(Math.max(a, b)), (i + 1) / (c + 1)))));
    const pts = onEdge.get(k)!;
    return a < b ? pts : [...pts].reverse();
  };
  quads = quads.flatMap(([a, b, c, d]) => {
    const bottom = [a, ...along(a, b), b], top = [d, ...along(d, c), c], left = [a, ...along(a, d), d], right = [b, ...along(b, c), c];
    const m = bottom.length - 1, k = left.length - 1;
    if (m === 1 && k === 1) return [[a, b, c, d]];
    const g: number[][] = Array.from({ length: m + 1 }, (_, i) =>
      Array.from({ length: k + 1 }, (_, j) => {
        if (j === 0) return bottom[i];
        if (j === k) return top[i];
        if (i === 0) return left[j];
        if (i === m) return right[j];
        return push(mix(mix(row(a), row(b), i / m), mix(row(d), row(c), i / m), j / k));
      }),
    );
    const out: number[][] = [];
    for (let i = 0; i < m; i++) for (let j = 0; j < k; j++) out.push([g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]]);
    return out;
  });
  const W2 = new Float64Array((n + rows.length) * D);
  W2.set(V);
  rows.forEach((r, i) => W2.set(r, (n + i) * D));
  V = W2;
  console.log(`${added} loops added: ${V.length / D} vertices, ${quads.length} quads`);
}

for (let level = 0; level < LEVELS; level++) {
  const n = V.length / D;
  const { edgeFaces, near } = topology(quads, n);
  const edges = [...edgeFaces.keys()];
  const eIndex = new Map(edges.map((k, i) => [k, n + quads.length + i]));
  const out = new Float64Array((n + quads.length + edges.length) * D);
  const add = (to: number, from: Float64Array, at: number, w: number) => {
    for (let k = 0; k < D; k++) out[to * D + k] += w * from[at * D + k];
  };
  // a point in each quad's middle
  quads.forEach((q, f) => {
    for (const v of q) add(n + f, V, v, 0.25);
  });
  // one on each edge: its ends and its two quads' middles (an open edge: its own middle)
  edges.forEach((k, i) => {
    const a = Math.floor(k / 1e6), b = k % 1e6, fs = edgeFaces.get(k)!, e = n + quads.length + i;
    if (fs.length === 2) {
      add(e, V, a, 0.25);
      add(e, V, b, 0.25);
      for (const f of fs) add(e, out, n + f, 0.25);
    } else {
      add(e, V, a, 0.5);
      add(e, V, b, 0.5);
    }
  });
  // and each old vertex eased towards its neighbours
  const facesOf: number[][] = Array.from({ length: n }, () => []);
  quads.forEach((q, f) => q.forEach((v) => facesOf[v].push(f)));
  for (let v = 0; v < n; v++) {
    const open = [...near[v]].filter((o) => edgeFaces.get(edgeKey(v, o))!.length === 1);
    if (open.length) {
      add(v, V, v, 0.75);
      for (const o of open) add(v, V, o, 0.125);
      continue;
    }
    const k = near[v].size;
    add(v, V, v, (k - 3) / k);
    for (const f of facesOf[v]) add(v, out, n + f, 1 / (k * facesOf[v].length));
    for (const o of near[v]) {
      add(v, V, v, 1 / (k * k));
      add(v, V, o, 1 / (k * k));
    }
  }
  quads = quads.flatMap((q, f) => q.map((v, j) => [v, eIndex.get(edgeKey(v, q[(j + 1) % 4]))!, n + f, eIndex.get(edgeKey(q[(j + 3) % 4], v))!]));
  V = out;
  const m = V.length / D;
  console.log(`level ${level + 1}: ${m} vertices, ${quads.length} quads`);
  // the rim stays where the subdivision put it (the base mesh has a head above it, not an edge)
  project(quads, m, new Set(rimOf(quads, m)));
}

// ------------------------------------------------------------------ write
const n = V.length / D;
if (n > 65535) throw new Error(`${n} vertices: too many for 16-bit indices`);
const rim = rimOf(quads, n);
const r2 = (x: number) => Math.round(x * 100) / 100;
const xs = rim.map((i) => V[i * D]), ys = rim.map((i) => V[i * D + 1]), zs = rim.map((i) => Math.abs(V[i * D + 2]));
/** at most this many bones move each vertex */
const INF = 8;
const header = {
  source: `MakeHuman (CC0): ${PROXY} on the base mesh hm08 (adult male), MPFB2's Rigify human rig and weights; subdivided ${LEVELS}×`,
  licence: "CC0 (MakeHuman assets)",
  bones: KEEP.map((name, i) => {
    const e = ends(name);
    return { name, parent: parentOf(name), head: e.head.map(r2), tail: e.tail.map(r2), ...(SPECS[i].helper ? { helper: true } : {}) };
  }),
  points: Object.fromEntries(Object.entries({ ...fingers("R"), ...fingers("L") }).map(([k, v]) => [k, v.map(r2)])),
  rim,
  rimShape: { centre: [r2((Math.max(...xs) + Math.min(...xs)) / 2), r2(ys.reduce((a, b) => a + b) / ys.length), 0], a: r2((Math.max(...xs) - Math.min(...xs)) / 2), b: r2(Math.max(...zs)) },
  influences: INF,
  vertices: n,
  quads: quads.length,
};
// padded to 4 bytes (bytes, not characters: the arrays after it must start aligned)
let json = JSON.stringify(header);
while (new TextEncoder().encode(json).length % 4) json += " ";
const H = new TextEncoder().encode(json);
const size = 4 + H.length + n * 12 + n * INF + n * INF * 2 + quads.length * 8;
const buf = new ArrayBuffer(size + (size % 4 ? 4 - (size % 4) : 0));
const dv = new DataView(buf);
dv.setUint32(0, H.length, true);
new Uint8Array(buf, 4, H.length).set(H);
let o = 4 + H.length;
for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++, o += 4) dv.setFloat32(o, V[i * D + k], true);
const skin: [number, number][][] = [];
let most = 0, dropped = 0;
for (let i = 0; i < n; i++) {
  const all: [number, number][] = [];
  for (let b = 0; b < NB; b++) if (V[i * D + 3 + b] > 1e-4) all.push([b, V[i * D + 3 + b]]);
  all.sort((a, b) => b[1] - a[1]);
  most = Math.max(most, all.length);
  const top = all.slice(0, INF);
  const sum = top.reduce((s, [, w]) => s + w, 0), total = all.reduce((s, [, w]) => s + w, 0);
  dropped = Math.max(dropped, 1 - sum / total);
  if (!top.length) throw new Error(`vertex ${i} has no weights`);
  const q = top.map(([b, w]) => [b, Math.round((65535 * w) / sum)] as [number, number]);
  q[0][1] += 65535 - q.reduce((s, [, w]) => s + w, 0);
  while (q.length < INF) q.push([0, 0]);
  skin.push(q);
}
console.log(`bones per vertex: at most ${most}; the most weight dropped past ${INF}: ${(100 * dropped).toFixed(2)}%`);
for (const q of skin) for (const [b] of q) dv.setUint8(o++, b);
for (const q of skin) for (const [, w] of q) {
  dv.setUint16(o, w, true);
  o += 2;
}
for (const q of quads) for (const v of q) {
  dv.setUint16(o, v, true);
  o += 2;
}
writeFileSync(OUT, new Uint8Array(buf));
// how many edges meet at each vertex: the mesh's poles
const { near } = topology(quads, n);
const valence = new Map<number, number>();
for (const s of near) valence.set(s.size, (valence.get(s.size) ?? 0) + 1);
console.log(`${n} vertices, ${quads.length} quads, ${(buf.byteLength / 1024).toFixed(0)} KB; rim ${rim.length} vertices`, header.rimShape, `edges per vertex`, Object.fromEntries([...valence].sort((a, b) => a[0] - b[0])));

