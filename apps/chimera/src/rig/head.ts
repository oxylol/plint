/**
 * The CHIMERA head worn by the body rig: the app's own 3D head (src/ui/render/anatomy3d.ts),
 * drawn live, carried by the rig's head bone. Its skin is a grid grown from the rig's neck rim:
 * the bottom row is the rim, vertex for vertex, so every line up the neck runs on over the head;
 * the rows above are the head's own cross-sections (Anatomy3D.skinAt), closed over the crown by a
 * patch, and run on over the mouth square for square, cut open along the lips' line. The grid is
 * attached to the rig's mesh and skinned with it, so the neck bends as the body does; each frame
 * the jaw, cheeks and lips move the face's part of it as the head draws them.
 *
 * Use (after `await initChimera()`: the head reads the engine's tables when it loads):
 *   const model = new VoiceModel(body, sampleRate);     // yours, for the rig's lifetime
 *   const head = wearHead(rig, model);                    // before the RigView: it adds to the mesh
 *   const view = new RigView(rig);
 *   scene.add(view.group, head.mount);                    // the same parent
 *   // each frame: rig.pose(p); rig.skin(posed); head.update(camera, artic, tele, dt);
 *   //             head.face(posed, view.posed); view.update();
 * A different body (a child's voice, say) is a new rig and a new head.
 *
 * Frames: the head is drawn in its own cm (glottis at the origin); the body is `s` times smaller
 * than life against it (the voice's scale × SIZE), and joins it at the neck:
 *   head = s (body − rim centre) + anchor.
 * Colours are the app's, as display values: render with `renderer.outputColorSpace =
 * THREE.LinearSRGBColorSpace`, as the app's 3D view does (RigView's lines are encoded to match).
 */
import * as THREE from "three";
import type { Rig } from "./rig.ts";
import { Anatomy3D, APP_DRAWING, REST_ARTIC } from "../ui/render/anatomy3d.ts";
import type { ViewFlags } from "../ui/render/anatomy.ts";
import type { VoiceModel } from "../ui/analysis.ts";
import type { HumanArtic } from "../wasm/types.ts";
import type { TelemetryMessage } from "../worklet/protocol.ts";

/** how far up the head's neck the body reaches (template cm), the body's size against the head's */
const NECK_BLEND = 5.5, SIZE = 0.88;
/** the band under the rim that is drawn in to the head's neck (body cm) */
const FIT_BAND = 9;
/** the head's turn is taken up the neck from this far under the rim (body cm) to this far above it (template cm) */
const TURN_DOWN = 10, TURN_UP = 9;
/** how bright the scalp's lines are against the face's, and over how far (template cm) across the hairline they fall to it */
const SCALP_DIM = 0.22, SCALP_BLEND = 5.5;
/** how many rings of the body's neck under the rim settle with the head's lower rows */
const SETTLE = 3;
/** the crown is closed where the head has narrowed to this (mean radius, template cm) */
const CAP = 6.4;
/** how far round the mouth the skin takes a share of the lips' move: rows above and below the
 *  lips' line, and columns past the mouth's corners */
const MOUTH = { rows: 3, beyond: 3 };
const VIEW: ViewFlags = { pressure: true, particles: true, grid: false, calipers: false };
/** points along the lips' line (Anatomy3D.lipEdges) */
const LIP_POINTS = 24;

/**
 * How the head is drawn: how bright its insides are (`inner`: the airway, tongue, palate, brain,
 * vertebrae), its jaw and teeth (`bone`), the profile on its midline (`outline`), the app's own
 * wireframe of the head (`shell`), the rig's grid over the head (`skin`), and the eyes, nose and
 * lips (`features`: 1 as the app draws them; more is brighter and wider). 1 everywhere is the
 * app's head (with the rig's grid over it).
 */
export interface HeadLook {
  inner: number;
  bone: number;
  outline: number;
  shell: number;
  skin: number;
  features: number;
}
/**
 * The head's looks, from the app's head to the outside of one:
 *  - `classic`: the app's head as it draws itself (its own wireframe, its insides at full), the
 *    body's grid faded out over it
 *  - `skinned`: the app's insides at full under the grid, half bright; the features a little brighter
 *  - `vocal`: the grid and bright features in front, the insides at half: for singing and speaking
 *  - `outside`: the grid and the brightest features, the insides faint: for acting, the voice aside
 */
export const HEAD_LOOKS: Readonly<Record<string, HeadLook>> = {
  classic: { inner: 1, bone: 1, outline: 1, shell: 1, skin: 0.12, features: 1 },
  skinned: { inner: 1, bone: 1, outline: 1, shell: 0, skin: 0.55, features: 1.4 },
  vocal: { inner: 0.9, bone: 0.88, outline: 0.95, shell: 0, skin: 1, features: 1.6 },
  outside: { inner: 0.7, bone: 0.67, outline: 0.7, shell: 0, skin: 1, features: 1.8 },
};
export const HEAD_LOOK = "vocal";

const smooth = (x: number) => {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3 - 2 * t);
};

/** the hairline: the lowest height hair grows at (template cm), by the angle round from the front */
const HAIRLINE: [number, number][] = [[0, 19.4], [0.7, 18.7], [1.2, 15.2], [1.65, 11.5], [2.4, 8.2], [Math.PI, 6.6]];
const hairline = (a: number) => {
  for (let i = 1; i < HAIRLINE.length; i++) {
    if (a <= HAIRLINE[i][0]) {
      const [a0, y0] = HAIRLINE[i - 1], [a1, y1] = HAIRLINE[i];
      return y0 + ((y1 - y0) * (a - a0)) / (a1 - a0);
    }
  }
  return HAIRLINE[HAIRLINE.length - 1][1];
};

export interface RigHead {
  /** what to add to the scene (beside the RigView's group): the head's lines, placed each frame on the head bone */
  readonly mount: THREE.Group;
  /** the app's head renderer drawing them */
  readonly head: Anatomy3D;
  /** the voice model the head was fitted to (the caller's: it isn't disposed here) */
  readonly model: VoiceModel;
  /** body cm ↔ head cm */
  readonly s: number;
  /** the head's look now (see HEAD_LOOKS) */
  readonly look: HeadLook;
  /** draw the head another way (the grid's brightness is the rig's tone: call view.toneChanged() after) */
  setLook(look: HeadLook): void;
  /** Place and draw the head for this frame, after rig.pose: the voice's articulation now, its
   *  telemetry (the live worklet's message, or a TeleTrack's; null: none), seconds since the last. */
  update(camera: THREE.Camera, a: HumanArtic, tele: TelemetryMessage | null, dt: number): void;
  /** the skinned body `base` with the face's own motion this frame (the jaw, the cheeks, the lips) into `out` (may be `base`) */
  face(base: Float32Array, out: Float32Array): void;
  /** the canvas's size in CSS px (the head's lines' widths are in px) */
  resize(w: number, h: number): void;
  dispose(): void;
}

/**
 * Put the CHIMERA head on the rig, fitted to `model` (the voice's body: its size and shape): its
 * skin is added to the rig's mesh, and the head bone moved to the head's pivot. Call it once per
 * rig, before making a RigView of it.
 */
export function wearHead(rig: Rig, model: VoiceModel, opts: { look?: Partial<HeadLook> } = {}): RigHead {
  let look: HeadLook = { ...HEAD_LOOKS[HEAD_LOOK], ...opts.look };
  const k: number = model.tract.scale;
  const head = new Anatomy3D();
  head.centred = true;
  const shape = head.headShape(model);
  /** the head's drawing for a look: mounted on the body, the lips' patch's own skin left out (the grid covers the mouth) */
  const drawAs = (l: HeadLook) => {
    head.drawing = { ...APP_DRAWING, inner: l.inner, bone: l.bone, outline: l.outline, shell: l.shell, features: l.features, lipSkin: false, mounted: true, fade: { y: shape.y0 + 3.5 * k, band: 6.5 * k } };
  };
  drawAs(look);
  /** the lips' line at rest (jaw and lips closed), as wide as the mouth spreads: where the grid
   *  is cut open, so the lips can part anywhere the mouth's corners can reach */
  const lipsRest = head.lipEdges(model, { ...REST_ARTIC, lipWidth: 1 }, LIP_POINTS).upper;

  // ------------------------------------------------------------------ the body's neck fitted to the head's
  const y = shape.y0 + NECK_BLEND * k, sec = shape.at(y);
  const f = { neckX: sec ? (sec.F + sec.Bk) / 2 : -2 * k, neckY: y, neckA: sec ? (sec.F - sec.Bk) / 2 : 5 * k, neckB: sec ? sec.W : 5.3 * k };
  const s = k * SIZE;
  const rc = new THREE.Vector3().fromArray(rig.header.rimShape.centre);
  const anchor = new THREE.Vector3(f.neckX, f.neckY, 0);
  const P = rig.rest, top = rc.y;
  {
    // over the band under the rim each vertex is drawn in or out to the head's ring (a
    // superellipse, as the head's cross-sections), the more the nearer the rim, so the neck runs
    // up into the head as one column; the rim itself is made level
    const a = f.neckA / s, b = f.neckB / s, E = 2.2;
    for (let i = 0; i < rig.bodyVertices; i++) {
      const t = THREE.MathUtils.smoothstep(P[3 * i + 1], top - FIT_BAND, top - 0.3);
      if (t <= 0) continue;
      const dx = P[3 * i] - rc.x, dz = P[3 * i + 2], r = Math.hypot(dx, dz) || 1;
      const ring = Math.pow(Math.pow(Math.abs(dx / r) / a, E) + Math.pow(Math.abs(dz / r) / b, E), -1 / E);
      const to = (r + (ring - r) * t) / r;
      P[3 * i] = rc.x + dx * to;
      P[3 * i + 2] = dz * to;
    }
    for (const i of rig.rim) P[3 * i + 1] = top;
  }
  /** a body vertex's place in the head's frame (the trunk and neck are straight at rest) */
  const toHead = (i: number, out: number[], o = 0) => {
    out[o] = s * (P[3 * i] - rc.x) + anchor.x;
    out[o + 1] = s * (P[3 * i + 1] - rc.y) + anchor.y;
    out[o + 2] = s * (P[3 * i + 2] - rc.z) + anchor.z;
  };
  /** how much of the head's turn the neck takes at a height above the rim (head cm; negative: under it) */
  const neckShare = (h: number) => smooth((h + TURN_DOWN * s) / (TURN_UP * k + TURN_DOWN * s));

  // ------------------------------------------------------------------ the head's skin
  const N = rig.rim.length, step = 1.5;
  const rim: number[] = new Array(3 * N);
  rig.rim.forEach((v, i) => toHead(v, rim, 3 * i));
  let mean = 0;
  for (let i = 0; i < N; i++) mean += rim[3 * i + 1] / N;
  const hTop = anchor.y;
  // each column's turn round the head, from where its rim vertex is, read off one cross-section
  const turns = Array.from({ length: N }, (_, i) => head.skinTurn(model, mean, rim[3 * i], rim[3 * i + 2]) as number);
  // the rim crowds its vertices at the throat; up the head the columns ease to an even spacing
  const order = turns.map((t, i) => [t, i] as const).sort((p, q) => p[0] - q[0]).map((p) => p[1]);
  const rank = new Array<number>(N);
  order.forEach((i, n) => (rank[i] = n));
  const ahead = turns.reduce((best, t, i) => (Math.abs(t) < Math.abs(turns[best]) ? i : best), 0);
  const even = turns.map((_, i) => {
    const t = ((2 * Math.PI * (rank[i] - rank[ahead])) / N) % (2 * Math.PI);
    return t > Math.PI ? t - 2 * Math.PI : t <= -Math.PI ? t + 2 * Math.PI : t;
  });
  if (N % 4) throw new Error(`the neck's rim has ${N} vertices: the crown's patch needs a multiple of 4`);
  const radius = (yy: number) => {
    const q = shape.at(yy);
    return q ? ((q.F - q.Bk) / 2 + q.W) / 2 : 0;
  };
  const ys: number[] = [];
  for (let yy = hTop + 0.8 * step * k; yy < shape.y1 - 0.45 * k; yy += step * k) {
    ys.push(yy);
    if (yy > (shape.y0 + shape.y1) / 2 && radius(yy) < CAP * k) break;
  }
  // the row nearest the lips' line is brought onto it (the rows round it give way a little), so
  // the mouth's loops sit evenly in the block they replace
  const lipMid = 3 * Math.floor(lipsRest.length / 6), yc = lipsRest[lipMid + 1];
  const rm = ys.reduce((b, yy, r) => (Math.abs(yy - yc) < Math.abs(ys[b] - yc) ? r : b), 0);
  {
    const dy = yc - ys[rm], reach = MOUTH.rows + 2;
    ys.forEach((yy, r) => (ys[r] = yy + dy * Math.max(0, 1 - Math.abs(r - rm) / reach)));
  }
  const R = ys.length, M = N / 4, K = (M - 1) * (M - 1);
  /** the column `j` places round from the front one (+: towards +z) */
  const col = (j: number) => order[(((rank[ahead] + j) % N) + N) % N];
  const count = N * R + K;
  const rest = new Float32Array(3 * count), jaw = new Float32Array(count), cheek = new Float32Array(count), turn = new Float32Array(count);
  const at: number[][] = [];
  const facing = new Float32Array(N * R);
  ys.forEach((yy, r) => {
    const ease = smooth((yy - hTop) / (1.1 * k)), spread = smooth((yy - hTop) / (5 * k));
    const row: number[] = [];
    for (let i = 0; i < N; i++) {
      const th = turns[i] + (even[i] - turns[i]) * spread;
      row.push(th);
      const p = head.skinAt(model, yy, th) ?? { x: rim[3 * i], z: rim[3 * i + 2], face: 0 };
      rest.set([rim[3 * i] + (p.x - rim[3 * i]) * ease, yy, rim[3 * i + 2] + (p.z - rim[3 * i + 2]) * ease], 3 * (N * r + i));
      facing[N * r + i] = p.face * ease;
    }
    at.push(row);
  });
  // from the rim up to the mouth the mesh is relaxed (the neck runs into the jaw in one curve);
  // the rim and the top of the body's neck settle with it, less the further under the rim
  const hold = ys.map((yy) => 1 - smooth((yy - hTop - 3.6 * k) / (2.4 * k)));
  const patch = neckPatch(rig, SETTLE + 1), B = patch.ids.length;
  const bp: number[] = new Array(3 * B);
  patch.ids.forEach((v, n) => toHead(v, bp, 3 * n));
  const was0 = bp.slice();
  const loose = patch.hops.map((h) => Math.max(0, 1 - h / (SETTLE + 1)));
  for (let pass = 0; pass < 24; pass++) {
    const wb = bp.slice(), wp = rest.slice();
    for (let n = 0; n < B; n++) {
      if (!loose[n]) continue;
      const nb = patch.near[n], cnt = nb.length + (n < N ? 1 : 0);
      for (let c = 0; c < 3; c++) {
        let sum = n < N ? wp[3 * n + c] : 0;
        for (const j of nb) sum += wb[3 * j + c];
        bp[3 * n + c] += 0.5 * loose[n] * (sum / cnt - wb[3 * n + c]);
      }
    }
    for (let r = 0; r < R - 2 && hold[r] > 0; r++) {
      for (let i = 0; i < N; i++) {
        const o = 3 * (N * r + i), e = 3 * (N * r + ((i + 1) % N)), w = 3 * (N * r + ((i + N - 1) % N));
        for (let c = 0; c < 3; c++) rest[o + c] += 0.5 * hold[r] * ((wp[e + c] + wp[w + c] + (r > 0 ? wp[o - 3 * N + c] : wb[3 * i + c]) + wp[o + 3 * N + c]) / 4 - wp[o + c]);
      }
    }
  }
  // the settled neck back into the body's rest
  patch.ids.forEach((v, n) => {
    for (let c = 0; c < 3; c++) P[3 * v + c] += (bp[3 * n + c] - was0[3 * n + c]) / s;
  });
  for (let r = 0; r < R; r++) {
    for (let i = 0; i < N; i++) {
      const o = N * r + i, w = head.skinWeights(model, rest[3 * o], rest[3 * o + 1]);
      jaw[o] = w.jaw;
      cheek[o] = w.cheek;
      turn[o] = Math.max(facing[o], neckShare(rest[3 * o + 1] - hTop));
    }
  }
  const lipLine = head.lipBox(model).bottom;
  for (let pass = 0; pass < 8; pass++) {
    const was = jaw.slice();
    for (let r = 0; r < R - 1; r++) {
      for (let i = 0; i < N; i++) {
        const o = N * r + i, away = Math.max(smooth((Math.abs(at[r][i]) - 0.55) / 0.5), smooth((lipLine - 0.6 * k - ys[r]) / k));
        const dn = r > 0 ? was[o - N] : 0;
        jaw[o] += 0.5 * away * ((was[N * r + ((i + 1) % N)] + was[N * r + ((i + N - 1) % N)] + dn + was[o + N]) / 4 - was[o]);
      }
    }
  }
  const rimShare = Array.from({ length: N }, (_, i) => neckShare(bp[3 * i + 1] - hTop));
  for (let pass = 0; pass < 10; pass++) {
    const was = turn.slice();
    for (let r = 0; r < R - 1 && ys[r] < lipLine; r++) {
      const give = smooth((lipLine - ys[r]) / (1.5 * k));
      for (let i = 0; i < N; i++) {
        const o = N * r + i, dn = r > 0 ? was[o - N] : rimShare[i];
        turn[o] += 0.5 * give * ((was[N * r + ((i + 1) % N)] + was[N * r + ((i + N - 1) % N)] + dn + was[o + N]) / 4 - was[o]);
      }
    }
  }
  const pivot: THREE.Vector3 = head.pivot(model);

  // the crown: M × M quads inside the last ring, lifted on to the head's top
  const ringAt = N * (R - 1), j0 = -Math.floor(M / 2);
  const ringP = (j: number) => rest.subarray(3 * (ringAt + col(j)), 3 * (ringAt + col(j)) + 3);
  const capAt = (i: number, j: number) =>
    j === 0 ? ringAt + col(j0 + i) : i === M ? ringAt + col(j0 + M + j) : j === M ? ringAt + col(j0 + 3 * M - i) : i === 0 ? ringAt + col(j0 + 4 * M - j) : N * R + (i - 1) * (M - 1) + (j - 1);
  const domeY = (x: number, z: number) => {
    let lo = ys[R - 1], hi = shape.y1;
    for (let n = 0; n < 26; n++) {
      const yy = (lo + hi) / 2, q = shape.at(yy), th = head.skinTurn(model, yy, x, z), p = q && Math.abs(z) < q.W ? head.skinAt(model, yy, th) : null;
      if (p && (Math.abs(th) <= Math.PI / 2 ? x <= p.x : x >= p.x)) lo = yy;
      else hi = yy;
    }
    return (lo + hi) / 2;
  };
  const cA = ringP(j0), cB = ringP(j0 + M), cC = ringP(j0 + 2 * M), cD = ringP(j0 + 3 * M);
  for (let i = 1; i < M; i++) {
    for (let j = 1; j < M; j++) {
      const u = i / M, v = j / M, o = 3 * capAt(i, j);
      const b = ringP(j0 + i), t = ringP(j0 + 3 * M - i), l = ringP(j0 + 4 * M - j), r = ringP(j0 + M + j);
      for (const c of [0, 2]) rest[o + c] = (1 - v) * b[c] + v * t[c] + (1 - u) * l[c] + u * r[c] - ((1 - u) * (1 - v) * cA[c] + u * (1 - v) * cB[c] + u * v * cC[c] + (1 - u) * v * cD[c]);
      rest[o + 1] = domeY(rest[o], rest[o + 2]);
      turn[capAt(i, j)] = 1;
    }
  }

  // the quads (grid numbering: the rim's N, then the rows', then the crown's)
  const grid: number[] = [];
  for (let r = 0; r < R; r++) {
    for (let i = 0; i < N; i++) {
      const j = (i + 1) % N;
      grid.push(N * r + i, N * r + j, N * (r + 1) + j, N * (r + 1) + i);
    }
  }
  // the crown's quads, wound as the sides' are
  const pt = (P: ArrayLike<number>, g: number) => new THREE.Vector3(P[3 * g], P[3 * g + 1], P[3 * g + 2]);
  /** which way a quad (rest numbering) faces, against outwards from the head's axis: + or − */
  const facingOf = (P: ArrayLike<number>, a: number, b: number, c: number, d: number) => {
    const n = new THREE.Vector3().crossVectors(pt(P, c).sub(pt(P, a)), pt(P, d).sub(pt(P, b)));
    return n.dot(pt(P, a).sub(new THREE.Vector3(pivot.x, P[3 * a + 1], 0)));
  };
  const side = Math.sign(facingOf(rest, N * Math.floor(R / 2), N * Math.floor(R / 2) + 1, N * Math.floor(R / 2) + N + 1, N * Math.floor(R / 2) + N));
  {
    const up = new THREE.Vector3().crossVectors(pt(rest, capAt(1, 1)).sub(pt(rest, capAt(0, 0))), pt(rest, capAt(0, 1)).sub(pt(rest, capAt(1, 0)))).y;
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < M; j++) {
        const q = [capAt(i, j), capAt(i + 1, j), capAt(i + 1, j + 1), capAt(i, j + 1)].map((g) => g + N);
        grid.push(...(side * up > 0 ? q : q.reverse()));
      }
    }
  }
  // how bright each point's lines are: the face's full, the scalp's dimmer, eased across the hairline
  const tone0 = new Float32Array(count);
  for (let p = 0; p < count; p++) {
    if (p >= N * R) {
      tone0[p] = SCALP_DIM;
      continue;
    }
    const r = Math.floor(p / N), i = p % N, above = rest[3 * p + 1] / k - hairline(Math.min(Math.PI, Math.abs(at[r][i])));
    tone0[p] = 1 - (1 - SCALP_DIM) * smooth((above + 1) / SCALP_BLEND);
  }

  // ------------------------------------------------------------------ the mouth
  // The grid runs on over the mouth, square for square. Its row on the lips' line (see `rm`) is
  // cut open between the columns nearest the corners of the mouth at its widest: each point there
  // has an upper lip's copy (the quads above take it) and a lower lip's (the quads below). At rest
  // the cut lies on the lips' line, evenly from corner to corner, the two copies together. Each
  // frame the cut is laid on the lips the head draws, from their corners as they are (a rounded
  // mouth gathers the skin in towards it, as lips do), and the rows above and below, and the
  // columns past the corners, take a share of the lips' move that lessens away from them.
  const rv = (r: number, j: number) => N * r + col(j);
  const m0 = lipsRest.length / 3 - 1, zc = (Math.abs(lipsRest[2]) + Math.abs(lipsRest[3 * m0 + 2])) / 2;
  const zOf = (r: number, j: number) => rest[3 * rv(r, j) + 2];
  let J = 1;
  for (let j = 2; j < N / 4; j++) if (Math.abs(Math.abs(zOf(rm, j)) - zc) < Math.abs(Math.abs(zOf(rm, J)) - zc)) J = j;
  const { rows: MR, beyond: MB } = MOUTH;
  // The lips' share fades over MOUTH.rows rows each way, and below the mouth it's gone by the first
  // row over the rim (a small head has few rows there: a child's mouth is two up): that row
  // still blends the neck's motion with the head's, and the lips' move on it folded the grid
  // under the chin with the head turned.
  const fadeUp = MR + 1, fadeDown = Math.min(MR + 1, rm);
  if (rm < 1 || rm + MR > R - 2) throw new Error(`the mouth's row (${rm}) is too near the head's rim or crown (rows 0 to ${R - 1})`);
  const extra = 2 * J - 1, kept = count + extra;
  /** the lower lip's copy of the slit's point j (−J < j < J; rest numbering) */
  const lower = (j: number) => count + j + J - 1;
  const restC = new Float32Array(3 * kept), jawC = new Float32Array(kept), cheekC = new Float32Array(kept), turnC = new Float32Array(kept), toneC = new Float32Array(kept);
  /** each point's column of the grid (its bones are its rim vertex's), or −1 (the crown's: the head's alone) */
  const colC = new Int32Array(kept);
  for (let p = 0; p < kept; p++) {
    const from = p < count ? p : rv(rm, p - count - J + 1);
    restC.set(rest.subarray(3 * from, 3 * from + 3), 3 * p);
    jawC[p] = jaw[from];
    cheekC[p] = cheek[from];
    turnC[p] = turn[from];
    toneC[p] = tone0[from];
    colC[p] = from < N * R ? from % N : -1;
  }
  // the cut laid on the lips' line at rest; its points' places along the lips (0 to 1, −z to +z)
  const along = (j: number) => Math.max(0, Math.min(1, (j + J) / (2 * J)));
  for (let j = -J; j <= J; j++) {
    sample(lipsRest, along(j), restC, 3 * rv(rm, j));
    if (Math.abs(j) < J) sample(lipsRest, along(j), restC, 3 * lower(j));
  }
  // the quads under the slit take the lower lip's copies
  const quads = grid.slice();
  const cut = new Map<number, number>();
  for (let j = -J + 1; j < J; j++) cut.set(N + rv(rm, j), N + lower(j));
  for (let f = 4 * N * rm; f < 4 * N * (rm + 1); f++) quads[f] = cut.get(quads[f]) ?? quads[f];
  /** the points that take a share of the lips' move: where along the lips (0 to 1, −z to +z), which lip (−1 lower), how much */
  const lipShare: { v: number; s: number; lip: number; w: number }[] = [];
  for (let r = rm - fadeDown + 1; r < rm + fadeUp; r++) {
    for (let j = -J - MB; j <= J + MB; j++) {
      const w = (1 - Math.abs(r - rm) / (r < rm ? fadeDown : fadeUp)) * (Math.abs(j) <= J ? 1 : 1 - (Math.abs(j) - J) / (MB + 1));
      lipShare.push({ v: rv(r, j), s: along(j), lip: r < rm ? -1 : 1, w });
      if (r === rm && Math.abs(j) < J) lipShare.push({ v: lower(j), s: along(j), lip: -1, w });
    }
  }

  // ------------------------------------------------------------------ onto the rig
  const n0 = rig.n;
  const head0 = rig.bone("head");
  /** a skin point's index in the rig's mesh (grid numbering: the rim's first) */
  const gi = (g: number) => (g < N ? rig.rim[g] : n0 + g - N);
  const points = new Float32Array(3 * kept);
  for (let p = 0; p < kept; p++) for (let c = 0; c < 3; c++) points[3 * p + c] = (restC[3 * p + c] - anchor.getComponent(c)) / s + rc.getComponent(c);
  // The body's neck under the rim takes its share of the head's turn: the rim's bones give way to
  // the head's from TURN_DOWN under the rim, the same at every turn round the neck. Each point of
  // the skin above is then its column's rim vertex's bones and the head's, by how much of the
  // head's turn it takes.
  neckWeights(rig, rc.y, (yb) => neckShare(s * (yb - rc.y)), head0);
  const rimSkin = rig.rim.map((v) => {
    // the rim vertex's bones other than the head, as a whole
    const list: [number, number][] = [];
    for (let j = 0; j < rig.INF; j++) {
      const b = rig.skinIndex[rig.INF * v + j], w = rig.skinWeight[rig.INF * v + j];
      if (w > 0 && b !== head0) list.push([b, w]);
    }
    const sum = list.reduce((a, [, w]) => a + w, 0) || 1;
    return list.map(([b, w]) => [b, w / sum] as [number, number]);
  });
  const skin: [number, number][][] = [];
  for (let p = 0; p < kept; p++) {
    const i = colC[p], tau = i < 0 ? 1 : turnC[p];
    const list: [number, number][] = i < 0 ? [] : rimSkin[i].map(([b, w]) => [b, (1 - tau) * w] as [number, number]);
    list.push([head0, tau]);
    skin.push(list);
  }
  rig.attach(points, quads.map(gi), skin, toneC);
  // the grid over the head at the look's brightness, eased in from the rim to the jaw
  const skinEase = Array.from({ length: kept }, (_, p) => smooth((restC[3 * p + 1] - hTop) / (4 * k)));
  const skinAs = (b: number) => {
    for (let p = 0; p < kept; p++) rig.tone[n0 + p] = toneC[p] * (1 - (1 - b) * skinEase[p]);
  };
  skinAs(look.skin);
  // the head turns about its own pivot
  const pivotBody = pivot.clone().sub(anchor).divideScalar(s).add(rc);
  rig.H[head0].copy(pivotBody);
  rig.weightsChanged();

  // ------------------------------------------------------------------ the face's skin, moving
  // The jaw carries the lower face and the cheeks puff (Anatomy3D.skinNow, by each point's
  // weights); the grid's cut follows the lips as the head draws them, and the points round it take
  // a share of the lips' move. What each point has moved (head cm) is added to the skinned body by
  // the head bone (`face`).
  const now = new Float32Array(restC.length), moved = new Float32Array(restC.length), lip = new Float32Array(6);
  const faceNow = (a: HumanArtic) => {
    head.skinNow(model, a, restC, jawC, cheekC, now);
    const mouth = head.lipEdges(model, a, LIP_POINTS);
    for (const { v, s: at, lip: which, w } of lipShare) {
      sample(which < 0 ? mouth.lower : mouth.upper, at, lip, 0);
      sample(lipsRest, at, lip, 3);
      for (let c = 0; c < 3; c++) now[3 * v + c] = restC[3 * v + c] + (1 - w) * (now[3 * v + c] - restC[3 * v + c]) + w * (lip[c] - lip[3 + c]);
    }
    for (let i = 0; i < now.length; i++) moved[i] = now[i] - restC[i];
  };

  // ------------------------------------------------------------------ the head's lines, on the head bone
  const mount = new THREE.Group();
  mount.matrixAutoUpdate = false;
  mount.add(head.headGroup);
  const toBody = new THREE.Matrix4().makeTranslation(-anchor.x, -anchor.y, -anchor.z).premultiply(new THREE.Matrix4().makeScale(1 / s, 1 / s, 1 / s)).premultiply(new THREE.Matrix4().makeTranslation(rc.x, rc.y, rc.z));
  const m = new THREE.Matrix4(), one = new THREE.Vector3(1, 1, 1), unH = new THREE.Matrix4();
  const eye = new THREE.Vector3(), piv = new THREE.Vector3(), fwd = new THREE.Vector3(), v3 = new THREE.Vector3();
  return {
    mount,
    head,
    model,
    s,
    get look() {
      return look;
    },
    setLook(l: HeadLook) {
      look = { ...l };
      drawAs(look);
      skinAs(look.skin);
    },
    resize(w: number, h: number) {
      head.resize(w, h);
    },
    update(camera: THREE.Camera, a: HumanArtic, tele: TelemetryMessage | null, dt: number) {
      // head cm → body cm → the head bone's transform (the rig's frame: the mount's parent is the RigView's)
      const i = head0;
      m.compose(rig.P[i], rig.S[i], one).multiply(unH.makeTranslation(-rig.H[i].x, -rig.H[i].y, -rig.H[i].z)).multiply(toBody);
      mount.matrix.copy(m);
      // its world transform now, through its parents' as they are now (they may have moved since
      // they were last drawn; a matrix set by hand isn't flagged as changed by itself)
      mount.matrixWorldNeedsUpdate = true;
      mount.updateWorldMatrix(true, true);
      // the features full from in front, dimmer as the face turns away (they're seen through the head); in the world's frame
      camera.getWorldPosition(eye);
      piv.copy(pivot).applyMatrix4(mount.matrixWorld);
      fwd.set(1, 0, 0).transformDirection(mount.matrixWorld);
      head.drawing.features = look.features * (0.12 + 0.88 * smooth((fwd.dot(v3.copy(eye).sub(piv).normalize()) + 0.35) / 0.6));
      head.face.update(tele, dt);
      head.camera.position.copy(eye);
      camera.getWorldQuaternion(head.camera.quaternion);
      head.camera.updateMatrixWorld();
      head.update(model, model.frame(a), a, tele, VIEW, dt);
      faceNow(a);
    },
    face(base: Float32Array, out: Float32Array) {
      if (out !== base) out.set(base);
      const S = rig.S[head0], sc = rig.sc[head0];
      // every point of the head's skin (the rim is the body's, and doesn't move)
      for (let p = 0; p < kept; p++) {
        v3.set(moved[3 * p], moved[3 * p + 1], moved[3 * p + 2]).multiplyScalar(sc / s).applyQuaternion(S);
        const o = 3 * (n0 + p);
        out[o] += v3.x;
        out[o + 1] += v3.y;
        out[o + 2] += v3.z;
      }
    },
    dispose() {
      mount.remove(head.headGroup);
      head.dispose();
    },
  };
}

/** a point `s` of the way along a polyline (flat xyz, evenly spaced points) into out[o…] */
function sample(line: ArrayLike<number>, s: number, out: Float32Array, o: number): void {
  const m = line.length / 3 - 1, f = Math.max(0, Math.min(1, s)) * m, i = Math.min(m - 1, Math.floor(f)), t = f - i;
  for (let c = 0; c < 3; c++) out[o + c] = line[3 * i + c] * (1 - t) + line[3 * i + 3 + c] * t;
}

/** the body's vertices within `rings` edges of the rim: the rim's first, in its order; how many edges each is from it; its neighbours as places in that list */
function neckPatch(rig: Rig, rings: number): { ids: number[]; hops: number[]; near: number[][] } {
  const at = new Map<number, number>(), ids: number[] = [], hops: number[] = [];
  let ring = rig.rim;
  for (let d = 0; d <= rings; d++) {
    const next: number[] = [];
    for (const v of ring) {
      if (at.has(v)) continue;
      at.set(v, ids.length);
      ids.push(v);
      hops.push(d);
    }
    for (const v of ring) for (const o of rig.near[v]) if (!at.has(o) && o < rig.bodyVertices) next.push(o);
    ring = next;
  }
  return { ids, hops, near: ids.map((v) => rig.near[v].map((o) => at.get(o) ?? -1).filter((o) => o >= 0)) };
}

/**
 * The body's neck under the rim: its weights to the neck and head bones (Rigify's: more at the
 * back than the front, so a turn skewed the lines) are given to the vertex's other bones, and
 * then it takes `share(height)` of the head's, the same at every turn round the neck.
 */
function neckWeights(rig: Rig, rimY: number, share: (y: number) => number, head: number): void {
  const neck = rig.bone("neck"), INF = rig.INF;
  for (let i = 0; i < rig.bodyVertices; i++) {
    const y = rig.rest[3 * i + 1];
    if (y < rimY - 12) continue;
    const list: [number, number][] = [];
    let other = 0;
    for (let j = 0; j < INF; j++) {
      const b = rig.skinIndex[INF * i + j], w = rig.skinWeight[INF * i + j];
      if (!w) continue;
      if (b !== neck && b !== head) {
        list.push([b, w]);
        other += w;
      }
    }
    if (!other) list.push([rig.bone("upperchest"), (other = 1)]);
    const tau = share(y);
    const out: [number, number][] = list.map(([b, w]) => [b, ((1 - tau) * w) / other]);
    out.push([head, tau]);
    out.sort((a, b) => b[1] - a[1]);
    // the strongest INF, summing to 1
    const top = out.slice(0, INF), sum = top.reduce((t, [, w]) => t + w, 0) || 1;
    for (let j = 0; j < INF; j++) {
      rig.skinIndex[INF * i + j] = top[j]?.[0] ?? 0;
      rig.skinWeight[INF * i + j] = (top[j]?.[1] ?? 0) / sum;
    }
  }
}
