/**
 * The body rig: the mesh tools/rig/gen-body.ts makes (MakeHuman's CC0 low-poly body on MPFB2's
 * Rigify human skeleton and weights: 33 bones and 10 helpers), posed by joint angles (pose.ts)
 * and skinned on the CPU.
 *
 * Posing. Every bone has its rest place (the mesh's, an A pose) and a "neutral" turn that takes
 * it from rest to anatomical zero (arms hanging, legs straight under the hips). A pose's angles
 * are turns about each joint relative to its parent, in that neutral frame; they're chained down
 * the skeleton. A bone's skinning transform takes a rest point to its posed place:
 *   x' = S (x − head) + P,   S its turn from rest, P where its head now is.
 * Twist about a limb's length is shared out along it (the twist bones), the shoulder girdle rises
 * as the arm does (scapulohumeral rhythm), and the spine's and neck's bending is shared down
 * their joints, each as much as its real joints take. Helper bones, as Rigify's: the shoulder's
 * cap and the hip's skin turn half way with the limb about its joint, and the elbow's and knee's
 * swell a little as they bend, so they keep their volume (`x' = s S (x − head) + P`, s ≥ 1).
 *
 * Skinning (`skin`): optimised centres of rotation (CoR, Le & Hodgins 2016: each vertex turns by
 * the blend of its bones' turns about a centre computed from where its weights are like its
 * neighbours'), the default; or plain linear blending (LBS: what other 3D tools do, which loses
 * volume at bends and twists to a "candy wrapper").
 *
 * Frames and units: body cm, x forward, y up, z out of the body's right side; the skeleton's rest
 * (bind) pose is the mesh's (an A pose), and every bone's transform is from rest: x' = S x + t.
 */
import * as THREE from "three";
import type { Arm, Leg, Pose } from "./pose.ts";

export interface BoneDef {
  name: string;
  parent: number;
  head: number[];
  tail: number[];
  helper?: boolean;
}
export interface Header {
  source: string;
  licence: string;
  bones: BoneDef[];
  points: Record<string, number[]>;
  rim: number[];
  rimShape: { centre: number[]; a: number; b: number };
  influences: number;
  vertices: number;
  quads: number;
}

export type Method = "lbs" | "cor";

/** How the rig shares motion between its bones. */
export interface Settings {
  /** of the upper arm's turn about its length, how much its first half (by the shoulder) takes */
  upperArmTwist: number;
  /** likewise the thigh's first half */
  thighTwist: number;
  /** how much the shoulder girdle rises and comes forward as the arm rises (the scapulohumeral rhythm; 1: as a body does) */
  rhythm: number;
  /** how far the shoulder's cap turns with the arm, about the shoulder joint (Rigify's: half) */
  deltoid: number;
  /** how far the hip's skin turns with the thigh, about the hip joint (Rigify's: half) */
  hip: number;
  /** how much the elbow and knee swell as they bend, at 90° (Rigify's: 0.1) */
  swell: number;
  /** how much the pelvis rolls back as the thighs come up (the lumbopelvic rhythm; 1: as a body does) */
  pelvis: number;
}
export const SETTINGS: Readonly<Settings> = { upperArmTwist: 0.25, thighTwist: 0.25, rhythm: 1, deltoid: 0.5, hip: 0.5, swell: 0.1, pelvis: 1 };

const smooth = (x: number) => {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3 - 2 * t);
};
/** how far the pelvis rolls back (degrees; negative: forward) for a thigh raised so far from the trunk */
function pelvicRoll(flex: number): number {
  if (flex < -10) return 0.5 * (flex + 10);
  if (flex <= 55) return 0;
  if (flex <= 110) return 0.4 * (flex - 55);
  return 22 + 0.9 * (flex - 110);
}

const X = new THREE.Vector3(1, 0, 0), Y = new THREE.Vector3(0, 1, 0), Z = new THREE.Vector3(0, 0, 1);
const DOWN = new THREE.Vector3(0, -1, 0);
const RAD = Math.PI / 180;
const rot = (axis: THREE.Vector3, deg: number) => new THREE.Quaternion().setFromAxisAngle(axis, deg * RAD);
/** a right-side local turn mirrored for the left (mirroring across z: x and y components flip) */
const mirror = (q: THREE.Quaternion, left: boolean) => (left ? q.set(-q.x, -q.y, q.z, q.w) : q);

/** how the spine's and neck's bending is shared among their joints, lowest first */
const SPINE = ["spine1", "spine2", "chest", "upperchest"];
const SPINE_SHARE = { flex: [0.32, 0.28, 0.22, 0.18], bend: [0.25, 0.25, 0.25, 0.25], twist: [0.1, 0.2, 0.3, 0.4] };
const NECK = ["neck", "head"];
const NECK_SHARE = { flex: [0.6, 0.4], bend: [0.65, 0.35], twist: [0.45, 0.55] };

export class Rig {
  readonly header: Header;
  n: number;
  nq: number;
  readonly INF: number;
  /** rest positions (cm), n × 3 */
  rest: Float32Array;
  quads: Uint32Array;
  skinIndex: Uint8Array;
  skinWeight: Float32Array;
  /** each vertex's lines' brightness (1: the body's) */
  tone: Float32Array;
  /** how many of the vertices are the body's own (the rest were attached: a head's skin) */
  readonly bodyVertices: number;
  readonly bones: BoneDef[];
  readonly B: number;
  readonly idx: Record<string, number>;
  /** each vertex's neighbours along the mesh's edges */
  near: number[][] = [];
  /** each vertex's quads */
  quadsOf: number[][] = [];
  readonly rim: number[];
  /** vertices on an open edge (the neck's rim, or a head's lips) */
  open: Uint8Array = new Uint8Array(0);
  readonly settings: Settings = { ...SETTINGS };

  /** rest joint places */
  readonly H: THREE.Vector3[];
  /** rest → neutral turn per bone */
  readonly N: THREE.Quaternion[];
  /** posed: each bone's turn from rest, where its head is, and the transform's shift (x' = S x + t) */
  readonly S: THREE.Quaternion[];
  readonly P: THREE.Vector3[];
  readonly t: THREE.Vector3[];
  /** each bone's world turn from neutral */
  readonly W: THREE.Quaternion[];
  /** each bone's swelling (helpers; 1: none) */
  readonly sc: number[];
  /** how far the pelvis rolled back for the last pose (degrees) */
  roll = 0;

  private corCache: Float32Array | null = null;

  constructor(buf: ArrayBuffer) {
    const dv = new DataView(buf);
    const len = dv.getUint32(0, true);
    const header = (this.header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, len))) as Header);
    const n = (this.n = header.vertices), nq = (this.nq = header.quads), INF = (this.INF = header.influences);
    let o = 4 + len;
    this.rest = new Float32Array(buf.slice(o, o + 12 * n));
    o += 12 * n;
    this.skinIndex = new Uint8Array(buf.slice(o, o + INF * n));
    o += INF * n;
    const w16 = new Uint16Array(buf.slice(o, o + 2 * INF * n));
    o += 2 * INF * n;
    this.skinWeight = Float32Array.from(w16, (w) => w / 65535);
    this.quads = Uint32Array.from(new Uint16Array(buf.slice(o, o + 8 * nq)));
    this.bones = header.bones;
    this.B = this.bones.length;
    this.idx = Object.fromEntries(this.bones.map((b, i) => [b.name, i]));
    this.rim = header.rim;
    this.tone = new Float32Array(n).fill(1);
    this.bodyVertices = n;
    this.topology();
    this.H = this.bones.map((b) => new THREE.Vector3().fromArray(b.head));
    this.S = this.bones.map(() => new THREE.Quaternion());
    this.W = this.bones.map(() => new THREE.Quaternion());
    this.P = this.H.map((h) => h.clone());
    this.t = this.bones.map(() => new THREE.Vector3());
    this.sc = this.bones.map(() => 1);
    this.N = this.neutralTurns();
  }

  /** each vertex's neighbours and quads, and which are on an open edge */
  private topology(): void {
    const { n, nq } = this;
    const near: Set<number>[] = Array.from({ length: n }, () => new Set());
    this.quadsOf = Array.from({ length: n }, () => []);
    const edges = new Map<number, number>();
    for (let f = 0; f < nq; f++) {
      for (let j = 0; j < 4; j++) {
        const a = this.quads[4 * f + j], b = this.quads[4 * f + ((j + 1) % 4)];
        near[a].add(b);
        near[b].add(a);
        this.quadsOf[a].push(f);
        const k = Math.min(a, b) * 1048576 + Math.max(a, b);
        edges.set(k, (edges.get(k) ?? 0) + 1);
      }
    }
    this.near = near.map((s) => [...s]);
    this.open = new Uint8Array(n);
    for (const [k, c] of edges) {
      if (c !== 1) continue;
      this.open[Math.floor(k / 1048576)] = 1;
      this.open[k % 1048576] = 1;
    }
  }

  /**
   * Add vertices and quads to the mesh (a head's skin, grown from the neck's rim): `points` at
   * rest (cm, flat xyz), `quads` indexing the whole mesh (the new points numbered on from the
   * existing vertices), each point's bones and weights, and its lines' brightness. The quads are
   * turned to face the way the body's do where they meet it.
   */
  attach(points: Float32Array, quads: ArrayLike<number>, skin: [number, number][][], tone?: ArrayLike<number>): void {
    const m = points.length / 3, n0 = this.n, INF = this.INF;
    const rest = new Float32Array(3 * (n0 + m));
    rest.set(this.rest);
    rest.set(points, 3 * n0);
    const SI = new Uint8Array(INF * (n0 + m)), SW = new Float32Array(INF * (n0 + m));
    SI.set(this.skinIndex);
    SW.set(this.skinWeight);
    skin.forEach((list, p) => {
      const top = [...list].sort((a, b) => b[1] - a[1]).slice(0, INF), sum = top.reduce((s, [, w]) => s + w, 0) || 1;
      top.forEach(([b, w], j) => {
        SI[INF * (n0 + p) + j] = b;
        SW[INF * (n0 + p) + j] = w / sum;
      });
    });
    // which way round: an edge the body and the new quads share runs opposite ways in each
    const dir = new Map<number, number>();
    for (let f = 0; f < this.nq; f++) for (let j = 0; j < 4; j++) dir.set(this.quads[4 * f + j] * 1048576 + this.quads[4 * f + ((j + 1) % 4)], 1);
    let same = 0, opposite = 0;
    for (let f = 0; f < quads.length; f += 4) {
      for (let j = 0; j < 4; j++) {
        const a = quads[f + j], b = quads[f + ((j + 1) % 4)];
        if (dir.has(a * 1048576 + b)) same++;
        if (dir.has(b * 1048576 + a)) opposite++;
      }
    }
    const Q = Uint32Array.from(quads);
    if (same > opposite) for (let f = 0; f < Q.length; f += 4) [Q[f + 1], Q[f + 3]] = [Q[f + 3], Q[f + 1]];
    const all = new Uint32Array(this.quads.length + Q.length);
    all.set(this.quads);
    all.set(Q, this.quads.length);
    const T = new Float32Array(n0 + m).fill(1);
    T.set(this.tone);
    if (tone) T.set(tone, n0);
    Object.assign(this, { rest, skinIndex: SI, skinWeight: SW, quads: all, tone: T, n: n0 + m, nq: all.length / 4 });
    this.topology();
    this.weightsChanged();
  }

  static async load(url: string): Promise<Rig> {
    return new Rig(await (await fetch(url)).arrayBuffer());
  }

  bone(name: string): number {
    const i = this.idx[name];
    if (i === undefined) throw new Error(`no bone ${name}`);
    return i;
  }

  // ------------------------------------------------------------------ neutral
  /**
   * The turn that takes each bone from the mesh's rest (an A pose: arms out and down at about
   * 40°, elbows a little bent, legs a little apart) to anatomical zero: each arm straight down,
   * the hand on along the forearm, each leg's hip-to-ankle line straight down seen from the
   * front. The forearm keeps its rest turn about its length (no twist at zero).
   */
  private neutralTurns(): THREE.Quaternion[] {
    const N = this.bones.map(() => new THREE.Quaternion());
    const h = (name: string) => this.H[this.bone(name)];
    const pt = (name: string) => new THREE.Vector3().fromArray(this.header.points[name]);
    for (const side of ["R", "L"]) {
      const b = (name: string) => this.bone(`${name}.${side}`);
      const sh = h(`upperarm.${side}`), el = h(`forearm.${side}`), wr = h(`hand.${side}`);
      const qa = new THREE.Quaternion().setFromUnitVectors(el.clone().sub(sh).normalize(), DOWN);
      N[b("upperarm")].copy(qa);
      N[b("upperarm2")].copy(qa);
      const f = wr.clone().sub(el).normalize().applyQuaternion(qa);
      const qf = new THREE.Quaternion().setFromUnitVectors(f, DOWN).multiply(qa);
      N[b("forearm")].copy(qf);
      N[b("forearm2")].copy(qf);
      const hand = pt(`finger3-1.${side}`).sub(wr).normalize().applyQuaternion(qf);
      N[b("hand")].copy(new THREE.Quaternion().setFromUnitVectors(hand, DOWN).multiply(qf));
      const hip = h(`thigh.${side}`), ankle = h(`foot.${side}`);
      const v = ankle.clone().sub(hip);
      v.x = 0;
      const ql = new THREE.Quaternion().setFromUnitVectors(v.normalize(), DOWN);
      for (const name of ["thigh", "thigh2", "shin", "shin2"]) N[b(name)].copy(ql);
    }
    return N;
  }

  // ------------------------------------------------------------------ pose
  /** Pose the skeleton: every bone's S, P and t from the pose's angles. */
  pose(p: Pose): void {
    const L = this.bones.map(() => new THREE.Quaternion());
    const set = (name: string, q: THREE.Quaternion) => L[this.bone(name)].copy(q);
    const S = this.settings;
    // The whole body; and the pelvis rolled back as the thighs come up (the lumbopelvic rhythm:
    // past about 55° of hip flexion the pelvis takes a share of each further degree, and past the
    // hip's own limit nearly all of it; it tips forward a little as a thigh goes back). The lower
    // back bends forward as much, so the chest stays where the spine's sliders put it, and the
    // thighs are measured from the trunk, so they stay where theirs do.
    this.roll = (S.pelvis * (pelvicRoll(p.legs.R.flex) + pelvicRoll(p.legs.L.flex))) / 2;
    set("hips", rot(Y, p.body.turn).multiply(rot(Z, -p.body.tip)).multiply(rot(X, p.body.lean)).multiply(rot(Z, this.roll)));
    const bendy = (names: string[], share: { flex: number[]; bend: number[]; twist: number[] }, a: { flex: number; bend: number; twist: number }) =>
      names.forEach((name, j) => set(name, rot(Y, share.twist[j] * a.twist).multiply(rot(Z, -share.flex[j] * a.flex)).multiply(rot(X, share.bend[j] * a.bend))));
    bendy(SPINE, SPINE_SHARE, p.spine);
    bendy(NECK, NECK_SHARE, p.neck);
    L[this.bone("spine1")].premultiply(rot(Z, -0.55 * this.roll));
    L[this.bone("spine2")].premultiply(rot(Z, -0.45 * this.roll));
    for (const side of ["R", "L"] as const) {
      const left = side === "L", b = (name: string) => `${name}.${side}`;
      this.arm(p.arms[side], left, b, set, S);
      this.leg(p.legs[side], left, b, set, S, this.roll);
    }
    // chained: each bone's world turn from neutral
    for (let i = 0; i < this.B; i++) {
      const par = this.bones[i].parent;
      if (par < 0) this.W[i].copy(L[i]);
      else this.W[i].copy(this.W[par]).multiply(L[i]);
    }
    this.poseWorld(this.W, this.H[0].clone().add(new THREE.Vector3(p.body.x, p.body.y, p.body.z)));
  }

  /**
   * Pose the skeleton from each bone's world turn from neutral (`W`, as `this.W`: helpers' are
   * ignored) and where the root's joint goes (cm): every bone's S, P and t. `pose` comes here
   * after working the turns out from angles; motion from elsewhere (motion.ts) comes straight here.
   */
  poseWorld(W: readonly THREE.Quaternion[], root: THREE.Vector3): void {
    for (let i = 0; i < this.B; i++) {
      this.sc[i] = 1;
      if (W !== this.W) this.W[i].copy(W[i]);
      const par = this.bones[i].parent;
      if (par < 0) this.P[i].copy(root);
      else this.P[i].copy(this.H[i]).sub(this.H[par]).applyQuaternion(this.S[par]).add(this.P[par]);
      this.S[i].copy(this.W[i]).multiply(this.N[i]);
    }
    // helpers: each turns part of the way from one neighbour's turn to the other's, about the
    // joint between them; the elbow's and knee's swell as they bend
    const S = this.settings;
    // the bend (degrees) at a hinge between two bones that hang down at anatomical zero, in the
    // hinge's plane: the elbow's forward, the knee's back (negative: straightened past straight)
    const bend = (a: string, c: string, forward: number) => {
      const v = DOWN.clone().applyQuaternion(this.W[this.bone(c)]).applyQuaternion(this.W[this.bone(a)].clone().invert());
      return Math.atan2(forward * v.x, -v.y) / RAD;
    };
    for (const side of ["R", "L"] as const) {
      const b = (n: string) => `${n}.${side}`;
      this.helper(b("shoulderHelper"), b("clavicle"), b("upperarm"), S.deltoid);
      this.helper(b("elbowHelper"), b("upperarm2"), b("forearm"), 0.5, 1 + (S.swell * bend(b("upperarm2"), b("forearm"), 1)) / 90);
      this.helper(b("kneeHelper"), b("thigh2"), b("shin"), 0.5, 1 + (S.swell * bend(b("thigh2"), b("shin"), -1)) / 90);
      this.helper(b("hipHelper"), "hips", b("thigh"), S.hip);
      this.helper(b("hipFront"), "hips", b("thigh"), S.hip);
    }
    for (let i = 0; i < this.B; i++) this.t[i].copy(this.H[i]).applyQuaternion(this.S[i]).multiplyScalar(-this.sc[i]).add(this.P[i]);
  }

  /** bone `name` (its head at the joint `to` hangs from) turns `share` of the way from `from`'s turn to `to`'s, and swells by `swell` */
  private helper(name: string, from: string, to: string, share: number, swell = 1): void {
    const i = this.bone(name), a = this.bone(from), c = this.bone(to);
    this.S[i].slerpQuaternions(this.S[a], this.S[c], share);
    this.P[i].copy(this.P[c]);
    this.sc[i] = swell;
  }

  private arm(a: Arm, left: boolean, b: (n: string) => string, set: (n: string, q: THREE.Quaternion) => void, S: Settings): void {
    // The shoulder girdle: what the pose asks, and what the arm's rise brings on its own (the
    // scapulohumeral rhythm, as measured in living shoulders: Ludewig 2009). Past the first 30°
    // the collarbone's outer end rises, but only some 10° by the time the arm is straight up (a
    // shrug reads as wrong); it comes forward as the arm reaches forward or across, and goes back
    // as the arm goes overhead. The arm also turns outward as it rises out to the side past 90°,
    // as an arm must to get there.
    const sn = Math.sin(a.plane * RAD), cs = Math.cos(a.plane * RAD);
    const up = smooth((a.raise - 30) / 150), level = Math.sin((Math.min(180, a.raise) * Math.PI) / 180);
    const across = Math.max(0, sn) + Math.max(0, (a.plane - 90) / 90);
    const shrug = a.shrug + S.rhythm * 10 * up;
    const reach = a.reach + S.rhythm * (14 * across * level - 12 * up);
    const turnOut = S.rhythm * 40 * smooth((a.raise - 90) / 90) * cs * cs;
    const clav = rot(Y, reach).multiply(rot(X, -shrug));
    // the arm's direction and turn in the chest's frame: raised `raise` about the horizontal axis
    // square to its plane (the "globe": plane of elevation, elevation, axial rotation)
    const axis = new THREE.Vector3(-Math.cos(a.plane * RAD), 0, Math.sin(a.plane * RAD));
    const globe = rot(axis, a.raise);
    const gh = clav.clone().invert().multiply(globe);
    const k = S.upperArmTwist;
    set(b("clavicle"), mirror(clav, left));
    const rotate = a.rotate + turnOut;
    set(b("upperarm"), mirror(gh.multiply(rot(DOWN, k * rotate)), left));
    set(b("upperarm2"), mirror(rot(DOWN, (1 - k) * rotate), left));
    // the elbow a hinge; the forearm's turn shared out along it: a quarter, three quarters, all
    set(b("forearm"), mirror(rot(Z, a.elbow).multiply(rot(Y, 0.25 * a.turn)), left));
    set(b("forearm2"), mirror(rot(Y, 0.5 * a.turn), left));
    set(b("hand"), mirror(rot(Y, 0.25 * a.turn).multiply(rot(Z, a.deviate)).multiply(rot(X, a.wrist)), left));
  }

  private leg(l: Leg, left: boolean, b: (n: string) => string, set: (n: string, q: THREE.Quaternion) => void, S: Settings, roll: number): void {
    const k = S.thighTwist;
    set(b("thigh"), mirror(rot(Z, l.flex - roll).multiply(rot(X, -l.abduct)).multiply(rot(DOWN, k * l.rotate)), left));
    set(b("thigh2"), mirror(rot(DOWN, (1 - k) * l.rotate), left));
    set(b("shin"), mirror(rot(Z, -l.knee), left));
    set(b("foot"), mirror(rot(Z, l.ankle).multiply(rot(X, l.invert)), left));
    set(b("toes"), mirror(rot(Z, l.toes), left));
  }

  // ------------------------------------------------------------------ skinning
  /** Each vertex's centre of rotation (Le & Hodgins 2016), from the weights (worked out once). */
  cor(): Float32Array {
    return (this.corCache ??= centresOfRotation(this));
  }

  /** forget what was worked out from the rest mesh and weights (after either changes) */
  weightsChanged(): void {
    this.corCache = null;
  }

  /** the posed mesh into `out` (n × 3), after `pose` */
  skin(out: Float32Array, method: Method = "cor"): void {
    const { n, INF, rest, skinIndex: SI, skinWeight: SW, S, t } = this;
    const sc = this.sc;
    const m = this.bones.map((_, i) => new THREE.Matrix4().compose(t[i], S[i], new THREE.Vector3(sc[i], sc[i], sc[i])).elements);
    if (method === "lbs") {
      for (let i = 0; i < n; i++) {
        const x = rest[3 * i], y = rest[3 * i + 1], z = rest[3 * i + 2];
        let ox = 0, oy = 0, oz = 0;
        for (let j = 0; j < INF; j++) {
          const w = SW[INF * i + j];
          if (!w) continue;
          const e = m[SI[INF * i + j]];
          ox += w * (e[0] * x + e[4] * y + e[8] * z + e[12]);
          oy += w * (e[1] * x + e[5] * y + e[9] * z + e[13]);
          oz += w * (e[2] * x + e[6] * y + e[10] * z + e[14]);
        }
        out[3 * i] = ox;
        out[3 * i + 1] = oy;
        out[3 * i + 2] = oz;
      }
    } else {
      const C = this.cor();
      const q = new THREE.Quaternion(), v = new THREE.Vector3();
      for (let i = 0; i < n; i++) {
        let piv = SI[INF * i];
        for (let j = 1; j < INF; j++) if (SW[INF * i + j] > SW[INF * i]) piv = SI[INF * i + j];
        let qx = 0, qy = 0, qz = 0, qw = 0, cx = 0, cy = 0, cz = 0, ss = 0;
        const px = C[3 * i], py = C[3 * i + 1], pz = C[3 * i + 2];
        for (let j = 0; j < INF; j++) {
          const w = SW[INF * i + j];
          if (!w) continue;
          const bi = SI[INF * i + j], s = S[bi], e = m[bi];
          const sg = s.dot(S[piv]) < 0 ? -w : w;
          ss += w * sc[bi];
          qx += sg * s.x;
          qy += sg * s.y;
          qz += sg * s.z;
          qw += sg * s.w;
          cx += w * (e[0] * px + e[4] * py + e[8] * pz + e[12]);
          cy += w * (e[1] * px + e[5] * py + e[9] * pz + e[13]);
          cz += w * (e[2] * px + e[6] * py + e[10] * pz + e[14]);
        }
        q.set(qx, qy, qz, qw).normalize();
        v.set(rest[3 * i] - px, rest[3 * i + 1] - py, rest[3 * i + 2] - pz).applyQuaternion(q).multiplyScalar(ss);
        out[3 * i] = v.x + cx;
        out[3 * i + 1] = v.y + cy;
        out[3 * i + 2] = v.z + cz;
      }
    }
  }

}


/** area-weighted vertex normals of the mesh at P */
export function vertexNormals(rig: Rig, P: Float32Array<ArrayBufferLike>, out: Float32Array<ArrayBufferLike> = new Float32Array(3 * rig.n)): Float32Array<ArrayBufferLike> {
  out.fill(0);
  const Q = rig.quads;
  for (let f = 0; f < rig.nq; f++) {
    const a = 3 * Q[4 * f], b = 3 * Q[4 * f + 1], c = 3 * Q[4 * f + 2], d = 3 * Q[4 * f + 3];
    // the diagonals' cross product: twice the quad's area along its normal
    const ux = P[c] - P[a], uy = P[c + 1] - P[a + 1], uz = P[c + 2] - P[a + 2];
    const vx = P[d] - P[b], vy = P[d + 1] - P[b + 1], vz = P[d + 2] - P[b + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const o of [a, b, c, d]) {
      out[o] += nx;
      out[o + 1] += ny;
      out[o + 2] += nz;
    }
  }
  for (let i = 0; i < rig.n; i++) {
    const l = Math.hypot(out[3 * i], out[3 * i + 1], out[3 * i + 2]) || 1;
    out[3 * i] /= l;
    out[3 * i + 1] /= l;
    out[3 * i + 2] /= l;
  }
  return out;
}

/**
 * Optimised centres of rotation (Le & Hodgins, "Real-time Skeletal Skinning with Optimized
 * Centers of Rotation", SIGGRAPH 2016): a vertex's centre is the area-weighted mean of the
 * mesh's triangles' centroids, each weighted by how alike its skin weights are to the vertex's:
 *   s(wp, wt) = Σ_{j≠k} wpj wpk wtj wtk exp(−(wpj wtk − wpk wtj)² / σ²).
 * Only bone pairs both present matter, so each vertex visits just the triangles that share a
 * pair of its bones. A vertex no triangle is like (one bone) keeps itself: it moves rigidly.
 */
export function centresOfRotation(rig: Rig, sigma = 0.1): Float32Array {
  const { n, nq, INF, B, rest: P, skinIndex: SI, skinWeight: SW, quads: Q } = rig;
  const s2 = sigma * sigma;
  // triangles (each quad as two), their areas, centroids and dense weights
  const T = 2 * nq;
  const area = new Float32Array(T), cen = new Float32Array(3 * T), TW = new Float32Array(T * B);
  const dense = (v: number, into: Float32Array, at: number, f: number) => {
    for (let j = 0; j < INF; j++) into[at + SI[INF * v + j]] += f * SW[INF * v + j];
  };
  for (let f = 0; f < nq; f++) {
    const tris = [[Q[4 * f], Q[4 * f + 1], Q[4 * f + 2]], [Q[4 * f], Q[4 * f + 2], Q[4 * f + 3]]];
    tris.forEach((tri, h) => {
      const t = 2 * f + h, [a, b, c] = tri;
      const ux = P[3 * b] - P[3 * a], uy = P[3 * b + 1] - P[3 * a + 1], uz = P[3 * b + 2] - P[3 * a + 2];
      const vx = P[3 * c] - P[3 * a], vy = P[3 * c + 1] - P[3 * a + 1], vz = P[3 * c + 2] - P[3 * a + 2];
      area[t] = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
      for (let k = 0; k < 3; k++) cen[3 * t + k] = (P[3 * a + k] + P[3 * b + k] + P[3 * c + k]) / 3;
      for (const v of tri) dense(v, TW, t * B, 1 / 3);
    });
  }
  // for each pair of bones, the triangles that have both
  const pairs = new Map<number, number[]>();
  for (let t = 0; t < T; t++) {
    const bs: number[] = [];
    for (let b = 0; b < B; b++) if (TW[t * B + b] > 1e-6) bs.push(b);
    for (let x = 0; x < bs.length; x++) for (let y = x + 1; y < bs.length; y++) {
      const k = bs[x] * B + bs[y];
      let l = pairs.get(k);
      if (!l) pairs.set(k, (l = []));
      l.push(t);
    }
  }
  const C = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const bs: number[] = [], ws: number[] = [];
    for (let j = 0; j < INF; j++) if (SW[INF * i + j] > 1e-6) {
      bs.push(SI[INF * i + j]);
      ws.push(SW[INF * i + j]);
    }
    let sx = 0, sy = 0, sz = 0, sw = 0;
    for (let x = 0; x < bs.length; x++) for (let y = x + 1; y < bs.length; y++) {
      const j = Math.min(bs[x], bs[y]), k = Math.max(bs[x], bs[y]);
      const wj = bs[x] === j ? ws[x] : ws[y], wk = bs[x] === j ? ws[y] : ws[x];
      const list = pairs.get(j * B + k);
      if (!list) continue;
      for (const t of list) {
        const tj = TW[t * B + j], tk = TW[t * B + k], dd = wj * tk - wk * tj;
        const s = 2 * wj * wk * tj * tk * Math.exp(-(dd * dd) / s2) * area[t];
        sx += s * cen[3 * t];
        sy += s * cen[3 * t + 1];
        sz += s * cen[3 * t + 2];
        sw += s;
      }
    }
    if (sw > 1e-12) C.set([sx / sw, sy / sw, sz / sw], 3 * i);
    else C.set([P[3 * i], P[3 * i + 1], P[3 * i + 2]], 3 * i);
  }
  return C;
}
