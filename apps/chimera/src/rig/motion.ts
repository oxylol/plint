/**
 * Motion from other tools: a clip (Mixamo's FBX, a BVH, a glTF's animation, or anything three.js
 * loads) played on its own skeleton and carried over onto the rig, frame by frame.
 *
 *   const motion = await Motion.load(rig, await file.arrayBuffer(), file.name);
 *   motion.pose(t);          // instead of rig.pose(…): the rig as the clip is t seconds in
 *   rig.skin(skinned);
 *
 * How. The source's bones are matched to the rig's by name (limbs and head; the trunk by where it
 * sits between them). Its bind pose is turned limb by limb until each limb points as the rig's do
 * at anatomical zero (its arms hanging, legs straight), which is the source's "neutral". Each
 * frame, a source bone's turn since its neutral, in the world, is the rig bone's turn since its
 * own (the world frames matched by where the source's head and hips are), so the motion carries
 * over whatever the source's bones' own axes are. A spine or neck of a different number of bones
 * is spread along the rig's. The limbs' twist is shared along their twist bones as the rig's
 * posing does, and the hips move as the source's do, scaled to the rig's legs.
 */
import * as THREE from "three";
import type { Rig } from "./rig.ts";

export interface MotionOptions {
  /** which of the file's clips: its name or index (default: the longest) */
  clip?: string | number;
  /** source bone names to rig bones, for names the matcher misses: { "DEF-spine.006": "head" } */
  map?: Record<string, string>;
}

const SIDES = ["L", "R"] as const;
const LIMB: Record<string, string> = {
  shoulder: "clavicle", clavicle: "clavicle", collar: "clavicle", collarbone: "clavicle",
  arm: "upperarm", upperarm: "upperarm", uparm: "upperarm", shldr: "upperarm",
  forearm: "forearm", lowerarm: "forearm",
  hand: "hand",
  upleg: "thigh", upperleg: "thigh", thigh: "thigh",
  leg: "shin", lowerleg: "shin", shin: "shin", calf: "shin",
  foot: "foot",
  toebase: "toes", toe: "toes", toes: "toes", toe0: "toes", ball: "toes",
};
/** name parts that say nothing about the bone: rig and exporter prefixes */
const NOISE = /^(mixamorig\d*|def|org|j|bip0*\d*|c|character\d*)$/i;

/**
 * A source bone's name as the rig's (`upperarm.L`, `head`), or null: Mixamo, CMU and DAZ BVH,
 * VRM/VRoid, Unreal, Rigify, 3ds Max Biped and the rig's own names. Twist bones, fingers and end
 * sites don't match.
 */
export function rigName(name: string): string | null {
  const parts = name
    .replace(/^.*:/, "")
    .replace(/^mixamorig\d*(?=[A-Z_])/, "")
    .split(/[\s._-]+/)
    .filter((p) => p && !NOISE.test(p));
  let side = "";
  const i = parts.findIndex((p) => /^(l|r|left|right)$/i.test(p));
  if (i >= 0) side = parts.splice(i, 1)[0][0].toUpperCase();
  for (let j = 0; j < parts.length && !side; j++) {
    const m = parts[j].match(/^(left|right)(?=.)/i) ?? parts[j].match(/^([lr])(?=[A-Z])/);
    if (m) {
      side = m[1][0].toUpperCase();
      parts[j] = parts[j].slice(m[1].length);
    }
  }
  const last = parts.length - 1;
  if (!side && last >= 0 && /[a-z][LR]$/.test(parts[last])) {
    side = parts[last].slice(-1);
    parts[last] = parts[last].slice(0, -1);
  }
  const key = parts.join("").toLowerCase().replace(/[^a-z0-9]/g, "");
  if (!side) return key === "head" ? "head" : null;
  return LIMB[key] ? `${LIMB[key]}.${side}` : null;
}

/**
 * Read a motion file (by its name's extension: .fbx, .bvh, .glb, or a .gltf with nothing in
 * separate files) with three.js's loader for it: its skeleton's root (or the scene holding it)
 * and its clips.
 */
export async function readMotion(data: ArrayBuffer, fileName: string): Promise<{ root: THREE.Object3D; clips: THREE.AnimationClip[] }> {
  const ext = fileName.toLowerCase().split(".").pop();
  if (ext === "bvh") {
    const { BVHLoader } = await import("three/addons/loaders/BVHLoader.js");
    const r = new BVHLoader().parse(new TextDecoder().decode(data));
    return { root: r.skeleton.bones[0], clips: [r.clip] };
  }
  if (ext === "fbx") {
    const { FBXLoader } = await import("three/addons/loaders/FBXLoader.js");
    const root = new FBXLoader().parse(data, "");
    return { root, clips: root.animations };
  }
  if (ext === "glb" || ext === "gltf") {
    const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
    const g = await new GLTFLoader().parseAsync(data, "").catch((e: unknown) => {
      throw new Error(`motion: can't read ${fileName} (${e instanceof Error ? e.message : e}); a .gltf with separate files: use a .glb, or load it yourself and pass its scene to new Motion()`);
    });
    return { root: g.scene, clips: g.animations };
  }
  throw new Error(`motion: can't read .${ext} (fbx, bvh, glb or gltf)`);
}

const DOWN = new THREE.Vector3(0, -1, 0);
/** the angle (radians) q turns about `axis`, as the twist in q = swing · twist */
function twistAbout(q: THREE.Quaternion, axis: THREE.Vector3): number {
  const s = q.w < 0 ? -1 : 1;
  return 2 * Math.atan2(s * (q.x * axis.x + q.y * axis.y + q.z * axis.z), s * q.w);
}
const about = (axis: THREE.Vector3, a: number) => new THREE.Quaternion().setFromAxisAngle(axis, a);

export class Motion {
  readonly duration: number;
  /** which source bone drives which rig bone */
  readonly bones: Record<string, string> = {};

  private readonly mixer: THREE.AnimationMixer;
  private readonly src = new Map<string, THREE.Object3D>();
  /** each matched source bone's world turn at its neutral, inverted */
  private readonly neutral = new Map<THREE.Object3D, THREE.Quaternion>();
  private readonly spine: THREE.Object3D[];
  private readonly neck: THREE.Object3D[];
  private readonly hips: THREE.Object3D;
  /** the source's world turned into the rig's body frame (x forward, y up, z right), and how much
   *  larger the rig is than the source's units: to show the source beside it, say */
  readonly C: THREE.Quaternion;
  readonly scale: number;
  private readonly up: THREE.Vector3;
  private readonly floor: number;
  private readonly W: THREE.Quaternion[];

  /**
   * Load a motion file for `rig` (`readMotion`), playing the clip `opts.clip` (its name or index;
   * default: the longest). For another clip, load the file again: a Motion takes its scene over.
   */
  static async load(rig: Rig, data: ArrayBuffer, fileName: string, opts: MotionOptions = {}): Promise<Motion> {
    const { root, clips } = await readMotion(data, fileName);
    const clip = typeof opts.clip === "string" ? clips.find((c) => c.name === opts.clip)
      : typeof opts.clip === "number" ? clips[opts.clip]
      : [...clips].sort((a, b) => b.duration - a.duration)[0];
    if (!clip) throw new Error(`motion: no clip ${opts.clip ?? ""} in ${fileName} (${clips.map((c) => c.name).join(", ") || "none"})`);
    return new Motion(rig, root, clip, opts);
  }

  /**
   * `root`: the source skeleton's root (or a scene holding it), as loaded: its pose then is the
   * reference its limbs are turned from (a T or an A pose, palms down or in). `clip` animates it
   * (track names are its bones' names, as three.js loaders make them). The Motion plays the clip
   * on `root` from then on: move it by its parent, not itself.
   */
  constructor(readonly rig: Rig, readonly root: THREE.Object3D, readonly clip: THREE.AnimationClip, opts: MotionOptions = {}) {
    this.duration = clip.duration;
    this.W = rig.bones.map(() => new THREE.Quaternion());
    // match: each name to the bone the clip moves by that name (an FBX has same-named copies);
    // opts.map's first, then the matcher's. Two bones for one of the rig's: the one the clip
    // moves, or neither (name it in opts.map)
    const find = (n: string) => THREE.PropertyBinding.findNode(root, n) as THREE.Object3D | null;
    for (const [n, name] of Object.entries(opts.map ?? {})) {
      const o = find(n);
      if (!o) throw new Error(`motion: no bone ${n} (opts.map)`);
      rig.bone(name);
      this.src.set(name, o);
    }
    const animated = new Set(clip.tracks.map((t) => THREE.PropertyBinding.parseTrackName(t.name).nodeName));
    const found = new Map<string, Set<THREE.Object3D>>();
    root.traverse((o) => {
      const name = opts.map?.[o.name] ? null : rigName(o.name), b = name && !this.src.has(name) ? find(o.name) : null;
      if (name && b) found.set(name, (found.get(name) ?? new Set()).add(b));
    });
    for (const [name, set] of found) {
      const all = [...set], moving = all.filter((o) => animated.has(o.name));
      const one = all.length === 1 ? all[0] : moving.length === 1 ? moving[0] : null;
      if (!one) throw new Error(`motion: the rig's ${name} could be ${all.map((o) => o.name).join(" or ")}; name one in opts.map`);
      this.src.set(name, one);
    }
    for (const [name, o] of this.src) this.bones[name] = o.name;
    const need = (name: string) => {
      const o = this.src.get(name);
      if (!o) throw new Error(`motion: no bone for the rig's ${name} (found ${Object.keys(this.bones).join(", ") || "none"}); name it in opts.map`);
      return o;
    };
    // the trunk: the hips join the legs and the head; the spine runs up to where the arms hang
    const head = need("head"), thighs = SIDES.map((s) => need(`thigh.${s}`));
    const up = (o: THREE.Object3D) => { const l: THREE.Object3D[] = []; for (let p: THREE.Object3D | null = o; p; p = p.parent) l.push(p); return l; };
    const join = (...os: THREE.Object3D[]) => {
      const chains = os.map(up);
      return chains[0].find((o) => chains.every((c) => c.includes(o)))!;
    };
    const hips = (this.hips = join(head, ...thighs));
    const shoulder = join(head, ...SIDES.map((s) => this.src.get(`clavicle.${s}`) ?? need(`upperarm.${s}`)));
    const between = (from: THREE.Object3D, to: THREE.Object3D) => {
      const l = up(to), k = l.indexOf(from);
      if (k < 0) throw new Error(`motion: ${to.name} isn't below ${from.name}`);
      return l.slice(0, k).reverse();
    };
    this.spine = between(hips, shoulder);
    this.neck = between(shoulder, head);

    // as loaded, the frames: the source's up (to the nearest axis), and its right from its hips
    this.update();
    // (a mirrored or unevenly scaled bone has no plain turn to read)
    for (const o of new Set([hips, ...this.spine, ...this.neck, ...this.src.values()])) {
      const m = this.m.multiplyMatrices(this.base, o.matrixWorld), sc = new THREE.Vector3().setFromMatrixScale(m);
      if (m.determinant() < 0) throw new Error(`motion: ${o.name} is mirrored (a negative scale)`);
      if (Math.max(sc.x, sc.y, sc.z) > 1.001 * Math.min(sc.x, sc.y, sc.z)) throw new Error(`motion: ${o.name} is scaled unevenly`);
    }
    const u = this.pos(head).sub(this.pos(hips)).normalize();
    const k = [0, 1, 2].reduce((a, b) => (Math.abs(u.getComponent(b)) > Math.abs(u.getComponent(a)) ? b : a));
    this.up = new THREE.Vector3().setComponent(k, Math.sign(u.getComponent(k)));
    const right = this.pos(thighs[1]).sub(this.pos(thighs[0])).projectOnPlane(this.up);
    if (right.length() < 1e-6) throw new Error("motion: can't tell the source's left from its right (its thighs)");
    right.normalize();
    const fwd = new THREE.Vector3().crossVectors(this.up, right);
    this.C = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(fwd, this.up, right)).invert();
    const Ci = (this.Ci = this.C.clone().invert());

    // the rig at anatomical zero: its joints, and a few points on the hand
    const { H, N, bones } = rig;
    const P0 = H.map((h) => h.clone());
    bones.forEach((b, i) => b.parent >= 0 && P0[i].copy(H[i]).sub(H[b.parent]).applyQuaternion(N[b.parent]).add(P0[b.parent]));
    const at0 = (name: string) => P0[rig.bone(name)];
    const onHand = (side: string, point: string) => {
      const i = rig.bone(`hand.${side}`);
      return new THREE.Vector3().fromArray(rig.header.points[`${point}.${side}`]).sub(H[i]).applyQuaternion(N[i]).add(P0[i]);
    };

    // the source's neutral: each bone's turn as loaded, the limbs' each swung (on its own, so a
    // parent's swing doesn't roll it) to point as the rig's does at anatomical zero
    const neutral = (o: THREE.Object3D, q = new THREE.Quaternion()) => this.neutral.set(o, this.turn(o).premultiply(q).invert());
    for (const o of [hips, ...this.spine, ...this.neck, ...this.src.values()]) neutral(o);
    // (a child at its parent's joint says nothing about its direction: FBX's copies of shared bones)
    const child = (o: THREE.Object3D, like = /./) => o.children.find((c) => like.test(c.name) && c.position.lengthSq() > 1e-12) ?? null;
    const aim = (o: THREE.Object3D | undefined, to: THREE.Object3D | null, want: THREE.Vector3, across?: [THREE.Object3D | null, THREE.Object3D | null, THREE.Vector3]) => {
      if (!o || !to) return;
      if (!up(to).includes(o)) throw new Error(`motion: ${to.name} isn't below ${o.name}`);
      const d = this.pos(to).sub(this.pos(o)).normalize(), w = want.clone().applyQuaternion(Ci).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(d, w);
      if (across && across[0] && across[1]) {
        const a = this.pos(across[1]).sub(this.pos(across[0])).applyQuaternion(q).projectOnPlane(w).normalize();
        const b = across[2].clone().applyQuaternion(Ci).projectOnPlane(w).normalize();
        q.premultiply(new THREE.Quaternion().setFromUnitVectors(a, b));
      }
      // and the bones between it and the next joint (a source's own twist bones) likewise
      for (let x: THREE.Object3D | null = to.parent; x && x !== o.parent; x = x.parent) neutral(x, q);
    };
    for (const s of SIDES) {
      const b = (n: string) => this.src.get(`${n}.${s}`);
      const dir = (a: string, c: THREE.Vector3) => c.clone().sub(at0(`${a}.${s}`));
      aim(b("clavicle"), b("upperarm") ?? null, dir("clavicle", at0(`upperarm.${s}`)));
      aim(b("upperarm"), b("forearm") ?? null, dir("upperarm", at0(`forearm.${s}`)));
      aim(b("forearm"), b("hand") ?? null, dir("forearm", at0(`hand.${s}`)));
      const hand = b("hand");
      if (hand) {
        const mid = child(hand, /mid/i), pinky = child(hand, /pinky|little/i);
        const fingers = mid ?? child(hand, /^(?!.*thumb)/i);
        aim(hand, fingers, dir("hand", onHand(s, "finger3-1")), [mid, pinky, onHand(s, "finger5-1").sub(onHand(s, "finger3-1"))]);
      }
      aim(b("thigh"), b("shin") ?? null, dir("thigh", at0(`shin.${s}`)));
      aim(b("shin"), b("foot") ?? null, dir("shin", at0(`foot.${s}`)));
      const foot = b("foot"), toes = b("toes");
      aim(foot, toes ?? (foot ? child(foot) : null), dir("foot", at0(`toes.${s}`)));
      const i = rig.bone(`toes.${s}`);
      aim(toes, toes ? child(toes) : null, new THREE.Vector3().fromArray(bones[i].tail).sub(H[i]).applyQuaternion(N[i]));
    }

    // where the trunk's joints sit along it, at rest: the source's and the rig's
    const along = (ps: THREE.Vector3[]) => {
      const c = [0];
      for (let i = 1; i < ps.length; i++) c.push(c[i - 1] + ps[i].distanceTo(ps[i - 1]));
      return c.map((x) => x / (c.at(-1) || 1));
    };
    const shoulderTop = this.spine.at(-1) ?? hips;
    this.chains = [
      { names: ["spine1", "spine2", "chest", "upperchest"], from: hips, chain: this.spine },
      { names: ["neck", "head"], from: shoulderTop, chain: this.neck },
    ].map((c) => ({
      ...c,
      at: along([c.from, ...c.chain].map((o) => this.pos(o))),
      to: along([bones[rig.bone(c.names[0])].parent, ...c.names.map((n) => rig.bone(n))].map((i) => H[i])),
    }));

    // how tall the source's legs stand (hip joints over ankles), against the rig's
    const ankles = SIDES.map((s) => need(`foot.${s}`));
    const ankle = () => Math.min(...ankles.map((o) => this.pos(o).dot(this.up)));
    const mid = () => this.pos(thighs[0]).add(this.pos(thighs[1])).multiplyScalar(0.5);
    const rigMid = at0("thigh.L").clone().add(at0("thigh.R")).multiplyScalar(0.5);
    const rigAnkle = Math.min(...SIDES.map((s) => at0(`foot.${s}`).y));
    const stand = mid().dot(this.up) - ankle();
    if (!(stand > 0.25 * this.pos(head).distanceTo(this.pos(hips)))) throw new Error("motion: the source as loaded isn't standing (its hips over its ankles), so its height can't be told");
    this.scale = (rigMid.y - rigAnkle) / stand;

    // the clip, and the source's floor: where its lower ankle usually is (a low percentile over
    // the clip, so a foot through the floor for a frame or two doesn't count). An estimate: a
    // clip that never stands (all in the air) stands on its lowest.
    this.mixer = new THREE.AnimationMixer(root);
    this.mixer.clipAction(clip).play();
    if (!(clip.duration >= 0 && clip.duration < 1e6)) throw new Error(`motion: the clip lasts ${clip.duration} s`);
    const lows: number[] = [];
    for (let f = 0, n = Math.min(900, Math.max(1, Math.ceil(30 * clip.duration))); f <= n; f++) {
      this.setTime((f / n) * clip.duration);
      lows.push(ankle());
    }
    lows.sort((a, b) => a - b);
    this.floor = lows[Math.floor(0.05 * (lows.length - 1))];
    this.rigMid = rigMid;
    this.rigAnkle = rigAnkle;
    this.mid = mid;
  }

  /** the rig's spine and neck, and the source's chains that drive them: where each joint sits along its chain (0 to 1) */
  private readonly chains: { names: string[]; from: THREE.Object3D; chain: THREE.Object3D[]; at: number[]; to: number[] }[];

  /** the hip joints' midpoint: the rig's at rest, the source's now */
  private readonly rigMid: THREE.Vector3;
  private readonly rigAnkle: number;
  private readonly mid: () => THREE.Vector3;
  private readonly Ci: THREE.Quaternion;

  private setTime(t: number): void {
    this.mixer.setTime(t);
    this.update();
  }

  /** The source is read in its root's parent's frame: put it anywhere in a scene, by that parent. */
  private readonly base = new THREE.Matrix4();
  private readonly m = new THREE.Matrix4();
  private update(): void {
    this.root.updateMatrixWorld(true);
    if (this.root.parent) this.base.copy(this.root.parent.matrixWorld).invert();
    else this.base.identity();
  }
  private pos(o: THREE.Object3D): THREE.Vector3 {
    return new THREE.Vector3().setFromMatrixPosition(this.m.multiplyMatrices(this.base, o.matrixWorld));
  }
  private turn(o: THREE.Object3D): THREE.Quaternion {
    const q = new THREE.Quaternion();
    this.m.multiplyMatrices(this.base, o.matrixWorld).decompose(new THREE.Vector3(), q, new THREE.Vector3());
    return q;
  }

  /** a source bone's world turn since its neutral, in the rig's frame */
  private moved(o: THREE.Object3D): THREE.Quaternion {
    return this.turn(o).multiply(this.neutral.get(o)!).premultiply(this.C).multiply(this.Ci);
  }

  /** Pose the rig as the clip is `t` seconds in (looping). */
  pose(t: number): void {
    const { rig, W, src } = this;
    const { bones, settings, H, N } = rig;
    this.setTime(this.duration > 0 ? ((t % this.duration) + this.duration) % this.duration : 0);
    rig.roll = 0;
    // each matched bone's turn
    const driven = new Map<number, THREE.Quaternion>();
    for (const [name, o] of src) driven.set(rig.bone(name), this.moved(o));
    // a limb's whole turn is on the bone the next joint hangs from (a source's own twist bones
    // between them carry the rest of its twist): it drives the rig's second half
    for (const side of SIDES) {
      for (const [part, next] of [["upperarm", "forearm"], ["thigh", "shin"], ["forearm", "hand"]]) {
        const o = src.get(`${next}.${side}`)?.parent;
        if (src.has(`${part}.${side}`) && o && this.neutral.has(o)) driven.set(rig.bone(`${part}2.${side}`), this.moved(o));
      }
    }
    // the trunk: each rig joint takes the source's turn where it sits along the chain (by length,
    // so its top bone, the chest the arms hang from or the head, is the source's)
    driven.set(0, this.moved(this.hips));
    for (const { names, from, chain, at, to } of this.chains) {
      const qs = [from, ...chain].map((o) => this.moved(o));
      names.forEach((name, j) => {
        const f = to[j + 1];
        let k = 0;
        while (k + 1 < at.length && at[k + 1] <= f) k++;
        driven.set(rig.bone(name), k + 1 >= at.length ? qs[k] : qs[k].clone().slerp(qs[k + 1], (f - at[k]) / (at[k + 1] - at[k])));
      });
    }
    // parents first: what isn't driven turns with its parent
    for (let i = 0; i < bones.length; i++) W[i].copy(driven.get(i) ?? W[bones[i].parent]);
    // the limbs' twist shared along them
    for (const side of SIDES) {
      const b = (n: string) => rig.bone(`${n}.${side}`);
      for (const [part, k] of [["upperarm", settings.upperArmTwist], ["thigh", settings.thighTwist]] as const) {
        // the whole turn on the second half; the first half untwisted (swung from its rest
        // direction in its parent's frame, as a damped track does) plus its share of the twist
        if (!src.has(`${part}.${side}`)) continue;
        const i = b(part), D = W[b(`${part}2`)], par = W[bones[i].parent];
        const r0 = new THREE.Vector3().fromArray(bones[i].tail).sub(H[i]).normalize(), a = r0.clone().applyQuaternion(N[i]);
        const d = a.clone().applyQuaternion(D).applyQuaternion(par.clone().invert());
        const U = par.clone().multiply(new THREE.Quaternion().setFromUnitVectors(r0, d)).multiply(new THREE.Quaternion().setFromUnitVectors(r0, a).invert());
        W[i].copy(U).multiply(about(a, k * twistAbout(U.clone().invert().multiply(D), a)));
        // pointing nearly opposite its rest direction, the untwisted turn is ill-defined: there
        // the first half turns as the second
        const t = Math.max(0, Math.min(1, (-d.normalize().dot(r0) - 0.75) / 0.2));
        if (t > 0) W[i].slerp(D, t * t * (3 - 2 * t));
      }
      if (src.has(`forearm.${side}`)) {
        // the forearm's and hand's twists about the forearm (the source's forearm, and its own
        // twist bones and hand after it), shared: a quarter, a half, a quarter
        const up = W[b("upperarm2")], D = W[b("forearm")].clone(), D2 = W[b("forearm2")].clone(), Dh = W[b("hand")];
        const L = up.clone().invert().multiply(D), tf = twistAbout(L, DOWN), th = twistAbout(D.clone().invert().multiply(Dh), DOWN);
        const tau = tf + th;
        W[b("forearm")].copy(up).multiply(L.multiply(about(DOWN, tau / 4 - tf)));
        W[b("forearm2")].copy(W[b("forearm")]).multiply(about(DOWN, tau / 2));
        // near half a turn either twist flips sign: there the forearm turns as the source's
        const t = Math.max(0, Math.min(1, (Math.max(Math.abs(tf), Math.abs(th)) - (150 * Math.PI) / 180) / ((25 * Math.PI) / 180)));
        if (t > 0) {
          W[b("forearm")].slerp(D, t * t * (3 - 2 * t));
          W[b("forearm2")].slerp(D2, t * t * (3 - 2 * t));
        }
      }
    }
    // the hips: their joints' midpoint where the source's is (from the source's origin), scaled
    // to the rig's legs, its height over the source's floor as the rig's over its ankles
    const m = this.mid(), s = this.scale, c = m.clone().applyQuaternion(this.C);
    const goal = new THREE.Vector3(s * c.x, this.rigAnkle + s * (m.dot(this.up) - this.floor), s * c.z);
    rig.poseWorld(W, goal.sub(this.rigMid.clone().sub(H[0]).applyQuaternion(W[0])));
  }
}
