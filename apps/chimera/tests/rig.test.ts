/**
 * The body rig (src/rig/): posing, skinning, and the CHIMERA head worn on it. On the real engine
 * (the head is fitted to a VoiceModel), without a GPU.
 */
import { beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { initEngine } from "./harness.ts";
import type { Rig as RigT } from "../src/rig/rig.ts";
import type { Pose } from "../src/rig/pose.ts";

type Mods = {
  Rig: typeof import("../src/rig/rig.ts").Rig;
  motion: typeof import("../src/rig/motion.ts");
  pose: typeof import("../src/rig/pose.ts");
  head: typeof import("../src/rig/head.ts");
  VoiceModel: typeof import("../src/ui/analysis.ts").VoiceModel;
  BODIES: typeof import("../src/ui/body.ts").PRESETS;
  REST_ARTIC: typeof import("../src/ui/render/anatomy3d.ts").REST_ARTIC;
};
let M: Mods;
beforeAll(async () => {
  await initEngine();
  // the head reads the engine's tables as it loads: after the engine
  const [rig, pose, head, analysis, body, a3d, motion] = await Promise.all([import("../src/rig/rig.ts"), import("../src/rig/pose.ts"), import("../src/rig/head.ts"), import("../src/ui/analysis.ts"), import("../src/ui/body.ts"), import("../src/ui/render/anatomy3d.ts"), import("../src/rig/motion.ts")]);
  M = { Rig: rig.Rig, motion, pose, head, VoiceModel: analysis.VoiceModel, BODIES: body.PRESETS, REST_ARTIC: a3d.REST_ARTIC };
});

function loadRig(): RigT {
  const b = readFileSync(new URL("../src/rig/body.rig.bin", import.meta.url));
  return new M.Rig(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
}

/** the mesh's edges that only one quad has, as vertex pairs */
function openEdges(rig: RigT): [number, number][] {
  const count = new Map<number, number>();
  for (let f = 0; f < rig.nq; f++) {
    for (let j = 0; j < 4; j++) {
      const a = rig.quads[4 * f + j], b = rig.quads[4 * f + ((j + 1) % 4)];
      const k = Math.min(a, b) * 1048576 + Math.max(a, b);
      count.set(k, (count.get(k) ?? 0) + 1);
    }
  }
  return [...count].filter(([, c]) => c === 1).map(([k]) => [Math.floor(k / 1048576), k % 1048576]);
}

const at = (P: ArrayLike<number>, i: number) => new THREE.Vector3(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
const maxMove = (A: ArrayLike<number>, B: ArrayLike<number>, ids: Iterable<number>) => {
  let m = 0;
  for (const i of ids) m = Math.max(m, at(A, i).distanceTo(at(B, i)));
  return m;
};

describe("the body", () => {
  test("its only open edges are the neck's rim", () => {
    const rig = loadRig(), open = openEdges(rig);
    expect(open.length).toBe(rig.rim.length);
    const rim = new Set(rig.rim);
    expect(open.every(([a, b]) => rim.has(a) && rim.has(b))).toBe(true);
  });

  test("unposed, both skinnings give the rest mesh back", () => {
    const rig = loadRig(), out = new Float32Array(3 * rig.n);
    for (const method of ["cor", "lbs"] as const) {
      rig.skin(out, method);
      for (let i = 0; i < out.length; i++) expect(Math.abs(out[i] - rig.rest[i])).toBeLessThan(1e-3);
    }
  });

  test("at anatomical zero the arms hang straight down and the legs stand under the hips", () => {
    const rig = loadRig();
    rig.pose(M.pose.neutral());
    const down = (from: string, to: string) => rig.P[rig.bone(to)].clone().sub(rig.P[rig.bone(from)]).normalize().y;
    for (const s of ["R", "L"]) {
      expect(down(`upperarm.${s}`, `forearm.${s}`)).toBeLessThan(-0.999);
      expect(down(`forearm.${s}`, `hand.${s}`)).toBeLessThan(-0.999);
      // seen from the front the hip, knee and ankle are in one vertical line
      const hip = rig.P[rig.bone(`thigh.${s}`)], ankle = rig.P[rig.bone(`foot.${s}`)];
      expect(Math.abs(ankle.z - hip.z)).toBeLessThan(0.05);
    }
  });
});

describe("the head on the body", () => {
  /** the voices' bodies: the head is fitted to each (a child's has fewer rows under its mouth) */
  const BODIES = ["male", "female", "child"] as const;
  const wear = (body: (typeof BODIES)[number] = "male") => {
    const rig = loadRig(), model = new M.VoiceModel(M.BODIES[body], 48000), head = M.head.wearHead(rig, model);
    return { rig, model, head };
  };
  /** the lips' cut: pairs of head points that lie together at rest (each upper lip point and its lower copy) */
  const lipPairs = (rig: RigT) => {
    const pairs: [number, number][] = [];
    for (let i = rig.bodyVertices; i < rig.n; i++) for (let j = i + 1; j < rig.n; j++) if (at(rig.rest, i).distanceTo(at(rig.rest, j)) < 1e-5) pairs.push([i, j]);
    return pairs;
  };
  /** the body posed and skinned, with the head drawn and its face moved for an articulation */
  const frame = (w: ReturnType<typeof wear>, p: import("../src/rig/pose.ts").Pose, a: import("../src/wasm/types.ts").HumanArtic) => {
    const { rig, head } = w, base = new Float32Array(3 * rig.n), out = new Float32Array(3 * rig.n);
    rig.pose(p);
    rig.skin(base);
    head.update(new THREE.PerspectiveCamera(), a, null, 1 / 60);
    head.face(base, out);
    return { base, out };
  };

  test.each(BODIES)("the skin closes over the head: the only open edges are the lips' cut (%s)", (body) => {
    const { rig } = wear(body), open = openEdges(rig), pairs = lipPairs(rig);
    expect(pairs.length).toBeGreaterThan(2);
    const cut = new Set(pairs.flat());
    // the cut runs between the mouth's corners, which both lips share: each edge has at least one cut point, none a body point
    expect(open.length).toBe(2 * (pairs.length + 1));
    expect(open.every(([a, b]) => a >= rig.bodyVertices && b >= rig.bodyVertices && (cut.has(a) || cut.has(b)))).toBe(true);
  });

  /** a point's distance from a polyline (flat xyz) carried by a matrix */
  const fromLine = (p: THREE.Vector3, line: Float32Array, m: THREE.Matrix4) => {
    const a = new THREE.Vector3(), b = new THREE.Vector3(), seg = new THREE.Line3(), q = new THREE.Vector3();
    let d = Infinity;
    for (let i = 0; i + 5 < line.length; i += 3) {
      seg.set(a.fromArray(line, i).applyMatrix4(m), b.fromArray(line, i + 3).applyMatrix4(m));
      d = Math.min(d, seg.closestPointToPoint(p, true, q).distanceTo(p));
    }
    return d;
  };

  /** a point's distance from a polyline of points */
  const fromPoints = (p: THREE.Vector3, pts: THREE.Vector3[]) => {
    const seg = new THREE.Line3(), q = new THREE.Vector3();
    let d = Infinity;
    for (let i = 0; i + 1 < pts.length; i++) d = Math.min(d, seg.set(pts[i], pts[i + 1]).closestPointToPoint(p, true, q).distanceTo(p));
    return d;
  };

  test.each(BODIES)("the lips' cut lies on the lips the head draws: together at rest; apart, narrow or wide, with the head turned (%s)", (body) => {
    const w = wear(body), { rig } = w, pairs = lipPairs(rig);
    const shut = frame(w, M.pose.neutral(), M.REST_ARTIC).out;
    for (const [u, l] of pairs) expect(at(shut, u).distanceTo(at(shut, l))).toBeLessThan(1e-3);
    // the cut's corners: its ends, which both lips share
    const cut = new Set(pairs.flat()), corners = [...new Set(openEdges(rig).flat())].filter((i) => !cut.has(i));
    expect(corners.length).toBe(2);
    // each lip's side of the cut, corner to corner (at rest the cut runs across z)
    const byZ = (ids: number[]) => ids.sort((i, j) => rig.rest[3 * i + 2] - rig.rest[3 * j + 2]);
    const sides = [byZ([...corners, ...pairs.map(([u]) => u)]), byZ([...corners, ...pairs.map(([, l]) => l)])];
    const turned = M.pose.neutral();
    turned.neck.twist = 40;
    turned.neck.flex = 20;
    for (const p of [M.pose.neutral(), turned]) {
      for (const lipWidth of [0, 1]) {
        const a = { ...M.REST_ARTIC, jaw: 1.8, lipAp: 1.8, lipWidth };
        const { out } = frame(w, p, a);
        const e = w.head.head.lipEdges(w.model, a), m = w.head.mount.matrix;
        // apart (the drawn lips' gap at the middle, in body cm)
        const mid = 3 * Math.floor(e.upper.length / 6);
        expect(new THREE.Vector3().fromArray(e.upper, mid).distanceTo(new THREE.Vector3().fromArray(e.lower, mid)) / w.head.s).toBeGreaterThan(0.5);
        for (const [u, l] of pairs) {
          expect(fromLine(at(out, u), e.upper, m)).toBeLessThan(0.01);
          expect(fromLine(at(out, l), e.lower, m)).toBeLessThan(0.01);
        }
        // the cut's ends are the mouth's corners, however wide it is
        const ends = [new THREE.Vector3().fromArray(e.upper, 0), new THREE.Vector3().fromArray(e.upper, e.upper.length - 3)].map((c) => c.applyMatrix4(m));
        for (const c of corners) expect(Math.min(...ends.map((x) => x.distanceTo(at(out, c))))).toBeLessThan(0.01);
        // and the drawn lips are near the cut's edges all along (straight between the grid's
        // columns, so within a fraction of a column's width of the curve)
        [e.upper, e.lower].forEach((line, k) => {
          const edge = sides[k].map((i) => at(out, i));
          for (let i = 0; i < line.length; i += 3) expect(fromPoints(new THREE.Vector3().fromArray(line, i).applyMatrix4(m), edge)).toBeLessThan(0.4);
        });
      }
    }
  });

  test.each(BODIES)("the head's quads don't fold, the mouth wide open or pursed with the head bent and turned (%s)", (body) => {
    const w = wear(body), { rig } = w, p = M.pose.neutral();
    p.neck.flex = 40;
    p.neck.twist = 60;
    for (const a of [{ ...M.REST_ARTIC, jaw: 2.4, lipAp: 2.2, lipWidth: 1 }, { ...M.REST_ARTIC, jaw: 2.4, lipAp: 2.2, lipWidth: 0.05, lipProt: 1.4 }]) {
      const { out } = frame(w, p, a);
      for (let f = 0; f < rig.nq; f++) {
        const q = [0, 1, 2, 3].map((j) => rig.quads[4 * f + j]);
        if (q.every((i) => i < rig.bodyVertices)) continue;
        // the two triangles the view draws (a b c, a c d) face the same way
        const [A, B, C, D] = q.map((i) => at(out, i));
        const n1 = new THREE.Vector3().crossVectors(B.clone().sub(A), C.clone().sub(A)), n2 = new THREE.Vector3().crossVectors(C.clone().sub(A), D.clone().sub(A));
        if (n1.length() < 1e-6 || n2.length() < 1e-6) continue;
        expect(n1.normalize().dot(n2.normalize())).toBeGreaterThan(0);
      }
    }
  });

  test("every point's weights sum to 1 (a fitted neck once dropped a share)", () => {
    const { rig } = wear();
    for (let i = 0; i < rig.n; i++) {
      let sum = 0;
      for (let j = 0; j < rig.INF; j++) sum += rig.skinWeight[rig.INF * i + j];
      expect(Math.abs(sum - 1)).toBeLessThan(1e-5);
    }
  });

  test("the head reads its parent's transform as it is now (the features dim with the face turned away)", () => {
    const w = wear(), parent = new THREE.Group(), camera = new THREE.PerspectiveCamera();
    parent.add(w.head.mount);
    camera.position.set(400, 60, 0);
    w.rig.pose(M.pose.neutral());
    w.head.update(camera, M.REST_ARTIC, null, 0);
    const facing = w.head.head.drawing.features;
    // the body turned round under its parent since the last frame: the face is away from the camera now
    parent.rotation.y = Math.PI;
    w.head.update(camera, M.REST_ARTIC, null, 0);
    expect(facing).toBeCloseTo(w.head.look.features, 3);
    expect(w.head.head.drawing.features).toBeLessThan(0.2 * w.head.look.features);
  });

  test("the face's motion moves the face only: not the crown, not the rim (a mouth opening once dented the crown)", () => {
    const w = wear(), { rig } = w;
    const p = M.pose.neutral();
    p.neck.twist = 40;
    p.neck.flex = 20;
    const { base, out } = frame(w, p, { ...M.REST_ARTIC, jaw: 1.8, lipAp: 1.8 });
    // the crown: the head's points above its eyes
    let top = -Infinity;
    for (let i = rig.bodyVertices; i < rig.n; i++) top = Math.max(top, rig.rest[3 * i + 1]);
    const crown = [...Array(rig.n).keys()].filter((i) => i >= rig.bodyVertices && rig.rest[3 * i + 1] > top - 6);
    expect(crown.length).toBeGreaterThan(20);
    expect(maxMove(base, out, crown)).toBeLessThan(1e-4);
    expect(maxMove(base, out, rig.rim)).toBe(0);
    expect(maxMove(base, out, lipPairs(rig).map(([, l]) => l))).toBeGreaterThan(0.5);
  });
});

/**
 * The rig's own skeleton as a BVH file, as another tool would write it: bound at `bind`, then
 * posed through `frames` (half a second apart), turned `yaw`° about the vertical and measured in
 * `unit`s per cm, standing on the file's floor (y = 0) with its root's OFFSET at 0 as BVHs have;
 * without the `leave` joints (their parents end there).
 */
function toBVH(rig: RigT, bind: Pose, frames: Pose[], yaw: number, unit: number, leave: string[] = []): string {
  const G = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), (yaw * Math.PI) / 180), Gi = G.clone().invert();
  const joints = rig.bones.map((_, i) => i).filter((i) => !rig.bones[i].helper && !leave.includes(rig.bones[i].name));
  const kids = (i: number) => joints.filter((j) => rig.bones[j].parent === i);
  rig.pose(bind);
  const Pb = rig.P.map((p) => p.clone()), Wb = rig.W.map((q) => q.clone().invert()), Sb = rig.S.map((q) => q.clone());
  const floor = Math.min(...rig.P.map((p) => p.y));
  const file = (v: THREE.Vector3) => v.clone().applyQuaternion(G).multiplyScalar(unit);
  const lines = ["HIERARCHY"], order: number[] = [];
  const write = (i: number, pad: string) => {
    order.push(i);
    const par = rig.bones[i].parent, o = par < 0 ? new THREE.Vector3() : file(Pb[i].clone().sub(Pb[par]));
    lines.push(`${pad}${par < 0 ? "ROOT" : "JOINT"} ${rig.bones[i].name.replace(".", "_")}`, `${pad}{`, `${pad}  OFFSET ${o.x} ${o.y} ${o.z}`);
    lines.push(`${pad}  CHANNELS ${par < 0 ? "6 Xposition Yposition Zposition " : "3 "}Zrotation Xrotation Yrotation`);
    for (const j of kids(i)) write(j, pad + "  ");
    if (!kids(i).length) {
      const t = file(new THREE.Vector3().fromArray(rig.bones[i].tail).sub(rig.H[i]).applyQuaternion(Sb[i]));
      lines.push(`${pad}  End Site`, `${pad}  {`, `${pad}    OFFSET ${t.x} ${t.y} ${t.z}`, `${pad}  }`);
    }
    lines.push(`${pad}}`);
  };
  write(0, "");
  lines.push("MOTION", `Frames: ${frames.length}`, "Frame Time: 0.5");
  for (const f of frames) {
    rig.pose(f);
    const R = rig.W.map((w, i) => G.clone().multiply(w).multiply(Wb[i]).multiply(Gi));
    const vals: number[] = [];
    for (const i of order) {
      const par = rig.bones[i].parent;
      if (par < 0) vals.push(...file(rig.P[0].clone().sub(new THREE.Vector3(0, floor, 0))).toArray());
      const e = new THREE.Euler().setFromQuaternion(par < 0 ? R[i] : R[par].clone().invert().multiply(R[i]), "ZXY");
      vals.push(...[e.z, e.x, e.y].map((a) => (a * 180) / Math.PI));
    }
    lines.push(vals.join(" "));
  }
  return lines.join("\n");
}
const buffer = (text: string) => new TextEncoder().encode(text).buffer as ArrayBuffer;
/** how far (degrees) the hands of `b` are turned from `a`'s */
const handsApart = (a: RigT, b: RigT) => Math.max(...["hand.L", "hand.R"].map((h) => (a.S[a.bone(h)].angleTo(b.S[b.bone(h)]) * 180) / Math.PI));
/** the farthest any joint of `b` is from `a`'s, measured from each one's root (or not) */
function jointsApart(a: RigT, b: RigT, fromRoot: boolean): { cm: number; at: string } {
  let cm = 0, at = "";
  a.bones.forEach((bone, i) => {
    if (bone.helper) return;
    const d = fromRoot ? a.P[i].clone().sub(a.P[0]).distanceTo(b.P[i].clone().sub(b.P[0])) : a.P[i].distanceTo(b.P[i]);
    if (!(d <= cm)) [cm, at] = [Number.isNaN(d) ? Infinity : d, bone.name];
  });
  return { cm, at };
}

/**
 * A small skeleton as other tools make them: Mixamo's names, a T pose (palms down), y up, facing
 * +z, in metres. `change` can add to it or move it before it's used.
 */
function mixamo(change?: (b: Record<string, THREE.Bone>) => void): Record<string, THREE.Bone> {
  const b: Record<string, THREE.Bone> = {};
  const bone = (name: string, parent: string | null, x: number, y: number, z: number) => {
    const o = (b[name] = new THREE.Bone());
    o.name = name;
    o.position.set(x, y, z);
    if (parent) b[parent].add(o);
  };
  bone("Hips", null, 0, 1, 0);
  bone("Spine", "Hips", 0, 0.1, 0);
  bone("Chest", "Spine", 0, 0.2, 0);
  bone("Neck", "Chest", 0, 0.2, 0);
  bone("Head", "Neck", 0, 0.1, 0);
  bone("HeadTop_End", "Head", 0, 0.18, 0);
  for (const [s, x] of [["Left", 1], ["Right", -1]] as const) {
    bone(`${s}Shoulder`, "Chest", 0.05 * x, 0.15, 0);
    bone(`${s}Arm`, `${s}Shoulder`, 0.12 * x, 0, 0);
    bone(`${s}ForeArm`, `${s}Arm`, 0.27 * x, 0, 0);
    bone(`${s}Hand`, `${s}ForeArm`, 0.27 * x, 0, 0);
    bone(`${s}HandMiddle1`, `${s}Hand`, 0.1 * x, 0, 0);
    bone(`${s}HandPinky1`, `${s}Hand`, 0.09 * x, 0, -0.03);
    bone(`${s}UpLeg`, "Hips", 0.09 * x, -0.05, 0);
    bone(`${s}Leg`, `${s}UpLeg`, 0, -0.45, 0);
    bone(`${s}Foot`, `${s}Leg`, 0, -0.42, 0);
    bone(`${s}ToeBase`, `${s}Foot`, 0, -0.06, 0.12);
  }
  change?.(b);
  return b;
}
/** a clip turning bones (by name) about an axis, through angles (degrees), a key every 0.05 s */
function turning(turns: Record<string, { axis: [number, number, number]; degrees: number[] }>): THREE.AnimationClip {
  return new THREE.AnimationClip("turning", -1, Object.entries(turns).map(([name, { axis, degrees }]) => new THREE.QuaternionKeyframeTrack(
    `${name}.quaternion`, degrees.map((_, i) => 0.05 * i),
    degrees.flatMap((d) => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(...axis), (d * Math.PI) / 180).toArray()),
  )));
}
/** how far (degrees) a rig bone points from the source's bone (its joint to the next one's), in the rig's frame */
function aimedApart(motion: InstanceType<Mods["motion"]["Motion"]>, bone: string, next: string, from: THREE.Object3D, to: THREE.Object3D): number {
  const r = motion.rig, p = (o: THREE.Object3D) => o.getWorldPosition(new THREE.Vector3());
  const want = p(to).sub(p(from)).applyQuaternion(motion.C).normalize();
  return (r.P[r.bone(next)].clone().sub(r.P[r.bone(bone)]).normalize().angleTo(want) * 180) / Math.PI;
}

describe("motion from other tools", () => {
  test("bones are matched by name across tools' conventions; twist bones, fingers and end sites aren't", () => {
    const cases: Record<string, string | null> = {
      // Mixamo (FBX and glTF, as three.js's loaders name them), CMU and DAZ BVHs
      "mixamorig:LeftArm": "upperarm.L", mixamorigRightForeArm: "forearm.R", mixamorig1LeftUpLeg: "thigh.L", mixamorigLeftShoulder: "clavicle.L",
      mixamorigRightToeBase: "toes.R", mixamorigHead: "head", mixamorigLeftHandIndex1: null, mixamorigLeftToe_End: null, mixamorigHeadTop_End: null,
      LeftLeg: "shin.L", RightFoot: "foot.R", LHipJoint: null, LeftFingerBase: null, lShldr: "upperarm.L", rCollar: "clavicle.R", rShin: "shin.R", lButtock: null, head: "head",
      // VRM and VRoid, Unreal, Rigify, 3ds Max Biped, and the rig's own
      leftUpperArm: "upperarm.L", rightLowerLeg: "shin.R", J_Bip_L_UpperArm: "upperarm.L", J_Bip_C_Head: "head", J_Bip_R_ToeBase: "toes.R",
      upperarm_l: "upperarm.L", calf_r: "shin.R", ball_l: "toes.L", upperarm_twist_01_l: null, "DEF-upper_arm.L": "upperarm.L", "DEF-upper_armL": "upperarm.L",
      "DEF-upper_arm.L.001": null, "Bip01 L Thigh": "thigh.L", "Bip01 R Calf": "shin.R", "Bip01 L Toe0": "toes.L",
      "forearm.R": "forearm.R", "upperarm2.R": null, "shoulderHelper.L": null, "pelvis.R": null,
    };
    for (const [name, want] of Object.entries(cases)) expect([name, M.motion.rigName(name)]).toEqual([name, want]);
  });

  test("the rig's own poses come back through a BVH, whatever its bind, facing and units", async () => {
    const { PRESETS } = M.pose, names = Object.keys(PRESETS);
    // the clip comes back round to neutral, so its last pose isn't the loop's wrap
    const frames = [...names.map((n) => PRESETS[n]), PRESETS.neutral];
    for (const [bind, yaw, unit] of [["neutral", 90, 1 / 2.54], ["T pose", -30, 0.01]] as const) {
      const src = loadRig(), dst = loadRig();
      // the source's thigh twists about the thigh: the rig's own posing twists it about the
      // vertical, a few degrees off, which bends its second half off the line
      src.settings.thighTwist = 1;
      const motion = await M.motion.Motion.load(dst, buffer(toBVH(src, PRESETS[bind], frames, yaw, unit)), "rig.bvh");
      names.forEach((name, k) => {
        src.pose(PRESETS[name]);
        motion.pose(0.5 * k);
        expect([bind, name, jointsApart(src, dst, true).cm < 0.1, handsApart(src, dst) < 0.5]).toEqual([bind, name, true, true]);
      });
    }
  });

  test("a skeleton without toes, or with a clip of one frame, still plays; seeking goes anywhere in any order", async () => {
    const { PRESETS } = M.pose, names = ["run", "lotus", "twist & look", "wrists"];
    const src = loadRig(), dst = loadRig();
    src.settings.thighTwist = 1;
    const motion = await M.motion.Motion.load(dst, buffer(toBVH(src, PRESETS.neutral, [...names.map((n) => PRESETS[n]), PRESETS.neutral], 0, 1, ["toes.L", "toes.R"])), "rig.bvh");
    expect(motion.bones["toes.L"]).toBeUndefined();
    // [time in frames, the frame it shows]: past the end (2 s) it loops
    for (const [k, shows] of [[3, 3], [0, 0], [2, 2], [1, 1], [6, 2]]) {
      src.pose(PRESETS[names[shows]]);
      motion.pose(0.5 * k);
      expect(jointsApart(src, dst, true).cm).toBeLessThan(0.1);
    }
    const ref = loadRig(), still = await M.motion.Motion.load(loadRig(), buffer(toBVH(loadRig(), PRESETS.neutral, [PRESETS.run], 0, 1)), "rig.bvh");
    expect(still.duration).toBe(0);
    still.pose(3.7);
    ref.pose(PRESETS.run);
    expect(jointsApart(ref, still.rig, true).cm).toBeLessThan(0.1);
  });

  test("standing on its floor, the source's hips put the rig's where they'd stand", async () => {
    const { PRESETS } = M.pose, frames = ["neutral", "T pose", "arms forward", "flex biceps", "neutral"].map((n) => PRESETS[n]);
    const src = loadRig(), dst = loadRig();
    src.settings.thighTwist = 1;
    const motion = await M.motion.Motion.load(dst, buffer(toBVH(src, PRESETS["T pose"], frames, 45, 1)), "rig.bvh");
    frames.forEach((f, k) => {
      src.pose(f);
      motion.pose(0.5 * k);
      expect(jointsApart(src, dst, false).cm).toBeLessThan(0.1);
    });
  });

  test("an explicit map wins; of two bones that could be one of the rig's, the one the clip moves", () => {
    const { Motion } = M.motion;
    const withControl = () => mixamo((b) => {
      // a control bone that also reads as the left upper arm
      b.upperarm_L = new THREE.Bone();
      b.upperarm_L.name = "upperarm_L";
      b.Hips.add(b.upperarm_L);
    });
    const raise = { axis: [0, 0, 1] as [number, number, number], degrees: [0, 45] };
    let b = withControl();
    expect(new Motion(loadRig(), b.Hips, turning({ LeftArm: raise })).bones["upperarm.L"]).toBe("LeftArm");
    b = withControl();
    expect(() => new Motion(loadRig(), b.Hips, turning({ LeftArm: raise, upperarm_L: raise }))).toThrow(/could be/);
    b = withControl();
    expect(new Motion(loadRig(), b.Hips, turning({ LeftArm: raise, upperarm_L: raise }), { map: { LeftArm: "upperarm.L" } }).bones["upperarm.L"]).toBe("LeftArm");
  });

  test("a pivot between the chest and a shoulder is fine; an unevenly scaled or unstanding source is refused", () => {
    const { Motion } = M.motion;
    const b = mixamo((b) => {
      const pivot = new THREE.Bone();
      pivot.name = "ShoulderPivot";
      b.Chest.add(pivot);
      pivot.add(b.LeftShoulder);
    });
    const motion = new Motion(loadRig(), b.Hips, turning({ LeftArm: { axis: [0, 1, 0], degrees: [0, 90] }, LeftForeArm: { axis: [0, 0, 1], degrees: [0, 60] } }));
    motion.pose(0.05);
    expect(aimedApart(motion, "upperarm.L", "forearm.L", b.LeftArm, b.LeftForeArm)).toBeLessThan(0.1);
    expect(aimedApart(motion, "forearm.L", "hand.L", b.LeftForeArm, b.LeftHand)).toBeLessThan(0.1);
    const squashed = mixamo((b) => b.Chest.scale.set(1, 2, 1));
    expect(() => new Motion(loadRig(), squashed.Hips, turning({ LeftArm: { axis: [0, 1, 0], degrees: [0, 90] } }))).toThrow(/unevenly/);
    // sitting: the knees forward, the feet as high as the hips
    const seated = mixamo((b) => {
      for (const s of ["Left", "Right"]) b[`${s}Leg`].position.set(0, 0, 0.45);
      for (const s of ["Left", "Right"]) b[`${s}Foot`].position.set(0, 0, 0.42);
    });
    expect(() => new Motion(loadRig(), seated.Hips, turning({ LeftArm: { axis: [0, 1, 0], degrees: [0, 90] } }))).toThrow(/standing/);
  });

  test("the forearm doesn't snap as the wrist turns through half a turn", () => {
    const b = mixamo(), rig = loadRig();
    const motion = new M.motion.Motion(rig, b.Hips, turning({ LeftHand: { axis: [1, 0, 0], degrees: Array.from({ length: 41 }, (_, i) => 160 + i) } }));
    const bones = ["forearm.L", "forearm2.L", "hand.L"].map((n) => rig.bone(n));
    let last: THREE.Quaternion[] | null = null, most = 0;
    for (let t = 0; t <= 2; t += 0.01) {
      motion.pose(t);
      const now = bones.map((i) => rig.W[i].clone());
      if (last) most = Math.max(most, ...now.map((q, k) => (q.angleTo(last![k]) * 180) / Math.PI));
      last = now;
    }
    // the wrist turns 0.2° a step
    expect(most).toBeLessThan(2);
  });

  test("the source can be put anywhere in a scene, by its parent", async () => {
    const { PRESETS } = M.pose, frames = ["neutral", "run", "neutral"].map((n) => PRESETS[n]);
    const src = loadRig(), dst = loadRig(), still = loadRig();
    const text = toBVH(src, PRESETS.neutral, frames, 0, 1);
    const moved = await M.motion.Motion.load(dst, buffer(text), "rig.bvh");
    const motion = await M.motion.Motion.load(still, buffer(text), "rig.bvh");
    const parent = new THREE.Group();
    parent.position.set(30, -20, 5);
    parent.quaternion.setFromEuler(new THREE.Euler(0.3, 1.2, -0.2));
    parent.scale.setScalar(2.5);
    parent.add(moved.root);
    parent.updateMatrixWorld(true);
    moved.pose(0.5);
    motion.pose(0.5);
    expect(jointsApart(dst, still, false).cm).toBeLessThan(0.01);
  });
});
