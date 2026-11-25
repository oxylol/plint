/**
 * The rig drawn as a wireframe: the mesh's own quads as glowing lines (each triangle carries its
 * corners' places in their quad, so the diagonal isn't drawn), a glow where the surface turns
 * away, over a dark fill that hides what is behind; the far side seen faintly through the near
 * one. In CHIMERA's blue by default. For finding faults it can colour the lines by a score per
 * quad (`heat`: 1 where a fault starts to show) or by which bones move them (`weights`), and draw
 * the skeleton.
 *
 * The colours are worked in linear light and encoded for the screen here, to sit with the head's
 * lines (display values, as the app's): render with `renderer.outputColorSpace =
 * THREE.LinearSRGBColorSpace`.
 */
import * as THREE from "three";
import { vertexNormals, type Rig } from "./rig.ts";

const VERT = /* glsl */ `
attribute vec2 q;
attribute float heat;
attribute vec3 tint;
attribute float tone;
varying float vTone;
varying vec2 vQ;
varying vec3 vN;
varying vec3 vV;
varying float vHeat;
varying vec3 vTint;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalMatrix * normal;
  vV = -mv.xyz;
  vQ = q;
  vHeat = heat;
  vTint = tint;
  vTone = tone;
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = /* glsl */ `
uniform vec3 uFill;
uniform vec3 uLine;
uniform vec3 uEdge;
uniform float uMode;
uniform float uBack;
uniform float uAlpha;
varying vec2 vQ;
varying vec3 vN;
varying vec3 vV;
varying float vHeat;
varying vec3 vTint;
varying float vTone;
vec3 heatColour(float h) {
  // calm to sore: the look's own line colour, then amber, red, white-hot
  vec3 c = mix(uLine * 0.55, vec3(1.0, 0.62, 0.1), smoothstep(0.35, 0.8, h));
  c = mix(c, vec3(1.0, 0.12, 0.08), smoothstep(0.8, 1.3, h));
  return mix(c, vec3(1.0, 0.9, 0.9), smoothstep(1.6, 2.6, h));
}
void main() {
  vec3 n = normalize(vN), v = normalize(vV);
  float facing = abs(dot(n, v));
  float rim = pow(1.0 - facing, 2.5);
  vec2 fw = max(fwidth(vQ), vec2(1e-6));
  vec2 dq = min(vQ, 1.0 - vQ) / fw;
  float wire = (1.0 - smoothstep(0.35, 1.15, min(dq.x, dq.y))) * (1.0 - smoothstep(0.16, 0.42, max(fw.x, fw.y)));
  vec3 line = uLine, edge = uEdge, fill = uFill;
  if (uMode > 0.5 && uMode < 1.5) {
    line = heatColour(vHeat);
    fill = mix(uFill, vec3(0.35, 0.03, 0.02), smoothstep(0.8, 1.6, vHeat) * 0.6);
  } else if (uMode > 1.5) {
    line = vTint;
    fill = uFill + 0.12 * vTint;
  }
  vec3 col = fill + edge * 1.3 * rim + line * wire * (0.55 + 0.45 * facing) * 1.4 * (uMode > 0.5 ? 1.0 : vTone);
  col = max(col, 0.0);
  col = mix(col * 12.92, 1.055 * pow(col, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, col));
  gl_FragColor = vec4(col, uAlpha * (1.0 - 0.75 * uBack));
}`;

export type Mode = "wire" | "heat" | "weights";
export interface Look {
  fill: [number, number, number];
  line: [number, number, number];
  edge: [number, number, number];
}
/** CHIMERA's blue: the app's #4fb6f2 for the lines, its mid blue at the silhouette (linear light) */
export const CHIMERA_LOOK: Readonly<Look> = { fill: [0.003, 0.008, 0.016], line: [0.06, 0.36, 0.68], edge: [0.03, 0.2, 0.45] };

export class RigView {
  readonly group = new THREE.Group();
  readonly posed: Float32Array<ArrayBufferLike>;
  readonly normals: Float32Array<ArrayBufferLike>;
  private readonly geo = new THREE.BufferGeometry();
  private readonly front: THREE.Mesh;
  private readonly back: THREE.Mesh;
  private readonly mat: THREE.ShaderMaterial;
  private readonly backMat: THREE.ShaderMaterial;
  private readonly bones: THREE.LineSegments;
  private readonly joints: THREE.Points;
  /** each corner's vertex */
  private readonly corner: Uint32Array;

  constructor(readonly rig: Rig) {
    const nq = rig.nq, N = 6 * nq;
    this.posed = rig.rest.slice();
    this.normals = new Float32Array(3 * rig.n);
    this.corner = new Uint32Array(N);
    const QC = new Float32Array(2 * N);
    const AT = [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]];
    for (let f = 0; f < nq; f++) {
      [0, 1, 2, 0, 2, 3].forEach((c, j) => {
        this.corner[6 * f + j] = rig.quads[4 * f + c];
        QC.set(AT[j], 2 * (6 * f + j));
      });
    }
    this.geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * N), 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(3 * N), 3).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("q", new THREE.BufferAttribute(QC, 2));
    this.geo.setAttribute("heat", new THREE.BufferAttribute(new Float32Array(N), 1).setUsage(THREE.DynamicDrawUsage));
    this.geo.setAttribute("tint", new THREE.BufferAttribute(new Float32Array(3 * N), 3));
    this.geo.setAttribute("tone", new THREE.BufferAttribute(Float32Array.from(this.corner, (v) => rig.tone[v]), 1));
    const uniforms = () => ({
      uFill: { value: new THREE.Vector3() },
      uLine: { value: new THREE.Vector3() },
      uEdge: { value: new THREE.Vector3() },
      uMode: { value: 0 },
      uBack: { value: 0 },
      uAlpha: { value: 0.92 },
    });
    this.mat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: uniforms(), transparent: true, side: THREE.FrontSide });
    this.backMat = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: uniforms(), transparent: true, side: THREE.BackSide, depthWrite: false });
    this.backMat.uniforms.uBack.value = 1;
    this.back = new THREE.Mesh(this.geo, this.backMat);
    this.front = new THREE.Mesh(this.geo, this.mat);
    this.back.renderOrder = -3;
    this.front.renderOrder = -2;
    this.back.frustumCulled = this.front.frustumCulled = false;
    this.group.add(this.back, this.front);
    const bg = new THREE.BufferGeometry();
    bg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6 * rig.B), 3));
    this.bones = new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ color: 0x7fffd4, depthTest: false, transparent: true, opacity: 0.9 }));
    this.bones.renderOrder = 10;
    this.bones.frustumCulled = false;
    const jg = new THREE.BufferGeometry();
    jg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * rig.B), 3));
    this.joints = new THREE.Points(jg, new THREE.PointsMaterial({ color: 0xffffff, size: 5, sizeAttenuation: false, depthTest: false }));
    this.joints.renderOrder = 11;
    this.joints.frustumCulled = false;
    this.bones.visible = this.joints.visible = false;
    this.group.add(this.bones, this.joints);
    this.look = CHIMERA_LOOK;
    this.weightsOf(-1);
  }

  set look(l: Look) {
    for (const m of [this.mat, this.backMat]) {
      m.uniforms.uFill.value.set(...l.fill);
      m.uniforms.uLine.value.set(...l.line);
      m.uniforms.uEdge.value.set(...l.edge);
    }
  }
  set mode(m: Mode) {
    for (const mat of [this.mat, this.backMat]) mat.uniforms.uMode.value = m === "wire" ? 0 : m === "heat" ? 1 : 2;
  }
  set showBones(on: boolean) {
    this.bones.visible = this.joints.visible = on;
  }
  set backFaces(on: boolean) {
    this.back.visible = on;
  }

  /** free the GPU's copies of its geometry and materials (take `group` out of the scene too) */
  dispose(): void {
    this.geo.dispose();
    for (const m of [this.mat, this.backMat]) m.dispose();
    for (const o of [this.bones, this.joints]) {
      o.geometry.dispose();
      (o.material as THREE.Material).dispose();
    }
  }

  /** the lines' brightness again from the rig's tone (after it changed) */
  toneChanged(): void {
    const T = this.geo.getAttribute("tone") as THREE.BufferAttribute, A = T.array as Float32Array;
    for (let c = 0; c < this.corner.length; c++) A[c] = this.rig.tone[this.corner[c]];
    T.needsUpdate = true;
  }

  /** the weights view: every bone its own colour (bone < 0), or one bone's weight as brightness */
  weightsOf(bone: number): void {
    const rig = this.rig, T = this.geo.getAttribute("tint") as THREE.BufferAttribute, A = T.array as Float32Array;
    const hue = (b: number) => new THREE.Color().setHSL(((b * 0.61803) % 1 + 1) % 1, 0.85, 0.55);
    const cols = rig.bones.map((_, b) => hue(b));
    const per = new Float32Array(3 * rig.n);
    for (let i = 0; i < rig.n; i++) {
      for (let j = 0; j < rig.INF; j++) {
        const w = rig.skinWeight[rig.INF * i + j], b = rig.skinIndex[rig.INF * i + j];
        if (!w) continue;
        if (bone < 0) {
          per[3 * i] += w * cols[b].r;
          per[3 * i + 1] += w * cols[b].g;
          per[3 * i + 2] += w * cols[b].b;
        } else if (b === bone) {
          const c = new THREE.Color().setHSL(0.66 - 0.66 * w, 1, 0.25 + 0.35 * w);
          per.set([c.r, c.g, c.b], 3 * i);
        }
      }
      if (bone >= 0 && per[3 * i] + per[3 * i + 1] + per[3 * i + 2] === 0) per.set([0.05, 0.05, 0.2], 3 * i);
    }
    for (let c = 0; c < this.corner.length; c++) A.set(per.subarray(3 * this.corner[c], 3 * this.corner[c] + 3), 3 * c);
    T.needsUpdate = true;
  }

  /** draw the mesh at `posed` (call after rig.skin into view.posed), with each quad's heat */
  update(heat?: Float32Array): void {
    const rig = this.rig, P = this.posed;
    vertexNormals(rig, P, this.normals);
    const gp = this.geo.getAttribute("position") as THREE.BufferAttribute, gn = this.geo.getAttribute("normal") as THREE.BufferAttribute, gh = this.geo.getAttribute("heat") as THREE.BufferAttribute;
    const A = gp.array as Float32Array, NN = gn.array as Float32Array, HH = gh.array as Float32Array;
    for (let c = 0; c < this.corner.length; c++) {
      const v = 3 * this.corner[c];
      A[3 * c] = P[v];
      A[3 * c + 1] = P[v + 1];
      A[3 * c + 2] = P[v + 2];
      NN[3 * c] = this.normals[v];
      NN[3 * c + 1] = this.normals[v + 1];
      NN[3 * c + 2] = this.normals[v + 2];
      if (heat) HH[c] = heat[Math.floor(c / 6)];
    }
    gp.needsUpdate = gn.needsUpdate = true;
    if (heat) gh.needsUpdate = true;
    // the skeleton
    const bp = this.bones.geometry.getAttribute("position") as THREE.BufferAttribute, jp = this.joints.geometry.getAttribute("position") as THREE.BufferAttribute;
    rig.bones.forEach((b, i) => {
      const par = b.parent < 0 ? i : b.parent;
      (bp.array as Float32Array).set([...rig.P[par].toArray(), ...rig.P[i].toArray()], 6 * i);
      (jp.array as Float32Array).set(rig.P[i].toArray(), 3 * i);
    });
    bp.needsUpdate = jp.needsUpdate = true;
  }
}
