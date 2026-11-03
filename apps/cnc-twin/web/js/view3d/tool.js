/* The tool of every path where it really is: its envelope (scene.tools[].envelope, a GLB in T: origin at
 * the tip, +z_T toward the spindle) and holder (holder_envelope) placed with the frames.bin tip and tool
 * axis, turned by the spindle angle, and the deflected tool drawn over it.
 *
 * Deflected tool: the envelope bent as a cantilever clamped at the holder face (z_T = stickout) under
 * a load at the tip (Euler-Bernoulli beam with an end load, S. Timoshenko and J. Gere, "Mechanics of
 * Materials", 1972, table of beam deflections: w(x) = P x^2 (3L - x) / (6 E I)), normalised to the
 * recorded tip deflection u: lateral displacement u_xy (3 xi^2 - xi^3) / 2 with xi = (z_c - z) / z_c
 * the distance from the clamp over the stickout (0 at the holder face, 1 at the tip), axial u_z xi
 * (uniform strain), nothing inside the holder; all times the exaggeration factor shown beside it. The
 * shape is exact for a uniform prismatic beam loaded at its tip; a load spread evenly along the flutes
 * bends it, for the same tip value, by a shape within 5 % of the tip deflection of this one (uniform load:
 * x^2 (6L^2 - 4Lx + x^2) / (3 L^4) against x^2 (3L - x) / (2 L^3); largest difference 0.043 u at
 * x = 0.58 L). The deflection is the
 * keyframe's tool_deflection_m (T0 components) or, when the run records a tool displacement channel,
 * its tick-tier value at the time shown. */

import * as THREE from './three.js';
import { bakeGltf, parseGlb } from './scene3d.js';
import { toolMatrix } from './poses.js';

const GHOST_DECL = `
uniform vec3 uDeflect;
uniform float uClampZ;
`;
const GHOST_BODY = `
float bendXi = clamp( ( uClampZ - transformed.z ) / max( uClampZ, 1e-9 ), 0.0, 1.0 );
transformed += vec3( uDeflect.xy * ( 3.0 * bendXi * bendXi - bendXi * bendXi * bendXi ) * 0.5, uDeflect.z * bendXi );
`;

/** The deflected tool's displacement at height z_T (the shader's formula, for tests and picking): lateral
 * u_xy (3 xi^2 - xi^3) / 2 and axial u_z xi, xi = clamp((z_c - z) / z_c, 0, 1). */
export function bendOffset(z, clampZ, u) {
  const xi = Math.max(0, Math.min(1, (clampZ - z) / Math.max(clampZ, 1e-9)));
  const lateral = (3 * xi * xi - xi * xi * xi) / 2;
  return [u[0] * lateral, u[1] * lateral, u[2] * xi];
}

function cylinder(radius, length, sides) {
  const geometry = new THREE.CylinderGeometry(radius, radius, length, sides, 1, false);
  geometry.rotateX(Math.PI / 2);
  geometry.translate(0, 0, length / 2);
  return geometry;
}

export class ToolRig {
  constructor({ run, params, root, materials, notes, nPaths }) {
    this.run = run;
    this.params = params;
    this.root = root;
    this.materials = materials;
    this.notes = notes;
    this.tools = [];
    this.paths = [];
    this.visible = { tool: true, deflectedTool: false };
    this.exaggeration = params.deflection_exaggeration_default;
    this.ghostUniforms = [];
    for (let p = 0; p < nPaths; p++) {
      const group = new THREE.Group();
      group.name = `tool path ${p}`;
      group.matrixAutoUpdate = false;
      const spin = new THREE.Group();
      spin.matrixAutoUpdate = false;
      group.add(spin);
      root.add(group);
      const toolMaterial = materials.tool.clone();
      const uniforms = { uDeflect: { value: new THREE.Vector3() }, uClampZ: { value: 0.03 } };
      const ghostMaterial = new THREE.MeshStandardMaterial({
        color: new THREE.Color().setRGB(...params.deflected_tool_srgb, THREE.SRGBColorSpace),
        transparent: true, opacity: params.deflected_tool_opacity, depthWrite: false, roughness: 0.5, metalness: 0.1,
      });
      ghostMaterial.onBeforeCompile = (shader) => {
        Object.assign(shader.uniforms, uniforms);
        shader.vertexShader = shader.vertexShader
          .replace('#include <common>', `#include <common>\n${GHOST_DECL}`)
          .replace('#include <begin_vertex>', `#include <begin_vertex>\n${GHOST_BODY}`);
      };
      ghostMaterial.customProgramCacheKey = () => 'cnctwin-deflected-tool-v1';
      this.paths.push({ group, spin, toolMesh: null, holderMesh: null, ghost: null, toolMaterial, ghostMaterial, uniforms, current: null, deflection: null });
    }
  }

  async load() {
    const scene = this.run.scene;
    const list = Array.isArray(scene.tools) ? scene.tools : [];
    const crease = this.params.crease_angle_rad;
    const tasks = list.map(async (entry, index) => {
      const diameter = Number(entry.diameter_m) || null;
      const stickout = Number(entry.stickout_m ?? entry.length_m) || null;
      let envelope = null;
      let holder = null;
      if (entry.envelope && this.run.has(entry.envelope)) {
        try { envelope = bakeGltf(await parseGlb(await this.run.file(entry.envelope)), crease); } catch (error) {
          this.notes.add(`tool T${entry.number ?? index}: envelope not readable (${error.message})`);
        }
      }
      if (!envelope && diameter && stickout) envelope = cylinder(diameter / 2, stickout, this.params.tool_sides);
      if (entry.holder_envelope && this.run.has(entry.holder_envelope)) {
        try { holder = bakeGltf(await parseGlb(await this.run.file(entry.holder_envelope)), crease); } catch (error) {
          this.notes.add(`tool T${entry.number ?? index}: holder not readable (${error.message})`);
        }
      }
      if (!envelope) { this.notes.add(`tool T${entry.number ?? index}: no envelope and no size`); return; }
      envelope.computeBoundingBox();
      const clampZ = stickout || envelope.boundingBox.max.z;
      this.tools.push({
        slot: Number(entry.slot ?? index), number: entry.number === undefined || entry.number === null ? null : Number(entry.number),
        id: entry.tool_id || null, envelope, holder, clampZ, radius: diameter ? diameter / 2 : Math.max(envelope.boundingBox.max.x, envelope.boundingBox.max.y),
      });
    });
    await Promise.all(tasks);
    this.tools.sort((a, b) => a.slot - b.slot);
  }

  /** The tool of `path` at the sample: frames' tool field, else timeline.json, else the first tool. */
  toolFor(sample, path, layout) {
    if (!this.tools.length) return null;
    if (sample.hasTool && sample.tool[path] >= 0) {
      const value = sample.tool[path];
      const found = layout.toolField === 'tool_slot' ? this.tools.find((t) => t.slot === value) : this.tools.find((t) => t.number === value);
      if (found) return found;
      if (value === 0 && layout.toolField === 'tool_number') return null;
    }
    const number = this.run.toolAt(sample.tick, path);
    if (number !== null) {
      const found = this.tools.find((t) => t.number === number);
      if (found) return found;
    }
    return this.tools[0];
  }

  _mount(state, tool) {
    if (state.current === tool) return;
    for (const mesh of [state.toolMesh, state.holderMesh, state.ghost]) if (mesh) mesh.removeFromParent();
    state.current = tool;
    state.toolMesh = state.holderMesh = state.ghost = null;
    if (!tool) return;
    state.toolMesh = new THREE.Mesh(tool.envelope, state.toolMaterial);
    state.toolMesh.name = `tool ${tool.number ?? tool.slot}`;
    state.toolMesh.userData.kind = 'tool';
    state.spin.add(state.toolMesh);
    if (tool.holder) {
      state.holderMesh = new THREE.Mesh(tool.holder, this.materials.holder);
      state.holderMesh.name = `holder of tool ${tool.number ?? tool.slot}`;
      state.holderMesh.userData.kind = 'holder';
      state.spin.add(state.holderMesh);
    }
    state.ghost = new THREE.Mesh(tool.envelope, state.ghostMaterial);
    state.ghost.name = `deflected tool ${tool.number ?? tool.slot}`;
    state.ghost.renderOrder = 8;
    state.uniforms.uClampZ.value = tool.clampZ;
    state.group.add(state.ghost);
  }

  /** Place the tools: sample (frames), xHint (x of T0 in S), deflections [path] -> {vector_m, frame} or
   * null, colours [path] -> THREE.Color or null (layer colour of the tool). */
  update(sample, layout, xHint, deflections, colours) {
    const T0 = [];
    const T = [];
    for (let p = 0; p < this.paths.length; p++) {
      const state = this.paths[p];
      const tip = [sample.tip[p * 3], sample.tip[p * 3 + 1], sample.tip[p * 3 + 2]];
      const axis = [sample.toolAxis[p * 3], sample.toolAxis[p * 3 + 1], sample.toolAxis[p * 3 + 2]];
      const valid = sample.valid && tip.every(Number.isFinite);
      if (!valid) { state.group.visible = false; T0.push(null); T.push(null); continue; }
      const tool = this.toolFor(sample, p, layout);
      this._mount(state, tool);
      toolMatrix(state.group.matrix, tip, axis, xHint, 0);
      state.group.matrixWorldNeedsUpdate = true;
      state.spin.matrix.makeRotationZ(sample.spindle[p]);
      state.spin.matrixWorldNeedsUpdate = true;
      state.group.visible = !!tool;
      if (state.toolMesh) state.toolMesh.visible = this.visible.tool;
      if (state.holderMesh) state.holderMesh.visible = this.visible.tool;
      const deflection = deflections ? deflections[p] : null;
      state.deflection = deflection;
      if (state.ghost) {
        const show = this.visible.deflectedTool && !!deflection;
        state.ghost.visible = show;
        if (deflection) {
          const v = new THREE.Vector3(...deflection.vector_m);
          if (deflection.frame === 'S') v.applyMatrix3(new THREE.Matrix3().setFromMatrix4(state.group.matrix).transpose());
          state.uniforms.uDeflect.value.copy(v.multiplyScalar(this.exaggeration));
        }
      }
      const colour = colours ? colours[p] : null;
      if (colour) state.toolMaterial.color.copy(colour); else state.toolMaterial.color.copy(this.materials.tool.color);
      T0.push(state.group.matrix.clone());
      T.push(state.group.matrix.clone().multiply(state.spin.matrix));
    }
    return { T0, T };
  }

  setVisible({ tool, deflectedTool }) {
    this.visible.tool = tool !== false;
    this.visible.deflectedTool = deflectedTool === true;
  }

  setExaggeration(factor) {
    const p = this.params;
    this.exaggeration = Math.max(p.min_deflection_exaggeration, Math.min(p.max_deflection_exaggeration, Number(factor) || p.deflection_exaggeration_default));
  }

  /** The tool radius of a path (m), for coolant zones. */
  radius(path = 0) {
    const state = this.paths[path];
    return state && state.current ? state.current.radius : null;
  }

  pickables() {
    const out = [];
    for (const state of this.paths) {
      if (!state.group.visible) continue;
      if (state.toolMesh && state.toolMesh.visible) out.push(state.toolMesh);
      if (state.holderMesh && state.holderMesh.visible) out.push(state.holderMesh);
    }
    return out;
  }

  dispose() {
    for (const state of this.paths) {
      state.group.removeFromParent();
      state.toolMaterial.dispose();
      state.ghostMaterial.dispose();
    }
    for (const tool of this.tools) { tool.envelope.dispose(); if (tool.holder) tool.holder.dispose(); }
  }
}
