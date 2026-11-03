/* Section views: one clipping plane normal to x, y or z of S at an offset, cutting the part, the
 * fixture, the chips and the piles (machine, tool, enclosure and coolant stay whole, as in CAD section
 * views), with the cut faces capped.
 *
 * Caps use the stencil parity of closed solids (P. McReynolds and D. Blythe, "Advanced Graphics
 * Programming Using OpenGL", Morgan Kaufmann 2005, 16.10 "Capping clipped geometry"; the technique of
 * three.js's webgl_clipping_stencil example): the clipped solids' back faces increment and their
 * front faces decrement the stencil (no colour, no depth); a plane drawn where the stencil is not zero
 * is where the section plane runs inside material. The part and the fixture are capped one after the
 * other, each in its own colour (cap_srgb, fixture_cap_srgb). It is exact for closed meshes (the union of the
 * workpiece bricks and the fixture solids are); an open mesh leaks cap colour along its opening. The
 * plane keeps the points with n.p + c >= 0: for `flip` false the material beyond the offset (larger
 * coordinate) is removed, for `flip` true the material before it. */

import * as THREE from './three.js';

const AXIS = { x: 0, y: 1, z: 2 };

function stencilMaterials(plane) {
  const base = new THREE.MeshBasicMaterial();
  base.depthWrite = false;
  base.depthTest = false;
  base.colorWrite = false;
  base.stencilWrite = true;
  base.stencilFunc = THREE.AlwaysStencilFunc;
  const back = base.clone();
  back.side = THREE.BackSide;
  back.clippingPlanes = [plane];
  back.stencilFail = THREE.IncrementWrapStencilOp;
  back.stencilZFail = THREE.IncrementWrapStencilOp;
  back.stencilZPass = THREE.IncrementWrapStencilOp;
  const front = base.clone();
  front.side = THREE.FrontSide;
  front.clippingPlanes = [plane];
  front.stencilFail = THREE.DecrementWrapStencilOp;
  front.stencilZFail = THREE.DecrementWrapStencilOp;
  front.stencilZPass = THREE.DecrementWrapStencilOp;
  base.dispose();
  return { back, front };
}

/** One capped group (part, or fixture): its stencil materials and its cap plane, drawn flat (unlit) in its
 * colour as CAD section views are. The part's passes run
 * first (render order 1, cap 1.1), the fixture's after (1.2, 1.3); each cap clears the stencil after it
 * is drawn. Where the two solids' sections overlap (bodies that intersect), the part's cap stays: the
 * fixture cap is drawn with a strict depth test against the coplanar part cap. */
class CapGroup {
  constructor(scene, plane, rgb, order, name) {
    Object.assign(this, stencilMaterials(plane));
    this.order = order;
    this.back.userData.order = order;
    const colour = new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
    this.material = new THREE.MeshBasicMaterial({
      color: colour, side: THREE.DoubleSide,
      stencilWrite: true, stencilRef: 0, stencilFunc: THREE.NotEqualStencilFunc,
      stencilFail: THREE.ReplaceStencilOp, stencilZFail: THREE.ReplaceStencilOp, stencilZPass: THREE.ReplaceStencilOp,
    });
    if (order > 1) this.material.depthFunc = THREE.LessDepth;
    this.cap = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.material);
    this.cap.name = `section cap (${name})`;
    this.cap.renderOrder = order + 0.1;
    this.cap.visible = false;
    this.cap.onAfterRender = (renderer) => renderer.clearStencil();
    scene.add(this.cap);
  }

  dispose() {
    this.cap.removeFromParent();
    this.cap.geometry.dispose();
    this.material.dispose();
    this.back.dispose();
    this.front.dispose();
  }
}

export class Section {
  constructor(scene, { capSrgb, fixtureCapSrgb }) {
    this.scene = scene;
    this.plane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0);
    this.active = false;
    this.state = { axis: null, offset_m: 0, flip: false };
    this.partCap = new CapGroup(scene, this.plane, capSrgb, 1, 'part');
    this.fixtureCap = new CapGroup(scene, this.plane, fixtureCapSrgb, 1.2, 'fixture');
    this.back = this.partCap.back;
    this.front = this.partCap.front;
    this.clipped = new Set();
    this.solids = new Set();
  }

  /** Materials cut by the plane (part, fixture, chips, piles). */
  clip(material) {
    this.clipped.add(material);
    material.clippingPlanes = this.active ? [this.plane] : [];
    material.needsUpdate = true;
  }

  /** Closed fixture meshes that take part in the fixture caps: stencil companions as children. */
  addSolid(mesh) {
    const group = this.fixtureCap;
    const back = new THREE.Mesh(mesh.geometry, group.back);
    const front = new THREE.Mesh(mesh.geometry, group.front);
    back.renderOrder = group.order;
    front.renderOrder = group.order;
    back.visible = this.active;
    front.visible = this.active;
    back.name = `${mesh.name || 'solid'} stencil back`;
    front.name = `${mesh.name || 'solid'} stencil front`;
    mesh.add(back, front);
    mesh.renderOrder = 6;
    this.solids.add({ mesh, back, front });
  }

  /** Apply {axis, offset_m, flip}; `bounds` (THREE.Box3 in the view frame) sizes the caps. */
  set(section, bounds) {
    const axis = section && section.axis ? String(section.axis).toLowerCase() : null;
    const valid = axis !== null && AXIS[axis] !== undefined && Number.isFinite(Number(section.offset_m));
    this.state = valid ? { axis, offset_m: Number(section.offset_m), flip: !!section.flip } : { axis: null, offset_m: 0, flip: false };
    this.active = valid;
    if (valid) {
      const normal = new THREE.Vector3();
      normal.setComponent(AXIS[axis], section.flip ? 1 : -1);
      const offset = Number(section.offset_m);
      this.plane.normal.copy(normal);
      this.plane.constant = section.flip ? -offset : offset;
    }
    for (const material of this.clipped) {
      material.clippingPlanes = this.active ? [this.plane] : [];
      material.needsUpdate = true;
    }
    for (const solid of this.solids) { solid.back.visible = this.active; solid.front.visible = this.active; }
    for (const group of [this.partCap, this.fixtureCap]) {
      group.cap.visible = this.active && (group === this.partCap || this.solids.size > 0);
      if (this.active) this._place(group.cap, bounds);
    }
    return this.active;
  }

  _place(cap, bounds) {
    const box = bounds && !bounds.isEmpty() ? bounds : new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5));
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() * 1.5 || 1;
    this.plane.projectPoint(center, center);
    cap.position.copy(center);
    cap.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), this.plane.normal.clone().negate());
    cap.scale.set(size, size, 1);
    cap.updateMatrixWorld();
  }

  /** {normal, constant} for picking, or null. */
  pickPlane() {
    return this.active ? { normal: this.plane.normal.clone(), constant: this.plane.constant } : null;
  }

  dispose() {
    this.partCap.dispose();
    this.fixtureCap.dispose();
  }
}
