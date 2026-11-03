/* Camera of the 3D view: perspective, +z of S up (machine z), orbit / pan / zoom with mouse, pen and
 * touch (three.js OrbitControls: one finger turns, two fingers pinch to zoom and drag to pan; mouse
 * left turns, right or shift-drag pans, wheel zooms toward the pointer), framing of the setup, and
 * the follow-tool mode that carries the orbit centre with the tool tip (the camera keeps its offset, so
 * the user can still orbit and zoom while it follows). Near and far planes follow the orbit distance,
 * so a 10 um deflection and a 2 m machine share one depth buffer without z-fighting. */

import * as THREE from './three.js';
import { OrbitControls } from './three.js';

export class CameraRig {
  constructor(dom, params, onChange) {
    this.params = params;
    this.camera = new THREE.PerspectiveCamera(THREE.MathUtils.radToDeg(params.fov_rad), 1, 1e-4, 10);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.3, -0.3, 0.3);
    this.controls = new OrbitControls(this.camera, dom);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = params.orbit_damping;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    this.controls.addEventListener('change', () => onChange());
    this.following = false;
    this.sceneRadius = 1;
    this.home = null;
    this.followTarget = new THREE.Vector3();
    this.hasFollowTarget = false;
  }

  resize(width, height) {
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
  }

  /** Camera direction from azimuth (about +z from +x) and elevation. */
  direction(azimuth, elevation) {
    return new THREE.Vector3(Math.cos(elevation) * Math.cos(azimuth), Math.cos(elevation) * Math.sin(azimuth), Math.sin(elevation));
  }

  /** Frame a box from the default direction (the setup view). */
  frame(box, { remember = true } = {}) {
    if (!box || box.isEmpty()) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const radius = Math.max(sphere.radius, 1e-4) * this.params.framing_margin;
    const vertical = THREE.MathUtils.degToRad(this.camera.fov) / 2;
    const horizontal = Math.atan(Math.tan(vertical) * this.camera.aspect);
    const distance = radius / Math.sin(Math.min(vertical, horizontal));
    const dir = this.direction(this.params.view_azimuth_rad, this.params.view_elevation_rad);
    this.controls.target.copy(sphere.center);
    this.camera.position.copy(sphere.center).addScaledVector(dir, distance);
    this.controls.update();
    if (remember) this.home = { target: sphere.center.clone(), position: this.camera.position.clone() };
  }

  reset() {
    if (!this.home) return;
    this.controls.target.copy(this.home.target);
    this.camera.position.copy(this.home.position);
    this.controls.update();
  }

  /** Follow the tool: the first time, frame follow_view_height_m around the tip. */
  follow(on, tip = null) {
    this.following = !!on;
    if (this.following && tip) {
      const height = this.params.follow_view_height_m;
      const distance = height / 2 / Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
      const offset = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
      if (offset.length() > distance * 3 || offset.length() < distance * 0.1) offset.setLength(distance);
      this.controls.target.copy(tip);
      this.camera.position.copy(tip).add(offset);
      this.followTarget.copy(tip);
      this.hasFollowTarget = true;
      this.controls.update();
    }
  }

  /** Move the orbit centre with the tip: exponential smoothing over follow_smoothing_s of wall time (dt of
   * the first frame after an idle pause is taken as one 60 Hz frame); a jump of more than half the view
   * (a seek) is followed at once. Returns true while the centre is still catching up. */
  track(tip, dt) {
    if (!this.following || !tip) return false;
    if (!this.hasFollowTarget) { this.followTarget.copy(tip); this.hasFollowTarget = true; }
    const viewHeight = 2 * this.camera.position.distanceTo(this.controls.target) * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
    const gap = this.followTarget.distanceTo(tip);
    const step = dt > 0 ? dt : 1 / 60;
    const k = gap > viewHeight / 2 || this.params.follow_smoothing_s <= 0 ? 1 : 1 - Math.exp(-step / this.params.follow_smoothing_s);
    const next = this.followTarget.clone().lerp(tip, k);
    const delta = next.clone().sub(this.controls.target);
    this.followTarget.copy(next);
    if (delta.lengthSq() > 0) {
      this.controls.target.add(delta);
      this.camera.position.add(delta);
    }
    return this.followTarget.distanceTo(tip) > viewHeight * 1e-4;
  }

  /** Near/far from the orbit distance; returns whether the controls are still moving (damping). */
  update() {
    const moving = this.controls.update();
    const distance = this.camera.position.distanceTo(this.controls.target);
    const near = Math.max(distance * this.params.near_fraction, 1e-6);
    const far = Math.max(distance + this.sceneRadius * this.params.far_scene_factor, near * 100);
    if (Math.abs(this.camera.near - near) > near * 1e-3 || Math.abs(this.camera.far - far) > far * 1e-3) {
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }
    return moving;
  }

  dispose() { this.controls.dispose(); }
}
