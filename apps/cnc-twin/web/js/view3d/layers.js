/* Result layers painted on the part (docs/ui-3d.md "Layers"): what each layer reads, its colour map
 * (the shell's layerColormap of web/js/plot/colormap.js, so the part matches the shell's colour bar),
 * its display unit, and the part shader that colours fragments by the layer.
 *
 * | layer       | part                                              | also                         |
 * |-------------|---------------------------------------------------|------------------------------|
 * | temperature | 3-D temperature grid (else the 2-D surface one)   | tool: edge temperature; chips by temp_K |
 * | chips       | chip pile height over the part (2-D grid)         | piles coloured by height     |
 * | coolant     | wetted / dry cells (2-D grid, categories)         | jets, shadowed flow          |
 * | finish      | Ra per vertex (final brick attributes)            |                              |
 * | deviation   | deviation per vertex (final brick attributes)     |                              |
 * | deflection  | displacement magnitude (3-D grid, when recorded)  | tool: |deflection|           |
 * | wear        | VB of the generating edge per vertex              | tool: max VB of the teeth    |
 *
 * Fields are sampled per fragment from half-float textures (normalised convolution: value/weight of
 * an RG texture, so empty cells do not bleed); a 2-D field (a map of the top of the material) colours
 * only faces whose normal is within acos(min_surface_field_cos) of the map's normal, so walls keep
 * the material colour instead of the colour of the map cell above them; attributes come per vertex (NO_VALUE where a brick has
 * none, e.g. a brick that is not yet at its final version: those keep the material colour). A share
 * `layer_unlit_fraction` of the layer colour is shown unlit, so shading never makes a colour read as
 * a different value on the bar. */

import * as THREE from './three.js';
import { lut, LAYER_MAPS, mode as schemeMode, chrome } from '../plot/colormap.js';

export const LAYERS = ['temperature', 'chips', 'coolant', 'finish', 'deviation', 'deflection', 'wear'];

export const ATTRIBUTE_NAMES = {
  finish: ['Ra_m', 'finish_Ra_m', '_FINISH', 'FINISH', '_finish'],
  deviation: ['deviation_m', 'dev_m', '_DEV', 'DEV', '_dev'],
  wear: ['generating_VB_m', 'VB_m', 'wear_m', '_WEAR', 'WEAR', '_wear'],
};

/** Display unit of a layer's SI values: shown = value * scale + offset. */
export const DISPLAY = {
  temperature: { unit: '°C', scale: 1, offset: -273.15, si: 'K' },
  chips: { unit: 'mm', scale: 1e3, offset: 0, si: 'm' },
  coolant: { unit: '', scale: 1, offset: 0, si: '-' },
  finish: { unit: 'µm', scale: 1e6, offset: 0, si: 'm' },
  deviation: { unit: 'µm', scale: 1e6, offset: 0, si: 'm' },
  deflection: { unit: 'µm', scale: 1e6, offset: 0, si: 'm' },
  wear: { unit: 'µm', scale: 1e6, offset: 0, si: 'm' },
};

export const MODE = { none: 0, attribute: 1, field3: 2, field2: 3, categories2: 4 };

/** sRGB [0, 1] of a hex or rgb()/rgba() colour text. */
export function cssColor(text) {
  const c = new THREE.Color();
  const m = /rgba?\(([^)]+)\)/.exec(text || '');
  if (m) {
    const [r, g, b] = m[1].split(',').map((s) => Number(s.trim()) / 255);
    return [r, g, b];
  }
  c.setStyle(text || '#808080', THREE.SRGBColorSpace);
  const out = c.clone().convertLinearToSRGB();
  return [out.r, out.g, out.b];
}

/** The coolant categories (colours of the shell's coolant legend): wetted, dry, jet, shadowed flow. */
export function coolantColors() {
  const c = chrome();
  return { wetted: c.series[0], dry: c.ink2, jet: c.series[2], shadow: c.series[1], dryAlpha: 0.35 };
}

/** A 256 x 1 sRGB lookup texture of the layer's colour map in the page's light/dark mode. */
export function lutTexture(layer, scheme = schemeMode()) {
  const name = LAYER_MAPS[layer] || 'sequential';
  const table = lut(name === 'categorical' ? 'sequential' : name, scheme, 256);
  const data = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    data[i * 4] = table[i * 3];
    data[i * 4 + 1] = table[i * 3 + 1];
    data[i * 4 + 2] = table[i * 3 + 2];
    data[i * 4 + 3] = 255;
  }
  const texture = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

/** The CPU twin of the shader's lookup: linear THREE.Color of `value` on [lo, hi]. */
export function lutColor(table, value, lo, hi, out = new THREE.Color()) {
  const t = hi > lo ? Math.max(0, Math.min(1, (value - lo) / (hi - lo))) : 0;
  const x = t * 255;
  const i = Math.min(254, Math.floor(x));
  const f = x - i;
  const r = (table[i * 3] + (table[(i + 1) * 3] - table[i * 3]) * f) / 255;
  const g = (table[i * 3 + 1] + (table[(i + 1) * 3 + 1] - table[i * 3 + 1]) * f) / 255;
  const b = (table[i * 3 + 2] + (table[(i + 1) * 3 + 2] - table[i * 3 + 2]) * f) / 255;
  return out.setRGB(r, g, b, THREE.SRGBColorSpace);
}

const VERTEX_DECL = `
attribute float aLayer;
varying float vLayer;
varying vec3 vLayerPos;
uniform mat4 uSFromWorld;
`;
const VERTEX_BODY = `
vLayer = aLayer;
vLayerPos = ( uSFromWorld * modelMatrix * vec4( transformed, 1.0 ) ).xyz;
`;
const FRAGMENT_DECL = `
uniform int uLayerMode;
uniform sampler2D uLut;
uniform vec2 uRange;
uniform sampler3D uField3;
uniform sampler2D uField2;
uniform vec2 uFieldSpan;
uniform vec3 uFieldOrigin;
uniform float uFieldCell;
uniform vec3 uFieldN;
uniform mat3 uFieldAxes;
uniform vec3 uWetColor;
uniform vec3 uDryColor;
uniform float uDryAlpha;
uniform float uUnlit;
uniform float uFaceCos;
uniform float uFieldFloor;
varying float vLayer;
varying vec3 vLayerPos;
vec3 layerLut( float v ) {
  float t = clamp( ( v - uRange.x ) / max( uRange.y - uRange.x, 1e-30 ), 0.0, 1.0 );
  return texture2D( uLut, vec2( t * ( 255.0 / 256.0 ) + 0.5 / 256.0, 0.5 ) ).rgb;
}
bool layerField( out float value ) {
  vec3 rel = uFieldAxes * ( vLayerPos - uFieldOrigin );
  if ( uLayerMode != 2 ) {
    vec3 faceNormal = normalize( cross( dFdx( vLayerPos ), dFdy( vLayerPos ) ) );
    vec3 gridNormal = vec3( uFieldAxes[ 0 ][ 2 ], uFieldAxes[ 1 ][ 2 ], uFieldAxes[ 2 ][ 2 ] );
    if ( abs( dot( faceNormal, gridNormal ) ) < uFaceCos ) return false;
  }
  vec3 uvw = rel / ( uFieldCell * uFieldN );
  vec3 margin = 0.5 / uFieldN;
  bool three = uLayerMode == 2;
  if ( uvw.x < -margin.x || uvw.x > 1.0 + margin.x || uvw.y < -margin.y || uvw.y > 1.0 + margin.y ) return false;
  if ( three && ( uvw.z < -margin.z || uvw.z > 1.0 + margin.z ) ) return false;
  vec2 s = three ? texture( uField3, uvw ).rg : texture2D( uField2, uvw.xy ).rg;
  if ( s.g < 0.05 ) return false;
  value = uFieldSpan.x + ( s.r / s.g ) * ( uFieldSpan.y - uFieldSpan.x );
  return value >= uFieldFloor;
}
`;
const FRAGMENT_COLOR = `
float layerOn = 0.0;
if ( uLayerMode == 1 ) {
  if ( vLayer > -1e29 ) { diffuseColor.rgb = layerLut( vLayer ); layerOn = 1.0; }
} else if ( uLayerMode >= 2 ) {
  float fieldValue;
  if ( layerField( fieldValue ) ) {
    if ( uLayerMode == 4 ) {
      diffuseColor.rgb = fieldValue >= 0.5 ? uWetColor : mix( diffuseColor.rgb, uDryColor, uDryAlpha );
    } else {
      diffuseColor.rgb = layerLut( fieldValue );
    }
    layerOn = 1.0;
  }
}
`;
const FRAGMENT_UNLIT = `
outgoingLight = mix( outgoingLight, diffuseColor.rgb, layerOn * uUnlit );
`;

/** Uniforms shared by every part material, and the patch that adds the layer to a material. */
export class PartShading {
  constructor({ unlitFraction, faceCos }) {
    const dummy3 = new THREE.Data3DTexture(new Uint16Array(2), 1, 1, 1);
    dummy3.format = THREE.RGFormat;
    dummy3.type = THREE.HalfFloatType;
    dummy3.needsUpdate = true;
    const dummy2 = new THREE.DataTexture(new Uint16Array(2), 1, 1, THREE.RGFormat, THREE.HalfFloatType);
    dummy2.needsUpdate = true;
    this.dummies = [dummy3, dummy2];
    this.uniforms = {
      uLayerMode: { value: 0 },
      uLut: { value: lutTexture('chips') },
      uRange: { value: new THREE.Vector2(0, 1) },
      uField3: { value: dummy3 },
      uField2: { value: dummy2 },
      uFieldSpan: { value: new THREE.Vector2(0, 1) },
      uFieldOrigin: { value: new THREE.Vector3() },
      uFieldCell: { value: 1 },
      uFieldN: { value: new THREE.Vector3(1, 1, 1) },
      uFieldAxes: { value: new THREE.Matrix3() },
      uWetColor: { value: new THREE.Color() },
      uDryColor: { value: new THREE.Color() },
      uDryAlpha: { value: 0.35 },
      uUnlit: { value: unlitFraction },
      uFaceCos: { value: faceCos },
      uFieldFloor: { value: -1e30 },
      uSFromWorld: { value: new THREE.Matrix4() },
    };
    this.lutLayer = 'chips';
    this.lutScheme = null;
  }

  patch(material) {
    const uniforms = this.uniforms;
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>\n${VERTEX_DECL}`)
        .replace('#include <project_vertex>', `#include <project_vertex>\n${VERTEX_BODY}`);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\n${FRAGMENT_DECL}`)
        .replace('#include <color_fragment>', `#include <color_fragment>\n${FRAGMENT_COLOR}`)
        .replace('#include <opaque_fragment>', `${FRAGMENT_UNLIT}\n#include <opaque_fragment>`);
    };
    material.customProgramCacheKey = () => 'cnctwin-part-layer-v1';
    return material;
  }

  useLut(layer) {
    const scheme = schemeMode();
    if (this.lutLayer === layer && this.lutScheme === scheme) return;
    this.uniforms.uLut.value.dispose();
    this.uniforms.uLut.value = lutTexture(layer, scheme);
    this.lutLayer = layer;
    this.lutScheme = scheme;
  }

  /** No layer on the part. */
  off() { this.uniforms.uLayerMode.value = MODE.none; }

  /** Per-vertex values on [lo, hi]. */
  attribute(layer, lo, hi) {
    this.useLut(layer);
    this.uniforms.uRange.value.set(lo, hi);
    this.uniforms.uLayerMode.value = MODE.attribute;
  }

  /** A grid (fields.js) with its texture {texture, min, max}; `categories` for the coolant layer; values
   * below `floor` (chip piles thinner than min_pile_height_m) keep the material colour. */
  field(layer, grid, tex, lo, hi, categories = null, floor = null) {
    this.useLut(layer);
    const u = this.uniforms;
    u.uFieldFloor.value = floor === null || floor === undefined ? -1e30 : floor;
    u.uRange.value.set(lo, hi);
    u.uFieldSpan.value.set(tex.min, tex.max);
    const o = grid.origin;
    u.uFieldOrigin.value.set(o[0], o[1], Number.isFinite(o[2]) ? o[2] : 0);
    u.uFieldCell.value = grid.cell;
    u.uFieldN.value.set(grid.n[0], grid.n[1], grid.n[2]);
    const a = grid.axes;
    u.uFieldAxes.value.set(a[0][0], a[0][1], a[0][2], a[1][0], a[1][1], a[1][2], a[2][0], a[2][1], a[2][2]);
    if (grid.dims === 3) { u.uField3.value = tex.texture; u.uLayerMode.value = MODE.field3; } else {
      u.uField2.value = tex.texture;
      u.uLayerMode.value = categories ? MODE.categories2 : MODE.field2;
    }
    if (categories) {
      u.uWetColor.value.setStyle(categories.wetted, THREE.SRGBColorSpace);
      u.uDryColor.value.setStyle(categories.dry, THREE.SRGBColorSpace);
      u.uDryAlpha.value = categories.dryAlpha;
    }
  }

  dispose() {
    this.uniforms.uLut.value.dispose();
    for (const t of this.dummies) t.dispose();
  }
}
