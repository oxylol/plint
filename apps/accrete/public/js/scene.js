// Accrete · the picture. One WebGL canvas behind the page: the core, every body you own on its orbit, the mass
// falling in, comets, and the wider views (sky, galaxy, universe). It only draws; main.js tells it each frame
// what there is to draw, and calls it when something happens that should be seen.
(function (root) {
  "use strict";
  const TAU = Math.PI * 2;
  const NK = 8;

  // How each kind of body looks: colour, the band it orbits in (the cloud is about 1.1 across), size in px, how
  // thick its band is. Colour arrives with progress: the first kinds are pale dust, the last are saturated.
  const KIND = [
    { col: [0.70, 0.78, 0.92], r: [0.17, 0.33], size: 2.6,  thick: 0.050 },   // Mote
    { col: [0.94, 0.76, 0.52], r: [0.31, 0.43], size: 3.6,  thick: 0.030 },   // Pebble
    { col: [0.98, 0.52, 0.30], r: [0.42, 0.52], size: 5.2,  thick: 0.022 },   // Boulder
    { col: [0.93, 0.38, 0.58], r: [0.52, 0.62], size: 6.6,  thick: 0.018 },   // Asteroid
    { col: [0.76, 0.68, 1.00], r: [0.63, 0.73], size: 9.0,  thick: 0.014 },   // Moon
    { col: [0.32, 0.72, 1.00], r: [0.75, 0.85], size: 12.5, thick: 0.012 },   // Planet
    { col: [0.36, 0.93, 0.68], r: [0.88, 0.98], size: 17.0, thick: 0.010 },   // Giant
    { col: [1.00, 0.84, 0.34], r: [1.02, 1.12], size: 21.0, thick: 0.008 },   // Dwarf
  ];
  // The class of a star, cool to hot.
  const CLASS_COL = { M: [1.0, 0.45, 0.30], K: [1.0, 0.66, 0.36], G: [1.0, 0.90, 0.58], F: [1.0, 0.98, 0.92], A: [0.82, 0.90, 1.0], B: [0.62, 0.76, 1.0], O: [0.48, 0.62, 1.0] };
  const CLASSES = ["M", "K", "G", "F", "A", "B", "O"];
  const MAX_BODY = 1000, MAX_IN = 700, SPIN = 0.028, TILT = 0.42;

  const VS_QUAD = "attribute vec2 a_p; void main(){ gl_Position = vec4(a_p, 0.0, 1.0); }";
  const FS_BG = (oct) => `
precision highp float;
uniform vec2 u_res, u_c;
uniform float u_t, u_S, u_tilt, u_dpr, u_coreR, u_heat, u_ign, u_ignShow, u_surge, u_flash, u_dust, u_pulse, u_tintAmt, u_dim, u_hole, u_cloud, u_web;
uniform vec3 u_tintA, u_tintB, u_tintC, u_haze;
uniform vec4 u_ring[6];
uniform vec4 u_ringCol[6];
uniform float u_orbR[8], u_orbA[8];
uniform vec3 u_kcol[8];
float hash(vec2 p){ p = fract(p * vec2(0.1031, 0.1030)); p += dot(p, p.yx + 33.33); return fract((p.x + p.y) * p.x); }
float noise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*f*(f*(f*6.0-15.0)+10.0);
  return mix(mix(hash(i), hash(i+vec2(1.0,0.0)), f.x), mix(hash(i+vec2(0.0,1.0)), hash(i+vec2(1.0,1.0)), f.x), f.y); }
float fbm(vec2 p){ float v = 0.0, a = 0.5; for (int i = 0; i < ${oct}; i++){ v += a*noise(p); p = mat2(0.8, -0.6, 0.6, 0.8) * p * 2.02 + 11.0; a *= 0.5; } return v; }
vec3 coreColor(float h){
  vec3 a = vec3(0.95, 0.30, 0.10), b = vec3(1.0, 0.56, 0.18), c = vec3(1.0, 0.86, 0.56), d = vec3(1.0, 0.97, 0.92);
  return h < 0.33 ? mix(a, b, h/0.33) : h < 0.66 ? mix(b, c, (h-0.33)/0.33) : mix(c, d, (h-0.66)/0.34);
}
void main(){
  vec2 px = vec2(gl_FragCoord.x, u_res.y - gl_FragCoord.y);
  vec2 uv = (px - u_c) / u_S;
  vec2 sc = px / u_res - 0.5;
  vec2 wide = (px - u_c) / min(u_res.x, u_res.y);       // the same on any zoom: the far sky does not move
  vec3 col = vec3(0.012, 0.014, 0.030) + vec3(0.018, 0.020, 0.046) * (1.0 - length(sc * 1.3));
  // far stars, one to a cell; a few of them bright
  vec2 g = px / (26.0 * u_dpr); vec2 id = floor(g); float h = hash(id);
  if (h > 0.80) {
    vec2 o = vec2(hash(id + 1.3), hash(id + 2.7)) - 0.5;
    float d = length(fract(g) - 0.5 - o * 0.7);
    float tw = 0.65 + 0.35 * sin(u_t * (0.4 + h * 2.0) + h * 40.0);
    float big = step(0.985, h);
    col += mix(vec3(0.72, 0.80, 1.0), vec3(1.0, 0.90, 0.78), hash(id + 5.1)) * smoothstep(0.07 + 0.10 * big, 0.0, d) * tw * ((h - 0.80) * 4.0 + big * 1.5);
  }
  // the far nebula: always there, cool; it warms and gains colour with the kinds of body you own
  vec2 q = wide * 1.5 + vec2(u_t * 0.0025, 0.0);
  vec2 warp = vec2(fbm(q * 1.4 + 3.1), fbm(q * 1.4 - 7.7 + u_t * 0.004));
  float n1 = fbm(q * 1.1 + warp * 1.3);
  float n2 = fbm(q * 2.2 + warp * 0.7 + 9.0);
  vec3 cool = mix(vec3(0.05, 0.09, 0.26), vec3(0.20, 0.07, 0.30), n2);
  vec3 warm = mix(mix(u_tintA, u_tintB, smoothstep(0.30, 0.70, n2)), u_tintC, smoothstep(0.42, 0.75, fbm(q * 0.8 - 4.0)));
  float neb = smoothstep(0.32, 0.85, n1);
  col += mix(cool, warm * 0.75, u_tintAmt) * neb * (0.42 + 0.34 * u_tintAmt);
  col += vec3(0.50, 0.62, 1.0) * pow(max(0.0, n1 - 0.56), 2.0) * 0.9;
  // the cloud itself, around the core: what the star is being made from
  vec2 e = vec2(uv.x, uv.y / u_tilt); float er = length(e); float ea = atan(e.y, e.x);
  if (u_cloud > 0.0) {
    float sw = fbm(vec2(ea * 1.2 + er * 3.2 - u_t * 0.035, er * 4.2 - u_t * 0.02) + warp);
    float fog = exp(-pow(er / 0.85, 1.6)) * (0.30 + 0.95 * sw);
    col += mix(vec3(0.20, 0.28, 0.55), u_haze, 0.4 + 0.5 * u_dust) * fog * (0.16 + 0.20 * u_dust) * u_cloud;
    // the lanes the bodies run in
    float pw = 1.4 * u_dpr / u_S;
    for (int i = 0; i < 8; i++) {
      if (u_orbA[i] > 0.0) col += u_kcol[i] * smoothstep(pw * (1.0 + 5.0 * u_orbA[i]), 0.0, abs(er - u_orbR[i])) * (0.035 + 0.5 * u_orbA[i]) * u_cloud;
    }
  }
  float d = length(px - u_c);
  if (u_coreR > 0.0) {
    float R = u_coreR;
    vec3 cc = coreColor(u_heat);
    float ang = atan(px.x - u_c.x, -(px.y - u_c.y)); if (ang < 0.0) ang += 6.2831853;
    float glow = pow(R / (d + R * 0.5), 1.45);
    vec2 around = (px - u_c) / max(d, 0.001);
    float rays = pow(fbm(around * 2.4 + u_t * 0.07) * 1.25, 3.0) + 0.6 * pow(fbm(around * 5.5 + 4.0 - u_t * 0.11), 3.0);
    col += cc * glow * (0.62 + 0.5 * u_surge + 0.9 * u_pulse + 0.10 * u_heat) * (0.80 + 0.9 * rays * smoothstep(R * 9.0, R, d));
    col += cc * exp(-pow(er / 0.26, 2.0)) * 0.14 * (0.5 + u_heat);
    float disc = smoothstep(R, R - 1.5 * u_dpr, d);
    float limb = sqrt(max(0.0, 1.0 - (d * d) / (R * R)));
    float gran = fbm((px - u_c) / R * 2.6 + vec2(u_t * 0.12, -u_t * 0.07));
    // a cool core shows its grain; a hot one is mostly light
    float grain = mix(0.76, 0.34, u_heat);
    vec3 surf = cc * (1.05 + 0.75 * limb) * (1.0 - grain * 0.5 + grain * gran) * (1.0 + 0.5 * u_heat) + vec3(1.0, 0.92, 0.80) * pow(limb, 4.0) * (0.10 + 0.14 * u_heat + 0.5 * u_pulse);
    col = mix(col, surf, disc);
    // how near ignition is: a thin ring that closes
    float on = step(ang / 6.2831853, u_ign);
    float line = smoothstep(1.25 * u_dpr, 0.2 * u_dpr, abs(d - R * 1.55 - 9.0 * u_dpr));
    col += mix(vec3(0.13, 0.14, 0.21), cc * 1.1 + 0.1, on) * line * u_ignShow;
  }
  if (u_web > 0.0) {
    // the web the galaxies hang on: thin bright threads of ridged noise
    vec2 w = wide * 2.6 + warp * 0.5;
    float th = pow(1.0 - abs(2.0 * fbm(w) - 1.0), 7.0) + 0.6 * pow(1.0 - abs(2.0 * fbm(w * 2.1 + 5.0) - 1.0), 9.0);
    col += mix(vec3(0.30, 0.22, 0.75), vec3(0.20, 0.62, 0.85), n2) * th * 0.42 * u_web;
  }
  if (u_hole > 0.0) {
    // a black hole: nothing, then a bright thin ring
    float R = u_hole;
    col *= smoothstep(R * 0.92, R * 1.02, d);
    col += vec3(1.0, 0.80, 0.55) * smoothstep(R * 0.30, 0.0, abs(d - R * 1.12)) * 0.9;
    col += vec3(1.0, 0.62, 0.36) * pow(R / (d + R), 2.2) * 0.8 * step(R, d);
  }
  for (int i = 0; i < 6; i++) {
    vec4 r = u_ring[i];
    if (r.w > 0.0) col += u_ringCol[i].rgb * u_ringCol[i].a * smoothstep(r.w, 0.0, abs(length(px - r.xy) - r.z));
  }
  col *= 1.0 - u_dim;
  col += vec3(1.0, 0.96, 0.88) * u_flash;
  col *= 1.0 - 0.42 * pow(length(sc) * 1.2, 2.6);
  col = 1.0 - exp(-col * 1.25);                         // bright things roll off instead of burning out
  col += (hash(px + u_t) - 0.5) * 0.012;                // a little grain, so dark gradients do not band
  gl_FragColor = vec4(col, 1.0);
}`;

  const VS_BODY = `
attribute vec4 a_seed;
attribute vec2 a_who;
uniform float u_own[8], u_popFrom[8], u_popT[8];
uniform vec3 u_col[8];
uniform vec4 u_band[8];
uniform vec2 u_res, u_c;
uniform float u_t, u_S, u_tilt, u_dpr, u_coreR, u_fall, u_now, u_alpha;
varying vec3 v_col; varying vec2 v_light; varying float v_kind, v_a, v_px, v_hot, v_seed;
void main(){
  int k = int(a_who.x + 0.5);
  vec4 band = u_band[k];
  float vis = step(a_who.y + 0.5, u_own[k]);
  float r0 = mix(band.x, band.y, a_seed.x);
  float a = a_seed.y * 6.2831853 + u_t * ${SPIN.toFixed(3)} / pow(r0, 1.5) + u_fall * u_fall * (6.0 + 5.0 * a_seed.w);
  float r = r0 * (1.0 + 0.03 * sin(a * 2.0 + a_seed.w * 6.2831853)) * (1.0 - u_fall);
  float z = (a_seed.z - 0.5) * band.w * (1.0 - u_fall);
  vec2 p = vec2(cos(a), sin(a)) * r;
  vec2 px = u_c + u_S * vec2(p.x, p.y * u_tilt - z);
  float depth = p.y / max(r, 0.001);
  if (depth < 0.0 && length(px - u_c) < u_coreR * 0.96) vis = 0.0;
  float age = u_now - u_popT[k];
  float fresh = step(u_popFrom[k] - 0.5, a_who.y) * (1.0 - clamp(age / 0.55, 0.0, 1.0));
  float size = band.z * u_dpr * (0.86 + 0.20 * depth) * (1.0 + 1.5 * fresh * fresh);
  v_px = size;
  v_col = u_col[k];
  v_kind = a_who.x;
  v_hot = fresh;
  v_seed = a_seed.w;
  v_a = vis * u_alpha * (0.72 + 0.28 * depth) * (1.0 - u_fall * u_fall);
  v_light = normalize(u_c - px + vec2(0.0001));
  gl_PointSize = max(2.0, size * 2.6);
  vec2 clip = px / u_res * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}`;
  const FS_BODY = `
precision highp float;
varying vec3 v_col; varying vec2 v_light; varying float v_kind, v_a, v_px, v_hot, v_seed;
void main(){
  if (v_a <= 0.001) discard;
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float d = length(q);
  if (v_px < 4.6) {                                   // dust and pebbles: a soft point of light
    float a = pow(max(0.0, 1.0 - d), 2.2) * 0.9 + smoothstep(0.42, 0.18, d) * 0.9;
    gl_FragColor = vec4(mix(v_col, vec3(1.0), 0.25) * a * v_a * (1.0 + 2.0 * v_hot), 0.0);
    return;
  }
  float R = 1.0 / 2.6;
  float edge = 2.2 / (v_px * 2.6);
  float inside = smoothstep(R, R - edge, d);
  vec2 n2 = q / R; float nz = sqrt(max(0.0, 1.0 - dot(n2, n2)));
  vec3 L = normalize(vec3(v_light, 0.55));
  float diff = max(0.0, dot(vec3(n2, nz), L));
  vec3 base = v_col;
  float rim = pow(1.0 - nz, 2.5);
  vec3 lit;
  if (v_kind > 6.5) {                                 // a dwarf shines by itself
    lit = mix(vec3(1.0, 0.96, 0.82), base, 0.35 + 0.45 * (1.0 - nz)) * (0.85 + 0.35 * nz);
  } else if (v_kind > 5.5) {                          // a giant has bands
    float bands = 0.5 + 0.5 * sin(n2.y * 9.0 + v_seed * 20.0 + sin(n2.x * 3.0) * 0.6);
    base = mix(base * 0.62, base * 1.15 + 0.06, bands);
    lit = base * (0.07 + 1.10 * diff) + base * rim * 0.30;
  } else if (v_kind > 4.5) {                          // a planet has air
    lit = base * (0.06 + 1.05 * diff) + vec3(0.55, 0.85, 1.0) * rim * (0.25 + 0.6 * diff);
  } else {                                            // rock
    float rough = 0.86 + 0.28 * fract(sin(dot(floor(n2 * 3.0 + v_seed * 9.0), vec2(12.9898, 78.233))) * 43758.5453);
    lit = base * rough * (0.05 + 1.12 * diff);
  }
  float halo = pow(max(0.0, 1.0 - d), 2.4) * (v_kind > 6.5 ? 0.95 : 0.34);
  vec3 rgb = lit * inside + v_col * halo * (1.0 - inside * 0.6) + vec3(1.0) * v_hot * (0.9 * inside + halo);
  gl_FragColor = vec4(rgb * v_a, inside * v_a);
}`;

  const VS_IN = `
attribute vec4 a_seed;
attribute vec2 a_who;
uniform float u_in[8];
uniform vec3 u_col[8];
uniform vec4 u_band[8];
uniform vec2 u_res, u_c;
uniform float u_t, u_S, u_tilt, u_dpr, u_coreR, u_flow, u_alpha, u_fall;
uniform vec3 u_core;
varying vec3 v_col; varying float v_a;
void main(){
  int k = int(a_who.x + 0.5);
  vec4 band = u_band[k];
  float vis = step(a_who.y + 0.5, u_in[k]);
  float life = fract(a_seed.z + u_t * (0.035 + 0.05 * a_seed.w) * u_flow);
  float r0 = mix(band.x, band.y, a_seed.x);
  float fall = pow(life, 1.7);
  float r = mix(r0, 0.02, fall) * (1.0 - u_fall);
  // three arms that wind in toward the core, like water down a drain
  float arm = floor(a_seed.y * 3.0) * 2.0943951 + (fract(a_seed.y * 3.0) - 0.5) * 0.9;
  float a = arm + u_t * 0.05 + fall * (4.5 + 1.5 * a_seed.w) - r0 * 2.0;
  vec2 p = vec2(cos(a), sin(a)) * r;
  vec2 px = u_c + u_S * vec2(p.x, p.y * u_tilt - (a_seed.w - 0.5) * band.w * (1.0 - fall));
  if (p.y < 0.0 && length(px - u_c) < u_coreR * 0.96) vis = 0.0;
  v_col = mix(u_col[k], u_core, fall * fall);
  v_a = vis * u_alpha * sin(life * 3.14159) * (0.55 + 0.85 * fall);
  gl_PointSize = max(1.5, (1.7 + 2.6 * fall) * u_dpr);
  vec2 clip = px / u_res * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}`;
  const FS_IN = `
precision highp float;
varying vec3 v_col; varying float v_a;
void main(){
  float d = length(gl_PointCoord * 2.0 - 1.0);
  gl_FragColor = vec4(v_col * v_a * pow(max(0.0, 1.0 - d), 1.4), 0.0);
}`;

  // Loose points placed from the page side: sparks, the comet, stars of the sky, a galaxy, the universe.
  // a_pos is in scene units and goes through u_xf (centre, scale, turn) and u_squash; a_look.x picks the shape.
  const VS_PT = `
attribute vec2 a_pos;
attribute float a_size;
attribute vec4 a_col;
attribute vec2 a_look;
uniform vec2 u_res;
uniform vec4 u_xf;
uniform float u_squash, u_dpr, u_t, u_alpha, u_n, u_k;
varying vec4 v_col; varying float v_shape, v_px;
void main(){
  float cs = cos(u_xf.w), sn = sin(u_xf.w);
  vec2 p = vec2(a_pos.x * cs - a_pos.y * sn, a_pos.x * sn + a_pos.y * cs);
  vec2 px = u_xf.xy + u_xf.z * vec2(p.x, p.y * u_squash);
  float tw = a_look.y > 0.0 ? 0.72 + 0.28 * sin(u_t * (0.6 + a_look.y * 2.2) + a_look.y * 50.0) : 1.0;
  float vis = a_look.x < 3.5 ? 1.0 : step(a_look.y * 1.0e4, u_n);
  v_col = vec4(a_col.rgb, a_col.a * u_alpha * tw * vis);
  v_shape = a_look.x;
  v_px = a_size * u_dpr * u_k;
  gl_PointSize = max(1.5, v_px);
  vec2 clip = px / u_res * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}`;
  const FS_PT = `
precision highp float;
varying vec4 v_col; varying float v_shape, v_px;
void main(){
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float d = length(q);
  float a;
  if (v_shape > 0.5 && v_shape < 1.5) {               // a star with a glint
    float core = pow(max(0.0, 1.0 - d), 3.0);
    float cross = max(0.0, 1.0 - abs(q.x) * 14.0) * max(0.0, 1.0 - abs(q.y)) + max(0.0, 1.0 - abs(q.y) * 14.0) * max(0.0, 1.0 - abs(q.x));
    a = core * 1.3 + cross * 0.55 * pow(max(0.0, 1.0 - d), 1.2);
  } else if (v_shape > 1.5 && v_shape < 2.5) {        // a ring: something that can be lit
    a = smoothstep(0.16, 0.0, abs(d - 0.62)) * 0.9 + pow(max(0.0, 1.0 - d), 3.0) * 0.25;
  } else {                                            // a soft point
    a = pow(max(0.0, 1.0 - d), 1.9);
  }
  gl_FragColor = vec4(v_col.rgb * v_col.a * a, 0.0);
}`;
  const VS_LINE = `
attribute vec2 a_pos; attribute vec4 a_col;
uniform vec2 u_res; uniform vec4 u_xf; uniform float u_squash, u_alpha;
varying vec4 v_col;
void main(){
  vec2 px = u_xf.xy + u_xf.z * vec2(a_pos.x, a_pos.y * u_squash);
  v_col = vec4(a_col.rgb, a_col.a * u_alpha);
  vec2 clip = px / u_res * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
}`;
  const FS_LINE = "precision highp float; varying vec4 v_col; void main(){ gl_FragColor = vec4(v_col.rgb * v_col.a, 0.0); }";

  function lcg(seed) { let s = seed >>> 0; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); }

  function Scene(canvas, opts) {
    const o = opts || {};
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, powerPreference: "high-performance" })
      || canvas.getContext("webgl", { antialias: false, alpha: false });
    if (!gl) { this.dead = true; return; }
    const quality = o.quality || 2;                    // 2 the PC, 1 a phone
    const self = this;
    this.gl = gl;

    function prog(vs, fs) {
      const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + "\n" + src.split("\n").slice(0, 12).join("\n")); return s; };
      const p = gl.createProgram();
      gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      const u = {}, a = {};
      for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) { const info = gl.getActiveUniform(p, i); const name = info.name.replace(/\[0\]$/, ""); u[name] = gl.getUniformLocation(p, name); }
      for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES); i < n; i++) { const info = gl.getActiveAttrib(p, i); a[info.name] = gl.getAttribLocation(p, info.name); }
      return { p, u, a };
    }
    const P = {
      bg: prog(VS_QUAD, FS_BG(quality > 1 ? 5 : 3)),
      body: prog(VS_BODY, FS_BODY),
      inflow: prog(VS_IN, FS_IN),
      pt: prog(VS_PT, FS_PT),
      line: prog(VS_LINE, FS_LINE),
    };
    const buf = (data, usage) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, usage || gl.STATIC_DRAW); return b; };
    const quad = buf(new Float32Array([-1, -1, 3, -1, -1, 3]));

    // one vertex for every body there could ever be, and for every grain falling in
    function cloudBuffer(per, seed) {
      const rnd = lcg(seed), d = new Float32Array(NK * per * 6);
      for (let k = 0, i = 0; k < NK; k++) for (let n = 0; n < per; n++) { d[i++] = rnd(); d[i++] = rnd(); d[i++] = rnd(); d[i++] = rnd(); d[i++] = k; d[i++] = n; }
      return d;
    }
    const bodyData = cloudBuffer(MAX_BODY, 4242), inData = cloudBuffer(MAX_IN, 977);
    const bodyBuf = buf(bodyData), inBuf = buf(inData);
    const bandArr = new Float32Array(NK * 4), colArr = new Float32Array(NK * 3);
    KIND.forEach((k, i) => { bandArr.set([k.r[0], k.r[1], k.size, k.thick], i * 4); colArr.set(k.col, i * 3); });

    // points and lines that change: 7 floats a point (x, y, size, r, g, b, a) + 2 (shape, twinkle)
    const PT = 9;
    function Points(max) { this.max = max; this.data = new Float32Array(max * PT); this.n = 0; this.buf = buf(this.data, gl.DYNAMIC_DRAW); this.dirty = true; }
    Points.prototype.clear = function () { this.n = 0; this.dirty = true; };
    Points.prototype.add = function (x, y, size, c, a, shape, tw) {
      if (this.n >= this.max) return;
      const d = this.data, i = this.n++ * PT;
      d[i] = x; d[i + 1] = y; d[i + 2] = size; d[i + 3] = c[0]; d[i + 4] = c[1]; d[i + 5] = c[2]; d[i + 6] = a; d[i + 7] = shape || 0; d[i + 8] = tw || 0;
      this.dirty = true;
    };
    function drawPoints(pts, xf, squash, alpha, n, k) {
      if (!pts.n || alpha <= 0) return;
      const p = P.pt;
      gl.useProgram(p.p);
      gl.bindBuffer(gl.ARRAY_BUFFER, pts.buf);
      if (pts.dirty) { gl.bufferSubData(gl.ARRAY_BUFFER, 0, pts.data.subarray(0, pts.n * PT)); pts.dirty = false; }
      const st = PT * 4;
      gl.enableVertexAttribArray(p.a.a_pos); gl.vertexAttribPointer(p.a.a_pos, 2, gl.FLOAT, false, st, 0);
      gl.enableVertexAttribArray(p.a.a_size); gl.vertexAttribPointer(p.a.a_size, 1, gl.FLOAT, false, st, 8);
      gl.enableVertexAttribArray(p.a.a_col); gl.vertexAttribPointer(p.a.a_col, 4, gl.FLOAT, false, st, 12);
      gl.enableVertexAttribArray(p.a.a_look); gl.vertexAttribPointer(p.a.a_look, 2, gl.FLOAT, false, st, 28);
      gl.uniform2f(p.u.u_res, W, H); gl.uniform4f(p.u.u_xf, xf[0], xf[1], xf[2], xf[3]);
      gl.uniform1f(p.u.u_squash, squash); gl.uniform1f(p.u.u_dpr, dpr); gl.uniform1f(p.u.u_t, time); gl.uniform1f(p.u.u_alpha, alpha);
      gl.uniform1f(p.u.u_n, n === undefined ? 1e9 : n); gl.uniform1f(p.u.u_k, k || 1);
      gl.drawArrays(gl.POINTS, 0, pts.n);
    }
    const LN = 6;
    function Lines(max) { this.max = max; this.data = new Float32Array(max * 2 * LN); this.n = 0; this.buf = buf(this.data, gl.DYNAMIC_DRAW); this.dirty = true; }
    Lines.prototype.clear = function () { this.n = 0; this.dirty = true; };
    Lines.prototype.add = function (x0, y0, x1, y1, c, a) {
      if (this.n >= this.max) return;
      const d = this.data, i = this.n++ * 2 * LN;
      d.set([x0, y0, c[0], c[1], c[2], a, x1, y1, c[0], c[1], c[2], a], i);
      this.dirty = true;
    };
    function drawLines(ln, xf, squash, alpha) {
      if (!ln.n || alpha <= 0) return;
      const p = P.line;
      gl.useProgram(p.p);
      gl.bindBuffer(gl.ARRAY_BUFFER, ln.buf);
      if (ln.dirty) { gl.bufferSubData(gl.ARRAY_BUFFER, 0, ln.data.subarray(0, ln.n * 2 * LN)); ln.dirty = false; }
      gl.enableVertexAttribArray(p.a.a_pos); gl.vertexAttribPointer(p.a.a_pos, 2, gl.FLOAT, false, LN * 4, 0);
      gl.enableVertexAttribArray(p.a.a_col); gl.vertexAttribPointer(p.a.a_col, 4, gl.FLOAT, false, LN * 4, 8);
      gl.uniform2f(p.u.u_res, W, H); gl.uniform4f(p.u.u_xf, xf[0], xf[1], xf[2], xf[3]); gl.uniform1f(p.u.u_squash, squash); gl.uniform1f(p.u.u_alpha, alpha);
      gl.drawArrays(gl.LINES, 0, ln.n * 2);
    }

    // ── what the scene keeps between frames ──
    let W = 1, H = 1, dpr = 1, time = 0, now = 0;
    let cx = 0, cy = 0, S = 100;                       // where the core is, and how many px one unit of cloud is
    let reach = 0.37, reachWant = 0.37, zs = 1;        // how far out the picture reaches; it pulls back as kinds arrive
    const orbPulse = new Float32Array(NK), orbR = new Float32Array(NK), orbA = new Float32Array(NK);
    KIND.forEach((k, i) => { orbR[i] = (k.r[0] + k.r[1]) / 2; });
    const popFrom = new Float32Array(NK).fill(1e9), popT = new Float32Array(NK).fill(-99);
    const ownArr = new Float32Array(NK), inArr = new Float32Array(NK), inWant = new Float32Array(NK);
    const sparks = [];                                 // loose sparks: x, y, vx, vy, life, max, size, colour
    const rings = [];                                  // shockwaves: x, y, r, v, max, width, colour, alpha
    const fx = new Points(2400), cometPts = new Points(160);
    const skyStars = new Points(4200), skyNodes = new Points(64), skyLines = new Lines(64);
    const galaxy = new Points(quality > 1 ? 14000 : 5000), uniPts = new Points(5200), uniLines = new Lines(600);
    // the dust of the cloud: always there, thickest near the core; a flat disc of it and a rounder haze
    const dustDisc = new Points(quality > 1 ? 4200 : 1500), dustHalo = new Points(quality > 1 ? 1800 : 600);
    (function () {
      const r = lcg(808);
      for (const [set, flat] of [[dustDisc, true], [dustHalo, false]]) {
        for (let i = 0; i < set.max; i++) {
          const rad = 0.06 + 2.6 * Math.pow(r(), flat ? 1.7 : 1.4), a = r() * TAU, warm = r();
          const col = warm > 0.86 ? [1.0, 0.80, 0.62] : warm > 0.5 ? [0.74, 0.82, 1.0] : [0.58, 0.66, 0.92];
          set.add(Math.cos(a) * rad, Math.sin(a) * rad * (flat ? 1 : 0.8), 1.6 + r() * 2.6, col, (flat ? 0.34 : 0.22) * (0.4 + 0.6 * r()), 0, 0.05 + r() * 0.95);
        }
      }
    })();
    let flash = 0, pulse = 0, fall = 0, dim = 0, shake = 0;
    let coreR = 8, coreRShown = 8, heat = 0;
    let comet = null;                                  // { id, path, t0, ttl, type }
    let cometHead = null;
    const mixv = { cloud: 1, sky: 0, galaxy: 0, universe: 0 };
    let view = "cloud";
    let layout = { left: 0, right: 0, top: 0, bottom: 0 };
    let last = null;                                   // the last thing main.js said there was to draw
    let motion = 1;
    this.kinds = KIND; this.classCol = CLASS_COL;

    this.resize = function (w, h, ratio, pad) {
      dpr = Math.min(ratio || 1, quality > 1 ? 2 : 1.5);
      W = Math.max(1, Math.round(w * dpr)); H = Math.max(1, Math.round(h * dpr));
      if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
      layout = pad || layout;
      fit();
      gl.viewport(0, 0, W, H);
    };
    function fit() {
      const fw = W - (layout.left + layout.right) * dpr, fh = H - (layout.top + layout.bottom) * dpr;
      cx = layout.left * dpr + fw / 2; cy = layout.top * dpr + fh / 2;
      S = Math.min((fw * 0.94) / (2 * reach), (fh * 0.80) / (2 * reach * TILT));
      zs = Math.pow(S / (450 * dpr), 0.55);            // bodies grow when the picture is close, but less than the picture does
    }
    this.setMotion = function (m) { motion = m; };
    this.centre = function () { return { x: cx / dpr, y: cy / dpr, r: coreRShown / dpr, S: S / dpr }; };

    // Where body n of kind k is on the page right now (the same sum the shader does).
    this.bodyAt = function (k, n) {
      const i = (k * MAX_BODY + Math.min(n, MAX_BODY - 1)) * 6, b = KIND[k];
      const r0 = b.r[0] + (b.r[1] - b.r[0]) * bodyData[i];
      const a = bodyData[i + 1] * TAU + time * SPIN / Math.pow(r0, 1.5);
      const r = r0 * (1 + 0.03 * Math.sin(a * 2 + bodyData[i + 3] * TAU));
      const z = (bodyData[i + 2] - 0.5) * b.thick;
      return { x: (cx + S * Math.cos(a) * r) / dpr, y: (cy + S * (Math.sin(a) * r * TILT - z)) / dpr };
    };

    // ── things that happen ──
    this.pop = function (k, from, loud) { popFrom[k] = from; popT[k] = now; if (loud) orbPulse[k] = 1; };
    this.spark = function (x, y, col, n, speed, size, life) {
      if (motion < 0.5) n = Math.ceil(n / 3);
      for (let i = 0; i < n && sparks.length < 2200; i++) {
        const a = Math.random() * TAU, v = (0.25 + Math.random() * 0.75) * (speed || 90);
        sparks.push({ x: x * dpr, y: y * dpr, vx: Math.cos(a) * v * dpr, vy: Math.sin(a) * v * dpr, life: 0, max: (life || 0.7) * (0.6 + Math.random() * 0.8), size: (size || 3) * (0.6 + Math.random() * 0.8), col });
      }
    };
    // sparks that fall into the core from a ring around it: the hand pulling
    this.inward = function (col, n) {
      if (motion < 0.5) n = Math.ceil(n / 3);
      for (let i = 0; i < n && sparks.length < 2200; i++) {
        const a = Math.random() * TAU, r = coreRShown * (2.4 + Math.random() * 2.2);
        const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r * 0.8, t = 0.34 + Math.random() * 0.2;
        sparks.push({ x, y, vx: (cx - x) / t, vy: (cy - y) / t, life: 0, max: t, size: 2.6, col });
      }
    };
    this.ring = function (x, y, col, o2) {
      const q = o2 || {};
      if (rings.length >= 6) rings.shift();
      rings.push({ x: x * dpr, y: y * dpr, r: (q.from || 0) * dpr, v: (q.speed || 520) * dpr, max: (q.to || 420) * dpr, w: (q.width || 9) * dpr, col, a: q.alpha || 0.8 });
    };
    this.coreRing = function (col, o2) { this.ring(cx / dpr, cy / dpr, col, Object.assign({ from: coreRShown / dpr }, o2)); };
    this.pulse = function (v) { pulse = Math.max(pulse, v); };
    this.flash = function (v) { flash = Math.max(flash, motion < 0.5 ? v * 0.35 : v); };
    this.setComet = function (c) { comet = c ? { id: c.id, path: c.path, type: c.type, t0: now, ttl: c.ttl } : null; if (!c) cometHead = null; };
    this.cometAt = function () { return cometHead ? { x: cometHead.x / dpr, y: cometHead.y / dpr } : null; };
    this.setView = function (v) { view = v; };
    this.view = function () { return view; };
    // the cloud falls into the core; returns a number the caller can watch (0 → 1)
    this.fall = function (v) { fall = v; };
    this.dim = function (v) { dim = v; };

    // ── the sky: one star for every ignition in this galaxy, and the constellations ──
    let skyBuilt = -1, skyClasses = "";
    const CON_AT = { furnace: [-0.62, -0.30], deep: [0.62, -0.34], seed: [-0.70, 0.30], shepherd: [0.02, -0.58], net: [0.70, 0.28], lens: [0.0, 0.52] };
    const CON_SHAPE = {
      furnace: [[0, 0.12], [0.07, 0.0], [-0.02, -0.11], [0.12, -0.16]],
      deep: [[-0.08, -0.10], [0.0, 0.02], [0.11, 0.10]],
      seed: [[0.0, 0.12], [0.02, 0.0], [-0.08, -0.08], [0.09, -0.13]],
      shepherd: [[-0.14, 0.02], [-0.04, -0.05], [0.06, 0.0], [0.15, -0.07]],
      net: [[-0.09, 0.08], [0.02, 0.0], [0.11, -0.09]],
      lens: [[-0.11, 0.0], [0.0, -0.06], [0.11, 0.0]],
    };
    this.nodeAt = function (con, i) {                  // on the page, for the labels
      const c = CON_AT[con], p = CON_SHAPE[con][i], s = skyScale();
      return { x: (cx + (c[0] + p[0]) * s) / dpr, y: (cy + (c[1] + p[1]) * s) / dpr };
    };
    this.conAt = function (con) { const c = CON_AT[con], s = skyScale(); return { x: (cx + c[0] * s) / dpr, y: (cy + (c[1] + 0.17) * s) / dpr + 6 }; };
    function skyScale() { const fw = W - (layout.left + layout.right) * dpr, fh = H - (layout.top + layout.bottom) * dpr; return Math.min(fw * 0.52, fh * 0.70); }
    function buildSky(v) {
      const have = CLASSES.filter((c) => v.classes[c]);
      const key = have.join("");
      const n = Math.min(v.stars, skyStars.max);
      if (n === skyBuilt && key === skyClasses) return;
      // stars are added to, never moved: star i is always in the same place
      if (key !== skyClasses || n < skyBuilt) { skyStars.clear(); skyBuilt = 0; }
      const rnd = lcg(31337 + skyBuilt * 7919);
      for (let i = skyBuilt; i < n; i++) {
        const r = lcg(99991 + i * 7919);
        const x = (r() * 2 - 1) * 1.9, y = (r() * 2 - 1) * 1.05, pick = r(), tw = 0.05 + r() * 0.95;
        // most stars are small and cool; the hot ones are few
        const ci = have.length ? Math.min(have.length - 1, Math.floor(Math.pow(pick, 2.2) * have.length)) : 0;
        const col = CLASS_COL[have[ci] || "M"];
        skyStars.add(x, y, 13 + ci * 2.2 + r() * 8, col, 0.7 + 0.3 * r(), 1, tw);
      }
      void rnd;
      skyBuilt = n; skyClasses = key;
    }
    function buildNodes(v) {
      skyNodes.clear(); skyLines.clear();
      for (const con in CON_SHAPE) {
        const c = CON_AT[con], sh = CON_SHAPE[con], st = v.cons[con] || [];
        for (let i = 0; i < sh.length; i++) {
          const x = c[0] + sh[i][0], y = c[1] + sh[i][1], state = st[i] || 0;   // 0 shut · 1 can be lit · 2 lit
          if (i > 0) {
            const px0 = c[0] + sh[i - 1][0], py0 = c[1] + sh[i - 1][1], lit = state === 2;
            skyLines.add(px0, py0, x, y, lit ? [1.0, 0.86, 0.60] : [0.5, 0.56, 0.75], lit ? 0.55 : 0.16);
          }
          if (state === 2) skyNodes.add(x, y, 30, [1.0, 0.90, 0.68], 1.0, 1, 0.3 + i * 0.13);
          else if (state === 1) skyNodes.add(x, y, 22, [0.75, 0.86, 1.0], 0.75 + 0.25 * Math.sin(time * 3 + i), 2, 0);
          else skyNodes.add(x, y, 14, [0.45, 0.50, 0.68], 0.5, 2, 0);
        }
      }
    }

    // ── a galaxy: its stars along four arms; more of them the more stars you have lit ──
    (function buildGalaxy() {
      const r = lcg(2718);
      for (let i = 0; i < galaxy.max; i++) {
        const kind = r(), core = kind < 0.14, loose = kind > 0.80;
        const arm = Math.floor(r() * 2), t = Math.pow(r(), 0.75);
        const rad = core ? Math.pow(r(), 1.9) * 0.15 : 0.06 + t * 1.0;
        const spread = (loose ? 1.5 : 0.36) * (0.45 + 0.7 * t);
        const ang = core ? r() * TAU : arm * Math.PI + 3.5 * Math.pow(t, 0.8) + (r() + r() - 1) * spread;
        const x = Math.cos(ang) * rad, y = Math.sin(ang) * rad;
        const hot = r(), knot = !core && !loose && r() < 0.05;
        const col = core ? [1.0, 0.86, 0.62] : knot ? [1.0, 0.42, 0.62] : hot > 0.62 ? [0.66, 0.80, 1.0] : hot > 0.28 ? [0.92, 0.94, 1.0] : [1.0, 0.82, 0.62];
        // a_look.y carries the order in which stars appear (÷1e4), so the galaxy fills as you light them
        galaxy.add(x, y, core ? 2.8 : knot ? 6.5 : 1.8 + r() * 2.6, col, core ? 0.6 : knot ? 0.8 : (loose ? 0.3 : 0.5) + 0.4 * r(), 4, (i + 1) / 1e4);
      }
    })();
    // ── the universe: every galaxy you have made, strung along filaments ──
    let uniBuilt = -1;
    function buildUniverse(n) {
      n = Math.min(n, uniPts.max - 1);
      if (n === uniBuilt) return;
      uniPts.clear(); uniLines.clear();
      const r = lcg(1618);
      const knots = [];
      for (let i = 0; i < 46; i++) knots.push([(r() * 2 - 1) * 1.7, (r() * 2 - 1) * 0.95]);
      const threads = [];
      for (const k of knots) uniPts.add(k[0], k[1], 60, [0.42, 0.36, 0.90], 0.16, 0, 0);     // where threads meet, a glow
      for (let i = 0; i < knots.length; i++) {
        const near = knots.map((k, j) => [Math.hypot(k[0] - knots[i][0], k[1] - knots[i][1]), j]).filter((x) => x[1] !== i).sort((a, b) => a[0] - b[0]).slice(0, 2);
        for (const [, j] of near) if (j > i) { threads.push([i, j]); uniLines.add(knots[i][0], knots[i][1], knots[j][0], knots[j][1], [0.46, 0.44, 0.85], 0.22); }
      }
      for (let i = 0; i < n; i++) {
        const th = threads[Math.floor(r() * threads.length)], t = r(), a = knots[th[0]], b = knots[th[1]];
        const pull = Math.pow(Math.abs(t - 0.5) * 2, 0.6), spread = 0.05 * (1 - pull) + 0.012;
        const x = a[0] + (b[0] - a[0]) * t + (r() - 0.5) * spread, y = a[1] + (b[1] - a[1]) * t + (r() - 0.5) * spread;
        const warm = r();
        uniPts.add(x, y, 9 + r() * 10 + (i === n - 1 ? 16 : 0), warm > 0.66 ? [0.70, 0.78, 1.0] : warm > 0.33 ? [0.86, 0.74, 1.0] : [1.0, 0.82, 0.70], 0.65 + 0.35 * r(), i % 3 === 0 ? 1 : 0, 0.1 + r() * 0.9);
      }
      uniBuilt = n;
    }

    // ── one frame ──
    this.frame = function (dt, v) {
      if (self.dead) return;
      last = v || last;
      if (!last) return;
      v = last;
      dt = Math.min(dt, 0.1);
      now += dt;
      time += dt * (1 + 1.6 * (v.surge ? 1 : 0)) * (motion < 0.5 ? 0.5 : 1);
      reachWant = v.reach || 0.37;
      if (Math.abs(reachWant - reach) > 0.0005) { reach += (reachWant - reach) * Math.min(1, dt * 1.6); fit(); }
      for (let k = 0; k < NK; k++) { orbPulse[k] *= Math.exp(-dt * 3.5); orbA[k] = v.own[k] > 0 ? 0.02 + orbPulse[k] * 0.5 : 0; }
      // the four views fade into one another
      for (const k in mixv) { const want = k === view ? 1 : 0; mixv[k] += (want - mixv[k]) * Math.min(1, dt * 5.5); if (Math.abs(want - mixv[k]) < 0.004) mixv[k] = want; }
      // the core grows with what the cloud has gathered, and heats up toward ignition
      const want = S * (0.016 + 0.094 * Math.pow(Math.max(0, Math.min(1.3, v.grown)), 1.3));
      coreR += (want - coreR) * Math.min(1, dt * 3);
      heat += (v.heat - heat) * Math.min(1, dt * 2);
      pulse *= Math.exp(-dt * 7); flash *= Math.exp(-dt * 3.2);
      coreRShown = coreR * (1 + 0.10 * pulse) * (1 - fall * 0.35);
      for (let k = 0; k < NK; k++) {
        ownArr[k] = Math.min(MAX_BODY, v.own[k]);
        inWant[k] = Math.min(MAX_IN * (quality > 1 ? 1 : 0.4), v.flow[k]);
        inArr[k] += (inWant[k] - inArr[k]) * Math.min(1, dt * 2.5);
      }

      const m = mixv.cloud;
      const zoom = 0.25 + 0.75 * m;                    // leaving the cloud, it shrinks toward a point
      const Sv = S * zoom, coreV = coreRShown * zoom;
      const sx = shake > 0.01 && motion >= 0.5 ? (Math.random() - 0.5) * shake * dpr : 0, sy = shake > 0.01 && motion >= 0.5 ? (Math.random() - 0.5) * shake * dpr : 0;
      shake *= Math.exp(-dt * 6);
      const ccx = cx + sx, ccy = cy + sy;

      // background: space, the cloud, the core
      const b = P.bg;
      gl.disable(gl.BLEND);
      gl.useProgram(b.p);
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.enableVertexAttribArray(b.a.a_p); gl.vertexAttribPointer(b.a.a_p, 2, gl.FLOAT, false, 0, 0);
      gl.uniform2f(b.u.u_res, W, H); gl.uniform2f(b.u.u_c, ccx, ccy);
      gl.uniform1f(b.u.u_t, time); gl.uniform1f(b.u.u_S, S); gl.uniform1f(b.u.u_tilt, TILT); gl.uniform1f(b.u.u_dpr, dpr);
      gl.uniform1f(b.u.u_coreR, m > 0.02 ? coreV * m : 0); gl.uniform1f(b.u.u_heat, Math.min(1, heat + fall * 0.6));
      gl.uniform1f(b.u.u_ign, v.ign); gl.uniform1f(b.u.u_ignShow, v.ignShow * m * m * (1 - fall));
      gl.uniform1f(b.u.u_surge, v.surge ? 1 : 0); gl.uniform1f(b.u.u_flash, flash); gl.uniform1f(b.u.u_pulse, pulse + fall * 1.5);
      gl.uniform1f(b.u.u_dust, Math.min(1, v.own[0] / 60) * m * (1 - fall)); gl.uniform1f(b.u.u_dim, dim);
      gl.uniform1f(b.u.u_cloud, m * (1 - fall * 0.7)); gl.uniform1f(b.u.u_web, mixv.universe);
      gl.uniform1fv(b.u.u_orbR, orbR); gl.uniform1fv(b.u.u_orbA, orbA); gl.uniform3fv(b.u.u_kcol, colArr);
      gl.uniform1f(b.u.u_hole, mixv.galaxy > 0.02 ? (10 + 5 * (v.hole || 0)) * dpr * mixv.galaxy : 0);
      const tint = v.tint;
      gl.uniform3fv(b.u.u_tintA, tint[0]); gl.uniform3fv(b.u.u_tintB, tint[1]); gl.uniform3fv(b.u.u_tintC, tint[2]);
      gl.uniform3fv(b.u.u_haze, KIND[0].col); gl.uniform1f(b.u.u_tintAmt, v.tintAmt);
      const ra = new Float32Array(24), rc = new Float32Array(24);
      for (let i = rings.length - 1; i >= 0; i--) {
        const r = rings[i]; r.r += r.v * dt; r.v *= Math.exp(-dt * 1.4);
        if (r.r >= r.max) { rings.splice(i, 1); continue; }
      }
      rings.forEach((r, i) => { const f = 1 - r.r / r.max; ra.set([r.x, r.y, r.r, r.w * (0.6 + 1.2 * (1 - f))], i * 4); rc.set([r.col[0], r.col[1], r.col[2], r.a * f * f], i * 4); });
      gl.uniform4fv(b.u.u_ring, ra); gl.uniform4fv(b.u.u_ringCol, rc);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

      // the wider views, behind the cloud
      if (mixv.universe > 0.01) {
        buildUniverse(v.galaxiesEver || 0);
        const s = skyScale() * (0.8 + 0.2 * mixv.universe);
        drawLines(uniLines, [cx, cy, s, 0], 1, mixv.universe);
        drawPoints(uniPts, [cx, cy, s, 0], 1, mixv.universe);
      }
      if (mixv.galaxy > 0.01) {
        const s = skyScale() * (0.7 + 0.3 * mixv.galaxy) * 1.05;
        const shown = Math.max(260, Math.min(galaxy.max, 260 + (v.stars || 0) * 3));
        drawPoints(galaxy, [cx, cy, s, time * 0.02], 0.56, mixv.galaxy, shown);
      }
      if (mixv.sky > 0.01) {
        buildSky(v); buildNodes(v);
        const s = skyScale() * (0.75 + 0.25 * mixv.sky);
        // many stars are drawn smaller, so a full sky is a field and not a glare
        drawPoints(skyStars, [cx, cy, s, 0], 1, mixv.sky, undefined, Math.max(0.42, Math.min(1.25, 1.5 - Math.log10(1 + skyStars.n) / 3.2)));
        drawLines(skyLines, [cx, cy, s, 0], 1, mixv.sky);
        drawPoints(skyNodes, [cx, cy, s, 0], 1, mixv.sky);
      }

      if (m > 0.01) {
        drawPoints(dustHalo, [ccx, ccy, Sv, -time * 0.006], 0.86, m * (1 - fall));
        drawPoints(dustDisc, [ccx, ccy, Sv, time * 0.012], TILT, m * (1 - fall));
        // grains falling in
        const f = P.inflow;
        gl.useProgram(f.p);
        gl.bindBuffer(gl.ARRAY_BUFFER, inBuf);
        gl.enableVertexAttribArray(f.a.a_seed); gl.vertexAttribPointer(f.a.a_seed, 4, gl.FLOAT, false, 24, 0);
        gl.enableVertexAttribArray(f.a.a_who); gl.vertexAttribPointer(f.a.a_who, 2, gl.FLOAT, false, 24, 16);
        gl.uniform1fv(f.u.u_in, inArr); gl.uniform3fv(f.u.u_col, colArr); gl.uniform4fv(f.u.u_band, bandArr);
        gl.uniform2f(f.u.u_res, W, H); gl.uniform2f(f.u.u_c, ccx, ccy);
        gl.uniform1f(f.u.u_t, time); gl.uniform1f(f.u.u_S, Sv); gl.uniform1f(f.u.u_tilt, TILT); gl.uniform1f(f.u.u_dpr, dpr * Math.max(1, zs * 0.8));
        gl.uniform1f(f.u.u_coreR, coreV); gl.uniform1f(f.u.u_flow, 1); gl.uniform1f(f.u.u_alpha, m * 0.85); gl.uniform1f(f.u.u_fall, fall);
        const h = Math.min(1, heat);
        gl.uniform3f(f.u.u_core, 1.0, 0.45 + 0.5 * h, 0.15 + 0.6 * h);
        gl.drawArrays(gl.POINTS, 0, NK * MAX_IN);
        // every body you own
        const p = P.body;
        gl.useProgram(p.p);
        gl.bindBuffer(gl.ARRAY_BUFFER, bodyBuf);
        gl.enableVertexAttribArray(p.a.a_seed); gl.vertexAttribPointer(p.a.a_seed, 4, gl.FLOAT, false, 24, 0);
        gl.enableVertexAttribArray(p.a.a_who); gl.vertexAttribPointer(p.a.a_who, 2, gl.FLOAT, false, 24, 16);
        gl.uniform1fv(p.u.u_own, ownArr); gl.uniform1fv(p.u.u_popFrom, popFrom); gl.uniform1fv(p.u.u_popT, popT);
        gl.uniform3fv(p.u.u_col, colArr); gl.uniform4fv(p.u.u_band, bandArr);
        gl.uniform2f(p.u.u_res, W, H); gl.uniform2f(p.u.u_c, ccx, ccy);
        gl.uniform1f(p.u.u_t, time); gl.uniform1f(p.u.u_S, Sv); gl.uniform1f(p.u.u_tilt, TILT); gl.uniform1f(p.u.u_dpr, dpr * zoom * zs);
        gl.uniform1f(p.u.u_coreR, coreV); gl.uniform1f(p.u.u_fall, fall); gl.uniform1f(p.u.u_now, now); gl.uniform1f(p.u.u_alpha, m);
        gl.drawArrays(gl.POINTS, 0, NK * MAX_BODY);
      }

      // the comet: a head that crosses the scene on an arc, and the tail behind it
      cometPts.clear(); cometHead = null;
      if (comet && view === "cloud") {
        const age = now - comet.t0, u = Math.min(1, age / comet.ttl);
        const fw = W - (layout.left + layout.right) * dpr, fh = H - (layout.top + layout.bottom) * dpr;
        const dir = comet.path < 0.5 ? 1 : -1, bend = (comet.path * 2 % 1) - 0.5;
        const at = (q) => {
          const x = cx + dir * (q - 0.5) * fw * 0.92;
          const y = cy - fh * (0.30 + 0.10 * bend) + Math.pow((q - 0.5) * 2, 2) * fh * 0.16 + bend * fh * 0.12 * (q - 0.5);
          return [x, y];
        };
        const head = at(u);
        cometHead = { x: head[0], y: head[1] };
        const col = comet.type === "surge" ? [0.55, 0.85, 1.0] : [1.0, 0.86, 0.52];
        const fade = Math.min(1, age * 3) * Math.min(1, (comet.ttl - age) * 2.5);
        for (let i = 0; i < 70; i++) {
          const q = u - i * 0.0042; if (q < 0) break;
          const p2 = at(q), w = 1 - i / 70;
          cometPts.add(p2[0] + (Math.sin(i * 12.9898 + time * 9) * 3 * (1 - w)) * dpr, p2[1] + (Math.cos(i * 78.233 + time * 7) * 3 * (1 - w)) * dpr, (i ? 5 + 13 * w : 34) , i ? col : [1, 1, 1], (i ? 0.55 * w * w : 1) * fade, i ? 0 : 1, 0);
        }
        cometPts.add(head[0], head[1], 46 + 8 * Math.sin(now * 6), col, 0.5 * fade, 2, 0);
        drawPoints(cometPts, [0, 0, 1, 0], 1, 1);
      }

      // sparks
      fx.clear();
      for (let i = sparks.length - 1; i >= 0; i--) {
        const s = sparks[i]; s.life += dt;
        if (s.life >= s.max) { sparks.splice(i, 1); continue; }
        s.x += s.vx * dt; s.y += s.vy * dt; s.vx *= Math.exp(-dt * 2.2); s.vy *= Math.exp(-dt * 2.2);
        const f = 1 - s.life / s.max;
        fx.add(s.x, s.y, s.size * (0.6 + f), s.col, f * f, 0, 0);
      }
      drawPoints(fx, [0, 0, 1, 0], 1, 1);
    };
    this.shake = function (v) { shake = Math.max(shake, v); };
    this.busy = function () { return sparks.length + rings.length; };
  }

  root.AccreteScene = Scene;
  root.AccreteSceneKinds = KIND;
  root.AccreteSceneClass = CLASS_COL;
})(typeof self !== "undefined" ? self : this);
