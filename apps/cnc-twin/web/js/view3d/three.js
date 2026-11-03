/* The one import path of three.js for the 3D view: the vendored ES module build (web/vendor/three,
 * three.js r170 from the npm registry tarball, MIT licence) and the example modules the view uses,
 * whose bare 'three' imports were rewritten to the same build file, so one module instance exists. */

export * from '../../vendor/three/build/three.module.min.js';
export { OrbitControls } from '../../vendor/three/examples/jsm/controls/OrbitControls.js';
export { GLTFLoader } from '../../vendor/three/examples/jsm/loaders/GLTFLoader.js';
export { mergeGeometries, toCreasedNormals, mergeVertices } from '../../vendor/three/examples/jsm/utils/BufferGeometryUtils.js';
