# three.js r170, vendored

Source: the npm registry tarball https://registry.npmjs.org/three/-/three-0.170.0.tgz
(sha512 `FQK+LEpYc0fBD+J8g6oSEyyNzjp+Q7Ks1C568WWaoMRLW+TkNNWmenWeGgJjV105Gd+p/2ql1ZcjYvNiPZBhuQ==`, sha1
`6087f97aab79e9e9312f9c89fcef6808642dfbb7`, both checked against the registry's record on download, 2026-11-03).
Licence: MIT (`LICENSE`, the package's own).

| file | from the package | changed |
|---|---|---|
| `build/three.module.min.js` | `build/three.module.min.js` (the ES module build; self-contained in r170) | no |
| `examples/jsm/controls/OrbitControls.js` | the same path | `from 'three';` -> `from '../../../build/three.module.min.js';` |
| `examples/jsm/loaders/GLTFLoader.js` | the same path | the same one import line |
| `examples/jsm/utils/BufferGeometryUtils.js` | the same path | the same one import line |
| `LICENSE` | `LICENSE` | no |

The import rewrite lets the browser load the example modules without an import map (the UI has no build step),
and makes every module share one three.js instance. tests/unit/view3d/test_view3d_vendor.py checks the hashes of
the release files and that the rewrite is the only change. Used by web/js/view3d (docs/ui-3d.md) through
web/js/view3d/three.js only; nothing is fetched from a CDN.
