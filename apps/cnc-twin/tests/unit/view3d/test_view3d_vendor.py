"""The vendored three.js (web/vendor/three): r170 from the npm registry tarball three-0.170.0.tgz (sha512
FQK+LEpY...BhuQ==), MIT licence, the ES module build and the example modules the view imports, whose bare
'three' imports point at the vendored build (one module instance, no import map, no CDN)."""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "web" / "vendor" / "three"
# sha256 of the files as they are in the tarball (the examples before their import line was rewritten)
BUILD_SHA256 = "08fd7545d13d2c7fb65ab691530a802dafefd638596501854f267d0fb13c39e7"
LICENSE_SHA256 = "4c40a1ef62450b857c3b2aaf294936304cd552d965fbcd9d32d4c5bcf4ba4454"
EXAMPLES_SHA256 = {"controls/OrbitControls.js": "80efaadea4f8a636a65fb0bd08bfef62f3d93a0bb94e2e7500f23176c5c07f4e",
                   "loaders/GLTFLoader.js": "45139faddd5aaf48ed2d62203d976e5cbd703db1a592de40527f9f6cf58abd44",
                   "utils/BufferGeometryUtils.js": "c25b7930e570e9ec56173cd3b866ec8d2e10016630db3937efb439daf1cedbf6"}
REWRITE = "from '../../../build/three.module.min.js';"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_build_and_licence_are_the_release_files():
    build = (VENDOR / "build" / "three.module.min.js").read_bytes()
    assert sha256(build) == BUILD_SHA256
    assert b'const t="170"' in build[:400]
    licence = (VENDOR / "LICENSE").read_bytes()
    assert sha256(licence) == LICENSE_SHA256 and b"MIT License" in licence


def test_examples_differ_only_in_their_three_import():
    for name, digest in EXAMPLES_SHA256.items():
        text = (VENDOR / "examples" / "jsm" / name).read_text(encoding="utf-8")
        assert "from 'three'" not in text
        assert text.count(REWRITE) == 1, name
        assert sha256(text.replace(REWRITE, "from 'three';").encode("utf-8")) == digest, name


def test_the_view_imports_three_only_through_the_vendored_files():
    folder = ROOT / "web" / "js" / "view3d"
    seen = {}
    for path in sorted(folder.glob("*.js")):
        code = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
        code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
        for target in re.findall(r"from\s+'([^']+)'", code):
            assert target.startswith("."), f"{path.name} imports {target} (only relative modules are served)"
            assert (path.parent / target).resolve().is_file(), f"{path.name} imports {target}, which does not exist"
            seen.setdefault(target, []).append(path.name)
    vendored = {t for t in seen if "vendor/three" in t}
    assert vendored == {"../../vendor/three/build/three.module.min.js", "../../vendor/three/examples/jsm/controls/OrbitControls.js",
                        "../../vendor/three/examples/jsm/loaders/GLTFLoader.js",
                        "../../vendor/three/examples/jsm/utils/BufferGeometryUtils.js"}
    assert all(name == "three.js" for t in vendored for name in seen[t]), "three.js is the only door to the vendored files"
    assert len(seen.get("./three.js", [])) >= 8
