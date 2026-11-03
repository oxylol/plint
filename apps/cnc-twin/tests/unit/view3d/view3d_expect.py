"""Expected values for the node unit tests of web/js/view3d (view3d_unit.mjs), computed from a synthetic
run directory with the run infrastructure's own readers (cnctwin.core.keyframes, numpy) and
cnctwin.geom.transform, so the browser code is checked against the Python side of every format."""

from __future__ import annotations

import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for entry in (str(ROOT), str(ROOT / "tests" / "fixtures")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _npz_doc(path: Path) -> dict:
    out = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            a = archive[name]
            if a.dtype.kind in "US":
                data = [str(x) for x in a.reshape(-1)]
            else:
                data = [None if (isinstance(x, float) and not math.isfinite(x)) else x
                        for x in a.astype(np.float64).reshape(-1).tolist()]
            out[name] = {"dtype": a.dtype.str, "shape": list(a.shape), "data": data}
    return out


def write_extra_npz(path: Path) -> None:
    """An .npz with the dtypes the reader must handle: float16, bool, unicode, int64, Fortran order."""
    arrays = {"h": np.array([0.5, -2.0, 65504.0, 1e-5], np.float16), "b": np.array([True, False, True]),
              "u": np.array(["n1", "jet two", ""], dtype="<U8"), "i8": np.array([2**40, -3], np.int64),
              "f": np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4)),
              "big": np.array([1.5, 2.5], dtype=">f8")}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED if name == "u" else zipfile.ZIP_DEFLATED
            archive.writestr(info, member.getvalue())
    path.write_bytes(buffer.getvalue())


def expectations(run_dir: Path, out_dir: Path) -> dict:
    from cnctwin.core.keyframes import KeyframeReader
    from cnctwin.core.rundir import read_frames
    from cnctwin.geom import transform as tf

    run_dir = Path(run_dir)
    scene = json.loads((run_dir / "scene.json").read_text())
    header = scene["frames"]
    records, info = read_frames(run_dir, None, None)
    picks = [0, 1, 500, len(records) // 2, len(records) - 1]
    frames = {"count": int(len(records)), "picks": []}
    for i in picks:
        r = records[i]
        frames["picks"].append({"index": int(i), "tick": int(r["tick"]), "axis": [float(v) for v in np.ravel(r["axis"])],
                                "spindle_unwrapped": float(info["spindle_unwrapped_rad"][i].ravel()[0]),
                                "tip": [float(v) for v in np.ravel(r["tip_S_m"])[:3]],
                                "axis_T": [float(v) for v in np.ravel(r["axis_T_S"])[:3]]})
    reader = KeyframeReader(run_dir)
    # the API's rule for /bricks/{b}?t= (cnctwin/api/routes_runs.py): the newest entry whose keyframe time_s is
    # <= t + 1e-12; KeyframeReader rounds t to base ticks instead, which differs only within half a tick
    keyframe_time = {int(e["k"]): float(e["time_s"]) for e in reader.entries()}
    times = [0.0, 0.25, 0.5, 0.5 + 1e-13, 1.0 - 1e-9, 1.0, 1.2, 1.5, 2.3]
    versions = {}
    for t in times:
        chosen = {}
        for key, entries in sorted(reader.index().items()):
            earlier = [e for e in entries if keyframe_time.get(int(e[0]), math.inf) <= t + 1e-12]
            chosen[key] = int(earlier[-1][0]) if earlier else None
        versions[repr(t)] = chosen
    bricks = {}
    for key in sorted(reader.index()):
        v, f = reader.brick(key, time_s=2.3)
        bricks[key] = {"n": int(len(v)), "m": int(len(f)), "sum": v.sum(axis=0).tolist(),
                       "first": v[0].tolist(), "faces0": f[0].tolist()}
    poses = []
    for spec in ({"translate_mm": [10, -20, 5], "rotate_deg": [30, -45, 60], "order": "xyz"},
                 {"translate_m": [0.1, 0.2, 0.3], "rotate_rad": [0.3, 0.2, -1.1], "order": "zyx"},
                 {"translate_mm": [0, 0, 0], "rotate_deg": [0, 90, 0], "order": "yzx"}):
        poses.append({"spec": spec, "matrix": tf.pose_from_dict(spec).tolist()})
    perpendiculars = []
    for z in ([0, 0, 1], [1, 0, 0], [0.3, -0.5, 0.8], [0, 1, 0], [-1, 1, 0]):
        perpendiculars.append({"z": z, "x": tf.perpendicular(z).tolist()})
    extra = out_dir / "extra.npz"
    write_extra_npz(extra)
    fields = _npz_doc(run_dir / "keyframes" / "kf_2.npz")
    attr = _npz_doc(run_dir / "keyframes" / "attr" / "brick_0.npz")
    return {"run_dir": str(run_dir), "extra_npz": str(extra), "frames_header": header, "frames": frames,
            "versions": versions, "bricks": bricks, "poses": poses, "perpendiculars": perpendiculars,
            "fields_kf2": fields, "attr_brick0": attr, "extra": _npz_doc(extra),
            "stock_size_m": scene["stock"]["size_m"]}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    run_dir, out = Path(argv[0]), Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(expectations(run_dir, out.parent), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
