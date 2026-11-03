"""The synthetic run directory (tests/fixtures/make_synthetic_run.py, extended for production):
every file of docs/architecture.md 16, the same bytes for the same seed."""

import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

from make_synthetic_run import make_synthetic_run, result_set_digest, sha256_file  # noqa: E402

ARCHITECTURE_16 = ("run.json", "status.json", "findings.json", "trace.jsonl", "channels.json", "timeline.json",
                   "series/index.jsonl", "scene.json", "frames.bin", "keyframes/bricks.idx", "keyframes/index.jsonl",
                   "part_OP10.glb", "part_OP10.stl", "part_OP10.3mf", "part_OP10_unload.stl", "inspection.json",
                   "surface.json", "tools.json", "production.json", "report.html", "report.pdf", "setup_sheet.html",
                   "setup_sheet.pdf", "video.mp4", "audio.wav")
PATTERNS = ("series/fast/part-*.parquet", "series/tick/part-*.parquet", "series/macro/part-*.parquet",
            "scene/*.glb", "keyframes/kf_*.glb", "keyframes/kf_*.npz", "keyframes/attr/*", "chips/kf_*.npz",
            "sld/*.json", "images/*.png", "checkpoints/ck_*.npz", "checkpoints/ck_*.json")


def test_every_file_and_determinism(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    make_synthetic_run(first, seed=11)
    make_synthetic_run(second, seed=11)
    for name in ARCHITECTURE_16:
        assert (first / name).is_file(), name
    for pattern in PATTERNS:
        assert list(first.glob(pattern)), pattern
    assert result_set_digest(first) == result_set_digest(second)
    for name in ("timeline.json", "production.json", "video.mp4", "report.pdf", "audio.wav"):
        assert sha256_file(first / name) == sha256_file(second / name), name
