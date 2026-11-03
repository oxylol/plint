"""``python -m cnctwin.sound RUN_DIR`` writes audio.wav and audio.json;
``python -m cnctwin.sound RUN_DIR --t0 1.0 --t1 2.0 --out window.wav`` writes one window."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import dsp as D
from . import params as P
from .inputs import from_run
from .synth import LISTENERS, SOURCES, render_audio, render_window


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cnctwin.sound", description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--t0", type=float, default=None, help="window start (simulated s)")
    parser.add_argument("--t1", type=float, default=None, help="window end (simulated s)")
    parser.add_argument("--out", default=None, help="WAV file (default RUN_DIR/audio.wav)")
    parser.add_argument("--listener", choices=LISTENERS, default=None)
    parser.add_argument("--sources", default=None, help=f"comma-separated subset of {','.join(SOURCES)}")
    args = parser.parse_args(argv)
    sources = args.sources.split(",") if args.sources else None
    if args.t0 is None and args.t1 is None:
        result = render_audio(args.run_dir, args.out, sources, args.listener)
    else:
        if not args.out:
            parser.error("a window needs --out")
        inputs = from_run(args.run_dir)
        window = render_window(inputs, args.t0 or 0.0, args.t1 if args.t1 is not None else float("inf"), sources,
                               args.listener)
        full_scale = P.number(P.SOUND, "full_scale_Pa")
        # the run's seed: the window's samples are the same as those of the run's audio.wav
        samples, clipped = D.quantise(window.pressure_Pa, window.n0, inputs.seed, full_scale, bool(P.sound()["dither"]))
        writer = D.WavWriter(args.out, int(round(window.rate_Hz)), f"cnctwin sound window from sample {window.n0}")
        writer.write(samples)
        writer.close()
        result = {"file": args.out, "n0": window.n0, "samples": int(len(samples)), "clipped_samples": clipped,
                  "rms_Pa": float(np.sqrt(np.mean(window.pressure_Pa ** 2))) if len(samples) else 0.0,
                  "notes": window.notes}
    json.dump(result, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
