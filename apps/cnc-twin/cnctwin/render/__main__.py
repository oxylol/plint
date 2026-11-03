"""``python -m cnctwin.render RUN_DIR [--no-images] [--no-video] [--speed S] [--fps F] [--layer L]``
``python -m cnctwin.render RUN_DIR --at T --out picture.png [--camera follow|part|setup|overview] [--layer L]``"""

from __future__ import annotations

import argparse
import json
import sys

from .compose import CAMERA_MODES, FrameOptions
from .layers import LAYERS
from .pipeline import render_image, render_run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cnctwin.render", description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--speed", type=float, default=None, help="simulated seconds per video second")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--layer", choices=LAYERS, default=None)
    parser.add_argument("--at", type=float, default=None, help="one picture at this simulated time (s)")
    parser.add_argument("--out", default=None, help="PNG file for --at")
    parser.add_argument("--camera", choices=CAMERA_MODES, default="follow")
    parser.add_argument("--projection", choices=("perspective", "orthographic"), default="perspective")
    args = parser.parse_args(argv)
    if args.at is not None:
        if not args.out:
            parser.error("--at needs --out")
        result = render_image(args.run_dir, args.at, args.out, FrameOptions(camera=args.camera, layer=args.layer,
                                                                           projection=args.projection))
    else:
        result = render_run(args.run_dir, images=not args.no_images, video=not args.no_video, speed=args.speed,
                            fps=args.fps, layer=args.layer)
    json.dump(result, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
