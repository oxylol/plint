#!/usr/bin/env bash
# Plint · judge the engine as it is built in target/release: the real speakers (TSP) against an
# earlier run, then Whisper against the round's baseline. Results in the rig's hill/<name>.
#   tools/plint/judge.sh <name> [--tuning t.json] [--against <run>] [--jobs n]
#   (about 15 minutes on Plint's 4 cores with the default 3 jobs)
# --tuning t.json  judge that tuning instead of the default: exported to both judges as
#                  CHIMERA_TUNING (render.rs reads it) and kept in the run as hill/<name>/tuning.json
# --against <run>  compare the real-speaker judge and the terms table with hill/<run> (default: best),
#                  and Whisper with hill/<run>/speech/wer.json when that file exists (without the
#                  flag, and when it does not exist: hill/baseline/speech/wer.json)
# --jobs n         threads for the real-speaker judge (default 3). Whisper's script has no such
#                  option: it renders and transcribes on every core
# The run folder is wiped first, so <name> may not be ref442, baseline, best or the --against run.
# JUDGE_DRY=1 prints what would run and touches nothing.
# Keep a change with:  rm -rf $HILL/best && cp -a $HILL/<name> $HILL/best
set -euo pipefail
from="$PWD"
cd "$(dirname "$0")/../.."
HILL="${HILL:-/mnt/storage/plint/chimera/hill}"
usage="usage: tools/plint/judge.sh <name> [--tuning t.json] [--against <run>] [--jobs n]"
die() { echo "judge.sh: $*" >&2; exit 2; }
name="" tuning="" against="" jobs=3
while [ $# -gt 0 ]; do
  case "$1" in
    --tuning|--against|--jobs)
      [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value
$usage"
      case "$1" in --tuning) tuning=$2 ;; --against) against=$2 ;; --jobs) jobs=$2 ;; esac
      shift 2 ;;
    -*) die "unknown option $1
$usage" ;;
    *) [ -z "$name" ] || die "one name only (got $name and $1)
$usage"
      name=$1; shift ;;
  esac
done
[ -n "$name" ] || die "name of this try
$usage"
ref="${against:-best}"
case "$name" in */*|.|..) die "<name> is a folder name under $HILL, not a path: $name" ;; esac
case "$ref" in */*|.|..) die "--against is a run's name under $HILL, not a path: $ref" ;; esac
case "$name" in ref442|baseline|best) die "refusing to wipe the reference run $name: pick another name" ;; esac
[ "$name" != "$ref" ] || die "refusing to wipe $name, the run it is compared against"
case "$jobs" in ''|*[!0-9]*|0) die "--jobs needs a whole number above 0, got $jobs" ;; esac
[ -f "$HILL/$ref/tsp/score.json" ] || die "no run to compare against: $HILL/$ref/tsp/score.json"
if [ -n "$tuning" ]; then
  tuning="$(cd "$from" && realpath -m -- "$tuning")"   # as the caller wrote it, from where they stood
  [ -f "$tuning" ] || die "--tuning: no such file: $tuning"
  export CHIMERA_TUNING="$tuning"
else
  # a tuning only by the flag, so a variable left in the shell cannot judge something unrecorded
  unset CHIMERA_TUNING
fi
out="$HILL/$name"
wer="$HILL/baseline/speech/wer.json"
if [ -n "$against" ] && [ -f "$HILL/$against/speech/wer.json" ]; then wer="$HILL/$against/speech/wer.json"; fi
run() { if [ -n "${JUDGE_DRY:-}" ]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi; }
[ -z "${JUDGE_DRY:-}" ] || echo "+ CHIMERA_TUNING=${CHIMERA_TUNING:-}"
run rm -rf "$out"
if [ -n "$tuning" ]; then
  run mkdir -p "$out"
  run cp "$tuning" "$out/tuning.json"
fi
run target/release/chimera compare tsp ~/datasets/tsp "$out/tsp" --jobs "$jobs" --against "$HILL/$ref/tsp/score.json" | tail -n 12
run ~/.cache/chimera-asr/.venv/bin/python tools/asr/speech_wer.py "$out/speech" --against "$wer" | tail -n 4
run python3 tools/plint/terms.py "$out/tsp/score.json" "$HILL/$ref/tsp/score.json" --sex male --groups voice,intonation | head -n 20
