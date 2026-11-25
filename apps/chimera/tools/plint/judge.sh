#!/usr/bin/env bash
# Plint · judge the engine as it is built in target/release: the real speakers (TSP) against the
# best run so far, then Whisper against the round's baseline. Results in the rig's hill/<name>.
#   tools/plint/judge.sh <name>      (about 15 minutes on Plint's 4 cores, 3 of them used)
# Keep a change with:  rm -rf $HILL/best && cp -a $HILL/<name> $HILL/best
set -euo pipefail
cd "$(dirname "$0")/../.."
HILL="${HILL:-/mnt/storage/plint/chimera/hill}"
out="$HILL/${1:?name of this try}"
rm -rf "$out"
target/release/chimera compare tsp ~/datasets/tsp "$out/tsp" --jobs 3 --against "$HILL/best/tsp/score.json" | tail -n 12
~/.cache/chimera-asr/.venv/bin/python tools/asr/speech_wer.py "$out/speech" --against "$HILL/baseline/speech/wer.json" | tail -n 4
python3 tools/plint/terms.py "$out/tsp/score.json" "$HILL/best/tsp/score.json" --sex male --groups voice,intonation | head -n 20
