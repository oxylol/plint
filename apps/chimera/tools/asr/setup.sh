#!/usr/bin/env bash
# One-time setup for the ASR judges, in ~/.cache/chimera-asr:
# - tools/asr/speech_wer.py: faster-whisper in .venv (the small.en model, ~0.5 GB, downloads on
#   first use). The hill-climb judges with it, so nothing else is installed there.
# - tools/asr/scramble.py: its own pinned env in scramble/.venv: SenseVoice-Small through
#   sherpa-onnx, with pypinyin and zhconv to score Chinese. The model archive (1.05 GB; the int8
#   model inside is 0.23 GB) is checked against its sha256 before it's unpacked.
set -euo pipefail
dir="$HOME/.cache/chimera-asr"
mkdir -p "$dir/models"
[ -x "$dir/.venv/bin/python" ] || uv venv -q --python 3.12 "$dir/.venv"
uv pip install -q --python "$dir/.venv/bin/python" faster-whisper==1.2.1

sc="$dir/scramble/.venv"
[ -x "$sc/bin/python" ] || uv venv -q --python 3.12 "$sc"
uv pip install -q --python "$sc/bin/python" sherpa-onnx==2.0.1 numpy==2.5.4 pypinyin==0.55.0 zhconv==1.4.3
sv=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
sum=f6b2a72ebcb1ac7a764d4cfccd886e6bcb2a95c4657c2199d0ba95ed4b9ea71a
if [ ! -f "$dir/models/$sv/model.int8.onnx" ]; then
  tmp="$dir/models/$sv.tar.bz2.part"
  curl -fsSL -o "$tmp" "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$sv.tar.bz2"
  got=$( (command -v sha256sum >/dev/null && sha256sum "$tmp" || shasum -a 256 "$tmp") | cut -d' ' -f1)
  if [ "$got" != "$sum" ]; then
    rm -f "$tmp"
    echo "$sv.tar.bz2: sha256 $got, expected $sum" >&2
    exit 1
  fi
  tar xjf "$tmp" -C "$dir/models" && rm -f "$tmp"
fi
echo "run: $dir/.venv/bin/python tools/asr/speech_wer.py <outdir> [--against earlier/wer.json]"
echo "     $sc/bin/python tools/asr/scramble.py <outdir> [--langs zh,ko,en] [--shuffles 8]"
