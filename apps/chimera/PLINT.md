# Chimera on Plint

Plint's fork of [jamesondh/chimera-vocal-synth](https://github.com/jamesondh/chimera-vocal-synth)
(MIT), taken at commit `2c96403` (2026-11-25). A voice from a simulated airway: lungs, vocal folds,
tongue, lips. No samples, no neural net. The upstream `README.md`, `AGENTS.md` and `docs/` are
kept as they came and still describe the engine and the page.

Filip's aim (2026-11-25): an experiment on its own, to see how real the simulation can get,
driven like ordinary text-to-speech and by a JSON script. English, the male voice first. The
stretch goal, not started: fit a real voice to the control inputs.

## What Plint added

- `crates/chimera-serve/`: a small Rust server. It hands out the page (`dist/`) and answers
  `/api/speak`.
- `plint-build.sh`: builds the wasm, the page and the server on the host.
- `stacks/chimera`: runs the host-built server in a plain Ubuntu container, port 5270,
  `https://chimera.$DOMAIN`.
- The engine changes of the realism rounds, logged in `docs/hill-climb/log.md` under "Plint".

## The API

```sh
# plain text → WAV (48 kHz mono)
curl -o hi.wav "https://chimera.$DOMAIN/api/speak?text=Hello+there.&body=male&rate=1.0"

# a JSON script → WAV
curl -o q.wav -X POST https://chimera.$DOMAIN/api/speak -d '{
  "body": {"female": 0.3, "child": 0},
  "rate": 0.9,
  "source": {"breathiness": 0.3},
  "phrases": [
    {"text": "Is this a real voice", "intonation": "question", "pause": 0.6, "f0": 130},
    {"ipa": "nˈoʊ", "f0": 95, "rate": 0.7}
  ]
}'

# the same script → what the engine will do with it, as JSON (phones, then gestures)
curl -X POST "https://chimera.$DOMAIN/api/speak?plan=1" -d '{"text":"Hi."}'
```

A script has three depths. Each answer to `?plan=1` is itself a script, so one can start from
text and take over by hand one layer at a time.

1. **Text.** `text`, with `body` (`male`, `female`, `child` or a mix), `f0` (Hz), `rate`,
   `accent` (how far pitch moves on stressed syllables) and `source` (`pressure`, `breathiness`,
   `tension`, `fry`, `roughness`, `vibratoDepth`, `vibratoRate`, `drift`, `brightness`).
2. **Phrases.** Each is `text`, `ipa` or `segments` (phones: `ipa`, `dur` in seconds, `stress`,
   `f0`, and the IPA marks the engine knows), with its own `intonation` (`statement`, `question`,
   `flat`), `pause`, `rate`, `f0` and `accent`.
3. **Gestures.** The score itself: for each gesture `t0`, `t1`, `rampIn`, `rampOut`, `strength`
   and `targets` over the 17 articulators (`jaw`, `tbX`, `tbY`, `ttX`, `ttY`, `lipAp`, `velum`, …)
   and the source channels (`abduction`, `closure`, `breath`, `f0`, `vibrato`). When a script has
   `gestures`, they are what is rendered: remove them to plan again from the phrases.

Limits: 120 s of sound and 4 MB a request, two renders at once (`CHIMERA_JOBS`). A sentence
renders in a little under half the time it takes to say (3.3 s of speech in 1.4 s on Plint).
A mistake in a script is answered with `400` and a line saying what was wrong. The response
headers `X-Chimera-Seconds` and `X-Chimera-Render-Ms` give the length and the render time.

No login, like the other pages: it is reachable from the house and the tailnet only.

## Building and running

```sh
apps/chimera/plint-build.sh && plint restart chimera
```

Needs rustup's Rust in `~/.cargo` (with the `wasm32-unknown-unknown` target) and Bun in `~/.bun`,
both installed for the `plint` user on 2026-11-25. Ubuntu's own Rust (1.75) and Node (18) are too
old for it. `target/`, `dist/` and `node_modules/` are not in git: after a fresh checkout, run the
build before `plint up chimera`, or the container has nothing to start.

Checks: `cargo test --release -p chimera-serve` (the script: text to sound, a plan sent back
gives the same sound, bad scripts refused). The page: load it in the Playwright image and look
for console errors, as `apps/stocks/tests/page.js` shows.

## The measuring rig

The author's judges (`docs/hill-climb/README.md`), on the array in `/mnt/storage/plint/chimera`:
the TSP recordings of 23 real speakers (`tsp/`, linked from `~/datasets/tsp`), the Montreal Forced
Aligner (`mfa/`, linked from `~/.cache/chimera-mfa`) and Whisper (`asr/`, linked from
`~/.cache/chimera-asr`). Runs are kept in `hill/` (`baseline`, `best`, `try`).

```sh
target/release/chimera compare tsp ~/datasets/tsp /mnt/storage/plint/chimera/hill/try/tsp \
  --jobs 3 --against /mnt/storage/plint/chimera/hill/best/tsp/score.json
~/.cache/chimera-asr/.venv/bin/python tools/asr/speech_wer.py /mnt/storage/plint/chimera/hill/try/speech \
  --against /mnt/storage/plint/chimera/hill/baseline/speech/wer.json
```

Left out on Plint: the singer judge (Filip's aim is speech), the Chinese scramble judge, the
plugin, the body rig.
