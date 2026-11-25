# CHIMERA

A physical-modeling voice synthesizer in the browser. Every sound comes from a simulated
airway (lungs, vocal folds or syrinx, a branched vocal tract, lips, nose), not from samples.
The engine is Rust, compiled to WebAssembly for an AudioWorklet; the same code runs natively
for tests and the `chimera` CLI.

## Vision & scope

1. **One instrument**: a physically modeled voice (lungs, vocal folds, throat, mouth). No
   samples, no neural audio.
2. **The web app is the primary surface**: anyone should be able to make songs in it without
   an LLM's help. Mobile is the likely next surface. The plugin (CLAP/VST3/AU) still ships, but
   it's secondary.
3. **As realistic as we can make it**: the main push is to see how close the simulation gets to
   a real voice. The robotic, Vocaloid-like sound people like today stays playable as a preset
   ([#93](https://github.com/jamesondh/chimera-vocal-synth/issues/93)).
4. **Languages enter through one shared front-end design**, not one-off hacks. Each needs a
   native speaker's check and must leave English unchanged.
5. **The design language (the vector display) is settled**; the look within it can still evolve.
6. **Nothing copyrighted in git**; MIT; a deterministic, real-time-safe engine.
7. **The 3D head can be worn by a body** ([docs/rig.md](docs/rig.md)), for the videos people
   make with the voice, posed by joint angles or moved by other tools' motion (Mixamo, BVH,
   glTF): a small library, not animation software (no timeline, no renderer).
8. **Not now**: a DAW or sequencer, a light theme, multi-voice dialogue, translating the
   interface (the voice speaks and sings many languages; the labels stay in English).

## What it does

- **Play**: drag the tongue, lips and jaw in a live cutaway of the head; vowel pad, encoders,
  computer keyboard and Web MIDI.
- **Speak**: English text (CMUdict with a letter-to-sound fallback), Turkish text (EN/TR) or
  IPA, turned into gestures and prosody, with the gestural score and a consonant chart on screen.
  In IPA mode a notice under the box lists the symbols and marks the voice doesn't say
  (`chimera render speech` prints the same as warnings).
- **Sing**: type words and the **Sing** button sings them to a tune picked by its mood, each line
  fitted to a phrase of it (Tempo and Key on the Voice screen); **Record video** sings it again into a
  9:16 clip to share. The keys sing the same words a syllable at a time, legato or not, onto a
  pitch roll; consonants land before the beat. Lyrics can be English, Turkish (chosen with EN/TR:
  Latin script can't tell them apart), Korean (Hangul: one block per note, with Korean sound
  rules), Japanese (kana, one mora per note) or Russian; each language is a front end
  ([docs/language-front-ends.md](docs/language-front-ends.md)).
- **Beatbox**: pads and a step sequencer, every drum an articulatory gesture.
- **Bestiary**: songbird (two-voice syrinx), seal, alligator and elephant.
- **Duet** (`?show=<id>`): two 3D heads replay a pre-rendered song.

Male, female and child bodies blend continuously. Everything on screen animates from live
engine state: pressure in the airway, airflow, the vocal folds, spectra and formants.

## Run it

Needs Rust with the `wasm32-unknown-unknown` target, and Bun.

```sh
bun install
bun run dev        # builds the wasm, then Vite on 127.0.0.1:5317
bun run test       # wasm build, cargo test --workspace, bun test tests
```

The AudioWorklet needs a secure context, so open it on localhost or over HTTPS (for example
`tailscale serve`). Vercel builds it with `vercel.json`.

## As a plugin

Early: a CLAP, VST3 and AU instrument for macOS, and CLAP and VST3 for Windows, with the same UI in
a webview, in a window you can resize. On Linux (CLAP and VST3) the same UI opens in your web browser.
MIDI notes sing the lyrics, one syllable per note (Sing); play the posture you've set (Play); say
the next line of text (Speak); or play mouth drums and hum, with a beat that follows the host's tempo
(Beatbox). Everything saves with the project, and the voice parameters can be automated.

On a Mac (signed and notarised) or on Linux (x86_64), the release installs with one command (or
download it from [chimera.audio/plugin](https://chimera.audio/plugin/)):

```sh
curl -fsSL https://chimera.audio/install.sh | sh
```

To build it yourself, with Xcode's command line tools as well as the above:

```sh
crates/chimera-plugin/macos/build.sh --install
```

Or, with GitHub's `gh` logged in, install CI's newest build without building anything:

```sh
crates/chimera-plugin/macos/update.sh
```

To build it on Linux: `crates/chimera-plugin/linux/build.sh --install`. On Windows, in Git Bash with
Rust's MSVC toolchain: `crates/chimera-plugin/windows/build.sh --install`. The Windows build isn't
released yet. See [docs/plugin.md](docs/plugin.md), "Releases".

Offline renders and measurement:

```sh
cargo run --release -p chimera-cli -- render speech|text|song|notes|beatbox|creature|vowels …
cargo run --release -p chimera-cli -- diagnose       # nasal leakage, formants, bandwidths, pitch
cargo run --release -p chimera-cli -- compare csd    # against a real singer (docs/csd-comparison.md)
cargo run --release -p chimera-cli -- compare tsp    # against 23 real speakers (docs/speech-comparison.md)
bun tools/spectrogram.ts in.wav out.png
bun tools/musicxml-to-notes.ts score.mxl P1 --out p1.json   # a score's part → `render notes` melody
```

## Docs

- [docs/architecture.md](docs/architecture.md): the acoustic model (FDTD airway, sources,
  anatomy, gestures), and its [references and credits](docs/architecture.md#references-and-credits).
- [docs/rust-engine.md](docs/rust-engine.md): the engine's real-time and determinism contract,
  and the browser boundary.
- [docs/ui-redesign.md](docs/ui-redesign.md) and [docs/mockups/](docs/mockups/): the vector
  display look.
- [docs/csd-comparison.md](docs/csd-comparison.md): scoring the voice against the Children's
  Song Dataset.
- [docs/speech-comparison.md](docs/speech-comparison.md): scoring the speech against real
  speakers, phone by phone (the TSP speech database).
- [docs/russian.md](docs/russian.md): how Russian text is read, and what is approximate.
- [docs/turkish.md](docs/turkish.md): how Turkish text is read: k, g and l, ğ, stress.
- [docs/hill-climb/](docs/hill-climb/): the rules and log for tuning the voice one measured
  step at a time.
- [docs/rig.md](docs/rig.md): the body rig the 3D head can be worn on.

Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md). Agents: see [AGENTS.md](AGENTS.md).

## License

[MIT](LICENSE). Bundled third-party files keep their own licenses: the CMU Pronouncing
Dictionary ([public/data/CMUDICT-LICENSE.txt](public/data/CMUDICT-LICENSE.txt)), the fonts
([public/fonts/](public/fonts/)) and the body rig's mesh (CC0, MakeHuman and MPFB2:
[docs/rig.md](docs/rig.md#provenance)).
