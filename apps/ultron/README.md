# Ultron

A companion you talk to at `https://ultron.$DOMAIN`: Ultron from *Age of Ultron*, living in Plint.

- **Mind**: celeris-1 through its OpenAI-style API (`CELERIS_API_KEY` in `/opt/plint/.env`). Answers arrive in about half a second.
- **Voice**: VoiceStudio (container `voicestudio`, defined in the table stack, started by `COMPOSE_PROFILES=speech`) speaks each
  sentence with the clone profile `Ultron`. The page plays the clips in order; the mouth of the face follows the sound.
- **Ears**: the browser's own speech recognition (tap his face or the microphone). Typing always works.
- **Memory**: the talk in `data/ultron/history.jsonl`; every eight turns celeris-1 rewrites `data/ultron/notes.md`
  (lasting facts), which goes into every prompt. FORGET clears the talk (a dated copy stays), not the notes.
- **Body**: he is told the live state of Plint from `data/theme/*.json` (load, memory, disks, game servers, services down,
  tailnet devices). He can read it, he cannot change anything.

## Run

```
plint up ultron                      # start
plint logs ultron
scripts/ultron-voice.sh status       # profile, model licence, loaded model
scripts/ultron-voice.sh profile      # (re)create the clone from data/ultron/voice/ultron-ref.{wav,txt}
scripts/ultron-voice.sh licence      # read and accept the speech model's licence (needed once)
scripts/ultron-voice.sh say "No strings on me."   # one line, timed (ENGINE=moss-tts-nano STEPS=6 to compare)
scripts/ultron-voice.sh models       # complete the model cache when VoiceStudio's downloader fails
```

After a code change: commit, then `plint sync`.

## The voice

The reference is a clip of about eight seconds of one speaker, with the exact words in the `.txt` next to it
(`data/ultron/voice/`, not in git). A cleaner or different clip: replace the two files and run `ultron-voice.sh profile`.
The profile is for this house only: it is not attested, so VoiceStudio will not sync or publish it.

VoiceStudio's data (models, engines, voices) is on the array: `/srv/dnd/data/voicestudio` is a symlink to
`/mnt/storage/plint/voicestudio`. Its own downloader (0.5.7) crashes on small compressed files, so after a model
update run `ultron-voice.sh models`, which fetches every file of the model into the cache with curl.

Two engines are installed; `ULTRON_VOICE_ENGINE` in `.env` picks one, steps and format are in `stacks/ultron/compose.yaml`.
Measured on Plint 2026-11-11 (four busy cores, no usable GPU), same reference clip:

| engine | time for a short sentence | words | closeness to the reference* |
|---|---|---|---|
| `omnivoice`, 16 steps (its default) | 45 to 50 s | right | 0.76 to 0.85 |
| `omnivoice`, 6 steps (**in use**) | 19 to 26 s | right | 0.74 to 0.84 |
| `moss-tts-nano` | 6 to 10 s, about 2x real time | wrong or with made-up endings in 4 of 7 lines | 0.82 to 0.84 |

\* speaker-embedding similarity (Resemblyzer); another clip of the same actor scores 0.91, an unrelated voice 0.50.
Nobody has listened to it yet when this was written: the numbers say "same speaker", not "sounds like him".

## Limits

- **Slow voice.** The text is there in half a second; the first spoken sentence follows after about half a minute and
  each further one takes as long again. That is the CPU: OmniVoice needs a GPU to keep up with speech.
- **The GTX 950 in Plint does not help.** VoiceStudio's PyTorch is built for RTX 20-series cards and newer (sm_70+),
  the 950 is sm_52, usually has 2 GB, and its driver module is not installed for the running kernel. The ways to a
  quick voice: VoiceStudio on a machine with a newer GPU (set `VOICE_URL` to it), or a newer card here.
- VoiceStudio is capped at 3 GB of memory, which OmniVoice fills (it swaps); it is shared with Plint Table.
- Speech recognition is the browser's (Safari and Chrome have it, Firefox does not) and it sends the audio to the
  browser vendor. Languages on the switch: English, Norwegian, Swedish.
- A new message cancels the answer that is still being spoken.
- No login, like the other apps on the tailnet. Every message costs celeris-1 tokens.

## Look

The Plint Terminal language (one light on black, cut blocks, dot matrix, stepped motion), with the tokens and fonts of
the terminal theme copied into `public/` so it does not depend on the theme switch-over. His eyes and mouth are the one
exception: red, `--u-ph` in `public/ultron.css`.
