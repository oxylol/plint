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
scripts/ultron-voice.sh say "No strings on me."   # one line, timed
```

After a code change: commit, then `plint sync`.

## The voice

The reference is a clip of about eight seconds of one speaker, with the exact words in the `.txt` next to it
(`data/ultron/voice/`, not in git). A cleaner or different clip: replace the two files and run `ultron-voice.sh profile`.
The profile is for this house only: it is not attested, so VoiceStudio will not sync or publish it.

VoiceStudio does not synthesise until the licence of its model (OmniVoice, non-commercial weights) is accepted. Until then,
and whenever VoiceStudio is down, the page says so and he answers in text.

Settings (`.env`): `ULTRON_VOICE` (profile name, default `Ultron`), `ULTRON_VOICE_STEPS` (unmasking steps; 0 = VoiceStudio's
16; fewer is faster and rougher).

## Limits

- Plint has no GPU: speech is made on four CPU cores, one sentence at a time, so his voice trails the text.
- VoiceStudio is capped at 3 GB of memory and is shared with Plint Table.
- Speech recognition is the browser's (Safari and Chrome have it, Firefox does not) and it sends the audio to the
  browser vendor. Languages on the switch: English, Norwegian, Swedish.
- No login, like the other apps on the tailnet. Every message costs celeris-1 tokens.

## Look

The Plint Terminal language (one light on black, cut blocks, dot matrix, stepped motion), with the tokens and fonts of
the terminal theme copied into `public/` so it does not depend on the theme switch-over. His eyes and mouth are the one
exception: red, `--u-ph` in `public/ultron.css`.
