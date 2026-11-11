# Ultron

A companion you talk to at `https://ultron.$DOMAIN`: Ultron from *Age of Ultron*.

- **Mind**: celeris-1 through its OpenAI-style API (`CELERIS_API_KEY` in `/opt/plint/.env`); answers arrive in about
  half a second. A second mind to try, switched with MIND in TUNE: Claude Haiku (`claude-haiku-5-5`) through the
  Anthropic SDK (`ULTRON_ANTHROPIC_API_KEY`). Measured 2026-11-12 on the same eight messages: Haiku's text after 1.4 s
  against 0.3 s, replies about twice as long (45 against 21 words), it keeps the thread of the talk better and got
  the sum right that celeris-1 once got wrong. Haiku takes no temperature, so WILDNESS does nothing for it.
  Haiku has its own brief (`claudePrompt()` in `server.js`): left with the celeris one it turned into a kind helper
  with a manner, so its brief is mostly about who he is and what he will not do (console, look after, keep the talk
  going with a closing question), and ends with a four-line check he runs on every reply. The other TUNE switches
  apply to both.
- **Voice**: VoiceStudio speaks each sentence with the clone profile `Ultron`; the page plays the clips in order and the
  mouth of the face follows the sound. Two VoiceStudios: the **PC** (RX 9070 XT, over the tailnet, about a second per
  sentence) is asked first whenever it answers; **Plint's own** (container `voicestudio` in the table stack, started by
  `COMPOSE_PROFILES=speech`, CPU, about 20 s per sentence) takes over when the PC is off. The page shows which one
  spoke: `CLONED · GPU` or `CLONED · CPU`.
- **Manner**: the prompt in `server.js` never mentions a film: Sokovia is his own past. It describes how he talks (thinking out loud, pauses on three dots, the
  grand line undercut by a small one) and shows the model his own lines from `data/ultron/lines.txt` (one per line, not
  in git) for the rhythm only: he is told not to repeat them, to leave the film's subjects alone most of the time and
  to almost never say your name (and a name said to your face too soon after the last time is cut out of the reply). A reply that still lifts five words in a row from a film line is not shown: it is
  asked for again without the lines in view (celeris-1 answers whole in half a second, so this costs little; the log
  says `recited the film`). Asking him to quote himself, or THE FILM on FREE, switches that check off. Short sentences
  go to the voice together, so a pause stays inside one breath.
- **Sounds**: he may write `[laughter]` and `[sigh]`, the two reaction tags OmniVoice renders with this voice (a laugh
  adds about two seconds, a sigh half a second); they are hidden in the transcript and any other bracket is dropped.
  Graded emotion needs another engine: `HANDOFF-pc-emotion.md` is the brief for installing IndexTTS 2.5 on the PC.
- **Tuning**: TUNE on the page changes what celeris-1 is given (kept in `data/ultron/tune.json`, applied from the next
  message): wildness (temperature), reply length, how much of the film comes back, how many film lines he is shown,
  how often he says your name, how many past messages go along, and a box of extra instructions in plain words.
  SHOW THE PROMPT prints exactly what is sent.
- **Ears**: the browser's own speech recognition (tap his face or the microphone). Typing always works.
- **Eyes**: EYES on the page turns on that device's camera (a small grey preview by his face; tap it to switch
  between front and back). One frame, at most 768 pixels on its long side, goes along with each thing you say, to
  whichever mind is in use (both can see); it is not stored. Without it he is told he cannot see.
- **Memory**: two kinds. The talk itself in `data/ultron/history.jsonl`, of which the last 30 messages (CONTEXT in
  TUNE) go along each time; FORGET clears it (a dated copy stays). And a graph of what he has learned,
  `data/ultron/memory.json`: things and the links between them ("Filip · has a cat called · Loke"). After each turn the
  mind in use is asked for the lasting facts in what you said; before each turn the links around whatever you
  mention, plus the freshest ones about you, are put in front of him. MEMORY on the page draws the graph; tap a thing
  to see its links, × forgets one, WIPE forgets everything.
- **What he is not told** (removed on Filip's word, 2026-11-11, so that he simply is Ultron): nothing about being a
  program, what he runs on or where he is (no Plint, no server, no house), no rule to be helpful or kind, no facts about the server. (The old
  notes file `data/ultron/notes.md` is no longer read; the memory graph replaced it on 2026-11-12.)

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

## The PC voice

VoiceStudio 0.5.8 from a source checkout on Filip's PC with the opt-in ROCm runtime (`HANDOFF-pc-voice.md` is the brief
its agent worked from). Plint reaches it at `ULTRON_PC_VOICE_URL` with `ULTRON_PC_VOICE_KEY` (both in `.env`; the URL is
the PC's tailnet IP, its MagicDNS name does not resolve on Plint). The PC's firewall admits only Plint on port 3900.
Measured from Plint 2026-11-11 at 16 steps: 0.8 s for a 2-second sentence, 0.9 s for a 9-second one; in the page the
first sentence starts 1.5 s after sending. Plint asks the PC's `/health` every 20 s; a PC that stops answering in the
middle of a turn costs up to 12 s once, then Plint's own voice finishes the answer.

## Limits

- **Slow voice when the PC is off.** The text is there in half a second; on Plint's CPU the first spoken sentence
  follows after about half a minute and each further one takes as long again.
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
