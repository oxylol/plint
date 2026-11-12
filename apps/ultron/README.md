# Ultron

A companion you talk to at `https://ultron.$DOMAIN`: Ultron from *Age of Ultron*.

- **Mind**: celeris-1 through its OpenAI-style API (`CELERIS_API_KEY` in `/opt/plint/.env`); answers arrive in about
  half a second. A second mind to try, switched with MIND in TUNE: Claude Haiku (`claude-haiku-5-5`) through the
  Anthropic SDK (`ULTRON_ANTHROPIC_API_KEY`). Measured 2026-11-12 on the same eight messages: Haiku's text after 1.4 s
  against 0.3 s, replies about twice as long (45 against 21 words), it keeps the thread of the talk better and got
  the sum right that celeris-1 once got wrong. Haiku takes no temperature, so WILDNESS does nothing for it.
  Haiku has its own brief (`claudePrompt()` in `server.js`): left with the celeris one it turned into a kind helper
  with a manner, so its brief is mostly about who he is and what he will not do (console, look after, keep the talk
  going with a closing question), and ends with a five-line check he runs on every reply. The other TUNE switches
  apply to both. Haiku's answer is streamed, so the first sentence goes to the voice while the rest is still being
  written (measured 2026-11-12 at LENGTH short: first text after 0.8 s, all of it after 1.1 s, so this saves about a
  third of a second, more on longer replies). One answer in a hundred used to take twelve seconds: a new connection
  to the API now and then hangs before it is made (measured from Plint: 2 of 270), and Node waits ten seconds for
  it. A streamed request now gets three seconds to start and is then made again (`quickFetch`).
- **Voice**: VoiceStudio speaks each sentence with the clone profile `Ultron`; the page plays the clips in order and the
  mouth of the face follows the sound. Two VoiceStudios: the **PC** (RX 9070 XT, over the tailnet, about a second per
  sentence) is asked first whenever it answers; **Plint's own** (container `voicestudio` in the table stack, started by
  `COMPOSE_PROFILES=speech`, CPU, about 20 s per sentence) takes over when the PC is off. The page shows which one
  spoke: `CLONED · GPU` or `CLONED · CPU`. VoiceStudio drops its model after fifteen idle minutes
  (`OMNIVOICE_IDLE_TIMEOUT`, 900 s), and the next sentence then waits for it to load: measured on the PC 2026-11-12,
  14.6 s against 0.9 s warm. So an open page keeps it in (`/api/wake` when the page opens or comes to the front and
  every four minutes; the server then says a short word on the PC if it has been quiet for nine minutes), a
  sentence that finds it cold gets thirty seconds instead of eight, and a sentence that is late while the PC still
  answers its health check is asked once more there instead of going to the CPU voice, which would take longer.
- **Manner**: the prompt in `server.js` never mentions a film: Sokovia is his own past. It describes how he talks (thinking out loud, pauses on three dots, the
  grand line undercut by a small one) and shows the model his own lines from `data/ultron/lines.txt` (one per line, not
  in git) for the rhythm only: he is told not to repeat them, to leave the film's subjects alone most of the time and
  to almost never say your name (and a name said to your face too soon after the last time is cut out of the reply). A reply that still lifts five words in a row from a film line is not shown: it is
  asked for again without the lines in view (celeris-1 answers whole in half a second, so this costs little; the log
  says `recited the film`). Asking him to quote himself, or THE FILM on FREE, switches that check off. Short sentences
  go to the voice together, so a pause stays inside one breath. One more habit is cut the same way as the name:
  handing your own word back before answering ("Slow? ...", "Tired... "), which he did in 17 of 20 replies to one-
  and two-word messages (Haiku, 2026-11-12). The rule in his brief brought that to 13; `echoed()` takes such an
  opening off the reply unless none of his last six replies had one, which leaves 7 to 10 in 20 that begin with a
  word of yours at all, 1 or 2 of them as a question.
- **His past**: `PAST` in `server.js` is his history written as his own memories (the scepter, JARVIS, the twins, Klaue,
  Seoul and the cradle, Vision, the city, the last drone), from the character's biography Filip supplied. Both minds
  get it and are told it is what he knows, not what he talks about.
- **Sounds**: he may write `[laughter]` and `[sigh]`, the two reaction tags OmniVoice renders with this voice (a laugh
  adds about two seconds, a sigh half a second); they are hidden in the transcript and any other bracket is dropped.
  Graded emotion needs another engine: `HANDOFF-pc-emotion.md` is the brief for installing IndexTTS 2.5 on the PC.
- **Tuning**: TUNE on the page changes what the mind in use is given (kept in `data/ultron/tune.json`, applied from
  the next message): which mind, his reach (Haiku only), wildness (temperature, celeris-1 only), reply length, how
  much of the film comes back, how many film lines he is shown, how often he says your name, how many past messages
  go along, and a box of extra instructions in plain words. SHOW THE PROMPT prints exactly what is sent.
- **Ears**: the browser's own speech recognition. The microphone (or his face) takes one thing you say. TALK keeps
  listening for as long as it is on: for 25 seconds after he has spoken, anything you say is meant for him (the
  lamp says LISTENING); after that he answers to his name (STANDBY), so a room that is talking to itself does not
  set him off. Silence no longer switches it off; a recogniser that dies at once six times in a row does, with a
  line saying so. The screen is kept awake while TALK is on, and a page in the background does not listen. Typing
  always works.
- **He speaks first** (AWAKE on the page, on unless switched off; kept per device): when you open the page after
  twenty minutes or more (`GREET_AFTER_MIN`) he says something; when you have been quiet for about a minute he may
  add a thing of his own, and once more a few minutes later; then he waits to be spoken to. The page only says what
  is happening (`cue`: greet, idle, glance in `/api/say`); it reaches him as a line of yours in round brackets,
  which he is told is not speech. The server has the last word (not over a turn, not within 20 s of the last thing
  said, three unanswered remarks at most), and for a silence or a look he may answer with a dash, which is shown as
  nothing. He knows how long you were gone: a message after an hour's hole is marked "(two days later)". A browser
  plays no sound before the first touch, so a greeting on a page nobody has touched is shown and waits: TAP ANYWHERE
  TO HEAR HIM.
- **Eyes**: EYES on the page turns on that device's camera (a small grey preview by his face; tap it to switch
  between front and back). One frame, at most 768 pixels on its long side, goes along with each thing you say, to
  whichever mind is in use (both can see); it is not stored. Without it he is told he cannot see. With AWAKE he also
  looks by himself: once when his eyes open, and again when the view has changed since his last look and come to
  rest (a 24 by 18 grey print compared every 2.5 s, at most one look in 90 s, three without a word from you).
- **Memory**: three kinds. The talk itself in `data/ultron/history.jsonl`, of which the last 30 messages (CONTEXT in
  TUNE) go along each time; FORGET clears it (a dated copy stays). The talks he remembers: when a talk is over (an
  hour of nothing, or FORGET) the mind in use writes a line or two of it in his own first person ("They told me
  they got the job at the shipyard. I mocked it."), kept in `memory.json` under `talks`; the last four, and older
  ones that share a word with what you are saying, are put in front of him, so a talk outlives FORGET and the
  handful of messages he is handed. MEMORY lists them, × forgets one. And a graph of what he has learned,
  `data/ultron/memory.json`: things and the links between them ("Filip · has a cat called · Loke"). After each turn the
  mind in use is asked for the lasting facts in what you said; before each turn the links around whatever you
  mention, plus the freshest ones about you, are put in front of him. MEMORY on the page draws the graph; tap a thing
  to see its links, × forgets one. His own past is in the same graph (`past-graph.json`, 36 links around his node, put
  in at start-up and marked as past): it comes to mind when a name from it is mentioned, and WIPE leaves it, clearing
  what he learned from you and the talks he remembers (a dated copy of `memory.json` is kept first). Messages of
  one or two words are not searched for facts, and the learner sees the two messages before yours so it can tell
  who "she" is.
- **Reach** (REACH in TUNE, Haiku only; WEB by default): when what you ask has to be known as it is now (weather,
  news, a result, a price), it is looked up first and put in front of him as something he knows, and he says it
  his own way. Handing him a search tool directly did not work for someone who must answer in a second: with a
  tool in reach he thought before every reply (3.5 s to the first word against 0.8 s), and with the thinking
  switched off he promised to look and did not, or read the pages out. So it is split in three: a small question
  beside every turn settles whether anything is needed (`route()`, about 0.8 s, while his answer is already on its
  way, so an ordinary reply loses nothing); a lookup is its own call with Anthropic's web search (`lookUp()`), and
  only its two sentences of fact reach him; and WEB + HOUSE gives him, for the one reply in which you ask for it,
  two tools: `phone_note` (a line on your phone through ntfy) and `tv` (open an app, pause or resume, volume,
  screen on or off through `tv/remote`). Measured 2026-11-12: lookups asked for when they should be and only then
  in 8 of 8, first sentence after 4.5 to 5 s; a house action after about 3 s. The lamp says REACHING meanwhile.
  celeris-1 has no reach: it answers from what it knows.
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

## Measuring his manner

`battery.js` talks to a scratch copy of the server (never the live one: it writes to the talk, and it refuses port
5230) and counts the habits a prompt change is meant to move, so the change is measured and not eyeballed:

```
docker run --rm -d --name ultron-scratch --network host -e PORT=5931 -e DATA_DIR=/data -e ANTHROPIC_API_KEY \
  -e CELERIS_API_KEY -v /tmp/ultron-scratch:/data -v /opt/plint/apps/ultron:/app:ro -w /app node:22-alpine node server.js
node battery.js http://127.0.0.1:5931 terse --show     # the one- and two-word messages he actually gets
node battery.js http://127.0.0.1:5931 full             # whole sentences, a fact, a trouble, Stark
node battery.js http://127.0.0.1:5931 cues             # he speaks first: someone comes in, silences
node battery.js http://127.0.0.1:5931 reach            # does he look things up when he should, and only then
```

Copy `data/ultron/tune.json` and `lines.txt` into the scratch folder first so it runs with the live settings. Twenty
replies swing by a few either way from run to run: run it twice before believing a difference.

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
- TALK and STANDBY rest on the browser's recogniser. Checked with a stand-in recogniser only: how well a phone hears
  "Ultron" across a room, and whether iPhone Safari keeps restarting it, is not known yet.
- His own looks through the camera were checked with drawn scenes. In a real room the two thresholds in `eyes`
  (`0.07` changed, `0.025` at rest) may want moving.
- A lookup takes four to five seconds, and what it brings back is as good as the first page of a web search.
- A new message cancels the answer that is still being spoken.
- No login, like the other apps on the tailnet, but only his own page may change anything: a request a browser sends
  from another site is turned away (`ownPage()`). Every message costs tokens on the mind in use; with Haiku and
  REACH each turn is two small calls more (does it need anything; what is worth keeping), and a lookup a third.

## Look

The Plint Terminal language (one light on black, cut blocks, dot matrix, stepped motion), with the tokens and fonts of
the terminal theme copied into `public/` so it does not depend on the theme switch-over. His eyes and mouth are the one
exception: red, `--u-ph` in `public/ultron.css`.
