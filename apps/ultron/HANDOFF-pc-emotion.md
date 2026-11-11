# Handoff: emotion for Ultron's voice on Filip's PC (IndexTTS 2.5)

Written 2026-11-11 by the Claude session on Plint for an agent working **on Filip's Windows PC** (`filips-pc`).
It continues `ultron-pc-voice-handoff.md`: VoiceStudio 0.5.8 already runs here from `C:\Users\Filip\VoiceStudio` with
the ROCm runtime on the RX 9070 XT, serves Plint on port 3900 with an API key, and has the clone profile `Ultron`.

## Goal

Filip wants the voice to **convey emotion**. The engine in use, `omnivoice`, cannot (VoiceStudio says so itself). The
engine that can grade emotion is **IndexTTS 2.5** (`indextts2`). Install it next to OmniVoice, prove that it runs on
the GPU with the `Ultron` profile, measure how fast it is, and let Filip hear whether the emotions come through and
the voice is still his. Plint keeps using `omnivoice` until Filip decides, so **do not change the active engine**,
the API key, the firewall rule or the service.

Done means the report at the end is filled in with real timings and Filip has listened to the test clips.

## What is known (VoiceStudio's `docs/engines/indextts.md` and `docs/expressive-speech.md`; read both first)

- Install is in the app: **Model Catalogue → IndexTTS 2.5 → Install**. It needs `uv` and at least 12 GB free, builds an
  isolated Python 3.11 environment, downloads `IndexTeam/IndexTTS-2.5`, and **on an AMD ROCm host installs PyTorch's
  ROCm build so the engine uses the GPU**. Keep VoiceStudio open while it runs.
- Its licence is the **bilibili Model Use License** (free below very large company thresholds, with conditions).
  **Filip reads and accepts it himself**; do not accept it for him.
- Emotion goes through `POST /generate` (not the OpenAI-style `/v1/audio/speech`), as form fields:
  `text`, `profile_id` (the id of the `Ultron` profile, see `GET /profiles`), `engine=indextts2`, and either
  `emo_vector` = a JSON array of eight weights 0..1 in the order
  **happy, angry, sad, afraid, disgusted, melancholic, surprised, calm**, or `emo_text` (a description in words) with
  `emo_alpha` 0..0.6. Sending both is an error (422).
- IndexTTS speaks Chinese, English, Japanese, Spanish and Arabic. Not Norwegian or Swedish.

Not known, find out:

- Whether IndexTTS 2.5 actually runs on this GPU (`gfx1201`, ROCm 7.2.1). Nobody has reported it. If the installer
  leaves it on the CPU or it fails on the GPU, report the exact error and stop.
- How fast it is. OmniVoice here makes a 2-second sentence in about 0.8 s. IndexTTS is a heavier, different kind of
  model; a conversation needs the first sentence within a few seconds.
- Whether both engines can stay loaded (16 GB of video memory) and be used alternately per request with the `engine`
  field, or whether each switch reloads a model. Time the first request after switching in both directions.
- Whether `POST /generate` is allowed from another machine with the API key, the way `/v1/audio/speech` is.

## Steps

1. Check free disk and that `uv` is on the PATH of the VoiceStudio service. Tell Filip the size before starting.
2. Ask Filip to open Model Catalogue, read the IndexTTS 2.5 licence and press Install. Watch the install to the end.
3. Confirm with `GET http://127.0.0.1:3900/engines/tts` that `indextts2` is available and routed to the GPU, and that
   the active engine is still `omnivoice`.
4. Make these clips with the `Ultron` profile, all with the sentence
   `You really thought that would work. I almost admire it.`:
   neutral (no emotion fields) · angry `[0,0.8,0,0,0,0,0,0]` · sad `[0,0,0.7,0,0,0.3,0,0]` ·
   amused `[0.6,0,0,0,0,0,0.2,0]` · afraid `[0,0,0,0.8,0,0,0,0]` · calm `[0,0,0,0,0,0,0,1]` ·
   and one with `emo_text=dry, contemptuous, quietly amused` and `emo_alpha=0.5`.
   Note for each: seconds to make (run twice, keep the warm time) and seconds of speech.
5. Also time the long line used before:
   `You want to protect the world, but you do not want it to change. How is humanity saved if it is not allowed to evolve?`
6. Check the words of every clip (a transcript is enough) and check in the log that the GPU did the work.
7. Have **Filip listen**: do the emotions differ clearly, and is it still Ultron's voice? Write down his words.
8. Test one `POST /generate` against this PC's Tailscale address with the API key, and one without the key.
9. Leave OmniVoice as the active engine and confirm Plint's normal request still answers in about a second.

## Rules

- Ask Filip before a reboot, a driver change, or anything beyond this install. Never accept a licence for him.
- Do not attest, sync, export or publish the `Ultron` profile. Do not paste the API key anywhere.
- Nothing new listens to the network; the firewall rule stays limited to Plint.
- Say what you verified and what you did not. A clip that exists is not a clip somebody heard.

## Report (fill in and give to Filip for the Plint session)

```
Emotion engine for Ultron
- IndexTTS 2.5 installed: yes / no (error text)       Licence accepted by Filip: yes / no
- Runs on: GPU (device) / CPU                          Install size on disk:
- Active engine still omnivoice: yes / no
- Warm time, short sentence: neutral __ s · angry __ s · with emo_text __ s   (for __ s of speech)
- Warm time, long line: __ s for __ s of speech
- First request after switching omnivoice → indextts2: __ s      indextts2 → omnivoice: __ s
- Words right in all clips: yes / no (which not)
- Filip's verdict on the emotions and on the voice:
- POST /generate from the tailnet address with key: works / refused      without key: refused / not
- Profile id of Ultron:
- Anything that did not work, or that Filip still has to do:
```
