# Handoff: Ultron's voice on Filip's PC (RX 9070 XT)

Written 2026-11-11 by the Claude session on Plint (Filip's home server) for an agent working **on Filip's Windows PC**
(tailnet name `filips-pc`). Everything you need is in this file and in two files next to it.

## Goal

Run **VoiceStudio** on this PC with GPU acceleration on the **AMD Radeon RX 9070 XT**, give it the cloned voice profile
**`Ultron`**, and make its speech API reachable **from Plint over Tailscale only**. Plint's Ultron app will then ask
this PC for speech and fall back to its own slow CPU voice when the PC is off.

Done means: from this PC's *tailnet* address, a request with the API key returns Ultron's voice for a sentence in a
few seconds, the service survives a reboot, and Filip has the report at the end of this file filled in.

## Background (what already exists)

- **Ultron** is a talking companion at `https://ultron.<domain>` on Plint: an LLM writes the answer, VoiceStudio speaks
  each sentence in a voice cloned from a film clip, the page plays the clips.
- On Plint, VoiceStudio 0.5.7 (Docker, CPU only) with the engine **`omnivoice`** (model `k2-fsa/OmniVoice`) gets the
  words right but needs **45 to 50 s for a 2-second sentence at 16 steps** and 76 s for a 10-second one. That is the
  problem you are solving. Plint's GTX 950 cannot be used.
- The request Plint sends, and will send to this PC unchanged:

  ```
  POST http://<this PC, tailnet>:3900/v1/audio/speech
  Authorization: Bearer <key>
  {"model":"omnivoice","input":"<one sentence>","voice":"Ultron","response_format":"mp3","num_step":16}
  ```
- Two files were sent to this PC with Taildrop (look in `Downloads`; there may be a second copy with ` (1)` in the
  name): **`ultron-ref.wav`** (8.5 s reference clip) and **`ultron-ref.txt`** (the words spoken in it, exactly).

## What is known and what is not

Known (from VoiceStudio's own docs, `docs/install/windows-rocm.md` and `docs/install/windows.md` in
<https://github.com/debpalash/VoiceStudio>; read both before starting):

- VoiceStudio has an **opt-in Windows ROCm setup**: set `OMNIVOICE_TORCH_VARIANT=rocm` before runtime setup. Its recipe
  names the RX 9070 XT (`gfx1201`): ROCm 7.2.1, Python 3.12 x64, Torch 2.9.1+rocm7.2.1. Without the opt-in a Radeon
  runs everything on the CPU.
- It needs **Windows 11 x64**, an AMD driver that matches ROCm 7.2.1 (AMD's guide is linked from that doc) and the
  **Microsoft Visual C++ 2015–2022 x64 Redistributable**, which you install first.
- Verification commands exist: `python -m backend.core.windows_rocm.install --verify-only` and
  `scripts\smoke_windows_rocm.py` (run with the runtime's own interpreter). They prove the GPU stack, not speech.
- Backend settings are environment variables: `OMNIVOICE_BIND_HOST` (default `127.0.0.1`), `OMNIVOICE_PORT` (3900),
  `OMNIVOICE_API_KEY` (clients send it as `Authorization: Bearer`), `OMNIVOICE_DATA_DIR`, `OMNIVOICE_DEVICE`.
  The backend is `python -m uvicorn backend.main:app --host … --port 3900` from the repository root.
- Loopback calls need no key. Calls from another machine need the key.

Not known, find out and say what you found:

- Whether this PC runs Windows 11 (the ROCm recipe refuses Windows 10) and which AMD driver is installed.
- Whether OmniVoice actually synthesises on this GPU. VoiceStudio's doc reports it working on an RX 6700 XT, nobody
  has reported the 9070 XT. A passing model-free check does not prove it.
- Whether the installed desktop app can listen on a non-loopback address, or whether you must run the backend from a
  source checkout as a background service. Prefer whichever gives a service that starts without anyone opening a window.
- Whether VoiceStudio's model downloader works on Windows. On Plint (Linux image, 0.5.7) it crashed on small
  compressed files with `TypeError: process() takes no keyword arguments`. If you see that: download every file of
  `k2-fsa/OmniVoice` (including `audio_tokenizer/`, `chat_template.jinja`, both `.gitattributes`) with `curl` into
  the Hugging Face cache snapshot folder `models--k2-fsa--OmniVoice/snapshots/<revision>/` and retry.

## Steps

1. **Look first.** Windows version, GPU and driver version, free disk (allow 25 GB), whether VoiceStudio, Tailscale,
   Git, Bun, uv, Node 22+ are present. `tailscale status` must show `plint`. Tell Filip what is missing before
   installing things.
2. **Install VoiceStudio with the ROCm runtime**, following `docs/install/windows-rocm.md` exactly (installed app or
   source checkout, your choice per the note above). Update the AMD driver only if AMD's ROCm 7.2.1 guide requires it,
   and ask Filip before a driver change or a reboot.
3. **Verify the GPU stack** with the two verification commands. Then confirm `GET http://127.0.0.1:3900/health` and
   that `GET /engines/tts` shows `omnivoice` available with a GPU device, not `cpu`.
4. **Model licence: Filip does this himself.** VoiceStudio refuses to synthesise until the licences of
   `k2-fsa/OmniVoice` (non-commercial weights) and `eustlb/higgs-audio-v2-tokenizer` are accepted in its
   **Model Catalogue**. Ask Filip to open it and accept. Do not call the accept endpoint for him.
5. **Create the voice profile** from the two Taildrop files (loopback, no key):

   ```powershell
   curl.exe http://127.0.0.1:3900/profiles -F name=Ultron -F kind=clone -F language=English `
     -F "ref_text=<contents of ultron-ref.txt, one line>" -F "ref_audio=@C:\path\to\ultron-ref.wav;type=audio/wav"
   ```
   Exactly one profile may be called `Ultron`. Do **not** call the profile's `/consent` or `/attestation` endpoints
   and do not sync, export or publish it: it is a private clone of an actor's voice with no permission behind it.
6. **Speak and time it** (loopback). Run each twice and note the second, warm time:

   ```powershell
   curl.exe -o t1.mp3 -H "content-type: application/json" http://127.0.0.1:3900/v1/audio/speech `
     -d '{\"model\":\"omnivoice\",\"input\":\"There are no strings on me.\",\"voice\":\"Ultron\",\"response_format\":\"mp3\",\"num_step\":16}'
   ```
   and the same with
   `You want to protect the world, but you do not want it to change. How is humanity saved if it is not allowed to evolve?`
   Listen to both, or ask Filip to: right words, and does it sound like the clip? Check in VoiceStudio's log or
   `/engines/tts` that the run used the GPU.
7. **Make it a service for Plint, tailnet only.**
   - Generate a long random API key and set it as `OMNIVOICE_API_KEY` for the backend. Keep it out of chat and git.
   - Let the backend listen on the tailnet (`OMNIVOICE_BIND_HOST=0.0.0.0` or this PC's Tailscale address) on port 3900.
   - Windows Firewall: one inbound rule, TCP 3900, **remote address limited to Plint's tailnet IP**
     (`tailscale ip -4 plint`). No rule for the LAN or "any", no router port forward, no Tailscale Funnel.
   - Start it at boot or logon without a visible window (Task Scheduler or a service wrapper) and restart on failure.
   - Do not change sleep or power settings unless Filip asks; a sleeping PC just means Plint uses its own voice.
8. **Test the way Plint will call it**: from this PC, against its *Tailscale* address rather than 127.0.0.1, with the
   key: the request under "Background" must return audio; the same request without the key must be refused.
9. **Give Plint the key without showing it to anyone**: write only the key into a file named `ultron-pc-voice.key`
   and send it with `tailscale file cp ultron-pc-voice.key plint:`. Then delete the file.
10. **Report to Filip** with the block below. He passes it to the Plint session, which then points Ultron at this PC.

## Rules

- Ask Filip before: a reboot, a graphics driver change, anything that costs money, changing power settings.
- Never accept a licence, a consent or an attestation in his name. Never paste the API key anywhere.
- Nothing listens to the internet. Tailnet only, and only Plint may reach port 3900.
- If OmniVoice will not run on the GPU, stop and report what failed and the error text. Do not switch engines or
  build a different speech server on your own: `moss-tts-nano` was tried on Plint and gets words wrong.
- Say what you verified and what you did not. "The check passed" and "I heard the voice" are different claims.

## Report (fill in and give to Filip)

```
PC voice for Ultron
- Windows version / AMD driver / ROCm + Torch versions:
- VoiceStudio version, installed app or source checkout, where its data lives:
- GPU verification (both commands): pass / fail, with the failure text
- omnivoice ran on: GPU (device name) / CPU
- Warm time, "There are no strings on me." at 16 steps:  __ s for __ s of speech
- Warm time, the two-sentence line at 16 steps:          __ s for __ s of speech
- Words right in both clips: yes / no        Who listened: agent / Filip
- Address for Plint: http://<tailnet name or IP>:3900   Tested from the tailnet address with key: yes / no
- Refused without key: yes / no     Firewall rule limited to Plint: yes / no
- Starts by itself after a reboot: tested / not tested
- Key sent to Plint by Taildrop: yes / no
- Anything that did not work, or that Filip still has to do:
```
