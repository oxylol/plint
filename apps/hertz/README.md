# Hertz

A music maker made of waves, at `https://music.${DOMAIN}`. No keys, no piano roll: dials, buttons and sines.

## The idea

Every voice (Kick, Bass, Snap, Pad, Pluck, Hat) is two waves:

- **Pulse** (wave A): sine, triangle, saw or square, at a rate locked to the tempo (shown in Hz).
- **Drift** (wave B): a slower sine added on top. **Bend** sets how much it pushes the timing,
  **Melody** how many scale steps it moves the pitch. A bassline literally follows a sine.

The sum is compared with a **threshold**. Every upward crossing plays a note; how steeply the wave
crosses sets how hard it hits. **Snap** (Song page) moves hits to the nearest 16th; off, they land
wherever the wave crosses.

The sound of each hit is a sine bent by another sine (FM): **Freq** (melodic voices click between the
notes of the key), **Ratio** (modulator frequency), **Mod** (FM depth), **Decay**, **Color** (low-pass),
**Drive**, **Echo** (dotted 8th) and **Space** (reverb).

The **Tide** is one very slow sine for the whole song (2–32 bars): it moves the chords, thins out the
hits at low tide (a hit only counts if its wave reaches the tide line) and opens the filters.

## Screen

- The **orbit**: one ring per voice, each its wave drawn around one bar, dots where it fires. Tap a
  ring to pick it, tap the centre to play. The centre is the live spectrum, coloured low (red) to high
  (violet), the same order as the voices.
- The **strip**: the picked voice unrolled, pulse (grey), drift (dashed) and their sum. Drag up or down
  anywhere on it to move the threshold.
- Double-tap a dial to reset it. Space plays/stops, 1–7 picks a voice.
- **Rec** records what you hear (m4a on Apple, webm elsewhere) and offers it to share or save.

## Files

- `public/engine.js`: waves, trigger search, Web Audio voices, scheduler. Pure functions also load in
  node (`require('./public/engine.js')`) for tests.
- `public/app.js`, `hertz.css`, `index.html`: the interface.
- `server.js`: serves the page and stores songs in `DATA_DIR/songs.json` (`GET/POST /api/songs`,
  `GET/DELETE /api/songs/<id>`), so every device sees the same list.

## Limits

All sound is made by the browser. On an iPhone it stops when the screen locks (the page keeps the
screen awake while playing). No MIDI, no samples, no export of separate tracks.
