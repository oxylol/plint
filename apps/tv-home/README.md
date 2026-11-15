# TV home

The home screen of the bedroom TV stick (a Strong Leap-Una: Amlogic, 1 GB of memory, Android 14, on an
old Samsung). Built on Filip's brief: the stick is very slow, he wants a TV interface with just YouTube
and Stremio, and Claude able to control the stick and the TV from Plint.

It is an Android app of one screen, `se.plint.tvhome`, about 100 KB, with no permissions, no network
and nothing running in the background. The stock home screen (`com.google.android.tvlauncher`) is
switched off on the stick, so the Home button lands here.

## What is in it

- **Two tiles**, YOUTUBE and STREMIO. OK opens the app. A tile whose app is not on the stick stays dim.
- **MORE**, under the tiles: Settings first, then every other TV app on the stick by name (and any app
  he has put on himself that is not made for a TV). Back returns to the tiles.
- **The look** is Plint Terminal (`theme/plint.css`): the ground `#0f0e11`, the one light `228,248,234`,
  Geist Mono, chamfered blocks, the block in focus in inverse video, the mark in the corner. On a cold
  start the parts switch on one after another with a stutter; nothing is eased.
- It is made of ordinary text views, so `plint tv screen` reads it and shows what is in focus.

Not in it: a clock, a search, recommendations, anything from Plint. The tiles are a list at the top of
`src/se/plint/tvhome/Home.java`.

## Build and install

```
apps/tv-home/build.sh             → data/tv/plint-home.apk
apps/tv-home/build.sh --install   and put it on the stick
```

No Gradle: Google's build tools (aapt2, d8, zipalign, apksigner) and `javac`, run in a
`eclipse-temurin:17-jdk-jammy` container. The tools (about 300 MB, under Google's SDK licence) are fetched
on the first build and kept on the array in `/mnt/storage/plint/tv-home/sdk`. The key that signs the app
is made on the first build and stays in `data/tv/home.keystore`: an update has to carry the same one, or
the stick refuses it (then: `plint tv sh pm uninstall se.plint.tvhome`, with the stock home screen
switched back on first).

## Checks

By text, on the stick (there is no scratch stick): install it beside the stock home screen, open it with
`plint tv sh am start -n se.plint.tvhome/.Home`, then `plint tv screen` after each `plint tv key …`.
Only when it works is the stock one switched off.

## Putting the stock home screen back

```
plint tv enable com.google.android.tvlauncher
plint tv enable com.google.android.tvrecommendations
```

The stock one outranks this one, so the Home button goes back to it at once.

`assets/GeistMono.ttf` is Geist Mono (The Geist Project Authors, SIL Open Font License 1.1), the
variable font from Google Fonts.
