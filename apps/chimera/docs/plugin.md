# CHIMERA as a plugin

*Design, 2026-11-08. The spike (`crates/chimera-plugin`, November 2026) proved the webview works
inside FL Studio on macOS: typing, parameters both ways, two instances, reopening the window.*

The goal of the MVP: sing lyrics from a DAW's piano roll with the real React UI, as CLAP, VST3 and
AU on macOS, under the repo's MIT licence. Windows and Linux came after (see "Systems").

## What the MVP is

- **Modes:** Sing, Play, Speak and Beatbox. Bestiary stays out for now: it needs its own MIDI mapping
  and persisted state.
- **No tunes, Record video, language chooser or Web MIDI.** The host plays the notes. Latin script is
  read as English; other scripts go to their own front ends, as on the website.
- **The sound lives in Rust.** A plugin's window is usually closed while it plays, so nothing the
  sound depends on may live in the page: lyrics, cursor, body, posture, and every note decision.
  The page is a view that sends commands.

## Formats and licence

`nih_export_clap!` plus the `clap-wrapper` crate (MIT or Apache-2.0; no CMake; it embeds the
MIT-licensed VST3 SDK and Apple's AudioUnitSDK) make one dylib export CLAP, VST3 and AUv2. A script
copies the same fat binary into `CHIMERA.clap`, `CHIMERA.vst3` and `CHIMERA.component`. nih-plug's
own VST3 export pulls in GPLv3 `vst3-sys`, so `default-features = false` turns it off.

CI validates with clap-validator, `auval` and pluginval. The workflow's jobs only call scripts in
`crates/chimera-plugin/<system>`, because the agents' GitHub App can't edit `.github/workflows/`:
`crates/chimera-plugin/plugin.yml` is the copy the owner pastes.

**Signing.** On a push to this repo, CI signs the macOS bundles with the owner's Developer ID and
notarises them (`macos/sign.sh`), so a downloaded build opens without clearing quarantine. It needs five
Actions secrets: `MACOS_CERT_P12` (the .p12 in base64), `MACOS_CERT_PASSWORD`, and an App Store Connect
API key as `NOTARY_KEY_P8`, `NOTARY_KEY_ID` and `NOTARY_ISSUER_ID`. A fork's pull request gets no
secrets, so its build is signed ad hoc, and `build.sh` on a Mac signs ad hoc unless
`CHIMERA_SIGN_IDENTITY` names a certificate in the keychain.

## Systems

| | Formats | Editor | Built by |
|---|---|---|---|
| macOS 11+, Apple silicon and Intel | CLAP, VST3, AU | WKWebView | `macos/build.sh` |
| Windows 10 and 11, x64 | CLAP, VST3 | WebView2 | `windows/build.sh`, in Git Bash |
| Linux, x86_64 | CLAP, VST3 | the user's web browser | `linux/build.sh` |

The editor's code is shared (`src/editor/mod.rs`), and what differs is in `src/editor/macos.rs` and
`src/editor/windows.rs`: the host's window handle, the view's size, the pointer, the screen, and how
the webview follows the window.

**Windows.** The host sizes windows in physical pixels and tells the editor its scale
(`set_scale_factor` says yes there, and no on macOS, which scales by itself), so sizes stay logical and
are multiplied at the edges. WebView2 comes with Windows 10 and 11. It keeps its data beside the host's
executable unless told otherwise, which a host in Program Files can't write to, so it's told
`%LOCALAPPDATA%\CHIMERA\WebView2`. Unlike WKWebView it doesn't follow its parent's size, so `tick`
sets its bounds when the window changes. It scales the page by the monitor's DPI, and a host may pick
another scale for the editor (100 % on a 150 % monitor), so the zoom also undoes the difference. wry serves the custom protocol as `http://chimera.localhost/`
there, which the navigation handler allows. The screen that the window has to fit is the pointer's,
because the host asks the size before there's a window.

## Linux

The editor is the page in the user's web browser. Opening the plugin's window in the DAW starts a
small server on 127.0.0.1 and opens its address with `xdg-open`, and closing the window stops it; the
page then says it has no answer. The DAW's own window for the plugin stays small. That's
`src/browser.rs`, and `src/http.rs`, which `examples/ui_harness.rs` serves the page with too.

A webview inside the DAW's window would be WebKitGTK, which needs GTK running on its thread. The host's
thread only runs it if the host does (REAPER does, Bitwig doesn't, and nih-plug gives an editor no timer
to run it with), and a thread of the plugin's own would share a process with REAPER's GTK. A helper
process could embed WebKitGTK in the DAW's window by its X11 window ID, showing this same served page,
at the cost of a second executable to ship, WebKitGTK for users to install, and X11 only. A browser
needs none of that, and works on Wayland.

- **Bitwig** (6.1.7, at least) destroys the editor after 1 s if its parent gets no child window, so
  the editor maps a blank X11 child there, through x11rb: an X error is then the plugin's to log,
  where Xlib's default handler would exit the DAW. The host's window is X11 either way (XWayland on a
  Wayland desktop; nih-plug offers no other on Linux). The child's connection is its own, to
  `$DISPLAY` with `$XAUTHORITY`, as nih-plug's other editors open theirs: a host on another display
  is assumed away, and with no X display there's no child and Bitwig closes the editor. [#158](https://github.com/jamesondh/chimera-vocal-synth/issues/158)
- **Who can reach it:** the page's own files are public (they're the open-source site), but anything
  about the plugin needs the random token in the page's address, which the page keeps and sends as a
  header: a cookie would go to every service on 127.0.0.1, whatever its port. A request must also name
  `127.0.0.1:<port>` as its host, so a web page can't reach it by pointing a name of its own at
  127.0.0.1, and the page asks browsers not to pass its address on as a referrer.
- **Threads:** the server's thread makes the bridge's calls into the host, as nih-plug's own editors on
  Linux do from their window's thread. Closing the window doesn't wait for that thread (a request half
  sent could keep it, and the host's GUI thread with it): a gate cuts it off from the host instead.
- **The page knows** (the `in-browser` class): no grip, since the browser's window resizes itself.
- **A server that's gone** can leave a request hanging rather than failing, so the page's requests time
  out, and after 3 s without an answer it says so and polls once a second until there is one. That also
  covers a host whose main thread is busy, on the other systems. The first request has no timeout, as
  nothing would try it again.
- **Tested:** the token check by unit tests, and under Xvfb on CI, the X11 child: mapped, gone
  with the editor, and a parent already gone an error rather than an exit. In REAPER for Linux here,
  by hand: the plugin opened its page, a click on Beatbox reached it, and closing the window stopped
  the server and the page said so.
  In Bitwig 6.1.7 (demo, under Xvfb), CLAP and VST3: the server answered for 15 s with the child
  and stopped at 1 s without it; closing and opening the window again worked. REAPER keeps an editor
  with no child, so it never showed this. The host test on CI renders on Linux, and doesn't drive the
  browser.

## Threads

| Thread | Owns | Runs |
|---|---|---|
| Audio | `Core`: the `Engine`, the `Singer`, the telemetry scratch | `process`, `reset` |
| Main (host's UI thread) | the webview, `Settings` | custom-protocol and IPC handlers |
| Whichever thread the host restores or activates on | `Core` while replacing it | `initialize` |

nih-plug takes its plugin lock around `process` and `initialize`, so they never overlap. A state
restore that arrives while playing blocks the audio thread for the length of `initialize`, which
glitches, as loading a preset usually does. **The plugin must never call `GuiContext::set_state`:**
nih-plug runs that restore on the audio thread.

They share `Shared`:

- **`Pending`, a `Mutex` the audio thread only `try_lock`s**, once per block and only if an
  `AtomicBool` says it is dirty. If the lock is busy the changes wait a block. Everything in it
  *merges* rather than queues, so a host that stops processing can't overflow it:
  `artic: ArticPatch`, `tip_free`, `source: SourcePatch`, `hold`, `mode`, `cursor`,
  `voice: Option<(Box<HumanVoice>, BodyMix)>`, `syllables: Option<Vec<LiveSyllable>>`. `initialize`
  clears it, and records the sample rate there, so a voice built for another rate is dropped when
  the main thread tries to publish it, never on the audio side.
- **UI notes** are the exception that must queue, in a bounded array. On overflow the array empties
  and a `clear_ui_keys` flag is set instead: Core then releases every key the UI holds, so a
  dropped release can't leave a stuck note. Closing the window sets the same flag.
- **Nothing is freed on the audio thread.** Installing a voice or syllables swaps the old one into
  a `retired` slot in the same struct, under the same lock. The main thread empties that slot under
  the lock before it publishes another, so it holds at most one of each.
- **Telemetry:** the audio thread swaps its flat block with one in a `Mutex`, by `try_lock`, and
  skips the frame if the lock is busy. Both are sized for the largest body at this sample rate, so
  a body change can't make the next snapshot allocate.
- **Atomics:** cursor, sung syllable, sounding note, load, fault, playing, and a state revision.

A body change builds the `HumanVoice` on the main thread and sends it through `Pending`. The audio
thread calls `Engine::install_voice`, which swaps it in and carries the performer's settings over
without allocating. `Engine::set_body` becomes `prepare_voice` + `install_voice`, so the browser
keeps one code path.

**Who holds a note.** A pitch can be held by the host's MIDI and by the on-screen keys at once. Core
keeps one mask for each and tells the `Singer` only about the first press and the last release, as
`src/ui/midi.ts` does across keyboards. A repeated note-on from the same owner is still a new note, so
overlapping notes in a piano roll keep the lyrics in step.

## Real-time safety

`plan_note_on` allocates today (strings per phoneme, a `Vec` per list). The live path becomes
allocation-free without changing a single planned value:

- `plan_utterance` splits at the point its durations are known. `plan_timed` takes borrowed
  segments, uses `SmallVec` for its working lists and a caller's `Vec<Gesture>`, and builds
  `PlannedSegment`s only when asked. Live notes give every segment a duration, so they skip
  `klatt_durations`, which allocates.
- `LiveSyllable` is `Copy`: inline labels, at most 8 consonants on each side of the vowel. Longer
  clusters are cut, in the browser too; no English word has one. `Singer` keeps these, so a note's
  coda survives a lyrics change.
- `Singer` moves from the plugin crate to `chimera::control::singer`. The TypeScript controller
  stays in the browser for now; moving it is a later cleanup.

Checks: the golden planning tests prove the plans are unchanged, and counting-allocator tests cover
`Singer` and `Core` from note-on to render, including the longest syllable and a body swap.

## Determinism

`process` splits blocks at each event, so note timing is sample accurate and output doesn't depend
on block size. A test compares block sizes bit for bit, from a fresh `Core`. Taking telemetry doesn't
change the sound either, so whether the window is open makes no difference; a test checks that too.

- **Flush-to-zero.** nih-plug turns it on around `process`, which `docs/rust-engine.md` forbids for
  the sake of identity with the browser. A plugin has nothing to be identical to, so the plugin
  accepts it: a test renders with it on and checks the voice stays stable and close to the same.
- **`reset` is not a full DSP reset.** It stops what is sung, clears the score, and puts the cursor
  back. Glottal phase, the noise generators and the voice clock carry on, so a second render after
  a bare `reset` matches the first to the ear, not to the bit. A host that re-activates the plugin
  (`initialize`) gets a fresh voice and an exact match. Clearing every DSP state in place, without
  allocating, is a larger engine change, deferred until someone hears a difference.

## Lyrics and the transport

Each note sings the next syllable, so the cursor has to line up with the song:

- Transport start, or a backward jump of the play position (a loop, a seek), puts the cursor back
  on the *start syllable*. It is 0 unless the user picked another in the UI while stopped. The
  position is read in beats where the host gives them, because a tempo change makes a position
  computed in samples jump.
- Transport stop releases everything sung.
- A boolean host parameter, Restart lyrics on play (default on), turns the first of these off for
  live playing.

Known limits: starting mid-song can't know how many notes came before, so the user picks the
syllable. A loop point inside a block is seen at the next block, so a note in that same block sings
the old syllable.

## MIDI

Supported: notes, velocity (breath), pitch bend, channel and poly pressure, CC 74 (slide), and
all-notes-off / all-sound-off. Other controllers are ignored. Expression follows the note sounding;
after it ends, its channel's expression still moves the release.

- **Pressure** raises the breath, up to 75% above the Breath dial at full pressure, never past the
  dial's top. Poly pressure counts only on the sounding key.
- **MPE** (host parameter, default on), lower zone only. Channel 1 is the master channel: its bend
  is the wheel, ±2 semitones. Channels 2–16 are note channels: a note follows its own channel's
  bend at ±48 semitones, and its slide moves the brightness up or down from the dial, counted from
  where the finger landed. With MPE off, a bend on any channel is the ±2 wheel and slide does
  nothing. Ableton Live sends MPE only with "Enable MPE Mode" on in the device's context menu.

Pressure and brightness (the Breath and Bright dials included) glide to a new value with a 12 ms
time constant, so a controller's steps aren't heard. Bend glides with the Glide dial.

Through clap-wrapper's VST3 layer a controller keeps only its last value per block, so controllers
move in block-sized steps there (the glides cover them), and a controller sent with a note arrives
after it, by up to a block. So a slide in the 50 ms after its note starts counts as where the finger
landed, in every format. A bend left on a channel by its last note can sound for up to a block at
the next note's start. Note timing isn't affected.

Not handled: the same pitch held on two channels is one key (the first note-off ends it), RPN 0 and
6 (the note bend range is fixed at 48), the upper zone, CLAP note expressions.

## State

Host parameters (automatable): output gain, vibrato depth and rate, breath, pressed, bright,
breathy, rough, fry, glide, ring, restart lyrics, MPE. IDs are permanent once released.

Everything else is one `#[persist]` JSON field, `Settings`: lyrics, body, mode, posture (the
articulators changed from neutral, and `tip_free`), voiced, the Play-mode pitch, the start syllable,
and a version number. nih-plug restores it before `initialize`, which rebuilds `Core` from it, clears
`Pending` and bumps the state revision. The cursor, held notes and Play's Breathe hold aren't
saved.

Body isn't a host parameter: changing it rebuilds the airway and ends the sung notes.

## The bridge

The page comes from the same `dist/` as the website, embedded by `build.rs` (source maps and the
dictionary left out: Rust does the lyrics). `cargo test` works without a build: the page then says
the UI isn't built. `window.ipc`, which wry defines and the harness imitates, selects plugin mode at
run time, so there is no second bundle.

Page to Rust, `window.ipc.postMessage` with JSON, handled on the main thread:

`note`, `lyrics`, `cursor`, `body`, `mode`, `artic`, `source` (for `voiced` and `f0`), `hold`,
`param { id, value: plain }`.

**Copy and paste.** On macOS a host's Edit menu sends ⌘C and ⌘V to the webview, as `copy:` and `paste:`
down the responder chain. A host with no such menu (Bitwig's plugin process) leaves the text boxes taking
typing but not those, so in a text field the page takes ⌘X, ⌘C, ⌘V and ⌘A itself and sends `edit`, and
Rust calls the same action on the WKWebView. Pasting that way isn't the page reading the clipboard, so
WebKit doesn't ask first. Windows' WebView2 handles Ctrl+C itself, and on Linux it's a web browser.

Parameter gestures: the page's bridge sends `begin` before the first value and `end` 300 ms after
the last, so no control needs to know about hosts. Rust remembers which gestures are open and ends
them when the window closes. The page's timer dies with it, and clap-wrapper's VST3 layer ignores
automation on a parameter whose gesture never ended.

Rust to page, `fetch` over the custom protocol:

- `/poll`, binary, once per animation frame: the fast scalars, the telemetry block (the same flat
  layout and decoder as the worklet's, moved to shared code on both sides) and the active labels. A
  poll also tells the audio thread for the next 250 ms that someone is watching; without that it
  collects no telemetry.
- `/state`, JSON: the whole snapshot of `Settings`, parameters and syllables.

**A restore is a boundary.** Every command carries the revision the page last adopted, and Rust drops
one from before a restore (except the end of a gesture, which must still close). The page sends edits
at once instead of coalescing them per frame, so none can wait out a restore. A short transaction lock,
never on the audio thread, keeps a command's settings and its `Pending` change together against
`initialize`.

**One owner for each value.** `mount()` awaits a `/state` snapshot and fills the stores before the
first render, and in plugin mode `start()` sends nothing: replaying browser defaults would overwrite
a restored project. After that:

- `Settings` values are written only by the page. It reads them again only when the revision in
  `/poll` changes, which means a restore happened.
- Parameters can also change from the host, so the page polls them 20 times a second and skips any
  with a gesture open.
- The cursor, sung syllable, sounding note and a press count are read from every poll. The count tells
  a new note on the same pitch from one held. The piano roll and the words at the mouth follow these,
  and clear when the voice clock starts again.

`AudioEngine` keeps its interface: `send()` goes to the bridge instead of the worklet port. In plugin
mode `store.ts` and `singing.ts` send notes and lyrics to Rust and take the syllables, the cursor and
the piano roll's notes from it. The page's wasm stays: the anatomy model, geometry and vowel map
still use it.

## Window size

The window opens at the size the project saved (`#[persist = "window"]`, 1280 x 800 at first) and can
be anything from 720 x 500 to 3840 x 2160, but never larger than the screen has room for, host's window
included: a window hanging off the screen would have its grip there too, with no way back. The saved
size is left alone, so a project opened on a larger screen gets its size back.

- **Larger than the layout needs:** the screens take the room, as on the website.
- **Smaller than `LAYOUT_MIN` (1120 x 800):** the page is scaled down until it fits, so the layout stays
  the same, smaller. The website's own answer to a narrow window is a stacked page that scrolls, and in a
  plugin what's below the fold might as well not be there. So the page is always laid out at least
  `LAYOUT_MIN` in both directions, and the layout lint's `--plugin` mode fails if it scrolls at
  `LAYOUT_MIN` or at the shapes the grip makes when it's dragged one way only (2048 x 800, 1120 x 1244).
- **On macOS that's WKWebView's view scale, not its `pageZoom`** (`_setViewScale:` with the layout mode
  `_WKLayoutModeDynamicSizeComputedFromViewScale`, WebKit's since macOS 10.11 though not public; without
  them it falls back to `pageZoom`). Under a page zoom WebKit keeps any text of 9 px or more at 9 px or
  larger, its "smart minimum" font size, so below about a 0.75 zoom the labels outgrew their buttons and
  the page outgrew the window (seen in Bitwig, and in the REAPER test's own screenshots). WebKit's trunk
  has since scaled that minimum by the zoom, which shipping macOS doesn't have yet. A view scale scales the
  drawing instead, so text shrinks with everything else. On Windows WebView2's zoom does it; Chromium's
  minimum is 6 px, so only text under 9.6 px would stop shrinking, and it hasn't been looked at there.
- **The page has no scrollbar.** With a mouse, macOS shows classic scrollbars, which would sit beside the
  grip.
- **Known limit:** at the tallest, narrowest windows (720 x 2160) the 2D tract's lip-area label runs off
  the side of its canvas.

nih-plug answers `can_resize` with false, so a host isn't meant to drag the border. Patching that means
forking nih-plug, so the window has its own grip in the corner, as many plugins do. It sits 6 px in,
because macOS keeps the corner of a resizable window for resizing that window: in REAPER a drag from 12
points in shortens the window and clips the page, and one from 16 points in reaches the grip. While dragging, the page
tells Rust the pointer moved, at most once a frame, and Rust reads where to (`NSEvent.mouseLocation`):
the page's own coordinates change with its zoom, which changes during the drag. Double-clicking the grip
goes back to the first size.

- **`request_resize`'s answer can't be trusted.** clap-wrapper's VST3 layer tests `!resizeView(...)`, and
  `kResultOk` is 0, so success returns false and failure true. Its AU layer sets the frame and always
  returns false. So Rust ignores the answer, and half a second after the grip is let go it takes the
  host view's real size as the window's: a host may have refused or clamped it, and a CLAP host may
  resize a little after it's asked.
- **A host may refuse.** clap-wrapper's `onSize` reports failure while `can_resize` is false, and a host
  that honours that keeps its old size. REAPER doesn't, in any of the three formats. FL Studio is the
  one left to try; if it does, the way out is `can_resize` in a nih-plug fork, which would also give
  dragging the border.
- **An AU's view has no window at first.** clap-wrapper's AU calls `gui.set_parent` from
  `uiViewForAudioUnit`, before the host has put the view in a window, and wry panics without one
  (`ns_view.window().unwrap()`), which took the whole host down. So a run loop timer looks for the window
  and the webview is built then. Building in a temporary window and moving across doesn't work: the
  webview never draws in the window it's moved to. A view of ours watching for its window would need a
  class of ours, and bundles of one binary in one process can't all register the same class name. A
  panic from wry is also caught, so the worst case is an empty window.
- **An AU's editor goes when the host uninitializes it (known limit).** clap-wrapper's `Cleanup` closes the
  editor and leaves its empty view in the host's window, and nothing brings it back until the window is
  closed and opened again. REAPER does this for a bounce, and when its sound device starts after the
  editor has opened. The sound carries on. The fixes are a patched clap-wrapper, or a page that outlives
  `gui.destroy` until its view leaves the window, which then has to take care not to call a host that's gone.
- **The webview follows the host's view** through its autoresizing mask (wry fixes a child's size, so
  the editor changes the mask after building it). Nothing calls `set_bounds`: a host that resizes
  after the call returns would grow the webview twice. The zoom follows the view's real size too, read
  at every poll.

## Speak

Each note says the next line, as a note in Sing sings the next syllable. A line is what's between
line breaks, and punctuation inside one makes phrases with pauses, as on the website.

- **Lines are planned on the main thread**, when the text, the IPA toggle, the rate or the body
  changes, and at `initialize`: `to_phrases` (or the IPA tokenizer) then `plan_phrases`, for
  `default_f0(body)`. The audio thread gets a `Vec<Vec<Gesture>>` through `Pending`, and hands the old
  one back through a `retired` slot. A note only calls `Engine::schedule`, which copies into reserved
  buffers, so it allocates nothing. A line is cut at `MAX_SCHEDULE` (4000) gestures, about a minute.
- **The key transposes, like a sampler.** C4 (60) speaks at the body's natural pitch; each semitone
  moves the whole contour through `HumanVoice::bend`, on top of the pitch wheel. A plan isn't rebuilt
  per note. Velocity does nothing here.
- **A note is a one-shot.** Its release does nothing. A new note cuts the line sounding (`replace`).
- Transport start or a loop goes back to the *start line* if Restart Lyrics On Play is on, and a stop
  stops the speech. Picking a line while stopped sets the start line.
- The Speak button says the next line at natural pitch. The consonant chart's pads plan one `ˈɑpɑ` on
  the main thread and send it as a one-shot through `Pending` (latest wins).

The page takes what's being said from Rust: `/poll` carries a spoken serial, the line and the voice
clock of its time 0; when the serial changes the page fetches `/speech/<line>` (the phrases and the
plan), and the score, the sound readout and the words at the mouth work as they do on the website.
Every set of plans has a generation, published with the serial and returned with the plan. Audio can
be sounding from plans already replaced (the text changed during a line), so a reply from another
generation is dropped rather than drawn over the wrong sound. The pads' one-shot has its own generation
and its own route.

## The live spectrum

The audio thread copies each block's output into a bounded sample ring in `Shared`, by `try_lock` and
only while someone is watching; a busy lock loses a block of display, no more. `/poll` drains it. A
small class on the page has the part of `AnalyserNode` that `SpectrumView` and `Spectrogram` use
(Blackman window, FFT, smoothing, decibels), so those views don't change.

## Beatbox

**Keeping time.** Every mouth sound needs 100 to 174 ms of set-up before its hit (measured: kick 137,
snare 130, hat 106, click 174, for every body), so a hit started when its note arrives lands late and
unevenly. In Beatbox mode the plugin reports `BEAT_LATENCY` = 180 ms and puts each hit exactly that far
after the point on the host's timeline that asks for it; the host's delay compensation does the rest.
This makes the hard cases simple:

- The first hit after pressing play has its full set-up, and lands on the grid.
- A step is scheduled when the transport reaches it, not predicted ahead, so a loop or a seek leaves
  nothing to cancel: hits in flight belong to the time before the jump.
- Each step's time comes from the host's position in that block, so a tempo change needs no anchor.

**Changing mode.** The latency is 0 in every other mode, because live playing can't take it. A new mode
and its timing take effect on the audio thread at once, through `Pending`, and don't depend on
`initialize` running again. What the host then does depends on the format, because nih-plug calls
`request_restart` while processing and `latency.changed` otherwise:

- **CLAP:** the host restarts the plugin, so there's a short gap and `initialize` rebuilds `Core`.
- **VST3:** clap-wrapper turns it into `restartComponent(kLatencyChanged)`, and the host reads the new
  latency. It may or may not reactivate.
- **AU (known limit):** clap-wrapper ignores `request_restart`, so while processing the host isn't told.
  Compensation stays as it was, and hits are 180 ms off in one direction or the other until the host
  next activates the plugin (reloading the project does it). That is a lasting timing error, not a gap,
  and fixing it means patching clap-wrapper.
- A host with no delay compensation hears everything in Beatbox 180 ms late.

**The sequencer** follows the host while it is playing and gives a position and tempo (Follow Host in
the transport panel, on by default, saved). The pattern's step number comes from the host's position,
so it stays in phase with the bars, and swing delays the odd steps as before. `BeatPlayer` gets a
second way in beside the voice-clock `tick`: `follow`, which schedules the steps reached in this block.
While the host is stopped, the Play Beat button runs the pattern on the voice clock at its own tempo,
as on the website; a host starting takes over from it.

`follow` keeps the next step's number and asks only where each block starts:

- Steps fire when the position reaches them, and a step's number only moves forward, so a host that
  repeats a position for a block, or reports a slightly uneven one, can't double a hit. A step reached
  late still gets its true time, since `BEAT_LATENCY` leaves room.
- Only a jump of more than two blocks (or a quarter of a step, whichever is more) from where the last
  block ended looks the step up again.
- While the position is negative (a host's pre-roll) nothing fires; step 0 waits for beat 0.
- **Known limit:** nih-plug makes up beats from seconds and the current tempo for a host that gives no
  beats, and doesn't say so. In such a host a tempo change moves the position, and the pattern jumps
  phase. Hosts that give beats, FL Studio among them, aren't affected.

The voice clock moves in control steps of 64 samples, so a hit's time is good to within one (1.5 ms at
44.1 kHz). At block sizes that are multiples of 64 the output doesn't depend on block size, and it matches
the website's sequencer to the sample for the same delay. A kick's peak varies with exactly when it falls
(0.11 to 0.25 across delays, on the website's sequencer too), so a pattern's levels aren't all equal.

**Layout.** At the plugin's window sizes the pads spread across the width and the waterfall sits beside
the transport, and each pad's airstream moves to its tooltip, so the whole of Beatbox fits without
scrolling. The website's layout doesn't change.

**Notes.** Host notes below C4 (60) are pads, with General MIDI's drum kit mapped onto the ten mouth
sounds, so a drum clip plays straight away, as beatbox rather than as its original kit: 35, 36 kick; 37
rimshot; 38 snare; 39 click; 40 inward snare; 41, 43, 45, 47, 48, 50 lip pop; 42, 44, and the ride's 51,
53, 59 hi-hat, since they keep time; 46 open hat; 49, 52, 55, 57 crash. Others below 60 are ignored. C4 and up hum, two octaves down, so a bass
line sits on the keyboard's middle. Velocity sets the accent. The on-screen keys hum at the pitch they
show, and the pads have their own buttons.

Because the same MIDI number means a different pitch from each owner here, the shared press and release
masks don't apply in Beatbox: held hums are kept per owner, as sounding pitches.

While the host is playing, a note's hit goes `BEAT_LATENCY` after the note, on the grid. While it's
stopped, it goes as soon as the mouth can make it (the website's `beat_hit`), because someone is
auditioning. Which of these applies comes from the block's own transport, so a pad queued just before
the host starts is already on its timeline. The hum is the same sound the sequencer uses, built open-ended
with a tag; its release is `GestureScore::release` at the matching time, and it's monophonic, falling back
to the newest key held. Stopping the sequencer clears the score, so a hum still held starts again.

**Close mic.** The website's chain is Web Audio nodes (low shelf +9 dB at 120 Hz, peak +3 dB at 3.5 kHz,
compressor at -28 dB, 5:1, make-up), which a plugin doesn't have, and much of a beatbox's weight comes
from it. `CloseMic` is the same chain in Rust on the audio thread, per sample and allocation-free, used
only in Beatbox with Mic FX on (saved, on by default). It isn't bit-identical to the browser's: the
browser's compressor looks 6 ms ahead and this one doesn't, so a hit's first peak gets the whole make-up
gain. The default pattern peaked at +5 dBFS, and a host's bounce at 0 dB Gain clipped, so the chain ends in
a peak limiter at -1 dBFS (instant attack, 50 ms release). It changes nothing that stays under that,
except the 50 ms or so after a peak it turned down, while it lets go.

**State.** `Settings` gains `speech { text, ipa, rate, startLine }` and `beat { steps, bpm, swing,
preset, mic, follow }`, each with defaults, so older projects load. `playing` isn't saved. Entering
Beatbox sets the ready posture as the website does. Rust does that as part of the mode command, under
one transaction and in one `Pending` edit, so neither a held note nor a restore can separate them.

## Releases

A tag `plugin-v<version>` (the version in `Cargo.toml`; `release.ts` refuses another) makes CI build,
sign, notarise and test on each system as on any push. Only when both jobs pass does the `release` job
(`publish.sh`) make a GitHub release, from the archives the jobs uploaded:

- `CHIMERA-macos.zip` (`macos/release.sh`, which on a tag also requires the Developer ID signature
  and Apple's ticket on every bundle): a `CHIMERA` folder with the three bundles and
  `How to install.txt`.
- `CHIMERA-linux.tar.gz` (`linux/build.sh --package`): a `CHIMERA` folder with the CLAP (one `.so`),
  the VST3 bundle, the licences and `How to install.txt`.
- `SHA256SUMS`.
- `manifest.json` (`release.ts`): the version, and for each system its archive's address and SHA-256
  and each bundle's path in it and its folder. It's what an installer reads, so another tool (a package
  manager, a DAW's library) can install CHIMERA without knowing anything else about it. `schema` goes
  up when a field changes meaning; a new system or field doesn't.

`releases/latest/download/<file>` always names the newest release's, so the website's Download buttons
and `public/install.sh` never change. The installer (`curl -fsSL https://chimera.audio/install.sh | sh`,
served from `public/`) reads the latest manifest (with JavaScript for Automation on macOS, Python or jq
on Linux), checks the archive's SHA-256 and, on macOS, every bundle's signature before copying any,
and installs into `~/Library/Audio/Plug-Ins` or `~/.clap` and `~/.vst3`, so it needs no password.
`curl` doesn't quarantine what it downloads; the bundles are notarised for a browser's download.
`install-test.sh` runs it on CI on both systems, against each build's own archive served as `file://`.

The Linux build is made on Ubuntu 22.04, so it needs glibc 2.35 at most: built on a newer system, the
VST3 SDK's C++ calls glibc 2.38's `__isoc23_sscanf`, which Ubuntu 22.04 and Debian 12 lack. `linux/ci.sh`
fails a build that needs a newer glibc than the 2.35 the release promises. When GitHub retires its
22.04 runners, the build moves to the oldest one left and the promise with it.

To release: bump `version` in `Cargo.toml` if it has shipped, merge, then tag `main` and push the tag.
If either system's job fails, nothing is released: fix it, delete the tag, and tag again.

## Updating

`crates/chimera-plugin/macos/update.sh` installs CI's newest passing build into
`~/Library/Audio/Plug-Ins`. It needs a logged-in `gh`, because GitHub wants an account to download a run's
artifacts. It removes each old bundle rather than overwriting it, because changing the file of a binary a
host has loaded would crash the host. A running host keeps the old version until it restarts.

The plugin doesn't replace itself, though it could on a Mac:

- It would be a way to run new code on every install, so it would need its own signature check, and a key
  that can't leak: anyone who got past that would reach every user's machine.
- It still takes a host restart, Windows locks a loaded DLL, and a sandboxed host may not let it write.
- Musicians don't expect an instrument in their session to go online and change itself.

For the public, releases (above) and the installer stand in, and the plan is a notice in the plugin that a
new version exists (off with one click, and making no other network use): see Later.

## Testing without a Mac or a DAW

- `Core` has no nih-plug types, so its tests run anywhere: MIDI to audio, allocation, block-size
  independence, flush-to-zero, state restore, voice swap, note ownership, transport, spoken lines,
  beat steps against a simulated host transport (loops, seeks, tempo changes), the drum map, close mic.
- `examples/ui_harness.rs` serves the same `/poll`, `/state` and IPC over HTTP from a real `Core` on
  a timer. A Playwright script drives the UI against it in headless Chromium, and the layout lint
  checks the plugin's window sizes.
- CI on macOS builds all three bundles, runs the validators, and runs the test below.

## Testing in a host

GitHub's runners have a display, and their images let scripts capture it and post clicks, so CI runs the
real plugin in a real host: REAPER, driven by `host-test/run.sh`, with `host-test/host_test.lua` inside it
and `host-test/check.ts` judging what comes out. What differs by system (installing REAPER, the screen,
the mouse, its dialogs) is in `<system>/host-ui.sh`: System Events and `cliclick` on macOS, PowerShell
(`windows/ui.ps1`) on Windows, and on Linux a virtual display with no editor stage. On a Mac it takes
about seven minutes of a CI run, and it fails the run.

- **Each format, as a user would:** it loads, each mode renders, notes come out in tune, Beatbox's kicks and
  pads land on the grid after the host's delay compensation, a hum sits two octaves down, and the formats
  are about as loud as each other over time. Then the editor opens, notes play through the sound device, a
  real click picks Beatbox, a real drag on the grip resizes the window, and a bounce follows. Then REAPER
  quits and comes back: the saved project returns in its mode and at its size and bounces the same, and
  once its window is closed and opened again, a click stops the pattern.
- **Every format has to get through every case:** a run that did nothing mustn't pass. REAPER can't set an
  AU's state from a script, so AU covers Sing and what's reachable by clicking, not Speak, pads or the hum.
- **AU runs alone.** All three bundles hold one binary, so in one process their Objective-C classes share
  names, and a host's lookup of the AU's view fails. REAPER then quietly shows its generic sliders, which
  is what hid the crash above until AU ran by itself.
- **Formats are compared by loudness over time, not by samples.** With the sound device open, the voice runs
  on through silence, so it meets the first note in a different state each time. That's a coarse check: it
  catches a format gone silent or badly off, and doesn't prove they sound alike.
- **Rendering at the device's rate.** REAPER prepares plugins at the sound device's 48 kHz, so a bounce at
  another rate changes Beatbox's latency in samples, and a VST3 is then restarted partway through. It cost
  one kick its tail. Likely only that bounce at another rate is affected, and it isn't fixed.
- **Screenshots and renders** are uploaded as each job's `chimera-host-test-<system>` artifact, and on macOS
  also go to the log as base64 between `=== screenshot` lines. Read them: a check can't tell a
  good-looking page from a bad one.
- **To develop it** on Linux, `linux/build.sh --install` and then `host-test/run.sh` run what CI runs, with
  `xvfb-run` and REAPER for Linux's GTK and ALSA libraries installed.

It doesn't cover: looping playback in a host (the beat player's own tests do), what it sounds like,
dropouts on real hardware, or other hosts. FL Studio in particular still needs one look.

## Licences

Every bundle carries `LICENSE` and `THIRD-PARTY-LICENSES.txt` in `Contents/Resources`, generated at
build time by `tools/third-party-notices.ts` from what the build really uses: the plugin crate's
dependencies, what `clap-wrapper` compiles, the page's JavaScript and the dictionary. The script fails
if a dependency has no licence, or only a GPL one.

## Later

- **Idle sleep.** An idle voice costs as much as a singing one (29 % of a core at 44.1 kHz). Sleeping
  is cheap to write, but a sleeping voice freezes glottal phase, vibrato and noise, so the decision
  must not depend on the window being open or on wall-clock time, and the display needs a clock that
  keeps moving. FL Studio's Smart Disable covers much of this meanwhile.
- **Lookahead in Sing.** A vowel should land on the beat with its consonants ahead of it. Beatbox's
  reported latency shows how; live playing wouldn't want it, so in Sing it would be a toggle.
- The editor inside the DAW's window on Linux (see "Linux").
- Bestiary.
- Sustain pedal, mod wheel and other controllers.
- Dragging the host's window border (needs `can_resize` in nih-plug: a fork or an upstream change).
- A full DSP reset (see Determinism).
- A notice in the plugin when a new version exists, and a `.pkg` installer (needs a Developer ID
  Installer certificate). Releases, `install.sh` and signing are done on macOS.
- One `Singer` for the browser too, replacing the TypeScript controller.
