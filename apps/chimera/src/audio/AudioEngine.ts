/**
 * Main-thread side of the audio engine: AudioContext + worklet node + analyser, coalesced
 * parameter messages (at most one per animation frame per kind) and telemetry fan-out.
 */
import workletUrl from "../worklet/processor.ts?worker&url";
import { chimeraModule } from "../wasm/chimera.ts";
import { inPlugin, ipc, sendToPlugin, setParam, volumeToDb } from "./pluginBridge.ts";
import { SampleAnalyser, type Analyser } from "./sampleAnalyser.ts";
import { preferPlaybackSession } from "./session.ts";
import type { ToWorklet, FromWorklet, TelemetryMessage, PlannedMessage, CreatureTelemetryMessage } from "../worklet/protocol.ts";
import type { BodyMix } from "../wasm/types.ts";
import type { HumanArtic } from "../wasm/types.ts";
import type { VoiceSource } from "../wasm/types.ts";

type Listener = (t: TelemetryMessage) => void;

const ANALYSER_SIZE = 4096;
const ANALYSER_SMOOTHING = 0.6;
const FAST_ANALYSER_SIZE = 2048;
// 48 kHz whatever the device runs at (the browser resamples): the airway's cost grows with the
// square of its rate, and a 192 kHz device took ~3x the CPU of 48 kHz and starved the worklet
// (load 120-150%, no sound; issue #33). 32 kHz is the fallback for a device too slow for 48: its
// airway runs at 64 kHz instead of 96 (about 40 % less CPU), a little duller (output up to 16 kHz).
const FULL_RATE = 48000;
const LIGHT_RATE = 32000;
/** the averaged load (see watchLoad) past which the device can't keep up, and back under which it can */
const OVERLOADED = 0.9;
const KEEPING_UP = 0.6;
const AVERAGE_S = 2;
/** a reading counts as no more than this: a stall (a hidden tab, a laptop asleep, a busy moment)
 * mustn't alone make a fast device look slow, so it takes ~1 s of overload at least to fall back */
const MOST = 1.5;
/** how long a start or rebuild waits for the context to run */
const RESUME_WAIT_MS = 3000;
/** the first moments' load is the wasm compiling, not the device's speed */
const WARM_UP_MS = 2000;

export class AudioEngine {
  ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  analyser: Analyser | null = null;
  /** unsmoothed, short-window analyser for the spectrogram */
  analyserFast: Analyser | null = null;
  /** long time-domain window for the waveform scope */
  analyserScope: AnalyserNode | null = null;
  private master: GainNode | null = null;
  /** "close mic" chain: proximity bass, presence, compression (beatboxers cup the mic) */
  private micIn: AudioNode | null = null;
  private micOn = false;
  latest: TelemetryMessage | null = null;
  /** performance.now() when `latest` arrived */
  private latestAt = 0;
  latestCreature: CreatureTelemetryMessage | null = null;
  /** load of the most recent telemetry of either kind */
  load = 0;
  sampleRate = FULL_RATE;
  /** the device couldn't keep up at the full rate: the voice runs at the lighter one */
  light = false;
  /** the device can't keep up even so: the voice breaks up or goes silent */
  overloaded = false;
  private slowLoad = 0;
  private loadAt = 0;
  private builtAt = 0;
  private rebuilding = false;
  private triedLighter = false;
  private statusListeners = new Set<() => void>();
  private rebuiltListeners = new Set<() => void>();
  private listeners = new Set<Listener>();
  private pendingArtic: Partial<HumanArtic> | null = null;
  private pendingTipFree: boolean | undefined;
  private pendingSource: Partial<VoiceSource> | null = null;
  private pendingCParams: Record<string, number> | null = null;
  private flushScheduled = false;
  private readyListeners = new Set<() => void>();
  private plannedListeners = new Set<(p: PlannedMessage) => void>();
  /** most recent planned utterance (for timelines) */
  lastPlan: PlannedMessage | null = null;
  private speakId = 0;
  error: string | null = null;

  /** The human voice's clock now (s), extrapolated from the latest telemetry; null if none. */
  voiceClock(): number | null {
    if (!this.latest) return null;
    return this.latest.t.clock + Math.min(0.25, (performance.now() - this.latestAt) / 1000);
  }

  private tap: MediaStreamAudioDestinationNode | null = null;

  /** What the speakers play, as a media stream (to record a clip); null before start(). */
  recordStream(): MediaStream | null {
    if (!this.ctx || !this.master) return null;
    if (!this.tap) {
      this.tap = this.ctx.createMediaStreamDestination();
      this.master.connect(this.tap);
    }
    return this.tap.stream;
  }

  get running(): boolean {
    return !!this.ctx && this.ctx.state === "running";
  }

  async start(): Promise<void> {
    if (this.ctx) {
      await this.ctx.resume();
      return;
    }
    await this.build(FULL_RATE);
  }

  /**
   * The context, the worklet and the graph, at this rate; they replace the current ones only once
   * built (a failure leaves things as they were).
   */
  private async build(rate: number): Promise<void> {
    preferPlaybackSession();
    const ctx = new AudioContext({ latencyHint: "interactive", sampleRate: rate });
    // resume now, while the tap that called start() is still the current gesture: iOS Safari
    // refuses a resume() that comes after the awaits below (the context stays suspended, silent)
    const resumed = ctx.resume().catch(() => {});
    let node: AudioWorkletNode;
    try {
      await ctx.audioWorklet.addModule(workletUrl);
      // the worklet instantiates the engine from the module compiled at startup (main.tsx)
      node = new AudioWorkletNode(ctx, "chimera", { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2], processorOptions: { module: chimeraModule() } });
    } catch (e) {
      // without its worklet the context is empty: forget it, so a retry builds the graph again
      // instead of resuming a context with nothing in it (and reporting that the audio started)
      void ctx.close().catch(() => {});
      throw e;
    }
    const old = this.ctx;
    // the old engine's last telemetry mustn't read as the new one's
    if (this.node) this.node.port.onmessage = null;
    this.ctx = ctx;
    this.sampleRate = ctx.sampleRate;
    this.node = node;
    this.tap = null;
    this.latest = null;
    this.latestCreature = null;
    this.lastPlan = null;
    this.builtAt = this.loadAt = performance.now();
    this.slowLoad = 0;
    if (old) void old.close().catch(() => {});
    const master = ctx.createGain();
    master.gain.value = 0.8;
    this.master = master;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = ANALYSER_SIZE;
    analyser.smoothingTimeConstant = ANALYSER_SMOOTHING;
    this.analyser = analyser;
    const fast = ctx.createAnalyser();
    fast.fftSize = FAST_ANALYSER_SIZE;
    fast.smoothingTimeConstant = 0;
    this.analyserFast = fast;
    const scope = ctx.createAnalyser();
    scope.fftSize = 16384;
    this.analyserScope = scope;
    // close-mic chain: proximity effect (low shelf), presence peak, fast compressor, make-up
    const shelf = ctx.createBiquadFilter();
    shelf.type = "lowshelf";
    shelf.frequency.value = 120;
    shelf.gain.value = 9;
    const presence = ctx.createBiquadFilter();
    presence.type = "peaking";
    presence.frequency.value = 3500;
    presence.Q.value = 0.8;
    presence.gain.value = 3;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -28;
    comp.knee.value = 8;
    comp.ratio.value = 5;
    comp.attack.value = 0.002;
    comp.release.value = 0.12;
    const makeup = ctx.createGain();
    makeup.gain.value = 2.2;
    shelf.connect(presence).connect(comp).connect(makeup).connect(master);
    this.micIn = shelf;
    node.connect(this.micOn ? shelf : master);
    master.connect(ctx.destination);
    master.connect(analyser);
    master.connect(fast);
    master.connect(scope);
    node.port.onmessage = (e: MessageEvent<FromWorklet>) => this.receive(e.data);
    // a rebuild comes from no gesture, and a browser may hold the new context until there is one
    // (its resume() then doesn't settle until then): the next tap or key resumes it, and nothing
    // waits on it for long, so the rebuild still finishes (the state is sent, the old song stops)
    const resume = () => void ctx.resume().catch(() => {});
    for (const ev of ["pointerdown", "keydown"]) window.addEventListener(ev, resume, { once: true, capture: true });
    // a second chance where the first resume was refused (the result is what callers see)
    await Promise.race([resumed.then(() => ctx.resume()), new Promise((r) => setTimeout(r, RESUME_WAIT_MS))]);
  }

  /**
   * Whether the audio thread keeps up, from the worklet's load (render time over the block's
   * time). Averaged over ~2 s, past the start's warm-up: over 90 %, the device can't keep up,
   * and the voice breaks up or goes silent. At 48 kHz it then rebuilds at 32 kHz, once; at 32 kHz
   * it says so. A fast device never comes near (about 30-40 %), so its sound doesn't change.
   */
  private watchLoad(load: number): void {
    this.load = load;
    const t = performance.now();
    // a gap between readings counts as no more than a moment
    const dt = Math.min(0.1, (t - this.loadAt) / 1000);
    this.loadAt = t;
    if (t - this.builtAt < WARM_UP_MS || this.rebuilding) return;
    this.slowLoad += ((Math.min(MOST, load) - this.slowLoad) * dt) / AVERAGE_S;
    if (this.slowLoad > OVERLOADED && this.sampleRate > LIGHT_RATE && !this.triedLighter) void this.rebuildLighter();
    else if (this.slowLoad > OVERLOADED) this.setOverloaded(true);
    else if (this.slowLoad < KEEPING_UP) this.setOverloaded(false);
  }

  private setOverloaded(on: boolean): void {
    if (on === this.overloaded) return;
    this.overloaded = on;
    for (const l of this.statusListeners) l();
  }

  /** The lighter rate for a device that can't keep up: ~40 % less work, output up to 16 kHz. */
  private async rebuildLighter(): Promise<void> {
    this.rebuilding = this.triedLighter = true;
    const load = this.load;
    try {
      await this.build(LIGHT_RATE);
    } catch (e) {
      // the full-rate voice goes on (and says it can't keep up)
      console.error("audio: couldn't rebuild at the lighter rate:", e);
      return;
    } finally {
      this.rebuilding = false;
    }
    console.warn(`audio: this device can't keep up at ${FULL_RATE} Hz (engine load ${Math.round(load * 100)} %): running at ${this.sampleRate} Hz`);
    this.light = true;
    for (const l of this.statusListeners) l();
    // the new engine starts from scratch: what it plays is sent again
    for (const l of this.rebuiltListeners) l();
  }

  /** Called when `light` or `overloaded` changes. */
  onStatus(l: () => void): () => void {
    this.statusListeners.add(l);
    return () => this.statusListeners.delete(l);
  }

  /** Called after the audio was rebuilt lighter, with a fresh engine that knows nothing yet. */
  onRebuilt(l: () => void): () => void {
    this.rebuiltListeners.add(l);
    return () => this.rebuiltListeners.delete(l);
  }

  /** In the plugin there's no Web Audio: the same analysers, fed with what the audio thread sends. */
  pluginAnalysers(): { feed: (samples: Float32Array) => void } {
    const slow = new SampleAnalyser(ANALYSER_SIZE, ANALYSER_SMOOTHING);
    const fast = new SampleAnalyser(FAST_ANALYSER_SIZE, 0);
    this.analyser = slow;
    this.analyserFast = fast;
    return {
      feed: (samples) => {
        slow.push(samples);
        fast.push(samples);
      },
    };
  }

  /** The plugin's host sets the rate and runs the engine: there's nothing to rebuild lighter. */
  private noteLoad(load: number): void {
    if (inPlugin) this.load = load;
    else this.watchLoad(load);
  }

  /** A message from the engine: the worklet's, or the plugin's as the page's poll decodes it. */
  receive(m: FromWorklet): void {
    if (m.type === "telemetry") {
      this.latest = m;
      this.latestAt = performance.now();
      this.latestCreature = null;
      this.noteLoad(m.load);
      for (const l of this.listeners) l(m);
    } else if (m.type === "ctelemetry") {
      this.latestCreature = m;
      this.latest = null;
      this.noteLoad(m.load);
    } else if (m.type === "planned") {
      this.lastPlan = m;
      for (const l of this.plannedListeners) l(m);
    } else if (m.type === "ready") {
      for (const l of this.readyListeners) l();
    } else if (m.type === "error") {
      this.error = m.message;
      console.error("worklet:", m.message);
    }
  }

  onTelemetry(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  onPlanned(l: (p: PlannedMessage) => void): () => void {
    this.plannedListeners.add(l);
    return () => this.plannedListeners.delete(l);
  }

  /** Send a gestural plan (times relative to now). Returns the request id. */
  schedule(plan: { gestures: import("../wasm/types.ts").Gesture[]; segments: import("../wasm/types.ts").PlannedSegment[]; end: number }, replace = true): number {
    const id = ++this.speakId;
    this.send({ type: "schedule", gestures: plan.gestures, segments: plan.segments, end: plan.end, id, replace });
    return id;
  }

  /** Switch voice ("human" or a creature id), starting the creature from the given settings. */
  setCreature(id: string, params?: Record<string, number>): void {
    // settings still waiting for the next frame belong to the previous creature
    this.pendingCParams = null;
    this.send({ type: "creature", id, params });
  }

  setCreatureParams(values: Record<string, number>): void {
    this.pendingCParams = { ...(this.pendingCParams ?? {}), ...values };
    this.scheduleFlush();
  }

  playCall(id: string, pitch?: number, amp?: number): void {
    this.send({ type: "call", id, pitch, amp });
  }

  stopCall(): void {
    this.send({ type: "stopCall" });
  }

  stopSpeech(): void {
    this.send({ type: "stopSpeech" });
  }

  /** The output's RMS over the fast analyser's window (~40 ms); 0 before start(). */
  level(): number {
    const a = this.analyserFast;
    if (!a) return 0;
    const buf = new Float32Array(a.fftSize);
    a.getFloatTimeDomainData(buf);
    let s = 0;
    for (const x of buf) s += x * x;
    return Math.sqrt(s / buf.length);
  }

  /**
   * Resolves once the output has stayed below `floor` for `hold` s, or after `max` s: a clip
   * mustn't open on the tail of what was sounding (a stopped song keeps ringing for a moment).
   */
  async quiet(floor = 0.003, hold = 0.15, max = 2): Promise<void> {
    const t0 = performance.now();
    let since = t0;
    while (performance.now() - t0 < max * 1000) {
      const t = performance.now();
      if (this.level() >= floor) since = t;
      else if (t - since >= hold * 1000) return;
      await new Promise((r) => setTimeout(r, 20));
    }
  }

  onReady(l: () => void): () => void {
    this.readyListeners.add(l);
    return () => this.readyListeners.delete(l);
  }

  private send(m: ToWorklet): void {
    if (inPlugin) sendToPlugin(m);
    else this.node?.port.postMessage(m);
  }

  // ---- what only the plugin has: there the keys, the lyrics and the mode belong to Rust

  /** A key on the page went down or up. */
  pluginNote(midi: number, on: boolean, velocity?: number): void {
    ipc({ type: "note", midi, on, velocity });
  }

  pluginLyrics(text: string): void {
    ipc({ type: "lyrics", text });
  }

  pluginCursor(index: number): void {
    ipc({ type: "cursor", index });
  }

  pluginMode(mode: string): void {
    ipc({ type: "mode", mode });
  }

  pluginHold(on: boolean): void {
    ipc({ type: "hold", on });
  }

  /** The text to say changed: Rust plans its lines. */
  pluginSpeech(edit: { text?: string; ipa?: boolean; rate?: number }): void {
    ipc({ type: "speech", ...edit });
  }

  pluginSpeechCursor(index: number): void {
    ipc({ type: "speechCursor", index });
  }

  /** Say the next line, as a note at the natural pitch would. */
  pluginSpeak(): void {
    ipc({ type: "speak" });
  }

  /** Say this once. */
  pluginSayIpa(ipa: string): void {
    ipc({ type: "sayIpa", ipa });
  }

  /** Whether the sequencer follows the host's transport. */
  pluginFollowHost(on: boolean): void {
    ipc({ type: "followHost", on });
  }

  /** Route the voice through the close-mic chain (on) or straight to the output (off). */
  setMic(on: boolean): void {
    if (inPlugin) return ipc({ type: "mic", on });
    if (on === this.micOn) return;
    this.micOn = on;
    if (!this.node || !this.master || !this.micIn) return;
    this.node.disconnect();
    this.node.connect(on ? this.micIn : this.master);
  }

  setBeat(m: { steps?: import("../wasm/types.ts").BeatStep[] | (import("../wasm/types.ts").BeatStep | null)[]; bpm?: number; swing?: number; playing?: boolean; preset?: string | null }): void {
    this.send({ type: "beat", ...m });
  }

  /** End the sung note (open-ended gestures) with this tag. */
  release(tag: number): void {
    this.send({ type: "release", tag });
  }

  /** Gestures planned on the main thread, times relative to arrival + lead (live playing). */
  scheduleGestures(gestures: import("../wasm/types.ts").Gesture[], lead = 0.006): void {
    this.send({ type: "schedule", gestures, segments: [], end: 0, id: ++this.speakId, replace: false, lead });
  }

  beatHit(id: string, note?: number, accent?: number): void {
    this.send({ type: "beatHit", id, note, accent });
  }

  setVolume(v: number): void {
    if (inPlugin) setParam("gain", volumeToDb(v));
    else if (this.master && this.ctx) this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.02);
  }

  setBody(body: BodyMix): void {
    this.send({ type: "body", body });
  }

  gate(level: number): void {
    this.send({ type: "gate", level });
  }

  /** Coalesced: many calls within a frame become one message. (In the plugin Rust merges them.) */
  setArtic(values: Partial<HumanArtic>, tipFree?: boolean): void {
    if (inPlugin) return this.send({ type: "artic", values, tipFree });
    this.pendingArtic = { ...(this.pendingArtic ?? {}), ...values };
    if (tipFree !== undefined) this.pendingTipFree = tipFree;
    this.scheduleFlush();
  }

  setSource(values: Partial<VoiceSource>): void {
    if (inPlugin) return this.send({ type: "source", values });
    this.pendingSource = { ...(this.pendingSource ?? {}), ...values };
    this.scheduleFlush();
  }

  private scheduleFlush(): void {
    if (this.flushScheduled) return;
    this.flushScheduled = true;
    const flush = () => {
      this.flushScheduled = false;
      if (this.pendingArtic || this.pendingTipFree !== undefined) {
        this.send({ type: "artic", values: this.pendingArtic ?? {}, tipFree: this.pendingTipFree });
        this.pendingArtic = null;
        this.pendingTipFree = undefined;
      }
      if (this.pendingSource) {
        this.send({ type: "source", values: this.pendingSource });
        this.pendingSource = null;
      }
      if (this.pendingCParams) {
        this.send({ type: "cparams", values: this.pendingCParams });
        this.pendingCParams = null;
      }
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(flush);
    else setTimeout(flush, 16);
  }
}

export const engine = new AudioEngine();
