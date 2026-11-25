import { useEffect, useRef, useState } from "react";
import { useStore } from "../store.ts";
import { engine } from "../../audio/AudioEngine.ts";
import { inPlugin } from "../../audio/pluginBridge.ts";
import { BEAT_SOUNDS, BEAT_PRESETS, formatBeat, parseBeat, type Airstream } from "../../wasm/engine.ts";
import { Keys, MidiButton } from "./Keys.tsx";
import { ActionBar, PrimaryButton } from "./ActionBar.tsx";
import { Menu } from "./Controls.tsx";
import { Chip, Screen } from "./Screen.tsx";
import { EncoderPages } from "./Encoder.tsx";
import { Spectrogram } from "./Spectrogram.tsx";
import type { InfoFn, SlotsFn } from "./AnatomyView.tsx";
import { V } from "../colors.ts";
import { MAX_BEAT_STEPS } from "../../wasm/wire.ts";

const AIR: Record<Airstream, string> = {
  pulmonic: "Breath out",
  ejective: "Larynx piston",
  click: "Tongue suction",
  ingressive: "Breath in",
  voiced: "Voice",
};

const NOTE = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const noteName = (m: number) => `${NOTE[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`;

/** Step under the playhead, polled from telemetry (re-renders only when it changes). */
export function useBeatStep(): number {
  const [step, setStep] = useState(-1);
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const s = engine.latest?.beatStep ?? -1;
      setStep((prev) => (prev === s ? prev : s));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
  return step;
}

/** Beatbox's readouts under the chin: the articulators a mouth drum uses */
export const beatSlots: SlotsFn = (_m, a, f) => [
  { label: "Jaw", value: a.jaw.toFixed(1), unit: "cm", color: V.red },
  { label: "Lips", value: a.lipAp < 0.05 ? "Sealed" : a.lipWidth < 0.55 ? "Round" : "Open", color: V.pink },
  { label: "Larynx", value: `${a.larynx >= 0 ? "+" : ""}${a.larynx.toFixed(1)}`, unit: "cm", color: V.green },
  { label: "Soft palate", value: f.velumPort > 0.02 ? "Open" : "Shut", color: V.white },
];
/** the big symbol owns the top right, so nothing else is written there */
export const beatInfo: InfoFn = () => [];

function Pads() {
  const beatHit = useStore((s) => s.beatHit);
  const [lit, setLit] = useState<string | null>(null);
  const timer = useRef(0);
  const hit = (id: string) => {
    beatHit(id);
    setLit(id);
    clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setLit(null), 160);
  };
  const hitRef = useRef(hit);
  hitRef.current = hit;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target as HTMLElement;
      if (t && t.tagName === "SELECT") return;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA") && (t as HTMLInputElement).type !== "range" && (t as HTMLInputElement).type !== "checkbox") return;
      const i = "1234567890".indexOf(e.key);
      if (i >= 0 && i < BEAT_SOUNDS.length) {
        hitRef.current(BEAT_SOUNDS[i].id);
        e.preventDefault();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return (
    <div className="pads">
      {BEAT_SOUNDS.map((s, i) => (
        <button
          key={s.id}
          className={`pad pad-${s.role}${lit === s.id ? " lit" : ""}`}
          onPointerDown={() => hit(s.id)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              hit(s.id);
            }
          }}
          title={s.hint}
          aria-label={`${s.name}: ${s.hint}. Key ${(i + 1) % 10}`}
        >
          <span className="pad-sym">{s.id}</span>
          <span className="pad-name">{s.name}</span>
          <span className="pad-air">{AIR[s.airstream]}</span>
          <span className="pad-key" aria-hidden="true">
            {(i + 1) % 10}
          </span>
        </button>
      ))}
    </div>
  );
}

function Sequencer() {
  const beat = useStore((s) => s.beat);
  const toggle = useStore((s) => s.toggleBeatStep);
  const now = useBeatStep();
  return (
    <div className="seq-grid" style={{ "--steps": beat.steps.length } as React.CSSProperties} role="grid" aria-label="Beat pattern: one sound per step">
      {BEAT_SOUNDS.map((s) => (
        <div className="seq-row" role="row" key={s.id}>
          <span className="seq-label" role="rowheader">
            <b>{s.id}</b> {s.name}
          </span>
          {beat.steps.map((st, i) => {
            const on = st?.id === s.id;
            return (
              <button
                key={i}
                role="gridcell"
                className={`cell${on ? " on" : ""}${i === now ? " now" : ""}${i % 4 === 0 ? " downbeat" : ""}`}
                onClick={() => toggle(i, s.id)}
                aria-pressed={on}
                aria-label={`${s.name}, step ${i + 1}`}
              >
                {on && s.id === "m" && st?.note !== undefined ? <span className="cell-note">{noteName(st.note)}</span> : null}
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

/** Beatbox's main button: play or stop the beat; the pattern and the close-mic effect beside it. */
function BeatActions() {
  const playing = useStore((s) => s.beat.playing);
  const preset = useStore((s) => s.beat.preset);
  const setPlaying = useStore((s) => s.setBeatPlaying);
  const follow = useStore((s) => s.beat.follow);
  const following = useStore((s) => s.beat.following);
  const setFollow = useStore((s) => s.setBeatFollow);
  const loadPreset = useStore((s) => s.loadBeatPreset);
  const mic = useStore((s) => s.mic);
  const setMic = useStore((s) => s.setMic);
  const press = () => !following && setPlaying(!playing);
  return (
    <ActionBar label="Beat" onSpace={press}>
      <PrimaryButton label="Play beat" playing={playing} busy={following ? "With host" : null} title={following ? "The host's transport is running the beat" : undefined} onClick={press} />
      <Menu
        label="Pattern"
        value={preset ?? ""}
        options={[...(preset ? [] : [{ value: "", label: "Your beat", disabled: true }]), ...BEAT_PRESETS.map((p) => ({ value: p.name, label: p.name }))]}
        onChange={(name) => loadPreset(name)}
      />
      {/* an effect on the sound, not the microphone: "Close mic" read as "shut the mic" */}
      <Chip on={mic} onClick={() => setMic(!mic)} title="Close-mic effect: the bass, presence and squash of a mic held to the lips">
        Mic FX
      </Chip>
      {inPlugin && (
        <Chip on={follow} onClick={() => setFollow(!follow)} title="Run the pattern with the host's transport, at its tempo">
          Follow
        </Chip>
      )}
    </ActionBar>
  );
}

function Transport() {
  const beat = useStore((s) => s.beat);
  const setTempo = useStore((s) => s.setBeatTempo);
  return (
    <EncoderPages
      label="Transport"
      columns={2}
      pages={[
        {
          name: "Tempo",
          encoders: [
            { label: "Tempo", value: beat.following ? beat.hostBpm : beat.bpm, min: 60, max: 240, step: 1, onChange: (v) => setTempo({ bpm: Math.round(v) }), format: (v) => ({ value: String(Math.round(v)), unit: "bpm" }) },
            { label: "Swing", value: beat.swing, min: 0, max: 0.6, onChange: (v) => setTempo({ swing: v }), format: (v) => ({ value: String(Math.round(v * 100)), unit: "%" }) },
          ],
        },
      ]}
    />
  );
}

/** The pattern typed out, in the sequencer's header: edit either one. */
function BeatText() {
  const steps = useStore((s) => s.beat.steps);
  const setSteps = useStore((s) => s.setBeatSteps);
  const [text, setText] = useState(() => formatBeat(steps));
  const editing = useRef(false);
  useEffect(() => {
    if (!editing.current) setText(formatBeat(steps));
  }, [steps]);
  return (
    <label className="sbn">
      <span className="dim">Beat as text</span>
      <input
        value={text}
        title={`B kick, t hat, Pf snare, K rimshot, ^K inward, ! click, m hum (m:43 sets its note), . rest. Up to ${MAX_BEAT_STEPS} steps.`}
        placeholder="B . t . | Pf . t ."
        spellCheck={false}
        onFocus={() => (editing.current = true)}
        onBlur={() => {
          editing.current = false;
          setText(formatBeat(steps));
        }}
        onChange={(e) => {
          setText(e.target.value);
          const parsed = parseBeat(e.target.value);
          if (parsed.length) setSteps(parsed);
        }}
      />
    </label>
  );
}

/** Beatbox mode's right column: the beat button, the step sequencer (and the beat as text), the waterfall, the pads, tempo, the keys (hum bass). */
export function BeatboxRight() {
  return (
    <div className="right right-beatbox">
      <BeatActions />
      <Screen className="s-seq" title="Step sequencer" tools={<BeatText />}>
        <Sequencer />
      </Screen>
      <Screen watch className="s-wf" title="Spectrogram">
        <Spectrogram />
      </Screen>
      <Screen className="s-pads" title="Pads">
        <Pads />
      </Screen>
      <Screen className="s-transport" title="Tempo">
        <Transport />
      </Screen>
      {/* the keys hum a bass note under the beat */}
      <Screen className="s-keys s-keys-compact" title="Bass" tools={<MidiButton />}>
        <Keys />
      </Screen>
    </div>
  );
}

/** The sound being made, in big outlined letters over the face, fading after each hit. */
export function BeatOverlay() {
  const steps = useStore((s) => s.beat.steps);
  const playing = useStore((s) => s.beat.playing);
  const now = useBeatStep();
  const cell = now >= 0 ? steps[now] : null;
  if (!playing || !cell) return null;
  const s = BEAT_SOUNDS.find((x) => x.id === cell.id);
  return (
    <div className="beat-overlay" data-layout="overlay" aria-hidden="true" key={now}>
      <span className="beat-sym">{cell.id}</span>
      <span className="beat-name">{s?.name}</span>
      {s && <span className="beat-air">{AIR[s.airstream]}</span>}
      <span className="beat-step">
        Step {String(now + 1).padStart(2, "0")} / {steps.length}
      </span>
    </div>
  );
}
