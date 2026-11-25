import { memo, useState } from "react";
import { useStore } from "../store.ts";
import { CREATURES, creatureById } from "../../wasm/engine.ts";
import type { CreatureDef } from "../../wasm/types.ts";
import { useLiveCreature } from "../creatureLive.ts";
import { Keys, MidiButton } from "./Keys.tsx";
import { ActionBar, PrimaryButton } from "./ActionBar.tsx";
import { engine } from "../../audio/AudioEngine.ts";
import { Menu } from "./Controls.tsx";
import { Chip, Screen } from "./Screen.tsx";
import { EncoderPages, type EncoderPage } from "./Encoder.tsx";
import { Spectrogram } from "./Spectrogram.tsx";
import { CreatureTubes } from "./CreatureTubes.tsx";

/** Thumbnail: the creature's outline in blue with its airway traced in amber. */
const Thumb = memo(function Thumb({ def }: { def: CreatureDef }) {
  const v = def.visual;
  const vw = v.view.x1 - v.view.x0, vh = v.view.y1 - v.view.y0;
  const pad = Math.max(vw, vh) * 0.06;
  const stroke = Math.max(vw, vh) * 0.012;
  const pts = (p: [number, number][]) => p.map(([x, y]) => `${x},${-y}`).join(" ");
  return (
    <svg className="thumb" viewBox={`${v.view.x0 - pad} ${-v.view.y1 - pad} ${vw + 2 * pad} ${vh + 2 * pad}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <polygon points={pts(v.silhouette)} fill="none" stroke="var(--blue)" strokeWidth={stroke} strokeLinejoin="round" />
      {v.tubes.map((t) => (
        <polyline key={t.tube} points={pts(t.path)} fill="none" stroke="var(--amber)" strokeWidth={stroke * 0.9} strokeLinecap="round" strokeLinejoin="round" />
      ))}
    </svg>
  );
});

function Gallery() {
  const creature = useStore((s) => s.creature);
  const setCreature = useStore((s) => s.setCreature);
  return (
    <div className="specimens">
      {CREATURES.map((c) => (
        <button key={c.id} className={`specimen${c.id === creature ? " on" : ""}`} onClick={() => setCreature(c.id)} aria-pressed={c.id === creature}>
          <Thumb def={c} />
          <span className="specimen-name">{c.name}</span>
          <span className="specimen-latin">{c.latin}</span>
        </button>
      ))}
    </div>
  );
}

/**
 * Bestiary's main buttons: the animal's calls (each plays it; the one sounding reads Stop), and
 * Breathe, which keeps the breath on to shape the sound with the controls. Space stops the call
 * sounding, or plays the last one again. (The animal's description is written under its name, on
 * the creature screen.)
 */
function CallActions({ def }: { def: CreatureDef }) {
  const playCall = useStore((s) => s.playCall);
  const hold = useStore((s) => s.hold);
  const toggleBreath = useStore((s) => s.toggleBreath);
  const sounding = useLiveCreature(def.id)?.call ?? null;
  const [last, setLast] = useState<string | null>(null);
  const press = (id: string) => {
    if (sounding === id) return engine.stopCall();
    setLast(id);
    playCall(id);
  };
  const again = sounding ?? (def.calls.find((c) => c.id === last) ?? def.calls[0])?.id;
  return (
    <ActionBar label="Calls" onSpace={() => again && press(again)}>
      {def.calls.map((c) => (
        <PrimaryButton key={c.id} className="act-call" label={c.name} playing={sounding === c.id} pressed={last === c.id} title={c.hint} onClick={() => press(c.id)} />
      ))}
      <Chip on={hold} onClick={() => void toggleBreath()} title="Keep the breath on and shape the sound with the controls">
        Breathe
      </Chip>
    </ActionBar>
  );
}

/** What the keys play: the held voice at the key's pitch, or one of the calls. */
function KeysMode({ def }: { def: CreatureDef }) {
  const keyCall = useStore((s) => s.keyCall);
  const setKeyCall = useStore((s) => s.setKeyCall);
  return (
    <Menu
      label="What the keys play"
      value={keyCall ?? ""}
      options={[{ value: "", label: "Held voice" }, ...def.calls.map((c) => ({ value: c.id, label: c.name }))]}
      onChange={(id) => setKeyCall(id || null)}
    />
  );
}

function fmt(v: number, p: CreatureDef["params"][number]): { value: string; unit: string } {
  const digits = p.digits ?? (p.max - p.min <= 2 ? 2 : v < 100 && p.log ? 1 : 0);
  return { value: v.toFixed(digits), unit: p.unit ?? "" };
}

function Params({ def }: { def: CreatureDef }) {
  const cparams = useStore((s) => s.cparams[def.id]);
  const setCParam = useStore((s) => s.setCParam);
  const live = useLiveCreature(def.id);
  // while a call plays, the encoders follow its automation; afterwards they return to yours
  const calling = !!live?.call;
  const pages: EncoderPage[] = [];
  for (let i = 0; i < def.params.length; i += 4) {
    pages.push({
      name: String(pages.length + 1),
      encoders: def.params.slice(i, i + 4).map((p) => {
        const mine = cparams?.[p.key] ?? p.default;
        return { label: p.label, value: calling ? (live!.params[p.key] ?? mine) : mine, min: p.min, max: p.max, log: p.log, onChange: (v: number) => setCParam(p.key, v), format: (v: number) => fmt(v, p) };
      }),
    });
  }
  return (
    <>
      {/* a sentence, too long for the header: it wraps here instead of being cut off there */}
      {def.role && <p className="cparams-role">{def.role}</p>}
      <EncoderPages key={def.id} label={`${def.name} controls`} pages={pages} />
    </>
  );
}

/** Bestiary's right column: the calls, the creatures, the waterfall, the controls, the airway, the keys. */
export function BestiaryRight() {
  const creature = useStore((s) => s.creature);
  const def = creatureById(creature);
  return (
    <div className="right right-bestiary">
      {def && <CallActions def={def} />}
      <Screen className="s-gallery" title="Creatures">
        <Gallery />
      </Screen>
      <Screen watch className="s-wf" title="Spectrogram">
        <Spectrogram />
      </Screen>
      <Screen className="s-cparams" title="Controls">
        {def && <Params def={def} />}
      </Screen>
      <Screen watch className="s-airway" title="Airway" sub="Each tube unrolled">
        <CreatureTubes />
      </Screen>
      <Screen className="s-keys s-keys-compact" title="Keys" tools={<>{def && <KeysMode def={def} />}<MidiButton /></>}>
        <Keys />
      </Screen>
    </div>
  );
}
