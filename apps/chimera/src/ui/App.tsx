import { lazy, Suspense, useEffect, useState } from "react";
import { useStore } from "./store.ts";
import { engine } from "../audio/AudioEngine.ts";
import { inPlugin } from "../audio/pluginBridge.ts";
import { AnatomyView } from "./components/AnatomyView.tsx";
import { TubeView } from "./components/TubeView.tsx";
import { Stroboscope } from "./components/Stroboscope.tsx";
import { SpectrumView } from "./components/SpectrumView.tsx";
import { VowelMap } from "./components/VowelMap.tsx";
import { VoiceControls } from "./components/VoiceControls.tsx";
import { Keys, MidiButton } from "./components/Keys.tsx";
import { ActionBar, PrimaryButton } from "./components/ActionBar.tsx";
import { Segmented, Toggle } from "./components/Controls.tsx";
import { Chip, Screen } from "./components/Screen.tsx";
import { SpeakRight, speechInfo, speechSlots, speechWords } from "./components/SpeakPanel.tsx";
import { BestiaryRight } from "./components/BestiaryPanel.tsx";
import { CreatureView } from "./components/CreatureView.tsx";
import { BeatboxRight, BeatOverlay, beatInfo, beatSlots } from "./components/BeatboxPanel.tsx";
import { SingRight, singInfo, singWords } from "./components/SingPanel.tsx";
import type { Mode } from "./store.ts";
import type { BodyName } from "../worklet/protocol.ts";
import { BodyPad } from "./components/BodyPad.tsx";
import { ResizeGrip } from "./components/ResizeGrip.tsx";
import { presetOf, PRESETS } from "./body.ts";

// three.js only loads when the 3D view is first shown
const AnatomyView3D = lazy(() => import("./components/AnatomyView3D.tsx").then((m) => ({ default: m.AnatomyView3D })));
// the duet page (`?show=<id>`): two 3D heads replaying a pre-rendered song
const DuetPage = lazy(() => import("./components/DuetPage.tsx").then((m) => ({ default: m.DuetPage })));

/** the show to play, from `?show=<id>` (a plain folder name under public/shows) */
function showId(): string | null {
  const id = new URLSearchParams(window.location.search).get("show");
  return id && /^[\w-]+$/.test(id) ? id : null;
}

function TopBar() {
  const started = useStore((s) => s.started);
  const volume = useStore((s) => s.volume);
  const setVolume = useStore((s) => s.setVolume);
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const light = useStore((s) => s.lightAudio);
  const [load, setLoad] = useState(0);
  const [stopped, setStopped] = useState<string | null>(null);
  useEffect(() => {
    const id = setInterval(() => {
      setLoad(engine.load);
      if (inPlugin) setStopped(engine.error);
    }, 500);
    return () => clearInterval(id);
  }, []);
  return (
    <header className="topbar">
      <div className="brand">
        <span className="wordmark">Chimera</span>
        <span className="tagline">A voice grown from a simulated airway</span>
      </div>
      <Segmented<Mode>
        label="Mode"
        value={mode}
        onChange={setMode}
        options={
          inPlugin
            ? [
                { value: "sing", label: "Sing" },
                { value: "play", label: "Play" },
                { value: "speak", label: "Speak" },
                { value: "beatbox", label: "Beatbox" },
              ]
            : [
                { value: "play", label: "Play" },
                { value: "speak", label: "Speak" },
                { value: "sing", label: "Sing" },
                { value: "bestiary", label: "Bestiary" },
                { value: "beatbox", label: "Beatbox" },
              ]
        }
      />
      <div className="topbar-right">
        <label className="volume">
          <span>Volume</span>
          <input type="range" min={0} max={1} step={0.01} value={volume} onChange={(e) => setVolume(Number(e.target.value))} style={{ "--pos": `${volume * 100}%` } as React.CSSProperties} />
        </label>
        {stopped ? (
          <span className="load error">{stopped}</span>
        ) : started ? (
          <span className="load" title={"Share of the audio thread's time budget used by the simulation" + (light ? ". This device couldn't keep up at 48 kHz, so the voice runs at 32 kHz: lighter, a little duller. Reload to try 48 kHz again." : "")}>
            Engine load {Math.round(load * 100)} %{light && " · 32 kHz"}
          </span>
        ) : null}
        {!inPlugin && (
          <a className="plugin-link" href="/plugin/" title="CHIMERA as a CLAP, VST3 and AU plugin for your DAW">
            Plugin
          </a>
        )}
        {!inPlugin && (
          <a className="repo-link" href="https://github.com/jamesondh/chimera-vocal-synth" target="_blank" rel="noreferrer" aria-label="Source code on GitHub" title="Source code on GitHub">
            <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">
              <path
                fill="currentColor"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
              />
            </svg>
          </a>
        )}
      </div>
    </header>
  );
}

function ViewToggles() {
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const bestiary = useStore((s) => s.mode === "bestiary");
  return (
    <div className="view-toggles">
      {!bestiary && <Toggle label="3D" checked={view.threeD} onChange={(v) => setView({ threeD: v })} />}
      <Toggle label="Sound pressure" checked={view.pressure} onChange={(v) => setView({ pressure: v })} />
      {!bestiary && <Toggle label="Airflow" checked={view.particles} onChange={(v) => setView({ particles: v })} />}
      <Toggle label={bestiary ? "Labels" : "Measurements"} checked={view.calipers} onChange={(v) => setView({ calipers: v })} />
      {!bestiary && <Toggle label="Grid" checked={view.grid} onChange={(v) => setView({ grid: v })} />}
    </div>
  );
}

/** The body, in the head's header: three presets and the triangle between them. */
function BodyControl() {
  const body = useStore((s) => s.body);
  const setBody = useStore((s) => s.setBody);
  return (
    <div className="body-control">
      <Segmented<BodyName>
        label="Body"
        value={presetOf(body)}
        onChange={(b) => setBody(PRESETS[b])}
        options={[
          { value: "male", label: "Male" },
          { value: "female", label: "Female" },
          { value: "child", label: "Child" },
        ]}
      />
      <BodyPad />
    </div>
  );
}

/** The tall left screen: the face (or the creature) with its overlays. No title: it shows itself. */
function LeftScreen() {
  const mode = useStore((s) => s.mode);
  const threeD = useStore((s) => s.view.threeD);
  const bestiary = mode === "bestiary";
  const Face = threeD ? AnatomyView3D : AnatomyView;
  return (
    <Screen className="left" focus lead={bestiary ? undefined : <BodyControl />} label={bestiary ? "Live creature airway" : "Live vocal tract"} tools={<ViewToggles />}>
      <Suspense fallback={null}>
        {bestiary ? <CreatureView /> : mode === "speak" ? <Face slots={speechSlots} info={speechInfo} overlay={speechWords} /> : mode === "sing" ? <Face info={singInfo} overlay={singWords} /> : mode === "beatbox" ? <Face slots={beatSlots} info={beatInfo} showFormants={false} /> : <Face />}
      </Suspense>
      {mode === "beatbox" && <BeatOverlay />}
    </Screen>
  );
}

/** Play's main button: breathe (the voice sounds while it's on), and Whisper beside it. */
function PlayActions() {
  const hold = useStore((s) => s.hold);
  const toggleBreath = useStore((s) => s.toggleBreath);
  const voiced = useStore((s) => s.source.voiced);
  const setSource = useStore((s) => s.setSource);
  return (
    <ActionBar label="Voice" onSpace={() => void toggleBreath()}>
      <PrimaryButton label="Breathe" icon="breath" playing={hold} onClick={() => void toggleBreath()} />
      <Chip on={!voiced} onClick={() => setSource({ voiced: !voiced })}>
        Whisper
      </Chip>
    </ActionBar>
  );
}

function PlayRight() {
  const f0 = useStore((s) => s.source.f0);
  return (
    <div className="right right-play">
      <PlayActions />
      <Screen watch className="s-tube" title="Tube" sub="Glottis to lips">
        <TubeView />
      </Screen>
      <Screen watch className="s-folds" title="Vocal folds" sub="Slowed 100×">
        <Stroboscope />
      </Screen>
      <Screen watch className="s-spec" title="Spectrum">
        <SpectrumView />
      </Screen>
      <Screen className="s-vowels" title="Vowels" sub="F1 / F2">
        <VowelMap />
      </Screen>
      <Screen className="s-voice" title="Voice">
        <VoiceControls />
      </Screen>
      <Screen className="s-keys" title="Keys" sub={`${Math.round(f0)} Hz`} tools={<MidiButton />}>
        <Keys />
      </Screen>
    </div>
  );
}

export function App() {
  const show = showId();
  if (show)
    return (
      <Suspense fallback={null}>
        <DuetPage id={show} />
      </Suspense>
    );
  return <Wall />;
}

function Wall() {
  const mode = useStore((s) => s.mode);
  return (
    <div className={`app text-colors mode-${mode}${inPlugin ? " in-plugin" : ""}`}>
      <TopBar />
      <main className={`wall wall-${mode}`}>
        <LeftScreen />
        {mode === "play" ? <PlayRight /> : mode === "speak" ? <SpeakRight /> : mode === "sing" ? <SingRight /> : mode === "beatbox" ? <BeatboxRight /> : <BestiaryRight />}
      </main>
      {inPlugin && <ResizeGrip />}
    </div>
  );
}
