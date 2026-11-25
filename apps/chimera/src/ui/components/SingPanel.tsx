import { useEffect, useRef, useState } from "react";
import { inPlugin } from "../../audio/pluginBridge.ts";
import { useStore } from "../store.ts";
import { useSing, playTune, stopSinging, soundingSyllable, sungNow, TUNES, ensureDictionary } from "../singing.ts";
import { engine } from "../../audio/AudioEngine.ts";
import { canRecord, Clip } from "../../song/record.ts";
import { mouthWords } from "../mouthWords.ts";
import { V } from "../colors.ts";
import { Keys, MidiButton } from "./Keys.tsx";
import { ActionBar, PrimaryButton } from "./ActionBar.tsx";
import { Menu } from "./Controls.tsx";
import { PianoRoll } from "./PianoRoll.tsx";
import { SpectrumView } from "./SpectrumView.tsx";
import { Chip, Screen } from "./Screen.tsx";
import { LangPick } from "./LangPick.tsx";
import { EncoderPages } from "./Encoder.tsx";
import { noteName } from "./VoiceControls.tsx";
import type { InfoFn } from "./AnatomyView.tsx";

/** epilarynx narrowing for the singer's ring (measured: +3 to +12 dB at 2.4–3.6 kHz, adult male) */
const RING = 0.85;

/** the syllable being sung now: the tune's, or the one the held key is singing */
function syllableNow(): string | null {
  const s = useSing.getState();
  if (s.playing) return s.syllables[s.at]?.text ?? null;
  return soundingSyllable();
}

/** the syllable being sung, at the mouth */
export const singWords = mouthWords(sungNow, V.green);

/** Sing mode's lines over the face: the note, the syllable, the ring */
export const singInfo: InfoFn = (_m, _f, tele) => {
  const out: string[] = [];
  if (tele && tele.t.voicing > 0.3) out.push(`Note ${noteName(tele.t.f0)} ${tele.t.f0.toFixed(0)} Hz`);
  const syl = syllableNow();
  if (syl) out.push(`Syllable ${syl}`);
  out.push(useStore.getState().artic.epilarynx >= RING - 0.05 ? "Ring on" : "Ring off");
  return out;
};

function Lyrics() {
  const lyrics = useSing((s) => s.lyrics);
  const lang = useSing((s) => s.lang);
  const syllables = useSing((s) => s.syllables);
  const cursor = useSing((s) => s.cursor);
  const demo = useSing((s) => s.playing);
  const sungAt = useSing((s) => s.at);
  const setLyrics = useSing((s) => s.setLyrics);
  const setCursor = useSing((s) => s.setCursor);
  const at = demo ? sungAt : -1;
  const done = demo ? sungAt : cursor;
  return (
    <div className="lyrics">
      <textarea
        value={lyrics}
        onChange={(e) => setLyrics(e.target.value)}
        onKeyDown={(e) => e.stopPropagation()}
        rows={Math.min(4, lyrics.split("\n").length)}
        lang={lang}
        aria-label="Lyrics to sing"
        placeholder="Type the words to sing"
        title="Sing sings these words to the tune, a line to each of its phrases. Each key sings the next syllable; hold a key while you press the next one to sing legato. A lone “-” between words holds a vowel over several notes; hyphens split a word your way (“ti-ger”)."
      />
      <div className="syllables" role="group" aria-label="Syllables. Each note sings the marked one; choose another to start there.">
        {syllables.map((s, i) => (
          <span key={i} className={`syl-wrap${s.wordEnd ? " word-end" : ""}`}>
            <button
              className={`syl${!demo && i === cursor ? " next" : ""}${i === at ? " now" : ""}${i < done ? " sung" : ""}${s.melisma ? " held" : ""}`}
              onClick={() => setCursor(i)}
              aria-current={!demo && i === cursor ? "step" : undefined}
              title={s.melisma ? "Held vowel: carries the previous syllable onto this note" : `Sing from “${s.text}”`}
            >
              {s.text}
            </button>
            {s.phraseEnd && (
              <span className="breath-mark" aria-label="breath">
                ’
              </span>
            )}
          </span>
        ))}
        {!syllables.length && <span className="dim">No words: the keys sing “ah”.</span>}
        <Chip toggle={false} className="from-start" onClick={() => setCursor(0)} disabled={cursor === 0 && !demo}>
          From the start
        </Chip>
      </div>
    </div>
  );
}

/** The language Latin-script words are read in, in Sing and Speak (other scripts are read by their own). */
function LyricsLang() {
  const lang = useSing((s) => s.lang);
  const setLang = useSing((s) => s.setLang);
  return <LangPick value={lang} onChange={setLang} />;
}

function SongControls() {
  const source = useStore((s) => s.source);
  const setSource = useStore((s) => s.setSource);
  const epilarynx = useStore((s) => s.artic.epilarynx);
  const setArtic = useStore((s) => s.setArtic);
  const vibrato = useSing((s) => s.vibrato);
  const setVibrato = useSing((s) => s.setVibrato);
  const tempo = useSing((s) => s.tempo);
  const setTempo = useSing((s) => s.setTempo);
  const key = useSing((s) => s.key);
  const setKey = useSing((s) => s.setKey);
  return (
    <EncoderPages
      label="Song voice"
      pages={[
        {
          name: "Song",
          encoders: [
            { label: "Vibrato", value: vibrato, min: 0, max: 80, step: 1, onChange: (v) => setVibrato(Math.round(v)), format: (v) => ({ value: String(Math.round(v)), unit: "cents" }) },
            { label: "Rate", value: source.vibratoRate, min: 3.5, max: 7.5, onChange: (v) => setSource({ vibratoRate: v }), format: (v) => ({ value: v.toFixed(1), unit: "Hz" }) },
            { label: "Breathy", value: source.breathiness, min: 0, max: 1, onChange: (v) => setSource({ breathiness: v }), format: (v) => ({ value: String(Math.round(v * 100)), unit: "%" }) },
            // the singer's ring narrows the tube just above the folds, adding the bright band near 3 kHz
            { label: "Ring", value: epilarynx, min: 0, max: 1, onChange: (v) => setArtic({ epilarynx: v }), format: (v) => ({ value: String(Math.round(v * 100)), unit: "%" }) },
          ],
        },
        // the tune's speed and key, from the next time Sing is pressed (the plugin has no tunes: the host plays)
        ...(inPlugin
          ? []
          : [
              {
                name: "Tune",
                encoders: [
                  { label: "Tempo", value: tempo, min: 0.5, max: 1.5, step: 0.05, onChange: (v: number) => setTempo(Math.round(v * 20) / 20), format: (v: number) => ({ value: String(Math.round(v * 100)), unit: "%" }) },
                  { label: "Key", value: key, min: -12, max: 12, step: 1, onChange: (v: number) => setKey(Math.round(v)), format: (v: number) => ({ value: (v > 0.5 ? "+" : "") + String(Math.round(v)), unit: "semitones" }) },
                ],
              },
            ]),
        {
          name: "Tone",
          encoders: [{ label: "Bright", value: source.brightness, min: 0, max: 1, onChange: (v) => setSource({ brightness: v }), format: (v) => ({ value: String(Math.round(v * 100)), unit: "%" }) }],
        },
      ]}
    />
  );
}

/**
 * Sing's main button: sing the lyrics to the chosen tune, or stop. A song getting ready (audio
 * starting, the dictionary loading) counts as playing: Stop cancels it, and choosing another tune
 * replaces it. Record video sings it again into a clip of the face screen (Share and Save once it's done).
 */
function SingActions() {
  const start = useStore((s) => s.start);
  const playing = useSing((s) => s.playing);
  const tune = useSing((s) => s.tune);
  const setTune = useSing((s) => s.setTune);
  const [pending, setPending] = useState(false);
  const [recording, setRecording] = useState(false);
  const [clip, setClip] = useState<{ file: File; url: string } | null>(null);
  /** bumped by every start and stop: a start still waiting no longer counts */
  const request = useRef(0);
  /** bumped by every Record video: a clip still being finished for an earlier one isn't offered */
  const take = useRef(0);
  /** the clip recording now, if any (one at a time) */
  const recorder = useRef<Clip | null>(null);
  // the audio must be running: if it failed to start, nothing is planned and the error shows
  const ready = async () => {
    await start();
    if (!useStore.getState().started) throw new Error("no audio");
  };
  const drop = () => {
    recorder.current?.cancel();
    recorder.current = null;
    setRecording(false);
  };
  // the clip's life is the tune's run: it starts as the tune is sent (never for a start that was
  // cancelled), is saved when it's sung to the end, and goes when it's stopped from anywhere
  const sing = (id: string, record = false) => {
    const n = ++request.current;
    const k = record ? ++take.current : take.current;
    drop();
    setPending(true);
    let mine: Clip | null = null;
    const hooks = !record
      ? {}
      : {
          onStart: () => {
            const face = () => document.querySelector<HTMLCanvasElement>(".screen.left canvas");
            const audio = engine.recordStream();
            if (n !== request.current || !face() || !audio) throw new Error("not recording");
            mine = recorder.current = new Clip(face, audio);
            setRecording(true);
            setClip(null);
          },
          onEnd: () => {
            const c = mine;
            if (!c || recorder.current !== c) return;
            recorder.current = null;
            setRecording(false);
            c.stop(`chimera-${id}`)
              .then((file) => take.current === k && setClip({ file, url: URL.createObjectURL(file) }))
              .catch((e) => console.warn("Record video:", e));
          },
          onStop: () => {
            if (mine && recorder.current === mine) drop();
          },
        };
    // a clip waits for the last sound (the song this one replaces) to die away, or it opens on its tail
    const go = record ? async () => (await ready(), await engine.quiet()) : ready;
    playTune(id, useStore.getState().body, go, hooks)
      .catch(() => {})
      .finally(() => request.current === n && setPending(false));
  };
  const stop = () => {
    request.current++; // the song getting ready, if any, no longer counts
    setPending(false);
    stopSinging(); // its run's onStop drops the clip
    drop();
  };
  useEffect(() => () => void (clip && URL.revokeObjectURL(clip.url)), [clip]);
  // leaving Sing: nothing still starting may begin a clip, and one recording goes
  useEffect(
    () => () => {
      request.current++;
      take.current++;
      recorder.current?.cancel();
      recorder.current = null;
    },
    [],
  );
  const press = () => (playing || pending ? stop() : sing(tune));
  const share = clip && typeof navigator.canShare === "function" && navigator.canShare({ files: [clip.file] });
  return (
    <ActionBar label="Song" onSpace={press}>
      <PrimaryButton label="Sing" playing={(!!playing || pending) && !recording} busy={recording ? "Recording…" : null} onClick={press} />
      <Menu
        label="Tune"
        icon="note"
        value={tune}
        options={TUNES.map((t) => ({ value: t.id, label: t.name }))}
        onChange={(id) => {
          setTune(id);
          if ((playing || pending) && !recording) sing(id);
        }}
      />
      {canRecord() && (
        <Chip toggle={false} onClick={() => (recording ? stop() : sing(tune, true))} title="Sing it again and record the face into a video">
          {recording ? "Cancel" : "Record video"}
        </Chip>
      )}
      {clip && share && (
        <Chip toggle={false} tone="amber" onClick={() => void navigator.share({ files: [clip.file] }).catch(() => {})}>
          Share
        </Chip>
      )}
      {clip && (
        <a className="chip" href={clip.url} download={clip.file.name}>
          Save
        </a>
      )}
    </ActionBar>
  );
}

/** Sing mode's right column: the song button, the pitch roll, the lyrics, the spectrum with the ring, the song voice, the keys. */
export function SingRight() {
  // the pronouncing dictionary for the lyrics: here rather than on the mode switch, since the app
  // opens on Sing without one
  useEffect(ensureDictionary, []);
  const ring = useStore((s) => s.artic.epilarynx >= RING - 0.05);
  const next = useSing((s) => (s.playing ? null : (s.syllables[s.cursor]?.text ?? null)));
  return (
    <div className="right right-sing">
      {/* the plugin sings what the host plays: no tunes to sing or record */}
      {!inPlugin && <SingActions />}
      <Screen watch className="s-roll" title="Pitch">
        <PianoRoll />
      </Screen>
      <Screen className="s-lyrics" title="Lyrics" tools={inPlugin ? undefined : <LyricsLang />}>
        <Lyrics />
      </Screen>
      <Screen watch className="s-spec" title="Spectrum" sub={ring ? "Ring on" : ""}>
        <SpectrumView ring={ring} />
      </Screen>
      <Screen className="s-voice" title="Voice">
        <SongControls />
      </Screen>
      <Screen className="s-keys" title="Keys" sub={next ? `Next: ${next}` : ""} tools={<MidiButton />}>
        <Keys />
      </Screen>
    </div>
  );
}
