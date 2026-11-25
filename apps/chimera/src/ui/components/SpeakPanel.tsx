import { useEffect, useMemo, useState } from "react";
import { skippedLines } from "../skipped.ts";
import { useStore } from "../store.ts";
import { useSing } from "../singing.ts";
import { speakText, speakPhrases, stopSpeaking, onPlan, currentPlan, spokenWordNow, speakingNow, useSpeech } from "../speech.ts";
import { mouthWords } from "../mouthWords.ts";
import { tokenizeIpa, type Lang } from "../../wasm/engine.ts";
import { Menu, Slider } from "./Controls.tsx";
import { Chip, Screen } from "./Screen.tsx";
import { LangPick } from "./LangPick.tsx";
import { ActionBar, PrimaryButton } from "./ActionBar.tsx";
import { ScoreView, planNow } from "./ScoreView.tsx";
import { TubeView } from "./TubeView.tsx";
import { Stroboscope } from "./Stroboscope.tsx";
import type { InfoFn, SlotsFn } from "./AnatomyView.tsx";
import { V } from "../colors.ts";
import { engine } from "../../audio/AudioEngine.ts";
import { inPlugin } from "../../audio/pluginBridge.ts";

const EXAMPLES: Record<Lang, string[]> = {
  en: ["Hello. Can you hear me?", "She sells sea shells by the sea shore.", "Every sound here is air moving through a throat.", "The quick brown fox jumps over the lazy dog."],
  tr: ["Merhaba, nasılsın?", "Bugün hava çok güzel.", "Türkçe konuşabiliyor musun?", "Şu köşe yaz köşesi, şu köşe kış köşesi."],
};

// IPA pulmonic consonant chart (rows = manner, columns = place); empty cells are null
const PLACES = ["Bilabial", "Labiodental", "Dental", "Alveolar", "Postalveolar", "Retroflex", "Palatal", "Velar", "Uvular", "Pharyngeal", "Glottal"];
/** short headings for the chart's narrow columns (the full names are the cells' titles) */
const PLACE_SHORT = ["Bilab", "Labdent", "Dental", "Alveol", "Postalv", "Retrofl", "Palatal", "Velar", "Uvular", "Pharyn", "Glottal"];
const MANNER_SHORT: Record<string, string> = { "Lateral fricative": "Lat fric", Approximant: "Approx" };
const CHART: [string, (string | null)[][]][] = [
  ["Plosive", [["p", "b"], [null, null], [null, null], ["t", "d"], [null, null], ["ʈ", "ɖ"], ["c", "ɟ"], ["k", "g"], ["q", "ɢ"], [null, null], ["ʔ", null]]],
  ["Nasal", [[null, "m"], [null, "ɱ"], [null, null], [null, "n"], [null, null], [null, "ɳ"], [null, "ɲ"], [null, "ŋ"], [null, "ɴ"], [null, null], [null, null]]],
  ["Trill", [[null, "ʙ"], [null, null], [null, null], [null, "r"], [null, null], [null, null], [null, null], [null, null], [null, "ʀ"], [null, null], [null, null]]],
  ["Tap", [[null, null], [null, null], [null, null], [null, "ɾ"], [null, null], [null, "ɽ"], [null, null], [null, null], [null, null], [null, null], [null, null]]],
  ["Fricative", [["ɸ", "β"], ["f", "v"], ["θ", "ð"], ["s", "z"], ["ʃ", "ʒ"], ["ʂ", "ʐ"], ["ç", "ʝ"], ["x", "ɣ"], ["χ", "ʁ"], ["ħ", "ʕ"], ["h", "ɦ"]]],
  ["Lateral fricative", [[null, null], [null, null], [null, null], ["ɬ", "ɮ"], [null, null], [null, null], [null, null], [null, null], [null, null], [null, null], [null, null]]],
  ["Approximant", [[null, null], [null, "ʋ"], [null, null], [null, "ɹ"], [null, null], [null, "ɻ"], [null, "j"], [null, "ɰ"], [null, null], [null, null], [null, null]]],
  ["Lateral", [[null, null], [null, null], [null, null], [null, "l"], [null, null], [null, "ɭ"], [null, "ʎ"], [null, null], [null, null], [null, null], [null, null]]],
];

/** the segment sounding now in the current utterance */
function soundingSegment(): string | null {
  const sp = currentPlan();
  const now = planNow(sp);
  if (!sp || now < 0) return null;
  return sp.plan.segments.find((s) => now >= s.t0 && now < s.t1)?.ipa ?? null;
}

/** Speak mode's readouts under the chin: the sound being made, jaw, lip shape, soft palate */
/** the word being spoken, at the mouth */
export const speechWords = mouthWords(spokenWordNow, V.white);

export const speechSlots: SlotsFn = (_m, a, f) => {
  const sound = soundingSegment();
  const lips = a.lipAp < 0.05 ? "Closed" : a.lipWidth < 0.55 ? "Round" : a.lipAp > 1.3 ? "Open" : "Spread";
  return [
    sound ? { label: "Sound", value: sound, color: V.green, ipa: true } : { label: "Sound", value: "-", color: V.green },
    { label: "Jaw", value: a.jaw.toFixed(1), unit: "cm", color: V.red },
    { label: "Lips", value: lips, color: V.pink },
    { label: "Soft palate", value: f.velumPort > 0.02 ? "Open" : "Shut", color: V.white },
  ];
};

/** what the voice is doing: voiced or not, where the noise is, the lungs */
export const speechInfo: InfoFn = (m, _f, tele) => {
  if (!tele) return [];
  const t = tele.t;
  const out = [t.voicing > 0.3 ? `Voiced ${t.f0.toFixed(0)} Hz` : "Voiceless"];
  if (t.noiseLevel > 1 && t.noiseNode >= 0) {
    const s = m.branchS(t.noiseNode, t.phLen, t.orLen);
    if (s >= 0) out.push(`Noise at ${s.toFixed(1)} cm`);
  }
  out.push(`Lungs ${t.pLung.toFixed(1)} cmH₂O`);
  return out;
};

function ConsonantChart(props: { onPad: (c: string) => void }) {
  const [now, setNow] = useState<string | null>(null);
  useEffect(() => {
    const id = setInterval(() => setNow(engine.latest ? soundingSegment() : null), 60);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="chart-scroll">
      <table className="ipa-chart">
        <thead>
          <tr>
            <th />
            {PLACES.map((p, i) => (
              <th key={p} scope="col" title={p} aria-label={p}>
                {PLACE_SHORT[i]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {CHART.map(([manner, cells]) => (
            <tr key={manner}>
              <th scope="row" title={manner} aria-label={manner}>
                {MANNER_SHORT[manner] ?? manner}
              </th>
              {cells.map((pair, i) => (
                <td key={i}>
                  {pair.map((sym, j) =>
                    sym ? (
                      <button key={j} className={`ipa${sym === now ? " now" : ""}`} onClick={() => props.onPad(sym)} title={`Say it between two vowels`}>
                        {sym}
                      </button>
                    ) : (
                      <span key={j} className="ipa-empty" />
                    ),
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Whether speech is sounding, polled each frame (re-renders only when it changes). */
function useSpeaking(): boolean {
  const [on, setOn] = useState(false);
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      setOn(speakingNow());
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
  return on;
}

/** Speak mode's right column: the speak button, what to say, the gestural score, the consonant chart, the tube and folds. */
/** In the plugin: the lines a note says, with the next one marked. */
function Lines() {
  const lines = useSpeech((s) => s.lines);
  const cursor = useSpeech((s) => s.cursor);
  const setCursor = useSpeech((s) => s.setCursor);
  return (
    <div className="syllables say-lines" role="group" aria-label="Lines. Each note says the marked one; choose another to start there.">
      {lines.map((l, i) => (
        <button key={i} className={`syl say-line${i === cursor ? " next" : ""}`} onClick={() => setCursor(i)} aria-current={i === cursor ? "step" : undefined} title={`Say from “${l}”`}>
          {l}
        </button>
      ))}
      {!lines.length && <span className="dim">No words: the notes say nothing.</span>}
      <Chip toggle={false} className="from-start" onClick={() => setCursor(0)} disabled={cursor === 0}>
        From the start
      </Chip>
    </div>
  );
}

export function SpeakRight() {
  const body = useStore((s) => s.body);
  const f0 = useStore((s) => s.source.f0);
  const start = useStore((s) => s.start);
  // one language for Speak and Sing, kept with Sing's words
  const lang = useSing((s) => s.lang);
  const setLang = useSing((s) => s.setLang);
  // kept in a store, so the plugin saves them with the project and reads them on the audio thread
  const { text, ipa, rate, edit } = useSpeech();
  const setText = (v: string) => edit({ text: v });
  const setIpa = (v: boolean) => edit({ ipa: v });
  const setRate = (v: number) => edit({ rate: v });
  const [busy, setBusy] = useState(false);
  const [trans, setTrans] = useState(currentPlan()?.transcription ?? "");
  const [vowel, setVowel] = useState("ɑ");
  const speaking = useSpeaking();
  useEffect(() => onPlan((p) => setTrans(p?.transcription ?? "")), []);
  // in the browser each visit starts afresh, on the first example of the words' language (the
  // plugin's box is the project's, so it stays)
  useEffect(() => {
    if (!inPlugin) edit({ text: EXAMPLES[lang][0], ipa: false, rate: 1 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // in IPA mode: the symbols in the box that the voice will not say, updated as the text changes
  const skipped = useMemo(() => (ipa ? skippedLines(text) : []), [ipa, text]);

  const speak = async () => {
    // the plugin says the next line, as a note would
    if (inPlugin) return engine.pluginSpeak();
    setBusy(true);
    try {
      await start();
      // the audio didn't start (the error shows on the bar): speaking would only light Stop
      if (!useStore.getState().started) return;
      await speakText(text, { ipa, lang, body, rate, f0 });
    } finally {
      setBusy(false);
    }
  };
  const pad = async (c: string) => {
    if (inPlugin) return engine.pluginSayIpa(`ˈ${vowel}${c}${vowel}`);
    await start();
    if (!useStore.getState().started) return;
    speakPhrases([{ segments: tokenizeIpa(`ˈ${vowel}${c}${vowel}`), intonation: "flat" }], { body, rate, f0 });
  };

  const press = () => (speaking ? stopSpeaking() : void speak());
  // another language reads the words: an example of the old one gives way to the new one's first
  const pickLang = (l: Lang) => {
    if (l === lang && !ipa) return;
    if (l !== lang && EXAMPLES[lang].includes(text)) setText(EXAMPLES[l][0]);
    setLang(l);
    setIpa(false);
  };

  return (
    <div className="right right-speak">
      <ActionBar label="Speak" onSpace={press}>
        <PrimaryButton label="Speak" playing={speaking} busy={busy ? "Preparing…" : null} onClick={press} />
        {/* puts an example in the box: the menu always reads Examples, since the box shows the words */}
        {!inPlugin && (
        <Menu
          label="Examples"
          value=""
          options={[{ value: "", label: "Examples", disabled: true }, ...EXAMPLES[lang].map((ex) => ({ value: ex, label: ex }))]}
          onChange={(ex) => edit({ text: ex, ipa: false })}
        />
        )}
        <div className="act-slider">
          <Slider label="Rate" value={rate} min={0.5} max={1.8} onChange={setRate} format={(v) => `${v.toFixed(2)}×`} />
        </div>
        {/* the plugin reads Latin script as English (other scripts by their own front ends) */}
        {!inPlugin && <LangPick value={lang} onChange={pickLang} off={ipa} />}
        <Chip on={ipa} onClick={() => setIpa(!ipa)}>
          Text is IPA
        </Chip>
      </ActionBar>
      <Screen className="s-say" label="What to say">
        <div className={`say${inPlugin ? " say-lined" : ""}`}>
          <div className="say-row">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void speak();
                e.stopPropagation();
              }}
              rows={2}
              spellCheck={!ipa}
              lang={ipa ? undefined : lang}
              aria-label={ipa ? "IPA to speak" : "Text to speak"}
              placeholder={ipa ? "Type IPA, for example ˈhɛloʊ" : inPlugin ? "One line per note" : "Type something for the voice to say"}
            />
          </div>
          <div className="say-row">
            <div className="say-read">
              <p className="transcription" aria-live="polite">
                {trans ? `/${trans}/` : ""}
              </p>
              <div className="say-skipped" aria-live="polite">
                {skipped.map((l) => (
                  <p key={l.what}>
                    {l.what}: <span className="say-skipped-sym">{l.symbols.join(" ")}</span>
                  </p>
                ))}
              </div>
            </div>
          </div>
          {inPlugin && <Lines />}
        </div>
      </Screen>
      <Screen watch className="s-score" title="Gestural score">
        <ScoreView />
      </Screen>
      <Screen
        className="s-chart"
        title="Consonants"
        sub="Between two vowels"
        tools={
          <div className="vowel-pick" role="group" aria-label="Vowel around it">
            {["ɑ", "i", "u", "ə"].map((v) => (
              <Chip key={v} on={vowel === v} onClick={() => setVowel(v)} className="ipa-chip">
                {v}
              </Chip>
            ))}
          </div>
        }
      >
        <ConsonantChart onPad={(c) => void pad(c)} />
      </Screen>
      <Screen watch className="s-tube" title="Tube" sub="Glottis to lips">
        <TubeView />
      </Screen>
      <Screen watch className="s-folds" title="Vocal folds" sub="Slowed 100×">
        <Stroboscope />
      </Screen>
    </div>
  );
}
