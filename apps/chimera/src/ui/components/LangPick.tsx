import type { Lang } from "../../wasm/engine.ts";
import { Chip } from "./Screen.tsx";

const LANGS: { value: Lang; label: string; name: string }[] = [
  { value: "en", label: "EN", name: "English" },
  { value: "tr", label: "TR", name: "Turkish" },
];

/** The language typed words are read in: two chips for a screen's tools. `off` lights neither (IPA input). */
export function LangPick(props: { value: Lang; onChange: (l: Lang) => void; off?: boolean }) {
  return (
    <div className="lang-pick" role="group" aria-label="Language">
      {LANGS.map((l) => (
        <Chip key={l.value} on={!props.off && props.value === l.value} onClick={() => props.onChange(l.value)} title={l.name} aria-label={l.name}>
          {l.label}
        </Chip>
      ))}
    </div>
  );
}
