/** What the Speak box tells a person about IPA the engine will not say as written. */
import { skippedIpa, type SkippedIpa } from "../wasm/engine.ts";

/** the pieces of IPA text that become phrases (`.`, `|`, `?` and `!` end one) */
export function ipaPhraseTexts(text: string): string[] {
  return text
    .split(/[|.?!]+/)
    .map((p) => p.trim())
    .filter(Boolean);
}

/** One line of the notice: what the engine did with `symbols`. */
export interface SkippedLine {
  what: string;
  symbols: string[];
}

/** The notice's lines for IPA typed into Speak, one per kind that is not empty (none: nothing is skipped). */
export function skippedLines(text: string): SkippedLine[] {
  // the phrase separators are structure, as the Speak box reads them: check what it says, not them
  return skippedLinesOf(skippedIpa(ipaPhraseTexts(text).join(" | ")));
}

export function skippedLinesOf(r: SkippedIpa): SkippedLine[] {
  const out: SkippedLine[] = [];
  if (r.unknown.length) out.push({ what: "Not said", symbols: r.unknown });
  if (r.noSound.length) out.push({ what: "No sound yet", symbols: r.noSound });
  if (r.ignoredMarks.length) out.push({ what: "Mark ignored", symbols: r.ignoredMarks });
  return out;
}

/** a line as plain text (`Not said: ɕ ˥`) */
export function skippedText(l: SkippedLine): string {
  return `${l.what}: ${l.symbols.join(" ")}`;
}
