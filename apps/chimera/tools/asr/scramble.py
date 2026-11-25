"""The scramble test: how intelligible a language front end's singing is on sound alone.

A recogniser that knows a lyric fills it in, so the test that counts sings the same melody with the
lyric's units shuffled (syllables for Chinese and Korean, words for English): nothing to guess from.
Each language's text goes through CHIMERA's own front end (`chimera render notes` with "lyrics"),
on one generated tune, one note per syllable, a breath between lines, in the male and female voices.
SenseVoice (sherpa-onnx, CTC: it can't hallucinate a loop) transcribes one sung line at a time.
Score: syllable errors (Chinese: toneless pinyin; Korean: Hangul blocks) or word errors (English).

usage: scramble.py <outdir> [--langs zh,ko,en] [--shuffles 3] [--bodies male,female]
                   [--text lang:file]   (your own lines for one language, one line per phrase)
                   [--melody '{"consonantScale": 1.0}']   (extra melody-file settings, for A/B)
       scramble.py --analyze <outdir>/scramble.json   (the confusions of an earlier run)

Besides the error rates it reports what goes wrong: each sung line's syllables are aligned with what
the recogniser heard, and the substitutions split into the parts of a syllable (Chinese: initial,
final, tone; Korean: initial, vowel, final), with the commonest confusions. That's what says what to
fix next; one error rate doesn't.
Needs the release CLI (cargo build --release -p chimera-cli) and tools/asr/setup.sh.
The texts are public domain: two Tang poems (静夜思, 春晓), the folk song 아리랑, Harvard sentences.
"""

import json
import os
import random
import re
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CHIMERA_REPO", os.path.dirname(os.path.dirname(HERE)))
CLI = os.environ.get("CHIMERA_CLI", os.path.join(REPO, "target", "release", "chimera"))
MODEL = os.environ.get("SENSEVOICE_DIR", os.path.expanduser("~/.cache/chimera-asr/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"))

TEXTS = {
    "zh": ["床前明月光", "疑是地上霜", "举头望明月", "低头思故乡", "春眠不觉晓", "处处闻啼鸟", "夜来风雨声", "花落知多少"],
    "ko": ["아리랑 아리랑 아라리요", "아리랑 고개로 넘어간다", "나를 버리고 가시는 님은", "십리도 못 가서 발병 난다"],
    "en": ["The birch canoe slid on the smooth planks", "Glue the sheet to the dark blue background", "It's easy to tell the depth of a well", "These days a chicken leg is a rare dish", "Rice is often served in round bowls"],
}
BPM, T0 = 100.0, 0.3
TUNE = [0, 2, 4, 7, 9, 7, 4, 2, 0, -3, 0, 2]  # a pentatonic wander, from the voice's base note
BASE = {"male": 50, "female": 62}
MELODY_EXTRA = {}  # --melody: extra melody-file settings (sing-planner options) for an A/B


def units(lang, line):
    """The units the scramble shuffles: syllables (one per character) or words."""
    return line.split() if lang == "en" else [c for c in line if not c.isspace()]


def join(lang, us):
    return " ".join(us) if lang == "en" else "".join(us)


def scramble(lang, lines, seed):
    pool = [u for ln in lines for u in units(lang, ln)]
    random.Random(seed).shuffle(pool)
    out, k = [], 0
    for ln in lines:
        n = len(units(lang, ln))
        out.append(join(lang, pool[k:k + n]))
        k += n
    return out


def syllable_count(line):
    out = subprocess.run([CLI, "syllables", line], capture_output=True, text=True, check=True).stdout
    return sum(1 for s in json.loads(out) if not s.get("melisma"))


def render(lines, body, stem):
    """Render the lines on the tune; return each line's (start, end) in seconds."""
    beat = 60 / BPM
    notes, spans, t, k = [], [], T0, 0
    for ln in lines:
        n = syllable_count(ln)
        start = t
        for _ in range(n):
            notes.append([BASE[body] + TUNE[k % len(TUNE)], 1])
            k += 1
            t += beat
        spans.append((max(0.0, start - 0.3), t + 0.15))
        notes.append([0, 1])  # a breath between lines
        t += beat
    melody = {"bpm": BPM, "t0": T0, "lyrics": "\n".join(lines), "notes": notes, **MELODY_EXTRA}
    with open(stem + ".json", "w") as f:
        json.dump(melody, f, ensure_ascii=False)
    subprocess.run([CLI, "render", "notes", stem + ".json", stem, body], check=True, capture_output=True)
    return spans


def load(path):
    """The render at the recogniser's 16 kHz. It is low-passed first (a windowed sinc, cut at
    7.6 kHz): the renders are not band-limited, and taking every third sample alone folded the
    frication above 8 kHz (s, f, sh) back into the band the recogniser hears."""
    w = wave.open(path)
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    sr = w.getframerate()
    if sr == 16000:
        return x
    assert sr % 16000 == 0, f"{path}: {sr} Hz"
    fc, half = 7600 / sr, 96
    n = np.arange(-half, half + 1)
    h = 2 * fc * np.sinc(2 * fc * n) * np.blackman(2 * half + 1)
    return np.convolve(x, (h / h.sum()).astype(np.float32), "same")[:: sr // 16000].copy()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(h)]


def scorer(lang):
    """Reference and hypothesis → comparable unit lists."""
    if lang == "zh":
        from pypinyin import lazy_pinyin
        from zhconv import convert
        return lambda s: lazy_pinyin(re.sub(r"[^一-鿿]", "", convert(s, "zh-cn")))
    if lang == "ko":
        return lambda s: re.findall(r"[가-힣]", s)
    return lambda s: re.sub(r"[^a-z' ]+", " ", s.lower()).split()


# ---------------------------------------------------------------- confusions

PINYIN_INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]


def zh_parts(text):
    """Chinese text → (initial, final, tone) per character (toneless initial/final from pypinyin)."""
    from pypinyin import Style, pinyin
    from zhconv import convert
    han = re.sub(r"[^\u4e00-\u9fff]", "", convert(text, "zh-cn"))
    out = []
    for (p,) in pinyin(han, style=Style.TONE3, neutral_tone_with_five=True):
        tone = p[-1] if p[-1].isdigit() else "5"
        base = p.rstrip("012345")
        ini = next((i for i in PINYIN_INITIALS if base.startswith(i) and len(base) > len(i)), "")
        out.append((ini or "∅", base[len(ini):], tone))
    return out


KO_I = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
KO_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
KO_F = ["∅"] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def ko_parts(text):
    out = []
    for c in re.findall(r"[\uac00-\ud7a3]", text):
        k = ord(c) - 0xAC00
        out.append((KO_I[k // 588], KO_V[(k % 588) // 28], KO_F[k % 28]))
    return out


def align(r, h, same):
    """Edit-distance alignment of two unit lists: [(ref or None, hyp or None)]."""
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (0 if same(r[i - 1], h[j - 1]) else 1))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (0 if same(r[i - 1], h[j - 1]) else 1):
            out.append((r[i - 1], h[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((r[i - 1], None)); i -= 1
        else:
            out.append((None, h[j - 1])); j -= 1
    return out[::-1]


def confusions(results, lang, keep=lambda x: x["variant"] != "real"):
    """Where a language's syllables go wrong, over the runs `keep` selects."""
    from collections import Counter
    parts = zh_parts if lang == "zh" else ko_parts if lang == "ko" else None
    if parts is None:
        return None
    names = ("initial", "final", "tone") if lang == "zh" else ("initial", "vowel", "final")
    tally = Counter()
    pair = {n: Counter() for n in names}
    wrong_ref = {n: Counter() for n in names}
    seen = {n: Counter() for n in names}
    for x in results:
        if x["lang"] != lang or not keep(x):
            continue
        for ref, hyp in zip(x["ref"], x["hyp"]):
            r, h = parts(ref), parts(hyp)[: 2 * len(parts(ref))]
            # syllables align on initial + final (Chinese) or the whole block (Korean); tone is scored after
            key = (lambda p: p[:2]) if lang == "zh" else (lambda p: p)
            for a, b in align(r, h, lambda a, b: key(a) == key(b)):
                if a is None:
                    tally["inserted"] += 1
                    continue
                tally["syllables"] += 1
                if b is None:
                    tally["deleted"] += 1
                    continue
                for k, n in enumerate(names):
                    seen[n][a[k]] += 1
                    if a[k] != b[k]:
                        tally[n] += 1
                        pair[n][(a[k], b[k])] += 1
                        wrong_ref[n][a[k]] += 1
    return tally, pair, wrong_ref, seen, names


_CMU = None


def cmu():
    """CMUdict as the CLI reads it (public/data/cmudict.txt.gz), stress marks dropped."""
    global _CMU
    if _CMU is None:
        import gzip
        _CMU = {}
        with gzip.open(os.path.join(REPO, "public", "data", "cmudict.txt.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                w, _, ph = line.rstrip("\n").partition("\t")
                _CMU[w] = [re.sub(r"\d", "", p) for p in ph.split()]
    return _CMU


VOWELS_ARPA = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def en_phones(text):
    """English text → phonemes (CMUdict; words it doesn't know are skipped and counted)."""
    out, unknown = [], 0
    for w in re.sub(r"[^a-z' ]+", " ", text.lower()).split():
        ph = cmu().get(w.strip("'"))
        if ph:
            out.extend(ph)
        else:
            unknown += 1
    return out, unknown


def en_report(results, keep=lambda x: x["variant"] != "real"):
    from collections import Counter
    t = Counter(); sub = Counter(); dropped = Counter(); wrong = Counter(); seen = Counter()
    for x in results:
        if x["lang"] != "en" or not keep(x):
            continue
        for ref, hyp in zip(x["ref"], x["hyp"]):
            r, _ = en_phones(ref)
            h, u = en_phones(hyp)
            t["unknown words heard"] += u
            for a, b in align(r, h[: 2 * len(r)], lambda a, b: a == b):
                if a is None:
                    t["inserted"] += 1
                    continue
                kind = "vowel" if a in VOWELS_ARPA else "consonant"
                t[kind] += 1
                seen[a] += 1
                if b is None:
                    t[kind + " dropped"] += 1
                    dropped[a] += 1
                    wrong[a] += 1
                elif a != b:
                    t[kind + " wrong"] += 1
                    sub[(a, b)] += 1
                    wrong[a] += 1
    n = t["vowel"] + t["consonant"] or 1
    errs = sum(t[k] for k in ("vowel dropped", "vowel wrong", "consonant dropped", "consonant wrong")) + t["inserted"]
    print(f"\nen scrambled, {n} phonemes: phoneme error {errs / n:.0%}; vowels wrong {t['vowel wrong'] / max(1, t['vowel']):.0%}, "
          f"dropped {t['vowel dropped'] / max(1, t['vowel']):.0%}; consonants wrong {t['consonant wrong'] / max(1, t['consonant']):.0%}, "
          f"dropped {t['consonant dropped'] / max(1, t['consonant']):.0%}; inserted {t['inserted'] / n:.0%}; unknown words heard {t['unknown words heard']}")
    worst = sorted(((wrong[p] / seen[p], p, seen[p]) for p in seen if seen[p] >= 10), reverse=True)[:10]
    print("  most often wrong:  " + "  ".join(f"{p} {r:.0%} (of {s})" for r, p, s in worst))
    print("  top confusions:    " + "  ".join(f"{a}→{b} ×{m}" for (a, b), m in sub.most_common(10)))
    print("  most often dropped: " + "  ".join(f"{p} ×{m}" for p, m in dropped.most_common(10)))


def report(results, langs):
    for lang in langs:
        if lang == "en":
            en_report(results)
            continue
        c = confusions(results, lang)
        if not c:
            continue
        tally, pair, wrong_ref, seen, names = c
        n = tally["syllables"] or 1
        heard = n - tally["deleted"]
        print(f"\n{lang} scrambled, {n} syllables: deleted {tally['deleted'] / n:.0%}, inserted {tally['inserted'] / n:.0%}; "
              + ", ".join(f"{k} wrong {tally[k] / max(1, heard):.0%}" for k in names) + " (of the syllables heard)")
        for k in names:
            worst = sorted(((wrong_ref[k][u] / seen[k][u], u, seen[k][u]) for u in seen[k] if seen[k][u] >= 6), reverse=True)[:8]
            print(f"  {k:8s} most often wrong: " + "  ".join(f"{u} {r:.0%} (of {s})" for r, u, s in worst))
            print(f"  {k:8s} top confusions:   " + "  ".join(f"{a}→{b} ×{m}" for (a, b), m in pair[k].most_common(8)))


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--analyze":
        results = json.load(open(args[1]))
        for x in results:  # runs before refs were saved: the scrambles are deterministic per seed
            if "ref" not in x and x["lang"] in TEXTS:
                v = x["variant"]
                x["ref"] = TEXTS[x["lang"]] if v == "real" else scramble(x["lang"], TEXTS[x["lang"]], 100 + int(v.removeprefix("scramble")))
        report(results, sorted({x["lang"] for x in results}))
        return
    out = args[0]
    opt = lambda k, d: args[args.index(k) + 1] if k in args else d
    langs = opt("--langs", "zh,ko,en").split(",")
    shuffles = int(opt("--shuffles", "3"))
    bodies = opt("--bodies", "male,female").split(",")
    if "--melody" in args:
        MELODY_EXTRA.update(json.loads(opt("--melody", "{}")))
    if "--text" in args:
        lang, path = opt("--text", "").split(":", 1)
        TEXTS[lang] = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
        langs = [lang]
    os.makedirs(out, exist_ok=True)
    import sherpa_onnx
    sv = {}
    results = []
    for lang in langs:
        rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=f"{MODEL}/model.int8.onnx", tokens=f"{MODEL}/tokens.txt", language=lang, use_itn=False, num_threads=8)
        units_of = scorer(lang)
        if syllable_count(TEXTS[lang][0]) == 0:
            print(f"{lang}: the CLI's front ends read none of these lines (no front end for this script yet); skipped", flush=True)
            continue
        variants = [("real", TEXTS[lang])] + [(f"scramble{s}", scramble(lang, TEXTS[lang], 100 + s)) for s in range(shuffles)]
        for body in bodies:
            for name, lines in variants:
                stem = os.path.join(out, f"{lang}_{body}_{name}")
                spans = render(lines, body, stem)
                audio = load(stem + ".wav")
                errs = total = 0
                hyps = []
                for (a, b), ln in zip(spans, lines):
                    s = rec.create_stream()
                    s.accept_waveform(16000, audio[int(a * 16000):int(b * 16000)])
                    rec.decode_stream(s)
                    r, h = units_of(ln), units_of(s.result.text)[: 2 * len(units_of(ln))]
                    errs += edits(r, h)
                    total += len(r)
                    hyps.append(s.result.text)
                results.append({"lang": lang, "body": body, "variant": name, "error": errs / total, "units": total, "hyp": hyps, "ref": lines})
                print(f"{lang} {body:6s} {name:9s} error {errs / total:5.1%}  ({total} units)  {' | '.join(hyps)[:90]}", flush=True)
    with open(os.path.join(out, "scramble.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\nlang  voice   real   scrambled (mean)")
    langs = [lang for lang in langs if any(x["lang"] == lang for x in results)]
    for lang in langs:
        for body in bodies:
            rs = [x for x in results if x["lang"] == lang and x["body"] == body]
            real = next(x["error"] for x in rs if x["variant"] == "real")
            scr = [x["error"] for x in rs if x["variant"] != "real"]
            print(f"{lang:4s}  {body:6s}  {real:5.1%}  {sum(scr) / len(scr):5.1%}")
    report(results, langs)


if __name__ == "__main__":
    main()
