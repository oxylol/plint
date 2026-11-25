"""The neural gate: the one place a neural network judges Chimera's audio (Plint).

Two folders hold the same sentences as WAVs with the same file stems, say the renders of two
engine versions. Two networks listen to both and never make a sound:

  rating   UTMOS22 "strong learner": a predicted listener rating, 1 (bad) to 5 (like a person).
  phones   a wav2vec2 phoneme recogniser: which sounds a listener-like model hears.

usage: neural_gate.py <folder-A> <folder-B> --why "<reason>" [--reference <real recordings>]
                      [--texts file] [--limit n] [--only regex] [--jobs n] [--show n] [--trial]
       neural_gate.py --selftest [--reference <real recordings>] [--jobs n]

Prints the mean rating of A and of B with standard errors and the paired difference, the real
recordings' mean as the ceiling when --reference is given, and then, per phone, how often it is
heard in A and in B, worst first, with what is heard in its place.

WHEN IT MAY BE USED. Filip's word (2026-11-25): a network may judge "very sparingly, only when
needed". So this is run by hand at a gate and never by a fitting loop: once for the baseline,
then only when the realism judge and the author's judges disagree with each other, and once
before shipping. If it disagrees with them, all three are reported and Filip's ear decides:
nothing is tuned toward these networks. Nothing in crates/ may call or mention this file.
--why is required, and every run appends one line to
/mnt/storage/plint/chimera/hill/neural-uses.log (time, kind, folders, clips, reason, headline
numbers): that log is the proof it was used sparingly. A line's kind is `gate`, or `trial` with
--trial (trying the tool out, not judging a voice). --selftest writes no line.

THE FOLDERS. WAVs are found at any depth and matched by file stem, so a judge run works as it
lies: <run>/tsp (the renders of `chimera compare tsp`, in tsp/mfa/corpus/<stem>/<stem>.wav,
named male-LL_SS and female-LL_SS for list LL, sentence SS) or <run>/speech (speech_wer.py's
renders). Nothing is written to them. --reference is a folder of real recordings of the same
sentences: the TSP set (.../tsp/48k, or one speaker's folder). male-09_10 is matched to the
man who read it (MB09_10.wav), female-09_10 to the woman; any other stem to a file of the same
name. Stems without a recording are left out. --limit takes n sentences spread evenly over the
sorted stems; --only keeps the stems a regex matches (--only '^male-').

WHAT "HEARD" MEANS. The expected phones of a sentence are what the recogniser itself writes
for the REAL recording of that sentence; A's and B's phones are lined up against them by edit
distance, and a phone counts as heard when it lines up with itself. So the table says "is the
synth heard as the real speaker is", which is the question, and no aligner or pronouncing
dictionary is needed. The price: the recogniser's own mistakes on real speech become the
target (on the TSP men it writes clean English, a slip in about one sentence in ten), and the
speakers' accents with them. Without --reference there is no phone table, only ratings.
Renders made by `compare tsp` carry a noise floor like the recordings'; both networks hear that.
Every clip is brought to the same loudness before it is rated, so level is not what is judged.

THE NETWORKS (Plint has no GPU: CPU only, one clip at a time, one network in memory at a time).
  UTMOS22 strong, as repackaged in tarepan/SpeechMOS v1.2.0 (code MIT, (c) 2023 tarepan),
    checkpoint utmos22_strong_step7459_v1.pt (392 MB, sha256 38aa51ab…9bff), derived from
    sarulab-speech/UTMOS22 (MIT, (c) 2022 Saruwatari & Koyama laboratory, Univ. of Tokyo).
    Trained on ratings of synthetic and natural speech at 16 kHz (VoiceMOS 2022).
  facebook/wav2vec2-lv-60-espeak-cv-ft at revision ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4
    (Apache-2.0), 1.26 GB. Decoded greedily with the ids in its vocab.json: the espeak binary
    and the phonemizer package are not needed and not installed.
Measured on Plint with 2 threads: 0.3 s (rating) and 0.8 s (phones) for a 2.5 s sentence, 8 s
and 15 s to load, 1.8 GB of memory at the most.

HOW THE VENV WAS MADE (Python 3.12.3; everything on the array, nothing large on the root disk):
  python3 -m venv /mnt/storage/plint/chimera/neural-venv
  V=/mnt/storage/plint/chimera/neural-venv/bin
  $V/pip install --no-cache-dir --upgrade pip
  $V/pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  $V/pip install --no-cache-dir transformers numpy
which gave torch 2.15.0+cpu, torchaudio 2.11.1+cpu (SpeechMOS imports it), transformers 5.24.1,
numpy 2.5.5. The models download themselves on the first run into
/mnt/storage/plint/chimera/neural-cache (TORCH_HOME and HF_HOME); after that no network is used.
Run with any python3: the tool restarts itself in the venv.
"""

import os
import sys

RIG = "/mnt/storage/plint/chimera"
VENV = f"{RIG}/neural-venv"
CACHE = f"{RIG}/neural-cache"
USES = f"{RIG}/hill/neural-uses.log"
TSP = f"{RIG}/tsp/48k"
HERE = os.path.dirname(os.path.abspath(__file__))
MOS_REPO, MOS_TAG = "tarepan/SpeechMOS", "v1.2.0"
PHONE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
PHONE_REV = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
PHONE_FILES = ["config.json", "vocab.json", "preprocessor_config.json", "pytorch_model.bin"]
# a network is not loaded with less memory than this available (GB): Plint has 7.7 in all
MIN_FREE_GB = 2.0
# every clip is rated at this level (dB re full scale, RMS)
RATE_DB = -26.0
# a phone is ranked in the table only if the real recordings hold at least this many of it
MIN_COUNT = 3

if os.path.realpath(sys.prefix) != os.path.realpath(VENV) and os.path.exists(f"{VENV}/bin/python"):
    os.execv(f"{VENV}/bin/python", [f"{VENV}/bin/python"] + sys.argv)
os.environ["TORCH_HOME"] = f"{CACHE}/torch"
os.environ["HF_HOME"] = f"{CACHE}/hf"
PHONE_DIR = f"{CACHE}/hf/hub/models--{PHONE_MODEL.replace('/', '--')}/snapshots/{PHONE_REV}"
if all(os.path.exists(f"{PHONE_DIR}/{f}") for f in PHONE_FILES):
    os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import datetime
import json
import math
import re
import time
import wave

import numpy as np


def load16k(path):
    """Mono float32 at 16 kHz from a 16-bit PCM WAV (band-limited FFT resampling)."""
    with wave.open(path) as w:
        ch, rate, width = w.getnchannels(), w.getframerate(), w.getsampwidth()
        if width != 2:
            raise SystemExit(f"{path}: {8 * width}-bit, expected 16-bit PCM")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    x = x.reshape(-1, ch).mean(axis=1)
    n = int(round(len(x) * 16000 / rate))
    return (np.fft.irfft(np.fft.rfft(x)[: n // 2 + 1], n) * (n / len(x))).astype(np.float32)


def wavs(folder):
    """stem → path for every WAV under a folder; a stem found twice is an error."""
    found = {}
    for d, _, files in sorted(os.walk(folder)):
        for f in sorted(files):
            if f.lower().endswith(".wav"):
                stem = f[:-4]
                if stem in found:
                    raise SystemExit(f"{folder}: two files named {f} ({found[stem]} and {os.path.join(d, f)})")
                found[stem] = os.path.join(d, f)
    if not found:
        raise SystemExit(f"{folder}: no WAVs")
    return found


def real_for(stem, real):
    """The recording of a render's sentence: male-09_10 → M?09_10, else the same stem."""
    m = re.fullmatch(r"(male|female)-(\d\d_\d\d)", stem)
    if not m:
        return real.get(stem)
    hits = [s for s in real if re.fullmatch(("M" if m[1] == "male" else "F") + "[A-Z]" + m[2], s)]
    return real[hits[0]] if len(hits) == 1 else None


def free_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return float("inf")


def enough_memory(what):
    gb = free_gb()
    if gb < MIN_FREE_GB:
        return f"{what} not loaded: {gb:.1f} GB of memory available, {MIN_FREE_GB:.0f} needed (stop a fit or a build first)"
    return None


def rate_all(paths, torch):
    """Predicted listener rating per clip, or (None, why) when the network can't run."""
    short = enough_memory("the rating network")
    if short:
        return None, short
    try:
        local = f"{CACHE}/torch/hub/{MOS_REPO.replace('/', '_')}_{MOS_TAG}"
        if os.path.isdir(local):
            model = torch.hub.load(local, "utmos22_strong", source="local", progress=False)
        else:
            model = torch.hub.load(f"{MOS_REPO}:{MOS_TAG}", "utmos22_strong", trust_repo=True, progress=False)
        model.eval()
    except Exception as e:
        return None, f"the rating network did not load: {type(e).__name__}: {e}"
    out = {}
    target = 10.0 ** (RATE_DB / 20.0)
    with torch.inference_mode():
        for p in paths:
            x = load16k(p)
            x = x * (target / (float(np.sqrt(np.mean(x * x))) + 1e-9))
            out[p] = float(model(torch.from_numpy(np.clip(x, -1.0, 1.0)).unsqueeze(0), 16000)[0])
    del model
    return out, None


def hear_all(paths, torch):
    """The phones the recogniser writes per clip, or (None, why)."""
    short = enough_memory("the phone recogniser")
    if short:
        return None, short
    try:
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2ForCTC
        from transformers.utils import logging
        logging.set_verbosity_error()
        logging.disable_progress_bar()
        d = snapshot_download(PHONE_MODEL, revision=PHONE_REV, allow_patterns=PHONE_FILES)
        model = Wav2Vec2ForCTC.from_pretrained(d).eval()
        names = {i: s for s, i in json.load(open(os.path.join(d, "vocab.json"))).items()}
    except Exception as e:
        return None, f"the phone recogniser did not load: {type(e).__name__}: {e}"
    blank = model.config.pad_token_id
    skip = {blank} | {i for i, s in names.items() if s.startswith("<") or s == "|"}
    out = {}
    with torch.inference_mode():
        for p in paths:
            x = load16k(p)
            x = (x - x.mean()) / (x.std() + 1e-7)
            ids = model(torch.from_numpy(x).unsqueeze(0)).logits[0].argmax(-1).tolist()
            # CTC: runs of one id collapse, then blanks go
            out[p] = [names[i] for k, i in enumerate(ids) if (k == 0 or i != ids[k - 1]) and i not in skip]
    del model
    return out, None


def line_up(ref, hyp):
    """Edit-distance alignment. Per expected phone: what was heard there (None = nothing);
    and the number of phones heard that were not expected."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    heard, extra, i, j = [None] * n, 0, n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            heard[i - 1] = hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            extra += 1
            j -= 1
    return heard, extra


def mean_se(v):
    v = np.asarray(v, dtype=np.float64)
    return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"))


def phone_table(stems, real_ph, heard_ph):
    """Per folder: per phone [expected, heard, {heard instead: count}], and per clip (hits, expected, extra)."""
    per_phone, per_clip = {}, []
    for s in stems:
        heard, extra = line_up(real_ph[s], heard_ph[s])
        hits = 0
        for want, got in zip(real_ph[s], heard):
            row = per_phone.setdefault(want, [0, 0, {}])
            row[0] += 1
            if got == want:
                row[1] += 1
                hits += 1
            else:
                key = got if got is not None else "–"
                row[2][key] = row[2].get(key, 0) + 1
        per_clip.append((hits, len(real_ph[s]), extra))
    return per_phone, per_clip


def instead(row):
    if not row[2]:
        return ""
    return " ".join(f"{k}×{c}" for k, c in sorted(row[2].items(), key=lambda kc: (-kc[1], kc[0]))[:3])


def load_texts(path):
    texts = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            texts[parts[0]] = parts[-1]
    return texts


def judge(a_dir, b_dir, ref_dir, limit, only, jobs, texts_file, show, quiet=False):
    """Runs both networks. Returns a dict of everything measured."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(jobs)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        os.nice(10)
    except OSError:
        pass

    a, b = wavs(a_dir), wavs(b_dir)
    stems = sorted(set(a) & set(b))
    real = {}
    if ref_dir:
        recordings = wavs(ref_dir)
        real = {s: real_for(s, recordings) for s in stems}
        stems = [s for s in stems if real[s]]
    if only:
        stems = [s for s in stems if re.search(only, s)]
    if not stems:
        raise SystemExit("no sentence is in both folders" + (" and among the recordings" if ref_dir else ""))
    if limit and limit < len(stems):
        stems = [stems[int(k * len(stems) / limit)] for k in range(limit)]
    sides = [("A", a)] + [("B", b)] + ([("real", real)] if ref_dir else [])
    paths = [side[s] for _, side in sides for s in stems]
    say = (lambda *x: None) if quiet else (lambda *x: print(*x, flush=True))
    say(f"{len(stems)} sentences, {len(paths)} clips, {jobs} threads")
    say(f"  A    {a_dir}\n  B    {b_dir}" + (f"\n  real {ref_dir}" if ref_dir else ""))

    res = {"n": len(stems), "stems": stems, "notes": []}
    t = time.time()
    ratings, why_not = rate_all(paths, torch)
    res["rate_s"] = time.time() - t
    if ratings is None:
        res["notes"].append(why_not)
        say(f"\nrating: NOT DONE. {why_not}")
    else:
        r = {name: [ratings[side[s]] for s in stems] for name, side in sides}
        res["ratings"] = r
        say("\npredicted listener rating (UTMOS, 1 to 5), mean ± standard error")
        for name, _ in sides:
            m, se = mean_se(r[name])
            say(f"  {name:5s} {m:.3f} ± {se:.3f}   lowest {min(r[name]):.2f}, highest {max(r[name]):.2f}"
                + ("   (the ceiling)" if name == "real" else ""))
        dm, dse = mean_se(np.array(r["A"]) - np.array(r["B"]))
        res["mos"] = {name: mean_se(r[name])[0] for name, _ in sides}
        res["mos_diff"] = (dm, dse)
        clear = "" if not dse == dse else ("   clear" if abs(dm) > 2 * dse else "   not clear (within 2 standard errors)")
        say(f"  A − B {dm:+.3f} ± {dse:.3f} (paired){clear}")
        if ref_dir:
            gap = res["mos"]["real"] - max(res["mos"]["A"], res["mos"]["B"])
            if gap < 0.3:
                say(f"  NOTE: the real recordings rate only {gap:+.2f} above the better folder: "
                    "this rating tells real from synthetic poorly here")

    if not ref_dir:
        say("\nphones: not done (they need --reference: the expected phones are what is heard in the real recording)")
    else:
        t = time.time()
        phones, why_not = hear_all(paths, torch)
        res["hear_s"] = time.time() - t
        if phones is None:
            res["notes"].append(why_not)
            say(f"\nphones: NOT DONE. {why_not}")
        else:
            real_ph = {s: phones[real[s]] for s in stems}
            res["phones_real"] = real_ph
            tab, clips = {}, {}
            for name, side in sides[:2]:
                res["phones_" + name] = {s: phones[side[s]] for s in stems}
                tab[name], clips[name] = phone_table(stems, real_ph, res["phones_" + name])
            total = sum(len(v) for v in real_ph.values())
            res["heard"] = {}
            say(f"\nphones heard as in the real recording ({total} expected phones)")
            for name in ("A", "B"):
                hits = sum(c[0] for c in clips[name])
                extra = sum(c[2] for c in clips[name])
                _, se = mean_se([100.0 * c[0] / max(c[1], 1) for c in clips[name]])
                res["heard"][name] = 100.0 * hits / max(total, 1)
                say(f"  {name:5s} {res['heard'][name]:5.1f} % ± {se:.1f}   and {100.0 * extra / max(total, 1):.1f} extra phones per 100")
            dm, dse = mean_se([100.0 * (x[0] - y[0]) / max(x[1], 1) for x, y in zip(clips["A"], clips["B"])])
            res["heard_diff"] = (dm, dse)
            say(f"  A − B {dm:+.1f} ± {dse:.1f} points (paired, per sentence)")
            rows = []
            for ph, ra in tab["A"].items():
                rb = tab["B"][ph]
                rows.append((min(ra[1] / ra[0], rb[1] / rb[0]), -ra[0], ph, ra, rb))
            rows.sort(key=lambda r: (r[1] > -MIN_COUNT, r[0], r[1], r[2]))
            res["phone_rows"] = [(ph, ra[0], 100.0 * ra[1] / ra[0], 100.0 * rb[1] / rb[0]) for _, _, ph, ra, rb in rows]
            say(f"\nper phone, worst first (–: nothing heard there; phones with under {MIN_COUNT} in the recordings last)")
            say(f"  {'phone':6s} {'n':>4s} {'A %':>6s} {'B %':>6s}   A hears instead        B hears instead")
            for _, _, ph, ra, rb in rows:
                say(f"  {ph:6s} {ra[0]:4d} {100.0 * ra[1] / ra[0]:6.0f} {100.0 * rb[1] / rb[0]:6.0f}   {instead(ra):22s} {instead(rb)}")
            if show:
                texts = load_texts(texts_file) if texts_file else {}
                worst = sorted(range(len(stems)), key=lambda k: clips["A"][k][0] / max(clips["A"][k][1], 1))[:show]
                say(f"\nthe {len(worst)} sentences A is heard worst in")
                for k in worst:
                    s = stems[k]
                    lab = a[s][:-4] + ".lab"
                    text = texts.get(s) or texts.get(os.path.basename(real[s])[:-4]) or (open(lab).read().strip() if os.path.exists(lab) else "")
                    say(f"  {s}  {text}\n    real {' '.join(real_ph[s])}\n    A    {' '.join(res['phones_A'][s])}\n    B    {' '.join(res['phones_B'][s])}")
    return res


def headline(res):
    parts = []
    if "mos" in res:
        m = res["mos"]
        parts.append(f"rating A {m['A']:.3f} B {m['B']:.3f}" + (f" real {m['real']:.3f}" if "real" in m else "")
                     + f" A-B {res['mos_diff'][0]:+.3f}±{res['mos_diff'][1]:.3f}")
    if "heard" in res:
        parts.append(f"phones heard A {res['heard']['A']:.1f}% B {res['heard']['B']:.1f}%"
                     f" A-B {res['heard_diff'][0]:+.1f}±{res['heard_diff'][1]:.1f}")
        worst = [f"{ph} {x:.0f}/{y:.0f}" for ph, n, x, y in res["phone_rows"] if n >= MIN_COUNT][:5]
        parts.append("worst (A/B %) " + ", ".join(worst))
    return "; ".join(parts) + ("; " + "; ".join(res["notes"]) if res["notes"] else "")


def selftest(ref_dir, jobs):
    """Six real clips against themselves. Writes nothing to the uses log."""
    folder = ref_dir or f"{TSP}/MA"
    res = judge(folder, folder, folder, 6, None, jobs, None, 0, quiet=True)
    checks = []
    if "ratings" in res:
        r = res["ratings"]
        checks.append(("ratings are finite and between 1 and 5", all(math.isfinite(v) and 1.0 <= v <= 5.0 for v in r["A"] + r["B"])))
        checks.append(("the same clip rates the same twice", r["A"] == r["B"] == r["real"]))
        print(f"  ratings of the {res['n']} real clips: " + " ".join(f"{v:.2f}" for v in r["A"]))
    else:
        checks.append(("the rating network runs", False))
    if "heard" in res:
        checks.append(("a folder against itself is heard 100 %", res["heard"]["A"] == 100.0 and res["heard"]["B"] == 100.0))
        checks.append(("every clip gives phones", all(len(v) > 5 for v in res["phones_real"].values())))
        s = res["stems"]
        heard, _ = line_up(res["phones_real"][s[0]], res["phones_real"][s[1]])
        wrong = sum(g != w for g, w in zip(heard, res["phones_real"][s[0]])) / len(heard)
        checks.append((f"another sentence is not heard as this one ({100 * (1 - wrong):.0f} % heard)", wrong > 0.3))
        print(f"  {s[0]}: {' '.join(res['phones_real'][s[0]])}")
    else:
        checks.append(("the phone recogniser runs", False))
    for note in res["notes"]:
        print("  " + note)
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"  rating {res['rate_s']:.0f} s, phones {res.get('hear_s', 0):.0f} s for {3 * res['n']} clips (with loading)")
    ok = all(ok for _, ok in checks)
    print("selftest " + ("passed" if ok else "FAILED") + " (no line written to the uses log)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(usage=__doc__.split("usage: ")[1].split("\n\n")[0])
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--why")
    ap.add_argument("--reference")
    ap.add_argument("--texts")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(args.reference, max(1, args.jobs)))
    if len(args.folders) != 2:
        ap.error("two folders are needed")
    why = " ".join((args.why or "").split())
    if not why:
        ap.error('--why "<reason>" is required: every use of the network is logged with its reason')
    for d in args.folders + ([args.reference] if args.reference else []):
        if not os.path.isdir(d):
            ap.error(f"{d}: not a folder")
    a_dir, b_dir = (os.path.abspath(d) for d in args.folders)
    ref_dir = os.path.abspath(args.reference) if args.reference else None
    t = time.time()
    res = judge(a_dir, b_dir, ref_dir, args.limit, args.only, max(1, args.jobs), args.texts, args.show)
    if "mos" not in res and "heard" not in res:
        print("\nno network ran: nothing judged, nothing logged. The gate falls back to Whisper and Filip's ear.")
        sys.exit(2)
    line = "\t".join([datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "trial" if args.trial else "gate",
                      f"A={a_dir}", f"B={b_dir}", f"real={ref_dir or '-'}", f"n={res['n']}", f"why={why}", headline(res)])
    with open(USES, "a") as f:
        f.write(line + "\n")
    print(f"\n{time.time() - t:.0f} s. Logged as a {'trial' if args.trial else 'gate'} use in {USES}")


if __name__ == "__main__":
    main()
