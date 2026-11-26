#!/usr/bin/env bash
# Plint · the realism judge's own proof (`chimera realism`, crates/chimera-cli/src/realism.rs):
# is it sound enough to be a fitter's objective? Eight checks, each PASS or FAIL with its numbers;
# the whole output is kept in the rig's hill/realism/check.txt; exit 1 if any check fails.
#   tools/plint/realism_check.sh [--jobs 2]
#   (about 12 minutes on 2 of Plint's cores; CHIMERA_BIN=… judges another build than target/release)
# (a) each fitting man against a reference of the other five reads like his peers
# (b) the men the reference never saw (MH MI MJ, and MK ML) read like real men too
# (c) the default voice reads clearly worse than every real man
# (d) a source known to be better lowers the `source` group, one known to be worse raises it
# (e) the same input gives the same bytes, twice and on 1 or 2 threads
# (f) the plan's phone times and the aligner's give the same groups (needs a judge run's aligned
#     renders of the default voice in hill/ref442; SKIP, which is not a pass, when they are absent).
#     Timbre's bar is wider, 0.14: measured 2026-11-26 it reads 0.68 with plan times against 0.56,
#     a small one-sided offset in a dozen terms, none over 0.4 (realism.rs, PLAN_TO_SOUND, says what
#     was tried). It is the same for every tuning that leaves the timing alone, so a fit of the
#     source and the noise is not led by it; the Gate's own judge uses the aligner. (The bar was
#     0.12 for 0.097 while the fricatives' levels, which agree, were timbre terms: the same
#     offsets over 35 terms instead of 42 read 0.114.)
#     The same again for the first fit's voice (g), whose source is far from the default's, with
#     its own aligned renders in hill/fit1 (SKIP when absent): the plan-to-sound offsets were
#     measured on the default voice and must hold for another. Its bar is 0.15 for every group:
#     that voice's groups are all near 0.5, where a tenth is one term of twenty moving by 1.
#     Measured: timbre 0.07, source 0.08, noise 0.05, pitch 0.03, motion 0.04 apart, and
#     consonants 0.48 with plan times against 0.57: the plan's spans read that voice's quiet
#     consonants about 0.5 dB louder than the aligner's do, so the judge charges them a sixth too
#     little (on all 360 fitting sentences 0.55 against 0.60).
# (g) the first fit (tools/plint/fit/stage1-failed.json: stage (i) of 2026-11-26, which made the
#     source real and let the vowels gain 4-6 dB on every consonant; the author's judge and Whisper
#     caught it, this judge could not) reads clearly WORSE than the default on `consonants` and
#     BETTER on `source`: the judge sees both halves of what happened.
# (h) the hiss (the judge's version 3: before it no term heard a consonant above 8 kHz, and the
#     first fit turned the teeth noise's roll-off the wrong way, `noise.teethOrder` 2 → 1.32, while
#     the men and Filip's ear say the voice hisses too much up there): with the aspiration's and the
#     teeth's noise rolled off (`aspirationOrder` 4, `teethOrder` 4, which he heard as "way
#     better") `noise`, the group that holds the new terms, reads clearly BETTER than the
#     default, and with what the first fit did to the teeth it reads WORSE.
# Check (b) READS THE HELD-OUT MEN'S RECORDINGS (MK ML). That is allowed here and only here: it
# asks whether the judge is sound, and tunes nothing. Only their total scores are printed and
# kept; their per-term values are written nowhere, so no fitter can read them. (The validation men
# MH MI MJ are read here too, and nowhere else by this script; their groups are printed.)
# Not a check, so nobody tries it again: a reference of the women does not push the men far away
# (0.58 against 0.43), because the judge does not measure sex: pitch is relative to the speaker's
# own median, the long-term spectrum is removed, and no formant is scored.
set -euo pipefail
cd "$(dirname "$0")/../.."
HILL="${HILL:-/mnt/storage/plint/chimera/hill}"
TSP="${TSP:-/mnt/storage/plint/chimera/tsp}"
BIN="${CHIMERA_BIN:-target/release/chimera}"
OUT="$HILL/realism"
REF="$OUT/ref-fit6.json"
MFA="$HILL/ref442/tsp/mfa/aligned"
FAILED_FIT="tools/plint/fit/stage1-failed.json"
FAILED_MFA="$HILL/fit1/tsp/mfa/aligned"

# The bars, with what was measured when they were set (2026-11-26, engine efae71f and the judge's
# version 3: six groups, 107 scored terms of 114; `noise` has 25 terms where it had 8, `consonants`
# 23 scored of 28 where it had 22 of 27).
MEN_MAX=1.0        # (a) every fitting man, left out in turn: measured 0.26 to 0.58
MEN_MEDIAN=0.6     # (a) their median: measured 0.42
UNSEEN_MAX=1.5     # (b) MH MI MJ MK ML: measured 0.12 to 0.62
SYNTH_OVER=1.25    # (c) the default voice over the worst real man of (a) and (b): measured 1.13 / 0.62 = 1.82
SOURCE_STEP=0.25   # (d) the `source` group must move at least this far each way: measured −1.0 and +1.2 from 2.45; (g): −1.95
FIT_MEN="MA MB MC MD MF MG"
# (c) to (h) say these men's sentences (120; all 360 read 1.136 where these read 1.131)
SYNTH_SPEAKERS="MA,MB"
CONSONANTS_STEP=0.15   # (g) the first fit's `consonants` group over the default's: measured +0.26 (0.48 against 0.22)
BETTER='{"source":{"closureTension":0.6,"oqBase":0.6}}'   # sharper closure: what P1 found missing
WORSE='{"source":{"oqBase":0.9}}'                          # a softer source still
HISS_LESS='{"noise":{"aspirationOrder":4,"teethOrder":4}}'   # (h) the hiss rolled off: Filip heard the lisp get "way better"
HISS_MORE='{"noise":{"teethOrder":1.32}}'                    # (h) what the first fit did to the teeth noise
HISS_BETTER=0.75   # (h) the `noise` group with the hiss rolled off, under the default's: measured −1.59 (1.11 against 2.70)
HISS_WORSE=0.10    # (h) and with the first fit's teeth order, over it: measured +0.23 (2.93). With the new terms left out of the
                   #     score the same two read −0.42 and −0.01: the roll-off the first fit chose looked free.

jobs=2
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) [ $# -ge 2 ] || { echo "realism_check.sh: --jobs needs a value" >&2; exit 2; }; jobs=$2; shift 2 ;;
    *) echo "usage: tools/plint/realism_check.sh [--jobs n]" >&2; exit 2 ;;
  esac
done
case "$jobs" in ''|*[!0-9]*|0) echo "realism_check.sh: --jobs needs a whole number above 0, got $jobs" >&2; exit 2 ;; esac
[ -x "$BIN" ] || { echo "realism_check.sh: no binary at $BIN (cargo build --release -p chimera-cli, or CHIMERA_BIN=…)" >&2; exit 2; }
[ -d "$TSP/48k" ] || { echo "realism_check.sh: no recordings in $TSP/48k" >&2; exit 2; }
[ -f "$FAILED_FIT" ] || { echo "realism_check.sh: no $FAILED_FIT" >&2; exit 2; }
unset CHIMERA_TUNING
mkdir -p "$OUT"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
exec > >(tee "$OUT/check.txt") 2>&1
started=$(date +%s)
failed=0
verdict() { if [ "$1" = 0 ]; then echo "PASS ($2) $3"; else echo "FAIL ($2) $3"; failed=1; fi; }
# a voice's total from `realism score`'s output ("MA: realism 0.647  (60 sentences)")
totals() { sed -n 's/^\([A-Za-z]*\): realism \([0-9.]*\) .*/\1 \2/p'; }
# a value from a score's JSON: `field score.json score` or `field score.json groups.source`
field() { python3 - "$1" "$2" <<'EOF'
import json, sys
v = json.load(open(sys.argv[1]))["voices"][0]
for k in sys.argv[2].split("."):
    v = v[k]
print(v)
EOF
}
test_num() { python3 -c "import sys; sys.exit(0 if ($1) else 1)"; }

echo "realism check · $(date -u +%Y-%m-%dT%H:%MZ) · $BIN · $jobs jobs"
"$BIN" realism reference "$TSP" "$tmp/ref.json" --jobs "$jobs" > /dev/null
if [ -f "$REF" ]; then
  cmp -s "$tmp/ref.json" "$REF" && ok=0 || ok=1
  verdict $ok ref "$REF is what this binary builds from $FIT_MEN"
else
  cp "$tmp/ref.json" "$REF"
  echo "wrote $REF"
fi
REF="$tmp/ref.json"

# (a)
men=""
for m in $FIT_MEN; do
  others="$(echo "$FIT_MEN" | tr ' ' '\n' | grep -v "^$m\$" | paste -sd,)"
  "$BIN" realism reference "$TSP" "$tmp/loo-$m.json" --speakers "$others" --jobs "$jobs" > /dev/null
  "$BIN" realism score "$tmp/loo-$m.json" --recordings "$TSP" --speakers "$m" --jobs "$jobs" > "$tmp/loo-$m.txt"
  men="$men $(totals < "$tmp/loo-$m.txt" | tr ' ' '=')"
done
men_scores="$(echo "$men" | tr ' ' '\n' | sed -n 's/.*=//p' | paste -sd,)"
test_num "max([$men_scores]) < $MEN_MAX and sorted([$men_scores])[3] + sorted([$men_scores])[2] < 2 * $MEN_MEDIAN" && ok=0 || ok=1
verdict $ok a "each fitting man against the other five:$men (bars: all under $MEN_MAX, median under $MEN_MEDIAN)"
for m in $FIT_MEN; do echo "  $m: $(grep -A1 "^$m: realism" "$tmp/loo-$m.txt" | tail -1 | sed 's/^ *//')"; done

# (b) totals only: nothing of MK ML but these numbers leaves this pipe
"$BIN" realism score "$REF" --recordings "$TSP" --speakers MH,MI,MJ --jobs "$jobs" > "$tmp/validation.txt"
unseen="$({ totals < "$tmp/validation.txt"; "$BIN" realism score "$REF" --recordings "$TSP" --speakers MK,ML --jobs "$jobs" | totals; } | tr ' ' '=' | paste -sd' ')"
unseen_scores="$(echo "$unseen" | tr ' ' '\n' | sed -n 's/.*=//p' | paste -sd,)"
test_num "len([$unseen_scores]) == 5 and max([$unseen_scores]) < $UNSEEN_MAX" && ok=0 || ok=1
verdict $ok b "men the reference never saw: $unseen (bar: all under $UNSEEN_MAX)"
for m in MH MI MJ; do echo "  $m: $(grep -A1 "^$m: realism" "$tmp/validation.txt" | tail -1 | sed 's/^ *//')"; done

# (c)
"$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --jobs "$jobs" --json "$tmp/default.json" > "$tmp/default.txt"
synth="$(field "$tmp/default.json" score)"
worst_man="$(python3 -c "print(max([$men_scores,$unseen_scores]))")"
test_num "$synth > $SYNTH_OVER * $worst_man" && ok=0 || ok=1
verdict $ok c "the default voice on $SYNTH_SPEAKERS's sentences reads $(printf %.3f "$synth"), the worst real man $worst_man (bar: over $SYNTH_OVER times that)"

# (d)
echo "$BETTER" > "$tmp/better.json"
echo "$WORSE" > "$tmp/worse.json"
"$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --jobs "$jobs" --tuning "$tmp/better.json" --json "$tmp/better-score.json" > /dev/null
"$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --jobs "$jobs" --tuning "$tmp/worse.json" --json "$tmp/worse-score.json" > /dev/null
s0="$(field "$tmp/default.json" groups.source)"; s1="$(field "$tmp/better-score.json" groups.source)"; s2="$(field "$tmp/worse-score.json" groups.source)"
test_num "$s1 < $s0 - $SOURCE_STEP and $s2 > $s0 + $SOURCE_STEP" && ok=0 || ok=1
verdict $ok d "$(printf 'source group: default %.3f, sharper closure %.3f, softer source %.3f (bar: at least %s each way)' "$s0" "$s1" "$s2" "$SOURCE_STEP")"

# (e)
for run in two-a:2 two-b:2 one:1; do
  "$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --limit 30 --jobs "${run#*:}" --json "$tmp/same-${run%:*}.json" > /dev/null
done
"$BIN" realism score "$REF" --recordings "$TSP" --speakers MA --jobs 1 --json "$tmp/rec-1.json" > /dev/null
"$BIN" realism score "$REF" --recordings "$TSP" --speakers MA --jobs 2 --json "$tmp/rec-2.json" > /dev/null
{ cmp -s "$tmp/same-two-a.json" "$tmp/same-two-b.json" && cmp -s "$tmp/same-two-a.json" "$tmp/same-one.json" && cmp -s "$tmp/rec-1.json" "$tmp/rec-2.json"; } && ok=0 || ok=1
verdict $ok e "30 sentences rendered twice on 2 threads and once on 1, and MA's recordings on 1 and 2: the same bytes of JSON"

# (g) the first fit, before (f), which needs its score too
"$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --jobs "$jobs" --tuning "$FAILED_FIT" --json "$tmp/failed.json" > "$tmp/failed.txt"
c0="$(field "$tmp/default.json" groups.consonants 2> /dev/null || true)"; c1="$(field "$tmp/failed.json" groups.consonants 2> /dev/null || true)"; f1="$(field "$tmp/failed.json" groups.source)"
if [ -z "$c0" ] || [ -z "$c1" ]; then
  verdict 1 g "the score has no \`consonants\` group: the judge cannot see what the first fit did to the consonants"
else
  test_num "$c1 > $c0 + $CONSONANTS_STEP and $f1 < $s0 - $SOURCE_STEP" && ok=0 || ok=1
  verdict $ok g "$(printf 'the first fit against the default: consonants %.3f against %.3f (bar: at least %s worse), source %.3f against %.3f (bar: at least %s better)' "$c1" "$c0" "$CONSONANTS_STEP" "$f1" "$s0" "$SOURCE_STEP")"
fi
echo "  the default voice:"; sed -n 3,5p "$tmp/default.txt" | sed 's/^/  /'
echo "  the first fit ($FAILED_FIT):"; sed -n 3,4p "$tmp/failed.txt" | sed 's/^/  /'

# (h) the hiss, rolled off and turned up
echo "$HISS_LESS" > "$tmp/hiss-less.json"
echo "$HISS_MORE" > "$tmp/hiss-more.json"
"$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --jobs "$jobs" --tuning "$tmp/hiss-less.json" --json "$tmp/hiss-less-score.json" > "$tmp/hiss-less.txt"
"$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --jobs "$jobs" --tuning "$tmp/hiss-more.json" --json "$tmp/hiss-more-score.json" > "$tmp/hiss-more.txt"
n0="$(field "$tmp/default.json" groups.noise)"; h1="$(field "$tmp/hiss-less-score.json" groups.noise)"; h2="$(field "$tmp/hiss-more-score.json" groups.noise)"
test_num "$h1 < $n0 - $HISS_BETTER and $h2 > $n0 + $HISS_WORSE" && ok=0 || ok=1
verdict $ok h "$(printf 'noise group: default %.3f, hiss rolled off %.3f (bar: at least %s better), the first fit'"'"'s teeth %.3f (bar: at least %s worse)' "$n0" "$h1" "$HISS_BETTER" "$h2" "$HISS_WORSE")"
echo "  hiss rolled off ($HISS_LESS):"; sed -n 3,5p "$tmp/hiss-less.txt" | sed 's/^/  /'
echo "  the first fit's teeth ($HISS_MORE):"; sed -n 3,5p "$tmp/hiss-more.txt" | sed 's/^/  /'

# (f)
agree() { python3 - "$@" <<'EOF'
import json, sys
plan, mfa = (json.load(open(p))["voices"][0]["groups"] for p in sys.argv[1:3])
bad = False
for g in ["timbre", "source", "noise", "pitch", "motion", "consonants"]:
    if g not in plan or g not in mfa:
        bad = True
        print(f"    {g:10} is not in the score  <-- missing")
        continue
    room = float(sys.argv[3]) if len(sys.argv) > 3 else 0.14 if g == "timbre" else max(0.1 * mfa[g], 0.05)
    off = abs(plan[g] - mfa[g]) > room
    bad |= off
    print(f"    {g:10} plan times {plan[g]:.3f}  aligner's {mfa[g]:.3f}  apart {abs(plan[g] - mfa[g]):.3f} of {room:.3f} allowed{'  <-- too far' if off else ''}")
sys.exit(1 if bad else 0)
EOF
}
if [ -d "$MFA" ]; then
  "$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --jobs "$jobs" --times "mfa:$MFA" --json "$tmp/mfa.json" > /dev/null
  agree "$tmp/default.json" "$tmp/mfa.json" && ok=0 || ok=1
  verdict $ok f "plan times against the aligner's on the default voice, per group (bar: within 10 % or 0.05; timbre 0.14)"
else
  echo "SKIP (f) no aligned renders in $MFA (tools/plint/judge.sh makes them): plan times were NOT checked against the aligner's"
fi
if [ -d "$FAILED_MFA" ]; then
  "$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --jobs "$jobs" --tuning "$FAILED_FIT" --times "mfa:$FAILED_MFA" --json "$tmp/failed-mfa.json" > /dev/null
  agree "$tmp/failed.json" "$tmp/failed-mfa.json" 0.15 && ok=0 || ok=1
  verdict $ok f "the same on the first fit's voice (bar: within 0.15)"
else
  echo "SKIP (f) no aligned renders of the first fit in $FAILED_MFA: the plan-to-sound offsets were NOT checked on another voice than the default"
fi

echo "$(( $(date +%s) - started )) s"
if [ "$failed" = 0 ]; then echo "realism check: PASS"; else echo "realism check: FAIL"; fi
exit "$failed"
