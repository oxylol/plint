#!/usr/bin/env bash
# Plint · the realism judge's own proof (`chimera realism`, crates/chimera-cli/src/realism.rs):
# is it sound enough to be a fitter's objective? Six checks, each PASS or FAIL with its numbers;
# the whole output is kept in the rig's hill/realism/check.txt; exit 1 if any check fails.
#   tools/plint/realism_check.sh [--jobs 2]
#   (about 7 minutes on 2 of Plint's cores; CHIMERA_BIN=… judges another build than target/release)
# (a) each fitting man against a reference of the other five reads like his peers
# (b) the men the reference never saw (MH MI MJ, and MK ML) read like real men too
# (c) the default voice reads clearly worse than every real man
# (d) a source known to be better lowers the `source` group, one known to be worse raises it
# (e) the same input gives the same bytes, twice and on 1 or 2 threads
# (f) the plan's phone times and the aligner's give the same groups (needs a judge run's aligned
#     renders of the default voice in hill/ref442; SKIP, which is not a pass, when they are absent).
#     Timbre's bar is wider, 0.12: measured 2026-11-26 it reads 0.62 with plan times against 0.52,
#     a small one-sided offset in a dozen terms, none over 0.4 (realism.rs, PLAN_TO_SOUND, says what
#     was tried). It is the same for every tuning that leaves the timing alone, so a fit of the
#     source and the noise is not led by it; the Gate's own judge uses the aligner.
# Check (b) READS THE HELD-OUT MEN'S RECORDINGS (MK ML). That is allowed here and only here: it
# asks whether the judge is sound, and tunes nothing. Only their total scores are printed and
# kept; their per-term values are written nowhere, so no fitter can read them.
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

# The bars, with what was measured when they were set (2026-11-26, engine 95b7092, 74 scored terms).
MEN_MAX=1.0        # (a) every fitting man, left out in turn: measured 0.23 to 0.69
MEN_MEDIAN=0.6     # (a) their median: measured 0.37
UNSEEN_MAX=1.5     # (b) MH MI MJ MK ML: measured 0.08 to 0.61
SYNTH_OVER=1.25    # (c) the default voice over the worst real man of (a) and (b): measured 1.04 / 0.69 = 1.51
SOURCE_STEP=0.25   # (d) the `source` group must move at least this far each way: measured −1.0 and +1.2 from 2.45
FIT_MEN="MA MB MC MD MF MG"
# (c) to (f) say these men's sentences (120; all 360 read 1.064 where these read 1.043)
SYNTH_SPEAKERS="MA,MB"
BETTER='{"source":{"closureTension":0.6,"oqBase":0.6}}'   # sharper closure: what P1 found missing
WORSE='{"source":{"oqBase":0.9}}'                          # a softer source still

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
  men="$men $("$BIN" realism score "$tmp/loo-$m.json" --recordings "$TSP" --speakers "$m" --jobs "$jobs" | totals | tr ' ' '=')"
done
men_scores="$(echo "$men" | tr ' ' '\n' | sed -n 's/.*=//p' | paste -sd,)"
test_num "max([$men_scores]) < $MEN_MAX and sorted([$men_scores])[3] + sorted([$men_scores])[2] < 2 * $MEN_MEDIAN" && ok=0 || ok=1
verdict $ok a "each fitting man against the other five:$men (bars: all under $MEN_MAX, median under $MEN_MEDIAN)"

# (b) totals only: nothing of MK ML but these numbers leaves this pipe
unseen="$("$BIN" realism score "$REF" --recordings "$TSP" --speakers MH,MI,MJ,MK,ML --jobs "$jobs" | totals | tr ' ' '=' | paste -sd' ')"
unseen_scores="$(echo "$unseen" | tr ' ' '\n' | sed -n 's/.*=//p' | paste -sd,)"
test_num "len([$unseen_scores]) == 5 and max([$unseen_scores]) < $UNSEEN_MAX" && ok=0 || ok=1
verdict $ok b "men the reference never saw: $unseen (bar: all under $UNSEEN_MAX)"

# (c)
"$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --jobs "$jobs" --json "$tmp/default.json" > "$tmp/default.txt"
synth="$(field "$tmp/default.json" score)"
worst_man="$(python3 -c "print(max([$men_scores,$unseen_scores]))")"
test_num "$synth > $SYNTH_OVER * $worst_man" && ok=0 || ok=1
verdict $ok c "the default voice on $SYNTH_SPEAKERS's sentences reads $(printf %.3f "$synth"), the worst real man $worst_man (bar: over $SYNTH_OVER times that)"
sed -n 3p "$tmp/default.txt"

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

# (f)
if [ -d "$MFA" ]; then
  "$BIN" realism score "$REF" --speakers "$SYNTH_SPEAKERS" --jobs "$jobs" --times "mfa:$MFA" --json "$tmp/mfa.json" > /dev/null
  python3 - "$tmp/default.json" "$tmp/mfa.json" <<'EOF' && ok=0 || ok=1
import json, sys
plan, mfa = (json.load(open(p))["voices"][0]["groups"] for p in sys.argv[1:3])
bad = False
for g in ["timbre", "source", "noise", "pitch", "motion"]:
    room = 0.12 if g == "timbre" else max(0.1 * mfa[g], 0.05)
    off = abs(plan[g] - mfa[g]) > room
    bad |= off
    print(f"    {g:7} plan times {plan[g]:.3f}  aligner's {mfa[g]:.3f}  apart {abs(plan[g] - mfa[g]):.3f} of {room:.3f} allowed{'  <-- too far' if off else ''}")
sys.exit(1 if bad else 0)
EOF
  verdict $ok f "plan times against the aligner's on the default voice, per group (bar: within 10 % or 0.05; timbre 0.12)"
else
  echo "SKIP (f) no aligned renders in $MFA (tools/plint/judge.sh makes them): plan times were NOT checked against the aligner's"
fi

echo "$(( $(date +%s) - started )) s"
if [ "$failed" = 0 ]; then echo "realism check: PASS"; else echo "realism check: FAIL"; fi
exit "$failed"
