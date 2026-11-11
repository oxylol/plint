#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Ultron's voice in VoiceStudio (container `voicestudio`, table stack).
#    ultron-voice.sh status              profile, licence, loaded model
#    ultron-voice.sh profile [wav] [txt] (re)create the clone profile from a reference
#                                        clip and what is said in it
#                                        (default data/ultron/voice/ultron-ref.{wav,txt})
#    ultron-voice.sh licence             show the speech model's licence and ask
#                                        whether you accept it (VoiceStudio will not
#                                        synthesise before that)
#    ultron-voice.sh say "text" [out]    synthesise one line and time it
#  VoiceStudio is not published; this talks to it from inside its container.
# ─────────────────────────────────────────────────────────────
set -euo pipefail
PLINT_DIR="${PLINT_DIR:-/opt/plint}"
NAME="${ULTRON_VOICE:-Ultron}"
VS="docker exec voicestudio curl -s"
API="http://localhost:3900"
REPOS=(k2-fsa/OmniVoice eustlb/higgs-audio-v2-tokenizer)

docker inspect -f '{{.State.Running}}' voicestudio 2>/dev/null | grep -q true \
  || { echo "voicestudio is not running: COMPOSE_PROFILES=speech plint up table" >&2; exit 1; }

profile_id() { $VS "$API/profiles" | python3 -c 'import json,sys; n=sys.argv[1].lower(); print(next((p["id"] for p in json.load(sys.stdin) if p["name"].lower()==n), ""))' "$NAME"; }

case "${1:-status}" in
  status)
    id="$(profile_id)"; echo "profile  ${id:-none} ($NAME)"
    for r in "${REPOS[@]}"; do
      $VS "$API/models/licenses/acceptance/$r" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("licence ", d["repo_id"], "·", d["license"], "·", "accepted" if d["accepted"] else "NOT accepted")'
    done
    $VS "$API/model/status" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("model   ", d["status"], d.get("checkpoint") or "")'
    ;;
  profile)
    wav="${2:-$PLINT_DIR/data/ultron/voice/ultron-ref.wav}"; txt="${3:-${wav%.wav}.txt}"
    [[ -f "$wav" && -f "$txt" ]] || { echo "need $wav and $txt" >&2; exit 1; }
    id="$(profile_id)"
    [[ -n "$id" ]] && $VS -X DELETE "$API/profiles/$id" >/dev/null
    docker cp -q "$wav" voicestudio:/tmp/ultron-ref.wav
    $VS "$API/profiles" -F "name=$NAME" -F kind=clone -F language=English -F "ref_text=$(tr -d '\n' < "$txt")" \
      -F "ref_audio=@/tmp/ultron-ref.wav;type=audio/wav" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("profile", d["id"], d["name"])'
    ;;
  licence|license)
    for r in "${REPOS[@]}"; do
      $VS "$API/models/licenses/details/$r" | python3 -c '
import json,sys
d=json.load(sys.stdin); i=d["info"]
print(d["repo_id"]); print("  licence:", i["license"]); print("  category:", i["license_category"], "· commercial use:", i["commercial_inference"]); print("  notes:", i["notes"]); print("  terms:", i["source_url"])'
    done
    read -r -p "Accept these licences for use on this server? [y/N] " a
    [[ "$a" == y || "$a" == Y ]] || { echo "not accepted"; exit 1; }
    for r in "${REPOS[@]}"; do
      fp="$($VS "$API/models/licenses/acceptance/$r" | python3 -c 'import json,sys; print(json.load(sys.stdin)["fingerprint"])')"
      $VS -H 'content-type: application/json' "$API/models/licenses/accept" -d "{\"repo_id\":\"$r\",\"fingerprint\":\"$fp\",\"accepted\":true}" >/dev/null && echo "accepted $r"
    done
    ;;
  say)
    text="${2:?text}"; out="${3:-/tmp/ultron-say.wav}"
    body="$(python3 -c 'import json,sys; print(json.dumps({"model":"omnivoice","input":sys.argv[1],"voice":sys.argv[2],"response_format":"wav"}))' "$text" "$NAME")"
    t0=$(date +%s.%N)
    code="$($VS -m 900 -o /tmp/ultron-say.wav -w '%{http_code}' -H 'content-type: application/json' "$API/v1/audio/speech" -d "$body")"
    t1=$(date +%s.%N)
    if [[ "$code" != 200 ]]; then docker exec voicestudio head -c 400 /tmp/ultron-say.wav; echo; exit 1; fi
    docker cp -q voicestudio:/tmp/ultron-say.wav "$out"
    python3 -c 'import sys,wave; w=wave.open(sys.argv[1]); d=w.getnframes()/w.getframerate(); t=float(sys.argv[3])-float(sys.argv[2]); print(f"{sys.argv[1]}: {d:.1f}s of speech in {t:.1f}s ({t/d:.1f}x real time)")' "$out" "$t0" "$t1"
    ;;
  *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1;;
esac
