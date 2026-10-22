#!/usr/bin/env bash
# Plint · data for the Soft UI infographics that apps can't serve themselves.
# Writes data/theme/backups.json (Kopia snapshot times and sizes, no contents)
# which Caddy serves read-only as /_plint/data/backups.json. Run hourly by
# plint-theme-data.timer.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"

out="${PLINT_DIR}/data/theme"
install -d -m 0755 "$out"
tmp="$(mktemp "${out}/.backups.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

docker exec kopia kopia snapshot list --json --all 2>/dev/null | python3 -c '
import json, sys, time
snaps = json.load(sys.stdin)
cut = time.time() - 60 * 86400
rows = []
for s in snaps:
    st = s.get("stats", {})
    start = s.get("startTime", "")
    try:
        t = time.mktime(time.strptime(start[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        continue
    if t < cut:
        continue
    rows.append({
        "path": s.get("source", {}).get("path", ""),
        "start": start, "end": s.get("endTime", ""),
        "size": st.get("totalSize", 0), "files": st.get("fileCount", 0),
        "errors": st.get("errorCount", 0),
    })
rows.sort(key=lambda r: r["start"])
json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "snapshots": rows}, sys.stdout, separators=(",", ":"))
' > "$tmp"
chmod 0644 "$tmp"
mv "$tmp" "${out}/backups.json"
trap - EXIT
