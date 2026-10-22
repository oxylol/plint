#!/usr/bin/env bash
# Plint · Kopia: create the repository once, then a nightly policy. Idempotent.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
need_root; load_env

step "Backups (Kopia)"
mkdir -p "${BACKUP_PATH}"
kopia() { docker exec kopia kopia "$@"; }

wait_for "http://127.0.0.1:51515/" 60 || true

if ! kopia repository status >/dev/null 2>&1; then
  if [[ -f "${BACKUP_PATH}/kopia.repository.f" ]]; then
    say "Connecting to existing repository in ${BACKUP_PATH}"
    kopia repository connect filesystem --path=/repository --password="${ADMIN_PASSWORD}" >/dev/null
  else
    say "Creating repository in ${BACKUP_PATH}"
    kopia repository create filesystem --path=/repository --password="${ADMIN_PASSWORD}" >/dev/null
  fi
  # The server process needs a restart to pick up the connection.
  docker restart kopia >/dev/null
  wait_for "http://127.0.0.1:51515/" 60 || true
fi

say "Policy → nightly 03:30, keep 14 daily / 8 weekly / 6 monthly, zstd"
kopia policy set --global \
  --compression=zstd \
  --keep-latest=3 --keep-hourly=0 --keep-daily=14 --keep-weekly=8 --keep-monthly=6 --keep-annual=0 \
  --snapshot-time=03:30 >/dev/null

# The server only schedules sources that already have a snapshot, so each one
# needs a first snapshot. `snapshot list` succeeds even when empty: check output.
# It runs detached because AMP can be 100+ GB.
for src in /sources/plint-data /sources/plint.env /sources/amp; do
  docker exec kopia test -e "$src" 2>/dev/null || continue
  kopia policy set "$src" --snapshot-time=03:30 >/dev/null 2>&1 || true
  if [[ -z "$(kopia snapshot list "$src" 2>/dev/null)" ]]; then
    docker exec -d kopia kopia snapshot create "$src" && say "First snapshot of $src started in the background"
  fi
done

ok "Kopia ready at https://backup.${DOMAIN} (login: ${ADMIN_USER})"
