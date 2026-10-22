#!/usr/bin/env bash
# Plint · ntfy: one admin user, one topic family. Idempotent.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
need_root; load_env

step "ntfy"
wait_for "http://127.0.0.1:8070/v1/health" 60 || die "ntfy did not come up on :8070"

if ! docker exec ntfy ntfy user list 2>/dev/null | grep -q "^user ${ADMIN_USER} "; then
  say "Creating admin user ${ADMIN_USER}"
  docker exec -e NTFY_PASSWORD="${ADMIN_PASSWORD}" ntfy \
    ntfy user add --role=admin "${ADMIN_USER}" >/dev/null
fi

# A long-lived token for scripts and the other services to publish with.
if ! grep -q '^NTFY_TOKEN=' "$PLINT_ENV" || [[ -z "$(grep '^NTFY_TOKEN=' "$PLINT_ENV" | cut -d= -f2)" ]]; then
  token="$(docker exec ntfy ntfy token add --label plint "${ADMIN_USER}" 2>/dev/null | grep -oE 'tk_[A-Za-z0-9_]+' | head -n1 || true)"
  [[ -n "$token" ]] && env_set NTFY_TOKEN "$token"
fi

ok "ntfy ready at https://ntfy.${DOMAIN} — subscribe to the topic 'plint' in the app"
