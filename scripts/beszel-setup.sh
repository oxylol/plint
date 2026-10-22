#!/usr/bin/env bash
# Plint · Beszel: copy the hub's public key into .env so the agent trusts it.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
need_root; load_env

step "Metrics (Beszel)"
wait_for "http://127.0.0.1:8090/" 60 || die "Beszel hub did not come up on :8090"

# The hub writes only its private key (id_ed25519) into its data dir; the
# public key the agent needs is derived from it.
keyfile="${PLINT_DIR}/data/beszel/hub/id_ed25519"
for _ in $(seq 1 30); do [[ -s "$keyfile" ]] && break; sleep 1; done
if [[ ! -s "$keyfile" ]]; then
  warn "Hub key not found at ${keyfile}. Copy the public key from 'Add System' in the hub UI into BESZEL_KEY in .env, then: plint up ops"
  exit 0
fi

pub="$(ssh-keygen -y -f "$keyfile" | awk '{print $1" "$2}')"
[[ "$pub" == ssh-ed25519* ]] || die "Could not derive a public key from ${keyfile}"
if [[ "${BESZEL_KEY:-}" != "$pub" ]]; then
  env_set BESZEL_KEY "$pub"
  say "Restarting agent with the hub key"
  compose ops up -d beszel-agent >/dev/null
fi
ok "Beszel ready at https://metrics.${DOMAIN} — add system 'plint' with host /beszel_socket/beszel.sock"
