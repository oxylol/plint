#!/usr/bin/env bash
# Plint · AdGuard Home first-run configuration, done through its own API so it
# works across AdGuard versions. Idempotent: safe to run again.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
need_root; load_env

AG="http://127.0.0.1:3000"
AUTH=(-u "${ADMIN_USER}:${ADMIN_PASSWORD}")
json() { curl -fsS "$@" -H 'Content-Type: application/json'; }

step "AdGuard Home"
wait_for "$AG/" 90 || die "AdGuard did not come up on :3000"

# ── First run: create the admin user and bind DNS to :53 ─────
if curl -fsS "$AG/control/install/get_addresses" >/dev/null 2>&1; then
  say "First run — creating admin user and binding DNS"
  json -X POST "$AG/control/install/configure" -d "$(cat <<EOF
{
  "web":      {"ip": "0.0.0.0", "port": 3000},
  "dns":      {"ip": "0.0.0.0", "port": 53},
  "username": "${ADMIN_USER}",
  "password": "${ADMIN_PASSWORD}"
}
EOF
)" >/dev/null
  sleep 3
  wait_for "$AG/control/status" 60 || true
fi

# ── Upstreams: encrypted DNS, fast bootstrap, parallel ───────
say "Upstreams → Cloudflare + Quad9 over HTTPS"
json "${AUTH[@]}" -X POST "$AG/control/dns_config" -d "$(cat <<'EOF'
{
  "upstream_dns": [
    "https://dns.cloudflare.com/dns-query",
    "https://dns.quad9.net/dns-query"
  ],
  "upstream_mode": "parallel",
  "bootstrap_dns": ["1.1.1.1", "9.9.9.9"],
  "fallback_dns": ["1.0.0.1"],
  "cache_size": 4194304,
  "cache_optimistic": true,
  "dnssec_enabled": true,
  "protection_enabled": true,
  "blocking_mode": "default",
  "ratelimit": 0,
  "edns_cs_enabled": false
}
EOF
)" >/dev/null

# ── Rewrites: every name under the domain → this box ─────────
say "Rewrites → *.${DOMAIN} and ${DOMAIN} resolve to ${SERVER_IP}"
existing="$(json "${AUTH[@]}" "$AG/control/rewrite/list")"
for name in "${DOMAIN}" "*.${DOMAIN}"; do
  if ! grep -q "\"domain\":\"${name}\"" <<<"$existing"; then
    json "${AUTH[@]}" -X POST "$AG/control/rewrite/add" \
      -d "{\"domain\":\"${name}\",\"answer\":\"${SERVER_IP}\"}" >/dev/null
  fi
done

# ── Blocklists: a sane, low-breakage set ─────────────────────
say "Blocklists"
add_list() {
  local name="$1" url="$2"
  if ! json "${AUTH[@]}" "$AG/control/filtering/status" | grep -q "\"url\":\"${url}\""; then
    json "${AUTH[@]}" -X POST "$AG/control/filtering/add_url" \
      -d "{\"name\":\"${name}\",\"url\":\"${url}\",\"whitelist\":false}" >/dev/null || warn "could not add ${name}"
  fi
}
add_list "AdGuard DNS filter"        "https://adguardteam.github.io/HostlistsRegistry/assets/filter_1.txt"
add_list "HaGeZi Pro"                "https://adguardteam.github.io/HostlistsRegistry/assets/filter_48.txt"
add_list "HaGeZi Threat Intelligence" "https://adguardteam.github.io/HostlistsRegistry/assets/filter_44.txt"
add_list "OISD Small"                "https://adguardteam.github.io/HostlistsRegistry/assets/filter_5.txt"
json "${AUTH[@]}" -X POST "$AG/control/filtering/config" \
  -d '{"enabled":true,"interval":24}' >/dev/null
json "${AUTH[@]}" -X POST "$AG/control/filtering/refresh" -d '{"whitelist":false}' >/dev/null || true

# ── Safe browsing on, parental off, query log 30 days ────────
curl -fsS "${AUTH[@]}" -X POST "$AG/control/safebrowsing/enable" >/dev/null || true
json "${AUTH[@]}" -X PUT "$AG/control/querylog/config/update" \
  -d '{"enabled":true,"interval":2592000000,"anonymize_client_ip":false}' >/dev/null || true
json "${AUTH[@]}" -X PUT "$AG/control/stats/config/update" \
  -d '{"enabled":true,"interval":2592000000}' >/dev/null || true

ok "AdGuard ready at https://dns.${DOMAIN}"
