#!/usr/bin/env bash
# Plint · clear the way.
# Stops and removes whatever already holds a port Plint needs: leftover
# containers (never AMP's, never ours) and native services like Pi-hole,
# dnsmasq, nginx. Prints every thing it removes. Idempotent.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
need_root

PORTS=(53 80 443 3000 3100 5001 8090 3001 8070 51515 8095)
OURS=" caddy adguard homepage socket-proxy dockge beszel beszel-agent uptime-kuma ntfy kopia stremio "

step "Clearing the way"
removed=0

# ── 1. Containers that publish a port we need ────────────────
while IFS=$'\t' read -r id name image ports; do
  [[ -z "$id" ]] && continue
  [[ "$OURS" == *" $name "* ]] && continue
  [[ "$name" == AMP_* || "$image" == *cubecoders* || "$image" == *ampbase* ]] && continue
  for p in "${PORTS[@]}"; do
    if [[ "$ports" =~ :${p}-\> ]]; then
      warn "removing container ${name} (${image}) — it holds port ${p}"
      docker rm -f "$id" >/dev/null
      removed=$((removed + 1))
      break
    fi
  done
done < <(docker ps -a --format '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Ports}}' 2>/dev/null || true)

# ── 2. Native services on the ports that matter most ─────────
holder() { ss -H -lntup "sport = :$1" 2>/dev/null | grep -oP 'users:\(\("\K[^"]+' | sort -u | head -n1 || true; }
for p in 53 80 443 3000; do
  proc="$(holder "$p")"
  [[ -z "$proc" ]] && continue
  unit=""
  case "$proc" in
    docker-proxy|dockerd|systemd-resolve) continue;;
    pihole-FTL) unit=pihole-FTL;;   dnsmasq) unit=dnsmasq;;
    named) unit=named;;             unbound) unit=unbound;;
    AdGuardHome) unit=AdGuardHome;; nginx) unit=nginx;;
    apache2) unit=apache2;;         lighttpd) unit=lighttpd;;
    caddy) unit=caddy;;             haproxy) unit=haproxy;;
  esac
  if [[ -n "$unit" ]] && systemctl list-unit-files "${unit}.service" >/dev/null 2>&1; then
    warn "stopping ${unit} — it holds port ${p}"
    systemctl disable --now "${unit}.service" >/dev/null 2>&1 || true
    removed=$((removed + 1))
  else
    warn "port ${p} is held by '${proc}' and I don't know its service — stop it, then re-run"
  fi
done

if (( removed == 0 )); then ok "nothing in the way"; else ok "cleared ${removed} thing(s)"; fi
