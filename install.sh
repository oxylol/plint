#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Plint · install
#
#  Turns a fresh Ubuntu 24.04 / 26.04 box into Plint. Safe to re-run:
#  every step checks before it changes anything.
#
#    sudo -i
#    git clone git@github.com:oxylol/plint.git /opt/plint
#    /opt/plint/install.sh
#
#  Flags:  --skip-amp   --skip-claude   --skip-tailscale
# ─────────────────────────────────────────────────────────────
set -euo pipefail
PLINT_DIR="/opt/plint"
export PLINT_DIR
# shellcheck source=scripts/lib.sh
source "${PLINT_DIR}/scripts/lib.sh"
need_root

SKIP_AMP=0; SKIP_CLAUDE=0; SKIP_TS=0
for a in "$@"; do case "$a" in
  --skip-amp) SKIP_AMP=1;; --skip-claude) SKIP_CLAUDE=1;; --skip-tailscale) SKIP_TS=1;;
  *) die "unknown flag $a";; esac; done

printf '%s\n' "$C_BOLD" \
"   ___  _ _     _   " \
"  | _ \| (_)_ _| |_ " \
"  |  _/| | | ' \  _|" \
"  |_|  |_|_|_||_\__|   home server, as code" "$C_RESET"

# ── 0. Preflight ─────────────────────────────────────────────
step "Preflight"
. /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "Ubuntu only (found ${PRETTY_NAME:-unknown})"
[[ "${VERSION_ID%%.*}" -ge 24 ]] || die "Ubuntu 24.04 or newer required"
curl -fsS -o /dev/null --max-time 10 https://github.com || die "No internet"
ok "${PRETTY_NAME}"

# ── 1. Answers ───────────────────────────────────────────────
step "A few answers"
if [[ ! -f "$PLINT_ENV" ]]; then cp "${PLINT_DIR}/.env.example" "$PLINT_ENV"; chmod 600 "$PLINT_ENV"; fi
load_env
ask() { # ask VAR "Prompt" [default] [secret]
  local var="$1" prompt="$2" def="${3:-}" secret="${4:-}" cur val
  cur="${!var:-}"; [[ -n "$cur" ]] && def="$cur"
  if [[ -n "$secret" ]]; then
    read -r -s -p "  ${prompt}${def:+ [••••]}: " val; echo
  else
    read -r -p "  ${prompt}${def:+ [$def]}: " val
  fi
  val="${val:-$def}"
  [[ -n "$val" ]] || die "${var} is required"
  env_set "$var" "$val"; export "$var=$val"
}
ask DOMAIN         "Domain" "plint.net"
ask TZ             "Timezone" "Europe/Stockholm"
ask CF_API_TOKEN   "Cloudflare API token (Zone·DNS·Edit on ${DOMAIN})" "" secret
ask ADMIN_USER     "Admin username" "admin"
ask ADMIN_PASSWORD "Admin password (AdGuard, Kopia, ntfy)" "" secret
ask BACKUP_PATH    "Backup folder" "/srv/backups"
SERVER_IP="$(detect_ip || true)"; LAN_CIDR="$(detect_cidr 2>/dev/null || true)"
ask SERVER_IP "This box's LAN IP" "${SERVER_IP}"
ask LAN_CIDR  "LAN subnet (for Tailscale routing)" "${LAN_CIDR}"
load_env
ok "this box is ${SERVER_IP} on ${LAN_CIDR}"

# ── 2. Host basics ───────────────────────────────────────────
step "Host"
hostnamectl set-hostname plint
grep -q '127.0.1.1 plint' /etc/hosts || echo "127.0.1.1 plint" >> /etc/hosts
timedatectl set-timezone "$TZ"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg git tmux jq dnsutils \
  unattended-upgrades apt-listchanges python3 openssl wget >/dev/null
install -m 0644 "${PLINT_DIR}/systemd/unattended-upgrades-plint.conf" /etc/apt/apt.conf.d/52plint
install -m 0644 "${PLINT_DIR}/systemd/sysctl-plint.conf" /etc/sysctl.d/99-plint.conf
sysctl -q --system >/dev/null

# the plint user owns the repo; Claude runs as it; it can sudo and use docker
if ! id plint >/dev/null 2>&1; then
  useradd -m -s /bin/bash -U plint
fi
echo 'plint ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/plint; chmod 440 /etc/sudoers.d/plint
env_set PUID "$(id -u plint)"; env_set PGID "$(id -g plint)"
mkdir -p "${PLINT_DIR}/data" "${BACKUP_PATH}"
# the repo belongs to plint; container data stays as the containers wrote it
find "${PLINT_DIR}" -path "${PLINT_DIR}/data" -prune -o -exec chown plint:plint {} +
git config --global --add safe.directory "${PLINT_DIR}" >/dev/null 2>&1 || true
# let the plint user pull the repo with the same deploy key root cloned with
if [[ -f /root/.ssh/id_ed25519 && ! -f /home/plint/.ssh/id_ed25519 ]]; then
  install -d -m 700 -o plint -g plint /home/plint/.ssh
  install -m 600 -o plint -g plint /root/.ssh/id_ed25519 /home/plint/.ssh/id_ed25519
  install -m 644 -o plint -g plint /root/.ssh/id_ed25519.pub /home/plint/.ssh/id_ed25519.pub
  ssh-keyscan -t ed25519 github.com 2>/dev/null >> /home/plint/.ssh/known_hosts
  chown plint:plint /home/plint/.ssh/known_hosts
fi
ln -sf "${PLINT_DIR}/bin/plint" /usr/local/bin/plint
chmod +x "${PLINT_DIR}/bin/plint" "${PLINT_DIR}"/scripts/*.sh "${PLINT_DIR}/install.sh"
ok "hostname plint · user plint · plint CLI"

# ── 3. DNS plumbing: AdGuard owns :53 ───────────────────────
step "DNS plumbing"
install -d /etc/systemd/resolved.conf.d
install -m 0644 "${PLINT_DIR}/systemd/resolved-plint.conf" /etc/systemd/resolved.conf.d/plint.conf
ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf
systemctl restart systemd-resolved
ok "stub listener off; host resolves via 127.0.0.1 with public fallback"

# ── 4. Docker ────────────────────────────────────────────────
step "Docker"
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin >/dev/null
fi
usermod -aG docker plint
cat > /etc/docker/daemon.json <<'EOF'
{ "log-driver": "json-file", "log-opts": { "max-size": "10m", "max-file": "3" } }
EOF
systemctl enable --now docker >/dev/null
ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"

# ── 5. Cockpit ───────────────────────────────────────────────
step "Cockpit"
# (no cockpit-networkmanager: it would pull NetworkManager onto a netplan box)
apt-get install -y -qq cockpit cockpit-storaged cockpit-packagekit >/dev/null 2>&1 || apt-get install -y -qq cockpit >/dev/null
install -d /etc/cockpit
sed "s|\${DOMAIN}|${DOMAIN}|g" "${PLINT_DIR}/systemd/cockpit.conf" > /etc/cockpit/cockpit.conf
systemctl enable --now cockpit.socket >/dev/null
ok "cockpit on :9090 → https://host.${DOMAIN}"

# ── 6. Stacks ────────────────────────────────────────────────
"${PLINT_DIR}/scripts/cleanup.sh"
step "Stacks"
for s in core ops media; do ln -sf "${PLINT_ENV}" "${PLINT_DIR}/stacks/${s}/.env"; done
say "building Caddy (first time takes a minute)"
compose core build --quiet caddy
compose core up -d --remove-orphans
compose ops  up -d --remove-orphans
compose media up -d --remove-orphans
ok "core · ops · media are up"

"${PLINT_DIR}/scripts/adguard-setup.sh"
"${PLINT_DIR}/scripts/beszel-setup.sh"
"${PLINT_DIR}/scripts/ntfy-setup.sh"
"${PLINT_DIR}/scripts/kopia-setup.sh"

# ── 7. Tailscale ─────────────────────────────────────────────
if (( ! SKIP_TS )); then
  step "Tailscale"
  if ! command -v tailscale >/dev/null 2>&1; then
    curl -fsSL https://tailscale.com/install.sh | sh >/dev/null 2>&1
  fi
  systemctl enable --now tailscaled >/dev/null
  if ! tailscale status >/dev/null 2>&1; then
    printf '%s\n' "${C_DIM}A login link appears below. Open it on any device and approve this machine.${C_RESET}"
    tailscale up --ssh --advertise-routes="${LAN_CIDR}" --accept-dns=false --hostname=plint
  else
    tailscale set --ssh --advertise-routes="${LAN_CIDR}" --accept-dns=false --hostname=plint >/dev/null 2>&1 || true
  fi
  ok "tailscale ip $(tailscale ip -4 | head -n1)"
fi

# ── 8. AMP ───────────────────────────────────────────────────
(( SKIP_AMP )) || "${PLINT_DIR}/scripts/amp-install.sh"

# ── 9. Claude ────────────────────────────────────────────────
(( SKIP_CLAUDE )) || "${PLINT_DIR}/scripts/claude-setup.sh"

# ── Done ─────────────────────────────────────────────────────
ts_ip="$(tailscale ip -4 2>/dev/null | head -n1 || echo '<tailscale ip>')"
cat <<EOF

${C_BOLD}Plint is up.${C_RESET}

  Home          https://${DOMAIN}
  Ad blocking   https://dns.${DOMAIN}
  Games         https://games.${DOMAIN}
  Stremio       https://stremio.${DOMAIN}
  Host          https://host.${DOMAIN}
  Stacks        https://stacks.${DOMAIN}
  Metrics       https://metrics.${DOMAIN}
  Status        https://status.${DOMAIN}
  Push          https://ntfy.${DOMAIN}
  Backups       https://backup.${DOMAIN}

${C_BOLD}Three things only you can do:${C_RESET}
  1. Router → DHCP → DNS server = ${SERVER_IP}   (every device gets ad blocking + the names)
  2. Tailscale admin → Machines → plint → approve subnet route ${LAN_CIDR}
     Tailscale admin → DNS → Nameservers → add ${ts_ip}, enable "Override DNS servers"
  3. Open https://${DOMAIN} and enjoy it.

  Then: plint status · plint doctor · plint claude
EOF
"${PLINT_DIR}/scripts/notify.sh" "Plint is up" "Install finished on $(hostname). Open https://${DOMAIN}" default "white_check_mark" || true
