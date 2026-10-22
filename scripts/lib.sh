#!/usr/bin/env bash
# Plint · shared helpers for install.sh and scripts/*.sh
# shellcheck disable=SC2034

PLINT_DIR="${PLINT_DIR:-/opt/plint}"
PLINT_ENV="${PLINT_DIR}/.env"

if [[ -t 1 ]]; then
  C_DIM=$'\e[2m'; C_BOLD=$'\e[1m'; C_BLUE=$'\e[34m'; C_GREEN=$'\e[32m'
  C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_RESET=$'\e[0m'
else
  C_DIM=""; C_BOLD=""; C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_RESET=""
fi

say()   { printf '%s▸ %s%s\n' "$C_BLUE" "$*" "$C_RESET"; }
ok()    { printf '%s✓ %s%s\n' "$C_GREEN" "$*" "$C_RESET"; }
warn()  { printf '%s! %s%s\n' "$C_YELLOW" "$*" "$C_RESET"; }
die()   { printf '%s✗ %s%s\n' "$C_RED" "$*" "$C_RESET" >&2; exit 1; }
step()  { printf '\n%s%s%s\n%s%s%s\n' "$C_BOLD" "$*" "$C_RESET" "$C_DIM" "$(printf '─%.0s' $(seq 1 60))" "$C_RESET"; }

need_root() { [[ $EUID -eq 0 ]] || die "Run as root (sudo -i first)."; }

load_env() {
  [[ -f "$PLINT_ENV" ]] || die "Missing $PLINT_ENV — run install.sh first."
  # A damaged line must not take the whole script down; env_set repairs it.
  set -a; set +e
  # shellcheck disable=SC1090
  source "$PLINT_ENV"
  set -e; set +a
}

# env_set KEY VALUE — create or update a key in .env without touching the rest.
# Values with spaces or shell-special characters are single-quoted, which both
# bash and Docker Compose read back literally.
env_set() {
  local key="$1" val="$2" out
  if [[ "$val" =~ [^A-Za-z0-9_./:@,=+-] ]]; then
    out="'${val//\'/\'\\\'\'}'"
  else
    out="$val"
  fi
  touch "$PLINT_ENV"
  if grep -qE "^${key}=" "$PLINT_ENV"; then
    NEWVAL="$out" awk -v k="$key" 'BEGIN{v=ENVIRON["NEWVAL"]} index($0,k"=")==1 {print k"="v; next} {print}' \
      "$PLINT_ENV" > "${PLINT_ENV}.tmp" && cat "${PLINT_ENV}.tmp" > "$PLINT_ENV" && rm -f "${PLINT_ENV}.tmp"
  else
    printf '%s=%s\n' "$key" "$out" >> "$PLINT_ENV"
  fi
}

# wait_for URL [seconds] — poll until an HTTP endpoint answers.
wait_for() {
  local url="$1" secs="${2:-60}" i=0
  until curl -fsS -o /dev/null --max-time 3 "$url" 2>/dev/null; do
    (( i++ >= secs )) && return 1
    sleep 1
  done
}

# compose STACK args... — run docker compose for a stack by name.
compose() {
  local stack="$1"; shift
  docker compose --project-directory "${PLINT_DIR}/stacks/${stack}" "$@"
}

# detect_ip / detect_cidr — the LAN address this box answers on.
detect_ip()   { ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | head -n1; }
detect_cidr() {
  local ip; ip="$(detect_ip)"
  ip -4 -o addr show | awk -v ip="$ip" '$4 ~ "^"ip"/" {print $4}' | head -n1 \
    | python3 -c 'import ipaddress,sys; print(ipaddress.ip_network(sys.stdin.read().strip(), strict=False))'
}
