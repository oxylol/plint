#!/usr/bin/env bash
# Plint · Claude on the box.
# Installs Claude Code from Anthropic's apt repo, signs in with your claude.ai
# account, and runs it as a Remote Control server so the machine appears as
# "Plint" in the Claude app. Idempotent.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
need_root; load_env

step "Claude on the box"

# ── Package ──────────────────────────────────────────────────
if ! command -v claude >/dev/null 2>&1; then
  say "Installing Claude Code (stable channel)"
  install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://downloads.claude.ai/keys/claude-code.asc -o /etc/apt/keyrings/claude-code.asc
  fp="$(gpg --show-keys --with-colons /etc/apt/keyrings/claude-code.asc 2>/dev/null | awk -F: '$1=="fpr"{print $10; exit}')"
  [[ "$fp" == "31DDDE24DDFAB679F42D7BD2BAA929FF1A7ECACE" ]] || die "Claude Code signing key fingerprint mismatch: $fp"
  echo "deb [signed-by=/etc/apt/keyrings/claude-code.asc] https://downloads.claude.ai/claude-code/apt/stable stable main" \
    > /etc/apt/sources.list.d/claude-code.list
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq claude-code tmux >/dev/null
fi
ok "claude $(claude --version 2>/dev/null | head -n1)"

# ── Sign in (one time, interactive) ──────────────────────────
if ! sudo -u plint -H claude auth status >/dev/null 2>&1; then
  cat <<EOF
${C_DIM}Sign in with the claude.ai account you use on your phone (Pro or Max).
A link appears; open it on any device, approve, paste the code back here.${C_RESET}
EOF
  sudo -u plint -H claude auth login
fi

# ── Skills ───────────────────────────────────────────────────
# Emil Kowalski's design-engineering skills (emil-design-eng, review-animations,
# mobile-native, break-ui, …) for design reviews of the Plint Soft UI.
if [[ ! -d /home/plint/.claude/skills/emil-design-eng ]]; then
  say "Installing emilkowalski/skills"
  tmp="$(mktemp -d)"
  git clone -q --depth 1 https://github.com/emilkowalski/skills "$tmp/skills"
  install -d -o plint -g plint /home/plint/.claude /home/plint/.claude/skills
  cp -r "$tmp/skills/skills/." /home/plint/.claude/skills/
  chown -R plint:plint /home/plint/.claude/skills
  rm -rf "$tmp"
fi
ok "design skills (emilkowalski/skills)"

# ── Service ──────────────────────────────────────────────────
install -m 0644 "${PLINT_DIR}/systemd/plint-claude.service" /etc/systemd/system/plint-claude.service
systemctl daemon-reload
systemctl enable --now plint-claude.service >/dev/null
sleep 3

cat <<EOF

${C_BOLD}Last step — answer two one-time prompts:${C_RESET}
  1. run:  ${C_BOLD}plint claude${C_RESET}
  2. answer ${C_BOLD}y${C_RESET} to "Trust /opt/plint?" and ${C_BOLD}y${C_RESET} to "Enable Remote Control?"
  3. leave it running: press ${C_BOLD}Ctrl+B${C_RESET}, then ${C_BOLD}D${C_RESET}

The server then shows up as ${C_BOLD}"Plint"${C_RESET} in the Claude app and at claude.ai/code.
EOF
