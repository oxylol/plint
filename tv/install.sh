#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Plint TV · install
#
#  Turns the Mac on the TV into the second node of Plint. Run as the
#  user who is logged in on the TV (not root); it asks for sudo when it
#  needs it. No Homebrew: every app comes straight from its publisher.
#  Safe to re-run.
#
#    git clone git@github.com:oxylol/plint.git ~/plint
#    ~/plint/tv/install.sh
#
#  Flags:  --agents-only   (re)install the launch agents and stop
#          --power         redo the power settings and stop
#          --skip-claude
# ─────────────────────────────────────────────────────────────
set -euo pipefail

TV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$HOME/.config/plint-tv"
LOGS="$HOME/Library/Logs/plint-tv"
AGENTS="$HOME/Library/LaunchAgents"
BIN="$HOME/.local/bin"
ME="$(id -un)"
export PATH="$BIN:/usr/local/bin:$PATH"

if [[ -t 1 ]]; then C_DIM=$'\e[2m'; C_BOLD=$'\e[1m'; C_BLUE=$'\e[34m'; C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_RESET=$'\e[0m'
else C_DIM=""; C_BOLD=""; C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_RESET=""; fi
say()  { printf '%s▸ %s%s\n' "$C_BLUE" "$*" "$C_RESET"; }
ok()   { printf '%s✓ %s%s\n' "$C_GREEN" "$*" "$C_RESET"; }
warn() { printf '%s! %s%s\n' "$C_YELLOW" "$*" "$C_RESET"; }
die()  { printf '%s✗ %s%s\n' "$C_RED" "$*" "$C_RESET" >&2; exit 1; }
step() { printf '\n%s%s%s\n%s%s%s\n' "$C_BOLD" "$*" "$C_RESET" "$C_DIM" "$(printf '─%.0s' $(seq 1 60))" "$C_RESET"; }

AGENTS_ONLY=0; POWER_ONLY=0; SKIP_CLAUDE=0
for a in "$@"; do case "$a" in
  --agents-only) AGENTS_ONLY=1;; --power) POWER_ONLY=1;; --skip-claude) SKIP_CLAUDE=1;;
  *) die "unknown flag $a";; esac; done

[[ "$(uname -s)" == Darwin ]] || die "This is the Mac installer."
[[ $EUID -ne 0 ]] || die "Run as your normal user, not root."

# ── launch agents (also used by `tv sync`) ───────────────────
install_agents() {
  step "Launch agents"
  mkdir -p "$AGENTS" "$LOGS"
  # shellcheck disable=SC1091
  [[ -f "$CONF/env" ]] && source "$CONF/env"
  for src in "$TV_DIR"/launchd/*.plist; do
    local label; label="$(basename "$src" .plist)"
    if [[ "$label" == net.plint.tv.beszel && -z "${BESZEL_KEY:-}" ]]; then continue; fi
    sed -e "s|__HOME__|$HOME|g" -e "s|__KEY__|${BESZEL_KEY:-}|g" "$src" > "$AGENTS/$label.plist"
    chmod 600 "$AGENTS/$label.plist"
    plutil -lint "$AGENTS/$label.plist" >/dev/null
    launchctl bootout "gui/$(id -u)/$label" >/dev/null 2>&1 || true
    launchctl bootstrap "gui/$(id -u)" "$AGENTS/$label.plist"
    launchctl enable "gui/$(id -u)/$label" >/dev/null 2>&1 || true
    ok "$label"
  done
}

# ── power: never sleep, screen off at night ──────────────────
power_settings() {
  step "Power"
  sudo pmset -a sleep 0 disksleep 0 displaysleep 30 womp 1 autorestart 1 powernap 0 ttyskeepawake 1 lidwake 1 >/dev/null
  sudo pmset -a disablesleep 1 >/dev/null 2>&1 || true   # the lid sensor can't put it to sleep
  sudo systemsetup -setrestartfreeze on >/dev/null 2>&1 || true
  sudo systemsetup -setwakeonnetworkaccess on >/dev/null 2>&1 || true
  sudo systemsetup -setremotelogin on >/dev/null 2>&1 || true
  ok "never sleeps · display off after 30 min and at 01:00 · restarts after a freeze · SSH on"
}

if (( AGENTS_ONLY )); then install_agents; exit 0; fi
if (( POWER_ONLY )); then power_settings; exit 0; fi

printf '%s\n' "$C_BOLD" "  Plint TV" "  the Mac on the TV, as code" "$C_RESET"

# ── 0. Preflight ─────────────────────────────────────────────
step "Preflight"
ver="$(sw_vers -productVersion)"
[[ "${ver%%.*}" -ge 13 ]] || die "macOS 13 or newer required (found $ver)"
ok "macOS $ver on $(uname -m)"
if ! xcode-select -p >/dev/null 2>&1; then
  say "Installing the command line tools (a dialog appears; click Install, not Get Xcode)"
  xcode-select --install >/dev/null 2>&1 || true
  until xcode-select -p >/dev/null 2>&1; do sleep 10; done
fi
ok "command line tools"
curl -fsS -o /dev/null --max-time 10 https://github.com || die "No internet"

# ── 1. Answers ───────────────────────────────────────────────
step "A few answers"
mkdir -p "$CONF" "$BIN"; chmod 700 "$CONF"
# shellcheck disable=SC1091
[[ -f "$CONF/env" ]] && source "$CONF/env"
read -r -p "  Plint domain [${DOMAIN:-plint.net}]: " v; DOMAIN="${v:-${DOMAIN:-plint.net}}"
read -r -p "  Beszel hub key (BESZEL_KEY from Plint's .env; blank to skip) [${BESZEL_KEY:+••••}]: " v; BESZEL_KEY="${v:-${BESZEL_KEY:-}}"
printf 'DOMAIN=%s\nBESZEL_KEY=%q\n' "$DOMAIN" "$BESZEL_KEY" > "$CONF/env"
if [[ ! -s "$CONF/key" ]]; then openssl rand -hex 24 > "$CONF/key"; fi
chmod 600 "$CONF/key" "$CONF/env"
KEY="$(cat "$CONF/key")"
ok "saved to $CONF"

# ── 2. Identity ──────────────────────────────────────────────
step "Identity"
sudo scutil --set ComputerName "Plint TV"
sudo scutil --set HostName tv
sudo scutil --set LocalHostName tv
ok "this Mac is 'tv' → https://tv.${DOMAIN}"

# ── 3. Apps ──────────────────────────────────────────────────
step "Apps"
# shellcheck source=apps.sh
source "$TV_DIR/apps.sh"
install_all
chmod +x "$TV_DIR/bin/tv" "$TV_DIR/install.sh" "$TV_DIR/remote/tv-remote.py"
sudo mkdir -p /usr/local/bin && sudo ln -sf "$TV_DIR/bin/tv" /usr/local/bin/tv
ok "the tv command is ready"

# YouTube in its TV layout: Chrome in its own profile with a TV user agent.
say "YouTube TV app"
mkdir -p "$HOME/Applications"
tmp="$(mktemp -t ytv).applescript"
cat > "$tmp" <<EOF
do shell script "/usr/bin/open -na 'Google Chrome' --args '--app=https://www.youtube.com/tv' '--user-agent=Mozilla/5.0 (SMART-TV; Linux; Tizen 5.0) AppleWebKit/538.1 (KHTML, like Gecko) Version/5.0 NativeTVAds Safari/538.1' '--user-data-dir=$HOME/Library/Application Support/PlintTV/youtube' --start-fullscreen --no-first-run --no-default-browser-check"
EOF
rm -rf "$HOME/Applications/YouTube TV.app"
osacompile -o "$HOME/Applications/YouTube TV.app" "$tmp" && rm -f "$tmp"
ok "$HOME/Applications/YouTube TV.app"

# ── 4. Power ─────────────────────────────────────────────────
power_settings

# ── 5. TV mode ───────────────────────────────────────────────
step "TV mode"
defaults write com.apple.dock autohide -bool true
defaults write com.apple.dock autohide-delay -float 0.2
defaults write com.apple.dock show-recents -bool false
defaults write com.apple.dock tilesize -int 56
for c in tl tr bl br; do defaults write com.apple.dock "wvous-$c-corner" -int 0; done
defaults write NSGlobalDomain _HIHideMenuBar -bool true
defaults write com.apple.finder CreateDesktop -bool false
defaults write com.apple.CrashReporter DialogType -string none
defaults write NSGlobalDomain NSAutomaticSpellingCorrectionEnabled -bool false
defaults -currentHost write com.apple.screensaver idleTime -int 900
if [[ -d "$HOME/Library/Screen Savers/Aerial.saver" ]]; then
  defaults -currentHost write com.apple.screensaver moduleDict -dict moduleName Aerial path "$HOME/Library/Screen Savers/Aerial.saver" type 0 || true
fi
defaults write com.apple.universalaccess mouseDriverCursorSize -float 2.0 2>/dev/null || true
killall Dock Finder SystemUIServer 2>/dev/null || true
# updates download themselves; nothing installs or reboots on its own
sudo defaults write /Library/Preferences/com.apple.SoftwareUpdate AutomaticDownload -bool true
sudo defaults write /Library/Preferences/com.apple.SoftwareUpdate AutomaticallyInstallMacOSUpdates -bool false
sudo defaults write /Library/Preferences/com.apple.SoftwareUpdate ConfigDataInstall -bool true
sudo defaults write /Library/Preferences/com.apple.SoftwareUpdate CriticalUpdateInstall -bool true
sudo defaults write /Library/Preferences/com.apple.commerce AutoUpdate -bool true
# screen sharing on, for Screens / VNC over the tailnet
sudo /System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart \
  -activate -configure -access -on -users "$ME" -privs -all -restart -agent -menu >/dev/null 2>&1 || warn "could not enable Screen Sharing — turn it on in System Settings → General → Sharing"
# sound to the TV if we can see it
if [[ -x "$BIN/audiodev" ]]; then
  tvout="$("$BIN/audiodev" list | grep -iE 'hdmi|tv|display|receiver' | head -n1 || true)"
  [[ -n "$tvout" ]] && "$BIN/audiodev" set "$tvout" && ok "sound → $tvout"
fi
ok "dock and menu bar hidden · no desktop icons · Aerial after 15 min · updates download only"

# ── 6. Tailscale ─────────────────────────────────────────────
step "Tailscale"
open -a Tailscale 2>/dev/null || true
ok "Tailscale opened — sign in with the same account as Plint if it asks (menu bar icon)"

# ── 7. Launch agents: remote · Claude · screen-off · metrics ─
install_agents
[[ -n "$BESZEL_KEY" ]] || warn "no Beszel key given — metrics agent not started (re-run with the key later)"

# ── 8. Claude on the box ─────────────────────────────────────
if (( ! SKIP_CLAUDE )); then
  step "Claude on the box"
  if ! claude auth status >/dev/null 2>&1; then
    printf '%s\n' "${C_DIM}Sign in with the claude.ai account you use on your phone (Pro or Max).${C_RESET}"
    claude auth login
  fi
  ok "claude $(claude --version 2>/dev/null | head -n1)"
  open -a Claude 2>/dev/null || true
  ok "Claude desktop opened — sign in, then Settings → This computer → turn on Computer use"
fi

# ── Done ─────────────────────────────────────────────────────
ip="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo '<this Mac IP>')"
cat <<EOF

${C_BOLD}Plint TV is set up.${C_RESET}

${C_BOLD}1. Register it with Plint${C_RESET} (on the server, or ask Plint in the app):
   plint tv register ${ip} ${KEY}
   Then https://tv.${DOMAIN} is the remote, and TV gets a card on the home screen.

${C_BOLD}2. Two one-time prompts for Claude:${C_RESET}   tv claude   → answer y, y → Ctrl+A then D

${C_BOLD}3. A few clicks only you can do:${C_RESET}
   · System Settings → Users & Groups → Automatic login → ${ME}   (needs FileVault off)
   · System Settings → Privacy & Security → Accessibility → allow osascript (for remote keys)
   · Stremio → Settings → Streaming server URL → https://stremio.${DOMAIN}/
   · App Store → Steam Link → Get (if it isn't installed yet)
   · metrics.${DOMAIN} → Add System → name 'tv', host ${ip}, port 45876
   · Phone: Screens or any VNC app → tv (over Tailscale) for a full view when you need it

   Then: tv status · tv doctor · tv open stremio
EOF
