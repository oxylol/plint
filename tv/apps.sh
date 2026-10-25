#!/usr/bin/env bash
# Plint TV · apps, straight from their publishers. No Homebrew.
# Sourced by install.sh; each installer is idempotent (skips what's there).
#
#   install_all            everything below
#   app_installed "Name"   true if /Applications/Name.app exists

DL="$HOME/Library/Caches/plint-tv"
BIN="$HOME/.local/bin"
mkdir -p "$DL" "$BIN"

app_installed() { [[ -d "/Applications/$1.app" || -d "$HOME/Applications/$1.app" ]]; }

dl() {  # dl URL FILE — resumable, follows redirects, quiet
  curl -fL --retry 3 --retry-delay 2 -C - -# -o "$DL/$2" "$1" 2>&1 | tr '\r' '\n' | tail -n1 >/dev/null
  [[ -s "$DL/$2" ]]
}

github_asset() {  # github_asset owner/repo <regex> → download URL of the newest matching asset
  curl -fsSL "https://api.github.com/repos/$1/releases/latest" \
    | python3 -c 'import json,re,sys; r=re.compile(sys.argv[1]); print(next(a["browser_download_url"] for a in json.load(sys.stdin)["assets"] if r.search(a["name"])))' "$2"
}

install_dmg() {  # install_dmg "App Name" URL [file]
  local name="$1" url="$2" file="${3:-$1.dmg}" mnt app
  app_installed "$name" && { ok "$name (already installed)"; return 0; }
  say "$name"
  dl "$url" "$file" || { warn "could not download $name"; return 0; }
  mnt="$(mktemp -d /tmp/plint-tv.XXXXXX)"
  if ! hdiutil attach -nobrowse -quiet -mountpoint "$mnt" "$DL/$file"; then warn "could not open $file"; return 0; fi
  app="$(find "$mnt" -maxdepth 2 -name '*.app' -print -quit)"
  if [[ -n "$app" ]]; then
    ditto "$app" "/Applications/$(basename "$app")" && ok "$name → /Applications"
  else
    local pkg; pkg="$(find "$mnt" -maxdepth 2 -name '*.pkg' -print -quit)"
    [[ -n "$pkg" ]] && sudo installer -pkg "$pkg" -target / >/dev/null && ok "$name (pkg)"
  fi
  hdiutil detach -quiet "$mnt" || true
}

install_pkg() {  # install_pkg "Label" URL file
  say "$1"
  dl "$2" "$3" || { warn "could not download $1"; return 0; }
  sudo installer -pkg "$DL/$3" -target / >/dev/null && ok "$1"
}

install_zip_app() {  # install_zip_app "App Name" URL file [dest dir]
  local name="$1" url="$2" file="$3" dest="${4:-/Applications}" tmp
  app_installed "$name" && { ok "$name (already installed)"; return 0; }
  say "$name"
  dl "$url" "$file" || { warn "could not download $name"; return 0; }
  tmp="$(mktemp -d /tmp/plint-tv.XXXXXX)"
  ditto -xk "$DL/$file" "$tmp"
  local app; app="$(find "$tmp" -maxdepth 3 -name '*.app' -print -quit)"
  [[ -n "$app" ]] && ditto "$app" "$dest/$(basename "$app")" && ok "$name → $dest"
  rm -rf "$tmp"
}

# ── the apps ─────────────────────────────────────────────────
install_tailscale() {
  app_installed Tailscale && { ok "Tailscale (already installed)"; return 0; }
  local pkg; pkg="$(curl -fsSL https://pkgs.tailscale.com/stable/ | grep -oE 'Tailscale-[0-9.]+-macos\.pkg' | sort -V | tail -n1)"
  [[ -n "$pkg" ]] || { warn "could not find the Tailscale package"; return 0; }
  install_pkg "Tailscale" "https://pkgs.tailscale.com/stable/$pkg" "$pkg"
}

install_stremio() {
  app_installed Stremio && { ok "Stremio (already installed)"; return 0; }
  local url; url="$(curl -fsSL https://www.stremio.com/downloads | grep -oE 'https://dl\.strem\.io/[^"]+\.dmg' | head -n1)"
  [[ -n "$url" ]] || { warn "could not find the Stremio download"; return 0; }
  install_dmg "Stremio" "$url" "Stremio.dmg"
}

install_chrome()    { install_dmg "Google Chrome" "https://dl.google.com/chrome/mac/stable/GGRO/googlechrome.dmg" "googlechrome.dmg"; }
install_moonlight() { local u; u="$(github_asset moonlight-stream/moonlight-qt '\.dmg$')" && install_dmg "Moonlight" "$u" "Moonlight.dmg"; }
install_claude_app(){ install_dmg "Claude" "https://claude.ai/api/desktop/darwin/universal/dmg/latest/redirect" "Claude.dmg"; }

install_spotify() {
  app_installed Spotify && { ok "Spotify (already installed)"; return 0; }
  say "Spotify"
  dl "https://download.scdn.co/SpotifyInstaller.zip" "SpotifyInstaller.zip" || { warn "could not download Spotify"; return 0; }
  local tmp; tmp="$(mktemp -d /tmp/plint-tv.XXXXXX)"; ditto -xk "$DL/SpotifyInstaller.zip" "$tmp"
  local inst; inst="$(find "$tmp" -maxdepth 2 -name 'Install Spotify.app' -print -quit)"
  if [[ -n "$inst" ]] && open -W "$inst"; then ok "Spotify"; else warn "run the Spotify installer by hand later"; fi
  rm -rf "$tmp"
}

install_aerial() {
  [[ -d "$HOME/Library/Screen Savers/Aerial.saver" ]] && { ok "Aerial (already installed)"; return 0; }
  say "Aerial"
  local u; u="$(github_asset JohnCoates/Aerial 'Aerial\.saver\.zip$')" || { warn "could not find Aerial"; return 0; }
  dl "$u" "Aerial.saver.zip" || return 0
  mkdir -p "$HOME/Library/Screen Savers"
  ditto -xk "$DL/Aerial.saver.zip" "$HOME/Library/Screen Savers/" && ok "Aerial → ~/Library/Screen Savers"
}

install_beszel_agent() {
  [[ -x "$BIN/beszel-agent" ]] && { ok "beszel-agent (already installed)"; return 0; }
  say "Beszel agent"
  local u; u="$(github_asset henrygd/beszel 'beszel-agent_darwin_amd64\.tar\.gz$')" || { warn "could not find the Beszel agent"; return 0; }
  dl "$u" "beszel-agent.tgz" || return 0
  tar -xzf "$DL/beszel-agent.tgz" -C "$BIN" beszel-agent && chmod +x "$BIN/beszel-agent" && ok "beszel-agent → ~/.local/bin"
}

install_claude_code() {
  if [[ -x "$HOME/.local/bin/claude" ]] || command -v claude >/dev/null 2>&1; then ok "claude $("$HOME/.local/bin/claude" --version 2>/dev/null || claude --version 2>/dev/null | head -n1)"; return 0; fi
  say "Claude Code"
  if curl -fsSL https://claude.ai/install.sh | bash -s stable >/dev/null 2>&1; then ok "claude → ~/.local/bin"; else warn "Claude Code install failed — run: curl -fsSL https://claude.ai/install.sh | bash"; fi
}


install_steam_link() {
  app_installed "Steam Link" && { ok "Steam Link (already installed)"; return 0; }
  warn "Steam Link comes from the App Store — opening it; click Get"
  open "macappstore://apps.apple.com/app/id1246969117" 2>/dev/null || true
}

install_all() {
  install_tailscale
  install_stremio
  install_chrome
  install_moonlight
  install_spotify
  install_aerial
  install_claude_app
  install_claude_code
  install_beszel_agent
  install_steam_link
}
