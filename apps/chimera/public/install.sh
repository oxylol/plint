#!/bin/sh
# Install CHIMERA, a voice instrument, as plugins for this user: CLAP, VST3 and AU on macOS, CLAP and
# VST3 on Linux (x86_64).
#
#   curl -fsSL https://chimera.audio/install.sh | sh
#   curl -fsSL https://chimera.audio/install.sh | sh -s -- --formats clap,vst3
#   curl -fsSL https://chimera.audio/install.sh | sh -s -- --version 0.1.0
#   curl -fsSL https://chimera.audio/install.sh | sh -s -- --uninstall
#
# It reads the release's manifest.json, downloads the archive it names for this system, checks its
# SHA-256, and copies the plugins into ~/Library/Audio/Plug-Ins on macOS, ~/.clap and ~/.vst3 on Linux
# (no sudo). Source:
# https://github.com/jamesondh/chimera-vocal-synth (public/install.sh)
set -eu

repo=jamesondh/chimera-vocal-synth
manifest=${CHIMERA_MANIFEST:-https://github.com/$repo/releases/latest/download/manifest.json}
formats=""
action=install

say() { printf '%s\n' "$*"; }
die() { printf 'chimera: %s\n' "$*" >&2; exit 1; }
usage() {
  cat << 'EOF'
Install CHIMERA's plugins: CLAP, VST3 and AU into ~/Library/Audio/Plug-Ins on macOS, CLAP and VST3
into ~/.clap and ~/.vst3 on Linux.

  --formats LIST   only these, of clap, vst3 and (macOS) au (default: all)
  --version X      that release instead of the latest
  --uninstall      remove them
EOF
}

while [ $# -gt 0 ]; do
  case $1 in
    --formats) [ $# -ge 2 ] || die "--formats needs a list, like clap,vst3,au"; formats=$2; shift 2 ;;
    --formats=*) formats=${1#*=}; shift ;;
    --version) [ $# -ge 2 ] || die "--version needs a version, like 0.1.0"; manifest="https://github.com/$repo/releases/download/plugin-v$2/manifest.json"; shift 2 ;;
    --version=*) manifest="https://github.com/$repo/releases/download/plugin-v${1#*=}/manifest.json"; shift ;;
    --uninstall) action=uninstall; shift ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown option $1 (--formats LIST, --version X, --uninstall)" ;;
  esac
done

case $(uname -s) in
  Darwin) system=macos ;;
  Linux) system=linux ;;
  *) die "this installer is for macOS and Linux. A Windows build isn't released yet: https://chimera.audio/plugin/" ;;
esac
if [ $system = linux ]; then
  [ "$(uname -m)" = x86_64 ] || die "the Linux build is x86_64 only, and this is $(uname -m)"
fi

# where each format goes, and its bundle in the archive
plugins="$HOME/Library/Audio/Plug-Ins"
dest() {
  case $system:$1 in
    macos:clap) say "$plugins/CLAP/CHIMERA.clap" ;;
    macos:vst3) say "$plugins/VST3/CHIMERA.vst3" ;;
    macos:au) say "$plugins/Components/CHIMERA.component" ;;
    linux:clap) say "$HOME/.clap/CHIMERA.clap" ;;
    linux:vst3) say "$HOME/.vst3/CHIMERA.vst3" ;;
    linux:au) die "AU is a macOS format: on Linux, clap or vst3" ;;
    *) die "unknown format $1 (clap, vst3, au)" ;;
  esac
}
[ -n "$formats" ] || { [ $system = macos ] && formats=clap,vst3,au || formats=clap,vst3; }
chosen=$(say "$formats" | tr ',' ' ')
for f in $chosen; do dest "$f" > /dev/null; done
[ -n "$chosen" ] || die "no formats chosen"

# the AU registry keeps what it has seen; restarting it makes hosts notice the change
refresh_au() {
  [ $system = macos ] || return 0
  case " $chosen " in *" au "*) killall -9 AudioComponentRegistrar 2> /dev/null || true ;; esac
}

if [ "$action" = uninstall ]; then
  for f in $chosen; do
    d=$(dest "$f")
    if [ -e "$d" ]; then rm -rf "$d" && say "removed $d"; fi
  done
  refresh_au
  say "CHIMERA is uninstalled. Rescan plugins in your DAW."
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM
curl -fsSL "$manifest" -o "$tmp/manifest.json" || die "couldn't download $manifest"
# a string from the manifest, by its path ("platforms.macos.url"), with JSON.parse: on macOS in
# JavaScript for Automation (on every macOS since 10.10; plutil reads JSON only on recent ones), on
# Linux with Python or jq
if [ $system = macos ]; then
  cat > "$tmp/field.js" << 'JS'
function run(argv) {
  ObjC.import("Foundation");
  const text = ObjC.unwrap($.NSString.stringWithContentsOfFileEncodingError(argv[0], $.NSUTF8StringEncoding, null));
  let v = JSON.parse(text);
  for (const k of argv[1].split(".")) v = v == null ? undefined : v[k];
  return v == null ? "" : String(v);
}
JS
  read_field() { osascript -l JavaScript "$tmp/field.js" "$tmp/manifest.json" "$1"; }
elif command -v python3 > /dev/null; then
  read_field() {
    python3 -c 'import json, sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): v = v.get(k) if isinstance(v, dict) else None
print("" if v is None else v)' "$tmp/manifest.json" "$1"
  }
elif command -v jq > /dev/null; then
  read_field() { jq -r --arg p "$1" 'getpath($p / ".") // ""' "$tmp/manifest.json"; }
else
  die "reading the release's manifest needs python3 or jq. Or download CHIMERA-linux.tar.gz from https://chimera.audio/plugin/"
fi
field() {
  v=$(read_field "$1" 2> /dev/null) || true
  [ -n "$v" ] || die "the manifest has no $1"
  say "$v"
}
version=$(field version)
[ -n "$(read_field "platforms.$system.url" 2> /dev/null)" ] || die "release $version has no $system build: https://chimera.audio/plugin/"
url=$(field "platforms.$system.url")
sha=$(field "platforms.$system.sha256")

say "Downloading CHIMERA $version..."
curl -fL --progress-bar "$url" -o "$tmp/download" || die "couldn't download $url"
if [ $system = macos ]; then
  got=$(shasum -a 256 "$tmp/download" | cut -d' ' -f1)
else
  got=$(sha256sum "$tmp/download" | cut -d' ' -f1)
fi
[ "$got" = "$sha" ] || die "the download's checksum is $got, not the manifest's $sha: nothing was installed"
mkdir "$tmp/unpacked"
if [ $system = macos ]; then
  ditto -x -k "$tmp/download" "$tmp/unpacked"
else
  tar -xzf "$tmp/download" -C "$tmp/unpacked"
fi

bundle() { say "$tmp/unpacked/CHIMERA/$(basename "$(dest "$1")")"; }
# every bundle checked before any is copied, so a bad download installs nothing (on Linux a CLAP is
# one file, not a folder)
for f in $chosen; do
  src=$(bundle "$f")
  [ -e "$src" ] || die "the download has no $(basename "$src"): nothing was installed"
  if [ $system = macos ]; then
    codesign --verify --strict "$src" 2> /dev/null || die "$(basename "$src")'s signature doesn't verify: nothing was installed"
  fi
done

# Each new bundle is copied in beside the old one first (hidden, and not named like a plugin, so a
# host scanning meanwhile ignores it); only when every copy has worked are the old ones swapped out.
# A failed copy leaves the installed plugins as they were. A host with the old one loaded keeps it
# until it restarts: removing a bundle doesn't disturb a binary in use, overwriting one would.
# ditto keeps everything a signed bundle carries
copy() { if [ $system = macos ]; then ditto "$1" "$2"; else cp -R "$1" "$2"; fi; }
staged() { say "$(dirname "$(dest "$1")")/.$(basename "$(dest "$1")").installing"; }
unstage() { for f in $chosen; do rm -rf "$(staged "$f")"; done; }
trap 'unstage; rm -rf "$tmp"' EXIT INT TERM
for f in $chosen; do
  mkdir -p "$(dirname "$(dest "$f")")"
  rm -rf "$(staged "$f")"
  copy "$(bundle "$f")" "$(staged "$f")" || die "couldn't copy $(basename "$(dest "$f")") into $(dirname "$(dest "$f")"): nothing was changed"
done
say "CHIMERA $version is installed:"
for f in $chosen; do
  d=$(dest "$f")
  rm -rf "$d"
  mv "$(staged "$f")" "$d"
  say "  $d"
done
refresh_au
say "Rescan plugins in your DAW, and put CHIMERA on an instrument track."
