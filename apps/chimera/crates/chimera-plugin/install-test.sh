#!/usr/bin/env bash
# public/install.sh as a user runs it, on macOS or Linux, against this build's archive in out/release
# (from macos/release.sh or linux/build.sh --package) with a manifest that serves it as a file:// URL,
# and into a HOME of its own: all the formats, one, an update that can't finish, a checksum that
# doesn't match, and uninstalling. CI runs it on every push, on both.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bun crates/chimera-plugin/release.ts out/release --base "file://$PWD/out/release"
manifest="file://$PWD/out/release/manifest.json"
home=$(mktemp -d)
if [ "$(uname -s)" = Darwin ]; then
  plugins="$home/Library/Audio/Plug-Ins"
  all=("$plugins/CLAP/CHIMERA.clap" "$plugins/VST3/CHIMERA.vst3" "$plugins/Components/CHIMERA.component")
else
  all=("$home/.clap/CHIMERA.clap" "$home/.vst3/CHIMERA.vst3")
fi
failures=0
check() { # name, then a command that should succeed
  local name=$1
  shift
  if "$@"; then echo "ok   $name"; else echo "FAIL $name"; failures=$((failures + 1)); fi
}
installer() { HOME="$home" CHIMERA_MANIFEST="$manifest" sh public/install.sh "$@"; }
none() { for p in "${all[@]}"; do [ ! -e "$p" ] || return 1; done; }

installer
for p in "${all[@]}"; do
  check "installs $(basename "$p")" test -e "$p"
  if [ "$(uname -s)" = Darwin ]; then check "$(basename "$p") verifies" codesign --verify --strict "$p"; fi
done
check "the VST3's binary is there" bash -c 'ls "$0"/Contents/*/* > /dev/null' "${all[1]}"

installer --uninstall
check "uninstall removes them all" none

installer --formats clap
check "--formats clap installs the CLAP" test -e "${all[0]}"
check "and only the CLAP" bash -c 'for p in "${@:1}"; do [ ! -e "$p" ] || exit 1; done' _ "${all[@]:1}"
installer --uninstall

# an update that can't finish leaves what was installed as it was: the VST3 folder read-only, so its
# copy fails after the CLAP's has been made. The old CLAP is marked: a file in its bundle on macOS, its
# whole content on Linux, where it's one file.
installer
if [ -d "${all[0]}" ]; then marker="${all[0]}/old"; else marker="${all[0]}"; fi
echo old > "$marker"
vst3_dir=$(dirname "${all[1]}")
chmod a-w "$vst3_dir"
check "an update that can't copy fails" bash -c '! HOME="$0" CHIMERA_MANIFEST="$1" sh public/install.sh 2> /dev/null' "$home" "$manifest"
chmod u+w "$vst3_dir"
check "and leaves the CLAP it had" grep -qx old "$marker"
check "and the VST3" test -d "${all[1]}"
check "and nothing half-copied" bash -c 'for p in "$@"; do ! ls -a "$(dirname "$p")" | grep -q installing || exit 1; done' _ "${all[@]}"
installer --uninstall

# a download that isn't what the manifest says installs nothing
sed 's/"sha256": "[0-9a-f]*"/"sha256": "0000"/' out/release/manifest.json > "$home/bad.json"
if HOME="$home" CHIMERA_MANIFEST="file://$home/bad.json" sh public/install.sh 2> "$home/err"; then
  check "a wrong checksum fails" false
else
  check "a wrong checksum fails, and says so" grep -q checksum "$home/err"
fi
check "and installs nothing" none

check "the manifest's version is Cargo.toml's" grep -q "\"version\": \"$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' Cargo.toml)\"" out/release/manifest.json

rm -rf "$home"
[ "$failures" = 0 ] || { echo "$failures failed"; exit 1; }
