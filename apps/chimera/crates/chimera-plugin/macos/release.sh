#!/usr/bin/env bash
# The macOS part of a release, from build.sh's bundles (signed and notarised by then, on CI), in
# out/release: CHIMERA-macos.zip, a CHIMERA folder with the three bundles and a how-to, then
# ../release.ts's SHA256SUMS and manifest.json for it (CI's publish job makes the release's own, with
# Linux's archive too).
#
#   release.sh              the manifest points at the GitHub release plugin-v<version>
#   release.sh --base URL   ... at URL instead (install-test.sh uses a file:// one)
#   release.sh --tag TAG    and fail unless TAG is plugin-v<version> and the bundles carry a Developer
#                           ID signature and Apple's ticket (CI, on a tag, before the release is made)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

version=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' Cargo.toml)
base=""
tag=""
while [ $# -gt 0 ]; do
  case $1 in
    --base) base=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    *) echo "usage: release.sh [--base URL] [--tag TAG]" >&2; exit 2 ;;
  esac
done

out=out/release
rm -rf "$out" && mkdir -p "$out/CHIMERA"
for b in clap vst3 component; do
  ditto "target/bundled/CHIMERA.$b" "$out/CHIMERA/CHIMERA.$b"
done
cat > "$out/CHIMERA/How to install.txt" <<EOF
CHIMERA $version, a voice instrument: CLAP, VST3 and AU for macOS 11 or later.

Copy each plugin into its folder (in Finder: Go > Go to Folder, and paste the path):

  CHIMERA.clap       ~/Library/Audio/Plug-Ins/CLAP
  CHIMERA.vst3       ~/Library/Audio/Plug-Ins/VST3
  CHIMERA.component  ~/Library/Audio/Plug-Ins/Components

Then rescan plugins in your DAW and put CHIMERA on an instrument track.

Or in Terminal, without this download:  curl -fsSL https://chimera.audio/install.sh | sh

More: https://chimera.audio/plugin/
EOF
ditto -c -k --keepParent "$out/CHIMERA" "$out/CHIMERA-macos.zip"
rm -rf "$out/CHIMERA"

bun crates/chimera-plugin/release.ts "$out" ${base:+--base "$base"} ${tag:+--tag "$tag"}
ls -l "$out"

if [ -n "$tag" ]; then
  # sign.sh does nothing without its secrets (a fork's pull request), and an ad hoc signature passes
  # codesign --verify: a release has to show the Developer ID and Apple's stapled ticket itself
  for b in clap vst3 component; do
    bundle="target/bundled/CHIMERA.$b"
    signature=$(codesign -dvv "$bundle" 2>&1) || true
    grep -q '^Authority=Developer ID Application:' <<< "$signature" ||
      { echo "$bundle isn't signed with a Developer ID: not releasing" >&2; exit 1; }
    xcrun stapler validate "$bundle" > /dev/null || { echo "$bundle has no notarisation ticket: not releasing" >&2; exit 1; }
  done
fi
