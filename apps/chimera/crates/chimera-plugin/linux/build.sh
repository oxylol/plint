#!/usr/bin/env bash
# Build CHIMERA.clap and CHIMERA.vst3 for Linux (x86_64) into target/bundled: the same .so in both.
# Its editor is the page in a web browser (docs/plugin.md, "Linux").
#
#   crates/chimera-plugin/linux/build.sh [--install] [--package]
#   --install  copy into ~/.clap and ~/.vst3
#   --package  both, with the licences and a how-to, as a CHIMERA folder in
#              out/release/CHIMERA-linux.tar.gz (../release.ts adds the manifest)
set -euo pipefail
cd "$(dirname "$0")/../../.."

# The page the editor serves is embedded by build.rs, so it has to exist first.
bun install --frozen-lockfile
bun run build
cargo build -p chimera-plugin --profile plugin

out=target/bundled
rm -rf "$out"
mkdir -p "$out/CHIMERA.vst3/Contents/x86_64-linux" "$out/CHIMERA.vst3/Contents/Resources"
bun tools/third-party-notices.ts > "$out/THIRD-PARTY-LICENSES.txt"
cp LICENSE "$out/"
cp target/plugin/libchimera_plugin.so "$out/CHIMERA.clap"
cp target/plugin/libchimera_plugin.so "$out/CHIMERA.vst3/Contents/x86_64-linux/CHIMERA.so"
cp LICENSE "$out/THIRD-PARTY-LICENSES.txt" "$out/CHIMERA.vst3/Contents/Resources/"

for arg in "$@"; do
  case "$arg" in
    --install)
      mkdir -p "$HOME/.clap" "$HOME/.vst3"
      rm -rf "$HOME/.vst3/CHIMERA.vst3"
      cp "$out/CHIMERA.clap" "$HOME/.clap/"
      cp -R "$out/CHIMERA.vst3" "$HOME/.vst3/"
      echo "installed into ~/.clap and ~/.vst3"
      ;;
    --package)
      version=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' Cargo.toml)
      pkg=$(mktemp -d)
      mkdir "$pkg/CHIMERA"
      cp -R "$out/CHIMERA.clap" "$out/CHIMERA.vst3" "$out/LICENSE" "$out/THIRD-PARTY-LICENSES.txt" "$pkg/CHIMERA/"
      cat > "$pkg/CHIMERA/How to install.txt" <<EOF
CHIMERA $version, a voice instrument: CLAP and VST3 for Linux (x86_64, glibc 2.35 or later).

Copy each plugin into its folder:

  CHIMERA.clap  ~/.clap
  CHIMERA.vst3  ~/.vst3

Then rescan plugins in your DAW and put CHIMERA on an instrument track. Its editor opens in your web
browser.

Or in a terminal, without this download:  curl -fsSL https://chimera.audio/install.sh | sh

More: https://chimera.audio/plugin/
EOF
      mkdir -p out/release
      tar -czf out/release/CHIMERA-linux.tar.gz -C "$pkg" CHIMERA
      rm -rf "$pkg"
      ;;
  esac
done
