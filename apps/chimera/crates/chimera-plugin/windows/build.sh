#!/usr/bin/env bash
# Build CHIMERA.clap and CHIMERA.vst3 for Windows (x64), in Git Bash, into target/bundled. Both hold
# the same DLL: it exports clap_entry, and clap-wrapper adds the VST3 entry point. Unsigned.
#
#   crates/chimera-plugin/windows/build.sh [--install] [--package]
#   --install  copy into Common Files\CLAP and Common Files\VST3 (needs an administrator, as CI is)
#   --package  zip both, with the licences, into out/CHIMERA-windows.zip
set -euo pipefail
cd "$(dirname "$0")/../../.."

# The page the editor shows is embedded by build.rs, so it has to exist first.
bun install --frozen-lockfile
bun run build
cargo build -p chimera-plugin --profile plugin

out=target/bundled
rm -rf "$out"
mkdir -p "$out/CHIMERA.vst3/Contents/x86_64-win" "$out/CHIMERA.vst3/Contents/Resources"
bun tools/third-party-notices.ts > "$out/THIRD-PARTY-LICENSES.txt"
cp LICENSE "$out/"
cp target/plugin/chimera_plugin.dll "$out/CHIMERA.clap"
cp target/plugin/chimera_plugin.dll "$out/CHIMERA.vst3/Contents/x86_64-win/CHIMERA.vst3"
cp LICENSE "$out/THIRD-PARTY-LICENSES.txt" "$out/CHIMERA.vst3/Contents/Resources/"

for arg in "$@"; do
  case "$arg" in
    --install)
      common=$(cygpath -u "${COMMONPROGRAMFILES:-C:\\Program Files\\Common Files}")
      mkdir -p "$common/CLAP" "$common/VST3"
      rm -rf "$common/VST3/CHIMERA.vst3"
      cp "$out/CHIMERA.clap" "$common/CLAP/"
      cp -R "$out/CHIMERA.vst3" "$common/VST3/"
      echo "installed into $common"
      ;;
    --package)
      mkdir -p out
      rm -f out/CHIMERA-windows.zip
      (cd "$out" && 7z a -tzip -bso0 ../../out/CHIMERA-windows.zip CHIMERA.clap CHIMERA.vst3 LICENSE THIRD-PARTY-LICENSES.txt)
      ;;
  esac
done
