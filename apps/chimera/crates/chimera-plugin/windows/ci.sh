#!/usr/bin/env bash
# What CI runs on Windows, in Git Bash: the core's tests, a build, the validators, then the plugin
# in REAPER. A failure anywhere fails the run.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here/../.."
tools=$(mktemp -d)

echo "::group::tests"
cargo test -p chimera-plugin --release
echo "::endgroup::"

echo "::group::build"
"$here/windows/build.sh" --install --package
echo "::endgroup::"

echo "::group::clap-validator"
gh release download -R free-audio/clap-validator -p '*windows.zip' -D "$tools/clap-validator"
(cd "$tools/clap-validator" && unzip -q ./*.zip && for t in ./*.tar.gz; do [ -e "$t" ] && tar xzf "$t"; done; true)
validator=$(find "$tools/clap-validator" -iname 'clap-validator.exe' | head -1)
# the same exclusions as macos/validate.sh, for the same reasons
"$validator" validate --only-failed \
  -x 'state-reproducibility-.*' -x state-invalid-random -x process-varying-sample-rates \
  "$(cygpath -w "$PWD/target/bundled/CHIMERA.clap")"
echo "::endgroup::"

echo "::group::pluginval (VST3)"
gh release download -R Tracktion/pluginval -p 'pluginval_Windows.zip' -D "$tools/pluginval"
(cd "$tools/pluginval" && unzip -q pluginval_Windows.zip)
"$(find "$tools/pluginval" -iname 'pluginval.exe' | head -1)" --strictness-level 5 --skip-gui-tests \
  --validate "$(cygpath -w "$PWD/target/bundled/CHIMERA.vst3")"
echo "::endgroup::"

echo "::group::REAPER"
"$here/host-test/run.sh"
echo "::endgroup::"
