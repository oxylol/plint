#!/usr/bin/env bash
# What CI runs on Linux: the core's tests, a build and its release archive, the installer on it, the
# validators, then the plugin in REAPER under a virtual display. A failure anywhere fails the run.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here/../.."
tools=$(mktemp -d)

echo "::group::packages"
# a virtual display, and the GTK and ALSA that REAPER for Linux needs (renamed with t64 in Ubuntu 24.04)
sudo apt-get update -qq
if apt-cache show libgtk-3-0t64 > /dev/null 2>&1; then libs="libgtk-3-0t64 libasound2t64"; else libs="libgtk-3-0 libasound2"; fi
# shellcheck disable=SC2086
sudo apt-get install -y -qq xvfb $libs >/dev/null
echo "::endgroup::"

echo "::group::tests"
cargo test -p chimera-plugin --release
xvfb-run -a cargo test -p chimera-plugin --release -- --ignored the_x11_editor_maps_and_removes_its_child
echo "::endgroup::"

echo "::group::build"
"$here/linux/build.sh" --install --package
echo "::endgroup::"

echo "::group::glibc"
# The release says which glibc it needs (release.ts's minimum_glibc): built on an older system, the
# build can't need a newer one by accident. (C++ built against glibc 2.38's headers calls its
# __isoc23_ functions, which Ubuntu 22.04 and Debian 12 don't have.) Weak symbols are optional.
glibc=2.35
newest=$(objdump -T target/plugin/libchimera_plugin.so | grep -v '^[0-9a-f]* *w ' | grep -o 'GLIBC_[0-9.]*' | cut -d_ -f2 | sort -uV | tail -1)
echo "needs glibc $newest"
[ "$(printf '%s\n' "$newest" "$glibc" | sort -V | tail -1)" = "$glibc" ] ||
  { echo "the build needs glibc $newest, newer than the $glibc it promises" >&2; exit 1; }
echo "::endgroup::"

echo "::group::the release archive and the installer"
"$here/install-test.sh"
echo "::endgroup::"

echo "::group::clap-validator"
gh release download -R free-audio/clap-validator -p '*ubuntu-22.04.zip' -D "$tools/clap-validator"
(cd "$tools/clap-validator" && unzip -q ./*.zip && for t in ./*.tar.gz; do [ -e "$t" ] && tar xzf "$t"; done; true)
validator=$(find "$tools/clap-validator" -type f -name clap-validator | head -1)
chmod +x "$validator"
# the same exclusions as macos/validate.sh, for the same reasons
"$validator" validate --only-failed \
  -x 'state-reproducibility-.*' -x state-invalid-random -x process-varying-sample-rates \
  "$PWD/target/bundled/CHIMERA.clap"
echo "::endgroup::"

echo "::group::pluginval (VST3)"
gh release download -R Tracktion/pluginval -p 'pluginval_Linux.zip' -D "$tools/pluginval"
(cd "$tools/pluginval" && unzip -q pluginval_Linux.zip)
pluginval=$(find "$tools/pluginval" -type f -name pluginval | head -1)
chmod +x "$pluginval"
xvfb-run -a "$pluginval" --strictness-level 5 --skip-gui-tests --validate "$PWD/target/bundled/CHIMERA.vst3"
echo "::endgroup::"

echo "::group::REAPER"
"$here/host-test/run.sh"
echo "::endgroup::"
