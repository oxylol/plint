#!/usr/bin/env bash
# Make the GitHub release TAG from the systems' archives in out/release (CI's release job, on a tag,
# after the macOS and Linux jobs have built, signed and tested them): both archives, SHA256SUMS and
# manifest.json (release.ts, which refuses a tag that isn't plugin-v<the version in Cargo.toml>).
#
#   crates/chimera-plugin/publish.sh plugin-v0.2.0
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

repo=jamesondh/chimera-vocal-synth
tag=${1:?usage: publish.sh TAG}
out=out/release
for f in CHIMERA-macos.zip CHIMERA-linux.tar.gz; do
  [ -f "$out/$f" ] || { echo "no $out/$f: not releasing without every system's" >&2; exit 1; }
done
bun crates/chimera-plugin/release.ts "$out" --tag "$tag"
version=${tag#plugin-v}
ls -l "$out"

gh release create "$tag" "$out/CHIMERA-macos.zip" "$out/CHIMERA-linux.tar.gz" "$out/SHA256SUMS" "$out/manifest.json" \
  --repo "$repo" --title "CHIMERA plugin $version" --notes "CHIMERA $version as a voice instrument plugin:

- **macOS 11 or later** (Apple silicon and Intel): CLAP, VST3 and AU, signed and notarised.
- **Linux** (x86_64, glibc 2.35 or later): CLAP and VST3. The editor opens in your web browser.

Install in a terminal, on either:

\`\`\`sh
curl -fsSL https://chimera.audio/install.sh | sh
\`\`\`

Or download the archive for your system below and follow its How to install.txt. More at https://chimera.audio/plugin/."
