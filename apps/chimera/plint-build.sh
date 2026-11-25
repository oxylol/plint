#!/usr/bin/env bash
# Plint · build Chimera on the host: the engine as wasm, the page (dist/), and the server the
# `chimera` stack runs. Needs rustup's Rust (~/.cargo, with the wasm32-unknown-unknown target) and
# Bun (~/.bun); Plint's own Node is too old for the page's bundler, so Bun runs it.
#   apps/chimera/plint-build.sh && plint restart chimera
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.bun/bin:$HOME/.cargo/bin:$PATH"
bun install --frozen-lockfile
cargo build -p chimera-wasm --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/chimera_wasm.wasm src/wasm/chimera.wasm
bunx --bun tsc --noEmit
bunx --bun vite build
cargo build --release -p chimera-serve -p chimera-cli
echo "built: dist/ and target/release/chimera-serve"
