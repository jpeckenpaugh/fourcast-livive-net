#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Building Rust WebAssembly Module ==="
RUSTFLAGS="" cargo build --target wasm32-unknown-unknown --release

echo "=== Running wasm-bindgen ==="
wasm-bindgen target/wasm32-unknown-unknown/release/fourcast.wasm \
  --out-dir pkg \
  --target web \
  --no-typescript

echo "=== Wasm build complete -> pkg/ ==="
ls -lh pkg/
