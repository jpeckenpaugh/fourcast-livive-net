#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Serving fourcast.livive.net at http://localhost:${PORT}/"
python3 -m http.server "$PORT"
