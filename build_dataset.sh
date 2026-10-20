#!/usr/bin/env bash
# One-command dataset build: sets up the venv (if needed) and builds the
# balanced cat/dog crop set from the small Zenodo CC BY sample.
#
#   ./build_dataset.sh                       # downloads 32.6MB sample, writes 128 crops + manifest
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found (required)" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "creating .venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet

exec python data/build_dataset.py "$@"
