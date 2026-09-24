#!/usr/bin/env bash
# One-command launcher for macOS / Linux: creates a virtual environment on first run, then starts SignScribe.
# Extra arguments are passed through, e.g.  ./scripts/run.sh --demo
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
    echo "Setting up SignScribe for the first time (downloads a few packages)..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -e .
fi

exec .venv/bin/python -m signscribe "$@"
