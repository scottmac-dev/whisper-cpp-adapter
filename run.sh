#!/usr/bin/env bash
# One command to a running server: venv, dependencies, whisper.cpp discovery, launch.
#   ./run.sh [port]        (default 8000)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PORT="${1:-8000}"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required - see README"; exit 1; }

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

# whisper-cli: an existing WHISPER_CLI wins, then a local or sibling checkout, else build one.
if [ -z "${WHISPER_CLI:-}" ]; then
  for c in ./whisper.cpp ../whisper.cpp; do
    if [ -x "$c/build/bin/whisper-cli" ]; then WHISPER_CLI="$c/build/bin/whisper-cli"; break; fi
  done
fi
if [ -z "${WHISPER_CLI:-}" ]; then
  echo "No whisper.cpp build found - running scripts/setup_whisper.sh"
  ./scripts/setup_whisper.sh
  WHISPER_CLI="./whisper.cpp/build/bin/whisper-cli"
fi

# Model: an existing WHISPER_MODEL wins, else the first real model beside the binary.
if [ -z "${WHISPER_MODEL:-}" ]; then
  WHISPER_MODEL="$(ls "$(dirname "$WHISPER_CLI")"/../../models/ggml-*.bin 2>/dev/null \
                   | grep -v for-tests | head -1 || true)"
fi
[ -n "${WHISPER_MODEL:-}" ] || { echo "No model found - run ./scripts/setup_whisper.sh"; exit 1; }

export WHISPER_CLI WHISPER_MODEL
echo "cli:   $WHISPER_CLI"
echo "model: $WHISPER_MODEL"
echo "docs:  http://localhost:$PORT/docs"
exec .venv/bin/hypercorn app:app --bind "127.0.0.1:$PORT"
