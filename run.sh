#!/usr/bin/env bash
# One command to a running server: venv, dependencies, whisper.cpp discovery, launch.
#   ./run.sh [--server] [port]        (default port 8000)
#   --server  use the whisper-server backend, which adds verbose_json (see README "Backends")
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BACKEND="${WHISPER_BACKEND:-cli}"
PORT=8000
for arg in "$@"; do
  case "$arg" in
    --server) BACKEND=server ;;
    -*) echo "unknown option: $arg (usage: ./run.sh [--server] [port])"; exit 1 ;;
    *) PORT="$arg" ;;
  esac
done

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

# whisper-server: an existing WHISPER_SERVER wins, else the one beside whisper-cli, built if missing.
if [ "$BACKEND" = server ]; then
  WHISPER_SERVER="${WHISPER_SERVER:-$(dirname "$WHISPER_CLI")/whisper-server}"
  if [ ! -x "$WHISPER_SERVER" ]; then
    build="$(cd "$(dirname "$WHISPER_CLI")" && pwd)"
    while [ "$build" != / ] && [ ! -f "$build/CMakeCache.txt" ]; do build="$(dirname "$build")"; done
    [ -f "$build/CMakeCache.txt" ] || { echo "whisper-server not found at $WHISPER_SERVER, and no CMake build to make it in"; exit 1; }
    echo "Building whisper-server in $build"
    cmake --build "$build" --config Release -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)" \
      --target whisper-server
  fi
  export WHISPER_SERVER
fi

export WHISPER_CLI WHISPER_MODEL WHISPER_BACKEND="$BACKEND"
echo "backend: $BACKEND"
echo "cli:     $WHISPER_CLI"
if [ "$BACKEND" = server ]; then echo "server:  $WHISPER_SERVER"; fi
echo "model:   $WHISPER_MODEL"
echo "docs:    http://localhost:$PORT/docs"
exec .venv/bin/hypercorn app:app --bind "127.0.0.1:$PORT"
