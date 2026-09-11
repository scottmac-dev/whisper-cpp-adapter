#!/usr/bin/env bash
# Clones and builds whisper.cpp, then downloads a model. Works on Linux and macOS.
# On Windows use scripts/setup_whisper.ps1 instead.
# Skip this entirely if you already have whisper.cpp: just set WHISPER_CLI and WHISPER_MODEL.
#
#   ./scripts/setup_whisper.sh [model]     # model defaults to base.en (~150 MB)
set -euo pipefail

MODEL="${1:-base.en}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/whisper.cpp"

if [ "$(uname -s)" = "Darwin" ]; then INSTALL="brew install"; else INSTALL="sudo apt install"; fi
for tool in git cmake ffmpeg; do
  command -v "$tool" >/dev/null || { echo "$tool is required: $INSTALL $tool"; exit 1; }
done

# nproc on Linux, sysctl on macOS.
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

if [ ! -d "$DIR" ]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$DIR"
fi

cmake -B "$DIR/build" -S "$DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$DIR/build" --config Release -j "$JOBS" --target whisper-cli

"$DIR/models/download-ggml-model.sh" "$MODEL" "$DIR/models"

echo
echo "Done. Run the server with:"
echo "  export WHISPER_CLI=$DIR/build/bin/whisper-cli"
echo "  export WHISPER_MODEL=$DIR/models/ggml-$MODEL.bin"
echo "  hypercorn app:app --bind 127.0.0.1:8000"
