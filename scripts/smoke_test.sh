#!/usr/bin/env bash
# Checks a running server end to end with a real .webm upload.
#   ./scripts/smoke_test.sh [port]        (default 8000)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE="http://localhost:${1:-8000}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "--- health ---"
curl -sf "$BASE/health" || { echo "no server on $BASE - start it with ./run.sh"; exit 1; }
echo

SRC=""
for c in ./whisper.cpp/samples/jfk.wav ../whisper.cpp/samples/jfk.wav; do
  [ -f "$c" ] && SRC="$c" && break
done
[ -n "$SRC" ] || { echo "no sample audio found in whisper.cpp/samples"; exit 1; }
ffmpeg -nostdin -loglevel error -y -i "$SRC" -c:a libopus "$TMP/sample.webm"

echo "--- response_format=json ---"
curl -sf -X POST "$BASE/v1/audio/transcriptions" -H "Authorization: Bearer sk-not-needed" \
  -F file=@"$TMP/sample.webm" -F model=gpt-4o-mini-transcribe -F response_format=json
echo
echo "--- response_format=text ---"
curl -sf -X POST "$BASE/v1/audio/transcriptions" \
  -F file=@"$TMP/sample.webm" -F response_format=text
echo
echo "OK"
