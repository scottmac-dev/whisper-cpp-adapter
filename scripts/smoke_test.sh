#!/usr/bin/env bash
# Checks a running server end to end with a real .webm upload, plus the checks for its backend.
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

# Status of a request whose body is wanted only for display.
status() { curl -s -o "$TMP/body" -w '%{http_code}' -X POST "$BASE/v1/audio/transcriptions" "$@"; }

read -r BACKEND DTW < <(curl -sf "$BASE/health" | python3 -c \
  'import json, sys; h = json.load(sys.stdin); print(h.get("backend", "cli"), h.get("dtw") or "off")')

if [ "$BACKEND" = cli ]; then
  echo "--- response_format=verbose_json is refused by the cli backend ---"
  code="$(status -F file=@"$TMP/sample.webm" -F response_format=verbose_json)"
  cat "$TMP/body"; echo
  [ "$code" = 400 ] || { echo "expected 400, got $code"; exit 1; }
else
  # 5 s of noise before the sample: a word timed from the clip's start instead of when it is spoken
  # means DTW is not working.
  ffmpeg -nostdin -loglevel error -y -f lavfi -i "anoisesrc=d=5:c=pink:a=0.05:seed=1" -i "$SRC" \
    -filter_complex "[0:a]aresample=16000,aformat=channel_layouts=mono[n];[1:a]aresample=16000,aformat=channel_layouts=mono[s];[n][s]concat=n=2:v=0:a=1" \
    "$TMP/late.wav"

  echo "--- response_format=verbose_json with word and segment timestamps, as an OpenAI whisper-1 client sends ---"
  curl -sf -X POST "$BASE/v1/audio/transcriptions" -H "Authorization: Bearer sk-not-needed" \
    -F file=@"$TMP/late.wav" -F model=whisper-1 -F response_format=verbose_json \
    -F "timestamp_granularities[]=word" -F "timestamp_granularities[]=segment" > "$TMP/verbose.json"
  python3 - "$TMP/verbose.json" "$DTW" <<'EOF'
import json, math, sys
body, dtw = json.load(open(sys.argv[1])), sys.argv[2]
for key in ("task", "language", "duration", "text", "words", "segments", "usage"):
    assert key in body, f"missing {key}"
assert body["words"] and body["segments"], "no words or segments"
assert body["usage"] == {"type": "duration", "seconds": math.ceil(body["duration"])}, body["usage"]
first = body["words"][0]
print(f'{body["text"]!r}\n{len(body["words"])} words, {len(body["segments"])} segments, first word {first}')
if dtw == "off":
    print("DTW is off, skipping the word timing check")
else:
    assert first["start"] >= 4.5, f'first word at {first["start"]}s, but speech starts after 5 s of noise'
EOF

  echo "--- timestamp_granularities[] without verbose_json is refused ---"
  code="$(status -F file=@"$TMP/sample.webm" -F response_format=json -F "timestamp_granularities[]=word")"
  cat "$TMP/body"; echo
  [ "$code" = 400 ] || { echo "expected 400, got $code"; exit 1; }
fi
echo "OK ($BACKEND backend)"
