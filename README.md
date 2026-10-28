# whisper-cpp-adapter

An OpenAI-compatible transcription endpoint backed by a local [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
build. 

Point any OpenAI client at it and `gpt-4o-mini-transcribe` calls run on your machine with no API key.
Start it with `--server` and it also answers `whisper-1` style `verbose_json` requests with word and
segment timestamps (see [Backends](#backends)).

## Setup

### 1. Install the prerequisites

Python 3.10+ and `ffmpeg` are needed to run the server. `git` and `cmake` are needed only to build
whisper.cpp, which the setup script does for you if you don't already have a build.

| | |
| --- | --- |
| Linux | `sudo apt install python3 python3-venv ffmpeg git cmake build-essential` |
| macOS | `brew install python ffmpeg git cmake` |
| Windows | `winget install Python.Python.3.12 Gyan.FFmpeg Git.Git Kitware.CMake` |

On Windows, building whisper.cpp also needs [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
with the "Desktop development with C++" workload. Confirm the basics are on your PATH:

```bash
python3 --version      # 3.10 or newer
ffmpeg -version        # any recent build
```

### 2. Get the code

```bash
git clone https://github.com/scottmac-dev/whisper-cpp-adapter.git
cd whisper-cpp-adapter
```

### 3. Run it

```bash
./run.sh                    # Linux/macOS — venv + deps + whisper.cpp + server on :8000
./run.sh --server           # the same, with the whisper-server backend for verbose_json
```

```powershell
.\run.ps1                   # Windows
.\run.ps1 -Server           # Windows, whisper-server backend
```

That one command does the whole setup: creates `.venv`, installs the Python dependencies, finds an
existing whisper.cpp (`./whisper.cpp` or `../whisper.cpp`) or builds one via `scripts/setup_whisper.*`,
picks a model, and starts the server.

**The first run takes several minutes** — it compiles whisper.cpp and downloads the `base.en` model
(~150 MB). Later runs start immediately. Pass a port to use something other than 8000
(`./run.sh 9000`).

### 4. Check that it works

```bash
curl http://localhost:8000/health      # reports the backend and whether the binaries, model and ffmpeg were found
./scripts/smoke_test.sh                # end-to-end: uploads a real sample and prints the transcript,
                                       # plus the verbose_json checks when running with --server
```

Interactive API docs are at <http://localhost:8000/docs>.

## Manual setup

`run.sh` is the short path; these are the same steps if you'd rather run them yourself, already have
a whisper.cpp build, or are wiring the server into another tool.

```bash
# 1. Virtual environment and Python dependencies (fastapi, python-multipart, hypercorn, httpx).
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Build whisper.cpp and download a model. Skip if you already have one.
./scripts/setup_whisper.sh          # Windows: .\scripts\setup_whisper.ps1
                                    # optional arg picks the model: ./scripts/setup_whisper.sh tiny.en

# 3. Point the server at the binary and model (paths the setup script prints at the end).
export WHISPER_CLI=./whisper.cpp/build/bin/whisper-cli
export WHISPER_MODEL=./whisper.cpp/models/ggml-base.en.bin

# 4. Start the server. Add WHISPER_BACKEND=server in front for the whisper-server backend.
hypercorn app:app --bind 127.0.0.1:8000
```

On Windows, set the environment variables with `$env:WHISPER_CLI = "..."` instead of `export`.

### Environment variables

All optional — the server and `run` discover these on their own. Set them to use a whisper.cpp
build that lives somewhere else, or a different model or language.

| Variable | Default |
| --- | --- |
| `WHISPER_CLI` | first `whisper-cli` found under `./whisper.cpp` or `../whisper.cpp` |
| `WHISPER_MODEL` | first `ggml-*.bin` found beside that binary |
| `WHISPER_LANGUAGE` | `en` (`auto` to detect). With `--server`, a request's `language` field overrides it. |
| `WHISPER_BACKEND` | `cli`; `server` for verbose_json. `run.sh --server` sets it for you. |

Only read by the server backend:

| Variable | Default |
| --- | --- |
| `WHISPER_SERVER` | `whisper-server` beside `WHISPER_CLI` (`run.sh --server` builds it if it is missing) |
| `WHISPER_SERVER_PORT` | `8178`, on 127.0.0.1. Internal: clients still talk to the adapter's port. |
| `WHISPER_THREADS` | whisper.cpp's default (4). Lower it to leave cores free for other work on the machine. |
| `WHISPER_DTW` | `auto`: the DTW preset matching the model file (`ggml-base.en.bin` → `base.en`). `off`, or a preset name. |
| `WHISPER_SUPPRESS_NST` | `true`: suppress non-speech tokens, so no `[BLANK_AUDIO]` or `[MUSIC PLAYING]` text |

The model matters more than anything else here: `base.en` or `small.en` give far better transcripts
than `tiny`, which is all some whisper.cpp checkouts ship with.

## Use it

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@recording.webm -F model=gpt-4o-mini-transcribe -F response_format=json
```

Or just substitute `https://api.openai.com/v1/audio/transcriptions` with `http://localhost:8000/v1/audio/transcriptions` in your API backend

## The endpoint

`POST /v1/audio/transcriptions`, `multipart/form-data`. 

Also `GET /health`, which reports the backend and whether the binaries, model and ffmpeg were found.

| Field | |
| --- | --- |
| `file` | **Required.** Anything ffmpeg reads (`.webm`, `.mp3`, `.wav`, ...). Keep the extension on the upload. |
| `model` | Accepted and ignored — the model is whatever `WHISPER_MODEL` points at. |
| `response_format` | `json` (default) or `text`. `verbose_json` with the server backend. Anything else is a 400. |
| `timestamp_granularities[]` | Server backend: `word` and/or `segment`, repeated, with `verbose_json` only (as OpenAI). The cli backend ignores it. |
| `language`, `prompt`, `temperature` | Server backend: passed to whisper. The cli backend ignores them. |
| `Authorization` | Accepted and ignored. |

`json` returns the OpenAI shape; `text` returns the bare transcript as `text/plain`.

```json
{
  "text": "Example of your transcript text....",
  "usage": {
    "type": "tokens",
    "input_tokens": 110,
    "input_token_details": { "text_tokens": 0, "audio_tokens": 110 },
    "output_tokens": 27,
    "total_tokens": 137
  }
}
```

## Backends

Picked once at startup. The default is exactly the original adapter, `--server` adds to it.

| | `cli` (default) | `server` (`./run.sh --server`) |
| --- | --- | --- |
| How | `whisper-cli` run per request | whisper.cpp's `whisper-server` started as a child process, model loaded once |
| `json` / `text` | Yes | Yes, same response shape |
| `verbose_json` | 400, pointing at `--server` | Yes, in OpenAI's `whisper-1` shape |
| Errors | `{"detail": ...}` | OpenAI's `{"error": {message, type, param, code}}` |

The server backend exists for clients that need what `whisper-1` returns with
`response_format=verbose_json` and `timestamp_granularities[]=word` and/or `segment`:

```bash
curl http://localhost:8000/v1/audio/transcriptions -F file=@chunk.wav -F model=whisper-1 \
  -F response_format=verbose_json -F "timestamp_granularities[]=word" -F "timestamp_granularities[]=segment"
```

```json
{
  "task": "transcribe", "language": "english", "duration": 16.0,
  "text": "And so my fellow Americans ask not what your country can do for you, ...",
  "words": [{ "word": "And", "start": 5.5, "end": 5.74 }, { "word": "so", "start": 5.74, "end": 6.1 }, "..."],
  "segments": [{
    "id": 0, "seek": 0, "start": 5.5, "end": 12.04,
    "text": " And so my fellow Americans ask not what your country can do",
    "tokens": [843, 523, "..."], "temperature": 0.0, "avg_logprob": -0.2,
    "compression_ratio": 0.97, "no_speech_prob": 0.055
  }],
  "usage": { "type": "duration", "seconds": 16 }
}
```

`whisper-server` already reports `avg_logprob` and `no_speech_prob` per segment. The adapter
reshapes the rest to match `whisper-1` (`verbose.py`):

- **Words.** whisper-server's per-segment "words" are BPE tokens (`" dishon"` + `"or"`, `"!"` on its
  own). They are merged into words, with punctuation stripped as in OpenAI's `words`. It stays in the
  segment text.
- **Word timing.** With DTW on, the default whenever the model has a preset, words are timed from
  DTW alignment. Without it, the first word of a clip that opens with music or noise is stamped at
  the start of the clip rather than when it is spoken. DTW only gives onsets, so a word runs until
  the next one starts. As in the reference implementation, an over-long word that ends a sentence is
  trimmed to twice the median word duration, so the pause after a sentence shows as a gap.
- **Segments** are bounded by their words, and given `compression_ratio` (computed as the reference
  implementation does) and an approximate `seek`.
- **Non-speech.** Non-speech tokens are suppressed, and any segment that is only an annotation like
  `[BLANK_AUDIO]` is dropped. Hallucinated speech on silence (a lone `you` is common) is passed
  through with its scores, as `whisper-1` would return it. Filtering on `no_speech_prob` and
  `avg_logprob` is the caller's job.
- **Usage** is `{"type": "duration", "seconds": N}`, the audio's length rounded up to a whole second.

`whisper-server`'s output goes to `whisper-server.log` next to `app.py`, and the last of it is
shown if it fails to start.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

Unit tests cover the `verbose_json` reshape (against whisper-server output captured in
`tests/fixtures`), the whisper-server process wrapper, and the endpoint in both backends with
whisper-server faked. They need `ffmpeg` but no whisper.cpp build. `scripts/smoke_test.*` exercises a
real running server.

## Limitations

- **Token counts are estimated.** whisper.cpp reports none, so `usage` is derived from audio
  duration (10 tokens/sec) and transcript length (4 chars/token) — plausibly shaped, not accurate.
  Constants are at the top of `app.py`. `verbose_json` reports seconds, as `whisper-1` does.
- One request at a time per worker, blocking until whisper.cpp finishes. The server backend queues
  concurrent requests in whisper-server instead. No auth, no streaming, no `srt`/`vtt`. Run it on
  localhost.
- **verbose_json is close to `whisper-1`, not identical.** whisper.cpp reports `no_speech_prob` once
  per 30-second decode window, not per segment. Local models are far smaller than `whisper-1`, so
  transcripts and hallucinations differ. DTW word timing is plausible but rougher than OpenAI's.
- Verified on Linux. The macOS and Windows scripts follow the same steps but were not run on those
  machines. The Windows scripts were syntax-checked only.
