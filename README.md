# whisper-cpp-adapter

An OpenAI-compatible transcription endpoint backed by a local [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
build. 

Point any OpenAI client at it and `gpt-4o-mini-transcribe` calls run on your machine with no API key.

## Quick start

```bash
./run.sh                    # Linux/macOS: venv + deps + whisper.cpp + server on :8000
```

```powershell
.\run.ps1                   # Windows
```

`run` finds an existing whisper.cpp (`./whisper.cpp` or `../whisper.cpp`), picks a model, and
builds one via `scripts/setup_whisper.*` if there is none. 

Override either path with `WHISPER_CLI` and `WHISPER_MODEL`.

**Prerequisites:** Python 3.10+ and `ffmpeg`. Building whisper.cpp also needs `git` and `cmake`
(plus Visual Studio Build Tools with the C++ workload on Windows).

| | |
| --- | --- |
| Linux | `sudo apt install ffmpeg git cmake build-essential` |
| macOS | `brew install ffmpeg git cmake` |
| Windows | `winget install Gyan.FFmpeg Git.Git Kitware.CMake` |

## Use it

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@recording.webm -F model=gpt-4o-mini-transcribe -F response_format=json
```

Or just substitute `https://api.openai.com/v1/audio/transcriptions` with `http://localhost:8000/v1/audio/transcriptions` in your API backend

## The endpoint

`POST /v1/audio/transcriptions`, `multipart/form-data`. 

Also `GET /health`, which reports whether the binary, model and ffmpeg were found.

| Field | |
| --- | --- |
| `file` | **Required.** Anything ffmpeg reads (`.webm`, `.mp3`, `.wav`, ...). Keep the extension on the upload. |
| `model` | Accepted and ignored — the model is whatever `WHISPER_MODEL` points at. |
| `response_format` | `json` (default) or `text`. Anything else is a 400. |
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

| Variable | Default |
| --- | --- |
| `WHISPER_CLI` | first `whisper-cli` found under `./whisper.cpp` or `../whisper.cpp` |
| `WHISPER_MODEL` | first `ggml-*.bin` found beside that binary |
| `WHISPER_LANGUAGE` | `en` (`auto` to detect) |

## Limitations

- **Token counts are estimated.** whisper.cpp reports none, so `usage` is derived from audio
  duration (10 tokens/sec) and transcript length (4 chars/token) — plausibly shaped, not accurate.
  Constants are at the top of `app.py`.
- One request at a time per worker, blocking until whisper.cpp finishes. No auth, no streaming, no
  `verbose_json`/`srt`/`vtt`. Run it on localhost.
- Verified on Linux. The macOS and Windows scripts follow the same steps but were not run on those
  machines.
