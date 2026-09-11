# whisper-local-server

An OpenAI-compatible transcription endpoint backed by a local [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
build. Point any OpenAI client at it and `gpt-4o-mini-transcribe` calls run on your machine — no API key.

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
```

```powershell
.\run.ps1                   # Windows
```

That one command does the whole setup: creates `.venv`, installs the Python dependencies, finds an
existing whisper.cpp (`./whisper.cpp` or `../whisper.cpp`) or builds one via `scripts/setup_whisper.*`,
picks a model, and starts the server.

**The first run takes several minutes** — it compiles whisper.cpp and downloads the `base.en` model
(~150 MB). Later runs start immediately. Pass a port to use something other than 8000
(`./run.sh 9000`).

Use `./run.sh`, not `uvicorn` — see [HTTP/2](#http2-and-why-not-uvicorn) below if you start the
server by hand.

### 4. Check that it works

```bash
curl http://localhost:8000/health      # reports whether the binary, model and ffmpeg were found
./scripts/smoke_test.sh                # end-to-end: uploads a real sample and prints the transcript
```

Interactive API docs are at <http://localhost:8000/docs>.

## Manual setup

`run.sh` is the short path; these are the same steps if you'd rather run them yourself, already have
a whisper.cpp build, or are wiring the server into another tool.

```bash
# 1. Virtual environment and Python dependencies (fastapi, python-multipart, hypercorn).
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Build whisper.cpp and download a model. Skip if you already have one.
./scripts/setup_whisper.sh          # Windows: .\scripts\setup_whisper.ps1
                                    # optional arg picks the model: ./scripts/setup_whisper.sh tiny.en

# 3. Point the server at the binary and model (paths the setup script prints at the end).
export WHISPER_CLI=./whisper.cpp/build/bin/whisper-cli
export WHISPER_MODEL=./whisper.cpp/models/ggml-base.en.bin

# 4. Start the server.
hypercorn app:app --bind 127.0.0.1:8000
```

On Windows, set the environment variables with `$env:WHISPER_CLI = "..."` instead of `export`.

### Environment variables

All optional — the server and `run.sh` discover these on their own. Set them to use a whisper.cpp
build that lives somewhere else, or a different model or language.

| Variable | Default |
| --- | --- |
| `WHISPER_CLI` | first `whisper-cli` found under `./whisper.cpp` or `../whisper.cpp` |
| `WHISPER_MODEL` | first `ggml-*.bin` found beside that binary |
| `WHISPER_LANGUAGE` | `en` (`auto` to detect) |

## Use it

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@recording.webm -F model=gpt-4o-mini-transcribe -F response_format=json
```

## The endpoint

`POST /v1/audio/transcriptions`, `multipart/form-data`. Also `GET /health`, which reports whether
the binary, model and ffmpeg were found.

| Field | |
| --- | --- |
| `file` | **Required.** Anything ffmpeg reads (`.webm`, `.mp3`, `.wav`, ...). Keep the extension on the upload. |
| `model` | Accepted and ignored — the model is whatever `WHISPER_MODEL` points at. |
| `response_format` | `json` (default) or `text`. Anything else is a 400. |
| `Authorization` | Accepted and ignored. |

`json` returns the OpenAI shape; `text` returns the bare transcript as `text/plain`.

```json
{
  "text": "And so my fellow Americans ask not what your country can do for you...",
  "usage": {
    "type": "tokens",
    "input_tokens": 110,
    "input_token_details": { "text_tokens": 0, "audio_tokens": 110 },
    "output_tokens": 27,
    "total_tokens": 137
  }
}
```

## Troubleshooting

A 422 means the multipart body was not what the server expected. The error message reports what
actually arrived — match it here:

| Message says | Cause | Fix |
| --- | --- | --- |
| `parts: ['audio']` or similar | Part is not named `file` | Name it exactly `file` |
| `parts: none`, Content-Type not multipart | Sent as JSON | Send `multipart/form-data` |
| `parts: ['file']` + `Expected UploadFile, received str` | Part has no filename | Give the part a filename |
| `Missing boundary in multipart.` | Content-Type set by hand | Let the client set it |
| `parts: none` **with** a valid boundary, plus `Unsupported upgrade request` in the log | You are running uvicorn, and the client upgraded to HTTP/2 | Start with `./run.sh` — below |

A 500 naming `WHISPER_CLI` or `WHISPER_MODEL` means the build or the model was not found. Check
`GET /health` for the paths the server is actually using, then set the variables above.

### HTTP/2, and why not uvicorn

This server runs on **hypercorn**. Uvicorn speaks HTTP/1.1 only: when a client asks to upgrade to
HTTP/2 it logs `Unsupported upgrade request` and **silently discards the request body**, so a
well-formed upload arrives with zero parts and fails as a confusing 422.

Java's `HttpClient` — what Spring's `RestClient` uses by default — does exactly this, sending
`Upgrade: h2c` on plain `http://` URLs. It never happens against the real OpenAI API, where HTTPS
negotiates the protocol via ALPN, so it only appears against a local server.

`./run.sh` uses hypercorn, which handles the upgrade correctly. If you start the server by hand,
use `hypercorn app:app --bind 127.0.0.1:8000`.

### Java / Spring: the file part

A `ByteArrayResource` has no filename unless you give it one, which makes the part arrive as a
plain text field rather than a file.

```java
ByteArrayResource audio = new ByteArrayResource(bytes) {
    @Override public String getFilename() { return "recording.webm"; }   // required
};
MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
body.add("file", audio);                                                 // exact name: "file"
body.add("model", "gpt-4o-mini-transcribe");
body.add("response_format", "json");
```

## Limitations

- **Token counts are estimated.** whisper.cpp reports none, so `usage` is derived from audio
  duration (10 tokens/sec) and transcript length (4 chars/token) — plausibly shaped, not accurate.
  Constants are at the top of `app.py`.
- One request at a time per worker, blocking until whisper.cpp finishes. No auth, no streaming, no
  `verbose_json`/`srt`/`vtt`. Run it on localhost.
- Verified on Linux. The macOS and Windows scripts follow the same steps but were not run on those
  machines.
