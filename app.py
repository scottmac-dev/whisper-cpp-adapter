"""Minimal OpenAI-compatible transcription endpoint backed by a local whisper.cpp build."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

# Where a local whisper.cpp build usually lands. Windows keeps the .exe suffix and MSVC adds a
# config subdirectory, so check both; the setup scripts print the exact paths they produced.
_CHECKOUTS = ("./whisper.cpp", "../whisper.cpp")
_BIN_NAMES = ("build/bin/whisper-cli", "build/bin/Release/whisper-cli.exe", "build/bin/whisper-cli.exe")
_FALLBACK_CLI = "../whisper.cpp/build/bin/whisper-cli"
_FALLBACK_MODEL = "../whisper.cpp/models/ggml-base.en.bin"


def _find(candidates, fallback: str) -> str:
    """First candidate that exists, or the fallback so errors name a sensible path."""
    return next((str(c) for c in candidates if Path(c).is_file()), fallback)


# Point these at an existing whisper.cpp checkout, or run scripts/setup_whisper.* to build one.
WHISPER_CLI = Path(os.environ.get("WHISPER_CLI") or _find(
    (f"{d}/{n}" for d in _CHECKOUTS for n in _BIN_NAMES), _FALLBACK_CLI)).expanduser()
WHISPER_MODEL = Path(os.environ.get("WHISPER_MODEL") or _find(
    (m for d in _CHECKOUTS for m in sorted(Path(f"{d}/models").glob("ggml-*.bin"))
     if not m.name.startswith("for-tests")), _FALLBACK_MODEL)).expanduser()
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")

# whisper.cpp does not report token counts, so usage is estimated. See README.
AUDIO_TOKENS_PER_SECOND = 10
CHARS_PER_OUTPUT_TOKEN = 4

app = FastAPI(title="whisper-local-server")


def _subprocess_env() -> dict:
    """whisper-cli links against libwhisper sitting next to it; make sure the loader finds it.

    Windows already searches the executable's own directory, so it needs no help.
    """
    env = dict(os.environ)
    if sys.platform == "win32":
        return env
    var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    lib_dir = str(WHISPER_CLI.resolve().parent)
    existing = env.get(var, "")
    env[var] = f"{lib_dir}{os.pathsep}{existing}" if existing else lib_dir
    return env


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_subprocess_env())
    if result.returncode != 0:
        raise HTTPException(500, f"{Path(cmd[0]).name} failed: {result.stderr[-2000:]}")
    return result


def _usage(text: str, wav_path: Path) -> dict:
    # 16 kHz mono 16-bit PCM => 32000 bytes per second, minus the 44-byte header.
    seconds = max(0.0, (wav_path.stat().st_size - 44) / 32000)
    audio_tokens = round(seconds * AUDIO_TOKENS_PER_SECOND)
    output_tokens = -(-len(text) // CHARS_PER_OUTPUT_TOKEN)  # ceil
    return {
        "type": "tokens",
        "input_tokens": audio_tokens,
        "input_token_details": {"text_tokens": 0, "audio_tokens": audio_tokens},
        "output_tokens": output_tokens,
        "total_tokens": audio_tokens + output_tokens,
    }


@app.exception_handler(RequestValidationError)
async def explain_bad_request(request: Request, exc: RequestValidationError):
    """Say what actually arrived. Nearly every client bug here is a mis-built multipart body."""
    content_type = request.headers.get("content-type", "(none)")
    try:
        parts = list((await request.form()).keys())
    except Exception:
        parts = []
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": (
                    "Expected multipart/form-data with a file part named 'file'. "
                    f"Received Content-Type: {content_type}; parts: {parts or 'none'}. "
                    "Check that the part is named exactly 'file' and that it carries a "
                    "filename (so the extension survives)."
                ),
                "type": "invalid_request_error",
                "param": "file",
                "code": None,
            },
            "detail": [{"loc": e.get("loc"), "msg": e.get("msg")} for e in exc.errors()],
        },
    )


@app.get("/health")
def health() -> dict:
    return {
        "whisper_cli": str(WHISPER_CLI),
        "whisper_cli_found": WHISPER_CLI.is_file(),
        "model": str(WHISPER_MODEL),
        "model_found": WHISPER_MODEL.is_file(),
        "ffmpeg_found": shutil.which("ffmpeg") is not None,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("gpt-4o-mini-transcribe"),  # accepted and ignored
    response_format: str = Form("json"),
    authorization: str | None = Header(None),  # accepted and ignored
):
    if response_format not in ("json", "text"):
        raise HTTPException(400, "response_format must be 'json' or 'text'")
    for path, label in ((WHISPER_CLI, "WHISPER_CLI"), (WHISPER_MODEL, "WHISPER_MODEL")):
        if not path.is_file():
            raise HTTPException(500, f"{label} not found at {path} - see README setup")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / f"input{Path(file.filename or '').suffix or '.webm'}"
        source.write_bytes(await file.read())

        # Any ffmpeg-readable container (.webm, .mp3, ...) -> the 16 kHz mono WAV whisper.cpp wants.
        wav = tmp / "audio.wav"
        _run(["ffmpeg", "-nostdin", "-y", "-i", str(source), "-ar", "16000", "-ac", "1",
              "-c:a", "pcm_s16le", str(wav)])

        result = _run([str(WHISPER_CLI), "-m", str(WHISPER_MODEL), "-f", str(wav),
                       "-l", WHISPER_LANGUAGE, "-nt", "-np"])
        text = " ".join(result.stdout.split())

        if response_format == "text":
            return PlainTextResponse(text)
        return {"text": text, "usage": _usage(text, wav)}
