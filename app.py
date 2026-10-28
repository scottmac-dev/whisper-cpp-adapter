"""Minimal OpenAI-compatible transcription endpoint backed by a local whisper.cpp build.

Two backends, picked once at startup by WHISPER_BACKEND:
  cli     (default) whisper-cli per request; json and text only.
  server  a long-running whisper-server child; adds verbose_json with word and segment timestamps.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

import verbose
from server_backend import DTW_PRESETS, WhisperServer, WhisperServerError, dtw_preset_for

log = logging.getLogger("whisper-local-server")

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

WHISPER_BACKEND = os.environ.get("WHISPER_BACKEND", "cli").strip().lower()
if WHISPER_BACKEND not in ("cli", "server"):
    raise RuntimeError(f"WHISPER_BACKEND must be 'cli' or 'server', not {WHISPER_BACKEND!r}")


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in ("0", "false", "no", "off")


def _dtw_setting(value: str) -> str | None:
    """auto: the preset matching WHISPER_MODEL. off: no DTW. Anything else must name a preset."""
    value = value.strip().lower()
    if value == "auto":
        return dtw_preset_for(WHISPER_MODEL)
    if value == "off":
        return None
    if value not in DTW_PRESETS:
        raise RuntimeError(f"WHISPER_DTW must be auto, off or one of {sorted(DTW_PRESETS)}, not {value!r}")
    return value


# Server backend settings. whisper-server is built alongside whisper-cli, so look beside it first.
WHISPER_SERVER = Path(os.environ.get("WHISPER_SERVER") or WHISPER_CLI.with_name(
    "whisper-server.exe" if WHISPER_CLI.suffix == ".exe" else "whisper-server")).expanduser()
WHISPER_SERVER_PORT = int(os.environ.get("WHISPER_SERVER_PORT", "8178"))
WHISPER_THREADS = int(os.environ["WHISPER_THREADS"]) if os.environ.get("WHISPER_THREADS") else None
WHISPER_DTW = _dtw_setting(os.environ.get("WHISPER_DTW", "auto"))
WHISPER_SUPPRESS_NST = _flag("WHISPER_SUPPRESS_NST", True)
WHISPER_SERVER_LOG = Path(__file__).resolve().parent / "whisper-server.log"

# whisper.cpp does not report token counts, so usage is estimated. See README.
AUDIO_TOKENS_PER_SECOND = 10
CHARS_PER_OUTPUT_TOKEN = 4


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """In server mode, whisper-server lives exactly as long as the app."""
    app.state.whisper_server = None
    if WHISPER_BACKEND == "server":
        if WHISPER_DTW is None:
            log.warning("DTW timestamps are off (no preset for %s, or WHISPER_DTW=off): word timings will be coarse",
                        WHISPER_MODEL.name)
        server = WhisperServer(WHISPER_SERVER, WHISPER_MODEL, port=WHISPER_SERVER_PORT, threads=WHISPER_THREADS,
                               dtw=WHISPER_DTW, env=_subprocess_env(WHISPER_SERVER), log_path=WHISPER_SERVER_LOG)
        await asyncio.to_thread(server.start)
        app.state.whisper_server = server
    try:
        yield
    finally:
        if app.state.whisper_server is not None:
            app.state.whisper_server.stop()


app = FastAPI(title="whisper-local-server", lifespan=_lifespan)


def _subprocess_env(binary: Path = WHISPER_CLI) -> dict:
    """whisper.cpp binaries link against libwhisper sitting next to them; make sure the loader finds it.

    Windows already searches the executable's own directory, so it needs no help.
    """
    env = dict(os.environ)
    if sys.platform == "win32":
        return env
    var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    lib_dir = str(binary.resolve().parent)
    existing = env.get(var, "")
    env[var] = f"{lib_dir}{os.pathsep}{existing}" if existing else lib_dir
    return env


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_subprocess_env())
    if result.returncode != 0:
        raise HTTPException(500, f"{Path(cmd[0]).name} failed: {result.stderr[-2000:]}")
    return result


def _wav_command(source: Path, wav: Path) -> list[str]:
    """Any ffmpeg-readable container (.webm, .mp3, ...) -> the 16 kHz mono WAV whisper.cpp wants."""
    return ["ffmpeg", "-nostdin", "-y", "-i", str(source), "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", str(wav)]


def _usage(text: str, wav_path: Path) -> dict:
    # 16 kHz mono 16-bit PCM => 32000 bytes per second, minus the 44-byte header.
    return _estimated_usage(text, max(0.0, (wav_path.stat().st_size - 44) / 32000))


def _estimated_usage(text: str, seconds: float) -> dict:
    audio_tokens = round(seconds * AUDIO_TOKENS_PER_SECOND)
    output_tokens = -(-len(text) // CHARS_PER_OUTPUT_TOKEN)  # ceil
    return {
        "type": "tokens",
        "input_tokens": audio_tokens,
        "input_token_details": {"text_tokens": 0, "audio_tokens": audio_tokens},
        "output_tokens": output_tokens,
        "total_tokens": audio_tokens + output_tokens,
    }


class OpenAIError(Exception):
    """Rendered in OpenAI's error shape, so clients built for the real API handle it unchanged."""

    def __init__(self, status: int, message: str, param: str | None = None,
                 type_: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.param, self.type = status, message, param, type_


@app.exception_handler(OpenAIError)
async def render_openai_error(request: Request, exc: OpenAIError):
    return JSONResponse(
        status_code=exc.status,
        content={"error": {"message": exc.message, "type": exc.type, "param": exc.param, "code": None}},
    )


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
    info = {
        "whisper_cli": str(WHISPER_CLI),
        "whisper_cli_found": WHISPER_CLI.is_file(),
        "model": str(WHISPER_MODEL),
        "model_found": WHISPER_MODEL.is_file(),
        "ffmpeg_found": shutil.which("ffmpeg") is not None,
        "backend": WHISPER_BACKEND,
    }
    if WHISPER_BACKEND == "server":
        server = getattr(app.state, "whisper_server", None)
        info.update({
            "whisper_server": str(WHISPER_SERVER),
            "whisper_server_found": WHISPER_SERVER.is_file(),
            "whisper_server_reachable": server is not None and server.reachable(),
            "dtw": WHISPER_DTW,
            "suppress_nst": WHISPER_SUPPRESS_NST,
            "threads": WHISPER_THREADS,
        })
    return info


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("gpt-4o-mini-transcribe"),  # accepted and ignored
    response_format: str = Form("json"),
    timestamp_granularities: list[str] = Form([], alias="timestamp_granularities[]"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    temperature: float | None = Form(None),
    authorization: str | None = Header(None),  # accepted and ignored
):
    if WHISPER_BACKEND == "server":
        audio = await file.read()
        # Conversion and inference block, so they run off the event loop and /health stays responsive.
        return await run_in_threadpool(
            _transcribe_with_server, app.state.whisper_server, audio, file.filename, response_format,
            timestamp_granularities, language, prompt, temperature)

    # The cli backend: json and text only, timestamp_granularities, language, prompt and temperature ignored.
    if response_format == "verbose_json":
        raise HTTPException(400, "verbose_json needs the server backend - start with ./run.sh --server "
                                 "(or set WHISPER_BACKEND=server)")
    if response_format not in ("json", "text"):
        raise HTTPException(400, "response_format must be 'json' or 'text'")
    for path, label in ((WHISPER_CLI, "WHISPER_CLI"), (WHISPER_MODEL, "WHISPER_MODEL")):
        if not path.is_file():
            raise HTTPException(500, f"{label} not found at {path} - see README setup")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / f"input{Path(file.filename or '').suffix or '.webm'}"
        source.write_bytes(await file.read())

        wav = tmp / "audio.wav"
        _run(_wav_command(source, wav))

        result = _run([str(WHISPER_CLI), "-m", str(WHISPER_MODEL), "-f", str(wav),
                       "-l", WHISPER_LANGUAGE, "-nt", "-np"])
        text = " ".join(result.stdout.split())

        if response_format == "text":
            return PlainTextResponse(text)
        return {"text": text, "usage": _usage(text, wav)}


def _transcribe_with_server(server: WhisperServer, audio: bytes, filename: str | None, response_format: str,
                            granularities: list[str], language: str | None, prompt: str | None,
                            temperature: float | None):
    """The server backend: validates like OpenAI does, then reshapes whisper-server's verbose_json."""
    if response_format not in ("json", "text", "verbose_json"):
        raise OpenAIError(400, "response_format must be one of json, text, verbose_json", "response_format")
    unknown = sorted(set(granularities) - set(verbose.GRANULARITIES))
    if unknown:
        raise OpenAIError(400, f"timestamp_granularities[] must be 'word' and/or 'segment', got {unknown}",
                          "timestamp_granularities[]")
    if granularities and response_format != "verbose_json":
        raise OpenAIError(400, "timestamp_granularities[] requires response_format=verbose_json",
                          "timestamp_granularities[]")
    if not audio:
        raise OpenAIError(400, "The uploaded file is empty", "file")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / f"input{Path(filename or '').suffix or '.webm'}"
        source.write_bytes(audio)
        wav = tmp / "audio.wav"
        if subprocess.run(_wav_command(source, wav), capture_output=True).returncode != 0:
            raise OpenAIError(400, "Could not decode the uploaded file as audio", "file")
        wav_bytes = wav.read_bytes()

    # whisper-server always answers in verbose_json; json and text are derived from it, so every
    # format gets the same suppression of non-speech annotations like [BLANK_AUDIO].
    fields = {
        "response_format": "verbose_json",
        "language": language or WHISPER_LANGUAGE,
        "suppress_nst": "true" if WHISPER_SUPPRESS_NST else "false",
    }
    if prompt:
        fields["prompt"] = prompt
    if temperature is not None:
        fields["temperature"] = str(temperature)
    try:
        raw = server.inference(wav_bytes, fields)
    except WhisperServerError as e:
        raise OpenAIError(502, str(e), type_="server_error") from e

    result = verbose.to_openai_verbose(raw, granularities)
    if response_format == "verbose_json":
        return result
    text = " ".join(result["text"].split())
    if response_format == "text":
        return PlainTextResponse(text)
    return {"text": text, "usage": _estimated_usage(text, result["duration"])}
