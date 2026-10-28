"""whisper.cpp's whisper-server run as a child process: the backend behind WHISPER_BACKEND=server.

Unlike whisper-cli, the server keeps the model loaded between requests and reports avg_logprob and
no_speech_prob per segment, which verbose_json needs. The adapter converts uploads to WAV itself,
so whisper-server never needs its own ffmpeg (--convert) step.
"""

import re
import socket
import subprocess
import time
from pathlib import Path

import httpx

# whisper.cpp's alignment-head presets for DTW token timestamps (whisper-server --dtw <preset>).
DTW_PRESETS = frozenset({
    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
    "large.v1", "large.v2", "large.v3", "large.v3.turbo",
})

# Loading a large model from a cold disk can take a while.
STARTUP_TIMEOUT_SECONDS = 120


def dtw_preset_for(model: Path) -> str | None:
    """The DTW preset matching a ggml model file, or None if there isn't one.

    ggml-base.en.bin -> base.en, ggml-large-v3-turbo-q5_0.bin -> large.v3.turbo
    """
    name = re.sub(r"^ggml-|\.bin$", "", model.name)
    name = re.sub(r"-q\d+(_\w+)?$", "", name)  # quantisation suffix, eg -q5_0, -q8_0
    preset = name.replace("-", ".")
    return preset if preset in DTW_PRESETS else None


class WhisperServerError(Exception):
    """whisper-server could not be reached, or answered with an error."""


class WhisperServer:
    """Owns one whisper-server process on a loopback port, and the HTTP client that talks to it."""

    def __init__(self, binary: Path, model: Path, *, port: int, threads: int | None,
                 dtw: str | None, env: dict, log_path: Path):
        self.binary = binary
        self.model = model
        self.port = port
        self.threads = threads
        self.dtw = dtw
        self._env = env
        self._log_path = log_path
        self._process: subprocess.Popen | None = None
        self._log = None
        self._client = httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            # Inference on a long file can take minutes on CPU; connecting should be instant.
            timeout=httpx.Timeout(600.0, connect=2.0),
        )

    @property
    def command(self) -> list[str]:
        # -nlp: skip language probabilities, which verbose_json would otherwise compute on every request.
        cmd = [str(self.binary), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(self.port), "-nlp"]
        if self.dtw:
            # DTW timestamps do not work with flash attention, so it is switched off (-nfa).
            cmd += ["-nfa", "-dtw", self.dtw]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        return cmd

    def start(self) -> None:
        """Launches whisper-server and blocks until it answers, or raises with the tail of its log."""
        for path, label in ((self.binary, "whisper-server binary"), (self.model, "model")):
            if not path.is_file():
                raise RuntimeError(f"{label} not found at {path} - see README")
        # Fail clearly rather than talk to a stray process that already holds the port.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(
                    f"port {self.port} is already in use - stop the other process or set WHISPER_SERVER_PORT")

        self._log = open(self._log_path, "ab")
        self._process = subprocess.Popen(
            self.command, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=self._env)

        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                self.stop()
                raise RuntimeError(f"whisper-server exited with code {code} during startup:\n{self._log_tail()}")
            if self.reachable():
                return
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"whisper-server did not start within {STARTUP_TIMEOUT_SECONDS}s:\n{self._log_tail()}")

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None
        self._client.close()

    def reachable(self) -> bool:
        if self._process is None or self._process.poll() is not None:
            return False
        try:
            return self._client.get("/").status_code == 200
        except httpx.HTTPError:
            return False

    def inference(self, wav: bytes, fields: dict[str, str]) -> dict:
        """POSTs 16 kHz mono WAV to /inference and returns whisper-server's JSON response."""
        try:
            response = self._client.post("/inference", files={"file": ("audio.wav", wav, "audio/wav")}, data=fields)
        except httpx.HTTPError as e:
            raise WhisperServerError(f"whisper-server unreachable: {e}") from e
        try:
            body = response.json()
        except ValueError:
            body = {"error": response.text[-500:]}
        if response.status_code != 200 or "error" in body:
            raise WhisperServerError(f"whisper-server returned {response.status_code}: {body.get('error')}")
        return body

    def _log_tail(self, limit: int = 2000) -> str:
        try:
            return self._log_path.read_bytes()[-limit:].decode(errors="replace")
        except OSError:
            return "(no log)"
