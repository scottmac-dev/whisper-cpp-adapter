"""The /v1/audio/transcriptions endpoint in each backend, with whisper-server replaced by a fake.

Requests are shaped like the ones an OpenAI whisper-1 client sends. Uploads really go through ffmpeg,
so ffmpeg must be installed.
"""

import copy
import io
import json
import shutil
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
from server_backend import WhisperServerError

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

FIXTURES = Path(__file__).parent / "fixtures"
URL = "/v1/audio/transcriptions"


class FakeWhisperServer:
    def __init__(self, response: dict | None = None, error: str | None = None):
        self.response, self.error, self.calls = response, error, []

    def inference(self, wav: bytes, fields: dict) -> dict:
        self.calls.append((wav, fields))
        if self.error:
            raise WhisperServerError(self.error)
        return copy.deepcopy(self.response)

    def reachable(self) -> bool:
        return True


def wav_bytes(seconds: float = 0.5) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return buffer.getvalue()


@pytest.fixture
def fake(monkeypatch):
    server = FakeWhisperServer(json.loads((FIXTURES / "jfk_after_noise_dtw.json").read_text()))
    monkeypatch.setattr(app_module, "WHISPER_BACKEND", "server")
    monkeypatch.setattr(app_module.app.state, "whisper_server", server, raising=False)
    return server


@pytest.fixture
def client():
    return TestClient(app_module.app)  # not entered, so the lifespan never starts a real whisper-server


def post(client, data=None, audio=None, filename="chunk.wav"):
    files = {"file": (filename, wav_bytes() if audio is None else audio, "audio/wav")}
    return client.post(URL, files=files, data=data or {}, headers={"Authorization": "Bearer sk-ignored"})


def whisper_1_request(**extra):
    """The request the AS3 subtitle server sends."""
    return {"model": "whisper-1", "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"], **extra}


# --- server backend -----------------------------------------------------------------------------

def test_whisper_1_verbose_json_request(client, fake):
    response = post(client, whisper_1_request())
    assert response.status_code == 200
    body = response.json()
    assert body["words"][0] == {"word": "And", "start": 5.5, "end": 5.74}
    assert len(body["segments"]) == 3
    assert body["usage"] == {"type": "duration", "seconds": 16}

    wav, fields = fake.calls[0]
    assert wav[:4] == b"RIFF"  # converted by ffmpeg before reaching whisper-server
    assert fields == {"response_format": "verbose_json", "language": "en", "suppress_nst": "true"}


def test_language_prompt_and_temperature_are_forwarded(client, fake):
    post(client, whisper_1_request(language="fr", prompt="Previously: hello", temperature="0.2"))
    _, fields = fake.calls[0]
    assert (fields["language"], fields["prompt"], fields["temperature"]) == ("fr", "Previously: hello", "0.2")


def test_json_keeps_the_estimated_token_usage_shape(client, fake):
    response = post(client, {"response_format": "json"})
    body = response.json()
    assert body["text"].startswith("And so my fellow Americans")
    assert body["usage"]["type"] == "tokens"
    assert body["usage"]["input_tokens"] == 160  # 16 s at 10 tokens per second


def test_text_is_plain(client, fake):
    response = post(client, {"response_format": "text"})
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text.endswith("for your country.")


@pytest.mark.parametrize("data, param", [
    ({"response_format": "json", "timestamp_granularities[]": "word"}, "timestamp_granularities[]"),
    ({"response_format": "verbose_json", "timestamp_granularities[]": "sentence"}, "timestamp_granularities[]"),
    ({"response_format": "srt"}, "response_format"),
])
def test_invalid_requests_get_openai_errors(client, fake, data, param):
    response = post(client, data)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert fake.calls == []


@pytest.mark.parametrize("audio, message", [(b"", "empty"), (b"not audio at all", "Could not decode")])
def test_unusable_uploads_are_rejected(client, fake, audio, message):
    response = post(client, whisper_1_request(), audio=audio)
    assert response.status_code == 400
    assert message in response.json()["error"]["message"]


def test_whisper_server_failure_is_a_502(client, fake):
    fake.error = "whisper-server unreachable: connection refused"
    response = post(client, whisper_1_request())
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "server_error"


def test_health_reports_the_server_backend(client, fake):
    body = client.get("/health").json()
    assert body["backend"] == "server"
    assert body["whisper_server_reachable"] is True
    assert {"dtw", "suppress_nst", "threads", "whisper_server"} <= set(body)


# --- cli backend --------------------------------------------------------------------------------

def test_cli_backend_explains_how_to_get_verbose_json(client, monkeypatch):
    monkeypatch.setattr(app_module, "WHISPER_BACKEND", "cli")
    response = post(client, whisper_1_request())
    assert response.status_code == 400
    assert "--server" in response.json()["detail"]


def test_cli_backend_health_is_unchanged_apart_from_backend(client, monkeypatch):
    monkeypatch.setattr(app_module, "WHISPER_BACKEND", "cli")
    body = client.get("/health").json()
    assert set(body) == {"whisper_cli", "whisper_cli_found", "model", "model_found", "ffmpeg_found", "backend"}
    assert body["backend"] == "cli"
