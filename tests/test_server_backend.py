"""server_backend: DTW preset discovery, the whisper-server command line, and startup failures."""

import socket
from pathlib import Path

import pytest

from server_backend import WhisperServer, WhisperServerError, dtw_preset_for


@pytest.mark.parametrize("model, preset", [
    ("ggml-base.en.bin", "base.en"),
    ("ggml-tiny.bin", "tiny"),
    ("ggml-small.en-q8_0.bin", "small.en"),
    ("ggml-large-v3.bin", "large.v3"),
    ("ggml-large-v3-turbo-q5_0.bin", "large.v3.turbo"),
    ("ggml-distil-small.en.bin", None),
    ("my-custom-model.bin", None),
])
def test_dtw_preset_for_model_file(model, preset):
    assert dtw_preset_for(Path("/models") / model) == preset


def server(tmp_path, **overrides) -> WhisperServer:
    options = {"port": 1, "threads": None, "dtw": "base.en", "env": {}, "log_path": tmp_path / "server.log"}
    options.update(overrides)
    return WhisperServer(Path("/bin/whisper-server"), Path("/models/ggml-base.en.bin"), **options)


def test_command_turns_on_dtw_without_flash_attention(tmp_path):
    cmd = server(tmp_path, port=9100, threads=3).command
    assert cmd[:2] == ["/bin/whisper-server", "-m"]
    assert ["--host", "127.0.0.1", "--port", "9100"] == cmd[3:7]
    assert "-nlp" in cmd
    assert cmd[cmd.index("-dtw") + 1] == "base.en" and "-nfa" in cmd
    assert cmd[cmd.index("-t") + 1] == "3"


def test_command_without_dtw_keeps_flash_attention(tmp_path):
    cmd = server(tmp_path, dtw=None).command
    assert "-dtw" not in cmd and "-nfa" not in cmd and "-t" not in cmd


def test_start_names_a_missing_binary(tmp_path):
    with pytest.raises(RuntimeError, match="whisper-server binary not found"):
        server(tmp_path).start()


def test_start_refuses_a_port_already_in_use(tmp_path):
    binary, model = tmp_path / "whisper-server", tmp_path / "ggml-base.en.bin"
    binary.touch()
    model.touch()
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        ws = WhisperServer(binary, model, port=port, threads=None, dtw=None, env={}, log_path=tmp_path / "log")
        with pytest.raises(RuntimeError, match=f"port {port} is already in use"):
            ws.start()


def test_inference_reports_an_unreachable_server(tmp_path):
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    with pytest.raises(WhisperServerError, match="unreachable"):
        server(tmp_path, port=port).inference(b"RIFF", {})
