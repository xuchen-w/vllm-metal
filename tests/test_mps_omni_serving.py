# SPDX-License-Identifier: Apache-2.0
"""Real HTTP regressions for local Qwen3-TTS CustomVoice and Ming dense 0.5B.

Set QWEN3_TTS_TEST_MODEL and/or MING_TTS_TEST_MODEL to enable each checkpoint.
"""

import asyncio
import base64
import importlib.util
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.backends.mps.is_available()
        or importlib.util.find_spec("vllm_omni") is None,
        reason="Requires MPS and the Omni extra",
    ),
]


@dataclass(frozen=True)
class SpeechModel:
    family: str
    checkpoint: str
    sample_rate: int


@pytest.fixture(scope="module", params=["qwen3", "ming"])
def speech_model(request):
    variable, sample_rate = {
        "qwen3": ("QWEN3_TTS_TEST_MODEL", 24000),
        "ming": ("MING_TTS_TEST_MODEL", 44100),
    }[request.param]
    checkpoint = os.environ.get(variable)
    if not checkpoint:
        pytest.skip(f"Set {variable} to a local checkpoint")
    return SpeechModel(request.param, checkpoint, sample_rate)


@pytest.fixture(scope="module")
def speech_server(tmp_path_factory, speech_model):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    directory = tmp_path_factory.mktemp(f"mps-omni-{speech_model.family}")
    log = directory / "server.log"
    env = dict(
        os.environ,
        VLLM_METAL_BACKEND="mps",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        VLLM_NO_USAGE_STATS="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        HF_HUB_OFFLINE="1",
    )
    command = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        speech_model.checkpoint,
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if speech_model.family == "ming":
        import vllm_omni

        command += [
            "--deploy-config",
            str(Path(vllm_omni.__file__).parent / "deploy" / "ming_tts.yaml"),
        ]
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            with httpx.Client(timeout=2, trust_env=False) as client:
                for _ in range(240):
                    assert process.poll() is None, log.read_text()
                    try:
                        if client.get(f"{url}/health").status_code == 200:
                            break
                    except (httpx.ConnectError, httpx.TimeoutException):
                        pass
                    time.sleep(1)
                else:
                    pytest.fail(f"Omni server did not become ready:\n{log.read_text()}")
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def speech_request(index, speech_model):
    request = {
        "input": [
            "Hello, have a wonderful day.",
            "Please remember to bring your ticket. The library closes at six.",
        ][index % 2],
        "max_new_tokens": 128,
        "seed": 42,
        "response_format": "pcm",
        "stream": True,
    }
    if speech_model.family == "qwen3":
        request.update(task_type="CustomVoice", voice="Ryan", language="English")
    else:
        request["input"] = [
            "你好，欢迎使用语音合成。",
            "请记得带上车票。图书馆将在晚上六点关闭。",
        ][index % 2]
    return request


async def speech_audio(client, speech_server, speech_model, request):
    events, audio = [], bytearray()
    async with client.stream(
        "POST", f"{speech_server}/v1/audio/speech", json=request
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            assert event["type"] != "speech.audio.error", event
            events.append(event["type"])
            if event["type"] == "speech.audio.delta":
                assert event["sample_rate"] == speech_model.sample_rate
                assert event["channels"] == 1
                audio.extend(base64.b64decode(event["audio"]))
            elif event["type"] == "speech.audio.done":
                assert 0 < event["usage"]["output_tokens"] < request["max_new_tokens"]
    assert len(audio) % 2 == 0
    samples = np.frombuffer(audio, dtype="<i2")
    assert samples.size > speech_model.sample_rate // 10
    assert np.count_nonzero(samples) > 0
    assert events.count("speech.audio.delta") > 1
    assert events.count("speech.audio.done") == 1
    assert events[0] == "speech.audio.delta"
    assert events[-1] == "speech.audio.done"


@pytest.mark.parametrize("concurrency", [1, 4, 8])
async def test_speech_streams_complete_for_concurrent_requests(
    speech_server, speech_model, concurrency
):
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        await asyncio.gather(
            *(
                speech_audio(
                    client,
                    speech_server,
                    speech_model,
                    speech_request(index, speech_model),
                )
                for index in range(concurrency)
            )
        )


@pytest.mark.parametrize("speech_model", ["qwen3"], indirect=True)
async def test_qwen3_speech_rejects_random_sampling(speech_server, speech_model):
    request = speech_request(0, speech_model)
    request["extra_params"] = {"temperature": 0.9}
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        response = await client.post(f"{speech_server}/v1/audio/speech", json=request)
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == ["speech.audio.error"]
    assert "temperature" in events[0]["error"]["message"]


async def test_speech_returns_complete_nonstreaming_wave(speech_server, speech_model):
    request = speech_request(0, speech_model)
    request.update(
        input="Hello, this is a speech synthesis test.",
        stream=False,
        response_format="wav",
    )
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        response = await client.post(f"{speech_server}/v1/audio/speech", json=request)
    assert response.status_code == 200, response.text[:500]
    with wave.open(io.BytesIO(response.content)) as audio:
        assert audio.getframerate() == speech_model.sample_rate
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getnframes() > speech_model.sample_rate // 10
        assert (
            np.count_nonzero(
                np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
            )
            > 0
        )


async def test_cancelled_speech_does_not_poison_live_or_next_request(
    speech_server, speech_model
):
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:

        async def cancel_after_first_audio():
            request = speech_request(1, speech_model)
            request["input"] *= 10
            request["max_new_tokens"] = 512
            async with client.stream(
                "POST", f"{speech_server}/v1/audio/speech", json=request
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    assert event["type"] == "speech.audio.delta", event
                    assert base64.b64decode(event["audio"])
                    return  # Close the HTTP stream before consuming its terminal event.
            pytest.fail("Cancelled request produced no audio")

        await asyncio.gather(
            cancel_after_first_audio(),
            speech_audio(
                client, speech_server, speech_model, speech_request(0, speech_model)
            ),
        )
        await speech_audio(
            client, speech_server, speech_model, speech_request(1, speech_model)
        )
