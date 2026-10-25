# SPDX-License-Identifier: Apache-2.0
"""Optional real HTTP regression using a local Qwen3-TTS CustomVoice checkpoint."""

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

import httpx
import numpy as np
import pytest
import torch
from prometheus_client.parser import text_string_to_metric_families

MODEL = os.environ.get("QWEN3_TTS_TEST_MODEL")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not MODEL
        or not torch.backends.mps.is_available()
        or importlib.util.find_spec("vllm_omni") is None,
        reason="Requires MPS, the Omni extra, and QWEN3_TTS_TEST_MODEL (CustomVoice)",
    ),
]


@pytest.fixture(scope="module")
def speech_server(tmp_path_factory):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    directory = tmp_path_factory.mktemp("mps-omni")
    log = directory / "server.log"
    env = dict(
        os.environ,
        VLLM_METAL_BACKEND="mps",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        VLLM_NO_USAGE_STATS="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        HF_HUB_OFFLINE="1",
    )
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm_omni.entrypoints.cli.main",
                "serve",
                MODEL,
                "--omni",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
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


def speech_request(index):
    return {
        "input": [
            "Hello, have a wonderful day.",
            "Please remember to bring your ticket. The library closes at six.",
        ][index % 2],
        "task_type": "CustomVoice",
        "voice": "Ryan",
        "language": "English",
        "max_new_tokens": 128,
        "response_format": "pcm",
        "stream": True,
    }


@dataclass(frozen=True)
class SpeechAudio:
    pcm: bytes
    chunk_sizes: tuple[int, ...]
    input_tokens: int
    output_tokens: int


async def speech_audio(client, speech_server, request):
    events, chunks = [], []
    usage = None
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
                assert event["sample_rate"] == 24000
                assert event["channels"] == 1
                chunk = base64.b64decode(event["audio"])
                assert chunk and len(chunk) % 2 == 0
                chunks.append(chunk)
            elif event["type"] == "speech.audio.done":
                usage = event["usage"]
    assert usage is not None and 0 < usage["output_tokens"] < request["max_new_tokens"]
    assert events.count("speech.audio.delta") > 1
    assert events.count("speech.audio.done") == 1
    assert events[0] == "speech.audio.delta"
    assert events[-1] == "speech.audio.done"
    pcm = b"".join(chunks)
    samples = np.frombuffer(pcm, dtype="<i2")
    assert samples.size > 2400
    assert samples.min() < samples.max()
    return SpeechAudio(
        pcm,
        tuple(len(chunk) for chunk in chunks),
        usage["input_tokens"],
        usage["output_tokens"],
    )


async def pipeline_inflight(client, speech_server):
    response = await client.get(f"{speech_server}/metrics")
    response.raise_for_status()
    names = {"vllm_omni:num_requests_running", "vllm_omni:num_requests_waiting"}
    values = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name in names:
                values[sample.name] = values.get(sample.name, 0) + sample.value
    assert values.keys() == names, "Pipeline request gauges must be exported"
    return values


async def wait_for_pipeline_idle(client, speech_server):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        values = await pipeline_inflight(client, speech_server)
        if not any(values.values()):
            return
        await asyncio.sleep(0.1)
    pytest.fail(f"Pipeline retained requests after completion/cancellation: {values}")


@pytest.mark.parametrize("concurrency", [1, 4, 8])
async def test_speech_streams_complete_for_concurrent_requests(
    speech_server, concurrency
):
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        await asyncio.gather(
            *(
                speech_audio(client, speech_server, speech_request(index))
                for index in range(concurrency)
            )
        )
        await wait_for_pipeline_idle(client, speech_server)


async def test_speech_rejects_random_sampling(speech_server):
    request = speech_request(0)
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


async def test_long_stream_preserves_the_complete_nonstreaming_wave(speech_server):
    request = speech_request(0)
    request.update(
        input=(
            "Good morning. Today we are testing a speech service that can handle "
            "several requests at once. It returns audio as each part is ready. "
            "This longer example also checks that the final words are preserved. "
            "Thank you for listening."
        ),
        max_new_tokens=512,
    )
    async with httpx.AsyncClient(timeout=240, trust_env=False) as client:
        streamed = await speech_audio(client, speech_server, request)
        assert len(streamed.chunk_sizes) >= 4
        assert streamed.output_tokens > 75
        await wait_for_pipeline_idle(client, speech_server)
        response = await client.post(
            f"{speech_server}/v1/audio/speech",
            json={**request, "stream": False, "response_format": "wav"},
        )
        assert response.status_code == 200, response.text[:500]
        with wave.open(io.BytesIO(response.content)) as audio:
            assert audio.getframerate() == 24000
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            complete = audio.readframes(audio.getnframes())
        # Both requests run alone with the same greedy input and chunk pipeline.
        # Comparing the entire PCM catches a missing, repeated or truncated tail.
        assert streamed.pcm == complete
        await wait_for_pipeline_idle(client, speech_server)


async def cancel_after_first_audio(client, speech_server):
    request = speech_request(1)
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
            return
    pytest.fail("Cancelled request produced no audio")


async def test_cancelled_speech_releases_capacity_for_live_and_new_requests(
    speech_server,
):
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        await asyncio.gather(
            cancel_after_first_audio(client, speech_server),
            speech_audio(client, speech_server, speech_request(0)),
        )
        await wait_for_pipeline_idle(client, speech_server)
        await speech_audio(client, speech_server, speech_request(1))
        await wait_for_pipeline_idle(client, speech_server)


async def test_queued_mixed_lengths_do_not_poison_reused_request_slots(speech_server):
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        anchor = await speech_audio(client, speech_server, speech_request(0))
        await wait_for_pipeline_idle(client, speech_server)

        async def generate(index):
            request = speech_request(1 if index % 2 == 0 else 0)
            if index % 2 == 0:
                request["input"] *= 2
                request["max_new_tokens"] = 256
            return index, await speech_audio(client, speech_server, request)

        tasks = [asyncio.create_task(generate(index)) for index in range(12)]
        completed = [await task for task in asyncio.as_completed(tasks)]
        assert [index for index, _ in completed] != list(range(12))
        long_inputs = [
            audio.input_tokens for index, audio in completed if index % 2 == 0
        ]
        short_inputs = [audio.input_tokens for index, audio in completed if index % 2]
        assert min(long_inputs) > max(short_inputs)
        await wait_for_pipeline_idle(client, speech_server)
        replay = await speech_audio(client, speech_server, speech_request(0))
        assert replay.pcm == anchor.pcm
        assert replay.output_tokens == anchor.output_tokens
        await wait_for_pipeline_idle(client, speech_server)
