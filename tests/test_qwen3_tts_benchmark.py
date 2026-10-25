# SPDX-License-Identifier: Apache-2.0

import base64
import json
import struct
import wave

import httpx
import pytest

from tools.benchmark.qwen3_tts_benchmark import request_audio, save_audio, summarize

PCM = struct.pack("<4h", 100, -200, 300, -400)
DELTA = {
    "type": "speech.audio.delta",
    "sample_rate": 24000,
    "channels": 1,
    "audio": base64.b64encode(PCM).decode(),
    "id": "语音",
}
DONE = {"type": "speech.audio.done", "usage": {"output_tokens": 12}}


def _sse(event):
    return ("data: " + json.dumps(event) + "\n\n").encode()


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


async def _request(chunks):
    def respond(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_Chunks(chunks),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        return await request_audio(
            client, "http://test/v1/audio/speech", ({"index": 0}, {}), 24000, 1024, 1
        )


async def test_speech_decodes_multiline_sse_across_transport_and_utf8_boundaries():
    data = json.dumps(DELTA, ensure_ascii=False, indent=2)
    message = (
        ": keepalive\r\n"
        + "\r\n".join("data: " + line for line in data.splitlines())
        + "\r\n\r\n"
    ).encode() + _sse(DONE)
    split = message.index("语".encode()) + 1
    record, pcm = await _request([message[:5], message[5:split], message[split:]])

    assert record["success"]
    assert pcm == PCM
    assert record["usage"]["output_tokens"] == 12
    assert record["chunk_samples"] == [4]
    assert 0 <= record["first_audio_s"] <= record["elapsed_s"]


@pytest.mark.parametrize(
    ("following", "error"),
    [
        ([], "missing its single final done"),
        ([DONE, DONE], "after speech.audio.done"),
        ([{"type": "speech.audio.error", "error": "failed"}], "Speech error"),
        ([{**DONE, "usage": {"output_tokens": 1024}}], "token cap"),
        ([{**DELTA, "sample_rate": 16000}], "audio dimensions"),
        ([{**DELTA, "audio": "!invalid!"}], "base64"),
        ([{**DELTA, "audio": base64.b64encode(b"x").decode()}], "incomplete sample"),
    ],
    ids=["truncated", "duplicate-done", "error", "cap", "rate", "base64", "pcm"],
)
async def test_invalid_stream_is_failed_with_partial_audio_preserved(following, error):
    record, pcm = await _request([_sse(event) for event in [DELTA, *following]])

    assert not record["success"]
    assert error in record["error"]
    assert pcm == PCM


async def test_saved_truncated_audio_cannot_count_as_successful_throughput(tmp_path):
    result = await _request([_sse(DELTA)])
    records = save_audio([result], tmp_path, 24000)
    summary = summarize(records, 1.0)

    with wave.open(records[0]["path"], "rb") as audio:
        assert audio.getframerate() == 24000
        assert audio.readframes(audio.getnframes()) == PCM
    assert summary["completed"] == 0
    assert summary["failed"] == 1
    assert summary["audio_seconds_per_second"] == 0
    assert summary["mean_first_audio_ms"] is None
