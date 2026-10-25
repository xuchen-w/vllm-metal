# SPDX-License-Identifier: Apache-2.0
"""Validate reference-conditioned Qwen3-TTS HTTP streaming and save its audio.

Use an already-running Base checkpoint with the greedy MPS deployment profile
(including repetition_penalty=1.5 and eager predictor KV reuse). The speech API
does not expose a repetition-penalty override through extra_params.

The JSON/JSONL manifest contains sample_id, text, language, reference and
ref_text. Reference files are read/encoded before timing. Client concurrency
limits outstanding HTTP requests; it is not a fixed model decode batch size.
WAV writing and waveform statistics happen after each measurement window.

    python tools/benchmark/qwen3_tts_benchmark.py --manifest samples.json \\
        --base-url http://127.0.0.1:8091 --concurrency 4 --output-dir result-c4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def prepare_samples(path, max_tokens, model):
    text = path.read_text()
    rows = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    if not isinstance(rows, list) or not rows:
        raise ValueError("The manifest must contain at least one sample")
    references, samples = {}, []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("sample_id", "text", "language", "reference", "ref_text")
        ):
            raise ValueError(
                f"Invalid sample {index}: required fields must be nonempty strings"
            )
        reference = Path(row["reference"])
        if not reference.is_absolute():
            reference = path.parent / reference
        reference = reference.resolve(strict=True)
        if reference not in references:
            data = reference.read_bytes()
            references[reference] = (
                "data:audio/wav;base64," + base64.b64encode(data).decode(),
                hashlib.sha256(data).hexdigest(),
            )
        audio, digest = references[reference]
        if row.get("reference_sha256", digest) != digest:
            raise ValueError(f"Reference SHA256 mismatch for sample {index}")
        request = {
            "input": row["text"],
            "task_type": "Base",
            "language": "Auto",
            "ref_audio": audio,
            "ref_text": row["ref_text"],
            "max_new_tokens": max_tokens,
            "response_format": "pcm",
            "stream": True,
            "extra_params": {"temperature": 0.0, "top_k": -1, "top_p": 1.0},
        }
        if model:
            request["model"] = model
        samples.append(
            (
                {
                    **row,
                    "index": index,
                    "reference": str(reference),
                    "reference_sha256": digest,
                },
                request,
            )
        )
    return samples


async def sse_data(response):
    lines = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
        elif not line and lines:
            yield "\n".join(lines)
            lines.clear()
    if lines:
        yield "\n".join(lines)


async def request_audio(client, endpoint, sample, sample_rate, max_tokens, timeout):
    metadata, request = sample
    record = {**metadata, "success": False, "usage": None, "first_audio_s": None}
    pcm = bytearray()
    events, chunk_samples = [], []
    started = time.perf_counter()
    try:
        async with (
            asyncio.timeout(timeout),
            client.stream("POST", endpoint, json=request) as response,
        ):
            record["http_status"] = response.status_code
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                raise ValueError(f"HTTP {response.status_code}: {body[:2000]}")
            if not response.headers.get("content-type", "").startswith(
                "text/event-stream"
            ):
                raise ValueError("Expected text/event-stream")
            async for data in sse_data(response):
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise ValueError("Speech event must be a JSON object")
                kind = event.get("type")
                if record["usage"] is not None:
                    raise ValueError("Received an event after speech.audio.done")
                events.append(kind)
                if kind == "speech.audio.error":
                    raise ValueError(f"Speech error: {event.get('error')}")
                if kind == "speech.audio.delta":
                    if (
                        event.get("sample_rate") != sample_rate
                        or event.get("channels") != 1
                    ):
                        raise ValueError(
                            f"Unexpected audio dimensions: {event.get('sample_rate')}/{event.get('channels')}"
                        )
                    if not isinstance(event.get("audio"), str):
                        raise ValueError("Audio delta must contain base64 text")
                    chunk = base64.b64decode(event["audio"], validate=True)
                    if len(chunk) % 2:
                        raise ValueError("PCM16 delta contains an incomplete sample")
                    if chunk and record["first_audio_s"] is None:
                        record["first_audio_s"] = time.perf_counter() - started
                    pcm.extend(chunk)
                    chunk_samples.append(len(chunk) // 2)
                elif kind == "speech.audio.done":
                    usage = event.get("usage")
                    if not isinstance(usage, dict):
                        raise ValueError("Done event must contain token usage")
                    record["usage"] = usage
                    record["done"] = event
                else:
                    raise ValueError(f"Unexpected speech event: {kind!r}")
        output_tokens = (record["usage"] or {}).get("output_tokens")
        if (
            not pcm
            or events[-1:] != ["speech.audio.done"]
            or events.count("speech.audio.done") != 1
        ):
            raise ValueError("Stream is empty or missing its single final done event")
        if type(output_tokens) is not int or not 0 < output_tokens < max_tokens:
            raise ValueError(
                f"Invalid output token count or reached token cap: {output_tokens}"
            )
        record["success"] = True
    except (httpx.HTTPError, ValueError, TimeoutError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record.update(
        elapsed_s=time.perf_counter() - started,
        events=events,
        chunk_samples=chunk_samples,
    )
    return record, bytes(pcm)


async def run_requests(
    client, endpoint, samples, concurrency, sample_rate, max_tokens, timeout
):
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(sample):
        queued = time.perf_counter()
        async with semaphore:
            queue_s = time.perf_counter() - queued
            result = await request_audio(
                client, endpoint, sample, sample_rate, max_tokens, timeout
            )
            result[0]["client_queue_s"] = queue_s
            return result

    started = time.perf_counter()
    results = await asyncio.gather(*(limited(sample) for sample in samples))
    return results, time.perf_counter() - started


def save_audio(results, directory, sample_rate):
    records = []
    for output_index, (record, pcm) in enumerate(results):
        if pcm:
            path = directory / f"{output_index:04d}.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(pcm)
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            tail = audio[-max(1, sample_rate // 10) :]
            record.update(
                path=str(path.resolve()),
                sample_rate=sample_rate,
                channels=1,
                duration_s=len(audio) / sample_rate,
                pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                rms=float(np.sqrt(np.mean(audio**2))),
                peak=float(np.abs(audio).max()),
                full_scale_fraction=float(np.mean(np.abs(audio) >= 32767 / 32768)),
                tail_100ms_rms=float(np.sqrt(np.mean(tail**2))),
                last_chunk_samples=record["chunk_samples"][-1],
            )
            if not np.count_nonzero(audio):
                record.update(success=False, error="Audio is entirely silent")
        records.append(record)
    return records


def summarize(records, elapsed):
    successful = [row for row in records if row["success"]]
    total_audio = sum(row["duration_s"] for row in successful)
    latencies = [row["first_audio_s"] for row in successful]
    return {
        "requested": len(records),
        "completed": len(successful),
        "failed": len(records) - len(successful),
        "wall_time_s": elapsed,
        "audio_s": total_audio,
        "audio_seconds_per_second": total_audio / elapsed,
        "total_output_tokens": sum(row["usage"]["output_tokens"] for row in successful),
        "mean_first_audio_ms": statistics.mean(latencies) * 1000 if latencies else None,
        "p50_first_audio_ms": float(np.percentile(latencies, 50) * 1000)
        if latencies
        else None,
        "p95_first_audio_ms": float(np.percentile(latencies, 95) * 1000)
        if latencies
        else None,
    }


async def benchmark(args):
    samples = prepare_samples(args.manifest, args.max_new_tokens, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir()  # Reject reusing a previous run's output directory.
    versions = {}
    for name in ("httpx", "numpy", "torch", "vllm", "vllm-metal", "vllm-omni"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    metadata = {
        "command": sys.argv,
        "client_platform": platform.platform(),
        "client_python": sys.version,
        "client_package_versions": versions,
        "client_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "client_checkout_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip(),
        "server": json.loads(args.server_metadata.read_text())
        if args.server_metadata
        else None,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "concurrency": args.concurrency,
        "warmups": args.warmups,
        "max_new_tokens": args.max_new_tokens,
        "request_timeout_s": args.timeout,
        "request_language": "Auto",
        "sampling_request": {"temperature": 0.0, "top_k": -1, "top_p": 1.0},
        "required_server_profile": "greedy code predictor, eager KV reuse, repetition_penalty=1.5",
        "timing": "HTTP request through final event; TTFA is first nonempty audio, excludes client semaphore wait. Reference preparation, WAV writes and audio statistics excluded; JSON/base64 processing included.",
    }
    write_json(args.output_dir / "run.json", metadata)
    endpoint = args.base_url.rstrip("/") + "/v1/audio/speech"
    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        if args.warmups:
            warmups = [samples[i % len(samples)] for i in range(args.warmups)]
            warmed, _ = await run_requests(
                client,
                endpoint,
                warmups,
                args.concurrency,
                args.sample_rate,
                args.max_new_tokens,
                args.timeout,
            )
            warmup_dir = args.output_dir / "warmup_audio"
            warmup_dir.mkdir()
            warmup_records = save_audio(warmed, warmup_dir, args.sample_rate)
            write_json(args.output_dir / "warmups.json", warmup_records)
            if any(not row["success"] for row in warmup_records):
                raise RuntimeError("Warmup failed; see warmups.json")
        results, elapsed = await run_requests(
            client,
            endpoint,
            samples,
            args.concurrency,
            args.sample_rate,
            args.max_new_tokens,
            args.timeout,
        )
    records = save_audio(results, audio_dir, args.sample_rate)
    write_json(args.output_dir / "manifest.json", records)
    summary = summarize(records, elapsed)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary["failed"] != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--model", help="Optional served model name")
    parser.add_argument(
        "--server-metadata",
        type=Path,
        help="JSON describing the server's checkpoints, revisions and launch configuration",
    )
    args = parser.parse_args()
    if (
        min(args.concurrency, args.max_new_tokens, args.sample_rate) <= 0
        or args.warmups < 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error(
            "concurrency, token limit, sample rate and timeout must be positive; warmups must be nonnegative"
        )
    raise SystemExit(asyncio.run(benchmark(args)))


if __name__ == "__main__":
    main()
