# Qwen3-TTS on Apple Silicon

Generate speech with the optional vLLM-Omni integration.

> **Development preview:** The optional dependency installs a compatible Omni preview.

After the [development setup](CONTRIBUTING.md#development-setup), install:

```bash
VLLM_OMNI_TARGET_DEVICE=cpu uv pip install -e '.[omni]' 'torch==2.13.0'
```

Start the server:

```bash
VLLM_METAL_BACKEND=mps vllm-omni serve \
  Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni \
  --host 127.0.0.1 --port 8091
```

Save generated speech to a WAV file:

```bash
curl http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, have a wonderful day.","voice":"Ryan","language":"English","response_format":"wav"}' \
  --output speech.wav
```

For streaming audio, add `"stream": true`, `"stream_format": "audio"`, and
`"response_format": "pcm"` to the request. See the
[Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
for request formats and examples.

This preview uses greedy decoding for both the talker and code predictor.
Random sampling, quantized models, and distributed serving are not supported.

The compatible Omni snapshot constructs codec parameters at the configured
compute dtype while preserving explicitly FP32 rotary-frequency buffers.
Casting the entire decoder to BF16 after loading would also round those
frequencies, even though attention subsequently upcasts them for computation.

## Validate a checkout

Run the optional real HTTP suite with a local CustomVoice checkpoint:

```bash
VLLM_METAL_BACKEND=mps PYTORCH_ENABLE_MPS_FALLBACK=0 \
QWEN3_TTS_TEST_MODEL=/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  pytest tests/test_mps_omni_serving.py -q
```

It covers concurrent streaming, final audio, cancellation while another
request remains active, queue drainage and request reuse. Serial greedy
replays are compared; concurrent output is not assumed to be bit-identical.

For a Base checkpoint, the standalone streaming validation client accepts a
JSON array or JSONL manifest. Each row supplies `sample_id`, `text`, `language`,
`reference` (a WAV path, relative to the manifest or absolute), and `ref_text`.
An optional `reference_sha256` verifies the reference file. Start the Base
server before running:

```bash
python tools/benchmark/qwen3_tts_benchmark.py \
  --manifest samples.json --base-url http://127.0.0.1:8091 \
  --concurrency 4 --warmups 8 --output-dir result-c4 \
  --server-metadata server.json
```

`server.json` records the checkpoint revisions, package versions and launch
configuration. Use a fresh output directory for each run. The client retains
every WAV, terminal event, token usage, failure and timing sample for separate
ASR or audio analysis. Missing completion events and outputs reaching the token
cap fail validation. First-audio latency starts when the request acquires its
client concurrency slot; `client_queue_s` is recorded separately. Reference
preparation, WAV writes and waveform statistics are outside the timed window.
The server's MPS profile supplies repetition penalty 1.5 and eager predictor
KV reuse; the speech API does not expose a repetition-penalty override through
`extra_params`.

## Compare model components

`tools/qwen3_tts_component_parity.py` compares the residual-code predictor and
codec against an explicit [official Qwen3-TTS checkout](https://github.com/QwenLM/Qwen3-TTS).
Use that checkout's compatible environment for the `reference` command and
the Metal/Omni environment for `omni`; the tool downloads no source code.
Both commands require the same checkpoint and optional codec-prefix settings.
See `--help` and the examples in the tool's module docstring.

The tool records checkpoint/input hashes, real loader dtypes, predictor
first-divergence results and complete codec outputs. `--codec-prefix-frames 64`
exercises reference-conditioned streaming across the 72-frame cache window.
Full-codec parity and FP32 streaming checks are separate from BF16 cached
streaming drift against prefix recomputation. `--strict-stream` additionally
enforces the reported one-epsilon waveform diagnostic; that bound is not a
promised invariant for different BF16 computation graphs.

This component coverage does not include full Talker prompting or VoiceDesign.
Keep those limits and the actual ASR model/normalization alongside any reported
end-to-end quality results.
