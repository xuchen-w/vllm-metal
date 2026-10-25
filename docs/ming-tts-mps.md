# Ming dense TTS on Apple Silicon

Serve the unquantized `inclusionAI/Ming-omni-tts-0.5B` checkpoint through the
optional vLLM-Omni integration. The MPS deployment uses Omni's Ming model,
flow head and audio codec, with streaming and concurrent requests.

> **Development preview:** Use the compatible Omni snapshot installed by the
> `omni` extra. The upstream Omni integration dependencies are still in review.

After the [development setup](CONTRIBUTING.md#development-setup), install:

```bash
VLLM_OMNI_TARGET_DEVICE=cpu uv pip install -e '.[omni]' 'torch==2.13.0'
```

Select the installed Ming deployment explicitly, because its checkpoint uses
the generic `dense` model type:

```bash
MING_DEPLOY=$(python -c 'from pathlib import Path; import vllm_omni; print(Path(vllm_omni.__file__).parent / "deploy/ming_tts.yaml")')
VLLM_METAL_BACKEND=mps PYTORCH_ENABLE_MPS_FALLBACK=0 \
  vllm-omni serve inclusionAI/Ming-omni-tts-0.5B --omni \
  --deploy-config "$MING_DEPLOY" --host 127.0.0.1 --port 8091
```

A local checkpoint directory can replace the model ID. The MPS profile selects
the v2 runners, eager execution, up to eight requests per stage, and a 512 MiB
AR KV-cache budget. The codec stage uses the existing shared-memory transport.

Generate a WAV:

```bash
curl http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好，欢迎使用语音合成。","response_format":"wav"}' \
  --output ming.wav
```

Audio is mono, 44.1 kHz. For streaming events, add `"stream": true` and
`"response_format": "pcm"`. Adding `"stream_format": "audio"` selects raw PCM
streaming. Reference conditioning uses the existing `ref_audio` and `ref_text`
fields; see the [Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/).

Validation covers dense 0.5B at client concurrency 1, 4 and 8, non-streaming
WAV, stream cancellation followed by new requests, and final audio chunks.
Other Ming variants have not been validated by this integration. Flow sampling
remains stochastic: an API seed does not guarantee identical audio across
independent runs or different batch schedules.
