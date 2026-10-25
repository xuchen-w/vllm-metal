# SPDX-License-Identifier: Apache-2.0
"""Compare real Qwen3-TTS predictor/codec weights with the official implementation.

Run ``reference`` in the official Qwen package's environment, then ``omni`` in
the Metal/Omni environment. Both use MPS with optional CPU fallback disabled.
The reference checkout is supplied explicitly; this tool downloads no code.

The predictor receives identical synthetic hidden states, checkpoint-derived
CB0 embeddings and IDs. This isolates residual-code prediction, including
frame-local KV reuse; it does not test Talker prompting or text generation.
The codec receives identical legal code IDs, including a short final chunk.
``--codec-prefix-frames`` additionally exercises the ICL reference-prefix cache.
FP32 streaming is checked against the existing decoder tolerance. BF16 cached
streaming reports numerical drift separately: recomputation and caching use
different rounding graphs. ``--strict-stream`` also enforces a one-epsilon
whole-waveform NRMSE diagnostic; it is not a promised BF16 model invariant.

Official reference: https://github.com/QwenLM/Qwen3-TTS

    PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/qwen3_tts_component_parity.py \\
        reference --model /path/to/checkpoint --reference-source /path/to/Qwen3-TTS \\
        --output reference-results
    VLLM_METAL_BACKEND=mps PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. python \\
        tools/qwen3_tts_component_parity.py omni --model /path/to/checkpoint \\
        --reference reference-results --output omni-results
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from contextlib import contextmanager
from functools import partial
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

if __package__:
    from .parity_utils import parity_report
else:
    from parity_utils import parity_report

CODEC_CHUNK_FRAMES = (4, 25, 8)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@contextmanager
def parameter_dtype(dtype):
    """Match loader construction while retaining explicit FP32 RoPE buffers."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def weights(directory, prefix, names=None):
    result = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as archive:
            for name in archive.keys():
                if name.startswith(prefix) and (
                    names is None or name[len(prefix) :] in names
                ):
                    result[name[len(prefix) :]] = archive.get_tensor(name)
    if not result:
        raise ValueError(f"No weights for {prefix!r} in {directory}")
    return result


def provenance(model, reference_source=None):
    versions = {}
    for name in ("torch", "transformers", "vllm", "vllm-metal", "vllm-omni"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "optional_cpu_fallback": False,
        "tool_sha256": digest(Path(__file__)),
        "checkpoint_sha256": {
            str(path.relative_to(model)): digest(path)
            for directory in (model, model / "speech_tokenizer")
            for path in sorted(directory.iterdir())
            if path.suffix == ".safetensors" or path.name == "config.json"
        },
    }
    if reference_source is not None:
        result["reference_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=reference_source, text=True
        ).strip()
        result["reference_source_sha256"] = {
            str(path.relative_to(reference_source)): digest(path)
            for path in (
                reference_source / "qwen_tts/core/models/modeling_qwen3_tts.py",
                reference_source
                / "qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py",
            )
        }
    else:
        import vllm_omni

        import vllm_metal

        result["source_revisions"] = {}
        result["source_sha256"] = {}
        for package, files in (
            (
                vllm_metal,
                (
                    "pytorch_backend/models/qwen3_code_predictor.py",
                    "pytorch_backend/normalization.py",
                ),
            ),
            (
                vllm_omni,
                (
                    "model_executor/models/common/qwen3_code_predictor.py",
                    "model_executor/models/qwen3_tts/qwen3_tts_code2wav.py",
                    "model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py",
                ),
            ),
        ):
            directory = Path(package.__file__).parent
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=directory,
                text=True,
                capture_output=True,
                check=False,
            )
            result["source_revisions"][package.__name__] = (
                revision.stdout.strip() if revision.returncode == 0 else None
            )
            result["source_sha256"].update(
                {
                    f"{package.__name__}/{name}": digest(directory / name)
                    for name in files
                }
            )
    return result


def predictor_config(model, implementation):
    config = json.loads((model / "config.json").read_text())["talker_config"]
    if implementation == "reference":
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    else:
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSTalkerConfig,
        )
    talker = Qwen3TTSTalkerConfig(**config)
    talker.code_predictor_config._attn_implementation = "sdpa"
    return talker


def build_predictor(args, talker):
    state = weights(args.model, "talker.code_predictor.")
    if args.implementation == "reference":
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        )

        with parameter_dtype(torch.bfloat16):
            model = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
                talker.code_predictor_config, talker
            )
        model.load_state_dict(state, strict=True)
    else:
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm import (
            Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM,
        )

        from vllm_metal.pytorch_backend.models.qwen3_code_predictor import install

        install()
        # The isolated component uses these configuration fields only. Its
        # weights, operators, cache and greedy decoding are the real modules.
        config = SimpleNamespace(
            model_config=None,
            additional_config={"code_predictor_kv_cache": "eager"},
            scheduler_config=SimpleNamespace(max_num_seqs=3),
        )
        with parameter_dtype(torch.bfloat16):
            model = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
                vllm_config=config,
                config=talker.code_predictor_config,
                talker_config=talker,
            )
        loaded = model.load_weights(state.items())
        missing = set(dict(model.named_parameters())) - loaded
        if missing:
            raise ValueError(f"Unloaded predictor parameters: {sorted(missing)}")
        assert type(model.model.norm).__name__ == "MPSPredictorRMSNorm"
    return model.to("mps").eval()


def make_inputs(model, talker, seed, prefix_frames):
    generator = torch.Generator().manual_seed(seed)
    state = weights(model, "talker.model.", {"codec_embedding.weight", "norm.weight"})
    embedding, scale = state["codec_embedding.weight"], state["norm.weight"]
    data = {}
    for index, batch in enumerate((1, 3)):
        codes = torch.randint(
            talker.code_predictor_config.vocab_size, (batch, 1), generator=generator
        )
        hidden = (
            torch.randn(batch, 1, talker.hidden_size, generator=generator).to(
                torch.bfloat16
            )
            * scale
        )
        data[f"{index}.code"] = codes
        data[f"{index}.embedding"] = embedding[codes].contiguous()
        data[f"{index}.hidden"] = hidden
    for name in ("code", "embedding", "hidden"):
        data[f"2.{name}"] = data[f"0.{name}"].clone()
        data[f"3.{name}"] = data[f"1.{name}"][[2, 0, 1]].contiguous()
    codec_config = json.loads((model / "speech_tokenizer/config.json").read_text())
    decoder = codec_config["decoder_config"]
    for batch in (1, 3):
        data[f"codec.{batch}"] = torch.randint(
            decoder["codebook_size"],
            (batch, decoder["num_quantizers"], prefix_frames + sum(CODEC_CHUNK_FRAMES)),
            generator=generator,
        )
    return data


def record_logits(_module, _inputs, output, *, step, traces):
    # Preserve pre-sampling values without synchronizing every head to CPU.
    traces[step] = output.detach().clone()


def run_predictor(model, inputs, implementation):
    traces = {}
    handles = []
    if implementation == "omni":
        handles = [
            head.register_forward_hook(partial(record_logits, step=i, traces=traces))
            for i, head in enumerate(model.lm_head)
        ]
    result = {}
    try:
        for index in range(4):
            code, embedding, hidden = (
                inputs[f"{index}.{name}"].to("mps")
                for name in ("code", "embedding", "hidden")
            )
            traces.clear()
            if implementation == "reference":
                logits = []
                output = model(
                    inputs_embeds=torch.cat((hidden, embedding), dim=1),
                    use_cache=True,
                    return_dict=True,
                )
                selected = [code.reshape(-1)]
                for step in range(model.config.num_code_groups - 1):
                    scores = output.logits[:, -1]
                    token = scores.argmax(-1)
                    logits.append(scores.detach().clone())
                    selected.append(token)
                    if step + 1 < model.config.num_code_groups - 1:
                        output = model(
                            input_ids=token[:, None],
                            past_key_values=output.past_key_values,
                            generation_steps=output.generation_steps,
                            use_cache=True,
                            return_dict=True,
                        )
                codes = torch.stack(selected, dim=1)
            else:
                if model._kv_buf is not None:
                    model._kv_buf.fill_(float("nan"))
                codes = model(code, embedding, hidden, do_sample=False)
                assert model._eager_kv_cache and model._kv_cache_enabled
                assert len(traces) == model.config.num_code_groups - 1
                logits = [traces[i] for i in range(len(traces))]
            result[f"{index}.codes"] = codes.cpu().contiguous()
            result[f"{index}.logits"] = (
                torch.stack(logits, dim=1).float().cpu().contiguous()
            )
        torch.testing.assert_close(result["0.codes"], result["2.codes"], atol=0, rtol=0)
        torch.testing.assert_close(
            result["1.codes"][[2, 0, 1]], result["3.codes"], atol=0, rtol=0
        )
    finally:
        for handle in handles:
            handle.remove()
    return result


def run_codec(args, inputs):
    dtype = getattr(torch, args.codec_dtype)
    if args.implementation == "reference":
        from qwen_tts import Qwen3TTSTokenizer

        wrapper = Qwen3TTSTokenizer.from_pretrained(
            str(args.model / "speech_tokenizer"),
            device_map="mps",
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model = wrapper.model.decoder
    else:
        from vllm.config import LoadConfig
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import (
            Qwen3TTSCode2Wav,
        )

        config = SimpleNamespace(
            model_config=SimpleNamespace(
                model=str(args.model),
                revision=None,
                dtype=dtype,
                async_chunk=True,
                enforce_eager=True,
                stage_connector_config={"extra": {}},
            ),
            device_config=SimpleNamespace(device=torch.device("mps")),
            load_config=LoadConfig(),
        )
        wrapper = Qwen3TTSCode2Wav(vllm_config=config)
        loaded = wrapper.load_weights([])
        missing = set(dict(wrapper.named_parameters())) - loaded
        if missing:
            raise ValueError(f"Unloaded codec parameters: {sorted(missing)}")
        model = wrapper.decoder
    model.eval()
    result = {}
    for batch in (1, 3):
        codes = inputs[f"codec.{batch}"].to("mps")
        suffix_samples = sum(CODEC_CHUNK_FRAMES) * model.total_upsample
        result[f"codec.{batch}.full"] = (
            model(codes)[..., -suffix_samples:].float().cpu().contiguous()
        )
        if args.implementation == "reference":
            chunks, end = [], args.codec_prefix_frames
            for length in CODEC_CHUNK_FRAMES:
                end += length
                if end == codes.shape[-1]:
                    chunk = result[f"codec.{batch}.full"][
                        ..., -length * model.total_upsample :
                    ]
                else:
                    chunk = (
                        model(codes[..., :end])[..., -length * model.total_upsample :]
                        .float()
                        .cpu()
                    )
                chunks.append(chunk)
            result[f"codec.{batch}.stream"] = torch.cat(chunks, -1).contiguous()
        else:
            cache = {"prefix_frames": args.codec_prefix_frames}
            chunks, start = [], 0
            for index, length in enumerate(CODEC_CHUNK_FRAMES):
                end = start + length + (args.codec_prefix_frames if index == 0 else 0)
                chunk = model(codes[..., start:end], cache)
                if index == 0:
                    chunk = chunk[..., -length * model.total_upsample :]
                assert chunk.shape[-1] == length * model.total_upsample
                chunks.append(chunk.float().cpu())
                start = end
            result[f"codec.{batch}.stream"] = torch.cat(chunks, -1).contiguous()
    return result, {
        "loader": "official from_pretrained"
        if args.implementation == "reference"
        else "Omni Code2Wav.load_weights",
        "parameter_dtype": str(next(model.parameters()).dtype),
        "attention_implementation": model.pre_transformer.config._attn_implementation,
        "rotary_inv_freq_dtype": str(model.pre_transformer.rotary_emb.inv_freq.dtype),
        "snake_caches": args.implementation == "omni",
    }


def rows(tensors):
    result = []
    for index in range(4):
        for row, (codes, logits) in enumerate(
            zip(tensors[f"{index}.codes"], tensors[f"{index}.logits"], strict=True)
        ):
            candidates = []
            for scores in logits.log_softmax(-1):
                ids = torch.argsort(scores, descending=True, stable=True)[:5]
                candidates.append(
                    [
                        {
                            "id": int(i),
                            "text": str(int(i)),
                            "rank": rank,
                            "logprob": float(scores[i]),
                        }
                        for rank, i in enumerate(ids, 1)
                    ]
                )
            result.append(
                {
                    "prompt": f"frame {index}, row {row}",
                    "input_ids": [int(codes[0])],
                    "tokens": codes[1:].tolist(),
                    "text": str(codes[1:].tolist()),
                    "top_logprobs": candidates,
                }
            )
    return result


def compare(reference, candidate, codec_dtype, prefix_frames, strict_stream=False):
    report = parity_report(
        rows(reference),
        rows(candidate),
        max_tokens=reference["0.codes"].shape[1] - 1,
        top_k=5,
        labels=("official Qwen MPS", "Omni Metal MPS"),
    )
    result_index = 0
    for index in range(4):
        ref_logits, got_logits = (
            reference[f"{index}.logits"],
            candidate[f"{index}.logits"],
        )
        for row in range(ref_logits.shape[0]):
            item = report["results"][result_index]
            stop = item["first_divergence"]
            stop = ref_logits.shape[1] if stop is None else stop + 1
            error = (ref_logits[row, :stop] - got_logits[row, :stop]).abs()
            item["max_abs_logit_error_common_prefix"] = float(error.max())
            result_index += 1
    codec = {}
    for batch in (1, 3):
        for mode in ("full", "stream", "stream-vs-full"):
            expected = reference[
                f"codec.{batch}.{'full' if mode == 'stream-vs-full' else mode}"
            ]
            actual = candidate[
                f"codec.{batch}.{'stream' if mode == 'stream-vs-full' else mode}"
            ]
            assert actual.shape == expected.shape
            for row in range(batch):
                error = actual[row] - expected[row]
                reference_rms = float(expected[row].square().mean().sqrt())
                if reference_rms == 0:
                    raise ValueError("A silent reference cannot establish codec parity")
                nrmse = float(error.square().mean().sqrt()) / reference_rms
                codec[f"batch-{batch}-{mode}-row-{row}"] = {
                    "samples": actual.shape[-1],
                    "max_abs_error": float(error.abs().max()),
                    "nrmse": nrmse,
                    "snr_db": -20 * math.log10(nrmse) if nrmse else None,
                    "bit_exact": nrmse == 0,
                    "within_nrmse_tolerance": nrmse
                    <= (
                        torch.finfo(torch.bfloat16).eps
                        if codec_dtype == "bfloat16"
                        else 1e-4
                    ),
                    "within_fp32_tolerance": bool(
                        torch.allclose(actual[row], expected[row], atol=1e-5, rtol=1e-4)
                    )
                    if codec_dtype == "float32"
                    else None,
                }
    report["codec"] = codec
    full = [
        value
        for key, value in codec.items()
        if "-full-" in key and "stream-vs-full" not in key
    ]
    stream = [
        value
        for key, value in codec.items()
        if "-stream-" in key and "stream-vs-full" not in key
    ]
    report["validation_checks"] = {
        "predictor": report["passed"],
        "full_codec": all(v["within_nrmse_tolerance"] for v in full),
        "stream_shape_and_finiteness": True,
        "fp32_stream": all(v["within_fp32_tolerance"] for v in stream)
        if codec_dtype == "float32"
        else None,
    }
    report["stream_nrmse_diagnostic"] = {
        "within_threshold": all(v["within_nrmse_tolerance"] for v in stream),
        "max_nrmse": max(v["nrmse"] for v in stream),
        "enforced": strict_stream,
    }
    report["passed"] = all(
        value is not False for value in report["validation_checks"].values()
    )
    if strict_stream:
        report["passed"] &= report["stream_nrmse_diagnostic"]["within_threshold"]
    report["status"] = (
        "FAIL"
        if not report["passed"]
        else "PASS_WITH_STREAM_DRIFT_DIAGNOSTIC"
        if not report["stream_nrmse_diagnostic"]["within_threshold"]
        else "PASS"
    )
    report["codec_reference_protocol"] = {
        "reference_prefix_frames": prefix_frames,
        "cumulative_prefix_frames": [
            prefix_frames + n for n in accumulate(CODEC_CHUNK_FRAMES)
        ],
        "emitted_frames": list(CODEC_CHUNK_FRAMES),
        "note": "Official decoder recomputes matching prefixes; stream-vs-full is a separate chunk-shape diagnostic.",
    }
    report["codec_nrmse_tolerance"] = (
        torch.finfo(torch.bfloat16).eps if codec_dtype == "bfloat16" else 1e-4
    )
    report["scope"] = (
        "Synthetic matching-input predictor and codec components; not full Talker or text-to-audio parity."
    )
    return report


def main():
    import os

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("implementation", choices=("reference", "omni"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument(
        "--codec-dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--codec-prefix-frames", type=int, default=0)
    parser.add_argument("--strict-stream", action="store_true")
    args = parser.parse_args()
    if args.codec_prefix_frames < 0:
        parser.error("codec-prefix-frames must be nonnegative")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("Set PYTORCH_ENABLE_MPS_FALLBACK=0 before starting Python")
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if args.implementation == "reference":
        if args.reference_source is None:
            parser.error("reference requires --reference-source")
        sys.path.insert(0, str(args.reference_source.resolve()))
    elif args.reference is None:
        parser.error("omni requires --reference")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = provenance(args.model, args.reference_source)
    talker = predictor_config(args.model, args.implementation)
    if args.implementation == "reference":
        inputs = make_inputs(args.model, talker, args.seed, args.codec_prefix_frames)
        save_file(inputs, args.output / "inputs.safetensors")
    else:
        previous = json.loads((args.reference / "metadata.json").read_text())
        if previous["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
            raise ValueError("Reference and candidate checkpoints differ")
        if digest(args.reference / "inputs.safetensors") != previous["input_sha256"]:
            raise ValueError("Reference inputs changed after capture")
        if previous["codec_dtype"] != args.codec_dtype:
            raise ValueError("Reference and candidate codec dtypes differ")
        if previous.get("codec_prefix_frames", 0) != args.codec_prefix_frames:
            raise ValueError("Reference and candidate codec prefixes differ")
        inputs = load_file(args.reference / "inputs.safetensors")
    input_path = (
        args.output if args.implementation == "reference" else args.reference
    ) / "inputs.safetensors"
    metadata["input_sha256"] = digest(input_path)
    metadata["predictor_dtype"] = "bfloat16"
    metadata["codec_dtype"] = args.codec_dtype
    metadata["predictor_batches"] = [1, 3, 1, 3]
    metadata["codec_chunk_frames"] = list(CODEC_CHUNK_FRAMES)
    metadata["codec_prefix_frames"] = args.codec_prefix_frames
    with torch.inference_mode():
        predictor = build_predictor(args, talker)
        outputs = run_predictor(predictor, inputs, args.implementation)
        metadata["predictor_operators"] = {
            "norm": type(predictor.model.norm).__name__,
            "rotary": type(predictor.model.rotary_emb).__name__,
            "rotary_inv_freq_dtype": str(predictor.model.rotary_emb.inv_freq.dtype),
        }
        del predictor
        gc.collect()
        torch.mps.empty_cache()
        codec_outputs, codec_settings = run_codec(args, inputs)
        outputs.update(codec_outputs)
        metadata["codec_postload"] = codec_settings
    save_file(outputs, args.output / "outputs.safetensors")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name, tensor in outputs.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite component output: {name}")
    if args.implementation == "omni":
        report = compare(
            load_file(args.reference / "outputs.safetensors"),
            outputs,
            args.codec_dtype,
            args.codec_prefix_frames,
            args.strict_stream,
        )
        (args.output / "comparison.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(f"Component validation: {report['status']}")
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
