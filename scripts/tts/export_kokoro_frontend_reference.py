#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export deterministic Kokoro v1.0 frontend and model reference fixtures.

The script is developer-only. Without ``--model-root`` it exports the exact
KPipeline/Misaki frontend observations. With ``--model-root`` it additionally
runs the pinned PyTorch checkpoint and writes every major inference boundary as
a little-endian raw tensor plus a JSON manifest. It is never installed and is
not a dependency of the native runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MISAKI_COMMIT = "fba1236595f2d2bf21d414ba6e57d25256afada3"
MISAKI_VERSION = "0.9.4"
KOKORO_COMMIT = "dfb907a02bba8152ca444717ca5d78747ccb4bec"
KOKORO_HF_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
KOKORO_MODEL_SHA256 = "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
KOKORO_MODEL_NAME = "kokoro-v1_0.pth"
_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _uniform_open(seed: int, counter: int) -> float:
    bits = _splitmix64((seed & _UINT64_MASK) ^ _splitmix64(counter))
    return ((bits >> 11) + 0.5) / 9007199254740992.0


def _normal_counter(seed: int, counter: int) -> float:
    import math  # pylint: disable=import-outside-toplevel

    u1 = _uniform_open(seed, counter * 2)
    u2 = _uniform_open(seed, counter * 2 + 1)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * float(3.1415927410125732) * u2)


def _native_sine_source(f0_tensor: Any, seed: int) -> tuple[Any, Any, Any]:
    """Reproduce the installed runtime's tile-independent source generator."""
    import math  # pylint: disable=import-outside-toplevel

    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    harmonics = 9
    upsample = 300
    f0_sampled = f0_tensor.detach().cpu().to(torch.float32).numpy().reshape(-1)
    if len(f0_sampled) % upsample != 0:
        raise ValueError("Kokoro source F0 does not match the 300x upsample contract")
    # Generator.forward applies nearest-neighbor F0 upsampling before calling
    # SineGen. Recover the model-frame curve consumed by the native generator.
    f0 = f0_sampled[::upsample]
    pi = np.float32(3.1415927410125732)
    coarse = np.empty((len(f0), harmonics), dtype=np.float32)
    for harmonic in range(harmonics):
        accumulator = 0.0 if harmonic == 0 else _uniform_open(seed, harmonic)
        for frame, frequency in enumerate(f0):
            increment = math.fmod(float(frequency) * (harmonic + 1) / 24000.0, 1.0)
            if increment < 0.0:
                increment += 1.0
            accumulator += increment
            coarse[frame, harmonic] = np.float32(accumulator * 2.0 * float(pi) * upsample)

    def splitmix64_array(values: Any) -> Any:
        values = values.astype(np.uint64, copy=False)
        with np.errstate(over="ignore"):
            values = values + np.uint64(0x9E3779B97F4A7C15)
            values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))

    def uniform_open_array(counters: Any) -> Any:
        mixed = splitmix64_array(np.uint64(seed & _UINT64_MASK) ^ splitmix64_array(counters))
        return ((mixed >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0

    samples = len(f0) * upsample
    sample_indices = np.arange(samples, dtype=np.int64)
    source = (sample_indices.astype(np.float64) + 0.5) / upsample - 0.5
    floor_source = np.floor(source).astype(np.int64)
    left = np.clip(floor_source, 0, len(f0) - 1)
    right = np.minimum(left + 1, len(f0) - 1)
    fraction = (source - floor_source).astype(np.float32)
    voiced_flat = (f0[np.minimum(sample_indices // upsample, len(f0) - 1)] > 10.0).astype(
        np.float32
    )
    noise_scale = np.where(voiced_flat != 0.0, 0.003, 0.1 / 3.0).astype(np.float32)
    waves = np.empty((samples, harmonics), dtype=np.float32)
    noise = np.empty_like(waves)
    counter_base = np.uint64(0x100000000) + np.arange(samples, dtype=np.uint64) * np.uint64(
        harmonics
    )
    for harmonic in range(harmonics):
        a = coarse[left, harmonic]
        b = coarse[right, harmonic]
        phase = (a + fraction * (b - a).astype(np.float32)).astype(np.float32)
        counters = counter_base + np.uint64(harmonic)
        u1 = uniform_open_array(counters * np.uint64(2))
        u2 = uniform_open_array(counters * np.uint64(2) + np.uint64(1))
        normal = (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * float(pi) * u2)).astype(np.float32)
        noise[:, harmonic] = (noise_scale * normal).astype(np.float32)
        waves[:, harmonic] = (
            np.float32(0.1) * np.sin(phase).astype(np.float32) * voiced_flat + noise[:, harmonic]
        ).astype(np.float32)
    voiced = voiced_flat[:, None]

    device = f0_tensor.device
    return (
        torch.from_numpy(waves).unsqueeze(0).to(device),
        torch.from_numpy(voiced).unsqueeze(0).to(device),
        torch.from_numpy(noise).unsqueeze(0).to(device),
    )


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export pinned Misaki/Kokoro frontend and model reference tensors"
    )
    parser.add_argument("--misaki-root", type=Path, required=True)
    parser.add_argument("--kokoro-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        help=(
            "optional immutable Hugging Face snapshot containing config.json, "
            f"{KOKORO_MODEL_NAME}, and voices/*.pt"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser


def _json_value(value: Any) -> Any:
    """Convert all observable MToken fields to deterministic JSON values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "case"


def _store_tensor(root: Path, name: str, tensor: Any) -> dict[str, Any]:
    """Write a contiguous little-endian tensor and return its manifest entry."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().contiguous().numpy()
    else:
        array = np.asarray(tensor)
    if array.dtype.hasobject:
        raise TypeError(f"cannot serialize object tensor {name}")
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and sys.byteorder == "big"):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    elif array.dtype.byteorder not in ("|", "<"):
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    array = np.ascontiguousarray(array)
    relative = Path(f"{_safe_name(name)}.bin")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return {
        "path": relative.as_posix(),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_model_root(model_root: Path) -> tuple[Path, Path, Path]:
    config = model_root / "config.json"
    checkpoint = model_root / KOKORO_MODEL_NAME
    voices = model_root / "voices"
    if not config.is_file():
        raise FileNotFoundError(f"missing Kokoro config: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Kokoro checkpoint: {checkpoint}")
    if not voices.is_dir():
        raise FileNotFoundError(f"missing Kokoro voices directory: {voices}")
    actual = _sha256(checkpoint)
    if actual != KOKORO_MODEL_SHA256:
        raise RuntimeError(f"Kokoro checkpoint SHA256 is {actual}; expected {KOKORO_MODEL_SHA256}")
    return config, checkpoint, voices


def _capture_decoder_boundaries(model: Any) -> tuple[dict[str, Any], list[Any]]:
    """Install hooks for stochastic source and pre-iSTFT decoder boundaries."""
    captured: dict[str, Any] = {}
    handles = []

    def capture(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            captured[f"{name}.input"] = inputs
            captured[f"{name}.output"] = output

        return hook

    generator = model.decoder.generator
    handles.append(model.decoder.encode.register_forward_hook(capture("decoder.encode")))
    for index, block in enumerate(model.decoder.decode):
        handles.append(block.register_forward_hook(capture(f"decoder.decode.{index}")))
    handles.append(generator.register_forward_hook(capture("generator")))
    handles.append(generator.m_source.register_forward_hook(capture("generator.source")))
    handles.append(generator.m_source.l_sin_gen.register_forward_hook(capture("generator.sine")))
    handles.append(generator.conv_post.register_forward_hook(capture("generator.conv_post")))
    return captured, handles


def _flatten_captures(captured: dict[str, Any]) -> dict[str, Any]:
    """Give tuples returned by hooks stable, reviewable tensor names."""
    flattened: dict[str, Any] = {}
    aliases = {
        "generator.sine.output": ("sine_waves", "voiced_mask", "sine_noise"),
        "generator.source.output": ("harmonic_source", "noise_source", "source_uv"),
    }
    for name, value in captured.items():
        if isinstance(value, (tuple, list)):
            labels = aliases.get(name)
            for index, item in enumerate(value):
                suffix = labels[index] if labels and index < len(labels) else str(index)
                if hasattr(item, "detach"):
                    flattened[f"{name}.{suffix}"] = item
        elif hasattr(value, "detach"):
            flattened[name] = value
    return flattened


def _export_model_chunk(
    model: Any,
    voice_pack: Any,
    phonemes: str,
    speed: float,
    seed: int,
    tensor_root: Path,
) -> dict[str, Any]:
    """Run the public Kokoro v1.0 forward definition while naming boundaries."""
    import torch  # pylint: disable=import-outside-toplevel

    ids = [model.vocab[phoneme] for phoneme in phonemes if phoneme in model.vocab]
    if not ids or len(ids) > 510:
        raise ValueError(f"model fixture has {len(ids)} unframed phoneme IDs")
    if len(phonemes) < 1 or len(phonemes) > voice_pack.shape[0]:
        raise ValueError(f"voice row {len(phonemes) - 1} is unavailable")

    input_ids = torch.tensor([[0, *ids, 0]], dtype=torch.long)
    input_lengths = torch.tensor([input_ids.shape[-1]], dtype=torch.long)
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    attention_mask = (~text_mask).to(torch.int32)
    ref_s = voice_pack[len(phonemes) - 1].reshape(1, -1).to(torch.float32)
    decoder_style = ref_s[:, :128]
    duration_style = ref_s[:, 128:]

    captured, handles = _capture_decoder_boundaries(model)
    sine_generator = model.decoder.generator.m_source.l_sin_gen
    original_sine_forward = sine_generator.forward
    sine_generator.forward = lambda f0: _native_sine_source(f0, seed)
    try:
        with torch.inference_mode():
            torch.manual_seed(seed)
            bert_output = model.bert(input_ids, attention_mask=attention_mask)
            duration_projected = model.bert_encoder(bert_output).transpose(-1, -2)
            duration_encoded = model.predictor.text_encoder(
                duration_projected, duration_style, input_lengths, text_mask
            )
            duration_lstm, _ = model.predictor.lstm(duration_encoded)
            duration_logits = model.predictor.duration_proj(duration_lstm)
            duration_continuous = torch.sigmoid(duration_logits).sum(axis=-1) / speed
            pred_dur = torch.round(duration_continuous).clamp(min=1).long().squeeze(0)
            alignment_indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
            alignment = torch.zeros((input_ids.shape[1], alignment_indices.shape[0]))
            alignment[alignment_indices, torch.arange(alignment_indices.shape[0])] = 1
            alignment = alignment.unsqueeze(0)
            duration_aligned = duration_encoded.transpose(-1, -2) @ alignment
            prosody_shared, _ = model.predictor.shared(duration_aligned.transpose(-1, -2))
            f0, noise = model.predictor.F0Ntrain(duration_aligned, duration_style)
            text_encoded = model.text_encoder(input_ids, input_lengths, text_mask)
            decoder_asr = text_encoded @ alignment
            waveform = (
                model.decoder(decoder_asr, f0, noise, decoder_style).squeeze().to(torch.float32)
            )
    finally:
        sine_generator.forward = original_sine_forward
        for handle in handles:
            handle.remove()

    tensors: dict[str, Any] = {
        "phoneme_ids": torch.tensor(ids, dtype=torch.int32),
        "input_ids": input_ids.to(torch.int32),
        "input_lengths": input_lengths.to(torch.int32),
        "text_mask": text_mask,
        "attention_mask": attention_mask,
        "voice_style": ref_s,
        "decoder_style": decoder_style,
        "duration_style": duration_style,
        "bert_output": bert_output,
        "duration_projected": duration_projected,
        "duration_encoded": duration_encoded,
        "duration_lstm": duration_lstm,
        "duration_logits": duration_logits,
        "duration_continuous": duration_continuous,
        "predicted_durations": pred_dur.to(torch.int32),
        "alignment_indices": alignment_indices.to(torch.int32),
        "alignment": alignment,
        "duration_aligned": duration_aligned,
        "prosody_shared": prosody_shared,
        "f0": f0,
        "noise": noise,
        "text_encoded": text_encoded,
        "decoder_asr": decoder_asr,
        "waveform": waveform,
    }
    tensors.update(_flatten_captures(captured))
    conv_post = tensors.get("generator.conv_post.output")
    if conv_post is None:
        raise RuntimeError("failed to capture Kokoro pre-iSTFT decoder output")
    bins = model.decoder.generator.post_n_fft // 2 + 1
    tensors["pre_istft_magnitude"] = torch.exp(conv_post[:, :bins, :])
    tensors["pre_istft_phase"] = torch.sin(conv_post[:, bins:, :])

    entries = {
        name: _store_tensor(tensor_root, name, value) for name, value in sorted(tensors.items())
    }
    return {
        "speed": speed,
        "seed": seed,
        "phoneme_count": len(phonemes),
        "unframed_id_count": len(ids),
        "voice_row": len(phonemes) - 1,
        "sample_count": waveform.numel(),
        "tensors": entries,
    }


def main() -> int:
    args = _parser().parse_args()
    if _git_head(args.misaki_root) != MISAKI_COMMIT:
        raise RuntimeError(f"Misaki checkout must be pinned to {MISAKI_COMMIT}")
    if _git_head(args.kokoro_root) != KOKORO_COMMIT:
        raise RuntimeError(f"Kokoro checkout must be pinned to {KOKORO_COMMIT}")
    if not (0.5 <= args.speed <= 2.0):
        raise ValueError("--speed must be in [0.5,2.0]")

    sys.path.insert(0, str(args.misaki_root))
    sys.path.insert(0, str(args.kokoro_root))
    from kokoro.pipeline import KPipeline  # pylint: disable=import-outside-toplevel
    from misaki import __version__ as actual_version  # pylint: disable=import-outside-toplevel

    if actual_version != MISAKI_VERSION:
        raise RuntimeError(f"Misaki package must be {MISAKI_VERSION}, got {actual_version}")

    model = None
    voices_root = None
    model_manifest = None
    if args.model_root:
        import torch  # pylint: disable=import-outside-toplevel
        from kokoro.model import KModel  # pylint: disable=import-outside-toplevel

        config, checkpoint, voices_root = _validate_model_root(args.model_root)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model = (
            KModel(
                repo_id="hexgrad/Kokoro-82M",
                config=str(config),
                model=str(checkpoint),
            )
            .cpu()
            .eval()
        )
        model_manifest = {
            "repository": "hexgrad/Kokoro-82M",
            "revision": KOKORO_HF_REVISION,
            "config_sha256": _sha256(config),
            "checkpoint": KOKORO_MODEL_NAME,
            "checkpoint_sha256": _sha256(checkpoint),
        }

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("--cases must contain a non-empty JSON array")
    output_root = args.output.parent / f"{args.output.stem}.tensors"
    pipelines: dict[str, Any] = {}
    loaded_voices: dict[str, Any] = {}
    voice_hashes: dict[str, str] = {}
    fixtures = []
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict) or not isinstance(case.get("text"), str):
            raise ValueError(f"case {case_index} must contain string field 'text'")
        normalized_text = unicodedata.normalize("NFKC", case["text"])
        code = case["kokoro_language"]
        pipeline = pipelines.setdefault(
            code,
            KPipeline(code, repo_id="hexgrad/Kokoro-82M", model=False),
        )
        direct_phonemes = case.get("phonemes")
        if direct_phonemes is not None:
            if not isinstance(direct_phonemes, str) or not direct_phonemes:
                raise ValueError(f"case {case_index} field 'phonemes' must be non-empty")
            results = [SimpleNamespace(graphemes=case["text"], phonemes=direct_phonemes, tokens=[])]
        else:
            results = list(pipeline(normalized_text, split_pattern=None))
        fixture_id = _safe_name(str(case.get("id", f"case-{case_index:04d}")))
        fixture_chunks = []
        voice = case.get("voice")
        if model is not None and not voice:
            raise ValueError(f"model case {fixture_id} has no voice")
        voice_pack = None
        if model is not None:
            import torch  # pylint: disable=import-outside-toplevel

            assert voices_root is not None
            voice_path = voices_root / f"{voice}.pt"
            if not voice_path.is_file():
                raise FileNotFoundError(f"missing fixture voice: {voice_path}")
            if voice not in loaded_voices:
                loaded_voices[voice] = torch.load(
                    voice_path, map_location="cpu", weights_only=True
                ).to(torch.float32)
                voice_hashes[voice] = _sha256(voice_path)
            voice_pack = loaded_voices[voice]

        for chunk_index, result in enumerate(results):
            tokens = [
                {key: _json_value(value) for key, value in sorted(vars(token).items())}
                for token in result.tokens or []
            ]
            chunk: dict[str, Any] = {
                "text": result.graphemes,
                "phonemes": result.phonemes,
                "tokens": tokens,
                "phoneme_ids": (
                    [
                        model.vocab[phoneme]
                        for phoneme in result.phonemes
                        if model is not None and phoneme in model.vocab
                    ]
                    if model is not None
                    else None
                ),
            }
            if model is not None and result.phonemes:
                chunk_speed = float(case.get("speed", args.speed))
                chunk_seed = int(case.get("seed", args.seed)) + chunk_index
                chunk["model"] = _export_model_chunk(
                    model,
                    voice_pack,
                    result.phonemes,
                    chunk_speed,
                    chunk_seed,
                    output_root / fixture_id / f"chunk-{chunk_index:04d}",
                )
            fixture_chunks.append(chunk)
        fixtures.append(
            {
                **case,
                "id": fixture_id,
                "source_text": case["text"],
                "normalized_text": normalized_text,
                "chunks": fixture_chunks,
            }
        )

    manifest = {
        "format": 2,
        "endianness": "little",
        "source_generator": "splitmix64-box-muller-v1",
        "misaki_commit": MISAKI_COMMIT,
        "misaki_version": MISAKI_VERSION,
        "kokoro_commit": KOKORO_COMMIT,
        "kokoro_pipeline_sha256": _sha256(args.kokoro_root / "kokoro" / "pipeline.py"),
        "model": model_manifest,
        "voice_sha256": dict(sorted(voice_hashes.items())),
        "fixtures": fixtures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
