#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the pinned upstream Python OmniVoice implementation."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import omnivoice_benchmark as benchmark_support
import torch_audio_compat


def synchronize(torch, backend: str) -> None:
    if backend == "cuda":
        torch.cuda.synchronize()


def benchmark(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(args.omnivoice_root.resolve()))

    import numpy as np
    import torch
    import torchcodec
    import transformers
    from torchcodec.decoders import AudioDecoder

    # The pinned model still imports torchaudio.functional for its internal
    # 24-to-16 kHz semantic path. Media decode itself uses TorchCodec below.
    torch_audio_compat.install()

    from omnivoice import OmniVoice

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = "cuda:0" if args.backend == "cuda" else "cpu"
    dtype = torch.float16 if args.backend == "cuda" else torch.float32
    records = benchmark_support.source_records(args.input_file, args.reference_manifest, args.limit)
    reference_path = benchmark_support.resolve_reference(
        records[0]["ref_audio"], args.reference_root
    )

    synchronize(torch, args.backend)
    started = time.perf_counter()
    model = OmniVoice.from_pretrained(
        str(args.model.resolve()),
        device_map=device,
        dtype=dtype,
        load_asr=False,
    )
    synchronize(torch, args.backend)
    model_load_s = time.perf_counter() - started

    acceleration = "pytorch-default"
    if args.flashinfer:
        from omnivoice.models.omnivoice_flashinfer import apply_flashinfer

        apply_flashinfer(model, enable_cuda_graph=args.cuda_graph)
        acceleration = "flashinfer-cuda-graph" if args.cuda_graph else "flashinfer"

    synchronize(torch, args.backend)
    started = time.perf_counter()
    decoded_reference = AudioDecoder(
        str(reference_path), sample_rate=int(model.sampling_rate), num_channels=1
    ).get_all_samples()
    prompt = model.create_voice_clone_prompt(
        ref_audio=(decoded_reference.data, decoded_reference.sample_rate),
        ref_text=records[0]["ref_text"],
        preprocess_prompt=True,
    )
    synchronize(torch, args.backend)
    prompt_encode_s = time.perf_counter() - started

    def generate(item: dict) -> dict:
        seed = args.seed + item["source_line"] - 1
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
        if args.backend == "cuda":
            torch.cuda.manual_seed_all(seed)
        synchronize(torch, args.backend)
        started = time.perf_counter()
        audio = model.generate(
            text=item["text"],
            language="en",
            voice_clone_prompt=prompt,
            normalize_text=False,
            num_step=args.steps,
            guidance_scale=args.guidance,
            position_temperature=args.position_temperature,
            class_temperature=args.class_temperature,
        )[0]
        synchronize(torch, args.backend)
        elapsed_s = time.perf_counter() - started
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError(f"{item['id']}: upstream returned invalid audio")
        audio_s = audio.size / model.sampling_rate
        return {
            **item,
            "seed": seed,
            "samples": int(audio.size),
            "sample_rate": int(model.sampling_rate),
            "audio_s": audio_s,
            "wall_elapsed_s": elapsed_s,
            "wall_rtfx": audio_s / elapsed_s,
        }

    for index in range(args.warmup):
        warmup = dict(records[index % len(records)])
        warmup["id"] = f"warmup-{index + 1:02d}"
        generate(warmup)

    rows = []
    for item in records:
        row = generate(item)
        rows.append(row)
        print(
            f"upstream-{args.backend} {item['id']}: {row['wall_rtfx']:.3f} RTFx, "
            f"{row['wall_elapsed_s']:.3f} s wall",
            flush=True,
        )

    total_audio_s = sum(row["audio_s"] for row in rows)
    total_elapsed_s = sum(row["wall_elapsed_s"] for row in rows)
    result = {
        "schema_version": 1,
        "implementation": "upstream-python",
        "upstream_revision": benchmark_support.shell_output(
            ["git", "-C", str(args.omnivoice_root), "rev-parse", "HEAD"]
        ),
        "backend": args.backend,
        "acceleration": acceleration,
        "configuration": {
            "steps": args.steps,
            "guidance": args.guidance,
            "position_temperature": args.position_temperature,
            "class_temperature": args.class_temperature,
            "seed_base": args.seed,
            "threads": args.threads,
            "warmup_utterances": args.warmup,
            "dtype": str(dtype),
            "voice_mode": "reused voice-clone prompt",
            "reference_decoder": "torchcodec.AudioDecoder",
            "torchaudio_compatibility": "pure-PyTorch band-limited sinc resample",
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchcodec": torchcodec.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
        "artifacts": {
            "input_file": str(args.input_file.resolve()),
            "input_sha256": benchmark_support.sha256(args.input_file),
            "model": str(args.model.resolve()),
            "model_sha256": benchmark_support.sha256(args.model / "model.safetensors"),
            "codec_sha256": benchmark_support.sha256(
                args.model / "audio_tokenizer" / "model.safetensors"
            ),
            "reference_audio": str(reference_path.resolve()),
            "reference_audio_sha256": benchmark_support.sha256(reference_path),
        },
        "model_load_s": model_load_s,
        "prompt_encode_s": prompt_encode_s,
        "summary": {
            "utterances": len(rows),
            "audio_s": total_audio_s,
            "wall_elapsed_s": total_elapsed_s,
            "aggregate_rtfx": total_audio_s / total_elapsed_s,
            "mean_utterance_rtfx": sum(row["wall_rtfx"] for row in rows) / len(rows),
            "median_utterance_rtfx": benchmark_support.percentile(
                [row["wall_rtfx"] for row in rows], 0.5
            ),
        },
        "utterances": rows,
    }
    benchmark_support.write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--omnivoice-root", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--input-file", type=Path, required=True)
    result.add_argument("--reference-manifest", type=Path, required=True)
    result.add_argument("--reference-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    result.add_argument("--limit", type=int, default=10)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--steps", type=int, default=32)
    result.add_argument("--guidance", type=float, default=2.0)
    result.add_argument("--position-temperature", type=float, default=5.0)
    result.add_argument("--class-temperature", type=float, default=0.0)
    result.add_argument("--seed", type=int, default=20260904)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--flashinfer", action="store_true")
    result.add_argument("--cuda-graph", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.cuda_graph and not arguments.flashinfer:
        raise SystemExit("--cuda-graph requires --flashinfer")
    if arguments.flashinfer and arguments.backend != "cuda":
        raise SystemExit("--flashinfer requires --backend cuda")
    benchmark(arguments)
