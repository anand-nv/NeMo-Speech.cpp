#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark native OmniVoice synthesis with process memory sampling.

The benchmark loads one synthesizer, creates one reusable voice prompt, runs an
optional warm-up, and then measures each selected utterance independently.
CUDA memory uses process-scoped allocation counters when the supplied
cuda_allocation_tracker.cpp helper is preloaded.  Otherwise it falls back to
an exclusive-device cudaMemGetInfo delta.  The helper is useful on coherent-
memory devices such as GB10 where NVML process memory is unsupported.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import subprocess
import threading
import time
from pathlib import Path

import omnivoice_minimax_eval as eval_support

MIB = 1024 * 1024


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * MIB):
            digest.update(block)
    return digest.hexdigest()


def proc_status_kib() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if not line.startswith(("VmRSS:", "RssAnon:", "RssFile:", "VmHWM:")):
            continue
        key, value = line.split(":", 1)
        result[key] = int(value.split()[0])
    return result


def host_memory_total_kib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    raise RuntimeError("MemTotal is missing from /proc/meminfo")


class CudaMemory:
    def __init__(self) -> None:
        self.tracker = ctypes.CDLL(None)
        try:
            self.tracker.omnivoice_cuda_current_allocation_bytes.restype = ctypes.c_size_t
            self.tracker.omnivoice_cuda_peak_allocation_bytes.restype = ctypes.c_size_t
            self.tracker.omnivoice_cuda_reset_peak_allocation_bytes.restype = None
            self.process_tracker = True
        except AttributeError:
            self.process_tracker = False
        self.library = None
        for name in ("libcudart.so", "libcudart.so.13"):
            try:
                self.library = ctypes.CDLL(name)
                break
            except OSError:
                pass
        if self.library is None:
            raise RuntimeError("CUDA backend requested but libcudart was not found")
        self.library.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.library.cudaMemGetInfo.restype = ctypes.c_int

        _free, self.total = self.device_memory()
        self.baseline_free = _free

    def device_memory(self) -> tuple[int, int]:
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        status = self.library.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if status != 0:
            raise RuntimeError(f"cudaMemGetInfo failed with CUDA status {status}")
        return free.value, total.value

    def used(self) -> int:
        if self.process_tracker:
            return self.tracker.omnivoice_cuda_current_allocation_bytes()
        free, _total = self.device_memory()
        return max(0, self.baseline_free - free)

    def peak(self) -> int:
        if self.process_tracker:
            return self.tracker.omnivoice_cuda_peak_allocation_bytes()
        return self.used()

    def reset_peak(self) -> None:
        if self.process_tracker:
            self.tracker.omnivoice_cuda_reset_peak_allocation_bytes()

    @property
    def method(self) -> str:
        if self.process_tracker:
            return (
                "process-scoped high-water sum of successful cudaMalloc and "
                "cudaMallocManaged allocations recorded by LD_PRELOAD interposition; "
                "NVML per-process framebuffer memory is unsupported on this GB10 "
                "coherent-memory system"
            )
        return (
            "exclusive-device high-water delta from cudaMemGetInfo; NVML per-process "
            "framebuffer memory is unsupported on this coherent-memory system"
        )


class MemorySampler:
    def __init__(self, backend: str, interval_ms: int) -> None:
        self.interval_s = interval_ms / 1000.0
        self.host_total_kib = host_memory_total_kib()
        self.cuda = CudaMemory() if backend == "cuda" else None
        self.gpu_total = self.cuda.total if self.cuda is not None else 0
        initial = proc_status_kib()
        self.overall_peak_rss_kib = initial["VmRSS"]
        self.overall_peak_gpu_bytes = self.cuda.peak() if self.cuda is not None else 0
        self._label: str | None = None
        self._label_peak_rss_kib = 0
        self._label_peak_gpu_bytes = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        rss = proc_status_kib()["VmRSS"]
        gpu_used = self.cuda.used() if self.cuda is not None else 0
        with self._lock:
            self.overall_peak_rss_kib = max(self.overall_peak_rss_kib, rss)
            if self.cuda is not None:
                self.overall_peak_gpu_bytes = max(
                    self.overall_peak_gpu_bytes, gpu_used, self.cuda.peak()
                )
            if self._label is not None:
                self._label_peak_rss_kib = max(self._label_peak_rss_kib, rss)
                if self.cuda is not None:
                    self._label_peak_gpu_bytes = max(self._label_peak_gpu_bytes, gpu_used)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def begin(self, label: str) -> None:
        self._sample()
        if self.cuda is not None:
            self.cuda.reset_peak()
        with self._lock:
            self._label = label
            self._label_peak_rss_kib = proc_status_kib()["VmRSS"]
            self._label_peak_gpu_bytes = self.cuda.used() if self.cuda is not None else 0

    def end(self, label: str) -> dict[str, float]:
        self._sample()
        with self._lock:
            if self._label != label:
                raise RuntimeError(f"memory sampler label mismatch: {self._label!r} != {label!r}")
            peak_rss_kib = self._label_peak_rss_kib
            peak_gpu_bytes = self._label_peak_gpu_bytes
            if self.cuda is not None:
                peak_gpu_bytes = max(peak_gpu_bytes, self.cuda.peak())
                self.overall_peak_gpu_bytes = max(self.overall_peak_gpu_bytes, peak_gpu_bytes)
            self._label = None
        gpu_mib = peak_gpu_bytes / MIB
        return {
            "process_cpu_peak_rss_mib": peak_rss_kib / 1024.0,
            "process_cpu_peak_rss_percent": 100.0 * peak_rss_kib / self.host_total_kib,
            "process_gpu_peak_mib": gpu_mib,
            "process_gpu_peak_percent": (
                100.0 * gpu_mib / (self.gpu_total / MIB) if self.gpu_total else 0.0
            ),
        }

    def close(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join()
        self._sample()
        gpu_mib = self.overall_peak_gpu_bytes / MIB
        return {
            "host_memory_total_mib": self.host_total_kib / 1024.0,
            "process_cpu_peak_rss_mib": self.overall_peak_rss_kib / 1024.0,
            "process_cpu_peak_rss_percent": (
                100.0 * self.overall_peak_rss_kib / self.host_total_kib
            ),
            "cuda_addressable_memory_total_mib": self.gpu_total / MIB,
            "process_gpu_peak_mib": gpu_mib,
            "process_gpu_peak_percent": (
                100.0 * gpu_mib / (self.gpu_total / MIB) if self.gpu_total else 0.0
            ),
        }


def source_records(input_file: Path, reference_manifest: Path, limit: int) -> list[dict]:
    selected = []
    for source_line, line in enumerate(input_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if "|" not in line:
            raise RuntimeError(f"malformed input at {input_file}:{source_line}")
        voice, text = line.split("|", 1)
        selected.append(
            {
                "id": f"english-{len(selected) + 1:02d}",
                "source_line": source_line,
                "voice": voice,
                "text": text,
            }
        )
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise RuntimeError(f"{input_file} contains fewer than {limit} non-empty utterances")

    references = {record["text"]: record for record in read_jsonl(reference_manifest)}
    for record in selected:
        reference = references.get(record["text"])
        if reference is None:
            raise RuntimeError(f"reference manifest has no exact match for {record['id']}")
        record["ref_audio"] = reference["ref_audio"]
        record["ref_text"] = reference["ref_text"]
        record["language_id"] = reference["language_id"]
    return selected


def resolve_reference(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    direct = root / candidate
    if direct.exists():
        return direct
    if "minimax_multilingual_24" in candidate.parts:
        tail = candidate.parts[candidate.parts.index("minimax_multilingual_24") :]
        normalized = root.joinpath(*tail)
        if normalized.exists():
            return normalized
    raise RuntimeError(f"reference audio does not exist under {root}: {path}")


def shell_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate(rows: list[dict]) -> dict[str, float]:
    total_audio = sum(row["audio_s"] for row in rows)
    total_elapsed = sum(row["wall_elapsed_s"] for row in rows)
    rtfx = [row["wall_rtfx"] for row in rows]
    ttfa = [row["ttfa_ms"] for row in rows]
    cpu = [row["process_cpu_peak_rss_mib"] for row in rows]
    gpu = [row["process_gpu_peak_mib"] for row in rows]
    return {
        "utterances": len(rows),
        "audio_s": total_audio,
        "wall_elapsed_s": total_elapsed,
        "aggregate_rtfx": total_audio / total_elapsed,
        "mean_utterance_rtfx": sum(rtfx) / len(rtfx),
        "median_utterance_rtfx": percentile(rtfx, 0.5),
        "mean_ttfa_ms": sum(ttfa) / len(ttfa),
        "median_ttfa_ms": percentile(ttfa, 0.5),
        "p95_ttfa_ms": percentile(ttfa, 0.95),
        "max_process_cpu_peak_rss_mib": max(cpu),
        "max_process_gpu_peak_mib": max(gpu),
    }


def benchmark(args: argparse.Namespace) -> dict:
    records = source_records(args.input_file, args.reference_manifest, args.limit)
    if len({record["voice"] for record in records}) != 1:
        raise RuntimeError("selected records do not share one reference voice")
    reference_path = resolve_reference(records[0]["ref_audio"], args.reference_root)
    samples, sample_rate = eval_support.read_reference_audio(reference_path)
    sample_array = (ctypes.c_float * len(samples))(*samples)

    sampler = MemorySampler(args.backend, args.sample_interval_ms)
    lib = ctypes.CDLL(str(args.library.resolve()))
    eval_support.configure_native_api(lib)
    model_path = str(args.model.resolve()).encode()
    codec_path = str(args.codec.resolve()).encode()
    omni = lib.nemo_speech_tts_omnivoice_options_default()
    omni.num_steps = args.steps
    omni.guidance_scale = args.guidance
    omni.position_temperature = args.position_temperature
    omni.class_temperature = args.class_temperature
    model = eval_support.ModelConfig(
        ctypes.sizeof(eval_support.ModelConfig),
        None,
        None,
        None,
        None,
        model_path,
        codec_path,
    )
    runtime = lib.nemo_speech_tts_runtime_config_default_v2()
    runtime.lt_backend = 1 if args.backend == "cpu" else 2
    runtime.threads = args.threads
    runtime.codec_threads = args.threads
    runtime.omnivoice_options = ctypes.pointer(omni)
    config = eval_support.SynthConfig(
        ctypes.sizeof(eval_support.SynthConfig),
        ctypes.pointer(model),
        ctypes.pointer(runtime),
        b"en",
        b"auto",
    )

    handle = ctypes.c_void_p()
    start = time.perf_counter()
    status = lib.nemo_speech_tts_create(ctypes.byref(config), ctypes.byref(handle))
    model_load_s = time.perf_counter() - start
    if status != 0:
        sampler.close()
        raise RuntimeError(lib.nemo_speech_tts_last_error().decode())

    prompt = ctypes.c_void_p()
    start = time.perf_counter()
    status = lib.nemo_speech_tts_voice_prompt_create(
        handle,
        sample_array,
        len(samples),
        1,
        sample_rate,
        records[0]["ref_text"].encode(),
        True,
        ctypes.byref(prompt),
    )
    prompt_encode_s = time.perf_counter() - start
    if status != 0:
        lib.nemo_speech_tts_destroy(handle)
        sampler.close()
        raise RuntimeError(lib.nemo_speech_tts_last_error().decode())

    def synthesize(item: dict, measured: bool) -> dict:
        request = lib.nemo_speech_tts_synthesis_options_default_v2()
        request.language_code = b"en"
        request.voice_name = b"auto"
        request.seed = args.seed + item["source_line"] - 1
        request.omnivoice_options = ctypes.pointer(omni)
        request.voice_prompt = prompt
        stats = lib.nemo_speech_tts_synthesis_stats_default()
        started = time.perf_counter()
        first_callback = None

        @eval_support.CALLBACK
        def discard(_data, count, _user_data):
            nonlocal first_callback
            if count and first_callback is None:
                first_callback = time.perf_counter()
            return True

        if measured:
            sampler.begin(item["id"])
        status = lib.nemo_speech_tts_synthesize_text(
            handle,
            ctypes.byref(request),
            item["text"].encode(),
            discard,
            None,
            ctypes.byref(stats),
        )
        finished = time.perf_counter()
        memory = sampler.end(item["id"]) if measured else {}
        if status != 0:
            raise RuntimeError(f"{item['id']}: {lib.nemo_speech_tts_last_error().decode()}")
        if first_callback is None or stats.samples_written == 0:
            raise RuntimeError(f"{item['id']}: synthesis returned no audio callback")
        wall_elapsed_s = finished - started
        audio_s = stats.samples_written / stats.sample_rate
        return {
            **item,
            "seed": request.seed,
            "audio_s": audio_s,
            "wall_elapsed_s": wall_elapsed_s,
            "wall_rtfx": audio_s / wall_elapsed_s,
            "ttfa_ms": 1000.0 * (first_callback - started),
            "native_elapsed_s": stats.elapsed_s,
            "native_rtfx": stats.rtfx,
            "samples": stats.samples_written,
            "sample_rate": stats.sample_rate,
            "generated_frames": stats.generated_frames,
            "chunks": stats.chunks,
            **memory,
        }

    rows = []
    try:
        for index in range(args.warmup):
            warmup = dict(records[index % len(records)])
            warmup["id"] = f"warmup-{index + 1:02d}"
            synthesize(warmup, False)
        for item in records:
            result = synthesize(item, True)
            rows.append(result)
            print(
                f"{args.backend} {item['id']}: {result['wall_rtfx']:.3f} RTFx, "
                f"{result['ttfa_ms']:.1f} ms TTFA, "
                f"{result['process_cpu_peak_rss_mib']:.1f} MiB RSS, "
                f"{result['process_gpu_peak_mib']:.1f} MiB CUDA",
                flush=True,
            )
    finally:
        lib.nemo_speech_tts_voice_prompt_destroy(prompt)
        lib.nemo_speech_tts_destroy(handle)
    overall_memory = sampler.close()

    uname = platform.uname()
    result = {
        "schema_version": 1,
        "backend": args.backend,
        "configuration": {
            "steps": args.steps,
            "guidance": args.guidance,
            "position_temperature": args.position_temperature,
            "class_temperature": args.class_temperature,
            "seed_base": args.seed,
            "threads_requested": args.threads,
            "warmup_utterances": args.warmup,
            "memory_sample_interval_ms": args.sample_interval_ms,
            "voice_mode": "reused voice-clone prompt",
        },
        "environment": {
            "hostname": uname.node,
            "kernel": uname.release,
            "machine": uname.machine,
            "python": platform.python_version(),
            "cpu": shell_output(["lscpu", "-J"]),
            "gpu": (shell_output(["nvidia-smi", "-L"]) if args.backend == "cuda" else None),
            "driver": (
                shell_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ]
                )
                if args.backend == "cuda"
                else None
            ),
        },
        "artifacts": {
            "input_file": str(args.input_file.resolve()),
            "input_sha256": sha256(args.input_file),
            "library": str(args.library.resolve()),
            "library_sha256": sha256(args.library.resolve()),
            "model": str(args.model.resolve()),
            "model_sha256": sha256(args.model),
            "codec": str(args.codec.resolve()),
            "codec_sha256": sha256(args.codec),
            "reference_audio": str(reference_path.resolve()),
            "reference_audio_sha256": sha256(reference_path),
        },
        "model_load_s": model_load_s,
        "prompt_encode_s": prompt_encode_s,
        "memory": overall_memory,
        "summary": aggregate(rows),
        "utterances": rows,
        "gpu_memory_method": (
            sampler.cuda.method
            if args.backend == "cuda"
            else "no CUDA backend initialized; expected process GPU allocation is zero"
        ),
    }
    write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--library", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--codec", type=Path, required=True)
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
    result.add_argument("--sample-interval-ms", type=int, default=50)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    benchmark(arguments)
