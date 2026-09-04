#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned MiniMax subset selection and Whisper large-v3-turbo scoring.

The normal release flow is:
  select -> synthesize (python oracle, native CPU, native CUDA) -> score -> compare
The installed native runtime remains Python-free; this is release-only tooling.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import random
import sys
import wave
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "omnivoice_minimax_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def records_from_sources(spec_path: Path, source_root: Path) -> list[dict]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    limit = int(spec["utterances_per_language"])
    records: list[dict] = []
    for language in spec["languages"]:
        source = source_root / language["file"]
        raw = source.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != language["sha256"]:
            raise RuntimeError(f"MiniMax source hash mismatch for {source}: {actual}")
        selected = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()][
            :limit
        ]
        if len(selected) != limit:
            raise RuntimeError(f"{source} has only {len(selected)} non-empty utterances")
        stem = Path(language["file"]).stem
        for index, line in enumerate(selected, 1):
            if "|" not in line:
                raise RuntimeError(f"malformed MiniMax line in {source}: {line!r}")
            voice, text = line.split("|", 1)
            records.append(
                {
                    "id": f"{stem}-{index:02d}",
                    "source_file": language["file"],
                    "source_line": index,
                    "voice": voice,
                    "text": text,
                    "language_id": language["omnivoice_id"],
                    "iso639_3": language["iso639_3"],
                    "whisper_language": language["whisper"],
                    "metric": language["metric"],
                }
            )
    if len(records) != len(spec["languages"]) * limit:
        raise RuntimeError("MiniMax selection count is inconsistent")
    return records


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def jsonl_sha256(records: list[dict]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_omnivoice_text_normalize(omnivoice_root: Path):
    """Load the pinned WER normalizer without importing the TTS model package."""
    import importlib.util
    import types

    package_paths = {
        "omnivoice": omnivoice_root / "omnivoice",
        "omnivoice.eval": omnivoice_root / "omnivoice" / "eval",
        "omnivoice.eval.wer": omnivoice_root / "omnivoice" / "eval" / "wer",
    }
    norm_name = "omnivoice.eval.wer.norm_config_module"
    text_name = "omnivoice.eval.wer.text_norm_omni"
    module_paths = {
        norm_name: package_paths["omnivoice.eval.wer"] / "norm_config_module.py",
        text_name: package_paths["omnivoice.eval.wer"] / "text_norm_omni.py",
    }
    for path in (*package_paths.values(), *module_paths.values()):
        if not path.exists():
            raise RuntimeError(f"pinned OmniVoice normalizer path is missing: {path}")

    names = (*package_paths.keys(), *module_paths.keys())
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name, path in package_paths.items():
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        loaded = {}
        for name, path in module_paths.items():
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load pinned OmniVoice normalizer: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        return loaded[text_name].text_normalize
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def command_select(args: argparse.Namespace) -> None:
    records = records_from_sources(args.spec, args.source_root)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    actual_selection_hash = jsonl_sha256(records)
    expected_selection_hash = spec.get("selected_jsonl_sha256")
    if expected_selection_hash and actual_selection_hash != expected_selection_hash:
        raise RuntimeError(
            "MiniMax selected manifest hash mismatch: "
            f"expected {expected_selection_hash}, got {actual_selection_hash}"
        )
    if args.reference_manifest:
        references = load_jsonl(args.reference_manifest)
        by_text = {item["text"]: item for item in references}
        for record in records:
            reference = by_text.get(record["text"])
            if reference is None:
                raise RuntimeError(
                    f"pinned reference manifest is missing exact text {record['id']}"
                )
            for field in ("ref_audio", "ref_text"):
                if not reference.get(field):
                    raise RuntimeError(f"reference item for {record['id']} is missing {field}")
                record[field] = reference[field]
            if reference.get("language_id") != record["language_id"]:
                raise RuntimeError(f"reference language mismatch for {record['id']}")
            parts = Path(record["ref_audio"]).parts
            if "minimax_multilingual_24" in parts:
                start = parts.index("minimax_multilingual_24")
                record["ref_audio"] = str(Path(*parts[start:]))
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} deterministic records to {args.output}")


class OmniOptions(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("num_steps", ctypes.c_int32),
        ("guidance_scale", ctypes.c_float),
        ("t_shift", ctypes.c_float),
        ("layer_penalty_factor", ctypes.c_float),
        ("position_temperature", ctypes.c_float),
        ("class_temperature", ctypes.c_float),
        ("denoise", ctypes.c_bool),
        ("postprocess_output", ctypes.c_bool),
        ("audio_chunk_duration_s", ctypes.c_double),
        ("audio_chunk_threshold_s", ctypes.c_double),
        ("pad_duration_s", ctypes.c_double),
        ("fade_duration_s", ctypes.c_double),
        ("speed", ctypes.c_double),
        ("duration_s", ctypes.c_double),
    ]


class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magpie_model", ctypes.c_char_p),
        ("codec_model", ctypes.c_char_p),
        ("tokenizer_model_dir", ctypes.c_char_p),
        ("text_normalizer_model_dir", ctypes.c_char_p),
        ("omnivoice_model", ctypes.c_char_p),
        ("omnivoice_audio_tokenizer_model", ctypes.c_char_p),
    ]


class RuntimeConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("speaker", ctypes.c_int32),
        ("threads", ctypes.c_int32),
        ("codec_threads", ctypes.c_int32),
        ("seed", ctypes.c_int32),
        ("steps", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("chunk_frames", ctypes.c_int32),
        ("codec_queue_depth", ctypes.c_int32),
        ("codec_history_frames", ctypes.c_int32),
        ("codec_future_frames", ctypes.c_int32),
        ("window_ms", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("override_temperature", ctypes.c_bool),
        ("cfg_scale", ctypes.c_float),
        ("override_cfg_scale", ctypes.c_bool),
        ("use_cfg", ctypes.c_bool),
        ("use_local_transformer", ctypes.c_bool),
        ("use_kv_cache", ctypes.c_bool),
        ("use_stateful_codec", ctypes.c_bool),
        ("codec_cpu", ctypes.c_bool),
        ("flush_partial_chunk", ctypes.c_bool),
        ("verbose", ctypes.c_bool),
        ("lt_backend", ctypes.c_int),
        ("sampling_backend", ctypes.c_int),
        ("uma_mode", ctypes.c_int),
        ("longform_mode", ctypes.c_int),
        ("lt_fp32", ctypes.c_bool),
        ("omnivoice_options", ctypes.POINTER(OmniOptions)),
    ]


class SynthConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model", ctypes.POINTER(ModelConfig)),
        ("runtime", ctypes.POINTER(RuntimeConfig)),
        ("default_language_code", ctypes.c_char_p),
        ("default_voice_name", ctypes.c_char_p),
    ]


class RequestOptions(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("request_id", ctypes.c_char_p),
        ("language_code", ctypes.c_char_p),
        ("speaker", ctypes.c_int32),
        ("seed", ctypes.c_int32),
        ("steps", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("override_temperature", ctypes.c_bool),
        ("cfg_scale", ctypes.c_float),
        ("override_cfg_scale", ctypes.c_bool),
        ("voice_name", ctypes.c_char_p),
        ("output_sample_rate", ctypes.c_int32),
        ("omnivoice_options", ctypes.POINTER(OmniOptions)),
        ("voice_prompt", ctypes.c_void_p),
        ("instruction", ctypes.c_char_p),
    ]


class Stats(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("sample_rate", ctypes.c_int32),
        ("generated_frames", ctypes.c_int32),
        ("chunks", ctypes.c_int32),
        ("e2e_chunks", ctypes.c_int32),
        ("samples_written", ctypes.c_uint64),
    ] + [
        (name, ctypes.c_double)
        for name in (
            "tokenizer_ms",
            "encoder_ms",
            "audio_s",
            "elapsed_s",
            "rtf",
            "rtfx",
            "ttfa_ms",
            "icl_avg_ms",
            "icl_min_ms",
            "icl_max_ms",
            "decoder_audio_s",
            "decoder_elapsed_s",
            "decoder_rtfx",
            "decoder_ttft_ms",
            "decoder_itl_avg_ms",
            "decoder_itl_min_ms",
            "decoder_itl_max_ms",
            "decoder_itl_p95_ms",
            "decoder_itl_p99_ms",
            "codec_audio_s",
            "codec_elapsed_s",
            "codec_rtfx",
            "codec_ttfa_ms",
            "codec_icl_avg_ms",
            "codec_icl_min_ms",
            "codec_icl_max_ms",
            "codec_icl_p95_ms",
            "codec_icl_p99_ms",
            "e2e_ttfa_ms",
            "e2e_icl_avg_ms",
            "e2e_icl_min_ms",
            "e2e_icl_max_ms",
            "e2e_icl_p95_ms",
            "e2e_icl_p99_ms",
            "e2e_rtfx",
        )
    ]


CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_void_p
)


def read_reference_audio(path: Path) -> tuple[list[float], int]:
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[0] == 0 or rate <= 0:
        raise RuntimeError(f"reference audio is empty: {path}")
    return np.mean(audio, axis=1, dtype=np.float32).tolist(), int(rate)


def configure_native_api(lib: ctypes.CDLL) -> None:
    """Declare the complete pointer-width-safe subset of the stable C ABI."""
    lib.nemo_speech_tts_runtime_config_default_v2.restype = RuntimeConfig
    lib.nemo_speech_tts_omnivoice_options_default.restype = OmniOptions
    lib.nemo_speech_tts_synthesis_options_default_v2.restype = RequestOptions
    lib.nemo_speech_tts_synthesis_stats_default.restype = Stats
    lib.nemo_speech_tts_last_error.restype = ctypes.c_char_p
    lib.nemo_speech_tts_create.argtypes = [
        ctypes.POINTER(SynthConfig),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.nemo_speech_tts_create.restype = ctypes.c_int
    lib.nemo_speech_tts_destroy.argtypes = [ctypes.c_void_p]
    lib.nemo_speech_tts_destroy.restype = None
    lib.nemo_speech_tts_voice_prompt_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.nemo_speech_tts_voice_prompt_create.restype = ctypes.c_int
    lib.nemo_speech_tts_voice_prompt_destroy.argtypes = [ctypes.c_void_p]
    lib.nemo_speech_tts_voice_prompt_destroy.restype = None
    lib.nemo_speech_tts_synthesize_text.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(RequestOptions),
        ctypes.c_char_p,
        CALLBACK,
        ctypes.c_void_p,
        ctypes.POINTER(Stats),
    ]
    lib.nemo_speech_tts_synthesize_text.restype = ctypes.c_int


def command_synthesize(args: argparse.Namespace) -> None:
    records = load_jsonl(args.manifest)
    if not records or any("ref_audio" not in item or "ref_text" not in item for item in records):
        raise RuntimeError("synthesis manifest must include pinned ref_audio and ref_text fields")
    lib = ctypes.CDLL(str(args.library.resolve()))
    configure_native_api(lib)
    model_path, codec_path = str(args.model).encode(), str(args.codec).encode()
    omni = lib.nemo_speech_tts_omnivoice_options_default()
    omni.num_steps, omni.guidance_scale = args.steps, args.guidance
    omni.position_temperature, omni.class_temperature = (
        args.position_temperature,
        args.class_temperature,
    )
    model = ModelConfig(ctypes.sizeof(ModelConfig), None, None, None, None, model_path, codec_path)
    runtime = lib.nemo_speech_tts_runtime_config_default_v2()
    runtime.lt_backend = 1 if args.backend == "cpu" else 2
    runtime.omnivoice_options = ctypes.pointer(omni)
    config = SynthConfig(
        ctypes.sizeof(SynthConfig),
        ctypes.pointer(model),
        ctypes.pointer(runtime),
        b"en",
        b"auto",
    )
    handle = ctypes.c_void_p()
    status = lib.nemo_speech_tts_create(ctypes.byref(config), ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(lib.nemo_speech_tts_last_error().decode())
    output_records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for ordinal, item in enumerate(records):
            reference_path = Path(item["ref_audio"])
            if not reference_path.is_absolute():
                reference_path = args.reference_root / reference_path
            samples, reference_rate = read_reference_audio(reference_path)
            sample_array = (ctypes.c_float * len(samples))(*samples)
            prompt = ctypes.c_void_p()
            status = lib.nemo_speech_tts_voice_prompt_create(
                handle,
                sample_array,
                len(samples),
                1,
                reference_rate,
                item["ref_text"].encode(),
                True,
                ctypes.byref(prompt),
            )
            if status != 0:
                raise RuntimeError(lib.nemo_speech_tts_last_error().decode())
            pcm = bytearray()

            @CALLBACK
            def collect(data, count, _):
                pcm.extend(ctypes.string_at(data, count))
                return True

            request = lib.nemo_speech_tts_synthesis_options_default_v2()
            language = item["language_id"].encode()
            request.language_code, request.voice_name = language, b"auto"
            request.seed = args.seed + ordinal
            request.omnivoice_options, request.voice_prompt = (
                ctypes.pointer(omni),
                prompt,
            )
            stats = lib.nemo_speech_tts_synthesis_stats_default()
            status = lib.nemo_speech_tts_synthesize_text(
                handle,
                ctypes.byref(request),
                item["text"].encode(),
                collect,
                None,
                ctypes.byref(stats),
            )
            lib.nemo_speech_tts_voice_prompt_destroy(prompt)
            if status != 0:
                raise RuntimeError(f"{item['id']}: {lib.nemo_speech_tts_last_error().decode()}")
            if not pcm:
                raise RuntimeError(f"{item['id']}: native synthesis returned empty PCM")
            wav_path = args.output_dir / f"{item['id']}.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(stats.sample_rate)
                wav.writeframes(pcm)
            result = dict(item)
            result.update(
                {
                    "wav_path": str(wav_path),
                    "backend": args.backend,
                    "seed": request.seed,
                    "sample_rate": stats.sample_rate,
                    "samples": stats.samples_written,
                    "generated_frames": stats.generated_frames,
                    "chunks": stats.chunks,
                    "elapsed_s": stats.elapsed_s,
                    "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                }
            )
            output_records.append(result)
            print(f"[{ordinal + 1}/{len(records)}] {item['id']} {stats.samples_written} samples")
    finally:
        lib.nemo_speech_tts_destroy(handle)
    write_jsonl(args.output_manifest, output_records)


def command_oracle_synthesize(args: argparse.Namespace) -> None:
    """Generate the pinned Python-oracle baseline with the same request fields."""
    records = load_jsonl(args.manifest)
    if not records or any("ref_audio" not in item or "ref_text" not in item for item in records):
        raise RuntimeError("oracle manifest must include pinned ref_audio and ref_text fields")
    if not args.model.is_dir():
        raise RuntimeError("--model must be the locally downloaded pinned OmniVoice snapshot")
    expected = {
        "model.safetensors": "730839316de585f4c8298ec0e1712efc10fb19c6fa4e36eb741cb8d51ebcf6aa",
        "audio_tokenizer/model.safetensors": "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d",
        "tokenizer.json": "408f669b7e2b045fdf54201d815bd364e6667dbd845115da81239c40bc6dcfd1",
    }
    for name, wanted in expected.items():
        digest = sha256_file(args.model / name)
        if digest != wanted:
            raise RuntimeError(f"pinned OmniVoice hash mismatch for {name}: {digest}")

    sys.path.insert(0, str(args.omnivoice_root))
    import numpy as np
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    device = "cuda:0" if args.device == "cuda" else "cpu"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = OmniVoice.from_pretrained(
        str(args.model), device_map=device, dtype=dtype, load_asr=False
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_records = []
    for ordinal, item in enumerate(records):
        seed = args.seed + ordinal
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        reference_path = Path(item["ref_audio"])
        if not reference_path.is_absolute():
            reference_path = args.reference_root / reference_path
        audio = model.generate(
            text=item["text"],
            language=item["language_id"],
            ref_audio=str(reference_path),
            ref_text=item["ref_text"],
            normalize_text=False,
            num_step=args.steps,
            guidance_scale=args.guidance,
            position_temperature=args.position_temperature,
            class_temperature=args.class_temperature,
        )[0]
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError(f"{item['id']}: Python oracle returned invalid audio")
        wav_path = args.output_dir / f"{item['id']}.wav"
        sf.write(wav_path, audio, model.sampling_rate, subtype="PCM_16")
        result = dict(item)
        result.update(
            {
                "wav_path": str(wav_path),
                "backend": "python-oracle",
                "seed": seed,
                "sample_rate": int(model.sampling_rate),
                "samples": int(audio.size),
                "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            }
        )
        output_records.append(result)
        print(f"[{ordinal + 1}/{len(records)}] {item['id']} {audio.size} samples")
    write_jsonl(args.output_manifest, output_records)


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    rows = [[(0, 0, 0, 0)] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1):
        rows[i][0] = (i, i, 0, 0)
    for j in range(1, len(hypothesis) + 1):
        rows[0][j] = (j, 0, j, 0)
    for i, ref in enumerate(reference, 1):
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                rows[i][j] = rows[i - 1][j - 1]
            else:
                deletion = rows[i - 1][j]
                insertion = rows[i][j - 1]
                substitution = rows[i - 1][j - 1]
                choices = [
                    (deletion[0] + 1, deletion[1] + 1, deletion[2], deletion[3]),
                    (insertion[0] + 1, insertion[1], insertion[2] + 1, insertion[3]),
                    (
                        substitution[0] + 1,
                        substitution[1],
                        substitution[2],
                        substitution[3] + 1,
                    ),
                ]
                rows[i][j] = min(choices)
    _, deletions, insertions, substitutions = rows[-1][-1]
    return deletions, insertions, substitutions, len(reference)


def command_score(args: argparse.Namespace) -> None:
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    import zhconv
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    text_normalize = load_omnivoice_text_normalize(args.omnivoice_root)

    revision = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    device = "cuda:0" if args.device == "cuda" else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.whisper_model, revision=revision, torch_dtype=dtype
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(args.whisper_model, revision=revision)
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )
    records = load_jsonl(args.manifest)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    expected_languages = {item["omnivoice_id"] for item in spec["languages"]}
    actual_languages = {item["language_id"] for item in records}
    expected_count = len(spec["languages"]) * int(spec["utterances_per_language"])
    if len(records) != expected_count or actual_languages != expected_languages:
        raise RuntimeError(
            f"scoring manifest must contain all {expected_count} pinned utterances and languages"
        )
    details, totals = [], defaultdict(lambda: [0, 0])
    failed = 0
    empty = 0
    for item in records:
        audio, rate = sf.read(item["wav_path"], dtype="float32", always_2d=True)
        mono = np.mean(audio, axis=1, dtype=np.float32)
        if rate != 16000:
            mono = librosa.resample(mono, orig_sr=rate, target_sr=16000)
        asr_failed = False
        try:
            recognize_options = {
                "generate_kwargs": {
                    "language": item["whisper_language"],
                    "task": "transcribe",
                }
            }
            if mono.size > 30 * 16000:
                recognize_options["return_timestamps"] = True
            raw_hypothesis = recognizer(
                {"array": mono, "sampling_rate": 16000},
                **recognize_options,
            )["text"].strip()
        except Exception as error:
            raw_hypothesis, failed, asr_failed = "", failed + 1, True
            print(f"ASR failure for {item['id']}: {error}", file=sys.stderr)
        if not raw_hypothesis:
            empty += 1

        def normalize(text: str) -> str:
            value = (
                text_normalize(
                    text,
                    iso_code=item["iso639_3"],
                    lower_case=True,
                    remove_numbers=False,
                    remove_brackets=False,
                )
                .lower()
                .strip()
            )
            value = " ".join(value.split())
            language = item["language_id"]
            if language in ("zh", "yue"):
                value = zhconv.convert(value, "zh-cn")
            if language in ("zh", "yue", "ja"):
                value = " ".join(value.replace(" ", ""))
            elif language in ("ko", "th", "arb", "vi", "hi", "el"):
                value = " ".join(value.replace(" ", "|"))
            return value.lower().strip()

        truth, hypothesis = normalize(item["text"]), normalize(raw_hypothesis)
        truth_tokens, hypothesis_tokens = truth.split(), hypothesis.split()
        deletions, insertions, substitutions, reference_units = edit_counts(
            truth_tokens, hypothesis_tokens
        )
        errors = deletions + insertions + substitutions
        totals[item["language_id"]][0] += errors
        totals[item["language_id"]][1] += reference_units
        details.append(
            dict(
                item,
                raw_hypothesis=raw_hypothesis,
                asr_failed=asr_failed,
                normalized_truth=truth,
                normalized_hypothesis=hypothesis,
                deletions=deletions,
                insertions=insertions,
                substitutions=substitutions,
                reference_units=reference_units,
                error_rate=errors / max(reference_units, 1),
            )
        )
    write_jsonl(args.output_details, details)
    per_language = {
        language: {
            "errors": value[0],
            "reference_units": value[1],
            "error_rate": value[0] / max(value[1], 1),
        }
        for language, value in sorted(totals.items())
    }
    micro_errors = sum(value[0] for value in totals.values())
    micro_units = sum(value[1] for value in totals.values())
    summary = {
        "asr_model": args.whisper_model,
        "asr_revision": revision,
        "utterances": len(details),
        "failed_asr": failed,
        "empty_asr": empty,
        "failed_or_empty_asr": empty,
        "micro_error_rate": micro_errors / max(micro_units, 1),
        "macro_language_error_rate": sum(v["error_rate"] for v in per_language.values())
        / max(len(per_language), 1),
        "per_language": per_language,
    }
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_compare(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    failures = []
    if candidate.get("failed_or_empty_asr", 0):
        failures.append("candidate has failed/empty ASR jobs")
    macro_delta = candidate["macro_language_error_rate"] - baseline["macro_language_error_rate"]
    if macro_delta > 0.01:
        failures.append(f"macro regression {macro_delta:.4%} exceeds 1.0 point")
    for language, expected in baseline["per_language"].items():
        if language not in candidate["per_language"]:
            failures.append(f"missing language {language}")
            continue
        delta = candidate["per_language"][language]["error_rate"] - expected["error_rate"]
        if delta > 0.05:
            failures.append(f"{language} regression {delta:.4%} exceeds 5.0 points")
    if failures:
        raise SystemExit("\n".join(failures))
    print("candidate satisfies MiniMax WER/CER regression thresholds")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.set_defaults(func=command_select)
    select.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    select.add_argument("--source-root", type=Path, required=True)
    select.add_argument("--reference-manifest", type=Path)
    select.add_argument("--output", type=Path, required=True)
    synth = sub.add_parser("synthesize")
    synth.set_defaults(func=command_synthesize)
    synth.add_argument("--library", type=Path, required=True)
    synth.add_argument("--model", type=Path, required=True)
    synth.add_argument("--codec", type=Path, required=True)
    synth.add_argument("--manifest", type=Path, required=True)
    synth.add_argument("--reference-root", type=Path, default=Path("."))
    synth.add_argument("--output-dir", type=Path, required=True)
    synth.add_argument("--output-manifest", type=Path, required=True)
    synth.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    synth.add_argument("--steps", type=int, default=32)
    synth.add_argument("--guidance", type=float, default=2.0)
    synth.add_argument("--position-temperature", type=float, default=5.0)
    synth.add_argument("--class-temperature", type=float, default=0.0)
    synth.add_argument("--seed", type=int, default=20260903)
    oracle = sub.add_parser("oracle-synthesize")
    oracle.set_defaults(func=command_oracle_synthesize)
    oracle.add_argument("--omnivoice-root", type=Path, required=True)
    oracle.add_argument("--model", type=Path, required=True)
    oracle.add_argument("--manifest", type=Path, required=True)
    oracle.add_argument("--reference-root", type=Path, default=Path("."))
    oracle.add_argument("--output-dir", type=Path, required=True)
    oracle.add_argument("--output-manifest", type=Path, required=True)
    oracle.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    oracle.add_argument("--steps", type=int, default=32)
    oracle.add_argument("--guidance", type=float, default=2.0)
    oracle.add_argument("--position-temperature", type=float, default=5.0)
    oracle.add_argument("--class-temperature", type=float, default=0.0)
    oracle.add_argument("--seed", type=int, default=20260903)
    score = sub.add_parser("score")
    score.set_defaults(func=command_score)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--omnivoice-root", type=Path, required=True)
    score.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    score.add_argument("--whisper-model", default="openai/whisper-large-v3-turbo")
    score.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    score.add_argument("--output-details", type=Path, required=True)
    score.add_argument("--output-summary", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.set_defaults(func=command_compare)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
