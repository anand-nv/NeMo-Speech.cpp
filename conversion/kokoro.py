#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the published hexgrad/Kokoro-82M v1.0 checkpoint to GGUF."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import gguf
import numpy as np
import torch

KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_MODEL = "kokoro-v1_0.pth"
KOKORO_V1_SHA256 = "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
KOKORO_SAMPLE_RATE = 24_000
KOKORO_CONTEXT_LENGTH = 512
KOKORO_PHONEME_LIMIT = 510
KOKORO_STYLE_DIM = 256
MISAKI_VERSION = "0.9.4"
MISAKI_COMMIT = "fba1236595f2d2bf21d414ba6e57d25256afada3"
MISAKI_DATA_SHA256 = {
    "us_gold.json": "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f",
    "us_silver.json": "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475",
    "gb_gold.json": "29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71",
    "gb_silver.json": "48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce",
    "ja_words.txt": "a93a8e8aee24db307a32becb8bf01c4c2908ecf37e6c91f7a705fafdfeba67ff",
}

VOICE_NAMES = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
)

LANGUAGE_BY_PREFIX = {
    "a": "en-US",
    "b": "en-GB",
    "e": "es-ES",
    "f": "fr-FR",
    "h": "hi-IN",
    "i": "it-IT",
    "j": "ja-JP",
    "p": "pt-BR",
    "z": "zh-CN",
}

REQUIRED_GROUPS = ("bert", "bert_encoder", "predictor", "decoder", "text_encoder")
VOICE_RE = re.compile(r"^[abefhijpz][fm]_[a-z0-9_]+$")

KOKORO_V1_CONFIG = {
    "dim_in": 64,
    "hidden_dim": 512,
    "max_conv_dim": 512,
    "max_dur": 50,
    "n_layer": 3,
    "n_mels": 80,
    "n_token": 178,
    "style_dim": 128,
    "text_encoder_kernel_size": 5,
}
KOKORO_V1_PLBERT = {
    "hidden_size": 768,
    "intermediate_size": 2048,
    "max_position_embeddings": KOKORO_CONTEXT_LENGTH,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}
KOKORO_V1_ISTFTNET = {
    "gen_istft_hop_size": 5,
    "gen_istft_n_fft": 20,
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "resblock_kernel_sizes": [3, 7, 11],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [20, 12],
    "upsample_rates": [10, 6],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_misaki_data(cache_dir: Path) -> dict[str, str]:
    """Fetch and verify immutable English lexicons used by Kokoro's Misaki path."""
    data_dir = cache_dir / "misaki" / MISAKI_COMMIT / "misaki" / "data"
    result: dict[str, str] = {}
    for filename, expected_hash in MISAKI_DATA_SHA256.items():
        path = data_dir / filename
        payload = path.read_bytes() if path.is_file() else b""
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            url = (
                "https://raw.githubusercontent.com/hexgrad/misaki/"
                f"{MISAKI_COMMIT}/misaki/data/{filename}"
            )
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
            except (OSError, urllib.error.URLError) as error:
                raise RuntimeError(
                    f"failed to fetch pinned Misaki tokenizer data {filename}: {error}"
                ) from error
            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"pinned Misaki tokenizer data hash mismatch for {filename}: "
                    f"{actual_hash}; expected {expected_hash}"
                )
            data_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"invalid pinned Misaki tokenizer data {filename}") from error
        if filename.endswith(".json"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid pinned Misaki tokenizer data {filename}") from error
            if not isinstance(parsed, dict) or not parsed:
                raise RuntimeError(f"pinned Misaki tokenizer data {filename} is not an object")
        elif not text.strip():
            raise RuntimeError(f"pinned Misaki tokenizer data {filename} is empty")
        result[filename] = text
    return result


def _read_config(root: Path) -> dict[str, Any]:
    path = root / "config.json"
    if not path.is_file():
        raise RuntimeError(f"Kokoro source contains no config.json: {root}")
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise RuntimeError(f"Kokoro config is not an object: {path}")
    required = {
        "istftnet",
        "plbert",
        "vocab",
        "dim_in",
        "hidden_dim",
        "max_conv_dim",
        "max_dur",
        "n_layer",
        "n_mels",
        "n_token",
        "style_dim",
        "text_encoder_kernel_size",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError("Kokoro config is missing: " + ", ".join(missing))
    if not all(isinstance(config[name], dict) for name in ("istftnet", "plbert", "vocab")):
        raise ValueError("Kokoro istftnet, plbert, and vocab config entries must be objects")

    def require_integer(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Kokoro config {label} must be an integer")
        return value

    for name, expected in KOKORO_V1_CONFIG.items():
        actual = require_integer(config[name], name)
        if actual != expected:
            label = {
                "n_token": "vocabulary size",
                "style_dim": "style half-dimension",
            }.get(name, name)
            raise ValueError(f"unsupported Kokoro {label}: {config[name]} (expected {expected})")

    for section, expected_values in (
        ("plbert", KOKORO_V1_PLBERT),
        ("istftnet", KOKORO_V1_ISTFTNET),
    ):
        values = config[section]
        missing_nested = sorted(expected_values.keys() - values.keys())
        if missing_nested:
            raise ValueError(f"Kokoro {section} config is missing: " + ", ".join(missing_nested))
        for name, expected in expected_values.items():
            actual = values[name]
            if isinstance(expected, int):
                actual = require_integer(actual, f"{section}.{name}")
            if actual != expected:
                label = (
                    "context length"
                    if section == "plbert" and name == "max_position_embeddings"
                    else f"{section}.{name}"
                )
                raise ValueError(
                    f"unsupported Kokoro {label}: {values[name]} (expected {expected})"
                )
    return config


def _resolve_source(source: str, revision: str | None, cache_dir: Path) -> tuple[Path, str | None]:
    local = Path(source).expanduser()
    if local.exists():
        if not local.is_dir():
            raise RuntimeError("Kokoro source must be a directory or Hugging Face repository")
        return local, None

    from huggingface_hub import snapshot_download

    print(f"[download] fetching Kokoro model and voices from {source}")
    snapshot = Path(
        snapshot_download(
            repo_id=source,
            revision=revision,
            cache_dir=str(cache_dir),
            allow_patterns=["config.json", KOKORO_MODEL, "voices/*.pt"],
        )
    )
    # huggingface_hub resolves refs to an immutable snapshot directory named
    # with the full commit hash. Record that resolved revision in the GGUF even
    # when the caller supplied a branch, tag, or abbreviated hash.
    resolved_revision = snapshot.name if re.fullmatch(r"[0-9a-f]{40}", snapshot.name) else revision
    return snapshot, resolved_revision


def _load_state(root: Path) -> tuple[dict[str, dict[str, torch.Tensor]], Path, str]:
    path = root / KOKORO_MODEL
    if not path.is_file():
        raise RuntimeError(f"Kokoro source contains no {KOKORO_MODEL}: {root}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise RuntimeError(f"Kokoro checkpoint is not a mapping: {path}")
    missing = [name for name in REQUIRED_GROUPS if not isinstance(value.get(name), dict)]
    if missing:
        raise ValueError("Kokoro checkpoint is missing model groups: " + ", ".join(missing))
    groups = {name: value[name] for name in REQUIRED_GROUPS}
    return groups, path, _sha256(path)


def _voice_language(name: str) -> str:
    if not VOICE_RE.fullmatch(name) or name[0] not in LANGUAGE_BY_PREFIX:
        raise ValueError(f"unsupported Kokoro voice name: {name}")
    return LANGUAGE_BY_PREFIX[name[0]]


def _load_voices(root: Path) -> dict[str, torch.Tensor]:
    voice_dir = root / "voices"
    if not voice_dir.is_dir():
        raise RuntimeError(f"Kokoro source contains no voices directory: {root}")
    paths = {path.stem: path for path in voice_dir.glob("*.pt") if path.is_file()}
    missing = sorted(set(VOICE_NAMES) - paths.keys())
    extra = sorted(paths.keys() - set(VOICE_NAMES))
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unexpected=" + ",".join(extra))
        raise ValueError("Kokoro v1.0 voice set mismatch: " + " ".join(details))

    voices: dict[str, torch.Tensor] = {}
    for name in VOICE_NAMES:
        _voice_language(name)
        value = torch.load(paths[name], map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Kokoro voice {name} is not a tensor")
        if tuple(value.shape) != (KOKORO_PHONEME_LIMIT, 1, KOKORO_STYLE_DIM):
            raise ValueError(
                f"Kokoro voice {name} has shape {tuple(value.shape)}; "
                f"expected ({KOKORO_PHONEME_LIMIT}, 1, {KOKORO_STYLE_DIM})"
            )
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Kokoro voice {name} must contain finite floating-point values")
        voices[name] = value[:, 0, :].to(torch.float32).contiguous()
    return voices


def _strip_module(name: str) -> str:
    return name[7:] if name.startswith("module.") else name


def _runtime_tensor_name(name: str) -> str:
    """Use semantic abbreviations required by GGML's 63-byte name limit.

    Only the shared ALBERT layer prefix exceeds GGML_MAX_NAME in v1.0. Keep
    every module component while abbreviating the two repeated container names.
    """
    name = re.sub(
        r"\.albert_layer_groups\.(\d+)\.albert_layers\.(\d+)\.",
        r".g\1.l\2.",
        name,
    )
    if len(name.encode("utf-8")) >= 64:
        raise ValueError(f"converted Kokoro tensor name exceeds GGML_MAX_NAME: {name}")
    return name


def _bake_weight_norm(g: torch.Tensor, v: torch.Tensor, name: str) -> torch.Tensor:
    if g.shape[0] != v.shape[0]:
        raise ValueError(f"weight-normalization output dimension mismatch for {name}")
    dims = tuple(range(1, v.ndim))
    if not dims:
        norm = v.abs()
    else:
        norm = torch.linalg.vector_norm(v.to(torch.float32), dim=dims, keepdim=True)
    if torch.any(norm == 0):
        raise ValueError(f"weight-normalization vector contains a zero norm: {name}")
    return (v.to(torch.float32) * (g.to(torch.float32) / norm)).contiguous()


def flatten_and_bake(groups: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Flatten checkpoint groups and bake legacy PyTorch weight normalization."""
    flat: dict[str, torch.Tensor] = {}
    for group in REQUIRED_GROUPS:
        state = groups[group]
        normalized = {_strip_module(name): tensor for name, tensor in state.items()}
        consumed: set[str] = set()
        for name in sorted(normalized):
            if name in consumed:
                continue
            tensor = normalized[name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Kokoro checkpoint value is not a tensor: {group}.{name}")
            if name.endswith(".weight_g"):
                stem = name[: -len(".weight_g")]
                v_name = stem + ".weight_v"
                if v_name not in normalized:
                    raise ValueError(f"weight-normalization pair is missing {group}.{v_name}")
                value = _bake_weight_norm(tensor, normalized[v_name], f"{group}.{stem}")
                output_name = _runtime_tensor_name(f"kokoro.{group}.{stem}.weight")
                consumed.update((name, v_name))
            elif name.endswith(".weight_v"):
                stem = name[: -len(".weight_v")]
                if stem + ".weight_g" not in normalized:
                    raise ValueError(
                        f"weight-normalization pair is missing {group}.{stem}.weight_g"
                    )
                continue
            else:
                value = tensor.detach().cpu().contiguous()
                output_name = _runtime_tensor_name(f"kokoro.{group}.{name}")
                consumed.add(name)
            if output_name in flat:
                raise ValueError(f"duplicate converted Kokoro tensor: {output_name}")
            flat[output_name] = value
    return flat


def add_stft_constants(tensors: dict[str, torch.Tensor], config: dict[str, Any]) -> None:
    """Materialize the TorchSTFT Hann/DFT constants required by native GGML.

    The published checkpoint constructs these non-parameter buffers at runtime,
    so they are not present in its state dict. Keeping them in the GGUF makes
    the decoder graph self-contained and fixes their numerical convention.
    """
    n_fft = int(config["istftnet"]["gen_istft_n_fft"])
    bins = n_fft // 2 + 1
    n = torch.arange(n_fft, dtype=torch.float64)
    k = torch.arange(bins, dtype=torch.float64).unsqueeze(1)
    angle = 2.0 * torch.pi * k * n / n_fft
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
    forward_real = (torch.cos(angle) * window).to(torch.float32).unsqueeze(1)
    forward_imag = (-torch.sin(angle) * window).to(torch.float32).unsqueeze(1)

    scale = torch.full((bins, 1), 2.0 / n_fft, dtype=torch.float64)
    scale[0] = 1.0 / n_fft
    if n_fft % 2 == 0:
        scale[-1] = 1.0 / n_fft
    inverse_real = (torch.cos(angle) * window * scale).to(torch.float32).unsqueeze(1)
    inverse_imag = (torch.sin(angle) * window * scale).to(torch.float32).unsqueeze(1)
    constants = {
        "kokoro.decoder.generator.stft.forward_real": forward_real,
        "kokoro.decoder.generator.stft.forward_imag": forward_imag,
        "kokoro.decoder.generator.stft.inverse_real": inverse_real,
        "kokoro.decoder.generator.stft.inverse_imag": inverse_imag,
    }
    overlap = set(tensors).intersection(constants)
    if overlap:
        raise ValueError(f"Kokoro STFT tensor collision: {sorted(overlap)}")
    tensors.update(constants)


def _tokens_by_id(config: dict[str, Any]) -> list[str]:
    count = int(config["n_token"])
    tokens = [""] * count
    for symbol, raw_index in config["vocab"].items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Kokoro vocabulary symbols must be non-empty strings")
        index = int(raw_index)
        if index <= 0 or index >= count:
            raise ValueError(f"Kokoro vocabulary ID outside 1..{count - 1}: {symbol!r}={index}")
        if tokens[index]:
            raise ValueError(f"duplicate Kokoro vocabulary ID {index}")
        tokens[index] = symbol
    return tokens


def _store_f32(name: str) -> bool:
    return (
        name.startswith("kokoro.voice.")
        or ".lstm." in name
        or any(
            marker in name for marker in (".weight_ih_", ".weight_hh_", ".bias_ih_", ".bias_hh_")
        )
        or name.endswith((".bias", ".gamma", ".beta"))
        or ".LayerNorm." in name
        or "layer_norm." in name.lower()
        or ".norm." in name
        or name.endswith((".window", ".weight_forward_real", ".weight_forward_imag"))
        or name.endswith((".weight_backward_real", ".weight_backward_imag"))
        or ".generator.stft." in name
        or ".alpha1." in name
        or ".alpha2." in name
    )


def _tensor_numpy(name: str, tensor: torch.Tensor, outtype: str) -> np.ndarray:
    value = tensor.detach().cpu().contiguous()
    if value.is_floating_point():
        value = value.to(torch.float32 if outtype == "f32" or _store_f32(name) else torch.float16)
    elif value.dtype == torch.int64:
        value = value.to(torch.int32)
    return value.numpy()


def add_metadata(
    writer: gguf.GGUFWriter,
    config: dict[str, Any],
    model_hash: str,
    source: str,
    revision: str | None,
    model_tensors: dict[str, torch.Tensor] | None = None,
    outtype: str = "f16",
    misaki_data: dict[str, str] | None = None,
) -> dict[str, Any]:
    tokens = _tokens_by_id(config)
    voice_languages = [_voice_language(name) for name in VOICE_NAMES]
    writer.add_name("Kokoro-82M v1.0")
    writer.add_description("Kokoro StyleTTS2/iSTFTNet text-to-speech model")
    writer.add_string("kokoro.source.repository", source)
    writer.add_string("kokoro.source.revision", revision or "")
    writer.add_string("kokoro.source.sha256", model_hash)
    writer.add_string("kokoro.config_json", json.dumps(config, ensure_ascii=False, sort_keys=True))
    writer.add_int32("kokoro.sample_rate", KOKORO_SAMPLE_RATE)
    writer.add_int32("kokoro.context_length", KOKORO_CONTEXT_LENGTH)
    writer.add_int32("kokoro.phoneme_limit", KOKORO_PHONEME_LIMIT)
    writer.add_int32("kokoro.vocab_size", int(config["n_token"]))
    writer.add_int32("kokoro.voice.style_dim", KOKORO_STYLE_DIM)
    writer.add_float32("kokoro.speed.default", 1.0)
    writer.add_float32("kokoro.speed.min", 0.5)
    writer.add_float32("kokoro.speed.max", 2.0)
    writer.add_string("kokoro.lstm.gate_order", "i,f,g,o")
    writer.add_string("kokoro.istftnet.window", "hann_periodic")
    writer.add_string("kokoro.noise.generator", "splitmix64-box-muller-v1")
    writer.add_string("kokoro.tokenizer.misaki.version", MISAKI_VERSION)
    writer.add_string("kokoro.tokenizer.misaki.commit", MISAKI_COMMIT)
    if misaki_data is not None:
        if set(misaki_data) != set(MISAKI_DATA_SHA256):
            raise ValueError("Misaki tokenizer data set is incomplete")
        for filename, expected_hash in MISAKI_DATA_SHA256.items():
            payload = misaki_data[filename]
            actual_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Misaki tokenizer data hash mismatch for {filename}: {actual_hash}"
                )
            stem = filename.removesuffix(".json").removesuffix(".txt")
            writer.add_string(f"kokoro.tokenizer.misaki.{stem}.sha256", expected_hash)
            suffix = "json" if filename.endswith(".json") else "text"
            writer.add_string(f"kokoro.tokenizer.misaki.{stem}.{suffix}", payload)
    writer.add_array("kokoro.tokenizer.tokens", tokens)
    writer.add_array("kokoro.voice.names", list(VOICE_NAMES))
    writer.add_array("kokoro.voice.languages", voice_languages)

    plbert = config["plbert"]
    istftnet = config["istftnet"]
    for key in (
        "dim_in",
        "hidden_dim",
        "max_conv_dim",
        "max_dur",
        "n_layer",
        "n_mels",
        "n_token",
        "style_dim",
        "text_encoder_kernel_size",
    ):
        if key in config:
            writer.add_int32(f"kokoro.{key}", int(config[key]))
    for key in (
        "hidden_size",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
        "num_hidden_layers",
    ):
        writer.add_int32(f"kokoro.plbert.{key}", int(plbert[key]))
    for key in (
        "gen_istft_hop_size",
        "gen_istft_n_fft",
        "upsample_initial_channel",
    ):
        writer.add_int32(f"kokoro.istftnet.{key}", int(istftnet[key]))
    for key in (
        "upsample_kernel_sizes",
        "upsample_rates",
        "resblock_kernel_sizes",
    ):
        writer.add_array(f"kokoro.istftnet.{key}", [int(x) for x in istftnet[key]])
    writer.add_array(
        "kokoro.istftnet.resblock_dilation_sizes",
        [int(x) for values in istftnet["resblock_dilation_sizes"] for x in values],
    )

    if model_tensors is not None:
        tensor_names = sorted(model_tensors)
        tensor_shapes = [
            ",".join(str(int(dimension)) for dimension in model_tensors[name].shape)
            for name in tensor_names
        ]
        tensor_types = [
            str(_tensor_numpy(name, model_tensors[name], outtype).dtype) for name in tensor_names
        ]
        writer.add_int32("kokoro.model.tensor_count", len(tensor_names))
        writer.add_array("kokoro.model.tensor_names", tensor_names)
        writer.add_array("kokoro.model.tensor_shapes", tensor_shapes)
        writer.add_array("kokoro.model.tensor_types", tensor_types)

    return {
        "architecture": "kokoro",
        "source": source,
        "revision": revision,
        "model_sha256": model_hash,
        "sample_rate": KOKORO_SAMPLE_RATE,
        "context_length": KOKORO_CONTEXT_LENGTH,
        "phoneme_limit": KOKORO_PHONEME_LIMIT,
        "vocab_size": int(config["n_token"]),
        "style_dim": int(config["style_dim"]),
        "voice_style_dim": KOKORO_STYLE_DIM,
        "voice_count": len(VOICE_NAMES),
        "voice_names": list(VOICE_NAMES),
        "voice_languages": voice_languages,
    }


def _write_tensors(
    writer: gguf.GGUFWriter,
    tensors: Iterable[tuple[str, torch.Tensor]],
    outtype: str,
) -> int:
    count = 0
    for name, tensor in tensors:
        writer.add_tensor(name, _tensor_numpy(name, tensor, outtype))
        count += 1
    return count


def convert(
    source: str,
    output: Path,
    outtype: str = "f16",
    metadata_json: Path | None = None,
    revision: str | None = None,
    cache_dir: Path = Path.home() / ".cache" / "huggingface" / "hub",
) -> None:
    if outtype not in ("f16", "f32"):
        raise ValueError("Kokoro supports only f16 and f32 output")
    root, resolved_revision = _resolve_source(source, revision, cache_dir)
    config = _read_config(root)
    groups, _model_path, model_hash = _load_state(root)
    if model_hash != KOKORO_V1_SHA256:
        raise ValueError(
            f"published Kokoro v1.0 SHA256 mismatch: {model_hash}; expected {KOKORO_V1_SHA256}"
        )
    voices = _load_voices(root)
    tensors = flatten_and_bake(groups)
    add_stft_constants(tensors, config)
    misaki_data = _load_misaki_data(cache_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(output, "kokoro")
    summary = add_metadata(
        writer,
        config,
        model_hash,
        source,
        resolved_revision or revision,
        model_tensors=tensors,
        outtype=outtype,
        misaki_data=misaki_data,
    )
    model_count = _write_tensors(writer, sorted(tensors.items()), outtype)
    voice_count = _write_tensors(
        writer,
        ((f"kokoro.voice.{name}", voices[name]) for name in VOICE_NAMES),
        outtype,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    summary.update(
        {
            "model_tensors_written": model_count,
            "voice_tensors_written": voice_count,
            "output": str(output),
            "outtype": outtype,
        }
    )
    if metadata_json:
        metadata_json.parent.mkdir(parents=True, exist_ok=True)
        metadata_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote {output}")
    print(f"stored {model_count} model tensors and {voice_count} voices")
