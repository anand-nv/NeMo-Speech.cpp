# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the pinned k2-fsa OmniVoice denoiser and Qwen tokenizer to GGUF."""

from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path
from typing import Any

import numpy as np

from .omnivoice_tables import (
    DURATION_RANGES,
    DURATION_WEIGHTS,
    INSTRUCTION_CATEGORIES,
    INSTRUCTION_EN_TO_ZH,
    NONVERBAL_TAGS,
    language_tables,
)
from .registry import ConversionRequest
from .source import read_json_config, resolve_hugging_face_source

MODEL_REPO = "k2-fsa/OmniVoice"
MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
MODEL_SHA256 = "730839316de585f4c8298ec0e1712efc10fb19c6fa4e36eb741cb8d51ebcf6aa"
AUDIO_TOKENIZER_SHA256 = "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d"
TOKENIZER_SHA256 = "408f669b7e2b045fdf54201d815bd364e6667dbd845115da81239c40bc6dcfd1"

REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
    "audio_tokenizer/preprocessor_config.json",
)

EXPECTED_FILE_INFO = {
    "model.safetensors": (2_450_344_112, MODEL_SHA256),
    "audio_tokenizer/model.safetensors": (805_665_628, AUDIO_TOKENIZER_SHA256),
    "tokenizer.json": (11_423_986, TOKENIZER_SHA256),
}

PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_revision(value: str) -> bool:
    return len(value) == 40 and all(char in string.hexdigits for char in value)


def _snapshot_revision(root: Path, request: ConversionRequest) -> str:
    for part in (root, *root.parents):
        if _looks_like_revision(part.name):
            return part.name.lower()
        metadata = part / ".cache" / "huggingface" / "download" / "config.json.metadata"
        if metadata.is_file():
            try:
                revision = metadata.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, UnicodeDecodeError, IndexError):
                revision = ""
            if _looks_like_revision(revision):
                return revision.lower()
    return request.revision or "local"


def _effective_revision(request: ConversionRequest) -> str | None:
    if request.source == MODEL_REPO:
        return request.revision or MODEL_REVISION
    return request.revision


def resolve_snapshot(request: ConversionRequest) -> Path:
    revision = _effective_revision(request)
    root = resolve_hugging_face_source(
        request.source,
        request.cache_dir,
        revision,
        allow_patterns=list(REQUIRED_SNAPSHOT_FILES),
    )
    missing = [name for name in REQUIRED_SNAPSHOT_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError("OmniVoice snapshot is incomplete; missing " + ", ".join(missing))
    return root


def _validate_pinned_files(root: Path, revision: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    pinned = revision == MODEL_REVISION
    for name, (expected_size, expected_hash) in EXPECTED_FILE_INFO.items():
        path = root / name
        size = path.stat().st_size
        # A local synthetic/exported checkpoint is allowed. A file claiming to
        # be the pinned artifact (by revision or exact release size) is not.
        check_hash = pinned or size == expected_size
        actual_hash = _sha256(path)
        hashes[name] = actual_hash
        if check_hash and (size != expected_size or actual_hash != expected_hash):
            raise RuntimeError(
                f"pinned OmniVoice artifact checksum mismatch: {name}; "
                f"size={size} sha256={actual_hash}"
            )
    return hashes


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm_config")
    if config.get("model_type") != "omnivoice" or not isinstance(llm, dict):
        raise RuntimeError("source is not an OmniVoice checkpoint")

    expected = {
        "audio_vocab_size": 1025,
        "audio_mask_id": 1024,
        "num_audio_codebook": 8,
    }
    llm_expected = {
        "model_type": "qwen3",
        "vocab_size": 151676,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1.0e-6,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"unsupported OmniVoice {key}: {config.get(key)!r}")
    for key, value in llm_expected.items():
        if llm.get(key) != value:
            raise RuntimeError(f"unsupported OmniVoice llm_config.{key}: {llm.get(key)!r}")
    rope = llm.get("rope_parameters")
    if not isinstance(rope, dict) or rope.get("rope_theta") != 1_000_000:
        raise RuntimeError("unsupported OmniVoice RoPE parameters")

    weights = config.get("audio_codebook_weights")
    if weights != [8, 8, 6, 6, 4, 4, 2, 2]:
        raise RuntimeError(f"unsupported OmniVoice audio_codebook_weights: {weights!r}")
    return llm


def parse_tokenizer(path: Path, text_vocab_size: int) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read OmniVoice tokenizer: {path}") from error

    model = data.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise RuntimeError("OmniVoice tokenizer must use BPE")
    if data.get("normalizer") != {"type": "NFC"}:
        raise RuntimeError("OmniVoice tokenizer must use the NFC normalizer")
    expected_pre = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": PRETOKENIZER_REGEX},
                "behavior": "Isolated",
                "invert": False,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": False,
            },
        ],
    }
    if data.get("pre_tokenizer") != expected_pre:
        raise RuntimeError("unsupported OmniVoice tokenizer pre-tokenizer")

    vocab = model.get("vocab")
    merges = model.get("merges")
    added = data.get("added_tokens")
    if not isinstance(vocab, dict) or not isinstance(merges, list) or not isinstance(added, list):
        raise RuntimeError("malformed OmniVoice BPE model")

    base_vocab_size = len(vocab)
    ordered: list[str | None] = [None] * base_vocab_size
    for token, token_id in vocab.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise RuntimeError("malformed OmniVoice vocabulary entry")
        if token_id < 0 or token_id >= base_vocab_size or ordered[token_id] is not None:
            raise RuntimeError(f"duplicate or non-contiguous OmniVoice token ID: {token_id}")
        ordered[token_id] = token
    if any(token is None for token in ordered):
        raise RuntimeError("OmniVoice base vocabulary has missing token IDs")

    added_by_id: dict[int, dict[str, Any]] = {}
    contents = set(vocab)
    for item in added:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise RuntimeError("malformed OmniVoice added token")
        token_id = item["id"]
        content = item.get("content")
        if (
            not isinstance(content, str)
            or token_id in added_by_id
            or content in contents
            or content in {value.get("content") for value in added_by_id.values()}
        ):
            raise RuntimeError("duplicate or malformed OmniVoice added token")
        added_by_id[token_id] = item
    expected_added_ids = list(range(base_vocab_size, text_vocab_size))
    if sorted(added_by_id) != expected_added_ids:
        raise RuntimeError("OmniVoice added token IDs are not contiguous")

    merge_left: list[str] = []
    merge_right: list[str] = []
    seen_merges: set[tuple[str, str]] = set()
    for item in merges:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
        ):
            raise RuntimeError("malformed OmniVoice BPE merge")
        pair = (item[0], item[1])
        if pair in seen_merges:
            raise RuntimeError(f"duplicate OmniVoice BPE merge: {pair!r}")
        seen_merges.add(pair)
        merge_left.append(pair[0])
        merge_right.append(pair[1])

    ordered_added = [added_by_id[token_id] for token_id in expected_added_ids]
    tokens = [str(token) for token in ordered] + [item["content"] for item in ordered_added]
    if len(tokens) != text_vocab_size or len(set(tokens)) != len(tokens):
        raise RuntimeError("OmniVoice combined vocabulary is malformed")
    return {
        "tokens": tokens,
        "base_vocab_size": base_vocab_size,
        "merge_left": merge_left,
        "merge_right": merge_right,
        "added": ordered_added,
    }


def expected_tensor_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    llm = validate_config(config)
    hidden = int(llm["hidden_size"])
    ffn = int(llm["intermediate_size"])
    heads = int(llm["num_attention_heads"])
    kv_heads = int(llm["num_key_value_heads"])
    head_dim = int(llm["head_dim"])
    layers = int(llm["num_hidden_layers"])
    audio_rows = int(config["num_audio_codebook"]) * int(config["audio_vocab_size"])

    shapes: dict[str, tuple[int, ...]] = {
        "codebook_layer_offsets": (int(config["num_audio_codebook"]),),
        "audio_embeddings.weight": (audio_rows, hidden),
        "audio_heads.weight": (audio_rows, hidden),
        "llm.embed_tokens.weight": (int(llm["vocab_size"]), hidden),
        "llm.norm.weight": (hidden,),
    }
    for layer in range(layers):
        prefix = f"llm.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.mlp.down_proj.weight": (hidden, ffn),
                f"{prefix}.mlp.gate_proj.weight": (ffn, hidden),
                f"{prefix}.mlp.up_proj.weight": (ffn, hidden),
                f"{prefix}.self_attn.q_proj.weight": (heads * head_dim, hidden),
                f"{prefix}.self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                f"{prefix}.self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, heads * head_dim),
                f"{prefix}.self_attn.q_norm.weight": (head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (head_dim,),
            }
        )
    return shapes


def _validate_tensor_file(path: Path, config: dict[str, Any]) -> list[str]:
    from safetensors import safe_open

    expected = expected_tensor_shapes(config)
    with safe_open(path, framework="pt", device="cpu") as source:
        names = list(source.keys())
        if set(names) != set(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise RuntimeError(
                f"OmniVoice tensor set mismatch; missing={missing[:3]} extra={extra[:3]}"
            )
        for name in sorted(names):
            tensor_slice = source.get_slice(name)
            shape = tuple(int(value) for value in tensor_slice.get_shape())
            dtype = tensor_slice.get_dtype()
            if shape != expected[name]:
                raise RuntimeError(
                    f"OmniVoice tensor shape mismatch: {name}={shape}, expected={expected[name]}"
                )
            if name == "codebook_layer_offsets":
                if dtype != "I64":
                    raise RuntimeError(f"OmniVoice tensor dtype mismatch: {name}={dtype}")
            elif dtype != "F32":
                raise RuntimeError(f"OmniVoice tensor dtype mismatch: {name}={dtype}")
    return sorted(names)


def _add_metadata(
    writer: Any,
    config: dict[str, Any],
    tokenizer: dict[str, Any],
    revision: str,
    hashes: dict[str, str],
) -> None:
    llm = validate_config(config)
    language_ids, language_names = language_tables()
    added = tokenizer["added"]

    writer.add_name("k2-fsa OmniVoice")
    writer.add_description("OmniVoice diffusion-language-model TTS denoiser and text frontend")
    writer.add_source_url(f"https://huggingface.co/{MODEL_REPO}/tree/{MODEL_REVISION}")
    writer.add_license("cc-by-nc")
    writer.add_license_name("Creative Commons Attribution-NonCommercial")
    writer.add_license_link(
        f"https://huggingface.co/{MODEL_REPO}/blob/{MODEL_REVISION}/README.md#license"
    )
    writer.add_string("omnivoice.source.revision", revision)
    writer.add_string("omnivoice.source.model_sha256", hashes["model.safetensors"])
    writer.add_string("omnivoice.source.tokenizer_sha256", hashes["tokenizer.json"])
    writer.add_string(
        "omnivoice.source.audio_tokenizer_sha256",
        hashes["audio_tokenizer/model.safetensors"],
    )
    writer.add_string("omnivoice.config_json", json.dumps(config, sort_keys=True))

    writer.add_int32("omnivoice.text_vocab_size", int(llm["vocab_size"]))
    writer.add_int32("omnivoice.tokenizer.base_vocab_size", tokenizer["base_vocab_size"])
    writer.add_string("omnivoice.tokenizer.model", "BPE")
    writer.add_string("omnivoice.tokenizer.normalizer", "NFC")
    writer.add_string("omnivoice.tokenizer.pre_tokenizer", "Qwen2 regex + ByteLevel")
    writer.add_string("omnivoice.tokenizer.pre_tokenizer_regex", PRETOKENIZER_REGEX)
    writer.add_array("omnivoice.tokenizer.tokens", tokenizer["tokens"])
    writer.add_array("omnivoice.tokenizer.merge_left", tokenizer["merge_left"])
    writer.add_array("omnivoice.tokenizer.merge_right", tokenizer["merge_right"])
    writer.add_array("omnivoice.tokenizer.added.ids", [item["id"] for item in added])
    writer.add_array("omnivoice.tokenizer.added.content", [item["content"] for item in added])
    for flag in ("single_word", "lstrip", "rstrip", "normalized", "special"):
        writer.add_array(f"omnivoice.tokenizer.added.{flag}", [bool(item[flag]) for item in added])

    special = [item for item in added if item["special"]]
    writer.add_array("omnivoice.tokenizer.special.ids", [item["id"] for item in special])
    writer.add_array("omnivoice.tokenizer.special.content", [item["content"] for item in special])
    writer.add_int32("omnivoice.tokenizer.pad_id", int(config["pad_token_id"]))
    writer.add_int32("omnivoice.tokenizer.eos_id", int(config["eos_token_id"]))

    writer.add_int32("omnivoice.audio_vocab_size", int(config["audio_vocab_size"]))
    writer.add_int32("omnivoice.audio_mask_id", int(config["audio_mask_id"]))
    writer.add_int32("omnivoice.audio_codebooks", int(config["num_audio_codebook"]))
    writer.add_array("omnivoice.audio_codebook_weights", config["audio_codebook_weights"])
    writer.add_array(
        "omnivoice.audio_codebook_offsets",
        [index * int(config["audio_vocab_size"]) for index in range(8)],
    )

    writer.add_int32("omnivoice.embedding_length", int(llm["hidden_size"]))
    writer.add_int32("omnivoice.block_count", int(llm["num_hidden_layers"]))
    writer.add_int32("omnivoice.attention.head_count", int(llm["num_attention_heads"]))
    writer.add_int32("omnivoice.attention.head_count_kv", int(llm["num_key_value_heads"]))
    writer.add_int32("omnivoice.attention.key_length", int(llm["head_dim"]))
    writer.add_int32("omnivoice.feed_forward_length", int(llm["intermediate_size"]))
    writer.add_float32("omnivoice.attention.layer_norm_rms_epsilon", llm["rms_norm_eps"])
    writer.add_float32("omnivoice.rope.freq_base", llm["rope_parameters"]["rope_theta"])
    writer.add_int32("omnivoice.context_length", int(llm["max_position_embeddings"]))

    defaults: tuple[tuple[str, Any], ...] = (
        ("steps", 32),
        ("guidance_scale", 2.0),
        ("time_shift", 0.1),
        ("layer_penalty", 5.0),
        ("position_temperature", 5.0),
        ("class_temperature", 0.0),
        ("denoise", True),
        ("preprocess_prompt", True),
        ("postprocess_output", True),
        ("chunk_duration", 15.0),
        ("chunk_threshold", 30.0),
        ("pad_duration", 0.1),
        ("fade_duration", 0.1),
    )
    for name, value in defaults:
        key = f"omnivoice.inference.{name}"
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_int32(key, value)
        else:
            writer.add_float32(key, value)

    writer.add_array("omnivoice.languages.ids", language_ids)
    writer.add_array("omnivoice.languages.names", language_names)
    writer.add_array("omnivoice.nonverbal_tags", list(NONVERBAL_TAGS))
    writer.add_array("omnivoice.instructions.en_to_zh", list(INSTRUCTION_EN_TO_ZH))
    writer.add_array(
        "omnivoice.instructions.categories",
        [json.dumps(values, ensure_ascii=False) for values in INSTRUCTION_CATEGORIES],
    )
    writer.add_array("omnivoice.duration.weights", list(DURATION_WEIGHTS))
    writer.add_array("omnivoice.duration.range_ends", [end for end, _ in DURATION_RANGES])
    writer.add_array("omnivoice.duration.range_types", [kind for _, kind in DURATION_RANGES])
    writer.add_string("omnivoice.duration.reference_text", "Nice to meet you.")
    writer.add_int32("omnivoice.duration.reference_frames", 25)


def _numpy_tensor(tensor: Any, outtype: str) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.is_floating_point():
        if tensor.ndim > 1 and outtype == "f16":
            tensor = tensor.to(dtype=__import__("torch").float16)
        else:
            tensor = tensor.float()
    return tensor.numpy()


def _write_tensors(writer: Any, path: Path, names: list[str], outtype: str) -> int:
    from safetensors import safe_open

    # Register tensor info first, then materialize and write one tensor at a
    # time. This keeps peak converter memory independent of checkpoint size.
    with safe_open(path, framework="pt", device="cpu") as source:
        for name in names:
            tensor_slice = source.get_slice(name)
            shape = tuple(int(value) for value in tensor_slice.get_shape())
            if tensor_slice.get_dtype() == "I64":
                dtype = np.dtype(np.int64)
            elif outtype == "f16" and len(shape) > 1:
                dtype = np.dtype(np.float16)
            else:
                dtype = np.dtype(np.float32)
            writer.add_tensor_info(name, shape, dtype, int(np.prod(shape)) * dtype.itemsize)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    with safe_open(path, framework="pt", device="cpu") as source:
        for name in names:
            writer.write_tensor_data(_numpy_tensor(source.get_tensor(name), outtype))
    return len(names)


def convert(request: ConversionRequest, outtype: str = "f16") -> None:
    import gguf

    root = resolve_snapshot(request)
    config = read_json_config(root)
    llm = validate_config(config)
    tokenizer = parse_tokenizer(root / "tokenizer.json", int(llm["vocab_size"]))
    names = _validate_tensor_file(root / "model.safetensors", config)
    revision = _snapshot_revision(root, request)
    hashes = _validate_pinned_files(root, revision)

    request.outfile.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(request.outfile, "omnivoice")
    _add_metadata(writer, config, tokenizer, revision, hashes)
    n_written = _write_tensors(writer, root / "model.safetensors", names, outtype)
    writer.close()

    summary = {
        "architecture": "omnivoice",
        "output": str(request.outfile),
        "outtype": outtype,
        "revision": revision,
        "source_hashes": hashes,
        "tensors_written": n_written,
        "token_count": len(tokenizer["tokens"]),
        "merge_count": len(tokenizer["merge_left"]),
    }
    if request.metadata_json:
        request.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        request.metadata_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote {request.outfile}")
    print(f"stored {n_written} OmniVoice tensors and {len(tokenizer['tokens'])} text tokens")
