# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the Higgs Audio V2 prompt encoder and waveform decoder to GGUF."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .omnivoice import (
    AUDIO_TOKENIZER_SHA256,
    MODEL_REPO,
    MODEL_REVISION,
    _effective_revision,
    _sha256,
    _snapshot_revision,
)
from .registry import ConversionRequest
from .source import read_json_config, resolve_hugging_face_source

REQUIRED_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")
MODEL_SIZE = 805_665_628
GGML_MAX_NAME = 63

TRAINING_ONLY_SUFFIXES = (
    ".codebook.cluster_size",
    ".codebook.embed_avg",
    ".codebook.inited",
)


@dataclass(frozen=True)
class TensorPlan:
    target: str
    sources: tuple[str, ...]
    shape: tuple[int, ...]
    source_dtype: str
    materialize_weight_norm_dim: int | None = None


def resolve_snapshot(request: ConversionRequest) -> tuple[Path, Path]:
    """Return (repository root, audio-tokenizer root)."""
    revision = _effective_revision(request)
    root = resolve_hugging_face_source(
        request.source,
        request.cache_dir,
        revision,
        allow_patterns=[
            *REQUIRED_FILES,
            *(f"audio_tokenizer/{name}" for name in REQUIRED_FILES),
        ],
    )
    config = read_json_config(root)
    if config.get("model_type") == "omnivoice":
        codec_root = root / "audio_tokenizer"
    else:
        codec_root = root
    missing = [name for name in REQUIRED_FILES if not (codec_root / name).is_file()]
    if missing:
        raise RuntimeError("Higgs Audio V2 snapshot is incomplete; missing " + ", ".join(missing))
    return root, codec_root


def validate_config(config: dict[str, Any], preprocessor: dict[str, Any]) -> dict[str, int]:
    acoustic = config.get("acoustic_model_config")
    semantic = config.get("semantic_model_config")
    if config.get("model_type") != "higgs_audio_v2_tokenizer":
        raise RuntimeError("source is not a Higgs Audio V2 tokenizer checkpoint")
    if not isinstance(acoustic, dict) or not isinstance(semantic, dict):
        raise RuntimeError("Higgs Audio V2 nested model configuration is malformed")

    outer_expected = {
        "sample_rate": 24000,
        "codebook_size": 1024,
        "codebook_dim": 64,
        "downsample_factor": 320,
        "semantic_sample_rate": 16000,
        "kernel_size": 3,
        "unit_kernel_size": 3,
        "strides": [1, 1],
        "channel_ratios": [1, 1],
        "block_dilations": [1, 1],
        "target_bandwidths": [0.5, 1, 1.5, 2],
    }
    acoustic_expected = {
        "model_type": "dac",
        "encoder_hidden_size": 64,
        "decoder_hidden_size": 1024,
        "hidden_size": 256,
        "downsampling_ratios": [8, 5, 4, 2, 3],
        "upsampling_ratios": [8, 5, 4, 2, 3],
    }
    semantic_expected = {
        "model_type": "hubert",
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "conv_dim": [512] * 7,
        "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
        "conv_stride": [5, 2, 2, 2, 2, 2, 2],
        "num_conv_pos_embeddings": 128,
        "num_conv_pos_embedding_groups": 16,
        "layer_norm_eps": 1.0e-5,
    }
    for key, expected in outer_expected.items():
        if config.get(key) != expected:
            raise RuntimeError(f"unsupported Higgs Audio V2 {key}: {config.get(key)!r}")
    for key, expected in acoustic_expected.items():
        if acoustic.get(key) != expected:
            raise RuntimeError(
                f"unsupported Higgs Audio V2 acoustic_model_config.{key}: {acoustic.get(key)!r}"
            )
    for key, expected in semantic_expected.items():
        if semantic.get(key) != expected:
            raise RuntimeError(
                f"unsupported Higgs Audio V2 semantic_model_config.{key}: {semantic.get(key)!r}"
            )

    hop_length = math.prod(int(value) for value in acoustic["downsampling_ratios"])
    frame_rate = math.ceil(int(config["sample_rate"]) / hop_length)
    codebook_bits = math.ceil(math.log2(int(config["codebook_size"])))
    num_quantizers = int(
        1000 * float(config["target_bandwidths"][-1]) // (frame_rate * codebook_bits)
    )
    semantic_downsample = int(
        hop_length
        / (int(config["sample_rate"]) / int(config["semantic_sample_rate"]))
        / int(config["downsample_factor"])
    )
    if (hop_length, frame_rate, num_quantizers, semantic_downsample) != (960, 25, 8, 2):
        raise RuntimeError("inconsistent Higgs Audio V2 frame or quantizer configuration")
    if preprocessor != {
        "feature_extractor_type": "DacFeatureExtractor",
        "feature_size": 1,
        "hop_length": 960,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": True,
        "sampling_rate": 24000,
    }:
        raise RuntimeError("unsupported Higgs Audio V2 preprocessor configuration")
    return {
        "hop_length": hop_length,
        "frame_rate": frame_rate,
        "num_quantizers": num_quantizers,
        "semantic_downsample": semantic_downsample,
    }


def _manifest() -> dict[str, dict[str, Any]]:
    path = Path(__file__).with_name("higgs_audio_v2_tensor_manifest.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or len(value) != 527:
        raise RuntimeError("packaged Higgs Audio V2 tensor manifest is malformed")
    return value


def short_tensor_name(name: str) -> str:
    replacements = (
        ("acoustic_encoder.", "aenc."),
        ("acoustic_decoder.", "adec."),
        ("encoder_semantic.", "senc."),
        ("decoder_semantic.", "sdec."),
        ("quantizer.quantizers.", "rvq."),
        ("semantic_model.feature_extractor.conv_layers.", "hubert.fe."),
        ("semantic_model.feature_projection.", "hubert.fp."),
        ("semantic_model.encoder.layers.", "hubert.blk."),
        ("semantic_model.encoder.layer_norm.", "hubert.out_norm."),
        ("semantic_model.encoder.pos_conv_embed.conv.", "hubert.pos_conv."),
        (".feed_forward.intermediate_dense.", ".ffn.up."),
        (".feed_forward.output_dense.", ".ffn.down."),
        (".attention.out_proj.", ".attn.out."),
        (".attention.q_proj.", ".attn.q."),
        (".attention.k_proj.", ".attn.k."),
        (".attention.v_proj.", ".attn.v."),
        (".final_layer_norm.", ".final_norm."),
        (".layer_norm.", ".attn_norm."),
        (".codebook.", ".cb."),
        (".project_in.", ".in."),
        (".project_out.", ".out."),
        (".res_unit", ".ru"),
        (".conv_blocks.", ".blk."),
        (".res_units.", ".ru."),
    )
    result = name
    for old, new in replacements:
        result = result.replace(old, new)
    if result.endswith(".parametrizations.weight.original0"):
        result = result[: -len(".parametrizations.weight.original0")] + ".weight_g"
    elif result.endswith(".parametrizations.weight.original1"):
        result = result[: -len(".parametrizations.weight.original1")] + ".weight_v"
    if len(result.encode("utf-8")) >= GGML_MAX_NAME:
        raise RuntimeError(f"Higgs Audio V2 tensor name is too long after mapping: {result}")
    return result


def _plans(path: Path) -> tuple[list[TensorPlan], list[str]]:
    from safetensors import safe_open

    manifest = _manifest()
    with safe_open(path, framework="pt", device="cpu") as source:
        names = set(source.keys())
        if names != set(manifest):
            missing = sorted(set(manifest) - names)
            extra = sorted(names - set(manifest))
            raise RuntimeError(
                f"Higgs Audio V2 tensor set mismatch; missing={missing[:3]} extra={extra[:3]}"
            )
        for name, expected in manifest.items():
            tensor_slice = source.get_slice(name)
            actual_shape = [int(value) for value in tensor_slice.get_shape()]
            if actual_shape != expected["shape"] or tensor_slice.get_dtype() != expected["dtype"]:
                raise RuntimeError(
                    f"Higgs Audio V2 tensor mismatch: {name} "
                    f"shape={actual_shape} dtype={tensor_slice.get_dtype()}"
                )

    plans: list[TensorPlan] = []
    skipped: list[str] = []
    weight_g = "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"
    weight_v = "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"
    for name, info in manifest.items():
        if name.endswith(TRAINING_ONLY_SUFFIXES) or name == "semantic_model.masked_spec_embed":
            skipped.append(name)
            continue
        if name == weight_g:
            skipped.append(name)
            continue
        if name == weight_v:
            plans.append(
                TensorPlan(
                    target="hubert.pos_conv.weight",
                    sources=(weight_g, weight_v),
                    shape=tuple(info["shape"]),
                    source_dtype=info["dtype"],
                    materialize_weight_norm_dim=2,
                )
            )
            continue
        plans.append(
            TensorPlan(
                target=short_tensor_name(name),
                sources=(name,),
                shape=tuple(info["shape"]),
                source_dtype=info["dtype"],
            )
        )
    targets = [plan.target for plan in plans]
    if len(targets) != len(set(targets)):
        raise RuntimeError("Higgs Audio V2 tensor mapping produced duplicate names")
    return plans, skipped


def _preserve_f32(plan: TensorPlan) -> bool:
    source = plan.sources[-1]
    return (
        len(plan.shape) <= 1
        or source.endswith(".bias")
        or source.endswith(".alpha")
        or ".layer_norm." in source
        or ".codebook.embed" in source
    )


def _output_dtype(plan: TensorPlan, outtype: str) -> np.dtype[Any]:
    if plan.source_dtype != "F32":
        raise RuntimeError(f"unsupported Higgs Audio V2 source dtype: {plan.source_dtype}")
    if outtype == "f16" and not _preserve_f32(plan):
        return np.dtype(np.float16)
    return np.dtype(np.float32)


def materialize_weight_norm(weight_g: Any, weight_v: Any, dim: int) -> Any:
    """Match torch weight_norm: v * (g / norm_except_dim(v, dim))."""
    reduction_dims = tuple(index for index in range(weight_v.ndim) if index != dim)
    norm = weight_v.float().norm(2, dim=reduction_dims, keepdim=True).clamp_min(1.0e-12)
    return (weight_v.float() * (weight_g.float() / norm)).contiguous()


def _load_plan(source: Any, plan: TensorPlan, outtype: str) -> np.ndarray:
    if plan.materialize_weight_norm_dim is not None:
        tensor = materialize_weight_norm(
            source.get_tensor(plan.sources[0]),
            source.get_tensor(plan.sources[1]),
            plan.materialize_weight_norm_dim,
        )
    else:
        tensor = source.get_tensor(plan.sources[0]).detach().cpu().contiguous()
    if _output_dtype(plan, outtype) == np.dtype(np.float16):
        tensor = tensor.half()
    else:
        tensor = tensor.float()
    return tensor.numpy()


def _add_metadata(
    writer: Any,
    config: dict[str, Any],
    derived: dict[str, int],
    revision: str,
    model_hash: str,
) -> None:
    acoustic = config["acoustic_model_config"]
    semantic = config["semantic_model_config"]
    writer.add_name("Higgs Audio V2 Tokenizer for OmniVoice")
    writer.add_description("Higgs Audio V2 HuBERT/DAC prompt encoder and waveform decoder")
    writer.add_source_url(
        f"https://huggingface.co/{MODEL_REPO}/tree/{MODEL_REVISION}/audio_tokenizer"
    )
    writer.add_license("other")
    writer.add_license_name("Boson Higgs Audio 2 Community License Agreement")
    writer.add_license_link(
        f"https://huggingface.co/{MODEL_REPO}/blob/{MODEL_REVISION}/audio_tokenizer/LICENSE"
    )
    writer.add_string("higgs_audio_v2.source.revision", revision)
    writer.add_string("higgs_audio_v2.source.model_sha256", model_hash)
    writer.add_string("higgs_audio_v2.config_json", json.dumps(config, sort_keys=True))
    writer.add_string("higgs_audio_v2.tensor_layout", "PyTorch logical shapes; short GGML names")

    writer.add_int32("higgs_audio_v2.sample_rate", int(config["sample_rate"]))
    writer.add_int32("higgs_audio_v2.hop_length", derived["hop_length"])
    writer.add_float32("higgs_audio_v2.frame_rate", derived["frame_rate"])
    writer.add_int32("higgs_audio_v2.codebook_size", int(config["codebook_size"]))
    writer.add_int32("higgs_audio_v2.codebook_dim", int(config["codebook_dim"]))
    writer.add_int32("higgs_audio_v2.num_quantizers", derived["num_quantizers"])
    writer.add_array(
        "higgs_audio_v2.target_bandwidths",
        [float(value) for value in config["target_bandwidths"]],
    )
    writer.add_int32("higgs_audio_v2.semantic.sample_rate", int(config["semantic_sample_rate"]))
    writer.add_int32("higgs_audio_v2.semantic.pad_samples", 160)
    writer.add_int32("higgs_audio_v2.semantic.downsample_factor", derived["semantic_downsample"])
    writer.add_array("higgs_audio_v2.semantic.strides", config["strides"])
    writer.add_array("higgs_audio_v2.semantic.channel_ratios", config["channel_ratios"])
    writer.add_array("higgs_audio_v2.semantic.block_dilations", config["block_dilations"])
    writer.add_int32("higgs_audio_v2.semantic.kernel_size", int(config["kernel_size"]))
    writer.add_int32("higgs_audio_v2.semantic.unit_kernel_size", int(config["unit_kernel_size"]))

    writer.add_int32("higgs_audio_v2.hubert.hidden_size", int(semantic["hidden_size"]))
    writer.add_int32("higgs_audio_v2.hubert.intermediate_size", int(semantic["intermediate_size"]))
    writer.add_int32("higgs_audio_v2.hubert.layers", int(semantic["num_hidden_layers"]))
    writer.add_int32("higgs_audio_v2.hubert.heads", int(semantic["num_attention_heads"]))
    writer.add_float32("higgs_audio_v2.hubert.layer_norm_eps", semantic["layer_norm_eps"])
    writer.add_array("higgs_audio_v2.hubert.conv_dim", semantic["conv_dim"])
    writer.add_array("higgs_audio_v2.hubert.conv_kernel", semantic["conv_kernel"])
    writer.add_array("higgs_audio_v2.hubert.conv_stride", semantic["conv_stride"])
    writer.add_int32(
        "higgs_audio_v2.hubert.pos_conv_kernel",
        int(semantic["num_conv_pos_embeddings"]),
    )
    writer.add_int32(
        "higgs_audio_v2.hubert.pos_conv_groups",
        int(semantic["num_conv_pos_embedding_groups"]),
    )

    writer.add_int32("higgs_audio_v2.dac.encoder_hidden_size", acoustic["encoder_hidden_size"])
    writer.add_int32("higgs_audio_v2.dac.decoder_hidden_size", acoustic["decoder_hidden_size"])
    writer.add_int32("higgs_audio_v2.dac.hidden_size", acoustic["hidden_size"])
    writer.add_array("higgs_audio_v2.dac.downsampling_ratios", acoustic["downsampling_ratios"])
    writer.add_array("higgs_audio_v2.dac.upsampling_ratios", acoustic["upsampling_ratios"])
    writer.add_int32("higgs_audio_v2.dac.input_kernel_size", 7)
    writer.add_int32("higgs_audio_v2.dac.output_kernel_size", 7)
    writer.add_array("higgs_audio_v2.dac.residual_dilations", [1, 3, 9])
    writer.add_bool("higgs_audio_v2.dac.decoder_stride_parity_output_padding", True)
    writer.add_bool("higgs_audio_v2.dac.final_tanh", False)


def _write_tensors(writer: Any, path: Path, plans: list[TensorPlan], outtype: str) -> None:
    from safetensors import safe_open

    for plan in plans:
        dtype = _output_dtype(plan, outtype)
        writer.add_tensor_info(
            plan.target, plan.shape, dtype, int(np.prod(plan.shape)) * dtype.itemsize
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    with safe_open(path, framework="pt", device="cpu") as source:
        for plan in plans:
            writer.write_tensor_data(_load_plan(source, plan, outtype))


def convert(request: ConversionRequest, outtype: str = "f16") -> None:
    import gguf

    root, codec_root = resolve_snapshot(request)
    config = read_json_config(codec_root)
    preprocessor = json.loads((codec_root / "preprocessor_config.json").read_text(encoding="utf-8"))
    derived = validate_config(config, preprocessor)
    model_path = codec_root / "model.safetensors"
    plans, skipped = _plans(model_path)
    revision = _snapshot_revision(root, request)
    model_hash = _sha256(model_path)
    if (revision == MODEL_REVISION or model_path.stat().st_size == MODEL_SIZE) and (
        model_path.stat().st_size != MODEL_SIZE or model_hash != AUDIO_TOKENIZER_SHA256
    ):
        raise RuntimeError(
            "pinned Higgs Audio V2 artifact checksum mismatch: model.safetensors; "
            f"size={model_path.stat().st_size} sha256={model_hash}"
        )

    request.outfile.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(request.outfile, "higgs-audio-v2-tokenizer")
    _add_metadata(writer, config, derived, revision, model_hash)
    _write_tensors(writer, model_path, plans, outtype)
    writer.close()

    summary = {
        "architecture": "higgs-audio-v2-tokenizer",
        "output": str(request.outfile),
        "outtype": outtype,
        "revision": revision,
        "source_hash": model_hash,
        "tensors_written": len(plans),
        "tensors_skipped": skipped,
    }
    if request.metadata_json:
        request.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        request.metadata_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote {request.outfile}")
    print(f"stored {len(plans)} Higgs Audio V2 tensors; skipped {len(skipped)} training buffers")
