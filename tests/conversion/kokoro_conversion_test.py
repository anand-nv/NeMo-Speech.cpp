#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conversion.kokoro import (  # noqa: E402
    KOKORO_CONTEXT_LENGTH,
    KOKORO_PHONEME_LIMIT,
    KOKORO_STYLE_DIM,
    KOKORO_V1_CONFIG,
    KOKORO_V1_ISTFTNET,
    KOKORO_V1_PLBERT,
    LANGUAGE_BY_PREFIX,
    MISAKI_COMMIT,
    MISAKI_DATA_SHA256,
    MISAKI_VERSION,
    REQUIRED_GROUPS,
    VOICE_NAMES,
    _load_voices,
    _read_config,
    _runtime_tensor_name,
    _tensor_numpy,
    _tokens_by_id,
    _voice_language,
    add_metadata,
    add_stft_constants,
    flatten_and_bake,
)


def valid_config() -> dict:
    return {
        "istftnet": {
            "upsample_kernel_sizes": [20, 12],
            "upsample_rates": [10, 6],
            "gen_istft_hop_size": 5,
            "gen_istft_n_fft": 20,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_initial_channel": 512,
        },
        "plbert": {
            "hidden_size": 768,
            "num_attention_heads": 12,
            "intermediate_size": 2048,
            "max_position_embeddings": KOKORO_CONTEXT_LENGTH,
            "num_hidden_layers": 12,
        },
        "vocab": {";": 1, "a": 43, "ᵻ": 177},
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


class KokoroConversionTest(unittest.TestCase):
    def test_runtime_tensor_names_fit_ggml(self) -> None:
        source = (
            "kokoro.bert.encoder.albert_layer_groups.0.albert_layers.0."
            "full_layer_layer_norm.weight"
        )
        actual = _runtime_tensor_name(source)
        self.assertEqual(
            actual,
            "kokoro.bert.encoder.g0.l0.full_layer_layer_norm.weight",
        )
        self.assertLess(len(actual.encode("utf-8")), 64)

    def test_config_and_sparse_vocabulary_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(valid_config()), encoding="utf-8")
            config = _read_config(root)
        tokens = _tokens_by_id(config)
        self.assertEqual(len(tokens), 178)
        self.assertEqual(tokens[0], "")
        self.assertEqual(tokens[1], ";")
        self.assertEqual(tokens[43], "a")
        self.assertEqual(tokens[177], "ᵻ")

    def test_config_rejects_wrong_context_or_vocabulary(self) -> None:
        for path, value, message in (
            (("n_token",), 177, "vocabulary size"),
            (("style_dim",), 64, "style half-dimension"),
            (("plbert", "max_position_embeddings"), 256, "context length"),
        ):
            config = valid_config()
            target = config
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    _read_config(root)

    def test_config_rejects_every_incompatible_v1_dimension(self) -> None:
        sections = (
            (None, KOKORO_V1_CONFIG),
            ("plbert", KOKORO_V1_PLBERT),
            ("istftnet", KOKORO_V1_ISTFTNET),
        )
        for section, expected_values in sections:
            for name, expected in expected_values.items():
                with self.subTest(section=section, name=name):
                    config = valid_config()
                    target = config if section is None else config[section]
                    if isinstance(expected, int):
                        target[name] = expected + 1
                    else:
                        changed = json.loads(json.dumps(expected))
                        changed[0] = changed[0] + 1 if isinstance(changed[0], int) else [9]
                        target[name] = changed
                    with tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, "unsupported Kokoro"):
                            _read_config(root)

    def test_config_rejects_missing_fixed_dimensions(self) -> None:
        for section, name in (
            (None, "max_conv_dim"),
            ("plbert", "hidden_size"),
            ("istftnet", "upsample_rates"),
        ):
            with self.subTest(section=section, name=name):
                config = valid_config()
                del (config if section is None else config[section])[name]
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "missing"):
                        _read_config(root)

    def test_config_rejects_coercible_non_integer_dimensions(self) -> None:
        for value in (64.5, "64", True):
            with self.subTest(value=value):
                config = valid_config()
                config["dim_in"] = value
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "must be an integer"):
                        _read_config(root)

    def test_voice_order_and_language_mapping_are_stable(self) -> None:
        self.assertEqual(len(VOICE_NAMES), 54)
        self.assertEqual(tuple(sorted(VOICE_NAMES)), VOICE_NAMES)
        self.assertEqual(_voice_language("af_heart"), "en-US")
        self.assertEqual(_voice_language("bm_fable"), "en-GB")
        self.assertEqual(_voice_language("zf_xiaoxiao"), "zh-CN")
        self.assertEqual(set(map(_voice_language, VOICE_NAMES)), set(LANGUAGE_BY_PREFIX.values()))
        with self.assertRaisesRegex(ValueError, "unsupported Kokoro voice"):
            _voice_language("unknown")

    def test_voice_loader_requires_exact_published_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            voice_dir = root / "voices"
            voice_dir.mkdir()
            torch.save(
                torch.zeros(KOKORO_PHONEME_LIMIT, 1, KOKORO_STYLE_DIM), voice_dir / "af_heart.pt"
            )
            with self.assertRaisesRegex(ValueError, "voice set mismatch"):
                _load_voices(root)

    def test_flatten_bakes_legacy_weight_normalization(self) -> None:
        groups = {name: {} for name in REQUIRED_GROUPS}
        groups["decoder"] = {
            "module.block.weight_g": torch.tensor([[[2.0]], [[3.0]]]),
            "module.block.weight_v": torch.tensor(
                [[[3.0, 4.0]], [[0.0, 5.0]]], dtype=torch.float32
            ),
            "module.block.bias": torch.tensor([1.0, -1.0]),
        }
        tensors = flatten_and_bake(groups)
        self.assertEqual(
            set(tensors),
            {"kokoro.decoder.block.weight", "kokoro.decoder.block.bias"},
        )
        expected = torch.tensor([[[1.2, 1.6]], [[0.0, 3.0]]])
        torch.testing.assert_close(tensors["kokoro.decoder.block.weight"], expected)

    def test_missing_weight_normalization_pair_is_rejected(self) -> None:
        groups = {name: {} for name in REQUIRED_GROUPS}
        groups["decoder"] = {"module.block.weight_g": torch.ones(2, 1, 1)}
        with self.assertRaisesRegex(ValueError, "weight-normalization pair"):
            flatten_and_bake(groups)

    def test_stft_constants_are_self_contained(self) -> None:
        tensors: dict[str, torch.Tensor] = {}
        add_stft_constants(tensors, valid_config())
        self.assertEqual(len(tensors), 4)
        self.assertEqual(
            tensors["kokoro.decoder.generator.stft.forward_real"].shape,
            (11, 1, 20),
        )
        self.assertEqual(
            tensors["kokoro.decoder.generator.stft.inverse_imag"].dtype,
            torch.float32,
        )

    def test_f16_keeps_recurrent_bias_norm_and_voice_f32(self) -> None:
        matrix = torch.ones(2, 2)
        self.assertEqual(_tensor_numpy("kokoro.bert.linear.weight", matrix, "f16").dtype, "float16")
        self.assertEqual(
            _tensor_numpy("kokoro.predictor.lstm.weight_ih_l0", matrix, "f16").dtype, "float32"
        )
        self.assertEqual(
            _tensor_numpy(
                "kokoro.predictor.text_encoder.lstms.0.bias_hh_l0_reverse",
                matrix[0],
                "f16",
            ).dtype,
            "float32",
        )
        self.assertEqual(
            _tensor_numpy("kokoro.decoder.linear.bias", matrix[0], "f16").dtype, "float32"
        )
        self.assertEqual(
            _tensor_numpy(
                "kokoro.bert.encoder.g0.l0.full_layer_layer_norm.weight",
                matrix[0],
                "f16",
            ).dtype,
            "float32",
        )
        self.assertEqual(_tensor_numpy("kokoro.voice.af_heart", matrix, "f16").dtype, "float32")

    def test_metadata_records_model_contract(self) -> None:
        writer = mock.Mock()
        summary = add_metadata(writer, valid_config(), "abc", "local", "revision")
        self.assertEqual(summary["architecture"], "kokoro")
        self.assertEqual(summary["voice_count"], 54)
        writer.add_int32.assert_any_call("kokoro.sample_rate", 24_000)
        writer.add_int32.assert_any_call("kokoro.context_length", KOKORO_CONTEXT_LENGTH)
        writer.add_int32.assert_any_call("kokoro.style_dim", 128)
        writer.add_int32.assert_any_call("kokoro.voice.style_dim", KOKORO_STYLE_DIM)
        writer.add_string.assert_any_call("kokoro.lstm.gate_order", "i,f,g,o")
        writer.add_string.assert_any_call("kokoro.istftnet.window", "hann_periodic")
        writer.add_string.assert_any_call("kokoro.noise.generator", "splitmix64-box-muller-v1")
        writer.add_string.assert_any_call("kokoro.tokenizer.misaki.version", MISAKI_VERSION)
        writer.add_string.assert_any_call("kokoro.tokenizer.misaki.commit", MISAKI_COMMIT)
        writer.add_array.assert_any_call("kokoro.voice.names", list(VOICE_NAMES))
        writer.add_array.assert_any_call(
            "kokoro.voice.languages", [_voice_language(name) for name in VOICE_NAMES]
        )

    def test_metadata_requires_exact_misaki_data_set_and_hashes(self) -> None:
        writer = mock.Mock()
        with self.assertRaisesRegex(ValueError, "tokenizer data set is incomplete"):
            add_metadata(
                writer,
                valid_config(),
                "abc",
                "local",
                "revision",
                misaki_data={},
            )
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            add_metadata(
                writer,
                valid_config(),
                "abc",
                "local",
                "revision",
                misaki_data={name: "bad" for name in MISAKI_DATA_SHA256},
            )


if __name__ == "__main__":
    unittest.main()
