#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from conversion.higgs_audio_v2_tokenizer import (
    GGML_MAX_NAME,
    _manifest,
    materialize_weight_norm,
    short_tensor_name,
)
from conversion.higgs_audio_v2_tokenizer import (  # noqa: E402
    validate_config as validate_codec_config,
)
from conversion.omnivoice import (  # noqa: E402
    PRETOKENIZER_REGEX,
    expected_tensor_shapes,
    parse_tokenizer,
    validate_config,
)
from conversion.omnivoice_tables import language_tables  # noqa: E402


def main_config() -> dict:
    return {
        "model_type": "omnivoice",
        "audio_vocab_size": 1025,
        "audio_mask_id": 1024,
        "num_audio_codebook": 8,
        "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
        "llm_config": {
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
            "rope_parameters": {"rope_theta": 1_000_000},
        },
    }


def codec_config() -> tuple[dict, dict]:
    config = {
        "model_type": "higgs_audio_v2_tokenizer",
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
        "acoustic_model_config": {
            "model_type": "dac",
            "encoder_hidden_size": 64,
            "decoder_hidden_size": 1024,
            "hidden_size": 256,
            "downsampling_ratios": [8, 5, 4, 2, 3],
            "upsampling_ratios": [8, 5, 4, 2, 3],
        },
        "semantic_model_config": {
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
        },
    }
    preprocessor = {
        "feature_extractor_type": "DacFeatureExtractor",
        "feature_size": 1,
        "hop_length": 960,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": True,
        "sampling_rate": 24000,
    }
    return config, preprocessor


class OmniVoiceConverterTest(unittest.TestCase):
    def test_main_config_and_tensor_contract(self) -> None:
        config = main_config()
        self.assertEqual(validate_config(config)["hidden_size"], 1024)
        shapes = expected_tensor_shapes(config)
        self.assertEqual(len(shapes), 313)
        self.assertEqual(shapes["audio_embeddings.weight"], (8200, 1024))
        self.assertEqual(shapes["llm.layers.27.self_attn.q_proj.weight"], (2048, 1024))
        config["llm_config"]["head_dim"] = 64
        with self.assertRaisesRegex(RuntimeError, "head_dim"):
            validate_config(config)

    def test_tokenizer_contract_and_rejections(self) -> None:
        tokenizer = {
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {
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
            },
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]},
            "added_tokens": [
                {
                    "id": 2,
                    "content": "<special>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokenizer.json"
            path.write_text(json.dumps(tokenizer), encoding="utf-8")
            parsed = parse_tokenizer(path, 3)
            self.assertEqual(parsed["tokens"], ["a", "b", "<special>"])
            self.assertEqual(parsed["merge_left"], ["a"])

            tokenizer["model"]["vocab"] = {"a": 0, "b": 2}
            path.write_text(json.dumps(tokenizer), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-contiguous"):
                parse_tokenizer(path, 3)

    def test_codec_config_and_weight_norm(self) -> None:
        config, preprocessor = codec_config()
        self.assertEqual(
            validate_codec_config(config, preprocessor),
            {
                "hop_length": 960,
                "frame_rate": 25,
                "num_quantizers": 8,
                "semantic_downsample": 2,
            },
        )
        weight_g = torch.tensor([[[5.0, 13.0]]])
        weight_v = torch.tensor([[[3.0, 5.0]], [[4.0, 12.0]]])
        actual = materialize_weight_norm(weight_g, weight_v, dim=2)
        expected = torch.tensor([[[3.0, 5.0]], [[4.0, 12.0]]])
        torch.testing.assert_close(actual, expected)

    def test_codec_manifest_names_are_complete_and_ggml_safe(self) -> None:
        manifest = _manifest()
        self.assertEqual(len(manifest), 527)
        mapped = [short_tensor_name(name) for name in manifest]
        self.assertTrue(all(len(name.encode("utf-8")) < GGML_MAX_NAME for name in mapped))

    def test_language_table_is_complete(self) -> None:
        ids, names = language_tables()
        self.assertEqual(len(ids), 646)
        self.assertEqual(len(names), 646)
        self.assertEqual(ids[0], "aae")
        self.assertIn("en", ids)
        self.assertIn("English", names)


if __name__ == "__main__":
    unittest.main()
