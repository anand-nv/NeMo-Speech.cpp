#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "tts" / "export_kokoro_frontend_reference.py"
SPEC = importlib.util.spec_from_file_location("kokoro_reference_export", SCRIPT)
assert SPEC and SPEC.loader
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class KokoroReferenceExportTest(unittest.TestCase):
    def test_checked_in_corpus_covers_languages_and_boundaries(self) -> None:
        cases = json.loads(
            (ROOT / "scripts" / "tts" / "kokoro_reference_cases.json").read_text(encoding="utf-8")
        )
        self.assertEqual({case["kokoro_language"] for case in cases}, set("abefhijpz"))
        boundaries = {
            len(case["phonemes"]) for case in cases if case.get("id", "").startswith("boundary-")
        }
        self.assertEqual(boundaries, {1, 400, 509, 510})
        self.assertTrue(any(case["id"] == "en-us-long-paragraph" for case in cases))

    def test_tensor_payload_is_little_endian_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = EXPORT._store_tensor(  # pylint: disable=protected-access
                root, "bert/output", np.asarray([1.25, -2.5], dtype=">f4")
            )
            payload = (root / entry["path"]).read_bytes()
        self.assertEqual(entry["dtype"], "<f4")
        self.assertEqual(entry["shape"], [2])
        self.assertEqual(entry["nbytes"], 8)
        self.assertEqual(payload, struct.pack("<ff", 1.25, -2.5))
        self.assertEqual(len(entry["sha256"]), 64)

    def test_observable_token_fields_are_json_stable(self) -> None:
        value = EXPORT._json_value(  # pylint: disable=protected-access
            {"z": (2, None), "a": {"b": True}}
        )
        self.assertEqual(value, {"a": {"b": True}, "z": [2, None]})
        self.assertEqual(
            json.dumps(value, sort_keys=True),
            '{"a": {"b": true}, "z": [2, null]}',
        )

    def test_counter_based_source_is_repeatable(self) -> None:
        import torch

        coarse = torch.tensor([80.0, 120.0], dtype=torch.float32)
        sampled = coarse.repeat_interleave(300).reshape(1, 600, 1)
        first = EXPORT._native_sine_source(sampled, 1234)  # pylint: disable=protected-access
        second = EXPORT._native_sine_source(sampled, 1234)  # pylint: disable=protected-access
        self.assertEqual(
            [tuple(value.shape) for value in first], [(1, 600, 9), (1, 600, 1), (1, 600, 9)]
        )
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
        self.assertFalse(
            torch.equal(
                first[0],
                EXPORT._native_sine_source(sampled, 4321)[0],  # pylint: disable=protected-access
            )
        )

    def test_model_root_requires_published_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / EXPORT.KOKORO_MODEL_NAME).write_bytes(b"not-kokoro")
            (root / "voices").mkdir()
            with self.assertRaisesRegex(RuntimeError, "checkpoint SHA256"):
                EXPORT._validate_model_root(root)  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
