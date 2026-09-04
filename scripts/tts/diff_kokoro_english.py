#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differentially exercise pinned Misaki English entries against native C++.

This is a developer-only acceptance tool. It starts the C++ frontend once and
then sends every selected lexicon spelling over stdin, avoiding one GGUF load
per case. Use ``--limit 0`` for the exhaustive release-gate run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

MISAKI_COMMIT = "fba1236595f2d2bf21d414ba6e57d25256afada3"
DATA_SHA256 = {
    "us_gold.json": "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f",
    "us_silver.json": "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475",
    "gb_gold.json": "29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71",
    "gb_silver.json": "48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--misaki-root", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--native-runtime", type=Path, required=True)
    result.add_argument("--language", choices=("en-US", "en-GB"), required=True)
    result.add_argument("--offset", type=int, default=0)
    result.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="number of sorted unique entries to check; 0 checks every entry",
    )
    result.add_argument("--max-mismatches", type=int, default=20)
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parser().parse_args()
    if args.offset < 0 or args.limit < 0 or args.max_mismatches < 1:
        raise ValueError("offset/limit must be nonnegative and max-mismatches positive")
    head = subprocess.check_output(
        ["git", "-C", str(args.misaki_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != MISAKI_COMMIT:
        raise RuntimeError(f"Misaki checkout must be pinned to {MISAKI_COMMIT}, got {head}")

    prefix = "us" if args.language == "en-US" else "gb"
    data_root = args.misaki_root / "misaki" / "data"
    spellings: set[str] = set()
    for tier in ("gold", "silver"):
        name = f"{prefix}_{tier}.json"
        path = data_root / name
        actual = sha256(path)
        if actual != DATA_SHA256[name]:
            raise RuntimeError(f"pinned Misaki data hash mismatch for {name}: {actual}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"pinned Misaki data is not an object: {name}")
        spellings.update(value)
    words = sorted(word for word in spellings if "\n" not in word and "\r" not in word)
    words = words[args.offset : None if args.limit == 0 else args.offset + args.limit]
    if not words:
        raise ValueError("selected differential range is empty")

    sys.path.insert(0, str(args.misaki_root))
    from misaki import en, espeak  # pylint: disable=import-outside-toplevel

    british = args.language == "en-GB"
    oracle = en.G2P(trf=False, british=british, fallback=espeak.EspeakFallback(british=british))
    oracle_results = [oracle(word) for word in words]
    expected = [result[0] for result in oracle_results]
    voice = "bf_emma" if british else "af_heart"
    completed = subprocess.run(
        [
            str(args.native_runtime),
            str(args.model),
            "--prepare-stdin",
            args.language,
            voice,
        ],
        input="\n".join(words) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"native frontend exited {completed.returncode}:\n{completed.stderr[-4000:]}"
        )
    actual = completed.stdout.splitlines()
    if len(actual) != len(words):
        raise RuntimeError(f"native frontend returned {len(actual)} lines for {len(words)} inputs")

    mismatches = []
    for index, (word, want, got, oracle_result) in enumerate(
        zip(words, expected, actual, oracle_results)
    ):
        if want == got:
            continue
        mismatches.append(
            {
                "index": args.offset + index,
                "text": word,
                "misaki": want,
                "native": got,
                "misaki_tokens": [
                    {
                        "text": token.text,
                        "tag": token.tag,
                        "whitespace": token.whitespace,
                        "phonemes": token.phonemes,
                    }
                    for token in oracle_result[1]
                ],
            }
        )
    summary = {
        "language": args.language,
        "offset": args.offset,
        "checked": len(words),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[: args.max_mismatches],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
