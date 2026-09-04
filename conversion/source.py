# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint discovery, download, and safe NeMo archive handling."""

from __future__ import annotations

import json
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import yaml

CHECKPOINT_CONFIG_NAMES = ("model_config.yaml", "config.yaml")


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"archive member escapes extraction directory: {member_name}") from error
    return destination


def extract_archive(archive: Path, destination: Path, *, basenames: set[str] | None = None) -> None:
    """Extract regular files without accepting traversal or link members."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            if basenames is not None and Path(member.name).name not in basenames:
                continue
            target = _safe_destination(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported archive member: {member.name}")
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)


@contextmanager
def extracted_checkpoint(source: Path, prefix: str = "nemo-speech-convert-") -> Iterator[Path]:
    if source.is_dir():
        yield source
        return
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        root = Path(temporary)
        extract_archive(source, root)
        yield root


def find_checkpoint_files(root: Path) -> dict[str, Path | None]:
    found: dict[str, Path | None] = {"config": None, "weights": None, "tokenizer": None}
    for path in root.rglob("*"):
        name = path.name
        if name in CHECKPOINT_CONFIG_NAMES and found["config"] is None:
            found["config"] = path
        elif name == "model_weights.ckpt" and found["weights"] is None:
            found["weights"] = path
        elif (
            name == "tokenizer.model" or (name.endswith(".model") and "tokenizer" in name.lower())
        ) and found["tokenizer"] is None:
            found["tokenizer"] = path
    return found


def read_checkpoint_config(root: Path) -> dict[str, Any]:
    files = find_checkpoint_files(root)
    path = files["config"]
    if path is None:
        raise RuntimeError(f"checkpoint contains no {' or '.join(CHECKPOINT_CONFIG_NAMES)}")
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint configuration is not a mapping: {path}")
    return value


def read_nemo_config(source: Path) -> dict[str, Any]:
    if source.is_dir():
        return read_checkpoint_config(source)
    with tarfile.open(source, "r:*") as tar:
        candidates = [
            member
            for member in tar.getmembers()
            if member.isfile() and Path(member.name).name in CHECKPOINT_CONFIG_NAMES
        ]
        if not candidates:
            raise RuntimeError(f"NeMo archive contains no configuration: {source}")
        candidates.sort(key=lambda member: CHECKPOINT_CONFIG_NAMES.index(Path(member.name).name))
        stream = tar.extractfile(candidates[0])
        if stream is None:
            raise RuntimeError(f"cannot read {candidates[0].name} from {source}")
        with stream:
            value = yaml.safe_load(stream.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint configuration is not a mapping: {source}")
    return value


def load_state_dict(path: Path, *, weights_only: bool = False) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=weights_only)
    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint state dictionary is not a mapping: {path}")
    return value


def list_hugging_face_files(repo_id: str, revision: str | None = None) -> list[str]:
    from huggingface_hub import HfApi

    return list(HfApi().list_repo_files(repo_id, revision=revision))


def read_json_config(path: Path) -> dict[str, Any]:
    """Read a Hugging Face ``config.json`` without importing Transformers."""
    config_path = path / "config.json" if path.is_dir() else path
    if config_path.name != "config.json" or not config_path.is_file():
        raise RuntimeError(f"Hugging Face source contains no config.json: {path}")
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read Hugging Face configuration: {config_path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Hugging Face configuration is not a mapping: {config_path}")
    return value


def download_hugging_face_config(
    repo_id: str, cache_dir: Path, revision: str | None = None
) -> Path:
    """Download only config.json for architecture detection."""
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            cache_dir=str(cache_dir),
        )
    )


def resolve_hugging_face_source(
    source: str,
    cache_dir: Path,
    revision: str | None = None,
    *,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Resolve a local HF directory or download a repository snapshot."""
    local = Path(source).expanduser()
    if local.exists():
        if not local.is_dir():
            raise RuntimeError(f"Hugging Face checkpoint source is not a directory: {local}")
        return local

    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=source,
            revision=revision,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns,
        )
    )


def download_nemo_checkpoint(repo_id: str, cache_dir: Path, revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    files = [name for name in list_hugging_face_files(repo_id, revision) if name.endswith(".nemo")]
    if not files:
        raise RuntimeError(f"no .nemo checkpoint found in Hugging Face repository {repo_id}")
    if len(files) > 1:
        raise RuntimeError(
            f"multiple .nemo checkpoints found in {repo_id}; download one locally or select an "
            "architecture-specific source"
        )
    print(f"[download] fetching {repo_id}/{files[0]}")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=files[0],
            revision=revision,
            cache_dir=str(cache_dir),
        )
    )


def resolve_nemo_source(source: str, cache_dir: Path, revision: str | None = None) -> Path:
    local = Path(source).expanduser()
    if local.exists():
        if not local.is_file() and not local.is_dir():
            raise RuntimeError(f"checkpoint source is not a file or directory: {local}")
        return local
    cache_dir.mkdir(parents=True, exist_ok=True)
    return download_nemo_checkpoint(source, cache_dir, revision)
