# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow torchaudio.functional compatibility for the pinned OmniVoice oracle.

TorchCodec owns current PyTorch media decoding, but pinned OmniVoice and the
Higgs tokenizer still import ``torchaudio.functional.resample`` for in-memory
sample-rate conversion.  This module installs only that band-limited sinc
operation and does not emulate TorchAudio media I/O.
"""

from __future__ import annotations

import math
import sys
import types

import torch
import torch.nn.functional as torch_functional


def resample(
    waveform,
    orig_freq,
    new_freq,
    lowpass_filter_width=6,
    rolloff=0.99,
    resampling_method="sinc_interp_hann",
    beta=None,
):
    """Match the TorchAudio band-limited sinc resampling convention."""
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError("sample rates must be positive")
    if orig_freq == new_freq:
        return waveform
    if resampling_method not in ("sinc_interp_hann", "sinc_interp_kaiser"):
        raise ValueError(f"unsupported resampling method: {resampling_method}")
    divisor = math.gcd(int(orig_freq), int(new_freq))
    orig = int(orig_freq) // divisor
    new = int(new_freq) // divisor
    base = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base)
    dtype = waveform.dtype
    device = waveform.device
    index = torch.arange(-width, width + orig, dtype=dtype, device=device)[None, None] / orig
    time = torch.arange(0, -new, -1, dtype=dtype, device=device)[:, None, None] / new + index
    time *= base
    time = time.clamp_(-lowpass_filter_width, lowpass_filter_width)
    if resampling_method == "sinc_interp_hann":
        window = torch.cos(time * math.pi / lowpass_filter_width / 2) ** 2
    else:
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = torch.tensor(float(beta), dtype=dtype, device=device)
        window = torch.i0(
            beta_tensor * torch.sqrt(1 - (time / lowpass_filter_width) ** 2)
        ) / torch.i0(beta_tensor)
    time *= math.pi
    kernels = torch.where(
        time == 0,
        torch.tensor(1.0, dtype=dtype, device=device),
        time.sin() / time,
    )
    kernels *= window * (base / orig)
    shape = waveform.size()
    packed = waveform.view(-1, shape[-1])
    packed = torch_functional.pad(packed, (width, width + orig))
    output = torch_functional.conv1d(packed[:, None], kernels, stride=orig)
    output = output.transpose(1, 2).reshape(packed.shape[0], -1)
    target_length = math.ceil(new * shape[-1] / orig)
    output = output[..., :target_length]
    return output.view(shape[:-1] + output.shape[-1:])


def install() -> None:
    """Install the minimal import surface expected by pinned OmniVoice."""
    module = types.ModuleType("torchaudio")
    module.functional = types.SimpleNamespace(resample=resample)
    sys.modules["torchaudio"] = module
