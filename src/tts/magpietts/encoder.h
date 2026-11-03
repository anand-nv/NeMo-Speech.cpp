// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "model.h"

namespace nemo_speech::tts {

class MagpieEncoder {
   public:
    explicit MagpieEncoder(const magpietts_model& model) : model_(model) {}
    MagpieEncoder(magpietts_model&&) = delete;

    bool eval(const std::vector<int32_t>& tokens, int threads, std::vector<float>& out) const;
    // backend/keep_allocr: optional side backend (own stream) and reusable graph allocator for
    // background evaluation (longform chunk prefetch).
    bool evalDevice(
        const std::vector<int32_t>& tokens, int threads, magpietts_backend_tensor& out,
        ggml_backend_t backend = nullptr, ggml_gallocr_t* keep_allocr = nullptr) const;

    // Several token windows in one graph (the in-flight batching engine's batched prefill): the
    // windows are padded at the end to the longest one, which the causal encoder never attends
    // to, and each window's rows are written straight into its outs[i] (allocated here). The
    // results equal evalDevice's up to the summation order of the wider matmuls. backend: optional
    // side backend (own stream) to compute on.
    bool evalDeviceBatch(
        const std::vector<const std::vector<int32_t>*>& windows, int threads,
        const std::vector<magpietts_backend_tensor*>& outs, ggml_gallocr_t* keep_allocr,
        ggml_backend_t backend = nullptr) const;

   private:
    const magpietts_model& model_;
};

}  // namespace nemo_speech::tts
