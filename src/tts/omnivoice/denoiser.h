// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace ggml_runtime {
class BackendManager;
class GGUFLoader;
class Module;
class Session;
}  // namespace ggml_runtime

namespace nemo_speech::tts::omnivoice {

struct DenoiserInput {
    int32_t batch_size = 0;
    int32_t sequence_length = 0;

    // GGML-compatible row-major buffers. Text/audio-mask layout is [B,S];
    // shifted-audio layout is [B,8,S], and attention-mask layout is
    // [B,S(query),S(key)]. Mask values are 0 for visible entries and a large
    // negative value for padding/document boundaries.
    std::vector<int32_t> text_ids;
    std::vector<int32_t> shifted_audio_ids;
    std::vector<float> audio_mask;
    std::vector<int32_t> position_ids;
    std::vector<float> attention_mask;
};

struct DenoiserOutput {
    int32_t batch_size = 0;
    int32_t codebooks = 0;
    int32_t sequence_length = 0;
    int32_t vocabulary_size = 0;
    // Contiguous [B,codebook,S,vocabulary] logits.
    std::vector<float> logits;
};

// Backend-neutral GGML implementation of the 28-layer bidirectional Qwen3
// denoiser. BackendManager selects CPU or CUDA and Session schedules the same
// graph on either backend.
class Denoiser {
   public:
    Denoiser(ggml_runtime::BackendManager& backends, const std::string& gguf_path);
    ~Denoiser();

    Denoiser(const Denoiser&) = delete;
    Denoiser& operator=(const Denoiser&) = delete;

    const ModelConfig& config() const { return config_; }
    DenoiserOutput forward(const DenoiserInput& input);

   private:
    ModelConfig config_;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<ggml_runtime::Module> module_;
    std::unique_ptr<ggml_runtime::Session> session_;
};

}  // namespace nemo_speech::tts::omnivoice
