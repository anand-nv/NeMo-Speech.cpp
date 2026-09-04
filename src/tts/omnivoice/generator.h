// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "sampling.h"

namespace ggml_runtime {
class BackendManager;
class GGUFLoader;
}  // namespace ggml_runtime

namespace nemo_speech::tts::omnivoice {

class GenerationCancelled : public std::runtime_error {
   public:
    GenerationCancelled() : std::runtime_error("OmniVoice generation cancelled") {}
};

class Denoiser;
class FrontendTables;
class Tokenizer;

struct GenerationConfig {
    int32_t steps = 32;
    float guidance_scale = 2.0f;
    float time_shift = 0.1f;
    float layer_penalty = 5.0f;
    float position_temperature = 5.0f;
    float class_temperature = 0.0f;
    bool denoise = true;
    int64_t seed = -1;
    // Checked before preparation and each denoising/model step. Release and
    // transport code use this to stop an in-flight committed segment.
    std::function<bool()> cancelled;
};

struct TokenGenerationRequest {
    std::string text;
    std::optional<std::string> language;
    std::optional<std::string> instruction;
    std::optional<std::string> reference_text;
    // One vector per codebook, all with the same reference-frame count.
    std::optional<std::array<std::vector<int32_t>, 8>> reference_audio_codes;
    std::optional<int32_t> target_frames;
    std::optional<double> speed;
    std::optional<double> fixed_duration_seconds;
};

struct GeneratedAudioCodes {
    std::array<std::vector<int32_t>, 8> codebooks;
    int32_t frame_rate = 25;
    uint64_t effective_seed = 0;
};

// End-to-end text-to-audio-token generation (the waveform codec is a separate
// component). This object is safe for sequential requests; independent objects
// provide independent concurrent execution lanes through BackendManager.
class Generator {
   public:
    Generator(ggml_runtime::BackendManager& backends, const std::string& model_path);
    ~Generator();

    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    GeneratedAudioCodes generate(
        const TokenGenerationRequest& request, const GenerationConfig& config = {});

    int32_t estimate_target_frames(const TokenGenerationRequest& request) const;
    const ModelConfig& config() const { return config_; }
    const std::vector<std::string>& language_ids() const;
    const std::vector<std::string>& language_names() const;

   private:
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    ModelConfig config_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<FrontendTables> frontend_;
    std::unique_ptr<Denoiser> denoiser_;
};

}  // namespace nemo_speech::tts::omnivoice
