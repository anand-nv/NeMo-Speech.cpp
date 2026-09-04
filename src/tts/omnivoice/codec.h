// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ggml_runtime {
class BackendManager;
class GGUFLoader;
class Module;
class Session;
}  // namespace ggml_runtime

namespace nemo_speech::tts::omnivoice {

struct CodecConfig {
    int32_t sample_rate = 0;
    int32_t hop_length = 0;
    float frame_rate = 0.0f;
    int32_t codebook_size = 0;
    int32_t codebook_dim = 0;
    int32_t quantizers = 0;
    int32_t semantic_sample_rate = 0;
    int32_t semantic_pad_samples = 0;
    int32_t semantic_downsample_factor = 0;
    int32_t hubert_hidden_size = 0;
    int32_t hubert_intermediate_size = 0;
    int32_t hubert_layers = 0;
    int32_t hubert_heads = 0;
    float hubert_layer_norm_epsilon = 0.0f;
    int32_t dac_encoder_hidden_size = 0;
    int32_t dac_decoder_hidden_size = 0;
    int32_t dac_hidden_size = 0;
    std::vector<int32_t> dac_downsampling_ratios;
    std::vector<int32_t> dac_upsampling_ratios;
    std::vector<int32_t> dac_residual_dilations;
    std::string source_revision;
    std::string source_model_sha256;
};

// Validates the immutable Higgs Audio V2 configuration and all tensors needed
// by waveform decoding before a backend session allocates model storage.
CodecConfig load_codec_config(ggml_runtime::GGUFLoader& loader);

class CodecDecoder {
   public:
    CodecDecoder(ggml_runtime::BackendManager& backends, const std::string& gguf_path);
    ~CodecDecoder();

    CodecDecoder(const CodecDecoder&) = delete;
    CodecDecoder& operator=(const CodecDecoder&) = delete;

    const CodecConfig& config() const { return config_; }

    // Codes are unshifted [8,T] IDs in [0,1023]. The output is mono F32 PCM
    // at 24 kHz and always contains exactly T*960 samples.
    std::vector<float> decode(const std::array<std::vector<int32_t>, 8>& codes);

   private:
    CodecConfig config_;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<ggml_runtime::Module> module_;
    std::unique_ptr<ggml_runtime::Session> session_;
};

class CodecEncoder {
   public:
    CodecEncoder(ggml_runtime::BackendManager& backends, const std::string& gguf_path);
    ~CodecEncoder();

    CodecEncoder(const CodecEncoder&) = delete;
    CodecEncoder& operator=(const CodecEncoder&) = delete;

    const CodecConfig& config() const { return config_; }

    // Encodes mono 24 kHz F32 PCM. The input length must be a positive
    // multiple of 960 samples, as enforced by OmniVoice prompt preparation.
    std::array<std::vector<int32_t>, 8> encode(const std::vector<float>& mono_24khz);

   private:
    CodecConfig config_;
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    std::unique_ptr<ggml_runtime::Module> module_;
    std::unique_ptr<ggml_runtime::Session> session_;
};

}  // namespace nemo_speech::tts::omnivoice
