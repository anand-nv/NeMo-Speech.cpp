// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml_runtime {
class GGUFLoader;
}

namespace nemo_speech::tts::kokoro {

struct EnglishLexiconData;

struct KokoroHParams {
    int32_t sample_rate = 0;
    int32_t context_length = 0;
    int32_t phoneme_limit = 0;
    int32_t vocab_size = 0;
    int32_t voice_style_dim = 0;
    int32_t style_dim = 0;
    int32_t hidden_dim = 0;
    int32_t max_conv_dim = 0;
    int32_t dim_in = 0;
    int32_t max_dur = 0;
    int32_t n_layer = 0;
    int32_t n_mels = 0;
    int32_t text_encoder_kernel_size = 0;
    int32_t plbert_hidden_size = 0;
    int32_t plbert_attention_heads = 0;
    int32_t plbert_intermediate_size = 0;
    int32_t plbert_layers = 0;
    int32_t istft_hop_size = 0;
    int32_t istft_n_fft = 0;
    float speed_default = 0.0f;
    float speed_min = 0.0f;
    float speed_max = 0.0f;
};

// Lightweight, allocation-safe validation of a Kokoro GGUF. This runs before
// model buffers/workspaces are allocated and validates the converter's exact
// tensor manifest, all voice tables, and frontend metadata.
class KokoroModelMetadata {
   public:
    explicit KokoroModelMetadata(const ggml_runtime::GGUFLoader& loader);

    const KokoroHParams& hparams() const { return hparams_; }
    const std::vector<std::string>& vocabulary() const { return vocabulary_; }
    const std::vector<std::string>& voice_names() const { return voice_names_; }
    const std::vector<std::string>& voice_languages() const { return voice_languages_; }
    const std::string& source_sha256() const { return source_sha256_; }
    const std::string& misaki_lexicon_json(const std::string& name) const;
    EnglishLexiconData english_lexicon_data() const;

    size_t voice_index(const std::string& name) const;
    std::string voice_tensor_name(size_t index) const;
    std::vector<float> read_voice_style(
        ggml_runtime::GGUFLoader& loader, const std::string& voice,
        size_t unframed_phoneme_count) const;

   private:
    KokoroHParams hparams_;
    std::vector<std::string> vocabulary_;
    std::vector<std::string> voice_names_;
    std::vector<std::string> voice_languages_;
    std::string source_sha256_;
    std::unordered_map<std::string, std::string> misaki_lexicons_;
};

}  // namespace nemo_speech::tts::kokoro
