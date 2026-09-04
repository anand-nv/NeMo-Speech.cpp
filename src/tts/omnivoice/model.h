// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ggml_runtime {
class GGUFLoader;
}

namespace nemo_speech::tts::omnivoice {

struct ModelConfig {
    int32_t text_vocab_size = 0;
    int32_t audio_vocab_size = 0;
    int32_t audio_mask_id = 0;
    int32_t audio_codebooks = 0;
    int32_t hidden_size = 0;
    int32_t layer_count = 0;
    int32_t attention_heads = 0;
    int32_t kv_heads = 0;
    int32_t head_size = 0;
    int32_t feed_forward_size = 0;
    int32_t context_length = 0;
    float rms_epsilon = 0.0f;
    float rope_theta = 0.0f;
    int32_t pad_token_id = 0;
    int32_t eos_token_id = 0;
    std::vector<int32_t> audio_codebook_weights;
    std::vector<int32_t> audio_codebook_offsets;
    std::string source_revision;
    std::string source_model_sha256;
    std::string source_tokenizer_sha256;
    std::string source_audio_tokenizer_sha256;
};

// Reads and validates all runtime-relevant metadata and the complete tensor
// manifest before any model tensor or graph arena is allocated.
ModelConfig load_model_config(ggml_runtime::GGUFLoader& loader);

}  // namespace nemo_speech::tts::omnivoice
