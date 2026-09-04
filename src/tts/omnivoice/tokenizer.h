// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ggml_runtime {
class GGUFLoader;
}

namespace nemo_speech::tts::omnivoice {

// Native port of the pinned Qwen2 Tokenizers pipeline: strict UTF-8, NFC,
// Unicode split regex, GPT-2 byte-to-Unicode mapping, and ranked BPE merges.
class Tokenizer {
   public:
    explicit Tokenizer(const ggml_runtime::GGUFLoader& loader);
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    std::vector<int32_t> encode(const std::string& utf8) const;
    std::vector<int32_t> encode_with_nonverbal_tags(const std::string& utf8) const;
    int32_t token_id(const std::string& token) const;

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::omnivoice
