// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "tokenizer/kokoro_tokenizer.h"

namespace ggml_runtime {
class GGUFLoader;
}

namespace nemo_speech::tts::kokoro {

struct KokoroPreparedText {
    KokoroTokenization tokenization;
    std::string voice;
    size_t voice_index = 0;
};

// Owns all immutable model/frontend resources. Construction validates the
// complete GGUF contract before parsing its pinned Misaki dictionaries.
class KokoroFrontend {
   public:
    explicit KokoroFrontend(const std::string& model_path);
    ~KokoroFrontend();

    KokoroFrontend(const KokoroFrontend&) = delete;
    KokoroFrontend& operator=(const KokoroFrontend&) = delete;

    KokoroPreparedText prepare(
        const std::string& text, const std::string& language, const std::string& voice) const;
    KokoroChunk prepare_tokens(
        const std::vector<int32_t>& ids, const std::string& language,
        const std::string& voice) const;
    std::vector<float> voice_style(const std::string& voice, size_t unframed_phoneme_count) const;

    const KokoroModelMetadata& metadata() const { return metadata_; }

   private:
    std::unique_ptr<ggml_runtime::GGUFLoader> loader_;
    KokoroModelMetadata metadata_;
    KokoroTokenizer tokenizer_;
};

}  // namespace nemo_speech::tts::kokoro
