// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::tts::kokoro {

// Native GGML implementation of Kokoro's shared 12-layer ALBERT encoder plus
// the 768->512 bert_encoder projection. Output layout is [T, 512] in row-major
// host storage (GGML ne=[512,T]).
class KokoroPlbertEncoder {
   public:
    explicit KokoroPlbertEncoder(const std::string& model_path, bool use_gpu = false);
    ~KokoroPlbertEncoder();

    KokoroPlbertEncoder(const KokoroPlbertEncoder&) = delete;
    KokoroPlbertEncoder& operator=(const KokoroPlbertEncoder&) = delete;

    std::vector<float> encode(const std::vector<int32_t>& framed_ids);

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::kokoro
