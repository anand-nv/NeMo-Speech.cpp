// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::tts::kokoro {

struct KokoroDurationResult {
    // Output of DurationEncoder, GGML layout [640, token_count]. This is
    // repeated through the alignment for the shared prosody LSTM.
    std::vector<float> encoded_features;
    std::vector<float> values;
    std::vector<int32_t> frames;
};

// StyleTTS2 duration branch following the projected PL-BERT stage. Inputs use
// GGML's [channels,time] layout. `style` is the upper 128 values of a selected
// Kokoro voice row.
class KokoroDurationPredictor {
   public:
    explicit KokoroDurationPredictor(const std::string& model_path, bool use_gpu = false);
    ~KokoroDurationPredictor();

    KokoroDurationPredictor(const KokoroDurationPredictor&) = delete;
    KokoroDurationPredictor& operator=(const KokoroDurationPredictor&) = delete;

    KokoroDurationResult predict(
        const std::vector<float>& projected_plbert, size_t token_count,
        const std::vector<float>& style, float speed = 1.0f);

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::kokoro
