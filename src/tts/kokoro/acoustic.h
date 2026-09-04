// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::tts::kokoro {

struct KokoroAcousticFeatures {
    size_t frame_count = 0;
    // Alignment-expanded outputs in GGML [channels,time] memory order.
    std::vector<float> text;
    std::vector<float> prosody_shared;
};

// Runs the StyleTTS2 text encoder and the duration-aligned shared prosody
// LSTM. Alignment itself is constructed deterministically on the host from
// the already-rounded duration vector.
class KokoroAcousticEncoder {
   public:
    explicit KokoroAcousticEncoder(const std::string& model_path, bool use_gpu = false);
    ~KokoroAcousticEncoder();

    KokoroAcousticEncoder(const KokoroAcousticEncoder&) = delete;
    KokoroAcousticEncoder& operator=(const KokoroAcousticEncoder&) = delete;

    KokoroAcousticFeatures encode(
        const std::vector<int32_t>& framed_ids, const std::vector<int32_t>& durations,
        const std::vector<float>& duration_features);

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::kokoro
