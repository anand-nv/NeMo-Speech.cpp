// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::tts::kokoro {

class KokoroDecoderEncoder {
   public:
    explicit KokoroDecoderEncoder(const std::string& model_path, bool use_gpu = false);
    ~KokoroDecoderEncoder();

    KokoroDecoderEncoder(const KokoroDecoderEncoder&) = delete;
    KokoroDecoderEncoder& operator=(const KokoroDecoderEncoder&) = delete;

    // Returns generator input in [512, 2*frame_count] GGML memory order.
    std::vector<float> encode(
        const std::vector<float>& aligned_text, size_t frame_count, const std::vector<float>& f0,
        const std::vector<float>& noise, const std::vector<float>& decoder_style);

    // Returns only `[latent_begin, latent_end)` from the decoder's
    // `[512, 2*frame_count]` output. The implementation evaluates a bounded
    // source window with enough noncausal context for progressive vocoder
    // pulls instead of materializing the complete decoder output.
    std::vector<float> encode_range(
        const std::vector<float>& aligned_text, size_t frame_count, const std::vector<float>& f0,
        const std::vector<float>& noise, const std::vector<float>& decoder_style,
        size_t latent_begin, size_t latent_end);

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::kokoro
