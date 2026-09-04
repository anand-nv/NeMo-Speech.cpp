// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::tts::kokoro {

struct KokoroVocoderDebug {
    std::vector<float> sine;
    std::vector<float> harmonic;
    std::vector<float> magnitude;
    std::vector<float> phase;
};

struct KokoroVocoderStreamStats {
    uint64_t samples_written = 0;
    size_t generator_tiles = 0;
    size_t max_generator_input_frames = 0;
    bool cancelled = false;
};

class KokoroVocoder {
   public:
    using FloatCallback = std::function<bool(const std::vector<float>&)>;
    // Supplies `[512, end - begin]` latent columns for an absolute range.
    // The streaming path requests overlapping bounded ranges and may request
    // the same column more than once.
    using LatentProvider = std::function<std::vector<float>(size_t begin, size_t end)>;
    explicit KokoroVocoder(const std::string& model_path, bool use_gpu = false);
    ~KokoroVocoder();

    KokoroVocoder(const KokoroVocoder&) = delete;
    KokoroVocoder& operator=(const KokoroVocoder&) = delete;

    // `latent` is [512, f0.size()] in GGML memory order. A negative seed asks
    // the runtime to select a fresh seed; the selected seed is returned.
    std::vector<float> synthesize(
        const std::vector<float>& latent, const std::vector<float>& f0,
        const std::vector<float>& decoder_style, int64_t seed, uint64_t* selected_seed = nullptr,
        KokoroVocoderDebug* debug = nullptr);

    // Runs the same source and generator, then performs inverse-STFT with a
    // bounded overlap-add ring. Each callback contains newly finalized mono
    // samples and concatenates exactly to synthesize() for the same inputs.
    KokoroVocoderStreamStats synthesize_stream(
        const std::vector<float>& latent, const std::vector<float>& f0,
        const std::vector<float>& decoder_style, int64_t seed, size_t callback_samples,
        const FloatCallback& callback, uint64_t* selected_seed = nullptr);

    // Pull-based overload used by the complete runtime. It keeps both the
    // decoder output and the convolutional generator graph bounded while a
    // single linguistic chunk is emitted progressively.
    KokoroVocoderStreamStats synthesize_stream(
        const LatentProvider& latent_provider, size_t latent_frames, const std::vector<float>& f0,
        const std::vector<float>& decoder_style, int64_t seed, size_t callback_samples,
        const FloatCallback& callback, uint64_t* selected_seed = nullptr);

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::kokoro
