// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "frontend.h"

namespace nemo_speech::tts::kokoro {

struct KokoroRuntimeConfig {
    std::string model_path;
    bool use_gpu = false;
    int64_t seed = -1;
    float speed = 1.0f;
};

struct KokoroRuntimeStats {
    int sample_rate = 24000;
    int generated_frames = 0;
    int chunks = 0;
    uint64_t samples_written = 0;
    double audio_s = 0.0;
    double elapsed_s = 0.0;
    double rtf = 0.0;
    double rtfx = 0.0;
    double ttfa_ms = 0.0;
    double icl_avg_ms = 0.0;
    double icl_min_ms = 0.0;
    double icl_max_ms = 0.0;
    bool cancelled = false;
};

// Owns one initialized instance of every Kokoro inference stage. Synthesis is
// serialized because GGML Sessions reuse their graph cache and work buffers.
class KokoroRuntime {
   public:
    using PcmCallback = std::function<bool(const std::string& pcm_s16le)>;

    explicit KokoroRuntime(KokoroRuntimeConfig config);
    ~KokoroRuntime();

    KokoroRuntime(const KokoroRuntime&) = delete;
    KokoroRuntime& operator=(const KokoroRuntime&) = delete;

    KokoroPreparedText prepare(
        const std::string& text, const std::string& language, const std::string& voice) const;
    KokoroChunk prepare_tokens(
        const std::vector<int32_t>& ids, const std::string& language,
        const std::string& voice) const;

    KokoroRuntimeStats synthesize(
        const std::vector<KokoroChunk>& chunks, const std::string& voice, float speed, int64_t seed,
        const PcmCallback& callback = {});

    int sample_rate() const;
    const std::vector<std::string>& voice_names() const;
    const std::vector<std::string>& voice_languages() const;

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::kokoro
