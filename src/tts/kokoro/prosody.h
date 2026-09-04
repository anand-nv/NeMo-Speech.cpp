// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::tts::kokoro {

struct KokoroProsody {
    std::vector<float> f0;
    std::vector<float> noise;
};

class KokoroProsodyHeads {
   public:
    explicit KokoroProsodyHeads(const std::string& model_path, bool use_gpu = false);
    ~KokoroProsodyHeads();

    KokoroProsodyHeads(const KokoroProsodyHeads&) = delete;
    KokoroProsodyHeads& operator=(const KokoroProsodyHeads&) = delete;

    KokoroProsody predict(
        const std::vector<float>& shared_features, size_t frame_count,
        const std::vector<float>& style);

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::kokoro
