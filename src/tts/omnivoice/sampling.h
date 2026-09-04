// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace nemo_speech::tts::omnivoice {

struct SamplingConfig {
    float guidance_scale = 2.0f;
    float layer_penalty = 5.0f;
    float position_temperature = 5.0f;
    float class_temperature = 0.0f;
};

enum class RandomPurpose : uint32_t { ClassToken = 0, RevealPosition = 1 };

struct RandomCounter {
    int32_t step = 0;
    int32_t batch = 0;
    int32_t codebook = 0;
    int32_t position = 0;
    int32_t vocabulary = 0;
    RandomPurpose purpose = RandomPurpose::ClassToken;
};

using UniformSource = std::function<double(const RandomCounter&)>;

// Backend-independent counter-based generator. Each draw is a pure function
// of the request seed and semantic sampling coordinate.
class CounterRandom {
   public:
    explicit CounterRandom(int64_t seed);
    double uniform(const RandomCounter& counter) const;
    uint64_t seed() const { return seed_; }

   private:
    uint64_t seed_;
};

// Shifted-linear schedule from the reference implementation. Each entry is
// the exact number of codebook positions revealed by that denoising step.
std::vector<int32_t> reveal_schedule(
    int32_t target_frames, int32_t codebooks, int32_t steps, float time_shift);

// Select and commit one iterative-unmasking step. Logits are contiguous
// [codebook,target_frame,vocabulary]. Tokens use the same [codebook,frame]
// order and are mutated in place. Returns the number of newly revealed tokens.
int32_t sample_step(
    const float* conditional_logits, const float* unconditional_logits, int32_t codebooks,
    int32_t target_frames, int32_t vocabulary_size, int32_t mask_id, int32_t reveal_count,
    int32_t step, int32_t batch_item, const SamplingConfig& config, std::vector<int32_t>& tokens,
    const UniformSource& uniforms);

}  // namespace nemo_speech::tts::omnivoice
