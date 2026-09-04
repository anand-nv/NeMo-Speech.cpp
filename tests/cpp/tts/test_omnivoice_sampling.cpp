// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tts/omnivoice/sampling.h"

using nemo_speech::tts::omnivoice::CounterRandom;
using nemo_speech::tts::omnivoice::RandomCounter;
using nemo_speech::tts::omnivoice::SamplingConfig;

int
main() {
    using namespace nemo_speech::tts::omnivoice;
    if (reveal_schedule(25, 8, 32, 0.1f) !=
        std::vector<int32_t>({1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,  2,  2,  2,  2,  2,
                              3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 10, 13, 16, 22, 32, 34})) {
        throw std::runtime_error("OmniVoice shifted reveal schedule differs from oracle");
    }
    if (reveal_schedule(3, 2, 4, 1.0f) != std::vector<int32_t>({2, 2, 2, 0})) {
        throw std::runtime_error("OmniVoice linear reveal schedule differs from oracle");
    }

    constexpr int C = 2;
    constexpr int T = 2;
    constexpr int V = 5;
    constexpr int MASK = 4;
    std::vector<float> conditional(C * T * V, -4.0f);
    std::vector<float> unconditional(C * T * V, -4.0f);
    for (int position = 0; position < C * T; ++position) {
        conditional[position * V + (position % 4)] = 3.0f;
        conditional[position * V + MASK] = 100.0f;  // must never be selected
        unconditional[position * V] = 1.0f;
    }
    std::vector<int32_t> tokens(C * T, MASK);
    SamplingConfig config;
    config.guidance_scale = 0.0f;
    config.layer_penalty = 5.0f;
    config.position_temperature = 0.0f;
    CounterRandom random(1234);
    auto source = [&random](const RandomCounter& counter) { return random.uniform(counter); };
    if (sample_step(
            conditional.data(), unconditional.data(), C, T, V, MASK, 2, 0, 0, config, tokens,
            source) != 2 ||
        tokens != std::vector<int32_t>({0, 1, MASK, MASK})) {
        throw std::runtime_error("OmniVoice deterministic reveal selection is incorrect");
    }
    sample_step(
        conditional.data(), unconditional.data(), C, T, V, MASK, 2, 1, 0, config, tokens, source);
    if (tokens != std::vector<int32_t>({0, 1, 2, 3})) {
        throw std::runtime_error("OmniVoice sampling did not reveal every position");
    }

    const RandomCounter counter = {3, 2, 1, 7, 11, RandomPurpose::ClassToken};
    if (random.uniform(counter) != CounterRandom(1234).uniform(counter) ||
        random.uniform(counter) == CounterRandom(1235).uniform(counter)) {
        throw std::runtime_error("OmniVoice counter RNG is not seed-repeatable");
    }
    return 0;
}
