// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace nemo_speech::tts::omnivoice {
namespace {

uint64_t
mix(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

uint64_t
combine(uint64_t state, uint64_t value) {
    return mix(state ^ (mix(value) + 0x9e3779b97f4a7c15ULL + (state << 6) + (state >> 2)));
}

double
gumbel(double uniform) {
    // The oracle adds 1e-10 at both log boundaries. CounterRandom never emits
    // an endpoint, while injected fixtures are clamped for the same behavior.
    uniform = std::max(0.0, std::min(1.0, uniform));
    return -std::log(-std::log(uniform + 1.0e-10) + 1.0e-10);
}

void
log_softmax(const float* logits, int32_t size, std::vector<float>& output) {
    const float maximum = *std::max_element(logits, logits + size);
    double sum = 0.0;
    for (int32_t i = 0; i < size; ++i) sum += std::exp(double(logits[i] - maximum));
    const float log_sum = maximum + static_cast<float>(std::log(sum));
    output.resize(static_cast<size_t>(size));
    for (int32_t i = 0; i < size; ++i) output[static_cast<size_t>(i)] = logits[i] - log_sum;
}

}  // namespace

CounterRandom::CounterRandom(int64_t seed) {
    if (seed == -1) {
        std::random_device source;
        seed_ = (uint64_t(source()) << 32) ^ uint64_t(source());
    } else {
        seed_ = static_cast<uint64_t>(seed);
    }
}

double
CounterRandom::uniform(const RandomCounter& counter) const {
    uint64_t state = mix(seed_);
    state = combine(state, static_cast<uint32_t>(counter.step));
    state = combine(state, static_cast<uint32_t>(counter.batch));
    state = combine(state, static_cast<uint32_t>(counter.codebook));
    state = combine(state, static_cast<uint32_t>(counter.position));
    state = combine(state, static_cast<uint32_t>(counter.vocabulary));
    state = combine(state, static_cast<uint32_t>(counter.purpose));
    const uint64_t mantissa = mix(state) >> 11;
    return (static_cast<double>(mantissa) + 0.5) * (1.0 / 9007199254740992.0);
}

std::vector<int32_t>
reveal_schedule(int32_t target_frames, int32_t codebooks, int32_t steps, float time_shift) {
    if (target_frames <= 0 || codebooks <= 0 || steps <= 0 || !std::isfinite(time_shift) ||
        time_shift <= 0.0f) {
        throw std::invalid_argument("invalid OmniVoice reveal-schedule configuration");
    }
    const int64_t total = int64_t(target_frames) * codebooks;
    if (total > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("OmniVoice reveal schedule is too large");
    }
    auto shifted_time = [steps, time_shift](int32_t index) {
        const float t = static_cast<float>(index) / static_cast<float>(steps);
        return time_shift * t / (1.0f + (time_shift - 1.0f) * t);
    };

    int32_t remaining = static_cast<int32_t>(total);
    std::vector<int32_t> schedule;
    schedule.reserve(static_cast<size_t>(steps));
    for (int32_t step = 0; step < steps; ++step) {
        int32_t count = remaining;
        if (step != steps - 1) {
            const float delta = shifted_time(step + 1) - shifted_time(step);
            count = std::min(
                static_cast<int32_t>(std::ceil(static_cast<float>(total) * delta)), remaining);
        }
        schedule.push_back(count);
        remaining -= count;
    }
    return schedule;
}

int32_t
sample_step(
    const float* conditional_logits, const float* unconditional_logits, int32_t codebooks,
    int32_t target_frames, int32_t vocabulary_size, int32_t mask_id, int32_t reveal_count,
    int32_t step, int32_t batch_item, const SamplingConfig& config, std::vector<int32_t>& tokens,
    const UniformSource& uniforms) {
    if (conditional_logits == nullptr || unconditional_logits == nullptr || !uniforms ||
        codebooks <= 0 || target_frames <= 0 || vocabulary_size <= 1 || mask_id < 0 ||
        mask_id >= vocabulary_size || reveal_count < 0 || !std::isfinite(config.guidance_scale) ||
        !std::isfinite(config.layer_penalty) || config.position_temperature < 0.0f ||
        config.class_temperature < 0.0f) {
        throw std::invalid_argument("invalid OmniVoice sampling input");
    }
    const int32_t positions = codebooks * target_frames;
    if (tokens.size() != static_cast<size_t>(positions)) {
        throw std::invalid_argument("OmniVoice sampling token shape mismatch");
    }
    const int32_t remaining =
        static_cast<int32_t>(std::count(tokens.begin(), tokens.end(), mask_id));
    if (reveal_count > remaining) {
        throw std::invalid_argument("OmniVoice reveal count exceeds remaining mask positions");
    }
    if (reveal_count == 0)
        return 0;

    std::vector<int32_t> predicted(static_cast<size_t>(positions));
    std::vector<float> position_scores(static_cast<size_t>(positions));
    std::vector<float> conditional;
    std::vector<float> unconditional;
    std::vector<float> combined(static_cast<size_t>(vocabulary_size));
    std::vector<float> normalized;
    std::vector<int32_t> candidates(static_cast<size_t>(vocabulary_size));

    for (int32_t flat = 0; flat < positions; ++flat) {
        const int32_t codebook = flat / target_frames;
        const int32_t position = flat % target_frames;
        const size_t offset = static_cast<size_t>(flat) * vocabulary_size;
        log_softmax(conditional_logits + offset, vocabulary_size, conditional);
        if (config.guidance_scale != 0.0f) {
            log_softmax(unconditional_logits + offset, vocabulary_size, unconditional);
            for (int32_t token = 0; token < vocabulary_size; ++token) {
                combined[static_cast<size_t>(token)] =
                    conditional[static_cast<size_t>(token)] +
                    config.guidance_scale * (conditional[static_cast<size_t>(token)] -
                                             unconditional[static_cast<size_t>(token)]);
            }
            log_softmax(combined.data(), vocabulary_size, normalized);
        } else {
            normalized = conditional;
        }
        normalized[static_cast<size_t>(mask_id)] = -std::numeric_limits<float>::infinity();

        const auto maximum = std::max_element(normalized.begin(), normalized.end());
        position_scores[static_cast<size_t>(flat)] = *maximum - codebook * config.layer_penalty;
        int32_t selected = static_cast<int32_t>(maximum - normalized.begin());
        if (config.class_temperature > 0.0f) {
            std::iota(candidates.begin(), candidates.end(), 0);
            const int32_t top_count =
                static_cast<int32_t>(std::ceil(0.1f * static_cast<float>(vocabulary_size)));
            std::partial_sort(
                candidates.begin(), candidates.begin() + top_count, candidates.end(),
                [&normalized](int32_t a, int32_t b) {
                    if (normalized[static_cast<size_t>(a)] != normalized[static_cast<size_t>(b)])
                        return normalized[static_cast<size_t>(a)] >
                               normalized[static_cast<size_t>(b)];
                    return a < b;
                });
            double best = -std::numeric_limits<double>::infinity();
            for (int32_t index = 0; index < top_count; ++index) {
                const int32_t token = candidates[static_cast<size_t>(index)];
                const RandomCounter counter = {step,     batch_item, codebook,
                                               position, token,      RandomPurpose::ClassToken};
                const double score =
                    normalized[static_cast<size_t>(token)] / config.class_temperature +
                    gumbel(uniforms(counter));
                if (score > best || (score == best && token < selected)) {
                    best = score;
                    selected = token;
                }
            }
        }
        predicted[static_cast<size_t>(flat)] = selected;

        if (config.position_temperature > 0.0f) {
            const RandomCounter counter = {step,     batch_item, codebook,
                                           position, 0,          RandomPurpose::RevealPosition};
            position_scores[static_cast<size_t>(flat)] =
                position_scores[static_cast<size_t>(flat)] / config.position_temperature +
                static_cast<float>(gumbel(uniforms(counter)));
        }
        if (tokens[static_cast<size_t>(flat)] != mask_id) {
            position_scores[static_cast<size_t>(flat)] = -std::numeric_limits<float>::infinity();
        }
    }

    std::vector<int32_t> order(static_cast<size_t>(positions));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + reveal_count, order.end(),
        [&position_scores](int32_t a, int32_t b) {
            if (position_scores[static_cast<size_t>(a)] !=
                position_scores[static_cast<size_t>(b)]) {
                return position_scores[static_cast<size_t>(a)] >
                       position_scores[static_cast<size_t>(b)];
            }
            return a < b;
        });
    for (int32_t index = 0; index < reveal_count; ++index) {
        const int32_t position = order[static_cast<size_t>(index)];
        tokens[static_cast<size_t>(position)] = predicted[static_cast<size_t>(position)];
    }
    return reveal_count;
}

}  // namespace nemo_speech::tts::omnivoice
