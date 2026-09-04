// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tts/omnivoice/runtime.h"

using namespace nemo_speech::tts::omnivoice;

int
main(int argc, char** argv) {
    if (argc < 3 || argc > 4 || (argc == 4 && std::string(argv[3]) != "--gpu")) {
        std::cerr << "usage: test_omnivoice_runtime MODEL.gguf CODEC.gguf [--gpu]\n";
        return 2;
    }
    Runtime runtime(argv[1], argv[2], argc == 4);
    RuntimeSynthesisRequest request;
    request.text = "Test.";
    request.language = "en";
    request.target_frames = 1;
    RuntimeConfig config;
    config.generation.steps = 1;
    config.generation.seed = 1234;
    config.postprocess_output = false;
    std::vector<float> streamed;
    const auto streaming = runtime.synthesize(request, config, [&](const float* pcm, size_t count) {
        streamed.insert(streamed.end(), pcm, pcm + count);
        return true;
    });
    const auto offline = runtime.synthesize(request, config);
    if (streaming.stats.cancelled || offline.stats.cancelled || streamed != offline.pcm_24khz)
        throw std::runtime_error("OmniVoice offline and output-streaming PCM differ");
    if (offline.stats.generated_frames != 1 || offline.stats.chunks != 1 ||
        offline.pcm_24khz.size() != 960 + 2 * 2400)
        throw std::runtime_error("OmniVoice runtime returned invalid output dimensions");
    if (!std::all_of(offline.pcm_24khz.begin(), offline.pcm_24khz.end(), [](float value) {
            return std::isfinite(value);
        }))
        throw std::runtime_error("OmniVoice runtime output is not finite");

    RuntimeSynthesisRequest long_request;
    long_request.text = "One. Two.";
    long_request.language = "en";
    long_request.speed = 100.0;
    RuntimeConfig long_config = config;
    long_config.audio_chunk_duration_s = 0.01;
    long_config.audio_chunk_threshold_s = 0.01;
    long_config.pad_duration_s = 0.0;
    long_config.fade_duration_s = 0.0;
    std::vector<float> long_streamed;
    size_t callbacks = 0;
    const auto long_streaming =
        runtime.synthesize(long_request, long_config, [&](const float* pcm, size_t count) {
            ++callbacks;
            long_streamed.insert(long_streamed.end(), pcm, pcm + count);
            return true;
        });
    const auto long_offline = runtime.synthesize(long_request, long_config);
    if (long_streaming.stats.cancelled || long_offline.stats.cancelled ||
        long_offline.stats.chunks != 2 || callbacks < 2 ||
        long_streamed != long_offline.pcm_24khz) {
        throw std::runtime_error("OmniVoice multi-chunk output streaming differs from offline");
    }

    size_t cancelled_callbacks = 0;
    const auto cancelled = runtime.synthesize(long_request, long_config, [&](const float*, size_t) {
        ++cancelled_callbacks;
        return false;
    });
    if (!cancelled.stats.cancelled || cancelled_callbacks != 1 ||
        cancelled.stats.generated_frames >= long_offline.stats.generated_frames) {
        throw std::runtime_error("OmniVoice output-stream cancellation generated a later chunk");
    }
    return 0;
}
