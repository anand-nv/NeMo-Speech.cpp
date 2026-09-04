// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <string>
#include <vector>

#include "tts/omnivoice/frontend.h"
#include "tts/omnivoice/runtime.h"

using namespace nemo_speech::tts::omnivoice;

int
main(int argc, char** argv) {
    if (first_commit_boundary("Dr.") != 0 || first_commit_boundary("Dr. Smith.") == 0 ||
        first_commit_boundary("Hello. Next") != 6)
        throw std::runtime_error("OmniVoice stream punctuation boundary is incorrect");
    if (argc != 3 && argc != 4)
        return 0;

    Runtime runtime(argv[1], argv[2], argc == 4);
    RuntimeSynthesisRequest request;
    request.language = "en";
    RuntimeConfig config;
    config.generation.steps = 1;
    config.generation.seed = 42;
    config.postprocess_output = false;
    config.pad_duration_s = 0.0;
    config.fade_duration_s = 0.0;

    RuntimeSynthesisRequest cancelled_request = request;
    cancelled_request.text = "Never generated.";
    cancelled_request.target_frames = 1;
    RuntimeConfig cancelled_config = config;
    cancelled_config.generation.cancelled = [] { return true; };
    bool generation_cancelled = false;
    try {
        runtime.synthesize(cancelled_request, cancelled_config);
    }
    catch (const GenerationCancelled&) {
        generation_cancelled = true;
    }
    if (!generation_cancelled)
        throw std::runtime_error("OmniVoice runtime ignored pre-generation cancellation");

    {
        BidirectionalStream invalid_utf8(
            runtime, request, config, [](const float*, size_t) { return true; });
        bool rejected_invalid = false;
        try {
            const char invalid[] = {static_cast<char>(0xff)};
            invalid_utf8.push_text(invalid, sizeof(invalid), false);
        }
        catch (const std::invalid_argument&) {
            rejected_invalid = true;
        }
        invalid_utf8.cancel();
        if (!rejected_invalid)
            throw std::runtime_error("OmniVoice stream accepted invalid UTF-8");
    }
    {
        BidirectionalStream incomplete_utf8(
            runtime, request, config, [](const float*, size_t) { return true; });
        bool rejected_incomplete = false;
        try {
            const char incomplete[] = {static_cast<char>(0xe4)};
            incomplete_utf8.push_text(incomplete, sizeof(incomplete), true);
        }
        catch (const std::invalid_argument&) {
            rejected_incomplete = true;
        }
        incomplete_utf8.cancel();
        if (!rejected_incomplete)
            throw std::runtime_error("OmniVoice stream committed incomplete UTF-8");
    }
    {
        StreamLimits limits;
        limits.maximum_pending_bytes = 3;
        BidirectionalStream bounded(
            runtime, request, config, [](const float*, size_t) { return true; }, limits);
        bool rejected_pending = false;
        try {
            bounded.push_text("four", 4, false);
        }
        catch (const std::invalid_argument&) {
            rejected_pending = true;
        }
        bounded.cancel();
        if (!rejected_pending)
            throw std::runtime_error("OmniVoice stream exceeded its pending-text limit");
    }
    {
        StreamLimits limits;
        limits.maximum_queued_segments = 1;
        BidirectionalStream bounded(
            runtime, request, config, [](const float*, size_t) { return true; }, limits);
        bool rejected_queue = false;
        try {
            bounded.push_text("One. Two.", 9, false);
        }
        catch (const std::runtime_error&) {
            rejected_queue = true;
        }
        bounded.cancel();
        if (!rejected_queue)
            throw std::runtime_error("OmniVoice stream exceeded its committed-segment limit");
    }

    std::vector<float> pcm;
    BidirectionalStream stream(runtime, request, config, [&](const float* samples, size_t count) {
        pcm.insert(pcm.end(), samples, samples + count);
        return true;
    });
    const std::string chinese = "\xe4\xbd\xa0\xe5\xa5\xbd\xe3\x80\x82";
    stream.push_text(chinese.data(), 2, false);
    stream.push_text(chinese.data() + 2, chinese.size() - 2, false);
    const RuntimeStats stats = stream.finish();
    if (stats.cancelled || stats.chunks != 1 || pcm.empty())
        throw std::runtime_error("OmniVoice bidirectional stream did not drain one fragment");
    bool rejected = false;
    try {
        stream.push_text("x", 1, false);
    }
    catch (const std::logic_error&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("finished OmniVoice stream accepted more input");

    RuntimeSynthesisRequest fragmented_request;
    fragmented_request.language = "en";
    fragmented_request.speed = 100.0;
    RuntimeConfig fragmented_config = config;
    fragmented_config.audio_chunk_duration_s = 0.01;
    fragmented_config.audio_chunk_threshold_s = 0.01;
    std::vector<float> fragmented_pcm;
    BidirectionalStream fragmented(
        runtime, fragmented_request, fragmented_config, [&](const float* samples, size_t count) {
            fragmented_pcm.insert(fragmented_pcm.end(), samples, samples + count);
            return true;
        });
    const std::string committed_text = "One. Two.";
    for (char byte : committed_text) fragmented.push_text(&byte, 1, false);
    const RuntimeStats fragmented_stats = fragmented.finish();
    RuntimeSynthesisRequest complete_request = fragmented_request;
    complete_request.text = committed_text;
    const auto complete = runtime.synthesize(complete_request, fragmented_config);
    if (fragmented_stats.cancelled || fragmented_stats.chunks != 2 || complete.stats.chunks != 2 ||
        fragmented_pcm != complete.pcm_24khz) {
        throw std::runtime_error(
            "OmniVoice bidirectional fragmentation differs from complete-text streaming");
    }

    BidirectionalStream callback_cancelled(
        runtime, request, config, [](const float*, size_t) { return false; });
    callback_cancelled.push_text("Stop.", 5, true);
    const RuntimeStats cancelled_stats = callback_cancelled.finish();
    if (!cancelled_stats.cancelled)
        throw std::runtime_error("OmniVoice stream callback cancellation was ignored");

    BidirectionalStream explicitly_cancelled(
        runtime, request, config, [](const float*, size_t) { return true; });
    explicitly_cancelled.cancel();
    if (!explicitly_cancelled.finish().cancelled)
        throw std::runtime_error("OmniVoice explicit stream cancellation was ignored");
    return 0;
}
