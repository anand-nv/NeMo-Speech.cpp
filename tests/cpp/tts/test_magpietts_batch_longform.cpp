// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Longform requests in the MagpieTTS in-flight batching engine: a multi-chunk text decoded
// together with short requests (its chunk transitions re-prefill its slot while the others keep
// decoding) must produce what the serial runtime produces for every request. With the fused CUDA
// kernels (Q8_0 model) the audio must be sample-identical; on the ggml graphs every request must
// complete with about the same length. Also reports the longest audio gap the short requests saw.
//
// Usage: test_magpietts_batch_longform [--magpie M] [--codec C] [--tokenizer DIR] [--slots S]
// (or NEMO_SPEECH_TEST_TTS_MODEL / _CODEC_MODEL / _TOKENIZER_DIR); skipped without them.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "tts/magpietts/batch/batch_engine.h"
#include "tts/magpietts/magpietts.h"
#include "tts/nanocodec/model.h"
#include "tts/tokenizer/tokenizer.h"

namespace tts = nemo_speech::tts;

namespace {

const char*
env_or(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] ? v : "";
}

const char* kLong =
    "In the beginning of the twentieth century, the small coastal town was known mostly for its "
    "fishing fleet and its lighthouse. Over the following decades, however, artists and writers "
    "began to arrive, drawn by the quality of the light and the quiet of the harbor. By the "
    "nineteen fifties, galleries lined the main street, and summer visitors filled every guest "
    "house. Today the town balances its two identities, honoring the fishermen who built it while "
    "welcoming the painters who made it famous.";

const char* kShort[] = {
    "I guess it comes down a simple choice. Get busy living or get busy dying.",
    "Life was like a box of chocolates, you never know what you're gonna get.",
    "Whoever saves one life, saves the world entire.",
    "You can overcome anything, if and only if you love something enough.",
};

}  // namespace

int
main(int argc, char** argv) {
    std::string magpie_path = env_or("NEMO_SPEECH_TEST_TTS_MODEL");
    std::string codec_path = env_or("NEMO_SPEECH_TEST_CODEC_MODEL");
    std::string tokenizer_dir = env_or("NEMO_SPEECH_TEST_TOKENIZER_DIR");
    int slots = 4;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--magpie")
            magpie_path = argv[i + 1];
        else if (a == "--codec")
            codec_path = argv[i + 1];
        else if (a == "--tokenizer")
            tokenizer_dir = argv[i + 1];
        else if (a == "--slots")
            slots = std::atoi(argv[i + 1]);
    }
    if (magpie_path.empty() || codec_path.empty() || tokenizer_dir.empty()) {
        std::printf("SKIP: needs the Magpie model, the codec and the tokenizer\n");
        return 77;
    }
    tts::MagpieNativeTokenizer tokenizer(tokenizer_dir);
    std::vector<std::vector<std::vector<int32_t>>> requests;
    auto add = [&](const char* text) {
        std::vector<std::vector<int32_t>> chunks;
        for (const auto& c : tokenizer.tokenize(text, "en-US").chunks) chunks.push_back(c.tokens);
        requests.push_back(chunks);
    };
    add(kLong);
    for (const char* t : kShort) add(t);
    if (requests[0].size() < 3) {
        std::fprintf(
            stderr, "FAIL: the longform text tokenized into %zu chunk(s)\n", requests[0].size());
        return 1;
    }

    // serial reference
    std::vector<std::vector<uint8_t>> serial(requests.size());
    {
        tts::MagpieStreamingRuntime runtime;
        if (!runtime.load(magpie_path, codec_path, tts::MAGPIETTS_UMA_AUTO, false, false, false)) {
            std::fprintf(stderr, "FAIL: serial runtime load\n");
            return 1;
        }
        for (size_t i = 0; i < requests.size(); ++i) {
            tts::magpie_stream_params params;
            params.seed = 1234;
            tts::stream_run_metrics metrics;
            if (!runtime.synthesize(
                    params, requests[i],
                    [&](const std::vector<uint8_t>& b) {
                        serial[i].insert(serial[i].end(), b.begin(), b.end());
                        return true;
                    },
                    metrics, "riva_tts", false)) {
                std::fprintf(stderr, "FAIL: serial request %zu\n", i);
                return 1;
            }
        }
    }

    // the same requests, all at once, through the engine
    tts::MagpieModel magpie;
    tts::nanocodec::NanoCodecModel codec;
    if (!magpie.load(magpie_path, tts::MAGPIETTS_UMA_AUTO, false, false) ||
        !codec.load(codec_path, false, false)) {
        std::fprintf(stderr, "FAIL: model load\n");
        return 1;
    }
    tts::MagpieBatchConfig config;
    config.max_slots = slots;
    tts::MagpieBatchEngine engine(magpie, codec, config);
    std::string error;
    if (!engine.init(&error)) {
        std::fprintf(stderr, "FAIL: engine init: %s\n", error.c_str());
        return 1;
    }
    std::vector<std::vector<uint8_t>> batched(requests.size());
    std::vector<tts::stream_run_metrics> metrics(requests.size());
    std::vector<std::mutex> pcm_mutex(requests.size());
    std::mutex mutex;
    std::condition_variable cv;
    int remaining = static_cast<int>(requests.size());
    int failed = 0;
    for (size_t i = 0; i < requests.size(); ++i) {
        tts::magpie_stream_params params;
        params.seed = 1234;
        const int64_t id = engine.submit(
            params, requests[i],
            [&, i](const std::vector<uint8_t>& b) {
                std::lock_guard<std::mutex> lock(pcm_mutex[i]);
                batched[i].insert(batched[i].end(), b.begin(), b.end());
                return true;
            },
            &metrics[i], "riva_tts",
            [&](bool ok) {
                std::lock_guard<std::mutex> lock(mutex);
                failed += ok ? 0 : 1;
                --remaining;
                cv.notify_all();
            },
            &error);
        if (id < 0) {
            std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
            return 1;
        }
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return remaining == 0; });
    }
    engine.stop();
    if (failed) {
        std::fprintf(stderr, "FAIL: %d request(s) failed in the engine\n", failed);
        return 1;
    }
    const bool exact = engine.fusedKernels();
    int bad = 0;
    double worst_gap_ms = 0.0;
    for (size_t i = 0; i < requests.size(); ++i) {
        const bool same = serial[i] == batched[i];
        const double ratio = serial[i].empty() ? 0.0 : (double)batched[i].size() / serial[i].size();
        if (i > 0)
            worst_gap_ms = std::max(worst_gap_ms, metrics[i].e2e.inter_event_max_ms);
        std::printf(
            "request %zu (%zu chunk%s): serial %zu bytes, batched %zu bytes, %s\n", i,
            requests[i].size(), requests[i].size() > 1 ? "s" : "", serial[i].size(),
            batched[i].size(), same ? "identical" : "differs");
        if (exact ? !same : (ratio < 0.8 || ratio > 1.25))
            ++bad;
    }
    std::printf(
        "%s path; longest audio gap of a short request while batched with the longform request: "
        "%.1f ms\n",
        exact ? "fused" : "ggml", worst_gap_ms);
    if (bad) {
        std::fprintf(stderr, "FAIL: %d request(s) differ from the serial runtime\n", bad);
        return 1;
    }
    std::printf("batched longform OK\n");
    return 0;
}
