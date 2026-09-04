// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <stdexcept>
#include <string>

#include "tts/kokoro/kokoro_runtime.h"

int
main(int argc, char** argv) {
    if (argc != 2)
        throw std::runtime_error("usage: test_kokoro_voices MODEL.gguf");
    using namespace nemo_speech::tts::kokoro;
    KokoroRuntimeConfig config;
    config.model_path = argv[1];
    config.use_gpu = false;
    KokoroRuntime runtime(std::move(config));
    const auto& voices = runtime.voice_names();
    const auto& languages = runtime.voice_languages();
    if (voices.size() != 54 || languages.size() != voices.size()) {
        throw std::runtime_error("Kokoro published voice inventory is incomplete");
    }
    for (size_t index = 0; index < voices.size(); ++index) {
        const KokoroChunk chunk = runtime.prepare_tokens({43}, languages[index], voices[index]);
        size_t callbacks = 0;
        size_t bytes = 0;
        const KokoroRuntimeStats stats =
            runtime.synthesize({chunk}, voices[index], 2.0f, 1234, [&](const std::string& pcm) {
                ++callbacks;
                bytes += pcm.size();
                return true;
            });
        if (callbacks < 1 || bytes == 0 || bytes % sizeof(int16_t) != 0 || stats.chunks != 1 ||
            stats.samples_written * sizeof(int16_t) != bytes) {
            throw std::runtime_error("Kokoro voice smoke failed: " + voices[index]);
        }
    }
    return 0;
}
