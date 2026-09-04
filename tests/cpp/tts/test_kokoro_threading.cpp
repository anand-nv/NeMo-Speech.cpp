// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

#include "tts/kokoro/kokoro_runtime.h"

int
main(int argc, char** argv) {
    if (argc != 2) {
        throw std::runtime_error("usage: test_kokoro_threading MODEL.gguf");
    }

    using namespace nemo_speech::tts::kokoro;
    std::array<std::string, 2> outputs;
    std::array<std::exception_ptr, 2> errors;
    std::array<std::thread, 2> workers;
    for (size_t index = 0; index < workers.size(); ++index) {
        workers[index] = std::thread([&, index] {
            try {
                KokoroRuntimeConfig config;
                config.model_path = argv[1];
                config.use_gpu = false;
                KokoroRuntime runtime(std::move(config));
                const KokoroChunk input =
                    runtime.prepare_tokens({50, 83, 156, 54, 57}, "en-US", "af_heart");
                const KokoroRuntimeStats stats = runtime.synthesize(
                    {input}, "af_heart", 1.0f, 1234, [&](const std::string& pcm) {
                        outputs[index] += pcm;
                        return true;
                    });
                if (stats.cancelled || stats.chunks != 1 ||
                    stats.samples_written * sizeof(int16_t) != outputs[index].size()) {
                    throw std::runtime_error("independent Kokoro synthesis failed");
                }
            }
            catch (...) {
                errors[index] = std::current_exception();
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    for (const std::exception_ptr& error : errors) {
        if (error)
            std::rethrow_exception(error);
    }
    if (outputs[0].empty() || outputs[0] != outputs[1]) {
        throw std::runtime_error("independent Kokoro runtimes were not deterministic");
    }
    return 0;
}
