// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime.h"
#include "tts/kokoro/duration.h"
#include "tts/kokoro/model.h"
#include "tts/kokoro/plbert.h"

int
main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: test_kokoro_duration MODEL.gguf [OUTPUT.f32]\n";
        return 2;
    }
    constexpr size_t token_count = 7;
    nemo_speech::tts::kokoro::KokoroPlbertEncoder encoder(argv[1], false);
    const std::vector<float> projected = encoder.encode({0, 50, 83, 156, 54, 57, 0});
    ggml_runtime::GGUFLoader loader(argv[1]);
    nemo_speech::tts::kokoro::KokoroModelMetadata metadata(loader);
    const std::vector<float> voice = metadata.read_voice_style(loader, "af_heart", 5);
    const std::vector<float> style(voice.begin() + 128, voice.end());
    nemo_speech::tts::kokoro::KokoroDurationPredictor predictor(argv[1], false);
    const auto result = predictor.predict(projected, token_count, style, 1.0f);
    if (result.values.size() != token_count || result.frames.size() != token_count ||
        result.encoded_features.size() != token_count * 640) {
        throw std::runtime_error("Kokoro duration output shape mismatch");
    }
    for (size_t index = 0; index < token_count; ++index) {
        if (!std::isfinite(result.values[index]) || result.frames[index] < 1) {
            throw std::runtime_error("invalid Kokoro duration output");
        }
    }
    const std::vector<int32_t> expected = {18, 2, 3, 2, 3, 14, 12};
    if (result.frames != expected) {
        throw std::runtime_error("Kokoro duration reference mismatch");
    }
    if (argc == 3) {
        std::ofstream stream(argv[2], std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(result.values.data()),
            static_cast<std::streamsize>(result.values.size() * sizeof(float)));
        if (!stream)
            throw std::runtime_error("failed to write duration output");
    }
    std::cout << "Kokoro native duration smoke test passed\n";
    return 0;
}
