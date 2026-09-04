// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime.h"
#include "tts/kokoro/acoustic.h"
#include "tts/kokoro/duration.h"
#include "tts/kokoro/model.h"
#include "tts/kokoro/plbert.h"
#include "tts/kokoro/prosody.h"

int
main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: test_kokoro_prosody MODEL.gguf [OUTPUT_PREFIX]\n";
        return 2;
    }
    const std::vector<int32_t> ids = {0, 50, 83, 156, 54, 57, 0};
    ggml_runtime::GGUFLoader loader(argv[1]);
    nemo_speech::tts::kokoro::KokoroModelMetadata metadata(loader);
    const auto voice = metadata.read_voice_style(loader, "af_heart", 5);
    const std::vector<float> style(voice.begin() + 128, voice.end());
    nemo_speech::tts::kokoro::KokoroPlbertEncoder plbert(argv[1], false);
    const auto projected = plbert.encode(ids);
    nemo_speech::tts::kokoro::KokoroDurationPredictor duration(argv[1], false);
    const auto timing = duration.predict(projected, ids.size(), style);
    nemo_speech::tts::kokoro::KokoroAcousticEncoder acoustic(argv[1], false);
    const auto features = acoustic.encode(ids, timing.frames, timing.encoded_features);
    nemo_speech::tts::kokoro::KokoroProsodyHeads heads(argv[1], false);
    const auto result = heads.predict(features.prosody_shared, features.frame_count, style);
    if (result.f0.size() != 108 || result.noise.size() != 108) {
        throw std::runtime_error("Kokoro prosody output shape mismatch");
    }
    if (argc == 3) {
        auto write = [](const std::string& path, const std::vector<float>& values) {
            std::ofstream stream(path, std::ios::binary);
            stream.write(
                reinterpret_cast<const char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
            if (!stream)
                throw std::runtime_error("failed to write prosody output");
        };
        write(std::string(argv[2]) + ".f0.f32", result.f0);
        write(std::string(argv[2]) + ".noise.f32", result.noise);
    }
    std::cout << "Kokoro native prosody smoke test passed\n";
    return 0;
}
