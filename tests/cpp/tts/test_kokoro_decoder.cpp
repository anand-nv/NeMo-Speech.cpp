// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime.h"
#include "tts/kokoro/acoustic.h"
#include "tts/kokoro/decoder.h"
#include "tts/kokoro/duration.h"
#include "tts/kokoro/model.h"
#include "tts/kokoro/plbert.h"
#include "tts/kokoro/prosody.h"

int
main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: test_kokoro_decoder MODEL.gguf [OUTPUT.f32]\n";
        return 2;
    }
    const std::vector<int32_t> ids = {0, 50, 83, 156, 54, 57, 0};
    ggml_runtime::GGUFLoader loader(argv[1]);
    nemo_speech::tts::kokoro::KokoroModelMetadata metadata(loader);
    const auto voice = metadata.read_voice_style(loader, "af_heart", 5);
    const std::vector<float> decoder_style(voice.begin(), voice.begin() + 128);
    const std::vector<float> duration_style(voice.begin() + 128, voice.end());
    nemo_speech::tts::kokoro::KokoroPlbertEncoder plbert(argv[1], false);
    const auto projected = plbert.encode(ids);
    nemo_speech::tts::kokoro::KokoroDurationPredictor duration(argv[1], false);
    const auto timing = duration.predict(projected, ids.size(), duration_style);
    nemo_speech::tts::kokoro::KokoroAcousticEncoder acoustic(argv[1], false);
    const auto features = acoustic.encode(ids, timing.frames, timing.encoded_features);
    nemo_speech::tts::kokoro::KokoroProsodyHeads heads(argv[1], false);
    const auto prosody =
        heads.predict(features.prosody_shared, features.frame_count, duration_style);
    nemo_speech::tts::kokoro::KokoroDecoderEncoder decoder(argv[1], false);
    const auto output = decoder.encode(
        features.text, features.frame_count, prosody.f0, prosody.noise, decoder_style);
    if (output.size() != 108 * 512) {
        throw std::runtime_error("Kokoro decoder output shape mismatch");
    }
    std::vector<float> ranged;
    for (const auto& [begin, end] : {std::pair<size_t, size_t>{0, 40}, {40, 80}, {80, 108}}) {
        const auto tile = decoder.encode_range(
            features.text, features.frame_count, prosody.f0, prosody.noise, decoder_style, begin,
            end);
        ranged.insert(ranged.end(), tile.begin(), tile.end());
    }
    if (ranged != output) {
        throw std::runtime_error("Kokoro bounded decoder range parity failed");
    }
    if (argc == 3) {
        std::ofstream stream(argv[2], std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(output.data()),
            static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!stream)
            throw std::runtime_error("failed to write decoder output");
    }
    std::cout << "Kokoro native decoder encoder smoke test passed\n";
    return 0;
}
