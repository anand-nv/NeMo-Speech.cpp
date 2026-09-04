// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tts/kokoro/plbert.h"

int
main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: test_kokoro_plbert MODEL.gguf [OUTPUT.f32]\n";
        return 2;
    }
    nemo_speech::tts::kokoro::KokoroPlbertEncoder encoder(argv[1], false);
    const std::vector<float> output = encoder.encode({0, 50, 83, 156, 54, 57, 0});
    if (output.size() != 7 * 512) {
        throw std::runtime_error("Kokoro PL-BERT output size mismatch");
    }
    for (float value : output) {
        if (!std::isfinite(value))
            throw std::runtime_error("Kokoro PL-BERT output is not finite");
    }
    if (argc == 3) {
        std::ofstream stream(argv[2], std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(output.data()),
            static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!stream)
            throw std::runtime_error("failed to write PL-BERT output");
    }
    std::cout << "Kokoro PL-BERT native smoke test passed\n";
    return 0;
}
