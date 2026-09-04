// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "runtime/ggml/runtime.h"
#include "tts/omnivoice/codec.h"

int
main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--gpu")) {
        std::cerr << "usage: test_omnivoice_codec_encoder CODEC.gguf [--gpu]\n";
        return 2;
    }
    ggml_runtime::Params params;
    const bool use_gpu = argc == 3;
    params.use_gpu = use_gpu;
    ggml_runtime::BackendManager backends(params);
    nemo_speech::tts::omnivoice::CodecEncoder encoder(backends, argv[1]);

    std::vector<float> pcm(960);
    constexpr float pi = 3.14159265358979323846f;
    for (size_t index = 0; index < pcm.size(); ++index) {
        const float time = static_cast<float>(index) / 24000.0f;
        pcm[index] = 0.1f * std::sin(2.0f * pi * 440.0f * time) +
                     0.025f * std::sin(2.0f * pi * 997.0f * time);
    }
    const auto first = encoder.encode(pcm);
    const auto second = encoder.encode(pcm);
    if (first != second) {
        std::cerr << "first:";
        for (const auto& codebook : first) std::cerr << " " << codebook[0];
        std::cerr << "\nsecond:";
        for (const auto& codebook : second) std::cerr << " " << codebook[0];
        std::cerr << "\n";
        throw std::runtime_error("Higgs Audio V2 encoder is not repeatable");
    }
    for (const auto& codebook : first) {
        if (codebook.size() != 1 || codebook[0] < 0 || codebook[0] >= 1024) {
            throw std::runtime_error("Higgs Audio V2 encoder produced an invalid code");
        }
    }
    const std::array<int32_t, 8> expected_cpu = {605, 990, 995, 563, 495, 942, 461, 395};
    const std::array<int32_t, 8> expected_cuda = {605, 990, 995, 188, 495, 942, 476, 395};
    const auto& expected = use_gpu ? expected_cuda : expected_cpu;
    for (size_t q = 0; q < first.size(); ++q) {
        if (first[q][0] != expected[q]) {
            std::cerr << "codes:";
            for (const auto& codebook : first) std::cerr << " " << codebook[0];
            std::cerr << "\n";
            throw std::runtime_error("Higgs Audio V2 encoder disagrees with PyTorch oracle");
        }
    }
    std::cout << "Higgs Audio V2 encoder codes:";
    for (const auto& codebook : first) std::cout << " " << codebook[0];
    std::cout << "\n";
    return 0;
}
