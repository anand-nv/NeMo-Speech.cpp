// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "runtime/ggml/runtime.h"
#include "tts/omnivoice/codec.h"

int
main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: test_omnivoice_codec CODEC.gguf [WAVEFORM.f32] [--gpu]\n";
        return 2;
    }

    ggml_runtime::Params params;
    const bool use_gpu = std::string(argv[argc - 1]) == "--gpu";
    params.use_gpu = use_gpu;
    ggml_runtime::BackendManager backends(params);
    nemo_speech::tts::omnivoice::CodecDecoder decoder(backends, argv[1]);

    if (decoder.config().sample_rate != 24000 || decoder.config().hop_length != 960 ||
        decoder.config().quantizers != 8 || decoder.config().codebook_size != 1024) {
        throw std::runtime_error("unexpected Higgs Audio V2 codec configuration");
    }
    std::array<std::vector<int32_t>, 8> codes;
    for (size_t q = 0; q < codes.size(); ++q) {
        codes[q] = {static_cast<int32_t>(17 + 31 * q)};
    }
    const std::vector<float> first = decoder.decode(codes);
    const std::vector<float> second = decoder.decode(codes);
    if (first.size() != 960 || first != second ||
        !std::all_of(
            first.begin(), first.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("Higgs Audio V2 decoder smoke test failed");
    }
    const auto [minimum, maximum] = std::minmax_element(first.begin(), first.end());
    if (*minimum == *maximum) {
        throw std::runtime_error("Higgs Audio V2 decoder produced a constant waveform");
    }
    // F32 PyTorch oracle samples from the pinned Transformers DAC decoder.
    // F16 GGUF weights are permitted a small conversion/backend tolerance.
    constexpr std::array<std::pair<size_t, float>, 25> oracle = {{
        {0, 0.009410372f},     {1, 0.00919204764f},   {2, 0.0100879576f},     {3, 0.0119096525f},
        {7, 0.0158273019f},    {15, 0.0225561056f},   {31, -0.0124093099f},   {63, -0.00168563332f},
        {95, -0.0842303708f},  {127, -0.0206385087f}, {159, 0.0197277367f},   {191, -0.0532084182f},
        {255, -0.0231629964f}, {319, 0.0812508091f},  {383, 0.0627448633f},   {447, -0.0369332992f},
        {511, -0.0924761444f}, {575, 0.00527450256f}, {639, -0.00886745844f}, {703, 0.0698304698f},
        {767, 0.0509512797f},  {831, 0.0224618539f},  {895, 0.0163088664f},   {927, -0.0681121871f},
        {959, -0.0107599795f},
    }};
    for (const auto& [index, expected] : oracle) {
        if (std::fabs(first[index] - expected) > 5.0e-4f) {
            throw std::runtime_error("Higgs Audio V2 decoder disagrees with PyTorch oracle");
        }
    }
    if (argc >= 3 && std::string(argv[2]) != "--gpu") {
        std::ofstream output(argv[2], std::ios::binary);
        output.write(
            reinterpret_cast<const char*>(first.data()),
            static_cast<std::streamsize>(first.size() * sizeof(float)));
        if (!output)
            throw std::runtime_error("could not write codec test waveform");
    }
    std::cout << "Higgs Audio V2 decoder produced " << first.size()
              << " deterministic finite samples\n";
    return 0;
}
