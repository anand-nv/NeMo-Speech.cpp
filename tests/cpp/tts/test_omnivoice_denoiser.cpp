// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "runtime/ggml/runtime.h"
#include "tts/omnivoice/denoiser.h"

int
main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--gpu")) {
        std::cerr << "usage: test_omnivoice_denoiser MODEL.gguf [--gpu]\n";
        return 2;
    }
    ggml_runtime::Params params;
    params.use_gpu = argc == 3;
    ggml_runtime::BackendManager backends(params);
    nemo_speech::tts::omnivoice::Denoiser denoiser(backends, argv[1]);

    nemo_speech::tts::omnivoice::DenoiserInput input;
    input.batch_size = 1;
    input.sequence_length = 2;
    input.text_ids = {9707, 0};
    input.audio_mask = {0.0f, 0.0f};
    input.position_ids = {0, 1};
    input.attention_mask.assign(4, 0.0f);
    for (int codebook = 0; codebook < 8; ++codebook) {
        input.shifted_audio_ids.push_back(codebook * 1025);
        input.shifted_audio_ids.push_back(codebook * 1025);
    }

    const auto output = denoiser.forward(input);
    if (output.logits.size() != 8U * 2U * 1025U) {
        throw std::runtime_error("unexpected OmniVoice logit count");
    }
    for (float value : output.logits) {
        if (!std::isfinite(value))
            throw std::runtime_error("non-finite OmniVoice logit");
    }

    // Golden argmax values from the pinned Python reference model. This small
    // check is intentionally stronger than shape/finite-value smoke coverage:
    // it detects accidental head/sequence transposes around flash attention.
    constexpr std::array<int32_t, 16> expected_argmax = {
        643, 643, 339, 325, 821, 821, 581, 581, 966, 966, 659, 659, 736, 736, 598, 534,
    };
    for (size_t codebook = 0; codebook < 8; ++codebook) {
        for (size_t position = 0; position < 2; ++position) {
            const size_t offset = (codebook * 2 + position) * 1025;
            const auto begin = output.logits.begin() + static_cast<std::ptrdiff_t>(offset);
            const auto actual =
                static_cast<int32_t>(std::distance(begin, std::max_element(begin, begin + 1025)));
            const auto expected = expected_argmax[codebook * 2 + position];
            if (actual != expected) {
                throw std::runtime_error(
                    "OmniVoice denoiser golden argmax mismatch at codebook " +
                    std::to_string(codebook) + ", position " + std::to_string(position) +
                    ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
            }
        }
    }
    const auto repeated = denoiser.forward(input);
    if (repeated.logits != output.logits) {
        throw std::runtime_error("CPU OmniVoice denoiser is not repeatable");
    }
    std::cout << "OmniVoice denoiser parity test passed\n";
    return 0;
}
