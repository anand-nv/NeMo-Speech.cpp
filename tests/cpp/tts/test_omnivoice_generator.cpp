// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <stdexcept>

#include "runtime/ggml/runtime.h"
#include "tts/omnivoice/generator.h"

int
main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--gpu")) {
        std::cerr << "usage: test_omnivoice_generator MODEL.gguf [--gpu]\n";
        return 2;
    }
    ggml_runtime::Params params;
    params.use_gpu = argc == 3;
    ggml_runtime::BackendManager backends(params);
    nemo_speech::tts::omnivoice::Generator generator(backends, argv[1]);
    nemo_speech::tts::omnivoice::TokenGenerationRequest request;
    request.text = "Hello.";
    request.language = "English";
    request.target_frames = 1;
    nemo_speech::tts::omnivoice::GenerationConfig config;
    config.steps = 1;
    config.seed = 7;
    config.position_temperature = 0.0f;
    config.class_temperature = 0.0f;
    const auto first = generator.generate(request, config);
    const auto second = generator.generate(request, config);
    if (first.codebooks != second.codebooks || first.effective_seed != 7) {
        throw std::runtime_error("OmniVoice same-seed generation is not repeatable");
    }
    for (const auto& codebook : first.codebooks) {
        if (codebook.size() != 1 || codebook[0] < 0 || codebook[0] >= 1024) {
            throw std::runtime_error("OmniVoice generated an invalid codebook token");
        }
    }
    std::cout << "OmniVoice token-generation smoke test passed\n";
    return 0;
}
