// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <stdexcept>

#include "runtime/ggml/runtime.h"
#include "tts/omnivoice/model.h"

int
main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_omnivoice_loader MODEL.gguf\n";
        return 2;
    }
    ggml_runtime::GGUFLoader loader(argv[1]);
    const auto cfg = nemo_speech::tts::omnivoice::load_model_config(loader);
    if (cfg.text_vocab_size != 151676 || cfg.audio_vocab_size != 1025 || cfg.audio_codebooks != 8 ||
        cfg.hidden_size != 1024 || cfg.layer_count != 28 || cfg.context_length != 40960 ||
        loader.get_tensor_names().size() != 313) {
        throw std::runtime_error("unexpected OmniVoice model configuration");
    }
    std::cout << "validated OmniVoice GGUF revision " << cfg.source_revision << "\n";
    return 0;
}
