// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/ggml/runtime.h"
#include "tts/omnivoice/tokenizer.h"

namespace {

void
expect(
    const nemo_speech::tts::omnivoice::Tokenizer& tokenizer, const std::string& text,
    std::vector<int32_t> expected) {
    const auto actual = tokenizer.encode(text);
    if (actual != expected) {
        std::cerr << "token mismatch for: " << text << "\nactual:";
        for (int32_t id : actual) std::cerr << " " << id;
        std::cerr << "\n";
        std::abort();
    }
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_omnivoice_tokenizer MODEL.gguf\n";
        return 2;
    }
    ggml_runtime::GGUFLoader loader(argv[1]);
    nemo_speech::tts::omnivoice::Tokenizer tokenizer(loader);

    expect(tokenizer, "Hello, world!", {9707, 11, 1879, 0});
    expect(tokenizer, "  multiple   spaces  ", {220, 5248, 256, 12621, 256});
    expect(tokenizer, "Cafe\xCC\x81", {34, 2577, 963});
    expect(tokenizer, "中文和 English。", {104811, 33108, 6364, 1773});
    expect(tokenizer, "مرحبا بالعالم", {124122, 29825, 124671, 124476, 129634});
    expect(tokenizer, "👩🏽‍💻 test", {145233, 145375, 378, 235, 145851, 1273});
    expect(tokenizer, "line one\r\nline two", {1056, 825, 319, 1056, 1378});
    expect(
        tokenizer, "<|denoise|><|lang_start|>English<|lang_end|>", {151669, 151670, 22574, 151671});
    if (tokenizer.token_id("<|text_start|>") != 151674 ||
        tokenizer.token_id("<|text_end|>") != 151675 ||
        tokenizer.encode_with_nonverbal_tags("你好[laughter]world") !=
            std::vector<int32_t>({108386, 58, 51783, 60, 14615})) {
        throw std::runtime_error("OmniVoice special-token handling mismatch");
    }

    bool rejected = false;
    try {
        (void)tokenizer.encode(std::string("bad\xFF", 4));
    }
    catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("OmniVoice tokenizer accepted invalid UTF-8");
    std::cout << "OmniVoice tokenizer parity cases passed\n";
    return 0;
}
