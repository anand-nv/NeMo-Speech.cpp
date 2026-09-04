// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/ggml/runtime.h"
#include "tts/omnivoice/frontend.h"

int
main(int argc, char** argv) {
    using namespace nemo_speech::tts::omnivoice;
    if (combine_text("  world\n next  ", std::string(" Hello ")) != "Hello world next" ||
        combine_text(" 你好 \t 世界 ", std::string("测试（ok）")) != "测试(ok)你好世界") {
        throw std::runtime_error("OmniVoice combined-text behavior differs from oracle");
    }
    if (add_terminal_punctuation("hello") != "hello." ||
        add_terminal_punctuation("你好") != "你好。" || !add_terminal_punctuation("").empty()) {
        throw std::runtime_error("OmniVoice terminal punctuation behavior differs from oracle");
    }
    if (chunk_text_punctuation("Mr. Smith went home. Then slept!", 12) !=
            std::vector<std::string>({"Mr. Smith went home.", "Then slept!"}) ||
        chunk_text_punctuation("One. Two. Three.", 8, 5) !=
            std::vector<std::string>({"One. Two.", "Three."}) ||
        chunk_text_punctuation("第一句。第二句！第三句？", 5) !=
            std::vector<std::string>({"第一句。", "第二句！", "第三句？"})) {
        throw std::runtime_error("OmniVoice punctuation chunking differs from oracle");
    }

    if (argc == 2) {
        ggml_runtime::GGUFLoader loader(argv[1]);
        FrontendTables tables(loader);
        if (tables.resolve_language(std::string("English")) != std::optional<std::string>("en") ||
            tables.resolve_language(std::string("en")) != std::optional<std::string>("en") ||
            tables.resolve_language(std::string("not-a-language"))) {
            throw std::runtime_error("OmniVoice language resolution differs from oracle");
        }
        if (tables.resolve_instruction(std::string("male, high pitch"), false) !=
                std::optional<std::string>("male, high pitch") ||
            tables.resolve_instruction(std::string("male, high pitch"), true) !=
                std::optional<std::string>("男，高音调")) {
            throw std::runtime_error("OmniVoice instruction resolution differs from oracle");
        }
        bool conflict = false;
        try {
            (void)tables.resolve_instruction(std::string("male, female"), false);
        }
        catch (const std::invalid_argument&) {
            conflict = true;
        }
        if (!conflict)
            throw std::runtime_error("OmniVoice instruction conflict was accepted");
        if (std::fabs(tables.character_weight("Hello, world.") - 11.2) > 1.0e-9 ||
            std::fabs(tables.character_weight("你好，世界！") - 13.0) > 1.0e-9 ||
            std::fabs(tables.character_weight("नमस्ते दुनिया") - 12.8) > 1.0e-9 ||
            std::fabs(
                tables.estimate_frames("Hello, world.", "Nice to meet you.", 25) -
                36.75301533459202) > 1.0e-9) {
            throw std::runtime_error("OmniVoice duration estimation differs from oracle");
        }
        if (tables.target_frames("hello", {}, {}, 2.0, {}, 25) <= 0 ||
            tables.target_frames("hello", {}, {}, 3.0, 1.25, 25) != 31) {
            throw std::runtime_error("OmniVoice duration/speed precedence is incorrect");
        }
    } else if (argc != 1) {
        throw std::runtime_error("usage: test_omnivoice_frontend [MODEL.gguf]");
    }
    return 0;
}
