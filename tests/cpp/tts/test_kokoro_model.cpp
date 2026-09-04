// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime.h"
#include "tts/kokoro/frontend.h"
#include "tts/kokoro/model.h"
#include "tts/kokoro/tokenizer/english_g2p.h"

int
main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_kokoro_model MODEL.gguf\n";
        return 2;
    }
    ggml_runtime::GGUFLoader loader(argv[1]);
    nemo_speech::tts::kokoro::KokoroModelMetadata model(loader);
    const auto& h = model.hparams();
    if (h.sample_rate != 24000 || h.context_length != 512 || h.phoneme_limit != 510 ||
        model.vocabulary().size() != 178 || model.voice_names().size() != 54 ||
        model.misaki_lexicon_json("us_gold").size() < 1000000 ||
        model.voice_index("KOKORO.AF_HEART") != 3 ||
        model.voice_tensor_name(3) != "kokoro.voice.af_heart") {
        throw std::runtime_error("validated Kokoro metadata returned inconsistent values");
    }
    nemo_speech::tts::kokoro::EnglishG2P english(
        model.misaki_lexicon_json("us_gold"), model.misaki_lexicon_json("us_silver"), false);
    if (english.phonemize("hello world!").phonemes != "həlˈO wˈɜɹld!" ||
        english.phonemize("the apple").phonemes != "ði ˈæpᵊl" ||
        english.phonemize("to world").phonemes != "tə wˈɜɹld" ||
        english.phonemize("2024").phonemes != "twˈɛnti twˈɛnti fˈɔɹ") {
        throw std::runtime_error("embedded Misaki English lexicon behavior mismatch");
    }
    nemo_speech::tts::kokoro::KokoroFrontend frontend(argv[1]);
    const auto prepared = frontend.prepare("hello world!", "en-US", "kokoro.af_heart");
    if (prepared.voice != "af_heart" || prepared.voice_index != 3 ||
        prepared.tokenization.chunks.empty() ||
        prepared.tokenization.chunks.front().phonemes != "həlˈO wˈɜɹld!") {
        throw std::runtime_error("native Kokoro frontend integration mismatch");
    }
    if (prepared.tokenization.chunks[0].tokens.size() != 3 ||
        prepared.tokenization.chunks[0].tokens[0].text != "hello" ||
        prepared.tokenization.chunks[0].tokens[0].whitespace != " " ||
        prepared.tokenization.chunks[0].tokens[1].text != "world" ||
        prepared.tokenization.chunks[0].tokens[1].source_begin != 6 ||
        prepared.tokenization.chunks[0].tokens[2].text != "!") {
        throw std::runtime_error("native Misaki token metadata was not preserved");
    }
    const auto tokenized = frontend.prepare_tokens({50, 83, 156, 54, 57}, "en", "af_heart");
    if (tokenized.ids.size() != 5) {
        throw std::runtime_error("native Kokoro pretokenized preparation mismatch");
    }
    std::cout << "validated Kokoro model with " << model.voice_names().size() << " voices\n";
    return 0;
}
