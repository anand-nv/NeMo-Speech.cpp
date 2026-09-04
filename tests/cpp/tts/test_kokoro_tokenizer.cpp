// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tts/kokoro/sha256.h"
#include "tts/kokoro/tokenizer/english_g2p.h"
#include "tts/kokoro/tokenizer/espeak_g2p.h"
#include "tts/kokoro/tokenizer/japanese_g2p.h"
#include "tts/kokoro/tokenizer/kokoro_tokenizer.h"
#include "tts/kokoro/tokenizer/mandarin_g2p.h"

using nemo_speech::tts::kokoro::EnglishG2P;
using nemo_speech::tts::kokoro::EspeakG2P;
using nemo_speech::tts::kokoro::KokoroTokenizer;
using nemo_speech::tts::kokoro::MandarinG2P;

#ifndef KOKORO_MANDARIN_DATA_DIR
#define KOKORO_MANDARIN_DATA_DIR ""
#endif

namespace {

void
require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<std::string>
test_vocab() {
    std::vector<std::string> value(178);
    value[1] = ";";
    value[3] = ",";
    value[4] = ".";
    value[5] = "!";
    value[16] = " ";
    value[43] = "a";
    value[44] = "b";
    value[47] = "e";
    value[50] = "h";
    value[54] = "l";
    value[57] = "o";
    value[60] = "r";
    value[65] = "w";
    value[76] = "ɔ";
    value[83] = "ə";
    value[87] = "ɜ";
    value[123] = "ɹ";
    value[156] = "ˈ";
    return value;
}

template <typename Fn>
void
expect_throw(Fn&& fn) {
    try {
        fn();
    }
    catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("expected operation to throw");
}

void
test_language_and_voice() {
    require(KokoroTokenizer::canonicalize_language(" EN_us ") == "en-US", "en-US alias");
    require(KokoroTokenizer::canonicalize_language("cmn") == "zh-CN", "Mandarin alias");
    require(
        KokoroTokenizer::canonicalize_voice(" Kokoro.AF_HEART ") == "af_heart", "qualified voice");
    require(KokoroTokenizer::language_for_voice("bf_emma") == "en-GB", "voice locale");
    require(KokoroTokenizer::voice_matches_language("zf_xiaobei", "zh"), "voice match");
    require(!KokoroTokenizer::voice_matches_language("zf_xiaobei", "en"), "voice mismatch");
    expect_throw([] { KokoroTokenizer::canonicalize_language("de-DE"); });
    expect_throw([] { KokoroTokenizer::canonicalize_voice("not-a-voice"); });
}

void
test_utf8_and_vocabulary() {
    require(
        nemo_speech::tts::kokoro::sha256_hex("abc") ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "native SHA-256");
    KokoroTokenizer tokenizer(
        test_vocab(), [](const std::string& text, const std::string&) { return text; });
    require(KokoroTokenizer::normalize_nfkc("ＡＢＣ") == "ABC", "NFKC");
    const auto chunk = tokenizer.tokenize_phonemes("həˈlo!☃", "en-us", 7, 12);
    require(chunk.ids == std::vector<int32_t>({50, 83, 156, 54, 57, 5}), "vocabulary IDs");
    require(chunk.dropped.size() == 1, "unknown phoneme count");
    require(chunk.dropped[0].codepoint == 0x2603, "unknown phoneme code point");
    require(chunk.dropped[0].source_begin == 7, "unknown phoneme source span");
    expect_throw([&] { tokenizer.tokenize_phonemes("☃", "en-US"); });
    expect_throw([&] { KokoroTokenizer::normalize_nfkc(std::string("\xc0\xaf", 2)); });
}

void
test_chunking() {
    auto vocab = test_vocab();
    KokoroTokenizer tokenizer(std::move(vocab), [](const std::string& text, const std::string&) {
        std::string output;
        for (char byte : text) {
            if (byte == 'x')
                output += 'a';
            if (byte == '.')
                output += '.';
            if (byte == ' ')
                output += ' ';
        }
        return output;
    });
    std::string text(210, 'x');
    text += ". ";
    text += std::string(210, 'x');
    text += ".";
    const auto result = tokenizer.tokenize(text, "a");
    require(result.language == "en-US", "canonical result language");
    require(result.chunks.size() == 2, "chunk count");
    require(result.chunks[0].ids.size() == 212, "first chunk length");
    require(result.chunks[1].ids.size() == 211, "second chunk length");
    require(
        result.chunks[0].source_end == result.chunks[1].source_begin, "contiguous source spans");
    require(
        result.chunks[0].text + result.chunks[1].text == result.normalized_text,
        "source preservation");

    const auto unbroken = tokenizer.tokenize(std::string(511, 'x'), "en-US");
    require(unbroken.chunks.size() == 2, "unbroken overlong text is split");
    std::string reconstructed;
    for (const auto& chunk : unbroken.chunks) {
        require(chunk.ids.size() <= 400, "ordinary chunk limit");
        reconstructed += chunk.text;
    }
    require(reconstructed == unbroken.normalized_text, "unbroken source preservation");

    const auto whitespace =
        tokenizer.tokenize(std::string(250, 'x') + " " + std::string(250, 'x'), "en-US");
    require(whitespace.chunks.size() == 2, "whitespace split count");
    require(whitespace.chunks[0].source_end == 251, "whitespace-first split");
    require(
        whitespace.chunks[0].text + whitespace.chunks[1].text == whitespace.normalized_text,
        "whitespace source preservation");

    for (size_t length : {size_t{1}, size_t{400}, size_t{509}, size_t{510}}) {
        const auto direct = tokenizer.tokenize_phonemes(std::string(length, 'a'), "en-US");
        require(direct.ids.size() == length, "direct phoneme boundary");
    }
    expect_throw([&] { tokenizer.tokenize_phonemes(std::string(511, 'a'), "en-US"); });
}

void
test_native_espeak() {
    require(EspeakG2P::version().rfind("1.51", 0) == 0, "eSpeak version");
    require(!EspeakG2P::data_path().empty(), "eSpeak data path");
    require(EspeakG2P::phonemize("Ciao!", "it") == "ʧˈao!", "Italian eSpeak");
    require(EspeakG2P::fallback("hello", false) == "həlˈO", "English fallback");
}

void
test_native_english_lexicon() {
    const std::string gold = R"({
        "apple":"ˈæpəl","cat":"kˈæt","hello":"həlˈO","the":"ðə",
        "to":"tˈu","world":"wˈɜɹld","one":"wˈʌn",
        "hundred":"hˈʌndɹəd","twenty":"twˈɛnti","three":"θɹˈi",
        "and":"ænd"
    })";
    EnglishG2P american(gold, "{}", false);
    const auto greeting = american.phonemize("hello world!");
    require(greeting.phonemes == "həlˈO wˈɜɹld!", "English lexicon phrase");
    require(greeting.tokens.size() == 3, "English token metadata");
    require(greeting.tokens[0].tag == "UH", "English coarse POS");
    require(greeting.tokens[0].whitespace == " ", "English trailing whitespace");
    require(greeting.tokens[0].source_begin == 0, "English source begin");
    require(greeting.tokens[1].source_end == 11, "English source end");
    require(american.phonemize("the apple").phonemes == "ði ˈæpəl", "English contextual the");
    require(american.phonemize("to apple").phonemes == "tʊ ˈæpəl", "English contextual to");
    require(american.phonemize("cats").phonemes == "kˈæts", "English plural morphology");

    const auto direct = american.phonemize("[hello world](/hˈɛlo wˈɜɹld/)!");
    require(direct.text == "hello world!", "English inline pronunciation preprocessing");
    require(direct.phonemes == "hˈɛlo wˈɜɹld!", "English inline pronunciation");
    require(direct.tokens.size() == 2, "English inline token folding");
    require(direct.tokens[0].text == "hello world", "English folded token text");
    require(direct.tokens[0].features.rating == 5, "English inline rating");
    require(
        direct.tokens[0].source_begin == 1 && direct.tokens[0].source_end == 12,
        "English inline raw source span");

    const auto stress = american.phonemize("[hello](-2) [world](-1)");
    require(stress.phonemes == "həlO wˌɜɹld", "English inline stress controls");
    require(stress.tokens[0].features.stress == -2.0, "English stress metadata");
    require(stress.tokens[1].features.stress == -1.0, "English secondary stress metadata");

    require(
        american.phonemize("[123](#a#)").phonemes == "ə hˈʌndɹəd twˈɛnti θɹˈi",
        "English number article flag");
    require(
        american.phonemize("[123](#n#)").phonemes == "wˈʌn hˈʌndɹədən twˈɛnti θɹˈi",
        "English number reduced-and flag");
    require(
        american.phonemize("[123](#&#)").phonemes == "wˈʌn hˈʌndɹəd ænd twˈɛnti θɹˈi",
        "English number explicit-and flag");
}

void
test_mandarin() {
    require(MandarinG2P::pinyin_to_ipa("ni3") == "ni↓", "Mandarin third tone");
    require(MandarinG2P::pinyin_to_ipa("hao3") == "xau̯↓", "Mandarin h variant");
    const std::string shi = MandarinG2P::pinyin_to_ipa("shi4");
    if (shi != "ʂɨ↘") {
        throw std::runtime_error("Mandarin apical vowel: got '" + shi + "'");
    }
    require(MandarinG2P::pinyin_to_ipa("yuan2") == "ɥɛ↗n", "Mandarin strict final");
    MandarinG2P mandarin;
    const std::string greeting = mandarin.phonemize("你好！");
    if (greeting != "ni↓xau↓!") {
        throw std::runtime_error("Mandarin phrase and punctuation: got '" + greeting + "'");
    }
    require(mandarin.phonemize("还可以") == "xai↗ kʰɤ↓i↓", "Mandarin base pypinyin data");
    require(mandarin.phonemize("嗯") == "n↗", "Mandarin syllabic interjection");
    require(mandarin.phonemize("101") == "i↘pai↓ li↗ŋi→", "Mandarin interior zero");
    require(mandarin.phonemize("10001") == "i→wa↘n li↗ŋi→", "Mandarin group zero");
    require(mandarin.phonemize("2024年") == "ɚ↘li↗ŋɚ↘ sɨ↘njɛ↗n", "Mandarin year normalization");
}

void
test_mandarin_data_integrity() {
    namespace fs = std::filesystem;
    const fs::path source(KOKORO_MANDARIN_DATA_DIR);
    const fs::path temporary =
        fs::temp_directory_path() /
        ("nemo-speech-kokoro-mandarin-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        fs::path path;
        ~Cleanup() {
            std::error_code error;
            fs::remove_all(path, error);
        }
    } cleanup{temporary};
    fs::create_directories(temporary);
    for (const char* name :
         {"jieba.dict.utf8", "hmm_model.utf8", "misaki_pinyin_chars.tsv",
          "misaki_pinyin_phrases.tsv", "manifest.json"}) {
        fs::copy_file(source / name, temporary / name);
    }
    require(
        MandarinG2P(temporary).phonemize("你好！") == "ni↓xau↓!",
        "copied Mandarin data passes integrity validation");
    {
        std::ofstream corrupt(
            temporary / "misaki_pinyin_chars.tsv", std::ios::binary | std::ios::app);
        corrupt.put('\n');
        require(static_cast<bool>(corrupt), "corrupt Mandarin test fixture");
    }
    expect_throw([&] { MandarinG2P rejected(temporary); });
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--japanese") {
        nemo_speech::tts::kokoro::JapaneseG2P japanese;
        std::cout << japanese.phonemize(argv[2]);
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--mandarin") {
        MandarinG2P mandarin;
        std::cout << mandarin.phonemize(argv[2]);
        return 0;
    }
    if (argc >= 3 && std::string(argv[1]) == "--pinyin") {
        for (int index = 2; index < argc; ++index) {
            std::cout << argv[index] << '\t' << MandarinG2P::pinyin_to_ipa(argv[index]) << '\n';
        }
        return 0;
    }
    test_language_and_voice();
    test_utf8_and_vocabulary();
    test_chunking();
    test_native_espeak();
    test_native_english_lexicon();
    test_mandarin();
    test_mandarin_data_integrity();
    std::cout << "Kokoro tokenizer tests passed\n";
    return 0;
}
