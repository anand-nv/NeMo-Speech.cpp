// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/json.h"
#include "runtime.h"
#include "tts/kokoro/kokoro_runtime.h"
#include "tts/kokoro/model.h"
#include "tts/kokoro/tokenizer/kokoro_tokenizer.h"

#ifndef KOKORO_FRONTEND_REFERENCE
#error KOKORO_FRONTEND_REFERENCE must name the checked-in oracle manifest
#endif

namespace {

using nemo_speech::json::Value;
using nemo_speech::tts::kokoro::KokoroChunk;
using nemo_speech::tts::kokoro::KokoroModelMetadata;
using nemo_speech::tts::kokoro::KokoroPreparedText;
using nemo_speech::tts::kokoro::KokoroRuntime;
using nemo_speech::tts::kokoro::KokoroRuntimeConfig;
using nemo_speech::tts::kokoro::KokoroTokenizer;

std::string
read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open Kokoro reference: " + path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void
require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::string
expected_phonemes(const Value& fixture, const std::string& separator) {
    std::string output;
    for (const Value& chunk : fixture.at("chunks").array()) {
        if (!output.empty())
            output += separator;
        output += chunk.at("phonemes").string();
    }
    return output;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc != 2) {
        throw std::runtime_error("usage: test_kokoro_frontend_reference MODEL.gguf");
    }
    const Value reference = Value::parse(read_file(KOKORO_FRONTEND_REFERENCE));
    require(reference.at("format").number() == 2, "unexpected Kokoro reference format");
    require(
        reference.at("misaki_commit").string() == "fba1236595f2d2bf21d414ba6e57d25256afada3",
        "Kokoro reference has the wrong Misaki pin");

    ggml_runtime::GGUFLoader loader(argv[1]);
    KokoroModelMetadata metadata(loader);
    KokoroTokenizer vocabulary(metadata.vocabulary());
    KokoroRuntimeConfig config;
    config.model_path = argv[1];
    config.use_gpu = false;
    KokoroRuntime runtime(std::move(config));

    std::set<std::string> languages;
    size_t exact_cases = 0;
    size_t boundary_cases = 0;
    for (const Value& fixture : reference.at("fixtures").array()) {
        const std::string id = fixture.at("id").string();
        const std::string language = fixture.at("kokoro_language").string();
        const std::string voice = fixture.at("voice").string();
        languages.insert(KokoroTokenizer::canonicalize_language(language));

        if (const Value* direct = fixture.find("phonemes"); direct && direct->is_string()) {
            const KokoroChunk expected = vocabulary.tokenize_phonemes(direct->string(), language);
            const KokoroChunk actual = runtime.prepare_tokens(expected.ids, language, voice);
            require(actual.ids == expected.ids, id + ": direct phoneme IDs changed");
            require(actual.ids.size() == direct->string().size(), id + ": boundary size changed");
            ++boundary_cases;
            continue;
        }

        const std::string source = fixture.at("source_text").string();
        const KokoroPreparedText actual = runtime.prepare(source, language, voice);
        require(
            actual.tokenization.normalized_text == fixture.at("normalized_text").string(),
            id + ": normalized text mismatch");
        std::string reconstructed_text;
        std::string actual_phonemes;
        for (const KokoroChunk& chunk : actual.tokenization.chunks) {
            require(
                !chunk.ids.empty() && chunk.ids.size() <= 400,
                id + ": native chunk violates the ordinary limit");
            reconstructed_text += chunk.text;
            actual_phonemes += chunk.phonemes;
        }
        require(
            reconstructed_text == actual.tokenization.normalized_text,
            id + ": native splitting lost source text");

        if (id == "en-us-long-paragraph") {
            // Misaki strips the whitespace on both sides of its 510-character
            // chunk boundary. The native 400-ID policy retains that separator
            // on the preceding chunk, so reconstruct the unsplit oracle text.
            require(
                actual.tokenization.chunks.size() > 1,
                "long reference did not exercise native chunking");
            require(
                actual_phonemes == expected_phonemes(fixture, " "),
                "long reference phonemes changed outside chunk boundaries");
            continue;
        }

        const auto& chunks = fixture.at("chunks").array();
        require(
            actual.tokenization.chunks.size() == chunks.size(),
            id + ": reference chunk count mismatch");
        for (size_t index = 0; index < chunks.size(); ++index) {
            const KokoroChunk& got = actual.tokenization.chunks[index];
            const Value& expected = chunks[index];
            require(
                got.processed_text == KokoroTokenizer::normalize_nfkc(expected.at("text").string()),
                id + ": processed chunk text mismatch");
            require(
                got.phonemes == expected.at("phonemes").string(),
                id + ": byte-exact Misaki phoneme mismatch (native=" + got.phonemes +
                    ", Misaki=" + expected.at("phonemes").string() + ")");
            const KokoroChunk expected_ids =
                vocabulary.tokenize_phonemes(expected.at("phonemes").string(), language);
            require(got.ids == expected_ids.ids, id + ": phoneme ID mismatch");
            require(
                got.source_begin == 0 &&
                    got.source_end == actual.tokenization.normalized_text.size(),
                id + ": source span mismatch");

            if (language == "a" || language == "b") {
                const auto& expected_tokens = expected.at("tokens").array();
                require(
                    got.tokens.size() == expected_tokens.size(), id + ": MToken count mismatch");
                for (size_t token_index = 0; token_index < expected_tokens.size(); ++token_index) {
                    const auto& want = expected_tokens[token_index];
                    const auto& token = got.tokens[token_index];
                    require(token.text == want.at("text").string(), id + ": MToken text mismatch");
                    require(
                        token.tag == want.at("tag").string(),
                        id + ": MToken tag mismatch at " + std::to_string(token_index) +
                            " (native=" + token.tag + ", Misaki=" + want.at("tag").string() + ")");
                    require(
                        token.whitespace == want.at("whitespace").string(),
                        id + ": MToken whitespace mismatch");
                    require(
                        token.phonemes && *token.phonemes == want.at("phonemes").string(),
                        id + ": MToken phonemes mismatch");
                    const auto& features = want.at("_");
                    require(
                        token.features.is_head == features.at("is_head").boolean(),
                        id + ": MToken is_head mismatch");
                    require(
                        token.features.prespace == features.at("prespace").boolean(),
                        id + ": MToken prespace mismatch");
                    require(
                        token.features.num_flags == features.at("num_flags").string(),
                        id + ": MToken num_flags mismatch");
                    const Value* stress = features.find("stress");
                    require(
                        ((!stress || stress->is_null()) && !token.features.stress) ||
                            (stress && stress->is_number() && token.features.stress &&
                             *token.features.stress == stress->number()),
                        id + ": MToken stress mismatch");
                    const Value* currency = features.find("currency");
                    if (currency && currency->is_string()) {
                        require(
                            token.features.currency &&
                                *token.features.currency == currency->string(),
                            id + ": MToken currency mismatch");
                    } else {
                        require(!token.features.currency, id + ": unexpected MToken currency");
                    }
                    const Value* rating = want.find("rating");
                    if (!rating || rating->is_null())
                        rating = features.find("rating");
                    if (rating && rating->is_number()) {
                        require(
                            token.features.rating && *token.features.rating == rating->number(),
                            id + ": MToken rating mismatch");
                    }
                }
            }
        }
        ++exact_cases;
    }

    require(languages.size() == 9, "Kokoro reference did not cover all language modes");
    require(exact_cases == 22, "unexpected exact Kokoro frontend case count");
    require(boundary_cases == 4, "unexpected Kokoro boundary case count");
    return 0;
}
