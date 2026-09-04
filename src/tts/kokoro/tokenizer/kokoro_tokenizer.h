// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Native C++ behavioral port of the Kokoro-selected paths in Misaki 0.9.4.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nemo_speech::tts::kokoro {

inline constexpr const char* kMisakiVersion = "0.9.4";
inline constexpr const char* kMisakiCommit = "fba1236595f2d2bf21d414ba6e57d25256afada3";

struct MTokenFeatures {
    bool is_head = true;
    bool prespace = false;
    std::optional<std::string> alias;
    std::optional<double> stress;
    std::optional<std::string> currency;
    std::string num_flags;
    std::optional<int> rating;
};

struct MToken {
    std::string text;
    std::string tag;
    std::string whitespace;
    std::optional<std::string> phonemes;
    std::optional<double> start_ts;
    std::optional<double> end_ts;
    MTokenFeatures features;
    size_t source_begin = 0;
    size_t source_end = 0;
};

struct DroppedPhoneme {
    uint32_t codepoint = 0;
    std::string utf8;
    std::string language;
    size_t source_begin = 0;
    size_t source_end = 0;
};

struct KokoroChunk {
    // Text after language-frontend preprocessing (for example, with Misaki
    // inline-pronunciation markup removed). `text` remains the normalized raw
    // source slice addressed by source_begin/source_end.
    std::string processed_text;
    std::string text;
    std::string phonemes;
    std::vector<int32_t> ids;
    std::vector<MToken> tokens;
    std::vector<DroppedPhoneme> dropped;
    size_t source_begin = 0;
    size_t source_end = 0;
};

struct KokoroTokenization {
    std::string normalized_text;
    std::string language;
    std::vector<KokoroChunk> chunks;
};

struct EnglishLexiconData {
    std::string us_gold_json;
    std::string us_silver_json;
    std::string gb_gold_json;
    std::string gb_silver_json;

    bool empty() const {
        return us_gold_json.empty() && us_silver_json.empty() && gb_gold_json.empty() &&
               gb_silver_json.empty();
    }
};

class EnglishG2P;
class JapaneseG2P;

class KokoroTokenizer {
   public:
    using G2P = std::function<std::string(const std::string&, const std::string&)>;

    // Vocabulary must be indexed by Kokoro token ID and normally comes from
    // kokoro.tokenizer.tokens in the model GGUF. ID 0 is reserved for BOS/EOS.
    explicit KokoroTokenizer(
        std::vector<std::string> vocabulary, G2P g2p = {}, EnglishLexiconData english_lexicons = {},
        std::string japanese_words = {});

    KokoroTokenization tokenize(const std::string& utf8_text, const std::string& language) const;
    KokoroChunk tokenize_phonemes(
        const std::string& phonemes, const std::string& language, size_t source_begin = 0,
        size_t source_end = 0) const;

    static std::string canonicalize_language(const std::string& language);
    static std::string canonicalize_voice(const std::string& voice);
    static std::string language_for_voice(const std::string& voice);
    static bool voice_matches_language(const std::string& voice, const std::string& language);
    static std::string normalize_nfkc(const std::string& utf8_text);

   private:
    struct Phonemization {
        std::string text;
        std::string phonemes;
        std::vector<MToken> tokens;
    };

    struct SourceUnit {
        std::string text;
        size_t begin = 0;
        size_t end = 0;
    };

    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    G2P g2p_;
    std::shared_ptr<EnglishG2P> american_english_;
    std::shared_ptr<EnglishG2P> british_english_;
    mutable std::shared_ptr<JapaneseG2P> japanese_;
    mutable std::mutex japanese_mutex_;
    std::string japanese_words_;

    Phonemization phonemize(const std::string& text, const std::string& language) const;
    std::vector<SourceUnit> split_source(const std::string& normalized) const;
    std::vector<SourceUnit> split_unit_to_fit(
        const SourceUnit& unit, const std::string& language) const;
};

}  // namespace nemo_speech::tts::kokoro
