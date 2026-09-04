// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Native port of the deterministic Misaki 0.9.4 English lexicon path.
#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kokoro_tokenizer.h"

namespace nemo_speech::tts::kokoro {

struct EnglishG2PResult {
    // Misaki's preprocessing removes supported [text](feature) markup before
    // tokenization. Keep that observable text alongside raw-source spans.
    std::string text;
    std::string phonemes;
    std::vector<MToken> tokens;
};

class EnglishG2P {
   public:
    EnglishG2P(std::string gold_json, std::string silver_json, bool british);

    EnglishG2PResult phonemize(const std::string& text) const;

   private:
    struct Entry {
        std::optional<std::string> pronunciation;
        std::unordered_map<std::string, std::optional<std::string>> variants;
    };
    struct Context {
        std::optional<bool> future_vowel;
        bool future_to = false;
    };

    bool british_ = false;
    std::unordered_map<std::string, Entry> gold_;
    std::unordered_map<std::string, Entry> silver_;

    static std::unordered_map<std::string, Entry> parse_lexicon(const std::string& json);
    static std::string coarse_tag(const std::string& word);
    static std::string parent_tag(const std::string& tag);
    static std::string apply_stress(std::string phonemes, std::optional<double> stress);
    static bool begins_with_vowel(const std::string& phonemes);
    std::optional<std::string> spell_initialism(const std::string& word) const;
    static std::string cardinal(uint64_t value, bool british);
    static std::string ordinal(uint64_t value, bool british);

    std::optional<std::string> lookup(
        const std::string& word, const std::string& tag, std::optional<double> stress,
        const Context& context) const;
    std::optional<std::string> lookup_entry(
        const std::string& word, const std::string& tag, const Context* context = nullptr) const;
    int pronunciation_rating(const std::string& word) const;
    std::optional<std::string> number(
        const std::string& word, const std::string& tag, const Context& context,
        const std::optional<std::string>& currency = std::nullopt, bool is_head = true,
        const std::string& num_flags = {}) const;
};

}  // namespace nemo_speech::tts::kokoro
