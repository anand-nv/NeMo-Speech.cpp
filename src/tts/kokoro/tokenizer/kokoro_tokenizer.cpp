// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Native C++ behavioral port of the Kokoro-selected paths in Misaki 0.9.4.
#include "kokoro_tokenizer.h"

#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "english_g2p.h"
#include "espeak_g2p.h"
#include "japanese_g2p.h"
#include "mandarin_g2p.h"
#include "nvtx_utils.h"

namespace nemo_speech::tts::kokoro {
namespace {

struct Utf8Char {
    uint32_t codepoint;
    size_t begin;
    size_t end;
};

std::vector<Utf8Char>
decode_utf8(const std::string& text) {
    std::vector<Utf8Char> output;
    for (size_t offset = 0; offset < text.size();) {
        const size_t begin = offset;
        const unsigned char first = static_cast<unsigned char>(text[offset++]);
        uint32_t codepoint = 0;
        size_t continuation = 0;
        uint32_t minimum = 0;
        if (first < 0x80) {
            codepoint = first;
        } else if ((first & 0xe0) == 0xc0) {
            codepoint = first & 0x1f;
            continuation = 1;
            minimum = 0x80;
        } else if ((first & 0xf0) == 0xe0) {
            codepoint = first & 0x0f;
            continuation = 2;
            minimum = 0x800;
        } else if ((first & 0xf8) == 0xf0) {
            codepoint = first & 0x07;
            continuation = 3;
            minimum = 0x10000;
        } else {
            throw std::invalid_argument(
                "invalid UTF-8 leading byte at offset " + std::to_string(begin));
        }
        if (continuation > text.size() - offset) {
            throw std::invalid_argument(
                "truncated UTF-8 sequence at offset " + std::to_string(begin));
        }
        for (size_t index = 0; index < continuation; ++index) {
            const unsigned char value = static_cast<unsigned char>(text[offset++]);
            if ((value & 0xc0) != 0x80) {
                throw std::invalid_argument(
                    "invalid UTF-8 continuation byte at offset " + std::to_string(offset - 1));
            }
            codepoint = (codepoint << 6) | (value & 0x3f);
        }
        if ((minimum != 0 && codepoint < minimum) || codepoint > 0x10ffff ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            throw std::invalid_argument(
                "invalid UTF-8 code point at offset " + std::to_string(begin));
        }
        output.push_back({codepoint, begin, offset});
    }
    return output;
}

std::string
ascii_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char byte) {
        return static_cast<char>(std::tolower(byte));
    });
    return value;
}

bool
is_ascii_space(std::string_view value) {
    return value.size() == 1 && std::isspace(static_cast<unsigned char>(value[0]));
}

bool
is_sentence_boundary(uint32_t cp) {
    return cp == '.' || cp == '!' || cp == '?' || cp == 0x2026 || cp == 0x3002 || cp == 0xff01 ||
           cp == 0xff1f;
}

bool
is_clause_boundary(uint32_t cp) {
    return cp == ',' || cp == ';' || cp == ':' || cp == 0x2014 || cp == 0x3001 || cp == 0xff0c ||
           cp == 0xff1b || cp == 0xff1a;
}

std::string
trim_ascii_space(std::string value) {
    const auto not_space = [](unsigned char byte) { return !std::isspace(byte); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string
preserve_misaki_trailing_space(std::string phonemes, const std::string& source) {
    const auto chars = decode_utf8(source);
    if (!chars.empty() && u_isUWhiteSpace(static_cast<UChar32>(chars.back().codepoint)) &&
        (phonemes.empty() || phonemes.back() != ' ')) {
        // KPipeline.tokens_to_ps emits one ASCII space whenever MToken.whitespace
        // is non-empty, independently of the original whitespace width.
        phonemes += ' ';
    }
    return phonemes;
}

std::string
espeak_language(const std::string& canonical) {
    if (canonical == "en-US")
        return "en-us";
    if (canonical == "en-GB")
        return "en-gb";
    if (canonical == "es-ES")
        return "es";
    if (canonical == "fr-FR")
        return "fr-fr";
    if (canonical == "hi-IN")
        return "hi";
    if (canonical == "it-IT")
        return "it";
    if (canonical == "pt-BR")
        return "pt-br";
    return {};
}

}  // namespace

KokoroTokenizer::KokoroTokenizer(
    std::vector<std::string> vocabulary, G2P g2p, EnglishLexiconData english_lexicons,
    std::string japanese_words)
    : vocabulary_(std::move(vocabulary)), g2p_(std::move(g2p)),
      japanese_words_(std::move(japanese_words)) {
    if (vocabulary_.size() != 178) {
        throw std::invalid_argument("Kokoro vocabulary must contain exactly 178 ID slots");
    }
    if (!vocabulary_[0].empty()) {
        throw std::invalid_argument("Kokoro vocabulary ID 0 must be reserved for BOS/EOS");
    }
    for (size_t id = 1; id < vocabulary_.size(); ++id) {
        if (vocabulary_[id].empty()) {
            continue;
        }
        const auto chars = decode_utf8(vocabulary_[id]);
        if (chars.size() != 1) {
            throw std::invalid_argument("Kokoro vocabulary entries must be one Unicode code point");
        }
        if (!token_to_id_.emplace(vocabulary_[id], static_cast<int32_t>(id)).second) {
            throw std::invalid_argument("duplicate Kokoro vocabulary symbol: " + vocabulary_[id]);
        }
    }
    if (!english_lexicons.empty()) {
        if (english_lexicons.us_gold_json.empty() || english_lexicons.us_silver_json.empty() ||
            english_lexicons.gb_gold_json.empty() || english_lexicons.gb_silver_json.empty()) {
            throw std::invalid_argument("all four Misaki English lexicons are required");
        }
        american_english_ = std::make_shared<EnglishG2P>(
            std::move(english_lexicons.us_gold_json), std::move(english_lexicons.us_silver_json),
            false);
        british_english_ = std::make_shared<EnglishG2P>(
            std::move(english_lexicons.gb_gold_json), std::move(english_lexicons.gb_silver_json),
            true);
    }
}

std::string
KokoroTokenizer::canonicalize_language(const std::string& language) {
    std::string value = ascii_lower(trim_ascii_space(language));
    std::replace(value.begin(), value.end(), '_', '-');
    static const std::unordered_map<std::string, std::string> aliases = {
        {"a", "en-US"},      {"en", "en-US"},    {"en-us", "en-US"}, {"b", "en-GB"},
        {"en-gb", "en-GB"},  {"en-uk", "en-GB"}, {"e", "es-ES"},     {"es", "es-ES"},
        {"es-es", "es-ES"},  {"f", "fr-FR"},     {"fr", "fr-FR"},    {"fr-fr", "fr-FR"},
        {"h", "hi-IN"},      {"hi", "hi-IN"},    {"hi-in", "hi-IN"}, {"i", "it-IT"},
        {"it", "it-IT"},     {"it-it", "it-IT"}, {"j", "ja-JP"},     {"ja", "ja-JP"},
        {"ja-jp", "ja-JP"},  {"p", "pt-BR"},     {"pt", "pt-BR"},    {"pt-br", "pt-BR"},
        {"z", "zh-CN"},      {"zh", "zh-CN"},    {"zh-cn", "zh-CN"}, {"cmn", "zh-CN"},
        {"cmn-cn", "zh-CN"},
    };
    const auto found = aliases.find(value);
    if (found == aliases.end()) {
        throw std::invalid_argument("unsupported Kokoro language: '" + language + "'");
    }
    return found->second;
}

std::string
KokoroTokenizer::canonicalize_voice(const std::string& voice) {
    std::string value = ascii_lower(trim_ascii_space(voice));
    constexpr std::string_view qualifier = "kokoro.";
    if (value.rfind(qualifier, 0) == 0) {
        value.erase(0, qualifier.size());
    }
    if (value.size() < 4 || value[2] != '_' ||
        std::string("abefhijpz").find(value[0]) == std::string::npos ||
        (value[1] != 'f' && value[1] != 'm')) {
        throw std::invalid_argument("invalid Kokoro voice name: '" + voice + "'");
    }
    for (size_t index = 3; index < value.size(); ++index) {
        const unsigned char byte = static_cast<unsigned char>(value[index]);
        if (!std::islower(byte) && !std::isdigit(byte) && byte != '_') {
            throw std::invalid_argument("invalid Kokoro voice name: '" + voice + "'");
        }
    }
    return value;
}

std::string
KokoroTokenizer::language_for_voice(const std::string& voice) {
    const std::string value = canonicalize_voice(voice);
    static const std::unordered_map<char, std::string> languages = {
        {'a', "en-US"}, {'b', "en-GB"}, {'e', "es-ES"}, {'f', "fr-FR"}, {'h', "hi-IN"},
        {'i', "it-IT"}, {'j', "ja-JP"}, {'p', "pt-BR"}, {'z', "zh-CN"},
    };
    return languages.at(value[0]);
}

bool
KokoroTokenizer::voice_matches_language(const std::string& voice, const std::string& language) {
    return language_for_voice(voice) == canonicalize_language(language);
}

std::string
KokoroTokenizer::normalize_nfkc(const std::string& utf8_text) {
    decode_utf8(utf8_text);  // Give callers a byte-accurate validation error before ICU.
    UErrorCode status = U_ZERO_ERROR;
    const UNormalizer2* normalizer = unorm2_getNFKCInstance(&status);
    if (U_FAILURE(status)) {
        throw std::runtime_error("ICU could not create the NFKC normalizer");
    }

    status = U_ZERO_ERROR;
    int32_t utf16_size = 0;
    u_strFromUTF8(
        nullptr, 0, &utf16_size, utf8_text.data(), static_cast<int32_t>(utf8_text.size()), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        throw std::invalid_argument("invalid UTF-8 text");
    }
    status = U_ZERO_ERROR;
    std::vector<UChar> utf16(static_cast<size_t>(utf16_size) + 1);
    u_strFromUTF8(
        utf16.data(), static_cast<int32_t>(utf16.size()), &utf16_size, utf8_text.data(),
        static_cast<int32_t>(utf8_text.size()), &status);
    if (U_FAILURE(status)) {
        throw std::invalid_argument("invalid UTF-8 text");
    }

    status = U_ZERO_ERROR;
    int32_t normalized_size =
        unorm2_normalize(normalizer, utf16.data(), utf16_size, nullptr, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        throw std::runtime_error("ICU NFKC normalization failed");
    }
    status = U_ZERO_ERROR;
    std::vector<UChar> normalized(static_cast<size_t>(normalized_size) + 1);
    normalized_size = unorm2_normalize(
        normalizer, utf16.data(), utf16_size, normalized.data(),
        static_cast<int32_t>(normalized.size()), &status);
    if (U_FAILURE(status)) {
        throw std::runtime_error("ICU NFKC normalization failed");
    }

    status = U_ZERO_ERROR;
    int32_t output_size = 0;
    u_strToUTF8(nullptr, 0, &output_size, normalized.data(), normalized_size, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        throw std::runtime_error("ICU UTF-8 conversion failed");
    }
    status = U_ZERO_ERROR;
    std::string output(static_cast<size_t>(output_size), '\0');
    u_strToUTF8(
        output.data(), output_size, &output_size, normalized.data(), normalized_size, &status);
    if (U_FAILURE(status)) {
        throw std::runtime_error("ICU UTF-8 conversion failed");
    }
    return output;
}

KokoroTokenizer::Phonemization
KokoroTokenizer::phonemize(const std::string& text, const std::string& language) const {
    const std::string nvtx_name = "kokoro.tokenizer.phonemize language=" + language +
                                  " text_bytes=" + std::to_string(text.size());
    const ggml_nvtx::range nvtx_range(nvtx_name.c_str());
    if (g2p_) {
        return {text, g2p_(text, language), {}};
    }
    const std::string native_language = espeak_language(language);
    if (!native_language.empty()) {
        // Until the English lexicon stage resolves a token, the selected
        // Misaki behavior is its native EspeakFallback.
        if (language == "en-US" || language == "en-GB") {
            const auto& frontend = language == "en-GB" ? british_english_ : american_english_;
            if (frontend) {
                auto result = frontend->phonemize(text);
                return {
                    std::move(result.text), std::move(result.phonemes), std::move(result.tokens)};
            }
            return {text, EspeakG2P::fallback(text, language == "en-GB"), {}};
        }
        return {text, EspeakG2P::phonemize(text, native_language), {}};
    }
    if (language == "ja-JP") {
        std::lock_guard<std::mutex> lock(japanese_mutex_);
        if (!japanese_)
            japanese_ = std::make_shared<JapaneseG2P>(japanese_words_);
        return {text, japanese_->phonemize(text), {}};
    }
    static const MandarinG2P mandarin;
    return {text, mandarin.phonemize(text), {}};
}

KokoroChunk
KokoroTokenizer::tokenize_phonemes(
    const std::string& phonemes, const std::string& raw_language, size_t source_begin,
    size_t source_end) const {
    KokoroChunk chunk;
    const std::string language = canonicalize_language(raw_language);
    chunk.phonemes = phonemes;
    chunk.source_begin = source_begin;
    chunk.source_end = source_end;
    for (const auto& cp : decode_utf8(phonemes)) {
        const std::string symbol = phonemes.substr(cp.begin, cp.end - cp.begin);
        const auto found = token_to_id_.find(symbol);
        if (found == token_to_id_.end()) {
            chunk.dropped.push_back({cp.codepoint, symbol, language, source_begin, source_end});
            continue;
        }
        chunk.ids.push_back(found->second);
    }
    if (chunk.ids.empty()) {
        throw std::invalid_argument("Kokoro phonemization produced no vocabulary IDs");
    }
    if (chunk.ids.size() > 510) {
        throw std::length_error("Kokoro phoneme chunk exceeds 510 IDs");
    }
    return chunk;
}

std::vector<KokoroTokenizer::SourceUnit>
KokoroTokenizer::split_source(const std::string& normalized) const {
    const auto chars = decode_utf8(normalized);
    std::vector<SourceUnit> output;
    size_t begin = 0;
    size_t protected_until = 0;
    for (size_t index = 0; index < chars.size(); ++index) {
        const uint32_t cp = chars[index].codepoint;
        if (chars[index].begin < protected_until)
            continue;
        if (cp == '[') {
            const size_t label_end = normalized.find(']', chars[index].end);
            if (label_end != std::string::npos && label_end + 1 < normalized.size() &&
                normalized[label_end + 1] == '(') {
                const size_t feature_end = normalized.find(')', label_end + 2);
                if (feature_end != std::string::npos) {
                    // Misaki preprocesses a complete [text](feature) span
                    // before sentence chunking. Do not let punctuation in the
                    // label or feature split the annotation into invalid text.
                    protected_until = feature_end + 1;
                    continue;
                }
            }
        }
        const bool newline = cp == '\n' || cp == '\r';
        const bool numeric_separator =
            (cp == '.' || cp == ',') && index > 0 && index + 1 < chars.size() &&
            u_isdigit(static_cast<UChar32>(chars[index - 1].codepoint)) &&
            u_isdigit(static_cast<UChar32>(chars[index + 1].codepoint));
        if (numeric_separator) {
            continue;
        }
        if (!newline && !is_sentence_boundary(cp) && !is_clause_boundary(cp)) {
            continue;
        }
        size_t end = chars[index].end;
        while (index + 1 < chars.size()) {
            const auto& next = chars[index + 1];
            const std::string_view next_text(normalized.data() + next.begin, next.end - next.begin);
            if (!is_ascii_space(next_text)) {
                break;
            }
            end = next.end;
            ++index;
        }
        output.push_back({normalized.substr(begin, end - begin), begin, end});
        begin = end;
    }
    if (begin < normalized.size()) {
        output.push_back({normalized.substr(begin), begin, normalized.size()});
    }
    if (output.empty() && !normalized.empty()) {
        output.push_back({normalized, 0, normalized.size()});
    }
    return output;
}

std::vector<KokoroTokenizer::SourceUnit>
KokoroTokenizer::split_unit_to_fit(const SourceUnit& unit, const std::string& language) const {
    constexpr size_t kOrdinaryChunkLimit = 400;

    try {
        const std::string phonemes =
            preserve_misaki_trailing_space(phonemize(unit.text, language).phonemes, unit.text);
        const KokoroChunk encoded = tokenize_phonemes(phonemes, language, unit.begin, unit.end);
        if (encoded.ids.size() <= kOrdinaryChunkLimit) {
            return {unit};
        }
    }
    catch (const std::length_error&) {
        // The source must be split and phonemized again rather than truncating
        // the already-produced phoneme stream.
    }

    const auto chars = decode_utf8(unit.text);
    if (chars.size() < 2) {
        // Re-run the public encoder to retain its precise hard-limit error.
        tokenize_phonemes(phonemize(unit.text, language).phonemes, language, unit.begin, unit.end);
        return {unit};
    }

    const size_t middle = chars.size() / 2;
    std::optional<size_t> split_index;
    size_t best_distance = std::numeric_limits<size_t>::max();
    for (size_t index = 1; index < chars.size(); ++index) {
        const auto& previous = chars[index - 1];
        const std::string_view previous_text(
            unit.text.data() + previous.begin, previous.end - previous.begin);
        if (!is_ascii_space(previous_text)) {
            continue;
        }
        const size_t distance = index > middle ? index - middle : middle - index;
        if (distance < best_distance) {
            split_index = index;
            best_distance = distance;
        }
    }
    // A whitespace split is preferable, but every decoded-codepoint boundary
    // is a safe final fallback. This guarantees progress for long CJK text or
    // unusually long words without slicing a UTF-8 sequence.
    if (!split_index) {
        split_index = middle;
    }

    const size_t relative_byte = chars[*split_index].begin;
    if (relative_byte == 0 || relative_byte >= unit.text.size()) {
        throw std::length_error("unable to split an overlong Kokoro source segment");
    }
    const size_t absolute_byte = unit.begin + relative_byte;
    const SourceUnit left{unit.text.substr(0, relative_byte), unit.begin, absolute_byte};
    const SourceUnit right{unit.text.substr(relative_byte), absolute_byte, unit.end};

    auto output = split_unit_to_fit(left, language);
    auto tail = split_unit_to_fit(right, language);
    output.insert(
        output.end(), std::make_move_iterator(tail.begin()), std::make_move_iterator(tail.end()));
    return output;
}

KokoroTokenization
KokoroTokenizer::tokenize(const std::string& utf8_text, const std::string& raw_language) const {
    const ggml_nvtx::range nvtx_range("kokoro.tokenizer.tokenize");
    const std::string nvtx_parameters = "kokoro.parameters.tokenizer raw_language=" + raw_language +
                                        " input_bytes=" + std::to_string(utf8_text.size());
    ggml_nvtx::mark(nvtx_parameters.c_str());
    KokoroTokenization result;
    result.language = canonicalize_language(raw_language);
    result.normalized_text = normalize_nfkc(utf8_text);
    if (trim_ascii_space(result.normalized_text).empty()) {
        throw std::invalid_argument("Kokoro input text is empty or whitespace only");
    }

    // Sentence boundaries are candidate split points, not independent G2P
    // calls. Keep an ordinary request in one Misaki invocation so contextual
    // pronunciation, quote adjacency, whitespace, and token metadata match
    // the reference frontend across punctuation.
    Phonemization whole = phonemize(result.normalized_text, result.language);
    whole.phonemes =
        preserve_misaki_trailing_space(std::move(whole.phonemes), result.normalized_text);
    try {
        KokoroChunk encoded =
            tokenize_phonemes(whole.phonemes, result.language, 0, result.normalized_text.size());
        if (encoded.ids.size() <= 400) {
            encoded.processed_text = std::move(whole.text);
            encoded.text = result.normalized_text;
            encoded.tokens = std::move(whole.tokens);
            result.chunks.push_back(std::move(encoded));
            return result;
        }
    }
    catch (const std::length_error&) {
        // Continue through the source-preserving waterfall splitter.
    }

    const auto coarse_units = split_source(result.normalized_text);
    std::vector<SourceUnit> units;
    for (const auto& unit : coarse_units) {
        auto fitted = split_unit_to_fit(unit, result.language);
        units.insert(
            units.end(), std::make_move_iterator(fitted.begin()),
            std::make_move_iterator(fitted.end()));
    }
    KokoroChunk current;
    current.source_begin = units.front().begin;
    for (const auto& unit : units) {
        Phonemization phonemization = phonemize(unit.text, result.language);
        phonemization.phonemes =
            preserve_misaki_trailing_space(std::move(phonemization.phonemes), unit.text);
        KokoroChunk encoded =
            tokenize_phonemes(phonemization.phonemes, result.language, unit.begin, unit.end);

        const auto needs_sentence_separator = [&]() {
            if (current.phonemes.empty() || encoded.phonemes.empty() ||
                current.phonemes.back() == ' ' || encoded.phonemes.front() == ' ') {
                return false;
            }
            const char last = current.phonemes.back();
            return last == '.' || last == '!' || last == '?';
        };
        bool insert_sentence_separator = needs_sentence_separator();

        // KPipeline prefers punctuation boundaries and keeps ordinary chunks
        // comfortably below the absolute 510-ID model limit.
        if (current.ids.size() + encoded.ids.size() + (insert_sentence_separator ? 1 : 0) > 400 &&
            !current.ids.empty()) {
            current.source_end = unit.begin;
            current.text = result.normalized_text.substr(
                current.source_begin, current.source_end - current.source_begin);
            result.chunks.push_back(std::move(current));
            current = KokoroChunk{};
            current.source_begin = unit.begin;
            insert_sentence_separator = false;
        }
        if (current.ids.size() + encoded.ids.size() + (insert_sentence_separator ? 1 : 0) > 510) {
            throw std::logic_error("fitted Kokoro chunks exceeded the hard model limit");
        }
        if (insert_sentence_separator) {
            const auto space = token_to_id_.find(" ");
            if (space == token_to_id_.end()) {
                throw std::logic_error("Kokoro vocabulary does not contain the space token");
            }
            current.phonemes += ' ';
            current.ids.push_back(space->second);
        }
        current.processed_text += phonemization.text;
        current.phonemes += encoded.phonemes;
        current.ids.insert(current.ids.end(), encoded.ids.begin(), encoded.ids.end());
        current.dropped.insert(
            current.dropped.end(), encoded.dropped.begin(), encoded.dropped.end());
        if (phonemization.tokens.empty()) {
            MToken token;
            token.text = unit.text;
            token.phonemes = encoded.phonemes;
            token.source_begin = unit.begin;
            token.source_end = unit.end;
            current.tokens.push_back(std::move(token));
        } else {
            for (MToken& token : phonemization.tokens) {
                token.source_begin += unit.begin;
                token.source_end += unit.begin;
                current.tokens.push_back(std::move(token));
            }
        }
        current.source_end = unit.end;
    }
    if (!current.ids.empty()) {
        current.text = result.normalized_text.substr(
            current.source_begin, current.source_end - current.source_begin);
        result.chunks.push_back(std::move(current));
    }
    if (result.chunks.empty()) {
        throw std::invalid_argument("Kokoro phonemization produced no chunks");
    }
    return result;
}

}  // namespace nemo_speech::tts::kokoro
