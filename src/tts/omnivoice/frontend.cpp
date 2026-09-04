// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "frontend.h"

#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "common/json.h"
#include "runtime/ggml/runtime.h"

namespace nemo_speech::tts::omnivoice {
namespace {

void
check_icu(UErrorCode status, const char* operation) {
    if (U_FAILURE(status)) {
        throw std::invalid_argument(
            std::string("OmniVoice frontend ") + operation + " failed: " + u_errorName(status));
    }
}

std::vector<UChar32>
decode(const std::string& text) {
    std::vector<UChar32> output;
    output.reserve(text.size());
    int32_t offset = 0;
    while (offset < static_cast<int32_t>(text.size())) {
        UChar32 codepoint = 0;
        U8_NEXT(text.data(), offset, static_cast<int32_t>(text.size()), codepoint);
        if (codepoint < 0)
            throw std::invalid_argument("OmniVoice text is not valid UTF-8");
        output.push_back(codepoint);
    }
    return output;
}

std::string
encode(const std::vector<UChar32>& codepoints) {
    icu::UnicodeString value;
    for (UChar32 codepoint : codepoints) value.append(codepoint);
    std::string output;
    value.toUTF8String(output);
    return output;
}

bool
python_space(UChar32 cp) {
    return u_isUWhiteSpace(cp);
}

bool
cjk(UChar32 cp) {
    return cp >= 0x4e00 && cp <= 0x9fff;
}

std::string
trim(const std::string& text) {
    auto value = decode(text);
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && python_space(value[begin])) ++begin;
    while (end > begin && python_space(value[end - 1])) --end;
    return encode(std::vector<UChar32>(value.begin() + begin, value.begin() + end));
}

std::string
lowercase(const std::string& text) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t utf16_length = 0;
    u_strFromUTF8(
        nullptr, 0, &utf16_length, text.data(), static_cast<int32_t>(text.size()), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status))
        check_icu(status, "lowercase input");
    status = U_ZERO_ERROR;
    std::vector<UChar> utf16(static_cast<size_t>(utf16_length) + 1);
    u_strFromUTF8(
        utf16.data(), static_cast<int32_t>(utf16.size()), &utf16_length, text.data(),
        static_cast<int32_t>(text.size()), &status);
    check_icu(status, "lowercase input");
    icu::UnicodeString value(utf16.data(), utf16_length);
    value.toLower();
    std::string output;
    value.toUTF8String(output);
    return output;
}

bool
contains(const std::vector<UChar32>& values, UChar32 value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

const std::set<std::string>&
abbreviations() {
    static const std::set<std::string> values = {
        "Mr.",   "Mrs.",  "Ms.",  "Dr.",   "Prof.", "Sr.",     "Jr.",  "Rev.", "Fr.",
        "Hon.",  "Pres.", "Gov.", "Capt.", "Gen.",  "Sen.",    "Rep.", "Col.", "Maj.",
        "Lt.",   "Cmdr.", "Sgt.", "Cpl.",  "Co.",   "Corp.",   "Inc.", "Ltd.", "Est.",
        "Dept.", "St.",   "Ave.", "Blvd.", "Rd.",   "Mt.",     "Ft.",  "No.",  "Jan.",
        "Feb.",  "Mar.",  "Apr.", "Aug.",  "Sep.",  "Sept.",   "Oct.", "Nov.", "Dec.",
        "i.e.",  "e.g.",  "vs.",  "Vs.",   "Etc.",  "approx.", "fig.", "def."};
    return values;
}

std::string
last_word(const std::vector<UChar32>& value) {
    size_t begin = value.size();
    while (begin > 0 && python_space(value[begin - 1])) --begin;
    size_t start = begin;
    while (start > 0 && !python_space(value[start - 1])) --start;
    return encode(std::vector<UChar32>(value.begin() + start, value.begin() + begin));
}

double
lookup(const std::vector<std::pair<std::string, double>>& values, const std::string& key) {
    for (const auto& value : values) {
        if (value.first == key)
            return value.second;
    }
    throw std::runtime_error("OmniVoice duration table is missing key: " + key);
}

std::pair<std::string, std::string>
split_assignment(const std::string& value, const char* table) {
    const size_t separator = value.find('=');
    if (separator == std::string::npos || separator == 0 || separator + 1 == value.size()) {
        throw std::runtime_error(std::string("malformed OmniVoice ") + table + " entry");
    }
    return {value.substr(0, separator), value.substr(separator + 1)};
}

bool
ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

std::string
combine_text(const std::string& target, const std::optional<std::string>& reference) {
    std::string full;
    if (reference && !reference->empty()) {
        full = trim(*reference) + " " + trim(target);
    } else {
        full = trim(target);
    }
    auto input = decode(full);
    std::vector<UChar32> collapsed;
    collapsed.reserve(input.size());
    for (UChar32 cp : input) {
        if (cp == '\r' || cp == '\n')
            continue;
        if (cp == 0xff08)
            cp = '(';
        if (cp == 0xff09)
            cp = ')';
        if ((cp == ' ' || cp == '\t') && !collapsed.empty() && collapsed.back() == ' ')
            continue;
        if (cp == '\t')
            cp = ' ';
        collapsed.push_back(cp);
    }
    std::vector<UChar32> output;
    output.reserve(collapsed.size());
    for (size_t index = 0; index < collapsed.size(); ++index) {
        const bool adjacent_cjk = (index > 0 && cjk(collapsed[index - 1])) ||
                                  (index + 1 < collapsed.size() && cjk(collapsed[index + 1]));
        if (python_space(collapsed[index]) && adjacent_cjk)
            continue;
        output.push_back(collapsed[index]);
    }
    return encode(output);
}

std::string
add_terminal_punctuation(const std::string& text) {
    std::string stripped = trim(text);
    auto value = decode(stripped);
    if (value.empty())
        return stripped;
    static const std::vector<UChar32> endings = {
        ';',    ':',    ',',    '.',    '!',    '?',    0x2026, ')',    ']',
        '}',    '"',    '\'',   0x201c, 0x201d, 0x2018, 0x2019, 0xff1b, 0xff1a,
        0xff0c, 0x3002, 0xff01, 0xff1f, 0x3001, 0xff09, 0x3011};
    if (!contains(endings, value.back())) {
        const bool chinese = std::any_of(value.begin(), value.end(), cjk);
        value.push_back(chinese ? 0x3002 : '.');
    }
    return encode(value);
}

std::vector<std::string>
chunk_text_punctuation(
    const std::string& text, int32_t chunk_length, std::optional<int32_t> minimum_length) {
    if (chunk_length <= 0 || (minimum_length && *minimum_length < 0)) {
        throw std::invalid_argument("OmniVoice chunk lengths must be positive");
    }
    const auto input = decode(text);
    static const std::vector<UChar32> punctuation = {
        '.', ',', ';', ':', '!', '?', 0x3002, 0xff0c, 0xff1b, 0xff1a, 0xff01, 0xff1f};
    static const std::vector<UChar32> closing = {'"',    '\'', 0x201c, 0x201d, 0x2018, 0x2019,
                                                 0xff09, ']',  0x300b, '>',    0x300d, 0x3011};
    std::vector<std::vector<UChar32>> sentences;
    std::vector<UChar32> current;
    for (UChar32 cp : input) {
        if (current.empty() && !sentences.empty() &&
            (contains(punctuation, cp) || contains(closing, cp))) {
            sentences.back().push_back(cp);
            continue;
        }
        current.push_back(cp);
        if (contains(punctuation, cp)) {
            const bool abbreviation = cp == '.' && abbreviations().count(last_word(current)) != 0;
            if (!abbreviation) {
                sentences.push_back(std::move(current));
                current.clear();
            }
        }
    }
    if (!current.empty())
        sentences.push_back(std::move(current));

    std::vector<std::vector<UChar32>> merged;
    current.clear();
    for (auto& sentence : sentences) {
        if (current.size() + sentence.size() <= static_cast<size_t>(chunk_length)) {
            current.insert(current.end(), sentence.begin(), sentence.end());
        } else {
            if (!current.empty())
                merged.push_back(std::move(current));
            current = std::move(sentence);
        }
    }
    if (!current.empty())
        merged.push_back(std::move(current));

    std::vector<std::vector<UChar32>> final_chunks;
    if (minimum_length) {
        const bool first_short = !merged.empty() && merged[0].size() < size_t(*minimum_length);
        for (size_t index = 0; index < merged.size(); ++index) {
            if (index == 1 && first_short) {
                final_chunks.back().insert(
                    final_chunks.back().end(), merged[index].begin(), merged[index].end());
            } else if (merged[index].size() >= size_t(*minimum_length)) {
                final_chunks.push_back(std::move(merged[index]));
            } else if (final_chunks.empty()) {
                final_chunks.push_back(std::move(merged[index]));
            } else {
                final_chunks.back().insert(
                    final_chunks.back().end(), merged[index].begin(), merged[index].end());
            }
        }
    } else {
        final_chunks = std::move(merged);
    }
    std::vector<std::string> output;
    for (const auto& chunk : final_chunks) {
        std::string value = trim(encode(chunk));
        if (!value.empty())
            output.push_back(std::move(value));
    }
    return output;
}

size_t
first_commit_boundary(const std::string& valid_utf8_text) {
    const auto input = decode(valid_utf8_text);
    static const std::vector<UChar32> terminal = {'.',    ';',    ':',    '!',    '?',
                                                  0x3002, 0xff1b, 0xff1a, 0xff01, 0xff1f};
    static const std::vector<UChar32> closing = {'"',    '\'', 0x201c, 0x201d, 0x2018, 0x2019,
                                                 0xff09, ']',  0x300b, '>',    0x300d, 0x3011};
    std::vector<UChar32> current;
    for (size_t i = 0; i < input.size(); ++i) {
        current.push_back(input[i]);
        if (!contains(terminal, input[i]))
            continue;
        if (input[i] == '.' && abbreviations().count(last_word(current)) != 0)
            continue;
        size_t end = i + 1;
        while (end < input.size() &&
               (contains(terminal, input[end]) || contains(closing, input[end])))
            ++end;
        return encode(std::vector<UChar32>(input.begin(), input.begin() + end)).size();
    }
    return 0;
}

FrontendTables::FrontendTables(const ggml_runtime::GGUFLoader& loader) {
    language_ids_ = loader.get_str_array("omnivoice.languages.ids");
    language_names_ = loader.get_str_array("omnivoice.languages.names");
    if (language_ids_.size() != 646 || language_names_.size() != language_ids_.size()) {
        throw std::runtime_error("malformed OmniVoice language table");
    }

    const auto ends = loader.get_i32_array("omnivoice.duration.range_ends");
    const auto types = loader.get_str_array("omnivoice.duration.range_types");
    if (ends.empty() || ends.size() != types.size()) {
        throw std::runtime_error("malformed OmniVoice duration range table");
    }
    for (size_t index = 0; index < ends.size(); ++index) {
        if (index > 0 && ends[index] <= ends[index - 1]) {
            throw std::runtime_error("unsorted OmniVoice duration range table");
        }
        duration_ranges_.emplace_back(ends[index], types[index]);
    }
    for (const auto& item : loader.get_str_array("omnivoice.duration.weights")) {
        auto assignment = split_assignment(item, "duration weight");
        size_t consumed = 0;
        const double value = std::stod(assignment.second, &consumed);
        if (consumed != assignment.second.size() || !std::isfinite(value)) {
            throw std::runtime_error("malformed OmniVoice duration weight");
        }
        duration_weights_.emplace_back(assignment.first, value);
    }
    for (const char* key : {"latin", "space", "mark", "punctuation", "digit", "default", "cjk"}) {
        (void)lookup(duration_weights_, key);
    }

    for (const auto& item : loader.get_str_array("omnivoice.instructions.en_to_zh")) {
        instruction_translations_.push_back(split_assignment(item, "instruction translation"));
    }
    for (const auto& encoded : loader.get_str_array("omnivoice.instructions.categories")) {
        const auto parsed = json::Value::parse(encoded);
        if (!parsed.is_array())
            throw std::runtime_error("malformed OmniVoice instruction category");
        std::vector<std::string> category;
        for (const auto& item : parsed.array()) {
            if (!item.is_string())
                throw std::runtime_error("malformed OmniVoice instruction value");
            category.push_back(item.string());
        }
        if (category.empty())
            throw std::runtime_error("empty OmniVoice instruction category");
        instruction_categories_.push_back(std::move(category));
    }
    if (instruction_categories_.size() != 6) {
        throw std::runtime_error("OmniVoice instruction table must have six categories");
    }
}

std::optional<std::string>
FrontendTables::resolve_language(const std::optional<std::string>& language) const {
    if (!language || lowercase(*language) == "none")
        return std::nullopt;
    auto id = std::find(language_ids_.begin(), language_ids_.end(), *language);
    if (id != language_ids_.end())
        return *id;
    const std::string key = lowercase(*language);
    for (size_t index = 0; index < language_names_.size(); ++index) {
        if (lowercase(language_names_[index]) == key)
            return language_ids_[index];
    }
    std::fprintf(
        stderr, "warning: unsupported OmniVoice language '%s'; using language-agnostic mode\n",
        language->c_str());
    return std::nullopt;
}

std::optional<std::string>
FrontendTables::resolve_instruction(
    const std::optional<std::string>& instruction, bool use_chinese) const {
    if (!instruction)
        return std::nullopt;
    const std::string stripped = trim(*instruction);
    if (stripped.empty())
        return std::nullopt;
    const auto cps = decode(stripped);
    std::vector<std::string> items;
    std::vector<UChar32> current;
    for (UChar32 cp : cps) {
        if (cp == ',' || cp == 0xff0c) {
            const std::string item = trim(encode(current));
            if (!item.empty())
                items.push_back(lowercase(item));
            current.clear();
        } else {
            current.push_back(cp);
        }
    }
    const std::string final_item = trim(encode(current));
    if (!final_item.empty())
        items.push_back(lowercase(final_item));

    std::set<std::string> valid;
    for (const auto& category : instruction_categories_) {
        valid.insert(category.begin(), category.end());
    }
    for (const auto& item : items) {
        if (valid.count(item) == 0) {
            throw std::invalid_argument(
                "Unsupported instruct items found in " + stripped + ":\n  '" + item + "' -> '" +
                item + "' (unsupported)");
        }
    }
    const bool dialect = std::any_of(
        items.begin(), items.end(), [](const std::string& item) { return ends_with(item, "话"); });
    const bool accent = std::any_of(items.begin(), items.end(), [](const std::string& item) {
        return item.find(" accent") != std::string::npos;
    });
    if (dialect && accent) {
        throw std::invalid_argument(
            "Cannot mix Chinese dialect and English accent in a single instruct. Dialects are "
            "for Chinese speech, accents for English speech.");
    }
    if (dialect)
        use_chinese = true;
    if (accent)
        use_chinese = false;

    for (auto& item : items) {
        for (const auto& translation : instruction_translations_) {
            if (use_chinese && item == translation.first)
                item = translation.second;
            else if (!use_chinese && item == translation.second)
                item = translation.first;
        }
    }
    for (const auto& category : instruction_categories_) {
        std::vector<std::string> hits;
        for (const auto& item : items) {
            if (std::find(category.begin(), category.end(), item) != category.end()) {
                hits.push_back(item);
            }
        }
        if (hits.size() > 1) {
            throw std::invalid_argument(
                "Conflicting instruct items within the same category: '" + hits[0] + "' vs '" +
                hits[1] +
                "'. Each category (gender, age, pitch, style, accent, dialect) allows at most "
                "one item.");
        }
    }
    const bool has_chinese = std::any_of(items.begin(), items.end(), [](const std::string& item) {
        const auto codepoints = decode(item);
        return std::any_of(codepoints.begin(), codepoints.end(), cjk);
    });
    std::string result;
    const char* separator = has_chinese ? "，" : ", ";
    for (const auto& item : items) {
        if (!result.empty())
            result += separator;
        result += item;
    }
    return result.empty() ? std::nullopt : std::optional<std::string>(result);
}

double
FrontendTables::character_weight(const std::string& text) const {
    double total = 0.0;
    for (UChar32 cp : decode(text)) {
        std::string type;
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z'))
            type = "latin";
        else if (cp == 32)
            type = "space";
        else if (cp == 0x0640)
            type = "mark";
        else {
            const int8_t category = u_charType(cp);
            if (category == U_NON_SPACING_MARK || category == U_ENCLOSING_MARK ||
                category == U_COMBINING_SPACING_MARK) {
                type = "mark";
            } else if (
                category == U_DASH_PUNCTUATION || category == U_START_PUNCTUATION ||
                category == U_END_PUNCTUATION || category == U_CONNECTOR_PUNCTUATION ||
                category == U_OTHER_PUNCTUATION || category == U_INITIAL_PUNCTUATION ||
                category == U_FINAL_PUNCTUATION ||
                (category >= U_MATH_SYMBOL && category <= U_OTHER_SYMBOL)) {
                type = "punctuation";
            } else if (
                category == U_SPACE_SEPARATOR || category == U_LINE_SEPARATOR ||
                category == U_PARAGRAPH_SEPARATOR) {
                type = "space";
            } else if (
                category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
                category == U_OTHER_NUMBER) {
                type = "digit";
            } else {
                auto range = std::lower_bound(
                    duration_ranges_.begin(), duration_ranges_.end(), cp,
                    [](const auto& item, UChar32 value) { return item.first < value; });
                if (range != duration_ranges_.end())
                    type = range->second;
                else
                    type = cp > 0x20000 ? "cjk" : "default";
            }
        }
        total += lookup(duration_weights_, type);
    }
    return total;
}

double
FrontendTables::estimate_frames(
    const std::string& target, const std::string& reference, double reference_frames,
    std::optional<double> low_threshold, double boost_strength) const {
    if (reference_frames <= 0.0 || reference.empty())
        return 0.0;
    const double ref_weight = character_weight(reference);
    if (ref_weight == 0.0)
        return 0.0;
    const double estimated = character_weight(target) / (ref_weight / reference_frames);
    if (low_threshold && estimated < *low_threshold) {
        return *low_threshold * std::pow(estimated / *low_threshold, 1.0 / boost_strength);
    }
    return estimated;
}

int32_t
FrontendTables::target_frames(
    const std::string& target, const std::optional<std::string>& reference,
    std::optional<int32_t> reference_frames, std::optional<double> speed,
    std::optional<double> fixed_duration_seconds, int32_t frame_rate) const {
    if (target.empty() || frame_rate <= 0)
        throw std::invalid_argument("invalid OmniVoice target text");
    if (fixed_duration_seconds) {
        if (!std::isfinite(*fixed_duration_seconds) || *fixed_duration_seconds <= 0.0)
            throw std::invalid_argument("OmniVoice fixed duration must be positive");
        return std::max(1, static_cast<int32_t>(*fixed_duration_seconds * frame_rate));
    }
    if (speed && (!std::isfinite(*speed) || *speed <= 0.0))
        throw std::invalid_argument("OmniVoice speed must be positive");
    const std::string ref =
        reference && !reference->empty() ? *reference : std::string("Nice to meet you.");
    const int32_t frames =
        reference_frames && reference && !reference->empty() ? *reference_frames : 25;
    double estimate = estimate_frames(target, ref, frames);
    if (speed && *speed != 1.0)
        estimate /= *speed;
    return std::max(1, static_cast<int32_t>(estimate));
}

}  // namespace nemo_speech::tts::omnivoice
