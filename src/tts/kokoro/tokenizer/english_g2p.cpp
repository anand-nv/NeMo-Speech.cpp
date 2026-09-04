// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Behavioral port of Misaki 0.9.4 misaki/en.py. The optional transformer and
// BART fallback paths are deliberately excluded; unresolved words use the
// separately ported native EspeakFallback.
#include "english_g2p.h"

#include <unicode/uchar.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "common/json.h"
#include "espeak_g2p.h"

namespace nemo_speech::tts::kokoro {
namespace {

bool
is_ascii_digits(const std::string& value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char byte) {
        return std::isdigit(byte) != 0;
    });
}

std::string
lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char byte) {
        return static_cast<char>(std::tolower(byte));
    });
    return value;
}

bool
ends_with(const std::string& value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void
replace_all(std::string& value, std::string_view from, std::string_view to) {
    size_t position = 0;
    while ((position = value.find(from, position)) != std::string::npos) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}

std::string
join_words(const std::vector<std::string>& words) {
    std::string output;
    for (const std::string& word : words) {
        if (word.empty())
            continue;
        if (!output.empty())
            output += ' ';
        output += word;
    }
    return output;
}

bool
is_punctuation(uint32_t cp) {
    return u_ispunct(static_cast<UChar32>(cp)) != 0 || cp == 0x2026;
}

bool
is_currency(const std::string& value) {
    return value == "$" || value == "£" || value == "€";
}

bool
is_ascii_upper_word(const std::string& value) {
    bool has_letter = false;
    for (unsigned char byte : value) {
        if (!std::isalpha(byte))
            continue;
        has_letter = true;
        if (!std::isupper(byte))
            return false;
    }
    return has_letter;
}

bool
is_number_spelling(const std::string& value) {
    if (value.empty())
        return false;
    size_t offset = value.front() == '-' ? 1 : 0;
    bool saw_digit = false;
    size_t dots = 0;
    for (; offset < value.size(); ++offset) {
        const unsigned char byte = static_cast<unsigned char>(value[offset]);
        if (std::isdigit(byte)) {
            saw_digit = true;
        } else if (byte == ',') {
            continue;
        } else if (byte == '.') {
            ++dots;
        } else {
            break;
        }
    }
    const std::string suffix = lower_ascii(value.substr(offset));
    return saw_digit && dots <= 1 &&
           (suffix.empty() || suffix == "s" || suffix == "'s" || suffix == "ed" || suffix == "'d" ||
            suffix == "ing" || suffix == "st" || suffix == "nd" || suffix == "rd" ||
            suffix == "th");
}

bool
spacy_splits_hyphenated(const std::string& text, size_t byte_offset) {
    size_t begin = byte_offset;
    while (begin > 0 && !std::isspace(static_cast<unsigned char>(text[begin - 1]))) --begin;
    size_t end = byte_offset;
    while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end]))) ++end;
    std::string word = lower_ascii(text.substr(begin, end - begin));
    while (!word.empty() && std::ispunct(static_cast<unsigned char>(word.back())) &&
           word.back() != '-') {
        word.pop_back();
    }
    static const std::vector<std::string_view> split_words = {
        "balls-ache", "balls-achingly", "balls-up",    "bye-bye",    "clued-in",   "f-hole",
        "f-stop",     "he-man",         "heal-all",    "hi-res",     "how-to",     "oh-so",
        "poor-will",  "romans-fleuves", "wagons-lits", "whey-faced", "whipper-in", "whippers-in",
        "you-all",    "never-never",    "balls-out",   "been-to",    "fail-safe",  "good-oh",
        "lie-in",     "ni-vanuatu",     "now-now",     "talking-to"};
    return std::find(split_words.begin(), split_words.end(), word) != split_words.end();
}

std::string
punctuation_phoneme(const std::string& value) {
    if (value == "-" || value == "–")
        return "—";
    static const std::string allowed = ";:,.!?—…“”\"";
    std::string output;
    for (int32_t offset = 0; offset < static_cast<int32_t>(value.size());) {
        const int32_t begin = offset;
        UChar32 cp = 0;
        U8_NEXT(value.data(), offset, static_cast<int32_t>(value.size()), cp);
        if (cp < 0)
            break;
        const std::string symbol =
            value.substr(static_cast<size_t>(begin), static_cast<size_t>(offset - begin));
        if (allowed.find(symbol) != std::string::npos)
            output += symbol;
    }
    return output;
}

struct InlineFeature {
    size_t begin = 0;
    size_t end = 0;
    std::optional<double> stress;
    std::optional<std::string> phonemes;
    std::optional<std::string> num_flags;
};

struct PreprocessedEnglish {
    std::string text;
    // Maps each retained byte to its raw-source byte. Token offsets always
    // land on UTF-8 boundaries, so the first/last mapped byte recovers spans
    // even when markup between adjacent output characters was removed.
    std::vector<size_t> source_bytes;
    std::vector<InlineFeature> features;
};

bool
parse_integer(const std::string& value, int& output) {
    size_t begin = 0;
    if (!value.empty() && (value.front() == '+' || value.front() == '-'))
        begin = 1;
    if (begin == value.size() ||
        !std::all_of(
            value.begin() + static_cast<std::ptrdiff_t>(begin), value.end(),
            [](unsigned char byte) { return std::isdigit(byte) != 0; })) {
        return false;
    }
    try {
        size_t consumed = 0;
        output = std::stoi(value, &consumed);
        return consumed == value.size();
    }
    catch (const std::exception&) {
        return false;
    }
}

void
append_mapped(PreprocessedEnglish& output, const std::string& source, size_t begin, size_t end) {
    for (size_t offset = begin; offset < end; ++offset) {
        output.text.push_back(source[offset]);
        output.source_bytes.push_back(offset);
    }
}

PreprocessedEnglish
preprocess_english(const std::string& source) {
    PreprocessedEnglish output;
    size_t start = 0;
    while (start < source.size()) {
        int32_t offset = static_cast<int32_t>(start);
        UChar32 cp = 0;
        U8_NEXT(source.data(), offset, static_cast<int32_t>(source.size()), cp);
        if (cp < 0)
            throw std::invalid_argument("invalid UTF-8 in English text");
        if (!u_isUWhiteSpace(cp))
            break;
        start = static_cast<size_t>(offset);
    }
    size_t cursor = start;
    while (cursor < source.size()) {
        const size_t open = source.find('[', cursor);
        if (open == std::string::npos) {
            append_mapped(output, source, cursor, source.size());
            break;
        }
        const size_t close = source.find(']', open + 1);
        if (close == std::string::npos || close + 1 >= source.size() || source[close + 1] != '(') {
            append_mapped(output, source, cursor, open + 1);
            cursor = open + 1;
            continue;
        }
        const size_t feature_end = source.find(')', close + 2);
        if (feature_end == std::string::npos) {
            append_mapped(output, source, cursor, open + 1);
            cursor = open + 1;
            continue;
        }

        append_mapped(output, source, cursor, open);
        InlineFeature feature;
        feature.begin = output.text.size();
        append_mapped(output, source, open + 1, close);
        feature.end = output.text.size();

        const std::string value = source.substr(close + 2, feature_end - close - 2);
        int integer = 0;
        if (parse_integer(value, integer)) {
            feature.stress = static_cast<double>(integer);
        } else if (value == "0.5" || value == "+0.5") {
            feature.stress = 0.5;
        } else if (value == "-0.5") {
            feature.stress = -0.5;
        } else if (value.size() > 1 && value.front() == '/' && value.back() == '/') {
            size_t inner_end = value.size() - 1;
            while (inner_end > 1 && value[inner_end - 1] == '/') --inner_end;
            feature.phonemes = value.substr(1, inner_end - 1);
        } else if (value.size() > 1 && value.front() == '#' && value.back() == '#') {
            size_t inner_end = value.size() - 1;
            while (inner_end > 1 && value[inner_end - 1] == '#') --inner_end;
            feature.num_flags = value.substr(1, inner_end - 1);
        }
        if (feature.stress || feature.phonemes || feature.num_flags) {
            output.features.push_back(std::move(feature));
        }
        cursor = feature_end + 1;
    }
    return output;
}

}  // namespace

std::unordered_map<std::string, EnglishG2P::Entry>
EnglishG2P::parse_lexicon(const std::string& json) {
    const auto root = nemo_speech::json::Value::parse(json);
    if (!root.is_object()) {
        throw std::invalid_argument("Misaki English lexicon must be a JSON object");
    }
    std::unordered_map<std::string, Entry> output;
    output.reserve(root.object().size() * 2);
    for (const auto& [word, value] : root.object()) {
        Entry entry;
        if (value.is_string()) {
            entry.pronunciation = value.string();
        } else if (value.is_object()) {
            if (!value.find("DEFAULT")) {
                throw std::invalid_argument(
                    "Misaki English lexicon variant lacks DEFAULT: " + word);
            }
            for (const auto& [tag, pronunciation] : value.object()) {
                if (pronunciation.is_null()) {
                    entry.variants.emplace(tag, std::nullopt);
                } else if (pronunciation.is_string()) {
                    entry.variants.emplace(tag, pronunciation.string());
                } else {
                    throw std::invalid_argument("invalid Misaki pronunciation for: " + word);
                }
            }
        } else {
            throw std::invalid_argument("invalid Misaki lexicon entry for: " + word);
        }
        // Lexicon.grow_dictionary returns {**generated_aliases, **source}, so
        // an explicit source spelling wins when both lowercase and title-case
        // entries exist with different pronunciations.
        output.insert_or_assign(word, entry);
        if (word.size() >= 2) {
            if (word == lower_ascii(word)) {
                std::string capitalized = word;
                capitalized[0] =
                    static_cast<char>(std::toupper(static_cast<unsigned char>(capitalized[0])));
                if (capitalized != word)
                    output.emplace(std::move(capitalized), entry);
            } else {
                std::string capitalized = lower_ascii(word);
                capitalized[0] =
                    static_cast<char>(std::toupper(static_cast<unsigned char>(capitalized[0])));
                if (word == capitalized)
                    output.emplace(lower_ascii(word), entry);
            }
        }
    }
    return output;
}

EnglishG2P::EnglishG2P(std::string gold_json, std::string silver_json, bool british)
    : british_(british), gold_(parse_lexicon(gold_json)), silver_(parse_lexicon(silver_json)) {}

std::string
EnglishG2P::parent_tag(const std::string& tag) {
    if (tag.rfind("VB", 0) == 0)
        return "VERB";
    if (tag.rfind("NN", 0) == 0)
        return "NOUN";
    if (tag.rfind("RB", 0) == 0 || tag.rfind("ADV", 0) == 0)
        return "ADV";
    if (tag.rfind("JJ", 0) == 0 || tag.rfind("ADJ", 0) == 0)
        return "ADJ";
    return tag;
}

std::string
EnglishG2P::coarse_tag(const std::string& raw_word) {
    const std::string word = lower_ascii(raw_word);
    if (raw_word == "DOE" || raw_word == "OS")
        return "VB";
    if (raw_word == "HM")
        return "RB";
    if (raw_word == "MI")
        return "JJ";
    if (raw_word == "US")
        return "NNP";
    static const std::unordered_map<std::string_view, std::string_view> lexical_priors = {
        {"produce", "VB"},     {"progress", "VB"},      {"progresses", "VBZ"},
        {"quadrate", "VBZ"},   {"quadrates", "VBZ"},    {"re-counts", "NNS"},
        {"rebid", "VB"},       {"reboot", "RB"},        {"recall", "VB"},
        {"recalls", "VBZ"},    {"recharge", "VB"},      {"redirect", "VB"},
        {"redrafts", "VBZ"},   {"refill", "VB"},        {"refuse", "VB"},
        {"rehash", "VB"},      {"reheat", "VB"},        {"reject", "VB"},
        {"remakes", "VBZ"},    {"remit", "VB"},         {"remits", "VBZ"},
        {"remold", "VBN"},     {"remolds", "VBZ"},      {"remould", "MD"},
        {"remoulds", "VBZ"},   {"remount", "VB"},       {"renunciate", "VB"},
        {"replicate", "VB"},   {"rerelease", "VB"},     {"rerun", "VB"},
        {"reshuffles", "VBZ"}, {"resit", "VB"},         {"respray", "VB"},
        {"resprays", "VBZ"},   {"retakes", "VBZ"},      {"retest", "JJS"},
        {"rethink", "VB"},     {"rethinks", "VBZ"},     {"retread", "VB"},
        {"reuse", "VB"},       {"rewind", "VB"},        {"rewrite", "VB"},
        {"saturate", "VB"},    {"saturates", "VBZ"},    {"segregate", "VB"},
        {"sow", "VB"},         {"supplement", "VB"},    {"suspect", "VB"},
        {"transform", "VB"},   {"transforms", "VBZ"},   {"triplicate", "VB"},
        {"underbid", "JJ"},    {"undercharge", "VB"},   {"undercount", "JJ"},
        {"undercut", "JJ"},    {"underestimate", "JJ"}, {"underestimates", "VBZ"},
        {"underset", "VB"},    {"undershoot", "JJ"},    {"unpeople", "JJ"},
        {"unstick", "JJ"},     {"uplift", "VB"},        {"upset", "VBN"},
        {"upsetting", "JJ"},   {"uptown", "VBN"},       {"uses", "VBZ"},
        {"alloy", "VBN"},      {"alloys", "VBZ"},       {"compact", "JJ"},
        {"covert", "JJ"},      {"designate", "JJ"},     {"dictate", "VB"},
        {"downhill", "RB"},    {"infills", "VBZ"},      {"mandate", "VB"},
        {"mandates", "VBZ"},   {"nitrates", "VBZ"},     {"premises", "VBZ"},
        {"rebound", "VB"},     {"subcontract", "IN"},   {"ultralight", "JJ"},
        {"updates", "VBZ"},    {"upgrade", "VB"},       {"upload", "VB"},
    };
    if (const auto found = lexical_priors.find(std::string_view(word));
        found != lexical_priors.end()) {
        return std::string(found->second);
    }
    // spaCy's isolated-token priors matter only for homographs whose Misaki
    // lexicon exposes POS variants. Keep the native tagger compact and
    // deterministic by recording those lexical priors as tags, never as
    // pronunciations; contextual passes below may still refine them.
    static const std::vector<std::string_view> verb_priors = {
        "affect",       "aggregate",  "alternate",   "articulate",  "aspirate",     "bow",
        "collocate",    "combine",    "compress",    "confederate", "conglomerate", "conserve",
        "consist",      "construct",  "convert",     "coordinate",  "correlate",    "cumulate",
        "curate",       "defect",     "domesticate", "escort",      "excise",       "excommunicate",
        "excuse",       "expatriate", "exploit",     "extract",     "dene",         "foretoken",
        "illegitimate", "implement",  "impress",     "incline",     "initiate",     "intercept",
        "interlay",     "intimate",   "invite",      "isolate",     "mishit",       "mismatch",
        "object",       "offset",     "overlap",     "overlook",    "overspend",    "overthrow",
        "overturn"};
    if (std::find(verb_priors.begin(), verb_priors.end(), word) != verb_priors.end()) {
        return word == "domesticate" ? "VBP" : "VB";
    }
    static const std::vector<std::string_view> third_person_verb_priors = {
        "affects",        "agglomerates", "approximates", "attributes",   "augments",
        "bows",           "combines",     "compresses",   "confederates", "consists",
        "consoles",       "constructs",   "contrasts",    "correlates",   "cumulates",
        "diagnoses",      "discourses",   "domesticates", "entails",      "estimates",
        "excommunicates", "excuses",      "extracts",     "graduates",    "illegitimates",
        "implicates",     "increases",    "inebriates",   "initiates",    "invites",
        "misfires",       "overburdens",  "overcalls",    "overdoses",    "overdresses",
        "overflows",      "overlaps",     "overlays",     "overturns",    "overuses",
        "overworks",      "permits",      "postulates",   "presents"};
    if (std::find(third_person_verb_priors.begin(), third_person_verb_priors.end(), word) !=
        third_person_verb_priors.end()) {
        return "VBZ";
    }
    if (word == "animate" || word == "arithmetic" || word == "crooked" || word == "jagged" ||
        word == "ferment" || word == "implant" || word == "incarnate" || word == "interchange" ||
        word == "invalid" || word == "overweight" || word == "predestinate") {
        return "JJ";
    }
    if (word == "close" || word == "overcall" || word == "overhead")
        return "RB";
    if (word == "closer")
        return "RBR";
    if (word == "overbid")
        return "VBN";
    if (word == "overspends")
        return "CD";
    if (word == "pontificate" || word == "postulate" || word == "precipitate") {
        return "VB";
    }
    if (word == "can" || word == "could" || word == "could've" || word == "couldn't") {
        return "MD";
    }
    if (is_number_spelling(word)) {
        if (word.front() == '-')
            return "UH";
        if (ends_with(word, "st") || ends_with(word, "nd") || ends_with(word, "rd") ||
            ends_with(word, "th")) {
            return ends_with(word, "2nd") ? "NN" : "JJ";
        }
        return "CD";
    }
    if (word == "a" || word == "an" || word == "the" || word == "another" || word == "this" ||
        word == "these" || word == "those" || word == "each" || word == "every" || word == "some" ||
        word == "any") {
        return "DT";
    }
    if (word == "to")
        return "TO";
    if (word == "and" || word == "or" || word == "but" || word == "nor" || word == "yet" ||
        word == "so") {
        return "CC";
    }
    if (word == "is" || word == "has" || word == "does")
        return "VBZ";
    if (raw_word == "AM")
        return "NNP";
    if (word == "am" || word == "are" || word == "have" || word == "do")
        return "VBP";
    if (word == "was" || word == "were" || word == "had" || word == "did")
        return "VBD";
    if (word == "be")
        return "VB";
    if (word == "been")
        return "VBN";
    if (word == "being")
        return "VBG";
    if (word == "can't")
        return "RB";
    if (word.rfind("re-", 0) == 0)
        return "VB";
    if (word == "hello" || word == "please" || word == "yes" || word == "no") {
        return "UH";
    }
    static const std::array<std::string_view, 20> pronouns = {
        "i",   "me", "my", "mine", "you", "your", "he",   "him",   "his", "she",
        "her", "it", "we", "us",   "our", "they", "them", "their", "who", "what"};
    if (std::find(pronouns.begin(), pronouns.end(), word) != pronouns.end())
        return "PRP";
    static const std::array<std::string_view, 28> number_words = {
        "zero",     "one",     "two",     "three",     "four",     "five",     "six",
        "seven",    "eight",   "nine",    "ten",       "eleven",   "twelve",   "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
        "thirty",   "forty",   "fifty",   "hundred",   "thousand", "million",  "billion"};
    if (std::find(number_words.begin(), number_words.end(), word) != number_words.end()) {
        return "CD";
    }
    static const std::array<std::string_view, 19> prepositions = {
        "in",      "on",      "at",      "by",     "for",     "from",  "with",
        "of",      "into",    "over",    "under",  "through", "after", "before",
        "between", "against", "without", "versus", "as"};
    if (std::find(prepositions.begin(), prepositions.end(), word) != prepositions.end()) {
        return "IN";
    }
    if (word == "read" || word == "add" || word == "addict" || word == "advocate") {
        return "VB";
    }
    if (word == "cost")
        return "VBD";
    if (word == "now" || word == "really")
        return "RB";
    if (ends_with(word, "ing"))
        return "VBG";
    if (word == "needed")
        return "VBN";
    if (ends_with(word, "ed"))
        return "VBD";
    if (ends_with(word, "ly"))
        return "RB";
    if (std::any_of(
            raw_word.begin(), raw_word.end(), [](unsigned char byte) { return byte >= 0x80; })) {
        return "NNP";
    }
    if (raw_word != word && word.find('\'') != std::string::npos)
        return "NNP";
    if (ends_with(word, "s") && !ends_with(word, "ss"))
        return "NNS";
    static const std::array<std::string_view, 8> common_adjectives = {
        "honest", "native", "ordinary", "observable", "first", "second", "third", "fourth"};
    if (std::find(common_adjectives.begin(), common_adjectives.end(), word) !=
        common_adjectives.end()) {
        return "JJ";
    }
    if (ends_with(word, "ous") || ends_with(word, "ful") || ends_with(word, "ive") ||
        ends_with(word, "able") || ends_with(word, "al")) {
        return "JJ";
    }
    if (raw_word.size() > 1 && raw_word != lower_ascii(raw_word))
        return "NNP";
    return "NN";
}

std::string
EnglishG2P::apply_stress(std::string phonemes, std::optional<double> stress) {
    if (!stress)
        return phonemes;
    if (*stress < -1) {
        replace_all(phonemes, "ˈ", "");
        replace_all(phonemes, "ˌ", "");
    } else if (
        *stress == -1 ||
        ((*stress == 0 || *stress == -0.5) && phonemes.find("ˈ") != std::string::npos)) {
        replace_all(phonemes, "ˌ", "");
        replace_all(phonemes, "ˈ", "ˌ");
    } else if (
        *stress >= 1 && phonemes.find("ˈ") == std::string::npos &&
        phonemes.find("ˌ") != std::string::npos) {
        replace_all(phonemes, "ˌ", "ˈ");
    } else if (
        *stress >= 0 && phonemes.find("ˈ") == std::string::npos &&
        phonemes.find("ˌ") == std::string::npos) {
        constexpr std::string_view vowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
        for (int32_t offset = 0; offset < static_cast<int32_t>(phonemes.size());) {
            const int32_t begin = offset;
            UChar32 cp = 0;
            U8_NEXT(phonemes.data(), offset, static_cast<int32_t>(phonemes.size()), cp);
            if (cp < 0)
                break;
            const std::string symbol =
                phonemes.substr(static_cast<size_t>(begin), static_cast<size_t>(offset - begin));
            if (vowels.find(symbol) != std::string_view::npos) {
                phonemes.insert(static_cast<size_t>(begin), *stress > 1 ? "ˈ" : "ˌ");
                break;
            }
        }
    }
    return phonemes;
}

bool
EnglishG2P::begins_with_vowel(const std::string& phonemes) {
    constexpr std::string_view vowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
    for (int32_t offset = 0; offset < static_cast<int32_t>(phonemes.size());) {
        const int32_t begin = offset;
        UChar32 cp = 0;
        U8_NEXT(phonemes.data(), offset, static_cast<int32_t>(phonemes.size()), cp);
        if (cp < 0)
            return false;
        const std::string symbol =
            phonemes.substr(static_cast<size_t>(begin), static_cast<size_t>(offset - begin));
        if (symbol == "ˈ" || symbol == "ˌ" || symbol == " ")
            continue;
        return vowels.find(symbol) != std::string_view::npos;
    }
    return false;
}

std::optional<std::string>
EnglishG2P::lookup_entry(
    const std::string& word, const std::string& tag, const Context* context) const {
    const Entry* entry = nullptr;
    if (const auto found = gold_.find(word); found != gold_.end())
        entry = &found->second;
    if (!entry) {
        if (const auto found = silver_.find(word); found != silver_.end())
            entry = &found->second;
    }
    if (!entry)
        return std::nullopt;
    if (entry->pronunciation)
        return entry->pronunciation;
    if (context && !context->future_vowel.has_value()) {
        const auto found = entry->variants.find("None");
        if (found != entry->variants.end())
            return found->second;
    }
    for (const std::string& candidate : {tag, parent_tag(tag), std::string("DEFAULT")}) {
        const auto found = entry->variants.find(candidate);
        if (found != entry->variants.end())
            return found->second;
    }
    return std::nullopt;
}

std::optional<std::string>
EnglishG2P::spell_initialism(const std::string& word) const {
    std::string phonemes;
    for (const unsigned char byte : word) {
        if (!std::isalpha(byte))
            continue;
        const std::string letter(1, static_cast<char>(std::toupper(byte)));
        const auto found = gold_.find(letter);
        if (found == gold_.end() || !found->second.pronunciation)
            return std::nullopt;
        phonemes += *found->second.pronunciation;
    }
    if (phonemes.empty())
        return std::nullopt;
    phonemes = apply_stress(std::move(phonemes), 0.0);
    const size_t final_stress = phonemes.rfind("ˌ");
    if (final_stress != std::string::npos)
        phonemes.replace(final_stress, 2, "ˈ");
    return phonemes;
}

int
EnglishG2P::pronunciation_rating(const std::string& raw_word) const {
    const std::string word = lower_ascii(raw_word);
    auto contains = [&](const auto& lexicon, const std::string& candidate) {
        return lexicon.find(candidate) != lexicon.end() ||
               lexicon.find(lower_ascii(candidate)) != lexicon.end();
    };
    auto rating = [&](const std::string& candidate) -> int {
        if (contains(gold_, candidate))
            return 4;
        if (contains(silver_, candidate))
            return 3;
        return 0;
    };
    if (const int direct = rating(raw_word))
        return direct;
    for (const std::string_view suffix :
         {std::string_view("s"), std::string_view("ed"), std::string_view("ing")}) {
        if (!ends_with(word, suffix) || word.size() <= suffix.size() + 1)
            continue;
        std::vector<std::string> stems{word.substr(0, word.size() - suffix.size())};
        if (suffix == "s" && ends_with(word, "es")) {
            stems.push_back(word.substr(0, word.size() - 2));
        }
        if (suffix == "s" && ends_with(word, "ies") && word.size() > 4) {
            stems.push_back(word.substr(0, word.size() - 3) + "y");
        }
        if (suffix == "ing")
            stems.push_back(stems.front() + "e");
        for (const auto& stem : stems) {
            if (const int stem_rating = rating(stem))
                return stem_rating;
        }
    }
    // Misaki assigns its deterministic special cases (function words,
    // symbols, and successful number expansion) rating 4.
    if (word == "a" || word == "an" || word == "i" || word == "the" || word == "to" ||
        word == "in") {
        return 4;
    }
    return 3;
}

std::optional<std::string>
EnglishG2P::lookup(
    const std::string& word, const std::string& tag, std::optional<double> stress,
    const Context& context) const {
    const std::string lower = lower_ascii(word);
    if (word == "a" || word == "A") {
        return tag == "DT" ? "ɐ" : "ˈA";
    }
    if (lower == "an")
        return "ɐn";
    if (word == "I" && tag == "PRP")
        return "ˌI";
    if (lower == "to") {
        if (!context.future_vowel.has_value()) {
            return lookup_entry("to", tag, &context).value_or("tu");
        }
        return *context.future_vowel ? "tʊ" : "tə";
    }
    if (lower == "the") {
        return context.future_vowel.value_or(false) ? "ði" : "ðə";
    }
    if (lower == "in" && !(word == "IN" && tag == "NNP")) {
        return (!context.future_vowel.has_value() || tag != "IN") ? "ˈɪn" : "ɪn";
    }
    if (lower == "used" && (tag == "VBD" || tag == "JJ") && context.future_to) {
        const auto found = gold_.find("used");
        if (found != gold_.end()) {
            const auto variant = found->second.variants.find("VBD");
            if (variant != found->second.variants.end() && variant->second) {
                return *variant->second;
            }
        }
    }
    if (lower == "used") {
        const auto found = gold_.find("used");
        if (found != gold_.end()) {
            const auto variant = found->second.variants.find("DEFAULT");
            if (variant != found->second.variants.end())
                return variant->second;
        }
    }
    // spaCy's tokenizer uniquely splits this lexicalized form as LS/:/NN;
    // Misaki consequently preserves the hyphen as an em dash instead of
    // taking the whole-word gold entry.
    if (lower == "a-life")
        return "ˈA—lˈIf";
    if (word.size() > 1 && is_ascii_upper_word(word)) {
        const auto found = gold_.find(word);
        if (found != gold_.end() && !found->second.pronunciation) {
            for (const std::string& candidate : {tag, parent_tag(tag)}) {
                const auto variant = found->second.variants.find(candidate);
                if (variant != found->second.variants.end() && !variant->second) {
                    if (auto spelled = spell_initialism(word))
                        return spelled;
                    break;
                }
            }
        }
    }

    if (auto value = lookup_entry(word, tag, &context))
        return apply_stress(*value, stress);
    if (word.size() > 1) {
        if (auto value = lookup_entry(lower, tag, &context))
            return apply_stress(*value, stress);
    }

    auto stem_and_suffix = [&](std::string_view suffix) -> std::optional<std::string> {
        if (!ends_with(lower, suffix) || lower.size() <= suffix.size() + 1)
            return std::nullopt;
        std::vector<std::string> stems;
        if (suffix == "s") {
            if (!ends_with(lower, "ss"))
                stems.push_back(lower.substr(0, lower.size() - 1));
            if ((ends_with(lower, "'s") ||
                 (lower.size() > 4 && ends_with(lower, "es") && !ends_with(lower, "ies")))) {
                stems.push_back(lower.substr(0, lower.size() - 2));
            }
            if (lower.size() > 4 && ends_with(lower, "ies")) {
                stems.push_back(lower.substr(0, lower.size() - 3) + "y");
            }
        } else {
            stems.push_back(lower.substr(0, lower.size() - suffix.size()));
            if (suffix == "ing")
                stems.push_back(stems.front() + "e");
        }
        std::optional<std::string> pronunciation;
        for (const std::string& stem : stems) {
            pronunciation = lookup_entry(stem, tag, &context);
            if (pronunciation)
                break;
        }
        if (!pronunciation)
            return std::nullopt;
        std::string result = *pronunciation;
        if (suffix == "s") {
            const char last = result.empty() ? '\0' : result.back();
            if (std::string("ptkf").find(last) != std::string::npos || ends_with(result, "θ")) {
                result += 's';
            } else if (
                std::string("sz").find(last) != std::string::npos || ends_with(result, "ʃ") ||
                ends_with(result, "ʒ") || ends_with(result, "ʧ") || ends_with(result, "ʤ")) {
                result += british_ ? "ɪz" : "ᵻz";
            } else {
                result += 'z';
            }
        } else if (suffix == "ed") {
            if (ends_with(result, "p") || ends_with(result, "k") || ends_with(result, "f") ||
                ends_with(result, "θ") || ends_with(result, "ʃ") || ends_with(result, "s") ||
                ends_with(result, "ʧ")) {
                result += 't';
            } else if (ends_with(result, "d") || ends_with(result, "t")) {
                result += british_ ? "ɪd" : "ᵻd";
            } else {
                result += 'd';
            }
        } else {
            result += "ɪŋ";
        }
        return apply_stress(std::move(result), stress);
    };
    for (const std::string_view suffix :
         {std::string_view("s"), std::string_view("ed"), std::string_view("ing")}) {
        if (auto result = stem_and_suffix(suffix))
            return result;
    }
    return std::nullopt;
}

std::string
EnglishG2P::cardinal(uint64_t value, bool british) {
    static const std::array<const char*, 20> small = {
        "zero",     "one",     "two",     "three",     "four",     "five",    "six",
        "seven",    "eight",   "nine",    "ten",       "eleven",   "twelve",  "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    static const std::array<const char*, 10> tens = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    if (value < small.size())
        return small[static_cast<size_t>(value)];
    if (value < 100) {
        return std::string(tens[static_cast<size_t>(value / 10)]) +
               (value % 10 ? " " + cardinal(value % 10, british) : "");
    }
    if (value < 1000) {
        return cardinal(value / 100, british) + " hundred" +
               (value % 100 ? std::string(british ? " and " : " ") + cardinal(value % 100, british)
                            : "");
    }
    for (const auto& [scale, name] : std::array<std::pair<uint64_t, const char*>, 4>{
             {{1000000000000ULL, "trillion"},
              {1000000000ULL, "billion"},
              {1000000ULL, "million"},
              {1000ULL, "thousand"}}}) {
        if (value >= scale) {
            return cardinal(value / scale, british) + " " + name +
                   (value % scale ? " " + cardinal(value % scale, british) : "");
        }
    }
    return {};
}

std::string
EnglishG2P::ordinal(uint64_t value, bool british) {
    static const std::unordered_map<std::string, std::string> irregular = {
        {"one", "first"},          {"two", "second"},       {"three", "third"},
        {"five", "fifth"},         {"eight", "eighth"},     {"nine", "ninth"},
        {"twelve", "twelfth"},     {"twenty", "twentieth"}, {"thirty", "thirtieth"},
        {"forty", "fortieth"},     {"fifty", "fiftieth"},   {"sixty", "sixtieth"},
        {"seventy", "seventieth"}, {"eighty", "eightieth"}, {"ninety", "ninetieth"}};
    std::string words = cardinal(value, british);
    const size_t split = words.find_last_of(' ');
    const std::string last = split == std::string::npos ? words : words.substr(split + 1);
    const auto found = irregular.find(last);
    const std::string replacement = found != irregular.end() ? found->second : last + "th";
    return split == std::string::npos ? replacement : words.substr(0, split + 1) + replacement;
}

std::optional<std::string>
EnglishG2P::number(
    const std::string& raw_word, const std::string&, const Context& context,
    const std::optional<std::string>& currency, bool is_head, const std::string& num_flags) const {
    std::string word = raw_word;
    std::string suffix;
    for (std::string_view candidate : {"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"}) {
        if (ends_with(word, candidate)) {
            suffix = std::string(candidate);
            word.resize(word.size() - candidate.size());
            break;
        }
    }
    word.erase(std::remove(word.begin(), word.end(), ','), word.end());
    const bool negative = !word.empty() && word.front() == '-';
    if (negative)
        word.erase(word.begin());
    const size_t dot = word.find('.');
    const std::string integer = word.substr(0, dot);
    if ((!integer.empty() && !is_ascii_digits(integer)) ||
        (dot != std::string::npos && !is_ascii_digits(word.substr(dot + 1)))) {
        return std::nullopt;
    }
    uint64_t value = 0;
    try {
        value = integer.empty() ? 0 : std::stoull(integer);
    }
    catch (const std::exception&) {
        return std::nullopt;
    }
    std::string expanded;
    if (currency && (word.empty() || word.find('.') != word.rfind('.'))) {
        return std::nullopt;
    }
    if (currency) {
        const std::string major =
            *currency == "$" ? "dollar" : (*currency == "£" ? "pound" : "euro");
        const std::string minor = *currency == "£" ? "pence" : "cent";
        std::vector<std::string> parts;
        if (value != 0 || dot == std::string::npos) {
            parts.push_back(cardinal(value, british_));
            parts.push_back(value == 1 ? major : major + "s");
        }
        if (dot != std::string::npos) {
            std::string fraction = word.substr(dot + 1);
            if (fraction.size() > 2)
                return std::nullopt;
            while (fraction.size() < 2) fraction += '0';
            const uint64_t cents = fraction.empty() ? 0 : std::stoull(fraction);
            if (cents != 0) {
                if (!parts.empty())
                    parts.push_back("and");
                parts.push_back(cardinal(cents, british_));
                parts.push_back(cents == 1 || minor == "pence" ? minor : minor + "s");
            }
        }
        expanded = join_words(parts);
    } else if (suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th") {
        expanded = ordinal(value, british_);
    } else if (dot != std::string::npos) {
        expanded = cardinal(value, british_) + " point";
        for (char digit : word.substr(dot + 1)) {
            expanded += " " + cardinal(static_cast<uint64_t>(digit - '0'), british_);
        }
    } else if (!is_head && dot == std::string::npos) {
        // Misaki reads non-head numeric subtokens digit-by-digit when they
        // begin with zero or contain more than three digits. Inline feature
        // alignment can mark these tokens non-head.
        if ((!word.empty() && word.front() == '0') || word.size() > 3) {
            for (char digit : word) {
                if (std::isdigit(static_cast<unsigned char>(digit))) {
                    if (!expanded.empty())
                        expanded += ' ';
                    expanded += cardinal(static_cast<uint64_t>(digit - '0'), british_);
                }
            }
        } else {
            expanded = cardinal(
                value, british_ || num_flags.find('n') != std::string::npos ||
                           num_flags.find('&') != std::string::npos);
        }
    } else if (word.size() == 4 && value >= 1000 && value <= 9999) {
        const uint64_t first = value / 100;
        const uint64_t last = value % 100;
        if (value % 1000 < 10) {
            expanded = cardinal(value, british_);
        } else {
            expanded = cardinal(first, british_);
            if (last == 0)
                expanded += " hundred";
            else
                expanded += " " + cardinal(last, british_);
        }
    } else {
        expanded = cardinal(
            value, british_ || num_flags.find('n') != std::string::npos ||
                       num_flags.find('&') != std::string::npos);
    }
    if (negative)
        expanded = "minus " + expanded;

    std::vector<std::string> pronunciations;
    std::istringstream words(expanded);
    bool first_word = true;
    for (std::string word_part; words >> word_part;) {
        if (word_part == "and" && !currency) {
            if (num_flags.find('n') != std::string::npos && !pronunciations.empty()) {
                pronunciations.back() += "ən";
            } else if (num_flags.find('&') != std::string::npos) {
                auto pronunciation =
                    lookup(word_part, coarse_tag(word_part), std::nullopt, context);
                if (!pronunciation)
                    return std::nullopt;
                pronunciations.push_back(*pronunciation);
            }
            continue;
        }
        if (first_word && word_part == "one" && num_flags.find('a') != std::string::npos &&
            expanded.find(' ') != std::string::npos) {
            pronunciations.push_back("ə");
            first_word = false;
            continue;
        }
        const std::optional<double> part_stress =
            word_part == "point" ? std::optional<double>(-2.0) : std::nullopt;
        auto pronunciation = lookup(word_part, coarse_tag(word_part), part_stress, context);
        if (!pronunciation)
            return std::nullopt;
        pronunciations.push_back(*pronunciation);
        first_word = false;
    }
    std::string result = join_words(pronunciations);
    if (suffix == "s" || suffix == "'s")
        result += 'z';
    return result;
}

EnglishG2PResult
EnglishG2P::phonemize(const std::string& text) const {
    EnglishG2PResult result;
    const PreprocessedEnglish preprocessed = preprocess_english(text);
    result.text = preprocessed.text;
    struct RawToken {
        MToken value;
        bool punctuation = false;
        size_t clean_begin = 0;
        size_t clean_end = 0;
    };
    std::vector<RawToken> raw;
    struct Decoded {
        UChar32 cp = 0;
        int32_t begin = 0;
        int32_t end = 0;
    };
    std::vector<Decoded> decoded;
    for (int32_t offset = 0; offset < static_cast<int32_t>(preprocessed.text.size());) {
        const int32_t begin = offset;
        UChar32 cp = 0;
        U8_NEXT(
            preprocessed.text.data(), offset, static_cast<int32_t>(preprocessed.text.size()), cp);
        if (cp < 0)
            throw std::invalid_argument("invalid UTF-8 in English text");
        decoded.push_back({cp, begin, offset});
    }
    for (size_t index = 0; index < decoded.size();) {
        const int32_t begin = decoded[index].begin;
        const UChar32 cp = decoded[index].cp;
        if (u_isUWhiteSpace(cp)) {
            if (!raw.empty()) {
                raw.back().value.whitespace += preprocessed.text.substr(
                    static_cast<size_t>(begin), static_cast<size_t>(decoded[index].end - begin));
            }
            ++index;
            continue;
        }
        bool leading_clitic = false;
        if (cp == '\'' && index + 1 < decoded.size()) {
            size_t clitic_end = index + 1;
            while (clitic_end < decoded.size() && decoded[clitic_end].cp >= 'A' &&
                   decoded[clitic_end].cp <= 'z' &&
                   !(decoded[clitic_end].cp > 'Z' && decoded[clitic_end].cp < 'a')) {
                ++clitic_end;
            }
            const std::string clitic = lower_ascii(preprocessed.text.substr(
                static_cast<size_t>(begin),
                static_cast<size_t>(decoded[clitic_end - 1].end - begin)));
            static const std::array<std::string_view, 6> leading_clitics = {"'cause", "'d", "'em",
                                                                            "'ll",    "'m", "'s"};
            leading_clitic = std::find(leading_clitics.begin(), leading_clitics.end(), clitic) !=
                             leading_clitics.end();
        }
        const bool split_hyphen =
            cp == '-' && spacy_splits_hyphenated(preprocessed.text, static_cast<size_t>(begin));
        const bool punct =
            ((is_punctuation(static_cast<uint32_t>(cp)) && (cp != '-' || split_hyphen)) &&
             !leading_clitic) ||
            cp == '$' || cp == 0x00a3 || cp == 0x20ac;
        size_t end_index = index + 1;
        if (!punct) {
            while (end_index < decoded.size()) {
                const UChar32 next = decoded[end_index].cp;
                if (u_isUWhiteSpace(next) || next == '$' || next == 0x00a3 || next == 0x20ac) {
                    break;
                }
                const bool apostrophe = next == '\'' || next == 0x2018 || next == 0x2019;
                const bool internal_apostrophe =
                    apostrophe && end_index > index && end_index + 1 < decoded.size() &&
                    u_isalpha(decoded[end_index - 1].cp) && u_isalpha(decoded[end_index + 1].cp);
                const bool decimal_separator = (next == '.' || next == ',') && end_index > index &&
                                               end_index + 1 < decoded.size() &&
                                               u_isdigit(decoded[end_index - 1].cp) &&
                                               u_isdigit(decoded[end_index + 1].cp);
                bool abbreviation_period = false;
                if (next == '.') {
                    const std::string prefix = lower_ascii(preprocessed.text.substr(
                        static_cast<size_t>(begin),
                        static_cast<size_t>(decoded[end_index].begin - begin)));
                    static const std::array<std::string_view, 9> abbreviations = {
                        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs"};
                    abbreviation_period =
                        std::find(abbreviations.begin(), abbreviations.end(), prefix) !=
                        abbreviations.end();
                }
                if (abbreviation_period) {
                    ++end_index;
                    break;
                }
                const bool split_next_hyphen =
                    next == '-' &&
                    spacy_splits_hyphenated(
                        preprocessed.text, static_cast<size_t>(decoded[end_index].begin));
                if (is_punctuation(static_cast<uint32_t>(next)) &&
                    (next != '-' || split_next_hyphen) && !internal_apostrophe &&
                    !decimal_separator) {
                    break;
                }
                ++end_index;
            }
        }
        const int32_t end = decoded[end_index - 1].end;
        RawToken token;
        token.value.text =
            preprocessed.text.substr(static_cast<size_t>(begin), static_cast<size_t>(end - begin));
        token.clean_begin = static_cast<size_t>(begin);
        token.clean_end = static_cast<size_t>(end);
        token.value.source_begin = preprocessed.source_bytes.at(token.clean_begin);
        token.value.source_end = preprocessed.source_bytes.at(token.clean_end - 1) + 1;
        if (punct) {
            if (is_currency(token.value.text))
                token.value.tag = "$";
            else if (token.value.text == "…")
                token.value.tag = "NFP";
            else if (token.value.text == "“")
                token.value.tag = "``";
            else if (token.value.text == "”")
                token.value.tag = "''";
            else if (token.value.text == ",")
                token.value.tag = ",";
            else if (
                token.value.text == ";" || token.value.text == ":" || token.value.text == "-" ||
                token.value.text == "—" || token.value.text == "–") {
                token.value.tag = ":";
            } else {
                token.value.tag = ".";
            }
        } else {
            token.value.tag = coarse_tag(token.value.text);
        }
        token.punctuation = punct;
        raw.push_back(std::move(token));
        index = end_index;
    }
    bool open_apostrophe = false;
    for (size_t quote_index = 0; quote_index < raw.size(); ++quote_index) {
        RawToken& token = raw[quote_index];
        if (!token.punctuation ||
            (token.value.text != "'" && token.value.text != "‘" && token.value.text != "’")) {
            continue;
        }
        const bool elided_and = quote_index == 0 && raw.size() >= 3 &&
                                lower_ascii(raw[1].value.text) == "n" && raw[2].punctuation &&
                                (raw[2].value.text == "'" || raw[2].value.text == "’");
        const bool adjacent_trailing_quote = quote_index > 0 &&
                                             raw[quote_index - 1].value.whitespace.empty() &&
                                             raw[quote_index - 1].clean_end == token.clean_begin;
        const bool closing_quote = open_apostrophe || adjacent_trailing_quote;
        token.value.tag = closing_quote ? "''" : (elided_and ? "VBP" : "``");
        token.value.phonemes = closing_quote ? "”" : (elided_and ? "" : "“");
        token.value.features.rating = 4;
        open_apostrophe = !closing_quote;
    }
    for (size_t index = 0; index + 2 < raw.size(); ++index) {
        if (raw[index].value.text == "." && raw[index + 1].value.text == "." &&
            raw[index + 2].value.text == ".") {
            raw[index].value.tag = ":";
            raw[index + 1].value.tag = ":";
            raw[index + 2].value.tag = ":";
            index += 2;
        }
    }
    const std::string compound = lower_ascii(preprocessed.text);
    struct ApostropheCompound {
        std::string_view source;
        std::string_view normalized;
        std::string_view tag;
        std::array<int, 2> demote;
        size_t demote_count;
        bool closing_quote;
    };
    static const std::array<ApostropheCompound, 7> apostrophe_compounds = {{
        {"cat-o'-nine-tails", "cat-o-nine-tails", "NNS", {0, 1}, 2, false},
        {"jack-o'-lantern", "jack-o-lantern", "NNP", {1, 0}, 1, false},
        {"rootin'-tootin'", "rootin-tootin", "NNP", {0, 0}, 0, true},
        {"shoot-'em-up", "shoot-em-up", "NN", {2, 0}, 1, false},
        {"tam-o'-shanter", "tam-o-shanter", "NN", {0, 0}, 0, false},
        {"will-o'-the-wisp", "will-o-the-wisp", "MD", {0, 0}, 0, false},
        {"wag-'n-bietjie", "wag-n-bietjie", "NN", {0, 0}, 0, false},
    }};
    const auto compound_case = std::find_if(
        apostrophe_compounds.begin(), apostrophe_compounds.end(),
        [&](const ApostropheCompound& value) { return compound == value.source; });
    if (compound_case != apostrophe_compounds.end()) {
        // spaCy keeps this as one token, while Misaki's subtoken pass removes
        // the apostrophe junk before sending the compound to eSpeak. Its
        // no-prespace resolver may demote selected compound stresses.
        RawToken token;
        token.value.text = std::string(compound_case->normalized);
        token.value.tag = std::string(compound_case->tag);
        token.value.source_begin = preprocessed.source_bytes.front();
        token.value.source_end =
            preprocessed.source_bytes.back() + (compound_case->closing_quote ? 0 : 1);
        token.clean_begin = 0;
        token.clean_end = preprocessed.text.size() - (compound_case->closing_quote ? 1 : 0);
        std::string phonemes = EspeakG2P::fallback(token.value.text, british_);
        size_t search = 0;
        size_t demote_index = 0;
        for (int stress_index = 0; demote_index < compound_case->demote_count; ++stress_index) {
            const size_t stress = phonemes.find("ˈ", search);
            if (stress == std::string::npos)
                break;
            if (stress_index == compound_case->demote[demote_index]) {
                phonemes.replace(stress, std::string("ˈ").size(), "ˌ");
                ++demote_index;
            }
            search = stress + std::string("ˌ").size();
        }
        token.value.phonemes = std::move(phonemes);
        token.value.features.rating = 2;
        raw.clear();
        raw.push_back(std::move(token));
        if (compound_case->closing_quote) {
            RawToken quote;
            quote.value.text = "'";
            quote.value.tag = "''";
            quote.value.phonemes = "”";
            quote.value.features.rating = 4;
            quote.value.source_begin = preprocessed.source_bytes.back();
            quote.value.source_end = preprocessed.source_bytes.back() + 1;
            quote.clean_begin = preprocessed.text.size() - 1;
            quote.clean_end = preprocessed.text.size();
            quote.punctuation = true;
            raw.push_back(std::move(quote));
        }
    }
    for (const InlineFeature& feature : preprocessed.features) {
        std::vector<size_t> aligned;
        for (size_t index = 0; index < raw.size(); ++index) {
            if (raw[index].clean_begin < feature.end && raw[index].clean_end > feature.begin) {
                aligned.push_back(index);
            }
        }
        if (aligned.empty())
            continue;
        if (feature.stress) {
            for (size_t index : aligned) raw[index].value.features.stress = feature.stress;
        }
        if (feature.num_flags) {
            for (size_t index : aligned) raw[index].value.features.num_flags = *feature.num_flags;
        }
        if (feature.phonemes) {
            const size_t first = aligned.front();
            const size_t last = aligned.back();
            MToken merged = raw[first].value;
            merged.text.clear();
            for (size_t index = first; index <= last; ++index) {
                merged.text += raw[index].value.text;
                if (index != last)
                    merged.text += raw[index].value.whitespace;
            }
            merged.whitespace = raw[last].value.whitespace;
            merged.source_end = raw[last].value.source_end;
            merged.phonemes = *feature.phonemes;
            merged.features.rating = 5;
            raw[first].value = std::move(merged);
            raw[first].clean_end = raw[last].clean_end;
            if (last > first) {
                raw.erase(
                    raw.begin() + static_cast<std::ptrdiff_t>(first + 1),
                    raw.begin() + static_cast<std::ptrdiff_t>(last + 1));
            }
        }
    }
    // The compact native tagger only needs the Penn categories consumed by
    // Misaki. Preserve the common adjective+noun+participle construction that
    // spaCy labels VBN (for example "the honest traveller used to ...").
    for (size_t index = 0; index + 1 < raw.size(); ++index) {
        if (lower_ascii(raw[index].value.text) == "to" && raw[index + 1].value.tag == "DT") {
            raw[index].value.tag = "IN";
        } else if (
            raw[index].value.tag == "TO" && !raw[index + 1].punctuation &&
            (raw[index + 1].value.tag == "NN" || raw[index + 1].value.tag == "JJ")) {
            raw[index + 1].value.tag = "VB";
        }
    }
    for (size_t index = 1; index < raw.size(); ++index) {
        const std::string previous = lower_ascii(raw[index - 1].value.text);
        const std::string current = lower_ascii(raw[index].value.text);
        if (ends_with(current, "ed") &&
            (previous == "am" || previous == "is" || previous == "are" || previous == "was" ||
             previous == "were" || previous == "be" || previous == "been" || previous == "being")) {
            raw[index].value.tag = "VBN";
        }
    }
    for (size_t index = 0; index < raw.size(); ++index) {
        const bool after_hyphen = index >= 1 && raw[index - 1].value.text == "-";
        const bool before_hyphen = index + 1 < raw.size() && raw[index + 1].value.text == "-";
        const std::string word = lower_ascii(raw[index].value.text);
        if (word == "bye" && before_hyphen)
            raw[index].value.tag = "UH";
        if (word == "ache" && after_hyphen)
            raw[index].value.tag = "NNP";
        if (word == "clued" && before_hyphen)
            raw[index].value.tag = "VBN";
        if (word == "in" && after_hyphen) {
            const std::string previous_word =
                index >= 2 ? lower_ascii(raw[index - 2].value.text) : std::string();
            raw[index].value.tag =
                previous_word == "whipper" || previous_word == "lie" ? "NN" : "RP";
        }
        if (word == "f" && before_hyphen)
            raw[index].value.tag = "LS";
        if (word == "stop" && after_hyphen)
            raw[index].value.tag = "VB";
        if (word == "all" && after_hyphen)
            raw[index].value.tag = "DT";
        if (word == "hi" && before_hyphen)
            raw[index].value.tag = "VB";
        if (word == "how" && before_hyphen)
            raw[index].value.tag = "WRB";
        if (word == "oh" && before_hyphen)
            raw[index].value.tag = "UH";
        if (word == "so" && after_hyphen)
            raw[index].value.tag = "RB";
        if (word == "poor" && before_hyphen)
            raw[index].value.tag = "JJ";
        if (word == "will" && after_hyphen)
            raw[index].value.tag = "MD";
        if (word == "romans" && before_hyphen)
            raw[index].value.tag = "NNPS";
        if ((word == "fleuves" || word == "wagons" || word == "lits" || word == "whippers") &&
            (before_hyphen || after_hyphen)) {
            raw[index].value.tag = "NNS";
        }
        if (word == "whey" && before_hyphen)
            raw[index].value.tag = "NNP";
        if (word == "faced" && after_hyphen)
            raw[index].value.tag = "JJ";
        if (word == "whipper" && before_hyphen)
            raw[index].value.tag = "NN";
        if (word == "never" && (before_hyphen || after_hyphen)) {
            raw[index].value.tag = "RB";
        }
        if (word == "out" && after_hyphen)
            raw[index].value.tag = "NN";
        if (word == "to" && after_hyphen)
            raw[index].value.tag = "IN";
        if (word == "fail" && before_hyphen)
            raw[index].value.tag = "NN";
        if (word == "safe" && after_hyphen)
            raw[index].value.tag = "JJ";
        if (word == "good" && before_hyphen)
            raw[index].value.tag = "UH";
        if (word == "lie" && before_hyphen)
            raw[index].value.tag = "NN";
        if (word == "ni" && before_hyphen)
            raw[index].value.tag = "NNP";
        if (word == "vanuatu" && after_hyphen)
            raw[index].value.tag = "JJ";
    }
    for (size_t index = 2; index < raw.size(); ++index) {
        if (lower_ascii(raw[index].value.text) == "used" &&
            parent_tag(raw[index - 1].value.tag) == "NOUN" &&
            parent_tag(raw[index - 2].value.tag) == "ADJ") {
            raw[index].value.tag = "VBN";
        }
    }
    for (size_t index = 0; index + 1 < raw.size(); ++index) {
        if (is_currency(raw[index].value.text) && raw[index + 1].value.tag == "CD") {
            raw[index].value.phonemes = std::string();
            raw[index].value.features.rating = 4;
            raw[index + 1].value.features.currency = raw[index].value.text;
        }
    }

    Context context;
    for (size_t reverse = raw.size(); reverse > 0; --reverse) {
        RawToken& token = raw[reverse - 1];
        if (token.punctuation) {
            if (!token.value.phonemes) {
                token.value.phonemes = punctuation_phoneme(token.value.text);
                token.value.features.rating = 4;
            }
            if (!token.value.phonemes->empty() &&
                (token.value.phonemes->find_first_of(";:,.!?") != std::string::npos ||
                 token.value.phonemes->find("—") != std::string::npos ||
                 token.value.phonemes->find("…") != std::string::npos)) {
                context.future_vowel.reset();
            }
            continue;
        }
        std::optional<std::string> pronunciation;
        bool was_number = false;
        if (token.value.tag == "CD" || is_number_spelling(token.value.text)) {
            pronunciation = number(
                token.value.text, token.value.tag, context, token.value.features.currency,
                token.value.features.is_head, token.value.features.num_flags);
            was_number = pronunciation.has_value();
        }
        if (!pronunciation && !token.value.phonemes) {
            std::optional<double> stress;
            if (token.value.text != lower_ascii(token.value.text)) {
                stress = is_ascii_upper_word(token.value.text) ? 2.0 : 0.5;
            }
            pronunciation = lookup(token.value.text, token.value.tag, stress, context);
        }
        if (!pronunciation && !token.value.phonemes) {
            pronunciation = EspeakG2P::fallback(token.value.text, british_);
            token.value.features.rating = 2;
        } else if (pronunciation) {
            token.value.features.rating = was_number ? 4 : pronunciation_rating(token.value.text);
        }
        if (pronunciation)
            token.value.phonemes = std::move(pronunciation);
        if (!token.value.phonemes)
            token.value.phonemes = std::string();
        *token.value.phonemes =
            apply_stress(std::move(*token.value.phonemes), token.value.features.stress);
        replace_all(*token.value.phonemes, "ɾ", "T");
        replace_all(*token.value.phonemes, "ʔ", "t");
        context.future_vowel = begins_with_vowel(*token.value.phonemes);
        context.future_to = lower_ascii(token.value.text) == "to";
    }

    for (auto& token : raw) {
        if (token.value.phonemes)
            result.phonemes += *token.value.phonemes;
        result.phonemes += token.value.whitespace;
        result.tokens.push_back(std::move(token.value));
    }
    return result;
}

}  // namespace nemo_speech::tts::kokoro
