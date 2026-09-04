// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Port of Misaki 0.9.4 zh.py and transcription.py at commit
// fba1236595f2d2bf21d414ba6e57d25256afada3. transcription.py was adapted
// from pinyin-to-ipa by Stefan Taubert (MIT).
#include "mandarin_g2p.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cppjieba/MixSegment.hpp"
#include "resource_path.h"
#include "tts/kokoro/sha256.h"

#ifndef NEMO_SPEECH_KOKORO_MANDARIN_DATA_DIR
#define NEMO_SPEECH_KOKORO_MANDARIN_DATA_DIR ""
#endif
#ifndef NEMO_SPEECH_KOKORO_MANDARIN_INSTALLED_DATA_DIR
#define NEMO_SPEECH_KOKORO_MANDARIN_INSTALLED_DATA_DIR ""
#endif

namespace nemo_speech::tts::kokoro {
namespace {

namespace fs = std::filesystem;

struct Utf8Char {
    uint32_t codepoint;
    std::string text;
};

std::vector<Utf8Char>
decode_utf8(const std::string& text) {
    std::vector<Utf8Char> output;
    for (size_t offset = 0; offset < text.size();) {
        const size_t begin = offset;
        const unsigned char first = static_cast<unsigned char>(text[offset++]);
        uint32_t cp = 0;
        size_t extra = 0;
        uint32_t minimum = 0;
        if (first < 0x80) {
            cp = first;
        } else if ((first & 0xe0) == 0xc0) {
            cp = first & 0x1f;
            extra = 1;
            minimum = 0x80;
        } else if ((first & 0xf0) == 0xe0) {
            cp = first & 0x0f;
            extra = 2;
            minimum = 0x800;
        } else if ((first & 0xf8) == 0xf0) {
            cp = first & 0x07;
            extra = 3;
            minimum = 0x10000;
        } else {
            throw std::invalid_argument("invalid UTF-8 passed to Mandarin G2P");
        }
        if (extra > text.size() - offset) {
            throw std::invalid_argument("truncated UTF-8 passed to Mandarin G2P");
        }
        for (size_t index = 0; index < extra; ++index) {
            const unsigned char byte = static_cast<unsigned char>(text[offset++]);
            if ((byte & 0xc0) != 0x80) {
                throw std::invalid_argument("invalid UTF-8 passed to Mandarin G2P");
            }
            cp = (cp << 6) | (byte & 0x3f);
        }
        if ((minimum != 0 && cp < minimum) || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
            throw std::invalid_argument("invalid UTF-8 passed to Mandarin G2P");
        }
        output.push_back({cp, text.substr(begin, offset - begin)});
    }
    return output;
}

bool
is_legacy_han(uint32_t cp) {
    return cp >= 0x4e00 && cp <= 0x9fff;
}

void
replace_all(std::string& value, std::string_view from, std::string_view to) {
    for (size_t offset = 0;
         !from.empty() && (offset = value.find(from, offset)) != std::string::npos;) {
        value.replace(offset, from.size(), to);
        offset += to.size();
    }
}

std::vector<std::string>
split_spaces(const std::string& value) {
    std::istringstream input(value);
    std::vector<std::string> output;
    std::string part;
    while (input >> part) output.push_back(part);
    return output;
}

bool
valid_data_dir(const fs::path& path) {
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 5> files = {{
        {"jieba.dict.utf8", "7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8"},
        {"hmm_model.utf8", "6be21b9135e8b63b1a3a306492ad0bd1cc2fe5c97d250e17b9f2c9bf52d1b49e"},
        {"misaki_pinyin_chars.tsv",
         "db60cce26d1d601e7267781da11f55958e2cc71da43f7c00d0fb6baf1df9e3d5"},
        {"misaki_pinyin_phrases.tsv",
         "b3d1ee5d19f2e32d85bd52f5a0480a2a6727072ace7a76cfde48d16b498084d3"},
        {"manifest.json", "21480231a8558824b24a4e45316278164e035ae8b82bf9a0b837c285bca02779"},
    }};
    for (const auto& [name, expected] : files) {
        const fs::path file = path / name;
        if (!fs::is_regular_file(file))
            return false;
        std::ifstream input(file, std::ios::binary);
        if (!input.is_open())
            return false;
        const std::string payload{
            std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        // istreambuf_iterator reaches EOF through the stream buffer and is not
        // required to set std::ios::eofbit on the owning stream.
        if (input.bad() || sha256_hex(payload) != expected)
            return false;
    }
    return true;
}

fs::path
find_data_dir() {
    std::vector<fs::path> candidates;
    if (const char* configured = std::getenv("NEMO_SPEECH_KOKORO_MISAKI_DATA_DIR")) {
        candidates.emplace_back(configured);
    }
    candidates.emplace_back(installed_misaki_data_root() / "mandarin");
    candidates.emplace_back(NEMO_SPEECH_KOKORO_MANDARIN_DATA_DIR);
    candidates.emplace_back(NEMO_SPEECH_KOKORO_MANDARIN_INSTALLED_DATA_DIR);
    for (const auto& path : candidates) {
        if (!path.empty() && valid_data_dir(path))
            return path;
    }
    throw std::runtime_error(
        "Kokoro Mandarin data not found; set NEMO_SPEECH_KOKORO_MISAKI_DATA_DIR");
}

const std::unordered_map<std::string, std::string>&
initials() {
    static const std::unordered_map<std::string, std::string> values = {
        {"b", "p"}, {"c", "ʦʰ"}, {"ch", "ꭧʰ"}, {"d", "t"},  {"f", "f"}, {"g", "k"},  {"h", "x"},
        {"j", "ʨ"}, {"k", "kʰ"}, {"l", "l"},   {"m", "m"},  {"n", "n"}, {"p", "pʰ"}, {"q", "ʨʰ"},
        {"r", "ɻ"}, {"s", "s"},  {"sh", "ʂ"},  {"t", "tʰ"}, {"x", "ɕ"}, {"z", "ʦ"},  {"zh", "ꭧ"},
    };
    return values;
}

const std::unordered_map<std::string, std::string>&
finals() {
    static const std::unordered_map<std::string, std::string> values = {
        {"a", "a0"},     {"ai", "ai̯0"},   {"an", "a0n"},    {"ang", "a0ŋ"},   {"ao", "au̯0"},
        {"e", "ɤ0"},     {"ei", "ei̯0"},   {"en", "ə0n"},    {"eng", "ə0ŋ"},   {"i", "i0"},
        {"ia", "ja0"},   {"ian", "jɛ0n"}, {"iang", "ja0ŋ"}, {"iao", "jau̯0"},  {"ie", "je0"},
        {"in", "i0n"},   {"iou", "jou̯0"}, {"ing", "i0ŋ"},   {"iong", "jʊ0ŋ"}, {"ong", "ʊ0ŋ"},
        {"ou", "ou̯0"},   {"u", "u0"},     {"uei", "wei̯0"},  {"ua", "wa0"},    {"uai", "wai̯0"},
        {"uan", "wa0n"}, {"uen", "wə0n"}, {"uang", "wa0ŋ"}, {"ueng", "wə0ŋ"}, {"uo", "wo0"},
        {"o", "wo0"},    {"ü", "y0"},     {"üe", "ɥe0"},    {"üan", "ɥɛ0n"},  {"ün", "y0n"},
    };
    return values;
}

std::pair<std::string, std::string>
strict_initial_final(std::string syllable) {
    // pypinyin.contrib.tone_convert strict spelling expansions.
    static const std::vector<std::pair<std::string_view, std::string_view>> y_finals = {
        {"yong", "iong"}, {"ying", "ing"}, {"yang", "iang"}, {"yuan", "üan"}, {"you", "iou"},
        {"yan", "ian"},   {"yao", "iao"},  {"yin", "in"},    {"yun", "ün"},   {"yue", "üe"},
        {"ye", "ie"},     {"yi", "i"},     {"ya", "ia"},     {"yu", "ü"},     {"yo", "o"},
    };
    static const std::vector<std::pair<std::string_view, std::string_view>> w_finals = {
        {"wang", "uang"}, {"weng", "ueng"}, {"wai", "uai"}, {"wei", "uei"}, {"wan", "uan"},
        {"wen", "uen"},   {"wong", "ong"},  {"wu", "u"},    {"wa", "ua"},   {"wo", "uo"},
    };
    for (const auto& [prefix, final] : y_finals) {
        if (syllable == prefix)
            return {"", std::string(final)};
    }
    for (const auto& [prefix, final] : w_finals) {
        if (syllable == prefix)
            return {"", std::string(final)};
    }

    std::string initial;
    for (std::string_view candidate : {"zh", "ch", "sh"}) {
        if (syllable.rfind(candidate, 0) == 0) {
            initial = candidate;
            break;
        }
    }
    if (initial.empty() && !syllable.empty() && initials().count(syllable.substr(0, 1))) {
        initial = syllable.substr(0, 1);
    }
    std::string final = syllable.substr(initial.size());
    if (final == "iu")
        final = "iou";
    if (final == "ui")
        final = "uei";
    if (initial == "j" || initial == "q" || initial == "x") {
        if (final == "u")
            final = "ü";
        else if (final == "ue")
            final = "üe";
        else if (final == "uan")
            final = "üan";
        else if (final == "un")
            final = "ün";
    } else {
        if (final == "un")
            final = "uen";
    }
    return {initial, final};
}

std::string
tone_mark(char tone) {
    switch (tone) {
        case '1':
            return "→";
        case '2':
            return "↗";
        case '3':
            return "↓";
        case '4':
            return "↘";
        case '5':
            return "";
        default:
            throw std::invalid_argument("Mandarin pinyin has no tone 1-5");
    }
}

std::string
integer_to_chinese(std::string digits) {
    const std::array<std::string, 10> numerals = {"零", "一", "二", "三", "四",
                                                  "五", "六", "七", "八", "九"};
    while (digits.size() > 1 && digits.front() == '0') digits.erase(digits.begin());
    if (digits.size() > 12) {
        std::string output;
        for (char digit : digits) output += numerals[static_cast<size_t>(digit - '0')];
        return output;
    }
    static const std::array<std::string, 4> small_units = {"", "十", "百", "千"};
    static const std::array<std::string, 4> group_units = {"", "万", "亿", "万亿"};
    std::string output;
    bool skipped_group = false;
    const size_t groups = (digits.size() + 3) / 4;
    for (size_t group_index = 0; group_index < groups; ++group_index) {
        const size_t reverse_group = groups - group_index - 1;
        const size_t group_end = digits.size() - reverse_group * 4;
        const size_t group_begin = group_end >= 4 ? group_end - 4 : 0;
        const unsigned group_value =
            static_cast<unsigned>(std::stoul(digits.substr(group_begin, group_end - group_begin)));
        if (group_value == 0) {
            if (!output.empty())
                skipped_group = true;
            continue;
        }
        if (!output.empty() && (skipped_group || group_value < 1000))
            output += "零";
        skipped_group = false;
        bool group_nonzero = false;
        bool pending_zero = false;
        for (size_t index = group_begin; index < group_end; ++index) {
            const int digit = digits[index] - '0';
            const size_t position = group_end - index - 1;
            if (digit == 0) {
                if (group_nonzero)
                    pending_zero = true;
                continue;
            }
            if (pending_zero && !output.empty())
                output += "零";
            pending_zero = false;
            if (!(digit == 1 && position == 1 && output.empty()))
                output += numerals[digit];
            output += small_units[position];
            group_nonzero = true;
        }
        output += group_units[reverse_group];
    }
    return output.empty() ? "零" : output;
}

std::string
normalize_arabic_numbers(const std::string& text) {
    std::string output;
    for (size_t offset = 0; offset < text.size();) {
        if (!std::isdigit(static_cast<unsigned char>(text[offset]))) {
            output.push_back(text[offset++]);
            continue;
        }
        const size_t begin = offset;
        while (offset < text.size() && std::isdigit(static_cast<unsigned char>(text[offset]))) {
            ++offset;
        }
        const std::string integer = text.substr(begin, offset - begin);
        const bool year = integer.size() == 4 && text.compare(offset, std::strlen("年"), "年") == 0;
        if (year) {
            for (char digit : integer) output += integer_to_chinese(std::string(1, digit));
        } else {
            output += integer_to_chinese(integer);
        }
        if (offset + 1 < text.size() && text[offset] == '.' &&
            std::isdigit(static_cast<unsigned char>(text[offset + 1]))) {
            output += "点";
            ++offset;
            while (offset < text.size() && std::isdigit(static_cast<unsigned char>(text[offset]))) {
                output += integer_to_chinese(text.substr(offset, 1));
                ++offset;
            }
        }
    }
    return output;
}

std::string
map_punctuation(std::string text) {
    for (const auto& [from, to] : std::vector<std::pair<std::string_view, std::string_view>>{
             {"、", ", "},
             {"，", ", "},
             {"。", ". "},
             {"．", ". "},
             {"！", "! "},
             {"：", ": "},
             {"；", "; "},
             {"？", "? "},
             {"«", " “"},
             {"»", "” "},
             {"《", " “"},
             {"》", "” "},
             {"「", " “"},
             {"」", "” "},
             {"【", " “"},
             {"】", "” "},
             {"（", " ("},
             {"）", ") "},
         }) {
        replace_all(text, from, to);
    }
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

}  // namespace

class MandarinG2P::Impl {
   public:
    explicit Impl(fs::path data_dir) : data_dir_(data_dir.empty() ? find_data_dir() : data_dir) {
        if (!valid_data_dir(data_dir_)) {
            throw std::runtime_error(
                "invalid Kokoro Mandarin data directory: " + data_dir_.string());
        }
        segmenter_ = std::make_unique<cppjieba::MixSegment>(
            (data_dir_ / "jieba.dict.utf8").string(), (data_dir_ / "hmm_model.utf8").string());
        load_chars(data_dir_ / "misaki_pinyin_chars.tsv");
        load_phrases(data_dir_ / "misaki_pinyin_phrases.tsv");
    }

    std::string phonemize(const std::string& raw_text) const {
        std::lock_guard<std::mutex> lock(segmenter_mutex_);
        const std::string text = map_punctuation(normalize_arabic_numbers(raw_text));
        const auto chars = decode_utf8(text);
        std::string output;
        for (size_t begin = 0; begin < chars.size();) {
            const bool han = is_legacy_han(chars[begin].codepoint);
            size_t end = begin + 1;
            while (end < chars.size() && is_legacy_han(chars[end].codepoint) == han) ++end;
            if (!han) {
                for (size_t index = begin; index < end; ++index) output += chars[index].text;
            } else {
                std::string segment;
                for (size_t index = begin; index < end; ++index) segment += chars[index].text;
                std::vector<std::string> words;
                // jieba.lcut(..., cut_all=False) keeps HMM discovery enabled
                // by default; disabling it changes numeral and unknown-word
                // boundaries (for example 零一 and 二零二).
                segmenter_->Cut(segment, words, true);
                for (size_t index = 0; index < words.size(); ++index) {
                    if (index != 0)
                        output += ' ';
                    output += word_to_ipa(words[index]);
                }
            }
            begin = end;
        }
        replace_all(output, "\xcc\xaf", "");  // U+032F, matching zh.legacy_call.
        return output;
    }

   private:
    fs::path data_dir_;
    std::unique_ptr<cppjieba::MixSegment> segmenter_;
    std::unordered_map<uint32_t, std::string> characters_;
    std::unordered_map<std::string, std::vector<std::string>> phrases_;
    size_t max_phrase_chars_ = 0;
    mutable std::mutex segmenter_mutex_;

    void load_chars(const fs::path& path) {
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("failed to read " + path.string());
        std::string line;
        while (std::getline(input, line)) {
            const size_t tab = line.find('\t');
            if (tab == std::string::npos)
                continue;
            characters_[static_cast<uint32_t>(std::stoul(line.substr(0, tab), nullptr, 16))] =
                line.substr(tab + 1);
        }
    }

    void load_phrases(const fs::path& path) {
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("failed to read " + path.string());
        std::string line;
        while (std::getline(input, line)) {
            const size_t tab = line.find('\t');
            if (tab == std::string::npos)
                continue;
            const std::string phrase = line.substr(0, tab);
            phrases_[phrase] = split_spaces(line.substr(tab + 1));
            max_phrase_chars_ = std::max(max_phrase_chars_, decode_utf8(phrase).size());
        }
    }

    std::vector<std::string> word_pinyin(const std::string& word) const {
        const auto chars = decode_utf8(word);
        std::vector<std::string> output;
        for (size_t position = 0; position < chars.size();) {
            const std::vector<std::string>* match = nullptr;
            size_t match_length = 0;
            std::string candidate;
            for (size_t cursor = position;
                 cursor < std::min(chars.size(), position + max_phrase_chars_); ++cursor) {
                candidate += chars[cursor].text;
                const auto found = phrases_.find(candidate);
                if (found != phrases_.end()) {
                    match = &found->second;
                    match_length = cursor - position + 1;
                }
            }
            if (match != nullptr) {
                output.insert(output.end(), match->begin(), match->end());
                position += match_length;
                continue;
            }
            const auto found = characters_.find(chars[position].codepoint);
            output.push_back(found == characters_.end() ? chars[position].text : found->second);
            ++position;
        }
        return output;
    }

    std::string word_to_ipa(const std::string& word) const {
        std::string output;
        for (const auto& pinyin : word_pinyin(word)) output += MandarinG2P::pinyin_to_ipa(pinyin);
        return output;
    }
};

MandarinG2P::MandarinG2P(fs::path data_dir) : impl_(std::make_unique<Impl>(std::move(data_dir))) {}

MandarinG2P::~MandarinG2P() = default;

std::string
MandarinG2P::phonemize(const std::string& text) const {
    return impl_->phonemize(text);
}

std::string
MandarinG2P::pinyin_to_ipa(const std::string& raw_pinyin) {
    if (raw_pinyin.size() < 2 || raw_pinyin.back() < '1' || raw_pinyin.back() > '5') {
        throw std::invalid_argument("invalid tone-3 pinyin syllable: " + raw_pinyin);
    }
    const char tone = raw_pinyin.back();
    std::string syllable = raw_pinyin.substr(0, raw_pinyin.size() - 1);
    std::transform(syllable.begin(), syllable.end(), syllable.begin(), [](unsigned char byte) {
        return static_cast<char>(std::tolower(byte));
    });
    replace_all(syllable, "v", "ü");

    std::string value;
    if (syllable == "hm")
        value = "hm0";
    else if (syllable == "hng")
        value = "hŋ0";
    else if (syllable == "m")
        value = "m0";
    else if (syllable == "n")
        value = "n0";
    else if (syllable == "ng")
        value = "ŋ0";
    else if (syllable == "io")
        value = "jɔ0";
    else if (syllable == "ê")
        value = "ɛ0";
    else if (syllable == "er")
        value = "ɚ0";
    else if (syllable == "o")
        value = "ɔ0";
    else {
        const auto [initial, final] = strict_initial_final(syllable);
        auto final_found = finals().find(final);
        if (final_found == finals().end()) {
            throw std::invalid_argument("unsupported pinyin final in: " + raw_pinyin);
        }
        if ((initial == "zh" || initial == "ch" || initial == "sh" || initial == "r") &&
            final == "i") {
            value = initials().at(initial) + std::string("ɻ") + "\xcc\xa9" + "0";
        } else if ((initial == "z" || initial == "c" || initial == "s") && final == "i") {
            value = initials().at(initial) + std::string("ɹ") + "\xcc\xa9" + "0";
        } else {
            if (!initial.empty())
                value = initials().at(initial);
            value += final_found->second;
        }
    }
    replace_all(value, "0", tone_mark(tone));
    replace_all(value, "ɻ\xcc\xa9", "ɨ");
    replace_all(value, "ɹ\xcc\xa9", "ɨ");
    return value;
}

}  // namespace nemo_speech::tts::kokoro
