// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The kana/Hepburn, punctuation, nasal, sokuon, and long-vowel rules are a
// direct C++ port of hexgrad/misaki 0.9.4 misaki/cutlet.py, which adapts
// polm/cutlet (MIT). Morphological readings are obtained through native MeCab
// and the same UniDic-lite dictionary family used by first-generation Cutlet.
#include "japanese_g2p.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kokoro_tokenizer.h"
#include "mecab.h"
#include "resource_path.h"

#ifndef NEMO_SPEECH_KOKORO_UNIDIC_SOURCE_DIR
#define NEMO_SPEECH_KOKORO_UNIDIC_SOURCE_DIR ""
#endif
#ifndef NEMO_SPEECH_KOKORO_UNIDIC_INSTALLED_DIR
#define NEMO_SPEECH_KOKORO_UNIDIC_INSTALLED_DIR ""
#endif
#ifndef NEMO_SPEECH_KOKORO_JA_WORDS_SOURCE
#define NEMO_SPEECH_KOKORO_JA_WORDS_SOURCE ""
#endif
#ifndef NEMO_SPEECH_KOKORO_JA_WORDS_INSTALLED
#define NEMO_SPEECH_KOKORO_JA_WORDS_INSTALLED ""
#endif

namespace nemo_speech::tts::kokoro {
namespace {

namespace fs = std::filesystem;

struct Word {
    std::string surface;
    std::string hira;
    int char_type = 0;
};

struct Token {
    std::string surface;
    bool space = false;
};

std::vector<std::string>
split_utf8(const std::string& input) {
    std::vector<std::string> result;
    for (size_t offset = 0; offset < input.size();) {
        const unsigned char byte = static_cast<unsigned char>(input[offset]);
        size_t width = byte < 0x80 ? 1 : (byte & 0xe0) == 0xc0 ? 2 : (byte & 0xf0) == 0xe0 ? 3 : 4;
        if (offset + width > input.size())
            width = 1;
        result.push_back(input.substr(offset, width));
        offset += width;
    }
    return result;
}

std::string
katakana_to_hiragana(const std::string& input) {
    std::string result;
    for (const std::string& symbol : split_utf8(input)) {
        if (symbol.size() == 3) {
            const unsigned char b0 = static_cast<unsigned char>(symbol[0]);
            const unsigned char b1 = static_cast<unsigned char>(symbol[1]);
            const unsigned char b2 = static_cast<unsigned char>(symbol[2]);
            uint32_t cp = ((b0 & 0x0fU) << 12) | ((b1 & 0x3fU) << 6) | (b2 & 0x3fU);
            if (cp >= 0x30a1 && cp <= 0x30f6) {
                cp -= 0x60;
                result.push_back(static_cast<char>(0xe0U | (cp >> 12)));
                result.push_back(static_cast<char>(0x80U | ((cp >> 6) & 0x3fU)));
                result.push_back(static_cast<char>(0x80U | (cp & 0x3fU)));
                continue;
            }
        }
        result += symbol;
    }
    return result;
}

std::string
normalize_cutlet_text(const std::string& input) {
    const auto chars = split_utf8(input);
    static const std::unordered_map<std::string, std::string> extensions = {
        {"ㇰ", "ク"}, {"ㇱ", "シ"}, {"ㇲ", "ス"}, {"ㇳ", "ト"}, {"ㇴ", "ヌ"}, {"ㇵ", "ハ"},
        {"ㇶ", "ヒ"}, {"ㇷ", "フ"}, {"ㇸ", "ヘ"}, {"ㇹ", "ホ"}, {"ㇺ", "ム"}, {"ㇻ", "ラ"},
        {"ㇼ", "リ"}, {"ㇽ", "ル"}, {"ㇾ", "レ"}, {"ㇿ", "ロ"},
    };
    std::string prepared;
    for (size_t index = 0; index < chars.size(); ++index) {
        const bool wave = chars[index] == "〜" || chars[index] == "～";
        const bool next_digit = index + 1 < chars.size() &&
                                ((chars[index + 1].size() == 1 && chars[index + 1][0] >= '0' &&
                                  chars[index + 1][0] <= '9') ||
                                 (chars[index + 1] >= "０" && chars[index + 1] <= "９"));
        if (wave && next_digit) {
            prepared += "から";
        } else {
            const auto found = extensions.find(chars[index]);
            prepared += found == extensions.end() ? chars[index] : found->second;
        }
    }
    return KokoroTokenizer::normalize_nfkc(prepared);
}

const std::unordered_map<std::string, std::string>&
hepburn() {
    static const std::unordered_map<std::string, std::string> table = {
        {"ぁ", "a"},     {"あ", "a"},     {"ぃ", "i"},     {"い", "i"},     {"ぅ", "ɯ"},
        {"う", "ɯ"},     {"ぇ", "e"},     {"え", "e"},     {"ぉ", "o"},     {"お", "o"},
        {"か", "ka"},    {"が", "ɡa"},    {"き", "kʲi"},   {"ぎ", "ɡʲi"},   {"く", "kɯ"},
        {"ぐ", "ɡɯ"},    {"け", "ke"},    {"げ", "ɡe"},    {"こ", "ko"},    {"ご", "ɡo"},
        {"さ", "sa"},    {"ざ", "ʣa"},    {"し", "ɕi"},    {"じ", "ʥi"},    {"す", "sɨ"},
        {"ず", "zɨ"},    {"せ", "se"},    {"ぜ", "ʣe"},    {"そ", "so"},    {"ぞ", "ʣo"},
        {"た", "ta"},    {"だ", "da"},    {"ち", "ʨi"},    {"ぢ", "ʥi"},    {"つ", "ʦɨ"},
        {"づ", "zɨ"},    {"て", "te"},    {"で", "de"},    {"と", "to"},    {"ど", "do"},
        {"な", "na"},    {"に", "ɲi"},    {"ぬ", "nɯ"},    {"ね", "ne"},    {"の", "no"},
        {"は", "ha"},    {"ば", "ba"},    {"ぱ", "pa"},    {"ひ", "çi"},    {"び", "bʲi"},
        {"ぴ", "pʲi"},   {"ふ", "ɸɯ"},    {"ぶ", "bɯ"},    {"ぷ", "pɯ"},    {"へ", "he"},
        {"べ", "be"},    {"ぺ", "pe"},    {"ほ", "ho"},    {"ぼ", "bo"},    {"ぽ", "po"},
        {"ま", "ma"},    {"み", "mʲi"},   {"む", "mɯ"},    {"め", "me"},    {"も", "mo"},
        {"ゃ", "ja"},    {"や", "ja"},    {"ゅ", "jɯ"},    {"ゆ", "jɯ"},    {"ょ", "jo"},
        {"よ", "jo"},    {"ら", "ɾa"},    {"り", "ɾʲi"},   {"る", "ɾɯ"},    {"れ", "ɾe"},
        {"ろ", "ɾo"},    {"ゎ", "βa"},    {"わ", "βa"},    {"ゐ", "i"},     {"ゑ", "e"},
        {"を", "o"},     {"ゔ", "vɯ"},    {"ゕ", "ka"},    {"ゖ", "ke"},    {"ヷ", "va"},
        {"ヸ", "vʲi"},   {"ヹ", "ve"},    {"ヺ", "vo"},    {"いぇ", "je"},  {"うぃ", "βi"},
        {"うぇ", "βe"},  {"うぉ", "βo"},  {"きぇ", "kʲe"}, {"きゃ", "kʲa"}, {"きゅ", "kʲɨ"},
        {"きょ", "kʲo"}, {"ぎゃ", "ɡʲa"}, {"ぎゅ", "ɡʲɨ"}, {"ぎょ", "ɡʲo"}, {"くぁ", "kᵝa"},
        {"くぃ", "kᵝi"}, {"くぇ", "kᵝe"}, {"くぉ", "kᵝo"}, {"ぐぁ", "ɡᵝa"}, {"ぐぃ", "ɡᵝi"},
        {"ぐぇ", "ɡᵝe"}, {"ぐぉ", "ɡᵝo"}, {"しぇ", "ɕe"},  {"しゃ", "ɕa"},  {"しゅ", "ɕɨ"},
        {"しょ", "ɕo"},  {"じぇ", "ʥe"},  {"じゃ", "ʥa"},  {"じゅ", "ʥɨ"},  {"じょ", "ʥo"},
        {"ちぇ", "ʨe"},  {"ちゃ", "ʨa"},  {"ちゅ", "ʨɨ"},  {"ちょ", "ʨo"},  {"ぢゃ", "ʥa"},
        {"ぢゅ", "ʥɨ"},  {"ぢょ", "ʥo"},  {"つぁ", "ʦa"},  {"つぃ", "ʦʲi"}, {"つぇ", "ʦe"},
        {"つぉ", "ʦo"},  {"てぃ", "tʲi"}, {"てゅ", "tʲɨ"}, {"でぃ", "dʲi"}, {"でゅ", "dʲɨ"},
        {"とぅ", "tɯ"},  {"どぅ", "dɯ"},  {"にぇ", "ɲe"},  {"にゃ", "ɲa"},  {"にゅ", "ɲɨ"},
        {"にょ", "ɲo"},  {"ひぇ", "çe"},  {"ひゃ", "ça"},  {"ひゅ", "çɨ"},  {"ひょ", "ço"},
        {"びゃ", "bʲa"}, {"びゅ", "bʲɨ"}, {"びょ", "bʲo"}, {"ぴゃ", "pʲa"}, {"ぴゅ", "pʲɨ"},
        {"ぴょ", "pʲo"}, {"ふぁ", "ɸa"},  {"ふぃ", "ɸʲi"}, {"ふぇ", "ɸe"},  {"ふぉ", "ɸo"},
        {"ふゅ", "ɸʲɨ"}, {"ふょ", "ɸʲo"}, {"みゃ", "mʲa"}, {"みゅ", "mʲɨ"}, {"みょ", "mʲo"},
        {"りゃ", "ɾʲa"}, {"りゅ", "ɾʲɨ"}, {"りょ", "ɾʲo"}, {"ゔぁ", "va"},  {"ゔぃ", "vʲi"},
        {"ゔぇ", "ve"},  {"ゔぉ", "vo"},  {"ゔゅ", "bʲɨ"}, {"ゔょ", "bʲo"}, {"。", "."},
        {"、", ","},     {"？", "?"},     {"！", "!"},     {"「", "“"},     {"」", "”"},
        {"『", "“"},     {"』", "”"},     {"：", ":"},     {"；", ";"},     {"（", "("},
        {"）", ")"},     {"《", "("},     {"》", ")"},     {"【", "["},     {"】", "]"},
        {"・", " "},     {"，", ","},     {"～", "—"},     {"〜", "—"},     {"—", "—"},
        {"«", "“"},      {"»", "”"},      {"゚", ""},        {"゙", ""},
    };
    return table;
}

bool
all_ascii(const std::string& value) {
    return std::all_of(value.begin(), value.end(), [](unsigned char c) { return c < 0x80; });
}

std::string
trim_spaces(std::string value) {
    size_t read = 0;
    size_t write = 0;
    bool space = true;
    while (read < value.size()) {
        const bool is_space = std::isspace(static_cast<unsigned char>(value[read])) != 0;
        if (!is_space || !space)
            value[write++] = is_space ? ' ' : value[read];
        space = is_space;
        ++read;
    }
    value.resize(write);
    while (!value.empty() && value.back() == ' ') value.pop_back();
    return value;
}

const char*
digit_reading(char digit) {
    static const char* values[] = {"ゼロ", "いち", "に",   "さん", "よん",
                                   "ご",   "ろく", "なな", "はち", "きゅう"};
    return values[digit - '0'];
}

std::string
read_under_10000(int value, bool standalone) {
    if (value == 0)
        return {};
    std::string result;
    const int thousands = value / 1000;
    value %= 1000;
    if (thousands) {
        if (thousands == 1)
            result += standalone ? "せん" : "いっせん";
        else if (thousands == 3)
            result += "さんぜん";
        else if (thousands == 8)
            result += "はっせん";
        else
            result += std::string(digit_reading(static_cast<char>('0' + thousands))) + "せん";
    }
    const int hundreds = value / 100;
    value %= 100;
    if (hundreds) {
        if (hundreds == 1)
            result += "ひゃく";
        else if (hundreds == 3)
            result += "さんびゃく";
        else if (hundreds == 6)
            result += "ろっぴゃく";
        else if (hundreds == 8)
            result += "はっぴゃく";
        else
            result += std::string(digit_reading(static_cast<char>('0' + hundreds))) + "ひゃく";
    }
    const int tens = value / 10;
    const int ones = value % 10;
    if (tens) {
        if (tens != 1)
            result += digit_reading(static_cast<char>('0' + tens));
        result += "じゅう";
    }
    if (ones)
        result += digit_reading(static_cast<char>('0' + ones));
    return result;
}

std::string
number_to_kana(std::string digits) {
    while (digits.size() > 1 && digits.front() == '0') digits.erase(digits.begin());
    if (digits.size() > 9)
        return "Number length too long, choose less than 10 digits";
    uint64_t value = 0;
    for (char c : digits) value = value * 10 + static_cast<unsigned>(c - '0');
    if (value == 0)
        return "ゼロ";
    std::string result;
    const int oku = static_cast<int>(value / 100000000ULL);
    value %= 100000000ULL;
    const int man = static_cast<int>(value / 10000ULL);
    const int rest = static_cast<int>(value % 10000ULL);
    if (oku)
        result += read_under_10000(oku, false) + "おく";
    if (man)
        result += read_under_10000(man, false) + "まん";
    if (rest)
        result += read_under_10000(rest, oku == 0 && man == 0);
    return result;
}

std::string
normalize_numbers(const std::string& text) {
    std::string result;
    for (size_t index = 0; index < text.size();) {
        if (text[index] < '0' || text[index] > '9') {
            result.push_back(text[index++]);
            continue;
        }
        size_t end = index + 1;
        while (end < text.size() && text[end] >= '0' && text[end] <= '9') ++end;
        result.push_back(' ');
        result += number_to_kana(text.substr(index, end - index));
        index = end;
    }
    return result;
}

std::vector<std::string>
csv_fields(const char* raw) {
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (const char* p = raw ? raw : "";; ++p) {
        const char c = *p;
        if (c == '"') {
            if (quoted && p[1] == '"') {
                current.push_back('"');
                ++p;
            } else {
                quoted = !quoted;
            }
        } else if ((c == ',' && !quoted) || c == '\0') {
            fields.push_back(std::move(current));
            current.clear();
            if (c == '\0')
                break;
        } else {
            current.push_back(c);
        }
    }
    return fields;
}

std::string
map_reading(const std::string& reading) {
    const auto chars = split_utf8(katakana_to_hiragana(reading));
    const auto& table = hepburn();
    static const std::string sutegana = "ゃゅょぁぃぅぇぉ";
    static const std::string odoriji = "〃々ゝゞヽヾ";
    static const std::unordered_map<std::string, std::string> dakuten = {
        {"か", "が"}, {"き", "ぎ"}, {"く", "ぐ"}, {"け", "げ"}, {"こ", "ご"},
        {"さ", "ざ"}, {"し", "じ"}, {"す", "ず"}, {"せ", "ぜ"}, {"そ", "ぞ"},
        {"た", "だ"}, {"ち", "ぢ"}, {"つ", "づ"}, {"て", "で"}, {"と", "ど"},
        {"は", "ば"}, {"ひ", "び"}, {"ふ", "ぶ"}, {"へ", "べ"}, {"ほ", "ぼ"},
    };
    std::string output;
    for (size_t index = 0; index < chars.size(); ++index) {
        const std::string& previous = index ? chars[index - 1] : std::string();
        const std::string& current = chars[index];
        const std::string& next = index + 1 < chars.size() ? chars[index + 1] : std::string();
        if (odoriji.find(current) != std::string::npos) {
            if ((current == "ゝ" || current == "ヽ") && !previous.empty()) {
                // This deliberately matches Cutlet: an unvoiced iteration mark
                // returns the preceding kana itself rather than remapping it.
                output += previous;
            } else if ((current == "ゞ" || current == "ヾ") && !previous.empty()) {
                const auto voiced = dakuten.find(previous);
                if (voiced != dakuten.end()) {
                    const auto mapped = table.find(voiced->second);
                    if (mapped != table.end())
                        output += mapped->second;
                }
            }
            continue;
        }
        const auto pair_previous = table.find(previous + current);
        if (!previous.empty() && pair_previous != table.end()) {
            output += pair_previous->second;
            continue;
        }
        if (!next.empty() && table.count(current + next))
            continue;
        if (!next.empty() && sutegana.find(next) != std::string::npos) {
            if (current == "っ")
                continue;
            const auto base = table.find(current);
            const auto small = table.find(next);
            if (base != table.end() && small != table.end()) {
                auto stem = split_utf8(base->second);
                if (!stem.empty())
                    stem.pop_back();
                for (const auto& part : stem) output += part;
                output += small->second;
            }
            continue;
        }
        if (sutegana.find(current) != std::string::npos)
            continue;
        if (current == "ー") {
            output += "ː";
        } else if (current == "っ") {
            output += "ʔ";
        } else if (current == "ん") {
            const auto found = table.find(next);
            const std::string following = found == table.end() ? std::string() : found->second;
            if (!following.empty() && std::string("mpb").find(following[0]) != std::string::npos)
                output += "m";
            else if (following.rfind("k", 0) == 0 || following.rfind("ɡ", 0) == 0)
                output += "ŋ";
            else if (
                following.rfind("ɲ", 0) == 0 || following.rfind("ʨ", 0) == 0 ||
                following.rfind("ʥ", 0) == 0)
                output += "ɲ";
            else if (
                !following.empty() && std::string("ntdz").find(following[0]) != std::string::npos)
                output += "n";
            else if (following.rfind("ɾ", 0) == 0)
                output += "n";
            else
                output += "ɴ";
        } else {
            const auto found = table.find(current);
            if (found != table.end())
                output += found->second;
        }
    }
    return output;
}

bool
dictionary_dir(const fs::path& path) {
    return fs::is_regular_file(path / "sys.dic") && fs::is_regular_file(path / "unk.dic") &&
           fs::is_regular_file(path / "matrix.bin") && fs::is_regular_file(path / "version") &&
           fs::file_size(path / "sys.dic") == 187680870 &&
           fs::file_size(path / "matrix.bin") == 71544726;
}

fs::path
find_unidic() {
    std::vector<fs::path> candidates;
    if (const char* value = std::getenv("NEMO_SPEECH_KOKORO_UNIDIC_DIR")) {
        if (*value)
            candidates.emplace_back(value);
    }
    candidates.emplace_back(installed_misaki_data_root() / "unidic-lite");
    candidates.emplace_back(NEMO_SPEECH_KOKORO_UNIDIC_SOURCE_DIR);
    candidates.emplace_back(NEMO_SPEECH_KOKORO_UNIDIC_INSTALLED_DIR);
    for (const fs::path& path : candidates) {
        if (!path.empty() && dictionary_dir(path))
            return path;
    }
    return {};
}

fs::path
find_ja_words() {
    std::vector<fs::path> candidates;
    if (const char* value = std::getenv("NEMO_SPEECH_KOKORO_JA_WORDS")) {
        if (*value)
            candidates.emplace_back(value);
    }
    candidates.emplace_back(installed_misaki_data_root() / "ja_words.txt");
    candidates.emplace_back(NEMO_SPEECH_KOKORO_JA_WORDS_SOURCE);
    candidates.emplace_back(NEMO_SPEECH_KOKORO_JA_WORDS_INSTALLED);
    for (const fs::path& path : candidates) {
        if (!path.empty() && fs::is_regular_file(path))
            return path;
    }
    return {};
}

}  // namespace

class JapaneseG2P::Impl {
   public:
    explicit Impl(std::string words_payload) {
        const fs::path dictionary = find_unidic();
        if (dictionary.empty()) {
            throw std::runtime_error(
                "Kokoro Japanese requires the pinned UniDic-lite dictionary; set "
                "NEMO_SPEECH_KOKORO_UNIDIC_DIR or install the packaged dictionary");
        }
        const std::string args = "-r /dev/null -d " + dictionary.string();
        tagger.reset(MeCab::createTagger(args.c_str()));
        if (!tagger) {
            throw std::runtime_error(
                std::string("failed to initialize native MeCab/UniDic: ") +
                MeCab::getTaggerError());
        }
        std::string line;
        if (!words_payload.empty()) {
            size_t begin = 0;
            while (begin <= words_payload.size()) {
                const size_t end = words_payload.find('\n', begin);
                line = words_payload.substr(
                    begin, end == std::string::npos ? std::string::npos : end - begin);
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                if (!line.empty())
                    ja_words.insert(std::move(line));
                if (end == std::string::npos)
                    break;
                begin = end + 1;
            }
        } else {
            const fs::path words_path = find_ja_words();
            if (words_path.empty()) {
                throw std::runtime_error(
                    "Kokoro Japanese requires Misaki 0.9.4 ja_words.txt in its GGUF");
            }
            std::ifstream stream(words_path);
            while (std::getline(stream, line)) {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                if (!line.empty())
                    ja_words.insert(std::move(line));
            }
            if (!stream.eof()) {
                throw std::runtime_error("failed to load Kokoro Japanese grouping data");
            }
        }
        if (ja_words.empty()) {
            throw std::runtime_error("failed to load Kokoro Japanese grouping data");
        }
    }

    struct TaggerDeleter {
        void operator()(MeCab::Tagger* value) const { MeCab::deleteTagger(value); }
    };
    std::unique_ptr<MeCab::Tagger, TaggerDeleter> tagger;
    std::unordered_set<std::string> ja_words;
    mutable std::mutex mutex;
};

JapaneseG2P::JapaneseG2P(std::string words_payload)
    : impl_(std::make_shared<Impl>(std::move(words_payload))) {}
JapaneseG2P::~JapaneseG2P() = default;

std::string
JapaneseG2P::phonemize(const std::string& text) const {
    if (text.empty())
        return {};
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const std::string normalized = normalize_numbers(normalize_cutlet_text(text));
    const MeCab::Node* node = impl_->tagger->parseToNode(normalized.c_str());
    if (!node)
        throw std::runtime_error("native MeCab/UniDic analysis failed");
    std::vector<Word> words;
    for (; node; node = node->next) {
        if (node->stat == MECAB_BOS_NODE || node->stat == MECAB_EOS_NODE)
            continue;
        Word word;
        word.surface.assign(node->surface, node->length);
        const std::vector<std::string> fields = csv_fields(node->feature);
        std::string reading;
        if (fields.size() > 9 && fields[9] != "*" && !fields[9].empty())
            reading = fields[9];
        if (reading.empty() && fields.size() > 17 && fields[17] != "*")
            reading = fields[17];
        if (reading.empty())
            reading = word.surface;
        word.hira = katakana_to_hiragana(reading);
        word.char_type =
            (node->char_type == 7 || node->stat != MECAB_UNK_NODE) ? 6 : node->char_type;
        words.push_back(std::move(word));
    }

    std::vector<Word> grouped;
    for (size_t index = 0; index < words.size();) {
        size_t different = index + 1;
        while (different < words.size() && words[different].char_type == words[index].char_type)
            ++different;
        size_t end = 0;
        for (size_t candidate = different; candidate > index; --candidate) {
            std::string surface;
            for (size_t i = index; i < candidate; ++i) surface += words[i].surface;
            if (impl_->ja_words.count(surface)) {
                end = candidate;
                break;
            }
        }
        if (end == 0)
            end = index + 1;
        Word word;
        word.char_type = words[index].char_type;
        for (size_t i = index; i < end; ++i) {
            word.surface += words[i].surface;
            word.hira += words[i].hira;
        }
        grouped.push_back(std::move(word));
        index = end;
    }

    std::vector<Token> tokens;
    for (const Word& word : grouped) {
        std::string roma;
        if (all_ascii(word.surface)) {
            roma = word.surface;
        } else if (word.char_type == 3) {
            for (const std::string& symbol : split_utf8(word.surface)) {
                const auto found = hepburn().find(symbol);
                roma += found == hepburn().end() ? symbol : found->second;
            }
        } else if (word.char_type == 6) {
            roma = map_reading(word.hira);
        }
        Token token{roma, false};
        if (word.surface == "「" || word.surface == "『" || word.surface == "«" || roma == "(" ||
            roma == "[") {
            if (!tokens.empty())
                tokens.back().space = true;
        } else if (
            word.surface == "」" || word.surface == "』" || word.surface == "»" ||
            std::string("]).,?!:").find(roma) != std::string::npos) {
            if (!tokens.empty())
                tokens.back().space = false;
            token.space = true;
        } else if (roma != " ") {
            token.space = true;
        }
        roma.erase(std::remove(roma.begin(), roma.end(), '\0'), roma.end());
        token.surface.erase(
            std::remove(token.surface.begin(), token.surface.end(), '\0'), token.surface.end());
        tokens.push_back(std::move(token));
    }

    std::string output;
    for (Token& token : tokens) {
        size_t pos = 0;
        while ((pos = token.surface.find("っ", pos)) != std::string::npos)
            token.surface.erase(pos, std::string("っ").size());
        output += token.surface;
        if (token.space)
            output.push_back(' ');
    }
    output = trim_spaces(output);
    for (size_t pos = 0; (pos = output.find('(', pos)) != std::string::npos;)
        output.replace(pos, 1, "«");
    for (size_t pos = 0; (pos = output.find(')', pos)) != std::string::npos;)
        output.replace(pos, 1, "»");
    // Match Cutlet's two context-sensitive whitespace substitutions around a
    // sokuon. Keeping this code point based avoids byte-level UTF-8 edge cases.
    const auto out_chars = split_utf8(output);
    static const std::string keep_before_sokuon = "!\",.:;?»—…”";
    static const std::string keep_after_sokuon = "\"«“";
    std::string compact;
    for (size_t index = 0; index < out_chars.size(); ++index) {
        if (out_chars[index] == " ") {
            const std::string previous = index ? out_chars[index - 1] : std::string();
            const std::string next =
                index + 1 < out_chars.size() ? out_chars[index + 1] : std::string();
            if (next == "ʔ" && keep_before_sokuon.find(previous) == std::string::npos)
                continue;
            if (previous == "ʔ" && keep_after_sokuon.find(next) == std::string::npos)
                continue;
        }
        compact += out_chars[index];
    }
    return compact;
}

}  // namespace nemo_speech::tts::kokoro
