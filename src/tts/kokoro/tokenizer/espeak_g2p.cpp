// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Behavioral port of misaki/espeak.py at commit
// fba1236595f2d2bf21d414ba6e57d25256afada3. Misaki is Apache-2.0.
#include "espeak_g2p.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Minimal declarations from eSpeak-NG's public speak_lib.h API revision 12.
// Keeping these declarations local avoids making a development header a
// runtime packaging requirement. The implementation still links and calls the
// native C API directly.
extern "C" {
typedef struct {
    const char* name;
    const char* languages;
    const char* identifier;
    unsigned char gender;
    unsigned char age;
    unsigned char variant;
    unsigned char reserved;
    int score;
    void* spare;
} espeak_VOICE;

int espeak_Initialize(int output, int buflength, const char* path, int options);
int espeak_SetVoiceByProperties(espeak_VOICE* voice_spec);
const char* espeak_TextToPhonemes(const void** textptr, int textmode, int phonememode);
int espeak_Terminate(void);
const char* espeak_Info(const char** path_data);
}

namespace nemo_speech::tts::kokoro {
namespace {

constexpr int kAudioOutputSynchronous = 2;
constexpr int kCharsUtf8 = 1;
constexpr int kPhonemesIpa = 0x02;
constexpr int kPhonemesTie = 0x80;
constexpr int kInitializeDontExit = 0x8000;
constexpr int kTieCharacter = '^';

void
replace_all(std::string& value, std::string_view from, std::string_view to) {
    if (from.empty()) {
        return;
    }
    for (size_t offset = 0; (offset = value.find(from, offset)) != std::string::npos;) {
        value.replace(offset, from.size(), to);
        offset += to.size();
    }
}

// phonemizer preserves these marks outside the eSpeak call. Splitting them
// here reproduces that behavior while still obtaining all phonemes through the
// C API. The set is the punctuation used by Kokoro/Misaki.
bool
is_preserved_punctuation(std::string_view codepoint) {
    static constexpr std::array<std::string_view, 28> marks = {
        ";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "\"", "'",  "(",  ")",
        "[", "]", "{", "}", "“", "”", "‘", "’", "«", "»", "，", "。", "！", "？",
    };
    return std::find(marks.begin(), marks.end(), codepoint) != marks.end();
}

bool
is_internal_ascii_apostrophe(const std::string& text, size_t offset, size_t width) {
    if (width != 1 || text[offset] != '\'' || offset == 0 || offset + 1 >= text.size()) {
        return false;
    }
    return std::isalnum(static_cast<unsigned char>(text[offset - 1])) != 0 &&
           std::isalnum(static_cast<unsigned char>(text[offset + 1])) != 0;
}

size_t
utf8_width(unsigned char first) {
    if (first < 0x80) {
        return 1;
    }
    if ((first & 0xe0) == 0xc0) {
        return 2;
    }
    if ((first & 0xf0) == 0xe0) {
        return 3;
    }
    if ((first & 0xf8) == 0xf0) {
        return 4;
    }
    return 1;
}

class EspeakState {
   public:
    EspeakState() {
        if (espeak_Initialize(kAudioOutputSynchronous, 0, nullptr, kInitializeDontExit) <= 0) {
            throw std::runtime_error("eSpeak-NG initialization failed");
        }
        const char* path = nullptr;
        const char* reported = espeak_Info(&path);
        version_ = reported == nullptr ? std::string() : reported;
        data_path_ = path == nullptr ? std::string() : path;
        if (version_.rfind("1.51", 0) != 0) {
            espeak_Terminate();
            throw std::runtime_error(
                "Kokoro Misaki frontend requires eSpeak-NG 1.51; found '" + version_ + "'");
        }
    }

    ~EspeakState() { espeak_Terminate(); }

    EspeakState(const EspeakState&) = delete;
    EspeakState& operator=(const EspeakState&) = delete;

    std::mutex mutex;
    std::string version_;
    std::string data_path_;
};

EspeakState&
state() {
    static EspeakState instance;
    return instance;
}

std::string
phonemize_run_locked(const std::string& text, const std::string& language) {
    if (text.empty()) {
        return {};
    }
    espeak_VOICE voice{};
    voice.languages = language.c_str();
    if (espeak_SetVoiceByProperties(&voice) != 0) {
        throw std::runtime_error("eSpeak-NG has no voice for language '" + language + "'");
    }

    const void* cursor = text.c_str();
    std::string output;
    while (cursor != nullptr) {
        const void* before = cursor;
        const char* value = espeak_TextToPhonemes(
            &cursor, kCharsUtf8, kPhonemesIpa | kPhonemesTie | (kTieCharacter << 8));
        if (value != nullptr) {
            output += value;
        }
        if (cursor == before) {
            throw std::runtime_error("eSpeak-NG did not advance while phonemizing input");
        }
    }
    while (!output.empty() && std::isspace(static_cast<unsigned char>(output.back()))) {
        output.pop_back();
    }
    return output;
}

std::string
phonemize_preserving_punctuation_locked(const std::string& text, const std::string& language) {
    std::string output;
    std::string run;
    const auto flush = [&]() {
        if (!run.empty()) {
            size_t begin = 0;
            while (begin < run.size() && std::isspace(static_cast<unsigned char>(run[begin]))) {
                ++begin;
            }
            size_t end = run.size();
            while (end > begin && std::isspace(static_cast<unsigned char>(run[end - 1]))) {
                --end;
            }
            output.append(run, 0, begin);
            if (end > begin) {
                output += phonemize_run_locked(run.substr(begin, end - begin), language);
            }
            output.append(run, end, run.size() - end);
            run.clear();
        }
    };
    for (size_t offset = 0; offset < text.size();) {
        const size_t width =
            std::min(utf8_width(static_cast<unsigned char>(text[offset])), text.size() - offset);
        const std::string_view cp(text.data() + offset, width);
        if (is_preserved_punctuation(cp) && !is_internal_ascii_apostrophe(text, offset, width)) {
            flush();
            output.append(cp);
        } else {
            run.append(cp);
        }
        offset += width;
    }
    flush();
    return output;
}

void
rewrite_syllabic_consonants(std::string& phonemes) {
    // Python: re.sub(r'(\S)\u0329', r'ᵊ\1', ps)
    constexpr std::string_view mark = "\xcc\xa9";  // U+0329
    for (size_t position = phonemes.find(mark); position != std::string::npos;
         position = phonemes.find(mark, position)) {
        if (position == 0) {
            phonemes.erase(position, mark.size());
            continue;
        }
        size_t previous = position - 1;
        while (previous > 0 && (static_cast<unsigned char>(phonemes[previous]) & 0xc0) == 0x80) {
            --previous;
        }
        if (std::isspace(static_cast<unsigned char>(phonemes[previous]))) {
            phonemes.erase(position, mark.size());
            continue;
        }
        const std::string consonant = phonemes.substr(previous, position - previous);
        phonemes.replace(previous, position + mark.size() - previous, "ᵊ" + consonant);
        position = previous + std::strlen("ᵊ") + consonant.size();
    }
}

std::string
general_rewrites(std::string phonemes) {
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 9> rewrites = {{
        {"a^ɪ", "I"},
        {"a^ʊ", "W"},
        {"d^z", "ʣ"},
        {"d^ʒ", "ʤ"},
        {"e^ɪ", "A"},
        {"o^ʊ", "O"},
        {"s^s", "S"},
        {"t^s", "ʦ"},
        {"t^ʃ", "ʧ"},
    }};
    for (const auto& [from, to] : rewrites) {
        replace_all(phonemes, from, to);
    }
    replace_all(phonemes, "ə^ʊ", "Q");
    replace_all(phonemes, "ɔ^ɪ", "Y");
    replace_all(phonemes, "^", "");
    replace_all(phonemes, "-", "");
    replace_all(phonemes, "«", "(");
    replace_all(phonemes, "»", ")");
    return phonemes;
}

std::string
fallback_rewrites(std::string phonemes, bool british) {
    // Longest-first order is material (for example ʔˌn̩ before ʔn̩).
    static const std::vector<std::pair<std::string_view, std::string_view>> rewrites = {
        {"ʔˌn\xcc\xa9", "ʔn"},
        {"ʔn\xcc\xa9", "ʔn"},
        {"a^ɪ", "I"},
        {"a^ʊ", "W"},
        {"d^ʒ", "ʤ"},
        {"e^ɪ", "A"},
        {"t^ʃ", "ʧ"},
        {"ɔ^ɪ", "Y"},
        {"ə^l", "ᵊl"},
        {"ʲo", "jo"},
        {"ʲə", "jə"},
        {"ɚ", "əɹ"},
        {"ʲ", ""},
        {"r", "ɹ"},
        {"x", "k"},
        {"ç", "k"},
        {"ɐ", "ə"},
        {"ɬ", "l"},
        {"\xcc\x83", ""},
    };
    for (const auto& [from, to] : rewrites) {
        replace_all(phonemes, from, to);
    }
    rewrite_syllabic_consonants(phonemes);
    if (british) {
        replace_all(phonemes, "e^ə", "ɛː");
        replace_all(phonemes, "iə", "ɪə");
        replace_all(phonemes, "ə^ʊ", "Q");
    } else {
        replace_all(phonemes, "o^ʊ", "O");
        replace_all(phonemes, "ɜːɹ", "ɜɹ");
        replace_all(phonemes, "ɜː", "ɜɹ");
        replace_all(phonemes, "ɪə", "iə");
        replace_all(phonemes, "ː", "");
    }
    replace_all(phonemes, "o", "ɔ");
    replace_all(phonemes, "ɾ", "T");
    replace_all(phonemes, "ʔ", "t");
    replace_all(phonemes, "^", "");
    return phonemes;
}

}  // namespace

std::string
EspeakG2P::phonemize(const std::string& raw_text, const std::string& espeak_language) {
    std::string text = raw_text;
    replace_all(text, "«", "“");
    replace_all(text, "»", "”");
    replace_all(text, "(", "«");
    replace_all(text, ")", "»");
    auto& shared = state();
    std::lock_guard<std::mutex> lock(shared.mutex);
    return general_rewrites(phonemize_preserving_punctuation_locked(text, espeak_language));
}

std::string
EspeakG2P::fallback(const std::string& text, bool british) {
    auto& shared = state();
    std::lock_guard<std::mutex> lock(shared.mutex);
    return fallback_rewrites(
        phonemize_preserving_punctuation_locked(text, british ? "en-gb" : "en-us"), british);
}

std::string
EspeakG2P::version() {
    return state().version_;
}

std::string
EspeakG2P::data_path() {
    return state().data_path_;
}

}  // namespace nemo_speech::tts::kokoro
