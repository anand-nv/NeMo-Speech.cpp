// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer.h"

#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/stringpiece.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "runtime/ggml/runtime.h"

namespace nemo_speech::tts::omnivoice {
namespace {

constexpr const char* kRegex =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

void
check_icu(UErrorCode status, const char* operation) {
    if (U_FAILURE(status)) {
        throw std::runtime_error(
            std::string("OmniVoice tokenizer ") + operation + " failed: " + u_errorName(status));
    }
}

icu::UnicodeString
strict_unicode(const std::string& text) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t length = 0;
    u_strFromUTF8(nullptr, 0, &length, text.data(), static_cast<int32_t>(text.size()), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        check_icu(status, "UTF-8 validation");
    }
    status = U_ZERO_ERROR;
    std::vector<UChar> buffer(static_cast<size_t>(length) + 1);
    u_strFromUTF8(
        buffer.data(), static_cast<int32_t>(buffer.size()), &length, text.data(),
        static_cast<int32_t>(text.size()), &status);
    check_icu(status, "UTF-8 conversion");
    return icu::UnicodeString(buffer.data(), length);
}

std::string
to_utf8(const icu::UnicodeString& value) {
    std::string out;
    value.toUTF8String(out);
    return out;
}

struct Pair {
    std::string left;
    std::string right;

    bool operator==(const Pair& other) const { return left == other.left && right == other.right; }
};

struct PairHash {
    size_t operator()(const Pair& pair) const {
        const size_t first = std::hash<std::string>{}(pair.left);
        const size_t second = std::hash<std::string>{}(pair.right);
        return first ^ (second + 0x9e3779b97f4a7c15ULL + (first << 6) + (first >> 2));
    }
};

std::array<std::string, 256>
byte_encoder() {
    std::array<UChar32, 256> points{};
    std::array<bool, 256> assigned{};
    for (int value = 33; value <= 126; ++value) {
        points[value] = value;
        assigned[value] = true;
    }
    for (int value = 161; value <= 172; ++value) {
        points[value] = value;
        assigned[value] = true;
    }
    for (int value = 174; value <= 255; ++value) {
        points[value] = value;
        assigned[value] = true;
    }
    int extra = 0;
    for (int value = 0; value <= 255; ++value) {
        if (!assigned[value])
            points[value] = 256 + extra++;
    }

    std::array<std::string, 256> encoded;
    for (int value = 0; value <= 255; ++value) {
        encoded[value] = to_utf8(icu::UnicodeString(points[value]));
    }
    return encoded;
}

}  // namespace

class Tokenizer::Impl {
   public:
    explicit Impl(const ggml_runtime::GGUFLoader& loader) {
        const std::string regex = loader.get_str("omnivoice.tokenizer.pre_tokenizer_regex");
        if (loader.get_str("omnivoice.tokenizer.model") != "BPE" ||
            loader.get_str("omnivoice.tokenizer.normalizer") != "NFC" || regex != kRegex) {
            throw std::runtime_error("unsupported OmniVoice tokenizer metadata");
        }

        tokens_ = loader.get_str_array("omnivoice.tokenizer.tokens");
        const auto left = loader.get_str_array("omnivoice.tokenizer.merge_left");
        const auto right = loader.get_str_array("omnivoice.tokenizer.merge_right");
        const auto added_ids = loader.get_i32_array("omnivoice.tokenizer.added.ids");
        const auto added_content = loader.get_str_array("omnivoice.tokenizer.added.content");
        if (tokens_.size() != 151676 || left.size() != 151387 || right.size() != left.size() ||
            added_ids.size() != 33 || added_content.size() != added_ids.size()) {
            throw std::runtime_error("malformed OmniVoice tokenizer tables");
        }

        vocab_.reserve(tokens_.size());
        for (size_t id = 0; id < tokens_.size(); ++id) {
            if (!vocab_.emplace(tokens_[id], static_cast<int32_t>(id)).second) {
                throw std::runtime_error(
                    "duplicate OmniVoice vocabulary token at ID " + std::to_string(id));
            }
        }
        merges_.reserve(left.size());
        for (size_t rank = 0; rank < left.size(); ++rank) {
            if (!merges_.emplace(Pair{left[rank], right[rank]}, static_cast<int32_t>(rank))
                     .second) {
                throw std::runtime_error(
                    "duplicate OmniVoice BPE merge at rank " + std::to_string(rank));
            }
        }
        for (size_t index = 0; index < added_ids.size(); ++index) {
            if (added_ids[index] != static_cast<int32_t>(151643 + index) ||
                tokens_[static_cast<size_t>(added_ids[index])] != added_content[index]) {
                throw std::runtime_error("inconsistent OmniVoice added-token table");
            }
            added_.emplace_back(added_content[index], added_ids[index]);
        }
        std::stable_sort(added_.begin(), added_.end(), [](const auto& a, const auto& b) {
            return a.first.size() > b.first.size();
        });

        UErrorCode status = U_ZERO_ERROR;
        normalizer_ = icu::Normalizer2::getNFCInstance(status);
        check_icu(status, "NFC initialization");
        regex_.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(kRegex), 0, status));
        check_icu(status, "regex compilation");
        if (regex_ == nullptr)
            throw std::runtime_error("OmniVoice tokenizer regex is null");
        bytes_ = byte_encoder();
    }

    int32_t token_id(const std::string& token) const {
        auto it = vocab_.find(token);
        if (it == vocab_.end())
            throw std::invalid_argument("unknown OmniVoice token: " + token);
        return it->second;
    }

    std::vector<int32_t> encode(const std::string& text) const {
        // Validate the complete string before byte-scanning it for added tokens.
        (void)strict_unicode(text);
        std::vector<int32_t> output;
        size_t plain_start = 0;
        size_t cursor = 0;
        while (cursor < text.size()) {
            const std::pair<std::string, int32_t>* found = nullptr;
            for (const auto& item : added_) {
                if (item.first.size() <= text.size() - cursor &&
                    text.compare(cursor, item.first.size(), item.first) == 0) {
                    found = &item;
                    break;
                }
            }
            if (found == nullptr) {
                ++cursor;
                continue;
            }
            append_plain(text.substr(plain_start, cursor - plain_start), output);
            output.push_back(found->second);
            cursor += found->first.size();
            plain_start = cursor;
        }
        append_plain(text.substr(plain_start), output);
        return output;
    }

    std::vector<int32_t> encode_with_nonverbal_tags(const std::string& text) const {
        static const std::array<const char*, 13> tags = {
            "[laughter]",           "[sigh]",        "[confirmation-en]", "[question-en]",
            "[question-ah]",        "[question-oh]", "[question-ei]",     "[question-yi]",
            "[surprise-ah]",        "[surprise-oh]", "[surprise-wa]",     "[surprise-yo]",
            "[dissatisfaction-hnn]"};
        (void)strict_unicode(text);
        std::vector<int32_t> output;
        size_t last = 0;
        size_t cursor = 0;
        while (cursor < text.size()) {
            const char* match = nullptr;
            for (const char* tag : tags) {
                const size_t length = std::char_traits<char>::length(tag);
                if (length <= text.size() - cursor && text.compare(cursor, length, tag) == 0) {
                    match = tag;
                    break;
                }
            }
            if (match == nullptr) {
                ++cursor;
                continue;
            }
            append_encoded(text.substr(last, cursor - last), output);
            const size_t length = std::char_traits<char>::length(match);
            append_encoded(text.substr(cursor, length), output);
            cursor += length;
            last = cursor;
        }
        append_encoded(text.substr(last), output);
        return output;
    }

   private:
    void append_encoded(const std::string& text, std::vector<int32_t>& output) const {
        const auto ids = encode(text);
        output.insert(output.end(), ids.begin(), ids.end());
    }

    void append_plain(const std::string& text, std::vector<int32_t>& output) const {
        if (text.empty())
            return;
        const icu::UnicodeString input = strict_unicode(text);
        UErrorCode status = U_ZERO_ERROR;
        icu::UnicodeString normalized;
        normalizer_->normalize(input, normalized, status);
        check_icu(status, "NFC normalization");

        std::unique_ptr<icu::RegexMatcher> matcher(regex_->matcher(normalized, status));
        check_icu(status, "regex matcher creation");
        int32_t consumed = 0;
        while (matcher->find(status)) {
            check_icu(status, "regex match");
            if (matcher->start(status) != consumed) {
                throw std::runtime_error("OmniVoice split regex left unmatched input");
            }
            const icu::UnicodeString piece = matcher->group(status);
            check_icu(status, "regex group extraction");
            consumed = matcher->end(status);
            append_bpe(to_utf8(piece), output);
        }
        check_icu(status, "regex matching");
        if (consumed != normalized.length()) {
            throw std::runtime_error("OmniVoice split regex did not consume the full input");
        }
    }

    void append_bpe(const std::string& piece, std::vector<int32_t>& output) const {
        std::string encoded;
        for (unsigned char byte : piece) encoded += bytes_[byte];

        {
            std::shared_lock<std::shared_mutex> lock(cache_mutex_);
            auto cached = cache_.find(encoded);
            if (cached != cache_.end()) {
                output.insert(output.end(), cached->second.begin(), cached->second.end());
                return;
            }
        }

        std::vector<std::string> symbols;
        for (int32_t offset = 0; offset < static_cast<int32_t>(encoded.size());) {
            UChar32 cp = 0;
            const int32_t start = offset;
            U8_NEXT(encoded.data(), offset, static_cast<int32_t>(encoded.size()), cp);
            if (cp < 0)
                throw std::logic_error("invalid internal byte-level Unicode mapping");
            symbols.emplace_back(encoded.substr(static_cast<size_t>(start), offset - start));
        }

        while (symbols.size() > 1) {
            int32_t best_rank = std::numeric_limits<int32_t>::max();
            Pair best;
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                auto found = merges_.find(Pair{symbols[i], symbols[i + 1]});
                if (found != merges_.end() && found->second < best_rank) {
                    best_rank = found->second;
                    best = found->first;
                }
            }
            if (best_rank == std::numeric_limits<int32_t>::max())
                break;
            std::vector<std::string> merged;
            merged.reserve(symbols.size());
            for (size_t i = 0; i < symbols.size();) {
                if (i + 1 < symbols.size() && symbols[i] == best.left &&
                    symbols[i + 1] == best.right) {
                    merged.push_back(symbols[i] + symbols[i + 1]);
                    i += 2;
                } else {
                    merged.push_back(std::move(symbols[i++]));
                }
            }
            symbols = std::move(merged);
        }

        std::vector<int32_t> ids;
        ids.reserve(symbols.size());
        for (const auto& symbol : symbols) {
            auto found = vocab_.find(symbol);
            if (found == vocab_.end()) {
                throw std::runtime_error(
                    "OmniVoice BPE produced a token absent from its vocabulary");
            }
            ids.push_back(found->second);
        }
        {
            std::unique_lock<std::shared_mutex> lock(cache_mutex_);
            cache_.emplace(std::move(encoded), ids);
        }
        output.insert(output.end(), ids.begin(), ids.end());
    }

    const icu::Normalizer2* normalizer_ = nullptr;
    std::unique_ptr<icu::RegexPattern> regex_;
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<Pair, int32_t, PairHash> merges_;
    std::vector<std::pair<std::string, int32_t>> added_;
    std::array<std::string, 256> bytes_;
    mutable std::shared_mutex cache_mutex_;
    mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

Tokenizer::Tokenizer(const ggml_runtime::GGUFLoader& loader)
    : impl_(std::make_unique<Impl>(loader)) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<int32_t>
Tokenizer::encode(const std::string& utf8) const {
    return impl_->encode(utf8);
}

std::vector<int32_t>
Tokenizer::encode_with_nonverbal_tags(const std::string& utf8) const {
    return impl_->encode_with_nonverbal_tags(utf8);
}

int32_t
Tokenizer::token_id(const std::string& token) const {
    return impl_->token_id(token);
}

}  // namespace nemo_speech::tts::omnivoice
