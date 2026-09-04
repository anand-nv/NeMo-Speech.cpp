// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "frontend.h"

#include <algorithm>
#include <stdexcept>

#include "nvtx_utils.h"
#include "runtime.h"

namespace nemo_speech::tts::kokoro {

KokoroFrontend::KokoroFrontend(const std::string& model_path)
    : loader_(std::make_unique<ggml_runtime::GGUFLoader>(model_path)), metadata_(*loader_),
      tokenizer_(
          metadata_.vocabulary(), {}, metadata_.english_lexicon_data(),
          metadata_.misaki_lexicon_json("ja_words")) {}

KokoroFrontend::~KokoroFrontend() = default;

KokoroPreparedText
KokoroFrontend::prepare(
    const std::string& text, const std::string& raw_language, const std::string& raw_voice) const {
    const ggml_nvtx::range nvtx_range("kokoro.frontend.prepare");
    const std::string nvtx_parameters =
        "kokoro.parameters.frontend text_bytes=" + std::to_string(text.size()) +
        " raw_language=" + raw_language + " raw_voice=" + raw_voice;
    ggml_nvtx::mark(nvtx_parameters.c_str());
    const std::string voice = KokoroTokenizer::canonicalize_voice(raw_voice);
    const std::string language = raw_language.empty()
                                     ? KokoroTokenizer::language_for_voice(voice)
                                     : KokoroTokenizer::canonicalize_language(raw_language);
    if (!KokoroTokenizer::voice_matches_language(voice, language)) {
        throw std::invalid_argument(
            "Kokoro voice '" + voice + "' is not compatible with language " + language);
    }
    KokoroPreparedText prepared;
    prepared.voice = voice;
    prepared.voice_index = metadata_.voice_index(voice);
    prepared.tokenization = tokenizer_.tokenize(text, language);
    return prepared;
}

KokoroChunk
KokoroFrontend::prepare_tokens(
    const std::vector<int32_t>& ids, const std::string& raw_language,
    const std::string& raw_voice) const {
    const ggml_nvtx::range nvtx_range("kokoro.frontend.prepare_tokens");
    if (ids.empty())
        throw std::invalid_argument("Kokoro phoneme ID list is empty");
    if (ids.size() > static_cast<size_t>(metadata_.hparams().phoneme_limit)) {
        throw std::length_error("Kokoro phoneme ID list exceeds 510 IDs");
    }
    const std::string voice = KokoroTokenizer::canonicalize_voice(raw_voice);
    const std::string language = raw_language.empty()
                                     ? KokoroTokenizer::language_for_voice(voice)
                                     : KokoroTokenizer::canonicalize_language(raw_language);
    if (!KokoroTokenizer::voice_matches_language(voice, language)) {
        throw std::invalid_argument(
            "Kokoro voice '" + voice + "' is not compatible with language " + language);
    }
    for (const int32_t id : ids) {
        if (id <= 0 || id >= metadata_.hparams().vocab_size ||
            metadata_.vocabulary()[static_cast<size_t>(id)].empty()) {
            throw std::invalid_argument(
                "invalid unframed Kokoro phoneme ID: " + std::to_string(id));
        }
    }
    KokoroChunk chunk;
    chunk.ids = ids;
    for (const int32_t id : ids) {
        chunk.phonemes += metadata_.vocabulary()[static_cast<size_t>(id)];
    }
    return chunk;
}

std::vector<float>
KokoroFrontend::voice_style(const std::string& voice, size_t unframed_phoneme_count) const {
    const ggml_nvtx::range nvtx_range("kokoro.frontend.voice_style");
    return metadata_.read_voice_style(*loader_, voice, unframed_phoneme_count);
}

}  // namespace nemo_speech::tts::kokoro
