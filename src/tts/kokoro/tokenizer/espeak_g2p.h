// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Behavioral port of misaki/espeak.py at commit
// fba1236595f2d2bf21d414ba6e57d25256afada3.
#pragma once

#include <string>

namespace nemo_speech::tts::kokoro {

// Calls the eSpeak-NG C API in-process. The eSpeak library owns process-global
// frontend state, so calls are serialized internally.
class EspeakG2P {
   public:
    // Implements Misaki EspeakG2P(version=None).
    static std::string phonemize(const std::string& text, const std::string& espeak_language);

    // Implements Misaki EspeakFallback(version=None), used only for English
    // words unresolved by the native lexicon.
    static std::string fallback(const std::string& text, bool british);

    static std::string version();
    static std::string data_path();
};

}  // namespace nemo_speech::tts::kokoro
