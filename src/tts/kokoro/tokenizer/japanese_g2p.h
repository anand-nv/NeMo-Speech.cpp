// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Native behavioral port of Misaki 0.9.4's default Cutlet Japanese path.
#pragma once

#include <memory>
#include <string>

namespace nemo_speech::tts::kokoro {

class JapaneseG2P {
   public:
    explicit JapaneseG2P(std::string words_payload = {});
    ~JapaneseG2P();

    JapaneseG2P(const JapaneseG2P&) = delete;
    JapaneseG2P& operator=(const JapaneseG2P&) = delete;

    std::string phonemize(const std::string& text) const;

   private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::kokoro
