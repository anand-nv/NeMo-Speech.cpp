// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Port of Misaki 0.9.4 zh.ZHG2P(version=None). The pinyin-to-IPA transcription
// rules were adapted by Misaki from pinyin-to-ipa (MIT).
#pragma once

#include <filesystem>
#include <memory>
#include <string>

namespace nemo_speech::tts::kokoro {

class MandarinG2P {
   public:
    explicit MandarinG2P(std::filesystem::path data_dir = {});
    ~MandarinG2P();

    MandarinG2P(const MandarinG2P&) = delete;
    MandarinG2P& operator=(const MandarinG2P&) = delete;

    std::string phonemize(const std::string& text) const;
    static std::string pinyin_to_ipa(const std::string& tone3_pinyin);

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::kokoro
