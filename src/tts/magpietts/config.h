// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>

#include "magpietts.h"
#include "runtime.h"
#include "tts/omnivoice/options.h"

namespace nemo_speech::common {
class ParameterParser;
}

namespace nemo_speech::tts {

struct MagpieTtsServerConfig {
    MagpieRuntimeConfig runtime;
    std::string omnivoice_model;
    std::string omnivoice_audio_tokenizer_model;
    std::optional<OmniVoiceOptions> omnivoice_options;
    std::string tokenizer_model_dir;
    std::string tn_model_dir;
    MagpieTokenizerConfig tokenizer_config;
    std::string default_language_code = "en-US";
    std::string default_voice_name;
    bool benchmark = false;
    bool warmup = true;
    std::string warmup_text = "Hello from Magpie T T S.";
    int warmup_steps = 8;

    void Register(common::ParameterParser& parser);
};

void register_magpie_stream_params(common::ParameterParser& parser, magpie_stream_params& params);

}  // namespace nemo_speech::tts
