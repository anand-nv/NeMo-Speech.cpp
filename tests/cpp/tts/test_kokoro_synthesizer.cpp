// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tts/synthesizer.h"

namespace {

template <typename Function>
void
require_invalid(Function&& function, const char* label) {
    try {
        function();
    }
    catch (const std::invalid_argument&) {
        return;
    }
    catch (const std::length_error&) {
        return;
    }
    throw std::runtime_error(std::string("Kokoro accepted invalid ") + label);
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc != 2)
        throw std::runtime_error("usage: test_kokoro_synthesizer MODEL.gguf");
    using namespace nemo_speech::tts;

    SynthesizerConfig config;
    config.family = TtsModelFamily::Kokoro;
    config.kokoro_model = argv[1];
    config.runtime.lt_backend = MagpieBackendPreference::Cpu;
    config.default_voice_name = "af_heart";
    Synthesizer synthesizer(std::move(config));

    if (synthesizer.model_name() != "kokoro" || synthesizer.sample_rate() != 24000 ||
        synthesizer.speaker_count() != 54 || synthesizer.supported_language_codes().size() != 9 ||
        synthesizer.speaker_names_for_language("ja").size() != 5) {
        throw std::runtime_error("Kokoro unified synthesizer inventory is invalid");
    }

    SynthesisRequest request;
    request.text = "Hello!";
    request.voice_name = "KOKORO.AF_HEART";
    PreparedSynthesis prepared = synthesizer.prepare(request);
    if (prepared.family != TtsModelFamily::Kokoro || prepared.voice_name != "af_heart" ||
        prepared.metadata.language_code != "en-US" ||
        prepared.metadata.tokenizer_name != "misaki-0.9.4-native" ||
        prepared.metadata.token_count == 0 || prepared.metadata.chunk_count != 1) {
        throw std::runtime_error("Kokoro unified frontend metadata is invalid");
    }

    require_invalid(
        [&] {
            SynthesisRequest invalid = request;
            invalid.language_code = "ja-JP";
            (void)synthesizer.prepare(invalid);
        },
        "voice/language mismatch");
    require_invalid(
        [&] {
            SynthesisRequest invalid = request;
            invalid.options.steps = 4;
            (void)synthesizer.prepare(invalid);
        },
        "Magpie steps option");
    require_invalid(
        [&] {
            SynthesisRequest invalid = request;
            invalid.options.top_k = 10;
            (void)synthesizer.prepare(invalid);
        },
        "Magpie top_k option");
    require_invalid(
        [&] {
            SynthesisRequest invalid = request;
            invalid.options.override_temperature = true;
            invalid.options.temperature = 0.7f;
            (void)synthesizer.prepare(invalid);
        },
        "Magpie temperature option");
    require_invalid(
        [&] {
            SynthesisRequest invalid = request;
            invalid.options.override_cfg_scale = true;
            invalid.options.cfg_scale = 1.5f;
            (void)synthesizer.prepare(invalid);
        },
        "Magpie cfg_scale option");

    const MagpieSynthesisOptions defaults;
    require_invalid([&] { (void)synthesizer.synthesize_tokens({0}, defaults); }, "BOS token");
    require_invalid(
        [&] { (void)synthesizer.synthesize_tokens({178}, defaults); }, "out-of-vocabulary token");
    require_invalid(
        [&] { (void)synthesizer.synthesize_tokens(std::vector<int32_t>(511, 50), defaults); },
        "511-token request");
    require_invalid(
        [&] {
            MagpieSynthesisOptions invalid;
            invalid.override_speed = true;
            invalid.speed = std::numeric_limits<float>::quiet_NaN();
            (void)synthesizer.synthesize_tokens({50}, invalid);
        },
        "non-finite speed");

    SynthesizerConfig mixed;
    mixed.family = TtsModelFamily::Kokoro;
    mixed.kokoro_model = argv[1];
    mixed.runtime.magpie_model = "magpie.gguf";
    require_invalid([&] { Synthesizer invalid(std::move(mixed)); }, "mixed model family");

    size_t callbacks = 0;
    size_t bytes = 0;
    MagpieSynthesisOptions options;
    options.seed = 1234;
    options.override_speed = true;
    options.speed = 2.0f;
    const SynthesisResult result = synthesizer.synthesize_tokens(
        {50, 83, 156, 54, 57}, options, 16000,
        [&](const SynthesisMetadata& metadata, const std::string& pcm) {
            if (metadata.sample_rate != 16000 || pcm.empty() || pcm.size() % 2 != 0)
                throw std::runtime_error("invalid resampled Kokoro callback");
            ++callbacks;
            bytes += pcm.size();
            return true;
        },
        "kokoro.AF_HEART", "en-us");
    if (result.cancelled || callbacks < 2 || bytes != result.output_samples * 2 ||
        result.stats.sample_rate != 16000 ||
        result.stats.samples_written != result.output_samples || result.stats.audio_s <= 0.0 ||
        result.stats.e2e_ttfa_ms <= 0.0 || result.stats.tokenizer_ms != 0.0) {
        throw std::runtime_error("Kokoro unified resampling/statistics contract failed");
    }
    return 0;
}
