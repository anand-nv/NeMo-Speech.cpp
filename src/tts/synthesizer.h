// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Transport-neutral text-to-PCM orchestration.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "magpietts/runtime.h"
#include "omnivoice/options.h"
#include "tts/tokenizer/tokenizer.h"

namespace nemo_speech::tts {

namespace omnivoice {
class Runtime;
class BidirectionalStream;
struct VoicePrompt;
struct RuntimeConfig;
struct RuntimeSynthesisRequest;
}  // namespace omnivoice

struct SynthesizerConfig {
    MagpieRuntimeConfig runtime;
    std::string omnivoice_model;
    std::string omnivoice_audio_tokenizer_model;
    std::optional<OmniVoiceOptions> omnivoice_options;
    std::string tokenizer_model_dir;
    std::string text_normalizer_model_dir;
    MagpieTokenizerConfig tokenizer;
    std::string default_language_code = "en-US";
    std::string default_voice_name;
};

struct SynthesisRequest {
    std::string text;
    std::string language_code;
    std::string voice_name;
    int output_sample_rate = 0;  // 0 = model rate
    MagpieSynthesisOptions options;
    std::optional<OmniVoiceOptions> omnivoice_options;
    std::string instruction;
    const omnivoice::VoicePrompt* voice_prompt = nullptr;
};

struct SynthesisMetadata {
    std::string original_text;
    std::string processed_text;
    std::string language_code;
    std::string tokenizer_name;
    int speaker = 0;
    int sample_rate = 0;
    size_t token_count = 0;
    size_t chunk_count = 0;
};

struct PreparedSynthesis {
    SynthesisMetadata metadata;
    std::vector<int32_t> tokens;
    std::vector<std::vector<int32_t>> token_chunks;
    MagpieSynthesisOptions options;
    double normalizer_ms = 0.0;
    double tokenizer_ms = 0.0;
};

struct SynthesisResult {
    SynthesisMetadata metadata;
    MagpieSynthesisStats stats;
    uint64_t output_samples = 0;
    bool cancelled = false;
};

class Synthesizer {
   public:
    using PcmCallback = std::function<bool(const SynthesisMetadata&, const std::string& pcm_s16le)>;

    explicit Synthesizer(SynthesizerConfig config);
    ~Synthesizer();

    Synthesizer(const Synthesizer&) = delete;
    Synthesizer& operator=(const Synthesizer&) = delete;

    PreparedSynthesis prepare(const SynthesisRequest& request) const;
    SynthesisResult synthesize(const PreparedSynthesis& request, const PcmCallback& callback = {});
    SynthesisResult synthesize(const SynthesisRequest& request, const PcmCallback& callback = {});
    SynthesisResult synthesize_tokens(
        const std::vector<int32_t>& tokens, const MagpieSynthesisOptions& options = {},
        int output_sample_rate = 0, const PcmCallback& callback = {});

    // Initialize lazy runtime state through the normal synthesis path.
    SynthesisResult warmup(const std::string& text, int steps = -1);

    int sample_rate() const;
    int speaker_count() const;
    const std::vector<std::string>& speaker_names() const;
    std::vector<std::string> supported_language_codes() const;
    const std::string& model_name() const;
    const std::string& default_language_code() const;
    int default_speaker() const;
    bool text_normalization_enabled() const;
    bool is_omnivoice() const;
    OmniVoiceOptions omnivoice_defaults() const;

    std::unique_ptr<omnivoice::VoicePrompt> create_voice_prompt(
        const float* interleaved_pcm, size_t frames, int channels, int sample_rate,
        const std::string& transcript, bool preprocess = true);
    std::unique_ptr<omnivoice::VoicePrompt> load_voice_prompt(const std::string& path) const;
    void save_voice_prompt(const omnivoice::VoicePrompt& prompt, const std::string& path) const;

#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
    omnivoice::Runtime& omnivoice_runtime();
    omnivoice::RuntimeConfig resolve_omnivoice_config(const SynthesisRequest& request) const;
    omnivoice::RuntimeSynthesisRequest resolve_omnivoice_request(
        const SynthesisRequest& request) const;
#endif

   private:
    int resolve_speaker(const std::string& voice_name) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts
