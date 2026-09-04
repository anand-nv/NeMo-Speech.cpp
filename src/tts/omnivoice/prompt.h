// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nemo_speech::tts::omnivoice {

class CodecEncoder;

inline constexpr uint32_t kVoicePromptFormatVersion = 1;

struct PromptFingerprint {
    std::string source_revision;
    std::string omnivoice_model_sha256;
    std::string tokenizer_sha256;
    std::string audio_tokenizer_sha256;
};

struct VoicePrompt {
    uint32_t format_version = kVoicePromptFormatVersion;
    std::array<std::vector<int32_t>, 8> audio_codes;
    std::string transcript;
    float reference_rms = 0.0f;
    PromptFingerprint fingerprint;
};

struct PreparedReferenceAudio {
    std::vector<float> mono_24khz;
    // Measured after downmix/resampling but before the quiet-reference boost.
    float original_rms = 0.0f;
};

// Mirrors the pinned Python prompt path: channel mean, torchaudio's default
// sinc interpolation to 24 kHz, pre-gain RMS, optional quiet-reference boost,
// pydub-compatible -50 dBFS silence processing, and 960-sample hop clipping.
PreparedReferenceAudio preprocess_reference_audio(
    const float* interleaved_pcm, size_t frames, int32_t channels, int32_t sample_rate,
    bool remove_prompt_silence = true);

std::vector<float> remove_silence(
    const std::vector<float>& mono, int32_t sample_rate, int32_t middle_ms, int32_t leading_ms,
    int32_t trailing_ms);

VoicePrompt create_voice_prompt(
    CodecEncoder& encoder, const float* interleaved_pcm, size_t frames, int32_t channels,
    int32_t sample_rate, const std::string& transcript, const PromptFingerprint& fingerprint,
    bool preprocess = true);

void validate_voice_prompt(
    const VoicePrompt& prompt, const PromptFingerprint* expected_fingerprint = nullptr);
void save_voice_prompt(const VoicePrompt& prompt, const std::string& path);
VoicePrompt load_voice_prompt(
    const std::string& path, const PromptFingerprint* expected_fingerprint = nullptr);

}  // namespace nemo_speech::tts::omnivoice
