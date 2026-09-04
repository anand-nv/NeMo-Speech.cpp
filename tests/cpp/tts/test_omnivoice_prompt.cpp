// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/ggml/runtime.h"
#include "tts/omnivoice/codec.h"
#include "tts/omnivoice/prompt.h"

using namespace nemo_speech::tts::omnivoice;

namespace {

PromptFingerprint
fingerprint() {
    return {
        "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
        "730839316de585f4c8298ec0e1712efc10fb19c6fa4e36eb741cb8d51ebcf6aa",
        "408f669b7e2b045fdf54201d815bd364e6667dbd845115da81239c40bc6dcfd1",
        "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d"};
}

void
require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc > 3 || (argc == 3 && std::string(argv[2]) != "--gpu"))
        throw std::invalid_argument("usage: test_omnivoice_prompt [CODEC.gguf [--gpu]]");
    std::vector<float> stereo(4800 * 2, 0.0f);
    for (size_t frame = 1200; frame < 3600; ++frame) {
        stereo[frame * 2] = 0.02f;
        stereo[frame * 2 + 1] = 0.04f;
    }
    auto prepared = preprocess_reference_audio(stereo.data(), 4800, 2, 48000, false);
    require(prepared.mono_24khz.size() == 1920, "prompt hop clipping is incorrect");
    require(
        prepared.original_rms > 0.020f && prepared.original_rms < 0.022f,
        "prompt RMS must be measured before quiet-reference gain");

    VoicePrompt prompt;
    prompt.transcript = "Fixture.";
    prompt.reference_rms = 0.05f;
    prompt.fingerprint = fingerprint();
    for (size_t q = 0; q < prompt.audio_codes.size(); ++q)
        prompt.audio_codes[q] = {static_cast<int32_t>(q), static_cast<int32_t>(q + 10)};
    validate_voice_prompt(prompt, &prompt.fingerprint);

    const auto path = std::filesystem::temp_directory_path() / "nemo-speech-omnivoice-prompt.bin";
    save_voice_prompt(prompt, path.string());
    const VoicePrompt restored = load_voice_prompt(path.string(), &prompt.fingerprint);
    std::filesystem::remove(path);
    require(
        restored.audio_codes == prompt.audio_codes && restored.transcript == prompt.transcript &&
            restored.reference_rms == prompt.reference_rms,
        "saved prompt round trip failed");
    auto wrong = prompt.fingerprint;
    wrong.audio_tokenizer_sha256[0] = '0';
    bool rejected = false;
    try {
        validate_voice_prompt(prompt, &wrong);
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "mismatched prompt fingerprint was accepted");
    prompt.transcript = " \t\n";
    rejected = false;
    try {
        validate_voice_prompt(prompt);
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "whitespace-only prompt transcript was accepted");

    if (argc >= 2) {
        ggml_runtime::Params params;
        params.use_gpu = argc == 3;
        ggml_runtime::BackendManager backends(params);
        CodecEncoder encoder(backends, argv[1]);
        std::vector<float> audio(960);
        constexpr float pi = 3.14159265358979323846f;
        for (size_t i = 0; i < audio.size(); ++i) {
            const float time = static_cast<float>(i) / 24000.0f;
            audio[i] = 0.1f * std::sin(2.0f * pi * 440.0f * time) +
                       0.025f * std::sin(2.0f * pi * 997.0f * time);
        }
        const auto encoded = create_voice_prompt(
            encoder, audio.data(), audio.size(), 1, 24000, "Reference fixture", fingerprint(),
            false);
        require(encoded.audio_codes[0].size() == 1, "prompt encoder produced wrong frame count");
        require(
            encoded.transcript == "Reference fixture",
            "disabled prompt preprocessing changed text");
    }
    return 0;
}
