// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "nemo_speech/tts.h"

namespace {

bool
collect(const uint8_t* pcm, size_t bytes, void* opaque) {
    auto* output = static_cast<std::vector<uint8_t>*>(opaque);
    output->insert(output->end(), pcm, pcm + bytes);
    return true;
}

bool
reject_audio(const uint8_t*, size_t, void*) {
    return false;
}

void
check(nemo_speech_tts_status status, const char* operation) {
    if (status != NEMO_SPEECH_TTS_OK)
        throw std::runtime_error(std::string(operation) + ": " + nemo_speech_tts_last_error());
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: test_omnivoice_c_api MODEL.gguf CODEC.gguf [--gpu]\n";
        return 2;
    }
    nemo_speech_tts_model_config model{};
    model.size = sizeof(model);
    model.omnivoice_model = argv[1];
    model.omnivoice_audio_tokenizer_model = argv[2];
    auto runtime = nemo_speech_tts_runtime_config_default();
    runtime.lt_backend = argc == 4 ? NEMO_SPEECH_TTS_BACKEND_CUDA : NEMO_SPEECH_TTS_BACKEND_CPU;
    auto defaults = nemo_speech_tts_omnivoice_options_default();
    defaults.num_steps = 1;
    defaults.postprocess_output = false;
    defaults.pad_duration_s = 0.0;
    defaults.fade_duration_s = 0.0;
    defaults.duration_s = 0.04;
    runtime.omnivoice_options = &defaults;
    nemo_speech_tts_synthesizer_config config{};
    config.size = sizeof(config);
    config.model = &model;
    config.runtime = &runtime;
    config.default_language_code = "en";
    config.default_voice_name = "auto";

    nemo_speech_tts_synthesizer* synthesizer = nullptr;
    check(nemo_speech_tts_create(&config, &synthesizer), "create");
    if (nemo_speech_tts_sample_rate(synthesizer) != 24000 ||
        nemo_speech_tts_speaker_count(synthesizer) != 1)
        throw std::runtime_error("OmniVoice C inventory is incorrect");

    auto options = nemo_speech_tts_synthesis_options_default();
    options.language_code = "en";
    options.voice_name = "auto";
    options.seed = 7;
    std::vector<uint8_t> offline;
    auto stats = nemo_speech_tts_synthesis_stats_default();
    check(
        nemo_speech_tts_synthesize_text(synthesizer, &options, "Test.", collect, &offline, &stats),
        "synthesize");
    if (offline.size() != 960 * 2 || stats.generated_frames != 1)
        throw std::runtime_error("OmniVoice C synthesis dimensions are incorrect");

    std::vector<float> prompt_audio(960);
    constexpr float pi = 3.14159265358979323846f;
    for (size_t i = 0; i < prompt_audio.size(); ++i)
        prompt_audio[i] = 0.1f * std::sin(2.0f * pi * 440.0f * i / 24000.0f);
    nemo_speech_tts_voice_prompt* prompt = nullptr;
    check(
        nemo_speech_tts_voice_prompt_create(
            synthesizer, prompt_audio.data(), prompt_audio.size(), 1, 24000, "Prompt.", false,
            &prompt),
        "prompt create");
    const auto prompt_path =
        std::filesystem::temp_directory_path() / "nemo-speech-c-api-omnivoice.prompt";
    check(
        nemo_speech_tts_voice_prompt_save(synthesizer, prompt, prompt_path.c_str()), "prompt save");
    nemo_speech_tts_voice_prompt_destroy(prompt);
    prompt = nullptr;
    check(
        nemo_speech_tts_voice_prompt_load(synthesizer, prompt_path.c_str(), &prompt),
        "prompt load");
    std::filesystem::remove(prompt_path);

    std::vector<uint8_t> cloned;
    options.voice_prompt = prompt;
    check(
        nemo_speech_tts_synthesize_text(
            synthesizer, &options, "Cloned.", collect, &cloned, nullptr),
        "saved prompt synthesis");
    if (cloned.empty())
        throw std::runtime_error("saved OmniVoice prompt produced no audio");

    options.instruction = "male";
    if (nemo_speech_tts_synthesize_text(
            synthesizer, &options, "Invalid.", collect, &cloned, nullptr) !=
        NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT) {
        throw std::runtime_error("OmniVoice C API accepted prompt plus instruction");
    }
    options.instruction = nullptr;
    options.voice_prompt = nullptr;
    nemo_speech_tts_voice_prompt_destroy(prompt);

    auto invalid_omnivoice = defaults;
    invalid_omnivoice.duration_s = -1.0;
    options.omnivoice_options = &invalid_omnivoice;
    if (nemo_speech_tts_synthesize_text(
            synthesizer, &options, "Invalid.", collect, &cloned, nullptr) !=
        NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT) {
        throw std::runtime_error("OmniVoice C API accepted a negative duration");
    }
    options.omnivoice_options = nullptr;

    std::vector<uint8_t> streamed;
    nemo_speech_tts_stream* stream = nullptr;
    check(
        nemo_speech_tts_stream_create(synthesizer, &options, collect, &streamed, &stream),
        "stream create");
    check(nemo_speech_tts_stream_push_text(stream, "Hello", 5, false), "stream push");
    check(nemo_speech_tts_stream_push_text(stream, ".", 1, false), "stream commit");
    stats = nemo_speech_tts_synthesis_stats_default();
    check(nemo_speech_tts_stream_finish(stream, &stats), "stream finish");
    if (streamed.empty() || stats.chunks != 1)
        throw std::runtime_error("OmniVoice C stream produced no audio");
    nemo_speech_tts_stream_destroy(stream);

    stream = nullptr;
    check(
        nemo_speech_tts_stream_create(synthesizer, &options, reject_audio, nullptr, &stream),
        "cancel stream create");
    check(nemo_speech_tts_stream_push_text(stream, "Stop.", 5, true), "cancel stream push");
    stats = nemo_speech_tts_synthesis_stats_default();
    if (nemo_speech_tts_stream_finish(stream, &stats) != NEMO_SPEECH_TTS_ERROR_CANCELLED)
        throw std::runtime_error("OmniVoice C stream did not report callback cancellation");
    nemo_speech_tts_stream_destroy(stream);
    nemo_speech_tts_destroy(synthesizer);
    return 0;
}
