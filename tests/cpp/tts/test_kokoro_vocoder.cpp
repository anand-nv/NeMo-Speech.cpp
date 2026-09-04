// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tts/kokoro/vocoder.h"

namespace {

std::vector<float>
read_exact(const std::string& path, size_t count) {
    std::vector<float> result(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("invalid Kokoro vocoder fixture input");
    }
    return result;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cerr
            << "usage: test_kokoro_vocoder MODEL.gguf LATENT.f32 F0.f32 STYLE.f32 [WAVE.f32]\n";
        return 2;
    }
    const auto latent = read_exact(argv[2], 108 * 512);
    const auto f0 = read_exact(argv[3], 108);
    const auto style = read_exact(argv[4], 128);
    nemo_speech::tts::kokoro::KokoroVocoder vocoder(argv[1], false);
    uint64_t selected_seed = 0;
    nemo_speech::tts::kokoro::KokoroVocoderDebug debug;
    const auto waveform = vocoder.synthesize(latent, f0, style, 1234, &selected_seed, &debug);
    const auto repeated = vocoder.synthesize(latent, f0, style, 1234);
    if (selected_seed != 1234 || waveform.size() != 32400 || waveform != repeated) {
        throw std::runtime_error("Kokoro vocoder deterministic contract failed");
    }
    for (float value : waveform) {
        if (!std::isfinite(value))
            throw std::runtime_error("non-finite Kokoro waveform");
    }
    std::vector<float> streamed;
    size_t callbacks = 0;
    const auto stream_stats = vocoder.synthesize_stream(
        latent, f0, style, 1234, 4096, [&](const std::vector<float>& tile) {
            ++callbacks;
            streamed.insert(streamed.end(), tile.begin(), tile.end());
            return true;
        });
    if (callbacks < 2 || stream_stats.cancelled ||
        stream_stats.samples_written != waveform.size() || streamed != waveform ||
        stream_stats.generator_tiles != 3 || stream_stats.max_generator_input_frames != f0.size()) {
        throw std::runtime_error("Kokoro bounded iSTFT streaming parity failed");
    }
    size_t cancellation_callbacks = 0;
    const auto cancelled =
        vocoder.synthesize_stream(latent, f0, style, 1234, 4096, [&](const std::vector<float>&) {
            ++cancellation_callbacks;
            return false;
        });
    if (!cancelled.cancelled || cancellation_callbacks != 1 || cancelled.samples_written != 4096 ||
        cancelled.generator_tiles != 1 || cancelled.max_generator_input_frames != f0.size()) {
        throw std::runtime_error("Kokoro bounded iSTFT cancellation failed");
    }
    if (argc == 6) {
        std::ofstream stream(argv[5], std::ios::binary);
        stream.write(
            reinterpret_cast<const char*>(waveform.data()),
            static_cast<std::streamsize>(waveform.size() * sizeof(float)));
        if (!stream)
            throw std::runtime_error("failed to write Kokoro waveform");
        auto write_debug = [](const std::string& path, const std::vector<float>& values) {
            std::ofstream output(path, std::ios::binary);
            output.write(
                reinterpret_cast<const char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
            if (!output)
                throw std::runtime_error("failed to write Kokoro debug tensor");
        };
        write_debug(std::string(argv[5]) + ".harmonic", debug.harmonic);
        write_debug(std::string(argv[5]) + ".sine", debug.sine);
        write_debug(std::string(argv[5]) + ".magnitude", debug.magnitude);
        write_debug(std::string(argv[5]) + ".phase", debug.phase);
    }
    std::cout << "Kokoro native vocoder smoke test passed\n";
    return 0;
}
