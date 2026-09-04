// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/json.h"
#include "runtime.h"
#include "tts/kokoro/acoustic.h"
#include "tts/kokoro/decoder.h"
#include "tts/kokoro/duration.h"
#include "tts/kokoro/model.h"
#include "tts/kokoro/plbert.h"
#include "tts/kokoro/prosody.h"
#include "tts/kokoro/vocoder.h"

namespace {

using nemo_speech::json::Value;

std::string
read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open Kokoro reference manifest");
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::filesystem::path
tensor_path(
    const std::filesystem::path& manifest, const Value& fixture, size_t chunk_index,
    const Value& tensor) {
    std::ostringstream chunk;
    chunk << "chunk-" << std::setfill('0') << std::setw(4) << chunk_index;
    return manifest.parent_path() / (manifest.stem().string() + ".tensors") /
           fixture.at("id").string() / chunk.str() / tensor.at("path").string();
}

template <typename T>
std::vector<T>
read_tensor(
    const std::filesystem::path& manifest, const Value& fixture, size_t chunk_index,
    const Value& entry) {
    const auto path = tensor_path(manifest, fixture, chunk_index, entry);
    const size_t bytes = static_cast<size_t>(entry.at("nbytes").number());
    if (bytes % sizeof(T) != 0)
        throw std::runtime_error("invalid reference tensor size");
    std::vector<T> output(bytes / sizeof(T));
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(output.data()), static_cast<std::streamsize>(bytes));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("cannot read exact reference tensor: " + path.string());
    }
    return output;
}

std::vector<float>
channels_time_to_frames(const std::vector<float>& input, size_t channels, size_t frames) {
    if (input.size() != channels * frames)
        throw std::runtime_error("bad transpose shape");
    std::vector<float> output(input.size());
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t channel = 0; channel < channels; ++channel) {
            output[frame * channels + channel] = input[channel * frames + frame];
        }
    }
    return output;
}

void
compare_tensor(
    const std::string& name, const std::vector<float>& actual, const std::vector<float>& expected,
    float rtol = 1.0e-3f, float atol = 1.0e-4f) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + ": tensor size mismatch");
    }
    float maximum = 0.0f;
    double mean = 0.0;
    size_t failures = 0;
    size_t first_failure = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float difference = std::abs(actual[index] - expected[index]);
        maximum = std::max(maximum, difference);
        mean += difference;
        if (!std::isfinite(actual[index]) || difference > atol + rtol * std::abs(expected[index])) {
            if (failures++ == 0)
                first_failure = index;
        }
    }
    if (failures != 0) {
        std::ostringstream message;
        message << name << ": " << failures << " values exceeded tolerance; first=" << first_failure
                << ", max_abs=" << maximum << ", mae=" << mean / static_cast<double>(actual.size());
        throw std::runtime_error(message.str());
    }
}

void
compare_waveform(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("waveform sample count mismatch");
    }
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    double absolute_error = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]))
            throw std::runtime_error("non-finite waveform");
        dot += static_cast<double>(actual[index]) * expected[index];
        actual_norm += static_cast<double>(actual[index]) * actual[index];
        expected_norm += static_cast<double>(expected[index]) * expected[index];
        absolute_error += std::abs(static_cast<double>(actual[index]) - expected[index]);
    }
    const double cosine = dot / std::sqrt(actual_norm * expected_norm);
    const double mae = absolute_error / static_cast<double>(actual.size());
    if (cosine < 0.999 || mae > 1.0e-3) {
        std::ostringstream message;
        message << "waveform parity failed: cosine=" << cosine << ", mae=" << mae;
        throw std::runtime_error(message.str());
    }
}

}  // namespace

int
main(int argc, char** argv) {
    const bool use_gpu = argc == 4 && std::string(argv[3]) == "--gpu";
    if (argc != 3 && !use_gpu) {
        throw std::runtime_error("usage: test_kokoro_reference MODEL.gguf REFERENCE.json [--gpu]");
    }
    const std::filesystem::path manifest_path = argv[2];
    const Value manifest = Value::parse(read_text(manifest_path));
    if (manifest.at("format").number() != 2 || manifest.at("model").is_null()) {
        throw std::runtime_error("Kokoro reference does not contain model tensors");
    }
    const Value& fixture = manifest.at("fixtures").array().front();
    const Value& chunk = fixture.at("chunks").array().front();
    const Value& model = chunk.at("model");
    const Value& tensors = model.at("tensors");
    auto f32 = [&](const char* name) {
        return read_tensor<float>(manifest_path, fixture, 0, tensors.at(name));
    };
    auto i32 = [&](const char* name) {
        return read_tensor<int32_t>(manifest_path, fixture, 0, tensors.at(name));
    };

    const std::vector<int32_t> ids = i32("phoneme_ids");
    const std::vector<int32_t> framed = i32("input_ids");
    if (framed.size() != ids.size() + 2 || framed.front() != 0 || framed.back() != 0) {
        throw std::runtime_error("invalid framed-ID reference");
    }

    ggml_runtime::GGUFLoader loader(argv[1]);
    nemo_speech::tts::kokoro::KokoroModelMetadata metadata(loader);
    const std::string voice = fixture.at("voice").string();
    const std::vector<float> voice_style = metadata.read_voice_style(loader, voice, ids.size());
    compare_tensor("voice_style", voice_style, f32("voice_style"), 0.0f, 0.0f);
    const std::vector<float> decoder_style(voice_style.begin(), voice_style.begin() + 128);
    const std::vector<float> duration_style(voice_style.begin() + 128, voice_style.end());

    std::vector<float> projected;
    {
        nemo_speech::tts::kokoro::KokoroPlbertEncoder stage(argv[1], use_gpu);
        projected = stage.encode(framed);
    }
    compare_tensor(
        "duration_projected", projected,
        channels_time_to_frames(f32("duration_projected"), 512, framed.size()),
        use_gpu ? 1.0e-2f : 1.0e-3f, use_gpu ? 2.0e-2f : 1.0e-4f);

    nemo_speech::tts::kokoro::KokoroDurationResult timing;
    {
        nemo_speech::tts::kokoro::KokoroDurationPredictor stage(argv[1], use_gpu);
        timing = stage.predict(
            projected, framed.size(), duration_style,
            static_cast<float>(model.at("speed").number()));
    }
    if (timing.frames != i32("predicted_durations")) {
        throw std::runtime_error("predicted duration parity failed");
    }
    compare_tensor(
        "duration_continuous", timing.values, f32("duration_continuous"),
        use_gpu ? 5.0e-3f : 1.0e-3f, use_gpu ? 3.0e-3f : 1.0e-4f);
    compare_tensor(
        "duration_encoded", timing.encoded_features, f32("duration_encoded"),
        use_gpu ? 1.0e-2f : 1.0e-3f, use_gpu ? 2.0e-2f : 1.0e-4f);

    nemo_speech::tts::kokoro::KokoroAcousticFeatures acoustic;
    {
        nemo_speech::tts::kokoro::KokoroAcousticEncoder stage(argv[1], use_gpu);
        acoustic = stage.encode(framed, timing.frames, timing.encoded_features);
    }
    compare_tensor(
        "decoder_asr", acoustic.text,
        channels_time_to_frames(f32("decoder_asr"), 512, acoustic.frame_count),
        use_gpu ? 2.0e-3f : 1.0e-3f, use_gpu ? 1.2e-3f : 3.0e-4f);
    compare_tensor(
        "prosody_shared", acoustic.prosody_shared, f32("prosody_shared"),
        use_gpu ? 5.0e-3f : 1.0e-3f, use_gpu ? 5.0e-3f : 1.0e-4f);

    nemo_speech::tts::kokoro::KokoroProsody prosody;
    {
        nemo_speech::tts::kokoro::KokoroProsodyHeads stage(argv[1], use_gpu);
        prosody = stage.predict(acoustic.prosody_shared, acoustic.frame_count, duration_style);
    }
    // GGML's current CPU Conv1d implementation requires F16 im2col weights
    // even for an F32 GGUF. Keep these convolution-heavy boundaries under an
    // explicit regression ceiling while the stricter recurrent/linear stages
    // above retain the release tolerances.
    compare_tensor("f0", prosody.f0, f32("f0"), 1.0e-2f, 4.0e-2f);
    compare_tensor("noise", prosody.noise, f32("noise"), 1.0e-2f, 4.0e-2f);

    std::vector<float> latent;
    {
        nemo_speech::tts::kokoro::KokoroDecoderEncoder stage(argv[1], use_gpu);
        latent = stage.encode(
            acoustic.text, acoustic.frame_count, prosody.f0, prosody.noise, decoder_style);
    }
    compare_tensor(
        "decoder_output", latent,
        channels_time_to_frames(f32("decoder.decode.3.output"), 512, prosody.f0.size()),
        use_gpu ? 2.0e-2f : 1.0e-2f, use_gpu ? 8.0e-2f : 4.0e-2f);

    std::vector<float> waveform;
    std::vector<float> end_to_end_waveform;
    {
        nemo_speech::tts::kokoro::KokoroVocoder stage(argv[1], use_gpu);
        const std::vector<float> reference_f0 = f32("f0");
        const std::vector<float> reference_latent =
            channels_time_to_frames(f32("decoder.decode.3.output"), 512, reference_f0.size());
        waveform = stage.synthesize(
            reference_latent, reference_f0, decoder_style,
            static_cast<int64_t>(model.at("seed").number()));
        end_to_end_waveform = stage.synthesize(
            latent, prosody.f0, decoder_style, static_cast<int64_t>(model.at("seed").number()));
        std::vector<float> streamed;
        size_t callbacks = 0;
        const auto stream_stats = stage.synthesize_stream(
            reference_latent, reference_f0, decoder_style,
            static_cast<int64_t>(model.at("seed").number()), 12000,
            [&](const std::vector<float>& tile) {
                ++callbacks;
                streamed.insert(streamed.end(), tile.begin(), tile.end());
                return true;
            });
        if (callbacks < 2 || stream_stats.cancelled || streamed != waveform ||
            stream_stats.samples_written != waveform.size()) {
            throw std::runtime_error("streaming iSTFT changed the reference waveform");
        }
    }
    compare_waveform(waveform, f32("waveform"));
    compare_waveform(end_to_end_waveform, f32("waveform"));
    return 0;
}
