// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "vocoder.h"

#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ggml_ops.h"
#include "model.h"
#include "nvtx_utils.h"
#include "runtime.h"

namespace nemo_speech::tts::kokoro {
namespace {

constexpr int kSampleRate = 24000;
constexpr int kHarmonics = 9;
constexpr int kF0Upsample = 300;
constexpr int kNfft = 20;
constexpr int kHop = 5;
constexpr int kBins = 11;
constexpr float kPi = 3.14159265358979323846f;

uint64_t
splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

double
uniform_open(uint64_t seed, uint64_t counter) {
    const uint64_t bits = splitmix64(seed ^ splitmix64(counter));
    return (static_cast<double>(bits >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

float
normal_counter(uint64_t seed, uint64_t counter) {
    const double u1 = uniform_open(seed, counter * 2);
    const double u2 = uniform_open(seed, counter * 2 + 1);
    return static_cast<float>(
        std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * static_cast<double>(kPi) * u2));
}

std::vector<float>
make_sine_source(const std::vector<float>& f0, uint64_t seed) {
    const ggml_nvtx::range nvtx_range("kokoro.vocoder.source.generate_full");
    if (f0.empty() || f0.size() > std::numeric_limits<size_t>::max() / kF0Upsample) {
        throw std::invalid_argument("invalid Kokoro F0 source length");
    }
    const size_t samples = f0.size() * kF0Upsample;
    std::vector<float> coarse_phase(f0.size() * kHarmonics);
    for (int harmonic = 0; harmonic < kHarmonics; ++harmonic) {
        double accumulator = harmonic == 0 ? 0.0 : uniform_open(seed, harmonic);
        for (size_t frame = 0; frame < f0.size(); ++frame) {
            double increment =
                std::fmod(static_cast<double>(f0[frame]) * (harmonic + 1) / kSampleRate, 1.0);
            if (increment < 0.0)
                increment += 1.0;
            accumulator += increment;
            coarse_phase[frame * kHarmonics + static_cast<size_t>(harmonic)] =
                static_cast<float>(accumulator * 2.0 * kPi * kF0Upsample);
        }
    }

    std::vector<float> result(samples * kHarmonics);
    for (size_t sample = 0; sample < samples; ++sample) {
        // F.interpolate(..., mode="linear", align_corners=False).
        const double source = (static_cast<double>(sample) + 0.5) / kF0Upsample - 0.5;
        const int64_t left_raw = static_cast<int64_t>(std::floor(source));
        const size_t left = static_cast<size_t>(
            std::clamp<int64_t>(left_raw, 0, static_cast<int64_t>(f0.size() - 1)));
        const size_t right = std::min(left + 1, f0.size() - 1);
        const float fraction = static_cast<float>(source - std::floor(source));
        const size_t f0_frame = std::min(sample / kF0Upsample, f0.size() - 1);
        const float uv = f0[f0_frame] > 10.0f ? 1.0f : 0.0f;
        const float noise_scale = uv != 0.0f ? 0.003f : 0.1f / 3.0f;
        for (int harmonic = 0; harmonic < kHarmonics; ++harmonic) {
            const float a = coarse_phase[left * kHarmonics + static_cast<size_t>(harmonic)];
            const float b = coarse_phase[right * kHarmonics + static_cast<size_t>(harmonic)];
            const float phase = a + fraction * (b - a);
            const uint64_t counter =
                0x100000000ULL + sample * kHarmonics + static_cast<size_t>(harmonic);
            result[sample * kHarmonics + static_cast<size_t>(harmonic)] =
                0.1f * std::sin(phase) * uv + noise_scale * normal_counter(seed, counter);
        }
    }
    return result;
}

size_t
reflect_sample(int64_t index, size_t sample_count) {
    if (index < 0)
        return static_cast<size_t>(-index);
    if (static_cast<size_t>(index) >= sample_count) {
        return static_cast<size_t>(2 * static_cast<int64_t>(sample_count) - 2 - index);
    }
    return static_cast<size_t>(index);
}

// Produces a bounded window of the globally padded harmonic source input.
// `padded_begin == -kNfft/2` is the first sample seen by the STFT convolution.
// Phase and noise are keyed by the reflected absolute sample, so independently
// evaluated overlapping windows are bit-identical to make_sine_source().
std::vector<float>
make_sine_source_window(
    const std::vector<float>& f0, uint64_t seed, int64_t padded_begin, size_t count) {
    const ggml_nvtx::range nvtx_range("kokoro.vocoder.source.generate_window");
    if (f0.empty() || count == 0 || f0.size() > std::numeric_limits<size_t>::max() / kF0Upsample) {
        throw std::invalid_argument("invalid Kokoro F0 source window");
    }
    const size_t sample_count = f0.size() * kF0Upsample;
    if (padded_begin < -kNfft / 2 || padded_begin + static_cast<int64_t>(count) >
                                         static_cast<int64_t>(sample_count) + kNfft / 2) {
        throw std::out_of_range("Kokoro F0 source window exceeds reflection padding");
    }

    size_t minimum_frame = f0.size() - 1;
    size_t maximum_frame = 0;
    for (size_t offset = 0; offset < count; ++offset) {
        const size_t sample =
            reflect_sample(padded_begin + static_cast<int64_t>(offset), sample_count);
        const double source = (static_cast<double>(sample) + 0.5) / kF0Upsample - 0.5;
        const int64_t left_raw = static_cast<int64_t>(std::floor(source));
        const size_t left = static_cast<size_t>(
            std::clamp<int64_t>(left_raw, 0, static_cast<int64_t>(f0.size() - 1)));
        minimum_frame = std::min(minimum_frame, left);
        maximum_frame = std::max(maximum_frame, std::min(left + 1, f0.size() - 1));
    }

    const size_t phase_frames = maximum_frame - minimum_frame + 1;
    std::vector<float> coarse_phase(phase_frames * kHarmonics);
    for (int harmonic = 0; harmonic < kHarmonics; ++harmonic) {
        double accumulator = harmonic == 0 ? 0.0 : uniform_open(seed, harmonic);
        for (size_t frame = 0; frame <= maximum_frame; ++frame) {
            double increment =
                std::fmod(static_cast<double>(f0[frame]) * (harmonic + 1) / kSampleRate, 1.0);
            if (increment < 0.0)
                increment += 1.0;
            accumulator += increment;
            if (frame >= minimum_frame) {
                coarse_phase[(frame - minimum_frame) * kHarmonics + static_cast<size_t>(harmonic)] =
                    static_cast<float>(accumulator * 2.0 * kPi * kF0Upsample);
            }
        }
    }

    std::vector<float> result(count * kHarmonics);
    for (size_t offset = 0; offset < count; ++offset) {
        const size_t sample =
            reflect_sample(padded_begin + static_cast<int64_t>(offset), sample_count);
        const double source = (static_cast<double>(sample) + 0.5) / kF0Upsample - 0.5;
        const int64_t left_raw = static_cast<int64_t>(std::floor(source));
        const size_t left = static_cast<size_t>(
            std::clamp<int64_t>(left_raw, 0, static_cast<int64_t>(f0.size() - 1)));
        const size_t right = std::min(left + 1, f0.size() - 1);
        const float fraction = static_cast<float>(source - std::floor(source));
        const size_t f0_frame = std::min(sample / kF0Upsample, f0.size() - 1);
        const float uv = f0[f0_frame] > 10.0f ? 1.0f : 0.0f;
        const float noise_scale = uv != 0.0f ? 0.003f : 0.1f / 3.0f;
        for (int harmonic = 0; harmonic < kHarmonics; ++harmonic) {
            const float a =
                coarse_phase[(left - minimum_frame) * kHarmonics + static_cast<size_t>(harmonic)];
            const float b =
                coarse_phase[(right - minimum_frame) * kHarmonics + static_cast<size_t>(harmonic)];
            const float phase = a + fraction * (b - a);
            const uint64_t counter =
                0x100000000ULL + sample * kHarmonics + static_cast<size_t>(harmonic);
            result[offset * kHarmonics + static_cast<size_t>(harmonic)] =
                0.1f * std::sin(phase) * uv + noise_scale * normal_counter(seed, counter);
        }
    }
    return result;
}

class SourceStftModule final : public ggml_runtime::Module {
   public:
    void define_tensors(ggml_runtime::Session* session) override {
        add(session, "kokoro.decoder.generator.m_source.l_linear.weight", false);
        add(session, "kokoro.decoder.generator.m_source.l_linear.bias", false);
        add(session, "kokoro.decoder.generator.stft.forward_real", true);
        add(session, "kokoro.decoder.generator.stft.forward_imag", true);
    }

    void set_data(ggml_runtime::Session* session) override {
        for (const std::string& name : names_) session->load_weight(name);
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag inputs,
        ggml_runtime::TensorContainer* tensors) override {
        if (inputs.tensor_count() != 1 || inputs.get_tensor(0).tensor->ne[0] != kHarmonics) {
            throw std::logic_error("invalid Kokoro harmonic-source input");
        }
        const auto sine = inputs.get_tensor(0);
        ggml_context* ctx = tensors->get_ctx_of_buffer_type(sine.buft).ctx;
        ggml_tensor* source = ggml_add(
            ctx,
            kokoro_mul_mat(
                ctx, weight(session, "kokoro.decoder.generator.m_source.l_linear.weight"),
                sine.tensor),
            weight(session, "kokoro.decoder.generator.m_source.l_linear.bias"));
        source = ggml_tanh(ctx, source);
        source = ggml_cont(ctx, ggml_transpose(ctx, source));
        if (std::string_view(sine.tensor->name) != "kokoro.vocoder.sine_window") {
            source = ggml_pad_reflect_1d(ctx, source, kNfft / 2, kNfft / 2);
        }
        ggml_tensor* real = ggml_conv_1d(
            ctx, weight(session, "kokoro.decoder.generator.stft.forward_real"), source, kHop, 0, 1);
        ggml_tensor* imag = ggml_conv_1d(
            ctx, weight(session, "kokoro.decoder.generator.stft.forward_imag"), source, kHop, 0, 1);
        ggml_set_name(real, "kokoro.vocoder.harmonic_real");
        ggml_set_name(imag, "kokoro.vocoder.harmonic_imag");
        ggml_set_output(real);
        ggml_set_output(imag);
        ggml_runtime::TensorBag output;
        output.add_tensor({real, sine.buft});
        output.add_tensor({imag, sine.buft});
        return output;
    }

   private:
    void add(ggml_runtime::Session* session, const std::string& name, bool force_f16) {
        const auto shape = session->gguf_loader->get_tensor_shape(name);
        ggml_type type = force_f16 ? GGML_TYPE_F16 : session->gguf_loader->get_tensor_type(name);
        if (shape.size() == 1) {
            session->model_tensor_container->create_tensor_1d(name, type, shape[0]);
        } else if (shape.size() == 2) {
            session->model_tensor_container->create_tensor_2d(name, type, shape[0], shape[1]);
        } else if (shape.size() == 3) {
            session->model_tensor_container->create_tensor_3d(
                name, type, shape[0], shape[1], shape[2]);
        } else {
            throw std::runtime_error("unexpected Kokoro source tensor rank: " + name);
        }
        names_.push_back(name);
    }

    static ggml_tensor* weight(ggml_runtime::Session* session, const std::string& name) {
        return session->model_tensor_container->get_tensor_by_name(name).tensor;
    }

    std::vector<std::string> names_;
};

class GeneratorModule final : public ggml_runtime::Module {
   public:
    void define_tensors(ggml_runtime::Session* session) override {
        for (int stage = 0; stage < 2; ++stage) {
            add_pair("kokoro.decoder.generator.noise_convs." + std::to_string(stage));
            add_resblock_names("kokoro.decoder.generator.noise_res." + std::to_string(stage));
            add_pair("kokoro.decoder.generator.ups." + std::to_string(stage));
        }
        for (int block = 0; block < 6; ++block) {
            add_resblock_names("kokoro.decoder.generator.resblocks." + std::to_string(block));
        }
        add_pair("kokoro.decoder.generator.conv_post");

        for (const std::string& name : names_) {
            const auto shape = session->gguf_loader->get_tensor_shape(name);
            ggml_type type = session->gguf_loader->get_tensor_type(name);
            if (name.find(".weight") != std::string::npos &&
                name.find(".fc.weight") == std::string::npos) {
                type = GGML_TYPE_F16;
            }
            switch (shape.size()) {
                case 1:
                    session->model_tensor_container->create_tensor_1d(name, type, shape[0]);
                    break;
                case 2:
                    session->model_tensor_container->create_tensor_2d(
                        name, type, shape[0], shape[1]);
                    break;
                case 3:
                    session->model_tensor_container->create_tensor_3d(
                        name, type, shape[0], shape[1], shape[2]);
                    break;
                default:
                    throw std::runtime_error("unexpected Kokoro generator tensor rank: " + name);
            }
        }
    }

    void set_data(ggml_runtime::Session* session) override {
        for (const std::string& name : names_) session->load_weight(name);
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag inputs,
        ggml_runtime::TensorContainer* tensors) override {
        if (inputs.tensor_count() != 3) {
            throw std::logic_error("Kokoro generator requires latent, harmonic, and style");
        }
        const auto latent = inputs.get_tensor(0);
        const auto harmonic = inputs.get_tensor(1);
        const auto style = inputs.get_tensor(2);
        if (latent.tensor->ne[0] != 512 || harmonic.tensor->ne[1] != 22 ||
            style.tensor->ne[0] != 128) {
            throw std::logic_error("invalid Kokoro generator input shape");
        }
        ggml_context* ctx = tensors->get_ctx_of_buffer_type(latent.buft).ctx;
        ggml_tensor* x = ggml_cont(ctx, ggml_transpose(ctx, latent.tensor));
        for (int stage = 0; stage < 2; ++stage) {
            x = ggml_leaky_relu(ctx, x, 0.1f, false);
            const std::string stage_name = std::to_string(stage);
            ggml_tensor* x_source = conv(
                ctx, session, harmonic.tensor, "kokoro.decoder.generator.noise_convs." + stage_name,
                stage == 0 ? 6 : 1, stage == 0 ? 3 : 0, 1);
            x_source = adain_resblock(
                ctx, session, x_source, style.tensor,
                "kokoro.decoder.generator.noise_res." + stage_name, stage == 0 ? 7 : 11);
            x = transpose_conv(
                ctx, session, x, "kokoro.decoder.generator.ups." + stage_name, stage == 0 ? 10 : 6,
                stage == 0 ? 5 : 3);
            if (stage == 1)
                x = ggml_pad_reflect_1d(ctx, x, 1, 0);
            if (x->ne[0] != x_source->ne[0] || x->ne[1] != x_source->ne[1]) {
                throw std::logic_error("Kokoro generator source alignment mismatch");
            }
            x = ggml_add(ctx, x, x_source);
            ggml_tensor* sum = nullptr;
            for (int kernel_index = 0; kernel_index < 3; ++kernel_index) {
                const int block = stage * 3 + kernel_index;
                const int kernel = kernel_index == 0 ? 3 : (kernel_index == 1 ? 7 : 11);
                ggml_tensor* branch = adain_resblock(
                    ctx, session, x, style.tensor,
                    "kokoro.decoder.generator.resblocks." + std::to_string(block), kernel);
                sum = sum ? ggml_add(ctx, sum, branch) : branch;
            }
            x = ggml_scale(ctx, sum, 1.0f / 3.0f);
        }
        x = ggml_leaky_relu(ctx, x, 0.01f, false);
        x = conv(ctx, session, x, "kokoro.decoder.generator.conv_post", 1, 3, 1);
        ggml_tensor* channels_first = ggml_cont(ctx, ggml_transpose(ctx, x));
        ggml_tensor* magnitude = ggml_exp(
            ctx, ggml_cont(
                     ctx, ggml_view_2d(
                              ctx, channels_first, kBins, channels_first->ne[1],
                              channels_first->nb[1], 0)));
        ggml_tensor* phase = ggml_sin(
            ctx,
            ggml_cont(
                ctx, ggml_view_2d(
                         ctx, channels_first, kBins, channels_first->ne[1], channels_first->nb[1],
                         static_cast<size_t>(kBins) * channels_first->nb[0])));
        ggml_set_name(magnitude, "kokoro.vocoder.magnitude");
        ggml_set_name(phase, "kokoro.vocoder.phase");
        ggml_set_output(magnitude);
        ggml_set_output(phase);
        ggml_runtime::TensorBag output;
        output.add_tensor({magnitude, latent.buft});
        output.add_tensor({phase, latent.buft});
        return output;
    }

   private:
    void add_pair(const std::string& prefix) {
        names_.push_back(prefix + ".weight");
        names_.push_back(prefix + ".bias");
    }

    void add_resblock_names(const std::string& prefix) {
        for (int layer = 0; layer < 3; ++layer) {
            for (const std::string convs : {"convs1", "convs2"}) {
                add_pair(prefix + "." + convs + "." + std::to_string(layer));
            }
            for (const std::string adain : {"adain1", "adain2"}) {
                add_pair(prefix + "." + adain + "." + std::to_string(layer) + ".fc");
            }
            names_.push_back(prefix + ".alpha1." + std::to_string(layer));
            names_.push_back(prefix + ".alpha2." + std::to_string(layer));
        }
    }

    static ggml_tensor* weight(ggml_runtime::Session* session, const std::string& name) {
        return session->model_tensor_container->get_tensor_by_name(name).tensor;
    }

    static ggml_tensor* linear(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix) {
        return ggml_add(
            ctx, kokoro_mul_mat(ctx, weight(session, prefix + ".weight"), input),
            weight(session, prefix + ".bias"));
    }

    static ggml_tensor* conv(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix, int stride, int padding, int dilation) {
        ggml_tensor* output = ggml_conv_1d(
            ctx, weight(session, prefix + ".weight"), input, stride, padding, dilation);
        return ggml_add(
            ctx, output, ggml_reshape_2d(ctx, weight(session, prefix + ".bias"), 1, output->ne[1]));
    }

    static ggml_tensor* transpose_conv(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix, int stride, int padding) {
        ggml_tensor* full =
            ggml_conv_transpose_1d(ctx, weight(session, prefix + ".weight"), input, stride, 0, 1);
        const int64_t length = input->ne[0] * stride;
        ggml_tensor* output = ggml_cont(
            ctx, ggml_view_2d(
                     ctx, full, length, full->ne[1], full->nb[1],
                     static_cast<size_t>(padding) * full->nb[0]));
        return ggml_add(
            ctx, output, ggml_reshape_2d(ctx, weight(session, prefix + ".bias"), 1, output->ne[1]));
    }

    static ggml_tensor* adain(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input, ggml_tensor* style,
        const std::string& prefix) {
        ggml_tensor* affine = linear(ctx, session, style, prefix + ".fc");
        const int64_t channels = input->ne[1];
        ggml_tensor* gamma = ggml_cont(ctx, ggml_view_1d(ctx, affine, channels, 0));
        ggml_tensor* beta = ggml_cont(
            ctx,
            ggml_view_1d(ctx, affine, channels, static_cast<size_t>(channels) * affine->nb[0]));
        gamma = ggml_reshape_2d(ctx, gamma, 1, channels);
        beta = ggml_reshape_2d(ctx, beta, 1, channels);
        ggml_tensor* normalized = ggml_norm(ctx, input, 1.0e-5f);
        return ggml_add(
            ctx,
            ggml_add(
                ctx, normalized, ggml_mul(ctx, normalized, ggml_repeat(ctx, gamma, normalized))),
            ggml_repeat(ctx, beta, normalized));
    }

    static ggml_tensor* snake(ggml_context* ctx, ggml_tensor* input, ggml_tensor* alpha) {
        alpha = ggml_reshape_2d(ctx, alpha, 1, input->ne[1]);
        ggml_tensor* repeated = ggml_repeat(ctx, alpha, input);
        ggml_tensor* sine = ggml_sin(ctx, ggml_mul(ctx, repeated, input));
        return ggml_add(ctx, input, ggml_div(ctx, ggml_sqr(ctx, sine), repeated));
    }

    static ggml_tensor* adain_resblock(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input, ggml_tensor* style,
        const std::string& prefix, int kernel) {
        ggml_tensor* output = input;
        for (int layer = 0; layer < 3; ++layer) {
            ggml_tensor* branch =
                adain(ctx, session, output, style, prefix + ".adain1." + std::to_string(layer));
            branch =
                snake(ctx, branch, weight(session, prefix + ".alpha1." + std::to_string(layer)));
            const int dilation = layer == 0 ? 1 : (layer == 1 ? 3 : 5);
            branch = conv(
                ctx, session, branch, prefix + ".convs1." + std::to_string(layer), 1,
                (kernel * dilation - dilation) / 2, dilation);
            branch =
                adain(ctx, session, branch, style, prefix + ".adain2." + std::to_string(layer));
            branch =
                snake(ctx, branch, weight(session, prefix + ".alpha2." + std::to_string(layer)));
            branch = conv(
                ctx, session, branch, prefix + ".convs2." + std::to_string(layer), 1,
                (kernel - 1) / 2, 1);
            output = ggml_add(ctx, output, branch);
        }
        return output;
    }

    std::vector<std::string> names_;
};

std::vector<float>
inverse_stft(const std::vector<float>& magnitude, const std::vector<float>& phase, size_t frames) {
    const ggml_nvtx::range nvtx_range("kokoro.vocoder.istft.offline");
    if (magnitude.size() != frames * kBins || phase.size() != magnitude.size() || frames < 1) {
        throw std::invalid_argument("invalid Kokoro inverse-STFT shape");
    }
    const size_t full_length = (frames - 1) * kHop + kNfft;
    std::vector<double> waveform(full_length, 0.0);
    std::vector<double> envelope(full_length, 0.0);
    std::vector<double> window(kNfft);
    for (int sample = 0; sample < kNfft; ++sample) {
        window[static_cast<size_t>(sample)] =
            0.5 - 0.5 * std::cos(2.0 * static_cast<double>(kPi) * sample / kNfft);
    }
    for (size_t frame = 0; frame < frames; ++frame) {
        for (int sample = 0; sample < kNfft; ++sample) {
            double value = 0.0;
            for (int bin = 0; bin < kBins; ++bin) {
                const size_t index = frame * kBins + static_cast<size_t>(bin);
                const double real = magnitude[index] * std::cos(phase[index]);
                const double imag = magnitude[index] * std::sin(phase[index]);
                const double angle = 2.0 * static_cast<double>(kPi) * bin * sample / kNfft;
                const double scale = (bin == 0 || bin == kNfft / 2) ? 1.0 / kNfft : 2.0 / kNfft;
                value += scale * (real * std::cos(angle) - imag * std::sin(angle));
            }
            const size_t output = frame * kHop + static_cast<size_t>(sample);
            waveform[output] += value * window[static_cast<size_t>(sample)];
            envelope[output] +=
                window[static_cast<size_t>(sample)] * window[static_cast<size_t>(sample)];
        }
    }
    const size_t trim = kNfft / 2;
    if (full_length <= 2 * trim)
        return {};
    std::vector<float> result(full_length - 2 * trim);
    for (size_t index = trim; index < full_length - trim; ++index) {
        if (envelope[index] < 1.0e-11) {
            throw std::runtime_error("Kokoro inverse STFT violates the Hann overlap condition");
        }
        result[index - trim] = static_cast<float>(waveform[index] / envelope[index]);
    }
    return result;
}

// A generator tile contains complete consecutive spectral frames. This state
// carries overlap-add and window normalization across tile boundaries, so a
// tile boundary is numerically invisible and the tail is flushed exactly once.
class StreamingIstft {
   public:
    StreamingIstft(
        size_t total_frames, size_t callback_samples, const KokoroVocoder::FloatCallback& callback)
        : total_frames_(total_frames), callback_samples_(callback_samples), callback_(callback) {
        if (total_frames_ < 1 || callback_samples_ == 0) {
            throw std::invalid_argument("invalid streaming Kokoro inverse-STFT shape");
        }
        const size_t full_length = (total_frames_ - 1) * kHop + kNfft;
        output_end_ = full_length - kNfft / 2;
        expected_samples_ = full_length - kNfft;
        pcm_tile_.reserve(callback_samples_);
        for (int sample = 0; sample < kNfft; ++sample) {
            window_[static_cast<size_t>(sample)] =
                0.5 - 0.5 * std::cos(2.0 * static_cast<double>(kPi) * sample / kNfft);
        }
    }

    bool consume(
        const std::vector<float>& magnitude, const std::vector<float>& phase, size_t frames) {
        const ggml_nvtx::range nvtx_range("kokoro.vocoder.istft.consume");
        if (stats_.cancelled)
            return false;
        if (frames == 0 || magnitude.size() != frames * kBins || phase.size() != magnitude.size() ||
            frames > total_frames_ - frames_seen_) {
            throw std::invalid_argument("invalid Kokoro spectral tile");
        }
        for (size_t local_frame = 0; local_frame < frames; ++local_frame) {
            const size_t frame = frames_seen_++;
            for (int sample = 0; sample < kNfft; ++sample) {
                double value = 0.0;
                for (int bin = 0; bin < kBins; ++bin) {
                    const size_t index = local_frame * kBins + static_cast<size_t>(bin);
                    const double real = magnitude[index] * std::cos(phase[index]);
                    const double imag = magnitude[index] * std::sin(phase[index]);
                    const double angle = 2.0 * static_cast<double>(kPi) * bin * sample / kNfft;
                    const double scale = (bin == 0 || bin == kNfft / 2) ? 1.0 / kNfft : 2.0 / kNfft;
                    value += scale * (real * std::cos(angle) - imag * std::sin(angle));
                }
                const size_t absolute = frame * kHop + static_cast<size_t>(sample);
                const size_t ring = absolute % kRingSize;
                waveform_[ring] += value * window_[static_cast<size_t>(sample)];
                envelope_[ring] +=
                    window_[static_cast<size_t>(sample)] * window_[static_cast<size_t>(sample)];
            }

            const size_t finalized = frames_seen_ == total_frames_
                                         ? (total_frames_ - 1) * kHop + kNfft
                                         : frames_seen_ * static_cast<size_t>(kHop);
            while (cleanup_ < finalized) {
                const size_t ring = cleanup_ % kRingSize;
                if (cleanup_ >= kNfft / 2 && cleanup_ < output_end_) {
                    if (envelope_[ring] < 1.0e-11) {
                        throw std::runtime_error(
                            "Kokoro streaming inverse STFT violates the Hann overlap condition");
                    }
                    pcm_tile_.push_back(static_cast<float>(waveform_[ring] / envelope_[ring]));
                    if (pcm_tile_.size() == callback_samples_ && !flush())
                        return false;
                }
                waveform_[ring] = 0.0;
                envelope_[ring] = 0.0;
                ++cleanup_;
            }
        }
        return true;
    }

    KokoroVocoderStreamStats finish() {
        const ggml_nvtx::range nvtx_range("kokoro.vocoder.istft.finish");
        if (stats_.cancelled)
            return stats_;
        if (frames_seen_ != total_frames_) {
            throw std::runtime_error("Kokoro streaming inverse STFT is missing spectral frames");
        }
        if (!flush())
            return stats_;
        if (stats_.samples_written != expected_samples_) {
            throw std::runtime_error("Kokoro streaming inverse STFT sample count mismatch");
        }
        return stats_;
    }

    KokoroVocoderStreamStats& stats() { return stats_; }

   private:
    bool flush() {
        const ggml_nvtx::range nvtx_range("kokoro.vocoder.istft.flush_callback");
        if (pcm_tile_.empty())
            return true;
        stats_.samples_written += pcm_tile_.size();
        const bool keep_going = !callback_ || callback_(pcm_tile_);
        pcm_tile_.clear();
        if (!keep_going)
            stats_.cancelled = true;
        return keep_going;
    }

    static constexpr size_t kRingSize = 4 * kNfft;
    size_t total_frames_ = 0;
    size_t callback_samples_ = 0;
    KokoroVocoder::FloatCallback callback_;
    std::array<double, kRingSize> waveform_{};
    std::array<double, kRingSize> envelope_{};
    std::array<double, kNfft> window_{};
    std::vector<float> pcm_tile_;
    size_t frames_seen_ = 0;
    size_t cleanup_ = 0;
    size_t output_end_ = 0;
    size_t expected_samples_ = 0;
    KokoroVocoderStreamStats stats_;
};

}  // namespace

class KokoroVocoder::Impl {
   public:
    static constexpr size_t kGeneratorTileFrames = 40;
    // The true convolutional receptive field is substantially smaller; the
    // wider context also makes time-global AdaIN statistics identical to the
    // whole graph for ordinary chunks and stable for unusually long ones.
    static constexpr size_t kGeneratorHaloFrames = 512;
    static constexpr size_t kMaximumGeneratorWindow =
        kGeneratorTileFrames + 2 * kGeneratorHaloFrames;

    struct Spectra {
        std::vector<float> magnitude;
        std::vector<float> phase;
        size_t frames = 0;
        size_t sample_count = 0;
    };

    Impl(const std::string& model_path, bool use_gpu)
        : source_loader(model_path), source_metadata(source_loader),
          source_backend({use_gpu, 0, nullptr}),
          source_session(source_backend, &source_module, &source_loader),
          generator_loader(model_path), generator_metadata(generator_loader),
          generator_backend({use_gpu, 0, nullptr}),
          generator_session(generator_backend, &generator_module, &generator_loader) {
        source_session.setup();
        generator_session.setup();
        // Boundary tiles have different shapes. Retaining only the most recent
        // graph prevents the cache itself from making scratch grow with the
        // number of tiles in an utterance.
        source_session.set_run_cache_capacity(1);
        generator_session.set_run_cache_capacity(1);
    }

    struct Harmonic {
        std::vector<float> sine;
        std::vector<float> values;
        size_t frames = 0;
        size_t sample_count = 0;
    };

    Harmonic compute_harmonic(const std::vector<float>& f0, uint64_t seed) {
        const ggml_nvtx::range nvtx_range("kokoro.vocoder.harmonic.full");
        Harmonic result;
        result.sine = make_sine_source(f0, seed);
        result.sample_count = f0.size() * kF0Upsample;
        result.frames = result.sample_count / kHop + 1;
        const size_t harmonic_frames = result.frames;
        std::vector<float> real(harmonic_frames * kBins);
        std::vector<float> imag(harmonic_frames * kBins);
        std::vector<ggml_runtime::Session::Output> stft_outputs = {
            {0, "", real.data(), real.size() * sizeof(float)},
            {1, "", imag.data(), imag.size() * sizeof(float)}};
        {
            const ggml_nvtx::range session_range("kokoro.vocoder.source_stft.session");
            source_session.run(
                {{"kokoro.vocoder.sine",
                  GGML_TYPE_F32,
                  result.sine.data(),
                  {kHarmonics, static_cast<int64_t>(result.sample_count)}}},
                stft_outputs);
        }

        result.values.resize(harmonic_frames * 22);
        for (size_t frame = 0; frame < harmonic_frames; ++frame) {
            for (int bin = 0; bin < kBins; ++bin) {
                const size_t source = static_cast<size_t>(bin) * harmonic_frames + frame;
                const float re = real[source];
                const float im = imag[source];
                result.values[static_cast<size_t>(bin) * harmonic_frames + frame] =
                    std::sqrt(re * re + im * im);
                result.values[(kBins + static_cast<size_t>(bin)) * harmonic_frames + frame] =
                    std::atan2(im, re);
            }
        }
        return result;
    }

    std::vector<float> compute_harmonic_range(
        const std::vector<float>& f0, uint64_t seed, size_t frame_begin, size_t frame_count) {
        const std::string nvtx_name =
            "kokoro.vocoder.harmonic.range frame_begin=" + std::to_string(frame_begin) +
            " frame_count=" + std::to_string(frame_count);
        const ggml_nvtx::range nvtx_range(nvtx_name.c_str());
        const size_t total_frames = f0.size() * (kF0Upsample / kHop) + 1;
        if (frame_count == 0 || frame_begin > total_frames ||
            frame_count > total_frames - frame_begin) {
            throw std::out_of_range("invalid Kokoro harmonic frame range");
        }
        const size_t source_samples = (frame_count - 1) * kHop + kNfft;
        const int64_t padded_begin = static_cast<int64_t>(frame_begin * kHop) - kNfft / 2;
        const std::vector<float> sine =
            make_sine_source_window(f0, seed, padded_begin, source_samples);
        std::vector<float> real(frame_count * kBins);
        std::vector<float> imag(frame_count * kBins);
        std::vector<ggml_runtime::Session::Output> outputs = {
            {0, "", real.data(), real.size() * sizeof(float)},
            {1, "", imag.data(), imag.size() * sizeof(float)}};
        {
            const ggml_nvtx::range session_range("kokoro.vocoder.source_stft.session");
            source_session.run(
                {{"kokoro.vocoder.sine_window",
                  GGML_TYPE_F32,
                  sine.data(),
                  {kHarmonics, static_cast<int64_t>(source_samples)}}},
                outputs);
        }

        std::vector<float> harmonic(frame_count * 22);
        for (size_t frame = 0; frame < frame_count; ++frame) {
            for (int bin = 0; bin < kBins; ++bin) {
                const size_t source = static_cast<size_t>(bin) * frame_count + frame;
                const float re = real[source];
                const float im = imag[source];
                harmonic[static_cast<size_t>(bin) * frame_count + frame] =
                    std::sqrt(re * re + im * im);
                harmonic[(kBins + static_cast<size_t>(bin)) * frame_count + frame] =
                    std::atan2(im, re);
            }
        }
        return harmonic;
    }

    Spectra run_generator(
        const std::vector<float>& latent, size_t latent_frames, const std::vector<float>& harmonic,
        size_t harmonic_frames, const std::vector<float>& decoder_style) {
        const std::string nvtx_name =
            "kokoro.vocoder.generator.run latent_frames=" + std::to_string(latent_frames) +
            " harmonic_frames=" + std::to_string(harmonic_frames);
        const ggml_nvtx::range nvtx_range(nvtx_name.c_str());
        if (latent.size() != latent_frames * 512 || harmonic.size() != harmonic_frames * 22 ||
            harmonic_frames != latent_frames * 60 + 1) {
            throw std::invalid_argument("invalid Kokoro generator tile shape");
        }
        Spectra result;
        result.frames = harmonic_frames;
        result.sample_count = latent_frames * kF0Upsample;
        result.magnitude.resize(harmonic_frames * kBins);
        result.phase.resize(harmonic_frames * kBins);
        std::vector<ggml_runtime::Session::Output> generator_outputs = {
            {0, "", result.magnitude.data(), result.magnitude.size() * sizeof(float)},
            {1, "", result.phase.data(), result.phase.size() * sizeof(float)}};
        generator_session.run(
            {{"kokoro.vocoder.latent",
              GGML_TYPE_F32,
              latent.data(),
              {512, static_cast<int64_t>(latent_frames)}},
             {"kokoro.vocoder.harmonic",
              GGML_TYPE_F32,
              harmonic.data(),
              {static_cast<int64_t>(harmonic_frames), 22}},
             {"kokoro.vocoder.style", GGML_TYPE_F32, decoder_style.data(), {128}}},
            generator_outputs);
        return result;
    }

    Spectra compute_spectra(
        const std::vector<float>& latent, const std::vector<float>& f0,
        const std::vector<float>& decoder_style, uint64_t seed, KokoroVocoderDebug* debug) {
        const ggml_nvtx::range nvtx_range("kokoro.vocoder.compute_spectra");
        Harmonic harmonic = compute_harmonic(f0, seed);
        Spectra result =
            run_generator(latent, f0.size(), harmonic.values, harmonic.frames, decoder_style);
        if (debug) {
            debug->sine = harmonic.sine;
            debug->harmonic = harmonic.values;
            debug->magnitude = result.magnitude;
            debug->phase = result.phase;
        }
        return result;
    }

    KokoroVocoderStreamStats compute_stream(
        const KokoroVocoder::LatentProvider& latent_provider, size_t latent_frames,
        const std::vector<float>& f0, const std::vector<float>& decoder_style, uint64_t seed,
        size_t callback_samples, const KokoroVocoder::FloatCallback& callback) {
        const std::string stream_name =
            "kokoro.vocoder.compute_stream latent_frames=" + std::to_string(latent_frames) +
            " callback_samples=" + std::to_string(callback_samples) +
            " seed=" + std::to_string(seed);
        const ggml_nvtx::range stream_range(stream_name.c_str());
        const size_t total_harmonic_frames = f0.size() * (kF0Upsample / kHop) + 1;
        StreamingIstft istft(total_harmonic_frames, callback_samples, callback);

        for (size_t begin = 0; begin < latent_frames; begin += kGeneratorTileFrames) {
            const size_t end = std::min(begin + kGeneratorTileFrames, latent_frames);
            // A bounded leaf fits in one exact normalization window. Re-run
            // that window for each output tile so AdaIN continues to see the
            // complete time axis without retaining its activation graph.
            const bool exact_window = latent_frames <= kMaximumGeneratorWindow;
            const size_t input_begin =
                exact_window ? 0
                             : (begin > kGeneratorHaloFrames ? begin - kGeneratorHaloFrames : 0);
            const size_t input_end =
                exact_window ? latent_frames : std::min(latent_frames, end + kGeneratorHaloFrames);
            const size_t input_frames = input_end - input_begin;
            const std::string tile_name =
                "kokoro.vocoder.tile output=" + std::to_string(begin) + ":" + std::to_string(end) +
                " input=" + std::to_string(input_begin) + ":" + std::to_string(input_end) +
                " exact_window=" + std::string(exact_window ? "true" : "false");
            const ggml_nvtx::range tile_range(tile_name.c_str());
            std::vector<float> latent;
            {
                const ggml_nvtx::range provider_range(
                    "kokoro.vocoder.tile.decoder_latent_provider");
                latent = latent_provider(input_begin, input_end);
            }
            if (latent.size() != input_frames * 512) {
                throw std::runtime_error("Kokoro latent provider returned an unexpected shape");
            }

            const size_t harmonic_begin = input_begin * 60;
            const size_t local_harmonic_frames = input_frames * 60 + 1;
            std::vector<float> local_harmonic =
                compute_harmonic_range(f0, seed, harmonic_begin, local_harmonic_frames);
            Spectra local = run_generator(
                latent, input_frames, local_harmonic, local_harmonic_frames, decoder_style);

            const size_t crop_begin = (begin - input_begin) * 60;
            const size_t crop_frames = (end - begin) * 60 + (end == latent_frames ? 1 : 0);
            std::vector<float> magnitude(crop_frames * kBins);
            std::vector<float> phase(crop_frames * kBins);
            {
                const ggml_nvtx::range crop_range("kokoro.vocoder.tile.crop_spectra");
                std::copy_n(
                    local.magnitude.data() + crop_begin * kBins, magnitude.size(),
                    magnitude.data());
                std::copy_n(local.phase.data() + crop_begin * kBins, phase.size(), phase.data());
            }

            auto& stats = istft.stats();
            ++stats.generator_tiles;
            stats.max_generator_input_frames =
                std::max(stats.max_generator_input_frames, input_frames);
            if (!istft.consume(magnitude, phase, crop_frames))
                return stats;
        }
        return istft.finish();
    }

    ggml_runtime::GGUFLoader source_loader;
    KokoroModelMetadata source_metadata;
    SourceStftModule source_module;
    ggml_runtime::BackendManager source_backend;
    ggml_runtime::Session source_session;
    ggml_runtime::GGUFLoader generator_loader;
    KokoroModelMetadata generator_metadata;
    GeneratorModule generator_module;
    ggml_runtime::BackendManager generator_backend;
    ggml_runtime::Session generator_session;
};

KokoroVocoder::KokoroVocoder(const std::string& model_path, bool use_gpu)
    : impl_(std::make_unique<Impl>(model_path, use_gpu)) {}

KokoroVocoder::~KokoroVocoder() = default;

std::vector<float>
KokoroVocoder::synthesize(
    const std::vector<float>& latent, const std::vector<float>& f0,
    const std::vector<float>& decoder_style, int64_t seed, uint64_t* selected_seed,
    KokoroVocoderDebug* debug) {
    const std::string nvtx_name = "kokoro.vocoder.synthesize frames=" + std::to_string(f0.size()) +
                                  " requested_seed=" + std::to_string(seed);
    const ggml_nvtx::range nvtx_range(nvtx_name.c_str());
    if (f0.empty() || f0.size() > 2400 || latent.size() != f0.size() * 512) {
        throw std::invalid_argument("invalid Kokoro vocoder input shape");
    }
    if (decoder_style.size() != 128) {
        throw std::invalid_argument("Kokoro vocoder style must contain 128 values");
    }
    uint64_t actual_seed = 0;
    if (seed < 0) {
        std::random_device device;
        actual_seed =
            (static_cast<uint64_t>(device()) << 32) ^ static_cast<uint64_t>(device()) ^
            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    } else {
        actual_seed = static_cast<uint64_t>(seed);
    }
    if (selected_seed)
        *selected_seed = actual_seed;

    Impl::Spectra spectra = impl_->compute_spectra(latent, f0, decoder_style, actual_seed, debug);
    std::vector<float> waveform = inverse_stft(spectra.magnitude, spectra.phase, spectra.frames);
    if (waveform.size() != spectra.sample_count) {
        throw std::runtime_error("Kokoro vocoder returned an unexpected waveform length");
    }
    for (float value : waveform) {
        if (!std::isfinite(value))
            throw std::runtime_error("Kokoro waveform is not finite");
    }
    return waveform;
}

KokoroVocoderStreamStats
KokoroVocoder::synthesize_stream(
    const std::vector<float>& latent, const std::vector<float>& f0,
    const std::vector<float>& decoder_style, int64_t seed, size_t callback_samples,
    const FloatCallback& callback, uint64_t* selected_seed) {
    const ggml_nvtx::range nvtx_range("kokoro.vocoder.synthesize_stream.materialized");
    if (f0.empty() || f0.size() > 2400 || latent.size() != f0.size() * 512) {
        throw std::invalid_argument("invalid Kokoro vocoder input shape");
    }
    const LatentProvider provider = [&](size_t begin, size_t end) {
        if (begin >= end || end > f0.size()) {
            throw std::out_of_range("invalid Kokoro latent tile request");
        }
        return std::vector<float>(
            latent.begin() + static_cast<std::ptrdiff_t>(begin * 512),
            latent.begin() + static_cast<std::ptrdiff_t>(end * 512));
    };
    return synthesize_stream(
        provider, f0.size(), f0, decoder_style, seed, callback_samples, callback, selected_seed);
}

KokoroVocoderStreamStats
KokoroVocoder::synthesize_stream(
    const LatentProvider& latent_provider, size_t latent_frames, const std::vector<float>& f0,
    const std::vector<float>& decoder_style, int64_t seed, size_t callback_samples,
    const FloatCallback& callback, uint64_t* selected_seed) {
    const ggml_nvtx::range nvtx_range("kokoro.vocoder.synthesize_stream.pull");
    if (!latent_provider || f0.empty() || f0.size() > 2400 || latent_frames != f0.size() ||
        callback_samples == 0) {
        throw std::invalid_argument("invalid streaming Kokoro vocoder input shape");
    }
    if (decoder_style.size() != 128) {
        throw std::invalid_argument("Kokoro vocoder style must contain 128 values");
    }
    uint64_t actual_seed = 0;
    if (seed < 0) {
        std::random_device device;
        actual_seed =
            (static_cast<uint64_t>(device()) << 32) ^ static_cast<uint64_t>(device()) ^
            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    } else {
        actual_seed = static_cast<uint64_t>(seed);
    }
    if (selected_seed)
        *selected_seed = actual_seed;
    const std::string nvtx_parameters =
        "kokoro.parameters.vocoder latent_frames=" + std::to_string(latent_frames) +
        " callback_samples=" + std::to_string(callback_samples) +
        " requested_seed=" + std::to_string(seed) + " actual_seed=" + std::to_string(actual_seed);
    ggml_nvtx::mark(nvtx_parameters.c_str());
    KokoroVocoderStreamStats stats = impl_->compute_stream(
        latent_provider, latent_frames, f0, decoder_style, actual_seed, callback_samples, callback);
    const size_t sample_count = f0.size() * kF0Upsample;
    if (!stats.cancelled && stats.samples_written != sample_count) {
        throw std::runtime_error("Kokoro streaming vocoder returned an unexpected waveform length");
    }
    return stats;
}

}  // namespace nemo_speech::tts::kokoro
