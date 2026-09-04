// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "decoder.h"

#include <ggml.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ggml_ops.h"
#include "model.h"
#include "nvtx_utils.h"
#include "runtime.h"

namespace nemo_speech::tts::kokoro {
namespace {

class DecoderModule final : public ggml_runtime::Module {
   public:
    void define_tensors(ggml_runtime::Session* session) override {
        for (const std::string prefix : {"kokoro.decoder.F0_conv", "kokoro.decoder.N_conv"}) {
            names_.push_back(prefix + ".weight");
            names_.push_back(prefix + ".bias");
        }
        names_.push_back("kokoro.decoder.asr_res.0.weight");
        names_.push_back("kokoro.decoder.asr_res.0.bias");
        add_block_names("kokoro.decoder.encode", false);
        for (int block = 0; block < 4; ++block) {
            add_block_names("kokoro.decoder.decode." + std::to_string(block), block == 3);
        }

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
                    throw std::runtime_error("unexpected Kokoro decoder tensor rank: " + name);
            }
        }
    }

    void set_data(ggml_runtime::Session* session) override {
        for (const std::string& name : names_) session->load_weight(name);
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag inputs,
        ggml_runtime::TensorContainer* tensors) override {
        if (inputs.tensor_count() != 4) {
            throw std::logic_error("Kokoro decoder graph requires text, F0, noise, and style");
        }
        const auto text = inputs.get_tensor(0);
        const auto f0_curve = inputs.get_tensor(1);
        const auto noise_curve = inputs.get_tensor(2);
        const auto style = inputs.get_tensor(3);
        if (text.tensor->ne[0] != 512 || f0_curve.tensor->ne[0] != 2 * text.tensor->ne[1] ||
            noise_curve.tensor->ne[0] != f0_curve.tensor->ne[0] || style.tensor->ne[0] != 128) {
            throw std::logic_error("invalid Kokoro decoder input shape");
        }
        ggml_context* ctx = tensors->get_ctx_of_buffer_type(text.buft).ctx;
        ggml_tensor* convolution_text = ggml_cont(ctx, ggml_transpose(ctx, text.tensor));
        ggml_tensor* f0 = downsample_curve(ctx, session, f0_curve.tensor, "kokoro.decoder.F0_conv");
        ggml_tensor* noise =
            downsample_curve(ctx, session, noise_curve.tensor, "kokoro.decoder.N_conv");
        ggml_tensor* combined = ggml_concat(ctx, convolution_text, f0, 1);
        combined = ggml_concat(ctx, combined, noise, 1);
        ggml_tensor* output =
            residual_block(ctx, session, combined, style.tensor, "kokoro.decoder.encode", false);
        ggml_tensor* asr_res =
            pointwise(ctx, session, convolution_text, "kokoro.decoder.asr_res.0", true);
        for (int block = 0; block < 4; ++block) {
            output = ggml_concat(ctx, output, asr_res, 1);
            output = ggml_concat(ctx, output, f0, 1);
            output = ggml_concat(ctx, output, noise, 1);
            output = residual_block(
                ctx, session, output, style.tensor,
                "kokoro.decoder.decode." + std::to_string(block), block == 3);
        }
        // Restore recurrent/linear convention [channels,time].
        output = ggml_cont(ctx, ggml_transpose(ctx, output));
        ggml_set_name(output, "kokoro.decoder.generator_input");
        ggml_set_output(output);
        ggml_runtime::TensorBag outputs;
        outputs.add_tensor({output, text.buft});
        return outputs;
    }

   private:
    void add_block_names(const std::string& prefix, bool upsample) {
        for (const std::string conv : {"conv1", "conv2"}) {
            names_.push_back(prefix + "." + conv + ".weight");
            names_.push_back(prefix + "." + conv + ".bias");
        }
        for (const std::string norm : {"norm1", "norm2"}) {
            names_.push_back(prefix + "." + norm + ".fc.weight");
            names_.push_back(prefix + "." + norm + ".fc.bias");
        }
        names_.push_back(prefix + ".conv1x1.weight");
        if (upsample) {
            names_.push_back(prefix + ".pool.weight");
            names_.push_back(prefix + ".pool.bias");
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

    static ggml_tensor* pointwise(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix, bool bias) {
        ggml_tensor* channels_first = ggml_cont(ctx, ggml_transpose(ctx, input));
        ggml_tensor* matrix = weight(session, prefix + ".weight");
        if (matrix->ne[0] != 1) {
            throw std::logic_error("invalid Kokoro pointwise convolution shape");
        }
        matrix = ggml_reshape_2d(ctx, matrix, matrix->ne[1], matrix->ne[2]);
        ggml_tensor* output = kokoro_mul_mat(ctx, matrix, channels_first);
        if (bias)
            output = ggml_add(ctx, output, weight(session, prefix + ".bias"));
        return ggml_cont(ctx, ggml_transpose(ctx, output));
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

    static ggml_tensor* conv(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix, int stride = 1, int padding = 1, int dilation = 1) {
        ggml_tensor* output = ggml_conv_1d(
            ctx, weight(session, prefix + ".weight"), input, stride, padding, dilation);
        return ggml_add(
            ctx, output, ggml_reshape_2d(ctx, weight(session, prefix + ".bias"), 1, output->ne[1]));
    }

    static ggml_tensor* downsample_curve(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* curve,
        const std::string& prefix) {
        ggml_tensor* input = ggml_reshape_2d(ctx, curve, curve->ne[0], 1);
        return conv(ctx, session, input, prefix, 2, 1, 1);
    }

    static ggml_tensor* depthwise_transpose(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix) {
        ggml_tensor* kernel = weight(session, prefix + ".weight");
        const int64_t time = input->ne[0];
        const int64_t channels = input->ne[1];
        ggml_tensor* output = nullptr;
        for (int64_t channel = 0; channel < channels; ++channel) {
            ggml_tensor* kernel_channel = ggml_cont(
                ctx, ggml_view_3d(
                         ctx, kernel, 3, 1, 1, kernel->nb[1], kernel->nb[2],
                         static_cast<size_t>(channel) * kernel->nb[2]));
            ggml_tensor* input_channel = ggml_cont(
                ctx, ggml_view_2d(
                         ctx, input, time, 1, input->nb[1],
                         static_cast<size_t>(channel) * input->nb[1]));
            ggml_tensor* full = ggml_conv_transpose_1d(ctx, kernel_channel, input_channel, 2, 0, 1);
            ggml_tensor* cropped =
                ggml_cont(ctx, ggml_view_2d(ctx, full, 2 * time, 1, full->nb[1], full->nb[0]));
            output = output ? ggml_concat(ctx, output, cropped, 1) : cropped;
        }
        return ggml_add(
            ctx, output, ggml_reshape_2d(ctx, weight(session, prefix + ".bias"), 1, channels));
    }

    static ggml_tensor* residual_block(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input, ggml_tensor* style,
        const std::string& prefix, bool upsample) {
        ggml_tensor* shortcut = input;
        if (upsample) {
            shortcut = ggml_interpolate(
                ctx, shortcut, shortcut->ne[0] * 2, shortcut->ne[1], 1, 1, GGML_SCALE_MODE_NEAREST);
        }
        shortcut = pointwise(ctx, session, shortcut, prefix + ".conv1x1", false);

        ggml_tensor* residual = adain(ctx, session, input, style, prefix + ".norm1");
        residual = ggml_leaky_relu(ctx, residual, 0.2f, false);
        if (upsample) {
            residual = depthwise_transpose(ctx, session, residual, prefix + ".pool");
        }
        residual = conv(ctx, session, residual, prefix + ".conv1");
        residual = adain(ctx, session, residual, style, prefix + ".norm2");
        residual = ggml_leaky_relu(ctx, residual, 0.2f, false);
        residual = conv(ctx, session, residual, prefix + ".conv2");
        return ggml_scale(ctx, ggml_add(ctx, residual, shortcut), 0.7071067811865475f);
    }

    std::vector<std::string> names_;
};

}  // namespace

class KokoroDecoderEncoder::Impl {
   public:
    Impl(const std::string& model_path, bool use_gpu)
        : loader(model_path), metadata(loader), backend({use_gpu, 0, nullptr}),
          session(backend, &module, &loader) {
        session.setup();
        session.set_run_cache_capacity(1);
    }
    ggml_runtime::GGUFLoader loader;
    KokoroModelMetadata metadata;
    DecoderModule module;
    ggml_runtime::BackendManager backend;
    ggml_runtime::Session session;
};

KokoroDecoderEncoder::KokoroDecoderEncoder(const std::string& model_path, bool use_gpu)
    : impl_(std::make_unique<Impl>(model_path, use_gpu)) {}

KokoroDecoderEncoder::~KokoroDecoderEncoder() = default;

std::vector<float>
KokoroDecoderEncoder::encode(
    const std::vector<float>& aligned_text, size_t frame_count, const std::vector<float>& f0,
    const std::vector<float>& noise, const std::vector<float>& decoder_style) {
    const std::string nvtx_name =
        "kokoro.decoder.run source_frames=" + std::to_string(frame_count) +
        " latent_frames=" + std::to_string(frame_count * 2);
    const ggml_nvtx::range nvtx_range(nvtx_name.c_str());
    if (frame_count < 1 || frame_count > 1200 || aligned_text.size() != frame_count * 512 ||
        f0.size() != frame_count * 2 || noise.size() != f0.size()) {
        throw std::invalid_argument("invalid Kokoro decoder feature shape");
    }
    if (decoder_style.size() != 128) {
        throw std::invalid_argument("Kokoro decoder style must contain 128 values");
    }
    std::vector<float> output(frame_count * 2 * 512);
    std::vector<ggml_runtime::Session::Output> outputs = {
        {0, "", output.data(), output.size() * sizeof(float)}};
    impl_->session.run(
        {{"kokoro.decoder.text",
          GGML_TYPE_F32,
          aligned_text.data(),
          {512, static_cast<int64_t>(frame_count)}},
         {"kokoro.decoder.f0", GGML_TYPE_F32, f0.data(), {static_cast<int64_t>(f0.size())}},
         {"kokoro.decoder.noise",
          GGML_TYPE_F32,
          noise.data(),
          {static_cast<int64_t>(noise.size())}},
         {"kokoro.decoder.style", GGML_TYPE_F32, decoder_style.data(), {128}}},
        outputs);
    for (float value : output) {
        if (!std::isfinite(value))
            throw std::runtime_error("Kokoro decoder output is not finite");
    }
    return output;
}

std::vector<float>
KokoroDecoderEncoder::encode_range(
    const std::vector<float>& aligned_text, size_t frame_count, const std::vector<float>& f0,
    const std::vector<float>& noise, const std::vector<float>& decoder_style, size_t latent_begin,
    size_t latent_end) {
    const std::string nvtx_name =
        "kokoro.decoder.range latent_begin=" + std::to_string(latent_begin) +
        " latent_end=" + std::to_string(latent_end) +
        " total_latent=" + std::to_string(frame_count * 2);
    const ggml_nvtx::range nvtx_range(nvtx_name.c_str());
    if (frame_count < 1 || frame_count > 1200 || aligned_text.size() != frame_count * 512 ||
        f0.size() != frame_count * 2 || noise.size() != f0.size() || latent_begin >= latent_end ||
        latent_end > frame_count * 2) {
        throw std::invalid_argument("invalid Kokoro decoder range shape");
    }
    if (decoder_style.size() != 128) {
        throw std::invalid_argument("Kokoro decoder style must contain 128 values");
    }

    // Each decoder block has two radius-one convolutions; the final block
    // upsamples by two. This deliberately wider halo also stabilizes the
    // time-global AdaIN statistics for long inputs while imposing a fixed
    // upper bound on the graph built for an interior range.
    constexpr size_t kDecoderHaloFrames = 256;
    const size_t target_begin = latent_begin / 2;
    const size_t target_end = (latent_end + 1) / 2;
    const size_t source_begin =
        target_begin > kDecoderHaloFrames ? target_begin - kDecoderHaloFrames : 0;
    const size_t source_end = std::min(frame_count, target_end + kDecoderHaloFrames);
    const size_t source_frames = source_end - source_begin;
    const std::string nvtx_parameters =
        "kokoro.parameters.decoder_range source_begin=" + std::to_string(source_begin) +
        " source_end=" + std::to_string(source_end) +
        " source_frames=" + std::to_string(source_frames) +
        " crop_frames=" + std::to_string(latent_end - latent_begin);
    ggml_nvtx::mark(nvtx_parameters.c_str());

    std::vector<float> text(
        aligned_text.begin() + static_cast<std::ptrdiff_t>(source_begin * 512),
        aligned_text.begin() + static_cast<std::ptrdiff_t>(source_end * 512));
    std::vector<float> local_f0(
        f0.begin() + static_cast<std::ptrdiff_t>(source_begin * 2),
        f0.begin() + static_cast<std::ptrdiff_t>(source_end * 2));
    std::vector<float> local_noise(
        noise.begin() + static_cast<std::ptrdiff_t>(source_begin * 2),
        noise.begin() + static_cast<std::ptrdiff_t>(source_end * 2));
    const std::vector<float> local =
        encode(text, source_frames, local_f0, local_noise, decoder_style);

    const size_t crop_begin = latent_begin - source_begin * 2;
    const size_t crop_frames = latent_end - latent_begin;
    return std::vector<float>(
        local.begin() + static_cast<std::ptrdiff_t>(crop_begin * 512),
        local.begin() + static_cast<std::ptrdiff_t>((crop_begin + crop_frames) * 512));
}

}  // namespace nemo_speech::tts::kokoro
