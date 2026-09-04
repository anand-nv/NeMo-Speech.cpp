// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "prosody.h"

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

class ProsodyModule final : public ggml_runtime::Module {
   public:
    void define_tensors(ggml_runtime::Session* session) override {
        for (const std::string branch : {"F0", "N"}) {
            for (int block = 0; block < 3; ++block) {
                const std::string prefix =
                    "kokoro.predictor." + branch + "." + std::to_string(block);
                for (const std::string conv : {"conv1", "conv2"}) {
                    names_.push_back(prefix + "." + conv + ".weight");
                    names_.push_back(prefix + "." + conv + ".bias");
                }
                for (const std::string norm : {"norm1", "norm2"}) {
                    names_.push_back(prefix + "." + norm + ".fc.weight");
                    names_.push_back(prefix + "." + norm + ".fc.bias");
                }
                if (block == 1) {
                    names_.push_back(prefix + ".conv1x1.weight");
                    names_.push_back(prefix + ".pool.weight");
                    names_.push_back(prefix + ".pool.bias");
                }
            }
            names_.push_back("kokoro.predictor." + branch + "_proj.weight");
            names_.push_back("kokoro.predictor." + branch + "_proj.bias");
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
                    throw std::runtime_error("unexpected Kokoro prosody tensor rank: " + name);
            }
        }
    }

    void set_data(ggml_runtime::Session* session) override {
        for (const std::string& name : names_) session->load_weight(name);
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag inputs,
        ggml_runtime::TensorContainer* tensors) override {
        if (inputs.tensor_count() != 2) {
            throw std::logic_error("Kokoro prosody graph requires shared features and style");
        }
        const auto features = inputs.get_tensor(0);
        const auto style = inputs.get_tensor(1);
        if (features.tensor->ne[0] != 512 || style.tensor->ne[0] != 128) {
            throw std::logic_error("invalid Kokoro prosody input shape");
        }
        ggml_context* ctx = tensors->get_ctx_of_buffer_type(features.buft).ctx;
        // Convolution layout is [time,channels].
        ggml_tensor* input = ggml_cont(ctx, ggml_transpose(ctx, features.tensor));
        ggml_tensor* f0 = branch(ctx, session, input, style.tensor, "F0");
        ggml_tensor* noise = branch(ctx, session, input, style.tensor, "N");
        ggml_set_name(f0, "kokoro.prosody.f0");
        ggml_set_name(noise, "kokoro.prosody.noise");
        ggml_set_output(f0);
        ggml_set_output(noise);
        ggml_runtime::TensorBag output;
        output.add_tensor({f0, features.buft});
        output.add_tensor({noise, features.buft});
        return output;
    }

   private:
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

    static ggml_tensor* adain(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input, ggml_tensor* style,
        const std::string& prefix) {
        ggml_tensor* affine = linear(ctx, session, style, prefix + ".fc");
        const int64_t channels = input->ne[1];
        if (affine->ne[0] != 2 * channels) {
            throw std::logic_error("Kokoro AdaIN channel mismatch");
        }
        ggml_tensor* gamma = ggml_cont(ctx, ggml_view_1d(ctx, affine, channels, 0));
        ggml_tensor* beta = ggml_cont(
            ctx,
            ggml_view_1d(ctx, affine, channels, static_cast<size_t>(channels) * affine->nb[0]));
        gamma = ggml_reshape_2d(ctx, gamma, 1, channels);
        beta = ggml_reshape_2d(ctx, beta, 1, channels);
        ggml_tensor* normalized = ggml_norm(ctx, input, 1.0e-5f);
        ggml_tensor* gamma_rows = ggml_repeat(ctx, gamma, normalized);
        ggml_tensor* beta_rows = ggml_repeat(ctx, beta, normalized);
        return ggml_add(
            ctx, ggml_add(ctx, normalized, ggml_mul(ctx, normalized, gamma_rows)), beta_rows);
    }

    static ggml_tensor* conv(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix, int padding) {
        ggml_tensor* output =
            ggml_conv_1d(ctx, weight(session, prefix + ".weight"), input, 1, padding, 1);
        return ggml_add(
            ctx, output, ggml_reshape_2d(ctx, weight(session, prefix + ".bias"), 1, output->ne[1]));
    }

    static ggml_tensor* pointwise_no_bias(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& name) {
        ggml_tensor* channels_first = ggml_cont(ctx, ggml_transpose(ctx, input));
        ggml_tensor* matrix = weight(session, name);
        matrix = ggml_reshape_2d(ctx, matrix, matrix->ne[1], matrix->ne[2]);
        return ggml_cont(ctx, ggml_transpose(ctx, kokoro_mul_mat(ctx, matrix, channels_first)));
    }

    static ggml_tensor* depthwise_transpose(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix) {
        ggml_tensor* kernel = weight(session, prefix + ".weight");
        const int64_t time = input->ne[0];
        const int64_t channels = input->ne[1];
        if (kernel->ne[0] != 3 || kernel->ne[1] != 1 || kernel->ne[2] != channels) {
            throw std::logic_error("Kokoro depthwise transpose convolution shape mismatch");
        }
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
        const std::string& prefix, bool upsample, bool learned_shortcut) {
        ggml_tensor* shortcut = input;
        if (upsample) {
            shortcut = ggml_interpolate(
                ctx, shortcut, shortcut->ne[0] * 2, shortcut->ne[1], 1, 1, GGML_SCALE_MODE_NEAREST);
        }
        if (learned_shortcut) {
            shortcut = pointwise_no_bias(ctx, session, shortcut, prefix + ".conv1x1.weight");
        }

        ggml_tensor* residual = adain(ctx, session, input, style, prefix + ".norm1");
        residual = ggml_leaky_relu(ctx, residual, 0.2f, false);
        if (upsample) {
            residual = depthwise_transpose(ctx, session, residual, prefix + ".pool");
        }
        residual = conv(ctx, session, residual, prefix + ".conv1", 1);
        residual = adain(ctx, session, residual, style, prefix + ".norm2");
        residual = ggml_leaky_relu(ctx, residual, 0.2f, false);
        residual = conv(ctx, session, residual, prefix + ".conv2", 1);
        return ggml_scale(ctx, ggml_add(ctx, residual, shortcut), 0.7071067811865475f);
    }

    static ggml_tensor* branch(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input, ggml_tensor* style,
        const std::string& name) {
        ggml_tensor* output = input;
        output = residual_block(
            ctx, session, output, style, "kokoro.predictor." + name + ".0", false, false);
        output = residual_block(
            ctx, session, output, style, "kokoro.predictor." + name + ".1", true, true);
        output = residual_block(
            ctx, session, output, style, "kokoro.predictor." + name + ".2", false, false);
        ggml_tensor* channels_first = ggml_cont(ctx, ggml_transpose(ctx, output));
        ggml_tensor* projection = weight(session, "kokoro.predictor." + name + "_proj.weight");
        projection = ggml_reshape_2d(ctx, projection, projection->ne[1], 1);
        return ggml_add(
            ctx, kokoro_mul_mat(ctx, projection, channels_first),
            weight(session, "kokoro.predictor." + name + "_proj.bias"));
    }

    std::vector<std::string> names_;
};

}  // namespace

class KokoroProsodyHeads::Impl {
   public:
    Impl(const std::string& model_path, bool use_gpu)
        : loader(model_path), metadata(loader), backend({use_gpu, 0, nullptr}),
          session(backend, &module, &loader) {
        session.setup();
    }

    ggml_runtime::GGUFLoader loader;
    KokoroModelMetadata metadata;
    ProsodyModule module;
    ggml_runtime::BackendManager backend;
    ggml_runtime::Session session;
};

KokoroProsodyHeads::KokoroProsodyHeads(const std::string& model_path, bool use_gpu)
    : impl_(std::make_unique<Impl>(model_path, use_gpu)) {}

KokoroProsodyHeads::~KokoroProsodyHeads() = default;

KokoroProsody
KokoroProsodyHeads::predict(
    const std::vector<float>& shared_features, size_t frame_count,
    const std::vector<float>& style) {
    const std::string nvtx_name = "kokoro.prosody.run frames=" + std::to_string(frame_count);
    const ggml_nvtx::range nvtx_range(nvtx_name.c_str());
    if (frame_count < 1 || frame_count > 1200 || shared_features.size() != frame_count * 512) {
        throw std::invalid_argument("invalid Kokoro shared prosody feature shape");
    }
    if (style.size() != 128) {
        throw std::invalid_argument("Kokoro prosody style must contain 128 values");
    }
    KokoroProsody result;
    result.f0.resize(frame_count * 2);
    result.noise.resize(frame_count * 2);
    std::vector<ggml_runtime::Session::Output> outputs = {
        {0, "", result.f0.data(), result.f0.size() * sizeof(float)},
        {1, "", result.noise.data(), result.noise.size() * sizeof(float)}};
    impl_->session.run(
        {{"kokoro.prosody.shared",
          GGML_TYPE_F32,
          shared_features.data(),
          {512, static_cast<int64_t>(frame_count)}},
         {"kokoro.prosody.style", GGML_TYPE_F32, style.data(), {128}}},
        outputs);
    for (float value : result.f0) {
        if (!std::isfinite(value))
            throw std::runtime_error("Kokoro F0 is not finite");
    }
    for (float value : result.noise) {
        if (!std::isfinite(value))
            throw std::runtime_error("Kokoro noise is not finite");
    }
    return result;
}

}  // namespace nemo_speech::tts::kokoro
