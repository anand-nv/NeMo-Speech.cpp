// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "duration.h"

#include <ggml.h>

#include <algorithm>
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

class DurationModule final : public ggml_runtime::Module {
   public:
    void define_tensors(ggml_runtime::Session* session) override {
        for (int layer : {0, 2, 4}) {
            const std::string prefix =
                "kokoro.predictor.text_encoder.lstms." + std::to_string(layer);
            add_lstm_names(prefix);
            names_.push_back(
                "kokoro.predictor.text_encoder.lstms." + std::to_string(layer + 1) + ".fc.weight");
            names_.push_back(
                "kokoro.predictor.text_encoder.lstms." + std::to_string(layer + 1) + ".fc.bias");
        }
        add_lstm_names("kokoro.predictor.lstm");
        names_.push_back("kokoro.predictor.duration_proj.linear_layer.weight");
        names_.push_back("kokoro.predictor.duration_proj.linear_layer.bias");

        for (const std::string& name : names_) {
            const auto shape = session->gguf_loader->get_tensor_shape(name);
            const ggml_type type = session->gguf_loader->get_tensor_type(name);
            if (shape.size() == 1) {
                session->model_tensor_container->create_tensor_1d(name, type, shape[0]);
            } else if (shape.size() == 2) {
                session->model_tensor_container->create_tensor_2d(name, type, shape[0], shape[1]);
            } else {
                throw std::runtime_error("unexpected Kokoro duration tensor rank: " + name);
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
            throw std::logic_error("Kokoro duration graph requires features, style, and speed");
        }
        const auto features = inputs.get_tensor(0);
        const auto style = inputs.get_tensor(1);
        const auto speed = inputs.get_tensor(2);
        if (features.tensor->ne[0] != 512 || style.tensor->ne[0] != 128 ||
            speed.tensor->ne[0] != 1) {
            throw std::logic_error("invalid Kokoro duration graph input shape");
        }
        ggml_context* ctx = tensors->get_ctx_of_buffer_type(features.buft).ctx;

        ggml_tensor* hidden = concat_style(ctx, features.tensor, style.tensor);
        for (int layer : {0, 2, 4}) {
            hidden = bidirectional_lstm(
                ctx, session, hidden,
                "kokoro.predictor.text_encoder.lstms." + std::to_string(layer));
            hidden = adaptive_layer_norm(
                ctx, session, hidden, style.tensor,
                "kokoro.predictor.text_encoder.lstms." + std::to_string(layer + 1) + ".fc");
            hidden = concat_style(ctx, hidden, style.tensor);
        }

        ggml_tensor* encoded_features = hidden;
        hidden = bidirectional_lstm(ctx, session, encoded_features, "kokoro.predictor.lstm");
        ggml_tensor* logits =
            linear(ctx, session, hidden, "kokoro.predictor.duration_proj.linear_layer");
        ggml_tensor* duration = ggml_sum_rows(ctx, ggml_sigmoid(ctx, logits));
        duration = ggml_div(ctx, duration, ggml_repeat(ctx, speed.tensor, duration));
        ggml_set_name(duration, "kokoro.duration.values");
        ggml_set_output(duration);

        ggml_runtime::TensorBag output;
        output.add_tensor({duration, features.buft});
        output.add_tensor({encoded_features, features.buft});
        return output;
    }

   private:
    static void append_direction(
        std::vector<std::string>& names, const std::string& prefix, const std::string& suffix) {
        names.push_back(prefix + ".weight_ih_l0" + suffix);
        names.push_back(prefix + ".weight_hh_l0" + suffix);
        names.push_back(prefix + ".bias_ih_l0" + suffix);
        names.push_back(prefix + ".bias_hh_l0" + suffix);
    }

    void add_lstm_names(const std::string& prefix) {
        append_direction(names_, prefix, "");
        append_direction(names_, prefix, "_reverse");
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

    static ggml_tensor* concat_style(ggml_context* ctx, ggml_tensor* input, ggml_tensor* style) {
        ggml_tensor* style_columns = ggml_repeat(
            ctx, ggml_reshape_2d(ctx, style, style->ne[0], 1),
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style->ne[0], input->ne[1]));
        return ggml_concat(ctx, input, style_columns, 0);
    }

    static ggml_tensor* adaptive_layer_norm(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input, ggml_tensor* style,
        const std::string& prefix) {
        ggml_tensor* affine = linear(ctx, session, style, prefix);
        const int64_t channels = input->ne[0];
        if (affine->ne[0] != 2 * channels) {
            throw std::logic_error("Kokoro AdaLayerNorm channel mismatch");
        }
        ggml_tensor* gamma = ggml_cont(ctx, ggml_view_1d(ctx, affine, channels, 0));
        ggml_tensor* beta = ggml_cont(
            ctx,
            ggml_view_1d(ctx, affine, channels, static_cast<size_t>(channels) * affine->nb[0]));
        ggml_tensor* normalized = ggml_norm(ctx, input, 1.0e-5f);
        ggml_tensor* gamma_columns = ggml_repeat(ctx, gamma, normalized);
        ggml_tensor* beta_columns = ggml_repeat(ctx, beta, normalized);
        return ggml_add(
            ctx, ggml_add(ctx, normalized, ggml_mul(ctx, normalized, gamma_columns)), beta_columns);
    }

    static std::vector<ggml_tensor*> lstm_direction(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix, bool reverse) {
        constexpr int64_t hidden_size = 256;
        const std::string suffix = reverse ? "_reverse" : "";
        ggml_tensor* weight_ih = weight(session, prefix + ".weight_ih_l0" + suffix);
        ggml_tensor* weight_hh = weight(session, prefix + ".weight_hh_l0" + suffix);
        ggml_tensor* bias_ih = weight(session, prefix + ".bias_ih_l0" + suffix);
        ggml_tensor* bias_hh = weight(session, prefix + ".bias_hh_l0" + suffix);
        if (weight_ih->ne[0] != input->ne[0] || weight_ih->ne[1] != 4 * hidden_size ||
            weight_hh->ne[0] != hidden_size || weight_hh->ne[1] != 4 * hidden_size) {
            throw std::logic_error("Kokoro LSTM weight shape mismatch: " + prefix);
        }

        ggml_tensor* hidden =
            ggml_scale(ctx, ggml_cont(ctx, ggml_view_1d(ctx, input, hidden_size, 0)), 0.0f);
        ggml_tensor* cell =
            ggml_scale(ctx, ggml_cont(ctx, ggml_view_1d(ctx, input, hidden_size, 0)), 0.0f);
        const int64_t steps = input->ne[1];
        std::vector<ggml_tensor*> output(static_cast<size_t>(steps));
        for (int64_t iteration = 0; iteration < steps; ++iteration) {
            const int64_t step = reverse ? steps - iteration - 1 : iteration;
            ggml_tensor* x = ggml_cont(
                ctx,
                ggml_view_1d(ctx, input, input->ne[0], static_cast<size_t>(step) * input->nb[1]));
            ggml_tensor* gates = ggml_add(
                ctx, ggml_add(ctx, kokoro_mul_mat(ctx, weight_ih, x), bias_ih),
                ggml_add(ctx, kokoro_mul_mat(ctx, weight_hh, hidden), bias_hh));
            auto gate = [&](int index) {
                return ggml_cont(
                    ctx, ggml_view_1d(
                             ctx, gates, hidden_size,
                             static_cast<size_t>(index * hidden_size) * gates->nb[0]));
            };
            ggml_tensor* input_gate = ggml_sigmoid(ctx, gate(0));
            ggml_tensor* forget_gate = ggml_sigmoid(ctx, gate(1));
            ggml_tensor* candidate = ggml_tanh(ctx, gate(2));
            ggml_tensor* output_gate = ggml_sigmoid(ctx, gate(3));
            cell = ggml_add(
                ctx, ggml_mul(ctx, forget_gate, cell), ggml_mul(ctx, input_gate, candidate));
            hidden = ggml_mul(ctx, output_gate, ggml_tanh(ctx, cell));
            output[static_cast<size_t>(step)] = hidden;
        }
        return output;
    }

    static ggml_tensor* bidirectional_lstm(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix) {
        auto forward = lstm_direction(ctx, session, input, prefix, false);
        auto backward = lstm_direction(ctx, session, input, prefix, true);
        ggml_tensor* output = nullptr;
        for (size_t step = 0; step < forward.size(); ++step) {
            ggml_tensor* column = ggml_concat(ctx, forward[step], backward[step], 0);
            column = ggml_reshape_2d(ctx, column, column->ne[0], 1);
            output = output ? ggml_concat(ctx, output, column, 1) : column;
        }
        return output;
    }

    std::vector<std::string> names_;
};

}  // namespace

class KokoroDurationPredictor::Impl {
   public:
    Impl(const std::string& model_path, bool use_gpu)
        : loader(model_path), metadata(loader), backend({use_gpu, 0, nullptr}),
          session(backend, &module, &loader) {
        session.setup();
    }

    ggml_runtime::GGUFLoader loader;
    KokoroModelMetadata metadata;
    DurationModule module;
    ggml_runtime::BackendManager backend;
    ggml_runtime::Session session;
};

KokoroDurationPredictor::KokoroDurationPredictor(const std::string& model_path, bool use_gpu)
    : impl_(std::make_unique<Impl>(model_path, use_gpu)) {}

KokoroDurationPredictor::~KokoroDurationPredictor() = default;

KokoroDurationResult
KokoroDurationPredictor::predict(
    const std::vector<float>& projected_plbert, size_t token_count, const std::vector<float>& style,
    float speed) {
    const std::string nvtx_name = "kokoro.duration.run tokens=" + std::to_string(token_count) +
                                  " speed=" + std::to_string(speed);
    const ggml_nvtx::range nvtx_range(nvtx_name.c_str());
    if (token_count < 2 || token_count > 512 || projected_plbert.size() != token_count * 512) {
        throw std::invalid_argument("invalid projected PL-BERT shape for Kokoro duration");
    }
    if (style.size() != 128) {
        throw std::invalid_argument("Kokoro duration style must contain 128 values");
    }
    const auto& hparams = impl_->metadata.hparams();
    if (!std::isfinite(speed) || speed < hparams.speed_min || speed > hparams.speed_max) {
        throw std::invalid_argument("Kokoro speed is outside the model-supported range");
    }

    KokoroDurationResult result;
    result.values.resize(token_count);
    result.encoded_features.resize(token_count * 640);
    std::vector<ggml_runtime::Session::Output> outputs = {
        {0, "", result.values.data(), result.values.size() * sizeof(float)},
        {1, "", result.encoded_features.data(), result.encoded_features.size() * sizeof(float)}};
    impl_->session.run(
        {{"kokoro.duration.features",
          GGML_TYPE_F32,
          projected_plbert.data(),
          {512, static_cast<int64_t>(token_count)}},
         {"kokoro.duration.style", GGML_TYPE_F32, style.data(), {128}},
         {"kokoro.duration.speed", GGML_TYPE_F32, &speed, {1}}},
        outputs);

    result.frames.reserve(token_count);
    for (const float value : result.values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Kokoro duration prediction is not finite");
        }
        const float rounded = std::nearbyint(value);
        result.frames.push_back(static_cast<int32_t>(std::max(1.0f, rounded)));
    }
    return result;
}

}  // namespace nemo_speech::tts::kokoro
