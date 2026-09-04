// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "acoustic.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <limits>
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

class AcousticModule final : public ggml_runtime::Module {
   public:
    void define_tensors(ggml_runtime::Session* session) override {
        names_.push_back("kokoro.text_encoder.embedding.weight");
        for (int layer = 0; layer < 3; ++layer) {
            const std::string prefix = "kokoro.text_encoder.cnn." + std::to_string(layer);
            names_.push_back(prefix + ".0.weight");
            names_.push_back(prefix + ".0.bias");
            names_.push_back(prefix + ".1.gamma");
            names_.push_back(prefix + ".1.beta");
        }
        add_lstm_names("kokoro.text_encoder.lstm");
        add_lstm_names("kokoro.predictor.shared");

        for (const std::string& name : names_) {
            const auto shape = session->gguf_loader->get_tensor_shape(name);
            ggml_type type = session->gguf_loader->get_tensor_type(name);
            // ggml_conv_1d uses an F16 im2col kernel on CPU. F32 GGUFs are
            // converted on upload; all other tensor classes retain storage.
            if (name.find(".cnn.") != std::string::npos && name.size() >= 7 &&
                name.compare(name.size() - 7, 7, ".weight") == 0) {
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
                    throw std::runtime_error("unexpected Kokoro acoustic tensor rank: " + name);
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
            throw std::logic_error(
                "Kokoro acoustic graph requires IDs and aligned duration features");
        }
        const auto ids = inputs.get_tensor(0);
        const auto aligned_duration = inputs.get_tensor(1);
        if (aligned_duration.tensor->ne[0] != 640) {
            throw std::logic_error("invalid Kokoro aligned duration feature shape");
        }
        ggml_context* ctx = tensors->get_ctx_of_buffer_type(ids.buft).ctx;

        ggml_tensor* text =
            ggml_get_rows(ctx, weight(session, "kokoro.text_encoder.embedding.weight"), ids.tensor);
        // Conv1d uses [time,channels], while embeddings and recurrent layers
        // use [channels,time].
        text = ggml_cont(ctx, ggml_transpose(ctx, text));
        for (int layer = 0; layer < 3; ++layer) {
            const std::string prefix = "kokoro.text_encoder.cnn." + std::to_string(layer);
            text = ggml_conv_1d(ctx, weight(session, prefix + ".0.weight"), text, 1, 2, 1);
            text = ggml_add(
                ctx, text, ggml_reshape_2d(ctx, weight(session, prefix + ".0.bias"), 1, 512));
            ggml_tensor* channels_first = ggml_cont(ctx, ggml_transpose(ctx, text));
            channels_first = ggml_norm(ctx, channels_first, 1.0e-5f);
            channels_first = ggml_add(
                ctx,
                ggml_mul(
                    ctx, channels_first,
                    ggml_repeat(ctx, weight(session, prefix + ".1.gamma"), channels_first)),
                ggml_repeat(ctx, weight(session, prefix + ".1.beta"), channels_first));
            channels_first = ggml_leaky_relu(ctx, channels_first, 0.2f, false);
            text = ggml_cont(ctx, ggml_transpose(ctx, channels_first));
        }
        text = bidirectional_lstm(
            ctx, session, ggml_cont(ctx, ggml_transpose(ctx, text)), "kokoro.text_encoder.lstm");
        ggml_tensor* shared =
            bidirectional_lstm(ctx, session, aligned_duration.tensor, "kokoro.predictor.shared");
        ggml_set_name(text, "kokoro.acoustic.text_tokens");
        ggml_set_name(shared, "kokoro.acoustic.prosody_shared");
        ggml_set_output(text);
        ggml_set_output(shared);

        ggml_runtime::TensorBag output;
        output.add_tensor({text, ids.buft});
        output.add_tensor({shared, aligned_duration.buft});
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

    static std::vector<ggml_tensor*> lstm_direction(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix, bool reverse) {
        constexpr int64_t hidden_size = 256;
        const std::string suffix = reverse ? "_reverse" : "";
        ggml_tensor* weight_ih = weight(session, prefix + ".weight_ih_l0" + suffix);
        ggml_tensor* weight_hh = weight(session, prefix + ".weight_hh_l0" + suffix);
        ggml_tensor* bias_ih = weight(session, prefix + ".bias_ih_l0" + suffix);
        ggml_tensor* bias_hh = weight(session, prefix + ".bias_hh_l0" + suffix);
        if (weight_ih->ne[0] != input->ne[0] || weight_ih->ne[1] != 4 * hidden_size) {
            throw std::logic_error("Kokoro acoustic LSTM shape mismatch: " + prefix);
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
            column = ggml_reshape_2d(ctx, column, 512, 1);
            output = output ? ggml_concat(ctx, output, column, 1) : column;
        }
        return output;
    }

    std::vector<std::string> names_;
};

}  // namespace

class KokoroAcousticEncoder::Impl {
   public:
    Impl(const std::string& model_path, bool use_gpu)
        : loader(model_path), metadata(loader), backend({use_gpu, 0, nullptr}),
          session(backend, &module, &loader) {
        session.setup();
    }

    ggml_runtime::GGUFLoader loader;
    KokoroModelMetadata metadata;
    AcousticModule module;
    ggml_runtime::BackendManager backend;
    ggml_runtime::Session session;
};

KokoroAcousticEncoder::KokoroAcousticEncoder(const std::string& model_path, bool use_gpu)
    : impl_(std::make_unique<Impl>(model_path, use_gpu)) {}

KokoroAcousticEncoder::~KokoroAcousticEncoder() = default;

KokoroAcousticFeatures
KokoroAcousticEncoder::encode(
    const std::vector<int32_t>& framed_ids, const std::vector<int32_t>& durations,
    const std::vector<float>& duration_features) {
    const ggml_nvtx::range nvtx_range("kokoro.acoustic.run");
    if (framed_ids.size() < 2 || framed_ids.size() > 512 || durations.size() != framed_ids.size() ||
        duration_features.size() != framed_ids.size() * 640) {
        throw std::invalid_argument("invalid Kokoro acoustic encoder input shape");
    }
    size_t frame_count = 0;
    for (const int32_t duration : durations) {
        if (duration < 1 || duration > 100) {
            throw std::invalid_argument("Kokoro duration is outside [1,100]");
        }
        if (frame_count > std::numeric_limits<size_t>::max() - static_cast<size_t>(duration)) {
            throw std::length_error("Kokoro duration sum overflow");
        }
        frame_count += static_cast<size_t>(duration);
    }
    // The current GGML scheduler has a 65k-node graph ceiling. This bound is
    // deliberately explicit until the recurrent scan is tiled in milestone 2.
    if (frame_count == 0 || frame_count > 1200) {
        throw std::length_error("Kokoro acoustic frame count exceeds the current graph capacity");
    }
    const std::string nvtx_parameters =
        "kokoro.parameters.acoustic tokens=" + std::to_string(framed_ids.size()) +
        " frames=" + std::to_string(frame_count);
    ggml_nvtx::mark(nvtx_parameters.c_str());

    std::vector<float> aligned_duration(frame_count * 640);
    size_t frame = 0;
    for (size_t token = 0; token < framed_ids.size(); ++token) {
        const float* source = duration_features.data() + token * 640;
        for (int32_t repeat = 0; repeat < durations[token]; ++repeat, ++frame) {
            std::copy(source, source + 640, aligned_duration.data() + frame * 640);
        }
    }

    std::vector<float> text_tokens(framed_ids.size() * 512);
    KokoroAcousticFeatures result;
    result.frame_count = frame_count;
    result.prosody_shared.resize(frame_count * 512);
    std::vector<ggml_runtime::Session::Output> outputs = {
        {0, "", text_tokens.data(), text_tokens.size() * sizeof(float)},
        {1, "", result.prosody_shared.data(), result.prosody_shared.size() * sizeof(float)}};
    impl_->session.run(
        {{"kokoro.acoustic.ids",
          GGML_TYPE_I32,
          framed_ids.data(),
          {static_cast<int64_t>(framed_ids.size())}},
         {"kokoro.acoustic.aligned_duration",
          GGML_TYPE_F32,
          aligned_duration.data(),
          {640, static_cast<int64_t>(frame_count)}}},
        outputs);

    result.text.resize(frame_count * 512);
    frame = 0;
    for (size_t token = 0; token < framed_ids.size(); ++token) {
        const float* source = text_tokens.data() + token * 512;
        for (int32_t repeat = 0; repeat < durations[token]; ++repeat, ++frame) {
            std::copy(source, source + 512, result.text.data() + frame * 512);
        }
    }
    return result;
}

}  // namespace nemo_speech::tts::kokoro
