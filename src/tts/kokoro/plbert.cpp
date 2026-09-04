// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "plbert.h"

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

class PlbertModule final : public ggml_runtime::Module {
   public:
    void define_tensors(ggml_runtime::Session* session) override {
        names_ = {
            "kokoro.bert.embeddings.word_embeddings.weight",
            "kokoro.bert.embeddings.position_embeddings.weight",
            "kokoro.bert.embeddings.token_type_embeddings.weight",
            "kokoro.bert.embeddings.LayerNorm.weight",
            "kokoro.bert.embeddings.LayerNorm.bias",
            "kokoro.bert.encoder.embedding_hidden_mapping_in.weight",
            "kokoro.bert.encoder.embedding_hidden_mapping_in.bias",
            "kokoro.bert.encoder.g0.l0.attention.query.weight",
            "kokoro.bert.encoder.g0.l0.attention.query.bias",
            "kokoro.bert.encoder.g0.l0.attention.key.weight",
            "kokoro.bert.encoder.g0.l0.attention.key.bias",
            "kokoro.bert.encoder.g0.l0.attention.value.weight",
            "kokoro.bert.encoder.g0.l0.attention.value.bias",
            "kokoro.bert.encoder.g0.l0.attention.dense.weight",
            "kokoro.bert.encoder.g0.l0.attention.dense.bias",
            "kokoro.bert.encoder.g0.l0.attention.LayerNorm.weight",
            "kokoro.bert.encoder.g0.l0.attention.LayerNorm.bias",
            "kokoro.bert.encoder.g0.l0.ffn.weight",
            "kokoro.bert.encoder.g0.l0.ffn.bias",
            "kokoro.bert.encoder.g0.l0.ffn_output.weight",
            "kokoro.bert.encoder.g0.l0.ffn_output.bias",
            "kokoro.bert.encoder.g0.l0.full_layer_layer_norm.weight",
            "kokoro.bert.encoder.g0.l0.full_layer_layer_norm.bias",
            "kokoro.bert_encoder.weight",
            "kokoro.bert_encoder.bias",
        };
        for (const std::string& name : names_) {
            const auto shape = session->gguf_loader->get_tensor_shape(name);
            const ggml_type type = session->gguf_loader->get_tensor_type(name);
            switch (shape.size()) {
                case 1:
                    session->model_tensor_container->create_tensor_1d(name, type, shape[0]);
                    break;
                case 2:
                    session->model_tensor_container->create_tensor_2d(
                        name, type, shape[0], shape[1]);
                    break;
                default:
                    throw std::runtime_error("unexpected PL-BERT tensor rank: " + name);
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
            throw std::logic_error("PL-BERT graph requires IDs, positions, and token types");
        }
        auto ids = inputs.get_tensor(0);
        auto positions = inputs.get_tensor(1);
        auto token_types = inputs.get_tensor(2);
        auto word_weight = weight(session, "kokoro.bert.embeddings.word_embeddings.weight");
        auto context = tensors->get_ctx_of_buffer_type(word_weight.buft);
        ggml_context* ctx = context.ctx;

        ggml_tensor* hidden = ggml_get_rows(ctx, word_weight.tensor, ids.tensor);
        hidden = ggml_add(
            ctx, hidden,
            ggml_get_rows(
                ctx, weight(session, "kokoro.bert.embeddings.position_embeddings.weight").tensor,
                positions.tensor));
        hidden = ggml_add(
            ctx, hidden,
            ggml_get_rows(
                ctx, weight(session, "kokoro.bert.embeddings.token_type_embeddings.weight").tensor,
                token_types.tensor));
        hidden = layer_norm(ctx, session, hidden, "kokoro.bert.embeddings.LayerNorm", 1.0e-12f);
        hidden = linear(ctx, session, hidden, "kokoro.bert.encoder.embedding_hidden_mapping_in");

        constexpr int64_t kHeads = 12;
        constexpr int64_t kHeadDim = 64;
        constexpr int kLayers = 12;
        const int64_t token_count = hidden->ne[1];
        for (int layer = 0; layer < kLayers; ++layer) {
            (void)layer;  // ALBERT shares the single layer's parameters.
            ggml_tensor* query =
                linear(ctx, session, hidden, "kokoro.bert.encoder.g0.l0.attention.query");
            ggml_tensor* key =
                linear(ctx, session, hidden, "kokoro.bert.encoder.g0.l0.attention.key");
            ggml_tensor* value =
                linear(ctx, session, hidden, "kokoro.bert.encoder.g0.l0.attention.value");

            query = ggml_cont(
                ctx,
                ggml_permute(
                    ctx, ggml_reshape_3d(ctx, query, kHeadDim, kHeads, token_count), 0, 2, 1, 3));
            key = ggml_cont(
                ctx,
                ggml_permute(
                    ctx, ggml_reshape_3d(ctx, key, kHeadDim, kHeads, token_count), 0, 2, 1, 3));
            ggml_tensor* scores = kokoro_mul_mat(ctx, key, query);
            scores = ggml_scale(ctx, scores, 1.0f / std::sqrt(static_cast<float>(kHeadDim)));
            scores = ggml_soft_max(ctx, scores);

            ggml_tensor* value_heads = ggml_reshape_3d(ctx, value, kHeadDim, kHeads, token_count);
            ggml_tensor* value_transposed = ggml_cont(
                ctx, ggml_permute(ctx, ggml_permute(ctx, value_heads, 2, 1, 0, 3), 0, 2, 1, 3));
            ggml_tensor* attended =
                ggml_permute(ctx, kokoro_mul_mat(ctx, value_transposed, scores), 0, 2, 1, 3);
            attended =
                ggml_reshape_2d(ctx, ggml_cont(ctx, attended), kHeads * kHeadDim, token_count);
            attended = linear(ctx, session, attended, "kokoro.bert.encoder.g0.l0.attention.dense");
            attended = layer_norm(
                ctx, session, ggml_add(ctx, attended, hidden),
                "kokoro.bert.encoder.g0.l0.attention.LayerNorm", 1.0e-12f);

            ggml_tensor* feed_forward =
                linear(ctx, session, attended, "kokoro.bert.encoder.g0.l0.ffn");
            feed_forward = gelu_new(ctx, feed_forward);
            feed_forward =
                linear(ctx, session, feed_forward, "kokoro.bert.encoder.g0.l0.ffn_output");
            hidden = layer_norm(
                ctx, session, ggml_add(ctx, feed_forward, attended),
                "kokoro.bert.encoder.g0.l0.full_layer_layer_norm", 1.0e-12f);
        }
        hidden = linear(ctx, session, hidden, "kokoro.bert_encoder");
        hidden = ggml_cont(ctx, ggml_cast(ctx, hidden, GGML_TYPE_F32));
        ggml_set_name(hidden, "kokoro.plbert.output");

        ggml_runtime::TensorBag output;
        output.add_tensor({hidden, context.buft});
        return output;
    }

   private:
    std::vector<std::string> names_;

    static ggml_runtime::ggml_bf_tensor weight(
        ggml_runtime::Session* session, const std::string& name) {
        return session->model_tensor_container->get_tensor_by_name(name);
    }

    static ggml_tensor* linear(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix) {
        ggml_tensor* output =
            kokoro_mul_mat(ctx, weight(session, prefix + ".weight").tensor, input);
        return ggml_add(ctx, output, weight(session, prefix + ".bias").tensor);
    }

    static ggml_tensor* layer_norm(
        ggml_context* ctx, ggml_runtime::Session* session, ggml_tensor* input,
        const std::string& prefix, float epsilon) {
        ggml_tensor* normalized = ggml_norm(ctx, input, epsilon);
        normalized = ggml_mul(ctx, normalized, weight(session, prefix + ".weight").tensor);
        return ggml_add(ctx, normalized, weight(session, prefix + ".bias").tensor);
    }

    static ggml_tensor* gelu_new(ggml_context* ctx, ggml_tensor* input) {
        // Transformers' NewGELUActivation. Do not use ggml_gelu here: the CPU
        // backend intentionally evaluates that op through an F16 lookup table,
        // which exceeds Kokoro's F32 differential tolerance after 12 shared
        // ALBERT layers.
        ggml_tensor* squared = ggml_mul(ctx, input, input);
        ggml_tensor* cubed = ggml_mul(ctx, squared, input);
        ggml_tensor* inner = ggml_add(ctx, input, ggml_scale(ctx, cubed, 0.044715f));
        ggml_tensor* activated = ggml_tanh(ctx, ggml_scale(ctx, inner, 0.7978845608028654f));
        return ggml_scale(ctx, ggml_add(ctx, input, ggml_mul(ctx, input, activated)), 0.5f);
    }
};

}  // namespace

class KokoroPlbertEncoder::Impl {
   public:
    Impl(const std::string& model_path, bool use_gpu)
        : loader(model_path), metadata(loader), backend({use_gpu, 0, nullptr}),
          session(backend, &module, &loader) {
        session.setup();
    }

    ggml_runtime::GGUFLoader loader;
    KokoroModelMetadata metadata;
    PlbertModule module;
    ggml_runtime::BackendManager backend;
    ggml_runtime::Session session;
};

KokoroPlbertEncoder::KokoroPlbertEncoder(const std::string& model_path, bool use_gpu)
    : impl_(std::make_unique<Impl>(model_path, use_gpu)) {}

KokoroPlbertEncoder::~KokoroPlbertEncoder() = default;

std::vector<float>
KokoroPlbertEncoder::encode(const std::vector<int32_t>& framed_ids) {
    const std::string nvtx_name = "kokoro.plbert.run tokens=" + std::to_string(framed_ids.size());
    const ggml_nvtx::range nvtx_range(nvtx_name.c_str());
    if (framed_ids.size() < 2 ||
        framed_ids.size() > static_cast<size_t>(impl_->metadata.hparams().context_length)) {
        throw std::invalid_argument("Kokoro framed token count must be in [2,512]");
    }
    for (const int32_t id : framed_ids) {
        if (id < 0 || id >= impl_->metadata.hparams().vocab_size) {
            throw std::invalid_argument("Kokoro PL-BERT token ID is out of range");
        }
    }
    std::vector<int32_t> positions(framed_ids.size());
    std::vector<int32_t> token_types(framed_ids.size(), 0);
    for (size_t index = 0; index < positions.size(); ++index) {
        positions[index] = static_cast<int32_t>(index);
    }
    std::vector<float> output(framed_ids.size() * 512);
    std::vector<ggml_runtime::Session::Input> inputs = {
        {"kokoro.plbert.ids",
         GGML_TYPE_I32,
         framed_ids.data(),
         {static_cast<int64_t>(framed_ids.size())}},
        {"kokoro.plbert.positions",
         GGML_TYPE_I32,
         positions.data(),
         {static_cast<int64_t>(positions.size())}},
        {"kokoro.plbert.token_types",
         GGML_TYPE_I32,
         token_types.data(),
         {static_cast<int64_t>(token_types.size())}},
    };
    std::vector<ggml_runtime::Session::Output> outputs(1);
    outputs[0].index = 0;
    outputs[0].host_buffer = output.data();
    outputs[0].nbytes = output.size() * sizeof(float);
    impl_->session.run(inputs, outputs);
    if (outputs[0].out_shape[0] != 512 ||
        outputs[0].out_shape[1] != static_cast<int64_t>(framed_ids.size())) {
        throw std::runtime_error("Kokoro PL-BERT returned an unexpected shape");
    }
    return output;
}

}  // namespace nemo_speech::tts::kokoro
