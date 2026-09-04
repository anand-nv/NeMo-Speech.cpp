// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "denoiser.h"

#include <ggml.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "runtime/ggml/runtime.h"

namespace nemo_speech::tts::omnivoice {
namespace {

namespace rt = ggml_runtime;

std::string
layer_name(int layer, const char* suffix) {
    return "llm.layers." + std::to_string(layer) + "." + suffix;
}

ggml_tensor*
weight(rt::Session* session, const std::string& name) {
    return session->model_tensor_container->get_tensor_by_name(name).tensor;
}

ggml_tensor*
linear(ggml_context* ctx, rt::Session* session, const std::string& name, ggml_tensor* input) {
    return ggml_mul_mat(ctx, weight(session, name), input);
}

ggml_tensor*
rms_norm(
    ggml_context* ctx, rt::Session* session, ggml_tensor* input, const std::string& name,
    float epsilon) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, input, epsilon), weight(session, name));
}

class DenoiserModule final : public rt::Module {
   public:
    explicit DenoiserModule(ModelConfig config) : config_(std::move(config)) {}

    void define_tensors(rt::Session* session) override {
        auto* tc = session->model_tensor_container.get();
        auto* loader = session->gguf_loader;
        const int H = config_.hidden_size;
        const int V = config_.audio_vocab_size;
        const int C = config_.audio_codebooks;
        const int Q = config_.attention_heads * config_.head_size;
        const int K = config_.kv_heads * config_.head_size;

        tc->create_tensor_1d("codebook_layer_offsets", GGML_TYPE_I64, C);
        tc->create_tensor_2d(
            "audio_embeddings.weight", loader->get_tensor_type("audio_embeddings.weight"), H,
            C * V);
        tc->create_tensor_2d(
            "audio_heads.weight", loader->get_tensor_type("audio_heads.weight"), H, C * V);
        tc->create_tensor_2d(
            "llm.embed_tokens.weight", loader->get_tensor_type("llm.embed_tokens.weight"), H,
            config_.text_vocab_size);
        tc->create_tensor_1d("llm.norm.weight", GGML_TYPE_F32, H);

        for (int layer = 0; layer < config_.layer_count; ++layer) {
            declare_vector(session, layer_name(layer, "input_layernorm.weight"), H);
            declare_vector(session, layer_name(layer, "post_attention_layernorm.weight"), H);
            declare_matrix(
                session, layer_name(layer, "mlp.down_proj.weight"), config_.feed_forward_size, H);
            declare_matrix(
                session, layer_name(layer, "mlp.gate_proj.weight"), H, config_.feed_forward_size);
            declare_matrix(
                session, layer_name(layer, "mlp.up_proj.weight"), H, config_.feed_forward_size);
            declare_matrix(session, layer_name(layer, "self_attn.q_proj.weight"), H, Q);
            declare_matrix(session, layer_name(layer, "self_attn.k_proj.weight"), H, K);
            declare_matrix(session, layer_name(layer, "self_attn.v_proj.weight"), H, K);
            declare_matrix(session, layer_name(layer, "self_attn.o_proj.weight"), Q, H);
            declare_vector(
                session, layer_name(layer, "self_attn.q_norm.weight"), config_.head_size);
            declare_vector(
                session, layer_name(layer, "self_attn.k_norm.weight"), config_.head_size);
        }
    }

    void set_data(rt::Session* session) override {
        session->load_weight("codebook_layer_offsets");
        session->load_weight("audio_embeddings.weight");
        session->load_weight("audio_heads.weight");
        session->load_weight("llm.embed_tokens.weight");
        session->load_weight("llm.norm.weight");
        for (int layer = 0; layer < config_.layer_count; ++layer) {
            for (const char* suffix :
                 {"input_layernorm.weight", "post_attention_layernorm.weight",
                  "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                  "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                  "self_attn.o_proj.weight", "self_attn.q_norm.weight",
                  "self_attn.k_norm.weight"}) {
                session->load_weight(layer_name(layer, suffix));
            }
        }
    }

    rt::TensorBag build_graph(
        rt::Session* session, rt::TensorBag, rt::TensorContainer* tc) override {
        auto text_embedding =
            session->model_tensor_container->get_tensor_by_name("llm.embed_tokens.weight");
        auto audio_embedding =
            session->model_tensor_container->get_tensor_by_name("audio_embeddings.weight");
        ggml_context* ctx = tc->get_ctx_of_buffer_type(text_embedding.buft).ctx;
        ggml_tensor* text_ids = tc->get_tensor_by_name("omnivoice.in.text_ids").tensor;
        ggml_tensor* audio_ids = tc->get_tensor_by_name("omnivoice.in.audio_ids").tensor;
        ggml_tensor* audio_mask = tc->get_tensor_by_name("omnivoice.in.audio_mask").tensor;
        ggml_tensor* positions = tc->get_tensor_by_name("omnivoice.in.positions").tensor;
        ggml_tensor* mask_f32 = tc->get_tensor_by_name("omnivoice.in.attention_mask").tensor;

        const int64_t S = text_ids->ne[0];
        const int64_t B = text_ids->ne[1];
        const int H = config_.hidden_size;
        const int C = config_.audio_codebooks;
        const int heads = config_.attention_heads;
        const int kv_heads = config_.kv_heads;
        const int head_size = config_.head_size;

        ggml_tensor* hidden = nullptr;
        for (int64_t batch = 0; batch < B; ++batch) {
            ggml_tensor* text_ids_b =
                ggml_view_1d(ctx, text_ids, S, static_cast<size_t>(batch) * text_ids->nb[1]);
            ggml_tensor* text = ggml_get_rows(ctx, text_embedding.tensor, text_ids_b);
            ggml_tensor* audio = nullptr;
            for (int codebook = 0; codebook < C; ++codebook) {
                const size_t offset = static_cast<size_t>(batch) * audio_ids->nb[2] +
                                      static_cast<size_t>(codebook) * audio_ids->nb[1];
                ggml_tensor* ids = ggml_view_1d(ctx, audio_ids, S, offset);
                ggml_tensor* embedded = ggml_get_rows(ctx, audio_embedding.tensor, ids);
                audio = audio == nullptr ? embedded : ggml_add(ctx, audio, embedded);
            }
            ggml_tensor* mask =
                ggml_view_1d(ctx, audio_mask, S, static_cast<size_t>(batch) * audio_mask->nb[1]);
            mask = ggml_reshape_2d(ctx, mask, 1, S);
            ggml_tensor* selected =
                ggml_add(ctx, text, ggml_mul(ctx, ggml_sub(ctx, audio, text), mask));
            selected = ggml_reshape_3d(ctx, selected, H, S, 1);
            hidden = hidden == nullptr ? selected : ggml_concat(ctx, hidden, selected, 2);
        }

        ggml_tensor* attention_mask = ggml_cast(ctx, mask_f32, GGML_TYPE_F16);
        const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        for (int layer = 0; layer < config_.layer_count; ++layer) {
            ggml_tensor* residual = hidden;
            ggml_tensor* normalized = rms_norm(
                ctx, session, hidden, layer_name(layer, "input_layernorm.weight"),
                config_.rms_epsilon);
            ggml_tensor* query =
                linear(ctx, session, layer_name(layer, "self_attn.q_proj.weight"), normalized);
            ggml_tensor* key =
                linear(ctx, session, layer_name(layer, "self_attn.k_proj.weight"), normalized);
            ggml_tensor* value =
                linear(ctx, session, layer_name(layer, "self_attn.v_proj.weight"), normalized);

            query = ggml_reshape_4d(ctx, query, head_size, heads, S, B);
            key = ggml_reshape_4d(ctx, key, head_size, kv_heads, S, B);
            value = ggml_reshape_4d(ctx, value, head_size, kv_heads, S, B);
            query = rms_norm(
                ctx, session, query, layer_name(layer, "self_attn.q_norm.weight"),
                config_.rms_epsilon);
            key = rms_norm(
                ctx, session, key, layer_name(layer, "self_attn.k_norm.weight"),
                config_.rms_epsilon);
            query = ggml_rope_ext(
                ctx, query, positions, nullptr, head_size, GGML_ROPE_TYPE_NEOX,
                config_.context_length, config_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            key = ggml_rope_ext(
                ctx, key, positions, nullptr, head_size, GGML_ROPE_TYPE_NEOX,
                config_.context_length, config_.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            query = ggml_permute(ctx, query, 0, 2, 1, 3);
            key = ggml_permute(ctx, key, 0, 2, 1, 3);
            value = ggml_permute(ctx, value, 0, 2, 1, 3);
            if (key->type == GGML_TYPE_F32)
                key = ggml_cast(ctx, key, GGML_TYPE_F16);
            if (value->type == GGML_TYPE_F32)
                value = ggml_cast(ctx, value, GGML_TYPE_F16);
            ggml_tensor* attended = ggml_flash_attn_ext(
                ctx, query, key, value, attention_mask, attention_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
            // flash_attn_ext returns [head_size, heads, sequence, batch]
            // even though Q/K/V are supplied as [head_size, sequence, heads,
            // batch]. It is already in the layout needed to flatten heads;
            // permuting it again would mix head and sequence coordinates.
            attended = ggml_reshape_3d(ctx, attended, heads * head_size, S, B);
            attended = linear(ctx, session, layer_name(layer, "self_attn.o_proj.weight"), attended);
            hidden = ggml_add(ctx, residual, attended);

            residual = hidden;
            normalized = rms_norm(
                ctx, session, hidden, layer_name(layer, "post_attention_layernorm.weight"),
                config_.rms_epsilon);
            ggml_tensor* gate =
                linear(ctx, session, layer_name(layer, "mlp.gate_proj.weight"), normalized);
            ggml_tensor* up =
                linear(ctx, session, layer_name(layer, "mlp.up_proj.weight"), normalized);
            ggml_tensor* gated = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            ggml_tensor* mlp =
                linear(ctx, session, layer_name(layer, "mlp.down_proj.weight"), gated);
            hidden = ggml_add(ctx, residual, mlp);
        }

        hidden = rms_norm(ctx, session, hidden, "llm.norm.weight", config_.rms_epsilon);
        ggml_tensor* logits = linear(ctx, session, "audio_heads.weight", hidden);
        logits =
            ggml_reshape_4d(ctx, logits, config_.audio_vocab_size, config_.audio_codebooks, S, B);
        // PyTorch returns [B,C,S,V]; GGML stores that contiguously as [V,S,C,B].
        logits = ggml_cont(ctx, ggml_permute(ctx, logits, 0, 2, 1, 3));
        ggml_set_name(logits, "omnivoice.logits");
        ggml_set_output(logits);
        rt::TensorBag output;
        output.add_tensor(rt::ggml_bf_tensor(logits, text_embedding.buft));
        return output;
    }

   private:
    void declare_vector(rt::Session* session, const std::string& name, int64_t size) {
        session->model_tensor_container->create_tensor_1d(name, GGML_TYPE_F32, size);
    }

    void declare_matrix(rt::Session* session, const std::string& name, int64_t ne0, int64_t ne1) {
        session->model_tensor_container->create_tensor_2d(
            name, session->gguf_loader->get_tensor_type(name), ne0, ne1);
    }

    ModelConfig config_;
};

void
validate_input(const ModelConfig& cfg, const DenoiserInput& input) {
    if (input.batch_size <= 0 || input.sequence_length <= 0) {
        throw std::invalid_argument(
            "OmniVoice denoiser batch and sequence length must be positive");
    }
    if (input.sequence_length > cfg.context_length) {
        throw std::invalid_argument("OmniVoice denoiser context exceeds 40960 positions");
    }
    const size_t B = static_cast<size_t>(input.batch_size);
    const size_t S = static_cast<size_t>(input.sequence_length);
    const size_t BS = B * S;
    if (input.text_ids.size() != BS || input.shifted_audio_ids.size() != BS * cfg.audio_codebooks ||
        input.audio_mask.size() != BS || input.position_ids.size() != S ||
        input.attention_mask.size() != B * S * S) {
        throw std::invalid_argument("OmniVoice denoiser input buffer shape mismatch");
    }
    for (int32_t id : input.text_ids) {
        if (id < 0 || id >= cfg.text_vocab_size) {
            throw std::invalid_argument("OmniVoice text token is out of range");
        }
    }
    for (int32_t id : input.shifted_audio_ids) {
        if (id < 0 || id >= cfg.audio_codebooks * cfg.audio_vocab_size) {
            throw std::invalid_argument("OmniVoice shifted audio token is out of range");
        }
    }
    for (float value : input.audio_mask) {
        if (value != 0.0f && value != 1.0f) {
            throw std::invalid_argument("OmniVoice audio mask values must be zero or one");
        }
    }
    for (int32_t position : input.position_ids) {
        if (position < 0 || position >= cfg.context_length) {
            throw std::invalid_argument("OmniVoice position ID is out of range");
        }
    }
    for (float value : input.attention_mask) {
        if (!std::isfinite(value) && value != -std::numeric_limits<float>::infinity()) {
            throw std::invalid_argument("OmniVoice attention mask contains NaN or +infinity");
        }
        if (value > 0.0f) {
            throw std::invalid_argument("OmniVoice attention mask contains a positive bias");
        }
    }
}

}  // namespace

Denoiser::Denoiser(rt::BackendManager& backends, const std::string& gguf_path) {
    loader_ = std::make_unique<rt::GGUFLoader>(gguf_path);
    config_ = load_model_config(*loader_);
    module_ = std::make_unique<DenoiserModule>(config_);
    session_ = std::make_unique<rt::Session>(backends, module_.get(), loader_.get());
    session_->set_run_cache_capacity(4);
    session_->setup();
}

Denoiser::~Denoiser() = default;

DenoiserOutput
Denoiser::forward(const DenoiserInput& input) {
    validate_input(config_, input);
    DenoiserOutput output;
    output.batch_size = input.batch_size;
    output.codebooks = config_.audio_codebooks;
    output.sequence_length = input.sequence_length;
    output.vocabulary_size = config_.audio_vocab_size;
    output.logits.resize(
        static_cast<size_t>(output.batch_size) * output.codebooks * output.sequence_length *
        output.vocabulary_size);

    std::vector<rt::Session::Input> inputs = {
        {"omnivoice.in.text_ids",
         GGML_TYPE_I32,
         input.text_ids.data(),
         {input.sequence_length, input.batch_size}},
        {"omnivoice.in.audio_ids",
         GGML_TYPE_I32,
         input.shifted_audio_ids.data(),
         {input.sequence_length, config_.audio_codebooks, input.batch_size}},
        {"omnivoice.in.audio_mask",
         GGML_TYPE_F32,
         input.audio_mask.data(),
         {input.sequence_length, input.batch_size}},
        {"omnivoice.in.positions",
         GGML_TYPE_I32,
         input.position_ids.data(),
         {input.sequence_length}},
        {"omnivoice.in.attention_mask",
         GGML_TYPE_F32,
         input.attention_mask.data(),
         {input.sequence_length, input.sequence_length, 1, input.batch_size}},
    };
    std::vector<rt::Session::Output> outputs = {
        {0, "", output.logits.data(), output.logits.size() * sizeof(float)}};
    session_->run(inputs, outputs);
    if (outputs[0].out_shape[0] != config_.audio_vocab_size ||
        outputs[0].out_shape[1] != input.sequence_length ||
        outputs[0].out_shape[2] != config_.audio_codebooks ||
        outputs[0].out_shape[3] != input.batch_size) {
        throw std::runtime_error("OmniVoice denoiser returned an unexpected logit shape");
    }
    return output;
}

}  // namespace nemo_speech::tts::omnivoice
