// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "batch_lt.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "decoder.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "graph.h"
#include "nvtx_utils.h"

namespace nemo_speech::tts {

struct MagpieBatchLocalTransformer::Graph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor* logits = nullptr;  // [vocab, lanes] f32

    ~Graph() {
        if (allocr)
            ggml_gallocr_free(allocr);
        if (ctx)
            ggml_free(ctx);
    }
};

// Persistent tensors of one bucket (the per-frame attention cache, lane slots, per-round cache
// state, the round-0 input and the previous codes) and its round graphs.
struct MagpieBatchLocalTransformer::Bucket {
    int size = 0;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor*> arenas;  // per layer [lt_hidden * rounds, lanes, 2]
    ggml_tensor* lane_slots = nullptr;
    std::vector<ggml_tensor*> states;  // per round [lanes, 2]
    ggml_tensor* input = nullptr;      // [n_embd, lanes]
    ggml_tensor* prev_codes = nullptr;  // [bucket]
    std::vector<std::unique_ptr<Graph>> rounds;

    ~Bucket() {
        rounds.clear();
        if (buffer)
            ggml_backend_buffer_free(buffer);
        if (ctx)
            ggml_free(ctx);
    }
};

MagpieBatchLocalTransformer::MagpieBatchLocalTransformer(
    const magpietts_model& model, bool use_cfg, int threads)
    : model_(model), use_cfg_(use_cfg), threads_(std::max(1, threads)) {}

MagpieBatchLocalTransformer::~MagpieBatchLocalTransformer() = default;

bool
MagpieBatchLocalTransformer::init(std::string* error) {
    auto fail = [&](const char* message) {
        if (error)
            *error = message;
        return false;
    };
#if !defined(NEMO_SPEECH_GGML_PATCHED)
    return fail("the batched local transformer requires the patched ggml");
#endif
    const magpietts_hparams& h = model_.hparams;
    const magpietts_transformer& tr = model_.local;
    const int rounds = h.stacked_audio_codebooks();
    if (tr.layers.empty() || tr.kernel != 1 || tr.n_head <= 0 || tr.n_embd % tr.n_head != 0 ||
        !tr.pos_emb || tr.pos_emb->ne[1] < rounds) {
        return fail("the batched local transformer needs a kernel-size-1 local transformer");
    }
    if (static_cast<int>(model_.audio_embeddings.size()) < rounds ||
        static_cast<int>(model_.lt_out_w.size()) < rounds) {
        return fail("missing local-transformer tensors");
    }
    for (const magpietts_layer& layer : tr.layers) {
        if (layer.ff_proj.size() != 1 || layer.ff_out.size() != 1) {
            return fail("the batched local transformer needs kernel-size-1 feed-forward layers");
        }
    }
    return true;
}

bool
MagpieBatchLocalTransformer::buildBucket(int bucket, Bucket& b) {
    const magpietts_hparams& h = model_.hparams;
    const magpietts_transformer& tr = model_.local;
    const int rounds = h.stacked_audio_codebooks();
    const int L = lanes(bucket);
    const int64_t n = tr.n_embd;
    const int64_t d_head = n / tr.n_head;
    b.size = bucket;
    const size_t n_tensors = tr.layers.size() + static_cast<size_t>(rounds) + 4;
    ggml_init_params params = {ggml_tensor_overhead() * n_tensors, nullptr, true};
    b.ctx = ggml_init(params);
    for (size_t il = 0; il < tr.layers.size(); ++il) {
        b.arenas.push_back(ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, n * rounds, L, 2));
    }
    b.lane_slots = ggml_new_tensor_1d(b.ctx, GGML_TYPE_I32, L);
    for (int c = 0; c < rounds; ++c) {
        b.states.push_back(ggml_new_tensor_2d(b.ctx, GGML_TYPE_I32, L, 2));
    }
    b.input = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, h.n_embd, L);
    b.prev_codes = ggml_new_tensor_1d(b.ctx, GGML_TYPE_I32, bucket);
    b.buffer = ggml_backend_alloc_ctx_tensors(b.ctx, model_.backend);
    if (!b.buffer) {
        fprintf(stderr, "batched local transformer: buffer allocation failed\n");
        return false;
    }
    ggml_backend_buffer_clear(b.buffer, 0);
    std::vector<int32_t> slots(static_cast<size_t>(L));
    for (int l = 0; l < L; ++l) slots[static_cast<size_t>(l)] = l;
    ggml_backend_tensor_set(b.lane_slots, slots.data(), 0, slots.size() * sizeof(int32_t));
    std::vector<int32_t> state(static_cast<size_t>(L) * 2);
    for (int c = 0; c < rounds; ++c) {
        // round c appends cache row c and attends the c rows before it
        std::fill(state.begin(), state.end(), c);
        ggml_backend_tensor_set(b.states[static_cast<size_t>(c)], state.data(), 0, state.size() * sizeof(int32_t));
    }

    for (int c = 0; c < rounds; ++c) {
        auto g = std::make_unique<Graph>();
        g->ctx = new_graph_context();
        g->gf = ggml_new_graph_custom(g->ctx, MAGPIETTS_MAX_NODES, false);
        ggml_context* ctx = g->ctx;
        ggml_tensor* x = nullptr;
        if (c == 0) {
            x = b.input;
        } else {
            ggml_tensor* emb = ggml_get_rows(ctx, model_.audio_embeddings[c - 1], b.prev_codes);
            x = use_cfg_ ? ggml_concat(ctx, emb, emb, 1) : emb;
        }
        if (model_.lt_in_w) {
            x = linear(ctx, model_.lt_in_w, x, model_.lt_in_b);
        }
        ggml_tensor* pos = ggml_view_2d(
            ctx, tr.pos_emb, n, 1, tr.pos_emb->nb[1], static_cast<size_t>(c) * tr.pos_emb->nb[1]);
        x = ggml_add(ctx, x, ggml_cont(ctx, ggml_cast(ctx, pos, GGML_TYPE_F32)));
        for (size_t il = 0; il < tr.layers.size(); ++il) {
            const magpietts_layer& layer = tr.layers[il];
            ggml_tensor* residual = x;
            ggml_tensor* cur = layer_norm(ctx, x, layer.norm_self);
            ggml_tensor* qkv = ggml_reshape_3d(ctx, linear(ctx, layer.self_qkv, cur), 3 * n, 1, L);
            const size_t element = ggml_element_size(qkv);
            auto split_heads = [&](size_t offset) {
                return ggml_view_4d(
                    ctx, qkv, d_head, 1, tr.n_head, L, qkv->nb[1],
                    static_cast<size_t>(d_head) * element, qkv->nb[2], offset);
            };
#if defined(NEMO_SPEECH_GGML_PATCHED)
            ggml_tensor* heads = ggml_fused_attn_cached(
                ctx, split_heads(0), split_heads(static_cast<size_t>(n) * element),
                split_heads(static_cast<size_t>(2 * n) * element), nullptr, b.arenas[il],
                b.lane_slots, b.states[static_cast<size_t>(c)], rounds,
                1.0f / std::sqrt(static_cast<float>(d_head)), true);
#else
            (void)split_heads;
            ggml_tensor* heads = nullptr;
            throw std::runtime_error("the batched local transformer requires patched ggml");
#endif
            ggml_tensor* merged = ggml_reshape_2d(ctx, ggml_permute(ctx, heads, 0, 2, 1, 3), n, L);
            x = ggml_add(ctx, residual, linear(ctx, layer.self_o, merged));
            residual = x;
            cur = layer_norm(ctx, x, layer.norm_ff);
            cur = causal_conv1d(ctx, cur, layer.ff_proj);
            cur = ggml_gelu(ctx, cur);
            cur = causal_conv1d(ctx, cur, layer.ff_out);
            x = ggml_add(ctx, residual, cur);
        }
        if (tr.norm_out) {
            x = layer_norm(ctx, x, tr.norm_out);
        }
        ggml_tensor* logits = linear(
            ctx, model_.lt_out_w[static_cast<size_t>(c)], x,
            c < static_cast<int>(model_.lt_out_b.size()) ? model_.lt_out_b[static_cast<size_t>(c)]
                                                        : nullptr);
        g->logits = ggml_cont(ctx, ggml_cast(ctx, logits, GGML_TYPE_F32));
        ggml_set_output(g->logits);
        ggml_build_forward_expand(g->gf, g->logits);
        g->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_.backend));
        if (!g->allocr || !ggml_gallocr_alloc_graph(g->allocr, g->gf)) {
            fprintf(stderr, "batched local transformer: graph allocation failed\n");
            return false;
        }
        b.rounds.push_back(std::move(g));
    }
    return true;
}

bool
MagpieBatchLocalTransformer::sample(
    int bucket, const std::vector<MagpieBatchSampling>& sampling,
    const std::function<bool(ggml_tensor* input)>& fill_input,
    std::vector<std::vector<int32_t>>& codes, std::vector<std::vector<int32_t>>& argmax) {
    const ggml_nvtx::range nvtx_range("magpietts_batch_local_transformer");
    const magpietts_hparams& h = model_.hparams;
    const int rows = static_cast<int>(sampling.size());
    if (bucket < 1 || rows < 1 || rows > bucket) {
        return false;
    }
    if (static_cast<int>(buckets_.size()) <= bucket) {
        buckets_.resize(static_cast<size_t>(bucket) + 1);
    }
    std::unique_ptr<Bucket>& slot = buckets_[static_cast<size_t>(bucket)];
    if (!slot) {
        auto b = std::make_unique<Bucket>();
        if (!buildBucket(bucket, *b)) {
            return false;
        }
        slot = std::move(b);
    }
    Bucket& b = *slot;
    if (!fill_input(b.input)) {
        return false;
    }
    if (ggml_backend_is_cpu(model_.backend)) {
        ggml_backend_cpu_set_n_threads(model_.backend, threads_);
    }
    const int rounds = h.stacked_audio_codebooks();
    const int V = h.audio_vocab_size;
    codes.assign(static_cast<size_t>(rows), std::vector<int32_t>(static_cast<size_t>(rounds), 0));
    argmax.assign(static_cast<size_t>(rows), std::vector<int32_t>(static_cast<size_t>(rounds), 0));
    std::vector<int32_t> prev(static_cast<size_t>(bucket), 0);
    std::vector<float> combined(static_cast<size_t>(V));
    for (int c = 0; c < rounds; ++c) {
        Graph& g = *b.rounds[static_cast<size_t>(c)];
        if (c > 0) {
            for (int r = 0; r < rows; ++r) prev[static_cast<size_t>(r)] = codes[static_cast<size_t>(r)][static_cast<size_t>(c - 1)];
            ggml_backend_tensor_set(b.prev_codes, prev.data(), 0, prev.size() * sizeof(int32_t));
        }
        if (ggml_backend_graph_compute(model_.backend, g.gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "batched local transformer: round %d failed\n", c);
            return false;
        }
        // rows' cond lanes are the first `rows` columns, their uncond lanes start at `bucket`
        const size_t row_bytes = static_cast<size_t>(V) * sizeof(float);
        logits_.resize(static_cast<size_t>(V) * lanes(bucket));
        ggml_backend_tensor_get(g.logits, logits_.data(), 0, row_bytes * rows);
        if (use_cfg_) {
            ggml_backend_tensor_get(
                g.logits, logits_.data() + static_cast<size_t>(V) * bucket, row_bytes * bucket,
                row_bytes * rows);
        }
        for (int r = 0; r < rows; ++r) {
            const MagpieBatchSampling& s = sampling[static_cast<size_t>(r)];
            const float* cond = logits_.data() + static_cast<size_t>(r) * V;
            if (use_cfg_) {
                const float* uncond = logits_.data() + static_cast<size_t>(bucket + r) * V;
                for (int i = 0; i < V; ++i) {
                    combined[static_cast<size_t>(i)] =
                        s.cfg_scale * cond[i] + (1.0f - s.cfg_scale) * uncond[i];
                }
            } else {
                std::copy(cond, cond + V, combined.begin());
            }
            const int sampled = MagpieCodebookSampler::sampleFromLogits(
                combined, h, s.temperature, s.top_k, *s.rng, s.forbid_eos);
            const int greedy = MagpieCodebookSampler::argmaxFromLogits(combined, h, s.forbid_eos);
            codes[static_cast<size_t>(r)][static_cast<size_t>(c)] = sampled;
            argmax[static_cast<size_t>(r)][static_cast<size_t>(c)] = greedy;
        }
    }
    return true;
}

}  // namespace nemo_speech::tts
