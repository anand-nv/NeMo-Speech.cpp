// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "batch_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>

#include "graph.h"
#include "nvtx_utils.h"

namespace nemo_speech::tts {

namespace {

constexpr float kPadBias = -1.0e30f;

bool
layer_selected(const std::vector<int32_t>& layers, int layer) {
    return layers.empty() ||
           std::find(layers.begin(), layers.end(), static_cast<int32_t>(layer)) != layers.end();
}

std::string
self_arena_name(int layer) {
    return "magpietts.batch.kv." + std::to_string(layer);
}

constexpr const char* kCrossK = "magpietts.batch.cross_k";
constexpr const char* kCrossV = "magpietts.batch.cross_v";

}  // namespace

// Step graph of one bucket. Inputs (bucket B, lanes L = B or 2B):
//   tokens [B, n_cb] i32, positions [B] i32, lane_slots [L] i32, cache_meta [L, 2] i32
//   (ring heads, then valid lengths), cross_slots [B] i32, prior [TC, B] f32, mask [TC, B] f32.
// Outputs: hidden [E, L] f32, alignment [TC, B] f32 (when the model collects one).
class MagpieBatchDecoder::Module final : public ggml_runtime::Module {
   public:
    Module(
        const magpietts_model& model, const MagpieLaneLayout& layout, int cache_len,
        int text_capacity)
        : model_(model), layout_(layout), cache_len_(cache_len), text_capacity_(text_capacity) {
        const magpietts_transformer& tr = model_.decoder;
        for (int il = 0; il < static_cast<int>(tr.layers.size()); ++il) {
            if (tr.layers[il].has_cross && layer_selected(tr.estimate_alignment_from_layers, il)) {
                ++alignment_layers_;
            }
        }
    }

    int alignmentLayers() const { return alignment_layers_; }

    void define_tensors(ggml_runtime::Session* session) override {
        std::unordered_set<const ggml_tensor*> seen;
        int imported = 0;
        auto import = [&](ggml_tensor* tensor) {
            if (!tensor || !seen.insert(tensor).second) {
                return;
            }
            session->import_model_tensor(
                "magpietts.batch.external." + std::to_string(imported++), tensor);
        };
        for (ggml_tensor* embedding : model_.audio_embeddings) import(embedding);
        const magpietts_transformer& tr = model_.decoder;
        import(tr.pos_emb);
        import(tr.norm_out);
        for (const magpietts_layer& layer : tr.layers) {
            import(layer.norm_self);
            import(layer.self_qkv);
            import(layer.self_o);
            import(layer.norm_xattn_query);
            import(layer.cross_q);
            import(layer.cross_o);
            import(layer.norm_ff);
            for (ggml_tensor* tensor : layer.ff_proj) import(tensor);
            for (ggml_tensor* tensor : layer.ff_out) import(tensor);
        }
        const magpietts_hparams& h = model_.hparams;
        auto* tc = session->model_tensor_container.get();
        for (int layer = 0; layer < h.n_dec_layer; ++layer) {
            tc->create_tensor_3d(
                self_arena_name(layer), GGML_TYPE_F32, static_cast<int64_t>(h.n_embd) * cache_len_,
                layout_.arenaLanes(), 2);
        }
        const int64_t cross_dim = static_cast<int64_t>(tr.n_cross_head) * tr.n_cross_dhead;
        // [cross_dim * text_capacity, layers, slots]: a slot's rows of every layer are one block.
        tc->create_tensor_3d(
            kCrossK, GGML_TYPE_F32, cross_dim * text_capacity_, h.n_dec_layer, layout_.arena_slots);
        tc->create_tensor_3d(
            kCrossV, GGML_TYPE_F32, cross_dim * text_capacity_, h.n_dec_layer, layout_.arena_slots);
    }

    ggml_runtime::TensorBag build_graph(
        ggml_runtime::Session* session, ggml_runtime::TensorBag inputs,
        ggml_runtime::TensorContainer* tc) override {
        if (inputs.tensor_count() != 7) {
            throw std::runtime_error("Magpie batched decoder expects seven inputs");
        }
        const auto tokens = inputs.get_tensor(0);
        const auto positions = inputs.get_tensor(1);
        const auto lane_slots = inputs.get_tensor(2);
        const auto cache_meta = inputs.get_tensor(3);
        const auto cross_slots = inputs.get_tensor(4);
        const auto prior = inputs.get_tensor(5);
        const auto mask = inputs.get_tensor(6);
        const auto bf_ctx = tc->get_ctx_of_buffer_type(tokens.buft);
        ggml_context* ctx = bf_ctx.ctx;
        const magpietts_hparams& h = model_.hparams;
        const magpietts_transformer& tr = model_.decoder;
        const int64_t B = positions.tensor->ne[0];
        const int64_t L = B * layout_.lanesPerSlot();
        const int64_t E = tr.n_embd;
        const int64_t d_head = E / tr.n_head;
        const int64_t cd_head = tr.n_cross_dhead;
        const int64_t c_heads = tr.n_cross_head;
        const int64_t cross_dim = cd_head * c_heads;
        const int64_t TC = text_capacity_;

        // Input embedding: mean of the stacked codebook embeddings plus the position.
        ggml_tensor* audio = nullptr;
        for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
            ggml_tensor* ids =
                ggml_view_1d(ctx, tokens.tensor, B, static_cast<size_t>(c) * B * sizeof(int32_t));
            ggml_tensor* emb = ggml_get_rows(ctx, model_.audio_embeddings[c], ids);
            audio = audio ? ggml_add(ctx, audio, emb) : emb;
        }
        audio = ggml_scale(ctx, audio, 1.0f / static_cast<float>(h.stacked_audio_codebooks()));
        audio = ggml_add(ctx, audio, ggml_get_rows(ctx, tr.pos_emb, positions.tensor));
        ggml_tensor* x = layout_.use_cfg ? ggml_concat(ctx, audio, audio, 1) : audio;  // [E, L]

        auto cross_k = session->model_tensor_container->get_tensor_by_name(kCrossK).tensor;
        auto cross_v = session->model_tensor_container->get_tensor_by_name(kCrossV).tensor;
        ggml_tensor* prior_bias = ggml_reshape_4d(ctx, prior.tensor, TC, 1, 1, B);
        ggml_tensor* mask_bias = ggml_reshape_4d(ctx, mask.tensor, TC, 1, 1, B);
        ggml_tensor* alignment_sum = nullptr;

        for (int il = 0; il < static_cast<int>(tr.layers.size()); ++il) {
            const magpietts_layer& layer = tr.layers[il];
            ggml_tensor* residual = x;
            ggml_tensor* cur = layer_norm(ctx, x, layer.norm_self);
            ggml_tensor* qkv = linear(ctx, layer.self_qkv, cur);  // [3E, L]
            const size_t element = ggml_element_size(qkv);
            auto split_heads = [&](size_t offset) {
                return ggml_view_4d(
                    ctx, qkv, d_head, 1, tr.n_head, L, qkv->nb[1],
                    static_cast<size_t>(d_head) * element, qkv->nb[1], offset);
            };
            ggml_tensor* q = split_heads(0);
            ggml_tensor* k = split_heads(static_cast<size_t>(E) * element);
            ggml_tensor* v = split_heads(static_cast<size_t>(2 * E) * element);
            auto arena = session->model_tensor_container->get_tensor_by_name(self_arena_name(il));
#if defined(NEMO_SPEECH_GGML_PATCHED)
            ggml_tensor* heads = ggml_fused_attn_cached(
                ctx, q, k, v, nullptr, arena.tensor, lane_slots.tensor, cache_meta.tensor,
                cache_len_, 1.0f / std::sqrt(static_cast<float>(d_head)), true);
#else
            (void)q;
            (void)k;
            (void)v;
            (void)arena;
            ggml_tensor* heads = nullptr;
            throw std::runtime_error("the batched Magpie decoder requires patched ggml");
#endif
            ggml_tensor* merged = ggml_reshape_2d(ctx, ggml_permute(ctx, heads, 0, 2, 1, 3), E, L);
            x = ggml_add(ctx, residual, linear(ctx, layer.self_o, merged));

            if (tr.has_cross && layer.has_cross) {
                // Text cross-attention for the conditional half only.
                ggml_tensor* cond = ggml_view_2d(ctx, x, E, B, x->nb[1], 0);
                ggml_tensor* uncond =
                    layout_.use_cfg
                        ? ggml_view_2d(ctx, x, E, B, x->nb[1], static_cast<size_t>(B) * x->nb[1])
                        : nullptr;
                ggml_tensor* query =
                    linear(ctx, layer.cross_q, layer_norm(ctx, cond, layer.norm_xattn_query));
                ggml_tensor* qh = ggml_permute(
                    ctx, ggml_reshape_4d(ctx, query, cd_head, c_heads, 1, B), 0, 2, 1, 3);
                auto gather = [&](ggml_tensor* arena_plane) {
                    ggml_tensor* layer_rows = ggml_view_2d(
                        ctx, arena_plane, cross_dim * TC, layout_.arena_slots, arena_plane->nb[2],
                        static_cast<size_t>(il) * arena_plane->nb[1]);
                    return ggml_get_rows(ctx, layer_rows, cross_slots.tensor);  // [CD*TC, B]
                };
                ggml_tensor* kh = ggml_permute(
                    ctx, ggml_reshape_4d(ctx, gather(cross_k), cd_head, c_heads, TC, B), 0, 2, 1,
                    3);                                           // [d, TC, H, B]
                ggml_tensor* scores = ggml_mul_mat(ctx, kh, qh);  // [TC, 1, H, B]
                const bool apply_prior =
                    tr.apply_attention_prior && layer_selected(tr.apply_prior_to_layers, il);
                ggml_tensor* probs = ggml_soft_max_ext(
                    ctx, scores, apply_prior ? prior_bias : mask_bias,
                    1.0f / std::sqrt(static_cast<float>(cd_head)), 0.0f);
                if (layer_selected(tr.estimate_alignment_from_layers, il)) {
                    ggml_tensor* per_layer =
                        c_heads == 1
                            ? ggml_reshape_2d(ctx, probs, TC, B)
                            : ggml_reshape_2d(
                                  ctx,
                                  ggml_sum_rows(
                                      ctx, ggml_cont(ctx, ggml_permute(ctx, probs, 1, 2, 0, 3))),
                                  TC, B);
                    alignment_sum =
                        alignment_sum ? ggml_add(ctx, alignment_sum, per_layer) : per_layer;
                }
                ggml_tensor* vt = ggml_cont(
                    ctx, ggml_permute(
                             ctx, ggml_reshape_4d(ctx, gather(cross_v), cd_head, c_heads, TC, B), 1,
                             2, 0, 3));                                 // [TC, d, H, B]
                ggml_tensor* ctx_heads = ggml_mul_mat(ctx, vt, probs);  // [d, 1, H, B]
                ggml_tensor* attended =
                    ggml_cont_2d(ctx, ggml_permute(ctx, ctx_heads, 0, 2, 1, 3), cross_dim, B);
                cond = ggml_add(ctx, cond, linear(ctx, layer.cross_o, attended));
                x = uncond ? ggml_concat(ctx, cond, uncond, 1) : cond;
            }

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
        ggml_tensor* hidden = ggml_cont(ctx, x);
        ggml_set_name(hidden, "magpietts_batch_decoder_hidden");
        ggml_runtime::TensorBag outputs;
        outputs.add_tensor({hidden, bf_ctx.buft});
        if (alignment_sum) {
            ggml_tensor* mean = ggml_scale(
                ctx, alignment_sum,
                1.0f / static_cast<float>(alignment_layers_ * static_cast<int>(c_heads)));
            ggml_set_name(mean, "magpietts_batch_decoder_alignment");
            outputs.add_tensor({mean, bf_ctx.buft});
        }
        return outputs;
    }

    void set_data(ggml_runtime::Session* session) override {
        auto* tc = session->model_tensor_container.get();
        for (int layer = 0; layer < model_.hparams.n_dec_layer; ++layer) {
            auto arena = tc->get_tensor_by_name(self_arena_name(layer));
            ggml_backend_tensor_memset(arena.tensor, 0, 0, ggml_nbytes(arena.tensor));
        }
        for (const char* name : {kCrossK, kCrossV}) {
            auto t = tc->get_tensor_by_name(name);
            ggml_backend_tensor_memset(t.tensor, 0, 0, ggml_nbytes(t.tensor));
        }
    }

   private:
    const magpietts_model& model_;
    MagpieLaneLayout layout_;
    int cache_len_ = 0;
    int text_capacity_ = 0;
    int alignment_layers_ = 0;
};

MagpieBatchDecoder::MagpieBatchDecoder(
    const magpietts_model& model, const MagpieBatchConfig& config)
    : model_(model), config_(config) {}

MagpieBatchDecoder::~MagpieBatchDecoder() = default;

ggml_backend_t
MagpieBatchDecoder::backend() const {
    return model_.backend;
}

bool
MagpieBatchDecoder::init(std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error) {
            *error = message;
        }
        return false;
    };
#if !defined(NEMO_SPEECH_GGML_PATCHED)
    return fail("the batched Magpie decoder requires the patched ggml (NEMO_SPEECH_GGML_PATCHED)");
#endif
    const magpietts_hparams& h = model_.hparams;
    const magpietts_transformer& tr = model_.decoder;
    if (config_.max_slots < 1) {
        return fail("max_slots must be >= 1");
    }
    if (h.dec_kernel != 1 || tr.n_head <= 0 || tr.n_embd % tr.n_head != 0 || !tr.pos_emb) {
        return fail("the batched decoder needs a kernel-size-1 decoder with positional embeddings");
    }
    if (!tr.has_cross || tr.n_cross_head <= 0 || tr.n_cross_dhead <= 0) {
        return fail("the batched decoder needs a text cross-attention decoder");
    }
    for (const magpietts_layer& layer : tr.layers) {
        if (layer.ff_proj.size() != 1 || layer.ff_out.size() != 1) {
            return fail("the batched decoder needs kernel-size-1 feed-forward layers");
        }
    }
    if (static_cast<int>(model_.audio_embeddings.size()) < h.stacked_audio_codebooks()) {
        return fail("missing audio embeddings");
    }
    layout_.arena_slots = config_.max_slots + 1;  // + dummy slot
    layout_.use_cfg = config_.use_cfg;
    buckets_ = config_.buckets;
    std::sort(buckets_.begin(), buckets_.end());
    buckets_.erase(
        std::remove_if(buckets_.begin(), buckets_.end(), [](int b) { return b < 1; }),
        buckets_.end());
    buckets_.erase(std::unique(buckets_.begin(), buckets_.end()), buckets_.end());
    if (buckets_.empty() || buckets_.back() < config_.max_slots) {
        buckets_.push_back(config_.max_slots);
    }
    const int stacking = std::max(1, static_cast<int>(h.frame_stacking_factor));
    const int steps = config_.max_decoder_positions > 0 ? config_.max_decoder_positions * stacking
                                                        : h.max_decoder_steps;
    max_positions_ = (steps + stacking - 1) / stacking;
    // The prefill holds the baked context plus the BOS frame; every step appends one row and
    // the last step's row is never attended (the request ends after it).
    cache_len_ = h.baked_context_length + max_positions_;
    if (cache_len_ + 1 > h.n_ctx) {
        return fail("decoder positions exceed the model context");
    }
    text_capacity_ = config_.text_capacity;
    if (text_capacity_ < 1 || text_capacity_ > h.n_ctx) {
        return fail("text_capacity must be in [1, n_ctx]");
    }
    slots_.assign(static_cast<size_t>(layout_.arena_slots), SlotState{});

    const bool gpu = magpietts_backend_is_cuda(model_.backend);
    ggml_runtime::Params params;
    params.use_gpu = gpu;
    params.n_threads = std::max(1, config_.threads);
    try {
        backend_manager_ =
            gpu ? std::make_unique<ggml_runtime::BackendManager>(params, model_.backend)
                : std::make_unique<ggml_runtime::BackendManager>(params);
        module_ = std::make_unique<Module>(model_, layout_, cache_len_, text_capacity_);
        session_ =
            std::make_unique<ggml_runtime::Session>(*backend_manager_, module_.get(), nullptr);
        session_->set_run_cache_capacity(buckets_.size() + 1);
        session_->setup();
    }
    catch (const std::exception& e) {
        session_.reset();
        module_.reset();
        backend_manager_.reset();
        return fail(std::string("batched decoder setup failed: ") + e.what());
    }
    // the dummy slot pads steps: one valid (zero) cache row and one (zero) text row
    SlotState& dummy = slots_[static_cast<size_t>(dummySlot())];
    dummy.n_tokens = 1;
    dummy.valid = 1;
    dummy.text_len = 1;
    return true;
}

float*
MagpieBatchDecoder::selfArena(int layer) const {
    return static_cast<float*>(
        session_->model_tensor_container->get_tensor_by_name(self_arena_name(layer)).tensor->data);
}

float*
MagpieBatchDecoder::crossK() const {
    return static_cast<float*>(
        session_->model_tensor_container->get_tensor_by_name(kCrossK).tensor->data);
}

float*
MagpieBatchDecoder::crossV() const {
    return static_cast<float*>(
        session_->model_tensor_container->get_tensor_by_name(kCrossV).tensor->data);
}

bool
MagpieBatchDecoder::seedSlot(int slot, const DecoderKvCache& cond, const DecoderKvCache* uncond) {
    if (!session_ || slot < 0 || slot >= dummySlot()) {
        return false;
    }
    const int n = cond.n_tokens;
    if (n <= 0 || n > cache_len_ || cond.n_embd != model_.hparams.n_embd ||
        cond.n_layers != model_.hparams.n_dec_layer ||
        (layout_.use_cfg &&
         (!uncond || uncond->n_tokens != n || uncond->n_layers != cond.n_layers))) {
        fprintf(stderr, "batched decoder: incompatible prefill caches for slot %d\n", slot);
        return false;
    }
    // two views per copy: layers x lanes x {K, V}
    ggml_init_params params = {
        ggml_tensor_overhead() * static_cast<size_t>(8 * cond.n_layers + 8), nullptr, true};
    ggml_context* ctx = ggml_init(params);
    const size_t f = sizeof(float);
    const size_t rows = static_cast<size_t>(n) * cond.n_embd;
    const size_t src_layer_bytes = static_cast<size_t>(cond.n_ctx) * cond.n_embd * f;
    const size_t dst_row0 = static_cast<size_t>(cache_len_ - n) * cond.n_embd * f;
    const DecoderKvCache* sources[2] = {&cond, layout_.use_cfg ? uncond : nullptr};
    const int lanes[2] = {layout_.condLane(slot), layout_.use_cfg ? layout_.uncondLane(slot) : -1};
    for (int layer = 0; layer < cond.n_layers; ++layer) {
        ggml_tensor* arena =
            session_->model_tensor_container->get_tensor_by_name(self_arena_name(layer)).tensor;
        for (int lane = 0; lane < 2; ++lane) {
            if (!sources[lane]) {
                continue;
            }
            for (int plane = 0; plane < 2; ++plane) {
                ggml_tensor* src_base =
                    plane == 0 ? sources[lane]->memory_k : sources[lane]->memory_v;
                ggml_tensor* src = ggml_view_1d(
                    ctx, src_base, static_cast<int64_t>(rows), layer * src_layer_bytes);
                ggml_tensor* dst = ggml_view_1d(
                    ctx, arena, static_cast<int64_t>(rows),
                    plane * arena->nb[2] + static_cast<size_t>(lanes[lane]) * arena->nb[1] +
                        dst_row0);
                ggml_backend_view_init(src);
                ggml_backend_view_init(dst);
                ggml_backend_tensor_copy(src, dst);
            }
        }
    }
    ggml_free(ctx);
    SlotState& s = slots_[static_cast<size_t>(slot)];
    s.n_tokens = n;
    s.valid = n;
    s.ring_head = 0;
    return true;
}

bool
MagpieBatchDecoder::setSlotCross(int slot, const DecoderCrossKvCache& cross, int text_len) {
    if (!session_ || slot < 0 || slot >= dummySlot() || !cross.memory_k || !cross.memory_v ||
        text_len <= 0 || text_len > text_capacity_ || text_len > cross.text_len) {
        fprintf(
            stderr, "batched decoder: invalid cross cache for slot %d (text_len %d)\n", slot,
            text_len);
        return false;
    }
    const int64_t cross_dim = cross.n_cross_dim;
    const int stride = cross.capacity > 0 ? cross.capacity : cross.text_len;
    ggml_init_params params = {
        ggml_tensor_overhead() * static_cast<size_t>(4 * cross.n_layers + 8), nullptr, true};
    ggml_context* ctx = ggml_init(params);
    const size_t f = sizeof(float);
    for (const char* name : {kCrossK, kCrossV}) {
        ggml_tensor* arena = session_->model_tensor_container->get_tensor_by_name(name).tensor;
        ggml_tensor* src_base = name == kCrossK ? cross.memory_k : cross.memory_v;
        for (int layer = 0; layer < cross.n_layers; ++layer) {
            ggml_tensor* src = ggml_view_1d(
                ctx, src_base, cross_dim * text_len,
                static_cast<size_t>(layer) * stride * cross_dim * f);
            ggml_tensor* dst = ggml_view_1d(
                ctx, arena, cross_dim * text_len,
                static_cast<size_t>(slot) * arena->nb[2] +
                    static_cast<size_t>(layer) * arena->nb[1]);
            ggml_backend_view_init(src);
            ggml_backend_view_init(dst);
            ggml_backend_tensor_copy(src, dst);
        }
    }
    ggml_free(ctx);
    slots_[static_cast<size_t>(slot)].text_len = text_len;
    return true;
}

void
MagpieBatchDecoder::releaseSlot(int slot) {
    if (slot >= 0 && slot < dummySlot()) {
        slots_[static_cast<size_t>(slot)] = SlotState{};
    }
}

int
MagpieBatchDecoder::slotTokens(int slot) const {
    return slot >= 0 && slot < static_cast<int>(slots_.size())
               ? slots_[static_cast<size_t>(slot)].n_tokens
               : 0;
}

int
MagpieBatchDecoder::slotRingHead(int slot) const {
    return slots_.at(static_cast<size_t>(slot)).ring_head;
}

int
MagpieBatchDecoder::slotValid(int slot) const {
    return slots_.at(static_cast<size_t>(slot)).valid;
}

void
MagpieBatchDecoder::advanceSlot(int slot) {
    SlotState& s = slots_.at(static_cast<size_t>(slot));
    ++s.n_tokens;
    s.valid = std::min(cache_len_, s.valid + 1);
    s.ring_head = (s.ring_head + 1) % cache_len_;
}

bool
MagpieBatchDecoder::step(
    const std::vector<MagpieBatchDecoderRow>& rows, int bucket, ggml_runtime::DeviceTensor& hidden,
    std::vector<std::vector<float>>& alignment) {
    const ggml_nvtx::range nvtx_range("magpietts_batch_decoder_step");
    const magpietts_hparams& h = model_.hparams;
    const int B = static_cast<int>(rows.size());
    if (!session_ || B < 1 || bucket < B ||
        std::find(buckets_.begin(), buckets_.end(), bucket) == buckets_.end()) {
        fprintf(stderr, "batched decoder: %d rows do not fit bucket %d\n", B, bucket);
        return false;
    }
    const int n_cb = h.stacked_audio_codebooks();
    const int L = lanes(bucket);
    const int TC = text_capacity_;
    tokens_.assign(static_cast<size_t>(n_cb) * bucket, 0);
    positions_.assign(static_cast<size_t>(bucket), 0);
    cross_slots_.assign(static_cast<size_t>(bucket), dummySlot());
    lane_slots_.assign(static_cast<size_t>(L), 0);
    cache_meta_.assign(static_cast<size_t>(L) * 2, 0);
    mask_.assign(static_cast<size_t>(TC) * bucket, kPadBias);
    prior_.assign(static_cast<size_t>(TC) * bucket, kPadBias);
    bool any_alignment = false;
    for (int b = 0; b < bucket; ++b) {
        const bool real = b < B;
        const int slot = real ? rows[static_cast<size_t>(b)].slot : dummySlot();
        if (real) {
            const MagpieBatchDecoderRow& row = rows[static_cast<size_t>(b)];
            if (slot < 0 || slot >= dummySlot()) {
                fprintf(stderr, "batched decoder: row %d has invalid slot %d\n", b, slot);
                return false;
            }
            const SlotState& s = slots_[static_cast<size_t>(slot)];
            if (s.n_tokens <= 0 || !row.tokens || row.text_len <= 0 || row.text_len > s.text_len ||
                s.n_tokens >= h.n_ctx) {
                fprintf(stderr, "batched decoder: row %d (slot %d) is not ready\n", b, slot);
                return false;
            }
            for (int c = 0; c < n_cb; ++c) {
                tokens_[static_cast<size_t>(c) * bucket + b] = row.tokens[c];
            }
            any_alignment = any_alignment || row.collect_alignment;
        }
        const SlotState& s = slots_[static_cast<size_t>(slot)];
        const int text_len = real ? rows[static_cast<size_t>(b)].text_len : 1;
        positions_[static_cast<size_t>(b)] = real ? s.n_tokens : 0;
        cross_slots_[static_cast<size_t>(b)] = slot;
        const int lane_count = layout_.lanesPerSlot();
        for (int lane = 0; lane < lane_count; ++lane) {
            const int col = lane * bucket + b;
            lane_slots_[static_cast<size_t>(col)] =
                lane == 0 ? layout_.condLane(slot) : layout_.uncondLane(slot);
            cache_meta_[static_cast<size_t>(col)] = s.ring_head;
            cache_meta_[static_cast<size_t>(L + col)] = s.valid;
        }
        float* mask_row = mask_.data() + static_cast<size_t>(b) * TC;
        float* prior_row = prior_.data() + static_cast<size_t>(b) * TC;
        const std::vector<float>* p = real ? rows[static_cast<size_t>(b)].prior : nullptr;
        for (int j = 0; j < text_len; ++j) {
            mask_row[j] = 0.0f;
            prior_row[j] = p && j < static_cast<int>(p->size())
                               ? std::log(std::max((*p)[static_cast<size_t>(j)], 1.0e-20f))
                               : 0.0f;
        }
    }

    using Input = ggml_runtime::Session::Input;
    std::vector<Input> inputs = {
        {"magpietts.batch.tokens", GGML_TYPE_I32, tokens_.data(), {bucket, n_cb}},
        {"magpietts.batch.positions", GGML_TYPE_I32, positions_.data(), {bucket}},
        {"magpietts.batch.lane_slots", GGML_TYPE_I32, lane_slots_.data(), {L}},
        {"magpietts.batch.cache_meta", GGML_TYPE_I32, cache_meta_.data(), {L, 2}},
        {"magpietts.batch.cross_slots", GGML_TYPE_I32, cross_slots_.data(), {bucket}},
        {"magpietts.batch.prior", GGML_TYPE_F32, prior_.data(), {TC, bucket}},
        {"magpietts.batch.mask", GGML_TYPE_F32, mask_.data(), {TC, bucket}},
    };
    const bool has_alignment = module_->alignmentLayers() > 0;
    std::vector<ggml_runtime::Session::Output> outputs(has_alignment && any_alignment ? 2 : 1);
    outputs[0].index = 0;
    outputs[0].device_tensor = &hidden;
    if (outputs.size() > 1) {
        alignment_.assign(static_cast<size_t>(TC) * bucket, 0.0f);
        outputs[1].index = 1;
        outputs[1].host_buffer = alignment_.data();
        outputs[1].nbytes = alignment_.size() * sizeof(float);
    }
    try {
        session_->run(inputs, outputs);
    }
    catch (const std::exception& e) {
        fprintf(stderr, "batched decoder step failed: %s\n", e.what());
        return false;
    }
    alignment.resize(static_cast<size_t>(B));
    for (int b = 0; b < B; ++b) {
        const MagpieBatchDecoderRow& row = rows[static_cast<size_t>(b)];
        std::vector<float>& out = alignment[static_cast<size_t>(b)];
        if (row.collect_alignment && outputs.size() > 1) {
            const float* src = alignment_.data() + static_cast<size_t>(b) * TC;
            out.assign(src, src + row.text_len);
        } else {
            out.clear();
        }
        advanceSlot(row.slot);
    }
    return true;
}

}  // namespace nemo_speech::tts
