// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "batch_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "ggml-alloc.h"
#include "nvtx_utils.h"
#if defined(MAGPIETTS_CUDA_SAMPLING)
#include <cuda_runtime.h>

#include "ggml-cuda.h"
#include "magpietts_batch_fused.h"
#endif

namespace nemo_speech::tts {

namespace {

bool
layer_selected(const std::vector<int32_t>& layers, int layer) {
    return layers.empty() ||
           std::find(layers.begin(), layers.end(), static_cast<int32_t>(layer)) != layers.end();
}

}  // namespace

#if defined(MAGPIETTS_CUDA_SAMPLING)
// Fused CUDA step: the batched persistent decoder and local-transformer kernels, with the step's
// per-row inputs uploaded in one copy and the codes read back in one.
struct MagpieBatchModelBackend::Fused {
    magpietts_batch_decoder_fused* decoder = nullptr;
    magpietts_batch_lt_fused* lt = nullptr;
    cudaStream_t stream = nullptr;
    int max_rows = 0, max_lanes = 0, n_cb = 0, text_capacity = 0, E = 0;
    // device / pinned host mirrors of the step inputs, one block
    char* d_in = nullptr;
    char* h_in = nullptr;
    size_t in_bytes = 0;
    size_t off_tokens = 0, off_positions = 0, off_arena_lane = 0, off_ring = 0, off_valid = 0,
           off_cross_slot = 0, off_text_len = 0, off_prior = 0, off_mask = 0, off_configs = 0;
    // device / pinned host outputs: codes, argmax, then alignment
    char* d_out = nullptr;
    char* h_out = nullptr;
    size_t out_bytes = 0, off_argmax = 0, off_align = 0;
    float* hidden = nullptr;  // [lanes][E]

    ~Fused() {
        if (decoder)
            magpietts_batch_decoder_fused_free(decoder);
        if (lt)
            magpietts_batch_lt_fused_free(lt);
        cudaFree(d_in);
        cudaFreeHost(h_in);
        cudaFree(d_out);
        cudaFreeHost(h_out);
        cudaFree(hidden);
    }
    template <class T>
    T* in(size_t off) {
        return reinterpret_cast<T*>(h_in + off);
    }
    template <class T>
    T* dev(size_t off) const {
        return reinterpret_cast<T*>(d_in + off);
    }
};
#else
struct MagpieBatchModelBackend::Fused {};
#endif

MagpieBatchModelBackend::MagpieBatchModelBackend(
    const magpietts_model& model, const MagpieBatchConfig& config)
    : model_(model), config_(config), decoder_(model, config),
      lt_(model, config.use_cfg, config.threads), prefill_decoder_(model) {}

MagpieBatchModelBackend::~MagpieBatchModelBackend() {
    if (prefill_allocr_)
        ggml_gallocr_free(prefill_allocr_);
    if (cross_allocr_)
        ggml_gallocr_free(cross_allocr_);
}

bool
MagpieBatchModelBackend::init(std::string* error) {
    if (!decoder_.init(error) || !lt_.init(error)) {
        return false;
    }
    const int E = model_.hparams.n_embd;
    if (!hidden_cond_.alloc2d(model_, GGML_TYPE_F32, E, 1, "magpietts_batch_prefill_hidden_cond") ||
        !hidden_uncond_.alloc2d(
            model_, GGML_TYPE_F32, E, 1, "magpietts_batch_prefill_hidden_uncond")) {
        if (error)
            *error = "prefill hidden-state allocation failed";
        return false;
    }
    if (config_.use_fused_kernels && magpietts_backend_is_cuda(model_.backend)) {
        std::string reason;
        if (initFused(&reason)) {
            fprintf(
                stderr, "MagpieTTS batch: fused CUDA decoder + local-transformer kernels (%d slots)\n",
                config_.max_slots);
        } else {
            fprintf(stderr, "MagpieTTS batch: using the ggml graphs (%s)\n", reason.c_str());
        }
    }
    return true;
}

bool
MagpieBatchModelBackend::initFused(std::string* reason) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    auto fail = [&](const std::string& why) {
        *reason = why;
        fused_.reset();
        return false;
    };
    const magpietts_hparams& h = model_.hparams;
    const magpietts_transformer& tr = model_.decoder;
    const int lanes_per = config_.use_cfg ? 2 : 1;
    if (config_.max_slots * lanes_per > MAGPIETTS_BATCH_FUSED_MAX_LANES) {
        return fail("too many slots for the fused kernels");
    }
    if (tr.n_cross_head != 1) {
        return fail("the fused decoder needs one cross-attention head");
    }
    if (h.top_k > MAGPIETTS_BATCH_FUSED_MAX_TOPK) {
        return fail("top-k above the in-kernel sampler limit");
    }
    std::vector<magpietts_decoder_fused_layer_weights> layers(tr.layers.size());
    for (size_t i = 0; i < tr.layers.size(); ++i) {
        const magpietts_layer& L = tr.layers[i];
        magpietts_decoder_fused_layer_weights& d = layers[i];
        if (!L.self_qkv || !L.self_o || !L.norm_self || !L.norm_ff || L.ff_proj.size() != 1 ||
            L.ff_out.size() != 1 || !L.has_cross || !L.cross_q || !L.cross_o || !L.norm_xattn_query) {
            return fail("unsupported decoder layout");
        }
        d.qkv = L.self_qkv->data;
        d.qkv_type = (int)L.self_qkv->type;
        d.o = L.self_o->data;
        d.o_type = (int)L.self_o->type;
        d.cross_q = L.cross_q->data;
        d.cross_q_type = (int)L.cross_q->type;
        d.cross_o = L.cross_o->data;
        d.cross_o_type = (int)L.cross_o->type;
        d.ff1 = L.ff_proj[0]->data;
        d.ff1_type = (int)L.ff_proj[0]->type;
        d.ff2 = L.ff_out[0]->data;
        d.ff2_type = (int)L.ff_out[0]->type;
        d.norm_self = (const float*)L.norm_self->data;
        d.norm_xq = (const float*)L.norm_xattn_query->data;
        d.norm_ff = (const float*)L.norm_ff->data;
        d.has_cross = true;
        d.collect_alignment = layer_selected(tr.estimate_alignment_from_layers, (int)i);
        d.apply_prior = tr.apply_attention_prior && layer_selected(tr.apply_prior_to_layers, (int)i);
    }
    std::vector<const void*> emb(static_cast<size_t>(h.stacked_audio_codebooks()));
    for (int c = 0; c < h.stacked_audio_codebooks(); ++c) {
        if (model_.audio_embeddings[c]->type != model_.audio_embeddings[0]->type)
            return fail("mixed audio embedding types");
        emb[c] = model_.audio_embeddings[c]->data;
    }
    magpietts_decoder_fused_weights w{};
    w.n_embd = tr.n_embd;
    w.n_head = tr.n_head;
    w.n_ff = h.n_ffn;
    w.cross_dim = tr.n_cross_head * tr.n_cross_dhead;
    w.n_layers = (int)tr.layers.size();
    w.n_codebooks = h.stacked_audio_codebooks();
    w.ln_eps = MAGPIETTS_LN_EPS;
    w.audio_emb = emb.data();
    w.emb_type = (int)model_.audio_embeddings[0]->type;
    w.pos_emb = tr.pos_emb->data;
    w.pos_type = (int)tr.pos_emb->type;
    w.norm_out = tr.norm_out ? (const float*)tr.norm_out->data : nullptr;
    w.layers = layers.data();
    std::vector<float*> arena(tr.layers.size());
    for (size_t i = 0; i < tr.layers.size(); ++i) arena[i] = decoder_.selfArena((int)i);
    magpietts_batch_decoder_fused_cache cache{};
    cache.cache_len = decoder_.cacheLen();
    cache.arena_lanes = decoder_.layout().arenaLanes();
    cache.kv_arena = arena.data();
    cache.text_capacity = decoder_.textCapacity();
    cache.arena_slots = decoder_.layout().arena_slots;
    cache.cross_k = decoder_.crossK();
    cache.cross_v = decoder_.crossV();
    auto f = std::make_unique<Fused>();
    char error[256] = {};
    f->decoder = magpietts_batch_decoder_fused_create(w, cache, error, sizeof(error));
    if (!f->decoder) {
        return fail(error);
    }

    // local transformer
    const magpietts_transformer& lt = model_.local;
    if (model_.lt_in_w || lt.kernel != 1 || lt.norm_out || lt.layers.empty() || lt.n_embd != h.n_embd ||
        !lt.pos_emb) {
        return fail("unsupported local-transformer layout");
    }
    const int rounds = h.stacked_audio_codebooks();
    std::vector<magpietts_lt_fused_layer_weights> lt_layers(lt.layers.size());
    for (size_t il = 0; il < lt.layers.size(); ++il) {
        const magpietts_layer& L = lt.layers[il];
        if (!L.self_qkv || !L.self_o || L.ff_proj.size() != 1 || L.ff_out.size() != 1 ||
            !L.norm_self || !L.norm_ff) {
            return fail("unsupported local-transformer layer");
        }
        lt_layers[il].qkv = L.self_qkv->data;
        lt_layers[il].qkv_type = (int)L.self_qkv->type;
        lt_layers[il].o = L.self_o->data;
        lt_layers[il].o_type = (int)L.self_o->type;
        lt_layers[il].ff1 = L.ff_proj[0]->data;
        lt_layers[il].ff1_type = (int)L.ff_proj[0]->type;
        lt_layers[il].ff2 = L.ff_out[0]->data;
        lt_layers[il].ff2_type = (int)L.ff_out[0]->type;
        lt_layers[il].norm_self = (const float*)L.norm_self->data;
        lt_layers[il].norm_ff = (const float*)L.norm_ff->data;
    }
    std::vector<const void*> lt_emb(rounds), out_w(rounds);
    std::vector<const float*> out_b(rounds);
    for (int r = 0; r < rounds; ++r) {
        lt_emb[r] = model_.audio_embeddings[r]->data;
        out_w[r] = model_.lt_out_w[r]->data;
        out_b[r] = r < (int)model_.lt_out_b.size() && model_.lt_out_b[r]
                       ? (const float*)model_.lt_out_b[r]->data
                       : nullptr;
    }
    magpietts_lt_fused_weights lw{};
    lw.n_embd = lt.n_embd;
    lw.n_head = lt.n_head;
    lw.n_ff = (int)lt.layers[0].ff_proj[0]->ne[1];
    lw.n_layers = (int)lt.layers.size();
    lw.vocab = h.audio_vocab_size;
    lw.n_rounds = rounds;
    lw.pos_rows = (int)lt.pos_emb->ne[1];
    lw.ln_eps = MAGPIETTS_LN_EPS;
    lw.layers = lt_layers.data();
    lw.pos_emb = lt.pos_emb->data;
    lw.pos_type = (int)lt.pos_emb->type;
    lw.audio_emb = lt_emb.data();
    lw.audio_emb_type = (int)model_.audio_embeddings[0]->type;
    lw.out_w = out_w.data();
    lw.out_type = (int)model_.lt_out_w[0]->type;
    lw.out_b = out_b.data();
    f->lt = magpietts_batch_lt_fused_create(lw, error, sizeof(error));
    if (!f->lt) {
        return fail(error);
    }

    f->stream = (cudaStream_t)ggml_backend_cuda_get_stream(model_.backend);
    f->max_rows = config_.max_slots;
    f->max_lanes = config_.max_slots * lanes_per;
    f->n_cb = rounds;
    f->text_capacity = decoder_.textCapacity();
    f->E = h.n_embd;
    const size_t R = (size_t)f->max_rows, L = (size_t)f->max_lanes, TC = (size_t)f->text_capacity;
    size_t off = 0;
    auto take = [&](size_t bytes) {
        const size_t at = off;
        off += (bytes + 255) & ~(size_t)255;
        return at;
    };
    f->off_tokens = take(R * rounds * sizeof(int32_t));
    f->off_positions = take(R * sizeof(int32_t));
    f->off_arena_lane = take(L * sizeof(int32_t));
    f->off_ring = take(L * sizeof(int32_t));
    f->off_valid = take(L * sizeof(int32_t));
    f->off_cross_slot = take(R * sizeof(int32_t));
    f->off_text_len = take(R * sizeof(int32_t));
    f->off_prior = take(R * TC * sizeof(float));
    f->off_mask = take(R * TC * sizeof(float));
    f->off_configs = take(R * magpietts_batch_fused_config_size());
    f->in_bytes = off;
    off = 0;
    take(R * rounds * sizeof(int32_t));
    f->off_argmax = take(R * rounds * sizeof(int32_t));
    f->off_align = take(R * TC * sizeof(float));
    f->out_bytes = off;
    if (cudaMalloc(&f->d_in, f->in_bytes) != cudaSuccess ||
        cudaMallocHost(&f->h_in, f->in_bytes) != cudaSuccess ||
        cudaMalloc(&f->d_out, f->out_bytes) != cudaSuccess ||
        cudaMallocHost(&f->h_out, f->out_bytes) != cudaSuccess ||
        cudaMalloc(&f->hidden, (L + 1) * h.n_embd * sizeof(float)) != cudaSuccess) {
        (void)cudaGetLastError();
        return fail("fused step buffer allocation failed");
    }
    fused_ = std::move(f);
    return true;
#else
    *reason = "CUDA sampling was not compiled in";
    return false;
#endif
}

bool
MagpieBatchModelBackend::prefill(int slot, const MagpieBatchRowInput& row, MagpieBatchRowOutput& out) {
    const ggml_nvtx::range nvtx_range("magpietts_batch_backend_prefill");
    const magpietts_hparams& h = model_.hparams;
    if (row.text_len <= 0 || row.text_len > decoder_.textCapacity() ||
        (!row.text_cond_device && (!row.text_cond || row.text_cond->empty()))) {
        fprintf(stderr, "MagpieTTS batch: invalid prefill input (text_len %d)\n", row.text_len);
        return false;
    }
    static const std::vector<float> no_text;
    const std::vector<float>& text_cond = row.text_cond ? *row.text_cond : no_text;
    const magpietts_backend_tensor* text_device =
        row.text_cond_device && row.text_cond_device->tensor ? row.text_cond_device : nullptr;
    std::vector<std::vector<int32_t>> bos(static_cast<size_t>(h.audio_codebooks));
    for (auto& codes : bos) codes.assign(static_cast<size_t>(h.frame_stacking_factor), h.audio_bos_id);
    out.alignment.clear();
    magpietts_decoder_attention attention;
    attention.prior = row.prior;
    attention.alignment_scores = row.collect_alignment ? &out.alignment : nullptr;
    const magpietts_decoder_attention* attention_arg =
        (attention.prior || attention.alignment_scores) ? &attention : nullptr;
    cross_.clear();
    bool ok = false;
    if (config_.use_cfg) {
        ok = prefill_decoder_.prefillPair(
            text_cond, row.text_len, bos, row.speaker, config_.threads, cond_kv_, uncond_kv_,
            text_device, &hidden_cond_, &hidden_uncond_, &cross_, attention_arg, staging_, nullptr,
            &prefill_allocr_, &cross_allocr_);
    } else {
        cond_kv_.clear();
        decoder_result result;
        result.logits_required = false;
        ok = prefill_decoder_.evalCached(
            text_cond, row.text_len, bos, row.speaker, true, config_.threads, cond_kv_, result,
            nullptr, text_device, &hidden_cond_, &cross_, attention_arg);
    }
    if (!ok || !decoder_.seedSlot(slot, cond_kv_, config_.use_cfg ? &uncond_kv_ : nullptr) ||
        !decoder_.setSlotCross(slot, cross_, row.text_len)) {
        fprintf(stderr, "MagpieTTS batch: prefill failed for slot %d\n", slot);
        return false;
    }
    if (fused_) {
        std::vector<const MagpieBatchRowInput*> rows = {&row};
        std::vector<MagpieBatchRowOutput> outs;
        if (!sampleFused(1, rows, outs, true)) {
            return false;
        }
        out.codes = std::move(outs[0].codes);
        out.argmax = std::move(outs[0].argmax);
        return true;
    }
    std::vector<MagpieBatchSampling> sampling(1);
    sampling[0].temperature = row.temperature;
    sampling[0].cfg_scale = row.cfg_scale;
    sampling[0].top_k = row.top_k;
    sampling[0].forbid_eos = row.forbid_eos;
    sampling[0].rng = row.rng;
    std::vector<std::vector<int32_t>> codes, argmax;
    const bool cfg = config_.use_cfg;
    const bool sampled = lt_.sample(
        1, sampling,
        [&](ggml_tensor* input) {
            ggml_init_params params = {ggml_tensor_overhead() * 2, nullptr, true};
            ggml_context* ctx = ggml_init(params);
            const size_t col = input->nb[1];
            for (int lane = 0; lane < (cfg ? 2 : 1); ++lane) {
                ggml_tensor* dst = ggml_view_1d(ctx, input, model_.hparams.n_embd, lane * col);
                ggml_backend_view_init(dst);
                // stream-ordered after the prefill graph on the model backend
                ggml_backend_tensor_copy_async(
                    model_.backend, model_.backend,
                    lane == 0 ? hidden_cond_.tensor : hidden_uncond_.tensor, dst);
            }
            ggml_free(ctx);
            return true;
        },
        codes, argmax);
    if (!sampled) {
        return false;
    }
    out.codes = std::move(codes[0]);
    out.argmax = std::move(argmax[0]);
    return true;
}

bool
MagpieBatchModelBackend::decode(
    const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
    std::vector<MagpieBatchRowOutput>& out) {
    if (fused_) {
        return decodeFused(slots, rows, out);
    }
    const int n = static_cast<int>(rows.size());
    const int bucket = magpie_batch_bucket(decoder_.buckets(), n);
    if (n < 1 || static_cast<int>(slots.size()) != n || bucket < 0) {
        return false;
    }
    std::vector<MagpieBatchDecoderRow> decoder_rows(static_cast<size_t>(n));
    std::vector<MagpieBatchSampling> sampling(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const MagpieBatchRowInput& in = *rows[static_cast<size_t>(i)];
        if (static_cast<int>(in.tokens.size()) != model_.hparams.stacked_audio_codebooks()) {
            fprintf(stderr, "MagpieTTS batch: row %d has %zu codes\n", i, in.tokens.size());
            return false;
        }
        MagpieBatchDecoderRow& r = decoder_rows[static_cast<size_t>(i)];
        r.slot = slots[static_cast<size_t>(i)];
        r.tokens = in.tokens.data();
        r.text_len = in.text_len;
        r.prior = in.prior;
        r.collect_alignment = in.collect_alignment;
        MagpieBatchSampling& s = sampling[static_cast<size_t>(i)];
        s.temperature = in.temperature;
        s.cfg_scale = in.cfg_scale;
        s.top_k = in.top_k;
        s.forbid_eos = in.forbid_eos;
        s.rng = in.rng;
    }
    ggml_runtime::DeviceTensor hidden;
    std::vector<std::vector<float>> alignment;
    if (!decoder_.step(decoder_rows, bucket, hidden, alignment)) {
        return false;
    }
    std::vector<std::vector<int32_t>> codes, argmax;
    if (!lt_.sample(
            bucket, sampling,
            [&](ggml_tensor* input) {
                if (ggml_nbytes(input) != ggml_nbytes(hidden.tensor)) {
                    fprintf(stderr, "MagpieTTS batch: decoder/LT lane mismatch\n");
                    return false;
                }
                // the decoder graph ran asynchronously on the same backend stream
                ggml_backend_tensor_copy_async(model_.backend, model_.backend, hidden.tensor, input);
                return true;
            },
            codes, argmax)) {
        return false;
    }
    out.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)].codes = std::move(codes[static_cast<size_t>(i)]);
        out[static_cast<size_t>(i)].argmax = std::move(argmax[static_cast<size_t>(i)]);
        out[static_cast<size_t>(i)].alignment = std::move(alignment[static_cast<size_t>(i)]);
    }
    return true;
}

bool
MagpieBatchModelBackend::sampleFused(
    int rows, const std::vector<const MagpieBatchRowInput*>& in, std::vector<MagpieBatchRowOutput>& out,
    bool from_prefill) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    Fused& f = *fused_;
    const int n_cb = f.n_cb;
    if (from_prefill) {
        // prefill: the first frame's hidden states come from the prefill scratch
        const size_t bytes = (size_t)f.E * sizeof(float);
        if (cudaMemcpyAsync(f.hidden, hidden_cond_.tensor->data, bytes, cudaMemcpyDeviceToDevice, f.stream) !=
                cudaSuccess ||
            (config_.use_cfg &&
             cudaMemcpyAsync(f.hidden + f.E, hidden_uncond_.tensor->data, bytes, cudaMemcpyDeviceToDevice,
                             f.stream) != cudaSuccess)) {
            return false;
        }
    }
    for (int b = 0; b < rows; ++b) {
        const MagpieBatchRowInput& r = *in[static_cast<size_t>(b)];
        magpietts_batch_fused_write_config(
            f.h_in + f.off_configs, b, config_.use_cfg, r.cfg_scale, r.temperature, r.top_k,
            r.forbid_eos, r.seed, r.frame_index);
    }
    const size_t cfg_bytes = (size_t)rows * magpietts_batch_fused_config_size();
    if (cudaMemcpyAsync(f.d_in + f.off_configs, f.h_in + f.off_configs, cfg_bytes, cudaMemcpyHostToDevice,
                        f.stream) != cudaSuccess) {
        return false;
    }
    magpietts_batch_lt_fused_args la{};
    la.rows = rows;
    la.lanes = rows * (config_.use_cfg ? 2 : 1);
    la.hidden = f.hidden;
    la.configs = f.d_in + f.off_configs;
    la.codes = reinterpret_cast<int32_t*>(f.d_out);
    la.argmax = reinterpret_cast<int32_t*>(f.d_out + f.off_argmax);
    la.audio_codebook_size = model_.hparams.audio_codebook_size;
    la.audio_eos_id = model_.hparams.audio_eos_id;
    char error[256] = {};
    if (!magpietts_batch_lt_fused_run(f.lt, f.stream, la, error, sizeof(error))) {
        fprintf(stderr, "MagpieTTS batch: %s\n", error);
        return false;
    }
    const size_t code_bytes = (size_t)rows * n_cb * sizeof(int32_t);
    if (cudaMemcpyAsync(f.h_out, f.d_out, code_bytes, cudaMemcpyDeviceToHost, f.stream) != cudaSuccess ||
        cudaMemcpyAsync(f.h_out + f.off_argmax, f.d_out + f.off_argmax, code_bytes, cudaMemcpyDeviceToHost,
                        f.stream) != cudaSuccess ||
        cudaStreamSynchronize(f.stream) != cudaSuccess) {
        return false;
    }
    out.resize(static_cast<size_t>(rows));
    const int32_t* codes = reinterpret_cast<const int32_t*>(f.h_out);
    const int32_t* argmax = reinterpret_cast<const int32_t*>(f.h_out + f.off_argmax);
    for (int b = 0; b < rows; ++b) {
        out[b].codes.assign(codes + (size_t)b * n_cb, codes + (size_t)(b + 1) * n_cb);
        out[b].argmax.assign(argmax + (size_t)b * n_cb, argmax + (size_t)(b + 1) * n_cb);
    }
    return true;
#else
    (void)rows;
    (void)in;
    (void)out;
    (void)from_prefill;
    return false;
#endif
}

bool
MagpieBatchModelBackend::decodeFused(
    const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
    std::vector<MagpieBatchRowOutput>& out) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    const ggml_nvtx::range nvtx_range("magpietts_batch_fused_step");
    Fused& f = *fused_;
    const int B = static_cast<int>(rows.size());
    const int lanes_per = config_.use_cfg ? 2 : 1;
    const int L = B * lanes_per;
    const int TC = f.text_capacity;
    if (B < 1 || B > f.max_rows || static_cast<int>(slots.size()) != B) {
        return false;
    }
    const MagpieLaneLayout& layout = decoder_.layout();
    int32_t* tokens = f.in<int32_t>(f.off_tokens);
    int32_t* positions = f.in<int32_t>(f.off_positions);
    int32_t* arena_lane = f.in<int32_t>(f.off_arena_lane);
    int32_t* ring = f.in<int32_t>(f.off_ring);
    int32_t* valid = f.in<int32_t>(f.off_valid);
    int32_t* cross_slot = f.in<int32_t>(f.off_cross_slot);
    int32_t* text_len = f.in<int32_t>(f.off_text_len);
    float* prior = f.in<float>(f.off_prior);
    float* mask = f.in<float>(f.off_mask);
    bool any_alignment = false;
    for (int b = 0; b < B; ++b) {
        const MagpieBatchRowInput& r = *rows[static_cast<size_t>(b)];
        const int slot = slots[static_cast<size_t>(b)];
        if (static_cast<int>(r.tokens.size()) != f.n_cb || r.text_len <= 0 || r.text_len > TC ||
            decoder_.slotTokens(slot) <= 0) {
            fprintf(stderr, "MagpieTTS batch: row %d (slot %d) is not ready\n", b, slot);
            return false;
        }
        std::memcpy(tokens + (size_t)b * f.n_cb, r.tokens.data(), (size_t)f.n_cb * sizeof(int32_t));
        positions[b] = decoder_.slotTokens(slot);
        cross_slot[b] = slot;
        text_len[b] = r.text_len;
        for (int lane = 0; lane < lanes_per; ++lane) {
            const int l = lane * B + b;
            arena_lane[l] = lane == 0 ? layout.condLane(slot) : layout.uncondLane(slot);
            ring[l] = decoder_.slotRingHead(slot);
            valid[l] = decoder_.slotValid(slot);
        }
        float* prior_row = prior + (size_t)b * TC;
        float* mask_row = mask + (size_t)b * TC;
        for (int j = 0; j < TC; ++j) {
            const bool real = j < r.text_len;
            mask_row[j] = real ? 0.0f : -1.0e30f;
            prior_row[j] = !real ? -1.0e30f
                           : r.prior && j < static_cast<int>(r.prior->size())
                               ? std::log(std::max((*r.prior)[static_cast<size_t>(j)], 1.0e-20f))
                               : 0.0f;
        }
        any_alignment = any_alignment || r.collect_alignment;
        magpietts_batch_fused_write_config(
            f.h_in + f.off_configs, b, config_.use_cfg, r.cfg_scale, r.temperature, r.top_k,
            r.forbid_eos, r.seed, r.frame_index);
    }
    if (cudaMemcpyAsync(f.d_in, f.h_in, f.in_bytes, cudaMemcpyHostToDevice, f.stream) != cudaSuccess) {
        return false;
    }
    magpietts_batch_decoder_fused_step_args st{};
    st.rows = B;
    st.lanes = L;
    st.tokens = f.dev<int32_t>(f.off_tokens);
    st.positions = f.dev<int32_t>(f.off_positions);
    st.arena_lane = f.dev<int32_t>(f.off_arena_lane);
    st.ring_head = f.dev<int32_t>(f.off_ring);
    st.valid_len = f.dev<int32_t>(f.off_valid);
    st.cross_slot = f.dev<int32_t>(f.off_cross_slot);
    st.text_len = f.dev<int32_t>(f.off_text_len);
    st.prior = f.dev<float>(f.off_prior);
    st.mask = f.dev<float>(f.off_mask);
    st.hidden = f.hidden;
    st.alignment = any_alignment ? reinterpret_cast<float*>(f.d_out + f.off_align) : nullptr;
    char error[256] = {};
    if (!magpietts_batch_decoder_fused_step(f.decoder, f.stream, st, error, sizeof(error))) {
        fprintf(stderr, "MagpieTTS batch: %s\n", error);
        return false;
    }
    if (any_alignment &&
        cudaMemcpyAsync(f.h_out + f.off_align, f.d_out + f.off_align, (size_t)B * TC * sizeof(float),
                        cudaMemcpyDeviceToHost, f.stream) != cudaSuccess) {
        return false;
    }
    // sampleFused uploads the configs again (already current) and reads the codes back
    if (!sampleFused(B, rows, out, false)) {
        return false;
    }
    const float* align = reinterpret_cast<const float*>(f.h_out + f.off_align);
    for (int b = 0; b < B; ++b) {
        const MagpieBatchRowInput& r = *rows[static_cast<size_t>(b)];
        if (r.collect_alignment) {
            out[b].alignment.assign(align + (size_t)b * TC, align + (size_t)b * TC + r.text_len);
        }
        decoder_.advanceSlot(slots[static_cast<size_t>(b)]);
    }
    return true;
#else
    (void)slots;
    (void)rows;
    (void)out;
    return false;
#endif
}

void
MagpieBatchModelBackend::kernelTimingReport() const {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    magpietts_batch_fused_timing_report();
#endif
}

void
MagpieBatchModelBackend::release(int slot) {
    decoder_.releaseSlot(slot);
}

}  // namespace nemo_speech::tts
