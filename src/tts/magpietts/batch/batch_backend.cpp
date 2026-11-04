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
    bool owns_stream = false;
    cudaEvent_t ready = nullptr;     // orders a secondary stream after the model backend's stream
    bool pending_alignment = false;  // the launched chunk copied alignments back

    ~Fused() {
        if (ready)
            cudaEventDestroy(ready);
        if (decoder)
            magpietts_batch_decoder_fused_free(decoder);
        if (lt)
            magpietts_batch_lt_fused_free(lt);
        cudaFree(d_in);
        cudaFreeHost(h_in);
        cudaFree(d_out);
        cudaFreeHost(h_out);
        cudaFree(hidden);
        if (owns_stream && stream)
            cudaStreamDestroy(stream);
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
      lt_(model, config.use_cfg, config.threads), prefill_decoder_(model), encoder_(model) {}

MagpieBatchModelBackend::~MagpieBatchModelBackend() {
    if (prefill_allocr_)
        ggml_gallocr_free(prefill_allocr_);
    if (cross_allocr_)
        ggml_gallocr_free(cross_allocr_);
    if (prefill_batch_allocr_)
        ggml_gallocr_free(prefill_batch_allocr_);
    if (encode_batch_allocr_)
        ggml_gallocr_free(encode_batch_allocr_);
#if defined(MAGPIETTS_CUDA_SAMPLING)
    if (prefill_ready_)
        cudaEventDestroy((cudaEvent_t)prefill_ready_);
#endif
    if (prefill_backend_)
        ggml_backend_free(prefill_backend_);
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
                stderr,
                "MagpieTTS batch: fused CUDA decoder + local-transformer kernels (%d slots)\n",
                config_.max_slots);
        } else {
            fprintf(stderr, "MagpieTTS batch: using the ggml graphs (%s)\n", reason.c_str());
        }
    }
    if (config_.prefill_stream && config_.prefill_batch_size > 1) {
        std::string reason;
        if (initPrefillStream(&reason)) {
            fprintf(stderr, "MagpieTTS batch: batched prefills on their own stream\n");
        } else {
            fprintf(
                stderr, "MagpieTTS batch: prefills on the decode stream (%s)\n", reason.c_str());
        }
    }
    return true;
}

bool
MagpieBatchModelBackend::initPrefillStream(std::string* reason) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    if (!magpietts_backend_is_cuda(model_.backend) || !config_.use_cfg) {
        *reason = "needs CUDA and CFG";
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(model_.backend);
    ggml_backend_t backend = device ? ggml_backend_dev_init(device, nullptr) : nullptr;
    if (!backend) {
        *reason = "no second backend on the model's device";
        return false;
    }
#if defined(NEMO_SPEECH_GGML_PATCHED)
    ggml_backend_cuda_set_graphs_enabled(backend, false);  // shapes vary per batch
#endif
    cudaEvent_t ready = nullptr;
    if (cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) != cudaSuccess) {
        (void)cudaGetLastError();
        ggml_backend_free(backend);
        *reason = "event creation failed";
        return false;
    }
    // The shared unconditional prefill comes from the first one-at-a-time prefill, which runs on
    // the model stream: run it now, on slot 0, before any request can be decoding.
    if (!prefill_decoder_.uncondPrefillHidden()) {
        std::vector<float> text((size_t)model_.hparams.n_embd, 0.0f);
        std::mt19937 rng(0);
        MagpieBatchRowInput row;
        row.text_len = 1;
        row.text_cond = &text;
        row.rng = &rng;
        MagpieBatchRowOutput out;
        const bool ok = prefill(0, row, out);
        release(0);
        if (!ok || !prefill_decoder_.uncondPrefillHidden()) {
            cudaEventDestroy(ready);
            ggml_backend_free(backend);
            *reason = "unconditional prefill failed";
            return false;
        }
    }
    if (!ensurePrefillHidden()) {
        cudaEventDestroy(ready);
        ggml_backend_free(backend);
        *reason = "allocation failed";
        return false;
    }
    prefill_backend_ = backend;
    prefill_ready_ = ready;
    return true;
#else
    *reason = "CUDA sampling was not compiled in";
    return false;
#endif
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
    // The kernels handle at most MAGPIETTS_BATCH_FUSED_MAX_LANES lanes per launch; a step with
    // more requests runs as several launches over chunks of this many rows.
    const int chunk_rows = MAGPIETTS_BATCH_FUSED_MAX_LANES / lanes_per;
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
            L.ff_out.size() != 1 || !L.has_cross || !L.cross_q || !L.cross_o ||
            !L.norm_xattn_query) {
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
        d.apply_prior =
            tr.apply_attention_prior && layer_selected(tr.apply_prior_to_layers, (int)i);
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
    cache.arena_lanes = decoder_.arenaLaneCount();
    cache.kv_arena = arena.data();
    cache.text_capacity = decoder_.textCapacity();
    cache.arena_slots = decoder_.layout().arena_slots;
    cache.cross_k = decoder_.crossK();
    cache.cross_v = decoder_.crossV();
    auto build = [&](cudaStream_t stream, bool owns_stream) -> std::unique_ptr<Fused> {
        // The primary context reports through fail(); a failed second context only logs, so the
        // first one keeps working.
        auto bad = [&](const std::string& why) {
            if (owns_stream)
                fprintf(stderr, "MagpieTTS batch: second fused context failed (%s)\n", why.c_str());
            else
                fail(why);
        };
        auto f = std::make_unique<Fused>();
        char error[256] = {};
        f->decoder = magpietts_batch_decoder_fused_create(w, cache, error, sizeof(error));
        if (!f->decoder) {
            {
                bad(error);
                return nullptr;
            }
        }

        // local transformer
        const magpietts_transformer& lt = model_.local;
        if (model_.lt_in_w || lt.kernel != 1 || lt.norm_out || lt.layers.empty() ||
            lt.n_embd != h.n_embd || !lt.pos_emb) {
            {
                bad("unsupported local-transformer layout");
                return nullptr;
            }
        }
        const int rounds = h.stacked_audio_codebooks();
        std::vector<magpietts_lt_fused_layer_weights> lt_layers(lt.layers.size());
        for (size_t il = 0; il < lt.layers.size(); ++il) {
            const magpietts_layer& L = lt.layers[il];
            if (!L.self_qkv || !L.self_o || L.ff_proj.size() != 1 || L.ff_out.size() != 1 ||
                !L.norm_self || !L.norm_ff) {
                {
                    bad("unsupported local-transformer layer");
                    return nullptr;
                }
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
            {
                bad(error);
                return nullptr;
            }
        }

        f->stream = stream;
        f->owns_stream = owns_stream;
        f->max_rows = std::min(config_.max_slots, chunk_rows);
        f->max_lanes = f->max_rows * lanes_per;
        f->n_cb = rounds;
        f->text_capacity = decoder_.textCapacity();
        f->E = h.n_embd;
        const size_t R = (size_t)f->max_rows, L = (size_t)f->max_lanes,
                     TC = (size_t)f->text_capacity;
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
            {
                bad("fused step buffer allocation failed");
                return nullptr;
            }
        }
        if (cudaEventCreateWithFlags(&f->ready, cudaEventDisableTiming) != cudaSuccess) {
            (void)cudaGetLastError();
            bad("fused step event creation failed");
            return nullptr;
        }
        return f;
    };
    fused_ = build((cudaStream_t)ggml_backend_cuda_get_stream(model_.backend), false);
    if (!fused_) {
        return false;
    }
    // A second context lets two chunks of a large step run at the same time. The persistent
    // kernels spin on a grid barrier, so both contexts' grids must be co-resident: two grids
    // that do not fit on the GPU together could deadlock, and then one context is used.
    if (config_.fused_streams > 1 && config_.max_slots > fused_->max_rows) {
        int device = 0, sms = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        const int grid = std::max(
            magpietts_batch_decoder_fused_grid(fused_->decoder),
            magpietts_batch_lt_fused_grid(fused_->lt));
        cudaStream_t second = nullptr;
        if (2 * grid <= sms &&
            cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking) == cudaSuccess) {
            fused2_ = build(second, true);
            if (fused2_) {
                fprintf(
                    stderr, "MagpieTTS batch: 2 concurrent fused streams (grid %d on %d SMs)\n",
                    grid, sms);
            } else {
                cudaStreamDestroy(second);
            }
        } else {
            (void)cudaGetLastError();
            fprintf(
                stderr, "MagpieTTS batch: one fused stream (2 grids of %d blocks need %d SMs)\n",
                grid, 2 * grid);
        }
    }
    return true;
#else
    *reason = "CUDA sampling was not compiled in";
    return false;
#endif
}

bool
MagpieBatchModelBackend::prefill(
    int slot, const MagpieBatchRowInput& row, MagpieBatchRowOutput& out) {
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
    for (auto& codes : bos)
        codes.assign(static_cast<size_t>(h.frame_stacking_factor), h.audio_bos_id);
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
    sampling[0].seed = row.seed;
    sampling[0].frame_index = row.frame_index;
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
MagpieBatchModelBackend::preparePrefillRows(
    const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
    size_t first, std::vector<MagpieBatchRowOutput>& out,
    std::vector<MagpieDecoderPrefillRow>& prow, MagpieDecoderPrefillTargets& targets) const {
    const MagpieLaneLayout& layout = decoder_.layout();
    const size_t n = rows.size() - first;
    prow.assign(n, MagpieDecoderPrefillRow{});
    for (size_t i = 0; i < n; ++i) {
        const MagpieBatchRowInput& r = *rows[first + i];
        const int slot = slots[first + i];
        if (r.text_len <= 0 || r.text_len > decoder_.textCapacity()) {
            fprintf(stderr, "MagpieTTS batch: invalid prefill input (text_len %d)\n", r.text_len);
            return false;
        }
        MagpieDecoderPrefillRow& p = prow[i];
        p.speaker = r.speaker;
        p.text_len = r.text_len;
        p.text_device = r.text_cond_device;
        p.prior = r.prior;
        p.alignment = r.collect_alignment ? &out[first + i].alignment : nullptr;
        p.cond_lane = layout.condLane(slot);
        p.uncond_lane = layout.uncondLane(slot);
        p.cross_slot = slot;
    }
    targets = MagpieDecoderPrefillTargets{};
    for (int il = 0; il < (int)model_.decoder.layers.size(); ++il)
        targets.self_arena.push_back(decoder_.selfArenaTensor(il));
    targets.cache_len = decoder_.cacheLen();
    targets.cross_k = decoder_.crossKTensor();
    targets.cross_v = decoder_.crossVTensor();
    return true;
}

bool
MagpieBatchModelBackend::sharedUncondPrefix() {
    if (!decoder_.sharedUncondPrefix()) {
        if (const magpietts_uncond_prefill_cache* uc = prefill_decoder_.uncondPrefillCache()) {
            decoder_.setSharedUncondPrefix(*uc);
        }
    }
    return decoder_.sharedUncondPrefix();
}

bool
MagpieBatchModelBackend::ensurePrefillHidden() {
    return prefill_batch_hidden_.tensor || prefill_batch_hidden_.alloc2d(
                                               model_, GGML_TYPE_F32, model_.hparams.n_embd,
                                               config_.max_slots, "prefill_batch_hidden");
}

bool
MagpieBatchModelBackend::prefillBatch(
    const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
    std::vector<MagpieBatchRowOutput>& out) {
    const size_t B = rows.size();
    auto one_at_a_time = [&](size_t from) {
        for (size_t i = from; i < B; ++i)
            if (!prefill(slots[i], *rows[i], out[i]))
                return false;
        return true;
    };
    out.assign(B, MagpieBatchRowOutput{});
    if (!config_.use_cfg || B == 0) {
        return one_at_a_time(0);
    }
    const ggml_nvtx::range nvtx_range("magpietts_batch_backend_prefill_batch");
    size_t first = 0;
    if (!prefill_decoder_.uncondPrefillHidden()) {
        // the first prefill fills the shared unconditional cache
        if (!prefill(slots[0], *rows[0], out[0])) {
            return false;
        }
        first = 1;
    }
    const ggml_tensor* uncond_hidden = prefill_decoder_.uncondPrefillHidden();
    if (!uncond_hidden || !ensurePrefillHidden()) {
        return one_at_a_time(first);
    }
    const size_t n = B - first;
    if (n == 0) {
        return true;
    }
    std::vector<MagpieDecoderPrefillRow> prow;
    MagpieDecoderPrefillTargets targets;
    if (!preparePrefillRows(slots, rows, first, out, prow, targets)) {
        return false;
    }
    const int n_tokens = prefill_decoder_.prefillBatch(
        prow, targets, config_.threads, &prefill_batch_allocr_, prefill_batch_hidden_);
    if (n_tokens <= 0) {
        return one_at_a_time(first);
    }
    const bool shared = sharedUncondPrefix();
    for (size_t i = 0; i < n; ++i) {
        if (!decoder_.markPrefilled(
                slots[first + i], n_tokens, rows[first + i]->text_len, shared)) {
            return false;
        }
    }
    return firstFrames(rows, first, n, uncond_hidden, out);
}

bool
MagpieBatchModelBackend::firstFrames(
    const std::vector<const MagpieBatchRowInput*>& rows, size_t first, size_t n,
    const ggml_tensor* uncond_hidden, std::vector<MagpieBatchRowOutput>& out) {
    if (!fused_) {
        return sampleFirstFrames(rows, first, n, uncond_hidden, out);
    }
#if defined(MAGPIETTS_CUDA_SAMPLING)
    const magpietts_hparams& h = model_.hparams;
    // first frame: the local transformer for every row, in chunks on the fused contexts
    const size_t E = (size_t)h.n_embd;
    const size_t chunk = static_cast<size_t>(fused_->max_rows);
    Fused* contexts[2] = {fused_.get(), fused2_.get()};
    const size_t n_contexts = fused2_ ? 2 : 1;
    if (fused2_ && (cudaEventRecord(fused2_->ready, fused_->stream) != cudaSuccess ||
                    cudaStreamWaitEvent(fused2_->stream, fused2_->ready, 0) != cudaSuccess)) {
        return false;
    }
    const float* cond = static_cast<const float*>(prefill_batch_hidden_.tensor->data);
    const float* uncond = static_cast<const float*>(uncond_hidden->data);
    for (size_t start = 0; start < n; start += chunk * n_contexts) {
        struct Part {
            size_t first = 0, n = 0;
            std::vector<const MagpieBatchRowInput*> rows;
            std::vector<MagpieBatchRowOutput> out;
        } parts[2];
        size_t used = 0;
        bool ok = true;
        for (size_t c = 0; c < n_contexts && ok; ++c) {
            const size_t p0 = start + c * chunk;
            if (p0 >= n) {
                break;
            }
            Part& part = parts[c];
            part.first = p0;
            part.n = std::min(chunk, n - p0);
            part.rows.assign(rows.begin() + first + p0, rows.begin() + first + p0 + part.n);
            Fused& f = *contexts[c];
            // planar LT input: the rows' conditional states, then the shared unconditional one
            ok = cudaMemcpyAsync(
                     f.hidden, cond + p0 * E, part.n * E * sizeof(float), cudaMemcpyDeviceToDevice,
                     f.stream) == cudaSuccess;
            for (size_t b = 0; b < part.n && ok; ++b) {
                ok = cudaMemcpyAsync(
                         f.hidden + (part.n + b) * E, uncond, E * sizeof(float),
                         cudaMemcpyDeviceToDevice, f.stream) == cudaSuccess;
            }
            ok = ok && launchSampleFused(f, (int)part.n, part.rows, false);
            used = c + 1;
        }
        for (size_t c = 0; c < used; ++c) {
            ok = collectSampleFused(*contexts[c], (int)parts[c].n, parts[c].out) && ok;
        }
        if (!ok) {
            return false;
        }
        for (size_t c = 0; c < used; ++c) {
            for (size_t b = 0; b < parts[c].n; ++b) {
                MagpieBatchRowOutput& o = out[first + parts[c].first + b];
                o.codes = std::move(parts[c].out[b].codes);
                o.argmax = std::move(parts[c].out[b].argmax);
            }
        }
    }
    return true;
#else
    return false;
#endif
}

bool
MagpieBatchModelBackend::prefillStream() const {
    return prefill_backend_ != nullptr;
}

bool
MagpieBatchModelBackend::prefillBegin(MagpieBatchPrefillJob& job) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    // the prefill stream starts after everything already queued on the model stream (the
    // requests' text, slot handoffs)
    cudaStream_t main = (cudaStream_t)ggml_backend_cuda_get_stream(model_.backend);
    job.out.assign(job.rows.size(), MagpieBatchRowOutput{});
    return cudaEventRecord((cudaEvent_t)prefill_ready_, main) == cudaSuccess;
#else
    (void)job;
    return false;
#endif
}

bool
MagpieBatchModelBackend::prefillCompute(MagpieBatchPrefillJob& job) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    cudaStream_t stream = (cudaStream_t)ggml_backend_cuda_get_stream(prefill_backend_);
    if (cudaStreamWaitEvent(stream, (cudaEvent_t)prefill_ready_, 0) != cudaSuccess) {
        return false;
    }
    if (!job.windows.empty() &&
        !encoder_.evalDeviceBatch(
            job.windows, config_.threads, job.encoded, &encode_batch_allocr_, prefill_backend_)) {
        fprintf(stderr, "MagpieTTS batch: batched encoder pass failed\n");
        return false;
    }
    std::vector<MagpieDecoderPrefillRow> prow;
    MagpieDecoderPrefillTargets targets;
    if (!preparePrefillRows(job.slots, job.rows, 0, job.out, prow, targets)) {
        return false;
    }
    // writes only the job's slots in the arenas; completes before returning
    job.n_tokens = prefill_decoder_.prefillBatch(
        prow, targets, config_.threads, &prefill_batch_allocr_, prefill_batch_hidden_,
        prefill_backend_);
    return job.n_tokens > 0;
#else
    (void)job;
    return false;
#endif
}

bool
MagpieBatchModelBackend::prefillEnd(MagpieBatchPrefillJob& job) {
    const size_t n = job.rows.size();
    const bool shared = sharedUncondPrefix();
    for (size_t i = 0; i < n; ++i) {
        if (!decoder_.markPrefilled(job.slots[i], job.n_tokens, job.rows[i]->text_len, shared)) {
            return false;
        }
    }
    return firstFrames(job.rows, 0, n, prefill_decoder_.uncondPrefillHidden(), job.out);
}

bool
MagpieBatchModelBackend::sampleFirstFrames(
    const std::vector<const MagpieBatchRowInput*>& rows, size_t first, size_t n,
    const ggml_tensor* uncond_hidden, std::vector<MagpieBatchRowOutput>& out) {
    // first frame on the ggml local transformer: one bucket for the n rows, planar input (the
    // rows' conditional states, then the shared unconditional state for each row)
    const int bucket = magpie_batch_bucket(decoder_.buckets(), static_cast<int>(n));
    if (bucket < static_cast<int>(n)) {
        return false;
    }
    std::vector<MagpieBatchSampling> sampling(n);
    for (size_t i = 0; i < n; ++i) {
        const MagpieBatchRowInput& r = *rows[first + i];
        MagpieBatchSampling& s = sampling[i];
        s.temperature = r.temperature;
        s.cfg_scale = r.cfg_scale;
        s.top_k = r.top_k;
        s.forbid_eos = r.forbid_eos;
        s.rng = r.rng;
        s.seed = r.seed;
        s.frame_index = r.frame_index;
    }
    std::vector<std::vector<int32_t>> codes, argmax;
    const bool sampled = lt_.sample(
        bucket, sampling,
        [&](ggml_tensor* input) {
            const size_t E = static_cast<size_t>(model_.hparams.n_embd);
            ggml_init_params params = {ggml_tensor_overhead() * (n + 2), nullptr, true};
            ggml_context* ctx = ggml_init(params);
            // stream-ordered after the prefill graph on the model backend
            ggml_tensor* src = ggml_view_1d(ctx, prefill_batch_hidden_.tensor, (int64_t)(n * E), 0);
            ggml_tensor* dst = ggml_view_1d(ctx, input, (int64_t)(n * E), 0);
            ggml_backend_view_init(src);
            ggml_backend_view_init(dst);
            ggml_backend_tensor_copy_async(model_.backend, model_.backend, src, dst);
            for (size_t b = 0; b < n; ++b) {
                ggml_tensor* u = ggml_view_1d(
                    ctx, input, (int64_t)E, (static_cast<size_t>(bucket) + b) * input->nb[1]);
                ggml_backend_view_init(u);
                ggml_backend_tensor_copy_async(
                    model_.backend, model_.backend, const_cast<ggml_tensor*>(uncond_hidden), u);
            }
            ggml_free(ctx);
            return true;
        },
        codes, argmax);
    if (!sampled) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        out[first + i].codes = std::move(codes[i]);
        out[first + i].argmax = std::move(argmax[i]);
    }
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
        s.seed = in.seed;
        s.frame_index = in.frame_index;
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
                ggml_backend_tensor_copy_async(
                    model_.backend, model_.backend, hidden.tensor, input);
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
MagpieBatchModelBackend::launchSampleFused(
    Fused& f, int rows, const std::vector<const MagpieBatchRowInput*>& in, bool from_prefill) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    const int n_cb = f.n_cb;
    if (from_prefill) {
        // prefill: the first frame's hidden states come from the prefill scratch
        const size_t bytes = (size_t)f.E * sizeof(float);
        if (cudaMemcpyAsync(
                f.hidden, hidden_cond_.tensor->data, bytes, cudaMemcpyDeviceToDevice, f.stream) !=
                cudaSuccess ||
            (config_.use_cfg && cudaMemcpyAsync(
                                    f.hidden + f.E, hidden_uncond_.tensor->data, bytes,
                                    cudaMemcpyDeviceToDevice, f.stream) != cudaSuccess)) {
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
    if (cudaMemcpyAsync(
            f.d_in + f.off_configs, f.h_in + f.off_configs, cfg_bytes, cudaMemcpyHostToDevice,
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
    return cudaMemcpyAsync(f.h_out, f.d_out, code_bytes, cudaMemcpyDeviceToHost, f.stream) ==
               cudaSuccess &&
           cudaMemcpyAsync(
               f.h_out + f.off_argmax, f.d_out + f.off_argmax, code_bytes, cudaMemcpyDeviceToHost,
               f.stream) == cudaSuccess;
#else
    (void)f;
    (void)rows;
    (void)in;
    (void)from_prefill;
    return false;
#endif
}

bool
MagpieBatchModelBackend::collectSampleFused(
    Fused& f, int rows, std::vector<MagpieBatchRowOutput>& out) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    if (cudaStreamSynchronize(f.stream) != cudaSuccess) {
        return false;
    }
    const int n_cb = f.n_cb;
    out.resize(static_cast<size_t>(rows));
    const int32_t* codes = reinterpret_cast<const int32_t*>(f.h_out);
    const int32_t* argmax = reinterpret_cast<const int32_t*>(f.h_out + f.off_argmax);
    for (int b = 0; b < rows; ++b) {
        out[b].codes.assign(codes + (size_t)b * n_cb, codes + (size_t)(b + 1) * n_cb);
        out[b].argmax.assign(argmax + (size_t)b * n_cb, argmax + (size_t)(b + 1) * n_cb);
    }
    return true;
#else
    (void)f;
    (void)rows;
    (void)out;
    return false;
#endif
}

bool
MagpieBatchModelBackend::sampleFused(
    int rows, const std::vector<const MagpieBatchRowInput*>& in,
    std::vector<MagpieBatchRowOutput>& out, bool from_prefill) {
    return launchSampleFused(*fused_, rows, in, from_prefill) &&
           collectSampleFused(*fused_, rows, out);
}

bool
MagpieBatchModelBackend::decodeFused(
    const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
    std::vector<MagpieBatchRowOutput>& out) {
    const size_t B = rows.size();
    const size_t chunk = static_cast<size_t>(fused_->max_rows);
    // More requests than one launch holds: run the step in chunks, two at a time when a second
    // context exists. A row's arithmetic and sampling depend only on its own slot, seed and frame
    // index, so the codes do not depend on how the rows are chunked or on which stream runs them.
    Fused* contexts[2] = {fused_.get(), fused2_.get()};
    const size_t n_contexts = fused2_ ? 2 : 1;
    out.clear();
    out.reserve(B);
    struct Part {
        std::vector<int> slots;
        std::vector<const MagpieBatchRowInput*> rows;
        std::vector<MagpieBatchRowOutput> out;
    };
    if (fused2_) {
        // The slots' caches were written on the model backend's stream (prefill, handoff): order
        // the second stream after that work, once, before any chunk of this step is enqueued
        // (an event recorded later would also wait for the first chunk's kernels).
        if (cudaEventRecord(fused2_->ready, fused_->stream) != cudaSuccess ||
            cudaStreamWaitEvent(fused2_->stream, fused2_->ready, 0) != cudaSuccess) {
            return false;
        }
    }
    for (size_t start = 0; start < B; start += chunk * n_contexts) {
        Part parts[2];
        size_t used = 0;
        for (size_t c = 0; c < n_contexts; ++c) {
            const size_t first = start + c * chunk;
            if (first >= B) {
                break;
            }
            const size_t n = std::min(chunk, B - first);
            parts[c].slots.assign(slots.begin() + first, slots.begin() + first + n);
            parts[c].rows.assign(rows.begin() + first, rows.begin() + first + n);
            used = c + 1;
        }
        // enqueue every chunk before waiting on any, so the kernels overlap on the GPU
        bool ok = true;
        size_t launched = 0;
        for (size_t c = 0; c < used && ok; ++c) {
            ok = launchFusedChunk(*contexts[c], parts[c].slots, parts[c].rows);
            ++launched;
        }
        // always collect what was launched: the streams must be idle before returning
        for (size_t c = 0; c < launched; ++c) {
            ok = collectFusedChunk(*contexts[c], parts[c].slots, parts[c].rows, parts[c].out) && ok;
        }
        if (!ok) {
            return false;
        }
        for (size_t c = 0; c < used; ++c)
            for (MagpieBatchRowOutput& o : parts[c].out) out.push_back(std::move(o));
    }
    return true;
}

bool
MagpieBatchModelBackend::launchFusedChunk(
    Fused& f, const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    const ggml_nvtx::range nvtx_range("magpietts_batch_fused_step");
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
    if (cudaMemcpyAsync(f.d_in, f.h_in, f.in_bytes, cudaMemcpyHostToDevice, f.stream) !=
        cudaSuccess) {
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
    f.pending_alignment = any_alignment;
    if (any_alignment &&
        cudaMemcpyAsync(
            f.h_out + f.off_align, f.d_out + f.off_align, (size_t)B * TC * sizeof(float),
            cudaMemcpyDeviceToHost, f.stream) != cudaSuccess) {
        return false;
    }
    // the configs were uploaded with the inputs above; this adds the local-transformer frame and
    // the code readback
    return launchSampleFused(f, B, rows, false);
#else
    (void)f;
    (void)slots;
    (void)rows;
    return false;
#endif
}

bool
MagpieBatchModelBackend::collectFusedChunk(
    Fused& f, const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
    std::vector<MagpieBatchRowOutput>& out) {
#if defined(MAGPIETTS_CUDA_SAMPLING)
    const int B = static_cast<int>(rows.size());
    const int TC = f.text_capacity;
    if (!collectSampleFused(f, B, out)) {
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
    (void)f;
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

bool
MagpieBatchModelBackend::batchedEncoder() const {
    // requests defer their encoder pass only when their text stays on a CUDA device (see
    // MagpieRequestRunner::beginTextChunk), and only batched prefills collect them
    return magpietts_backend_is_cuda(model_.backend) && config_.use_cfg &&
           config_.prefill_batch_size > 1;
}

bool
MagpieBatchModelBackend::encodeTexts(
    const std::vector<const std::vector<int32_t>*>& windows,
    const std::vector<magpietts_backend_tensor*>& outs) {
    return encoder_.evalDeviceBatch(windows, config_.threads, outs, &encode_batch_allocr_);
}

void
MagpieBatchModelBackend::release(int slot) {
    decoder_.releaseSlot(slot);
}

}  // namespace nemo_speech::tts
