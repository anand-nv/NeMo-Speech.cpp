// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "graph.h"
#include "nvtx_utils.h"

namespace nemo_speech::tts {

bool
MagpieEncoder::eval(
    const std::vector<int32_t>& tokens, int threads, std::vector<float>& out) const {
    const ggml_nvtx::range nvtx_range("magpietts_encoder_eval");
    const int n = (int)tokens.size();
    if (n <= 0 || n > model_.hparams.n_ctx) {
        fprintf(stderr, "invalid token length %d (max %d)\n", n, model_.hparams.n_ctx);
        return false;
    }

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);

    ggml_tensor* tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(tok, "magpietts_encoder_tokens");
    ggml_set_input(tok);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(pos, "magpietts_encoder_positions");
    ggml_set_input(pos);

    ggml_tensor* x = ggml_get_rows(ctx, model_.text_embedding, tok);
    x = transformer_forward(ctx, model_.encoder, x, pos, nullptr);
    x = ggml_cont(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));
    ggml_set_name(x, "magpietts_encoder_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t allocr = nullptr;
    const bool ok = compute_graph(
        model_, ctx, gf,
        {{"magpietts_encoder_tokens", tokens}, {"magpietts_encoder_positions", positions(n)}}, {},
        threads, &allocr);
    if (!ok) {
        ggml_free(ctx);
        return false;
    }

    out.resize((size_t)model_.hparams.n_embd * n);
    MagpiePinnedHostScratch output_staging;
    magpietts_backend_tensor_get_staged(
        model_, output_staging, x, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return true;
}

bool
MagpieEncoder::evalDevice(
    const std::vector<int32_t>& tokens, int threads, magpietts_backend_tensor& out,
    ggml_backend_t backend, ggml_gallocr_t* keep_allocr) const {
    const ggml_nvtx::range nvtx_range("magpietts_encoder_eval_device");
    const int n = (int)tokens.size();
    if (n <= 0 || n > model_.hparams.n_ctx) {
        fprintf(stderr, "invalid token length %d (max %d)\n", n, model_.hparams.n_ctx);
        return false;
    }
    if (!out.alloc2d(model_, GGML_TYPE_F32, model_.hparams.n_embd, n, "text_cond_device")) {
        return false;
    }

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);

    ggml_tensor* tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(tok, "magpietts_encoder_tokens");
    ggml_set_input(tok);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(pos, "magpietts_encoder_positions");
    ggml_set_input(pos);

    ggml_tensor* x = ggml_get_rows(ctx, model_.text_embedding, tok);
    x = transformer_forward(ctx, model_.encoder, x, pos, nullptr);
    x = ggml_cont(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));
    ggml_set_name(x, "magpietts_encoder_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t local_allocr = nullptr;
    ggml_gallocr_t* allocr = keep_allocr ? keep_allocr : &local_allocr;
    const bool ok = compute_graph(
        model_, ctx, gf,
        {{"magpietts_encoder_tokens", tokens}, {"magpietts_encoder_positions", positions(n)}}, {},
        threads, allocr, backend);
    if (!ok) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_copy(x, out.tensor);
    if (!keep_allocr) {
        ggml_gallocr_free(local_allocr);
    }
    ggml_free(ctx);
    return true;
}

namespace {

// Causal shift of each of the B sequences of x [E, T * B] by `shift` rows (zeros enter at the
// front), as a contiguous [E, T * B] tensor (the Q8_0 matmuls take 2-D activations).
ggml_tensor*
causal_shift_batch(ggml_context* ctx, ggml_tensor* x, int T, int B, int shift) {
    if (shift == 0) {
        return x;
    }
    ggml_tensor* x3 = ggml_reshape_3d(ctx, x, x->ne[0], T, B);
    ggml_tensor* padded = ggml_pad_ext(ctx, x3, 0, 0, shift, 0, 0, 0, 0, 0);
    ggml_tensor* view = ggml_view_3d(ctx, padded, x->ne[0], T, B, padded->nb[1], padded->nb[2], 0);
    return ggml_cont_2d(ctx, view, x->ne[0], (int64_t)T * B);
}

ggml_tensor*
causal_conv1d_batch(
    ggml_context* ctx, ggml_tensor* x, int T, int B, const std::vector<ggml_tensor*>& kernels) {
    ggml_tensor* y = nullptr;
    const int kernel = (int)kernels.size();
    for (int k = 0; k < kernel; ++k) {
        ggml_tensor* part =
            ggml_mul_mat(ctx, kernels[k], causal_shift_batch(ctx, x, T, B, kernel - 1 - k));
        y = y ? ggml_add(ctx, y, part) : part;
    }
    return y;
}

}  // namespace

bool
MagpieEncoder::evalDeviceBatch(
    const std::vector<const std::vector<int32_t>*>& windows, int threads,
    const std::vector<magpietts_backend_tensor*>& outs, ggml_gallocr_t* keep_allocr,
    ggml_backend_t backend) const {
    const ggml_nvtx::range nvtx_range("magpietts_encoder_eval_device_batch");
    const magpietts_transformer& tr = model_.encoder;
    const int B = (int)windows.size();
    if (B <= 0 || outs.size() != windows.size() || !tr.causal || tr.portable_causal_mask ||
        tr.has_cross) {
        return false;
    }
    const int64_t E = model_.hparams.n_embd;
    int T = 0;
    for (int b = 0; b < B; ++b) {
        const int n = (int)windows[(size_t)b]->size();
        if (n <= 0 || n > model_.hparams.n_ctx) {
            return false;
        }
        T = std::max(T, n);
        if (!outs[(size_t)b]->alloc2d(model_, GGML_TYPE_F32, E, n, "text_cond_device")) {
            return false;
        }
    }
    std::vector<int32_t> tokens((size_t)T * B, 0);  // padding token 0: never attended to
    for (int b = 0; b < B; ++b)
        std::copy(
            windows[(size_t)b]->begin(), windows[(size_t)b]->end(), tokens.begin() + (size_t)b * T);

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);
    ggml_tensor* tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)T * B);
    ggml_set_name(tok, "magpietts_encoder_batch_tokens");
    ggml_set_input(tok);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(pos, "magpietts_encoder_batch_positions");
    ggml_set_input(pos);

    const int64_t n_head = tr.n_head, d_head = E / n_head;
    // causal mask, fused with the scale into the softmax (broadcast over heads and windows)
    ggml_tensor* causal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    ggml_set_name(causal, "magpietts_encoder_batch_causal_mask");
    ggml_set_input(causal);
    std::vector<float> causal_data((size_t)T * T, 0.0f);
    for (int q = 0; q < T; ++q)
        for (int k = q + 1; k < T; ++k) causal_data[(size_t)q * T + k] = -INFINITY;
    // activations stay 2-D [E, T * B] (sequence b in columns [b * T, (b + 1) * T))
    ggml_tensor* x = ggml_reshape_3d(ctx, ggml_get_rows(ctx, model_.text_embedding, tok), E, T, B);
    x = ggml_add(ctx, x, ggml_get_rows(ctx, tr.pos_emb, pos));
    x = ggml_reshape_2d(ctx, x, E, (int64_t)T * B);
    for (const magpietts_layer& layer : tr.layers) {
        ggml_tensor* residual = x;
        ggml_tensor* cur = layer_norm(ctx, x, layer.norm_self);
        ggml_tensor* qkv = linear(ctx, layer.self_qkv, cur);  // [3E, T * B]
        const size_t esz = ggml_element_size(qkv);
        auto part = [&](int i) {
            return ggml_cont_4d(
                ctx, ggml_view_3d(ctx, qkv, E, T, B, qkv->nb[1], qkv->nb[1] * T, esz * E * i),
                d_head, n_head, T, B);
        };
        ggml_tensor* q = ggml_permute(ctx, part(0), 0, 2, 1, 3);
        ggml_tensor* k = ggml_permute(ctx, part(1), 0, 2, 1, 3);
        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);  // [T, T, n_head, B]
        kq = ggml_soft_max_ext(ctx, kq, causal, 1.0f / std::sqrt((float)d_head), 0.0f);
        ggml_tensor* v_trans =
            ggml_cont_4d(ctx, ggml_permute(ctx, part(2), 1, 2, 0, 3), T, d_head, n_head, B);
        ggml_tensor* kqv = ggml_mul_mat(ctx, v_trans, kq);  // [d_head, T, n_head, B]
        cur = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), E, (int64_t)T * B);
        x = ggml_add(ctx, residual, linear(ctx, layer.self_o, cur));

        residual = x;
        cur = layer_norm(ctx, x, layer.norm_ff);
        cur = causal_conv1d_batch(ctx, cur, T, B, layer.ff_proj);
        cur = ggml_gelu(ctx, cur);
        cur = causal_conv1d_batch(ctx, cur, T, B, layer.ff_out);
        x = ggml_add(ctx, residual, cur);
    }
    if (tr.norm_out) {
        x = layer_norm(ctx, x, tr.norm_out);
    }
    x = ggml_cast(ctx, x, GGML_TYPE_F32);
    for (int b = 0; b < B; ++b) {
        const int n = (int)windows[(size_t)b]->size();
        ggml_tensor* src = ggml_view_2d(ctx, x, E, n, x->nb[1], (size_t)b * T * x->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, src, outs[(size_t)b]->tensor));
    }
    std::vector<int32_t> pos_ids(T);
    for (int i = 0; i < T; ++i) pos_ids[(size_t)i] = i;
    const bool ok = compute_graph(
        model_, ctx, gf,
        {{"magpietts_encoder_batch_tokens", tokens},
         {"magpietts_encoder_batch_positions", pos_ids}},
        {{"magpietts_encoder_batch_causal_mask", causal_data}}, threads, keep_allocr, backend);
    ggml_free(ctx);
    return ok;
}

}  // namespace nemo_speech::tts
