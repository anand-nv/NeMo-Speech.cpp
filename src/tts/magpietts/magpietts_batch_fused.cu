// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Batched persistent MagpieTTS kernels (see magpietts_batch_fused.h). They follow the two-lane
// kernels phase for phase (magpietts_decoder_fused.cu, the chain kernel of magpietts_lt_fused.cu)
// with L lanes: every GEMV phase loads a warp's weight rows into registers once and runs the
// unchanged two-lane dot (ltf_dot_q8_prefetched*) over consecutive lane pairs, LayerNorm
// statistics are reduced pair by pair in the same order, and attention, cross-attention and
// sampling work items are per lane or per request with the same splits. A lane's arithmetic is
// therefore identical to the two-lane kernels' whatever the batch holds.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "magpietts_batch_fused.h"
#include "magpietts_chain_common.cuh"

#ifdef BF_TIMING
// -DBF_TIMING: block 0 accumulates the time between consecutive grid barriers per phase slot
// ([kernel][slot]: 0 decoder, 1 local transformer); magpietts_batch_fused_timing_report prints it.
__device__ unsigned long long g_bf_timing[2][16];
__device__ unsigned long long g_bf_launches[2];
static __device__ __forceinline__ unsigned long long
bf_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#define BF_STAMP(kernel, slot)                                     \
    do {                                                           \
        if (blockIdx.x == 0 && threadIdx.x == 0) {                 \
            const unsigned long long now_ = bf_now();              \
            atomicAdd(&g_bf_timing[kernel][slot], now_ - bf_prev); \
            bf_prev = now_;                                        \
        }                                                          \
    } while (0)
#else
#define BF_STAMP(kernel, slot) \
    do {                       \
    } while (0)
#endif

namespace {

constexpr int BF_MAX_LANES = MAGPIETTS_BATCH_FUSED_MAX_LANES;
// Lane buffers hold one extra lane: a lane pair at the end of an odd lane count reads it (the
// result is discarded).
constexpr int BF_LANE_ALLOC = BF_MAX_LANES + 1;
constexpr int BF_MAX_ROWS = BF_MAX_LANES;
// Widest input a GEMV phase quantizes in its own prologue (n_embd, cross dim); wider inputs
// (the FFN hidden) arrive producer-quantized.
constexpr int BF_MAX_PROLOGUE_K = 768;
constexpr int BF_MAX_CODEBOOKS = 16;
constexpr int BF_MAX_LAYERS = 16;
constexpr int BF_ATTN_SPLITS = 2;  // = DF_ATTN_SPLITS of the two-lane decoder
constexpr int BF_ATTN_MAX_RANGE = 2048 / BF_ATTN_SPLITS;
constexpr int BF_XATTN_BLOCKS = 8;  // = DF_XATTN_BLOCKS
constexpr int BF_XATTN_MAX_KEYS = 64;
constexpr int BF_SX = 2048;  // floats of the shared scratch (attention, cross context, staging)

struct bf_gemv_smem {
    __align__(16) int s_q[BF_LANE_ALLOC * (BF_MAX_PROLOGUE_K / 4)];
    __align__(16) float s_d[BF_LANE_ALLOC * (BF_MAX_PROLOGUE_K / 32)];
    float s_red[4 * LTF_WARPS];
    float s_part[2 * BF_LANE_ALLOC * LTF_WARPS];  // per-lane partial sums of a reduction
    float s_mean[BF_LANE_ALLOC + 1];
    float s_rstd[BF_LANE_ALLOC + 1];
    __align__(16) float s_x[BF_SX];
};
union bf_smem_union {
    bf_gemv_smem g;
    magpietts_sample_fast_shared samp;
};
template <class Tier>
struct bf_smem {
    __align__(16) int8_t s_w[Tier::stage_bytes];
    __align__(16) __half s_wd[Tier::stage_scales];
    bf_smem_union u;
};

// One GEMV phase over L lanes (see ltf_chain_phase_args for the two-lane original).
struct bf_phase_args {
    int NL = 0;            // lanes
    int lanes_active = 0;  // lanes [0, lanes_active) get the product; the others keep their
                           // residual (decoder cross-attention: conditional lanes only); 0 = all
    int N = 0, K = 0;
    // input: producer-quantized int8 [L][K] with one scale per xgroup elements ...
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
    int xgroup = 0;
    // ... or quantized in the prologue from one of these
    int assemble = 0;                       // 1: decoder input embedding, 2: LT round-0 input
    const void* const* emb_rows = nullptr;  // [rows][BF_MAX_CODEBOOKS] embedding rows (assemble 1)
    int n_emb = 0;
    float emb_scale = 1.f;
    const void* const* pos_rows = nullptr;  // [rows] positional rows (assemble 1)
    int rows = 1;                           // requests: lane l belongs to request l % rows
    const float* h_in = nullptr;            // [L][K] (assemble 2)
    const void* pos_row = nullptr;          // positional row (assemble 2)
    int emb_type = 0, pos_type = 0;
    float* x_store = nullptr;       // residual stream [L][K] initialized by tile 0 (assemble)
    const float* norm_w = nullptr;  // in-block LayerNorm weight (assemble)
    float ln_eps = 0.f;
    const float* attn_part = nullptr;  // self-attention partials [L][H][SPLITS][66] (combined)
    int H = 0;
    const float* x_smem = nullptr;  // [lanes_active][K] in shared memory
    // LayerNorm folded into the consumer, from the producers' partial statistics
    const float* stats = nullptr;  // [lane][sum, sumsq][grid]
    const float* ln_c = nullptr;
    // outputs
    float* out = nullptr;  // [L][N]
    const float* bias = nullptr;
    int8_t* out_q = nullptr;
    float* out_d = nullptr;
    int out_group = 0;
    int out_keep_f32 = 0;
    const float* out_gamma = nullptr;
    float* out_stats = nullptr;
    // K/V rows (qkv phase): lane l writes row kv_row[l] (or kv_row_const) of lane kv_lane[l] (or l)
    float* kcache = nullptr;
    float* vcache = nullptr;
    size_t kv_lane_stride = 0, kv_row_stride = 0;
    const int32_t* kv_lane = nullptr;
    const int32_t* kv_row = nullptr;
    int kv_row_const = 0;
};

__device__ __forceinline__ float
bf_input(const bf_phase_args& a, int l, int k) {
    if (a.assemble == 1) {
        const int row = l % a.rows;
        const float p = ltf_load_scalar(a.pos_rows[row], a.pos_type, (size_t)k);
        const void* const* er = a.emb_rows + (size_t)row * BF_MAX_CODEBOOKS;
        float s = 0.f;
        for (int i = 0; i < a.n_emb; ++i) s += ltf_load_scalar(er[i], a.emb_type, (size_t)k);
        return s * a.emb_scale + p;
    }
    if (a.assemble == 2) {
        const float p = ltf_load_scalar(a.pos_row, a.pos_type, (size_t)k);
        return a.h_in[(size_t)l * a.K + k] + p;
    }
    if (a.attn_part) {
        // df_combine_self: merge the key-range splits of (lane, head) for feature d
        const int hd = k >> 6, d = k & 63;
        const float* part = a.attn_part + ((size_t)l * a.H + hd) * BF_ATTN_SPLITS * 66;
        float M = -INFINITY;
#pragma unroll
        for (int sp = 0; sp < BF_ATTN_SPLITS; ++sp) M = fmaxf(M, part[sp * 66]);
        float S = 0.f, acc = 0.f;
#pragma unroll
        for (int sp = 0; sp < BF_ATTN_SPLITS; ++sp) {
            const float m = part[sp * 66];
            const float wgt = (m > -INFINITY) ? __expf(m - M) : 0.f;
            S += wgt * part[sp * 66 + 1];
            acc += wgt * part[sp * 66 + 2 + d];
        }
        return S > 0.f ? acc / S : 0.f;
    }
    return a.x_smem[(size_t)l * a.K + k];
}

// LayerNorm mean / rstd of every lane from the producers' per-block partial sums. The two-lane
// kernels reduce a (lane, sum|sumsq) row with 64 threads: thread t adds blocks t, t+64, ... and
// two warp sums are added; here one warp computes each 32-thread half the same way.
__device__ __forceinline__ void
bf_stats_all(const float* stats, int NL, int K, float eps, bf_gemv_smem& g) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nb = (int)gridDim.x;
    for (int item = warp; item < 4 * NL; item += LTF_WARPS) {
        const int row = item >> 1;  // lane * 2 + (0: sum, 1: sumsq)
        const int half = item & 1;
        float v = 0.f;
        for (int b = half * 32 + lane; b < nb; b += 64) v += stats[(size_t)row * nb + b];
        v = ltf_warp_sum(v);
        if (lane == 0)
            g.s_part[item] = v;
    }
    __syncthreads();
    for (int l = tid; l < NL; l += LTF_THREADS) {
        const float S = g.s_part[4 * l + 0] + g.s_part[4 * l + 1];
        const float Q = g.s_part[4 * l + 2] + g.s_part[4 * l + 3];
        const float m = S / (float)K;
        g.s_mean[l] = m;
        g.s_rstd[l] = rsqrtf(fmaxf(Q / (float)K - m * m, 0.f) + eps);
    }
    __syncthreads();
}

// One GEMV phase over all lanes on this block's staged weight tile (see ltf_chain_gemv).
template <int EPI, int R, int MB, class Smem>
__device__ __forceinline__ void
bf_chain_gemv(const bf_phase_args& a, int tile, Smem& sm) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K = a.K;
    const int blocks = K >> 5;
    const int NL = a.NL;
    const int active = a.lanes_active > 0 ? a.lanes_active : NL;
    bf_gemv_smem& g = sm.u.g;
    int row_begin, rows_valid;
    ltf_chain_rows(a.N, tile, gridDim.x, row_begin, rows_valid);
    if (rows_valid <= 0)
        return;
    const int wrow0 = warp * R;
    const bool staged = a.norm_w != nullptr;

    // LayerNorm statistics: in-block over the assembled input, or from the producers.
    if (staged) {
        // an unconditional lane's assembled input is its request's conditional input
        const int distinct = a.assemble == 1 ? min(NL, a.rows) : NL;
        for (int l = 0; l < distinct; ++l) {
            float s0 = 0.f, q0 = 0.f;
#pragma unroll 3
            for (int k = tid; k < K; k += LTF_THREADS) {
                const float x0 = bf_input(a, l, k);
                if (tile == 0)
                    for (int dup = l; dup < NL; dup += distinct)
                        a.x_store[(size_t)dup * K + k] = x0;
                s0 += x0;
                q0 += x0 * x0;
            }
            s0 = ltf_warp_sum(s0);
            q0 = ltf_warp_sum(q0);
            if (lane == 0) {
                g.s_part[(l * 2 + 0) * LTF_WARPS + warp] = s0;
                g.s_part[(l * 2 + 1) * LTF_WARPS + warp] = q0;
            }
        }
        __syncthreads();
        for (int l = tid; l < NL; l += LTF_THREADS) {
            const int src = l % distinct;
            float S0 = 0.f, Q0 = 0.f;
#pragma unroll
            for (int w = 0; w < LTF_WARPS; ++w) {
                S0 += g.s_part[(src * 2 + 0) * LTF_WARPS + w];
                Q0 += g.s_part[(src * 2 + 1) * LTF_WARPS + w];
            }
            const float m0 = S0 / (float)K;
            g.s_mean[l] = m0;
            g.s_rstd[l] = rsqrtf(fmaxf(Q0 / (float)K - m0 * m0, 0.f) + a.ln_eps);
        }
        __syncthreads();
    } else if (a.stats) {
        bf_stats_all(a.stats, NL, K, a.ln_eps, g);
    }

    // Prologue: quantize the inputs of the lanes that need the product (int8, 32-blocks).
    if (!a.xq) {
        const int group = lane >> 3;
        const int sub = lane & 7;
        const int total = active * blocks;
        for (int base = warp * 4; base < total; base += LTF_WARPS * 4) {
            const int idx = base + group;
            const bool valid = idx < total;
            const int l = valid ? idx / blocks : 0;
            const int b = valid ? idx - l * blocks : 0;
            const int k0 = b * 32 + sub * 4;
            float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
            if (valid) {
                if (a.x_smem && !staged) {
                    x = *reinterpret_cast<const float4*>(a.x_smem + (size_t)l * K + k0);
                } else {
                    x = make_float4(
                        bf_input(a, l, k0), bf_input(a, l, k0 + 1), bf_input(a, l, k0 + 2),
                        bf_input(a, l, k0 + 3));
                }
                if (staged) {
                    const float m = g.s_mean[l];
                    const float r = g.s_rstd[l];
                    const float4 w4 = *reinterpret_cast<const float4*>(a.norm_w + k0);
                    x.x = (x.x - m) * r * w4.x;
                    x.y = (x.y - m) * r * w4.y;
                    x.z = (x.z - m) * r * w4.z;
                    x.w = (x.w - m) * r * w4.w;
                }
            }
            float amax = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            if (valid) {
                const float inv = amax > 0.f ? 127.f / amax : 0.f;
                const int q0 = __float2int_rn(x.x * inv) & 0xff;
                const int q1 = __float2int_rn(x.y * inv) & 0xff;
                const int q2 = __float2int_rn(x.z * inv) & 0xff;
                const int q3 = __float2int_rn(x.w * inv) & 0xff;
                g.s_q[((size_t)l * (K >> 2)) + b * 8 + sub] =
                    q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
                if (sub == 0)
                    g.s_d[l * blocks + b] = amax / 127.f;
            }
        }
    }
    ltf_cp_async_wait_all();
    __syncthreads();

    // Products of the active lanes go to shared memory ([lane][rows_valid], lane 0's warp sum as
    // in the two-lane kernels); the epilogue then runs one (lane, row) per thread.
    float* acc_s = g.s_x;
    if (wrow0 < rows_valid) {
        ltf_q8_prefetch<R, MB> pf;
        ltf_chain_load_staged<R, MB>(sm, K, rows_valid, wrow0, lane, pf);
        for (int l0 = 0; l0 < active; l0 += 2) {
            float acc0[R], acc1[R];
#pragma unroll
            for (int r = 0; r < R; ++r) acc0[r] = acc1[r] = 0.f;
            {
                if (!a.xq) {
                    ltf_dot_q8_prefetched<R, MB>(
                        pf, K, g.s_q + (size_t)l0 * (K >> 2), g.s_q + (size_t)(l0 + 1) * (K >> 2),
                        g.s_d + (size_t)l0 * blocks, g.s_d + (size_t)(l0 + 1) * blocks, lane, acc0,
                        acc1);
                } else if (a.xgroup == 32) {
                    ltf_dot_q8_prefetched_xq<R, MB, 32>(
                        pf, K, a.xq + (size_t)l0 * K, a.xd + (size_t)l0 * (K / 32), lane, acc0,
                        acc1);
                } else if (a.xgroup == 16) {
                    ltf_dot_q8_prefetched_xq<R, MB, 16>(
                        pf, K, a.xq + (size_t)l0 * K, a.xd + (size_t)l0 * (K / 16), lane, acc0,
                        acc1);
                } else {
                    ltf_dot_q8_prefetched_xq<R, MB, 4>(
                        pf, K, a.xq + (size_t)l0 * K, a.xd + (size_t)l0 * (K / 4), lane, acc0,
                        acc1);
                }
            }
            if (lane == 0) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (wrow0 + r >= rows_valid)
                        break;  // rows past this block's range belong to another block
                    acc_s[(size_t)l0 * rows_valid + wrow0 + r] = acc0[r];
                    if (l0 + 1 < active)
                        acc_s[(size_t)(l0 + 1) * rows_valid + wrow0 + r] = acc1[r];
                }
            }
        }
    }
    __syncthreads();  // all warps done with the staged weights; products complete
    for (int it = tid; it < NL * rows_valid; it += LTF_THREADS) {
        const int l = it / rows_valid;
        const int row = row_begin + (it - l * rows_valid);
        const float res = EPI == LTF_EPI_RESIDUAL ? a.out[(size_t)l * a.N + row] : 0.f;
        float v;
        if (l < active) {
            v = acc_s[it];
            if (a.ln_c)  // LN fold: LN(x) W = r * ((x*gamma) W - mu * c)
                v = g.s_rstd[l] * (v - g.s_mean[l] * a.ln_c[row]);
            v += a.bias ? a.bias[row] : 0.f;
            if (EPI == LTF_EPI_GELU)
                v = ltf_gelu(v);
            if (EPI == LTF_EPI_RESIDUAL)
                v = v + res;
        } else {
            v = res;
        }
        if (a.out_q)
            acc_s[it] = v;
        if (!a.out_q || a.out_keep_f32)
            a.out[(size_t)l * a.N + row] = v;
        if (a.kcache && row >= K) {
            float* cache = row < 2 * K ? a.kcache : a.vcache;
            const int f = row < 2 * K ? row - K : row - 2 * K;
            const int al = a.kv_lane ? a.kv_lane[l] : l;
            const int kr = a.kv_row ? a.kv_row[l] : a.kv_row_const;
            cache[(size_t)al * a.kv_lane_stride + (size_t)kr * a.kv_row_stride + f] = v;
        }
    }
    __syncthreads();
    if (a.out_q) {
        const int G = a.out_group;
        const int n_groups = rows_valid / G;
        if (a.out_stats) {
            for (int l = warp; l < NL; l += LTF_WARPS) {
                float sv = 0.f, sq = 0.f;
                for (int i = lane; i < rows_valid; i += 32) {
                    const float v = g.s_x[(size_t)l * rows_valid + i];
                    sv += v;
                    sq += v * v;
                }
                sv = ltf_warp_sum(sv);
                sq = ltf_warp_sum(sq);
                if (lane == 0) {
                    a.out_stats[(size_t)(l * 2 + 0) * gridDim.x + tile] = sv;
                    a.out_stats[(size_t)(l * 2 + 1) * gridDim.x + tile] = sq;
                }
            }
        }
        const int tasks = NL * n_groups;
        const int per_pass = LTF_THREADS / G;
        for (int t0 = 0; t0 < tasks; t0 += per_pass) {
            const int task = t0 + tid / G;
            const int j = tid % G;
            const bool valid = task < tasks;
            const int l = valid ? task / n_groups : 0;
            const int grp = valid ? task - l * n_groups : 0;
            const int grow = row_begin + grp * G;
            float v = valid ? g.s_x[(size_t)l * rows_valid + grp * G + j] : 0.f;
            if (valid && a.out_gamma)
                v *= a.out_gamma[grow + j];
            float amax = fabsf(v);
            for (int o = G / 2; o > 0; o >>= 1)
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            if (valid) {
                const float inv = amax > 0.f ? 127.f / amax : 0.f;
                a.out_q[(size_t)l * a.N + grow + j] = (int8_t)__float2int_rn(v * inv);
                if (j == 0)
                    a.out_d[(size_t)l * (a.N / G) + grow / G] = amax / 127.f;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------
// batched decoder step
// ---------------------------------------------------------------------------------------
struct bdf_layer {
    ltf_q8p qkv, o, cq, co, ff1, ff2;
    const float* norm_self;
    const float* norm_xq;
    const float* norm_ff;
    const float* qkv_c;
    const float* cq_c;
    const float* ff1_c;
    float* kv_arena;  // [2 planes][arena_lanes][cache_len][E]
    int flags;        // 1: has_cross, 2: collect alignment, 4: apply prior
};

struct bdf_args {
    int E, H, NFF, CD, n_layers, n_codebooks, cache_len, TC;
    int prefetch;
    float ln_eps, attn_scale, xattn_scale, emb_scale;
    bdf_layer layers[BF_MAX_LAYERS];
    const void* audio_emb[BF_MAX_CODEBOOKS];
    int emb_type;
    const void* pos_emb;
    int pos_type;
    const float* norm_out;
    const float* cross_k;
    const float* cross_v;
    size_t plane_stride;  // floats between the K and V planes of a layer arena
    int n_collect;
    // step
    int B, NL;
    const int32_t* tokens;
    const int32_t* positions;
    const int32_t* arena_lane;
    const int32_t* ring_head;
    const int32_t* valid_len;
    const int32_t* cross_slot;
    const int32_t* text_len;
    const float* prior;
    const float* mask;
    float* hidden;
    float* alignment;
    // scratch
    float* h;          // [L][E]
    int8_t* h_q;       // [L][E]
    float* h_d;        // [L][E/4]
    float* stats;      // [L][2][grid]
    float* qkv;        // [L][3E]
    float* apart;      // [L][H][SPLITS][66]
    float* qc;         // [B][CD]
    float* xpart;      // [B][XATTN_BLOCKS][2 + CD]
    float* xscores;    // [B][TC]
    float* align_acc;  // [B][TC]
    int8_t* ctx_q;     // [L][E] combined self-attention context, int8 groups of 32
    float* ctx_d;      // [L][E/32]
    int8_t* ff_q;      // [L][NFF]
    float* ff_d;       // [L][NFF/16]
    unsigned int* barrier;
};

template <class Smem>
__device__ __forceinline__ void
bdf_self_attention(
    const bdf_args& a, const float* kbase, const float* vbase, int tile, int nblocks, Smem& sm) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int E = a.E;
    const size_t cache_ss = (size_t)a.cache_len * E;
    const int n_items = a.NL * a.H * BF_ATTN_SPLITS;
    float* query = sm.u.g.s_x;
    float* scores = query + 64;
    float* red = scores + BF_ATTN_MAX_RANGE;
    float* vpart = red + LTF_WARPS;
    for (int item = tile; item < n_items; item += nblocks) {
        const int l = item / (a.H * BF_ATTN_SPLITS);
        const int rem = item - l * a.H * BF_ATTN_SPLITS;
        const int hd = rem / BF_ATTN_SPLITS;
        const int sp = rem - hd * BF_ATTN_SPLITS;
        const int valid = a.valid_len[l];
        const int ring_head = a.ring_head[l];
        const int n_keys = valid + 1;
        const int per_split = (n_keys + BF_ATTN_SPLITS - 1) / BF_ATTN_SPLITS;
        const int j0 = sp * per_split;
        const int j1 = min(n_keys, j0 + per_split);
        const int range = max(0, j1 - j0);
        const size_t lane_off = (size_t)a.arena_lane[l] * cache_ss;
        const float* Kh = kbase + lane_off + hd * 64;
        const float* Vh = vbase + lane_off + hd * 64;
        const int phys0 = ring_head + a.cache_len - valid;
        __syncthreads();
        if (tid < 64)
            query[tid] = a.qkv[(size_t)l * 3 * E + hd * 64 + tid];
        __syncthreads();
        const int group = lane >> 3, sub = lane & 7, d8 = sub * 8;
        const float4 qa = *reinterpret_cast<const float4*>(query + d8);
        const float4 qb = *reinterpret_cast<const float4*>(query + d8 + 4);
        float local_max = -INFINITY;
        for (int base = j0 + warp * 4; base < j1; base += LTF_WARPS * 4) {
            const int j = base + group;
            float score = 0.f;
            if (j < j1) {
                int p = phys0 + j;
                p -= (p >= a.cache_len) ? a.cache_len : 0;
                const float4 ka = *reinterpret_cast<const float4*>(Kh + (size_t)p * E + d8);
                const float4 kb = *reinterpret_cast<const float4*>(Kh + (size_t)p * E + d8 + 4);
                score = ka.x * qa.x + ka.y * qa.y + ka.z * qa.z + ka.w * qa.w + kb.x * qb.x +
                        kb.y * qb.y + kb.z * qb.z + kb.w * qb.w;
            }
            score += __shfl_xor_sync(0xffffffffu, score, 1);
            score += __shfl_xor_sync(0xffffffffu, score, 2);
            score += __shfl_xor_sync(0xffffffffu, score, 4);
            if (sub == 0 && j < j1) {
                score *= a.attn_scale;
                scores[j - j0] = score;
                local_max = fmaxf(local_max, score);
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, o));
        if (lane == 0)
            red[warp] = local_max;
        __syncthreads();
        float m = red[0];
#pragma unroll
        for (int w = 1; w < LTF_WARPS; ++w) m = fmaxf(m, red[w]);
        __syncthreads();
        float local_sum = 0.f;
        for (int i = tid; i < range; i += LTF_THREADS) {
            const float wgt = (m > -INFINITY) ? __expf(scores[i] - m) : 0.f;
            scores[i] = wgt;
            local_sum += wgt;
        }
        local_sum = ltf_warp_sum(local_sum);
        if (lane == 0)
            red[warp] = local_sum;
        __syncthreads();
        float s = red[0];
#pragma unroll
        for (int w = 1; w < LTF_WARPS; ++w) s += red[w];
        const int d = tid & 63, part = tid >> 6;
        float acc = 0.f;
        for (int i = part; i < range; i += 4) {
            int p = phys0 + j0 + i;
            p -= (p >= a.cache_len) ? a.cache_len : 0;
            acc += scores[i] * Vh[(size_t)p * E + d];
        }
        vpart[part * 64 + d] = acc;
        __syncthreads();
        float* out = a.apart + (((size_t)l * a.H + hd) * BF_ATTN_SPLITS + sp) * 66;
        if (tid < 64)
            out[2 + tid] = vpart[tid] + vpart[64 + tid] + vpart[128 + tid] + vpart[192 + tid];
        if (tid == 64) {
            out[0] = m;
            out[1] = s;
        }
    }
}

// Combine the self-attention splits (df_combine_self) and quantize the context into int8 groups
// of 32 (as the o-projection prologue would), one (lane, group) per warp, once for the whole
// grid instead of in every block's o-projection prologue.
__device__ __forceinline__ void
bdf_combine_quantize(const bdf_args& a, int tile, int nblocks) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int groups = a.E / 32;
    for (int item = tile * LTF_WARPS + warp; item < a.NL * groups; item += nblocks * LTF_WARPS) {
        const int l = item / groups;
        const int gidx = item - l * groups;
        const int k = gidx * 32 + lane;
        const int hd = k >> 6, d = k & 63;
        const float* part = a.apart + ((size_t)l * a.H + hd) * BF_ATTN_SPLITS * 66;
        float M = -INFINITY;
#pragma unroll
        for (int sp = 0; sp < BF_ATTN_SPLITS; ++sp) M = fmaxf(M, part[sp * 66]);
        float S = 0.f, acc = 0.f;
#pragma unroll
        for (int sp = 0; sp < BF_ATTN_SPLITS; ++sp) {
            const float m = part[sp * 66];
            const float wgt = (m > -INFINITY) ? __expf(m - M) : 0.f;
            S += wgt * part[sp * 66 + 1];
            acc += wgt * part[sp * 66 + 2 + d];
        }
        const float x = S > 0.f ? acc / S : 0.f;
        float amax = fabsf(x);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float inv = amax > 0.f ? 127.f / amax : 0.f;
        a.ctx_q[(size_t)l * a.E + k] = (int8_t)(__float2int_rn(x * inv) & 0xff);
        if (lane == 0)
            a.ctx_d[(size_t)l * groups + gidx] = amax / 127.f;
    }
}

// Cross-attention query rows [slot*CD/8, +CD/8) of request b (conditional lane b), from the
// producer-quantized residual q(h * gamma_xq) with the LayerNorm fold (see df_cross_q_tail);
// then the 8 tail blocks of the request wait for each other.
template <class Smem>
__device__ __forceinline__ void
bdf_cross_q_tail(
    const bdf_args& a, int il, int slot, int b, const ltf_q8_prefetch<2, 1>& pf, int nblocks,
    Smem& sm, int* s_flag) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int K = a.E;
    const bdf_layer& L = a.layers[il];
    float* red = sm.u.g.s_red;
    const int rows_per = a.CD / BF_XATTN_BLOCKS;
    const int row0 = slot * rows_per + warp * 2;
    const bool has_rows = warp * 2 < rows_per;
    {
        {  // LN statistics of lane b from the producers' partials
            const int grp = tid >> 6;
            float v = 0.f;
            for (int blk = tid & 63; blk < nblocks; blk += 64)
                v += a.stats[(size_t)(2 * b + grp) * nblocks + blk];
            v = ltf_warp_sum(v);
            if (lane == 0)
                red[warp] = v;
        }
        __syncthreads();
        const float S0 = red[0] + red[1], Q0 = red[2] + red[3];
        const float m0 = S0 / (float)K;
        const float r0 = rsqrtf(fmaxf(Q0 / (float)K - m0 * m0, 0.f) + a.ln_eps);
        if (has_rows) {
            float acc0[2], acc1[2];
            ltf_dot_q8_prefetched_xq<2, 1, 4>(
                pf, K, a.h_q + (size_t)b * K, a.h_d + (size_t)b * (K / 4), lane, acc0, acc1);
            if (lane == 0) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int row = row0 + r;
                    if (row < a.CD)
                        a.qc[(size_t)b * a.CD + row] = r0 * (acc0[r] - m0 * L.cq_c[row]);
                }
            }
        }
        __syncthreads();
    }
    ltf_chain_arrival_rank(a.barrier + 3 + b, BF_XATTN_BLOCKS, BF_XATTN_BLOCKS, s_flag);
}

// Cross-attention of request b over this tail block's share of its text rows.
template <class Smem>
__device__ __forceinline__ void
bdf_cross_attention(const bdf_args& a, int il, int slot, int b, Smem& sm) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int CD = a.CD;
    const int T = a.text_len[b];
    const int keys_per = (T + BF_XATTN_BLOCKS - 1) / BF_XATTN_BLOCKS;
    const int j0 = slot * keys_per;
    const int n_keys = min(keys_per, T - j0);
    const bdf_layer& L = a.layers[il];
    const float* bias = ((L.flags & 4) ? a.prior : a.mask) + (size_t)b * a.TC;
    const size_t base = (((size_t)a.cross_slot[b] * a.n_layers + il) * a.TC + j0) * CD;
    const float* K = a.cross_k + base;
    const float* V = a.cross_v + base;
    float* sc = sm.u.g.s_x;
    float* red = sc + BF_XATTN_MAX_KEYS;
    float* q = red + LTF_WARPS;
    __syncthreads();
    for (int d = tid; d < CD; d += LTF_THREADS) q[d] = a.qc[(size_t)b * CD + d];
    __syncthreads();
    float local_max = -INFINITY;
    for (int jj = warp; jj < n_keys; jj += LTF_WARPS) {
        float s = 0.f;
        for (int d = lane; d < CD; d += 32) s += q[d] * K[(size_t)jj * CD + d];
        s = ltf_warp_sum(s);
        s = s * a.xattn_scale + bias[j0 + jj];
        if (lane == 0) {
            sc[jj] = s;
            a.xscores[(size_t)b * a.TC + j0 + jj] = s;
        }
        local_max = fmaxf(local_max, s);
    }
    if (lane == 0)
        red[warp] = local_max;
    __syncthreads();
    float m = red[0];
#pragma unroll
    for (int w = 1; w < LTF_WARPS; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    float local_sum = 0.f;
    for (int jj = tid; jj < n_keys; jj += LTF_THREADS) {
        const float wgt = __expf(sc[jj] - m);
        sc[jj] = wgt;
        local_sum += wgt;
    }
    local_sum = ltf_warp_sum(local_sum);
    if (lane == 0)
        red[warp] = local_sum;
    __syncthreads();
    float s = red[0];
#pragma unroll
    for (int w = 1; w < LTF_WARPS; ++w) s += red[w];
    float* out = a.xpart + ((size_t)b * BF_XATTN_BLOCKS + slot) * (2 + CD);
    for (int d = tid; d < CD; d += LTF_THREADS) {
        float acc = 0.f;
        for (int jj = 0; jj < n_keys; ++jj) acc += sc[jj] * V[(size_t)jj * CD + d];
        out[2 + d] = acc;
    }
    if (tid == 0) {
        out[0] = m;
        out[1] = s;
    }
}

// Every request's cross-attention context [B][CD] into s_x; block 0 accumulates the alignment of
// collect layers (see df_combine_cross).
template <class Smem>
__device__ __forceinline__ void
bdf_combine_cross(const bdf_args& a, int il, int tile, Smem& sm) {
    const int tid = threadIdx.x;
    const int CD = a.CD;
    float* ctx = sm.u.g.s_x;
    for (int b = 0; b < a.B; ++b) {
        const float* xp = a.xpart + (size_t)b * BF_XATTN_BLOCKS * (2 + CD);
        float M = -INFINITY;
#pragma unroll
        for (int sp = 0; sp < BF_XATTN_BLOCKS; ++sp) M = fmaxf(M, xp[(size_t)sp * (2 + CD)]);
        float S = 0.f;
#pragma unroll
        for (int sp = 0; sp < BF_XATTN_BLOCKS; ++sp) {
            const float* p = xp + (size_t)sp * (2 + CD);
            S += __expf(p[0] - M) * p[1];
        }
        const float inv = S > 0.f ? 1.f / S : 0.f;
        for (int d = tid; d < CD; d += LTF_THREADS) {
            float acc = 0.f;
#pragma unroll
            for (int sp = 0; sp < BF_XATTN_BLOCKS; ++sp) {
                const float* p = xp + (size_t)sp * (2 + CD);
                acc += __expf(p[0] - M) * p[2 + d];
            }
            ctx[(size_t)b * CD + d] = acc * inv;
        }
        if (tile == 0 && a.alignment && (a.layers[il].flags & 2)) {
            const int T = a.text_len[b];
            float* acc_row = a.align_acc + (size_t)b * a.TC;
            const float* sc_row = a.xscores + (size_t)b * a.TC;
            for (int j = tid; j < T; j += LTF_THREADS) acc_row[j] += __expf(sc_row[j] - M) * inv;
        }
    }
}

template <class Smem>
__device__ __forceinline__ void
bdf_final(const bdf_args& a, int tile, int nblocks, Smem& sm) {
    const int tid = threadIdx.x;
    const int E = a.E;
    bf_gemv_smem& g = sm.u.g;
    bf_stats_all(a.stats, a.NL, E, a.ln_eps, g);
    int row_begin, rows;
    ltf_chain_rows(E, tile, nblocks, row_begin, rows);
    for (int l = 0; l < a.NL; ++l) {
        const float m = g.s_mean[l], r = g.s_rstd[l];
        for (int i = tid; i < rows; i += LTF_THREADS) {
            const int row = row_begin + i;
            const float gm = a.norm_out ? a.norm_out[row] : 1.f;
            a.hidden[(size_t)l * E + row] = (a.h[(size_t)l * E + row] - m) * r * gm;
        }
    }
    if (tile == 0 && a.alignment) {
        const float inv = a.n_collect > 0 ? 1.f / (float)a.n_collect : 0.f;
        for (int i = tid; i < a.B * a.TC; i += LTF_THREADS) a.alignment[i] = a.align_acc[i] * inv;
    }
}

template <class Tier>
__global__ void LTF_CHAIN_LAUNCH_BOUNDS
bdf_kernel(const bdf_args a) {
    extern __shared__ __align__(16) unsigned char bdf_dyn_smem[];
    bf_smem<Tier>& sm = *reinterpret_cast<bf_smem<Tier>*>(bdf_dyn_smem);
    __shared__ int s_flag;
    __shared__ const void* s_emb[BF_MAX_ROWS * BF_MAX_CODEBOOKS];
    __shared__ const void* s_pos[BF_MAX_ROWS];
    const int tile = blockIdx.x;
    const int tid = threadIdx.x;
    const unsigned int nblocks = gridDim.x;
    unsigned int target = 0;
    const int E = a.E;
    const size_t emb_elem = a.emb_type == LTF_TYPE_F16 ? 2 : 4;
    const size_t pos_elem = a.pos_type == LTF_TYPE_F16 ? 2 : 4;
    auto sync = [&]() { ltf_grid_barrier(a.barrier, target, nblocks); };
#ifdef BF_TIMING
    unsigned long long bf_prev = 0;
    if (tile == 0 && tid == 0) {
        bf_prev = bf_now();
        atomicAdd(&g_bf_launches[0], 1ull);
    }
#endif

    for (int i = tid; i < a.B * a.n_codebooks; i += LTF_THREADS) {
        const int b = i / a.n_codebooks, c = i - b * a.n_codebooks;
        s_emb[b * BF_MAX_CODEBOOKS + c] =
            reinterpret_cast<const char*>(a.audio_emb[c]) + (size_t)a.tokens[i] * E * emb_elem;
    }
    for (int b = tid; b < a.B; b += LTF_THREADS)
        s_pos[b] = reinterpret_cast<const char*>(a.pos_emb) + (size_t)a.positions[b] * E * pos_elem;
    if (tile == 0 && a.alignment)
        for (int j = tid; j < a.B * a.TC; j += LTF_THREADS) a.align_acc[j] = 0.f;
    __syncthreads();

    ltf_chain_stage(a.layers[0].qkv, E, 3 * E, tile, sm);
    for (int il = 0; il < a.n_layers; ++il) {
        const bdf_layer& L = a.layers[il];
        float* kbase = L.kv_arena;
        float* vbase = kbase + a.plane_stride;
        if (a.prefetch) {
            ltf_chain_prefetch_l2(L.o, E, E, tile);
            if (L.flags & 1)
                ltf_chain_prefetch_l2(L.co, a.CD, E, tile);
            ltf_chain_prefetch_l2(L.ff1, E, a.NFF, tile);
        }
        {  // LN(norm_self) + qkv; K/V rows appended at every lane's ring head
            bf_phase_args g{};
            g.NL = a.NL;
            g.N = 3 * E;
            g.K = E;
            g.out = a.qkv;
            g.ln_eps = a.ln_eps;
            if (il == 0) {
                g.assemble = 1;
                g.emb_rows = s_emb;
                g.n_emb = a.n_codebooks;
                g.emb_scale = a.emb_scale;
                g.emb_type = a.emb_type;
                g.pos_rows = s_pos;
                g.pos_type = a.pos_type;
                g.rows = a.B;
                g.norm_w = L.norm_self;
                g.x_store = a.h;
            } else {
                g.xq = a.h_q;
                g.xd = a.h_d;
                g.xgroup = 4;
                g.stats = a.stats;
                g.ln_c = L.qkv_c;
            }
            g.kcache = kbase;
            g.vcache = vbase;
            g.kv_lane_stride = (size_t)a.cache_len * E;
            g.kv_row_stride = (size_t)E;
            g.kv_lane = a.arena_lane;
            g.kv_row = a.ring_head;
            bf_chain_gemv<LTF_EPI_STORE, Tier::r_qkv, 1>(g, tile, sm);
        }
        ltf_chain_stage(L.o, E, E, tile, sm);
        sync();
        BF_STAMP(0, 0);  // qkv
        bdf_self_attention(a, kbase, vbase, tile, (int)nblocks, sm);
        sync();
        // Beyond a request the context is combined once for the grid (one more barrier) instead
        // of by every block's o-projection prologue.
        const bool combine_pass = a.NL > 2;
        if (combine_pass) {
            bdf_combine_quantize(a, tile, (int)nblocks);
            sync();
        }
        BF_STAMP(0, 1);  // self-attention
        if (a.prefetch)
            ltf_chain_prefetch_l2(L.ff2, a.NFF, E, tile);
        {  // o + residual over the combined attention context
            bf_phase_args g{};
            g.NL = a.NL;
            if (combine_pass) {
                g.xq = a.ctx_q;
                g.xd = a.ctx_d;
                g.xgroup = 32;
            } else {
                g.attn_part = a.apart;
                g.H = a.H;
            }
            g.N = E;
            g.K = E;
            g.out = a.h;
            g.out_q = a.h_q;
            g.out_d = a.h_d;
            g.out_group = 4;
            g.out_keep_f32 = 1;
            g.out_gamma = (L.flags & 1) ? L.norm_xq : L.norm_ff;
            g.out_stats = a.stats;
            bf_chain_gemv<LTF_EPI_RESIDUAL, Tier::r_o, 1>(g, tile, sm);
        }
        if (L.flags & 1) {
            ltf_chain_stage(L.co, a.CD, E, tile, sm);
            {
                // the last blocks to finish their rows form groups of BF_XATTN_BLOCKS; group g
                // attends for requests g, g + groups, ... (the query of a request needs every
                // block's o-projection rows)
                const int groups = min(a.B, (int)nblocks / BF_XATTN_BLOCKS);
                const int n_tail = groups * BF_XATTN_BLOCKS;
                const int rank = ltf_chain_arrival_rank(a.barrier + 1, nblocks, n_tail, &s_flag);
                const int t = rank - ((int)nblocks - n_tail);
                if (t >= 0) {
                    const int group = t / BF_XATTN_BLOCKS;
                    const int slot = t - group * BF_XATTN_BLOCKS;
                    const int rows_per = a.CD / BF_XATTN_BLOCKS;
                    const int warp = threadIdx.x >> 5;
                    ltf_q8_prefetch<2, 1> pf;
                    if (warp * 2 < rows_per)
                        ltf_prefetch_q8<2, 1>(
                            L.cq.qs, L.cq.d, E, a.CD, slot * rows_per + warp * 2, threadIdx.x & 31,
                            pf);
                    for (int b = group; b < a.B; b += groups) {
                        bdf_cross_q_tail(a, il, slot, b, pf, (int)nblocks, sm, &s_flag);
                        bdf_cross_attention(a, il, slot, b, sm);
                    }
                }
            }
            sync();
            BF_STAMP(0, 2);  // o-proj + cross-attention tail
            {  // cross-o + residual on the conditional lanes; every lane re-quantized for ff1
                bdf_combine_cross(a, il, tile, sm);
                __syncthreads();
                bf_phase_args g{};
                g.NL = a.NL;
                g.lanes_active = a.B;
                g.x_smem = sm.u.g.s_x;
                g.N = E;
                g.K = a.CD;
                g.out = a.h;
                g.out_q = a.h_q;
                g.out_d = a.h_d;
                g.out_group = 4;
                g.out_keep_f32 = 1;
                g.out_gamma = L.norm_ff;
                g.out_stats = a.stats;
                bf_chain_gemv<LTF_EPI_RESIDUAL, Tier::r_o, 1>(g, tile, sm);
            }
        }
        ltf_chain_stage(L.ff1, E, a.NFF, tile, sm);
        sync();
        BF_STAMP(0, 3);  // cross-o
        if (a.prefetch && il + 1 < a.n_layers)
            ltf_chain_prefetch_l2(a.layers[il + 1].qkv, E, 3 * E, tile);
        {  // LN(norm_ff) + ff1 + gelu -> int8 groups of 16
            bf_phase_args g{};
            g.NL = a.NL;
            g.xq = a.h_q;
            g.xd = a.h_d;
            g.xgroup = 4;
            g.stats = a.stats;
            g.ln_c = L.ff1_c;
            g.N = a.NFF;
            g.K = E;
            g.out = a.qkv;  // unused (int8 only)
            g.out_q = a.ff_q;
            g.out_d = a.ff_d;
            g.out_group = 16;
            g.ln_eps = a.ln_eps;
            bf_chain_gemv<LTF_EPI_GELU, Tier::r_ff1, 1>(g, tile, sm);
        }
        ltf_chain_stage(L.ff2, a.NFF, E, tile, sm);
        sync();
        BF_STAMP(0, 4);  // ff1
        {                // ff2 + residual
            bf_phase_args g{};
            g.NL = a.NL;
            g.xq = a.ff_q;
            g.xd = a.ff_d;
            g.xgroup = 16;
            g.N = E;
            g.K = a.NFF;
            g.out = a.h;
            g.out_q = a.h_q;
            g.out_d = a.h_d;
            g.out_group = 4;
            g.out_keep_f32 = 1;
            g.out_gamma = (il + 1 < a.n_layers) ? a.layers[il + 1].norm_self : nullptr;
            g.out_stats = a.stats;
            bf_chain_gemv<LTF_EPI_RESIDUAL, Tier::r_o, 3>(g, tile, sm);
        }
        if (il + 1 < a.n_layers)
            ltf_chain_stage(a.layers[il + 1].qkv, E, 3 * E, tile, sm);
        sync();
        BF_STAMP(0, 5);  // ff2
    }
    bdf_final(a, tile, (int)nblocks, sm);
    BF_STAMP(0, 6);  // final
}

// ---------------------------------------------------------------------------------------
// batched local-transformer chain
// ---------------------------------------------------------------------------------------
constexpr int BLT_ATTN_SPLIT = 4;  // = LTF_ATTN_SPLIT
constexpr int BLT_ATTN_TASKS_PER_BLOCK = LTF_WARPS / BLT_ATTN_SPLIT;
constexpr int BLT_MAX_LAYERS = 4;
constexpr int BLT_MAX_ROUNDS = LTF_MAX_POS;

struct blt_layer {
    ltf_q8p qkv, o, ff1, ff2;
    const float* norm_self;
    const float* norm_ff;
    const float* qkv_c;
    const float* ff1_c;
};

struct blt_args {
    int K, NFF, H, P, vocab, n_layers, n_rounds;
    int prefetch_out;
    float ln_eps, attn_scale;
    blt_layer layers[BLT_MAX_LAYERS];
    const void* pos;
    int pos_type;
    const void* audio_emb[BLT_MAX_ROUNDS];
    int emb_type;
    ltf_q8p out_w[BLT_MAX_ROUNDS];
    const float* out_b[BLT_MAX_ROUNDS];
    int B, NL, use_cfg;
    const float* h_in;
    float* h;
    float* qkv;
    float* ctx;
    float* ff;
    float* logits;  // [L][vocab]
    float* kcache;  // [layers][L][P][K]
    float* vcache;
    size_t layer_cache;
    int8_t* ctx_q;
    float* ctx_d;
    int8_t* ff_q;
    float* ff_d;
    int8_t* h_q;
    float* h_d;
    float* stats;
    unsigned int* barrier;
    const magpietts_cuda_sampling_config* configs;
    int32_t* codes;
    int32_t* argmax;
    int sample_codebook_size;
    int sample_eos_id;
};

struct blt_attn_partial {
    float m[BLT_ATTN_SPLIT];
    float sum[BLT_ATTN_SPLIT];
    float acc[BLT_ATTN_SPLIT][64];
};

// ltf_chain_attention_split for lane l (identical arithmetic).
__device__ __forceinline__ void
blt_attention_split(
    const float* qkv, const float* kcache, const float* vcache, int8_t* ctx_q, float* ctx_d, int K,
    int pos_rows, int pos, float scale, int h, int l, int chunk, int lane, blt_attn_partial& part,
    int task_warp0) {
    constexpr int CH = (LTF_MAX_POS + BLT_ATTN_SPLIT - 1) / BLT_ATTN_SPLIT;
    const int d0 = h * 64 + lane * 2;
    const float2 q2 = *reinterpret_cast<const float2*>(qkv + (size_t)l * 3 * K + d0);
    const float* kbase = kcache + (size_t)l * pos_rows * K + d0;
    const float* vbase = vcache + (size_t)l * pos_rows * K + d0;
    const int j0 = chunk * CH;
    float2 k2[CH], v2[CH];
#pragma unroll
    for (int i = 0; i < CH; ++i) {
        const int j = j0 + i;
        const bool ok = j < pos_rows;
        k2[i] =
            ok ? *reinterpret_cast<const float2*>(kbase + (size_t)j * K) : make_float2(0.f, 0.f);
        v2[i] =
            ok ? *reinterpret_cast<const float2*>(vbase + (size_t)j * K) : make_float2(0.f, 0.f);
    }
    float sc[CH];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < CH; ++i) {
        const float part_dot = ltf_warp_sum(q2.x * k2[i].x + q2.y * k2[i].y) * scale;
        sc[i] = (j0 + i) <= pos ? part_dot : -INFINITY;
        m = fmaxf(m, sc[i]);
    }
    float sum = 0.f, c0 = 0.f, c1 = 0.f;
    if (m > -INFINITY) {
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            const float pj = __expf(sc[i] - m);
            sum += pj;
            c0 += pj * v2[i].x;
            c1 += pj * v2[i].y;
        }
    }
    if (lane == 0) {
        part.m[chunk] = m;
        part.sum[chunk] = sum;
    }
    part.acc[chunk][lane * 2] = c0;
    part.acc[chunk][lane * 2 + 1] = c1;
    __syncthreads();
    if ((threadIdx.x >> 5) == task_warp0) {
        float M = -INFINITY;
#pragma unroll
        for (int c = 0; c < BLT_ATTN_SPLIT; ++c) M = fmaxf(M, part.m[c]);
        float S = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int c = 0; c < BLT_ATTN_SPLIT; ++c) {
            const float w = part.m[c] > -INFINITY ? __expf(part.m[c] - M) : 0.f;
            S += w * part.sum[c];
            a0 += w * part.acc[c][lane * 2];
            a1 += w * part.acc[c][lane * 2 + 1];
        }
        const float inv = 1.0f / S;
        const float o0 = a0 * inv, o1 = a1 * inv;
        float amax = fmaxf(fabsf(o0), fabsf(o1));
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float qinv = amax > 0.f ? 127.f / amax : 0.f;
        const int q0 = __float2int_rn(o0 * qinv) & 0xff;
        const int q1 = __float2int_rn(o1 * qinv) & 0xff;
        *reinterpret_cast<int16_t*>(ctx_q + (size_t)l * K + d0) = (int16_t)(q0 | (q1 << 8));
        if ((lane & 15) == 0)
            ctx_d[(size_t)l * (K / 32) + d0 / 32] = amax / 127.f;
    }
}

template <class Tier>
__global__ void LTF_CHAIN_LAUNCH_BOUNDS
blt_kernel(const blt_args a) {
    extern __shared__ __align__(16) unsigned char blt_dyn_smem[];
    bf_smem<Tier>& sm = *reinterpret_cast<bf_smem<Tier>*>(blt_dyn_smem);
    __shared__ int s_flag;
    const int tile = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const unsigned int nblocks = gridDim.x;
    unsigned int target = 0;
    const int K = a.K;
    const size_t pos_elem = a.pos_type == LTF_TYPE_F16 ? 2 : 4;
    const size_t emb_elem = a.emb_type == LTF_TYPE_F16 ? 2 : 4;
    auto sync = [&]() { ltf_grid_barrier(a.barrier, target, nblocks); };
#ifdef BF_TIMING
    unsigned long long bf_prev = 0;
    if (tile == 0 && threadIdx.x == 0) {
        bf_prev = bf_now();
        atomicAdd(&g_bf_launches[1], 1ull);
    }
#endif

    ltf_chain_stage(a.layers[0].qkv, K, 3 * K, tile, sm);
    for (int c = 0; c < a.n_rounds; ++c) {
        for (int il = 0; il < a.n_layers; ++il) {
            const blt_layer& L = a.layers[il];
            float* kc = a.kcache + il * a.layer_cache;
            float* vc = a.vcache + il * a.layer_cache;
            {
                bf_phase_args g{};
                g.NL = a.NL;
                g.N = 3 * K;
                g.K = K;
                g.out = a.qkv;
                g.ln_eps = a.ln_eps;
                if (il == 0 && c == 0) {
                    g.assemble = 2;
                    g.h_in = a.h_in;
                    g.pos_row = a.pos;
                    g.pos_type = a.pos_type;
                    g.norm_w = L.norm_self;
                    g.x_store = a.h;
                } else {
                    g.xq = a.h_q;
                    g.xd = a.h_d;
                    g.xgroup = 4;
                    g.stats = a.stats;
                    g.ln_c = L.qkv_c;
                }
                g.kcache = kc;
                g.vcache = vc;
                g.kv_lane_stride = (size_t)a.P * K;
                g.kv_row_stride = (size_t)K;
                g.kv_row_const = c;
                bf_chain_gemv<LTF_EPI_STORE, Tier::r_qkv, 1>(g, tile, sm);
            }
            ltf_chain_stage(L.o, K, K, tile, sm);
            {  // attention over (head, lane) tasks by the last blocks to finish their qkv rows
                const int n_tasks = a.H * a.NL;
                const int attn_blocks =
                    min((n_tasks + BLT_ATTN_TASKS_PER_BLOCK - 1) / BLT_ATTN_TASKS_PER_BLOCK,
                        (int)nblocks);
                const int rank =
                    ltf_chain_arrival_rank(a.barrier + 1, nblocks, attn_blocks, &s_flag);
                const int slot = rank - ((int)nblocks - attn_blocks);
                if (slot >= 0) {
                    const int local_task = warp / BLT_ATTN_SPLIT;
                    blt_attn_partial* parts = reinterpret_cast<blt_attn_partial*>(sm.u.g.s_x);
                    for (int base = slot * BLT_ATTN_TASKS_PER_BLOCK; base < n_tasks;
                         base += attn_blocks * BLT_ATTN_TASKS_PER_BLOCK) {
                        const int task = base + local_task;
                        if (task < n_tasks) {
                            blt_attention_split(
                                a.qkv, kc, vc, a.ctx_q, a.ctx_d, K, a.P, c, a.attn_scale,
                                task / a.NL, task % a.NL, warp % BLT_ATTN_SPLIT, lane,
                                parts[local_task], local_task * BLT_ATTN_SPLIT);
                        } else {
                            __syncthreads();  // matches the barrier inside the split attention
                        }
                        __syncthreads();
                    }
                }
            }
            sync();
            BF_STAMP(1, 0);  // qkv + attention tail
            if (a.prefetch_out && il + 1 == a.n_layers)
                ltf_chain_prefetch_l2(a.out_w[c], K, a.vocab, tile);
            {
                bf_phase_args g{};
                g.NL = a.NL;
                g.xq = a.ctx_q;
                g.xd = a.ctx_d;
                g.xgroup = 32;
                g.N = K;
                g.K = K;
                g.out = a.h;
                g.out_q = a.h_q;
                g.out_d = a.h_d;
                g.out_group = 4;
                g.out_keep_f32 = 1;
                g.out_gamma = L.norm_ff;
                g.out_stats = a.stats;
                bf_chain_gemv<LTF_EPI_RESIDUAL, Tier::r_o, 1>(g, tile, sm);
            }
            ltf_chain_stage(L.ff1, K, a.NFF, tile, sm);
            sync();
            BF_STAMP(1, 1);  // o-proj
            {
                bf_phase_args g{};
                g.NL = a.NL;
                g.xq = a.h_q;
                g.xd = a.h_d;
                g.xgroup = 4;
                g.stats = a.stats;
                g.ln_c = L.ff1_c;
                g.N = a.NFF;
                g.K = K;
                g.out = a.ff;
                g.out_q = a.ff_q;
                g.out_d = a.ff_d;
                g.out_group = 16;
                g.ln_eps = a.ln_eps;
                bf_chain_gemv<LTF_EPI_GELU, Tier::r_ff1, 1>(g, tile, sm);
            }
            ltf_chain_stage(L.ff2, a.NFF, K, tile, sm);
            sync();
            BF_STAMP(1, 2);  // ff1
            {
                bf_phase_args g{};
                g.NL = a.NL;
                g.xq = a.ff_q;
                g.xd = a.ff_d;
                g.xgroup = 16;
                g.N = K;
                g.K = a.NFF;
                g.out = a.h;
                g.out_q = a.h_q;
                g.out_d = a.h_d;
                g.out_group = 4;
                g.out_keep_f32 = 1;
                if (il + 1 < a.n_layers) {
                    g.out_gamma = a.layers[il + 1].norm_self;
                    g.out_stats = a.stats;
                }
                bf_chain_gemv<LTF_EPI_RESIDUAL, Tier::r_o, 3>(g, tile, sm);
            }
            if (il + 1 < a.n_layers)
                ltf_chain_stage(a.layers[il + 1].qkv, K, 3 * K, tile, sm);
            else
                ltf_chain_stage(a.out_w[c], K, a.vocab, tile, sm);
            sync();
            BF_STAMP(1, 3);  // ff2
        }
        {  // output projection + bias -> logits [L][vocab]
            bf_phase_args g{};
            g.NL = a.NL;
            g.xq = a.h_q;
            g.xd = a.h_d;
            g.xgroup = 4;
            g.N = a.vocab;
            g.K = K;
            g.out = a.logits;
            g.bias = a.out_b[c];
            bf_chain_gemv<LTF_EPI_BIAS, Tier::r_out, 1>(g, tile, sm);
        }
        if (c + 1 < a.n_rounds)
            ltf_chain_stage(a.layers[0].qkv, K, 3 * K, tile, sm);
        // the last B blocks to finish their logit rows sample one request each
        const int rank = ltf_chain_arrival_rank(a.barrier + 1, nblocks, a.B, &s_flag);
        const int row = rank - ((int)nblocks - a.B);
        if (row >= 0) {
            const magpietts_cuda_sampling_config config = a.configs[row];
            int k = config.top_k < a.vocab ? config.top_k : a.vocab;
            if (k < 1)
                k = 1;
            if (k > MAGPIETTS_CUDA_MAX_FAST_TOPK)
                k = MAGPIETTS_CUDA_MAX_FAST_TOPK;
            const float* cond = a.logits + (size_t)row * a.vocab;
            const float* uncond = a.use_cfg ? a.logits + (size_t)(a.B + row) * a.vocab : cond;
            int32_t* code_out = a.codes + (size_t)row * a.n_rounds + c;
            magpietts_sample_codebook_fast_block<MAGPIETTS_CUDA_SMALL_ITEMS_PER_THREAD>(
                cond, uncond, a.vocab, a.sample_codebook_size, a.sample_eos_id, config, c, k,
                nullptr, nullptr, code_out, a.argmax + (size_t)row * a.n_rounds + c, sm.u.samp);
            __syncthreads();
            if (c + 1 < a.n_rounds) {
                // next round's layer-0 input of this request's lanes: x = emb[code] + pos[c+1]
                const int code = *code_out;
                const char* emb_row =
                    reinterpret_cast<const char*>(a.audio_emb[c]) + (size_t)code * K * emb_elem;
                const char* pos_row =
                    reinterpret_cast<const char*>(a.pos) + (size_t)(c + 1) * K * pos_elem;
                const float* gamma = a.layers[0].norm_self;
                const int lanes[2] = {row, a.B + row};
                const int n_lanes = a.use_cfg ? 2 : 1;
                float sv = 0.f, sq = 0.f;
                const int tid = (int)threadIdx.x;
                for (int k4 = tid * 4; k4 < K; k4 += LTF_THREADS * 4) {
                    float x[4];
                    float amax = 0.f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        x[i] = ltf_load_scalar(emb_row, a.emb_type, (size_t)(k4 + i)) +
                               ltf_load_scalar(pos_row, a.pos_type, (size_t)(k4 + i));
                        for (int li = 0; li < n_lanes; ++li)
                            a.h[(size_t)lanes[li] * K + k4 + i] = x[i];
                        sv += x[i];
                        sq += x[i] * x[i];
                        x[i] *= gamma[k4 + i];
                        amax = fmaxf(amax, fabsf(x[i]));
                    }
                    const float inv = amax > 0.f ? 127.f / amax : 0.f;
                    int packed = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        packed |= (__float2int_rn(x[i] * inv) & 0xff) << (8 * i);
                    for (int li = 0; li < n_lanes; ++li) {
                        *reinterpret_cast<int*>(a.h_q + (size_t)lanes[li] * K + k4) = packed;
                        a.h_d[(size_t)lanes[li] * (K / 4) + k4 / 4] = amax / 127.f;
                    }
                }
                sv = ltf_warp_sum(sv);
                sq = ltf_warp_sum(sq);
                if (lane == 0) {
                    sm.u.g.s_red[warp * 2] = sv;
                    sm.u.g.s_red[warp * 2 + 1] = sq;
                }
                __syncthreads();
                const int nb = (int)nblocks;
                for (int i = tid; i < n_lanes * 2 * nb; i += LTF_THREADS) {
                    const int li = i / (2 * nb);
                    const int grp = (i / nb) & 1;
                    const int bi = i % nb;
                    float v = 0.f;
                    if (bi == 0)
                        for (int w2 = 0; w2 < LTF_WARPS; ++w2) v += sm.u.g.s_red[w2 * 2 + grp];
                    a.stats[(size_t)(lanes[li] * 2 + grp) * nb + bi] = v;
                }
            }
        }
        if (c + 1 < a.n_rounds) {
            sync();
            BF_STAMP(1, 4);  // output projection + sampling + next input
        }
    }
}

template <class Tier, class Fits>
bool
bf_tier_usable(const void* kernel, int sms, Fits&& fits) {
    if (LTF_CHAIN_BLOCKS_PER_SM * sms < Tier::grid || Tier::grid > LTF_CHAIN_MAX_GRID ||
        !fits(Tier{}))
        return false;
    const int smem = (int)sizeof(bf_smem<Tier>);
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) !=
        cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    if (cudaFuncSetAttribute(
            kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared) != cudaSuccess)
        (void)cudaGetLastError();
    int per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, LTF_THREADS, smem) !=
        cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return per_sm >= LTF_CHAIN_BLOCKS_PER_SM;
}

bool
bf_repack(const void* w, int type, int N, int K, std::vector<void*>& owned, ltf_q8p& out) {
    if (type != LTF_TYPE_Q8_0 || !w)
        return false;
    return ltf_repack_q8(w, N, K, owned, out);
}

}  // namespace

void
magpietts_batch_fused_timing_report() {
#ifdef BF_TIMING
    unsigned long long t[2][16] = {}, n[2] = {};
    cudaMemcpyFromSymbol(t, g_bf_timing, sizeof(t));
    cudaMemcpyFromSymbol(n, g_bf_launches, sizeof(n));
    static const char* names[2][7] = {
        {"qkv", "self-attn", "o-proj+cross tail", "cross-o", "ff1", "ff2", "final"},
        {"qkv+attn", "o-proj", "ff1", "ff2", "out-proj+sample", "", ""}};
    for (int k = 0; k < 2; ++k) {
        if (!n[k])
            continue;
        fprintf(stderr, "batched %s: %llu launches, us per launch:", k ? "LT" : "decoder", n[k]);
        for (int p = 0; p < 7; ++p)
            if (names[k][p][0])
                fprintf(stderr, " %s=%.1f", names[k][p], t[k][p] / 1000.0 / n[k]);
        fprintf(stderr, "\n");
    }
    unsigned long long zero[2][16] = {}, zn[2] = {};
    cudaMemcpyToSymbol(g_bf_timing, zero, sizeof(zero));
    cudaMemcpyToSymbol(g_bf_launches, zn, sizeof(zn));
#endif
}

size_t
magpietts_batch_fused_config_size() {
    return sizeof(magpietts_cuda_sampling_config);
}

void
magpietts_batch_fused_write_config(
    void* configs, int row, bool use_cfg, float cfg_scale, float temperature, int top_k,
    bool forbid_audio_eos, uint64_t seed, int frame_index) {
    magpietts_cuda_sampling_config c{};
    c.cfg_scale = cfg_scale;
    c.temperature = temperature;
    c.top_k = top_k;
    c.frame_index = frame_index;
    c.seed = seed;
    c.use_cfg = use_cfg ? 1 : 0;
    c.forbid_audio_eos = forbid_audio_eos ? 1 : 0;
    static_cast<magpietts_cuda_sampling_config*>(configs)[row] = c;
}

// ---------------------------------------------------------------------------------------
// host: decoder
// ---------------------------------------------------------------------------------------
struct magpietts_batch_decoder_fused {
    std::vector<void*> owned;
    void* arena = nullptr;
    unsigned int* barrier = nullptr;
    bdf_args a{};
    int grid = 0;
};

magpietts_batch_decoder_fused*
magpietts_batch_decoder_fused_create(
    const magpietts_decoder_fused_weights& w, const magpietts_batch_decoder_fused_cache& cache,
    char* error, size_t error_size) {
    if (!ltf_kernel_arch_supported(
            reinterpret_cast<const void*>(bdf_kernel<ltf_chain_tier_large>))) {
        ltf_set_error(error, error_size, "batched fused decoder: needs sm_80+");
        return nullptr;
    }
    if (w.n_embd <= 0 || w.n_embd > BF_MAX_PROLOGUE_K || w.n_embd % 64 != 0 || w.n_head <= 0 ||
        w.n_embd / w.n_head != 64 || w.n_ff <= 0 || w.n_ff > LTF_MAX_K || w.n_ff % 64 != 0 ||
        w.cross_dim <= 0 || w.cross_dim % 32 != 0 || w.cross_dim > BF_MAX_PROLOGUE_K ||
        w.n_layers <= 0 || w.n_layers > BF_MAX_LAYERS || w.n_codebooks <= 0 ||
        w.n_codebooks > BF_MAX_CODEBOOKS || !w.audio_emb || !w.pos_emb || !w.layers ||
        cache.cache_len <= 0 || !cache.kv_arena || cache.arena_lanes <= 0 ||
        cache.text_capacity <= 0 || cache.text_capacity > BF_XATTN_BLOCKS * BF_XATTN_MAX_KEYS ||
        cache.cache_len + 1 > BF_ATTN_SPLITS * BF_ATTN_MAX_RANGE || !cache.cross_k ||
        !cache.cross_v || (w.emb_type != LTF_TYPE_F16 && w.emb_type != LTF_TYPE_F32) ||
        (w.pos_type != LTF_TYPE_F16 && w.pos_type != LTF_TYPE_F32) ||
        BF_MAX_LANES / 2 * w.cross_dim > BF_SX) {
        ltf_set_error(error, error_size, "batched fused decoder: unsupported shape");
        return nullptr;
    }
    auto* f = new magpietts_batch_decoder_fused;
    const int E = w.n_embd, NFF = w.n_ff, CD = w.cross_dim;
    bdf_args& a = f->a;
    int n_collect = 0;
    for (int i = 0; i < w.n_layers; ++i) {
        const magpietts_decoder_fused_layer_weights& L = w.layers[i];
        bdf_layer& d = a.layers[i];
        bool ok = bf_repack(L.qkv, L.qkv_type, 3 * E, E, f->owned, d.qkv) &&
                  bf_repack(L.o, L.o_type, E, E, f->owned, d.o) &&
                  bf_repack(L.ff1, L.ff1_type, NFF, E, f->owned, d.ff1) &&
                  bf_repack(L.ff2, L.ff2_type, E, NFF, f->owned, d.ff2) && L.norm_self && L.norm_ff;
        if (ok && L.has_cross)
            ok = bf_repack(L.cross_q, L.cross_q_type, CD, E, f->owned, d.cq) &&
                 bf_repack(L.cross_o, L.cross_o_type, E, CD, f->owned, d.co) && L.norm_xq;
        if (ok)
            ok = ltf_ln_fold_constants(d.qkv, L.norm_self, 3 * E, E, f->owned, d.qkv_c) &&
                 ltf_ln_fold_constants(d.ff1, L.norm_ff, NFF, E, f->owned, d.ff1_c);
        if (ok && L.has_cross)
            ok = ltf_ln_fold_constants(d.cq, L.norm_xq, CD, E, f->owned, d.cq_c);
        if (!ok) {
            ltf_set_error(error, error_size, "batched fused decoder: Q8_0 projections required");
            magpietts_batch_decoder_fused_free(f);
            return nullptr;
        }
        d.norm_self = L.norm_self;
        d.norm_xq = L.norm_xq;
        d.norm_ff = L.norm_ff;
        d.kv_arena = cache.kv_arena[i];
        d.flags = (L.has_cross ? 1 : 0) | (L.collect_alignment && L.has_cross ? 2 : 0) |
                  (L.apply_prior ? 4 : 0);
        if (L.has_cross && L.collect_alignment)
            ++n_collect;
    }
    int device = 0, sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    auto fits = [&](auto tier) {
        using Tier = decltype(tier);
        const int grid = Tier::grid;
        return E % grid == 0 && (E / grid) % 4 == 0 && NFF % grid == 0 && (NFF / grid) % 16 == 0 &&
               ltf_chain_phase_fits<Tier>(3 * E, E, Tier::r_qkv) &&
               ltf_chain_phase_fits<Tier>(E, E, Tier::r_o) &&
               ltf_chain_phase_fits<Tier>(E, CD, Tier::r_o) &&
               ltf_chain_phase_fits<Tier>(NFF, E, Tier::r_ff1) &&
               ltf_chain_phase_fits<Tier>(E, NFF, Tier::r_o) && (NFF >> 5) <= 96 &&
               BF_XATTN_BLOCKS <= grid && CD % (2 * BF_XATTN_BLOCKS) == 0 &&
               CD / BF_XATTN_BLOCKS <= 2 * LTF_WARPS &&
               // producer staging of every lane's rows of the widest phase
               BF_MAX_LANES * ((std::max(3 * E, NFF) + grid - 1) / grid) <= BF_SX;
    };
    if (bf_tier_usable<ltf_chain_tier_large>(
            reinterpret_cast<const void*>(bdf_kernel<ltf_chain_tier_large>), sms, fits)) {
        f->grid = ltf_chain_tier_large::grid;
    } else if (bf_tier_usable<ltf_chain_tier_small>(
                   reinterpret_cast<const void*>(bdf_kernel<ltf_chain_tier_small>), sms, fits)) {
        f->grid = ltf_chain_tier_small::grid;
    } else {
        ltf_set_error(error, error_size, "batched fused decoder: no persistent tier fits this GPU");
        magpietts_batch_decoder_fused_free(f);
        return nullptr;
    }
    const size_t L = BF_LANE_ALLOC;
    const size_t B = BF_MAX_ROWS;
    size_t total = 0;
    auto reserve = [&](size_t n_floats) {
        const size_t off = total;
        total += (n_floats * sizeof(float) + 255) & ~(size_t)255;
        return off;
    };
    const size_t off_h = reserve(L * E), off_hq = reserve(L * E / 4), off_hd = reserve(L * E / 4),
                 off_stats = reserve(L * 2 * LTF_CHAIN_MAX_GRID), off_qkv = reserve(L * 3 * E),
                 off_apart = reserve(L * w.n_head * BF_ATTN_SPLITS * 66), off_qc = reserve(B * CD),
                 off_xpart = reserve(B * BF_XATTN_BLOCKS * (2 + CD)),
                 off_xs = reserve(B * cache.text_capacity),
                 off_aa = reserve(B * cache.text_capacity), off_ffq = reserve(L * NFF / 4),
                 off_ffd = reserve(L * NFF / 16), off_cq = reserve(L * E / 4),
                 off_cd = reserve(L * E / 32);
    if (cudaMalloc(&f->arena, total) != cudaSuccess ||
        cudaMemset(f->arena, 0, total) != cudaSuccess ||
        cudaMalloc(&f->barrier, (3 + BF_MAX_ROWS) * sizeof(unsigned int)) != cudaSuccess) {
        ltf_set_error(error, error_size, "batched fused decoder: allocation failed");
        magpietts_batch_decoder_fused_free(f);
        return nullptr;
    }
    char* base = reinterpret_cast<char*>(f->arena);
    a.E = E;
    a.H = w.n_head;
    a.NFF = NFF;
    a.CD = CD;
    a.n_layers = w.n_layers;
    a.n_codebooks = w.n_codebooks;
    a.cache_len = cache.cache_len;
    a.TC = cache.text_capacity;
    {
        int l2 = 0;
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device);
        const size_t layer_q8 = ((size_t)4 * E * E + 2 * (size_t)CD * E + 2 * (size_t)NFF * E);
        a.prefetch = (size_t)l2 >= 2 * (layer_q8 + layer_q8 / 16) ? 1 : 0;
    }
    a.ln_eps = w.ln_eps;
    a.attn_scale = 1.0f / sqrtf(64.0f);
    a.xattn_scale = 1.0f / sqrtf((float)CD);
    a.emb_scale = 1.0f / (float)w.n_codebooks;
    for (int c = 0; c < w.n_codebooks; ++c) a.audio_emb[c] = w.audio_emb[c];
    a.emb_type = w.emb_type;
    a.pos_emb = w.pos_emb;
    a.pos_type = w.pos_type;
    a.norm_out = w.norm_out;
    a.cross_k = cache.cross_k;
    a.cross_v = cache.cross_v;
    a.plane_stride = (size_t)cache.arena_lanes * cache.cache_len * E;
    a.n_collect = n_collect;
    a.h = reinterpret_cast<float*>(base + off_h);
    a.h_q = reinterpret_cast<int8_t*>(base + off_hq);
    a.h_d = reinterpret_cast<float*>(base + off_hd);
    a.stats = reinterpret_cast<float*>(base + off_stats);
    a.qkv = reinterpret_cast<float*>(base + off_qkv);
    a.apart = reinterpret_cast<float*>(base + off_apart);
    a.qc = reinterpret_cast<float*>(base + off_qc);
    a.xpart = reinterpret_cast<float*>(base + off_xpart);
    a.xscores = reinterpret_cast<float*>(base + off_xs);
    a.align_acc = reinterpret_cast<float*>(base + off_aa);
    a.ff_q = reinterpret_cast<int8_t*>(base + off_ffq);
    a.ff_d = reinterpret_cast<float*>(base + off_ffd);
    a.ctx_q = reinterpret_cast<int8_t*>(base + off_cq);
    a.ctx_d = reinterpret_cast<float*>(base + off_cd);
    a.barrier = f->barrier;
    if (error && error_size)
        error[0] = '\0';
    return f;
}

void
magpietts_batch_decoder_fused_free(magpietts_batch_decoder_fused* f) {
    if (!f)
        return;
    for (void* p : f->owned) cudaFree(p);
    cudaFree(f->arena);
    cudaFree(f->barrier);
    delete f;
}

bool
magpietts_batch_decoder_fused_step(
    magpietts_batch_decoder_fused* f, cudaStream_t stream,
    const magpietts_batch_decoder_fused_step_args& s, char* error, size_t error_size) {
    if (!f || f->grid <= 0 || s.rows < 1 || s.lanes < s.rows || s.lanes > BF_MAX_LANES ||
        (s.lanes != s.rows && s.lanes != 2 * s.rows) || s.rows > BF_MAX_ROWS ||
        s.rows * f->a.CD > BF_SX || !s.tokens || !s.positions || !s.arena_lane || !s.ring_head ||
        !s.valid_len || !s.cross_slot || !s.text_len || !s.prior || !s.mask || !s.hidden) {
        ltf_set_error(error, error_size, "batched fused decoder: invalid step arguments");
        return false;
    }
    bdf_args a = f->a;
    a.B = s.rows;
    a.NL = s.lanes;
    a.tokens = s.tokens;
    a.positions = s.positions;
    a.arena_lane = s.arena_lane;
    a.ring_head = s.ring_head;
    a.valid_len = s.valid_len;
    a.cross_slot = s.cross_slot;
    a.text_len = s.text_len;
    a.prior = s.prior;
    a.mask = s.mask;
    a.hidden = s.hidden;
    a.alignment = s.alignment;
    cudaError_t err =
        cudaMemsetAsync(f->barrier, 0, (3 + BF_MAX_ROWS) * sizeof(unsigned int), stream);
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "batched fused decoder: barrier reset failed", err);
        return false;
    }
    if (f->grid == ltf_chain_tier_large::grid) {
        bdf_kernel<ltf_chain_tier_large>
            <<<f->grid, LTF_THREADS, sizeof(bf_smem<ltf_chain_tier_large>), stream>>>(a);
    } else {
        bdf_kernel<ltf_chain_tier_small>
            <<<f->grid, LTF_THREADS, sizeof(bf_smem<ltf_chain_tier_small>), stream>>>(a);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "batched fused decoder: launch failed", err);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// host: local transformer
// ---------------------------------------------------------------------------------------
struct magpietts_batch_lt_fused {
    std::vector<void*> owned;
    void* arena = nullptr;
    unsigned int* barrier = nullptr;
    blt_args a{};
    int grid = 0;
};

magpietts_batch_lt_fused*
magpietts_batch_lt_fused_create(
    const magpietts_lt_fused_weights& w, char* error, size_t error_size) {
    if (!ltf_kernel_arch_supported(
            reinterpret_cast<const void*>(blt_kernel<ltf_chain_tier_large>))) {
        ltf_set_error(error, error_size, "batched fused local transformer: needs sm_80+");
        return nullptr;
    }
    if (w.n_embd <= 0 || w.n_embd > BF_MAX_PROLOGUE_K || (w.n_embd % 64) != 0 || w.n_head <= 0 ||
        (w.n_embd / w.n_head) != 64 || w.n_ff <= 0 || (w.n_ff % 64) != 0 || w.n_ff > LTF_MAX_K ||
        w.n_layers <= 0 || w.n_layers > BLT_MAX_LAYERS || w.vocab <= 0 ||
        w.vocab > MAGPIETTS_CUDA_SMALL_VOCAB || w.n_rounds <= 0 || w.n_rounds > BLT_MAX_ROUNDS ||
        w.pos_rows <= 0 || w.pos_rows > LTF_MAX_POS || w.n_rounds > w.pos_rows ||
        w.out_type != LTF_TYPE_Q8_0 ||
        (w.audio_emb_type != LTF_TYPE_F16 && w.audio_emb_type != LTF_TYPE_F32) ||
        (w.pos_type != LTF_TYPE_F16 && w.pos_type != LTF_TYPE_F32) || (w.n_embd >> 5) > 32 ||
        (w.n_ff >> 5) > 96) {
        ltf_set_error(error, error_size, "batched fused local transformer: unsupported shape");
        return nullptr;
    }
    auto* f = new magpietts_batch_lt_fused;
    blt_args& a = f->a;
    const int K = w.n_embd;
    for (int i = 0; i < w.n_layers; ++i) {
        const magpietts_lt_fused_layer_weights& L = w.layers[i];
        blt_layer& d = a.layers[i];
        bool ok = bf_repack(L.qkv, L.qkv_type, 3 * K, K, f->owned, d.qkv) &&
                  bf_repack(L.o, L.o_type, K, K, f->owned, d.o) &&
                  bf_repack(L.ff1, L.ff1_type, w.n_ff, K, f->owned, d.ff1) &&
                  bf_repack(L.ff2, L.ff2_type, K, w.n_ff, f->owned, d.ff2) && L.norm_self &&
                  L.norm_ff;
        if (ok)
            ok = ltf_ln_fold_constants(d.qkv, L.norm_self, 3 * K, K, f->owned, d.qkv_c) &&
                 ltf_ln_fold_constants(d.ff1, L.norm_ff, w.n_ff, K, f->owned, d.ff1_c);
        if (!ok) {
            ltf_set_error(
                error, error_size, "batched fused local transformer: Q8_0 weights required");
            magpietts_batch_lt_fused_free(f);
            return nullptr;
        }
        d.norm_self = L.norm_self;
        d.norm_ff = L.norm_ff;
    }
    for (int r = 0; r < w.n_rounds; ++r) {
        if (!w.out_w[r] || (r > 0 && !w.audio_emb[r - 1]) ||
            !bf_repack(w.out_w[r], w.out_type, w.vocab, K, f->owned, a.out_w[r])) {
            ltf_set_error(error, error_size, "batched fused local transformer: output projection");
            magpietts_batch_lt_fused_free(f);
            return nullptr;
        }
        a.out_b[r] = w.out_b[r];
        a.audio_emb[r] = w.audio_emb[r];
    }
    int device = 0, sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    auto fits = [&](auto tier) {
        using Tier = decltype(tier);
        const int grid = Tier::grid;
        return ltf_chain_phase_fits<Tier>(3 * K, K, Tier::r_qkv) &&
               ltf_chain_phase_fits<Tier>(K, K, Tier::r_o) &&
               ltf_chain_phase_fits<Tier>(w.n_ff, K, Tier::r_ff1) &&
               ltf_chain_phase_fits<Tier>(K, w.n_ff, Tier::r_o) &&
               ltf_chain_phase_fits<Tier>(w.vocab, K, Tier::r_out) && w.n_ff % grid == 0 &&
               (w.n_ff / grid) % 16 == 0 && K % grid == 0 && (K / grid) % 4 == 0 &&
               BF_MAX_ROWS <= grid &&
               BF_MAX_LANES * ((std::max({3 * K, w.vocab, w.n_ff}) + grid - 1) / grid) <= BF_SX;
    };
    if (bf_tier_usable<ltf_chain_tier_large>(
            reinterpret_cast<const void*>(blt_kernel<ltf_chain_tier_large>), sms, fits)) {
        f->grid = ltf_chain_tier_large::grid;
    } else if (bf_tier_usable<ltf_chain_tier_small>(
                   reinterpret_cast<const void*>(blt_kernel<ltf_chain_tier_small>), sms, fits)) {
        f->grid = ltf_chain_tier_small::grid;
    } else {
        ltf_set_error(
            error, error_size, "batched fused local transformer: no persistent tier fits");
        magpietts_batch_lt_fused_free(f);
        return nullptr;
    }
    const size_t L = BF_LANE_ALLOC;
    const size_t cache = (size_t)w.n_layers * L * w.pos_rows * K;
    size_t total = 0;
    auto reserve = [&](size_t n) {
        const size_t off = total;
        total += ((n * sizeof(float)) + 255) & ~(size_t)255;
        return off;
    };
    const size_t off_h = reserve(L * K), off_qkv = reserve(L * 3 * K), off_ctx = reserve(L * K),
                 off_ff = reserve(L * (size_t)w.n_ff), off_logits = reserve(L * (size_t)w.vocab),
                 off_k = reserve(cache), off_v = reserve(cache), off_ctx_q = reserve(L * K / 4),
                 off_ctx_d = reserve(L * K / 32), off_ff_q = reserve(L * (size_t)w.n_ff / 4),
                 off_ff_d = reserve(L * (size_t)w.n_ff / 16), off_h_q = reserve(L * K / 4),
                 off_h_d = reserve(L * K / 4), off_stats = reserve(L * 2 * LTF_CHAIN_MAX_GRID);
    if (cudaMalloc(&f->arena, total) != cudaSuccess ||
        cudaMemset(f->arena, 0, total) != cudaSuccess ||
        cudaMalloc(&f->barrier, 2 * sizeof(unsigned int)) != cudaSuccess) {
        ltf_set_error(error, error_size, "batched fused local transformer: allocation failed");
        magpietts_batch_lt_fused_free(f);
        return nullptr;
    }
    char* base = reinterpret_cast<char*>(f->arena);
    a.K = K;
    a.NFF = w.n_ff;
    a.H = w.n_head;
    a.P = w.pos_rows;
    a.vocab = w.vocab;
    a.n_layers = w.n_layers;
    a.n_rounds = w.n_rounds;
    a.ln_eps = w.ln_eps;
    a.attn_scale = 1.0f / sqrtf((float)(K / w.n_head));
    {
        int l2 = 0;
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device);
        const size_t out_q8 = (size_t)w.vocab * K;
        a.prefetch_out = (size_t)l2 >= 4 * (out_q8 + out_q8 / 16) ? 1 : 0;
    }
    a.pos = w.pos_emb;
    a.pos_type = w.pos_type;
    a.emb_type = w.audio_emb_type;
    a.h = reinterpret_cast<float*>(base + off_h);
    a.qkv = reinterpret_cast<float*>(base + off_qkv);
    a.ctx = reinterpret_cast<float*>(base + off_ctx);
    a.ff = reinterpret_cast<float*>(base + off_ff);
    a.logits = reinterpret_cast<float*>(base + off_logits);
    a.kcache = reinterpret_cast<float*>(base + off_k);
    a.vcache = reinterpret_cast<float*>(base + off_v);
    a.layer_cache = L * w.pos_rows * K;
    a.ctx_q = reinterpret_cast<int8_t*>(base + off_ctx_q);
    a.ctx_d = reinterpret_cast<float*>(base + off_ctx_d);
    a.ff_q = reinterpret_cast<int8_t*>(base + off_ff_q);
    a.ff_d = reinterpret_cast<float*>(base + off_ff_d);
    a.h_q = reinterpret_cast<int8_t*>(base + off_h_q);
    a.h_d = reinterpret_cast<float*>(base + off_h_d);
    a.stats = reinterpret_cast<float*>(base + off_stats);
    a.barrier = f->barrier;
    if (error && error_size)
        error[0] = '\0';
    return f;
}

void
magpietts_batch_lt_fused_free(magpietts_batch_lt_fused* f) {
    if (!f)
        return;
    for (void* p : f->owned) cudaFree(p);
    cudaFree(f->arena);
    cudaFree(f->barrier);
    delete f;
}

bool
magpietts_batch_lt_fused_run(
    magpietts_batch_lt_fused* f, cudaStream_t stream, const magpietts_batch_lt_fused_args& s,
    char* error, size_t error_size) {
    if (!f || f->grid <= 0 || s.rows < 1 || s.rows > BF_MAX_ROWS || s.lanes > BF_MAX_LANES ||
        (s.lanes != s.rows && s.lanes != 2 * s.rows) || !s.hidden || !s.configs || !s.codes ||
        !s.argmax) {
        ltf_set_error(error, error_size, "batched fused local transformer: invalid arguments");
        return false;
    }
    blt_args a = f->a;
    a.B = s.rows;
    a.NL = s.lanes;
    a.use_cfg = s.lanes == 2 * s.rows ? 1 : 0;
    a.h_in = s.hidden;
    a.configs = reinterpret_cast<const magpietts_cuda_sampling_config*>(s.configs);
    a.codes = s.codes;
    a.argmax = s.argmax;
    a.sample_codebook_size = s.audio_codebook_size;
    a.sample_eos_id = s.audio_eos_id;
    cudaError_t err = cudaMemsetAsync(f->barrier, 0, 2 * sizeof(unsigned int), stream);
    if (err != cudaSuccess) {
        ltf_set_error(
            error, error_size, "batched fused local transformer: barrier reset failed", err);
        return false;
    }
    if (f->grid == ltf_chain_tier_large::grid) {
        blt_kernel<ltf_chain_tier_large>
            <<<f->grid, LTF_THREADS, sizeof(bf_smem<ltf_chain_tier_large>), stream>>>(a);
    } else {
        blt_kernel<ltf_chain_tier_small>
            <<<f->grid, LTF_THREADS, sizeof(bf_smem<ltf_chain_tier_small>), stream>>>(a);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        ltf_set_error(error, error_size, "batched fused local transformer: launch failed", err);
        return false;
    }
    return true;
}
