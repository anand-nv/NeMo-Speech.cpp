// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// ggml_fused_attn_cached over a slot arena, the layout the MagpieTTS in-flight batching decoder
// uses: L query lanes, each mapped by slot_ids to its own circular K/V row of a larger arena,
// with its own ring head, valid length and key-mask column. Checks the CPU forward against a
// scalar reference (output and cache append), and the CUDA kernel against the CPU forward when a
// GPU backend is present.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

constexpr int kHeadDim = 64;
constexpr int kHeads = 12;
constexpr int kFeatures = kHeadDim * kHeads;

struct Case {
    int lanes = 0;
    int slots = 0;  // arena rows (>= lanes)
    int cache_len = 0;
    bool mask = false;
    uint32_t seed = 0;
};

struct Inputs {
    std::vector<float> q, k, v, cache, mask;
    std::vector<int32_t> slot_ids, state;  // state: [lanes] ring heads, then [lanes] valid lengths
};

struct Outputs {
    std::vector<float> out, cache;
};

Inputs
make_inputs(const Case& c) {
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    Inputs in;
    const size_t n = static_cast<size_t>(kFeatures) * c.lanes;
    in.q.resize(n);
    in.k.resize(n);
    in.v.resize(n);
    for (auto* x : {&in.q, &in.k, &in.v})
        for (float& f : *x) f = u(rng);
    in.cache.resize(static_cast<size_t>(kFeatures) * c.cache_len * c.slots * 2);
    for (float& f : in.cache) f = u(rng);
    // Distinct slots in a shuffled, non-contiguous order.
    std::vector<int32_t> all(c.slots);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    in.slot_ids.assign(all.begin(), all.begin() + c.lanes);
    in.state.resize(static_cast<size_t>(c.lanes) * 2);
    std::uniform_int_distribution<int> ring(0, c.cache_len - 1);
    std::uniform_int_distribution<int> valid(1, c.cache_len);
    for (int l = 0; l < c.lanes; ++l) {
        in.state[l] = ring(rng);
        in.state[c.lanes + l] = l == 0 ? c.cache_len : valid(rng);  // lane 0: full cache
    }
    if (c.mask) {
        const int kv = c.cache_len + 1;
        in.mask.assign(static_cast<size_t>(kv) * c.lanes, 0.0f);
        for (int l = 0; l < c.lanes; ++l)
            for (int j = 0; j < kv; ++j)
                if ((j + l) % 5 == 0 && j != kv - 1)
                    in.mask[static_cast<size_t>(l) * kv + j] = -1.0e30f;
    }
    return in;
}

size_t
arena_index(const Case& c, int d, int head, int row, int slot, int plane) {
    return static_cast<size_t>(d) + static_cast<size_t>(kHeadDim) * head +
           static_cast<size_t>(kFeatures) * row +
           static_cast<size_t>(kFeatures) * c.cache_len * slot +
           static_cast<size_t>(kFeatures) * c.cache_len * c.slots * plane;
}

size_t
lane_index(int d, int head, int lane) {
    return static_cast<size_t>(d) + static_cast<size_t>(kHeadDim) * (head + kHeads * lane);
}

Outputs
reference(const Case& c, const Inputs& in) {
    Outputs r;
    r.out.assign(in.q.size(), 0.0f);
    r.cache = in.cache;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    const int kv = c.cache_len + 1;
    for (int l = 0; l < c.lanes; ++l) {
        const int slot = in.slot_ids[l];
        const int ring = in.state[l];
        const int valid = in.state[c.lanes + l];
        const int begin = c.cache_len - valid;
        for (int h = 0; h < kHeads; ++h) {
            std::vector<double> s(kv, 0.0);
            double mx = -INFINITY;
            for (int j = begin; j < kv; ++j) {
                double acc = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    const float key =
                        j == c.cache_len
                            ? in.k[lane_index(d, h, l)]
                            : in.cache[arena_index(c, d, h, (ring + j) % c.cache_len, slot, 0)];
                    acc += static_cast<double>(in.q[lane_index(d, h, l)]) * key;
                }
                acc = acc * scale + (c.mask ? in.mask[static_cast<size_t>(l) * kv + j] : 0.0f);
                s[j] = acc;
                mx = std::max(mx, acc);
            }
            double sum = 0.0;
            for (int j = begin; j < kv; ++j) sum += (s[j] = std::exp(s[j] - mx));
            for (int d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (int j = begin; j < kv; ++j) {
                    const float val =
                        j == c.cache_len
                            ? in.v[lane_index(d, h, l)]
                            : in.cache[arena_index(c, d, h, (ring + j) % c.cache_len, slot, 1)];
                    acc += s[j] * val;
                }
                r.out[lane_index(d, h, l)] = static_cast<float>(acc / sum);
                r.cache[arena_index(c, d, h, ring, slot, 0)] = in.k[lane_index(d, h, l)];
                r.cache[arena_index(c, d, h, ring, slot, 1)] = in.v[lane_index(d, h, l)];
            }
        }
    }
    return r;
}

bool
run_backend(ggml_backend_t backend, const Case& c, const Inputs& in, Outputs& result) {
    ggml_init_params params = {ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, 1, kHeads, c.lanes);
    ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, 1, kHeads, c.lanes);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, 1, kHeads, c.lanes);
    ggml_tensor* cache =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kFeatures * c.cache_len, c.slots, 2);
    ggml_tensor* slot_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.lanes);
    ggml_tensor* state = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.lanes, 2);
    ggml_tensor* mask =
        c.mask ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.cache_len + 1, c.lanes) : nullptr;
    ggml_tensor* out = ggml_fused_attn_cached(
        ctx, q, k, v, mask, cache, slot_ids, state, c.cache_len,
        1.0f / std::sqrt(static_cast<float>(kHeadDim)), false);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(q, in.q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, in.k.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, in.v.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(cache, in.cache.data(), 0, ggml_nbytes(cache));
    ggml_backend_tensor_set(slot_ids, in.slot_ids.data(), 0, ggml_nbytes(slot_ids));
    ggml_backend_tensor_set(state, in.state.data(), 0, ggml_nbytes(state));
    if (mask)
        ggml_backend_tensor_set(mask, in.mask.data(), 0, ggml_nbytes(mask));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    result.out.resize(in.q.size());
    result.cache.resize(in.cache.size());
    if (ok) {
        ggml_backend_tensor_get(out, result.out.data(), 0, ggml_nbytes(out));
        ggml_backend_tensor_get(cache, result.cache.data(), 0, ggml_nbytes(cache));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

float
max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            return INFINITY;
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

}  // namespace

int
main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    ggml_backend_t gpu = nullptr;
    if (ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        gpu = ggml_backend_dev_init(dev, nullptr);
    }
    if (!gpu)
        std::printf("no GPU backend: checking the CPU forward only\n");

    const std::vector<Case> cases = {
        {1, 1, 18, false, 1},    {2, 3, 18, false, 2},   {4, 9, 37, true, 3},
        {8, 17, 64, true, 4},    {16, 33, 111, true, 5}, {32, 33, 130, true, 6},
        {16, 20, 611, false, 7},
    };
    int failures = 0;
    for (const Case& c : cases) {
        const Inputs in = make_inputs(c);
        const Outputs ref = reference(c, in);
        Outputs got_cpu;
        if (!run_backend(cpu, c, in, got_cpu)) {
            std::fprintf(stderr, "FAIL: CPU compute lanes=%d\n", c.lanes);
            ++failures;
            continue;
        }
        const float e_out = max_diff(got_cpu.out, ref.out);
        const bool cache_ok = got_cpu.cache == ref.cache;
        std::printf(
            "lanes=%2d slots=%2d cache=%3d mask=%d  cpu-vs-ref out=%.3g cache=%s", c.lanes, c.slots,
            c.cache_len, c.mask ? 1 : 0, e_out, cache_ok ? "exact" : "MISMATCH");
        if (e_out > 2e-5f || !cache_ok)
            ++failures;
        if (gpu) {
            Outputs got_gpu;
            if (!run_backend(gpu, c, in, got_gpu)) {
                std::printf("  cuda=FAIL\n");
                ++failures;
                continue;
            }
            const float e_gpu = max_diff(got_gpu.out, got_cpu.out);
            const bool gpu_cache_ok = got_gpu.cache == ref.cache;
            std::printf(
                "  cuda-vs-cpu out=%.3g cache=%s", e_gpu, gpu_cache_ok ? "exact" : "MISMATCH");
            if (e_gpu > 2e-5f || !gpu_cache_ok)
                ++failures;
        }
        std::printf("\n");
    }
    if (gpu)
        ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    if (failures) {
        std::fprintf(stderr, "FAIL: %d slot-attention check(s)\n", failures);
        return 1;
    }
    std::printf("slot attention OK\n");
    return 0;
}
