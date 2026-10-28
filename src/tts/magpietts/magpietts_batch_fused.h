// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Persistent (single-launch) CUDA kernels for in-flight batched MagpieTTS (see GOAL.md): one
// decoder step and one local-transformer frame (all rounds plus in-kernel sampling) for B
// requests at once. Lanes are planar: lanes [0, B) are the requests' conditional lanes and, with
// CFG, lanes [B, 2B) their unconditional lanes. Every lane runs exactly the arithmetic of the
// two-lane kernels (magpietts_decoder_fused / magpietts_lt_fused chain), so a request's results
// do not depend on what else is in the batch. Requires Q8_0 projections on sm_80+.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "magpietts_decoder_fused.h"
#include "magpietts_lt_fused.h"

// Most lanes a batched step handles (16 requests with CFG, 32 without).
constexpr int MAGPIETTS_BATCH_FUSED_MAX_LANES = 32;
// Largest top-k of the in-kernel sampler.
constexpr int MAGPIETTS_BATCH_FUSED_MAX_TOPK = 256;

// Per-row sampling configs (device layout of magpietts_cuda_sampling_config) for
// magpietts_batch_lt_fused_args::configs.
size_t magpietts_batch_fused_config_size();
// -DBF_TIMING builds: print and reset the per-phase kernel timing (no-op otherwise).
void magpietts_batch_fused_timing_report();
void magpietts_batch_fused_write_config(
    void* configs, int row, bool use_cfg, float cfg_scale, float temperature, int top_k,
    bool forbid_audio_eos, uint64_t seed, int frame_index);

struct magpietts_batch_decoder_fused;

struct magpietts_batch_decoder_fused_cache {
    int cache_len = 0;                 // self-attention rows per lane
    int arena_lanes = 0;               // lanes of the slot arena
    float* const* kv_arena = nullptr;  // per layer: [2 planes][arena_lanes][cache_len][n_embd]
    int text_capacity = 0;             // cross-attention rows per slot and layer
    int arena_slots = 0;
    const float* cross_k = nullptr;  // [arena_slots][n_layers][text_capacity][cross_dim]
    const float* cross_v = nullptr;
};

// Device arrays of one batched step (B rows, L lanes).
struct magpietts_batch_decoder_fused_step_args {
    int rows = 0;                         // B
    int lanes = 0;                        // B (no CFG) or 2B
    const int32_t* tokens = nullptr;      // [B][n_codebooks]
    const int32_t* positions = nullptr;   // [B]
    const int32_t* arena_lane = nullptr;  // [L] slot-arena lane of every lane
    const int32_t* ring_head = nullptr;   // [L]
    const int32_t* valid_len = nullptr;   // [L]
    const int32_t* cross_slot = nullptr;  // [B]
    const int32_t* text_len = nullptr;    // [B]
    const float* prior = nullptr;         // [B][text_capacity] log prior (+ pad mask)
    const float* mask = nullptr;          // [B][text_capacity] pad mask
    float* hidden = nullptr;              // [L][n_embd] final hidden states
    float* alignment = nullptr;           // [B][text_capacity] mean cross-attention, or nullptr
};

magpietts_batch_decoder_fused* magpietts_batch_decoder_fused_create(
    const magpietts_decoder_fused_weights& w, const magpietts_batch_decoder_fused_cache& cache,
    char* error, size_t error_size);
void magpietts_batch_decoder_fused_free(magpietts_batch_decoder_fused* f);
bool magpietts_batch_decoder_fused_step(
    magpietts_batch_decoder_fused* f, cudaStream_t stream,
    const magpietts_batch_decoder_fused_step_args& step, char* error, size_t error_size);

struct magpietts_batch_lt_fused;

// One frame of the local transformer for B rows: every round of every lane, with each row
// sampled in-kernel from its own config.
struct magpietts_batch_lt_fused_args {
    int rows = 0;
    int lanes = 0;
    const float* hidden = nullptr;  // [L][n_embd] decoder hidden states (round-0 input)
    const void* configs = nullptr;  // [B] magpietts_cuda_sampling_config (device)
    int32_t* codes = nullptr;       // [B][n_rounds]
    int32_t* argmax = nullptr;      // [B][n_rounds]
    int audio_codebook_size = 0;
    int audio_eos_id = 0;
};

magpietts_batch_lt_fused* magpietts_batch_lt_fused_create(
    const magpietts_lt_fused_weights& w, char* error, size_t error_size);
void magpietts_batch_lt_fused_free(magpietts_batch_lt_fused* f);
bool magpietts_batch_lt_fused_run(
    magpietts_batch_lt_fused* f, cudaStream_t stream, const magpietts_batch_lt_fused_args& args,
    char* error, size_t error_size);
