// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Per-request MagpieTTS state, separated from the resources shared by every request (encoder,
// decoder runtimes, local-transformer samplers, codec decoder; see MagpieStreamingWorkspace).
// A request is one synthesis call: its resolved parameters, its progress through the text chunks
// and the buffers it decodes into. The streaming runtime owns one of each today; the in-flight
// batching engine (GOAL.md) owns one per slot.
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "decoder.h"
#include "model.h"
#include "tts/nanocodec/model.h"

namespace nemo_speech::tts {

// Device and host buffers a request decodes into. They belong to a workspace slot and are reused
// by successive requests on it: the allocations survive, and so does the persistent decoder
// runtime, which is keyed on the cross cache buffer.
struct MagpieRequestBuffers {
    std::vector<float> text_cond;
    magpietts_backend_tensor text_cond_device;
    magpietts_backend_tensor cond_hidden_device;
    magpietts_backend_tensor uncond_hidden_device;
    DecoderKvCache cond_kv;
    DecoderKvCache uncond_kv;
    DecoderCrossKvCache cond_cross_kv;
    nanocodec::NanoCodecStreamState codec_stream_state;

    // Longform chunk prefetch targets: the next text chunk's encoder output, cross K/V and
    // baked-context prefill, built on the workspace's side backend.
    DecoderKvCache prefetch_cond_kv;
    DecoderKvCache prefetch_uncond_kv;
    DecoderCrossKvCache prefetch_cross_kv;
    magpietts_backend_tensor prefetch_text_cond_device;
    magpietts_backend_tensor prefetch_cond_hidden;
    magpietts_backend_tensor prefetch_uncond_hidden;
    MagpiePinnedHostScratch prefetch_staging;

    // Drop the previous request's decoder and codec state (allocations are kept).
    void beginRequest() {
        cond_kv.clear();
        uncond_kv.clear();
        cond_cross_kv.clear();
        codec_stream_state.clear();
    }
};

// Longform chunk prefetch: while chunk k is being generated, a helper thread runs chunk k+1's
// encoder pass, cross K/V build and baked-context prefill. Everything it needs is known ahead
// except the text history length, which depends on where the attention prior leaves chunk k; it
// is speculated with the default window and verified at the boundary (any mismatch falls back to
// the regular synchronous path).
struct MagpieChunkPrefetch {
    std::thread thread;
    bool launched = false;
    bool ok = false;
    size_t chunk_index = 0;
    int text_len = 0;
    std::vector<int32_t> text_window;
    std::vector<float> text_cond;
    std::vector<float> prior;
    bool collect_alignment = false;
    std::vector<float> alignment;

    MagpieChunkPrefetch() = default;
    MagpieChunkPrefetch(const MagpieChunkPrefetch&) = delete;
    MagpieChunkPrefetch& operator=(const MagpieChunkPrefetch&) = delete;
    ~MagpieChunkPrefetch() { join(); }

    void join() {
        if (launched) {
            thread.join();
            launched = false;
        }
    }
};

// The text chunk a request is currently generating (one per longform chunk; a single chunk
// otherwise).
struct MagpieTextChunkState {
    size_t index = 0;
    const std::vector<int32_t>* tokens = nullptr;  // the chunk's own tokens (no history)
    std::vector<int32_t> text_window;              // history + chunk tokens fed to the encoder
    int text_len = 0;
    bool first = false;
    bool final = false;
    bool adopted_prefetch = false;
    int max_decoder_positions = 0;  // stacked decoder steps allowed for this chunk
    int step = 0;                   // next stacked decoder step
    // Audio codes fed back to the decoder, [audio_codebooks][frames] (BOS frame first).
    std::vector<std::vector<int32_t>> audio_codes;
    int frames_generated = 0;  // codec frames emitted for this chunk
    int near_end_frames = 0;
    bool suppress_nonfinal_codec_output = false;
    int suppressed_nonfinal_frames = 0;
};

// What one decoder step produced for a request, before it is applied to the request's state.
struct MagpieStepOutput {
    bool forbid_eos = false;
    int frames_remaining = 0;
    std::vector<int32_t> next_codes;    // sampled stacked codes [stacked_audio_codebooks]
    std::vector<int32_t> argmax_codes;  // greedy stacked codes [stacked_audio_codebooks]
    bool alignment_collected = false;
    std::vector<float> alignment_scores;  // cross-attention alignment over the text window
};

// Resolved parameters and generation progress of one synthesis request.
struct MagpieRequestState {
    // Model hyperparameters with the request's overrides (steps, top_k, temperature, cfg_scale).
    magpietts_hparams h;
    std::vector<std::vector<int32_t>> token_chunks;
    size_t total_text_tokens = 0;
    bool longform_active = false;
    bool use_cuda_lt = false;
    bool use_cuda_sampling = false;
    double codec_fps = 0.0;
    int window_samples = 0;

    std::mt19937 rng;
    std::mt19937 boundary_silence_rng;
    std::uniform_int_distribution<int> boundary_silence_dist{6, 10};
    std::vector<std::vector<int32_t>> forced_code_frames;  // MAGPIETTS_FORCE_CODES

    int frames_generated = 0;          // codec frames emitted over all chunks
    int decoder_frames_generated = 0;  // decoder frames over all chunks (sampler frame index)
    bool first_audio_pending = true;

    // Longform history carried across text chunks.
    MagpieLongformAttentionPriorState attention_prior;
    std::vector<int32_t> prior_text_tokens;
    std::vector<float> history_text_context;
    int history_text_context_len = 0;
    int absolute_token_offset = 0;
    MagpiePinnedHostScratch text_context_staging;

    bool prefetch_enabled = false;
    int prefetch_max_window = 0;
    MagpieChunkPrefetch chunk_prefetch;

    MagpieTextChunkState chunk;
};

}  // namespace nemo_speech::tts
