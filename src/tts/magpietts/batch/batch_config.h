// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Settings of the MagpieTTS in-flight batching engine (see GOAL.md).
#pragma once

#include <cstddef>
#include <vector>

namespace nemo_speech::tts {

struct MagpieBatchConfig {
    // Requests decoding together (slots of the K/V arenas). Each one owns its self-attention
    // cache rows and its cross-attention (text) rows until it finishes.
    int max_slots = 8;
    // Every request of the engine runs classifier-free guidance (two lanes per request:
    // conditional and unconditional), or none does (one lane).
    bool use_cfg = true;
    // Decode-step graph shapes: a step with B active requests runs the smallest bucket >= B,
    // the tail padded with the reserved dummy slot. Must be ascending; the largest bucket must
    // be >= max_slots (it is raised to max_slots otherwise).
    std::vector<int> buckets = {1, 2, 4, 8, 16};
    // Text rows per slot of the cross-attention arena: the longest text window (longform
    // history + chunk tokens) a request may use.
    int text_capacity = 512;
    // Decoder steps (stacked frames) a request may take per text chunk; 0 = the model's
    // max_decoder_steps. Sizes the per-slot self-attention ring cache.
    int max_decoder_positions = 0;
    // Prefill work (text encoder + baked-context decoder prefill + first frame) the scheduler
    // runs between two decode steps, and how many decode steps may run back to back while
    // requests wait for prefill.
    int max_prefills_per_iter = 2;
    int max_decode_steps_before_prefill = 1;
    // Requests prefilled together in one batched prefill (one decoder graph and one
    // local-transformer frame for all of them); it is then also the prefill budget between two
    // decode steps. 1 prefills requests one at a time (bit-identical to the serial runtime).
    int prefill_batch_size = 1;
    // With prefill_batch_size > 1: how long the first request waiting for prefill may wait for
    // others to join its batch. The batch then runs when it is full, when this time has passed,
    // or when no further request can join (every slot taken, none queued). Decode steps continue
    // meanwhile. 0 prefills whatever is waiting at once.
    int prefill_wait_ms = 0;
    // With prefill_batch_size > 1 on CUDA: batched prefills (encoder and decoder prefill) run on
    // their own stream and thread while decode steps continue; only the first frames join the
    // decode stream.
    bool prefill_stream = false;
    // Decode the requests' audio in one batched codec (CUDA): the chunks requests submit while a
    // batch runs are decoded together in the next one, bit-identical to decoding them alone.
    // codec_chunk_frames is the requests' chunk size (MagpieRuntimeConfig::chunk_frames).
    bool codec_batching = false;
    int codec_chunk_frames = 4;
    // Requests waiting for a slot; submissions beyond it are rejected.
    size_t max_queue_depth = 256;
    // CPU threads for CPU backends (encoder, decoder, local transformer).
    int threads = 4;
    // Use the fused CUDA kernels (Q8_0 models on sm_80+) when available.
    bool use_fused_kernels = true;
    // Fused-kernel contexts (streams) that run chunks of a step concurrently when a step has more
    // rows than one launch holds. Only used when every context's persistent grid fits on the GPU
    // at once; 1 runs the chunks one after another.
    int fused_streams = 2;
    // Reference policy for benchmarks (static batching): admit a new batch only when no request
    // is active, waiting up to static_batch_wait_ms for the queue to fill it.
    bool static_batching = false;
    int static_batch_wait_ms = 50;
};

}  // namespace nemo_speech::tts
