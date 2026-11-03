// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Batched MagpieTTS local transformer (in-flight batching, see GOAL.md): samples the stacked
// codebooks of one frame for every row of a decoder step. Each round is one fixed-shape ggml
// graph per bucket. On CUDA every row is then sampled on the GPU (the fused local transformer's
// sampler: counter-based draws from the row's seed and frame index), whose codes feed the next
// round on the device, so a frame's rounds run without a host round trip. Elsewhere the logits
// come back to the host, where every row is sampled with its own parameters and random
// generator, exactly as the serial CPU sampler does.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "model.h"

namespace nemo_speech::tts {

// Sampling parameters of one row.
struct MagpieBatchSampling {
    float temperature = 0.6f;
    float cfg_scale = 2.5f;
    int top_k = 80;
    bool forbid_eos = false;
    std::mt19937* rng = nullptr;  // host sampling
    uint64_t seed = 0;            // GPU sampling: counter-based draws per (seed, frame, codebook)
    int frame_index = 0;
};

class MagpieBatchLocalTransformer {
   public:
    MagpieBatchLocalTransformer(const magpietts_model& model, bool use_cfg, int threads);
    ~MagpieBatchLocalTransformer();

    MagpieBatchLocalTransformer(const MagpieBatchLocalTransformer&) = delete;
    MagpieBatchLocalTransformer& operator=(const MagpieBatchLocalTransformer&) = delete;

    bool init(std::string* error);

    // Lanes of a bucket's input: [n_embd, lanes] (planar: cond rows, then uncond rows).
    int lanes(int bucket) const { return bucket * (use_cfg_ ? 2 : 1); }

    // Samples one stacked frame for rows [0, sampling.size()) of `bucket`. `fill_input` copies
    // the decoder's final hidden states into the round-0 input tensor [n_embd, lanes(bucket)].
    // codes[b] / argmax[b] receive row b's sampled and greedy stacked codes.
    bool sample(
        int bucket, const std::vector<MagpieBatchSampling>& sampling,
        const std::function<bool(ggml_tensor* input)>& fill_input,
        std::vector<std::vector<int32_t>>& codes, std::vector<std::vector<int32_t>>& argmax);

   private:
    struct Graph;
    struct Bucket;
    struct DeviceSampling;
    bool buildBucket(int bucket, Bucket& out);
    // GPU sampling of every round (CUDA); false = not available here, use the host sampler.
    bool sampleDevice(
        Bucket& b, int bucket, const std::vector<MagpieBatchSampling>& sampling,
        std::vector<std::vector<int32_t>>& codes, std::vector<std::vector<int32_t>>& argmax,
        bool* failed);

    const magpietts_model& model_;
    bool use_cfg_ = true;
    int threads_ = 1;
    std::vector<std::unique_ptr<Bucket>> buckets_;  // indexed by bucket size
    std::vector<float> logits_;
    std::unique_ptr<DeviceSampling> device_;
};

}  // namespace nemo_speech::tts
