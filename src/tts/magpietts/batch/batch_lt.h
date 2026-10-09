// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Batched MagpieTTS local transformer (in-flight batching, see GOAL.md): samples the stacked
// codebooks of one frame for every row of a decoder step. Each round is one fixed-shape ggml
// graph per bucket; the logits come back to the host, where every row is sampled with its own
// parameters and random generator, exactly as the serial CPU sampler does.
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
    std::mt19937* rng = nullptr;
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
    bool buildBucket(int bucket, Bucket& out);

    const magpietts_model& model_;
    bool use_cfg_ = true;
    int threads_ = 1;
    std::vector<std::unique_ptr<Bucket>> buckets_;  // indexed by bucket size
    std::vector<float> logits_;
};

}  // namespace nemo_speech::tts
