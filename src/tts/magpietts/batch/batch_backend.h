// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Model side of the MagpieTTS in-flight batching engine: per-slot prefill (eager baked-context
// prefill into shared scratch caches, handed over to the slot) and batched decode steps (the
// batched decoder graph followed by the batched local transformer).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "batch_config.h"
#include "batch_decoder.h"
#include "batch_lt.h"
#include "batch_scheduler.h"
#include "decoder.h"
#include "model.h"

namespace nemo_speech::tts {

class MagpieBatchModelBackend final : public IMagpieBatchBackend {
   public:
    MagpieBatchModelBackend(const magpietts_model& model, const MagpieBatchConfig& config);
    ~MagpieBatchModelBackend() override;

    bool init(std::string* error);

    int textCapacity() const { return decoder_.textCapacity(); }
    int maxDecoderPositions() const { return decoder_.maxDecoderPositions(); }
    const MagpieBatchConfig& config() const { return config_; }

    bool prefill(int slot, const MagpieBatchRowInput& row, MagpieBatchRowOutput& out) override;
    bool decode(
        const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
        std::vector<MagpieBatchRowOutput>& out) override;
    void release(int slot) override;

    // Whether decode steps run the fused CUDA kernels (Q8_0 model on sm_80+).
    bool fused() const { return fused_ != nullptr; }
    // -DBF_TIMING builds of the fused kernels: print and reset their per-phase timing.
    void kernelTimingReport() const;

   private:
    struct Fused;
    bool initFused(std::string* reason);
    bool decodeFused(
        const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
        std::vector<MagpieBatchRowOutput>& out);
    // Local-transformer frame for `rows` rows; from_prefill takes the round-0 input from the
    // prefill scratch instead of the decoder step.
    bool sampleFused(
        int rows, const std::vector<const MagpieBatchRowInput*>& in,
        std::vector<MagpieBatchRowOutput>& out, bool from_prefill);

    const magpietts_model& model_;
    MagpieBatchConfig config_;
    MagpieBatchDecoder decoder_;
    MagpieBatchLocalTransformer lt_;
    MagpieDecoder prefill_decoder_;
    // prefill scratch, reused by every prefill (the scheduler runs them one at a time)
    DecoderKvCache cond_kv_;
    DecoderKvCache uncond_kv_;
    DecoderCrossKvCache cross_;
    magpietts_backend_tensor hidden_cond_;
    magpietts_backend_tensor hidden_uncond_;
    MagpiePinnedHostScratch staging_;
    ggml_gallocr_t prefill_allocr_ = nullptr;  // kept across prefills
    ggml_gallocr_t cross_allocr_ = nullptr;
    std::unique_ptr<Fused> fused_;
};

}  // namespace nemo_speech::tts
