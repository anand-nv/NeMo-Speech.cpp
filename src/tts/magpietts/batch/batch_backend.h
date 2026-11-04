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
#include "encoder.h"
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
    const MagpieBatchDecoder& decoder() const { return decoder_; }

    bool prefill(int slot, const MagpieBatchRowInput& row, MagpieBatchRowOutput& out) override;
    // One decoder graph for every row's baked-context prefill, then one local-transformer frame
    // for all of them (the fused kernels or the ggml local transformer). Requires CFG; without
    // it, or before the first prefill filled the unconditional cache, rows are prefilled one at a
    // time.
    bool prefillBatch(
        const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
        std::vector<MagpieBatchRowOutput>& out) override;
    bool decode(
        const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
        std::vector<MagpieBatchRowOutput>& out) override;
    void release(int slot) override;
    bool prefillStream() const override;
    bool prefillBegin(MagpieBatchPrefillJob& job) override;
    bool prefillCompute(MagpieBatchPrefillJob& job) override;
    bool prefillEnd(MagpieBatchPrefillJob& job) override;
    bool batchedEncoder() const override;
    bool encodeTexts(
        const std::vector<const std::vector<int32_t>*>& windows,
        const std::vector<magpietts_backend_tensor*>& outs) override;

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
    // One fused step of up to Fused::max_rows rows on one context, in two halves so that the
    // chunks of a larger step can run on several contexts at once: launch enqueues the step
    // (inputs, decoder kernel, local-transformer kernel, readback) on the context's stream;
    // collect waits for it and fills `out`.
    bool launchFusedChunk(
        Fused& f, const std::vector<int>& slots,
        const std::vector<const MagpieBatchRowInput*>& rows);
    bool collectFusedChunk(
        Fused& f, const std::vector<int>& slots,
        const std::vector<const MagpieBatchRowInput*>& rows,
        std::vector<MagpieBatchRowOutput>& out);
    // Local-transformer frame for `rows` rows; from_prefill takes the round-0 input from the
    // prefill scratch instead of the decoder step.
    bool sampleFused(
        int rows, const std::vector<const MagpieBatchRowInput*>& in,
        std::vector<MagpieBatchRowOutput>& out, bool from_prefill);
    // Rows [first, end) of a batched prefill as decoder prefill rows, and the arenas it fills.
    bool preparePrefillRows(
        const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
        size_t first, std::vector<MagpieBatchRowOutput>& out,
        std::vector<MagpieDecoderPrefillRow>& prow, MagpieDecoderPrefillTargets& targets) const;
    bool ensurePrefillHidden();
    // Whether batched prefills may mark their slots as sharing the unconditional prefix lane:
    // fills it from the prefill decoder's cache once that cache exists.
    bool sharedUncondPrefix();
    // First frames of rows [first, first + n) of a batched prefill (fused kernels or the ggml
    // local transformer), on the decode stream.
    bool firstFrames(
        const std::vector<const MagpieBatchRowInput*>& rows, size_t first, size_t n,
        const ggml_tensor* uncond_hidden, std::vector<MagpieBatchRowOutput>& out);
    // Prefill stream (MagpieBatchConfig::prefill_stream): a second backend on the model's device
    // with its own stream, and the event that orders it after the model stream.
    bool initPrefillStream(std::string* reason);
    // First frame of rows [first, first + n) of a batched prefill on the ggml local transformer,
    // from the prefill's conditional states (prefill_batch_hidden_) and the shared unconditional
    // one.
    bool sampleFirstFrames(
        const std::vector<const MagpieBatchRowInput*>& rows, size_t first, size_t n,
        const ggml_tensor* uncond_hidden, std::vector<MagpieBatchRowOutput>& out);
    bool launchSampleFused(
        Fused& f, int rows, const std::vector<const MagpieBatchRowInput*>& in, bool from_prefill);
    bool collectSampleFused(Fused& f, int rows, std::vector<MagpieBatchRowOutput>& out);

    const magpietts_model& model_;
    MagpieBatchConfig config_;
    MagpieBatchDecoder decoder_;
    MagpieBatchLocalTransformer lt_;
    MagpieDecoder prefill_decoder_;
    MagpieEncoder encoder_;  // batched encoder passes (encodeTexts)
    // prefill scratch, reused by every prefill (the scheduler runs them one at a time)
    DecoderKvCache cond_kv_;
    DecoderKvCache uncond_kv_;
    DecoderCrossKvCache cross_;
    magpietts_backend_tensor hidden_cond_;
    magpietts_backend_tensor hidden_uncond_;
    MagpiePinnedHostScratch staging_;
    ggml_gallocr_t prefill_allocr_ = nullptr;  // kept across prefills
    ggml_gallocr_t cross_allocr_ = nullptr;
    ggml_gallocr_t prefill_batch_allocr_ = nullptr;
    ggml_gallocr_t encode_batch_allocr_ = nullptr;
    ggml_backend_t prefill_backend_ = nullptr;  // prefill stream (nullptr: prefill on the model's)
    void* prefill_ready_ = nullptr;             // cudaEvent_t
    magpietts_backend_tensor
        prefill_batch_hidden_;  // [n_embd, max_slots] conditional hidden states
    std::unique_ptr<Fused> fused_;
    std::unique_ptr<Fused> fused2_;  // second context: chunks of a large step run concurrently
};

}  // namespace nemo_speech::tts
