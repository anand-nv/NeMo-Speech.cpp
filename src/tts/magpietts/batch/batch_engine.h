// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// MagpieTTS in-flight (continuous) batching engine (see GOAL.md). Concurrent synthesis requests
// share batched decoder steps; each one streams its own audio and leaves the batch as soon as it
// ends, freeing its slot for a waiting request.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "batch_config.h"
#include "batch_scheduler.h"
#include "magpietts.h"
#include "tts/nanocodec/model.h"

namespace nemo_speech::tts {

class MagpieBatchModelBackend;
class MagpieBatchRequestFactory;

class MagpieBatchEngine {
   public:
    using Done = std::function<void(bool ok)>;

    // The models must outlive the engine. The engine owns the Magpie model's backend while it
    // runs (do not synthesize through another runtime on the same model concurrently).
    MagpieBatchEngine(
        magpietts_model& magpie, const nanocodec::NanoCodecModel& codec,
        const MagpieBatchConfig& config);
    ~MagpieBatchEngine();

    MagpieBatchEngine(const MagpieBatchEngine&) = delete;
    MagpieBatchEngine& operator=(const MagpieBatchEngine&) = delete;

    // Allocates the arenas and starts the scheduler thread.
    bool init(std::string* error);
    void stop();

    // Queue a request; returns its id, or -1 (reason in error) when it cannot run in the engine or
    // the queue is full. `metrics` (optional) must outlive the request; `done` is called once.
    int64_t submit(
        const magpie_stream_params& params, const std::vector<std::vector<int32_t>>& token_chunks,
        const magpie_pcm_callback& pcm_callback, stream_run_metrics* metrics,
        const std::string& run_label, Done done, std::string* error = nullptr);
    bool cancel(int64_t id);

    // Blocking convenience over submit().
    bool synthesize(
        const magpie_stream_params& params, const std::vector<std::vector<int32_t>>& token_chunks,
        const magpie_pcm_callback& pcm_callback, stream_run_metrics& metrics,
        const std::string& run_label = "batch", std::string* error = nullptr);

    MagpieBatchStats stats() const;
    const MagpieBatchConfig& config() const { return config_; }
    // Whether decode steps run the fused CUDA kernels (else the ggml graphs).
    bool fusedKernels() const;

   private:
    magpietts_model& magpie_;
    const nanocodec::NanoCodecModel& codec_;
    MagpieBatchConfig config_;
    std::unique_ptr<MagpieBatchModelBackend> backend_;
    std::unique_ptr<MagpieBatchRequestFactory> factory_;
    std::unique_ptr<MagpieBatchScheduler> scheduler_;
};

}  // namespace nemo_speech::tts
