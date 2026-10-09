// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Synthesis requests for the MagpieTTS in-flight batching scheduler. A request runs the same
// per-request logic as the serial runtime (text windows, attention prior, EOS and chunk-end
// handling, codec streaming; see MagpieRequestRunner in magpietts.cpp); only its decoder steps
// are batched.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "batch_scheduler.h"
#include "magpietts.h"
#include "tts/nanocodec/model.h"

namespace nemo_speech::tts {

// What a request may use of the engine (from the batched decoder's arenas).
struct MagpieBatchRequestLimits {
    bool use_cfg = true;
    int text_capacity = 0;          // longest text window
    int max_decoder_positions = 0;  // decoder steps per text chunk
};

class MagpieBatchRequestFactory {
   public:
    MagpieBatchRequestFactory(magpietts_model& magpie, const nanocodec::NanoCodecModel& codec);
    ~MagpieBatchRequestFactory();

    MagpieBatchRequestFactory(const MagpieBatchRequestFactory&) = delete;
    MagpieBatchRequestFactory& operator=(const MagpieBatchRequestFactory&) = delete;

    // Resolves the request (parameter overrides, text chunks) and checks it fits the engine;
    // nullptr (with the reason in error) otherwise. `metrics` must outlive the request.
    std::shared_ptr<IMagpieBatchRequest> create(
        const magpie_stream_params& params, const std::vector<std::vector<int32_t>>& token_chunks,
        const magpie_pcm_callback& pcm_callback, stream_run_metrics* metrics,
        const std::string& run_label, bool write_codes, const MagpieBatchRequestLimits& limits,
        std::string* error);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts
