// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "batch_engine.h"

#include <condition_variable>
#include <mutex>

#include "batch_backend.h"
#include "batch_request.h"

namespace nemo_speech::tts {

MagpieBatchEngine::MagpieBatchEngine(
    magpietts_model& magpie, const nanocodec::NanoCodecModel& codec,
    const MagpieBatchConfig& config)
    : magpie_(magpie), codec_(codec), config_(config) {}

MagpieBatchEngine::~MagpieBatchEngine() {
    stop();
}

bool
MagpieBatchEngine::init(std::string* error) {
    if (scheduler_) {
        return true;
    }
    if (!magpie_.loaded() || !codec_.loaded()) {
        if (error)
            *error = "the Magpie and codec models must be loaded";
        return false;
    }
    if (magpie_.hparams.audio_codebooks != codec_.numCodebooks()) {
        if (error)
            *error = "MagpieTTS and NanoCodec codebook counts differ";
        return false;
    }
    auto backend = std::make_unique<MagpieBatchModelBackend>(magpie_, config_);
    if (!backend->init(error)) {
        return false;
    }
    backend_ = std::move(backend);
    // A request keeps decoding audio after it leaves the decoder batch, while its decoder slot
    // already serves the next request: the codec has twice the slots.
    factory_ = std::make_unique<MagpieBatchRequestFactory>(
        magpie_, codec_, config_.codec_batching ? 2 * config_.max_slots : 0,
        config_.codec_chunk_frames);
    scheduler_ = std::make_unique<MagpieBatchScheduler>(*backend_, config_);
    scheduler_->setCodecCommit([factory = factory_.get()] { factory->commitCodec(); });
    scheduler_->start();
    return true;
}

void
MagpieBatchEngine::stop() {
    if (scheduler_) {
        scheduler_->stop();
    }
}

int64_t
MagpieBatchEngine::submit(
    const magpie_stream_params& params, const std::vector<std::vector<int32_t>>& token_chunks,
    const magpie_pcm_callback& pcm_callback, stream_run_metrics* metrics,
    const std::string& run_label, Done done, std::string* error) {
    if (!scheduler_) {
        if (error)
            *error = "the batching engine is not initialized";
        return -1;
    }
    MagpieBatchRequestLimits limits;
    limits.use_cfg = config_.use_cfg;
    limits.text_capacity = backend_->textCapacity();
    limits.max_decoder_positions = backend_->maxDecoderPositions();
    std::shared_ptr<IMagpieBatchRequest> request = factory_->create(
        params, token_chunks, pcm_callback, metrics, run_label, false, limits, error);
    if (!request) {
        return -1;
    }
    const int64_t id = scheduler_->submit(std::move(request), std::move(done));
    if (id < 0 && error) {
        *error = "the batching queue is full";
    }
    return id;
}

bool
MagpieBatchEngine::cancel(int64_t id) {
    return scheduler_ && scheduler_->cancel(id);
}

bool
MagpieBatchEngine::synthesize(
    const magpie_stream_params& params, const std::vector<std::vector<int32_t>>& token_chunks,
    const magpie_pcm_callback& pcm_callback, stream_run_metrics& metrics,
    const std::string& run_label, std::string* error) {
    std::mutex mutex;
    std::condition_variable cv;
    bool finished = false;
    bool result = false;
    const int64_t id = submit(
        params, token_chunks, pcm_callback, &metrics, run_label,
        [&](bool ok) {
            std::lock_guard<std::mutex> lock(mutex);
            result = ok;
            finished = true;
            cv.notify_all();
        },
        error);
    if (id < 0) {
        return false;
    }
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return finished; });
    return result;
}

bool
MagpieBatchEngine::fusedKernels() const {
    return backend_ && backend_->fused();
}

MagpieBatchStats
MagpieBatchEngine::stats() const {
    return scheduler_ ? scheduler_->stats() : MagpieBatchStats{};
}

}  // namespace nemo_speech::tts
