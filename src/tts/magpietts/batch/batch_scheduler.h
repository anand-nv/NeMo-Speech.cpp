// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// In-flight (continuous) batching scheduler for MagpieTTS (see GOAL.md).
//
// One scheduler thread owns the model. Each iteration it admits queued requests into free slots,
// runs a bounded amount of prefill (text encoder + baked-context decoder prefill + first frame)
// and decodes one frame for every request in the decode phase as one batched step. A request
// that reaches EOS, its step limit or a cancellation leaves the batch in that iteration; its
// slot is reused from the next one. Draining the audio (codec flush) runs off the scheduler
// thread.
//
// The scheduler only sees the two interfaces below, so it is tested with fakes; the model-backed
// implementations live in batch_backend.* and in magpietts.cpp (the request adapter).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "batch_config.h"
#include "model.h"
#include "slot_pool.h"

namespace nemo_speech::tts {

// What a request feeds into one decoder step (or its text chunk's prefill).
struct MagpieBatchRowInput {
    // Decode steps: stacked codes of the previous frame.
    std::vector<int32_t> tokens;
    // Text rows of the current chunk, its attention prior (or nullptr) and whether the step
    // must return the cross-attention alignment.
    int text_len = 0;
    const std::vector<float>* prior = nullptr;
    bool collect_alignment = false;
    // Sampling.
    float temperature = 0.6f;
    float cfg_scale = 2.5f;
    int top_k = 80;
    bool forbid_eos = false;
    uint64_t seed = 0;
    int frame_index = 0;  // decoder frame index (counter-based samplers)
    std::mt19937* rng = nullptr;
    // Prefill (the first step of a text chunk): the encoded text and the speaker.
    const std::vector<float>* text_cond = nullptr;
    const magpietts_backend_tensor* text_cond_device = nullptr;
    int speaker = 0;
};

// What a step produced for a request.
struct MagpieBatchRowOutput {
    std::vector<int32_t> codes;    // sampled stacked codes
    std::vector<int32_t> argmax;   // greedy stacked codes
    std::vector<float> alignment;  // when collected: mean cross-attention over the text rows
};

// The per-request side driven by the scheduler (all calls on the scheduler thread except
// finish()/abort(), which run on a finishing thread).
class IMagpieBatchRequest {
   public:
    enum class Status { Continue, EndChunk, Failed };

    virtual ~IMagpieBatchRequest() = default;
    // Admission (a slot was assigned): start the request's audio output. False = failed.
    virtual bool start() = 0;
    // Prepare the next text chunk (text window, encoder). False = failed.
    virtual bool beginTextChunk() = 0;
    // As beginTextChunk, but the request may leave its encoder pass to the scheduler, which
    // encodes several requests' windows at once: textWindowToEncode then returns the window and
    // the tensor to encode it into, and textEncoded must follow before prepareStep.
    virtual bool beginTextChunkDeferred() { return beginTextChunk(); }
    virtual const std::vector<int32_t>* textWindowToEncode(magpietts_backend_tensor** out) {
        (void)out;
        return nullptr;
    }
    virtual void textEncoded() {}
    // Before every step: fill the row (prefill fields included on a chunk's first step).
    // EndChunk when the chunk's step budget is used up, Failed when the request broke.
    virtual Status prepareStep(MagpieBatchRowInput& row) = 0;
    // Apply the step's output; EndChunk when the chunk ended with this frame.
    virtual Status applyStep(const MagpieBatchRowOutput& out) = 0;
    // Close the current chunk; more = another chunk follows. False = failed.
    virtual bool endTextChunk(bool& more) = 0;
    // Drain the audio output; returns overall success.
    virtual bool finish() = 0;
    // Stop the audio output after a failure or cancellation.
    virtual void abort() = 0;
};

// The model side: per-slot prefill, batched decode steps.
// A batched prefill whose GPU work runs on the backend's prefill stream, on the scheduler's
// prefill thread, while decode steps continue (MagpieBatchConfig::prefill_stream).
struct MagpieBatchPrefillJob {
    std::vector<int> slots;
    std::vector<const MagpieBatchRowInput*> rows;
    // deferred encoder passes of these rows (IMagpieBatchRequest::textWindowToEncode)
    std::vector<const std::vector<int32_t>*> windows;
    std::vector<magpietts_backend_tensor*> encoded;
    std::vector<MagpieBatchRowOutput> out;  // the rows' first frames (and alignments)
    int n_tokens = 0;                       // decoder rows the prefill wrote per lane
};

class IMagpieBatchBackend {
   public:
    virtual ~IMagpieBatchBackend() = default;
    // Prefill `row`'s text chunk into `slot` and sample its first frame.
    virtual bool prefill(int slot, const MagpieBatchRowInput& row, MagpieBatchRowOutput& out) = 0;
    // Prefill several rows' text chunks at once (MagpieBatchConfig::prefill_batch_size); the
    // default prefills them one at a time. out[i] is row i's first frame.
    virtual bool prefillBatch(
        const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
        std::vector<MagpieBatchRowOutput>& out) {
        out.assign(rows.size(), MagpieBatchRowOutput{});
        for (size_t i = 0; i < rows.size(); ++i)
            if (!prefill(slots[i], *rows[i], out[i]))
                return false;
        return true;
    }
    // One frame for every (slot, row) pair, in one batched step.
    virtual bool decode(
        const std::vector<int>& slots, const std::vector<const MagpieBatchRowInput*>& rows,
        std::vector<MagpieBatchRowOutput>& out) = 0;
    // The slot's request left the batch.
    virtual void release(int slot) = 0;
    // Whether batched prefills can run on a stream of their own (the three calls below): begin
    // on the scheduler thread orders the prefill stream after the work already queued for
    // decoding, compute on the prefill thread runs the encoder and the decoder prefill there
    // (touching only the job's slots), end on the scheduler thread adds the first frames.
    virtual bool prefillStream() const { return false; }
    virtual bool prefillBegin(MagpieBatchPrefillJob& job) {
        (void)job;
        return false;
    }
    virtual bool prefillCompute(MagpieBatchPrefillJob& job) {
        (void)job;
        return false;
    }
    virtual bool prefillEnd(MagpieBatchPrefillJob& job) {
        (void)job;
        return false;
    }
    // Whether encodeTexts is supported.
    virtual bool batchedEncoder() const { return false; }
    // Encode several text windows at once into outs[i] (deferred encoder passes, see
    // IMagpieBatchRequest::beginTextChunkDeferred).
    virtual bool encodeTexts(
        const std::vector<const std::vector<int32_t>*>& windows,
        const std::vector<magpietts_backend_tensor*>& outs) {
        (void)windows;
        (void)outs;
        return false;
    }
};

struct MagpieBatchStats {
    uint64_t submitted = 0;
    uint64_t completed = 0;
    uint64_t failed = 0;
    uint64_t cancelled = 0;
    uint64_t rejected = 0;
    uint64_t decode_steps = 0;
    uint64_t decode_rows = 0;  // sum of rows over decode steps (mean batch = rows / steps)
    uint64_t prefills = 0;
    double prefill_ms = 0.0;  // scheduler time in prefill (encoder, decoder prefill, first frame)
    double decode_ms = 0.0;   // scheduler time in batched decode steps (incl. request updates)
    int max_active = 0;
    int queued = 0;
    int active = 0;
};

class MagpieBatchScheduler {
   public:
    using RequestPtr = std::shared_ptr<IMagpieBatchRequest>;
    // Called once per request, from the finishing thread or the scheduler thread.
    using Done = std::function<void(bool ok)>;

    MagpieBatchScheduler(IMagpieBatchBackend& backend, const MagpieBatchConfig& config);
    ~MagpieBatchScheduler();

    MagpieBatchScheduler(const MagpieBatchScheduler&) = delete;
    MagpieBatchScheduler& operator=(const MagpieBatchScheduler&) = delete;

    // Run the loop on an internal thread. Without start(), the caller drives it with runOnce().
    void start();
    // Fails queued requests, aborts active ones and joins every thread.
    void stop();

    // Queue a request; returns its id, or -1 when the queue is full or the scheduler stopped
    // (done is not called then).
    int64_t submit(RequestPtr request, Done done);
    // Cancel a queued or active request (its done(false) follows). False for unknown ids.
    bool cancel(int64_t id);

    // One scheduler iteration; false when there was nothing to do.
    bool runOnce();
    // Wait until no request is queued, active or finishing (manual mode helper for tests).
    void waitIdle();

    MagpieBatchStats stats() const;
    // Slot of an active request (-1 otherwise); for tests.
    int slotOf(int64_t id) const;

   private:
    // PrefillRunning: in a batched prefill on the prefill thread (not touched until it ends)
    enum class Phase { Queued, Prefill, PrefillRunning, Decode };
    struct Entry {
        int64_t id = 0;
        RequestPtr request;
        Done done;
        Phase phase = Phase::Queued;
        int slot = -1;
        bool cancel = false;
        bool ended_ok = false;  // left the batch after its last chunk (vs failed/cancelled)
        MagpieBatchRowInput row;
    };
    using EntryPtr = std::shared_ptr<Entry>;

    void loop();
    void ingest();
    void dropCancelled();
    void admit();
    bool runPrefills();
    // Whether a batched prefill of `pending` requests runs now
    // (MagpieBatchConfig::prefill_wait_ms).
    bool prefillWindowClosed(size_t pending);
    void runPrefillBatch(const std::vector<EntryPtr>& pending);
    // Asynchronous batched prefill (MagpieBatchConfig::prefill_stream).
    bool asyncPrefill() const;
    bool submitPrefillJob(const std::vector<EntryPtr>& pending);
    void finishPrefillJob();
    void prefillLoop();
    bool runDecode();
    void endChunk(const EntryPtr& e);
    void fail(const EntryPtr& e);
    void finishAsync(const EntryPtr& e);
    void reapFinishers(bool all);
    void removeActive(const EntryPtr& e);

    IMagpieBatchBackend& backend_;
    MagpieBatchConfig config_;
    MagpieSlotPool pool_;

    mutable std::mutex mutex_;  // guards incoming_, cancel_ids_, stats_, the id maps
    std::condition_variable wake_;
    std::condition_variable idle_;
    std::deque<EntryPtr> incoming_;
    std::vector<int64_t> cancel_ids_;
    std::atomic<bool> stopping_{false};
    std::thread thread_;
    int64_t next_id_ = 1;
    MagpieBatchStats stats_;
    std::chrono::steady_clock::time_point batch_wait_start_{};      // static batching
    std::chrono::steady_clock::time_point prefill_window_start_{};  // batched prefill window

    // scheduler-thread state
    std::deque<EntryPtr> waiting_;
    std::vector<EntryPtr> active_;  // admitted (prefill or decode), in admission order
    int decode_steps_since_prefill_ = 0;
    struct Finisher {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> done;
    };
    std::vector<Finisher> finishers_;
    std::atomic<int> finishing_{0};
    std::vector<int64_t> slot_owner_;  // slot -> request id (or 0), read by slotOf

    // asynchronous batched prefill: the scheduler thread owns job_ except while job_state_ is
    // Queued (the prefill thread computes it)
    enum class JobState { None, Queued, Done };
    std::thread prefill_thread_;
    std::mutex prefill_mutex_;  // guards job_state_, job_ok_
    std::condition_variable prefill_cv_;
    JobState job_state_ = JobState::None;
    bool job_ok_ = false;
    std::atomic<bool> job_done_{false};  // wakes the scheduler loop
    std::unique_ptr<MagpieBatchPrefillJob> job_;
    std::vector<EntryPtr> job_entries_;
};

}  // namespace nemo_speech::tts
