// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "batch_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "nvtx_utils.h"

namespace nemo_speech::tts {

MagpieBatchScheduler::MagpieBatchScheduler(
    IMagpieBatchBackend& backend, const MagpieBatchConfig& config)
    : backend_(backend), config_(config), pool_(config.max_slots),
      slot_owner_(static_cast<size_t>(std::max(0, config.max_slots)), 0) {
    config_.max_prefills_per_iter = std::max(1, config_.max_prefills_per_iter);
    config_.max_decode_steps_before_prefill = std::max(0, config_.max_decode_steps_before_prefill);
    config_.prefill_batch_size = std::max(1, config_.prefill_batch_size);
}

MagpieBatchScheduler::~MagpieBatchScheduler() {
    stop();
}

void
MagpieBatchScheduler::start() {
    if (thread_.joinable() || stopping_) {
        return;
    }
    if (asyncPrefill()) {
        prefill_thread_ = std::thread(&MagpieBatchScheduler::prefillLoop, this);
    }
    thread_ = std::thread(&MagpieBatchScheduler::loop, this);
}

void
MagpieBatchScheduler::stop() {
    if (stopping_.exchange(true)) {
        return;
    }
    wake_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
    { std::lock_guard<std::mutex> lock(prefill_mutex_); }
    prefill_cv_.notify_all();
    if (prefill_thread_.joinable()) {
        prefill_thread_.join();  // a running prefill completes first
    }
    // Fail whatever is left: never-admitted requests just report failure, admitted ones stop
    // their audio output.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (EntryPtr& e : incoming_) waiting_.push_back(e);
        incoming_.clear();
    }
    for (EntryPtr& e : waiting_) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --stats_.queued;
            ++stats_.failed;
        }
        e->done(false);
    }
    waiting_.clear();
    const std::vector<EntryPtr> active = active_;
    for (const EntryPtr& e : active) fail(e);
    reapFinishers(true);
}

int64_t
MagpieBatchScheduler::submit(RequestPtr request, Done done) {
    if (!request) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || static_cast<size_t>(std::max(0, stats_.queued)) >= config_.max_queue_depth) {
        ++stats_.rejected;
        return -1;
    }
    auto e = std::make_shared<Entry>();
    e->id = next_id_++;
    e->request = std::move(request);
    e->done = std::move(done);
    incoming_.push_back(e);
    ++stats_.submitted;
    ++stats_.queued;
    wake_.notify_all();
    return e->id;
}

bool
MagpieBatchScheduler::cancel(int64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id <= 0 || id >= next_id_) {
        return false;
    }
    cancel_ids_.push_back(id);
    wake_.notify_all();
    return true;
}

MagpieBatchStats
MagpieBatchScheduler::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

int
MagpieBatchScheduler::slotOf(int64_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t s = 0; s < slot_owner_.size(); ++s) {
        if (slot_owner_[s] == id) {
            return static_cast<int>(s);
        }
    }
    return -1;
}

void
MagpieBatchScheduler::ingest() {
    std::vector<int64_t> cancels;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (EntryPtr& e : incoming_) waiting_.push_back(std::move(e));
        incoming_.clear();
        cancels.swap(cancel_ids_);
    }
    for (int64_t id : cancels) {
        for (EntryPtr& e : waiting_)
            if (e->id == id)
                e->cancel = true;
        for (EntryPtr& e : active_)
            if (e->id == id)
                e->cancel = true;
    }
}

void
MagpieBatchScheduler::dropCancelled() {
    for (auto it = waiting_.begin(); it != waiting_.end();) {
        if ((*it)->cancel) {
            EntryPtr e = *it;
            it = waiting_.erase(it);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --stats_.queued;
                ++stats_.cancelled;
            }
            e->done(false);
        } else {
            ++it;
        }
    }
    const std::vector<EntryPtr> active = active_;
    for (const EntryPtr& e : active) {
        // a request in a running prefill is failed when the prefill ends (its slot is in use)
        if (e->cancel && e->phase != Phase::PrefillRunning) {
            fail(e);
        }
    }
}

void
MagpieBatchScheduler::admit() {
    if (config_.static_batching) {
        // a batch runs to completion; the next one forms once it is full or has waited enough
        if (!active_.empty() || waiting_.empty()) {
            batch_wait_start_ = {};
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (batch_wait_start_ == std::chrono::steady_clock::time_point{})
            batch_wait_start_ = now;
        if (static_cast<int>(waiting_.size()) < pool_.capacity() &&
            now - batch_wait_start_ < std::chrono::milliseconds(config_.static_batch_wait_ms))
            return;
        batch_wait_start_ = {};
    }
    while (!waiting_.empty() && pool_.available() > 0) {
        EntryPtr e = waiting_.front();
        waiting_.pop_front();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --stats_.queued;
        }
        e->slot = pool_.acquire();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            slot_owner_[static_cast<size_t>(e->slot)] = e->id;
        }
        e->phase = Phase::Prefill;
        active_.push_back(e);
        if (!e->request->start()) {
            fail(e);
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.active = static_cast<int>(active_.size());
    stats_.max_active = std::max(stats_.max_active, stats_.active);
}

void
MagpieBatchScheduler::removeActive(const EntryPtr& e) {
    active_.erase(std::remove(active_.begin(), active_.end(), e), active_.end());
    if (e->slot >= 0) {
        backend_.release(e->slot);
        pool_.release(e->slot);
        std::lock_guard<std::mutex> lock(mutex_);
        slot_owner_[static_cast<size_t>(e->slot)] = 0;
        e->slot = -1;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.active = static_cast<int>(active_.size());
}

void
MagpieBatchScheduler::fail(const EntryPtr& e) {
    if (std::find(active_.begin(), active_.end(), e) == active_.end()) {
        return;
    }
    removeActive(e);
    e->ended_ok = false;
    // abort on the finishing thread: stopping the codec worker joins it
    finishAsync(e);
}

void
MagpieBatchScheduler::endChunk(const EntryPtr& e) {
    bool more = false;
    if (!e->request->endTextChunk(more)) {
        fail(e);
        return;
    }
    if (more) {
        e->phase = Phase::Prefill;  // keeps its slot; the next prefill reseeds it
        return;
    }
    removeActive(e);
    e->ended_ok = true;
    finishAsync(e);
}

void
MagpieBatchScheduler::finishAsync(const EntryPtr& e) {
    const bool ok_path = e->ended_ok && !e->cancel;
    auto flag = std::make_shared<std::atomic<bool>>(false);
    ++finishing_;
    Finisher f;
    f.done = flag;
    f.thread = std::thread([this, e, ok_path, flag]() {
        bool ok = false;
        if (ok_path) {
            ok = e->request->finish();
        } else {
            e->request->abort();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (ok)
                ++stats_.completed;
            else if (e->cancel)
                ++stats_.cancelled;
            else
                ++stats_.failed;
        }
        e->done(ok);
        flag->store(true);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --finishing_;
        }
        idle_.notify_all();
        wake_.notify_all();
    });
    finishers_.push_back(std::move(f));
}

void
MagpieBatchScheduler::reapFinishers(bool all) {
    for (auto it = finishers_.begin(); it != finishers_.end();) {
        if (all || it->done->load()) {
            it->thread.join();
            it = finishers_.erase(it);
        } else {
            ++it;
        }
    }
}

bool
MagpieBatchScheduler::runPrefills() {
    std::vector<EntryPtr> pending;
    bool decoding = false;
    for (const EntryPtr& e : active_) {
        if (e->phase == Phase::Prefill)
            pending.push_back(e);
        else if (e->phase == Phase::Decode)
            decoding = true;
    }
    if (asyncPrefill()) {
        // one prefill in flight on the prefill stream; decode steps continue meanwhile
        JobState state;
        {
            std::lock_guard<std::mutex> lock(prefill_mutex_);
            state = job_state_;
        }
        if (state == JobState::Done) {
            finishPrefillJob();
            return true;
        }
        if (state != JobState::None || pending.empty() || !prefillWindowClosed(pending.size())) {
            return false;
        }
        return submitPrefillJob(pending);
    }
    if (pending.empty()) {
        return false;
    }
    // Prefill takes priority over decoding, but at most every max_decode_steps_before_prefill
    // decode steps while other requests are decoding, so it cannot stall them for long.
    if (decoding && decode_steps_since_prefill_ < config_.max_decode_steps_before_prefill) {
        return false;
    }
    if (config_.prefill_batch_size > 1 && !prefillWindowClosed(pending.size())) {
        return false;  // keep collecting; decode steps continue meanwhile
    }
    decode_steps_since_prefill_ = 0;
    const auto t0 = std::chrono::steady_clock::now();
    struct AddTime {
        MagpieBatchScheduler* s;
        std::chrono::steady_clock::time_point t0;
        ~AddTime() {
            std::lock_guard<std::mutex> lock(s->mutex_);
            s->stats_.prefill_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
        }
    } add_time{this, t0};
    if (config_.prefill_batch_size > 1) {
        runPrefillBatch(pending);
        return true;
    }
    const int budget =
        std::min<int>(config_.max_prefills_per_iter, static_cast<int>(pending.size()));
    for (int i = 0; i < budget; ++i) {
        const EntryPtr& e = pending[static_cast<size_t>(i)];
        const ggml_nvtx::range nvtx_range("magpietts_batch_prefill");
        if (!e->request->beginTextChunk()) {
            fail(e);
            continue;
        }
        e->row = MagpieBatchRowInput{};
        IMagpieBatchRequest::Status st = e->request->prepareStep(e->row);
        if (st == IMagpieBatchRequest::Status::Failed) {
            fail(e);
            continue;
        }
        if (st == IMagpieBatchRequest::Status::EndChunk) {
            endChunk(e);
            continue;
        }
        MagpieBatchRowOutput out;
        if (!backend_.prefill(e->slot, e->row, out)) {
            fail(e);
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.prefills;
        }
        st = e->request->applyStep(out);
        if (st == IMagpieBatchRequest::Status::Failed) {
            fail(e);
        } else if (st == IMagpieBatchRequest::Status::EndChunk) {
            endChunk(e);
        } else {
            e->phase = Phase::Decode;
        }
    }
    return true;
}

bool
MagpieBatchScheduler::prefillWindowClosed(size_t pending) {
    if (config_.prefill_wait_ms <= 0) {
        return true;
    }
    const auto now = std::chrono::steady_clock::now();
    if (prefill_window_start_ == std::chrono::steady_clock::time_point{}) {
        prefill_window_start_ = now;
    }
    // more requests can join while a slot is free or a request is queued for one
    const bool can_grow = pool_.available() > 0 || !waiting_.empty();
    if (static_cast<int>(pending) < config_.prefill_batch_size && can_grow &&
        now - prefill_window_start_ < std::chrono::milliseconds(config_.prefill_wait_ms)) {
        return false;
    }
    prefill_window_start_ = {};
    return true;
}

void
MagpieBatchScheduler::runPrefillBatch(const std::vector<EntryPtr>& pending) {
    // up to prefill_batch_size waiting requests: their text chunks are prepared one by one, then
    // prefilled together
    const int budget = std::min<int>(config_.prefill_batch_size, static_cast<int>(pending.size()));
    const bool defer = backend_.batchedEncoder();
    std::vector<EntryPtr> begun;
    std::vector<EntryPtr> encode;
    std::vector<const std::vector<int32_t>*> windows;
    std::vector<magpietts_backend_tensor*> encoded;
    for (int i = 0; i < budget; ++i) {
        const EntryPtr& e = pending[static_cast<size_t>(i)];
        if (!(defer ? e->request->beginTextChunkDeferred() : e->request->beginTextChunk())) {
            fail(e);
            continue;
        }
        begun.push_back(e);
        magpietts_backend_tensor* out = nullptr;
        if (const std::vector<int32_t>* window = e->request->textWindowToEncode(&out)) {
            encode.push_back(e);
            windows.push_back(window);
            encoded.push_back(out);
        }
    }
    if (!encode.empty()) {
        bool ok = false;
        {
            const ggml_nvtx::range nvtx_range("magpietts_batch_encode");
            ok = backend_.encodeTexts(windows, encoded);
        }
        for (const EntryPtr& e : encode) {
            if (ok) {
                e->request->textEncoded();
            } else {
                fail(e);
            }
        }
        if (!ok) {
            fprintf(
                stderr, "MagpieTTS batch: batched encoder pass of %zu requests failed\n",
                encode.size());
            begun.erase(
                std::remove_if(
                    begun.begin(), begun.end(),
                    [&](const EntryPtr& e) {
                        return std::find(encode.begin(), encode.end(), e) != encode.end();
                    }),
                begun.end());
        }
    }
    std::vector<EntryPtr> batch;
    std::vector<int> slots;
    std::vector<const MagpieBatchRowInput*> rows;
    for (const EntryPtr& e : begun) {
        e->row = MagpieBatchRowInput{};
        const IMagpieBatchRequest::Status st = e->request->prepareStep(e->row);
        if (st == IMagpieBatchRequest::Status::Failed) {
            fail(e);
        } else if (st == IMagpieBatchRequest::Status::EndChunk) {
            endChunk(e);
        } else {
            batch.push_back(e);
            slots.push_back(e->slot);
            rows.push_back(&e->row);
        }
    }
    if (batch.empty()) {
        return;
    }
    std::vector<MagpieBatchRowOutput> outs;
    bool ok = false;
    {
        const ggml_nvtx::range nvtx_range("magpietts_batch_prefill");
        ok = backend_.prefillBatch(slots, rows, outs) && outs.size() == batch.size();
    }
    if (!ok) {
        fprintf(stderr, "MagpieTTS batch: batched prefill of %zu requests failed\n", batch.size());
        for (const EntryPtr& e : batch) fail(e);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.prefills += batch.size();
    }
    for (size_t i = 0; i < batch.size(); ++i) {
        const EntryPtr& e = batch[i];
        const IMagpieBatchRequest::Status st = e->request->applyStep(outs[i]);
        if (st == IMagpieBatchRequest::Status::Failed) {
            fail(e);
        } else if (st == IMagpieBatchRequest::Status::EndChunk) {
            endChunk(e);
        } else {
            e->phase = Phase::Decode;
        }
    }
}

bool
MagpieBatchScheduler::asyncPrefill() const {
    return config_.prefill_stream && config_.prefill_batch_size > 1 && backend_.prefillStream();
}

bool
MagpieBatchScheduler::submitPrefillJob(const std::vector<EntryPtr>& pending) {
    // the requests' own steps run here, on the scheduler thread; the prefill thread only runs
    // the GPU work of the job
    const int budget = std::min<int>(config_.prefill_batch_size, static_cast<int>(pending.size()));
    const bool defer = backend_.batchedEncoder();
    auto job = std::make_unique<MagpieBatchPrefillJob>();
    std::vector<EntryPtr> entries;
    for (int i = 0; i < budget; ++i) {
        const EntryPtr& e = pending[static_cast<size_t>(i)];
        if (!(defer ? e->request->beginTextChunkDeferred() : e->request->beginTextChunk())) {
            fail(e);
            continue;
        }
        magpietts_backend_tensor* out = nullptr;
        const std::vector<int32_t>* window = e->request->textWindowToEncode(&out);
        if (window) {
            e->request->textEncoded();  // the job encodes it before its prefill reads it
        }
        e->row = MagpieBatchRowInput{};
        const IMagpieBatchRequest::Status st = e->request->prepareStep(e->row);
        if (st == IMagpieBatchRequest::Status::Failed) {
            fail(e);
            continue;
        }
        if (st == IMagpieBatchRequest::Status::EndChunk) {
            endChunk(e);
            continue;
        }
        if (window) {
            job->windows.push_back(window);
            job->encoded.push_back(out);
            e->row.text_cond_device = out;  // allocated by the job's encoder pass
        }
        job->slots.push_back(e->slot);
        job->rows.push_back(&e->row);
        entries.push_back(e);
    }
    if (entries.empty()) {
        return true;
    }
    if (!backend_.prefillBegin(*job)) {
        for (const EntryPtr& e : entries) fail(e);
        return true;
    }
    for (const EntryPtr& e : entries) e->phase = Phase::PrefillRunning;
    job_ = std::move(job);
    job_entries_ = std::move(entries);
    if (!prefill_thread_.joinable()) {
        // manual mode (runOnce without start): compute inline
        const bool ok = backend_.prefillCompute(*job_);
        std::lock_guard<std::mutex> lock(prefill_mutex_);
        job_ok_ = ok;
        job_state_ = JobState::Done;
        return true;
    }
    {
        std::lock_guard<std::mutex> lock(prefill_mutex_);
        job_state_ = JobState::Queued;
    }
    prefill_cv_.notify_all();
    return true;
}

void
MagpieBatchScheduler::prefillLoop() {
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(prefill_mutex_);
            prefill_cv_.wait(lock, [&] { return stopping_ || job_state_ == JobState::Queued; });
            if (job_state_ != JobState::Queued) {
                return;  // stopping
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        bool ok = false;
        {
            const ggml_nvtx::range nvtx_range("magpietts_batch_prefill_stream");
            ok = backend_.prefillCompute(*job_);
        }
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.prefill_ms += ms;
        }
        {
            std::lock_guard<std::mutex> lock(prefill_mutex_);
            job_ok_ = ok;
            job_state_ = JobState::Done;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_done_ = true;
        }
        wake_.notify_all();
    }
}

void
MagpieBatchScheduler::finishPrefillJob() {
    std::unique_ptr<MagpieBatchPrefillJob> job = std::move(job_);
    std::vector<EntryPtr> entries = std::move(job_entries_);
    bool ok;
    {
        std::lock_guard<std::mutex> lock(prefill_mutex_);
        ok = job_ok_;
        job_state_ = JobState::None;
    }
    job_done_ = false;
    {
        const ggml_nvtx::range nvtx_range("magpietts_batch_prefill_first_frames");
        ok = ok && backend_.prefillEnd(*job) && job->out.size() == entries.size();
    }
    if (!ok) {
        fprintf(stderr, "MagpieTTS batch: prefill of %zu requests failed\n", entries.size());
    } else {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.prefills += entries.size();
    }
    for (size_t i = 0; i < entries.size(); ++i) {
        const EntryPtr& e = entries[i];
        e->phase = Phase::Prefill;
        if (!ok || e->cancel) {
            fail(e);
            continue;
        }
        const IMagpieBatchRequest::Status st = e->request->applyStep(job->out[i]);
        if (st == IMagpieBatchRequest::Status::Failed) {
            fail(e);
        } else if (st == IMagpieBatchRequest::Status::EndChunk) {
            endChunk(e);
        } else {
            e->phase = Phase::Decode;
        }
    }
}

bool
MagpieBatchScheduler::runDecode() {
    const auto t0 = std::chrono::steady_clock::now();
    struct AddTime {
        MagpieBatchScheduler* s;
        std::chrono::steady_clock::time_point t0;
        ~AddTime() {
            std::lock_guard<std::mutex> lock(s->mutex_);
            s->stats_.decode_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
        }
    } add_time{this, t0};
    std::vector<EntryPtr> batch;
    const std::vector<EntryPtr> active = active_;
    for (const EntryPtr& e : active) {
        if (e->phase != Phase::Decode) {
            continue;
        }
        e->row = MagpieBatchRowInput{};
        const IMagpieBatchRequest::Status st = e->request->prepareStep(e->row);
        if (st == IMagpieBatchRequest::Status::Failed) {
            fail(e);
        } else if (st == IMagpieBatchRequest::Status::EndChunk) {
            endChunk(e);
        } else {
            batch.push_back(e);
        }
    }
    if (batch.empty()) {
        return false;
    }
    const ggml_nvtx::range nvtx_range("magpietts_batch_decode_step");
    std::vector<int> slots;
    std::vector<const MagpieBatchRowInput*> rows;
    for (const EntryPtr& e : batch) {
        slots.push_back(e->slot);
        rows.push_back(&e->row);
    }
    std::vector<MagpieBatchRowOutput> outs;
    if (!backend_.decode(slots, rows, outs) || outs.size() != batch.size()) {
        fprintf(stderr, "MagpieTTS batch: decode step failed for %zu requests\n", batch.size());
        for (const EntryPtr& e : batch) fail(e);
        return true;
    }
    ++decode_steps_since_prefill_;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.decode_steps;
        stats_.decode_rows += batch.size();
    }
    for (size_t i = 0; i < batch.size(); ++i) {
        const EntryPtr& e = batch[i];
        const IMagpieBatchRequest::Status st = e->request->applyStep(outs[i]);
        if (st == IMagpieBatchRequest::Status::Failed) {
            fail(e);
        } else if (st == IMagpieBatchRequest::Status::EndChunk) {
            endChunk(e);
        }
    }
    return true;
}

bool
MagpieBatchScheduler::runOnce() {
    reapFinishers(false);
    ingest();
    dropCancelled();
    admit();
    bool did = runPrefills();
    did = runDecode() || did;
    if (!did) {
        idle_.notify_all();
    }
    return did;
}

void
MagpieBatchScheduler::loop() {
    while (!stopping_) {
        if (runOnce()) {
            continue;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        // a forming batch (static batching, or a batched prefill waiting for requests) is
        // checked again soon so its timeout fires on time
        const bool forming = (config_.static_batching && !waiting_.empty()) ||
                             prefill_window_start_ != std::chrono::steady_clock::time_point{};
        wake_.wait_for(lock, std::chrono::milliseconds(forming ? 2 : 50), [&] {
            return stopping_ || !incoming_.empty() || !cancel_ids_.empty() || job_done_;
        });
    }
}

void
MagpieBatchScheduler::waitIdle() {
    for (;;) {
        if (thread_.joinable()) {
            std::unique_lock<std::mutex> lock(mutex_);
            idle_.wait_for(lock, std::chrono::milliseconds(20));
            if (incoming_.empty() && stats_.queued == 0 && stats_.active == 0 && finishing_ == 0)
                return;
            continue;
        }
        const bool did = runOnce();
        if (!did && waiting_.empty() && active_.empty()) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!incoming_.empty() || !cancel_ids_.empty())
                continue;
            if (finishing_ == 0)
                break;
            idle_.wait_for(lock, std::chrono::milliseconds(20));
        }
    }
    if (!thread_.joinable()) {
        reapFinishers(true);
    }
}

}  // namespace nemo_speech::tts
