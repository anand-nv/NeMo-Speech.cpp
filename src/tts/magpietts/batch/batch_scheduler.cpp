// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "batch_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "nvtx_utils.h"

namespace nemo_speech::tts {

MagpieBatchScheduler::MagpieBatchScheduler(IMagpieBatchBackend& backend, const MagpieBatchConfig& config)
    : backend_(backend), config_(config), pool_(config.max_slots),
      slot_owner_(static_cast<size_t>(std::max(0, config.max_slots)), 0) {
    config_.max_prefills_per_iter = std::max(1, config_.max_prefills_per_iter);
    config_.max_decode_steps_before_prefill = std::max(0, config_.max_decode_steps_before_prefill);
}

MagpieBatchScheduler::~MagpieBatchScheduler() {
    stop();
}

void
MagpieBatchScheduler::start() {
    if (thread_.joinable() || stopping_) {
        return;
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
        if (e->cancel) {
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
    if (pending.empty()) {
        return false;
    }
    // Prefill takes priority over decoding, but at most every max_decode_steps_before_prefill
    // decode steps while other requests are decoding, so it cannot stall them for long.
    if (decoding && decode_steps_since_prefill_ < config_.max_decode_steps_before_prefill) {
        return false;
    }
    decode_steps_since_prefill_ = 0;
    const auto t0 = std::chrono::steady_clock::now();
    struct AddTime {
        MagpieBatchScheduler* s;
        std::chrono::steady_clock::time_point t0;
        ~AddTime() {
            std::lock_guard<std::mutex> lock(s->mutex_);
            s->stats_.prefill_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
    } add_time{this, t0};
    const int budget = std::min<int>(config_.max_prefills_per_iter, static_cast<int>(pending.size()));
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
MagpieBatchScheduler::runDecode() {
    const auto t0 = std::chrono::steady_clock::now();
    struct AddTime {
        MagpieBatchScheduler* s;
        std::chrono::steady_clock::time_point t0;
        ~AddTime() {
            std::lock_guard<std::mutex> lock(s->mutex_);
            s->stats_.decode_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
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
        const bool forming = config_.static_batching && !waiting_.empty();
        wake_.wait_for(lock, std::chrono::milliseconds(forming ? 2 : 50), [&] {
            return stopping_ || !incoming_.empty() || !cancel_ids_.empty();
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
