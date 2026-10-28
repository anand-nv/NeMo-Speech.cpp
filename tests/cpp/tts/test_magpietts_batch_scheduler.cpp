// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// MagpieBatchScheduler with a fake backend and scripted requests (no model): admission order,
// the slot bound, eviction and slot reuse, the prefill budget, longform re-prefill on the same
// slot, cancellation, queue-depth rejection, failure handling and the static-batching reference
// policy.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "tts/magpietts/batch/batch_scheduler.h"

namespace tts = nemo_speech::tts;
using Status = tts::IMagpieBatchRequest::Status;

namespace {

int failures = 0;

void
expect(bool cond, const std::string& what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Log {
    std::mutex mutex;
    std::vector<std::string> events;
    void add(const std::string& e) {
        std::lock_guard<std::mutex> lock(mutex);
        events.push_back(e);
    }
};

class FakeBackend final : public tts::IMagpieBatchBackend {
   public:
    explicit FakeBackend(int max_slots) : max_slots_(max_slots) {}
    bool prefill(
        int slot, const tts::MagpieBatchRowInput& row, tts::MagpieBatchRowOutput& out) override {
        ++prefills;
        prefill_slots.push_back(slot);
        if (fail_prefill_slot == slot)
            return false;
        out.codes.assign(1, row.frame_index);
        return true;
    }
    bool decode(
        const std::vector<int>& slots, const std::vector<const tts::MagpieBatchRowInput*>& rows,
        std::vector<tts::MagpieBatchRowOutput>& out) override {
        ++decodes;
        max_rows = std::max(max_rows, static_cast<int>(slots.size()));
        std::set<int> distinct(slots.begin(), slots.end());
        if (distinct.size() != slots.size())
            duplicate_slots = true;
        for (int s : slots)
            if (s < 0 || s >= max_slots_)
                bad_slot = true;
        decode_sizes.push_back(static_cast<int>(slots.size()));
        if (fail_next_decode) {
            fail_next_decode = false;
            return false;
        }
        out.resize(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) out[i].codes.assign(1, rows[i]->frame_index);
        return true;
    }
    void release(int slot) override { released.push_back(slot); }

    int max_slots_;
    int prefills = 0, decodes = 0, max_rows = 0;
    bool duplicate_slots = false, bad_slot = false, fail_next_decode = false;
    int fail_prefill_slot = -2;
    std::vector<int> prefill_slots, released, decode_sizes;
};

// A request with scripted text chunks of `frames[i]` frames each (the last frame of a chunk ends
// it, like an EOS).
class FakeRequest final : public tts::IMagpieBatchRequest {
   public:
    FakeRequest(std::string name, std::vector<int> frames, Log& log)
        : name_(std::move(name)), frames_(std::move(frames)), log_(log) {}
    bool start() override {
        log_.add("start " + name_);
        started = true;
        return !fail_start;
    }
    bool beginTextChunk() override {
        step_ = 0;
        ++chunks_begun;
        return true;
    }
    Status prepareStep(tts::MagpieBatchRowInput& row) override {
        row.frame_index = frame_++;
        return Status::Continue;
    }
    Status applyStep(const tts::MagpieBatchRowOutput& out) override {
        if (out.codes.size() != 1)
            return Status::Failed;
        ++frames_done;
        ++step_;
        return step_ >= frames_[static_cast<size_t>(chunk_)] ? Status::EndChunk : Status::Continue;
    }
    bool endTextChunk(bool& more) override {
        ++chunk_;
        more = chunk_ < static_cast<int>(frames_.size());
        return true;
    }
    bool finish() override {
        log_.add("finish " + name_);
        finished = true;
        return true;
    }
    void abort() override {
        log_.add("abort " + name_);
        aborted = true;
    }

    std::string name_;
    std::vector<int> frames_;
    Log& log_;
    int chunk_ = 0, step_ = 0, frame_ = 0;
    std::atomic<int> frames_done{0};
    int chunks_begun = 0;
    bool started = false, fail_start = false;
    std::atomic<bool> finished{false}, aborted{false};
};

struct Result {
    std::atomic<int> calls{0};
    std::atomic<bool> ok{false};
};

tts::MagpieBatchScheduler::Done
done_into(Result& r) {
    return [&r](bool ok) {
        r.ok = ok;
        ++r.calls;
    };
}

}  // namespace

int
main() {
    // 1. Admission order, slot bound, eviction and reuse.
    {
        Log log;
        FakeBackend backend(2);
        tts::MagpieBatchConfig cfg;
        cfg.max_slots = 2;
        cfg.max_decode_steps_before_prefill = 0;
        tts::MagpieBatchScheduler sched(backend, cfg);
        std::vector<std::shared_ptr<FakeRequest>> reqs;
        std::vector<std::unique_ptr<Result>> results;
        const int lengths[5] = {3, 8, 2, 5, 4};
        for (int i = 0; i < 5; ++i) {
            reqs.push_back(std::make_shared<FakeRequest>(
                "r" + std::to_string(i), std::vector<int>{lengths[i]}, log));
            results.push_back(std::make_unique<Result>());
            expect(sched.submit(reqs.back(), done_into(*results.back())) > 0, "submit");
        }
        sched.runOnce();
        expect(sched.stats().active == 2, "two requests admitted into two slots");
        expect(!reqs[2]->started, "third request waits for a slot");
        sched.waitIdle();
        std::vector<std::string> starts;
        for (const auto& e : log.events)
            if (e.rfind("start ", 0) == 0)
                starts.push_back(e.substr(6));
        expect(
            starts == std::vector<std::string>({"r0", "r1", "r2", "r3", "r4"}),
            "FCFS admission order");
        for (int i = 0; i < 5; ++i) {
            expect(
                results[i]->calls == 1 && results[i]->ok,
                "request " + std::to_string(i) + " completed once");
            expect(
                reqs[i]->frames_done == lengths[i],
                "request " + std::to_string(i) + " frame count");
        }
        expect(
            backend.max_rows <= 2 && !backend.duplicate_slots && !backend.bad_slot, "slot bound");
        expect(backend.released.size() == 5, "every slot released once per request");
        const tts::MagpieBatchStats st = sched.stats();
        expect(
            st.completed == 5 && st.active == 0 && st.queued == 0 && st.max_active == 2,
            "stats after drain");
        // r0 (3 frames) ends first: r2 must take its slot while r1 is still decoding
        expect(
            backend.prefill_slots.size() == 5 &&
                backend.prefill_slots[2] == backend.prefill_slots[0],
            "a freed slot is reused by the next request");
    }

    // 2. Prefill budget: at most max_prefills_per_iter prefills per iteration, and decoding never
    //    runs more than max_decode_steps_before_prefill steps while a prefill waits.
    {
        Log log;
        FakeBackend backend(8);
        tts::MagpieBatchConfig cfg;
        cfg.max_slots = 8;
        cfg.max_prefills_per_iter = 1;
        cfg.max_decode_steps_before_prefill = 3;
        tts::MagpieBatchScheduler sched(backend, cfg);
        std::vector<std::shared_ptr<FakeRequest>> reqs;
        std::vector<std::unique_ptr<Result>> results;
        for (int i = 0; i < 4; ++i) {
            reqs.push_back(
                std::make_shared<FakeRequest>("p" + std::to_string(i), std::vector<int>{20}, log));
            results.push_back(std::make_unique<Result>());
            sched.submit(reqs.back(), done_into(*results.back()));
        }
        int decodes_before = 0;
        std::vector<int> decodes_between_prefills;
        int last_prefills = 0;
        for (int iter = 0; iter < 40 && backend.prefills < 4; ++iter) {
            const int p0 = backend.prefills;
            sched.runOnce();
            expect(backend.prefills - p0 <= 1, "one prefill per iteration");
            if (backend.prefills != last_prefills) {
                decodes_between_prefills.push_back(backend.decodes - decodes_before);
                decodes_before = backend.decodes;
                last_prefills = backend.prefills;
            }
        }
        expect(backend.prefills == 4, "all prefills ran");
        for (size_t i = 1; i < decodes_between_prefills.size(); ++i)
            expect(
                decodes_between_prefills[i] <= 3 + 1, "prefill waited at most the decode budget");
        sched.waitIdle();
        for (auto& r : results) expect(r->ok, "budgeted requests completed");
    }

    // 3. Longform: a request with three chunks re-prefills on its own slot.
    {
        Log log;
        FakeBackend backend(2);
        tts::MagpieBatchConfig cfg;
        cfg.max_slots = 2;
        tts::MagpieBatchScheduler sched(backend, cfg);
        auto longform = std::make_shared<FakeRequest>("long", std::vector<int>{4, 6, 3}, log);
        auto other = std::make_shared<FakeRequest>("other", std::vector<int>{10}, log);
        Result r1, r2;
        sched.submit(longform, done_into(r1));
        sched.submit(other, done_into(r2));
        sched.waitIdle();
        expect(r1.ok && r2.ok, "longform and its neighbour completed");
        expect(
            longform->chunks_begun == 3 && longform->frames_done == 13, "three chunks, 13 frames");
        std::vector<int> long_slots;
        for (size_t i = 0; i < backend.prefill_slots.size(); ++i)
            long_slots.push_back(backend.prefill_slots[i]);
        expect(backend.prefills == 4, "three chunk prefills plus the neighbour's");
        int first = backend.prefill_slots[0];
        int same = 0;
        for (int s : backend.prefill_slots) same += s == first ? 1 : 0;
        expect(same == 3, "every chunk of the longform request prefilled into the same slot");
        expect(backend.released.size() == 2, "slots released once per request, not per chunk");
    }

    // 4. Cancellation (queued and active), queue depth and failures.
    {
        Log log;
        FakeBackend backend(1);
        tts::MagpieBatchConfig cfg;
        cfg.max_slots = 1;
        cfg.max_queue_depth = 3;
        tts::MagpieBatchScheduler sched(backend, cfg);
        auto a = std::make_shared<FakeRequest>("a", std::vector<int>{50}, log);
        auto b = std::make_shared<FakeRequest>("b", std::vector<int>{5}, log);
        auto c = std::make_shared<FakeRequest>("c", std::vector<int>{5}, log);
        auto d = std::make_shared<FakeRequest>("d", std::vector<int>{5}, log);
        Result ra, rb, rc, rd;
        const int64_t ia = sched.submit(a, done_into(ra));
        const int64_t ib = sched.submit(b, done_into(rb));
        sched.submit(c, done_into(rc));
        expect(sched.submit(d, done_into(rd)) == -1, "queue depth rejects the fourth request");
        expect(sched.stats().rejected == 1, "rejection counted");
        sched.runOnce();
        sched.runOnce();
        expect(sched.slotOf(ia) == 0, "a is active");
        expect(sched.cancel(ib), "cancel queued b");
        expect(sched.cancel(ia), "cancel active a");
        sched.waitIdle();
        expect(ra.calls == 1 && !ra.ok && a->aborted && !a->finished, "active cancel aborts");
        expect(rb.calls == 1 && !rb.ok && !b->started, "queued cancel never starts");
        expect(rc.calls == 1 && rc.ok, "c ran after the cancellations");
        expect(sched.stats().cancelled == 2, "two cancellations counted");
        expect(!sched.cancel(999), "unknown id");
    }
    {
        Log log;
        FakeBackend backend(2);
        tts::MagpieBatchConfig cfg;
        cfg.max_slots = 2;
        tts::MagpieBatchScheduler sched(backend, cfg);
        auto bad_start = std::make_shared<FakeRequest>("bad", std::vector<int>{3}, log);
        bad_start->fail_start = true;
        auto x = std::make_shared<FakeRequest>("x", std::vector<int>{6}, log);
        auto y = std::make_shared<FakeRequest>("y", std::vector<int>{6}, log);
        Result r0, rx, ry;
        sched.submit(bad_start, done_into(r0));
        sched.runOnce();
        expect(r0.calls == 0 || !r0.ok, "failed start reports failure");
        sched.submit(x, done_into(rx));
        sched.submit(y, done_into(ry));
        sched.runOnce();  // prefills x, y
        backend.fail_next_decode = true;
        sched.runOnce();  // the batched step fails: both rows fail
        sched.waitIdle();
        expect(!r0.ok && !rx.ok && !ry.ok, "decode failure fails every row of the step");
        expect(x->aborted && y->aborted, "failed requests are aborted");
        expect(sched.stats().active == 0, "no request left active");
    }
    // 5. Threaded mode: many requests through the scheduler thread.
    {
        Log log;
        FakeBackend backend(4);
        tts::MagpieBatchConfig cfg;
        cfg.max_slots = 4;
        tts::MagpieBatchScheduler sched(backend, cfg);
        sched.start();
        std::vector<std::shared_ptr<FakeRequest>> reqs;
        std::vector<std::unique_ptr<Result>> results;
        for (int i = 0; i < 40; ++i) {
            reqs.push_back(std::make_shared<FakeRequest>(
                "t" + std::to_string(i), std::vector<int>{1 + (i * 7) % 13, 1 + i % 3}, log));
            results.push_back(std::make_unique<Result>());
            sched.submit(reqs.back(), done_into(*results.back()));
        }
        sched.waitIdle();
        int ok = 0;
        for (auto& r : results) ok += (r->calls == 1 && r->ok) ? 1 : 0;
        expect(ok == 40, "threaded: all 40 requests completed exactly once");
        expect(backend.max_rows <= 4 && !backend.duplicate_slots, "threaded: slot bound");
        sched.stop();
    }

    // 6. Static batching (benchmark reference): a batch runs to completion before the next one is
    //    admitted, even when a slot frees up early.
    {
        Log log;
        FakeBackend backend(2);
        tts::MagpieBatchConfig cfg;
        cfg.max_slots = 2;
        cfg.max_prefills_per_iter = 2;
        cfg.static_batching = true;
        cfg.static_batch_wait_ms = 0;  // the partial last batch starts at once (wait: test 7)
        tts::MagpieBatchScheduler sched(backend, cfg);
        std::vector<std::shared_ptr<FakeRequest>> reqs;
        std::vector<std::unique_ptr<Result>> results;
        const int lengths[5] = {3, 8, 2, 5, 4};
        for (int i = 0; i < 5; ++i) {
            reqs.push_back(std::make_shared<FakeRequest>(
                "s" + std::to_string(i), std::vector<int>{lengths[i]}, log));
            results.push_back(std::make_unique<Result>());
            sched.submit(reqs.back(), done_into(*results.back()));
        }
        // a full queue forms the first batch without waiting
        sched.runOnce();
        expect(sched.stats().active == 2, "static: the first batch fills both slots");
        bool s2_checked = false;
        int admissions = 1;
        for (int iter = 0; iter < 100 && backend.prefills < 5; ++iter) {
            const int active_before = sched.stats().active;
            const int p0 = backend.prefills;
            sched.runOnce();
            if (backend.prefills > p0) {
                expect(active_before == 0, "static: admission only after the previous batch ended");
                ++admissions;
            }
            if (!s2_checked && reqs[2]->started) {
                expect(
                    reqs[0]->frames_done == 3 && reqs[1]->frames_done == 8,
                    "static: s2 waits for the whole first batch, not just a free slot");
                s2_checked = true;
            }
        }
        expect(s2_checked, "static: the second batch started");
        sched.waitIdle();
        for (int i = 0; i < 5; ++i) {
            expect(
                results[i]->calls == 1 && results[i]->ok,
                "static: request " + std::to_string(i) + " completed");
            expect(
                reqs[i]->frames_done == lengths[i],
                "static: request " + std::to_string(i) + " frame count");
        }
        expect(admissions == 3, "static: batches {s0,s1}, {s2,s3}, {s4}");
        expect(sched.stats().max_active == 2 && !backend.duplicate_slots, "static: slot bound");
    }

    // 7. Static batching: a partial batch waits for the queue to fill, up to static_batch_wait_ms.
    {
        Log log;
        FakeBackend backend(4);
        tts::MagpieBatchConfig cfg;
        cfg.max_slots = 4;
        cfg.static_batching = true;
        cfg.static_batch_wait_ms = 40;
        tts::MagpieBatchScheduler sched(backend, cfg);
        auto a = std::make_shared<FakeRequest>("a", std::vector<int>{3}, log);
        Result ra;
        sched.submit(a, done_into(ra));
        sched.runOnce();
        expect(
            !a->started && sched.stats().active == 0,
            "static: a partial batch waits for more requests");
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        sched.runOnce();
        expect(
            a->started && sched.stats().active == 1,
            "static: a partial batch starts after the wait");
        sched.waitIdle();
        expect(ra.calls == 1 && ra.ok, "static: the partial batch completed");
    }

    // 8. Static batching through the scheduler thread.
    {
        Log log;
        FakeBackend backend(4);
        tts::MagpieBatchConfig cfg;
        cfg.max_slots = 4;
        cfg.static_batching = true;
        cfg.static_batch_wait_ms = 5;
        tts::MagpieBatchScheduler sched(backend, cfg);
        sched.start();
        std::vector<std::shared_ptr<FakeRequest>> reqs;
        std::vector<std::unique_ptr<Result>> results;
        for (int i = 0; i < 22; ++i) {
            reqs.push_back(std::make_shared<FakeRequest>(
                "st" + std::to_string(i), std::vector<int>{1 + (i * 5) % 11}, log));
            results.push_back(std::make_unique<Result>());
            sched.submit(reqs.back(), done_into(*results.back()));
        }
        sched.waitIdle();
        int ok = 0;
        for (auto& r : results) ok += (r->calls == 1 && r->ok) ? 1 : 0;
        expect(ok == 22, "static threaded: all 22 requests completed exactly once");
        expect(backend.max_rows <= 4 && !backend.duplicate_slots, "static threaded: slot bound");
        sched.stop();
    }

    if (failures) {
        std::fprintf(stderr, "%d scheduler check(s) failed\n", failures);
        return 1;
    }
    std::printf("batch scheduler OK\n");
    return 0;
}
