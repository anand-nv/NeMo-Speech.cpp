// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Slot pool, planar lane layout and bucket selection of the MagpieTTS in-flight batching engine.
#include <cstdio>
#include <vector>

#include "tts/magpietts/batch/slot_pool.h"

namespace tts = nemo_speech::tts;

namespace {

int failures = 0;

void
expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int
main() {
    tts::MagpieSlotPool pool(4);
    expect(pool.capacity() == 4 && pool.dummySlot() == 4 && pool.arenaSlots() == 5, "geometry");
    expect(pool.acquire() == 0 && pool.acquire() == 1 && pool.acquire() == 2, "lowest-free order");
    expect(pool.inUse() == 3 && pool.available() == 1, "counts after three acquires");
    expect(pool.release(1), "release held slot");
    expect(!pool.release(1), "double release is rejected");
    expect(!pool.release(4), "the dummy slot is never held");
    expect(!pool.release(-1) && !pool.release(7), "out-of-range release is rejected");
    expect(pool.acquire() == 1, "a released slot is reused first");
    expect(pool.acquire() == 3, "last free slot");
    expect(pool.acquire() == -1, "exhaustion returns -1");
    for (int s = 0; s < 4; ++s) expect(pool.held(s), "all slots held");
    expect(!pool.held(4), "dummy slot not held");
    for (int s = 0; s < 4; ++s) expect(pool.release(s), "release all");
    expect(pool.inUse() == 0 && pool.available() == 4, "empty after releasing all");

    tts::MagpieLaneLayout cfg{5, true};
    expect(cfg.lanesPerSlot() == 2 && cfg.arenaLanes() == 10, "CFG lane count");
    expect(cfg.condLane(3) == 3 && cfg.uncondLane(3) == 8, "CFG planar lanes");
    tts::MagpieLaneLayout nocfg{5, false};
    expect(nocfg.lanesPerSlot() == 1 && nocfg.arenaLanes() == 5 && nocfg.condLane(4) == 4,
           "no-CFG lanes");

    const std::vector<int> buckets = {1, 2, 4, 8, 16};
    expect(tts::magpie_batch_bucket(buckets, 1) == 1, "bucket 1");
    expect(tts::magpie_batch_bucket(buckets, 3) == 4, "bucket 3 -> 4");
    expect(tts::magpie_batch_bucket(buckets, 8) == 8, "bucket 8");
    expect(tts::magpie_batch_bucket(buckets, 9) == 16, "bucket 9 -> 16");
    expect(tts::magpie_batch_bucket(buckets, 17) == -1, "too many rows");
    expect(tts::magpie_batch_bucket(buckets, 0) == -1, "no rows");
    expect(tts::magpie_batch_bucket({8, 2, 4}, 3) == 4, "unsorted buckets");

    if (failures) {
        std::fprintf(stderr, "%d slot-pool check(s) failed\n", failures);
        return 1;
    }
    std::printf("slot pool OK\n");
    return 0;
}
