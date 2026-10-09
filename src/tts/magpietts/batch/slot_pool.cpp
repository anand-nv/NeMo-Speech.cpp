// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "slot_pool.h"

#include <algorithm>

namespace nemo_speech::tts {

MagpieSlotPool::MagpieSlotPool(int max_slots) : held_(static_cast<size_t>(std::max(0, max_slots))) {}

int
MagpieSlotPool::acquire() {
    for (size_t i = 0; i < held_.size(); ++i) {
        if (!held_[i]) {
            held_[i] = true;
            ++in_use_;
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool
MagpieSlotPool::release(int slot) {
    if (!held(slot)) {
        return false;
    }
    held_[static_cast<size_t>(slot)] = false;
    --in_use_;
    return true;
}

bool
MagpieSlotPool::held(int slot) const {
    return slot >= 0 && slot < capacity() && held_[static_cast<size_t>(slot)];
}

int
magpie_batch_bucket(const std::vector<int>& buckets, int rows) {
    if (rows < 1) {
        return -1;
    }
    int best = -1;
    for (int bucket : buckets) {
        if (bucket >= rows && (best < 0 || bucket < best)) {
            best = bucket;
        }
    }
    return best;
}

}  // namespace nemo_speech::tts
