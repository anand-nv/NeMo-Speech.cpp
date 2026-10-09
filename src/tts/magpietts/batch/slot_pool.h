// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Slots of the in-flight batching K/V arenas and the lane layout of a batched decoder step.
#pragma once

#include <vector>

namespace nemo_speech::tts {

// Hands out the slots [0, max_slots) of the batching arenas. Slot max_slots is reserved for the
// rows that pad a step up to its bucket and is never handed out.
class MagpieSlotPool {
   public:
    explicit MagpieSlotPool(int max_slots);

    int capacity() const { return static_cast<int>(held_.size()); }
    int dummySlot() const { return capacity(); }
    // Arena rows including the dummy slot.
    int arenaSlots() const { return capacity() + 1; }
    int inUse() const { return in_use_; }
    int available() const { return capacity() - in_use_; }

    // The lowest free slot, or -1 when every slot is held.
    int acquire();
    // False (and no change) for a slot that is not held: out of range, the dummy slot or a
    // double release.
    bool release(int slot);
    bool held(int slot) const;

   private:
    std::vector<bool> held_;
    int in_use_ = 0;
};

// Lanes of a slot arena. Lanes are planar: every slot's conditional lane first, then (with CFG)
// every slot's unconditional lane, so a step's conditional rows are one contiguous half.
struct MagpieLaneLayout {
    int arena_slots = 0;
    bool use_cfg = true;

    int lanesPerSlot() const { return use_cfg ? 2 : 1; }
    int arenaLanes() const { return arena_slots * lanesPerSlot(); }
    int condLane(int slot) const { return slot; }
    int uncondLane(int slot) const { return arena_slots + slot; }
};

// The smallest bucket >= rows, or -1 when rows exceeds every bucket (or is < 1).
int magpie_batch_bucket(const std::vector<int>& buckets, int rows);

}  // namespace nemo_speech::tts
