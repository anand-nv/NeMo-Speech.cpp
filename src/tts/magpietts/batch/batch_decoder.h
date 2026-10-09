// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Batched MagpieTTS decoder step over per-slot K/V arenas (in-flight batching, see GOAL.md).
//
// Every request owns a slot: one circular self-attention cache row per CFG lane (seeded from its
// baked-context prefill) and its text's cross-attention K/V rows. A step decodes one stacked
// frame for any set of seeded slots in one fixed-shape graph per bucket; rows reach their cache
// through slot indirection, so requests join and leave without moving any K/V.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "batch_config.h"
#include "decoder.h"
#include "model.h"
#include "runtime/ggml/runtime.h"
#include "slot_pool.h"

namespace nemo_speech::tts {

// One request's row of a decoder step.
struct MagpieBatchDecoderRow {
    int slot = -1;
    // Stacked codes of the request's previous frame ([stacked_audio_codebooks]).
    const int32_t* tokens = nullptr;
    // Real text rows in the slot's cross cache (<= text capacity).
    int text_len = 0;
    // Attention prior (probabilities over the text rows), or nullptr for none.
    const std::vector<float>* prior = nullptr;
    // Return the cross-attention alignment of this row.
    bool collect_alignment = false;
};

class MagpieBatchDecoder {
   public:
    MagpieBatchDecoder(const magpietts_model& model, const MagpieBatchConfig& config);
    ~MagpieBatchDecoder();

    MagpieBatchDecoder(const MagpieBatchDecoder&) = delete;
    MagpieBatchDecoder& operator=(const MagpieBatchDecoder&) = delete;

    // Builds the session and allocates the arenas; false (with a message) when the model or the
    // configuration is unsupported.
    bool init(std::string* error);

    const MagpieLaneLayout& layout() const { return layout_; }
    const std::vector<int>& buckets() const { return buckets_; }
    int dummySlot() const { return layout_.arena_slots - 1; }
    int cacheLen() const { return cache_len_; }
    int textCapacity() const { return text_capacity_; }
    // Decoder steps a request may take per text chunk.
    int maxDecoderPositions() const { return max_positions_; }
    // Lanes of a step graph for `bucket` requests.
    int lanes(int bucket) const { return bucket * layout_.lanesPerSlot(); }
    ggml_backend_t backend() const;
    ggml_runtime::BackendManager& backendManager() { return *backend_manager_; }

    // Prefill handoff: copy a baked-context prefill (rows [0, n_tokens) of the caches) into the
    // slot's lanes and make the slot's next step position n_tokens. `uncond` is required with
    // CFG and ignored without.
    bool seedSlot(int slot, const DecoderKvCache& cond, const DecoderKvCache* uncond);
    // Copy the text's cross-attention K/V rows [0, text_len) into the slot.
    bool setSlotCross(int slot, const DecoderCrossKvCache& cross, int text_len);
    void releaseSlot(int slot);
    // Decoder rows in the slot's cache: the position of its next step (0 when unseeded).
    int slotTokens(int slot) const;
    // The slot's cache state before its next step.
    int slotRingHead(int slot) const;
    int slotValid(int slot) const;
    // Device data of the arenas (fused kernels).
    float* selfArena(int layer) const;
    float* crossK() const;
    float* crossV() const;
    // Advance a slot's cache state by one step (the fused path appends the row itself).
    void advanceSlot(int slot);

    // Decode one frame for `rows` (distinct seeded slots) padded to `bucket` lanes. On success
    // `hidden` holds the final hidden states [n_embd, lanes(bucket)] (planar: cond row b in
    // column b, its uncond lane in column bucket + b), valid until the next step; and
    // alignment[b] holds row b's mean cross-attention over its text rows when it asked for it.
    bool step(
        const std::vector<MagpieBatchDecoderRow>& rows, int bucket,
        ggml_runtime::DeviceTensor& hidden, std::vector<std::vector<float>>& alignment);

   private:
    class Module;
    struct SlotState {
        int n_tokens = 0;
        int ring_head = 0;
        int valid = 0;
        int text_len = 0;
    };

    const magpietts_model& model_;
    MagpieBatchConfig config_;
    MagpieLaneLayout layout_;
    std::vector<int> buckets_;
    int cache_len_ = 0;
    int text_capacity_ = 0;
    int max_positions_ = 0;
    std::vector<SlotState> slots_;
    std::unique_ptr<ggml_runtime::BackendManager> backend_manager_;
    std::unique_ptr<Module> module_;
    std::unique_ptr<ggml_runtime::Session> session_;
    // host staging of the step inputs
    std::vector<int32_t> tokens_, positions_, lane_slots_, cache_meta_, cross_slots_;
    std::vector<float> prior_, mask_, alignment_;
};

}  // namespace nemo_speech::tts
