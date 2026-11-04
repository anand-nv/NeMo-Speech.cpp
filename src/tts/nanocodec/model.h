// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"

namespace nemo_speech::tts::nanocodec {

struct NanoCodecHParams {
    int32_t sample_rate = 22050;
    int32_t samples_per_frame = 1024;
    int32_t num_codebooks = 8;
    int32_t codebook_size = 2016;
    int32_t latent_dim = 32;
    int32_t group_dim = 4;

    std::vector<int32_t> levels = {8, 7, 6, 6};
    std::vector<int32_t> base = {1, 8, 56, 336};
    std::vector<int32_t> scale = {4, 3, 3, 3};
    std::vector<int32_t> offset = {4, 3, 3, 3};
    std::vector<int32_t> up_rates = {8, 8, 4, 2, 2};
    std::vector<int32_t> res_kernels = {3, 7, 11};
    std::vector<int32_t> res_dilations = {1, 3, 5};
};

using NanoCodecFrame = std::array<int32_t, 8>;
using NanoCodecFrames = std::vector<NanoCodecFrame>;

class NanoCodecDecoder;

class NanoCodecModel {
   public:
    NanoCodecModel();
    ~NanoCodecModel();

    NanoCodecModel(NanoCodecModel&& other) noexcept;
    NanoCodecModel& operator=(NanoCodecModel&& other) noexcept;

    NanoCodecModel(const NanoCodecModel&) = delete;
    NanoCodecModel& operator=(const NanoCodecModel&) = delete;

    bool load(const std::string& path, bool force_cpu = false, bool verbose = false);
    void reset();
    bool loaded() const;
    // Whether the decoder runs on an accelerator backend (not the CPU).
    bool onAccelerator() const;
    // A new compute backend on the model's accelerator (its own stream) that can run this model's
    // decode graphs alongside the default one; nullptr on the CPU. The caller frees it.
    ggml_backend_t createComputeBackend() const;

    const NanoCodecHParams& hparams() const;
    int sampleRate() const;
    int samplesPerFrame() const;
    int numCodebooks() const;
    int codebookSize() const;
    int decoderLeftContextFrames() const;
    int64_t decoderLeftContextSamples() const;

   private:
    friend class NanoCodecDecoder;
    friend class NanoCodecBatchDecoder;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NanoCodecStreamState {
   public:
    NanoCodecStreamState();
    ~NanoCodecStreamState();

    NanoCodecStreamState(NanoCodecStreamState&& other) noexcept;
    NanoCodecStreamState& operator=(NanoCodecStreamState&& other) noexcept;

    NanoCodecStreamState(const NanoCodecStreamState&) = delete;
    NanoCodecStreamState& operator=(const NanoCodecStreamState&) = delete;

    void clear();

   private:
    friend class NanoCodecDecoder;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NanoCodecStreamGraph {
   public:
    NanoCodecStreamGraph();
    ~NanoCodecStreamGraph();

    NanoCodecStreamGraph(NanoCodecStreamGraph&& other) noexcept;
    NanoCodecStreamGraph& operator=(NanoCodecStreamGraph&& other) noexcept;

    NanoCodecStreamGraph(const NanoCodecStreamGraph&) = delete;
    NanoCodecStreamGraph& operator=(const NanoCodecStreamGraph&) = delete;

    void reset();
    bool initialized() const;
    int chunkFrames() const;

   private:
    friend class NanoCodecDecoder;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NanoCodecDecoder {
   public:
    explicit NanoCodecDecoder(const NanoCodecModel& model);

    // `backend` (optional, from NanoCodecModel::createComputeBackend) computes the decode instead
    // of the model's default backend; callers sharing a backend must serialize.
    bool decode(
        const NanoCodecFrames& frames, int threads, std::vector<float>& audio,
        ggml_backend_t backend = nullptr) const;
    bool initStreamGraph(
        NanoCodecStreamState& state, int chunk_frames, NanoCodecStreamGraph& graph) const;
    // `backend` (optional, from NanoCodecModel::createComputeBackend) computes the chunk instead
    // of the model's default backend, so decodes on different backends can overlap.
    bool decodeStream(
        NanoCodecStreamState& state, NanoCodecStreamGraph& graph, const NanoCodecFrames& frames,
        int threads, std::vector<float>& audio, ggml_backend_t backend = nullptr) const;
    bool decodeStreamAll(
        const NanoCodecFrames& frames, int chunk_frames, int threads,
        std::vector<float>& audio) const;

   private:
    const NanoCodecModel* model_;
};

// Batched streaming decoder for many concurrent streams (a TTS server's requests). Every stream
// holds a slot of slot-indexed state arenas (the stateful codec's convolution histories), and the
// chunks that streams submit while a batch runs are decoded together in the next one: one graph
// with a batch dimension, launched on its own stream. A stream's audio is bit-identical to what
// decodeStream produces for it. CUDA with the patched ggml series only (init reports why
// otherwise).
class NanoCodecBatchDecoder {
   public:
    // max_streams: concurrent streams (slots); max_chunk_frames: largest chunk a stream submits.
    NanoCodecBatchDecoder(const NanoCodecModel& model, int max_streams, int max_chunk_frames);
    ~NanoCodecBatchDecoder();

    NanoCodecBatchDecoder(const NanoCodecBatchDecoder&) = delete;
    NanoCodecBatchDecoder& operator=(const NanoCodecBatchDecoder&) = delete;

    bool init(std::string* error);
    // A slot with empty history for a new stream, or -1 when all are taken.
    int acquire();
    void release(int slot);
    // Decode the next chunk (1 .. max_chunk_frames frames) of the slot's stream; blocks until the
    // batch holding it is done. Calls for one slot must not overlap.
    bool decode(int slot, const NanoCodecFrames& frames, std::vector<float>& audio);

    // Asynchronous submission (a server's batching scheduler): `done` receives the audio of the
    // first keep_frames frames (the rest is padding, e.g. silence that fills a final chunk) on
    // the decoder's thread. A stream's chunks decode in the order they are staged or submitted.
    // stage() holds chunks until commit(), so the chunks one decoder step completes go into the
    // same batch; submit() queues at once.
    using Done = std::function<void(bool ok, std::vector<float>&& audio)>;
    void stage(int slot, NanoCodecFrames frames, int keep_frames, Done done);
    void commit();
    void submit(int slot, NanoCodecFrames frames, int keep_frames, Done done);

   private:
    bool valid(int slot, size_t frames) const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// True when every sample is finite (no NaN or infinity).
bool is_finite_audio(const std::vector<float>& audio);

}  // namespace nemo_speech::tts::nanocodec
