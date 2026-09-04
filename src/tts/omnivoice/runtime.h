// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "generator.h"
#include "prompt.h"

namespace ggml_runtime {
class BackendManager;
}

namespace nemo_speech::tts::omnivoice {

struct RuntimeConfig {
    GenerationConfig generation;
    bool postprocess_output = true;
    double audio_chunk_duration_s = 15.0;
    double audio_chunk_threshold_s = 30.0;
    double pad_duration_s = 0.1;
    double fade_duration_s = 0.1;
};

struct RuntimeSynthesisRequest {
    std::string text;
    std::optional<std::string> language;
    std::optional<std::string> instruction;
    const VoicePrompt* voice_prompt = nullptr;
    std::optional<double> speed;
    std::optional<double> fixed_duration_seconds;
    // Test/developer override. Normal callers leave this unset.
    std::optional<int32_t> target_frames;
};

struct RuntimeStats {
    uint64_t effective_seed = 0;
    uint64_t output_samples = 0;
    int32_t generated_frames = 0;
    int32_t chunks = 0;
    double elapsed_seconds = 0.0;
    bool cancelled = false;
};

struct RuntimeSynthesisResult {
    // Populated only when no callback is supplied.
    std::vector<float> pcm_24khz;
    std::array<std::vector<int32_t>, 8> first_chunk_codes;
    RuntimeStats stats;
};

class Runtime {
   public:
    using PcmCallback = std::function<bool(const float* samples, size_t count)>;

    Runtime(
        const std::string& model_path, const std::string& audio_tokenizer_path,
        bool use_gpu = false, int32_t gpu_device = 0);
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    RuntimeSynthesisResult synthesize(
        const RuntimeSynthesisRequest& request, const RuntimeConfig& config = {},
        const PcmCallback& callback = {});
    VoicePrompt create_prompt(
        const float* interleaved_pcm, size_t frames, int32_t channels, int32_t sample_rate,
        const std::string& transcript, bool preprocess = true);

    int32_t sample_rate() const { return 24000; }
    const PromptFingerprint& fingerprint() const { return fingerprint_; }
    const std::vector<std::string>& language_ids() const;
    const std::vector<std::string>& language_names() const;

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    PromptFingerprint fingerprint_;
};

struct StreamLimits {
    size_t maximum_pending_bytes = 64 * 1024;
    size_t maximum_queued_segments = 8;
};

// One session accepts arbitrary UTF-8 byte fragments and runs committed
// segments on a private worker. Runtime model execution remains serialized;
// input enqueueing and callback delivery do not require the caller to block on
// synthesis. finish() drains and joins the worker.
class BidirectionalStream {
   public:
    BidirectionalStream(
        Runtime& runtime, RuntimeSynthesisRequest immutable_request, RuntimeConfig config,
        Runtime::PcmCallback callback, StreamLimits limits = {});
    ~BidirectionalStream();

    BidirectionalStream(const BidirectionalStream&) = delete;
    BidirectionalStream& operator=(const BidirectionalStream&) = delete;

    void push_text(const char* bytes, size_t count, bool explicit_commit = false);
    RuntimeStats finish();
    void cancel();
    RuntimeStats stats() const;

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts::omnivoice
