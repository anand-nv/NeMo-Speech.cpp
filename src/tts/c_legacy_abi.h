// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Internal copies of the v1 by-value return types. Keep these byte-for-byte
// compatible with the public header that predates OmniVoice.
#pragma once

#include <cstddef>
#include <cstdint>

struct NemoSpeechTtsRuntimeConfigV1 {
    size_t size;
    int32_t speaker;
    int32_t threads;
    int32_t codec_threads;
    int32_t seed;
    int32_t steps;
    int32_t top_k;
    int32_t chunk_frames;
    int32_t codec_queue_depth;
    int32_t codec_history_frames;
    int32_t codec_future_frames;
    int32_t window_ms;
    float temperature;
    bool override_temperature;
    float cfg_scale;
    bool override_cfg_scale;
    bool use_cfg;
    bool use_local_transformer;
    bool use_kv_cache;
    bool use_stateful_codec;
    bool codec_cpu;
    bool flush_partial_chunk;
    bool verbose;
    int32_t lt_backend;
    int32_t sampling_backend;
    int32_t uma_mode;
    int32_t longform_mode;
    bool lt_fp32;
};

struct NemoSpeechTtsSynthesisOptionsV1 {
    size_t size;
    const char* request_id;
    const char* language_code;
    int32_t speaker;
    int32_t seed;
    int32_t steps;
    int32_t top_k;
    float temperature;
    bool override_temperature;
    float cfg_scale;
    bool override_cfg_scale;
    const char* voice_name;
    int32_t output_sample_rate;
};
