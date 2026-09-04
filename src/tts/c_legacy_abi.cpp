// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Preserve the original v1 symbols for already-compiled callers. The current
// public header routes source builds to explicitly versioned by-value returns.
#include "c_legacy_abi.h"

#include <cstring>

#include "magpietts/runtime.h"

#if defined(_WIN32)
#if defined(NEMO_SPEECH_TTS_BUILD)
#define NEMO_SPEECH_TTS_LEGACY_API __declspec(dllexport)
#else
#define NEMO_SPEECH_TTS_LEGACY_API __declspec(dllimport)
#endif
#else
#define NEMO_SPEECH_TTS_LEGACY_API __attribute__((visibility("default")))
#endif

extern "C" {

NEMO_SPEECH_TTS_LEGACY_API NemoSpeechTtsRuntimeConfigV1
nemo_speech_tts_runtime_config_default(void) {
    const nemo_speech::tts::MagpieRuntimeConfig defaults;
    NemoSpeechTtsRuntimeConfigV1 result;
    std::memset(&result, 0, sizeof(result));
    result.size = sizeof(result);
    result.speaker = defaults.speaker;
    result.threads = defaults.threads;
    result.codec_threads = defaults.codec_threads;
    result.seed = defaults.seed;
    result.steps = defaults.steps;
    result.top_k = defaults.top_k;
    result.chunk_frames = defaults.chunk_frames;
    result.codec_queue_depth = defaults.codec_queue_depth;
    result.codec_history_frames = defaults.codec_history_frames;
    result.codec_future_frames = defaults.codec_future_frames;
    result.window_ms = defaults.window_ms;
    result.temperature = defaults.temperature;
    result.override_temperature = defaults.override_temperature;
    result.cfg_scale = defaults.cfg_scale;
    result.override_cfg_scale = defaults.override_cfg_scale;
    result.use_cfg = defaults.use_cfg;
    result.use_local_transformer = defaults.use_local_transformer;
    result.use_kv_cache = defaults.use_kv_cache;
    result.use_stateful_codec = defaults.use_stateful_codec;
    result.codec_cpu = defaults.codec_cpu;
    result.flush_partial_chunk = defaults.flush_partial_chunk;
    result.verbose = defaults.verbose;
    result.lt_backend = static_cast<int32_t>(defaults.lt_backend);
    result.sampling_backend = static_cast<int32_t>(defaults.sampling_backend);
    result.uma_mode = static_cast<int32_t>(defaults.uma_mode);
    result.longform_mode = static_cast<int32_t>(defaults.longform_mode);
    result.lt_fp32 = defaults.lt_fp32;
    return result;
}

NEMO_SPEECH_TTS_LEGACY_API NemoSpeechTtsSynthesisOptionsV1
nemo_speech_tts_synthesis_options_default(void) {
    NemoSpeechTtsSynthesisOptionsV1 result;
    std::memset(&result, 0, sizeof(result));
    result.size = sizeof(result);
    result.speaker = -1;
    result.seed = -1;
    result.steps = -1;
    result.top_k = -1;
    return result;
}

}  // extern "C"
