// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Deliberately compiled against the exact public declarations from before
// OmniVoice fields were appended. Do not include the current tts.h here.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum legacy_backend_preference {
    LEGACY_BACKEND_AUTO = 0,
    LEGACY_BACKEND_CPU = 1,
    LEGACY_BACKEND_CUDA = 2,
} legacy_backend_preference;

typedef enum legacy_uma_mode {
    LEGACY_UMA_AUTO = 0,
    LEGACY_UMA_OFF = 1,
    LEGACY_UMA_ON = 2,
} legacy_uma_mode;

typedef enum legacy_longform_mode {
    LEGACY_LONGFORM_AUTO = 0,
    LEGACY_LONGFORM_OFF = 1,
    LEGACY_LONGFORM_ON = 2,
} legacy_longform_mode;

typedef struct legacy_runtime_config {
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
    legacy_backend_preference lt_backend;
    legacy_backend_preference sampling_backend;
    legacy_uma_mode uma_mode;
    legacy_longform_mode longform_mode;
    bool lt_fp32;
} legacy_runtime_config;

typedef struct legacy_synthesis_options {
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
} legacy_synthesis_options;

extern legacy_runtime_config nemo_speech_tts_runtime_config_default(void);
extern legacy_synthesis_options nemo_speech_tts_synthesis_options_default(void);
extern const char* nemo_speech_tts_version(void);

int
main(void) {
    struct {
        uint64_t before;
        legacy_runtime_config value;
        uint64_t after;
    } runtime = {UINT64_C(0x123456789abcdef0), {0}, UINT64_C(0xfedcba9876543210)};
    struct {
        uint64_t before;
        legacy_synthesis_options value;
        uint64_t after;
    } options = {UINT64_C(0x0f1e2d3c4b5a6978), {0}, UINT64_C(0x8796a5b4c3d2e1f0)};

    runtime.value = nemo_speech_tts_runtime_config_default();
    options.value = nemo_speech_tts_synthesis_options_default();
    if (runtime.before != UINT64_C(0x123456789abcdef0) ||
        runtime.after != UINT64_C(0xfedcba9876543210) ||
        options.before != UINT64_C(0x0f1e2d3c4b5a6978) ||
        options.after != UINT64_C(0x8796a5b4c3d2e1f0)) {
        fputs("legacy by-value default overwrote its caller's stack\n", stderr);
        return 1;
    }
    if (runtime.value.size != sizeof(runtime.value) || runtime.value.threads != 4 ||
        runtime.value.seed != -1 || runtime.value.chunk_frames != 4 || !runtime.value.use_cfg ||
        runtime.value.lt_backend != LEGACY_BACKEND_AUTO) {
        fputs("legacy runtime defaults changed ABI or values\n", stderr);
        return 1;
    }
    if (options.value.size != sizeof(options.value) || options.value.speaker != -1 ||
        options.value.seed != -1 || options.value.steps != -1 || options.value.top_k != -1) {
        fputs("legacy synthesis defaults changed ABI or values\n", stderr);
        return 1;
    }
    if (!nemo_speech_tts_version() || !*nemo_speech_tts_version()) {
        fputs("legacy caller could not resolve the stable library\n", stderr);
        return 1;
    }
    return 0;
}
