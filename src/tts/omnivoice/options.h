// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace nemo_speech::tts {

// Family-specific options shared by the native API, CLI, HTTP, and gRPC
// adapters. Keeping this POD independent of the runtime lets configurations
// parse when OmniVoice is compiled out and fail with one clear load error.
struct OmniVoiceOptions {
    int num_steps = 32;
    float guidance_scale = 2.0f;
    float time_shift = 0.1f;
    float layer_penalty = 5.0f;
    float position_temperature = 5.0f;
    float class_temperature = 0.0f;
    bool denoise = true;
    bool postprocess_output = true;
    double audio_chunk_duration_s = 15.0;
    double audio_chunk_threshold_s = 30.0;
    double pad_duration_s = 0.1;
    double fade_duration_s = 0.1;
    double speed = 1.0;
    double duration_s = 0.0;
};

}  // namespace nemo_speech::tts
