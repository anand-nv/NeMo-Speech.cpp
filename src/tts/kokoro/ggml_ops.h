// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml.h"

namespace nemo_speech::tts::kokoro {

inline ggml_tensor*
kokoro_mul_mat(ggml_context* context, ggml_tensor* left, ggml_tensor* right) {
    ggml_tensor* result = ggml_mul_mat(context, left, right);
    // Kokoro's recurrent and vocoder paths amplify half-accumulation error.
    // GGML_PREC_F32 also makes F16-weight CUDA matmuls accumulate into F32.
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

}  // namespace nemo_speech::tts::kokoro
