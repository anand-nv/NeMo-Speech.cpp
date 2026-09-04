# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_subdirectory(nanocodec)
add_subdirectory(tokenizer)
add_subdirectory(preproc)
if(NEMO_SPEECH_TTS_WITH_KOKORO)
    add_subdirectory(kokoro)
endif()
add_subdirectory(magpietts)
