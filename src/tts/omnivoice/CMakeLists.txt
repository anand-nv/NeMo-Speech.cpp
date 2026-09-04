# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

target_sources(nemo_speech_tts PRIVATE
    codec.cpp
    codec_encoder.cpp
    denoiser.cpp
    frontend.cpp
    generator.cpp
    model.cpp
    prompt.cpp
    runtime.cpp
    sampling.cpp
    tokenizer.cpp
)
target_include_directories(nemo_speech_tts PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_compile_definitions(nemo_speech_tts PRIVATE NEMO_SPEECH_TTS_WITH_OMNIVOICE=1)
find_package(ICU 70 REQUIRED COMPONENTS uc i18n)
target_link_libraries(nemo_speech_tts PRIVATE ICU::uc ICU::i18n)
