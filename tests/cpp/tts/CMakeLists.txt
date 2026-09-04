# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(Threads REQUIRED)

if(NEMO_SPEECH_TTS_WITH_OMNIVOICE)
    add_executable(test_omnivoice_loader test_omnivoice_loader.cpp)
    target_link_libraries(test_omnivoice_loader PRIVATE nemo_speech_tts)
    add_executable(test_omnivoice_tokenizer test_omnivoice_tokenizer.cpp)
    target_link_libraries(test_omnivoice_tokenizer PRIVATE nemo_speech_tts)
    add_executable(test_omnivoice_denoiser test_omnivoice_denoiser.cpp)
    target_link_libraries(test_omnivoice_denoiser PRIVATE nemo_speech_tts)
    add_executable(test_omnivoice_sampling test_omnivoice_sampling.cpp)
    target_link_libraries(test_omnivoice_sampling PRIVATE nemo_speech_tts)
    add_test(NAME omnivoice_sampling COMMAND test_omnivoice_sampling)
    add_executable(test_omnivoice_frontend test_omnivoice_frontend.cpp)
    target_link_libraries(test_omnivoice_frontend PRIVATE nemo_speech_tts)
    add_test(NAME omnivoice_frontend COMMAND test_omnivoice_frontend)
    add_executable(test_omnivoice_generator test_omnivoice_generator.cpp)
    target_link_libraries(test_omnivoice_generator PRIVATE nemo_speech_tts)
    add_executable(test_omnivoice_codec test_omnivoice_codec.cpp)
    target_link_libraries(test_omnivoice_codec PRIVATE nemo_speech_tts)
    add_executable(test_omnivoice_codec_encoder test_omnivoice_codec_encoder.cpp)
    target_link_libraries(test_omnivoice_codec_encoder PRIVATE nemo_speech_tts)
    add_executable(test_omnivoice_prompt test_omnivoice_prompt.cpp)
    target_link_libraries(test_omnivoice_prompt PRIVATE nemo_speech_tts)
    add_test(NAME omnivoice_prompt COMMAND test_omnivoice_prompt)
    add_executable(test_omnivoice_runtime test_omnivoice_runtime.cpp)
    target_link_libraries(test_omnivoice_runtime PRIVATE nemo_speech_tts)
    add_executable(test_omnivoice_stream test_omnivoice_stream.cpp)
    target_link_libraries(test_omnivoice_stream PRIVATE nemo_speech_tts)
    add_test(NAME omnivoice_stream_frontend COMMAND test_omnivoice_stream)
    add_executable(test_omnivoice_c_api test_omnivoice_c_api.cpp)
    target_link_libraries(test_omnivoice_c_api PRIVATE nemo_speech_tts_c)
    if(DEFINED ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL})
        add_test(
            NAME omnivoice_loader
            COMMAND test_omnivoice_loader $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL})
        add_test(
            NAME omnivoice_tokenizer
            COMMAND test_omnivoice_tokenizer $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL})
        add_test(
            NAME omnivoice_denoiser_cpu
            COMMAND test_omnivoice_denoiser $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL})
        add_test(
            NAME omnivoice_frontend_tables
            COMMAND test_omnivoice_frontend $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL})
        add_test(
            NAME omnivoice_generator_cpu
            COMMAND test_omnivoice_generator $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL})
    endif()
    if(DEFINED ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC})
        add_test(
            NAME omnivoice_codec_cpu
            COMMAND test_omnivoice_codec $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC})
        add_test(
            NAME omnivoice_codec_encoder_cpu
            COMMAND test_omnivoice_codec_encoder $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC})
        add_test(
            NAME omnivoice_prompt_codec_cpu
            COMMAND test_omnivoice_prompt $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC})
    endif()
    if(DEFINED ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL} AND
       DEFINED ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC})
        add_test(
            NAME omnivoice_runtime_cpu
            COMMAND test_omnivoice_runtime
                $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL}
                $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC})
        add_test(
            NAME omnivoice_stream_cpu
            COMMAND test_omnivoice_stream
                $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL}
                $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC})
        add_test(
            NAME omnivoice_c_api_cpu
            COMMAND test_omnivoice_c_api
                $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL}
                $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC})
        if(GGML_CUDA)
            add_test(
                NAME omnivoice_denoiser_cuda
                COMMAND test_omnivoice_denoiser
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL} --gpu)
            add_test(
                NAME omnivoice_generator_cuda
                COMMAND test_omnivoice_generator
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL} --gpu)
            add_test(
                NAME omnivoice_codec_cuda
                COMMAND test_omnivoice_codec
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC} --gpu)
            add_test(
                NAME omnivoice_codec_encoder_cuda
                COMMAND test_omnivoice_codec_encoder
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC} --gpu)
            add_test(
                NAME omnivoice_prompt_codec_cuda
                COMMAND test_omnivoice_prompt
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC} --gpu)
            add_test(
                NAME omnivoice_runtime_cuda
                COMMAND test_omnivoice_runtime
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL}
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC} --gpu)
            add_test(
                NAME omnivoice_stream_cuda
                COMMAND test_omnivoice_stream
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL}
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC} --gpu)
            add_test(
                NAME omnivoice_c_api_cuda
                COMMAND test_omnivoice_c_api
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_MODEL}
                    $ENV{NEMO_SPEECH_TEST_OMNIVOICE_CODEC} --gpu)
        endif()
    endif()
endif()

add_executable(test_magpietts_file test_magpietts_file.cpp)
target_link_libraries(test_magpietts_file PRIVATE nemo_speech_tts_magpietts)

add_executable(test_magpietts_attention_prior test_magpietts_attention_prior.cpp)
target_link_libraries(test_magpietts_attention_prior PRIVATE nemo_speech_tts_magpietts)

add_executable(test_magpietts_frame_stacking test_magpietts_frame_stacking.cpp)
target_link_libraries(test_magpietts_frame_stacking PRIVATE nemo_speech_tts_magpietts)
add_test(NAME magpietts_frame_stacking COMMAND test_magpietts_frame_stacking)

if(GGML_CUDA AND NEMO_SPEECH_GGML_PATCHED)
    add_executable(test_magpietts_cached_attention test_magpietts_cached_attention.cpp)
    target_link_libraries(test_magpietts_cached_attention PRIVATE nemo_speech_tts_magpietts)
    add_test(NAME magpietts_cached_attention COMMAND test_magpietts_cached_attention)
endif()

if(NEMO_SPEECH_BUILD_ASR)
    add_executable(test_magpietts_asr test_magpietts_asr.cpp)
    target_link_libraries(test_magpietts_asr PRIVATE
        tests_cpp_wer
        nemo_speech_tts_magpietts
        nemo_speech_tts_tokenizer
        nemo_speech_asr
    )
endif()

# C ABI tests. tts_c_smoke is compiled as C (proves include/nemo_speech/tts.h
# is valid C) and links the unified TTS library; tts_c_dlopen loads it at runtime.
add_executable(tts_c_smoke ${CMAKE_SOURCE_DIR}/tests/c/tts_c_smoke.c)
target_link_libraries(tts_c_smoke PRIVATE nemo_speech_tts_c)
add_executable(tts_c_legacy_abi ${CMAKE_SOURCE_DIR}/tests/c/tts_c_legacy_abi.c)
target_link_libraries(tts_c_legacy_abi PRIVATE nemo_speech_tts_c)
add_test(NAME tts_c_legacy_abi COMMAND tts_c_legacy_abi)
# The C ABI .so links the TTS C++ libraries privately, so the CUDA::cuda_driver
# stub that ggml-cuda needs in driver-less build containers may not reach this
# executable. Add the stub directly, matching the ASR C ABI smoke target.
if(GGML_CUDA)
    target_link_libraries(tts_c_smoke PRIVATE CUDA::cuda_driver)
    target_link_libraries(tts_c_legacy_abi PRIVATE CUDA::cuda_driver)
endif()

add_executable(tts_c_dlopen ${CMAKE_SOURCE_DIR}/tests/c/tts_c_dlopen.c)
target_link_libraries(tts_c_dlopen PRIVATE ${CMAKE_DL_LIBS})

add_executable(test_tokenizer_single_chars test_tokenizer_single_chars.cpp)
target_link_libraries(test_tokenizer_single_chars PRIVATE nemo_speech_tts_tokenizer)
if(NEMO_SPEECH_TTS_WITH_JA)
    target_compile_definitions(test_tokenizer_single_chars PRIVATE NEMO_SPEECH_TTS_WITH_JA=1)
endif()
if(NEMO_SPEECH_TTS_WITH_ZH)
    target_compile_definitions(test_tokenizer_single_chars PRIVATE NEMO_SPEECH_TTS_WITH_ZH=1)
endif()
if(DEFINED ENV{NEMO_SPEECH_TEST_TOKENIZER_DIR_V2602})
    add_test(
        NAME tokenizer_v2602
        COMMAND test_tokenizer_single_chars $ENV{NEMO_SPEECH_TEST_TOKENIZER_DIR_V2602})
endif()
if(DEFINED ENV{NEMO_SPEECH_TEST_TOKENIZER_DIR_V2607})
    add_test(
        NAME tokenizer_v2607
        COMMAND test_tokenizer_single_chars $ENV{NEMO_SPEECH_TEST_TOKENIZER_DIR_V2607})
endif()

if(NEMO_SPEECH_TTS_WITH_ZH)
    add_executable(test_tokenizer_mandarin test_tokenizer_mandarin.cpp)
    target_link_libraries(
        test_tokenizer_mandarin PRIVATE nemo_speech_tts_tokenizer Threads::Threads)
    target_compile_definitions(test_tokenizer_mandarin PRIVATE
        MANDARIN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    if(DEFINED ENV{NEMO_SPEECH_TEST_TOKENIZER_DIR_V2602})
        add_test(
            NAME tokenizer_mandarin_v2602
            COMMAND test_tokenizer_mandarin $ENV{NEMO_SPEECH_TEST_TOKENIZER_DIR_V2602})
    endif()
    if(DEFINED ENV{NEMO_SPEECH_TEST_TOKENIZER_DIR_V2607})
        add_test(
            NAME tokenizer_mandarin_v2607
            COMMAND test_tokenizer_mandarin $ENV{NEMO_SPEECH_TEST_TOKENIZER_DIR_V2607})
    endif()
endif()

add_executable(test_tts_terminal_punctuation test_tts_terminal_punctuation.cpp)
target_link_libraries(test_tts_terminal_punctuation PRIVATE nemo_speech_tts_tokenizer)

add_executable(test_text_normalizer test_text_normalizer.cpp)
target_link_libraries(
    test_text_normalizer PRIVATE nemo_speech_tts_preproc nemo_speech_tts_tokenizer)

if(TARGET nemo_speech_grpc_tts)
    add_executable(test_grpc_tts_config test_grpc_tts_config.cpp)
    target_link_libraries(test_grpc_tts_config PRIVATE nemo_speech_grpc_tts)
    target_compile_definitions(test_grpc_tts_config PRIVATE
        NEMO_SPEECH_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    if(NEMO_SPEECH_TTS_WITH_OMNIVOICE)
        add_executable(test_grpc_tts_omnivoice test_grpc_tts_omnivoice.cpp)
        target_link_libraries(test_grpc_tts_omnivoice PRIVATE nemo_speech_grpc_tts)
        add_test(NAME grpc_tts_omnivoice COMMAND test_grpc_tts_omnivoice)
    endif()
endif()
