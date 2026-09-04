# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(Threads REQUIRED)

if(TARGET nemo_speech_tts_kokoro)
    add_executable(test_kokoro_tokenizer test_kokoro_tokenizer.cpp)
    target_link_libraries(test_kokoro_tokenizer PRIVATE nemo_speech_tts_kokoro)
    target_compile_definitions(test_kokoro_tokenizer PRIVATE
        KOKORO_MANDARIN_DATA_DIR="${CMAKE_SOURCE_DIR}/src/tts/tokenizer/mandarin_data")
    add_test(NAME kokoro_tokenizer COMMAND test_kokoro_tokenizer)

    add_executable(test_kokoro_frontend_reference test_kokoro_frontend_reference.cpp)
    target_link_libraries(test_kokoro_frontend_reference PRIVATE nemo_speech_tts_kokoro)
    target_compile_definitions(test_kokoro_frontend_reference PRIVATE
        KOKORO_FRONTEND_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/kokoro_frontend_reference.json")
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_frontend_reference
            COMMAND test_kokoro_frontend_reference $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
    endif()

    add_executable(test_kokoro_model test_kokoro_model.cpp)
    target_link_libraries(test_kokoro_model PRIVATE
        nemo_speech_tts_kokoro nemo_speech_runtime_ggml)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_model
            COMMAND test_kokoro_model $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
    endif()

    add_executable(test_kokoro_plbert test_kokoro_plbert.cpp)
    target_link_libraries(test_kokoro_plbert PRIVATE nemo_speech_tts_kokoro)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_plbert
            COMMAND test_kokoro_plbert $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
    endif()

    add_executable(test_kokoro_duration test_kokoro_duration.cpp)
    target_link_libraries(test_kokoro_duration PRIVATE nemo_speech_tts_kokoro)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_duration
            COMMAND test_kokoro_duration $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
    endif()

    add_executable(test_kokoro_runtime test_kokoro_runtime.cpp)
    target_link_libraries(test_kokoro_runtime PRIVATE nemo_speech_tts_kokoro)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_runtime
            COMMAND test_kokoro_runtime $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
        if(GGML_CUDA)
            add_test(NAME kokoro_runtime_cuda
                COMMAND test_kokoro_runtime
                    $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} --gpu)
        endif()
    endif()

    add_executable(test_kokoro_voices test_kokoro_voices.cpp)
    target_link_libraries(test_kokoro_voices PRIVATE nemo_speech_tts_kokoro)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_voices
            COMMAND test_kokoro_voices $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
    endif()

    add_executable(test_kokoro_threading test_kokoro_threading.cpp)
    target_link_libraries(test_kokoro_threading PRIVATE nemo_speech_tts_kokoro Threads::Threads)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_threading
            COMMAND test_kokoro_threading $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
    endif()

    add_executable(test_kokoro_synthesizer test_kokoro_synthesizer.cpp)
    target_link_libraries(test_kokoro_synthesizer PRIVATE nemo_speech_tts)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_synthesizer
            COMMAND test_kokoro_synthesizer $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
    endif()

    add_executable(test_kokoro_acoustic test_kokoro_acoustic.cpp)
    target_link_libraries(test_kokoro_acoustic PRIVATE nemo_speech_tts_kokoro)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_acoustic
            COMMAND test_kokoro_acoustic $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
    endif()

    add_executable(test_kokoro_prosody test_kokoro_prosody.cpp)
    target_link_libraries(test_kokoro_prosody PRIVATE nemo_speech_tts_kokoro)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_prosody
            COMMAND test_kokoro_prosody $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
    endif()

    add_executable(test_kokoro_decoder test_kokoro_decoder.cpp)
    target_link_libraries(test_kokoro_decoder PRIVATE nemo_speech_tts_kokoro)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "")
        add_test(NAME kokoro_decoder
            COMMAND test_kokoro_decoder $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL})
    endif()

    add_executable(test_kokoro_vocoder test_kokoro_vocoder.cpp)
    target_link_libraries(test_kokoro_vocoder PRIVATE nemo_speech_tts_kokoro)

    add_executable(test_kokoro_reference test_kokoro_reference.cpp)
    target_link_libraries(test_kokoro_reference PRIVATE
        nemo_speech_tts_kokoro nemo_speech_runtime_ggml)
    if(DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_MODEL} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}" STREQUAL "" AND
       DEFINED ENV{NEMO_SPEECH_TEST_KOKORO_REFERENCE} AND
       NOT "$ENV{NEMO_SPEECH_TEST_KOKORO_REFERENCE}" STREQUAL "")
        add_test(NAME kokoro_reference
            COMMAND test_kokoro_reference
                $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}
                $ENV{NEMO_SPEECH_TEST_KOKORO_REFERENCE})
        if(GGML_CUDA)
            add_test(NAME kokoro_reference_cuda
                COMMAND test_kokoro_reference
                    $ENV{NEMO_SPEECH_TEST_KOKORO_MODEL}
                    $ENV{NEMO_SPEECH_TEST_KOKORO_REFERENCE} --gpu)
            set_tests_properties(kokoro_runtime_cuda kokoro_reference_cuda PROPERTIES
                ENVIRONMENT
                    "NVIDIA_TF32_OVERRIDE=0;GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1")
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
# The C ABI .so links the TTS C++ libraries privately, so the CUDA::cuda_driver
# stub that ggml-cuda needs in driver-less build containers may not reach this
# executable. Add the stub directly, matching the ASR C ABI smoke target.
if(GGML_CUDA)
    target_link_libraries(tts_c_smoke PRIVATE CUDA::cuda_driver)
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
endif()
