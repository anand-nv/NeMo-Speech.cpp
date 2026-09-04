# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Kokoro's frontend is deliberately an ordinary native library. Python and
# ONNX are not linked or invoked by the installed runtime.
find_package(ICU REQUIRED COMPONENTS uc)
find_library(NEMO_SPEECH_ESPEAK_NG_LIBRARY
    NAMES espeak-ng libespeak-ng.so.1 espeak-ng.so.1
    REQUIRED)

add_library(nemo_speech_tts_kokoro STATIC
    acoustic.cpp
    decoder.cpp
    duration.cpp
    frontend.cpp
    kokoro_runtime.cpp
    model.cpp
    plbert.cpp
    prosody.cpp
    vocoder.cpp
    sha256.cpp
    tokenizer/english_g2p.cpp
    tokenizer/espeak_g2p.cpp
    tokenizer/kokoro_tokenizer.cpp
    tokenizer/japanese_g2p.cpp
    tokenizer/mandarin_g2p.cpp
    tokenizer/resource_path.cpp
)
set_target_properties(nemo_speech_tts_kokoro PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nemo_speech_tts_kokoro PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(nemo_speech_tts_kokoro SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/cppjieba/include
    ${CMAKE_SOURCE_DIR}/third_party/cppjieba/deps/limonp/include
)
set(KOKORO_MANDARIN_DATA_SRC_DIR
    ${CMAKE_SOURCE_DIR}/src/tts/tokenizer/mandarin_data)
set(KOKORO_MANDARIN_DATA_INSTALL_DIR
    ${CMAKE_INSTALL_DATADIR}/nemo-speech/kokoro/misaki-0.9.4/mandarin)
set(KOKORO_MANDARIN_DATA_INSTALL_FULL_DIR
    ${CMAKE_INSTALL_FULL_DATADIR}/nemo-speech/kokoro/misaki-0.9.4/mandarin)
if(NEMO_SPEECH_KOKORO_UNIDIC_DIR)
    set(KOKORO_UNIDIC_SOURCE_DIR ${NEMO_SPEECH_KOKORO_UNIDIC_DIR})
else()
    set(KOKORO_UNIDIC_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/unidic-lite/dicdir)
endif()
set(KOKORO_UNIDIC_INSTALL_DIR
    ${CMAKE_INSTALL_DATADIR}/nemo-speech/kokoro/misaki-0.9.4/unidic-lite)
set(KOKORO_UNIDIC_INSTALL_FULL_DIR
    ${CMAKE_INSTALL_FULL_DATADIR}/nemo-speech/kokoro/misaki-0.9.4/unidic-lite)
if(WIN32)
    set(KOKORO_MODULE_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
else()
    set(KOKORO_MODULE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
endif()
file(RELATIVE_PATH KOKORO_DATA_FROM_MODULE_DIR
    "/${KOKORO_MODULE_INSTALL_DIR}"
    "/${CMAKE_INSTALL_DATADIR}/nemo-speech/kokoro/misaki-0.9.4")
target_compile_definitions(nemo_speech_tts_kokoro PRIVATE
    NEMO_SPEECH_KOKORO_MANDARIN_DATA_DIR="${KOKORO_MANDARIN_DATA_SRC_DIR}"
    NEMO_SPEECH_KOKORO_MANDARIN_INSTALLED_DATA_DIR="${KOKORO_MANDARIN_DATA_INSTALL_FULL_DIR}"
    NEMO_SPEECH_KOKORO_UNIDIC_SOURCE_DIR="${KOKORO_UNIDIC_SOURCE_DIR}"
    NEMO_SPEECH_KOKORO_UNIDIC_INSTALLED_DIR="${KOKORO_UNIDIC_INSTALL_FULL_DIR}"
    NEMO_SPEECH_KOKORO_JA_WORDS_SOURCE="${CMAKE_SOURCE_DIR}/src/tts/kokoro/tokenizer/data/ja_words.txt"
    NEMO_SPEECH_KOKORO_JA_WORDS_INSTALLED="${CMAKE_INSTALL_FULL_DATADIR}/nemo-speech/kokoro/misaki-0.9.4/ja_words.txt"
    NEMO_SPEECH_KOKORO_DATA_FROM_MODULE_DIR="${KOKORO_DATA_FROM_MODULE_DIR}"
)
target_link_libraries(nemo_speech_tts_kokoro PUBLIC
    ICU::uc
    ${NEMO_SPEECH_ESPEAK_NG_LIBRARY}
    nemo_speech_common
    nemo_speech_runtime_ggml
    nemo_speech_openjtalk_frontend
    ${CMAKE_DL_LIBS}
)

if(WIN32)
    target_compile_definitions(nemo_speech_tts_kokoro PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

install(DIRECTORY ${KOKORO_MANDARIN_DATA_SRC_DIR}/
    DESTINATION ${KOKORO_MANDARIN_DATA_INSTALL_DIR}
    COMPONENT Runtime)
if(EXISTS "${KOKORO_UNIDIC_SOURCE_DIR}/sys.dic")
    install(DIRECTORY ${KOKORO_UNIDIC_SOURCE_DIR}/
        DESTINATION ${KOKORO_UNIDIC_INSTALL_DIR}
        COMPONENT Runtime)
    get_filename_component(KOKORO_UNIDIC_PACKAGE_ROOT
        "${KOKORO_UNIDIC_SOURCE_DIR}/../.." ABSOLUTE)
    set(KOKORO_UNIDIC_LICENSE_ROOT
        "${KOKORO_UNIDIC_PACKAGE_ROOT}/unidic_lite-1.0.8.dist-info/licenses")
    if(EXISTS "${KOKORO_UNIDIC_LICENSE_ROOT}/LICENSE")
        install(FILES
            "${KOKORO_UNIDIC_LICENSE_ROOT}/LICENSE"
            "${KOKORO_UNIDIC_LICENSE_ROOT}/LICENSE.unidic"
            DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/nemo-speech/third_party/unidic-lite)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/unidic-lite/LICENSE")
        install(FILES
            "${CMAKE_SOURCE_DIR}/third_party/unidic-lite/LICENSE"
            "${CMAKE_SOURCE_DIR}/third_party/unidic-lite/LICENSE.unidic"
            DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/nemo-speech/third_party/unidic-lite)
    else()
        message(WARNING
            "UniDic-lite data was found but its LICENSE and LICENSE.unidic files were not; "
            "binary distributors must include both notices")
    endif()
else()
    message(STATUS
        "Kokoro Japanese G2P will be unavailable until the pinned unidic-lite 1.0.8 "
        "dicdir is supplied with NEMO_SPEECH_KOKORO_UNIDIC_DIR")
endif()
