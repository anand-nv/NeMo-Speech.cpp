// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tts/kokoro/kokoro_runtime.h"

int
main(int argc, char** argv) {
    const bool prepare_only = argc == 6 && std::string(argv[2]) == "--prepare";
    const bool prepare_stdin = argc == 5 && std::string(argv[2]) == "--prepare-stdin";
    const bool use_gpu = argc == 3 && std::string(argv[2]) == "--gpu";
    if (argc != 2 && !use_gpu && !prepare_only && !prepare_stdin) {
        std::cerr << "usage: test_kokoro_runtime MODEL.gguf [--gpu]\n"
                     "       test_kokoro_runtime MODEL.gguf --prepare LANGUAGE VOICE TEXT\n"
                     "       test_kokoro_runtime MODEL.gguf --prepare-stdin LANGUAGE VOICE\n";
        return 2;
    }
    using namespace nemo_speech::tts::kokoro;
    KokoroRuntimeConfig config;
    config.model_path = argv[1];
    config.use_gpu = use_gpu;
    KokoroRuntime runtime(std::move(config));

    if (prepare_only) {
        const KokoroPreparedText prepared_text = runtime.prepare(argv[5], argv[3], argv[4]);
        for (const KokoroChunk& chunk : prepared_text.tokenization.chunks) {
            std::cout << chunk.phonemes << '\n';
        }
        return 0;
    }
    if (prepare_stdin) {
        std::string line;
        while (std::getline(std::cin, line)) {
            const KokoroPreparedText prepared_text = runtime.prepare(line, argv[3], argv[4]);
            for (const KokoroChunk& chunk : prepared_text.tokenization.chunks) {
                std::cout << chunk.phonemes;
            }
            std::cout << '\n';
        }
        if (!std::cin.eof())
            throw std::runtime_error("failed to read differential input");
        return 0;
    }

    struct FrontendCase {
        const char* text;
        const char* language;
        const char* voice;
    };
    const FrontendCase frontend_cases[] = {
        {"Hello!", "en-US", "af_heart"}, {"Hello!", "en-GB", "bf_emma"},
        {"Hola!", "es-ES", "ef_dora"},   {"Bonjour!", "fr-FR", "ff_siwis"},
        {"नमस्ते!", "hi-IN", "hf_alpha"},  {"Ciao!", "it-IT", "if_sara"},
        {"Olá!", "pt-BR", "pf_dora"},    {"你好！", "zh-CN", "zf_xiaobei"},
    };
    for (const auto& frontend_case : frontend_cases) {
        const KokoroPreparedText value =
            runtime.prepare(frontend_case.text, frontend_case.language, frontend_case.voice);
        if (value.tokenization.language != frontend_case.language ||
            value.tokenization.chunks.empty() ||
            value.tokenization.chunks.front().phonemes.empty() ||
            value.tokenization.chunks.front().ids.empty()) {
            throw std::runtime_error(
                std::string("Kokoro frontend integration failed for ") + frontend_case.language);
        }
    }

    const KokoroPreparedText mandarin = runtime.prepare("你好！", "zh-CN", "zf_xiaobei");
    if (mandarin.tokenization.chunks.size() != 1 ||
        mandarin.tokenization.chunks[0].phonemes != "ni↓xau↓!") {
        throw std::runtime_error("Kokoro Mandarin frontend integration failed");
    }
    if (std::getenv("NEMO_SPEECH_KOKORO_UNIDIC_DIR")) {
        const KokoroPreparedText japanese = runtime.prepare("日本語です。", "ja-JP", "jf_alpha");
        if (japanese.tokenization.chunks.size() != 1 ||
            japanese.tokenization.chunks[0].phonemes != "ɲiʔpoŋɡo desɨ.") {
            throw std::runtime_error("Kokoro Japanese Cutlet integration failed");
        }
    }

    const KokoroChunk prepared = runtime.prepare_tokens({50, 83, 156, 54, 57}, "en-US", "af_heart");
    std::vector<std::string> first;
    const KokoroRuntimeStats stats =
        runtime.synthesize({prepared}, "af_heart", 1.0f, 1234, [&](const std::string& pcm) {
            first.push_back(pcm);
            return true;
        });
    std::vector<std::string> second;
    const KokoroRuntimeStats repeated =
        runtime.synthesize({prepared}, "af_heart", 1.0f, 1234, [&](const std::string& pcm) {
            second.push_back(pcm);
            return true;
        });
    size_t first_bytes = 0;
    for (const std::string& pcm : first) first_bytes += pcm.size();
    if (runtime.sample_rate() != 24000 || first.size() < 2 || first != second ||
        first.front().empty() || first.front().size() % sizeof(int16_t) != 0 ||
        stats.samples_written * sizeof(int16_t) != first_bytes ||
        stats.samples_written != repeated.samples_written || stats.chunks != 1 ||
        stats.generated_frames <= 0 || stats.audio_s <= 0.0 || stats.elapsed_s <= 0.0) {
        throw std::runtime_error("Kokoro end-to-end runtime contract failed");
    }

    if (use_gpu) {
        for (const size_t token_count : {1U, 31U, 32U, 33U}) {
            const KokoroChunk boundary =
                runtime.prepare_tokens(std::vector<int32_t>(token_count, 50), "en-US", "af_heart");
            size_t callbacks = 0;
            const KokoroRuntimeStats boundary_stats =
                runtime.synthesize({boundary}, "af_heart", 2.0f, 1234, [&](const std::string& pcm) {
                    ++callbacks;
                    return !pcm.empty();
                });
            if (boundary_stats.cancelled || boundary_stats.chunks != 1 ||
                boundary_stats.samples_written == 0 || callbacks == 0) {
                throw std::runtime_error("Kokoro CUDA boundary-length synthesis failed");
            }
        }
    }

    std::vector<std::string> changed_seed;
    runtime.synthesize({prepared}, "af_heart", 1.0f, 4321, [&](const std::string& pcm) {
        changed_seed.push_back(pcm);
        return true;
    });
    if (changed_seed == first)
        throw std::runtime_error("Kokoro different seeds returned identical PCM");

    const KokoroRuntimeStats faster = runtime.synthesize({prepared}, "af_heart", 2.0f, 1234);
    if (faster.samples_written >= stats.samples_written)
        throw std::runtime_error("Kokoro speed did not shorten the output");

    size_t multi_chunk_callbacks = 0;
    const KokoroRuntimeStats multi_chunk =
        runtime.synthesize({prepared, prepared}, "af_heart", 2.0f, 1234, [&](const std::string&) {
            ++multi_chunk_callbacks;
            return true;
        });
    if (multi_chunk.chunks != 2 || multi_chunk_callbacks < 4 || multi_chunk.ttfa_ms <= 0.0 ||
        multi_chunk.icl_avg_ms <= 0.0 || multi_chunk.icl_min_ms <= 0.0 ||
        multi_chunk.icl_max_ms <= 0.0 || multi_chunk.icl_min_ms > multi_chunk.icl_avg_ms ||
        multi_chunk.icl_avg_ms > multi_chunk.icl_max_ms) {
        throw std::runtime_error("Kokoro callback timing statistics are invalid");
    }

    size_t cancellation_callbacks = 0;
    const KokoroRuntimeStats cancelled =
        runtime.synthesize({prepared, prepared}, "af_heart", 1.0f, 1234, [&](const std::string&) {
            ++cancellation_callbacks;
            return false;
        });
    if (!cancelled.cancelled || cancelled.chunks != 1 || cancellation_callbacks != 1)
        throw std::runtime_error("Kokoro callback cancellation did not stop later chunks");

    try {
        runtime.synthesize({prepared}, "af_heart", 2.01f, 1234);
        throw std::runtime_error("Kokoro accepted an out-of-range speed");
    }
    catch (const std::invalid_argument&) {
    }
    std::cout << "Kokoro native end-to-end runtime test passed\n";
    return 0;
}
