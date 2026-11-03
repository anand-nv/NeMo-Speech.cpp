// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// NanoCodecBatchDecoder against the per-stream streaming decoder: N streams of different lengths,
// decoded chunk by chunk (the last chunk shorter) once with decodeStream each and once through the
// batched decoder from one thread per stream, so its batches mix streams at different positions.
// Every stream's audio must be bit-identical. Also reports the time of both runs.
//
// Usage: test_nanocodec_batch_decode <codec.gguf> [--streams N] [--chunk F]
// The model may also come from NEMO_SPEECH_TEST_CODEC_MODEL; skipped without it or without CUDA.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "tts/nanocodec/model.h"

namespace nc = nemo_speech::tts::nanocodec;

int
main(int argc, char** argv) {
    std::string path;
    int streams = 24, chunk = 4;
    bool cpu_ref = false;  // also measure both GPU paths against the CPU (F32) decoder
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--streams" && i + 1 < argc)
            streams = std::atoi(argv[++i]);
        else if (a == "--chunk" && i + 1 < argc)
            chunk = std::atoi(argv[++i]);
        else if (a == "--cpu-ref")
            cpu_ref = true;
        else
            path = a;
    }
    if (path.empty()) {
        const char* env = std::getenv("NEMO_SPEECH_TEST_CODEC_MODEL");
        if (!env || !env[0]) {
            std::printf("SKIP: no NanoCodec model\n");
            return 77;
        }
        path = env;
    }
    nc::NanoCodecModel model;
    if (!model.load(path, false, false)) {
        std::fprintf(stderr, "FAIL: cannot load %s\n", path.c_str());
        return 1;
    }
    nc::NanoCodecBatchDecoder batch(model, streams, chunk);
    std::string error;
    if (!batch.init(&error)) {
        std::printf("SKIP: batched decoder unavailable: %s\n", error.c_str());
        return 77;
    }

    // codes: smooth random walks over the codebooks (closer to speech than white noise)
    std::mt19937 rng(5);
    std::vector<nc::NanoCodecFrames> codes((size_t)streams);
    for (int s = 0; s < streams; ++s) {
        std::uniform_int_distribution<int> len(9, 60), step(-40, 40),
            tok(0, model.codebookSize() - 1);
        nc::NanoCodecFrame f{};
        for (int c = 0; c < model.numCodebooks(); ++c) f[(size_t)c] = tok(rng);
        const int n = len(rng);
        for (int t = 0; t < n; ++t) {
            for (int c = 0; c < model.numCodebooks(); ++c)
                f[(size_t)c] = std::clamp(f[(size_t)c] + step(rng), 0, model.codebookSize() - 1);
            codes[(size_t)s].push_back(f);
        }
    }
    auto chunks_of = [&](const nc::NanoCodecFrames& all) {
        std::vector<nc::NanoCodecFrames> out;
        for (size_t i = 0; i < all.size(); i += (size_t)chunk)
            out.emplace_back(
                all.begin() + (ptrdiff_t)i,
                all.begin() + (ptrdiff_t)std::min(all.size(), i + (size_t)chunk));
        return out;
    };

    // reference: one stream at a time
    nc::NanoCodecDecoder decoder(model);
    std::vector<std::vector<float>> ref((size_t)streams), got((size_t)streams);
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < streams; ++s) {
        nc::NanoCodecStreamState state;
        std::map<int, nc::NanoCodecStreamGraph> graphs;
        for (const auto& c : chunks_of(codes[(size_t)s])) {
            nc::NanoCodecStreamGraph& g = graphs[(int)c.size()];
            if (!g.initialized() && !decoder.initStreamGraph(state, (int)c.size(), g)) {
                std::fprintf(stderr, "FAIL: stream graph\n");
                return 1;
            }
            std::vector<float> a;
            if (!decoder.decodeStream(state, g, c, 1, a)) {
                std::fprintf(stderr, "FAIL: decodeStream\n");
                return 1;
            }
            ref[(size_t)s].insert(ref[(size_t)s].end(), a.begin(), a.end());
        }
    }
    const double serial_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // batched: every stream from its own thread
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    std::vector<int> failed((size_t)streams, 0);
    for (int s = 0; s < streams; ++s) {
        threads.emplace_back([&, s] {
            const int slot = batch.acquire();
            if (slot < 0) {
                failed[(size_t)s] = 1;
                return;
            }
            for (const auto& c : chunks_of(codes[(size_t)s])) {
                std::vector<float> a;
                if (!batch.decode(slot, c, a)) {
                    failed[(size_t)s] = 1;
                    break;
                }
                got[(size_t)s].insert(got[(size_t)s].end(), a.begin(), a.end());
            }
            batch.release(slot);
        });
    }
    for (auto& t : threads) t.join();
    const double batched_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();

    if (cpu_ref) {
        nc::NanoCodecModel cpu_model;
        if (!cpu_model.load(path, true, false)) {
            std::fprintf(stderr, "FAIL: CPU model\n");
            return 1;
        }
        nc::NanoCodecDecoder cpu_decoder(cpu_model);
        double e_alone = 0.0, e_batch = 0.0;
        for (int s = 0; s < streams; ++s) {
            nc::NanoCodecStreamState state;
            std::map<int, nc::NanoCodecStreamGraph> graphs;
            std::vector<float> cref;
            for (const auto& c : chunks_of(codes[(size_t)s])) {
                nc::NanoCodecStreamGraph& g = graphs[(int)c.size()];
                if (!g.initialized())
                    cpu_decoder.initStreamGraph(state, (int)c.size(), g);
                std::vector<float> a;
                cpu_decoder.decodeStream(state, g, c, 8, a);
                cref.insert(cref.end(), a.begin(), a.end());
            }
            auto rel = [&](const std::vector<float>& x) {
                double err = 0.0, pow = 1e-30;
                for (size_t i = 0; i < cref.size() && i < x.size(); ++i) {
                    err += ((double)x[i] - cref[i]) * ((double)x[i] - cref[i]);
                    pow += (double)cref[i] * cref[i];
                }
                return std::sqrt(err / pow);
            };
            e_alone = std::max(e_alone, rel(ref[(size_t)s]));
            e_batch = std::max(e_batch, rel(got[(size_t)s]));
        }
        std::printf(
            "against the CPU decoder: worst relative RMS error GPU alone %.2e, GPU batched %.2e\n",
            e_alone, e_batch);
    }

    double worst_rel = 0.0, worst_abs = 0.0;
    size_t samples = 0;
    bool ok = true;
    for (int s = 0; s < streams; ++s) {
        if (failed[(size_t)s] || got[(size_t)s].size() != ref[(size_t)s].size()) {
            std::fprintf(
                stderr, "FAIL: stream %d: %zu samples batched, %zu alone%s\n", s,
                got[(size_t)s].size(), ref[(size_t)s].size(), failed[(size_t)s] ? " (failed)" : "");
            ok = false;
            continue;
        }
        double err = 0.0, pow = 1e-30;
        for (size_t i = 0; i < ref[(size_t)s].size(); ++i) {
            const double d = (double)got[(size_t)s][i] - ref[(size_t)s][i];
            err += d * d;
            pow += (double)ref[(size_t)s][i] * ref[(size_t)s][i];
            worst_abs = std::max(worst_abs, std::fabs(d));
        }
        worst_rel = std::max(worst_rel, std::sqrt(err / pow));
        samples += ref[(size_t)s].size();
    }
    std::printf(
        "%d streams, %zu samples, chunks of %d frames: batched vs alone: worst relative RMS error "
        "%.2e (SNR %.1f dB), worst absolute difference %.2e; decode time alone %.1f ms, batched "
        "%.1f ms\n",
        streams, samples, chunk, worst_rel, -20.0 * std::log10(std::max(worst_rel, 1e-30)),
        worst_abs, serial_ms, batched_ms);
    if (!ok || worst_abs != 0.0) {
        std::fprintf(stderr, "FAIL: batched decoding differs from decoding alone\n");
        return 1;
    }
    std::printf("batched codec matches\n");
    return 0;
}
