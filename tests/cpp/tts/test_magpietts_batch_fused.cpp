// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Batch invariance of the fused CUDA batched MagpieTTS step (decoder + local transformer with
// in-kernel sampling): B requests prefilled into slots and decoded together for K steps must get
// exactly the codes, greedy codes and alignments they get when each one is decoded alone. Also
// reports the cost of a step at each batch size.
//
// Usage: test_magpietts_batch_fused <magpie.q8_0.gguf> [--rows 1,2,4,8,16] [--steps K] [--no-cfg]
// The model may also come from NEMO_SPEECH_TEST_TTS_MODEL (a Q8_0 model is needed for the fused
// kernels); the test is skipped without a model, a GPU or fused-kernel support.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "tts/magpietts/batch/batch_backend.h"
#include "tts/magpietts/encoder.h"
#include "tts/magpietts/model.h"

namespace tts = nemo_speech::tts;

namespace {

struct Request {
    std::vector<int32_t> tokens;
    int speaker = 0;
    uint64_t seed = 0;
    tts::magpietts_backend_tensor text_device;
    std::vector<float> text_host;
    std::mt19937 rng;
    // per step: codes, argmax, alignment
    std::vector<std::vector<int32_t>> codes, argmax;
    std::vector<std::vector<float>> align;
};

tts::MagpieBatchRowInput
make_row(
    const tts::magpietts_hparams& h, Request& r, int step, const std::vector<int32_t>* prev,
    const std::vector<float>* prior) {
    tts::MagpieBatchRowInput row;
    if (prev)
        row.tokens = *prev;
    row.text_len = static_cast<int>(r.tokens.size());
    row.prior = prior;
    row.collect_alignment = true;
    row.temperature = 0.6f;
    row.cfg_scale = 2.5f;
    row.top_k = 80;
    row.forbid_eos = step < 2;
    row.seed = r.seed;
    row.frame_index = step * h.frame_stacking_factor;
    row.rng = &r.rng;
    row.text_cond = &r.text_host;
    row.text_cond_device = r.text_device.tensor ? &r.text_device : nullptr;
    row.speaker = r.speaker;
    return row;
}

std::vector<float>
prior_for(int step, int text_len) {
    std::vector<float> p(static_cast<size_t>(text_len));
    for (int j = 0; j < text_len; ++j) {
        const float d = static_cast<float>(j - step % text_len);
        p[static_cast<size_t>(j)] = std::exp(-d * d / 8.0f);
    }
    return p;
}

}  // namespace

int
main(int argc, char** argv) {
    std::string model_path;
    std::vector<int> row_counts = {1, 2, 4, 8, 16};
    int steps = 24;
    bool use_cfg = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) {
            row_counts.clear();
            std::stringstream ss(argv[++i]);
            for (std::string t; std::getline(ss, t, ',');)
                row_counts.push_back(std::atoi(t.c_str()));
        } else if (a == "--steps" && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        } else if (a == "--no-cfg") {
            use_cfg = false;
        } else {
            model_path = a;
        }
    }
    if (model_path.empty()) {
        const char* env = std::getenv("NEMO_SPEECH_TEST_TTS_MODEL");
        if (!env || !env[0]) {
            std::printf("SKIP: no MagpieTTS model\n");
            return 77;
        }
        model_path = env;
    }
    tts::MagpieModel model;
    if (!model.load(model_path, tts::MAGPIETTS_UMA_AUTO, false, false)) {
        std::fprintf(stderr, "FAIL: cannot load %s\n", model_path.c_str());
        return 1;
    }
    if (!tts::magpietts_backend_is_cuda(model.backend)) {
        std::printf("SKIP: the fused kernels need a CUDA backend\n");
        return 77;
    }
    const tts::magpietts_hparams& h = model.hparams;
    const int max_rows = *std::max_element(row_counts.begin(), row_counts.end());
    tts::MagpieBatchConfig config;
    config.max_slots = max_rows;
    config.use_cfg = use_cfg;
    tts::MagpieBatchModelBackend backend(model, config);
    std::string error;
    if (!backend.init(&error)) {
        std::fprintf(stderr, "FAIL: backend init: %s\n", error.c_str());
        return 1;
    }
    if (!backend.fused()) {
        std::printf("SKIP: fused kernels unavailable for this model/GPU (use a Q8_0 model)\n");
        return 77;
    }

    tts::MagpieEncoder encoder(model);
    std::mt19937 rng(77);
    std::vector<Request> reqs(static_cast<size_t>(max_rows));
    for (int i = 0; i < max_rows; ++i) {
        Request& r = reqs[static_cast<size_t>(i)];
        std::uniform_int_distribution<int> len(8, 120), tok(5, h.text_vocab_size - 10);
        const int n = len(rng);
        for (int j = 0; j < n; ++j) r.tokens.push_back(tok(rng));
        r.speaker = i % std::max(1, h.baked_speakers);
        r.seed = 1000 + 17 * i;
        if (!encoder.evalDevice(r.tokens, 4, r.text_device)) {
            std::fprintf(stderr, "FAIL: encoder\n");
            return 1;
        }
    }

    // Run `ids` together (slot = position in ids) for `steps` steps; record each request's outputs.
    auto run = [&](const std::vector<int>& ids, double* ms_per_step) -> bool {
        const int B = static_cast<int>(ids.size());
        std::vector<std::vector<int32_t>> prev(static_cast<size_t>(B));
        for (int b = 0; b < B; ++b) {
            Request& r = reqs[static_cast<size_t>(ids[b])];
            r.codes.clear();
            r.argmax.clear();
            r.align.clear();
            r.rng.seed(static_cast<uint32_t>(r.seed));
            std::vector<float> prior = prior_for(0, static_cast<int>(r.tokens.size()));
            tts::MagpieBatchRowInput row = make_row(h, r, 0, nullptr, &prior);
            tts::MagpieBatchRowOutput out;
            if (!backend.prefill(b, row, out))
                return false;
            r.codes.push_back(out.codes);
            r.argmax.push_back(out.argmax);
            r.align.push_back(out.alignment);
            prev[static_cast<size_t>(b)] = out.codes;
        }
        double total_ms = 0.0;
        for (int s = 1; s < steps; ++s) {
            std::vector<tts::MagpieBatchRowInput> rows(static_cast<size_t>(B));
            std::vector<std::vector<float>> priors(static_cast<size_t>(B));
            std::vector<int> slots;
            std::vector<const tts::MagpieBatchRowInput*> ptrs;
            for (int b = 0; b < B; ++b) {
                Request& r = reqs[static_cast<size_t>(ids[b])];
                priors[static_cast<size_t>(b)] = prior_for(s, static_cast<int>(r.tokens.size()));
                rows[static_cast<size_t>(b)] = make_row(
                    h, r, s, &prev[static_cast<size_t>(b)], &priors[static_cast<size_t>(b)]);
                slots.push_back(b);
                ptrs.push_back(&rows[static_cast<size_t>(b)]);
            }
            std::vector<tts::MagpieBatchRowOutput> outs;
            const auto t0 = std::chrono::steady_clock::now();
            if (!backend.decode(slots, ptrs, outs))
                return false;
            total_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            for (int b = 0; b < B; ++b) {
                Request& r = reqs[static_cast<size_t>(ids[b])];
                r.codes.push_back(outs[static_cast<size_t>(b)].codes);
                r.argmax.push_back(outs[static_cast<size_t>(b)].argmax);
                r.align.push_back(outs[static_cast<size_t>(b)].alignment);
                prev[static_cast<size_t>(b)] = outs[static_cast<size_t>(b)].codes;
            }
        }
        for (int b = 0; b < B; ++b) backend.release(b);
        if (ms_per_step)
            *ms_per_step = total_ms / std::max(1, steps - 1);
        return true;
    };

    // Reference: every request alone.
    std::vector<Request> alone(static_cast<size_t>(max_rows));
    double alone_ms = 0.0;
    for (int i = 0; i < max_rows; ++i) {
        double ms = 0.0;
        if (!run({i}, &ms)) {
            std::fprintf(stderr, "FAIL: request %d alone\n", i);
            return 1;
        }
        alone_ms += ms;
        alone[static_cast<size_t>(i)].codes = reqs[static_cast<size_t>(i)].codes;
        alone[static_cast<size_t>(i)].argmax = reqs[static_cast<size_t>(i)].argmax;
        alone[static_cast<size_t>(i)].align = reqs[static_cast<size_t>(i)].align;
    }
    std::printf("rows=1: %.3f ms/step\n", alone_ms / max_rows);
    backend.kernelTimingReport();
    int failures = 0;
    for (int B : row_counts) {
        if (B == 1)
            continue;
        std::vector<int> ids(static_cast<size_t>(B));
        // a different subset order each time: rows sit at different lanes than when alone
        for (int b = 0; b < B; ++b) ids[static_cast<size_t>(b)] = (b * 5 + B) % max_rows;
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        for (int i = 0; static_cast<int>(ids.size()) < B; ++i)
            if (std::find(ids.begin(), ids.end(), i) == ids.end())
                ids.push_back(i);
        std::reverse(ids.begin(), ids.end());
        double ms = 0.0;
        if (!run(ids, &ms)) {
            std::fprintf(stderr, "FAIL: batched run rows=%d\n", B);
            return 1;
        }
        int mismatched = 0;
        for (int id : ids) {
            const Request& r = reqs[static_cast<size_t>(id)];
            const Request& ref = alone[static_cast<size_t>(id)];
            if (r.codes != ref.codes || r.argmax != ref.argmax || r.align != ref.align)
                ++mismatched;
        }
        std::printf(
            "rows=%d: %.3f ms/step (%.2fx rows per ms vs alone), %d/%d requests bit-identical to "
            "their solo run\n",
            B, ms, (B / ms) / (1.0 / (alone_ms / max_rows)), B - mismatched, B);
        backend.kernelTimingReport();
        failures += mismatched;
    }
    if (failures) {
        std::fprintf(
            stderr, "FAIL: %d request(s) differ between batched and solo decoding\n", failures);
        return 1;
    }
    std::printf("batched fused step is batch-invariant\n");
    return 0;
}
