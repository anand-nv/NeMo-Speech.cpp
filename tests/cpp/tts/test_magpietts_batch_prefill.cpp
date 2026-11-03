// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Batched prefill of the MagpieTTS in-flight batching engine
// (MagpieBatchModelBackend::prefillBatch, one decoder graph) against prefilling each text chunk
// alone (prefill):
// - one row per graph must leave bit-identical caches (self-attention K/V of both lanes,
//   cross-attention K/V, alignment) and greedy codes, so the graph computes what prefill does;
// - N rows per graph cannot be bit-identical: the wider matmuls sum in another order, and the
//   Q8_0 linears re-quantize their activations, which amplifies that to a few percent of the
//   self-attention cache and about a third of the greedy codes (the same as perturbing a lone
//   prefill's input by 1e-7). The batch is checked against bounds that separate this noise from
//   a row reading another row's speaker, text or prior: the layer-0 K/V (the assembled inputs)
//   and the cross K/V must match closely, the deeper caches and the codes loosely.
//
// Usage: test_magpietts_batch_prefill <magpie.gguf> [--rows N] [--steps K]
// A Q8_0 model runs the fused kernels, an F16 model the ggml graphs. The model may also come from
// NEMO_SPEECH_TEST_TTS_MODEL; skipped without a model or a GPU.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "ggml-backend.h"
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
    std::vector<float> prior;
    std::mt19937 rng;
};

tts::MagpieBatchRowInput
make_row(const tts::magpietts_hparams& h, Request& r, int step, const std::vector<int32_t>* prev) {
    tts::MagpieBatchRowInput row;
    if (prev)
        row.tokens = *prev;
    row.text_len = static_cast<int>(r.tokens.size());
    row.prior = &r.prior;
    row.collect_alignment = true;
    row.temperature = 0.6f;
    row.cfg_scale = 2.5f;
    row.top_k = 80;
    row.forbid_eos = step < 2;
    row.seed = r.seed;
    row.frame_index = step * h.frame_stacking_factor;
    row.rng = &r.rng;
    row.text_cond = &r.text_host;
    row.text_cond_device = &r.text_device;
    row.speaker = r.speaker;
    return row;
}

// max |a - b| / max |a|
double
rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 1e-12;
    for (size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, (double)std::fabs(a[i] - b[i]));
        den = std::max(den, (double)std::fabs(a[i]));
    }
    return num / den;
}

std::vector<float>
read(const ggml_tensor* t, size_t offset, size_t floats) {
    std::vector<float> v(floats);
    ggml_backend_tensor_get(t, v.data(), offset, floats * sizeof(float));
    return v;
}

}  // namespace

int
main(int argc, char** argv) {
    std::string model_path;
    int n_rows = 12;
    int steps = 16;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc)
            n_rows = std::atoi(argv[++i]);
        else if (a == "--steps" && i + 1 < argc)
            steps = std::atoi(argv[++i]);
        else
            model_path = a;
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
        std::printf("SKIP: needs a CUDA backend\n");
        return 77;
    }
    const tts::magpietts_hparams& h = model.hparams;
    // slots [0, N): prefilled one at a time; [N, 2N): one graph per row; [2N, 3N): one batch;
    // [3N, 4N): the same batch on the prefill stream
    tts::MagpieBatchConfig config;
    config.max_slots = 4 * n_rows;
    config.use_cfg = true;
    config.prefill_batch_size = n_rows;
    config.prefill_stream = true;
    tts::MagpieBatchModelBackend backend(model, config);
    std::string error;
    if (!backend.init(&error)) {
        std::fprintf(stderr, "FAIL: backend init: %s\n", error.c_str());
        return 1;
    }
    std::printf("%s path\n", backend.fused() ? "fused" : "ggml graph");  // Q8_0 or F16 (and others)

    tts::MagpieEncoder encoder(model);
    std::mt19937 gen(91);
    std::vector<Request> reqs(static_cast<size_t>(n_rows));
    for (int i = 0; i < n_rows; ++i) {
        Request& r = reqs[static_cast<size_t>(i)];
        std::uniform_int_distribution<int> len(8, 160), tok(5, h.text_vocab_size - 10);
        const int n = len(gen);
        for (int j = 0; j < n; ++j) r.tokens.push_back(tok(gen));
        r.speaker = i % std::max(1, h.baked_speakers);
        r.seed = 4000 + 31 * i;
        for (int j = 0; j < n; ++j) r.prior.push_back(std::exp(-0.05f * j) + 0.01f);
        if (!encoder.evalDevice(r.tokens, 4, r.text_device)) {
            std::fprintf(stderr, "FAIL: encoder\n");
            return 1;
        }
    }

    // encoder: one window per batched graph must equal evalDevice exactly; all windows in one
    // graph differ by the summation-order noise (a few 1e-3, the same for unpadded and heavily
    // padded windows), far below what reading the padding would cause (order 1)
    {
        std::vector<tts::magpietts_backend_tensor> one(static_cast<size_t>(n_rows)),
            all(static_cast<size_t>(n_rows));
        std::vector<const std::vector<int32_t>*> windows;
        std::vector<tts::magpietts_backend_tensor*> outs_all;
        ggml_gallocr_t allocr = nullptr;
        double one_diff = 0.0, all_diff = 0.0;
        for (int i = 0; i < n_rows; ++i) {
            windows.push_back(&reqs[static_cast<size_t>(i)].tokens);
            outs_all.push_back(&all[static_cast<size_t>(i)]);
            if (!encoder.evalDeviceBatch(
                    {windows.back()}, 4, {&one[static_cast<size_t>(i)]}, &allocr)) {
                std::fprintf(stderr, "FAIL: batched encoder of request %d alone\n", i);
                return 1;
            }
        }
        if (!encoder.evalDeviceBatch(windows, 4, outs_all, &allocr)) {
            std::fprintf(stderr, "FAIL: batched encoder\n");
            return 1;
        }
        for (int i = 0; i < n_rows; ++i) {
            const ggml_tensor* ref = reqs[static_cast<size_t>(i)].text_device.tensor;
            const size_t n = static_cast<size_t>(ggml_nelements(ref));
            const auto x = read(ref, 0, n);
            one_diff =
                std::max(one_diff, rel_diff(x, read(one[static_cast<size_t>(i)].tensor, 0, n)));
            all_diff =
                std::max(all_diff, rel_diff(x, read(all[static_cast<size_t>(i)].tensor, 0, n)));
        }
        ggml_gallocr_free(allocr);
        std::printf(
            "encoder, %d windows: max relative difference vs evalDevice: one per graph %.2e, all "
            "in "
            "one graph %.2e\n",
            n_rows, one_diff, all_diff);
        if (one_diff != 0.0 || all_diff > 2e-2) {
            std::fprintf(stderr, "FAIL: batched encoder differs from evalDevice\n");
            return 1;
        }
    }

    // prefill: alone, one row per graph, then all rows in one graph
    std::vector<std::vector<tts::MagpieBatchRowOutput>> outs(3);
    std::vector<tts::MagpieBatchRowInput> rows(static_cast<size_t>(n_rows));
    outs[0].resize(static_cast<size_t>(n_rows));
    for (int i = 0; i < n_rows; ++i) {
        Request& r = reqs[static_cast<size_t>(i)];
        r.rng.seed(static_cast<uint32_t>(r.seed));
        rows[static_cast<size_t>(i)] = make_row(h, r, 0, nullptr);
        if (!backend.prefill(i, rows[static_cast<size_t>(i)], outs[0][static_cast<size_t>(i)])) {
            std::fprintf(stderr, "FAIL: prefill %d\n", i);
            return 1;
        }
    }
    for (int i = 0; i < n_rows; ++i) {
        std::vector<tts::MagpieBatchRowOutput> one;
        if (!backend.prefillBatch({n_rows + i}, {&rows[static_cast<size_t>(i)]}, one) ||
            one.size() != 1) {
            std::fprintf(stderr, "FAIL: prefillBatch of request %d alone\n", i);
            return 1;
        }
        outs[1].push_back(std::move(one[0]));
    }
    std::vector<int> bslots;
    std::vector<const tts::MagpieBatchRowInput*> bptrs;
    for (int i = 0; i < n_rows; ++i) {
        bslots.push_back(2 * n_rows + i);
        bptrs.push_back(&rows[static_cast<size_t>(i)]);
    }
    if (!backend.prefillBatch(bslots, bptrs, outs[2]) ||
        static_cast<int>(outs[2].size()) != n_rows) {
        std::fprintf(stderr, "FAIL: prefillBatch\n");
        return 1;
    }

    // the batch again through the prefill stream (prefillBegin / prefillCompute / prefillEnd):
    // the same graphs on another stream, so it must equal the batch in [2N, 3N) bit for bit
    if (!backend.prefillStream()) {
        std::fprintf(stderr, "FAIL: no prefill stream\n");
        return 1;
    }
    {
        tts::MagpieBatchPrefillJob job;
        for (int i = 0; i < n_rows; ++i) {
            job.slots.push_back(3 * n_rows + i);
            job.rows.push_back(&rows[static_cast<size_t>(i)]);
        }
        if (!backend.prefillBegin(job) || !backend.prefillCompute(job) ||
            !backend.prefillEnd(job) || static_cast<int>(job.out.size()) != n_rows) {
            std::fprintf(stderr, "FAIL: prefill on the prefill stream\n");
            return 1;
        }
        const tts::MagpieBatchDecoder& d = backend.decoder();
        int same = 0;
        for (int i = 0; i < n_rows; ++i) {
            const int a = 2 * n_rows + i, b = 3 * n_rows + i;
            bool eq =
                d.slotTokens(a) == d.slotTokens(b) &&
                job.out[static_cast<size_t>(i)].codes == outs[2][static_cast<size_t>(i)].codes &&
                job.out[static_cast<size_t>(i)].argmax == outs[2][static_cast<size_t>(i)].argmax &&
                job.out[static_cast<size_t>(i)].alignment ==
                    outs[2][static_cast<size_t>(i)].alignment;
            const size_t E = static_cast<size_t>(h.n_embd);
            const size_t n = static_cast<size_t>(d.slotTokens(a)) * E;
            const size_t row0 =
                static_cast<size_t>(d.cacheLen() - d.slotTokens(a)) * E * sizeof(float);
            for (int il = 0; il < h.n_dec_layer && eq; ++il) {
                const ggml_tensor* arena = d.selfArenaTensor(il);
                for (int plane = 0; plane < 2; ++plane) {
                    for (int lane = 0; lane < 2; ++lane) {
                        const int la =
                            lane == 0 ? d.layout().condLane(a) : d.layout().uncondLane(a);
                        const int lb =
                            lane == 0 ? d.layout().condLane(b) : d.layout().uncondLane(b);
                        eq = eq &&
                             read(arena, plane * arena->nb[2] + la * arena->nb[1] + row0, n) ==
                                 read(arena, plane * arena->nb[2] + lb * arena->nb[1] + row0, n);
                    }
                }
                const size_t tl = reqs[static_cast<size_t>(i)].tokens.size();
                for (const ggml_tensor* t : {d.crossKTensor(), d.crossVTensor()}) {
                    const size_t cd = static_cast<size_t>(t->ne[0]) / d.textCapacity();
                    eq = eq && read(t, a * t->nb[2] + il * t->nb[1], cd * tl) ==
                                   read(t, b * t->nb[2] + il * t->nb[1], cd * tl);
                }
            }
            same += eq;
        }
        std::printf(
            "prefill stream vs the same batch on the model stream: %d/%d requests bit-identical\n",
            same, n_rows);
        if (same != n_rows) {
            std::fprintf(stderr, "FAIL: the prefill stream changes the prefill\n");
            return 1;
        }
    }

    // caches of group g against the alone group
    const tts::MagpieBatchDecoder& dec = backend.decoder();
    const tts::MagpieLaneLayout& layout = dec.layout();
    const int n_tok = dec.slotTokens(0);
    const size_t E = static_cast<size_t>(h.n_embd);
    struct Diff {
        double self_l0 = 0.0, self = 0.0, cross = 0.0, align = 0.0, agreement = 1.0;
        int first_same = 0;
    };
    Diff diff[3];
    bool ok = true;
    for (int g = 1; g < 3; ++g) {
        Diff& d = diff[g];
        for (int i = 0; i < n_rows; ++i) {
            const int a = i, b = g * n_rows + i;
            if (dec.slotTokens(b) != n_tok || dec.slotTokens(a) != n_tok) {
                std::fprintf(
                    stderr, "FAIL: request %d: slot positions %d vs %d\n", i, dec.slotTokens(a),
                    dec.slotTokens(b));
                ok = false;
            }
            for (int il = 0; il < h.n_dec_layer; ++il) {
                const ggml_tensor* arena = dec.selfArenaTensor(il);
                const size_t row0 = static_cast<size_t>(dec.cacheLen() - n_tok) * E * sizeof(float);
                for (int plane = 0; plane < 2; ++plane) {
                    for (int lane = 0; lane < 2; ++lane) {
                        const int la = lane == 0 ? layout.condLane(a) : layout.uncondLane(a);
                        const int lb = lane == 0 ? layout.condLane(b) : layout.uncondLane(b);
                        const auto x =
                            read(arena, plane * arena->nb[2] + la * arena->nb[1] + row0, n_tok * E);
                        const auto y =
                            read(arena, plane * arena->nb[2] + lb * arena->nb[1] + row0, n_tok * E);
                        const double r = rel_diff(x, y);
                        d.self = std::max(d.self, r);
                        if (il == 0)
                            d.self_l0 = std::max(d.self_l0, r);
                    }
                }
                const int tl = static_cast<int>(reqs[static_cast<size_t>(i)].tokens.size());
                for (const ggml_tensor* t : {dec.crossKTensor(), dec.crossVTensor()}) {
                    const size_t cd = static_cast<size_t>(t->ne[0]) / dec.textCapacity();
                    const auto x = read(t, a * t->nb[2] + il * t->nb[1], cd * tl);
                    const auto y = read(t, b * t->nb[2] + il * t->nb[1], cd * tl);
                    d.cross = std::max(d.cross, rel_diff(x, y));
                }
            }
            d.align = std::max(
                d.align, rel_diff(
                             outs[0][static_cast<size_t>(i)].alignment,
                             outs[g][static_cast<size_t>(i)].alignment));
            d.first_same +=
                outs[0][static_cast<size_t>(i)].argmax == outs[g][static_cast<size_t>(i)].argmax;
        }
    }

    // follow every group for a few steps with the alone group's codes as input (teacher forcing),
    // so the greedy codes compare step by step
    int agree[3] = {}, total = 0;
    std::vector<std::vector<int32_t>> prev(static_cast<size_t>(n_rows));
    for (int i = 0; i < n_rows; ++i)
        prev[static_cast<size_t>(i)] = outs[0][static_cast<size_t>(i)].codes;
    for (int s = 1; s < steps; ++s) {
        std::vector<tts::MagpieBatchRowInput> srows(static_cast<size_t>(3 * n_rows));
        std::vector<int> slots;
        std::vector<const tts::MagpieBatchRowInput*> ptrs;
        for (int g = 0; g < 3; ++g) {
            for (int i = 0; i < n_rows; ++i) {
                Request& r = reqs[static_cast<size_t>(i)];
                srows[static_cast<size_t>(g * n_rows + i)] =
                    make_row(h, r, s, &prev[static_cast<size_t>(i)]);
                slots.push_back(g * n_rows + i);
                ptrs.push_back(&srows[static_cast<size_t>(g * n_rows + i)]);
            }
        }
        std::vector<tts::MagpieBatchRowOutput> step;
        if (!backend.decode(slots, ptrs, step)) {
            std::fprintf(stderr, "FAIL: decode step %d\n", s);
            return 1;
        }
        for (int i = 0; i < n_rows; ++i) {
            const auto& x = step[static_cast<size_t>(i)].argmax;
            for (int g = 1; g < 3; ++g) {
                const auto& y = step[static_cast<size_t>(g * n_rows + i)].argmax;
                for (size_t c = 0; c < x.size(); ++c) agree[g] += x[c] == y[c];
            }
            total += static_cast<int>(x.size());
            prev[static_cast<size_t>(i)] = step[static_cast<size_t>(i)].codes;
        }
    }
    for (int g = 1; g < 3; ++g) {
        const Diff& d = diff[g];
        diff[g].agreement = total ? (double)agree[g] / total : 1.0;
        std::printf(
            "%s vs alone, %d requests (%d decoder rows each): max relative difference self K/V "
            "%.2e "
            "(layer 0 %.2e), cross K/V %.2e, alignment %.2e; greedy codes identical for %d/%d "
            "first "
            "frames, %.2f%% over %d decode steps\n",
            g == 1 ? "one row per graph" : "one batch", n_rows, n_tok, d.self, d.self_l0, d.cross,
            d.align, d.first_same, n_rows, 100.0 * diff[g].agreement, steps - 1);
    }
    const Diff& one = diff[1];
    if (one.self != 0.0 || one.cross != 0.0 || one.align != 0.0 || one.first_same != n_rows ||
        one.agreement != 1.0) {
        std::fprintf(stderr, "FAIL: a one-row batched prefill is not bit-identical to prefill\n");
        ok = false;
    }
    // Measured with 12 requests: Q8_0 self K/V 4e-2 (layer 0 1e-7), cross K/V 3e-4, alignment
    // 2e-2, 66% of the greedy codes; F16 self K/V 6e-3, cross K/V 3e-3 (cuBLAS runs F16 weights
    // with FP16 compute, its kernel chosen by the column count), alignment 4e-3, 85% of the codes.
    // A row with another row's speaker or text is off by O(1).
    const Diff& all = diff[2];
    if (all.self_l0 > 1e-5 || all.cross > 1e-2 || all.self > 0.15 || all.align > 0.1 ||
        all.agreement < 0.5) {
        std::fprintf(stderr, "FAIL: batched prefill differs from prefilling alone beyond noise\n");
        ok = false;
    }
    if (!ok)
        return 1;
    std::printf("batched prefill matches\n");
    return 0;
}
