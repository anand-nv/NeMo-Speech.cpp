// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Teacher-forced parity of the batched MagpieTTS decoder step (slot arenas, in-flight batching)
// against the serial decoder. Several prompts of different text lengths and speakers join the
// batch at different steps (so rows sit at different positions), are fed the same forced code
// frames as their serial run, and each row's hidden states and cross-attention alignment must
// match the serial decoder's.
//
// Usage: test_magpietts_batch_decoder <magpie.gguf> [--cpu] [--steps N] [--tol X]
// The model may also come from NEMO_SPEECH_TEST_TTS_MODEL; without one the test is skipped.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "tts/magpietts/batch/batch_decoder.h"
#include "tts/magpietts/decoder.h"
#include "tts/magpietts/encoder.h"
#include "tts/magpietts/model.h"

namespace tts = nemo_speech::tts;

namespace {

struct Prompt {
    std::vector<int32_t> tokens;
    int speaker = 0;
    int join_step = 0;  // batch step at which the prompt is prefilled
    std::vector<float> text_cond;
    std::vector<std::vector<int32_t>> frames;  // forced stacked frames, one per decoder step
    // serial reference per step: hidden cond/uncond [E] and alignment
    std::vector<std::vector<float>> ref_cond, ref_uncond, ref_align;
};

// The decoders copy their hidden outputs asynchronously on the model backend's stream.
ggml_backend_t g_backend = nullptr;

std::vector<float>
read_hidden(const tts::magpietts_backend_tensor& t) {
    ggml_backend_synchronize(g_backend);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(t.tensor)));
    ggml_backend_tensor_get(t.tensor, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<float>
prior_for(int step, int text_len) {
    std::vector<float> p(static_cast<size_t>(text_len));
    const float center = static_cast<float>(step % text_len);
    for (int j = 0; j < text_len; ++j) {
        const float d = static_cast<float>(j) - center;
        p[static_cast<size_t>(j)] = std::exp(-d * d / 8.0f);
    }
    return p;
}

// Max |a - b| relative to the reference's peak magnitude (hidden states are LayerNorm outputs
// scaled by the learned gains, alignments are probabilities).
float
max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0.0f, peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            return INFINITY;
        m = std::max(m, std::fabs(a[i] - b[i]));
        peak = std::max(peak, std::fabs(b[i]));
    }
    return m / std::max(1.0f, peak);
}

// Stacked frame -> per-codebook audio code history used by the serial decoder.
void
append_frame(
    const tts::magpietts_hparams& h, const std::vector<int32_t>& stacked,
    std::vector<std::vector<int32_t>>& audio_codes) {
    for (int lane = 0; lane < h.frame_stacking_factor; ++lane)
        for (int c = 0; c < h.audio_codebooks; ++c)
            audio_codes[static_cast<size_t>(c)].push_back(
                stacked[static_cast<size_t>(c + lane * h.audio_codebooks)]);
}

}  // namespace

int
main(int argc, char** argv) {
    std::string model_path;
    bool force_cpu = false;
    int steps = 12;
    float tol = 1.0e-2f;  // CUDA F16 batched GEMMs round activations to F16
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cpu")
            force_cpu = true;
        else if (a == "--steps" && i + 1 < argc)
            steps = std::atoi(argv[++i]);
        else if (a == "--tol" && i + 1 < argc)
            tol = std::strtof(argv[++i], nullptr);
        else
            model_path = a;
    }
    if (model_path.empty()) {
        const char* env = std::getenv("NEMO_SPEECH_TEST_TTS_MODEL");
        if (!env || !env[0]) {
            std::printf("SKIP: no MagpieTTS model (argv[1] or NEMO_SPEECH_TEST_TTS_MODEL)\n");
            return 77;
        }
        model_path = env;
    }

    tts::MagpieModel model;
    if (!model.load(model_path, tts::MAGPIETTS_UMA_AUTO, force_cpu, false)) {
        std::fprintf(stderr, "FAIL: cannot load %s\n", model_path.c_str());
        return 1;
    }
    g_backend = model.backend;
    const tts::magpietts_hparams& h = model.hparams;
    const int threads = 8;
    const int E = h.n_embd;

    std::mt19937 rng(1234);
    std::vector<Prompt> prompts(4);
    const int lengths[4] = {23, 61, 9, 140};
    for (size_t i = 0; i < prompts.size(); ++i) {
        Prompt& p = prompts[i];
        std::uniform_int_distribution<int> tok(5, std::max(6, h.text_vocab_size - 10));
        for (int j = 0; j < lengths[i]; ++j) p.tokens.push_back(tok(rng));
        p.speaker = static_cast<int>(i) % std::max(1, h.baked_speakers);
        p.join_step = static_cast<int>(i);
        std::uniform_int_distribution<int> code(0, h.audio_codebook_size - 1);
        for (int s = 0; s < steps; ++s) {
            std::vector<int32_t> f(static_cast<size_t>(h.stacked_audio_codebooks()));
            for (int32_t& c : f) c = code(rng);
            p.frames.push_back(std::move(f));
        }
    }

    tts::MagpieEncoder encoder(model);
    for (Prompt& p : prompts) {
        if (!encoder.eval(p.tokens, threads, p.text_cond)) {
            std::fprintf(stderr, "FAIL: encoder\n");
            return 1;
        }
    }

    // Serial reference: one decoder per prompt, step 0 is the baked-context prefill.
    tts::magpietts_backend_tensor hc, hu;
    if (!hc.alloc2d(model, GGML_TYPE_F32, E, 1, "test_hc") ||
        !hu.alloc2d(model, GGML_TYPE_F32, E, 1, "test_hu")) {
        std::fprintf(stderr, "FAIL: hidden alloc\n");
        return 1;
    }
    for (Prompt& p : prompts) {
        tts::MagpieDecoder decoder(model);
        tts::DecoderKvCache cond_kv, uncond_kv;
        tts::DecoderCrossKvCache cross;
        const int text_len = static_cast<int>(p.tokens.size());
        std::vector<std::vector<int32_t>> audio(static_cast<size_t>(h.audio_codebooks));
        for (auto& c : audio) c.assign(static_cast<size_t>(h.frame_stacking_factor), h.audio_bos_id);
        for (int s = 0; s < steps; ++s) {
            tts::decoder_result rc, ru;
            rc.logits_required = ru.logits_required = false;
            std::vector<float> prior = prior_for(s, text_len);
            std::vector<float> align;
            tts::magpietts_decoder_attention att;
            att.prior = &prior;
            att.alignment_scores = &align;
            if (!decoder.evalCachedPair(
                    p.text_cond, text_len, audio, p.speaker, threads, cond_kv, uncond_kv, rc, ru,
                    (h.max_decoder_steps + h.frame_stacking_factor - 1) / h.frame_stacking_factor,
                    nullptr, nullptr, &hc, &hu, &cross, &att)) {
                std::fprintf(stderr, "FAIL: serial decoder step %d\n", s);
                return 1;
            }
            p.ref_cond.push_back(read_hidden(hc));
            p.ref_uncond.push_back(read_hidden(hu));
            p.ref_align.push_back(align);
            append_frame(h, p.frames[static_cast<size_t>(s)], audio);
        }
    }

    // Batched: prefill each prompt at its join step, then decode every seeded row together.
    tts::MagpieBatchConfig config;
    config.max_slots = 6;
    config.use_cfg = true;
    config.buckets = {1, 2, 4, 8};
    config.threads = threads;
    tts::MagpieBatchDecoder batch(model, config);
    std::string error;
    if (!batch.init(&error)) {
        std::fprintf(stderr, "FAIL: batched decoder init: %s\n", error.c_str());
        return 1;
    }
    tts::MagpieDecoder prefill_decoder(model);
    tts::MagpiePinnedHostScratch staging;
    const int slot_of[4] = {3, 0, 5, 1};  // non-contiguous slots
    std::vector<int> local_step(prompts.size(), -1);
    float worst_hidden = 0.0f, worst_align = 0.0f;
    int checked = 0;
    for (int t = 0; t < steps + 4; ++t) {
        std::vector<tts::MagpieBatchDecoderRow> rows;
        std::vector<size_t> row_prompt;
        std::vector<std::vector<float>> priors(prompts.size());
        for (size_t i = 0; i < prompts.size(); ++i) {
            Prompt& p = prompts[i];
            const int text_len = static_cast<int>(p.tokens.size());
            if (t == p.join_step) {
                tts::DecoderKvCache cond_kv, uncond_kv;
                tts::DecoderCrossKvCache cross;
                std::vector<std::vector<int32_t>> bos(static_cast<size_t>(h.audio_codebooks));
                for (auto& c : bos) c.assign(static_cast<size_t>(h.frame_stacking_factor), h.audio_bos_id);
                std::vector<float> prior = prior_for(0, text_len);
                std::vector<float> align;
                tts::magpietts_decoder_attention att;
                att.prior = &prior;
                att.alignment_scores = &align;
                if (!prefill_decoder.prefillPair(
                        p.text_cond, text_len, bos, p.speaker, threads, cond_kv, uncond_kv, nullptr,
                        &hc, &hu, &cross, &att, staging, nullptr, nullptr) ||
                    !batch.seedSlot(slot_of[i], cond_kv, &uncond_kv) ||
                    !batch.setSlotCross(slot_of[i], cross, text_len)) {
                    std::fprintf(stderr, "FAIL: prefill/seed prompt %zu\n", i);
                    return 1;
                }
                const float ec =
                    max_abs_diff(read_hidden(hc).data(), p.ref_cond[0].data(), static_cast<size_t>(E));
                const float eu =
                    max_abs_diff(read_hidden(hu).data(), p.ref_uncond[0].data(), static_cast<size_t>(E));
                const float e = std::max(ec, eu);
                if (std::getenv("BATCH_DECODER_TEST_VERBOSE"))
                    std::printf("  prefill prompt=%zu cond=%.3g uncond=%.3g\n", i, ec, eu);
                worst_hidden = std::max(worst_hidden, e);
                local_step[i] = 1;
                continue;
            }
            if (local_step[i] >= 1 && local_step[i] < steps) {
                const int s = local_step[i];
                priors[i] = prior_for(s, text_len);
                tts::MagpieBatchDecoderRow row;
                row.slot = slot_of[i];
                row.tokens = p.frames[static_cast<size_t>(s - 1)].data();
                row.text_len = text_len;
                row.prior = &priors[i];
                row.collect_alignment = true;
                rows.push_back(row);
                row_prompt.push_back(i);
            }
        }
        if (rows.empty())
            continue;
        const int bucket = tts::magpie_batch_bucket(batch.buckets(), static_cast<int>(rows.size()) + 1);
        ggml_runtime::DeviceTensor hidden;
        std::vector<std::vector<float>> align;
        if (!batch.step(rows, bucket, hidden, align)) {
            std::fprintf(stderr, "FAIL: batched step %d\n", t);
            return 1;
        }
        std::vector<float> host(static_cast<size_t>(ggml_nelements(hidden.tensor)));
        ggml_backend_tensor_get(hidden.tensor, host.data(), 0, host.size() * sizeof(float));
        for (size_t r = 0; r < rows.size(); ++r) {
            const size_t i = row_prompt[r];
            const int s = local_step[i];
            const float* cond = host.data() + r * E;
            const float* uncond = host.data() + (static_cast<size_t>(bucket) + r) * E;
            const float e = std::max(
                max_abs_diff(cond, prompts[i].ref_cond[static_cast<size_t>(s)].data(), static_cast<size_t>(E)),
                max_abs_diff(uncond, prompts[i].ref_uncond[static_cast<size_t>(s)].data(), static_cast<size_t>(E)));
            const auto& ref_align = prompts[i].ref_align[static_cast<size_t>(s)];
            float ea = 0.0f;
            if (ref_align.size() != align[r].size())
                ea = INFINITY;
            else
                ea = max_abs_diff(align[r].data(), ref_align.data(), ref_align.size());
            worst_hidden = std::max(worst_hidden, e);
            worst_align = std::max(worst_align, ea);
            if (std::getenv("BATCH_DECODER_TEST_VERBOSE"))
                std::printf(
                    "  t=%d prompt=%zu slot=%d step=%d rows=%zu bucket=%d hidden=%.3g align=%.3g\n", t,
                    i, rows[r].slot, s, rows.size(), bucket, e, ea);
            ++checked;
            ++local_step[i];
        }
    }
    std::printf(
        "batched decoder vs serial: %d row-steps over %zu prompts, max rel. hidden diff %.3g, "
        "max rel. alignment diff %.3g (tol %.3g)\n",
        checked, prompts.size(), worst_hidden, worst_align, tol);
    if (checked == 0 || !(worst_hidden <= tol) || !(worst_align <= tol)) {
        std::fprintf(stderr, "FAIL: batched decoder parity\n");
        return 1;
    }
    std::printf("batched decoder parity OK\n");
    return 0;
}
