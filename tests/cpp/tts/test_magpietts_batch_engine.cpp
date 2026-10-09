// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// End-to-end MagpieTTS in-flight batching: submits every text of a file at once to a
// MagpieBatchEngine, writes one WAV per request and reports aggregate throughput (total audio /
// wall time) and per-request latency. With --check it also asserts that every request completes
// with audio, that slots are reused when there are more requests than slots, and that cancelling
// a queued and an active request fails exactly those two.
//
// Usage: test_magpietts_batch_engine --magpie M.gguf --codec C.gguf --tokenizer DIR
//            --texts FILE [--n N] [--slots S] [--out DIR] [--seed X] [--cpu] [--no-cfg]
//            [--check] [--max-steps N] [--stress N]
// --stress N submits N requests at random times and cancels about one in six at random moments;
// every request must report exactly once, every uncancelled one must succeed and the engine must
// end empty.
// Model paths may also come from NEMO_SPEECH_TEST_TTS_MODEL / NEMO_SPEECH_TEST_CODEC_MODEL /
// NEMO_SPEECH_TEST_TOKENIZER_DIR; without them the test is skipped.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "tts/magpietts/batch/batch_engine.h"
#include "tts/magpietts/magpietts.h"
#include "tts/nanocodec/model.h"
#include "tts/tokenizer/tokenizer.h"

namespace tts = nemo_speech::tts;

namespace {

const char*
env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v && v[0] ? v : fallback;
}

bool
write_wav(const std::string& path, int sample_rate, const std::vector<uint8_t>& pcm) {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        return false;
    auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
    out.write("RIFF", 4);
    u32(36 + static_cast<uint32_t>(pcm.size()));
    out.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(1);
    u32(static_cast<uint32_t>(sample_rate));
    u32(static_cast<uint32_t>(sample_rate * 2));
    u16(2);
    u16(16);
    out.write("data", 4);
    u32(static_cast<uint32_t>(pcm.size()));
    out.write(reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(pcm.size()));
    return static_cast<bool>(out);
}

struct Job {
    std::string text;
    std::vector<std::vector<int32_t>> chunks;
    tts::magpie_stream_params params;
    tts::stream_run_metrics metrics;
    std::vector<uint8_t> pcm;
    std::mutex pcm_mutex;
    int64_t id = -1;
    bool done = false;
    bool ok = false;
    double submit_s = 0.0, finish_s = 0.0;
};

}  // namespace

int
main(int argc, char** argv) {
    std::string magpie_path = env_or("NEMO_SPEECH_TEST_TTS_MODEL", "");
    std::string codec_path = env_or("NEMO_SPEECH_TEST_CODEC_MODEL", "");
    std::string tokenizer_dir = env_or("NEMO_SPEECH_TEST_TOKENIZER_DIR", "");
    std::string texts_path, out_dir;
    int n = 0, slots = 8, seed = 1234, max_steps = 0, stress = 0;
    bool force_cpu = false, use_cfg = true, check = false, same_seed = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--magpie")
            magpie_path = next();
        else if (a == "--codec")
            codec_path = next();
        else if (a == "--tokenizer")
            tokenizer_dir = next();
        else if (a == "--texts")
            texts_path = next();
        else if (a == "--out")
            out_dir = next();
        else if (a == "--n")
            n = std::atoi(next().c_str());
        else if (a == "--slots")
            slots = std::atoi(next().c_str());
        else if (a == "--seed")
            seed = std::atoi(next().c_str());
        else if (a == "--max-steps")
            max_steps = std::atoi(next().c_str());
        else if (a == "--stress")
            stress = std::atoi(next().c_str());
        else if (a == "--cpu")
            force_cpu = true;
        else if (a == "--no-cfg")
            use_cfg = false;
        else if (a == "--check")
            check = true;
        else if (a == "--same-seed")
            same_seed = true;  // every request uses --seed (compare with serial runs)
    }
    if (magpie_path.empty() || codec_path.empty() || tokenizer_dir.empty()) {
        std::printf("SKIP: needs --magpie/--codec/--tokenizer (or NEMO_SPEECH_TEST_* env)\n");
        return 77;
    }
    std::vector<std::string> texts;
    if (!texts_path.empty()) {
        std::ifstream in(texts_path);
        for (std::string line; std::getline(in, line);)
            if (!line.empty())
                texts.push_back(line);
    } else {
        texts = {
            "The quick brown fox jumps over the lazy dog.",
            "In-flight batching lets several requests share every decoder step.",
            "Short one.",
            "A considerably longer sentence keeps decoding well after the short requests are done, "
            "so their slots are reused by the requests waiting behind them.",
            "Numbers like twenty three and four hundred are read aloud.",
            "Hello there, how are you doing today?",
            "This request arrives with the others but waits for a free slot.",
            "Speech synthesis on the GPU, one frame at a time.",
            "Cancel me if you can.",
            "The last request of the batch.",
        };
    }
    if (n > 0 && n < static_cast<int>(texts.size()))
        texts.resize(static_cast<size_t>(n));

    tts::MagpieModel magpie;
    tts::nanocodec::NanoCodecModel codec;
    if (!magpie.load(magpie_path, tts::MAGPIETTS_UMA_AUTO, force_cpu, false) ||
        !codec.load(codec_path, force_cpu, false)) {
        std::fprintf(stderr, "FAIL: cannot load models\n");
        return 1;
    }
    tts::MagpieBatchConfig config;
    config.max_slots = slots;
    config.use_cfg = use_cfg;
    config.threads = 8;
    tts::MagpieBatchEngine engine(magpie, codec, config);
    std::string error;
    if (!engine.init(&error)) {
        std::fprintf(stderr, "FAIL: engine init: %s\n", error.c_str());
        return 1;
    }
    tts::MagpieNativeTokenizer tokenizer(tokenizer_dir);
    std::vector<std::unique_ptr<Job>> jobs;
    for (size_t i = 0; i < texts.size(); ++i) {
        auto job = std::make_unique<Job>();
        job->text = texts[i];
        const tts::MagpieTokenizationResult tok = tokenizer.tokenize(texts[i], "en-US");
        for (const auto& chunk : tok.chunks) job->chunks.push_back(chunk.tokens);
        job->params.seed = same_seed ? seed : seed + static_cast<int>(i);
        job->params.use_cfg = use_cfg;
        if (max_steps > 0)
            job->params.steps = max_steps;
        jobs.push_back(std::move(job));
    }

    // Warm-up: one request alone (graph allocation, CUDA graph capture).
    {
        tts::stream_run_metrics m;
        tts::magpie_stream_params p;
        p.seed = 1;
        p.use_cfg = use_cfg;
        if (!engine.synthesize(p, jobs[0]->chunks, [](const std::vector<uint8_t>&) { return true; }, m, "warmup", &error)) {
            std::fprintf(stderr, "FAIL: warm-up request: %s\n", error.c_str());
            return 1;
        }
    }

    std::mutex mutex;
    std::condition_variable cv;
    int remaining = static_cast<int>(jobs.size());
    const auto t0 = std::chrono::steady_clock::now();
    auto now_s = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    for (size_t i = 0; i < jobs.size(); ++i) {
        Job* job = jobs[i].get();
        job->submit_s = now_s();
        job->id = engine.submit(
            job->params, job->chunks,
            [job](const std::vector<uint8_t>& bytes) {
                std::lock_guard<std::mutex> lock(job->pcm_mutex);
                job->pcm.insert(job->pcm.end(), bytes.begin(), bytes.end());
                return true;
            },
            &job->metrics, "batch" + std::to_string(i),
            [&, job](bool ok) {
                std::lock_guard<std::mutex> lock(mutex);
                job->ok = ok;
                job->done = true;
                job->finish_s = now_s();
                --remaining;
                cv.notify_all();
            },
            &error);
        if (job->id < 0) {
            std::fprintf(stderr, "FAIL: submit %zu: %s\n", i, error.c_str());
            return 1;
        }
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return remaining == 0; });
    }
    const double wall_s = now_s();
    const int sample_rate = codec.sampleRate();
    double audio_s = 0.0, ttfa_sum = 0.0;
    int failures = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        Job& job = *jobs[i];
        const double a = static_cast<double>(job.pcm.size() / 2) / sample_rate;
        audio_s += a;
        ttfa_sum += job.metrics.ttfa_ms;
        std::printf(
            "req %2zu ok=%d frames=%4d audio=%6.2fs ttfa=%7.1fms done_at=%6.2fs  %s\n", i,
            job.ok ? 1 : 0, job.metrics.generated_frames, a, job.metrics.ttfa_ms, job.finish_s,
            job.text.substr(0, 60).c_str());
        if (!job.ok || job.pcm.empty())
            ++failures;
        if (!out_dir.empty()) {
            char name[64];
            std::snprintf(name, sizeof(name), "/%02zu", i + 1);
            write_wav(out_dir + name + ".wav", sample_rate, job.pcm);
            std::ofstream(out_dir + name + ".txt") << job.text << "\n";
            std::ofstream(out_dir + name + ".log")
                << "generated " << job.metrics.generated_frames << " frames e2e_audio_s=" << a
                << " e2e_elapsed_s=" << (job.finish_s - job.submit_s)
                << " e2e_rtfx=" << (a / std::max(1e-9, job.finish_s - job.submit_s))
                << " e2e_ttfa_ms=" << job.metrics.ttfa_ms << "\n";
        }
    }
    const tts::MagpieBatchStats st = engine.stats();
    std::printf(
        "requests=%zu slots=%d cfg=%d audio=%.2fs wall=%.2fs aggregate_rtfx=%.2f mean_ttfa=%.1fms "
        "decode_steps=%llu mean_batch=%.2f prefills=%llu max_active=%d prefill_ms=%.1f "
        "decode_ms=%.1f (%.2f ms/step)\n",
        jobs.size(), slots, use_cfg ? 1 : 0, audio_s, wall_s, audio_s / wall_s,
        ttfa_sum / static_cast<double>(jobs.size()), (unsigned long long)st.decode_steps,
        st.decode_steps ? static_cast<double>(st.decode_rows) / st.decode_steps : 0.0,
        (unsigned long long)st.prefills, st.max_active, st.prefill_ms, st.decode_ms,
        st.decode_steps ? st.decode_ms / st.decode_steps : 0.0);

    if (check) {
        if (st.max_active > slots) {
            std::fprintf(stderr, "FAIL: %d requests active with %d slots\n", st.max_active, slots);
            ++failures;
        }
        // Cancellation: with every slot busy, cancel one active and one queued request.
        std::vector<std::unique_ptr<Job>> extra;
        int extra_remaining = 0;
        for (int i = 0; i < slots + 2; ++i) {
            auto job = std::make_unique<Job>();
            job->chunks = jobs[static_cast<size_t>(i) % jobs.size()]->chunks;
            job->params.seed = 99 + i;
            job->params.use_cfg = use_cfg;
            Job* jp = job.get();
            jp->id = engine.submit(
                jp->params, jp->chunks, [](const std::vector<uint8_t>&) { return true; },
                &jp->metrics, "cancel" + std::to_string(i),
                [&, jp](bool ok) {
                    std::lock_guard<std::mutex> lock(mutex);
                    jp->ok = ok;
                    jp->done = true;
                    --extra_remaining;
                    cv.notify_all();
                },
                &error);
            if (jp->id >= 0)
                ++extra_remaining;
            extra.push_back(std::move(job));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        const int64_t active_id = extra[0]->id;
        const int64_t queued_id = extra.back()->id;
        engine.cancel(active_id);
        engine.cancel(queued_id);
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return extra_remaining == 0; });
        }
        for (size_t i = 0; i < extra.size(); ++i) {
            const bool cancelled = extra[i]->id == active_id || extra[i]->id == queued_id;
            if (extra[i]->ok == cancelled) {
                std::fprintf(
                    stderr, "FAIL: request %zu ok=%d (cancelled=%d)\n", i, extra[i]->ok ? 1 : 0,
                    cancelled ? 1 : 0);
                ++failures;
            }
        }
        const tts::MagpieBatchStats after = engine.stats();
        std::printf(
            "cancellation: completed=%llu cancelled=%llu failed=%llu active=%d queued=%d\n",
            (unsigned long long)after.completed, (unsigned long long)after.cancelled,
            (unsigned long long)after.failed, after.active, after.queued);
        if (after.cancelled != 2 || after.active != 0 || after.queued != 0) {
            std::fprintf(stderr, "FAIL: unexpected engine state after cancellation\n");
            ++failures;
        }
    }
    if (stress > 0) {
        std::mt19937 rng(static_cast<uint32_t>(seed));
        std::uniform_int_distribution<int> delay_ms(0, 15);
        std::uniform_int_distribution<int> pick(0, static_cast<int>(jobs.size()) - 1);
        std::uniform_int_distribution<int> coin(0, 5);
        struct StressJob {
            Job job;
            std::atomic<int> calls{0};
            bool cancel_target = false;
        };
        std::vector<std::unique_ptr<StressJob>> sj;
        std::atomic<int> outstanding{0};
        const tts::MagpieBatchStats before = engine.stats();
        for (int i = 0; i < stress; ++i) {
            auto s = std::make_unique<StressJob>();
            s->job.chunks = jobs[static_cast<size_t>(pick(rng))]->chunks;
            s->job.params.seed = 5000 + i;
            s->job.params.use_cfg = use_cfg;
            if (max_steps > 0)
                s->job.params.steps = max_steps;
            s->cancel_target = coin(rng) == 0;
            StressJob* sp = s.get();
            sp->job.id = engine.submit(
                sp->job.params, sp->job.chunks, [](const std::vector<uint8_t>&) { return true; },
                &sp->job.metrics, "stress" + std::to_string(i),
                [&, sp](bool ok) {
                    sp->job.ok = ok;
                    ++sp->calls;
                    --outstanding;
                    std::lock_guard<std::mutex> lock(mutex);
                    cv.notify_all();
                },
                &error);
            if (sp->job.id < 0) {
                std::fprintf(stderr, "FAIL: stress submit %d: %s\n", i, error.c_str());
                ++failures;
                break;
            }
            ++outstanding;
            sj.push_back(std::move(s));
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms(rng)));
            // cancel an earlier target at a random moment (queued, active or already done)
            if (!sj.empty() && coin(rng) < 2) {
                StressJob* target = sj[static_cast<size_t>(rng() % sj.size())].get();
                if (target->cancel_target)
                    engine.cancel(target->job.id);
            }
        }
        for (auto& s : sj)
            if (s->cancel_target)
                engine.cancel(s->job.id);
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait_for(lock, std::chrono::seconds(600), [&] { return outstanding == 0; });
        }
        int ok = 0, bad = 0, targets = 0;
        for (auto& s : sj) {
            targets += s->cancel_target ? 1 : 0;
            if (s->calls != 1) {
                std::fprintf(stderr, "FAIL: stress request reported %d times\n", s->calls.load());
                ++failures;
            }
            if (s->job.ok)
                ++ok;
            else if (!s->cancel_target)
                ++bad;
        }
        // let the finishing threads update the counters
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const tts::MagpieBatchStats after = engine.stats();
        std::printf(
            "stress: %zu requests, %d cancel targets, %d ok, %d unexpected failures; "
            "engine: completed=%llu cancelled=%llu failed=%llu active=%d queued=%d max_active=%d\n",
            sj.size(), targets, ok, bad,
            (unsigned long long)(after.completed - before.completed),
            (unsigned long long)(after.cancelled - before.cancelled),
            (unsigned long long)(after.failed - before.failed), after.active, after.queued,
            after.max_active);
        if (bad || after.active != 0 || after.queued != 0 || after.max_active > slots) {
            std::fprintf(stderr, "FAIL: stress\n");
            ++failures;
        }
    }
    engine.stop();
    if (failures) {
        std::fprintf(stderr, "FAIL: %d batch-engine check(s)\n", failures);
        return 1;
    }
    std::printf("batch engine OK\n");
    return 0;
}
