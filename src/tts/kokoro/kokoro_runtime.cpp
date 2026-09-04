// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "kokoro_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "acoustic.h"
#include "decoder.h"
#include "duration.h"
#include "nvtx_utils.h"
#include "plbert.h"
#include "prosody.h"
#include "vocoder.h"

namespace nemo_speech::tts::kokoro {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t
splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::string
pcm_s16le(const std::vector<float>& waveform) {
    const ggml_nvtx::range nvtx_range("kokoro.output.float_to_pcm16");
    std::string pcm(waveform.size() * sizeof(int16_t), '\0');
    for (size_t index = 0; index < waveform.size(); ++index) {
        if (!std::isfinite(waveform[index])) {
            throw std::runtime_error("Kokoro generated a non-finite PCM sample");
        }
        const float clipped = std::clamp(waveform[index], -1.0f, 1.0f);
        const int value = static_cast<int>(std::lrint(clipped * 32767.0f));
        const uint16_t bits = static_cast<uint16_t>(static_cast<int16_t>(value));
        pcm[2 * index] = static_cast<char>(bits & 0xffU);
        pcm[2 * index + 1] = static_cast<char>(bits >> 8);
    }
    return pcm;
}

}  // namespace

class KokoroRuntime::Impl {
   public:
    explicit Impl(KokoroRuntimeConfig raw)
        : config(std::move(raw)), frontend(required_path(config.model_path)),
          plbert(config.model_path, config.use_gpu), duration(config.model_path, config.use_gpu),
          acoustic(config.model_path, config.use_gpu), prosody(config.model_path, config.use_gpu),
          decoder(config.model_path, config.use_gpu), vocoder(config.model_path, config.use_gpu) {
        validate_speed(config.speed);
    }

    static const std::string& required_path(const std::string& path) {
        if (path.empty())
            throw std::invalid_argument("Kokoro model path is required");
        return path;
    }

    static void validate_speed(float speed) {
        if (!std::isfinite(speed) || speed < 0.5f || speed > 2.0f) {
            throw std::invalid_argument("Kokoro speed must be finite and in [0.5,2.0]");
        }
    }

    bool synthesize_ids(
        const std::vector<int32_t>& unframed_ids, const std::string& voice, float speed,
        int64_t seed, uint64_t& leaf_index, const KokoroRuntime::PcmCallback& callback,
        KokoroRuntimeStats& stats) {
        if (unframed_ids.empty())
            return true;
        const std::string nvtx_name =
            "kokoro.runtime.inference_leaf tokens=" + std::to_string(unframed_ids.size()) +
            " leaf=" + std::to_string(leaf_index);
        const ggml_nvtx::range nvtx_range(nvtx_name.c_str());

        std::vector<int32_t> framed;
        framed.reserve(unframed_ids.size() + 2);
        framed.push_back(0);
        framed.insert(framed.end(), unframed_ids.begin(), unframed_ids.end());
        framed.push_back(0);

        std::vector<float> voice_style;
        {
            const ggml_nvtx::range phase("kokoro.runtime.voice_style");
            voice_style = frontend.voice_style(voice, unframed_ids.size());
        }
        const std::vector<float> decoder_style(voice_style.begin(), voice_style.begin() + 128);
        const std::vector<float> duration_style(voice_style.begin() + 128, voice_style.end());
        const std::vector<float> projected = plbert.encode(framed);
        const KokoroDurationResult timing =
            duration.predict(projected, framed.size(), duration_style, speed);

        size_t frame_count = 0;
        for (int32_t value : timing.frames) frame_count += static_cast<size_t>(value);
        const std::string leaf_parameters =
            "kokoro.parameters.leaf tokens=" + std::to_string(unframed_ids.size()) +
            " framed_tokens=" + std::to_string(framed.size()) +
            " acoustic_frames=" + std::to_string(frame_count) +
            " latent_frames=" + std::to_string(frame_count * 2) + " voice=" + voice +
            " speed=" + std::to_string(speed) + " seed=" + std::to_string(seed);
        ggml_nvtx::mark(leaf_parameters.c_str());
        // One decoder frame becomes two generator input frames. Keep each
        // internal leaf within the generator's fixed 1,064-frame exact AdaIN
        // window; recursive token splitting preserves the established
        // deterministic seed order and bounds decoder/vocoder scratch.
        if (frame_count > 532) {
            const ggml_nvtx::range split_range("kokoro.runtime.split_leaf");
            if (unframed_ids.size() == 1) {
                throw std::length_error("one Kokoro phoneme exceeds the inference frame limit");
            }
            const size_t middle = unframed_ids.size() / 2;
            const std::vector<int32_t> left(unframed_ids.begin(), unframed_ids.begin() + middle);
            const std::vector<int32_t> right(unframed_ids.begin() + middle, unframed_ids.end());
            return synthesize_ids(left, voice, speed, seed, leaf_index, callback, stats) &&
                   synthesize_ids(right, voice, speed, seed, leaf_index, callback, stats);
        }

        const KokoroAcousticFeatures features =
            acoustic.encode(framed, timing.frames, timing.encoded_features);
        const KokoroProsody predicted =
            prosody.predict(features.prosody_shared, features.frame_count, duration_style);
        int64_t chunk_seed = seed;
        if (seed >= 0) {
            const uint64_t current_leaf = leaf_index++;
            chunk_seed = static_cast<int64_t>(
                splitmix64(static_cast<uint64_t>(seed) ^ splitmix64(current_leaf)) &
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
        }
        stats.generated_frames += static_cast<int>(predicted.f0.size());
        ++stats.chunks;
        constexpr size_t kPcmTileSamples = 12000;  // 500 ms at native 24 kHz.
        const KokoroVocoderStreamStats streamed = vocoder.synthesize_stream(
            [&](size_t begin, size_t end) {
                return decoder.encode_range(
                    features.text, features.frame_count, predicted.f0, predicted.noise,
                    decoder_style, begin, end);
            },
            predicted.f0.size(), predicted.f0, decoder_style, chunk_seed, kPcmTileSamples,
            [&](const std::vector<float>& waveform) {
                const ggml_nvtx::range callback_range("kokoro.runtime.pcm_callback");
                return !callback || callback(pcm_s16le(waveform));
            });
        stats.samples_written += streamed.samples_written;
        if (streamed.cancelled) {
            stats.cancelled = true;
            return false;
        }
        return true;
    }

    KokoroRuntimeConfig config;
    KokoroFrontend frontend;
    KokoroPlbertEncoder plbert;
    KokoroDurationPredictor duration;
    KokoroAcousticEncoder acoustic;
    KokoroProsodyHeads prosody;
    KokoroDecoderEncoder decoder;
    KokoroVocoder vocoder;
    std::mutex mutex;
};

KokoroRuntime::KokoroRuntime(KokoroRuntimeConfig config) {
    const ggml_nvtx::range nvtx_range("kokoro.model_load.total");
    const std::string nvtx_parameters =
        "kokoro.parameters.model_load backend=" + std::string(config.use_gpu ? "cuda" : "cpu") +
        " model=" + config.model_path + " seed=" + std::to_string(config.seed) +
        " speed=" + std::to_string(config.speed);
    ggml_nvtx::mark(nvtx_parameters.c_str());
    impl_ = std::make_unique<Impl>(std::move(config));
}

KokoroRuntime::~KokoroRuntime() = default;

KokoroPreparedText
KokoroRuntime::prepare(
    const std::string& text, const std::string& language, const std::string& voice) const {
    return impl_->frontend.prepare(text, language, voice);
}

KokoroChunk
KokoroRuntime::prepare_tokens(
    const std::vector<int32_t>& ids, const std::string& language, const std::string& voice) const {
    return impl_->frontend.prepare_tokens(ids, language, voice);
}

KokoroRuntimeStats
KokoroRuntime::synthesize(
    const std::vector<KokoroChunk>& chunks, const std::string& voice, float speed, int64_t seed,
    const PcmCallback& callback) {
    if (chunks.empty())
        throw std::invalid_argument("Kokoro synthesis requires a phoneme chunk");
    if (voice.empty())
        throw std::invalid_argument("Kokoro synthesis requires a voice");
    if (speed == 0.0f)
        speed = impl_->config.speed;
    if (seed == -1)
        seed = impl_->config.seed;
    Impl::validate_speed(speed);
    const ggml_nvtx::range nvtx_range("kokoro.runtime.request");
    // Validate/canonicalize before taking the compute lock, including checking
    // that this GGUF actually contains the requested voice.
    const std::string canonical_voice = KokoroTokenizer::canonicalize_voice(voice);
    (void)impl_->frontend.metadata().voice_index(canonical_voice);
    size_t requested_tokens = 0;
    for (const KokoroChunk& chunk : chunks) requested_tokens += chunk.ids.size();
    const std::string request_parameters =
        "kokoro.parameters.request backend=" + std::string(impl_->config.use_gpu ? "cuda" : "cpu") +
        " chunks=" + std::to_string(chunks.size()) + " tokens=" + std::to_string(requested_tokens) +
        " voice=" + canonical_voice + " speed=" + std::to_string(speed) +
        " seed=" + std::to_string(seed) + " sample_rate=" + std::to_string(sample_rate());
    ggml_nvtx::mark(request_parameters.c_str());

    std::unique_lock<std::mutex> lock;
    {
        const ggml_nvtx::range lock_range("kokoro.runtime.lock_wait");
        lock = std::unique_lock<std::mutex>(impl_->mutex);
    }
    const Clock::time_point started = Clock::now();
    KokoroRuntimeStats stats;
    Clock::time_point previous_emission;
    size_t emission_count = 0;
    double interval_total_ms = 0.0;
    const PcmCallback measured_callback = [&](const std::string& pcm) {
        const ggml_nvtx::range callback_range("kokoro.runtime.measured_callback");
        const Clock::time_point now = Clock::now();
        if (emission_count == 0) {
            stats.ttfa_ms = std::chrono::duration<double, std::milli>(now - started).count();
        } else {
            const double interval =
                std::chrono::duration<double, std::milli>(now - previous_emission).count();
            interval_total_ms += interval;
            if (emission_count == 1) {
                stats.icl_min_ms = interval;
                stats.icl_max_ms = interval;
            } else {
                stats.icl_min_ms = std::min(stats.icl_min_ms, interval);
                stats.icl_max_ms = std::max(stats.icl_max_ms, interval);
            }
        }
        previous_emission = now;
        ++emission_count;
        return !callback || callback(pcm);
    };
    uint64_t leaf_index = 0;
    for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        const KokoroChunk& chunk = chunks[chunk_index];
        const std::string chunk_name = "kokoro.runtime.chunk index=" + std::to_string(chunk_index) +
                                       " tokens=" + std::to_string(chunk.ids.size());
        const ggml_nvtx::range chunk_range(chunk_name.c_str());
        if (chunk.ids.empty())
            throw std::invalid_argument("Kokoro synthesis chunk is empty");
        if (!impl_->synthesize_ids(
                chunk.ids, canonical_voice, speed, seed, leaf_index, measured_callback, stats)) {
            break;
        }
    }
    if (emission_count > 1) {
        stats.icl_avg_ms = interval_total_ms / static_cast<double>(emission_count - 1);
    }
    stats.elapsed_s = std::chrono::duration<double>(Clock::now() - started).count();
    stats.audio_s = static_cast<double>(stats.samples_written) / stats.sample_rate;
    stats.rtf = stats.audio_s > 0.0 ? stats.elapsed_s / stats.audio_s : 0.0;
    stats.rtfx = stats.elapsed_s > 0.0 ? stats.audio_s / stats.elapsed_s : 0.0;
    return stats;
}

int
KokoroRuntime::sample_rate() const {
    return impl_->frontend.metadata().hparams().sample_rate;
}

const std::vector<std::string>&
KokoroRuntime::voice_names() const {
    return impl_->frontend.metadata().voice_names();
}

const std::vector<std::string>&
KokoroRuntime::voice_languages() const {
    return impl_->frontend.metadata().voice_languages();
}

}  // namespace nemo_speech::tts::kokoro
