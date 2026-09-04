// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "runtime.h"

#include <unicode/utf8.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "codec.h"
#include "frontend.h"
#include "runtime/ggml/runtime.h"

namespace nemo_speech::tts::omnivoice {
namespace {

size_t
unicode_length(const std::string& value) {
    size_t count = 0;
    int32_t offset = 0;
    while (offset < static_cast<int32_t>(value.size())) {
        UChar32 codepoint = 0;
        U8_NEXT(value.data(), offset, static_cast<int32_t>(value.size()), codepoint);
        if (codepoint < 0)
            throw std::invalid_argument("OmniVoice text is not valid UTF-8");
        ++count;
    }
    return count;
}

void
validate_config(const RuntimeConfig& config) {
    if (config.generation.steps <= 0 || !std::isfinite(config.generation.guidance_scale) ||
        !std::isfinite(config.generation.time_shift) || config.generation.time_shift <= 0.0f ||
        !std::isfinite(config.generation.layer_penalty) ||
        !std::isfinite(config.generation.position_temperature) ||
        config.generation.position_temperature < 0.0f ||
        !std::isfinite(config.generation.class_temperature) ||
        config.generation.class_temperature < 0.0f ||
        !std::isfinite(config.audio_chunk_duration_s) || config.audio_chunk_duration_s <= 0.0 ||
        !std::isfinite(config.audio_chunk_threshold_s) || config.audio_chunk_threshold_s <= 0.0 ||
        !std::isfinite(config.pad_duration_s) || config.pad_duration_s < 0.0 ||
        !std::isfinite(config.fade_duration_s) || config.fade_duration_s < 0.0) {
        throw std::invalid_argument("invalid OmniVoice runtime configuration");
    }
}

void
fade_prefix(std::vector<float>& audio, size_t count) {
    count = std::min(count, audio.size());
    for (size_t i = 0; i < count; ++i) {
        const float weight = count == 1 ? 0.0f : static_cast<float>(i) / (count - 1);
        audio[i] *= weight;
    }
}

void
fade_suffix(std::vector<float>& audio, size_t count) {
    count = std::min(count, audio.size());
    for (size_t i = 0; i < count; ++i) {
        const float weight = count == 1 ? 0.0f : 1.0f - static_cast<float>(i) / (count - 1);
        audio[audio.size() - count + i] *= weight;
    }
}

std::vector<float>
clean_chunk(
    std::vector<float> audio, const std::optional<float>& reference_rms, bool remove_long_silence) {
    if (remove_long_silence)
        audio = remove_silence(audio, 24000, 500, 100, 100);
    if (reference_rms && *reference_rms < 0.1f) {
        const float scale = *reference_rms / 0.1f;
        for (float& sample : audio) sample *= scale;
    } else if (!reference_rms) {
        float peak = 0.0f;
        for (float sample : audio) peak = std::max(peak, std::abs(sample));
        if (peak > 1.0e-6f) {
            const float scale = 0.5f / peak;
            for (float& sample : audio) sample *= scale;
        }
    }
    return audio;
}

uint64_t
chunk_seed(uint64_t seed, size_t index) {
    uint64_t value = seed + 0x9e3779b97f4a7c15ULL * index;
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

enum class Utf8State { Valid, Incomplete, Invalid };

Utf8State
utf8_state(const std::string& value) {
    size_t i = 0;
    while (i < value.size()) {
        const uint8_t lead = static_cast<uint8_t>(value[i]);
        if (lead <= 0x7f) {
            ++i;
            continue;
        }
        size_t continuation = 0;
        uint32_t codepoint = 0;
        if (lead >= 0xc2 && lead <= 0xdf) {
            continuation = 1;
            codepoint = lead & 0x1f;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            continuation = 2;
            codepoint = lead & 0x0f;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            continuation = 3;
            codepoint = lead & 0x07;
        } else {
            return Utf8State::Invalid;
        }
        if (value.size() - i - 1 < continuation)
            return Utf8State::Incomplete;
        for (size_t j = 1; j <= continuation; ++j) {
            const uint8_t byte = static_cast<uint8_t>(value[i + j]);
            if ((byte & 0xc0) != 0x80)
                return Utf8State::Invalid;
            codepoint = (codepoint << 6) | (byte & 0x3f);
        }
        if ((continuation == 2 && codepoint < 0x800) ||
            (continuation == 3 && codepoint < 0x10000) ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff)
            return Utf8State::Invalid;
        i += continuation + 1;
    }
    return Utf8State::Valid;
}

}  // namespace

class Runtime::Impl {
   public:
    Impl(
        const std::string& model_path, const std::string& audio_tokenizer_path, bool use_gpu,
        int32_t gpu_device)
        : backends_([&] {
              ggml_runtime::Params params;
              params.use_gpu = use_gpu;
              params.gpu_device_idx = gpu_device;
              return params;
          }()),
          generator_(backends_, model_path), decoder_(backends_, audio_tokenizer_path),
          audio_tokenizer_path_(audio_tokenizer_path) {}

    CodecEncoder& encoder() {
        if (!encoder_)
            encoder_ = std::make_unique<CodecEncoder>(backends_, audio_tokenizer_path_);
        return *encoder_;
    }

    ggml_runtime::BackendManager backends_;
    Generator generator_;
    CodecDecoder decoder_;
    std::string audio_tokenizer_path_;
    std::unique_ptr<CodecEncoder> encoder_;
    std::mutex request_mutex_;
};

Runtime::Runtime(
    const std::string& model_path, const std::string& audio_tokenizer_path, bool use_gpu,
    int32_t gpu_device) {
    if (model_path.empty() || audio_tokenizer_path.empty())
        throw std::invalid_argument("both OmniVoice GGUF paths are required");
    impl_ = std::make_unique<Impl>(model_path, audio_tokenizer_path, use_gpu, gpu_device);
    const auto& model = impl_->generator_.config();
    const auto& codec = impl_->decoder_.config();
    if (model.source_revision != codec.source_revision ||
        model.source_audio_tokenizer_sha256 != codec.source_model_sha256) {
        throw std::invalid_argument(
            "OmniVoice denoiser and audio tokenizer fingerprints do not match");
    }
    fingerprint_ = {
        model.source_revision, model.source_model_sha256, model.source_tokenizer_sha256,
        model.source_audio_tokenizer_sha256};
}

Runtime::~Runtime() = default;

RuntimeSynthesisResult
Runtime::synthesize(
    const RuntimeSynthesisRequest& request, const RuntimeConfig& config,
    const PcmCallback& callback) {
    validate_config(config);
    if (config.generation.cancelled && config.generation.cancelled())
        throw GenerationCancelled();
    if (request.text.empty())
        throw std::invalid_argument("OmniVoice text must not be empty");
    if (request.voice_prompt && request.instruction)
        throw std::invalid_argument(
            "voice prompt and voice-design instruction are mutually exclusive");
    if (request.speed && (!std::isfinite(*request.speed) || *request.speed <= 0.0))
        throw std::invalid_argument("OmniVoice speed must be positive and finite");
    if (request.fixed_duration_seconds &&
        (!std::isfinite(*request.fixed_duration_seconds) || *request.fixed_duration_seconds <= 0.0))
        throw std::invalid_argument("OmniVoice duration must be positive and finite");
    if (request.target_frames && *request.target_frames <= 0)
        throw std::invalid_argument("OmniVoice target frame override must be positive");
    if (request.voice_prompt)
        validate_voice_prompt(*request.voice_prompt, &fingerprint_);

    std::lock_guard<std::mutex> lock(impl_->request_mutex_);
    if (config.generation.cancelled && config.generation.cancelled())
        throw GenerationCancelled();
    const auto start = std::chrono::steady_clock::now();
    TokenGenerationRequest estimate_request;
    estimate_request.text = request.text;
    estimate_request.language = request.language;
    estimate_request.instruction = request.instruction;
    estimate_request.speed = request.speed;
    estimate_request.fixed_duration_seconds = request.fixed_duration_seconds;
    estimate_request.target_frames = request.target_frames;
    if (request.voice_prompt) {
        estimate_request.reference_text = request.voice_prompt->transcript;
        estimate_request.reference_audio_codes = request.voice_prompt->audio_codes;
    }
    const int32_t total_frames = impl_->generator_.estimate_target_frames(estimate_request);

    std::vector<std::string> chunks = {request.text};
    if (!request.target_frames &&
        total_frames > static_cast<int32_t>(config.audio_chunk_threshold_s * 25.0)) {
        const size_t characters = unicode_length(request.text);
        if (characters == 0)
            throw std::invalid_argument("OmniVoice text is empty after decoding");
        const double frames_per_character = static_cast<double>(total_frames) / characters;
        const int32_t chunk_characters = std::max<int32_t>(
            1, static_cast<int32_t>(config.audio_chunk_duration_s * 25.0 / frames_per_character));
        chunks = chunk_text_punctuation(request.text, chunk_characters, 3);
        if (chunks.empty())
            throw std::invalid_argument("OmniVoice chunker produced no text");
    }

    std::optional<double> chunk_speed = request.speed;
    if (request.fixed_duration_seconds && chunks.size() > 1) {
        const int32_t requested_frames =
            std::max(1, static_cast<int32_t>(*request.fixed_duration_seconds * 25.0));
        chunk_speed = static_cast<double>(total_frames) / requested_frames;
    }

    RuntimeSynthesisResult result;
    std::array<std::vector<int32_t>, 8> automatic_reference;
    std::string automatic_reference_text;
    uint64_t base_seed = 0;
    std::optional<float> reference_rms =
        request.voice_prompt ? std::optional<float>(request.voice_prompt->reference_rms)
                             : std::nullopt;
    const size_t boundary_fade = 2400;
    const size_t boundary_silence = 2400;
    const size_t global_fade = static_cast<size_t>(config.fade_duration_s * 24000.0);
    const size_t global_pad = static_cast<size_t>(config.pad_duration_s * 24000.0);

    auto emit = [&](const std::vector<float>& samples) {
        result.stats.output_samples += samples.size();
        if (!callback) {
            result.pcm_24khz.insert(result.pcm_24khz.end(), samples.begin(), samples.end());
            return true;
        }
        if (samples.empty())
            return true;
        if (!callback(samples.data(), samples.size())) {
            result.stats.cancelled = true;
            return false;
        }
        return true;
    };

    for (size_t index = 0; index < chunks.size(); ++index) {
        TokenGenerationRequest generation_request;
        generation_request.text = chunks[index];
        generation_request.language = request.language;
        generation_request.instruction = request.instruction;
        generation_request.speed = chunks.size() > 1 ? chunk_speed : request.speed;
        if (chunks.size() == 1) {
            generation_request.fixed_duration_seconds = request.fixed_duration_seconds;
            generation_request.target_frames = request.target_frames;
        }
        if (request.voice_prompt) {
            generation_request.reference_text = request.voice_prompt->transcript;
            generation_request.reference_audio_codes = request.voice_prompt->audio_codes;
        } else if (index > 0 && !request.instruction) {
            generation_request.reference_text = automatic_reference_text;
            generation_request.reference_audio_codes = automatic_reference;
        }

        GenerationConfig generation_config = config.generation;
        if (index > 0) {
            if (generation_config.seed < 0)
                generation_config.seed = static_cast<int64_t>(chunk_seed(base_seed, index));
            else
                generation_config.seed = static_cast<int64_t>(
                    chunk_seed(static_cast<uint64_t>(generation_config.seed), index));
        }
        GeneratedAudioCodes generated =
            impl_->generator_.generate(generation_request, generation_config);
        if (index == 0) {
            result.first_chunk_codes = generated.codebooks;
            automatic_reference = generated.codebooks;
            automatic_reference_text = chunks[0];
            base_seed = generated.effective_seed;
            result.stats.effective_seed = generated.effective_seed;
        }
        result.stats.generated_frames += static_cast<int32_t>(generated.codebooks[0].size());
        if (generation_config.cancelled && generation_config.cancelled())
            throw GenerationCancelled();
        std::vector<float> audio = clean_chunk(
            impl_->decoder_.decode(generated.codebooks), reference_rms, config.postprocess_output);
        if (generation_config.cancelled && generation_config.cancelled())
            throw GenerationCancelled();

        if (index > 0)
            fade_prefix(audio, boundary_fade);
        if (index + 1 < chunks.size())
            fade_suffix(audio, boundary_fade);
        if (index == 0 && global_fade > 0) {
            const size_t count = std::min(global_fade, audio.size() / 2);
            for (size_t i = 0; i < count; ++i)
                audio[i] *= count == 1 ? 0.0f : static_cast<float>(i) / (count - 1);
        }
        if (index + 1 == chunks.size() && global_fade > 0)
            fade_suffix(audio, global_fade);

        if (index == 0 && global_pad > 0 && !emit(std::vector<float>(global_pad, 0.0f)))
            break;
        if (!emit(audio))
            break;
        if (index + 1 < chunks.size() && !emit(std::vector<float>(boundary_silence, 0.0f)))
            break;
        if (index + 1 == chunks.size() && global_pad > 0)
            if (!emit(std::vector<float>(global_pad, 0.0f)))
                break;
        ++result.stats.chunks;
    }
    result.stats.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

VoicePrompt
Runtime::create_prompt(
    const float* interleaved_pcm, size_t frames, int32_t channels, int32_t sample_rate,
    const std::string& transcript, bool preprocess) {
    std::lock_guard<std::mutex> lock(impl_->request_mutex_);
    return create_voice_prompt(
        impl_->encoder(), interleaved_pcm, frames, channels, sample_rate, transcript, fingerprint_,
        preprocess);
}

const std::vector<std::string>&
Runtime::language_ids() const {
    return impl_->generator_.language_ids();
}

const std::vector<std::string>&
Runtime::language_names() const {
    return impl_->generator_.language_names();
}

class BidirectionalStream::Impl {
   public:
    Impl(
        Runtime& runtime, RuntimeSynthesisRequest immutable_request, RuntimeConfig config,
        Runtime::PcmCallback callback, StreamLimits limits)
        : runtime_(runtime), request_(std::move(immutable_request)), config_(config),
          callback_(std::move(callback)), limits_(limits) {
        validate_config(config_);
        if (!callback_)
            throw std::invalid_argument("OmniVoice stream callback is required");
        if (limits_.maximum_pending_bytes == 0 || limits_.maximum_queued_segments == 0)
            throw std::invalid_argument("OmniVoice stream limits must be positive");
        if (!request_.text.empty())
            throw std::invalid_argument("stream request text must start empty");
        if (request_.target_frames)
            throw std::invalid_argument(
                "fixed target frames are not valid for bidirectional streams");
        if (request_.voice_prompt) {
            fixed_prompt_ = *request_.voice_prompt;
            request_.voice_prompt = &*fixed_prompt_;
        }
        worker_ = std::thread([this] { work(); });
    }

    ~Impl() {
        cancel();
        join();
    }

    void push(const char* bytes, size_t count, bool explicit_commit) {
        if (!bytes && count != 0)
            throw std::invalid_argument("OmniVoice stream text pointer is null");
        std::unique_lock<std::mutex> lock(mutex_);
        rethrow_worker_error();
        if (finished_ || finishing_ || cancelled_)
            throw std::logic_error("OmniVoice stream no longer accepts text");
        if (count > limits_.maximum_pending_bytes -
                        std::min(pending_.size(), limits_.maximum_pending_bytes))
            throw std::invalid_argument("OmniVoice stream pending-text limit exceeded");
        const size_t previous_size = pending_.size();
        if (count != 0)
            pending_.append(bytes, count);
        const Utf8State state = utf8_state(pending_);
        if (state == Utf8State::Invalid) {
            pending_.resize(previous_size);
            throw std::invalid_argument("OmniVoice stream contains invalid UTF-8");
        }
        if (state == Utf8State::Incomplete) {
            if (explicit_commit)
                throw std::invalid_argument("cannot commit an incomplete UTF-8 code point");
            return;
        }
        enqueue_boundaries(lock);
        if (explicit_commit && !pending_.empty()) {
            enqueue(lock, pending_);
            pending_.clear();
        }
    }

    RuntimeStats finish() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            rethrow_worker_error();
            if (!cancelled_ && !finishing_ && !finished_) {
                if (utf8_state(pending_) == Utf8State::Incomplete)
                    throw std::invalid_argument("OmniVoice stream ends with incomplete UTF-8");
                enqueue_boundaries(lock);
                if (!pending_.empty()) {
                    enqueue(lock, std::move(pending_));
                    pending_.clear();
                }
                finishing_ = true;
                condition_.notify_all();
            }
        }
        join();
        std::lock_guard<std::mutex> lock(mutex_);
        rethrow_worker_error();
        return stats_;
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (finished_)
            return;
        cancelled_ = true;
        queue_.clear();
        stats_.cancelled = true;
        condition_.notify_all();
    }

    RuntimeStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

   private:
    void enqueue_boundaries(std::unique_lock<std::mutex>& lock) {
        while (true) {
            const size_t boundary = first_commit_boundary(pending_);
            if (boundary == 0)
                break;
            std::string segment = pending_.substr(0, boundary);
            enqueue(lock, std::move(segment));
            pending_.erase(0, boundary);
        }
    }

    void enqueue(std::unique_lock<std::mutex>&, std::string segment) {
        if (segment.find_first_not_of(" \t\r\n") == std::string::npos)
            return;
        if (queue_.size() >= limits_.maximum_queued_segments)
            throw std::runtime_error("OmniVoice stream committed-segment queue is full");
        queue_.push_back(std::move(segment));
        condition_.notify_one();
    }

    bool emit(const std::vector<float>& audio) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (cancelled_)
                return false;
        }
        if (audio.empty())
            return true;
        if (!callback_(audio.data(), audio.size())) {
            cancel();
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.output_samples += audio.size();
        return true;
    }

    bool stitch(std::vector<float> audio) {
        constexpr size_t boundary = 2400;
        if (!started_) {
            started_ = true;
            const size_t fade = static_cast<size_t>(config_.fade_duration_s * 24000.0);
            fade_prefix(audio, fade);
            const size_t pad = static_cast<size_t>(config_.pad_duration_s * 24000.0);
            if (pad > 0 && !emit(std::vector<float>(pad, 0.0f)))
                return false;
        } else {
            fade_suffix(held_tail_, boundary);
            if (!emit(held_tail_) || !emit(std::vector<float>(boundary, 0.0f)))
                return false;
            fade_prefix(audio, boundary);
        }
        const size_t held = std::min(boundary, audio.size());
        if (audio.size() > held) {
            std::vector<float> prefix(audio.begin(), audio.end() - held);
            if (!emit(prefix))
                return false;
        }
        held_tail_.assign(audio.end() - held, audio.end());
        return true;
    }

    void flush_tail() {
        if (!started_)
            return;
        const size_t fade = static_cast<size_t>(config_.fade_duration_s * 24000.0);
        fade_suffix(held_tail_, fade);
        if (!emit(held_tail_))
            return;
        const size_t pad = static_cast<size_t>(config_.pad_duration_s * 24000.0);
        if (pad > 0)
            emit(std::vector<float>(pad, 0.0f));
        held_tail_.clear();
    }

    void work() noexcept {
        try {
            uint64_t base_seed = 0;
            size_t segment_index = 0;
            while (true) {
                std::string segment;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    condition_.wait(
                        lock, [&] { return cancelled_ || !queue_.empty() || finishing_; });
                    if (cancelled_)
                        break;
                    if (queue_.empty()) {
                        if (finishing_)
                            break;
                        continue;
                    }
                    segment = std::move(queue_.front());
                    queue_.pop_front();
                }
                RuntimeSynthesisRequest item = request_;
                item.text = segment;
                const bool automatic_continuation = automatic_prompt_.has_value();
                if (automatic_continuation)
                    item.voice_prompt = &*automatic_prompt_;
                RuntimeConfig item_config = config_;
                item_config.generation.cancelled = [this] {
                    std::lock_guard<std::mutex> lock(mutex_);
                    return cancelled_;
                };
                item_config.pad_duration_s = 0.0;
                item_config.fade_duration_s = 0.0;
                // The synthesized first chunk is a token-conditioning prompt,
                // not an external loudness reference. Keep later automatic
                // chunks on the same no-reference post-processing path as
                // Runtime's complete-text long-form implementation.
                if (automatic_continuation)
                    item_config.postprocess_output = false;
                if (segment_index > 0) {
                    const uint64_t root = config_.generation.seed < 0
                                              ? base_seed
                                              : static_cast<uint64_t>(config_.generation.seed);
                    item_config.generation.seed =
                        static_cast<int64_t>(chunk_seed(root, segment_index));
                }
                RuntimeSynthesisResult generated = runtime_.synthesize(item, item_config);
                if (automatic_continuation) {
                    generated.pcm_24khz = clean_chunk(
                        std::move(generated.pcm_24khz), std::nullopt, config_.postprocess_output);
                }
                if (segment_index == 0) {
                    base_seed = generated.stats.effective_seed;
                    std::lock_guard<std::mutex> lock(mutex_);
                    stats_.effective_seed = base_seed;
                }
                if (!request_.voice_prompt && !request_.instruction && !automatic_prompt_) {
                    VoicePrompt prompt;
                    prompt.audio_codes = generated.first_chunk_codes;
                    prompt.transcript = segment;
                    prompt.reference_rms = 0.1f;
                    prompt.fingerprint = runtime_.fingerprint();
                    automatic_prompt_ = std::move(prompt);
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stats_.generated_frames += generated.stats.generated_frames;
                    stats_.elapsed_seconds += generated.stats.elapsed_seconds;
                    stats_.chunks += generated.stats.chunks;
                }
                if (!stitch(std::move(generated.pcm_24khz)))
                    break;
                ++segment_index;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!cancelled_)
                    flush_requested_ = true;
            }
            if (flush_requested_)
                flush_tail();
        }
        catch (const GenerationCancelled&) {
            std::lock_guard<std::mutex> lock(mutex_);
            cancelled_ = true;
            stats_.cancelled = true;
        }
        catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            worker_error_ = std::current_exception();
            cancelled_ = true;
            stats_.cancelled = true;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        condition_.notify_all();
    }

    void join() {
        if (worker_.joinable())
            worker_.join();
    }

    void rethrow_worker_error() const {
        if (worker_error_)
            std::rethrow_exception(worker_error_);
    }

    Runtime& runtime_;
    RuntimeSynthesisRequest request_;
    RuntimeConfig config_;
    Runtime::PcmCallback callback_;
    StreamLimits limits_;
    std::optional<VoicePrompt> fixed_prompt_;
    std::optional<VoicePrompt> automatic_prompt_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::string> queue_;
    std::string pending_;
    std::vector<float> held_tail_;
    std::thread worker_;
    RuntimeStats stats_;
    std::exception_ptr worker_error_;
    bool finishing_ = false;
    bool finished_ = false;
    bool cancelled_ = false;
    bool started_ = false;
    bool flush_requested_ = false;
};

BidirectionalStream::BidirectionalStream(
    Runtime& runtime, RuntimeSynthesisRequest immutable_request, RuntimeConfig config,
    Runtime::PcmCallback callback, StreamLimits limits)
    : impl_(std::make_unique<Impl>(
          runtime, std::move(immutable_request), config, std::move(callback), limits)) {}

BidirectionalStream::~BidirectionalStream() = default;

void
BidirectionalStream::push_text(const char* bytes, size_t count, bool explicit_commit) {
    impl_->push(bytes, count, explicit_commit);
}

RuntimeStats
BidirectionalStream::finish() {
    return impl_->finish();
}

void
BidirectionalStream::cancel() {
    impl_->cancel();
}

RuntimeStats
BidirectionalStream::stats() const {
    return impl_->stats();
}

}  // namespace nemo_speech::tts::omnivoice
