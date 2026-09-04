// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "synthesizer.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "audio_resampler.h"
#include "tts/preproc/text_normalizer.h"
#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
#include "omnivoice/runtime.h"
#endif

namespace nemo_speech::tts {
namespace {

std::string
lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

template <typename T>
std::string
join(const std::vector<T>& values, const char* separator) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0)
            out << separator;
        out << values[i];
    }
    return out.str();
}

void
add_preparation_timing(MagpieSynthesisStats& stats, const PreparedSynthesis& request) {
    stats.tokenizer_ms = request.tokenizer_ms;
    const double preparation_ms = request.normalizer_ms + request.tokenizer_ms;
    stats.elapsed_s += preparation_ms / 1000.0;
    stats.rtf = stats.audio_s > 0.0 ? stats.elapsed_s / stats.audio_s : 0.0;
    stats.rtfx = stats.elapsed_s > 0.0 ? stats.audio_s / stats.elapsed_s : 0.0;
    stats.e2e_ttfa_ms += preparation_ms;
    stats.ttfa_ms = stats.e2e_ttfa_ms;
    stats.e2e_rtfx = stats.rtfx;
}

#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
omnivoice::RuntimeConfig
to_runtime_config(const OmniVoiceOptions& options, const MagpieSynthesisOptions& legacy) {
    omnivoice::RuntimeConfig config;
    config.generation.steps = options.num_steps;
    config.generation.guidance_scale = options.guidance_scale;
    config.generation.time_shift = options.time_shift;
    config.generation.layer_penalty = options.layer_penalty;
    config.generation.position_temperature = options.position_temperature;
    config.generation.class_temperature = options.class_temperature;
    config.generation.denoise = options.denoise;
    config.generation.seed = legacy.seed;
    config.postprocess_output = options.postprocess_output;
    config.audio_chunk_duration_s = options.audio_chunk_duration_s;
    config.audio_chunk_threshold_s = options.audio_chunk_threshold_s;
    config.pad_duration_s = options.pad_duration_s;
    config.fade_duration_s = options.fade_duration_s;
    return config;
}

std::string
float_pcm_to_s16(const std::vector<float>& samples) {
    std::string output(samples.size() * sizeof(int16_t), '\0');
    for (size_t i = 0; i < samples.size(); ++i) {
        const float sample = std::clamp(samples[i], -1.0f, 1.0f);
        const int16_t value = static_cast<int16_t>(std::lrintf(sample * 32767.0f));
        const uint16_t bits = static_cast<uint16_t>(value);
        output[i * 2] = static_cast<char>(bits & 0xff);
        output[i * 2 + 1] = static_cast<char>((bits >> 8) & 0xff);
    }
    return output;
}
#endif

}  // namespace

struct Synthesizer::Impl {
    explicit Impl(SynthesizerConfig config)
        : default_language_code(
              config.default_language_code.empty() ? "en-US"
                                                   : std::move(config.default_language_code)),
          default_voice_name(std::move(config.default_voice_name)),
          default_speaker(config.runtime.speaker), magpie_config(std::move(config.runtime)),
          text_normalizer(std::move(config.text_normalizer_model_dir)) {
        const bool any_omnivoice =
            !config.omnivoice_model.empty() || !config.omnivoice_audio_tokenizer_model.empty();
        const bool complete_omnivoice =
            !config.omnivoice_model.empty() && !config.omnivoice_audio_tokenizer_model.empty();
        const bool any_magpie = !magpie_config.magpie_model.empty() ||
                                !magpie_config.codec_model.empty() ||
                                !config.tokenizer_model_dir.empty();
        if (any_omnivoice) {
            if (!complete_omnivoice)
                throw std::invalid_argument("both OmniVoice model paths are required");
            if (any_magpie)
                throw std::invalid_argument("mixed Magpie and OmniVoice model configuration");
#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
            is_omnivoice = true;
            if (default_language_code == "en-US")
                default_language_code = "en";
            if (default_voice_name.empty())
                default_voice_name = "auto";
            omnivoice_defaults_explicit = config.omnivoice_options.has_value();
            omnivoice_defaults = config.omnivoice_options.value_or(OmniVoiceOptions{});
            if (!omnivoice_defaults_explicit) {
                if (magpie_config.steps > 0)
                    omnivoice_defaults.num_steps = magpie_config.steps;
                if (magpie_config.override_cfg_scale)
                    omnivoice_defaults.guidance_scale = magpie_config.cfg_scale;
            }
            const bool use_gpu = magpie_config.lt_backend != MagpieBackendPreference::Cpu;
            omnivoice_runtime = std::make_unique<omnivoice::Runtime>(
                config.omnivoice_model, config.omnivoice_audio_tokenizer_model, use_gpu);
            speaker_names = {"auto"};
            return;
#else
            throw std::invalid_argument(
                "OmniVoice support was not enabled when nemo-speech was built");
#endif
        }
        if (magpie_config.magpie_model.empty() || magpie_config.codec_model.empty())
            throw std::invalid_argument("both Magpie model paths are required");
        magpie_runtime = std::make_unique<MagpieTtsRuntime>(magpie_config);
        if (!config.tokenizer_model_dir.empty()) {
            tokenizer = std::make_unique<MagpieNativeTokenizer>(
                std::move(config.tokenizer_model_dir), config.tokenizer);
            if (tokenizer->profile_id() != magpie_runtime->tokenizer_profile() ||
                tokenizer->text_vocab_size() != magpie_runtime->text_vocab_size()) {
                throw std::invalid_argument(
                    "Magpie model/tokenizer mismatch: GGUF requires tokenizer profile '" +
                    magpie_runtime->tokenizer_profile() + "' with text vocabulary " +
                    std::to_string(magpie_runtime->text_vocab_size()) +
                    ", but tokenizer directory is '" + tokenizer->profile_id() +
                    "' with text vocabulary " + std::to_string(tokenizer->text_vocab_size()));
            }
        }
    }

    std::string default_language_code;
    std::string default_voice_name;
    int default_speaker = 0;
    bool is_omnivoice = false;
    MagpieRuntimeConfig magpie_config;
    std::unique_ptr<MagpieTtsRuntime> magpie_runtime;
#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
    std::unique_ptr<omnivoice::Runtime> omnivoice_runtime;
#endif
    OmniVoiceOptions omnivoice_defaults;
    bool omnivoice_defaults_explicit = false;
    std::vector<std::string> speaker_names;
    std::unique_ptr<MagpieNativeTokenizer> tokenizer;
    preproc::TextNormalizer text_normalizer;
};

Synthesizer::Synthesizer(SynthesizerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
    if (!impl_->is_omnivoice && !impl_->default_voice_name.empty())
        impl_->default_speaker = resolve_speaker(impl_->default_voice_name);
    if (impl_->default_speaker < 0 || impl_->default_speaker >= speaker_count()) {
        throw std::invalid_argument(
            "default speaker " + std::to_string(impl_->default_speaker) + " must be in [0, " +
            std::to_string(speaker_count() - 1) + "]");
    }
}

Synthesizer::~Synthesizer() = default;

int
Synthesizer::resolve_speaker(const std::string& voice_name) const {
    if (voice_name.empty())
        return impl_->default_speaker;

    if (impl_->is_omnivoice) {
        const std::string value = lower_ascii(voice_name);
        if (value.empty() || value == "auto" || value == "omnivoice.auto" || value == "0")
            return 0;
        throw std::invalid_argument("OmniVoice voice_name must be auto or speaker index 0");
    }
    size_t consumed = 0;
    try {
        const int value = std::stoi(voice_name, &consumed);
        if (consumed == voice_name.size()) {
            if (value >= 0 && value < impl_->magpie_runtime->speaker_count())
                return value;
            throw std::invalid_argument("voice speaker index is outside the model speaker range");
        }
    }
    catch (const std::invalid_argument&) {
        if (consumed == voice_name.size())
            throw;
    }
    catch (const std::out_of_range&) {
        throw std::invalid_argument("voice speaker index is outside the model speaker range");
    }

    std::string wanted = lower_ascii(voice_name);
    const std::string model_prefix = lower_ascii(impl_->magpie_runtime->model_name()) + ".";
    if (wanted.rfind(model_prefix, 0) == 0)
        wanted.erase(0, model_prefix.size());
    const auto& names = impl_->magpie_runtime->speaker_names();
    for (size_t i = 0; i < names.size(); ++i) {
        if (lower_ascii(names[i]) == wanted)
            return static_cast<int>(i);
    }
    throw std::invalid_argument("unknown voice_name '" + voice_name + "'");
}

PreparedSynthesis
Synthesizer::prepare(const SynthesisRequest& request) const {
    if (impl_->is_omnivoice)
        throw std::invalid_argument("prepared token synthesis is not available for OmniVoice");
    if (request.text.empty())
        throw std::invalid_argument("text is required");
    if (!impl_->tokenizer) {
        throw std::invalid_argument("text synthesis requires a tokenizer model directory");
    }
    const int output_rate =
        request.output_sample_rate == 0 ? sample_rate() : request.output_sample_rate;
    if (output_rate < 8000 || output_rate > sample_rate()) {
        throw std::invalid_argument(
            "output sample rate must be between 8000 and the model rate (" +
            std::to_string(sample_rate()) + ") Hz, or be 0 (auto)");
    }

    PreparedSynthesis prepared;
    prepared.metadata.original_text = request.text;
    prepared.metadata.language_code =
        request.language_code.empty() ? impl_->default_language_code : request.language_code;
    prepared.metadata.sample_rate = output_rate;
    prepared.options = request.options;
    if (prepared.options.speaker < 0) {
        prepared.options.speaker = resolve_speaker(
            request.voice_name.empty() ? impl_->default_voice_name : request.voice_name);
    }
    if (prepared.options.speaker < 0 || prepared.options.speaker >= speaker_count())
        throw std::invalid_argument("speaker is outside the model speaker range");
    prepared.metadata.speaker = prepared.options.speaker;

    const auto tokenizer_start = std::chrono::steady_clock::now();
    const auto tokenized = impl_->tokenizer->tokenize(
        request.text, prepared.metadata.language_code,
        [&](const std::string& chunk, bool is_final) {
            const auto normalizer_start = std::chrono::steady_clock::now();
            std::string normalized =
                impl_->text_normalizer.normalize(chunk, prepared.metadata.language_code);
            prepared.normalizer_ms += std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - normalizer_start)
                                          .count();
            if (is_final) {
                normalized =
                    ensure_terminal_punctuation(normalized, prepared.metadata.language_code);
            }
            return normalized;
        });
    const double total_preparation_ms = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - tokenizer_start)
                                            .count();
    prepared.tokenizer_ms = std::max(0.0, total_preparation_ms - prepared.normalizer_ms);
    prepared.metadata.language_code = tokenized.language;
    prepared.metadata.tokenizer_name = tokenized.tokenizer_name;
    prepared.tokens = tokenized.tokens;
    prepared.token_chunks.reserve(tokenized.chunks.size());
    std::vector<std::string> processed_chunks;
    processed_chunks.reserve(tokenized.chunks.size());
    for (const auto& chunk : tokenized.chunks) {
        prepared.token_chunks.push_back(chunk.tokens);
        processed_chunks.push_back(chunk.text);
    }
    prepared.metadata.processed_text = join(processed_chunks, " ");
    if (processed_chunks.empty() && !prepared.tokens.empty())
        prepared.metadata.processed_text = join(prepared.tokens, " ");
    if (prepared.token_chunks.empty() && !prepared.tokens.empty())
        prepared.token_chunks.push_back(prepared.tokens);
    if (prepared.token_chunks.empty())
        throw std::invalid_argument("tokenizer produced no text tokens");
    prepared.metadata.token_count = prepared.tokens.size();
    prepared.metadata.chunk_count = prepared.token_chunks.size();
    return prepared;
}

SynthesisResult
Synthesizer::synthesize(const PreparedSynthesis& request, const PcmCallback& callback) {
    SynthesisResult result;
    result.metadata = request.metadata;
    const int output_rate = request.metadata.sample_rate;
    const bool resample = output_rate != sample_rate();
    std::unique_ptr<audio::Pcm16Resampler> resampler;
    if (resample)
        resampler = std::make_unique<audio::Pcm16Resampler>(sample_rate(), output_rate);

    auto emit = [&](const std::string& pcm) {
        result.output_samples += pcm.size() / 2;
        if (pcm.empty() || !callback)
            return true;
        if (!callback(result.metadata, pcm)) {
            result.cancelled = true;
            return false;
        }
        return true;
    };
    auto process = [&](const std::string& pcm) {
        if (!resampler)
            return emit(pcm);
        std::vector<uint8_t> converted;
        resampler->process(reinterpret_cast<const uint8_t*>(pcm.data()), pcm.size(), &converted);
        if (converted.empty())
            return true;
        return emit(std::string(reinterpret_cast<const char*>(converted.data()), converted.size()));
    };

    result.stats =
        impl_->magpie_runtime->synthesize(request.token_chunks, request.options, process);
    if (!result.cancelled && resampler) {
        std::vector<uint8_t> tail;
        resampler->finish(&tail);
        if (!tail.empty()) {
            emit(std::string(reinterpret_cast<const char*>(tail.data()), tail.size()));
        }
    }
    add_preparation_timing(result.stats, request);
    result.stats.sample_rate = output_rate;
    result.stats.samples_written = result.output_samples;
    result.stats.audio_s =
        output_rate > 0 ? static_cast<double>(result.output_samples) / output_rate : 0.0;
    result.stats.rtf =
        result.stats.audio_s > 0.0 ? result.stats.elapsed_s / result.stats.audio_s : 0.0;
    result.stats.rtfx =
        result.stats.elapsed_s > 0.0 ? result.stats.audio_s / result.stats.elapsed_s : 0.0;
    result.stats.e2e_rtfx = result.stats.rtfx;
    return result;
}

SynthesisResult
Synthesizer::synthesize(const SynthesisRequest& request, const PcmCallback& callback) {
#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
    if (impl_->is_omnivoice) {
        if (request.text.empty())
            throw std::invalid_argument("text is required");
        const int output_rate =
            request.output_sample_rate == 0 ? sample_rate() : request.output_sample_rate;
        if (output_rate < 8000 || output_rate > sample_rate()) {
            throw std::invalid_argument(
                "output sample rate must be between 8000 and the model rate (" +
                std::to_string(sample_rate()) + ") Hz, or be 0 (auto)");
        }
        if (request.options.speaker > 0)
            throw std::invalid_argument("OmniVoice only supports automatic speaker index 0");
        resolve_speaker(
            request.voice_name.empty() ? impl_->default_voice_name : request.voice_name);

        const omnivoice::RuntimeConfig runtime_config = resolve_omnivoice_config(request);
        const omnivoice::RuntimeSynthesisRequest runtime_request =
            resolve_omnivoice_request(request);

        SynthesisResult result;
        result.metadata.original_text = request.text;
        result.metadata.processed_text = request.text;
        result.metadata.language_code = runtime_request.language.value_or("");
        result.metadata.tokenizer_name = "qwen2-byte-bpe";
        result.metadata.speaker = 0;
        result.metadata.sample_rate = output_rate;
        std::unique_ptr<audio::AudioResampler> resampler;
        if (output_rate != sample_rate())
            resampler = std::make_unique<audio::AudioResampler>(sample_rate(), output_rate);
        auto emit_float = [&](const float* samples, size_t count) {
            std::vector<float> converted;
            if (resampler) {
                resampler->process(samples, count, &converted);
            } else {
                converted.assign(samples, samples + count);
            }
            result.output_samples += converted.size();
            if (converted.empty() || !callback)
                return true;
            const std::string pcm = float_pcm_to_s16(converted);
            if (!callback(result.metadata, pcm)) {
                result.cancelled = true;
                return false;
            }
            return true;
        };
        const auto generated =
            impl_->omnivoice_runtime->synthesize(runtime_request, runtime_config, emit_float);
        if (!result.cancelled && resampler) {
            std::vector<float> tail;
            resampler->finish(&tail);
            result.output_samples += tail.size();
            if (!tail.empty() && callback) {
                const std::string pcm = float_pcm_to_s16(tail);
                if (!callback(result.metadata, pcm))
                    result.cancelled = true;
            }
        }
        result.metadata.chunk_count = generated.stats.chunks;
        result.stats.sample_rate = output_rate;
        result.stats.generated_frames = generated.stats.generated_frames;
        result.stats.chunks = generated.stats.chunks;
        result.stats.e2e_chunks = generated.stats.chunks;
        result.stats.samples_written = result.output_samples;
        result.stats.elapsed_s = generated.stats.elapsed_seconds;
        result.stats.audio_s = static_cast<double>(result.output_samples) / output_rate;
        result.stats.rtf =
            result.stats.audio_s > 0 ? result.stats.elapsed_s / result.stats.audio_s : 0;
        result.stats.rtfx =
            result.stats.elapsed_s > 0 ? result.stats.audio_s / result.stats.elapsed_s : 0;
        result.stats.e2e_rtfx = result.stats.rtfx;
        return result;
    }
#endif
    return synthesize(prepare(request), callback);
}

SynthesisResult
Synthesizer::synthesize_tokens(
    const std::vector<int32_t>& tokens, const MagpieSynthesisOptions& options,
    int output_sample_rate, const PcmCallback& callback) {
    if (impl_->is_omnivoice)
        throw std::invalid_argument("pre-tokenized synthesis is not supported by OmniVoice");
    if (tokens.empty())
        throw std::invalid_argument("text token list is empty");
    const int output_rate = output_sample_rate == 0 ? sample_rate() : output_sample_rate;
    if (output_rate < 8000 || output_rate > sample_rate()) {
        throw std::invalid_argument(
            "output sample rate must be between 8000 and the model rate (" +
            std::to_string(sample_rate()) + ") Hz, or be 0 (auto)");
    }
    PreparedSynthesis prepared;
    prepared.tokens = tokens;
    prepared.token_chunks.push_back(tokens);
    prepared.options = options;
    if (prepared.options.speaker < 0)
        prepared.options.speaker = impl_->default_speaker;
    if (prepared.options.speaker < 0 || prepared.options.speaker >= speaker_count())
        throw std::invalid_argument("speaker is outside the model speaker range");
    prepared.metadata.processed_text = join(tokens, " ");
    prepared.metadata.tokenizer_name = "pretokenized";
    prepared.metadata.speaker = prepared.options.speaker;
    prepared.metadata.sample_rate = output_rate;
    prepared.metadata.token_count = tokens.size();
    prepared.metadata.chunk_count = 1;
    return synthesize(prepared, callback);
}

SynthesisResult
Synthesizer::warmup(const std::string& text, int steps) {
    if (text.empty())
        return {};
    SynthesisRequest request;
    request.text = text;
    request.language_code = impl_->default_language_code;
    request.options.speaker = impl_->default_speaker;
    request.options.steps = steps;
    return synthesize(request);
}

int
Synthesizer::sample_rate() const {
    if (impl_->is_omnivoice)
        return 24000;
    return impl_->magpie_runtime->sample_rate();
}

int
Synthesizer::speaker_count() const {
    return impl_->is_omnivoice ? 1 : impl_->magpie_runtime->speaker_count();
}

const std::vector<std::string>&
Synthesizer::speaker_names() const {
    return impl_->is_omnivoice ? impl_->speaker_names : impl_->magpie_runtime->speaker_names();
}

std::vector<std::string>
Synthesizer::supported_language_codes() const {
#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
    if (impl_->is_omnivoice)
        return impl_->omnivoice_runtime->language_ids();
#endif
    return impl_->tokenizer ? impl_->tokenizer->supported_language_codes()
                            : std::vector<std::string>{};
}

const std::string&
Synthesizer::model_name() const {
    static const std::string omnivoice_name = "omnivoice";
    return impl_->is_omnivoice ? omnivoice_name : impl_->magpie_runtime->model_name();
}

const std::string&
Synthesizer::default_language_code() const {
    return impl_->default_language_code;
}

int
Synthesizer::default_speaker() const {
    return impl_->default_speaker;
}

bool
Synthesizer::text_normalization_enabled() const {
    return impl_->text_normalizer.enabled();
}

bool
Synthesizer::is_omnivoice() const {
    return impl_->is_omnivoice;
}

OmniVoiceOptions
Synthesizer::omnivoice_defaults() const {
    if (!impl_->is_omnivoice)
        throw std::invalid_argument("OmniVoice options require an OmniVoice model");
    return impl_->omnivoice_defaults;
}

std::unique_ptr<omnivoice::VoicePrompt>
Synthesizer::create_voice_prompt(
    const float* interleaved_pcm, size_t frames, int channels, int sample_rate,
    const std::string& transcript, bool preprocess) {
#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
    if (!impl_->is_omnivoice)
        throw std::invalid_argument("voice prompts require an OmniVoice model");
    return std::make_unique<omnivoice::VoicePrompt>(impl_->omnivoice_runtime->create_prompt(
        interleaved_pcm, frames, channels, sample_rate, transcript, preprocess));
#else
    (void)interleaved_pcm;
    (void)frames;
    (void)channels;
    (void)sample_rate;
    (void)transcript;
    (void)preprocess;
    throw std::invalid_argument("OmniVoice support is not enabled");
#endif
}

std::unique_ptr<omnivoice::VoicePrompt>
Synthesizer::load_voice_prompt(const std::string& path) const {
#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
    if (!impl_->is_omnivoice)
        throw std::invalid_argument("voice prompts require an OmniVoice model");
    return std::make_unique<omnivoice::VoicePrompt>(
        omnivoice::load_voice_prompt(path, &impl_->omnivoice_runtime->fingerprint()));
#else
    (void)path;
    throw std::invalid_argument("OmniVoice support is not enabled");
#endif
}

void
Synthesizer::save_voice_prompt(
    const omnivoice::VoicePrompt& prompt, const std::string& path) const {
#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
    if (!impl_->is_omnivoice)
        throw std::invalid_argument("voice prompts require an OmniVoice model");
    omnivoice::validate_voice_prompt(prompt, &impl_->omnivoice_runtime->fingerprint());
    omnivoice::save_voice_prompt(prompt, path);
#else
    (void)prompt;
    (void)path;
    throw std::invalid_argument("OmniVoice support is not enabled");
#endif
}

#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
omnivoice::Runtime&
Synthesizer::omnivoice_runtime() {
    if (!impl_->is_omnivoice)
        throw std::invalid_argument("operation requires an OmniVoice model");
    return *impl_->omnivoice_runtime;
}

omnivoice::RuntimeConfig
Synthesizer::resolve_omnivoice_config(const SynthesisRequest& request) const {
    if (!impl_->is_omnivoice)
        throw std::invalid_argument("operation requires an OmniVoice model");
    OmniVoiceOptions selected = request.omnivoice_options.value_or(impl_->omnivoice_defaults);
    if (!request.omnivoice_options) {
        if (request.options.steps > 0)
            selected.num_steps = request.options.steps;
        if (request.options.override_cfg_scale)
            selected.guidance_scale = request.options.cfg_scale;
    }
    MagpieSynthesisOptions legacy = request.options;
    if (legacy.seed < 0)
        legacy.seed = impl_->magpie_config.seed;
    return to_runtime_config(selected, legacy);
}

omnivoice::RuntimeSynthesisRequest
Synthesizer::resolve_omnivoice_request(const SynthesisRequest& request) const {
    if (!impl_->is_omnivoice)
        throw std::invalid_argument("operation requires an OmniVoice model");
    OmniVoiceOptions selected = request.omnivoice_options.value_or(impl_->omnivoice_defaults);
    if (!request.omnivoice_options) {
        if (request.options.steps > 0)
            selected.num_steps = request.options.steps;
        if (request.options.override_cfg_scale)
            selected.guidance_scale = request.options.cfg_scale;
    }
    omnivoice::RuntimeSynthesisRequest resolved;
    resolved.text = request.text;
    resolved.language = request.language_code.empty()
                            ? std::optional<std::string>(impl_->default_language_code)
                            : std::optional<std::string>(request.language_code);
    if (!request.instruction.empty())
        resolved.instruction = request.instruction;
    resolved.voice_prompt = request.voice_prompt;
    resolved.speed = selected.speed;
    if (!std::isfinite(selected.duration_s) || selected.duration_s < 0.0)
        throw std::invalid_argument("OmniVoice duration must be zero or positive and finite");
    if (selected.duration_s > 0.0)
        resolved.fixed_duration_seconds = selected.duration_s;
    return resolved;
}
#endif

}  // namespace nemo_speech::tts
