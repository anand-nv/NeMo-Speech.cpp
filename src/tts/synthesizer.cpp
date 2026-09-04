// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "synthesizer.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "audio_resampler.h"
#include "tts/preproc/text_normalizer.h"
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
#include "nvtx_utils.h"
#include "tts/kokoro/kokoro_runtime.h"
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

#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
void
validate_kokoro_options(const MagpieSynthesisOptions& options) {
    if (options.steps > 0)
        throw std::invalid_argument("Kokoro does not support the Magpie-only steps option");
    if (options.top_k > 0)
        throw std::invalid_argument("Kokoro does not support the Magpie-only top_k option");
    if (options.override_temperature)
        throw std::invalid_argument("Kokoro does not support the Magpie-only temperature option");
    if (options.override_cfg_scale)
        throw std::invalid_argument("Kokoro does not support the Magpie-only cfg_scale option");
}
#endif

}  // namespace

struct Synthesizer::Impl {
    explicit Impl(SynthesizerConfig config)
        : family(config.family),
          default_language_code(
              config.default_language_code.empty() ? "en-US"
                                                   : std::move(config.default_language_code)),
          default_voice_name(std::move(config.default_voice_name)),
          default_speaker(config.runtime.speaker),
          text_normalizer(std::move(config.text_normalizer_model_dir)) {
        if (family == TtsModelFamily::Magpie) {
            runtime = std::make_unique<MagpieTtsRuntime>(std::move(config.runtime));
            if (!config.tokenizer_model_dir.empty()) {
                tokenizer = std::make_unique<MagpieNativeTokenizer>(
                    std::move(config.tokenizer_model_dir), config.tokenizer);
            }
            if (tokenizer && (tokenizer->profile_id() != runtime->tokenizer_profile() ||
                              tokenizer->text_vocab_size() != runtime->text_vocab_size())) {
                throw std::invalid_argument(
                    "Magpie model/tokenizer mismatch: GGUF requires tokenizer profile '" +
                    runtime->tokenizer_profile() + "' with text vocabulary " +
                    std::to_string(runtime->text_vocab_size()) + ", but tokenizer directory is '" +
                    tokenizer->profile_id() + "' with text vocabulary " +
                    std::to_string(tokenizer->text_vocab_size()));
            }
            return;
        }
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
        if (!config.runtime.magpie_model.empty() || !config.runtime.codec_model.empty() ||
            !config.tokenizer_model_dir.empty()) {
            throw std::invalid_argument("Kokoro and Magpie model/tokenizer paths cannot be mixed");
        }
        MagpieSynthesisOptions startup_options;
        startup_options.steps = config.runtime.steps;
        startup_options.top_k = config.runtime.top_k;
        startup_options.override_temperature = config.runtime.override_temperature;
        startup_options.override_cfg_scale = config.runtime.override_cfg_scale;
        validate_kokoro_options(startup_options);
        if (default_voice_name.empty())
            default_voice_name = "af_heart";
        kokoro::KokoroRuntimeConfig kokoro_config;
        kokoro_config.model_path = std::move(config.kokoro_model);
        kokoro_config.use_gpu = config.runtime.lt_backend == MagpieBackendPreference::Cuda;
        kokoro_config.seed = config.runtime.seed;
        kokoro_config.speed = config.runtime.speed;
        kokoro_runtime = std::make_unique<kokoro::KokoroRuntime>(std::move(kokoro_config));
#else
        (void)config;
        throw std::runtime_error("this build does not include Kokoro support");
#endif
    }

    TtsModelFamily family = TtsModelFamily::Magpie;
    std::string default_language_code;
    std::string default_voice_name;
    int default_speaker = 0;
    std::unique_ptr<MagpieTtsRuntime> runtime;
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    std::unique_ptr<kokoro::KokoroRuntime> kokoro_runtime;
#endif
    std::unique_ptr<MagpieNativeTokenizer> tokenizer;
    preproc::TextNormalizer text_normalizer;
};

Synthesizer::Synthesizer(SynthesizerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
    if (!impl_->default_voice_name.empty())
        impl_->default_speaker = resolve_speaker(impl_->default_voice_name);
    if (impl_->default_speaker < 0 || impl_->default_speaker >= speaker_count()) {
        throw std::invalid_argument(
            "default speaker " + std::to_string(impl_->default_speaker) + " must be in [0, " +
            std::to_string(speaker_count() - 1) + "]");
    }
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    if (impl_->family == TtsModelFamily::Kokoro &&
        !kokoro::KokoroTokenizer::voice_matches_language(
            speaker_names()[static_cast<size_t>(impl_->default_speaker)],
            impl_->default_language_code)) {
        throw std::invalid_argument(
            "Kokoro default voice is not compatible with the default language");
    }
#endif
}

Synthesizer::~Synthesizer() = default;

int
Synthesizer::resolve_speaker(const std::string& voice_name) const {
    if (voice_name.empty())
        return impl_->default_speaker;

    size_t consumed = 0;
    try {
        const int value = std::stoi(voice_name, &consumed);
        if (consumed == voice_name.size()) {
            if (value >= 0 && value < speaker_count())
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
    const std::string model_prefix = lower_ascii(model_name()) + ".";
    if (wanted.rfind(model_prefix, 0) == 0)
        wanted.erase(0, model_prefix.size());
    const auto& names = speaker_names();
    for (size_t i = 0; i < names.size(); ++i) {
        if (lower_ascii(names[i]) == wanted)
            return static_cast<int>(i);
    }
    throw std::invalid_argument("unknown voice_name '" + voice_name + "'");
}

PreparedSynthesis
Synthesizer::prepare(const SynthesisRequest& request) const {
    if (request.text.empty())
        throw std::invalid_argument("text is required");
    const int output_rate =
        request.output_sample_rate == 0 ? sample_rate() : request.output_sample_rate;
    if (output_rate < 8000 || output_rate > sample_rate()) {
        throw std::invalid_argument(
            "output sample rate must be between 8000 and the model rate (" +
            std::to_string(sample_rate()) + ") Hz, or be 0 (auto)");
    }

    PreparedSynthesis prepared;
    prepared.family = impl_->family;
    prepared.metadata.original_text = request.text;
    prepared.metadata.language_code =
        request.language_code.empty() ? impl_->default_language_code : request.language_code;
    prepared.metadata.sample_rate = output_rate;
    prepared.options = request.options;
    std::string selected_voice =
        request.voice_name.empty() ? impl_->default_voice_name : request.voice_name;
    if (prepared.options.speaker >= 0) {
        if (prepared.options.speaker >= speaker_count())
            throw std::invalid_argument("speaker is outside the model speaker range");
        selected_voice = speaker_names()[static_cast<size_t>(prepared.options.speaker)];
    } else {
        prepared.options.speaker = resolve_speaker(selected_voice);
    }
    if (prepared.options.speaker < 0 || prepared.options.speaker >= speaker_count())
        throw std::invalid_argument("speaker is outside the model speaker range");
    prepared.metadata.speaker = prepared.options.speaker;

#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    if (impl_->family == TtsModelFamily::Kokoro) {
        const ggml_nvtx::range nvtx_range("kokoro.synthesizer.prepare");
        const std::string nvtx_parameters =
            "kokoro.parameters.prepare text_bytes=" + std::to_string(request.text.size()) +
            " text=" + request.text + " language=" + prepared.metadata.language_code +
            " voice=" + selected_voice + " speaker=" + std::to_string(prepared.options.speaker) +
            " output_rate=" + std::to_string(output_rate);
        ggml_nvtx::mark(nvtx_parameters.c_str());
        validate_kokoro_options(prepared.options);
        if (request.language_code.empty()) {
            prepared.metadata.language_code =
                kokoro::KokoroTokenizer::language_for_voice(selected_voice);
        }
        const auto tokenizer_start = std::chrono::steady_clock::now();
        const kokoro::KokoroPreparedText tokenized = impl_->kokoro_runtime->prepare(
            request.text, prepared.metadata.language_code, selected_voice);
        prepared.tokenizer_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - tokenizer_start)
                                    .count();
        prepared.metadata.language_code = tokenized.tokenization.language;
        prepared.metadata.tokenizer_name =
            std::string("misaki-") + kokoro::kMisakiVersion + "-native";
        prepared.metadata.processed_text.clear();
        for (const kokoro::KokoroChunk& chunk : tokenized.tokenization.chunks) {
            prepared.metadata.processed_text += chunk.processed_text;
        }
        prepared.voice_name = tokenized.voice;
        prepared.token_chunks.reserve(tokenized.tokenization.chunks.size());
        for (const kokoro::KokoroChunk& chunk : tokenized.tokenization.chunks) {
            prepared.token_chunks.push_back(chunk.ids);
            prepared.tokens.insert(prepared.tokens.end(), chunk.ids.begin(), chunk.ids.end());
        }
        if (prepared.token_chunks.empty())
            throw std::invalid_argument("Kokoro tokenizer produced no phoneme IDs");
        prepared.metadata.token_count = prepared.tokens.size();
        prepared.metadata.chunk_count = prepared.token_chunks.size();
        return prepared;
    }
#endif

    if (!impl_->tokenizer) {
        throw std::invalid_argument("text synthesis requires a tokenizer model directory");
    }

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
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
        const ggml_nvtx::range nvtx_range(
            impl_->family == TtsModelFamily::Kokoro ? "kokoro.output.callback" : nullptr);
#endif
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
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
        const ggml_nvtx::range nvtx_range(
            impl_->family == TtsModelFamily::Kokoro ? "kokoro.output.resample" : nullptr);
#endif
        std::vector<uint8_t> converted;
        resampler->process(reinterpret_cast<const uint8_t*>(pcm.data()), pcm.size(), &converted);
        if (converted.empty())
            return true;
        return emit(std::string(reinterpret_cast<const char*>(converted.data()), converted.size()));
    };

#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    if (impl_->family == TtsModelFamily::Kokoro) {
        const ggml_nvtx::range nvtx_range("kokoro.synthesizer.synthesize");
        const float requested_speed = request.options.override_speed ? request.options.speed : 0.0f;
        const std::string nvtx_parameters =
            "kokoro.parameters.synthesize chunks=" + std::to_string(request.token_chunks.size()) +
            " tokens=" + std::to_string(request.tokens.size()) +
            " language=" + request.metadata.language_code + " voice=" + request.voice_name +
            " speaker=" + std::to_string(request.options.speaker) +
            " seed=" + std::to_string(request.options.seed) +
            " speed_override=" + (request.options.override_speed ? "true" : "false") +
            " requested_speed=" +
            (request.options.override_speed ? std::to_string(requested_speed) : "default") +
            " model_rate=" + std::to_string(sample_rate()) +
            " output_rate=" + std::to_string(output_rate);
        ggml_nvtx::mark(nvtx_parameters.c_str());
        std::vector<kokoro::KokoroChunk> chunks;
        chunks.reserve(request.token_chunks.size());
        for (const auto& ids : request.token_chunks) {
            kokoro::KokoroChunk chunk;
            chunk.ids = ids;
            chunks.push_back(std::move(chunk));
        }
        const float speed = requested_speed;
        const kokoro::KokoroRuntimeStats stats = impl_->kokoro_runtime->synthesize(
            chunks, request.voice_name, speed, request.options.seed, process);
        result.cancelled = result.cancelled || stats.cancelled;
        result.stats.sample_rate = stats.sample_rate;
        result.stats.generated_frames = stats.generated_frames;
        result.stats.chunks = stats.chunks;
        result.stats.e2e_chunks = stats.chunks;
        result.stats.samples_written = stats.samples_written;
        result.stats.audio_s = stats.audio_s;
        result.stats.elapsed_s = stats.elapsed_s;
        result.stats.rtf = stats.rtf;
        result.stats.rtfx = stats.rtfx;
        result.stats.ttfa_ms = stats.ttfa_ms;
        result.stats.icl_avg_ms = stats.icl_avg_ms;
        result.stats.icl_min_ms = stats.icl_min_ms;
        result.stats.icl_max_ms = stats.icl_max_ms;
        result.stats.e2e_ttfa_ms = stats.ttfa_ms;
        result.stats.e2e_icl_avg_ms = stats.icl_avg_ms;
        result.stats.e2e_icl_min_ms = stats.icl_min_ms;
        result.stats.e2e_icl_max_ms = stats.icl_max_ms;
    } else
#endif
    {
        result.stats = impl_->runtime->synthesize(request.token_chunks, request.options, process);
    }
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
    return synthesize(prepare(request), callback);
}

SynthesisResult
Synthesizer::synthesize_tokens(
    const std::vector<int32_t>& tokens, const MagpieSynthesisOptions& options,
    int output_sample_rate, const PcmCallback& callback, const std::string& voice_name,
    const std::string& language_code) {
#if !defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    (void)language_code;
#endif
    if (tokens.empty())
        throw std::invalid_argument("text token list is empty");
    const int output_rate = output_sample_rate == 0 ? sample_rate() : output_sample_rate;
    if (output_rate < 8000 || output_rate > sample_rate()) {
        throw std::invalid_argument(
            "output sample rate must be between 8000 and the model rate (" +
            std::to_string(sample_rate()) + ") Hz, or be 0 (auto)");
    }
    PreparedSynthesis prepared;
    prepared.family = impl_->family;
    prepared.options = options;
    if (prepared.options.speaker < 0) {
        prepared.options.speaker =
            voice_name.empty() ? impl_->default_speaker : resolve_speaker(voice_name);
    }
    if (prepared.options.speaker < 0 || prepared.options.speaker >= speaker_count())
        throw std::invalid_argument("speaker is outside the model speaker range");
    prepared.metadata.processed_text = join(tokens, " ");
    prepared.metadata.tokenizer_name = "pretokenized";
    prepared.metadata.speaker = prepared.options.speaker;
    prepared.metadata.sample_rate = output_rate;
    prepared.metadata.token_count = tokens.size();
    prepared.metadata.chunk_count = 1;
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    if (impl_->family == TtsModelFamily::Kokoro) {
        validate_kokoro_options(prepared.options);
        const std::string voice = speaker_names()[static_cast<size_t>(prepared.options.speaker)];
        const kokoro::KokoroChunk chunk = impl_->kokoro_runtime->prepare_tokens(
            tokens,
            language_code.empty() ? kokoro::KokoroTokenizer::language_for_voice(voice)
                                  : language_code,
            voice);
        prepared.tokens = chunk.ids;
        prepared.token_chunks.push_back(chunk.ids);
        prepared.voice_name = voice;
        prepared.metadata.language_code = kokoro::KokoroTokenizer::language_for_voice(voice);
    } else
#endif
    {
        prepared.tokens = tokens;
        prepared.token_chunks.push_back(tokens);
    }
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
    if (impl_->family == TtsModelFamily::Magpie)
        request.options.steps = steps;
    return synthesize(request);
}

int
Synthesizer::sample_rate() const {
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    if (impl_->family == TtsModelFamily::Kokoro)
        return impl_->kokoro_runtime->sample_rate();
#endif
    return impl_->runtime->sample_rate();
}

int
Synthesizer::speaker_count() const {
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    if (impl_->family == TtsModelFamily::Kokoro)
        return static_cast<int>(impl_->kokoro_runtime->voice_names().size());
#endif
    return impl_->runtime->speaker_count();
}

const std::vector<std::string>&
Synthesizer::speaker_names() const {
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    if (impl_->family == TtsModelFamily::Kokoro)
        return impl_->kokoro_runtime->voice_names();
#endif
    return impl_->runtime->speaker_names();
}

std::vector<std::string>
Synthesizer::speaker_names_for_language(const std::string& language_code) const {
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    if (impl_->family == TtsModelFamily::Kokoro) {
        const std::string canonical = kokoro::KokoroTokenizer::canonicalize_language(language_code);
        const auto& names = impl_->kokoro_runtime->voice_names();
        const auto& languages = impl_->kokoro_runtime->voice_languages();
        std::vector<std::string> result;
        for (size_t index = 0; index < names.size(); ++index) {
            if (languages[index] == canonical)
                result.push_back(names[index]);
        }
        return result;
    }
#else
    (void)language_code;
#endif
    return speaker_names();
}

std::vector<std::string>
Synthesizer::supported_language_codes() const {
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    if (impl_->family == TtsModelFamily::Kokoro) {
        std::vector<std::string> languages = impl_->kokoro_runtime->voice_languages();
        std::sort(languages.begin(), languages.end());
        languages.erase(std::unique(languages.begin(), languages.end()), languages.end());
        return languages;
    }
#endif
    return impl_->tokenizer ? impl_->tokenizer->supported_language_codes()
                            : std::vector<std::string>{};
}

const std::string&
Synthesizer::model_name() const {
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    static const std::string kokoro_name = "kokoro";
    if (impl_->family == TtsModelFamily::Kokoro)
        return kokoro_name;
#endif
    return impl_->runtime->model_name();
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
#if defined(NEMO_SPEECH_TTS_WITH_KOKORO)
    if (impl_->family == TtsModelFamily::Kokoro)
        return false;
#endif
    return impl_->text_normalizer.enabled();
}

}  // namespace nemo_speech::tts
