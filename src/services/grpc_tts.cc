// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "grpc_tts.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "audio_resampler.h"
#include "riva/proto/riva_audio.pb.h"
#include "tts/omnivoice/prompt.h"
#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
#include "tts/omnivoice/runtime.h"
#endif

namespace nemo_speech {
namespace {

namespace nr_audio = nvidia::riva;

grpc::Status
invalid_arg(const std::string& message) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, message);
}

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

std::string
json_escape(const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 0x0f];
                } else {
                    out += (char)c;
                }
                break;
        }
    }
    return out;
}

std::vector<std::string>
dotted_voice_names(const std::string& model_name, const std::vector<std::string>& speakers) {
    std::vector<std::string> voices;
    voices.reserve(speakers.size());
    for (const auto& speaker : speakers) {
        voices.push_back(model_name + "." + speaker);
    }
    return voices;
}

std::string
json_string_array(const std::vector<std::string>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            out += ",";
        }
        out += "\"";
        out += json_escape(values[i]);
        out += "\"";
    }
    out += "]";
    return out;
}

std::string
voices_by_language_json(
    const std::vector<std::string>& language_codes, const std::vector<std::string>& voices) {
    std::string out = "{";
    const std::string rendered_voices = json_string_array(voices);
    for (size_t i = 0; i < language_codes.size(); ++i) {
        if (i) {
            out += ",";
        }
        out += "\"";
        out += json_escape(language_codes[i]);
        out += "\":{\"voices\":";
        out += rendered_voices;
        out += "}";
    }
    out += "}";
    return out;
}

int
parse_int(const std::string& key, const std::string& value) {
    size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    }
    catch (const std::exception&) {
        throw std::invalid_argument("custom_configuration '" + key + "' must be an integer");
    }
    if (consumed != value.size())
        throw std::invalid_argument("custom_configuration '" + key + "' must be an integer");
    return parsed;
}

float
parse_float(const std::string& key, const std::string& value) {
    size_t consumed = 0;
    float parsed = 0.0f;
    try {
        parsed = std::stof(value, &consumed);
    }
    catch (const std::exception&) {
        throw std::invalid_argument("custom_configuration '" + key + "' must be a float");
    }
    if (consumed != value.size())
        throw std::invalid_argument("custom_configuration '" + key + "' must be a float");
    return parsed;
}

bool
parse_bool(const std::string& key, std::string value) {
    value = lower_ascii(std::move(value));
    if (value == "1" || value == "true" || value == "yes" || value == "on")
        return true;
    if (value == "0" || value == "false" || value == "no" || value == "off")
        return false;
    throw std::invalid_argument("custom_configuration '" + key + "' must be a boolean");
}

struct MappedRequest {
    tts::SynthesisRequest request;
    std::unique_ptr<tts::omnivoice::VoicePrompt> prompt;
};

MappedRequest
map_request(const nr_tts::SynthesizeSpeechRequest& req, tts::Synthesizer& synthesizer) {
    const auto encoding = req.encoding();
    if (encoding != nr_audio::LINEAR_PCM && encoding != nr_audio::ENCODING_UNSPECIFIED)
        throw std::invalid_argument("Only LINEAR_PCM encoding is supported.");
    if (!req.custom_dictionary().empty())
        throw std::invalid_argument("custom_dictionary is not supported by MagpieTTS.");

    MappedRequest mapped;
    auto& out = mapped.request;
    out.text = req.text();
    out.language_code = req.language_code();
    out.voice_name = req.voice_name();
    out.output_sample_rate = req.sample_rate_hz();
    if (req.has_zero_shot_data()) {
        if (!synthesizer.is_omnivoice())
            throw std::invalid_argument("zero_shot_data is only supported by OmniVoice.");
        const auto& zero_shot = req.zero_shot_data();
        if (zero_shot.encoding() != nr_audio::LINEAR_PCM)
            throw std::invalid_argument("OmniVoice zero_shot_data requires LINEAR_PCM encoding.");
        if (zero_shot.sample_rate_hz() <= 0)
            throw std::invalid_argument("zero_shot_data sample_rate_hz must be positive.");
        if (zero_shot.transcript().empty())
            throw std::invalid_argument("OmniVoice zero_shot_data transcript is required.");
        if (zero_shot.audio_prompt().empty() || zero_shot.audio_prompt().size() % 2 != 0)
            throw std::invalid_argument("zero_shot_data audio_prompt must be non-empty PCM16.");
        std::vector<float> samples(zero_shot.audio_prompt().size() / 2);
        const auto* bytes = reinterpret_cast<const uint8_t*>(zero_shot.audio_prompt().data());
        for (size_t i = 0; i < samples.size(); ++i) {
            const uint16_t bits =
                static_cast<uint16_t>(bytes[i * 2]) | static_cast<uint16_t>(bytes[i * 2 + 1]) << 8;
            samples[i] = static_cast<int16_t>(bits) / 32768.0f;
        }
        mapped.prompt = synthesizer.create_voice_prompt(
            samples.data(), samples.size(), 1, zero_shot.sample_rate_hz(), zero_shot.transcript(),
            true);
        out.voice_prompt = mapped.prompt.get();
    }
    std::optional<tts::OmniVoiceOptions> omnivoice;
    auto omni = [&]() -> tts::OmniVoiceOptions& {
        if (!synthesizer.is_omnivoice())
            throw std::invalid_argument(
                "OmniVoice custom configuration requires an OmniVoice model");
        if (!omnivoice)
            omnivoice = synthesizer.omnivoice_defaults();
        return *omnivoice;
    };
    for (const auto& entry : req.custom_configuration()) {
        const std::string key = lower_ascii(entry.first);
        const std::string& value = entry.second;
        if (key == "speaker") {
            out.options.speaker = parse_int(entry.first, value);
        } else if (key == "seed") {
            out.options.seed = parse_int(entry.first, value);
        } else if (key == "steps" || key == "max_decoder_steps") {
            out.options.steps = parse_int(entry.first, value);
        } else if (key == "top_k" || key == "top-k") {
            out.options.top_k = parse_int(entry.first, value);
        } else if (key == "temperature") {
            out.options.temperature = parse_float(entry.first, value);
            out.options.override_temperature = true;
        } else if (key == "cfg_scale" || key == "cfg-scale") {
            out.options.cfg_scale = parse_float(entry.first, value);
            out.options.override_cfg_scale = true;
        } else if (key == "instruction" || key == "instructions") {
            out.instruction = value;
        } else if (key == "speed") {
            omni().speed = parse_float(entry.first, value);
        } else if (key == "duration" || key == "duration_s") {
            omni().duration_s = parse_float(entry.first, value);
        } else if (key == "omnivoice_steps") {
            omni().num_steps = parse_int(entry.first, value);
        } else if (key == "guidance_scale") {
            omni().guidance_scale = parse_float(entry.first, value);
        } else if (key == "time_shift") {
            omni().time_shift = parse_float(entry.first, value);
        } else if (key == "layer_penalty") {
            omni().layer_penalty = parse_float(entry.first, value);
        } else if (key == "position_temperature") {
            omni().position_temperature = parse_float(entry.first, value);
        } else if (key == "class_temperature") {
            omni().class_temperature = parse_float(entry.first, value);
        } else if (key == "denoise") {
            omni().denoise = parse_bool(entry.first, value);
        } else if (key == "postprocess_output") {
            omni().postprocess_output = parse_bool(entry.first, value);
        } else {
            throw std::invalid_argument("Unsupported custom_configuration key: " + entry.first);
        }
    }
    out.omnivoice_options = std::move(omnivoice);
    return mapped;
}

void
fill_response_metadata(
    nr_tts::SynthesizeSpeechResponse& response, const nvidia::riva::RequestId& id,
    const tts::SynthesisMetadata& metadata) {
    *response.mutable_id() = id;
    response.mutable_meta()->set_text(metadata.original_text);
    response.mutable_meta()->set_processed_text(metadata.processed_text);
}

grpc::Status
map_exception() {
    try {
        throw;
    }
    catch (const std::invalid_argument& e) {
        return invalid_arg(e.what());
    }
    catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

void
log_benchmark(
    bool enabled, const char* mode, const nr_tts::SynthesizeSpeechRequest& request,
    const tts::SynthesisResult& result) {
    if (!enabled)
        return;
    const std::string id = request.id().value().empty() ? "-" : request.id().value();
    const auto& metadata = result.metadata;
    const auto& stats = result.stats;
    std::cerr << "[riva_tts][benchmark]"
              << " mode=" << mode << " id=" << id << " text_chars=" << metadata.original_text.size()
              << " processed_text_chars=" << metadata.processed_text.size()
              << " text_tokens=" << metadata.token_count << " text_chunks=" << metadata.chunk_count
              << " speaker=" << metadata.speaker << " frames=" << stats.generated_frames
              << " samples=" << stats.samples_written << " audio_s=" << stats.audio_s
              << " output_sample_rate=" << metadata.sample_rate << " elapsed_s=" << stats.elapsed_s
              << " tokenizer_ms=" << stats.tokenizer_ms << " encoder_ms=" << stats.encoder_ms
              << " decoder_ttft_ms=" << stats.decoder_ttft_ms
              << " decoder_rtfx=" << stats.decoder_rtfx << " codec_ttfa_ms=" << stats.codec_ttfa_ms
              << " codec_rtfx=" << stats.codec_rtfx << " e2e_ttfa_ms=" << stats.e2e_ttfa_ms
              << " e2e_rtfx=" << stats.e2e_rtfx << " codec_chunks=" << stats.chunks
              << " e2e_chunks=" << stats.e2e_chunks << "\n";
}

bool
same_immutable_request(
    const nr_tts::SynthesizeSpeechRequest& first, const nr_tts::SynthesizeSpeechRequest& later) {
    if (first.language_code() != later.language_code() || first.encoding() != later.encoding() ||
        first.sample_rate_hz() != later.sample_rate_hz() ||
        first.voice_name() != later.voice_name() ||
        first.custom_dictionary() != later.custom_dictionary() ||
        first.has_zero_shot_data() != later.has_zero_shot_data() ||
        first.custom_configuration().size() != later.custom_configuration().size())
        return false;
    if (first.has_zero_shot_data() &&
        first.zero_shot_data().SerializeAsString() != later.zero_shot_data().SerializeAsString())
        return false;
    for (const auto& entry : first.custom_configuration()) {
        const auto found = later.custom_configuration().find(entry.first);
        if (found == later.custom_configuration().end() || found->second != entry.second)
            return false;
    }
    return true;
}

std::string
float_pcm_to_s16(const float* samples, size_t count) {
    std::string pcm(count * 2, '\0');
    for (size_t i = 0; i < count; ++i) {
        const float sample = std::clamp(samples[i], -1.0f, 1.0f);
        const int16_t value = static_cast<int16_t>(std::lrintf(sample * 32767.0f));
        const uint16_t bits = static_cast<uint16_t>(value);
        pcm[i * 2] = static_cast<char>(bits & 0xff);
        pcm[i * 2 + 1] = static_cast<char>(bits >> 8);
    }
    return pcm;
}

}  // namespace

GrpcTtsService::GrpcTtsService(std::shared_ptr<tts::Synthesizer> synthesizer, bool benchmark)
    : synthesizer_(std::move(synthesizer)), benchmark_(benchmark) {
    if (!synthesizer_)
        throw std::invalid_argument("GrpcTtsService requires a TTS synthesizer");
}

GrpcTtsService::~GrpcTtsService() = default;

grpc::Status
GrpcTtsService::Synthesize(
    grpc::ServerContext* ctx, const nr_tts::SynthesizeSpeechRequest* req,
    nr_tts::SynthesizeSpeechResponse* resp) {
    try {
        auto mapped = map_request(*req, *synthesizer_);
        std::string audio;
        auto result =
            synthesizer_->synthesize(mapped.request, [&](const auto&, const std::string& pcm) {
                if (ctx->IsCancelled())
                    return false;
                audio.append(pcm);
                return true;
            });
        if (ctx->IsCancelled())
            return grpc::Status(grpc::StatusCode::CANCELLED, "client cancelled");
        resp->set_audio(std::move(audio));
        fill_response_metadata(*resp, req->id(), result.metadata);
        log_benchmark(benchmark_, "unary", *req, result);
        return grpc::Status::OK;
    }
    catch (...) {
        return map_exception();
    }
}

grpc::Status
GrpcTtsService::SynthesizeOnline(
    grpc::ServerContext* ctx,
    grpc::ServerReaderWriter<nr_tts::SynthesizeSpeechResponse, nr_tts::SynthesizeSpeechRequest>*
        stream) {
    nr_tts::SynthesizeSpeechRequest req;
    if (!stream->Read(&req))
        return grpc::Status::OK;

#ifdef NEMO_SPEECH_TTS_WITH_OMNIVOICE
    if (synthesizer_->is_omnivoice()) {
        try {
            auto mapped = map_request(req, *synthesizer_);
            const int output_rate = mapped.request.output_sample_rate == 0
                                        ? synthesizer_->sample_rate()
                                        : mapped.request.output_sample_rate;
            const nr_tts::SynthesizeSpeechRequest first_request = req;
            const nvidia::riva::RequestId response_id = req.id();
            tts::SynthesisMetadata metadata;
            metadata.original_text = req.text();
            metadata.processed_text = req.text();
            metadata.language_code = mapped.request.language_code;
            metadata.tokenizer_name = "omnivoice-qwen2-bpe";
            metadata.sample_rate = output_rate;
            auto runtime_request = synthesizer_->resolve_omnivoice_request(mapped.request);
            runtime_request.text.clear();
            const auto runtime_config = synthesizer_->resolve_omnivoice_config(mapped.request);
            std::atomic<bool> write_failed{false};
            std::unique_ptr<audio::AudioResampler> resampler;
            if (output_rate != synthesizer_->sample_rate())
                resampler = std::make_unique<audio::AudioResampler>(
                    synthesizer_->sample_rate(), output_rate);
            tts::omnivoice::BidirectionalStream bidirectional(
                synthesizer_->omnivoice_runtime(), std::move(runtime_request), runtime_config,
                [&](const float* samples, size_t count) {
                    if (ctx->IsCancelled())
                        return false;
                    std::vector<float> converted;
                    if (resampler) {
                        resampler->process(samples, count, &converted);
                        samples = converted.data();
                        count = converted.size();
                    }
                    if (count == 0)
                        return true;
                    nr_tts::SynthesizeSpeechResponse response;
                    response.set_audio(float_pcm_to_s16(samples, count));
                    fill_response_metadata(response, response_id, metadata);
                    if (!stream->Write(response)) {
                        write_failed.store(true, std::memory_order_relaxed);
                        return false;
                    }
                    return true;
                });
            bidirectional.push_text(req.text().data(), req.text().size(), false);
            while (!write_failed.load(std::memory_order_relaxed) && !ctx->IsCancelled() &&
                   stream->Read(&req)) {
                if (!same_immutable_request(first_request, req)) {
                    bidirectional.cancel();
                    return invalid_arg(
                        "later SynthesizeOnline messages may change only text and request id");
                }
                bidirectional.push_text(req.text().data(), req.text().size(), false);
            }
            if (write_failed.load(std::memory_order_relaxed) || ctx->IsCancelled()) {
                bidirectional.cancel();
                return grpc::Status(grpc::StatusCode::CANCELLED, "client stopped reading");
            }
            const auto stats = bidirectional.finish();
            if (stats.cancelled || write_failed.load(std::memory_order_relaxed))
                return grpc::Status(grpc::StatusCode::CANCELLED, "client stopped reading");
            if (resampler) {
                std::vector<float> tail;
                resampler->finish(&tail);
                if (!tail.empty()) {
                    nr_tts::SynthesizeSpeechResponse response;
                    response.set_audio(float_pcm_to_s16(tail.data(), tail.size()));
                    fill_response_metadata(response, response_id, metadata);
                    if (!stream->Write(response))
                        return grpc::Status(grpc::StatusCode::CANCELLED, "client stopped reading");
                }
            }
            return grpc::Status::OK;
        }
        catch (...) {
            return map_exception();
        }
    }
#endif

    do {
        try {
            auto mapped = map_request(req, *synthesizer_);
            bool write_failed = false;
            auto result = synthesizer_->synthesize(
                mapped.request,
                [&](const tts::SynthesisMetadata& metadata, const std::string& pcm) {
                    if (ctx->IsCancelled())
                        return false;
                    nr_tts::SynthesizeSpeechResponse resp;
                    resp.set_audio(pcm);
                    fill_response_metadata(resp, req.id(), metadata);
                    if (!stream->Write(resp)) {
                        write_failed = true;
                        return false;
                    }
                    return true;
                });
            if (write_failed || ctx->IsCancelled())
                return grpc::Status(grpc::StatusCode::CANCELLED, "client stopped reading");
            log_benchmark(benchmark_, "streaming", req, result);
        }
        catch (...) {
            return map_exception();
        }
    } while (stream->Read(&req));
    return grpc::Status::OK;
}

grpc::Status
GrpcTtsService::GetRivaSynthesisConfig(
    grpc::ServerContext* /*ctx*/, const nr_tts::RivaSynthesisConfigRequest* req,
    nr_tts::RivaSynthesisConfigResponse* resp) {
    try {
        if (!req->model_name().empty() && req->model_name() != synthesizer_->model_name()) {
            return grpc::Status(
                grpc::StatusCode::NOT_FOUND, "unknown TTS model '" + req->model_name() + "'");
        }

        std::vector<std::string> language_codes = synthesizer_->supported_language_codes();
        if (language_codes.empty()) {
            language_codes.push_back(synthesizer_->default_language_code());
        }
        const std::vector<std::string> speaker_names = synthesizer_->speaker_names();
        const std::vector<std::string> dotted_voices =
            dotted_voice_names(synthesizer_->model_name(), speaker_names);
        const std::string subvoices = join(speaker_names, ",");
        const std::string voices_by_language =
            voices_by_language_json(language_codes, dotted_voices);
        for (const auto& language_code : language_codes) {
            auto* mc = resp->add_model_config();
            mc->set_model_name(synthesizer_->model_name());
            auto& params = *mc->mutable_parameters();
            params["language_code"] = language_code;
            params["sample_rate_hz"] = std::to_string(synthesizer_->sample_rate());
            params["encoding"] = "LINEAR_PCM";
            params["tokenizer"] = "native";
            params["voice_name"] = synthesizer_->model_name();
            params["subvoices"] = subvoices;
            params["voices"] = subvoices;
            params["voices_by_language"] = voices_by_language;
            if (synthesizer_->is_omnivoice()) {
                params["zero_shot"] = "true";
                params["voice_design"] = "true";
                params["streaming"] = "output,bidirectional";
            }
        }
        return grpc::Status::OK;
    }
    catch (...) {
        return map_exception();
    }
}

}  // namespace nemo_speech
