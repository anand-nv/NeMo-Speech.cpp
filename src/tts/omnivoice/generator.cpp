// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "generator.h"

#include <unicode/utf8.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "denoiser.h"
#include "frontend.h"
#include "runtime/ggml/runtime.h"
#include "tokenizer.h"

namespace nemo_speech::tts::omnivoice {
namespace {

bool
contains_chinese(const std::string& text) {
    int32_t offset = 0;
    while (offset < static_cast<int32_t>(text.size())) {
        UChar32 cp = 0;
        U8_NEXT(text.data(), offset, static_cast<int32_t>(text.size()), cp);
        if (cp < 0)
            throw std::invalid_argument("OmniVoice text is not valid UTF-8");
        if (cp >= 0x4e00 && cp <= 0x9fff)
            return true;
    }
    return false;
}

void
append(std::vector<int32_t>& destination, const std::vector<int32_t>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

}  // namespace

Generator::Generator(ggml_runtime::BackendManager& backends, const std::string& model_path) {
    loader_ = std::make_unique<ggml_runtime::GGUFLoader>(model_path);
    config_ = load_model_config(*loader_);
    tokenizer_ = std::make_unique<Tokenizer>(*loader_);
    frontend_ = std::make_unique<FrontendTables>(*loader_);
    denoiser_ = std::make_unique<Denoiser>(backends, model_path);
}

Generator::~Generator() = default;

int32_t
Generator::estimate_target_frames(const TokenGenerationRequest& request) const {
    std::optional<int32_t> reference_frames;
    if (request.reference_audio_codes) {
        reference_frames = static_cast<int32_t>((*request.reference_audio_codes)[0].size());
    }
    return request.target_frames.value_or(frontend_->target_frames(
        request.text, request.reference_text, reference_frames, request.speed,
        request.fixed_duration_seconds));
}

const std::vector<std::string>&
Generator::language_ids() const {
    return frontend_->language_ids();
}

const std::vector<std::string>&
Generator::language_names() const {
    return frontend_->language_names();
}

GeneratedAudioCodes
Generator::generate(const TokenGenerationRequest& request, const GenerationConfig& config) {
    if (config.cancelled && config.cancelled())
        throw GenerationCancelled();
    if (request.text.empty())
        throw std::invalid_argument("OmniVoice text must not be empty");
    if (config.steps <= 0 || !std::isfinite(config.guidance_scale) ||
        !std::isfinite(config.time_shift) || config.time_shift <= 0.0f ||
        !std::isfinite(config.layer_penalty) || !std::isfinite(config.position_temperature) ||
        config.position_temperature < 0.0f || !std::isfinite(config.class_temperature) ||
        config.class_temperature < 0.0f) {
        throw std::invalid_argument("invalid OmniVoice generation configuration");
    }
    if (request.reference_audio_codes && request.instruction) {
        throw std::invalid_argument(
            "OmniVoice voice cloning and voice design are mutually exclusive");
    }
    int32_t reference_frames = 0;
    if (request.reference_audio_codes) {
        if (!request.reference_text || request.reference_text->empty()) {
            throw std::invalid_argument("OmniVoice voice cloning requires a non-empty transcript");
        }
        reference_frames = static_cast<int32_t>((*request.reference_audio_codes)[0].size());
        if (reference_frames <= 0)
            throw std::invalid_argument("OmniVoice reference codes are empty");
        for (const auto& codebook : *request.reference_audio_codes) {
            if (codebook.size() != static_cast<size_t>(reference_frames)) {
                throw std::invalid_argument("OmniVoice reference codebooks have different lengths");
            }
            for (int32_t token : codebook) {
                if (token < 0 || token >= 1024) {
                    throw std::invalid_argument("OmniVoice reference audio token is out of range");
                }
            }
        }
    } else if (request.reference_text) {
        throw std::invalid_argument(
            "OmniVoice reference text was provided without reference audio");
    }

    const auto language = frontend_->resolve_language(request.language);
    const auto instruction =
        frontend_->resolve_instruction(request.instruction, contains_chinese(request.text));
    const int32_t target_frames = estimate_target_frames(request);
    if (target_frames <= 0)
        throw std::invalid_argument("OmniVoice target duration is empty");

    std::string style;
    if (config.denoise && request.reference_audio_codes)
        style += "<|denoise|>";
    style += "<|lang_start|>" + language.value_or("None") + "<|lang_end|>";
    style += "<|instruct_start|>" + instruction.value_or("None") + "<|instruct_end|>";
    const auto style_ids = tokenizer_->encode(style);
    const std::string full_text = combine_text(request.text, request.reference_text);
    const auto text_ids =
        tokenizer_->encode_with_nonverbal_tags("<|text_start|>" + full_text + "<|text_end|>");

    std::vector<int32_t> conditioning_text;
    append(conditioning_text, style_ids);
    append(conditioning_text, text_ids);
    const int32_t audio_start = static_cast<int32_t>(conditioning_text.size());
    const int32_t conditional_length = audio_start + reference_frames + target_frames;
    if (conditional_length > denoiser_->config().context_length) {
        throw std::invalid_argument("OmniVoice request exceeds the 40960-position context limit");
    }

    constexpr int32_t B = 2;
    constexpr int32_t C = 8;
    constexpr int32_t V = 1025;
    const int32_t S = conditional_length;
    const int32_t conditional_target_start = S - target_frames;
    DenoiserInput input;
    input.batch_size = B;
    input.sequence_length = S;
    input.text_ids.assign(static_cast<size_t>(B) * S, 1024);
    input.shifted_audio_ids.resize(static_cast<size_t>(B) * C * S);
    input.audio_mask.assign(static_cast<size_t>(B) * S, 0.0f);
    input.position_ids.resize(static_cast<size_t>(S));
    input.attention_mask.assign(static_cast<size_t>(B) * S * S, -1.0e9f);
    for (int32_t position = 0; position < S; ++position) input.position_ids[position] = position;
    std::copy(conditioning_text.begin(), conditioning_text.end(), input.text_ids.begin());

    // Conditional audio region: optional prompt followed by target masks.
    for (int32_t position = audio_start; position < S; ++position) {
        input.audio_mask[static_cast<size_t>(position)] = 1.0f;
    }
    for (int32_t codebook = 0; codebook < C; ++codebook) {
        const size_t cond_base = static_cast<size_t>(codebook) * S;
        for (int32_t position = 0; position < S; ++position) {
            int32_t token = 0;
            if (position >= audio_start && position < conditional_target_start) {
                token =
                    (*request.reference_audio_codes)[static_cast<size_t>(codebook)]
                                                    [static_cast<size_t>(position - audio_start)];
            } else if (position >= conditional_target_start) {
                token = 1024;
            }
            input.shifted_audio_ids[cond_base + position] = codebook * V + token;
        }
    }
    for (int32_t query = 0; query < S; ++query) {
        for (int32_t key = 0; key < S; ++key) {
            input.attention_mask[static_cast<size_t>(query) * S + key] = 0.0f;
        }
    }

    // Unconditional document occupies the first target_frames positions.
    const size_t uncond_text = static_cast<size_t>(S);
    const size_t uncond_audio_mask = static_cast<size_t>(S);
    const size_t uncond_audio = static_cast<size_t>(C) * S;
    const size_t uncond_attention = static_cast<size_t>(S) * S;
    for (int32_t position = 0; position < target_frames; ++position) {
        input.audio_mask[uncond_audio_mask + position] = 1.0f;
    }
    for (int32_t codebook = 0; codebook < C; ++codebook) {
        const size_t base = uncond_audio + static_cast<size_t>(codebook) * S;
        for (int32_t position = 0; position < S; ++position) {
            input.shifted_audio_ids[base + position] = codebook * V + 1024;
        }
    }
    for (int32_t query = 0; query < target_frames; ++query) {
        for (int32_t key = 0; key < target_frames; ++key) {
            input.attention_mask[uncond_attention + static_cast<size_t>(query) * S + key] = 0.0f;
        }
    }
    for (int32_t position = target_frames; position < S; ++position) {
        input.attention_mask[uncond_attention + static_cast<size_t>(position) * S + position] =
            0.0f;
    }

    std::vector<int32_t> tokens(static_cast<size_t>(C) * target_frames, 1024);
    const auto schedule = reveal_schedule(target_frames, C, config.steps, config.time_shift);
    const CounterRandom random(config.seed);
    const UniformSource source = [&random](const RandomCounter& counter) {
        return random.uniform(counter);
    };
    const SamplingConfig sampling = {
        config.guidance_scale, config.layer_penalty, config.position_temperature,
        config.class_temperature};

    std::vector<float> conditional_logits(static_cast<size_t>(C) * target_frames * V);
    std::vector<float> unconditional_logits(static_cast<size_t>(C) * target_frames * V);
    for (int32_t step = 0; step < config.steps; ++step) {
        if (config.cancelled && config.cancelled())
            throw GenerationCancelled();
        if (schedule[static_cast<size_t>(step)] == 0)
            continue;
        if (config.cancelled && config.cancelled())
            throw GenerationCancelled();
        const DenoiserOutput output = denoiser_->forward(input);
        if (config.cancelled && config.cancelled())
            throw GenerationCancelled();
        for (int32_t codebook = 0; codebook < C; ++codebook) {
            for (int32_t position = 0; position < target_frames; ++position) {
                const size_t target =
                    (static_cast<size_t>(codebook) * target_frames + position) * V;
                const size_t cond_source =
                    (static_cast<size_t>(codebook) * S + conditional_target_start + position) * V;
                const size_t uncond_source =
                    ((static_cast<size_t>(C) + codebook) * S + position) * V;
                std::copy_n(
                    output.logits.data() + cond_source, V, conditional_logits.data() + target);
                std::copy_n(
                    output.logits.data() + uncond_source, V, unconditional_logits.data() + target);
            }
        }
        sample_step(
            conditional_logits.data(), unconditional_logits.data(), C, target_frames, V, 1024,
            schedule[static_cast<size_t>(step)], step, 0, sampling, tokens, source);

        for (int32_t codebook = 0; codebook < C; ++codebook) {
            const size_t token_base = static_cast<size_t>(codebook) * target_frames;
            const size_t cond_base = static_cast<size_t>(codebook) * S;
            const size_t uncond_base = uncond_audio + static_cast<size_t>(codebook) * S;
            for (int32_t position = 0; position < target_frames; ++position) {
                const int32_t token = tokens[token_base + position];
                input.shifted_audio_ids[cond_base + conditional_target_start + position] =
                    codebook * V + token;
                input.shifted_audio_ids[uncond_base + position] = codebook * V + token;
                if (codebook == 0) {
                    input.text_ids[static_cast<size_t>(conditional_target_start + position)] =
                        token;
                    input.text_ids[uncond_text + position] = token;
                }
            }
        }
    }

    GeneratedAudioCodes result;
    result.effective_seed = random.seed();
    for (int32_t codebook = 0; codebook < C; ++codebook) {
        const auto begin = tokens.begin() + static_cast<size_t>(codebook) * target_frames;
        result.codebooks[static_cast<size_t>(codebook)].assign(begin, begin + target_frames);
        if (std::any_of(
                result.codebooks[static_cast<size_t>(codebook)].begin(),
                result.codebooks[static_cast<size_t>(codebook)].end(),
                [](int32_t token) { return token < 0 || token >= 1024; })) {
            throw std::runtime_error("OmniVoice generation left an invalid audio token");
        }
    }
    return result;
}

}  // namespace nemo_speech::tts::omnivoice
