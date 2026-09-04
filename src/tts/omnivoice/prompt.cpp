// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "prompt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "codec.h"
#include "frontend.h"

namespace nemo_speech::tts::omnivoice {
namespace {

constexpr int32_t kPromptSampleRate = 24000;
constexpr int32_t kPromptHopLength = 960;
constexpr size_t kMaxPromptFrames = 25 * 60 * 20;
constexpr size_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr char kMagic[8] = {'N', 'V', 'P', 'R', 'O', 'M', 'P', 'T'};

std::vector<float>
torchaudio_resample(const std::vector<float>& input, int32_t source_rate, int32_t target_rate) {
    if (source_rate <= 0 || target_rate <= 0)
        throw std::invalid_argument("reference sample rate must be positive");
    if (source_rate == target_rate)
        return input;
    if (input.empty())
        return {};

    const int32_t divisor = std::gcd(source_rate, target_rate);
    const int32_t source = source_rate / divisor;
    const int32_t target = target_rate / divisor;
    constexpr int32_t lowpass_width = 6;
    constexpr double rolloff = 0.99;
    const double base_frequency = std::min(source, target) * rolloff;
    const int32_t width = static_cast<int32_t>(std::ceil(lowpass_width * source / base_frequency));
    const int32_t kernel_length = 2 * width + source;
    std::vector<double> kernel(static_cast<size_t>(target) * kernel_length);
    constexpr double pi = 3.14159265358979323846264338327950288;
    for (int32_t phase = 0; phase < target; ++phase) {
        for (int32_t tap = 0; tap < kernel_length; ++tap) {
            double t =
                (static_cast<double>(tap - width) / source - static_cast<double>(phase) / target) *
                base_frequency;
            t = std::clamp(
                t, -static_cast<double>(lowpass_width), static_cast<double>(lowpass_width));
            const double window = std::pow(std::cos(t * pi / lowpass_width / 2.0), 2.0);
            t *= pi;
            const double sinc = std::abs(t) < 1.0e-12 ? 1.0 : std::sin(t) / t;
            kernel[static_cast<size_t>(phase) * kernel_length + tap] =
                sinc * window * base_frequency / source;
        }
    }

    const size_t output_length =
        (static_cast<uint64_t>(target) * input.size() + source - 1) / source;
    std::vector<float> output;
    output.reserve(output_length);
    for (size_t block = 0; output.size() < output_length; ++block) {
        for (int32_t phase = 0; phase < target && output.size() < output_length; ++phase) {
            double sum = 0.0;
            for (int32_t tap = 0; tap < kernel_length; ++tap) {
                const int64_t index =
                    static_cast<int64_t>(block) * source + tap - static_cast<int64_t>(width);
                if (index >= 0 && index < static_cast<int64_t>(input.size())) {
                    sum += input[static_cast<size_t>(index)] *
                           kernel[static_cast<size_t>(phase) * kernel_length + tap];
                }
            }
            output.push_back(static_cast<float>(sum));
        }
    }
    return output;
}

std::vector<int16_t>
to_pydub_pcm(const std::vector<float>& input) {
    std::vector<int16_t> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const float scaled = std::clamp(input[i] * 32768.0f, -32768.0f, 32767.0f);
        result[i] = static_cast<int16_t>(scaled);
    }
    return result;
}

bool
is_silent(const std::vector<int16_t>& pcm, size_t begin, size_t end, double threshold) {
    if (end <= begin)
        return true;
    long double power = 0.0;
    for (size_t i = begin; i < end; ++i) {
        const long double value = pcm[i];
        power += value * value;
    }
    return std::sqrt(power / static_cast<long double>(end - begin)) <= threshold;
}

std::vector<std::pair<size_t, size_t>>
detect_silence(
    const std::vector<int16_t>& pcm, int32_t sample_rate, int32_t minimum_ms,
    double silence_db = -50.0) {
    const size_t window = static_cast<size_t>(minimum_ms) * sample_rate / 1000;
    const size_t seek = static_cast<size_t>(10) * sample_rate / 1000;
    if (window == 0 || pcm.size() < window)
        return {};
    const double threshold = 32768.0 * std::pow(10.0, silence_db / 20.0);
    std::vector<size_t> starts;
    for (size_t start = 0; start + window <= pcm.size(); start += seek) {
        if (is_silent(pcm, start, start + window, threshold))
            starts.push_back(start);
    }
    const size_t last = pcm.size() - window;
    if (last % seek != 0 && is_silent(pcm, last, pcm.size(), threshold))
        starts.push_back(last);
    if (starts.empty())
        return {};

    std::vector<std::pair<size_t, size_t>> ranges;
    size_t range_start = starts[0];
    size_t previous = starts[0];
    for (size_t i = 1; i < starts.size(); ++i) {
        const bool continuous = starts[i] == previous + seek;
        const bool overlaps = starts[i] <= previous + window;
        if (!continuous && !overlaps) {
            ranges.emplace_back(range_start, previous + window);
            range_start = starts[i];
        }
        previous = starts[i];
    }
    ranges.emplace_back(range_start, std::min(pcm.size(), previous + window));
    return ranges;
}

std::vector<std::pair<size_t, size_t>>
detect_nonsilent(const std::vector<int16_t>& pcm, int32_t sample_rate, int32_t minimum_ms) {
    const auto silent = detect_silence(pcm, sample_rate, minimum_ms);
    if (silent.empty())
        return pcm.empty() ? std::vector<std::pair<size_t, size_t>>()
                           : std::vector<std::pair<size_t, size_t>>{{0, pcm.size()}};
    if (silent.size() == 1 && silent[0].first == 0 && silent[0].second == pcm.size())
        return {};
    std::vector<std::pair<size_t, size_t>> result;
    size_t previous = 0;
    for (const auto& range : silent) {
        if (range.first > previous)
            result.emplace_back(previous, range.first);
        previous = range.second;
    }
    if (previous < pcm.size())
        result.emplace_back(previous, pcm.size());
    return result;
}

size_t
leading_silence(const std::vector<int16_t>& pcm, int32_t sample_rate) {
    const size_t chunk = static_cast<size_t>(10) * sample_rate / 1000;
    const double threshold = 32768.0 * std::pow(10.0, -50.0 / 20.0);
    size_t position = 0;
    while (position + chunk <= pcm.size() && is_silent(pcm, position, position + chunk, threshold))
        position += chunk;
    return position;
}

std::vector<float>
remove_silence_like_pydub(
    const std::vector<float>& input, int32_t sample_rate, int32_t middle_ms, int32_t leading_ms,
    int32_t trailing_ms) {
    if (input.empty())
        return {};
    std::vector<float> joined;
    if (middle_ms > 0) {
        const auto pcm = to_pydub_pcm(input);
        auto nonsilent = detect_nonsilent(pcm, sample_rate, middle_ms);
        const size_t keep = static_cast<size_t>(middle_ms) * sample_rate / 1000;
        std::vector<std::pair<size_t, size_t>> ranges;
        ranges.reserve(nonsilent.size());
        for (const auto& range : nonsilent) {
            ranges.emplace_back(
                range.first > keep ? range.first - keep : 0,
                std::min(input.size(), range.second + keep));
        }
        for (size_t i = 0; i + 1 < ranges.size(); ++i) {
            if (ranges[i + 1].first < ranges[i].second) {
                const size_t middle = (ranges[i].second + ranges[i + 1].first) / 2;
                ranges[i].second = middle;
                ranges[i + 1].first = middle;
            }
        }
        for (const auto& range : ranges)
            joined.insert(joined.end(), input.begin() + range.first, input.begin() + range.second);
    } else {
        joined = input;
    }
    if (joined.empty())
        return {};

    auto pcm = to_pydub_pcm(joined);
    const size_t lead_keep = static_cast<size_t>(leading_ms) * sample_rate / 1000;
    const size_t start_silence = leading_silence(pcm, sample_rate);
    const size_t start = start_silence > lead_keep ? start_silence - lead_keep : 0;
    joined.erase(joined.begin(), joined.begin() + start);
    if (joined.empty())
        return {};

    pcm = to_pydub_pcm(joined);
    std::reverse(pcm.begin(), pcm.end());
    const size_t trail_keep = static_cast<size_t>(trailing_ms) * sample_rate / 1000;
    const size_t end_silence = leading_silence(pcm, sample_rate);
    const size_t remove = end_silence > trail_keep ? end_silence - trail_keep : 0;
    if (remove > 0)
        joined.resize(joined.size() - remove);
    return joined;
}

template <typename T>
void
write_le(std::ostream& output, T value) {
    static_assert(std::is_integral_v<T>);
    using U = std::make_unsigned_t<T>;
    U bits = static_cast<U>(value);
    for (size_t i = 0; i < sizeof(T); ++i) output.put(static_cast<char>((bits >> (8 * i)) & 0xff));
    if (!output)
        throw std::runtime_error("failed while writing OmniVoice prompt");
}

template <typename T>
T
read_le(std::istream& input) {
    static_assert(std::is_integral_v<T>);
    using U = std::make_unsigned_t<T>;
    U bits = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
        const int byte = input.get();
        if (byte == std::char_traits<char>::eof())
            throw std::invalid_argument("truncated OmniVoice prompt");
        bits |= static_cast<U>(static_cast<uint8_t>(byte)) << (8 * i);
    }
    return static_cast<T>(bits);
}

void
write_string(std::ostream& output, const std::string& value) {
    if (value.size() > kMaxStringBytes)
        throw std::invalid_argument("OmniVoice prompt string is too large");
    write_le<uint32_t>(output, static_cast<uint32_t>(value.size()));
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!output)
        throw std::runtime_error("failed while writing OmniVoice prompt");
}

std::string
read_string(std::istream& input) {
    const size_t size = read_le<uint32_t>(input);
    if (size > kMaxStringBytes)
        throw std::invalid_argument("OmniVoice prompt string is too large");
    std::string value(size, '\0');
    input.read(value.data(), static_cast<std::streamsize>(size));
    if (!input)
        throw std::invalid_argument("truncated OmniVoice prompt");
    return value;
}

bool
same_fingerprint(const PromptFingerprint& a, const PromptFingerprint& b) {
    return a.source_revision == b.source_revision &&
           a.omnivoice_model_sha256 == b.omnivoice_model_sha256 &&
           a.tokenizer_sha256 == b.tokenizer_sha256 &&
           a.audio_tokenizer_sha256 == b.audio_tokenizer_sha256;
}

}  // namespace

PreparedReferenceAudio
preprocess_reference_audio(
    const float* interleaved_pcm, size_t frames, int32_t channels, int32_t sample_rate,
    bool remove_prompt_silence) {
    if (!interleaved_pcm || frames == 0)
        throw std::invalid_argument("reference audio is empty");
    if (channels <= 0 || channels > 256)
        throw std::invalid_argument("invalid reference channel count");
    if (sample_rate <= 0 || sample_rate > 768000)
        throw std::invalid_argument("invalid reference sample rate");
    if (frames > std::numeric_limits<size_t>::max() / static_cast<size_t>(channels))
        throw std::overflow_error("reference audio is too large");

    std::vector<float> mono(frames);
    for (size_t frame = 0; frame < frames; ++frame) {
        double sum = 0.0;
        for (int32_t channel = 0; channel < channels; ++channel) {
            const float sample = interleaved_pcm[frame * static_cast<size_t>(channels) + channel];
            if (!std::isfinite(sample))
                throw std::invalid_argument("reference audio contains NaN or infinity");
            sum += sample;
        }
        mono[frame] = static_cast<float>(sum / channels);
    }
    mono = torchaudio_resample(mono, sample_rate, kPromptSampleRate);
    if (mono.size() > static_cast<size_t>(20 * kPromptSampleRate)) {
        std::fprintf(
            stderr, "warning: OmniVoice reference audio exceeds the recommended 20-second limit\n");
    }

    double power = 0.0;
    for (float sample : mono) power += static_cast<double>(sample) * sample;
    const float rms = mono.empty() ? 0.0f : static_cast<float>(std::sqrt(power / mono.size()));
    if (rms > 0.0f && rms < 0.1f) {
        const float scale = 0.1f / rms;
        for (float& sample : mono) sample *= scale;
    }
    if (remove_prompt_silence)
        mono = remove_silence_like_pydub(mono, kPromptSampleRate, 200, 100, 200);
    if (mono.empty())
        throw std::invalid_argument("reference audio is empty after silence removal");
    mono.resize(mono.size() - mono.size() % kPromptHopLength);
    if (mono.empty())
        throw std::invalid_argument("reference audio is shorter than one 960-sample codec hop");
    return {std::move(mono), rms};
}

std::vector<float>
remove_silence(
    const std::vector<float>& mono, int32_t sample_rate, int32_t middle_ms, int32_t leading_ms,
    int32_t trailing_ms) {
    if (sample_rate <= 0 || middle_ms < 0 || leading_ms < 0 || trailing_ms < 0)
        throw std::invalid_argument("invalid silence-removal configuration");
    return remove_silence_like_pydub(mono, sample_rate, middle_ms, leading_ms, trailing_ms);
}

VoicePrompt
create_voice_prompt(
    CodecEncoder& encoder, const float* interleaved_pcm, size_t frames, int32_t channels,
    int32_t sample_rate, const std::string& transcript, const PromptFingerprint& fingerprint,
    bool preprocess) {
    if (transcript.find_first_not_of(" \t\r\n") == std::string::npos)
        throw std::invalid_argument("voice cloning requires a non-empty transcript");
    PreparedReferenceAudio audio =
        preprocess_reference_audio(interleaved_pcm, frames, channels, sample_rate, preprocess);
    VoicePrompt prompt;
    prompt.audio_codes = encoder.encode(audio.mono_24khz);
    prompt.transcript = preprocess ? add_terminal_punctuation(transcript) : transcript;
    prompt.reference_rms = audio.original_rms;
    prompt.fingerprint = fingerprint;
    validate_voice_prompt(prompt);
    return prompt;
}

void
validate_voice_prompt(const VoicePrompt& prompt, const PromptFingerprint* expected_fingerprint) {
    if (prompt.format_version != kVoicePromptFormatVersion)
        throw std::invalid_argument("unsupported OmniVoice prompt format version");
    if (prompt.transcript.find_first_not_of(" \t\r\n") == std::string::npos)
        throw std::invalid_argument("OmniVoice prompt transcript is empty");
    if (!std::isfinite(prompt.reference_rms) || prompt.reference_rms < 0.0f)
        throw std::invalid_argument("OmniVoice prompt RMS is invalid");
    const size_t frames = prompt.audio_codes[0].size();
    if (frames == 0 || frames > kMaxPromptFrames)
        throw std::invalid_argument("OmniVoice prompt frame count is invalid");
    for (const auto& codebook : prompt.audio_codes) {
        if (codebook.size() != frames)
            throw std::invalid_argument("OmniVoice prompt codebooks have different lengths");
        for (int32_t code : codebook) {
            if (code < 0 || code >= 1024)
                throw std::invalid_argument("OmniVoice prompt code is out of range");
        }
    }
    const auto valid_hash = [](const std::string& value) {
        return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
                   return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
               });
    };
    if (prompt.fingerprint.source_revision.empty() ||
        !valid_hash(prompt.fingerprint.omnivoice_model_sha256) ||
        !valid_hash(prompt.fingerprint.tokenizer_sha256) ||
        !valid_hash(prompt.fingerprint.audio_tokenizer_sha256)) {
        throw std::invalid_argument("OmniVoice prompt fingerprint is malformed");
    }
    if (expected_fingerprint && !same_fingerprint(prompt.fingerprint, *expected_fingerprint))
        throw std::invalid_argument(
            "OmniVoice prompt model fingerprint does not match loaded models");
}

void
save_voice_prompt(const VoicePrompt& prompt, const std::string& path) {
    validate_voice_prompt(prompt);
    if (path.empty())
        throw std::invalid_argument("OmniVoice prompt path is empty");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("could not open OmniVoice prompt for writing: " + path);
    output.write(kMagic, sizeof(kMagic));
    write_le<uint32_t>(output, prompt.format_version);
    write_le<uint32_t>(output, 8);
    write_le<uint32_t>(output, static_cast<uint32_t>(prompt.audio_codes[0].size()));
    uint32_t rms_bits = 0;
    static_assert(sizeof(rms_bits) == sizeof(prompt.reference_rms));
    std::memcpy(&rms_bits, &prompt.reference_rms, sizeof(rms_bits));
    write_le<uint32_t>(output, rms_bits);
    write_string(output, prompt.fingerprint.source_revision);
    write_string(output, prompt.fingerprint.omnivoice_model_sha256);
    write_string(output, prompt.fingerprint.tokenizer_sha256);
    write_string(output, prompt.fingerprint.audio_tokenizer_sha256);
    write_string(output, prompt.transcript);
    for (const auto& codebook : prompt.audio_codes)
        for (int32_t code : codebook) write_le<uint16_t>(output, static_cast<uint16_t>(code));
}

VoicePrompt
load_voice_prompt(const std::string& path, const PromptFingerprint* expected_fingerprint) {
    if (path.empty())
        throw std::invalid_argument("OmniVoice prompt path is empty");
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("could not open OmniVoice prompt for reading: " + path);
    char magic[sizeof(kMagic)]{};
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
        throw std::invalid_argument("invalid OmniVoice prompt magic");
    VoicePrompt prompt;
    prompt.format_version = read_le<uint32_t>(input);
    const uint32_t codebooks = read_le<uint32_t>(input);
    const size_t frames = read_le<uint32_t>(input);
    if (codebooks != 8 || frames == 0 || frames > kMaxPromptFrames)
        throw std::invalid_argument("invalid OmniVoice prompt dimensions");
    const uint32_t rms_bits = read_le<uint32_t>(input);
    std::memcpy(&prompt.reference_rms, &rms_bits, sizeof(rms_bits));
    prompt.fingerprint.source_revision = read_string(input);
    prompt.fingerprint.omnivoice_model_sha256 = read_string(input);
    prompt.fingerprint.tokenizer_sha256 = read_string(input);
    prompt.fingerprint.audio_tokenizer_sha256 = read_string(input);
    prompt.transcript = read_string(input);
    for (auto& codebook : prompt.audio_codes) {
        codebook.resize(frames);
        for (int32_t& code : codebook) code = read_le<uint16_t>(input);
    }
    if (input.get() != std::char_traits<char>::eof())
        throw std::invalid_argument("OmniVoice prompt contains trailing data");
    validate_voice_prompt(prompt, expected_fingerprint);
    return prompt;
}

}  // namespace nemo_speech::tts::omnivoice
