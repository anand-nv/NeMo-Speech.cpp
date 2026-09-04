// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ggml_runtime {
class GGUFLoader;
}

namespace nemo_speech::tts::omnivoice {

std::string combine_text(const std::string& target, const std::optional<std::string>& reference);
std::string add_terminal_punctuation(const std::string& text);
std::vector<std::string> chunk_text_punctuation(
    const std::string& text, int32_t chunk_length, std::optional<int32_t> minimum_length = {});
// Byte length of the first complete, non-abbreviation sentence boundary, or
// zero when more text is required. Closing punctuation already present in the
// buffer is kept with the committed prefix.
size_t first_commit_boundary(const std::string& valid_utf8_text);

class FrontendTables {
   public:
    explicit FrontendTables(const ggml_runtime::GGUFLoader& loader);

    // Returns the canonical language ID. Unknown values deliberately fall
    // back to language-agnostic mode (nullopt), matching the oracle.
    std::optional<std::string> resolve_language(const std::optional<std::string>& language) const;
    std::optional<std::string> resolve_instruction(
        const std::optional<std::string>& instruction, bool use_chinese) const;

    double character_weight(const std::string& text) const;
    double estimate_frames(
        const std::string& target, const std::string& reference, double reference_frames,
        std::optional<double> low_threshold = 50.0, double boost_strength = 3.0) const;
    int32_t target_frames(
        const std::string& target, const std::optional<std::string>& reference,
        std::optional<int32_t> reference_frames, std::optional<double> speed,
        std::optional<double> fixed_duration_seconds, int32_t frame_rate = 25) const;

    const std::vector<std::string>& language_ids() const { return language_ids_; }
    const std::vector<std::string>& language_names() const { return language_names_; }

   private:
    std::vector<std::string> language_ids_;
    std::vector<std::string> language_names_;
    std::vector<std::pair<int32_t, std::string>> duration_ranges_;
    std::vector<std::vector<std::string>> instruction_categories_;
    std::vector<std::pair<std::string, std::string>> instruction_translations_;
    std::vector<std::pair<std::string, double>> duration_weights_;
};

}  // namespace nemo_speech::tts::omnivoice
