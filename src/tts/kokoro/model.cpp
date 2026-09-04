// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "model.h"

#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "runtime.h"
#include "sha256.h"
#include "tokenizer/kokoro_tokenizer.h"

namespace nemo_speech::tts::kokoro {
namespace {

constexpr const char* kV1Sha256 =
    "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4";

constexpr std::array<const char*, 54> kV1VoiceNames = {
    "af_alloy",    "af_aoede",  "af_bella",   "af_heart",      "af_jessica",  "af_kore",
    "af_nicole",   "af_nova",   "af_river",   "af_sarah",      "af_sky",      "am_adam",
    "am_echo",     "am_eric",   "am_fenrir",  "am_liam",       "am_michael",  "am_onyx",
    "am_puck",     "am_santa",  "bf_alice",   "bf_emma",       "bf_isabella", "bf_lily",
    "bm_daniel",   "bm_fable",  "bm_george",  "bm_lewis",      "ef_dora",     "em_alex",
    "em_santa",    "ff_siwis",  "hf_alpha",   "hf_beta",       "hm_omega",    "hm_psi",
    "if_sara",     "im_nicola", "jf_alpha",   "jf_gongitsune", "jf_nezumi",   "jf_tebukuro",
    "jm_kumo",     "pf_dora",   "pm_alex",    "pm_santa",      "zf_xiaobei",  "zf_xiaoni",
    "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi",      "zm_yunxia",   "zm_yunyang",
};

constexpr std::array<std::pair<const char*, const char*>, 4> kMisakiLexicons = {{
    {"us_gold", "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f"},
    {"us_silver", "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475"},
    {"gb_gold", "29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71"},
    {"gb_silver", "48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce"},
}};
constexpr const char* kJapaneseWordsSha256 =
    "a93a8e8aee24db307a32becb8bf01c4c2908ecf37e6c91f7a705fafdfeba67ff";

int32_t
required_i32(const ggml_runtime::GGUFLoader& loader, const std::string& key) {
    if (!loader.has_key(key)) {
        throw std::runtime_error("Kokoro GGUF is missing metadata key: " + key);
    }
    return loader.get_i32(key);
}

float
required_f32(const ggml_runtime::GGUFLoader& loader, const std::string& key) {
    if (!loader.has_key(key)) {
        throw std::runtime_error("Kokoro GGUF is missing metadata key: " + key);
    }
    const float value = loader.get_f32(key);
    if (!std::isfinite(value)) {
        throw std::runtime_error("Kokoro GGUF metadata is not finite: " + key);
    }
    return value;
}

std::vector<int32_t>
required_i32_array(const ggml_runtime::GGUFLoader& loader, const std::string& key) {
    if (!loader.has_key(key)) {
        throw std::runtime_error("Kokoro GGUF is missing metadata key: " + key);
    }
    const std::vector<int32_t> values = loader.get_i32_array(key);
    if (values.empty()) {
        throw std::runtime_error("Kokoro GGUF metadata is not an int32 array: " + key);
    }
    return values;
}

std::vector<int64_t>
parse_shape(const std::string& encoded, const std::string& tensor_name) {
    std::vector<int64_t> shape;
    std::istringstream input(encoded);
    std::string field;
    while (std::getline(input, field, ',')) {
        if (field.empty()) {
            throw std::runtime_error("empty shape dimension for Kokoro tensor: " + tensor_name);
        }
        size_t used = 0;
        const long long value = std::stoll(field, &used);
        if (used != field.size() || value <= 0) {
            throw std::runtime_error("invalid shape for Kokoro tensor: " + tensor_name);
        }
        shape.push_back(static_cast<int64_t>(value));
    }
    if (shape.empty()) {
        throw std::runtime_error("missing shape for Kokoro tensor: " + tensor_name);
    }
    std::reverse(shape.begin(), shape.end());
    // ggml_tensor does not retain the serialized rank. ggml_n_dims() reports
    // an effective rank, so [3,1,1] is represented as [3].
    while (shape.size() > 1 && shape.back() == 1) {
        shape.pop_back();
    }
    return shape;
}

ggml_type
parse_type(const std::string& value, const std::string& tensor_name) {
    if (value == "float32")
        return GGML_TYPE_F32;
    if (value == "float16")
        return GGML_TYPE_F16;
    if (value == "int32")
        return GGML_TYPE_I32;
    throw std::runtime_error(
        "unsupported stored type '" + value + "' for Kokoro tensor: " + tensor_name);
}

void
expect_value(const std::string& name, int32_t actual, int32_t expected) {
    if (actual != expected) {
        throw std::runtime_error(
            "unsupported Kokoro " + name + ": " + std::to_string(actual) + " (expected " +
            std::to_string(expected) + ")");
    }
}


void
expect_array(
    const std::string& name, const std::vector<int32_t>& actual,
    const std::vector<int32_t>& expected) {
    if (actual != expected) {
        throw std::runtime_error("unsupported Kokoro " + name + " metadata");
    }
}

}  // namespace

KokoroModelMetadata::KokoroModelMetadata(const ggml_runtime::GGUFLoader& loader) {
    if (loader.get_str("general.architecture") != "kokoro") {
        throw std::runtime_error("GGUF architecture is not 'kokoro'");
    }
    source_sha256_ = loader.get_str("kokoro.source.sha256");
    if (source_sha256_ != kV1Sha256) {
        throw std::runtime_error("unsupported Kokoro checkpoint SHA256: " + source_sha256_);
    }
    if (loader.get_str("kokoro.lstm.gate_order") != "i,f,g,o") {
        throw std::runtime_error("Kokoro GGUF has unsupported LSTM gate order");
    }
    if (loader.get_str("kokoro.istftnet.window") != "hann_periodic" ||
        loader.get_str("kokoro.noise.generator") != "splitmix64-box-muller-v1") {
        throw std::runtime_error("Kokoro GGUF has unsupported vocoder numerical metadata");
    }
    if (loader.get_str("kokoro.tokenizer.misaki.version") != kMisakiVersion ||
        loader.get_str("kokoro.tokenizer.misaki.commit") != kMisakiCommit) {
        throw std::runtime_error("Kokoro GGUF has unsupported Misaki tokenizer provenance");
    }
    for (const auto& [name, expected_hash] : kMisakiLexicons) {
        const std::string prefix = std::string("kokoro.tokenizer.misaki.") + name;
        if (loader.get_str(prefix + ".sha256") != expected_hash) {
            throw std::runtime_error(
                "Kokoro GGUF has invalid Misaki lexicon hash: " + std::string(name));
        }
        const std::string payload = loader.get_str(prefix + ".json");
        if (payload.empty()) {
            throw std::runtime_error("Kokoro GGUF is missing Misaki lexicon: " + std::string(name));
        }
        if (sha256_hex(payload) != expected_hash) {
            throw std::runtime_error(
                "Kokoro GGUF Misaki lexicon payload hash mismatch: " + std::string(name));
        }
        misaki_lexicons_.emplace(name, payload);
    }
    if (loader.get_str("kokoro.tokenizer.misaki.ja_words.sha256") != kJapaneseWordsSha256) {
        throw std::runtime_error("Kokoro GGUF has invalid Misaki Japanese word-list hash");
    }
    const std::string japanese_words = loader.get_str("kokoro.tokenizer.misaki.ja_words.text");
    if (sha256_hex(japanese_words) != kJapaneseWordsSha256) {
        throw std::runtime_error("Kokoro GGUF Misaki Japanese word-list payload hash mismatch");
    }
    misaki_lexicons_.emplace("ja_words", japanese_words);

    hparams_.sample_rate = required_i32(loader, "kokoro.sample_rate");
    hparams_.context_length = required_i32(loader, "kokoro.context_length");
    hparams_.phoneme_limit = required_i32(loader, "kokoro.phoneme_limit");
    hparams_.vocab_size = required_i32(loader, "kokoro.vocab_size");
    hparams_.voice_style_dim = required_i32(loader, "kokoro.voice.style_dim");
    hparams_.style_dim = required_i32(loader, "kokoro.style_dim");
    hparams_.hidden_dim = required_i32(loader, "kokoro.hidden_dim");
    hparams_.max_conv_dim = required_i32(loader, "kokoro.max_conv_dim");
    hparams_.dim_in = required_i32(loader, "kokoro.dim_in");
    hparams_.max_dur = required_i32(loader, "kokoro.max_dur");
    hparams_.n_layer = required_i32(loader, "kokoro.n_layer");
    hparams_.n_mels = required_i32(loader, "kokoro.n_mels");
    hparams_.text_encoder_kernel_size = required_i32(loader, "kokoro.text_encoder_kernel_size");
    hparams_.plbert_hidden_size = required_i32(loader, "kokoro.plbert.hidden_size");
    hparams_.plbert_attention_heads = required_i32(loader, "kokoro.plbert.num_attention_heads");
    hparams_.plbert_intermediate_size = required_i32(loader, "kokoro.plbert.intermediate_size");
    hparams_.plbert_layers = required_i32(loader, "kokoro.plbert.num_hidden_layers");
    const int32_t plbert_context = required_i32(loader, "kokoro.plbert.max_position_embeddings");
    hparams_.istft_hop_size = required_i32(loader, "kokoro.istftnet.gen_istft_hop_size");
    hparams_.istft_n_fft = required_i32(loader, "kokoro.istftnet.gen_istft_n_fft");
    const int32_t upsample_initial_channel =
        required_i32(loader, "kokoro.istftnet.upsample_initial_channel");
    const std::vector<int32_t> upsample_kernel_sizes =
        required_i32_array(loader, "kokoro.istftnet.upsample_kernel_sizes");
    const std::vector<int32_t> upsample_rates =
        required_i32_array(loader, "kokoro.istftnet.upsample_rates");
    const std::vector<int32_t> resblock_kernel_sizes =
        required_i32_array(loader, "kokoro.istftnet.resblock_kernel_sizes");
    const std::vector<int32_t> resblock_dilation_sizes =
        required_i32_array(loader, "kokoro.istftnet.resblock_dilation_sizes");
    hparams_.speed_default = required_f32(loader, "kokoro.speed.default");
    hparams_.speed_min = required_f32(loader, "kokoro.speed.min");
    hparams_.speed_max = required_f32(loader, "kokoro.speed.max");

    expect_value("sample rate", hparams_.sample_rate, 24000);
    expect_value("context length", hparams_.context_length, 512);
    expect_value("phoneme limit", hparams_.phoneme_limit, 510);
    expect_value("vocabulary size", hparams_.vocab_size, 178);
    expect_value("voice style dimension", hparams_.voice_style_dim, 256);
    expect_value("style dimension", hparams_.style_dim, 128);
    expect_value("input dimension", hparams_.dim_in, 64);
    expect_value("hidden dimension", hparams_.hidden_dim, 512);
    expect_value("maximum convolution dimension", hparams_.max_conv_dim, 512);
    expect_value("maximum duration", hparams_.max_dur, 50);
    expect_value("text encoder layer count", hparams_.n_layer, 3);
    expect_value("mel bin count", hparams_.n_mels, 80);
    expect_value("text encoder kernel size", hparams_.text_encoder_kernel_size, 5);
    expect_value("PLBERT hidden size", hparams_.plbert_hidden_size, 768);
    expect_value("PLBERT attention heads", hparams_.plbert_attention_heads, 12);
    expect_value("PLBERT intermediate size", hparams_.plbert_intermediate_size, 2048);
    expect_value("PLBERT layer count", hparams_.plbert_layers, 12);
    expect_value("PLBERT context length", plbert_context, 512);
    expect_value("iSTFT hop size", hparams_.istft_hop_size, 5);
    expect_value("iSTFT FFT size", hparams_.istft_n_fft, 20);
    expect_value("iSTFTNet initial channel count", upsample_initial_channel, 512);
    expect_array("iSTFTNet upsample kernels", upsample_kernel_sizes, {20, 12});
    expect_array("iSTFTNet upsample rates", upsample_rates, {10, 6});
    expect_array("iSTFTNet residual kernels", resblock_kernel_sizes, {3, 7, 11});
    expect_array(
        "iSTFTNet residual dilations", resblock_dilation_sizes, {1, 3, 5, 1, 3, 5, 1, 3, 5});
    if (!(hparams_.speed_min == 0.5f && hparams_.speed_default == 1.0f &&
          hparams_.speed_max == 2.0f)) {
        throw std::runtime_error("Kokoro GGUF has unsupported speed range");
    }

    vocabulary_ = loader.get_str_array("kokoro.tokenizer.tokens");
    // Constructor performs strict vocabulary and UTF-8 validation.
    KokoroTokenizer vocabulary_validator(
        vocabulary_, [](const std::string&, const std::string&) { return std::string(); });
    (void)vocabulary_validator;

    voice_names_ = loader.get_str_array("kokoro.voice.names");
    voice_languages_ = loader.get_str_array("kokoro.voice.languages");
    if (voice_names_.size() != 54 || voice_languages_.size() != voice_names_.size()) {
        throw std::runtime_error("Kokoro GGUF must contain 54 voice names and languages");
    }
    for (size_t index = 0; index < kV1VoiceNames.size(); ++index) {
        if (voice_names_[index] != kV1VoiceNames[index]) {
            throw std::runtime_error(
                "Kokoro GGUF voice order mismatch at index " + std::to_string(index));
        }
    }
    std::unordered_set<std::string> unique_voices;
    for (size_t index = 0; index < voice_names_.size(); ++index) {
        const std::string canonical = KokoroTokenizer::canonicalize_voice(voice_names_[index]);
        if (canonical != voice_names_[index] || !unique_voices.insert(canonical).second) {
            throw std::runtime_error(
                "invalid or duplicate Kokoro voice metadata: " + voice_names_[index]);
        }
        if (KokoroTokenizer::language_for_voice(canonical) != voice_languages_[index]) {
            throw std::runtime_error("Kokoro voice/language metadata mismatch: " + canonical);
        }
        const std::string tensor_name = "kokoro.voice." + canonical;
        if (!loader.has_tensor(tensor_name)) {
            throw std::runtime_error("Kokoro GGUF is missing voice tensor: " + tensor_name);
        }
        if (loader.get_tensor_type(tensor_name) != GGML_TYPE_F32 ||
            loader.get_tensor_shape(tensor_name) != std::vector<int64_t>({256, 510})) {
            throw std::runtime_error(
                "Kokoro voice tensor must be F32 with GGML shape [256,510]: " + tensor_name);
        }
    }

    const int32_t tensor_count = required_i32(loader, "kokoro.model.tensor_count");
    const auto tensor_names = loader.get_str_array("kokoro.model.tensor_names");
    const auto tensor_shapes = loader.get_str_array("kokoro.model.tensor_shapes");
    const auto tensor_types = loader.get_str_array("kokoro.model.tensor_types");
    if (tensor_count != 463 || tensor_names.size() != static_cast<size_t>(tensor_count) ||
        tensor_shapes.size() != tensor_names.size() || tensor_types.size() != tensor_names.size()) {
        throw std::runtime_error("Kokoro GGUF tensor manifest arrays are inconsistent");
    }
    std::unordered_set<std::string> unique_tensors;
    for (size_t index = 0; index < tensor_names.size(); ++index) {
        const std::string& name = tensor_names[index];
        if (name.rfind("kokoro.", 0) != 0 || !unique_tensors.insert(name).second) {
            throw std::runtime_error("invalid or duplicate Kokoro model tensor name: " + name);
        }
        if (!loader.has_tensor(name)) {
            throw std::runtime_error("Kokoro GGUF is missing required tensor: " + name);
        }
        if (loader.get_tensor_shape(name) != parse_shape(tensor_shapes[index], name)) {
            throw std::runtime_error("Kokoro tensor shape mismatch: " + name);
        }
        if (loader.get_tensor_type(name) != parse_type(tensor_types[index], name)) {
            throw std::runtime_error("Kokoro tensor type mismatch: " + name);
        }
    }

    std::unordered_set<std::string> expected_tensors = unique_tensors;
    for (const std::string& voice : voice_names_) {
        expected_tensors.insert("kokoro.voice." + voice);
    }
    const auto loaded_tensor_names = loader.get_tensor_names();
    if (loaded_tensor_names.size() != expected_tensors.size()) {
        throw std::runtime_error("Kokoro GGUF contains an unexpected tensor count");
    }
    for (const std::string& name : loaded_tensor_names) {
        if (!expected_tensors.count(name)) {
            throw std::runtime_error("Kokoro GGUF contains an unexpected tensor: " + name);
        }
    }
}

size_t
KokoroModelMetadata::voice_index(const std::string& raw_name) const {
    const std::string name = KokoroTokenizer::canonicalize_voice(raw_name);
    const auto found = std::find(voice_names_.begin(), voice_names_.end(), name);
    if (found == voice_names_.end()) {
        throw std::invalid_argument("Kokoro model contains no voice named '" + raw_name + "'");
    }
    return static_cast<size_t>(std::distance(voice_names_.begin(), found));
}

std::string
KokoroModelMetadata::voice_tensor_name(size_t index) const {
    if (index >= voice_names_.size()) {
        throw std::out_of_range("Kokoro voice index is out of range");
    }
    return "kokoro.voice." + voice_names_[index];
}

std::vector<float>
KokoroModelMetadata::read_voice_style(
    ggml_runtime::GGUFLoader& loader, const std::string& voice,
    size_t unframed_phoneme_count) const {
    if (unframed_phoneme_count < 1 ||
        unframed_phoneme_count > static_cast<size_t>(hparams_.phoneme_limit)) {
        throw std::invalid_argument("Kokoro voice row requires 1-510 unframed phoneme IDs");
    }
    const std::string tensor_name = voice_tensor_name(voice_index(voice));
    const size_t row_values = static_cast<size_t>(hparams_.voice_style_dim);
    const size_t total_values = row_values * static_cast<size_t>(hparams_.phoneme_limit);
    const char* bytes = loader.get_tensor_file_data(tensor_name, total_values * sizeof(float));
    std::vector<float> style(row_values);
    const size_t row = unframed_phoneme_count - 1;
    std::memcpy(style.data(), bytes + row * row_values * sizeof(float), row_values * sizeof(float));
    if (!std::all_of(
            style.begin(), style.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("Kokoro voice style contains a non-finite value");
    }
    return style;
}

const std::string&
KokoroModelMetadata::misaki_lexicon_json(const std::string& name) const {
    const auto found = misaki_lexicons_.find(name);
    if (found == misaki_lexicons_.end()) {
        throw std::invalid_argument("unknown Misaki lexicon: " + name);
    }
    return found->second;
}

EnglishLexiconData
KokoroModelMetadata::english_lexicon_data() const {
    EnglishLexiconData data;
    data.us_gold_json = misaki_lexicon_json("us_gold");
    data.us_silver_json = misaki_lexicon_json("us_silver");
    data.gb_gold_json = misaki_lexicon_json("gb_gold");
    data.gb_silver_json = misaki_lexicon_json("gb_silver");
    return data;
}

}  // namespace nemo_speech::tts::kokoro
