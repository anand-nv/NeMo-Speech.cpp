// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "model.h"

#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "runtime/ggml/runtime.h"

namespace nemo_speech::tts::omnivoice {
namespace {

namespace rt = ggml_runtime;

constexpr int32_t kTextVocab = 151676;
constexpr int32_t kBaseVocab = 151643;
constexpr int32_t kAudioVocab = 1025;
constexpr int32_t kAudioMask = 1024;
constexpr int32_t kCodebooks = 8;
constexpr int32_t kHidden = 1024;
constexpr int32_t kLayers = 28;
constexpr int32_t kHeads = 16;
constexpr int32_t kKvHeads = 8;
constexpr int32_t kHeadSize = 128;
constexpr int32_t kFeedForward = 3072;
constexpr int32_t kContext = 40960;
constexpr int32_t kAddedTokens = 33;
constexpr int32_t kMergeCount = 151387;
constexpr int32_t kLanguageCount = 646;
constexpr const char* kPinnedRevision = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4";
constexpr const char* kPinnedModelSha =
    "730839316de585f4c8298ec0e1712efc10fb19c6fa4e36eb741cb8d51ebcf6aa";
constexpr const char* kPinnedTokenizerSha =
    "408f669b7e2b045fdf54201d815bd364e6667dbd845115da81239c40bc6dcfd1";
constexpr const char* kPinnedAudioTokenizerSha =
    "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d";

int32_t
required_i32(const rt::GGUFLoader& loader, const char* key) {
    if (!loader.has_key(key)) {
        throw std::runtime_error(std::string("OmniVoice GGUF is missing metadata: ") + key);
    }
    return loader.get_i32(key);
}

float
required_f32(const rt::GGUFLoader& loader, const char* key) {
    if (!loader.has_key(key)) {
        throw std::runtime_error(std::string("OmniVoice GGUF is missing metadata: ") + key);
    }
    return loader.get_f32(key);
}

std::string
required_string(const rt::GGUFLoader& loader, const char* key) {
    if (!loader.has_key(key)) {
        throw std::runtime_error(std::string("OmniVoice GGUF is missing metadata: ") + key);
    }
    return loader.get_str(key);
}

template <typename T>
void
require_equal(const char* key, const T& actual, const T& expected) {
    if (actual != expected) {
        std::ostringstream out;
        out << "unsupported OmniVoice metadata " << key << "=" << actual << ", expected "
            << expected;
        throw std::runtime_error(out.str());
    }
}

void
require_float(const char* key, float actual, float expected) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > std::fabs(expected) * 1.0e-6f) {
        std::ostringstream out;
        out << "unsupported OmniVoice metadata " << key << "=" << actual << ", expected "
            << expected;
        throw std::runtime_error(out.str());
    }
}

void
require_array_size(
    const rt::GGUFLoader& loader, const char* key, size_t expected, bool strings = false) {
    if (!loader.has_key(key)) {
        throw std::runtime_error(std::string("OmniVoice GGUF is missing metadata: ") + key);
    }
    const size_t actual =
        strings ? loader.get_str_array(key).size() : loader.get_i32_array(key).size();
    if (actual != expected) {
        throw std::runtime_error(
            std::string("OmniVoice metadata ") + key + " has " + std::to_string(actual) +
            " entries, expected " + std::to_string(expected));
    }
}

struct TensorSpec {
    std::string name;
    ggml_type type;
    bool allow_float_matrix;
    int rank;
    std::array<int64_t, GGML_MAX_DIMS> shape;
};

TensorSpec
tensor(
    std::string name, ggml_type type, bool allow_float_matrix, int rank, int64_t ne0,
    int64_t ne1 = 1) {
    return {std::move(name), type, allow_float_matrix, rank, {ne0, ne1, 1, 1}};
}

std::vector<TensorSpec>
expected_tensors() {
    std::vector<TensorSpec> specs;
    specs.reserve(313);
    specs.push_back(tensor("codebook_layer_offsets", GGML_TYPE_I64, false, 1, kCodebooks));
    specs.push_back(tensor(
        "audio_embeddings.weight", GGML_TYPE_F32, true, 2, kHidden, kCodebooks * kAudioVocab));
    specs.push_back(
        tensor("audio_heads.weight", GGML_TYPE_F32, true, 2, kHidden, kCodebooks * kAudioVocab));
    specs.push_back(tensor("llm.embed_tokens.weight", GGML_TYPE_F32, true, 2, kHidden, kTextVocab));
    specs.push_back(tensor("llm.norm.weight", GGML_TYPE_F32, false, 1, kHidden));

    for (int layer = 0; layer < kLayers; ++layer) {
        const std::string p = "llm.layers." + std::to_string(layer);
        specs.push_back(tensor(p + ".input_layernorm.weight", GGML_TYPE_F32, false, 1, kHidden));
        specs.push_back(
            tensor(p + ".post_attention_layernorm.weight", GGML_TYPE_F32, false, 1, kHidden));
        specs.push_back(
            tensor(p + ".mlp.down_proj.weight", GGML_TYPE_F32, true, 2, kFeedForward, kHidden));
        specs.push_back(
            tensor(p + ".mlp.gate_proj.weight", GGML_TYPE_F32, true, 2, kHidden, kFeedForward));
        specs.push_back(
            tensor(p + ".mlp.up_proj.weight", GGML_TYPE_F32, true, 2, kHidden, kFeedForward));
        specs.push_back(tensor(
            p + ".self_attn.q_proj.weight", GGML_TYPE_F32, true, 2, kHidden, kHeads * kHeadSize));
        specs.push_back(tensor(
            p + ".self_attn.k_proj.weight", GGML_TYPE_F32, true, 2, kHidden, kKvHeads * kHeadSize));
        specs.push_back(tensor(
            p + ".self_attn.v_proj.weight", GGML_TYPE_F32, true, 2, kHidden, kKvHeads * kHeadSize));
        specs.push_back(tensor(
            p + ".self_attn.o_proj.weight", GGML_TYPE_F32, true, 2, kHeads * kHeadSize, kHidden));
        specs.push_back(tensor(p + ".self_attn.q_norm.weight", GGML_TYPE_F32, false, 1, kHeadSize));
        specs.push_back(tensor(p + ".self_attn.k_norm.weight", GGML_TYPE_F32, false, 1, kHeadSize));
    }
    return specs;
}

void
validate_tensor_manifest(rt::GGUFLoader& loader) {
    const std::vector<TensorSpec> specs = expected_tensors();
    require_equal("tensor count", specs.size(), size_t{313});
    const auto names = loader.get_tensor_names();
    const std::set<std::string> actual(names.begin(), names.end());

    for (const auto& spec : specs) {
        if (!loader.has_tensor(spec.name)) {
            throw std::runtime_error("OmniVoice GGUF is missing tensor: " + spec.name);
        }
        const ggml_type type = loader.get_tensor_type(spec.name);
        if (spec.allow_float_matrix) {
            if (type != GGML_TYPE_F16 && type != GGML_TYPE_F32) {
                throw std::runtime_error(
                    "OmniVoice tensor " + spec.name + " has unsupported type " +
                    std::to_string(static_cast<int>(type)) + "; expected F16 or F32");
            }
        } else if (type != spec.type) {
            throw std::runtime_error(
                "OmniVoice tensor " + spec.name + " has type " +
                std::to_string(static_cast<int>(type)) + ", expected " +
                std::to_string(static_cast<int>(spec.type)));
        }
        const int rank = loader.get_tensor_n_dims(spec.name);
        if (rank != spec.rank) {
            throw std::runtime_error(
                "OmniVoice tensor " + spec.name + " has rank " + std::to_string(rank) +
                ", expected " + std::to_string(spec.rank));
        }
        const auto shape = loader.get_tensor_shape(spec.name);
        if (shape != spec.shape) {
            std::ostringstream out;
            out << "OmniVoice tensor " << spec.name << " has shape [" << shape[0] << "," << shape[1]
                << "," << shape[2] << "," << shape[3] << "], expected [" << spec.shape[0] << ","
                << spec.shape[1] << "," << spec.shape[2] << "," << spec.shape[3] << "]";
            throw std::runtime_error(out.str());
        }
    }

    if (actual.size() != specs.size()) {
        std::set<std::string> expected;
        for (const auto& spec : specs) expected.insert(spec.name);
        for (const auto& name : actual) {
            if (expected.count(name) == 0) {
                throw std::runtime_error("OmniVoice GGUF has unexpected tensor: " + name);
            }
        }
        throw std::runtime_error(
            "OmniVoice GGUF tensor count is " + std::to_string(actual.size()) + ", expected " +
            std::to_string(specs.size()));
    }
}

}  // namespace

ModelConfig
load_model_config(rt::GGUFLoader& loader) {
    require_equal(
        "general.architecture", required_string(loader, "general.architecture"),
        std::string("omnivoice"));

    ModelConfig cfg;
    cfg.text_vocab_size = required_i32(loader, "omnivoice.text_vocab_size");
    cfg.audio_vocab_size = required_i32(loader, "omnivoice.audio_vocab_size");
    cfg.audio_mask_id = required_i32(loader, "omnivoice.audio_mask_id");
    cfg.audio_codebooks = required_i32(loader, "omnivoice.audio_codebooks");
    cfg.hidden_size = required_i32(loader, "omnivoice.embedding_length");
    cfg.layer_count = required_i32(loader, "omnivoice.block_count");
    cfg.attention_heads = required_i32(loader, "omnivoice.attention.head_count");
    cfg.kv_heads = required_i32(loader, "omnivoice.attention.head_count_kv");
    cfg.head_size = required_i32(loader, "omnivoice.attention.key_length");
    cfg.feed_forward_size = required_i32(loader, "omnivoice.feed_forward_length");
    cfg.context_length = required_i32(loader, "omnivoice.context_length");
    cfg.rms_epsilon = required_f32(loader, "omnivoice.attention.layer_norm_rms_epsilon");
    cfg.rope_theta = required_f32(loader, "omnivoice.rope.freq_base");
    cfg.pad_token_id = required_i32(loader, "omnivoice.tokenizer.pad_id");
    cfg.eos_token_id = required_i32(loader, "omnivoice.tokenizer.eos_id");
    cfg.audio_codebook_weights = loader.get_i32_array("omnivoice.audio_codebook_weights");
    cfg.audio_codebook_offsets = loader.get_i32_array("omnivoice.audio_codebook_offsets");
    cfg.source_revision = required_string(loader, "omnivoice.source.revision");
    cfg.source_model_sha256 = required_string(loader, "omnivoice.source.model_sha256");
    cfg.source_tokenizer_sha256 = required_string(loader, "omnivoice.source.tokenizer_sha256");
    cfg.source_audio_tokenizer_sha256 =
        required_string(loader, "omnivoice.source.audio_tokenizer_sha256");

    require_equal("text_vocab_size", cfg.text_vocab_size, kTextVocab);
    require_equal(
        "tokenizer.base_vocab_size", required_i32(loader, "omnivoice.tokenizer.base_vocab_size"),
        kBaseVocab);
    require_equal("audio_vocab_size", cfg.audio_vocab_size, kAudioVocab);
    require_equal("audio_mask_id", cfg.audio_mask_id, kAudioMask);
    require_equal("audio_codebooks", cfg.audio_codebooks, kCodebooks);
    require_equal("embedding_length", cfg.hidden_size, kHidden);
    require_equal("block_count", cfg.layer_count, kLayers);
    require_equal("attention.head_count", cfg.attention_heads, kHeads);
    require_equal("attention.head_count_kv", cfg.kv_heads, kKvHeads);
    require_equal("attention.key_length", cfg.head_size, kHeadSize);
    require_equal("feed_forward_length", cfg.feed_forward_size, kFeedForward);
    require_equal("context_length", cfg.context_length, kContext);
    require_float("attention.layer_norm_rms_epsilon", cfg.rms_epsilon, 1.0e-6f);
    require_float("rope.freq_base", cfg.rope_theta, 1.0e6f);
    require_equal("tokenizer.pad_id", cfg.pad_token_id, 151643);
    require_equal("tokenizer.eos_id", cfg.eos_token_id, 151645);
    if (cfg.audio_codebook_weights != std::vector<int32_t>({8, 8, 6, 6, 4, 4, 2, 2})) {
        throw std::runtime_error("unsupported OmniVoice metadata audio_codebook_weights");
    }
    if (cfg.audio_codebook_offsets !=
        std::vector<int32_t>({0, 1025, 2050, 3075, 4100, 5125, 6150, 7175})) {
        throw std::runtime_error("unsupported OmniVoice metadata audio_codebook_offsets");
    }

    require_equal(
        "tokenizer.model", required_string(loader, "omnivoice.tokenizer.model"),
        std::string("BPE"));
    require_equal(
        "tokenizer.normalizer", required_string(loader, "omnivoice.tokenizer.normalizer"),
        std::string("NFC"));
    require_equal(
        "tokenizer.pre_tokenizer", required_string(loader, "omnivoice.tokenizer.pre_tokenizer"),
        std::string("Qwen2 regex + ByteLevel"));
    require_array_size(loader, "omnivoice.tokenizer.tokens", kTextVocab, true);
    require_array_size(loader, "omnivoice.tokenizer.merge_left", kMergeCount, true);
    require_array_size(loader, "omnivoice.tokenizer.merge_right", kMergeCount, true);
    require_array_size(loader, "omnivoice.tokenizer.added.ids", kAddedTokens);
    require_array_size(loader, "omnivoice.tokenizer.added.content", kAddedTokens, true);
    require_array_size(loader, "omnivoice.languages.ids", kLanguageCount, true);
    require_array_size(loader, "omnivoice.languages.names", kLanguageCount, true);

    require_equal("source.revision", cfg.source_revision, std::string(kPinnedRevision));
    require_equal("source.model_sha256", cfg.source_model_sha256, std::string(kPinnedModelSha));
    require_equal(
        "source.tokenizer_sha256", cfg.source_tokenizer_sha256, std::string(kPinnedTokenizerSha));
    require_equal(
        "source.audio_tokenizer_sha256", cfg.source_audio_tokenizer_sha256,
        std::string(kPinnedAudioTokenizerSha));

    validate_tensor_manifest(loader);
    return cfg;
}

}  // namespace nemo_speech::tts::omnivoice
