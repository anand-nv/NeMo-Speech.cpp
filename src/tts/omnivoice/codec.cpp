// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "codec.h"

#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/ggml/runtime.h"

namespace nemo_speech::tts::omnivoice {
namespace {

namespace rt = ggml_runtime;

constexpr int32_t kSampleRate = 24000;
constexpr int32_t kHopLength = 960;
constexpr int32_t kFrameRate = 25;
constexpr int32_t kCodebookSize = 1024;
constexpr int32_t kCodebookDim = 64;
constexpr int32_t kQuantizers = 8;
constexpr int32_t kCombinedHidden = 1024;
constexpr int32_t kDacHidden = 256;
constexpr int32_t kDacDecoderHidden = 1024;
constexpr size_t kConvertedTensorCount = 502;
constexpr const char* kPinnedRevision = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4";
constexpr const char* kPinnedModelSha =
    "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d";

int32_t
required_i32(const rt::GGUFLoader& loader, const char* key) {
    if (!loader.has_key(key)) {
        throw std::runtime_error(std::string("Higgs Audio V2 GGUF is missing metadata: ") + key);
    }
    return loader.get_i32(key);
}

float
required_f32(const rt::GGUFLoader& loader, const char* key) {
    if (!loader.has_key(key)) {
        throw std::runtime_error(std::string("Higgs Audio V2 GGUF is missing metadata: ") + key);
    }
    return loader.get_f32(key);
}

bool
required_bool(const rt::GGUFLoader& loader, const char* key) {
    if (!loader.has_key(key)) {
        throw std::runtime_error(std::string("Higgs Audio V2 GGUF is missing metadata: ") + key);
    }
    return loader.get_bool(key);
}

std::string
required_string(const rt::GGUFLoader& loader, const char* key) {
    if (!loader.has_key(key)) {
        throw std::runtime_error(std::string("Higgs Audio V2 GGUF is missing metadata: ") + key);
    }
    return loader.get_str(key);
}

template <typename T>
void
require_equal(const char* key, const T& actual, const T& expected) {
    if (actual != expected) {
        std::ostringstream out;
        out << "unsupported Higgs Audio V2 metadata " << key << "=" << actual << ", expected "
            << expected;
        throw std::runtime_error(out.str());
    }
}

template <typename T>
void
require_equal(const char* key, const std::vector<T>& actual, const std::vector<T>& expected) {
    if (actual != expected) {
        throw std::runtime_error(std::string("unsupported Higgs Audio V2 metadata array ") + key);
    }
}

void
require_float(const char* key, float actual, float expected) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > 1.0e-5f) {
        std::ostringstream out;
        out << "unsupported Higgs Audio V2 metadata " << key << "=" << actual << ", expected "
            << expected;
        throw std::runtime_error(out.str());
    }
}

struct TensorSpec {
    std::string name;
    int rank;
    std::array<int64_t, GGML_MAX_DIMS> shape;
    bool f32_only;
};

TensorSpec
tensor(
    std::string name, int rank, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1,
    bool f32_only = false) {
    return {std::move(name), rank, {ne0, ne1, ne2, 1}, f32_only};
}

std::vector<TensorSpec>
decoder_tensors() {
    std::vector<TensorSpec> specs;
    specs.reserve(136);
    for (int q = 0; q < kQuantizers; ++q) {
        const std::string p = "rvq." + std::to_string(q);
        specs.push_back(tensor(p + ".cb.embed", 2, kCodebookDim, kCodebookSize, 1, true));
        specs.push_back(tensor(p + ".out.weight", 2, kCodebookDim, kCombinedHidden));
        specs.push_back(tensor(p + ".out.bias", 1, kCombinedHidden, 1, 1, true));
    }
    specs.push_back(tensor("fc2.weight", 2, kCombinedHidden, kDacHidden));
    specs.push_back(tensor("fc2.bias", 1, kDacHidden, 1, 1, true));
    specs.push_back(tensor("adec.conv1.weight", 3, 7, kDacHidden, kDacDecoderHidden));
    specs.push_back(tensor("adec.conv1.bias", 1, kDacDecoderHidden, 1, 1, true));

    constexpr std::array<int32_t, 5> in_channels = {1024, 512, 256, 128, 64};
    constexpr std::array<int32_t, 5> out_channels = {512, 256, 128, 64, 32};
    constexpr std::array<int32_t, 5> strides = {8, 5, 4, 2, 3};
    constexpr std::array<int32_t, 3> dilations = {1, 3, 9};
    for (int stage = 0; stage < 5; ++stage) {
        const std::string p = "adec.block." + std::to_string(stage);
        specs.push_back(tensor(p + ".snake1.alpha", 2, 1, in_channels[stage], 1, true));
        specs.push_back(tensor(
            p + ".conv_t1.weight", 3, 2 * strides[stage], out_channels[stage], in_channels[stage]));
        specs.push_back(tensor(p + ".conv_t1.bias", 1, out_channels[stage], 1, 1, true));
        for (int unit = 0; unit < 3; ++unit) {
            const std::string r = p + ".ru" + std::to_string(unit + 1);
            specs.push_back(tensor(r + ".snake1.alpha", 2, 1, out_channels[stage], 1, true));
            specs.push_back(
                tensor(r + ".conv1.weight", 3, 7, out_channels[stage], out_channels[stage]));
            specs.push_back(tensor(r + ".conv1.bias", 1, out_channels[stage], 1, 1, true));
            specs.push_back(tensor(r + ".snake2.alpha", 2, 1, out_channels[stage], 1, true));
            specs.push_back(
                tensor(r + ".conv2.weight", 3, 1, out_channels[stage], out_channels[stage]));
            specs.push_back(tensor(r + ".conv2.bias", 1, out_channels[stage], 1, 1, true));
            (void)dilations[unit];
        }
    }
    specs.push_back(tensor("adec.snake1.alpha", 2, 1, 32, 1, true));
    specs.push_back(tensor("adec.conv2.weight", 2, 7, 32, 1));
    specs.push_back(tensor("adec.conv2.bias", 1, 1, 1, 1, true));
    return specs;
}

void
validate_decoder_tensors(rt::GGUFLoader& loader) {
    const auto names = loader.get_tensor_names();
    if (names.size() != kConvertedTensorCount) {
        throw std::runtime_error(
            "Higgs Audio V2 GGUF tensor count is " + std::to_string(names.size()) + ", expected " +
            std::to_string(kConvertedTensorCount));
    }
    for (const auto& spec : decoder_tensors()) {
        if (!loader.has_tensor(spec.name)) {
            throw std::runtime_error("Higgs Audio V2 GGUF is missing decoder tensor: " + spec.name);
        }
        const ggml_type type = loader.get_tensor_type(spec.name);
        if ((spec.f32_only && type != GGML_TYPE_F32) ||
            (!spec.f32_only && type != GGML_TYPE_F16 && type != GGML_TYPE_F32)) {
            throw std::runtime_error(
                "Higgs Audio V2 decoder tensor " + spec.name + " has unsupported type " +
                std::to_string(static_cast<int>(type)));
        }
        if (loader.get_tensor_n_dims(spec.name) != spec.rank) {
            throw std::runtime_error(
                "Higgs Audio V2 decoder tensor " + spec.name + " has rank " +
                std::to_string(loader.get_tensor_n_dims(spec.name)) + ", expected " +
                std::to_string(spec.rank));
        }
        const auto shape = loader.get_tensor_shape(spec.name);
        if (shape != spec.shape) {
            std::ostringstream out;
            out << "Higgs Audio V2 decoder tensor " << spec.name << " has shape [" << shape[0]
                << "," << shape[1] << "," << shape[2] << "," << shape[3] << "], expected ["
                << spec.shape[0] << "," << spec.shape[1] << "," << spec.shape[2] << ","
                << spec.shape[3] << "]";
            throw std::runtime_error(out.str());
        }
    }
}

std::string
rvq_name(int q, const char* suffix) {
    return "rvq." + std::to_string(q) + "." + suffix;
}

std::string
block_name(int stage, const std::string& suffix) {
    return "adec.block." + std::to_string(stage) + "." + suffix;
}

ggml_tensor*
weight(rt::Session* session, const std::string& name) {
    return session->model_tensor_container->get_tensor_by_name(name).tensor;
}

ggml_tensor*
bias_3d(ggml_context* ctx, rt::Session* session, const std::string& name) {
    ggml_tensor* bias = weight(session, name);
    return ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
}

ggml_tensor*
conv1d(
    ggml_context* ctx, rt::Session* session, ggml_tensor* input, const std::string& prefix,
    int padding, int dilation = 1) {
    ggml_tensor* kernel = weight(session, prefix + ".weight");
    ggml_tensor* output = nullptr;
    if (kernel->type == GGML_TYPE_F16) {
        output = ggml_conv_1d(ctx, kernel, input, 1, padding, dilation);
    } else {
        ggml_tensor* columns =
            ggml_im2col(ctx, kernel, input, 1, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
        output = ggml_mul_mat(
            ctx, ggml_reshape_2d(ctx, columns, columns->ne[0], columns->ne[2] * columns->ne[1]),
            ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
        output = ggml_reshape_3d(ctx, output, columns->ne[1], kernel->ne[2], 1);
    }
    return ggml_add(ctx, output, bias_3d(ctx, session, prefix + ".bias"));
}

ggml_tensor*
snake(ggml_context* ctx, rt::Session* session, ggml_tensor* input, const std::string& name) {
    ggml_tensor* alpha = weight(session, name);
    ggml_tensor* phase = ggml_mul(ctx, input, alpha);
    ggml_tensor* periodic = ggml_sqr(ctx, ggml_sin(ctx, phase));
    ggml_tensor* epsilon = weight(session, "omnivoice.codec.snake_epsilon");
    periodic = ggml_div(ctx, periodic, ggml_add(ctx, alpha, epsilon));
    return ggml_add(ctx, input, periodic);
}

ggml_tensor*
conv_transpose1d(
    ggml_context* ctx, rt::Session* session, ggml_tensor* input, const std::string& prefix,
    int stride) {
    ggml_tensor* conv_weight = weight(session, prefix + ".weight");
    // GGML's transpose-convolution operator currently requires F32 weights and
    // exposes padding=0 only. PyTorch's padding/output_padding pair is exactly
    // represented by cropping the full convolution on both sides.
    conv_weight = ggml_cont(ctx, ggml_cast(ctx, conv_weight, GGML_TYPE_F32));
    ggml_tensor* full = ggml_conv_transpose_1d(ctx, conv_weight, input, stride, 0, 1);
    const int64_t out_length = input->ne[0] * stride;
    const size_t left_crop = static_cast<size_t>((stride + 1) / 2) * full->nb[0];
    ggml_tensor* cropped = ggml_view_3d(
        ctx, full, out_length, conv_weight->ne[1], 1, full->nb[1], full->nb[2], left_crop);
    return ggml_add(ctx, cropped, bias_3d(ctx, session, prefix + ".bias"));
}

class CodecDecoderModule final : public rt::Module {
   public:
    explicit CodecDecoderModule(CodecConfig config) : config_(std::move(config)) {}

    void define_tensors(rt::Session* session) override {
        auto* tc = session->model_tensor_container.get();
        auto* loader = session->gguf_loader;
        tc->create_tensor_1d("omnivoice.codec.snake_epsilon", GGML_TYPE_F32, 1);
        for (const auto& spec : decoder_tensors()) {
            const ggml_type type = loader->get_tensor_type(spec.name);
            if (spec.rank == 1) {
                tc->create_tensor_1d(spec.name, type, spec.shape[0]);
            } else if (spec.rank == 2) {
                tc->create_tensor_2d(spec.name, type, spec.shape[0], spec.shape[1]);
            } else {
                tc->create_tensor_3d(spec.name, type, spec.shape[0], spec.shape[1], spec.shape[2]);
            }
        }
    }

    void set_data(rt::Session* session) override {
        const float epsilon = 1.0e-9f;
        ggml_backend_tensor_set(
            weight(session, "omnivoice.codec.snake_epsilon"), &epsilon, 0, sizeof(epsilon));
        for (const auto& spec : decoder_tensors()) session->load_weight(spec.name);
    }

    rt::TensorBag build_graph(
        rt::Session* session, rt::TensorBag, rt::TensorContainer* tc) override {
        auto first = session->model_tensor_container->get_tensor_by_name("rvq.0.cb.embed");
        ggml_context* ctx = tc->get_ctx_of_buffer_type(first.buft).ctx;
        ggml_tensor* codes = tc->get_tensor_by_name("omnivoice.codec.codes").tensor;
        const int64_t frames = codes->ne[0];

        ggml_tensor* quantized = nullptr;
        for (int q = 0; q < config_.quantizers; ++q) {
            ggml_tensor* ids =
                ggml_view_1d(ctx, codes, frames, static_cast<size_t>(q) * codes->nb[1]);
            ggml_tensor* embedded =
                ggml_get_rows(ctx, weight(session, rvq_name(q, "cb.embed")), ids);
            ggml_tensor* projected =
                ggml_mul_mat(ctx, weight(session, rvq_name(q, "out.weight")), embedded);
            projected = ggml_add(ctx, projected, weight(session, rvq_name(q, "out.bias")));
            quantized = quantized == nullptr ? projected : ggml_add(ctx, quantized, projected);
        }

        ggml_tensor* hidden = ggml_mul_mat(ctx, weight(session, "fc2.weight"), quantized);
        hidden = ggml_add(ctx, hidden, weight(session, "fc2.bias"));
        hidden = ggml_cont_3d(ctx, ggml_transpose(ctx, hidden), frames, config_.dac_hidden_size, 1);
        hidden = conv1d(ctx, session, hidden, "adec.conv1", 3);

        constexpr std::array<int32_t, 5> strides = {8, 5, 4, 2, 3};
        constexpr std::array<int32_t, 3> dilations = {1, 3, 9};
        for (int stage = 0; stage < 5; ++stage) {
            const std::string p = block_name(stage, "");
            hidden = snake(ctx, session, hidden, p + "snake1.alpha");
            hidden = conv_transpose1d(ctx, session, hidden, p + "conv_t1", strides[stage]);
            for (int unit = 0; unit < 3; ++unit) {
                const std::string r = p + "ru" + std::to_string(unit + 1) + ".";
                ggml_tensor* residual = hidden;
                ggml_tensor* update = snake(ctx, session, hidden, r + "snake1.alpha");
                update =
                    conv1d(ctx, session, update, r + "conv1", 3 * dilations[unit], dilations[unit]);
                update = snake(ctx, session, update, r + "snake2.alpha");
                update = conv1d(ctx, session, update, r + "conv2", 0);
                hidden = ggml_add(ctx, residual, update);
            }
        }

        hidden = snake(ctx, session, hidden, "adec.snake1.alpha");
        ggml_tensor* waveform = conv1d(ctx, session, hidden, "adec.conv2", 3);
        waveform = ggml_reshape_1d(ctx, waveform, frames * config_.hop_length);
        ggml_set_name(waveform, "omnivoice.codec.waveform");
        ggml_set_output(waveform);
        rt::TensorBag output;
        output.add_tensor(rt::ggml_bf_tensor(waveform, first.buft));
        return output;
    }

   private:
    CodecConfig config_;
};

}  // namespace

CodecConfig
load_codec_config(rt::GGUFLoader& loader) {
    require_equal(
        "general.architecture", required_string(loader, "general.architecture"),
        std::string("higgs-audio-v2-tokenizer"));

    CodecConfig cfg;
    cfg.sample_rate = required_i32(loader, "higgs_audio_v2.sample_rate");
    cfg.hop_length = required_i32(loader, "higgs_audio_v2.hop_length");
    cfg.frame_rate = required_f32(loader, "higgs_audio_v2.frame_rate");
    cfg.codebook_size = required_i32(loader, "higgs_audio_v2.codebook_size");
    cfg.codebook_dim = required_i32(loader, "higgs_audio_v2.codebook_dim");
    cfg.quantizers = required_i32(loader, "higgs_audio_v2.num_quantizers");
    cfg.semantic_sample_rate = required_i32(loader, "higgs_audio_v2.semantic.sample_rate");
    cfg.semantic_pad_samples = required_i32(loader, "higgs_audio_v2.semantic.pad_samples");
    cfg.semantic_downsample_factor =
        required_i32(loader, "higgs_audio_v2.semantic.downsample_factor");
    cfg.hubert_hidden_size = required_i32(loader, "higgs_audio_v2.hubert.hidden_size");
    cfg.hubert_intermediate_size = required_i32(loader, "higgs_audio_v2.hubert.intermediate_size");
    cfg.hubert_layers = required_i32(loader, "higgs_audio_v2.hubert.layers");
    cfg.hubert_heads = required_i32(loader, "higgs_audio_v2.hubert.heads");
    cfg.hubert_layer_norm_epsilon = required_f32(loader, "higgs_audio_v2.hubert.layer_norm_eps");
    cfg.dac_encoder_hidden_size = required_i32(loader, "higgs_audio_v2.dac.encoder_hidden_size");
    cfg.dac_decoder_hidden_size = required_i32(loader, "higgs_audio_v2.dac.decoder_hidden_size");
    cfg.dac_hidden_size = required_i32(loader, "higgs_audio_v2.dac.hidden_size");
    cfg.dac_downsampling_ratios = loader.get_i32_array("higgs_audio_v2.dac.downsampling_ratios");
    cfg.dac_upsampling_ratios = loader.get_i32_array("higgs_audio_v2.dac.upsampling_ratios");
    cfg.dac_residual_dilations = loader.get_i32_array("higgs_audio_v2.dac.residual_dilations");
    cfg.source_revision = required_string(loader, "higgs_audio_v2.source.revision");
    cfg.source_model_sha256 = required_string(loader, "higgs_audio_v2.source.model_sha256");

    require_equal("sample_rate", cfg.sample_rate, kSampleRate);
    require_equal("hop_length", cfg.hop_length, kHopLength);
    require_float("frame_rate", cfg.frame_rate, static_cast<float>(kFrameRate));
    require_equal("codebook_size", cfg.codebook_size, kCodebookSize);
    require_equal("codebook_dim", cfg.codebook_dim, kCodebookDim);
    require_equal("num_quantizers", cfg.quantizers, kQuantizers);
    require_equal("semantic.sample_rate", cfg.semantic_sample_rate, 16000);
    require_equal("semantic.pad_samples", cfg.semantic_pad_samples, 160);
    require_equal("semantic.downsample_factor", cfg.semantic_downsample_factor, 2);
    require_equal("hubert.hidden_size", cfg.hubert_hidden_size, 768);
    require_equal("hubert.intermediate_size", cfg.hubert_intermediate_size, 3072);
    require_equal("hubert.layers", cfg.hubert_layers, 12);
    require_equal("hubert.heads", cfg.hubert_heads, 12);
    require_float("hubert.layer_norm_eps", cfg.hubert_layer_norm_epsilon, 1.0e-5f);
    require_equal("dac.encoder_hidden_size", cfg.dac_encoder_hidden_size, 64);
    require_equal("dac.decoder_hidden_size", cfg.dac_decoder_hidden_size, kDacDecoderHidden);
    require_equal("dac.hidden_size", cfg.dac_hidden_size, kDacHidden);
    require_equal(
        "dac.downsampling_ratios", cfg.dac_downsampling_ratios,
        std::vector<int32_t>({8, 5, 4, 2, 3}));
    require_equal(
        "dac.upsampling_ratios", cfg.dac_upsampling_ratios, std::vector<int32_t>({8, 5, 4, 2, 3}));
    require_equal(
        "dac.residual_dilations", cfg.dac_residual_dilations, std::vector<int32_t>({1, 3, 9}));
    require_equal(
        "target_bandwidths", loader.get_f32_array("higgs_audio_v2.target_bandwidths"),
        std::vector<float>({0.5f, 1.0f, 1.5f, 2.0f}));
    require_equal(
        "semantic.strides", loader.get_i32_array("higgs_audio_v2.semantic.strides"),
        std::vector<int32_t>({1, 1}));
    require_equal(
        "semantic.channel_ratios", loader.get_i32_array("higgs_audio_v2.semantic.channel_ratios"),
        std::vector<int32_t>({1, 1}));
    require_equal(
        "semantic.block_dilations", loader.get_i32_array("higgs_audio_v2.semantic.block_dilations"),
        std::vector<int32_t>({1, 1}));
    require_equal(
        "semantic.kernel_size", required_i32(loader, "higgs_audio_v2.semantic.kernel_size"), 3);
    require_equal(
        "semantic.unit_kernel_size",
        required_i32(loader, "higgs_audio_v2.semantic.unit_kernel_size"), 3);
    require_equal(
        "dac.input_kernel_size", required_i32(loader, "higgs_audio_v2.dac.input_kernel_size"), 7);
    require_equal(
        "dac.output_kernel_size", required_i32(loader, "higgs_audio_v2.dac.output_kernel_size"), 7);
    require_equal(
        "dac.decoder_stride_parity_output_padding",
        required_bool(loader, "higgs_audio_v2.dac.decoder_stride_parity_output_padding"), true);
    require_equal("dac.final_tanh", required_bool(loader, "higgs_audio_v2.dac.final_tanh"), false);
    require_equal("source.revision", cfg.source_revision, std::string(kPinnedRevision));
    require_equal("source.model_sha256", cfg.source_model_sha256, std::string(kPinnedModelSha));

    validate_decoder_tensors(loader);
    return cfg;
}

CodecDecoder::CodecDecoder(rt::BackendManager& backends, const std::string& gguf_path) {
    loader_ = std::make_unique<rt::GGUFLoader>(gguf_path);
    config_ = load_codec_config(*loader_);
    module_ = std::make_unique<CodecDecoderModule>(config_);
    session_ = std::make_unique<rt::Session>(backends, module_.get(), loader_.get());
    session_->set_run_cache_capacity(4);
    session_->setup();
}

CodecDecoder::~CodecDecoder() = default;

std::vector<float>
CodecDecoder::decode(const std::array<std::vector<int32_t>, 8>& codes) {
    const size_t frames = codes[0].size();
    if (frames == 0) {
        throw std::invalid_argument("Higgs Audio V2 decode requires at least one frame");
    }
    std::vector<int32_t> packed;
    packed.reserve(frames * config_.quantizers);
    for (const auto& codebook : codes) {
        if (codebook.size() != frames) {
            throw std::invalid_argument("Higgs Audio V2 codebooks must have equal frame counts");
        }
        for (int32_t id : codebook) {
            if (id < 0 || id >= config_.codebook_size) {
                throw std::invalid_argument("Higgs Audio V2 code ID is out of range");
            }
            packed.push_back(id);
        }
    }
    if (frames > static_cast<size_t>(std::numeric_limits<int32_t>::max() / config_.hop_length)) {
        throw std::invalid_argument("Higgs Audio V2 decode request is too long");
    }

    std::vector<float> waveform(frames * config_.hop_length);
    std::vector<rt::Session::Input> inputs = {
        {"omnivoice.codec.codes",
         GGML_TYPE_I32,
         packed.data(),
         {static_cast<int64_t>(frames), config_.quantizers}}};
    std::vector<rt::Session::Output> outputs = {
        {0, "", waveform.data(), waveform.size() * sizeof(float)}};
    session_->run(inputs, outputs);
    if (outputs[0].out_shape[0] != static_cast<int64_t>(waveform.size()) ||
        outputs[0].out_shape[1] != 1 || outputs[0].out_shape[2] != 1 ||
        outputs[0].out_shape[3] != 1) {
        throw std::runtime_error("Higgs Audio V2 decoder returned an unexpected waveform shape");
    }
    return waveform;
}

}  // namespace nemo_speech::tts::omnivoice
