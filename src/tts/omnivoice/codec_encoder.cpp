// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "codec.h"
#include "runtime/ggml/runtime.h"

namespace nemo_speech::tts::omnivoice {
namespace {

namespace rt = ggml_runtime;

constexpr int kHubertHidden = 768;
constexpr int kHubertFfn = 3072;
constexpr int kHubertHeads = 12;
constexpr int kHubertHeadSize = 64;
constexpr int kHubertLayers = 12;
constexpr int kDacHidden = 256;
constexpr int kCombinedHidden = 1024;

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
encoder_tensors() {
    std::vector<TensorSpec> specs;
    specs.reserve(370);

    constexpr std::array<int, 7> kernels = {10, 3, 3, 3, 3, 2, 2};
    for (int layer = 0; layer < 7; ++layer) {
        const int input_channels = layer == 0 ? 1 : 512;
        const std::string p = "hubert.fe." + std::to_string(layer);
        specs.push_back(tensor(p + ".conv.weight", 3, kernels[layer], input_channels, 512));
        if (layer == 0) {
            specs.push_back(tensor(p + ".attn_norm.weight", 1, 512, 1, 1, true));
            specs.push_back(tensor(p + ".attn_norm.bias", 1, 512, 1, 1, true));
        }
    }
    specs.push_back(tensor("hubert.fp.attn_norm.weight", 1, 512, 1, 1, true));
    specs.push_back(tensor("hubert.fp.attn_norm.bias", 1, 512, 1, 1, true));
    specs.push_back(tensor("hubert.fp.projection.weight", 2, 512, kHubertHidden));
    specs.push_back(tensor("hubert.fp.projection.bias", 1, kHubertHidden, 1, 1, true));
    specs.push_back(tensor("hubert.pos_conv.weight", 3, 128, 48, kHubertHidden));
    specs.push_back(tensor("hubert.pos_conv.bias", 1, kHubertHidden, 1, 1, true));
    specs.push_back(tensor("hubert.out_norm.weight", 1, kHubertHidden, 1, 1, true));
    specs.push_back(tensor("hubert.out_norm.bias", 1, kHubertHidden, 1, 1, true));

    for (int layer = 0; layer < kHubertLayers; ++layer) {
        const std::string p = "hubert.blk." + std::to_string(layer);
        for (const char* projection : {"q", "k", "v", "out"}) {
            specs.push_back(tensor(p + ".attn." + projection + ".weight", 2, 768, 768));
            specs.push_back(tensor(p + ".attn." + projection + ".bias", 1, 768, 1, 1, true));
        }
        specs.push_back(tensor(p + ".ffn.up.weight", 2, 768, kHubertFfn));
        specs.push_back(tensor(p + ".ffn.up.bias", 1, kHubertFfn, 1, 1, true));
        specs.push_back(tensor(p + ".ffn.down.weight", 2, kHubertFfn, 768));
        specs.push_back(tensor(p + ".ffn.down.bias", 1, 768, 1, 1, true));
        for (const char* norm : {"attn_norm", "final_norm"}) {
            specs.push_back(tensor(p + "." + norm + ".weight", 1, 768, 1, 1, true));
            specs.push_back(tensor(p + "." + norm + ".bias", 1, 768, 1, 1, true));
        }
    }

    specs.push_back(tensor("senc.conv.weight", 3, 3, 768, 768));
    for (int block = 0; block < 2; ++block) {
        const std::string p = "senc.blk." + std::to_string(block);
        for (int unit = 0; unit < 2; ++unit) {
            const std::string r = p + ".rus." + std::to_string(unit);
            specs.push_back(tensor(r + ".conv1.weight", 3, 3, 768, 768));
            specs.push_back(tensor(r + ".conv2.weight", 3, 1, 768, 768));
        }
        specs.push_back(tensor(p + ".conv.weight", 3, 3, 768, 768));
        specs.push_back(tensor(p + ".conv.bias", 1, 768, 1, 1, true));
    }

    specs.push_back(tensor("aenc.conv1.weight", 3, 7, 1, 64));
    specs.push_back(tensor("aenc.conv1.bias", 1, 64, 1, 1, true));
    constexpr std::array<int, 5> input_channels = {64, 128, 256, 512, 1024};
    constexpr std::array<int, 5> strides = {8, 5, 4, 2, 3};
    for (int stage = 0; stage < 5; ++stage) {
        const int channels = input_channels[stage];
        const std::string p = "aenc.block." + std::to_string(stage);
        for (int unit = 0; unit < 3; ++unit) {
            const std::string r = p + ".ru" + std::to_string(unit + 1);
            specs.push_back(tensor(r + ".snake1.alpha", 2, 1, channels, 1, true));
            specs.push_back(tensor(r + ".conv1.weight", 3, 7, channels, channels));
            specs.push_back(tensor(r + ".conv1.bias", 1, channels, 1, 1, true));
            specs.push_back(tensor(r + ".snake2.alpha", 2, 1, channels, 1, true));
            specs.push_back(tensor(r + ".conv2.weight", 3, 1, channels, channels));
            specs.push_back(tensor(r + ".conv2.bias", 1, channels, 1, 1, true));
        }
        specs.push_back(tensor(p + ".snake1.alpha", 2, 1, channels, 1, true));
        specs.push_back(tensor(p + ".conv1.weight", 3, 2 * strides[stage], channels, 2 * channels));
        specs.push_back(tensor(p + ".conv1.bias", 1, 2 * channels, 1, 1, true));
    }
    specs.push_back(tensor("aenc.snake1.alpha", 2, 1, 2048, 1, true));
    specs.push_back(tensor("aenc.conv2.weight", 3, 3, 2048, kDacHidden));
    specs.push_back(tensor("aenc.conv2.bias", 1, kDacHidden, 1, 1, true));
    specs.push_back(tensor("fc.weight", 2, kCombinedHidden, kCombinedHidden));
    specs.push_back(tensor("fc.bias", 1, kCombinedHidden, 1, 1, true));

    for (int q = 0; q < 8; ++q) {
        const std::string p = "rvq." + std::to_string(q);
        specs.push_back(tensor(p + ".in.weight", 2, kCombinedHidden, 64));
        specs.push_back(tensor(p + ".in.bias", 1, 64, 1, 1, true));
        specs.push_back(tensor(p + ".cb.embed", 2, 64, 1024, 1, true));
        specs.push_back(tensor(p + ".out.weight", 2, 64, kCombinedHidden));
        specs.push_back(tensor(p + ".out.bias", 1, kCombinedHidden, 1, 1, true));
    }
    return specs;
}

void
validate_encoder_tensors(rt::GGUFLoader& loader) {
    for (const auto& spec : encoder_tensors()) {
        if (!loader.has_tensor(spec.name)) {
            throw std::runtime_error("Higgs Audio V2 GGUF is missing encoder tensor: " + spec.name);
        }
        const ggml_type type = loader.get_tensor_type(spec.name);
        if ((spec.f32_only && type != GGML_TYPE_F32) ||
            (!spec.f32_only && type != GGML_TYPE_F16 && type != GGML_TYPE_F32)) {
            throw std::runtime_error(
                "Higgs Audio V2 encoder tensor " + spec.name + " has unsupported type " +
                std::to_string(static_cast<int>(type)));
        }
        if (loader.get_tensor_n_dims(spec.name) != spec.rank ||
            loader.get_tensor_shape(spec.name) != spec.shape) {
            const auto shape = loader.get_tensor_shape(spec.name);
            std::ostringstream out;
            out << "Higgs Audio V2 encoder tensor " << spec.name << " has rank/shape "
                << loader.get_tensor_n_dims(spec.name) << " [" << shape[0] << "," << shape[1] << ","
                << shape[2] << "," << shape[3] << "]";
            throw std::runtime_error(out.str());
        }
    }
}

std::vector<float>
torchaudio_resample_24k_to_16k(const std::vector<float>& input) {
    constexpr int orig = 3;
    constexpr int dest = 2;
    constexpr int lowpass_width = 6;
    constexpr float rolloff = 0.99f;
    constexpr float pi = 3.14159265358979323846f;
    const float base = static_cast<float>(std::min(orig, dest)) * rolloff;
    const int width = static_cast<int>(std::ceil(lowpass_width * orig / base));
    const int kernel_width = 2 * width + orig;
    std::array<std::vector<float>, dest> kernels;
    for (int phase = 0; phase < dest; ++phase) {
        kernels[phase].resize(kernel_width);
        for (int index = 0; index < kernel_width; ++index) {
            float t = -static_cast<float>(phase) / dest + static_cast<float>(index - width) / orig;
            t = std::clamp(
                t * base, -static_cast<float>(lowpass_width), static_cast<float>(lowpass_width));
            const float window = std::pow(std::cos(t * pi / lowpass_width / 2.0f), 2.0f);
            const float angle = t * pi;
            const float sinc = angle == 0.0f ? 1.0f : std::sin(angle) / angle;
            kernels[phase][index] = sinc * window * (base / orig);
        }
    }
    const size_t target = (dest * input.size() + orig - 1) / orig;
    std::vector<float> output;
    output.reserve(target);
    for (size_t block = 0; output.size() < target; ++block) {
        for (int phase = 0; phase < dest && output.size() < target; ++phase) {
            float sum = 0.0f;
            const int64_t start = static_cast<int64_t>(block * orig) - width;
            for (int tap = 0; tap < kernel_width; ++tap) {
                const int64_t source = start + tap;
                if (source >= 0 && source < static_cast<int64_t>(input.size())) {
                    sum += input[static_cast<size_t>(source)] * kernels[phase][tap];
                }
            }
            output.push_back(sum);
        }
    }
    return output;
}

std::string
hubert_block(int layer, const std::string& suffix) {
    return "hubert.blk." + std::to_string(layer) + "." + suffix;
}

ggml_tensor*
weight(rt::Session* session, const std::string& name) {
    return session->model_tensor_container->get_tensor_by_name(name).tensor;
}

ggml_tensor*
bias_3d(ggml_context* ctx, rt::Session* session, const std::string& name) {
    ggml_tensor* value = weight(session, name);
    return ggml_reshape_3d(ctx, value, 1, value->ne[0], 1);
}

ggml_tensor*
linear(ggml_context* ctx, rt::Session* session, ggml_tensor* input, const std::string& prefix) {
    ggml_tensor* output = ggml_mul_mat(ctx, weight(session, prefix + ".weight"), input);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    return ggml_add(ctx, output, weight(session, prefix + ".bias"));
}

ggml_tensor*
layer_norm(
    ggml_context* ctx, rt::Session* session, ggml_tensor* input, const std::string& prefix,
    float epsilon = 1.0e-5f) {
    ggml_tensor* output = ggml_norm(ctx, input, epsilon);
    output = ggml_mul(ctx, output, weight(session, prefix + ".weight"));
    return ggml_add(ctx, output, weight(session, prefix + ".bias"));
}

ggml_tensor*
conv1d(
    ggml_context* ctx, rt::Session* session, ggml_tensor* input, const std::string& prefix,
    int stride, int padding, int dilation, bool has_bias = true) {
    ggml_tensor* kernel = weight(session, prefix + ".weight");
    if (kernel->type == GGML_TYPE_F16)
        kernel = ggml_cast(ctx, kernel, GGML_TYPE_F32);
    ggml_tensor* columns =
        ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor* output = ggml_mul_mat(
        ctx, ggml_reshape_2d(ctx, columns, columns->ne[0], columns->ne[2] * columns->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    output = ggml_reshape_3d(ctx, output, columns->ne[1], kernel->ne[2], 1);
    if (has_bias)
        output = ggml_add(ctx, output, bias_3d(ctx, session, prefix + ".bias"));
    return output;
}

ggml_tensor*
snake(ggml_context* ctx, rt::Session* session, ggml_tensor* input, const std::string& name) {
    ggml_tensor* alpha = weight(session, name);
    ggml_tensor* periodic = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, input, alpha)));
    periodic = ggml_div(
        ctx, periodic, ggml_add(ctx, alpha, weight(session, "omnivoice.codec.snake_epsilon")));
    return ggml_add(ctx, input, periodic);
}

ggml_tensor*
dac_residual(
    ggml_context* ctx, rt::Session* session, ggml_tensor* input, const std::string& prefix,
    int dilation) {
    ggml_tensor* update = snake(ctx, session, input, prefix + ".snake1.alpha");
    update = conv1d(ctx, session, update, prefix + ".conv1", 1, 3 * dilation, dilation);
    update = snake(ctx, session, update, prefix + ".snake2.alpha");
    update = conv1d(ctx, session, update, prefix + ".conv2", 1, 0, 1);
    return ggml_add(ctx, input, update);
}

ggml_tensor*
hubert_positional_embedding(
    ggml_context* ctx, rt::Session* session, ggml_tensor* hidden, int64_t frames) {
    ggml_tensor* input = ggml_cont_3d(ctx, ggml_transpose(ctx, hidden), frames, kHubertHidden, 1);
    ggml_tensor* kernel = weight(session, "hubert.pos_conv.weight");
    ggml_tensor* result = nullptr;
    constexpr int groups = 16;
    constexpr int channels_per_group = kHubertHidden / groups;
    for (int group = 0; group < groups; ++group) {
        ggml_tensor* input_group = ggml_view_3d(
            ctx, input, frames, channels_per_group, 1, input->nb[1], input->nb[2],
            static_cast<size_t>(group * channels_per_group) * input->nb[1]);
        ggml_tensor* kernel_group = ggml_view_3d(
            ctx, kernel, 128, channels_per_group, channels_per_group, kernel->nb[1], kernel->nb[2],
            static_cast<size_t>(group * channels_per_group) * kernel->nb[2]);
        if (kernel_group->type == GGML_TYPE_F16)
            kernel_group = ggml_cast(ctx, kernel_group, GGML_TYPE_F32);
        ggml_tensor* columns =
            ggml_im2col(ctx, kernel_group, input_group, 1, 0, 64, 0, 1, 0, false, GGML_TYPE_F32);
        ggml_tensor* convolved = ggml_mul_mat(
            ctx, ggml_reshape_2d(ctx, columns, columns->ne[0], columns->ne[2] * columns->ne[1]),
            ggml_reshape_2d(
                ctx, kernel_group, kernel_group->ne[0] * kernel_group->ne[1], kernel_group->ne[2]));
        ggml_mul_mat_set_prec(convolved, GGML_PREC_F32);
        convolved = ggml_reshape_3d(ctx, convolved, columns->ne[1], kernel_group->ne[2], 1);
        convolved = ggml_view_3d(
            ctx, convolved, frames, channels_per_group, 1, convolved->nb[1], convolved->nb[2], 0);
        result = result == nullptr ? convolved : ggml_concat(ctx, result, convolved, 1);
    }
    result = ggml_add(ctx, result, bias_3d(ctx, session, "hubert.pos_conv.bias"));
    result = ggml_gelu_erf(ctx, result);
    return ggml_cont_3d(ctx, ggml_transpose(ctx, result), kHubertHidden, frames, 1);
}

class CodecEncoderModule final : public rt::Module {
   public:
    explicit CodecEncoderModule(CodecConfig config) : config_(std::move(config)) {}

    void define_tensors(rt::Session* session) override {
        auto* tc = session->model_tensor_container.get();
        auto* loader = session->gguf_loader;
        tc->create_tensor_1d("omnivoice.codec.snake_epsilon", GGML_TYPE_F32, 1);
        for (const auto& spec : encoder_tensors()) {
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
        for (const auto& spec : encoder_tensors()) session->load_weight(spec.name);
    }

    rt::TensorBag build_graph(
        rt::Session* session, rt::TensorBag, rt::TensorContainer* tc) override {
        auto first = session->model_tensor_container->get_tensor_by_name("hubert.fe.0.conv.weight");
        ggml_context* ctx = tc->get_ctx_of_buffer_type(first.buft).ctx;
        ggml_tensor* semantic_pcm = tc->get_tensor_by_name("omnivoice.codec.semantic_pcm").tensor;
        ggml_tensor* acoustic_pcm = tc->get_tensor_by_name("omnivoice.codec.acoustic_pcm").tensor;

        // HuBERT convolutional feature extractor.
        ggml_tensor* feature = ggml_reshape_3d(ctx, semantic_pcm, semantic_pcm->ne[0], 1, 1);
        constexpr std::array<int, 7> kernels = {10, 3, 3, 3, 3, 2, 2};
        constexpr std::array<int, 7> strides = {5, 2, 2, 2, 2, 2, 2};
        for (int layer = 0; layer < 7; ++layer) {
            const std::string p = "hubert.fe." + std::to_string(layer);
            feature = conv1d(ctx, session, feature, p + ".conv", strides[layer], 0, 1, false);
            if (layer == 0) {
                feature = ggml_norm(ctx, feature, 1.0e-5f);
                feature = ggml_mul(ctx, feature, bias_3d(ctx, session, p + ".attn_norm.weight"));
                feature = ggml_add(ctx, feature, bias_3d(ctx, session, p + ".attn_norm.bias"));
            }
            feature = ggml_gelu_erf(ctx, feature);
            (void)kernels[layer];
        }
        const int64_t hubert_frames = feature->ne[0];
        feature = ggml_cont_3d(ctx, ggml_transpose(ctx, feature), 512, hubert_frames, 1);
        feature = layer_norm(ctx, session, feature, "hubert.fp.attn_norm");
        ggml_tensor* hidden = linear(ctx, session, feature, "hubert.fp.projection");

        hidden =
            ggml_add(ctx, hidden, hubert_positional_embedding(ctx, session, hidden, hubert_frames));
        hidden = layer_norm(ctx, session, hidden, "hubert.out_norm");
        ggml_tensor* hidden_sum = hidden;

        const float attention_scale = 1.0f / std::sqrt(static_cast<float>(kHubertHeadSize));
        for (int layer = 0; layer < kHubertLayers; ++layer) {
            ggml_tensor* residual = hidden;
            ggml_tensor* query = linear(ctx, session, hidden, hubert_block(layer, "attn.q"));
            ggml_tensor* key = linear(ctx, session, hidden, hubert_block(layer, "attn.k"));
            ggml_tensor* value = linear(ctx, session, hidden, hubert_block(layer, "attn.v"));
            query = ggml_permute(
                ctx, ggml_cont_3d(ctx, query, kHubertHeadSize, kHubertHeads, hubert_frames), 0, 2,
                1, 3);
            key = ggml_permute(
                ctx, ggml_cont_3d(ctx, key, kHubertHeadSize, kHubertHeads, hubert_frames), 0, 2, 1,
                3);
            ggml_tensor* scores = ggml_mul_mat(ctx, key, query);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            scores = ggml_soft_max(ctx, ggml_scale(ctx, scores, attention_scale));
            ggml_tensor* value_transposed = ggml_cont_3d(
                ctx,
                ggml_permute(
                    ctx, ggml_cont_3d(ctx, value, kHubertHeadSize, kHubertHeads, hubert_frames), 1,
                    2, 0, 3),
                hubert_frames, kHubertHeadSize, kHubertHeads);
            ggml_tensor* attended = ggml_mul_mat(ctx, value_transposed, scores);
            ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
            attended = ggml_cont_2d(
                ctx, ggml_permute(ctx, attended, 0, 2, 1, 3), kHubertHidden, hubert_frames);
            attended = linear(ctx, session, attended, hubert_block(layer, "attn.out"));
            hidden = layer_norm(
                ctx, session, ggml_add(ctx, residual, attended), hubert_block(layer, "attn_norm"));
            residual = hidden;
            ggml_tensor* ffn = linear(ctx, session, hidden, hubert_block(layer, "ffn.up"));
            ffn = ggml_gelu_erf(ctx, ffn);
            ffn = linear(ctx, session, ffn, hubert_block(layer, "ffn.down"));
            hidden = layer_norm(
                ctx, session, ggml_add(ctx, residual, ffn), hubert_block(layer, "final_norm"));
            hidden_sum = ggml_add(ctx, hidden_sum, hidden);
        }
        hidden = ggml_scale(ctx, hidden_sum, 1.0f / 13.0f);
        const int64_t semantic_frames = (hubert_frames + 1) / 2;
        hidden = ggml_view_2d(ctx, hidden, kHubertHidden, semantic_frames, hidden->nb[1] * 2, 0);
        ggml_tensor* semantic =
            ggml_cont_3d(ctx, ggml_transpose(ctx, hidden), semantic_frames, kHubertHidden, 1);

        semantic = conv1d(ctx, session, semantic, "senc.conv", 1, 1, 1, false);
        for (int block = 0; block < 2; ++block) {
            const std::string p = "senc.blk." + std::to_string(block);
            for (int unit = 0; unit < 2; ++unit) {
                const std::string r = p + ".rus." + std::to_string(unit);
                ggml_tensor* update = ggml_elu(ctx, semantic);
                update = conv1d(ctx, session, update, r + ".conv1", 1, 1, 1, false);
                update = ggml_elu(ctx, update);
                update = conv1d(ctx, session, update, r + ".conv2", 1, 0, 1, false);
                semantic = ggml_add(ctx, semantic, update);
            }
            semantic = conv1d(ctx, session, semantic, p + ".conv", 1, 1, 1);
        }

        // DAC acoustic encoder.
        ggml_tensor* acoustic = ggml_reshape_3d(ctx, acoustic_pcm, acoustic_pcm->ne[0], 1, 1);
        acoustic = conv1d(ctx, session, acoustic, "aenc.conv1", 1, 3, 1);
        constexpr std::array<int, 5> down_strides = {8, 5, 4, 2, 3};
        constexpr std::array<int, 3> dilations = {1, 3, 9};
        for (int stage = 0; stage < 5; ++stage) {
            const std::string p = "aenc.block." + std::to_string(stage);
            for (int unit = 0; unit < 3; ++unit) {
                acoustic = dac_residual(
                    ctx, session, acoustic, p + ".ru" + std::to_string(unit + 1), dilations[unit]);
            }
            acoustic = snake(ctx, session, acoustic, p + ".snake1.alpha");
            acoustic = conv1d(
                ctx, session, acoustic, p + ".conv1", down_strides[stage],
                (down_strides[stage] + 1) / 2, 1);
        }
        acoustic = snake(ctx, session, acoustic, "aenc.snake1.alpha");
        acoustic = conv1d(ctx, session, acoustic, "aenc.conv2", 1, 1, 1);
        if (acoustic->ne[0] != semantic->ne[0]) {
            throw std::runtime_error("Higgs Audio V2 semantic/acoustic frame mismatch");
        }

        ggml_tensor* embeddings = ggml_concat(ctx, acoustic, semantic, 1);
        const int64_t output_frames = embeddings->ne[0];
        embeddings =
            ggml_cont_3d(ctx, ggml_transpose(ctx, embeddings), kCombinedHidden, output_frames, 1);
        embeddings = linear(ctx, session, embeddings, "fc");

        ggml_tensor* residual = embeddings;
        rt::TensorBag output;
        for (int q = 0; q < 8; ++q) {
            const std::string p = "rvq." + std::to_string(q);
            ggml_tensor* projected = linear(ctx, session, residual, p + ".in");
            ggml_tensor* codebook = weight(session, p + ".cb.embed");
            ggml_tensor* similarity = ggml_scale(ctx, ggml_mul_mat(ctx, codebook, projected), 2.0f);
            ggml_mul_mat_set_prec(similarity->src[0], GGML_PREC_F32);
            ggml_tensor* codebook_norm = ggml_sum_rows(ctx, ggml_sqr(ctx, codebook));
            codebook_norm = ggml_cont(ctx, ggml_transpose(ctx, codebook_norm));
            similarity = ggml_sub(ctx, similarity, codebook_norm);
            ggml_tensor* indices = ggml_argmax(ctx, similarity);
            indices = ggml_reshape_2d(ctx, indices, output_frames, 1);
            ggml_set_name(indices, ("omnivoice.codec.codebook." + std::to_string(q)).c_str());
            ggml_set_output(indices);
            output.add_tensor(rt::ggml_bf_tensor(indices, first.buft));

            ggml_tensor* quantized = ggml_get_rows(ctx, codebook, indices);
            quantized = linear(ctx, session, quantized, p + ".out");
            residual = ggml_sub(ctx, residual, quantized);
        }
        return output;
    }

   private:
    CodecConfig config_;
};

}  // namespace

CodecEncoder::CodecEncoder(rt::BackendManager& backends, const std::string& gguf_path) {
    loader_ = std::make_unique<rt::GGUFLoader>(gguf_path);
    config_ = load_codec_config(*loader_);
    validate_encoder_tensors(*loader_);
    module_ = std::make_unique<CodecEncoderModule>(config_);
    session_ = std::make_unique<rt::Session>(backends, module_.get(), loader_.get());
    session_->set_run_cache_capacity(2);
    session_->setup();
}

CodecEncoder::~CodecEncoder() = default;

std::array<std::vector<int32_t>, 8>
CodecEncoder::encode(const std::vector<float>& mono_24khz) {
    if (mono_24khz.empty() || mono_24khz.size() % config_.hop_length != 0) {
        throw std::invalid_argument(
            "Higgs Audio V2 encoder requires a positive multiple of 960 samples");
    }
    if (!std::all_of(mono_24khz.begin(), mono_24khz.end(), [](float sample) {
            return std::isfinite(sample);
        })) {
        throw std::invalid_argument("Higgs Audio V2 encoder input contains non-finite samples");
    }
    std::vector<float> semantic = torchaudio_resample_24k_to_16k(mono_24khz);
    semantic.insert(semantic.begin(), config_.semantic_pad_samples, 0.0f);
    semantic.insert(semantic.end(), config_.semantic_pad_samples, 0.0f);
    const size_t frames = mono_24khz.size() / config_.hop_length;
    if (semantic.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        mono_24khz.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("Higgs Audio V2 encoder input is too long");
    }

    std::array<std::vector<int32_t>, 8> result;
    std::vector<rt::Session::Input> inputs = {
        {"omnivoice.codec.semantic_pcm",
         GGML_TYPE_F32,
         semantic.data(),
         {static_cast<int64_t>(semantic.size())}},
        {"omnivoice.codec.acoustic_pcm",
         GGML_TYPE_F32,
         mono_24khz.data(),
         {static_cast<int64_t>(mono_24khz.size())}},
    };
    std::vector<rt::Session::Output> outputs;
    outputs.reserve(config_.quantizers);
    for (int q = 0; q < config_.quantizers; ++q) {
        result[static_cast<size_t>(q)].resize(frames);
        outputs.push_back({q, "", result[static_cast<size_t>(q)].data(), frames * sizeof(int32_t)});
    }
    session_->run(inputs, outputs);
    for (int q = 0; q < config_.quantizers; ++q) {
        if (outputs[static_cast<size_t>(q)].out_shape[0] != static_cast<int64_t>(frames) ||
            outputs[static_cast<size_t>(q)].out_shape[1] != 1) {
            throw std::runtime_error("Higgs Audio V2 encoder returned an unexpected code shape");
        }
        for (int32_t code : result[static_cast<size_t>(q)]) {
            if (code < 0 || code >= config_.codebook_size) {
                throw std::runtime_error("Higgs Audio V2 encoder produced an invalid code");
            }
        }
    }
    return result;
}

}  // namespace nemo_speech::tts::omnivoice
