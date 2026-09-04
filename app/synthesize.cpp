// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "commands.h"
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#include "audio_file.h"
#include "cli_util.h"
#include "config.h"
#include "engine_registry.h"
#include "json.h"
#include "model_utils.h"
#include "parameter_parser.h"
#include "tts/omnivoice/prompt.h"

namespace {

void
write_audio(const std::filesystem::path& path, const std::string& audio, bool force) {
    if (path == "-") {
#if defined(_WIN32)
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        if (std::fwrite(audio.data(), 1, audio.size(), stdout) != audio.size())
            throw std::runtime_error("could not write synthesized audio to stdout");
        return;
    }
    if (!force && std::filesystem::exists(path))
        throw std::runtime_error(path.string() + " already exists (use --force to replace it)");
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot write " + path.string());
    output.write(audio.data(), static_cast<std::streamsize>(audio.size()));
    output.close();
    if (!output)
        throw std::runtime_error("cannot finish writing " + path.string());
    std::error_code error;
    if (force)
        std::filesystem::remove(path, error);
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot finish writing " + path.string() + ": " + error.message());
    }
}

}  // namespace

void
print_synthesize_help(const char* program) {
    std::printf(
        "Usage: %s synthesize TEXT [options]\n\n"
        "Options:\n"
        "  --magpie-model MODEL      MagpieTTS GGUF path or indexed HF repo\n"
        "                            (default: nvidia/magpie_tts_multilingual_357m)\n"
        "  --codec-model MODEL       NanoCodec GGUF path or indexed HF repo\n"
        "                            (default: nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)\n"
        "  --tokenizer-dir MODEL     Tokenizer directory or indexed HF repo\n"
        "                            (default: MagpieTTS repository)\n"
        "  --omnivoice-model MODEL   OmniVoice denoiser GGUF path\n"
        "  --audio-tokenizer-model MODEL\n"
        "                            Higgs Audio V2 tokenizer GGUF path\n"
        "  --tn-model-dir DIR        Optional text-normalization grammars\n"
        "  -i, --input PATH          Read text from a UTF-8 file\n"
        "  -o, --output PATH         Output path (default: speech.wav; '-' = stdout)\n"
        "  --format wav|pcm          WAV container or raw signed PCM16\n"
        "  --language CODE           Text language (default: en-US)\n"
        "  --voice NAME              Voice name or speaker index\n"
        "  --instruction TEXT        OmniVoice voice-design instruction\n"
        "  --prompt-wav PATH         OmniVoice voice-clone reference WAV\n"
        "  --prompt-text TEXT        Required reference transcript\n"
        "  --prompt-file PATH        Load a reusable encoded voice prompt\n"
        "  --save-prompt PATH        Save the encoded voice prompt\n"
        "  --speaker N               Speaker index\n"
        "  --sample-rate HZ          Output rate (8 kHz through model rate)\n"
        "  --device, --backend DEVICE\n"
        "                            auto, cpu, cuda[:N], metal, or vulkan[:N]\n"
        "  --seed N --steps N --top-k N --temperature N --cfg-scale N\n"
        "  --speed N --duration N --position-temperature N --class-temperature N\n"
        "  --config FILE             Load the complete TTS YAML config tree\n"
        "  --tts.KEY VALUE           Override any C++ TTS setting\n"
        "  --no-warmup               Skip warmup\n"
        "  --force                   Replace an existing WAV\n",
        program);
}

int
command_synthesize(int argc, char** argv) {
    try {
        if (argc > 0 && is_help_argument(argv[0])) {
            print_synthesize_help("nemo-speech");
            return 0;
        }
        nemo_speech::tts::MagpieTtsServerConfig parsed;
        nemo_speech::common::ParameterParser parser;
        parser.Register("tts", parsed);
        std::string config_file;
        for (int i = 0; i < argc; ++i) {
            if (std::string(argv[i]) == "--config") {
                if (++i >= argc)
                    throw std::invalid_argument("--config requires a value");
                config_file = argv[i];
            }
        }
        if (!config_file.empty())
            parser.ApplyYaml(config_file);
        parser.ApplyEnv("NEMO_SPEECH");

        std::string text, input_path, output_path = "speech.wav", language, voice;
        std::string instruction, prompt_wav, prompt_text, prompt_file, save_prompt;
        std::string format = "wav";
        bool force = false;
        bool warmup = true;
        int output_rate = 0;
        bool device_set = false;
        std::string device_name = "auto";
        int gpu = default_gpu_index();
        nemo_speech::tts::MagpieSynthesisOptions request_options;
        std::optional<nemo_speech::tts::OmniVoiceOptions> request_omnivoice;
        auto omni_options = [&]() -> nemo_speech::tts::OmniVoiceOptions& {
            if (!request_omnivoice)
                request_omnivoice =
                    parsed.omnivoice_options.value_or(nemo_speech::tts::OmniVoiceOptions{});
            return *request_omnivoice;
        };
        auto value = [&](int& i, const std::string& option) {
            if (++i >= argc)
                throw std::invalid_argument(option + " requires a value");
            return std::string(argv[i]);
        };
        for (int i = 0; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--config")
                ++i;
            else if (arg == "--magpie-model")
                parsed.runtime.magpie_model = value(i, arg);
            else if (arg == "--codec-model")
                parsed.runtime.codec_model = value(i, arg);
            else if (arg == "--omnivoice-model")
                parsed.omnivoice_model = value(i, arg);
            else if (arg == "--audio-tokenizer-model")
                parsed.omnivoice_audio_tokenizer_model = value(i, arg);
            else if (arg == "--tokenizer-dir")
                parsed.tokenizer_model_dir = value(i, arg);
            else if (arg == "--tn-model-dir")
                parsed.tn_model_dir = value(i, arg);
            else if (arg == "--input" || arg == "-i")
                input_path = value(i, arg);
            else if (arg == "--output" || arg == "-o")
                output_path = value(i, arg);
            else if (arg == "--format")
                format = value(i, arg);
            else if (arg == "--language")
                language = value(i, arg);
            else if (arg == "--voice")
                voice = value(i, arg);
            else if (arg == "--instruction")
                instruction = value(i, arg);
            else if (arg == "--prompt-wav")
                prompt_wav = value(i, arg);
            else if (arg == "--prompt-text")
                prompt_text = value(i, arg);
            else if (arg == "--prompt-file")
                prompt_file = value(i, arg);
            else if (arg == "--save-prompt")
                save_prompt = value(i, arg);
            else if (arg == "--speaker")
                request_options.speaker = parse_int(value(i, arg), arg, 0, 100000);
            else if (arg == "--sample-rate")
                output_rate = parse_int(value(i, arg), arg, 8000, 192000);
            else if (arg == "--device" || arg == "--backend") {
                device_name = value(i, arg);
                gpu = parse_device(device_name, arg);
                device_set = true;
            } else if (arg == "--seed")
                request_options.seed = parse_int(value(i, arg), arg, -1, 2147483647);
            else if (arg == "--steps")
                request_options.steps = omni_options().num_steps =
                    parse_int(value(i, arg), arg, 1, 1000000);
            else if (arg == "--top-k")
                request_options.top_k = parse_int(value(i, arg), arg, 1, 1000000);
            else if (arg == "--temperature") {
                request_options.temperature = static_cast<float>(parse_double(value(i, arg), arg));
                request_options.override_temperature = true;
            } else if (arg == "--cfg-scale") {
                request_options.cfg_scale = static_cast<float>(parse_double(value(i, arg), arg));
                request_options.override_cfg_scale = true;
                omni_options().guidance_scale = request_options.cfg_scale;
            } else if (arg == "--speed") {
                omni_options().speed = parse_double(value(i, arg), arg);
            } else if (arg == "--duration") {
                omni_options().duration_s = parse_double(value(i, arg), arg);
            } else if (arg == "--position-temperature") {
                omni_options().position_temperature =
                    static_cast<float>(parse_double(value(i, arg), arg));
            } else if (arg == "--class-temperature") {
                omni_options().class_temperature =
                    static_cast<float>(parse_double(value(i, arg), arg));
            } else if (arg == "--no-warmup")
                warmup = false;
            else if (arg == "--force")
                force = true;
            else if (!arg.empty() && arg[0] == '-') {
                bool consumed = false;
                if (!parser.ParseCliArg(arg, i + 1 < argc ? argv[i + 1] : nullptr, &consumed))
                    throw std::invalid_argument("unknown option: " + arg);
                if (consumed)
                    ++i;
            } else if (text.empty())
                text = arg;
            else
                text += " " + arg;
        }
        if (!input_path.empty()) {
            if (!text.empty())
                throw std::invalid_argument("TEXT and --input cannot be used together");
            text = read_text_file(input_path);
        }
        const bool prompt_only = text.empty() && !save_prompt.empty();
        if (text.empty() && !prompt_only)
            throw std::invalid_argument("TEXT is required");
        if (format != "wav" && format != "pcm")
            throw std::invalid_argument("--format must be wav or pcm");
        if (cli_json() && output_path == "-")
            throw std::invalid_argument("--json cannot be combined with --output -");

        const bool any_omnivoice =
            !parsed.omnivoice_model.empty() || !parsed.omnivoice_audio_tokenizer_model.empty();
        if (any_omnivoice) {
            parsed.omnivoice_model =
                require_model_file(parsed.omnivoice_model, "OmniVoice model").string();
            parsed.omnivoice_audio_tokenizer_model =
                require_model_file(
                    parsed.omnivoice_audio_tokenizer_model, "Higgs Audio V2 tokenizer model")
                    .string();
            if (!parsed.runtime.magpie_model.empty() || !parsed.runtime.codec_model.empty() ||
                !parsed.tokenizer_model_dir.empty())
                throw std::invalid_argument(
                    "Magpie and OmniVoice model options are mutually exclusive");
        } else {
            parsed.runtime.magpie_model =
                resolve_model_file(parsed.runtime.magpie_model, "tts", "MagpieTTS model").string();
            parsed.runtime.codec_model =
                resolve_model_file(parsed.runtime.codec_model, "codec", "NanoCodec model").string();
            parsed.tokenizer_model_dir =
                resolve_model_directory(parsed.tokenizer_model_dir, "tokenizer", "tokenizer model")
                    .string();
        }
        if (!parsed.tn_model_dir.empty())
            parsed.tn_model_dir =
                require_model_directory(parsed.tn_model_dir, "text normalization model").string();
        if (device_set) {
            bool cuda_device = device_name == "cuda" || device_name.rfind("cuda:", 0) == 0;
#if defined(NEMO_SPEECH_CLI_CUDA)
            cuda_device = cuda_device || device_name == "auto" || device_name == "gpu" ||
                          device_name.rfind("gpu:", 0) == 0;
#endif
            if (gpu < 0) {
                parsed.runtime.lt_backend = nemo_speech::tts::MagpieBackendPreference::Cpu;
                parsed.runtime.sampling_backend = nemo_speech::tts::MagpieBackendPreference::Cpu;
                parsed.runtime.magpie_cpu = true;
                parsed.runtime.codec_cpu = true;
            } else if (cuda_device) {
                parsed.runtime.lt_backend = nemo_speech::tts::MagpieBackendPreference::Cuda;
                parsed.runtime.codec_cpu = false;
            } else {
                parsed.runtime.lt_backend = nemo_speech::tts::MagpieBackendPreference::Cpu;
                parsed.runtime.sampling_backend = nemo_speech::tts::MagpieBackendPreference::Cpu;
                parsed.runtime.codec_cpu = false;
            }
        }
        parsed.runtime.verbose = cli_verbose();

        nemo_speech::tts::SynthesizerConfig config;
        config.runtime = parsed.runtime;
        config.omnivoice_model = parsed.omnivoice_model;
        config.omnivoice_audio_tokenizer_model = parsed.omnivoice_audio_tokenizer_model;
        config.omnivoice_options = parsed.omnivoice_options;
        config.tokenizer_model_dir = parsed.tokenizer_model_dir;
        config.text_normalizer_model_dir = parsed.tn_model_dir;
        config.tokenizer = parsed.tokenizer_config;
        config.default_language_code = parsed.default_language_code;
        config.default_voice_name = parsed.default_voice_name;
        nemo_speech::EngineRegistry engines;
        auto synthesizer = engines.load_tts(std::move(config));
        if (warmup)
            synthesizer->warmup("Hello", 1);

        std::unique_ptr<nemo_speech::tts::omnivoice::VoicePrompt> prompt;
        if (!prompt_wav.empty() && !prompt_file.empty())
            throw std::invalid_argument("--prompt-wav and --prompt-file are mutually exclusive");
        if (!prompt_wav.empty()) {
            if (prompt_text.empty())
                throw std::invalid_argument("--prompt-text is required with --prompt-wav");
            const auto audio = nemo_speech::audio::load_wav_file(prompt_wav);
            prompt = synthesizer->create_voice_prompt(
                audio.samples.data(), audio.samples.size(), 1, audio.sample_rate, prompt_text,
                true);
        } else if (!prompt_file.empty()) {
            prompt = synthesizer->load_voice_prompt(prompt_file);
        }
        if (!save_prompt.empty()) {
            if (!prompt)
                throw std::invalid_argument("--save-prompt requires --prompt-wav or --prompt-file");
            synthesizer->save_voice_prompt(*prompt, save_prompt);
            if (prompt_only) {
                if (!cli_quiet())
                    std::fprintf(stderr, "wrote %s\n", save_prompt.c_str());
                return 0;
            }
        }

        nemo_speech::tts::SynthesisRequest request;
        request.text = text;
        request.language_code = language;
        request.voice_name = voice;
        request.output_sample_rate = output_rate;
        request.options = request_options;
        request.omnivoice_options = request_omnivoice;
        request.instruction = instruction;
        request.voice_prompt = prompt.get();
        std::string pcm;
        const auto result =
            synthesizer->synthesize(request, [&](const auto&, const std::string& chunk) {
                pcm += chunk;
                return true;
            });
        if (pcm.empty())
            throw std::runtime_error("synthesizer returned no audio");
        const std::string audio =
            format == "wav" ? nemo_speech::audio::pcm16_wav(pcm, result.metadata.sample_rate) : pcm;
        write_audio(output_path, audio, force);
        if (cli_json()) {
            nemo_speech::json::Value report(nemo_speech::json::Value::Object{});
            report["output"] = output_path;
            report["format"] = format;
            report["language"] = result.metadata.language_code;
            report["speaker"] = result.metadata.speaker;
            report["sample_rate"] = result.metadata.sample_rate;
            report["samples"] = static_cast<double>(result.output_samples);
            report["duration"] =
                static_cast<double>(result.output_samples) / result.metadata.sample_rate;
            report["rtfx"] = result.stats.e2e_rtfx;
            std::printf("%s\n", report.dump(2).c_str());
        } else if (!cli_quiet()) {
            std::fprintf(
                stderr, "wrote %s (%llu samples at %d Hz, %.2fx realtime)\n", output_path.c_str(),
                static_cast<unsigned long long>(result.output_samples), result.metadata.sample_rate,
                result.stats.e2e_rtfx);
        }
        return 0;
    }
    catch (const std::invalid_argument& error) {
        return print_cli_error(
            "synthesize", error.what(), kCliExitInvalidArgument, "invalid_argument");
    }
    catch (const std::exception& error) {
        return print_cli_exception("synthesize", error);
    }
}
