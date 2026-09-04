// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <grpcpp/grpcpp.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "grpc_tts.h"
#include "riva/proto/riva_audio.pb.h"
#include "tts/synthesizer.h"

namespace {

void
require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

}  // namespace

int
main() {
    const char* model = std::getenv("NEMO_SPEECH_TEST_OMNIVOICE_MODEL");
    const char* codec = std::getenv("NEMO_SPEECH_TEST_OMNIVOICE_CODEC");
    if (!model || !*model || !codec || !*codec) {
        std::cout << "SKIP: OmniVoice gRPC test needs model fixtures\n";
        return 0;
    }

    nemo_speech::tts::SynthesizerConfig config;
    config.omnivoice_model = model;
    config.omnivoice_audio_tokenizer_model = codec;
    config.runtime.lt_backend = nemo_speech::tts::MagpieBackendPreference::Cpu;
    config.default_language_code = "en";
    auto synthesizer = std::make_shared<nemo_speech::tts::Synthesizer>(std::move(config));
    nemo_speech::GrpcTtsService service(synthesizer);

    nr_tts::RivaSynthesisConfigRequest inventory_request;
    nr_tts::RivaSynthesisConfigResponse inventory;
    grpc::ServerContext inventory_context;
    require(
        service.GetRivaSynthesisConfig(&inventory_context, &inventory_request, &inventory).ok(),
        "OmniVoice gRPC inventory failed");
    require(inventory.model_config_size() == 646, "OmniVoice inventory language count is wrong");
    const auto& parameters = inventory.model_config(0).parameters();
    require(parameters.at("zero_shot") == "true", "zero-shot capability is missing");
    require(parameters.at("voice_design") == "true", "voice-design capability is missing");

    nr_tts::SynthesizeSpeechRequest request;
    request.set_text("Hello.");
    request.set_language_code("en");
    request.set_voice_name("auto");
    request.set_encoding(nvidia::riva::LINEAR_PCM);
    (*request.mutable_custom_configuration())["omnivoice_steps"] = "1";
    (*request.mutable_custom_configuration())["duration"] = "0.04";
    (*request.mutable_custom_configuration())["instruction"] = "female";
    nr_tts::SynthesizeSpeechResponse response;
    grpc::ServerContext context;
    const grpc::Status status = service.Synthesize(&context, &request, &response);
    require(status.ok(), status.error_message().c_str());
    require(
        !response.audio().empty() && response.audio().size() % 2 == 0,
        "OmniVoice unary gRPC synthesis returned no PCM16");
    require(response.meta().text() == "Hello.", "OmniVoice response metadata text is wrong");

    nr_tts::SynthesizeSpeechRequest clone_request;
    clone_request.set_text("Cloned.");
    clone_request.set_language_code("en");
    clone_request.set_encoding(nvidia::riva::LINEAR_PCM);
    auto* zero_shot = clone_request.mutable_zero_shot_data();
    zero_shot->set_audio_prompt(response.audio());
    zero_shot->set_sample_rate_hz(24000);
    zero_shot->set_encoding(nvidia::riva::LINEAR_PCM);
    zero_shot->set_transcript("Hello.");
    (*clone_request.mutable_custom_configuration())["omnivoice_steps"] = "1";
    (*clone_request.mutable_custom_configuration())["duration"] = "0.04";
    nr_tts::SynthesizeSpeechResponse clone_response;
    grpc::ServerContext clone_context;
    const grpc::Status clone_status =
        service.Synthesize(&clone_context, &clone_request, &clone_response);
    require(clone_status.ok(), clone_status.error_message().c_str());
    require(
        !clone_response.audio().empty(), "OmniVoice zero-shot gRPC synthesis returned no audio");

    grpc::ServerBuilder builder;
    int selected_port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    require(server && selected_port > 0, "could not start local OmniVoice gRPC test server");
    auto channel = grpc::CreateChannel(
        "127.0.0.1:" + std::to_string(selected_port), grpc::InsecureChannelCredentials());
    auto stub = nr_tts::RivaSpeechSynthesis::NewStub(channel);
    grpc::ClientContext stream_context;
    auto online = stub->SynthesizeOnline(&stream_context);
    nr_tts::SynthesizeSpeechRequest first;
    first.set_text("Hello. ");
    first.set_language_code("en");
    first.set_voice_name("auto");
    first.set_encoding(nvidia::riva::LINEAR_PCM);
    first.set_sample_rate_hz(16000);
    (*first.mutable_custom_configuration())["omnivoice_steps"] = "1";
    (*first.mutable_custom_configuration())["duration"] = "0.04";
    nr_tts::SynthesizeSpeechRequest second = first;
    second.set_text("World.");
    require(online->Write(first) && online->Write(second), "could not write bidirectional text");
    online->WritesDone();
    size_t streamed_bytes = 0;
    int responses = 0;
    nr_tts::SynthesizeSpeechResponse streamed;
    while (online->Read(&streamed)) {
        streamed_bytes += streamed.audio().size();
        ++responses;
    }
    const grpc::Status online_status = online->Finish();
    server->Shutdown();
    require(online_status.ok(), online_status.error_message().c_str());
    require(
        responses >= 2 && streamed_bytes > 0,
        "OmniVoice bidirectional gRPC stream did not emit progressive PCM");
    return 0;
}
