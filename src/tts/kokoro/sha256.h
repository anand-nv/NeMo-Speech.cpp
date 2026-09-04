// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <string_view>

namespace nemo_speech::tts::kokoro {

std::string sha256_hex(std::string_view input);

}  // namespace nemo_speech::tts::kokoro
