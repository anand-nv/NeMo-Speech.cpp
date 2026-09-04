// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>

namespace nemo_speech::tts::kokoro {

// Locate the installed Misaki data root relative to the shared library that
// contains this code. This keeps staged and relocated installations independent
// of the configure-time CMAKE_INSTALL_PREFIX.
std::filesystem::path installed_misaki_data_root();

}  // namespace nemo_speech::tts::kokoro
