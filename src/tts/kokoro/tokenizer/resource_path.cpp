// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "resource_path.h"

#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#include <dlfcn.h>
#endif

#ifndef NEMO_SPEECH_KOKORO_DATA_FROM_MODULE_DIR
#define NEMO_SPEECH_KOKORO_DATA_FROM_MODULE_DIR ""
#endif

namespace nemo_speech::tts::kokoro {
namespace {

std::filesystem::path
module_path() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&installed_misaki_data_root), &module)) {
        return {};
    }
    std::wstring buffer(32768, L'\0');
    const DWORD size = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (size == 0 || size == buffer.size())
        return {};
    buffer.resize(size);
    return std::filesystem::path(buffer);
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&installed_misaki_data_root), &info) == 0 ||
        info.dli_fname == nullptr) {
        return {};
    }
    return std::filesystem::path(info.dli_fname);
#else
    return {};
#endif
}

}  // namespace

std::filesystem::path
installed_misaki_data_root() {
    const std::filesystem::path module = module_path();
    if (module.empty() || std::string(NEMO_SPEECH_KOKORO_DATA_FROM_MODULE_DIR).empty())
        return {};
    std::error_code error;
    const std::filesystem::path absolute = std::filesystem::absolute(
        module.parent_path() / NEMO_SPEECH_KOKORO_DATA_FROM_MODULE_DIR, error);
    return error ? std::filesystem::path{} : absolute.lexically_normal();
}

}  // namespace nemo_speech::tts::kokoro
