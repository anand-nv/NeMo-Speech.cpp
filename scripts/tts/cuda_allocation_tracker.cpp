// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// LD_PRELOAD helper for process-scoped CUDA allocation high-water reporting.
//
// GGML allocates its device buffers through cudaMalloc/cudaMallocManaged.  The
// GB10 coherent-memory driver does not expose NVML per-process framebuffer
// usage, so this interposer records successful runtime allocations directly.
#include <dlfcn.h>
#include <pthread.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace {

constexpr size_t kMaxAllocations = 65536;

struct Allocation {
    void* pointer = nullptr;
    size_t bytes = 0;
};

Allocation allocations[kMaxAllocations];
pthread_mutex_t allocations_mutex = PTHREAD_MUTEX_INITIALIZER;
std::atomic<size_t> current_bytes{0};
std::atomic<size_t> peak_bytes{0};

void
update_peak(size_t current) {
    size_t peak = peak_bytes.load(std::memory_order_relaxed);
    while (current > peak &&
           !peak_bytes.compare_exchange_weak(
               peak, current, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

void
record_allocation(void* pointer, size_t bytes) {
    if (pointer == nullptr || bytes == 0)
        return;
    pthread_mutex_lock(&allocations_mutex);
    for (Allocation& allocation : allocations) {
        if (allocation.pointer == pointer) {
            pthread_mutex_unlock(&allocations_mutex);
            return;
        }
        if (allocation.pointer == nullptr) {
            allocation.pointer = pointer;
            allocation.bytes = bytes;
            const size_t current =
                current_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
            update_peak(current);
            pthread_mutex_unlock(&allocations_mutex);
            return;
        }
    }
    pthread_mutex_unlock(&allocations_mutex);
}

void
record_free(void* pointer) {
    if (pointer == nullptr)
        return;
    pthread_mutex_lock(&allocations_mutex);
    for (Allocation& allocation : allocations) {
        if (allocation.pointer == pointer) {
            current_bytes.fetch_sub(allocation.bytes, std::memory_order_relaxed);
            allocation = {};
            break;
        }
    }
    pthread_mutex_unlock(&allocations_mutex);
}

template <typename Function>
Function
next_symbol(const char* name) {
    void* symbol = dlsym(RTLD_NEXT, name);
    if (symbol == nullptr) {
        // ctypes loads the TTS shared object with RTLD_LOCAL, so libcudart may
        // not appear in the preloaded object's RTLD_NEXT lookup scope.
        static void* cudart = dlopen("libcudart.so.13", RTLD_NOW | RTLD_LOCAL);
        if (cudart != nullptr)
            symbol = dlsym(cudart, name);
    }
    return reinterpret_cast<Function>(symbol);
}

}  // namespace

extern "C" int
cudaMalloc(void** pointer, size_t bytes) {
    using Function = int (*)(void**, size_t);
    static Function next = next_symbol<Function>("cudaMalloc");
    if (next == nullptr)
        return 999;
    const int status = next(pointer, bytes);
    if (status == 0)
        record_allocation(*pointer, bytes);
    return status;
}

extern "C" int
cudaMallocManaged(void** pointer, size_t bytes, unsigned int flags) {
    using Function = int (*)(void**, size_t, unsigned int);
    static Function next = next_symbol<Function>("cudaMallocManaged");
    if (next == nullptr)
        return 999;
    const int status = next(pointer, bytes, flags);
    if (status == 0)
        record_allocation(*pointer, bytes);
    return status;
}

extern "C" int
cudaFree(void* pointer) {
    using Function = int (*)(void*);
    static Function next = next_symbol<Function>("cudaFree");
    if (next == nullptr)
        return 999;
    const int status = next(pointer);
    if (status == 0)
        record_free(pointer);
    return status;
}

extern "C" size_t
omnivoice_cuda_current_allocation_bytes() {
    return current_bytes.load(std::memory_order_relaxed);
}

extern "C" size_t
omnivoice_cuda_peak_allocation_bytes() {
    return peak_bytes.load(std::memory_order_relaxed);
}

extern "C" void
omnivoice_cuda_reset_peak_allocation_bytes() {
    peak_bytes.store(current_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
}
