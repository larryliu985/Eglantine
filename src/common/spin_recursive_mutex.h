// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#ifdef __linux__
#include <pthread.h>
#endif
#include <cstdint>
#include <thread>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace Common {

/// A recursive mutex that spins briefly before sleeping. The GPU caches' locks are taken tens of
/// thousands of times a second by the GPU thread, the fencing thread and CPU write traps, and
/// held for microseconds: std::recursive_mutex put a waiting thread to sleep at once, and its
/// wake-up took longer than the critical section (title 1's GPU thread spent ~8% asleep there).
class SpinRecursiveMutex {
public:
    void lock() {
        const std::thread::id self = std::this_thread::get_id();
        if (owner.load(std::memory_order_relaxed) == self) {
            ++depth;
            return;
        }
        std::uint32_t expected = 0;
        if (!state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            // DIAGNOSTIC (EDEN_DIAG_LOCK_TIMING): per-thread time waiting for contended locks.
            static const bool diag = std::getenv("EDEN_DIAG_LOCK_TIMING") != nullptr;
            if (diag) [[unlikely]] {
                const auto t0 = std::chrono::steady_clock::now();
                LockContended();
                DiagNote(std::chrono::steady_clock::now() - t0);
            } else {
                LockContended();
            }
        }
        owner.store(self, std::memory_order_relaxed);
        depth = 1;
    }

    static void DiagNote(std::chrono::steady_clock::duration waited) {
        thread_local std::chrono::steady_clock::duration total{};
        thread_local std::uint64_t count{};
        thread_local auto start = std::chrono::steady_clock::now();
        total += waited;
        ++count;
        const auto now = std::chrono::steady_clock::now();
        if (now - start < std::chrono::seconds(10)) {
            return;
        }
        char name[32] = "?";
#ifdef __linux__
        pthread_getname_np(pthread_self(), name, sizeof(name));
#endif
        const double secs = std::chrono::duration<double>(now - start).count();
        std::fprintf(stderr, "DIAG lock wait %s: %.1f ms/s over %.0f waits/s\n", name,
                     std::chrono::duration<double, std::milli>(total).count() / secs,
                     static_cast<double>(count) / secs);
        total = {};
        count = 0;
        start = now;
    }

    bool try_lock() {
        const std::thread::id self = std::this_thread::get_id();
        if (owner.load(std::memory_order_relaxed) == self) {
            ++depth;
            return true;
        }
        std::uint32_t expected = 0;
        if (!state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            return false;
        }
        owner.store(self, std::memory_order_relaxed);
        depth = 1;
        return true;
    }

    void unlock() {
        if (--depth != 0) {
            return;
        }
        owner.store(std::thread::id{}, std::memory_order_relaxed);
        if (state.exchange(0, std::memory_order_release) == 2) {
            state.notify_one();
        }
    }

private:
    // About 10 microseconds of spinning before sleeping.
    static constexpr int SpinCount = 1024;

    static void Pause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#endif
    }

    void LockContended() {
        for (int spin = 0; spin < SpinCount; ++spin) {
            Pause();
            if (state.load(std::memory_order_relaxed) == 0) {
                std::uint32_t expected = 0;
                if (state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                                std::memory_order_relaxed)) {
                    return;
                }
            }
        }
        // 2: held with (possible) sleepers; taken this way it stays marked, so its unlock wakes
        // the next sleeper.
        while (state.exchange(2, std::memory_order_acquire) != 0) {
            state.wait(2, std::memory_order_relaxed);
        }
    }

    std::atomic<std::uint32_t> state{0}; ///< 0 free, 1 held, 2 held with sleepers
    std::atomic<std::thread::id> owner{};
    std::uint32_t depth{}; ///< Recursion depth, touched only by the owner
};

} // namespace Common
