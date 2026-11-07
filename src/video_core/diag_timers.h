// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"

// EDEN_DIAG_SYNC (diagnostic): wall time per named step of the back's synchronous work (sync
// requests by kind, syncpoint signals by part), logged every 10 s as ms per second and count.
namespace VideoCommon::DiagTimers {

enum class Id : unsigned {
    SyncComposite, SyncCpuRead, SyncForceRead, SyncInvalidate, SyncOther,
    SyncPointGuest, SyncPointFence, GpuCacheInvalidate, GpuCacheRanges, SyncpointLock,
    SyncpointActions, SyncpointNotify, SlotSignal, SlotLock, TickPipeline, TickTexture,
    TickBuffer, TickStaging, TickFence, TickDescriptors, Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(Id::Count)> names{
    "sync composite", "sync cpu read", "sync force read", "sync invalidate", "sync other",
    "syncpoint guest", "syncpoint fence", "gpu cache invalidate", "(ranges)", "syncpoint lock wait",
    "syncpoint actions", "syncpoint notify", "slot event signal", "slot shared lock", "tick pipeline cache", "tick texture cache",
    "tick buffer cache", "tick staging", "tick fence", "tick descriptors"};

inline bool Enabled() {
    static const bool enabled = std::getenv("EDEN_DIAG_SYNC") != nullptr;
    return enabled;
}

struct Stat {
    std::atomic<unsigned long long> ns{};
    std::atomic<unsigned long long> count{};
    std::atomic<unsigned long long> max_ns{};
};
inline std::array<Stat, static_cast<size_t>(Id::Count)> stats{};
inline std::atomic<long long> last_report{};

inline void Add(Id id, unsigned long long ns, unsigned long long count = 1) {
    auto& stat = stats[static_cast<size_t>(id)];
    stat.ns.fetch_add(ns, std::memory_order_relaxed);
    stat.count.fetch_add(count, std::memory_order_relaxed);
    for (auto max = stat.max_ns.load(std::memory_order_relaxed);
         ns > max && !stat.max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed);) {
    }
    const long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
    long long last = last_report.load(std::memory_order_relaxed);
    if (last == 0) {
        last_report.compare_exchange_strong(last, now);
        return;
    }
    if (now - last < 10000 || !last_report.compare_exchange_strong(last, now)) {
        return;
    }
    const double seconds = static_cast<double>(now - last) / 1000.0;
    std::string text;
    for (size_t i = 0; i < stats.size(); ++i) {
        const auto ns_total = stats[i].ns.exchange(0);
        const auto n = stats[i].count.exchange(0);
        const auto max_ns = stats[i].max_ns.exchange(0);
        if (n != 0) {
            text += fmt::format(" | {} {:.1f} ms/s ({:.0f}/s, max {:.1f} ms)", names[i],
                                static_cast<double>(ns_total) / 1e6 / seconds,
                                static_cast<double>(n) / seconds,
                                static_cast<double>(max_ns) / 1e6);
        }
    }
    LOG_INFO(HW_GPU, "Diag sync:{}", text);
}

struct Scope {
    Id id;
    std::chrono::steady_clock::time_point begin{};
    bool on;
    explicit Scope(Id id_) : id{id_}, on{Enabled()} {
        if (on) begin = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (on) {
            Add(id, static_cast<unsigned long long>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - begin)
                            .count()));
        }
    }
};

} // namespace VideoCommon::DiagTimers
