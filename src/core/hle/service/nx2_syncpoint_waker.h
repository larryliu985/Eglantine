// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "common/common_types.h"
#include "common/thread.h"

namespace Service::Nx2 {

// Wakes host threads that wait for a guest syncpoint (nx2:vi2's frame fence wait) from a thread of
// its own. The syncpoint action used to take the waiter's mutex itself, on the GPU back thread
// that increments the syncpoint: whenever the waiter had been preempted while holding it, the
// back thread waited too (~3 ms an action, ~120 ms a second, while turning the camera in
// Ocarina of Time). Now the action only pushes the wake onto a lock-free list.
class SyncpointWaker {
public:
    struct Wake {
        std::mutex mutex;
        std::condition_variable cv;
        bool done{};
    };

    static SyncpointWaker& Get() {
        static SyncpointWaker waker;
        return waker;
    }

    /// Called from syncpoint actions: never blocks.
    void Post(std::shared_ptr<Wake> wake) {
        auto* const node = new Node{std::move(wake), head.load(std::memory_order_relaxed)};
        while (!head.compare_exchange_weak(node->next, node, std::memory_order_release,
                                           std::memory_order_relaxed)) {
        }
        pending.fetch_add(1, std::memory_order_release);
        pending.notify_one();
    }

private:
    struct Node {
        std::shared_ptr<Wake> wake;
        Node* next;
    };

    SyncpointWaker() : thread{[this](std::stop_token token) { Run(token); }} {}

    ~SyncpointWaker() {
        thread.request_stop();
        pending.fetch_add(1, std::memory_order_release);
        pending.notify_one();
        thread.join();
        Drain();
    }

    void Run(std::stop_token token) {
        Common::SetCurrentThreadName("nx2:SyncptWaker");
        u32 seen = 0;
        while (!token.stop_requested()) {
            // A post after the drain below changes `pending` again, so this returns at once.
            pending.wait(seen, std::memory_order_acquire);
            seen = pending.load(std::memory_order_acquire);
            Drain();
        }
    }

    void Drain() {
        Node* node = head.exchange(nullptr, std::memory_order_acquire);
        while (node != nullptr) {
            Node* const next = node->next;
            {
                std::scoped_lock lk{node->wake->mutex};
                node->wake->done = true;
            }
            node->wake->cv.notify_all();
            delete node;
            node = next;
        }
    }

    std::atomic<Node*> head{nullptr};
    std::atomic<u32> pending{0};
    std::jthread thread; // last: starts once the members above exist
};

} // namespace Service::Nx2
