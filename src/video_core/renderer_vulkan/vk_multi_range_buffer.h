// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

using SparseBuffer = vk::Handle<VkBuffer, VkDevice, vk::DeviceDispatch>;

class Device;
class Scheduler;

struct MultiRangeSource {
    VkBuffer handle{};
    VkDeviceMemory memory{};
    VkDeviceSize memory_offset{};
    VkDeviceSize offset{};
    VkDeviceSize size{};
    u64 write_tick{};
    u32 memory_type{};
};

struct MultiRangeRef {
    VkBuffer handle{};
    VkDeviceAddress address{};
    VkDeviceSize size{};
    bool sparse{};
    bool needs_gather{};
};

class MultiRangeBufferCache final {
public:
    static constexpr VkDeviceSize DEFAULT_BLOCK_SIZE = 64 * 1024;
    static constexpr size_t MAX_RETIRED = 256;
    /// Frames an entry may go unbound before TickFrame releases its buffers. A game that binds
    /// multi-range storage at ever new addresses left one entry per address behind (Kirby and the
    /// Forgotten Land, Fallen-Star Beach: 468 gathered buffers of 8 MiB after eight minutes).
    static constexpr u64 MAX_UNUSED_FRAMES = 60;

    [[nodiscard]] static constexpr bool IsUnused(u64 frame, u64 last_used) noexcept {
        return frame - last_used > MAX_UNUSED_FRAMES;
    }

    /// Removes every entry of map that IsUnused at frame, after handing it to retire. A template
    /// so the unit test can run it on a map of plain structs, without a Vulkan device.
    template <typename Map, typename Retire>
    static void EraseUnused(Map& map, u64 frame, Retire&& retire) {
        erase_if(map, [&](auto& item) {
            const bool unused = IsUnused(frame, item.second.last_used);
            if (unused) {
                retire(item.second);
            }
            return unused;
        });
    }

    explicit MultiRangeBufferCache(const Device& device);

    YUZU_NON_COPYABLE(MultiRangeBufferCache);

    /// Releases the entries no draw bound for MAX_UNUSED_FRAMES (port of Orchid #146).
    void TickFrame(Scheduler& scheduler);

    [[nodiscard]] MultiRangeRef Get(const Device& device, Scheduler& scheduler,
                                    MemoryAllocator& memory_allocator, u64 key,
                                    std::span<const MultiRangeSource> sources,
                                    VkDeviceSize total);

    void MarkGathered(u64 key);

    void Invalidate(u64 key);

    void DropOwner(Scheduler& scheduler, VkBuffer owner);

    VkDeviceSize block_size{DEFAULT_BLOCK_SIZE};
    bool use_sparse{};

private:
    struct Retired {
        SparseBuffer handle;
        vk::Buffer gathered;
        u64 tick{};
    };

    struct Entry {
        vk::Buffer gathered;
        SparseBuffer sparse_handle;
        std::vector<VkBuffer> owners;
        VkDeviceAddress address{};
        VkDeviceSize size{};
        u64 geometry{};
        u64 content{};
        u64 last_used{};
        bool dirty{true};
    };

    [[nodiscard]] u64 HashSources(std::span<const MultiRangeSource> sources) const;

    [[nodiscard]] u64 HashContent(std::span<const MultiRangeSource> sources) const;

    [[nodiscard]] bool CanBindSparse(std::span<const MultiRangeSource> sources) const;

    [[nodiscard]] SparseBuffer CreateSparse(const Device& device, Scheduler& scheduler,
                                            std::span<const MultiRangeSource> sources,
                                            VkDeviceSize total);

    [[nodiscard]] VkDeviceSize QueryBlockSize(const Device& device, u32& memory_type_bits) const;

    void RetireEntry(Scheduler& scheduler, Entry& entry);

    void DrainRetired(Scheduler& scheduler);

    ::Common::unordered_map<u64, Entry> entries;
    boost::container::static_vector<Retired, MAX_RETIRED> retired;
    u32 sparse_memory_type_bits{};
    VkBufferUsageFlags sparse_usage{};
    u64 frame{};
};

} // namespace Vulkan
