// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS through NVIDIA NGX directly, for Linux: the NGX runtime is part of NVIDIA's driver
// (libnvidia-ngx.so.1) and the DLSS library (libnvidia-ngx-dlss.so.<version>) is supplied by the
// user. Same interface as vulkan_common/streamline.h, which does this on Windows.

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

struct NVSDK_NGX_Parameter;

namespace Vulkan::Ngx {

/// The user folder's "ngx" folder: where libnvidia-ngx-dlss.so.<version> goes, and NGX's data.
[[nodiscard]] std::filesystem::path UserFolder();

/// When the NVIDIA DLSS setting is on and a DLSS library is found, gets NGX's Vulkan extensions.
/// Call before creating the renderer's instance.
bool Start();

/// Whether NGX was started and is waiting for the renderer's device.
[[nodiscard]] bool WantsDevice();

/// Instance and device extensions NGX asks for (empty when not started).
[[nodiscard]] std::vector<std::string> InstanceExtensions();
[[nodiscard]] std::vector<std::string> DeviceExtensions();

/// Initialises NGX on the renderer's device and checks that DLSS super resolution can run.
/// Shuts NGX down and returns false when it can't.
bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                  u32 graphics_family, vk::DeviceDispatch& dld);

/// Whether DLSS can run on the attached device.
[[nodiscard]] bool IsDlssAvailable();

/// NGX's capability parameters for the attached device, used to create DLSS features.
[[nodiscard]] NVSDK_NGX_Parameter* Capabilities();

/// Shuts NGX down; call before the device is destroyed, after every DLSS feature is released.
void Stop();

} // namespace Vulkan::Ngx
