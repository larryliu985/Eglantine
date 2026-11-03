// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS through NVIDIA NGX directly (Linux).

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/vulkan_common/ngx.h"

#include <nvsdk_ngx_vk.h>

namespace Vulkan::Ngx {
namespace {

// Same project id as the Streamline path: a custom engine names itself with it, and the driver may
// key behaviour off it.
constexpr const char* PROJECT_ID = "8d3e2f9a-6b41-4c7e-9a52-1f0c3b7d5e64";
constexpr std::string_view DLSS_LIBRARY = "libnvidia-ngx-dlss.so.";

struct State {
    std::vector<std::wstring> search_dirs; ///< Folders holding a DLSS library, searched in order
    std::vector<std::string> instance_extensions;
    std::vector<std::string> device_extensions;
    NVSDK_NGX_Parameter* capabilities{};
    VkDevice device{};
    bool started{};
    bool attached{};
};

State& GetState() {
    static State state;
    return state;
}

int LogLevel() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_LOG");
    return value ? std::atoi(value) : 0;
}

void LogFromNgx(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    std::string_view text{message ? message : ""};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    // The callback's level is the configured one, not the line's: NGX writes "] Error: " and
    // "] Warning: " into the line itself.
    if (text.find("] Error: ") != std::string_view::npos ||
        text.find("] Warning: ") != std::string_view::npos) {
        LOG_WARNING(Render_Vulkan, "NGX: {}", text);
    } else if (LogLevel() > 0) {
        LOG_INFO(Render_Vulkan, "NGX: {}", text);
    }
}

/// Where the user may put libnvidia-ngx-dlss.so.<version>: EDEN_NGX_DIR, the user folder's "ngx"
/// folder, then an "ngx" folder next to the executable.
std::vector<std::filesystem::path> SearchDirs() {
    std::vector<std::filesystem::path> dirs;
    if (const char* const dir = std::getenv("EDEN_NGX_DIR"); dir && *dir) {
        dirs.emplace_back(dir);
    }
    dirs.push_back(UserFolder());
    // Common::FS::GetExeDirectory is Windows-only.
    std::error_code ec;
    if (const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
        !ec) {
        dirs.push_back(exe.parent_path() / "ngx");
    }
    return dirs;
}

bool HasDlssLibrary(const std::filesystem::path& dir) {
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (entry.path().filename().string().starts_with(DLSS_LIBRARY)) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> Names(unsigned count, const char** names) {
    std::vector<std::string> result;
    for (unsigned i = 0; i < count; ++i) {
        if (names && names[i]) {
            result.emplace_back(names[i]);
        }
    }
    return result;
}

void ReleaseDevice(State& state) {
    if (state.capabilities) {
        NVSDK_NGX_VULKAN_DestroyParameters(state.capabilities);
        state.capabilities = nullptr;
    }
    if (state.device) {
        NVSDK_NGX_VULKAN_Shutdown1(state.device);
        state.device = VkDevice{};
    }
}

} // Anonymous namespace

std::filesystem::path UserFolder() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::NgxDir);
}

bool Start() {
    State& state = GetState();
    if (state.started) {
        return true;
    }
    if (!Settings::values.nvidia_dlss.GetValue()) {
        return false;
    }
    state.search_dirs.clear();
    for (const std::filesystem::path& dir : SearchDirs()) {
        if (HasDlssLibrary(dir)) {
            state.search_dirs.push_back(dir.wstring());
        }
    }
    if (state.search_dirs.empty()) {
        LOG_WARNING(Render_Vulkan,
                    "NVIDIA DLSS is on, but no {}<version> is in {} (it comes with NVIDIA's DLSS "
                    "SDK, lib/Linux_*/rel); using FSR 3.1",
                    DLSS_LIBRARY, Common::FS::PathToUTF8String(UserFolder()));
        return false;
    }
    // NGX can name its extensions before it is initialised.
    unsigned instance_count = 0;
    unsigned device_count = 0;
    const char** instance_names = nullptr;
    const char** device_names = nullptr;
    if (const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_RequiredExtensions(
            &instance_count, &instance_names, &device_count, &device_names);
        NVSDK_NGX_FAILED(result)) {
        LOG_WARNING(Render_Vulkan,
                    "NVIDIA NGX is not available (result {:#x}); is this NVIDIA's driver? Using "
                    "FSR 3.1",
                    static_cast<u32>(result));
        return false;
    }
    state.instance_extensions = Names(instance_count, instance_names);
    state.device_extensions = Names(device_count, device_names);
    // NGX names VK_EXT_buffer_device_address, which can't be enabled together with the core
    // bufferDeviceAddress feature the device keeps on for DLSS (see Device). DLSS evaluates with
    // the core feature alone (checked on a GB10 with driver 580).
    std::erase(state.device_extensions, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    std::string dirs;
    for (const std::wstring& dir : state.search_dirs) {
        dirs += (dirs.empty() ? "" : ", ") +
                Common::FS::PathToUTF8String(std::filesystem::path{dir});
    }
    LOG_INFO(Render_Vulkan, "NVIDIA NGX started; DLSS library from {}", dirs);
    state.started = true;
    return true;
}

bool WantsDevice() {
    const State& state = GetState();
    return state.started && !state.attached;
}

std::vector<std::string> InstanceExtensions() {
    const State& state = GetState();
    return state.started ? state.instance_extensions : std::vector<std::string>{};
}

std::vector<std::string> DeviceExtensions() {
    const State& state = GetState();
    return state.started ? state.device_extensions : std::vector<std::string>{};
}

bool AttachDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                  [[maybe_unused]] u32 graphics_family, vk::DeviceDispatch& dld) {
    State& state = GetState();
    if (!WantsDevice()) {
        return false;
    }
    std::vector<const wchar_t*> paths;
    for (const std::wstring& dir : state.search_dirs) {
        paths.push_back(dir.c_str());
    }
    NVSDK_NGX_FeatureCommonInfo info{};
    info.PathListInfo.Path = paths.data();
    info.PathListInfo.Length = static_cast<unsigned>(paths.size());
    info.LoggingInfo.LoggingCallback = LogFromNgx;
    info.LoggingInfo.MinimumLoggingLevel =
        LogLevel() >= 2 ? NVSDK_NGX_LOGGING_LEVEL_VERBOSE : NVSDK_NGX_LOGGING_LEVEL_ON;
    info.LoggingInfo.DisableOtherLoggingSinks = true;

    // NGX keeps its own data (logs, downloaded models) in the user folder's "ngx" folder.
    const std::wstring data_dir = UserFolder().wstring();
    if (const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
            PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", data_dir.c_str(), instance, physical,
            device, dld.vkGetInstanceProcAddr, dld.vkGetDeviceProcAddr, &info);
        NVSDK_NGX_FAILED(result)) {
        LOG_WARNING(Render_Vulkan, "NVIDIA NGX did not start on this device (result {:#x}); "
                                   "using FSR 3.1",
                    static_cast<u32>(result));
        Stop();
        return false;
    }
    state.device = device;

    int available = 0;
    int needs_driver = 0;
    unsigned min_major = 0;
    unsigned min_minor = 0;
    int init_result = 0;
    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&state.capabilities);
    if (NVSDK_NGX_SUCCEED(result)) {
        NVSDK_NGX_Parameter_GetI(state.capabilities, NVSDK_NGX_Parameter_SuperSampling_Available,
                                 &available);
        NVSDK_NGX_Parameter_GetI(state.capabilities,
                                 NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
                                 &needs_driver);
        NVSDK_NGX_Parameter_GetUI(state.capabilities,
                                  NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor,
                                  &min_major);
        NVSDK_NGX_Parameter_GetUI(state.capabilities,
                                  NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor,
                                  &min_minor);
        NVSDK_NGX_Parameter_GetI(state.capabilities,
                                 NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
                                 &init_result);
    }
    if (NVSDK_NGX_FAILED(result) || available != 1) {
        if (needs_driver != 0) {
            LOG_WARNING(Render_Vulkan,
                        "NVIDIA DLSS needs driver {}.{} or newer; using FSR 3.1", min_major,
                        min_minor);
        } else {
            LOG_WARNING(Render_Vulkan,
                        "NVIDIA DLSS is not supported on this GPU or driver (result {:#x}, "
                        "feature init {:#x}); using FSR 3.1",
                        static_cast<u32>(result), static_cast<u32>(init_result));
        }
        Stop();
        return false;
    }
    state.attached = true;
    LOG_INFO(Render_Vulkan, "NVIDIA DLSS is available through NGX");
    return true;
}

bool IsDlssAvailable() {
    const State& state = GetState();
    return state.started && state.attached;
}

NVSDK_NGX_Parameter* Capabilities() {
    return GetState().capabilities;
}

void Stop() {
    State& state = GetState();
    if (!state.started) {
        return;
    }
    ReleaseDevice(state);
    state.started = false;
    state.attached = false;
    LOG_INFO(Render_Vulkan, "NVIDIA NGX stopped");
}

} // namespace Vulkan::Ngx
