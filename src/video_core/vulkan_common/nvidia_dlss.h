// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS in place of a game's DLSS, whichever way this build reaches it: NVIDIA Streamline
// on Windows (vulkan_common/streamline.h), NVIDIA NGX directly on Linux (vulkan_common/ngx.h).
// Both have the same interface; EGLANTINE_NVIDIA_DLSS is defined with either.

#pragma once

#if defined(EGLANTINE_STREAMLINE)
#include "video_core/vulkan_common/streamline.h"
namespace Vulkan {
namespace NvidiaDlss = Streamline;
}
#elif defined(EGLANTINE_NGX)
#include "video_core/vulkan_common/ngx.h"
namespace Vulkan {
namespace NvidiaDlss = Ngx;
}
#endif
