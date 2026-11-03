// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS (through NVIDIA NGX, Linux) in place of a game's DLSS evaluation.

#include <array>
#include <cstdlib>
#include <string_view>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_ngx_dlss.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/ngx.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>

namespace Vulkan {
namespace {

constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
// DLSS writes the output as a storage image; it is cleared when DLSS writes nothing to it.
constexpr VkImageUsageFlags OUTPUT_USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;

int EnvInt(const char* name, int fallback) {
    const char* const value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

// DLSS validates the render size against the mode's range: pick the mode for this ratio. DLAA only
// when nothing is scaled: NGX requires equal sizes for it, unlike Streamline, which picks DLAA from a
// ratio of 0.99.
NVSDK_NGX_PerfQuality_Value ModeFor(VkExtent2D render, VkExtent2D upscale) {
    if (render.width == upscale.width && render.height == upscale.height) {
        return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
    const f32 ratio = static_cast<f32>(render.width) / static_cast<f32>(upscale.width);
    if (ratio >= 0.65f) {
        return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    }
    if (ratio >= 0.57f) {
        return NVSDK_NGX_PerfQuality_Value_Balanced;
    }
    if (ratio >= 0.49f) {
        return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    }
    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

// Preset K for every mode, as on Windows: NVIDIA's defaults for Performance/Ultra Performance
// (M, L) draw thin horizontal lines in Fire Emblem at 2x resolution scale.
// EDEN_NX2_DLSS_PRESET=J|K|L|M picks another model, EDEN_NX2_DLSS_PRESET=default leaves the
// choice to NVIDIA.
NVSDK_NGX_DLSS_Hint_Render_Preset Preset() {
    const char* const value = std::getenv("EDEN_NX2_DLSS_PRESET");
    if (value && value[0] >= 'J' && value[0] <= 'M') {
        return static_cast<NVSDK_NGX_DLSS_Hint_Render_Preset>(
            static_cast<u32>(NVSDK_NGX_DLSS_Hint_Render_Preset_J) + static_cast<u32>(value[0] - 'J'));
    }
    if (value && std::string_view{value} == "default") {
        return NVSDK_NGX_DLSS_Hint_Render_Preset_Default;
    }
    return NVSDK_NGX_DLSS_Hint_Render_Preset_K;
}

VkImageMemoryBarrier Barrier(const NgxDlssUpscaler::Image& image, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.image,
        .subresourceRange = range,
    };
}

NVSDK_NGX_Resource_VK Resource(const NgxDlssUpscaler::Image& image, VkExtent2D size) {
    return NVSDK_NGX_Create_ImageView_Resource_VK(image.view, image.image, image.range,
                                                  image.format, size.width, size.height, false);
}

} // Anonymous namespace

NgxDlssUpscaler::NgxDlssUpscaler(const Device& device_, Scheduler& scheduler_,
                                 MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    if (!Ngx::IsDlssAvailable()) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    if (const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_AllocateParameters(&params);
        NVSDK_NGX_FAILED(result) || !params) {
        LOG_ERROR(Render_Vulkan, "NGX gave no parameters for DLSS (result {:#x})",
                  static_cast<u32>(result));
        params = nullptr;
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    point_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
}

NgxDlssUpscaler::~NgxDlssUpscaler() {
    // Not scheduler.Finish(): this runs when the renderer goes away, after the game's channels are
    // released, and a submit would touch their state. Nothing more gets recorded by then.
    void(device.GetLogical().WaitIdle());
    if (Ngx::IsDlssAvailable()) {
        if (feature) {
            NVSDK_NGX_VULKAN_ReleaseFeature(feature);
        }
        if (params) {
            NVSDK_NGX_VULKAN_DestroyParameters(params);
        }
    }
    feature = nullptr;
    params = nullptr;
}

void NgxDlssUpscaler::CreateOutput(VkExtent2D upscale_size) {
    if (output_image) {
        // The old output, and the DLSS feature that the next evaluation replaces, may still be in
        // use.
        scheduler.Finish();
    }
    output_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .extent = {upscale_size.width, upscale_size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = OUTPUT_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    output_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *output_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .components{},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               VkImageMemoryBarrier{
                                   .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                   .pNext = nullptr,
                                   .srcAccessMask = 0,
                                   .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange{
                                       .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .baseMipLevel = 0,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = 1,
                                   },
                               });
    });
}

bool NgxDlssUpscaler::Upscale(const Inputs& in) {
    if (failed) {
        return false;
    }
    bool reset = in.reset;
    bool size_changed = false;
    if (!output_image || in.render_size.width != render_extent.width ||
        in.render_size.height != render_extent.height ||
        in.upscale_size.width != upscale_extent.width ||
        in.upscale_size.height != upscale_extent.height) {
        CreateOutput(in.upscale_size);
        render_extent = in.render_size;
        upscale_extent = in.upscale_size;
        size_changed = true;
        reset = true;
    }

    // The game's own setup (Fortune's Weave: render-size motion vectors, reversed depth). Its
    // colour is linear and goes above 1 with no exposure passed, so by default DLSS gets the HDR
    // path with its own auto exposure. EDEN_NX2_DLSS_HDR=0 / EDEN_NX2_DLSS_AUTO_EXPOSURE=0 turn
    // those off.
    const NVSDK_NGX_PerfQuality_Value mode = ModeFor(in.render_size, in.upscale_size);
    const NVSDK_NGX_DLSS_Hint_Render_Preset preset = Preset();
    int create_flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                       NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    if (EnvInt("EDEN_NX2_DLSS_HDR", 1) != 0) {
        create_flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    }
    if (EnvInt("EDEN_NX2_DLSS_AUTO_EXPOSURE", 1) != 0) {
        create_flags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    }
    if (size_changed) {
        LOG_INFO(Render_Vulkan, "NVIDIA DLSS (NGX): {}x{} -> {}x{}, mode {}, preset {}",
                 in.render_size.width, in.render_size.height, in.upscale_size.width,
                 in.upscale_size.height, static_cast<int>(mode), static_cast<int>(preset));
    }

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, reset, size_changed, mode, preset, create_flags,
                      output_image_handle = *output_image,
                      output_view_handle = *output_view](vk::CommandBuffer cmdbuf) {
        // The caller draws the output this frame even when DLSS writes nothing to it: clear it
        // then, rather than show undefined contents (FSR 3.1 takes over from the next frame).
        const auto clear_output = [&] {
            const VkImageSubresourceRange range{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            const auto barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = src_access,
                    .dstAccessMask = dst_access,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = output_image_handle,
                    .subresourceRange = range,
                };
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                   barrier(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                           VK_ACCESS_TRANSFER_WRITE_BIT));
            cmdbuf.ClearColorImage(output_image_handle, VK_IMAGE_LAYOUT_GENERAL,
                                   VkClearColorValue{}, range);
            cmdbuf.PipelineBarrier(
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
        };
        const auto fail = [&](const char* what, NVSDK_NGX_Result result) {
            if (!failed.exchange(true)) {
                LOG_ERROR(Render_Vulkan, "NVIDIA DLSS {} failed (NGX result {:#x}); using FSR 3.1 "
                                         "from now on",
                          what, static_cast<u32>(result));
            }
            clear_output();
        };
        if (!params) {
            fail("setup", NVSDK_NGX_Result_Fail);
            return;
        }

        // A new size needs a new feature. CreateOutput finished the GPU's work first, so the old
        // one is no longer in use.
        if (size_changed && feature) {
            NVSDK_NGX_VULKAN_ReleaseFeature(feature);
            feature = nullptr;
        }
        if (!feature) {
            NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                                      preset);
            NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                                      preset);
            NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
                                      preset);
            NVSDK_NGX_Parameter_SetUI(
                params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, preset);
            NVSDK_NGX_Parameter_SetUI(
                params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, preset);
            NVSDK_NGX_Parameter_SetUI(
                params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality, preset);
            NVSDK_NGX_DLSS_Create_Params create{};
            create.Feature.InWidth = in.render_size.width;
            create.Feature.InHeight = in.render_size.height;
            create.Feature.InTargetWidth = in.upscale_size.width;
            create.Feature.InTargetHeight = in.upscale_size.height;
            create.Feature.InPerfQualityValue = mode;
            create.InFeatureCreateFlags = create_flags;
            const NVSDK_NGX_Result result =
                NGX_VULKAN_CREATE_DLSS_EXT(*cmdbuf, 1, 1, &feature, params, &create);
            if (NVSDK_NGX_FAILED(result) || !feature) {
                feature = nullptr;
                fail("feature creation", result);
                return;
            }
        }

        constexpr VkAccessFlags WRITES =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        // The inputs come from the game's draws and dispatches and stay in GENERAL otherwise;
        // DLSS reads them as sampled images.
        const std::array to_read{
            Barrier(in.color, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.depth, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.motion_vectors, WRITES, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);

        NVSDK_NGX_Resource_VK color = Resource(in.color, in.render_size);
        NVSDK_NGX_Resource_VK depth = Resource(in.depth, in.render_size);
        NVSDK_NGX_Resource_VK motion = Resource(in.motion_vectors, in.render_size);
        NVSDK_NGX_Resource_VK output = NVSDK_NGX_Create_ImageView_Resource_VK(
            output_view_handle, output_image_handle,
            VkImageSubresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            OUTPUT_FORMAT, in.upscale_size.width, in.upscale_size.height, true);
        NVSDK_NGX_VK_DLSS_Eval_Params eval{};
        eval.Feature.pInColor = &color;
        eval.Feature.pInOutput = &output;
        eval.pInDepth = &depth;
        eval.pInMotionVectors = &motion;
        eval.InJitterOffsetX = in.jitter[0];
        eval.InJitterOffsetY = in.jitter[1];
        eval.InRenderSubrectDimensions = {in.render_size.width, in.render_size.height};
        eval.InReset = reset ? 1 : 0;
        // NGX scales motion vectors into render pixels, which is what motion_vector_scale does
        // (Streamline instead wants them in [-1,1] screen units).
        eval.InMVScaleX = in.motion_vector_scale[0];
        eval.InMVScaleY = in.motion_vector_scale[1];
        const NVSDK_NGX_Result result = NGX_VULKAN_EVALUATE_DLSS_EXT(*cmdbuf, feature, params, &eval);

        const std::array to_general{
            Barrier(in.color, VK_ACCESS_SHADER_READ_BIT, 0,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.depth, VK_ACCESS_SHADER_READ_BIT, 0,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.motion_vectors, VK_ACCESS_SHADER_READ_BIT, 0,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        const VkMemoryBarrier output_written{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, output_written, {},
                               to_general);
        if (NVSDK_NGX_FAILED(result)) {
            fail("evaluation", result);
        }
    });
    // DLSS binds its own pipelines and descriptors on the command buffer.
    scheduler.InvalidateState();
    return true;
}

} // namespace Vulkan
