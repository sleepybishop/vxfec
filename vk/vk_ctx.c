#include "vk_internal.h"
#include "vk_spv.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif

static bool find_mem_type(VkPhysicalDevice phys, uint32_t type_bits, VkMemoryPropertyFlags req, VkMemoryPropertyFlags pref,
                          uint32_t *out_idx)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    if (pref) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & (req | pref)) == (req | pref)) {
                *out_idx = i;
                return true;
            }
        }
    }
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & req) == req) {
            *out_idx = i;
            return true;
        }
    }
    return false;
}

static bool create_raw_buffer(vxfec_vk_ctx *ctx, size_t bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags req,
                              VkMemoryPropertyFlags pref, VkBuffer *out_buf, VkDeviceMemory *out_mem)
{
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes ? bytes : 16,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(ctx->dev, &bci, NULL, out_buf) != VK_SUCCESS)
        return false;

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx->dev, *out_buf, &mr);

    uint32_t mem_idx = 0;
    if (!find_mem_type(ctx->phys, mr.memoryTypeBits, req, pref, &mem_idx)) {
        vkDestroyBuffer(ctx->dev, *out_buf, NULL);
        *out_buf = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = mem_idx,
    };
    if (vkAllocateMemory(ctx->dev, &mai, NULL, out_mem) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->dev, *out_buf, NULL);
        *out_buf = VK_NULL_HANDLE;
        return false;
    }
    if (vkBindBufferMemory(ctx->dev, *out_buf, *out_mem, 0) != VK_SUCCESS) {
        vkFreeMemory(ctx->dev, *out_mem, NULL);
        vkDestroyBuffer(ctx->dev, *out_buf, NULL);
        *out_buf = VK_NULL_HANDLE;
        *out_mem = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

static bool build_all_pipelines(vxfec_vk_ctx *ctx)
{
    struct {
        VkPipelineLayout layout;
        const uint32_t *spv;
        size_t bytes;
        VkPipeline *dst;
    } specs[] = {
        {ctx->pl_run, spv_ops_run_vec4_gf256, spv_ops_run_vec4_gf256_bytes, &ctx->pipe_vec4_gf256},
        {ctx->pl_run, spv_ops_run_vec4_gf2, spv_ops_run_vec4_gf2_bytes, &ctx->pipe_vec4_gf2},
        {ctx->pl_run, spv_ops_run_vram_gf256, spv_ops_run_vram_gf256_bytes, &ctx->pipe_vram_gf256},
        {ctx->pl_run, spv_ops_run_vram_gf2, spv_ops_run_vram_gf2_bytes, &ctx->pipe_vram_gf2},
        {ctx->pl_mix, spv_ops_mix_vec4, spv_ops_mix_vec4_bytes, &ctx->pipe_mix_vec4},
        {ctx->pl_mix, spv_ops_mix, spv_ops_mix_bytes, &ctx->pipe_mix},
        {ctx->pl_mix, spv_ops_direct_gf2_vec4_16, spv_ops_direct_gf2_vec4_16_bytes, &ctx->pipe_direct_gf2_vec4_16},
        {ctx->pl_mix, spv_ops_direct_gf2_vec4_32, spv_ops_direct_gf2_vec4_32_bytes, &ctx->pipe_direct_gf2_vec4_32},
        {ctx->pl_mix, spv_ops_direct_gf2_16, spv_ops_direct_gf2_16_bytes, &ctx->pipe_direct_gf2_16},
        {ctx->pl_mix, spv_ops_direct_gf2_32, spv_ops_direct_gf2_32_bytes, &ctx->pipe_direct_gf2_32},
        {ctx->pl_mix, spv_ops_direct_gf256_nib_vec4_8, spv_ops_direct_gf256_nib_vec4_8_bytes, &ctx->pipe_direct_gf256_nib_vec4_8},
        {ctx->pl_mix, spv_ops_direct_gf256_nib_vec4_16, spv_ops_direct_gf256_nib_vec4_16_bytes,
         &ctx->pipe_direct_gf256_nib_vec4_16},
        {ctx->pl_mix, spv_ops_direct_gf256_nib_8, spv_ops_direct_gf256_nib_8_bytes, &ctx->pipe_direct_gf256_nib_8},
        {ctx->pl_mix, spv_ops_direct_gf256_nib_16, spv_ops_direct_gf256_nib_16_bytes, &ctx->pipe_direct_gf256_nib_16},
        {ctx->pl_mix, spv_ops_wire_gf2_vec4_16, spv_ops_wire_gf2_vec4_16_bytes, &ctx->pipe_wire_gf2_vec4_16},
        {ctx->pl_mix, spv_ops_wire_gf2_vec4_32, spv_ops_wire_gf2_vec4_32_bytes, &ctx->pipe_wire_gf2_vec4_32},
        {ctx->pl_mix, spv_ops_wire_gf256_nib_vec4_8, spv_ops_wire_gf256_nib_vec4_8_bytes, &ctx->pipe_wire_gf256_nib_vec4_8},
        {ctx->pl_mix, spv_ops_wire_gf256_nib_vec4_16, spv_ops_wire_gf256_nib_vec4_16_bytes, &ctx->pipe_wire_gf256_nib_vec4_16},
    };
    enum { N_PIPES = sizeof(specs) / sizeof(specs[0]) };

    VkShaderModule mods[N_PIPES] = {0};
    VkComputePipelineCreateInfo cpcis[N_PIPES] = {0};
    VkPipeline pipes[N_PIPES] = {0};
    bool ok = true;

    for (size_t i = 0; i < N_PIPES; i++) {
        VkShaderModuleCreateInfo smci = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = specs[i].bytes,
            .pCode = specs[i].spv,
        };
        if (vkCreateShaderModule(ctx->dev, &smci, NULL, &mods[i]) != VK_SUCCESS) {
            ok = false;
            break;
        }
        cpcis[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpcis[i].stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpcis[i].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpcis[i].stage.module = mods[i];
        cpcis[i].stage.pName = "main";
        cpcis[i].layout = specs[i].layout;
    }

    if (ok && vkCreateComputePipelines(ctx->dev, VK_NULL_HANDLE, N_PIPES, cpcis, NULL, pipes) != VK_SUCCESS)
        ok = false;

    for (size_t i = 0; i < N_PIPES; i++) {
        if (mods[i])
            vkDestroyShaderModule(ctx->dev, mods[i], NULL);
        if (ok) {
            *specs[i].dst = pipes[i];
        } else if (pipes[i]) {
            vkDestroyPipeline(ctx->dev, pipes[i], NULL);
        }
    }
    return ok;
}

static int device_score(VkPhysicalDeviceType t)
{
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return 400;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return 300;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return 200;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return 100;
    default:
        return 10;
    }
}

static bool vk_idle_confirmed(VkDevice dev)
{
    VkResult res = vkDeviceWaitIdle(dev);
    /* Vulkan guarantees all queue execution has ceased on VK_SUCCESS or VK_ERROR_DEVICE_LOST */
    return res == VK_SUCCESS || res == VK_ERROR_DEVICE_LOST;
}

static void reap_retired_bufs(vxfec_vk_ctx *ctx)
{
    while (ctx->retired_bufs) {
        vxfec_vk_buf *b = ctx->retired_bufs;
        ctx->retired_bufs = b->next_retired;
        b->next_retired = NULL;
        vxfec_vk_buf_free(ctx, b);
    }
}

static bool has_pending_async_slots(vxfec_vk_ctx *ctx)
{
    if (!ctx || !ctx->has_timeline_sem || !ctx->timeline_sem)
        return false;
    uint64_t done_val = 0;
    bool have_val = (vkGetSemaphoreCounterValue(ctx->dev, ctx->timeline_sem, &done_val) == VK_SUCCESS);
    bool any_pending = false;
    for (uint32_t i = 0; i < VXFEC_VK_ASYNC_SLOTS; i++) {
        if (ctx->async_slot_tickets[i] == 0)
            continue;
        if (have_val && ctx->async_slot_tickets[i] <= done_val)
            ctx->async_slot_tickets[i] = 0;
        else
            any_pending = true;
    }
    return any_pending;
}

static bool ensure_sync_cmd_idle(vxfec_vk_ctx *ctx)
{
    if (!ctx->last_submit_unconfirmed && !ctx->retired_bufs)
        return true;
    if (!ctx->last_submit_unconfirmed) {
        if (!has_pending_async_slots(ctx))
            reap_retired_bufs(ctx);
        return true;
    }
    if (!vk_idle_confirmed(ctx->dev))
        return false;
    ctx->last_submit_unconfirmed = false;
    for (uint32_t i = 0; i < VXFEC_VK_ASYNC_SLOTS; i++)
        ctx->async_slot_tickets[i] = 0;
    if (ctx->pending_release_buf) {
        ctx->pending_release_buf->owned_by_vk = false;
        ctx->pending_release_buf = NULL;
    }
    reap_retired_bufs(ctx);
    return true;
}

static void commit_pending_owned(vxfec_vk_ctx *ctx)
{
    for (uint32_t i = 0; i < ctx->pending_owned_cnt; i++) {
        if (ctx->pending_owned_bufs[i])
            ctx->pending_owned_bufs[i]->owned_by_vk = true;
    }
    ctx->pending_owned_cnt = 0;
}

bool vxfec_vk__submit_and_wait(vxfec_vk_ctx *ctx)
{
    ctx->last_submit_unconfirmed = false;
    if (vkEndCommandBuffer(ctx->cmd) != VK_SUCCESS) {
        ctx->pending_owned_cnt = 0;
        ctx->pending_release_buf = NULL;
        return false;
    }
    if (vkResetFences(ctx->dev, 1, &ctx->fence) != VK_SUCCESS) {
        ctx->pending_owned_cnt = 0;
        ctx->pending_release_buf = NULL;
        return false;
    }
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd,
    };
    if (vkQueueSubmit(ctx->queue, 1, &si, ctx->fence) != VK_SUCCESS) {
        ctx->pending_owned_cnt = 0;
        ctx->pending_release_buf = NULL;
        return false;
    }
    commit_pending_owned(ctx);
    if (vkWaitForFences(ctx->dev, 1, &ctx->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        if (vk_idle_confirmed(ctx->dev)) {
            if (ctx->pending_release_buf) {
                ctx->pending_release_buf->owned_by_vk = false;
                ctx->pending_release_buf = NULL;
            }
            reap_retired_bufs(ctx);
        } else {
            ctx->last_submit_unconfirmed = true;
        }
        return false;
    }
    if (ctx->pending_release_buf) {
        ctx->pending_release_buf->owned_by_vk = false;
        ctx->pending_release_buf = NULL;
    }
    return true;
}

/* reset only ctx->cmd so any in-flight async_cmds[0..31] remain undisturbed */
bool vxfec_vk__begin_cmd(vxfec_vk_ctx *ctx)
{
    if (!ensure_sync_cmd_idle(ctx))
        return false;
    ctx->pending_owned_cnt = 0;
    ctx->pending_release_buf = NULL;
    if (vkResetCommandBuffer(ctx->cmd, 0) != VK_SUCCESS)
        return false;
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    return vkBeginCommandBuffer(ctx->cmd, &bi) == VK_SUCCESS;
}

static bool wait_async_idle(vxfec_vk_ctx *ctx)
{
    if (!ctx || !ctx->has_timeline_sem || !ctx->timeline_sem || ctx->timeline_submitted <= 1u)
        return true;
    return vxfec_vk_wait(ctx, ctx->timeline_submitted, UINT64_MAX);
}

static bool parse_env_uint(const char *primary, bool *out_forced, uint32_t *out_idx)
{
    const char *env = getenv(primary);
    if (!env || env[0] == '\0') {
        *out_forced = false;
        *out_idx = 0;
        return true;
    }
    char *end = NULL;
    errno = 0;
    long val = strtol(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || val < 0 || val > INT_MAX)
        return false;
    *out_forced = true;
    *out_idx = (uint32_t)val;
    return true;
}

static bool has_dev_ext(const VkExtensionProperties *exts, uint32_t count, const char *name)
{
    for (uint32_t i = 0; i < count; i++) {
        if (strcmp(exts[i].extensionName, name) == 0)
            return true;
    }
    return false;
}

static vxfec_vk_ctx *ctx_new_attempt(int *status, VkInstance shared_inst, const VkPhysicalDevice *devs, uint32_t dev_count,
                                     bool forced_dev, uint32_t forced_dev_idx, bool forced_qfam, uint32_t forced_qfam_idx,
                                     int previous_score, int previous_index, int *selected_score, int *selected_index)
{
    if (status)
        *status = VXFEC_VK_ERROR;
    vxfec_vk_ctx *ctx = (vxfec_vk_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;

    int best_score = -1;
    for (uint32_t i = 0; i < dev_count; i++) {
        if (forced_dev && forced_dev_idx != i)
            continue;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        if (props.apiVersion < VK_API_VERSION_1_2)
            continue;
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qcount, NULL);
        if (qcount == 0)
            continue;
        VkQueueFamilyProperties *qprops = (VkQueueFamilyProperties *)calloc(qcount, sizeof(*qprops));
        if (!qprops)
            goto fail;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qcount, qprops);
        int found_q = -1;
        uint32_t ts_bits = 0;
        bool found_dedicated = false;
        for (uint32_t q = 0; q < qcount; q++) {
            if (forced_qfam && forced_qfam_idx != q)
                continue;
            VkQueueFlags fl = qprops[q].queueFlags;
            if (!(fl & VK_QUEUE_COMPUTE_BIT))
                continue;
            bool ded = ((fl & VK_QUEUE_GRAPHICS_BIT) == 0);
            if (found_q < 0 || (ded && !found_dedicated)) {
                found_q = (int)q;
                ts_bits = qprops[q].timestampValidBits;
                found_dedicated = ded;
                if (ded)
                    break;
            }
        }
        free(qprops);
        if (found_q < 0)
            continue;

        int sc = device_score(props.deviceType);
        if (!forced_dev && (sc > previous_score || (sc == previous_score && (int)i <= previous_index)))
            continue;
        if (sc > best_score) {
            best_score = sc;
            *selected_score = sc;
            *selected_index = (int)i;
            ctx->phys = devs[i];
            ctx->qfam = (uint32_t)found_q;
            ctx->dedicated_compute = found_dedicated;
            ctx->vendor_id = props.vendorID;
            memcpy(ctx->dev_name, props.deviceName, sizeof(ctx->dev_name) - 1);
            ctx->is_discrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
            ctx->max_shared_bytes = props.limits.maxComputeSharedMemorySize;
            ctx->max_wg_x = props.limits.maxComputeWorkGroupCount[0];
            ctx->max_wg_y = props.limits.maxComputeWorkGroupCount[1];
            ctx->max_wg_z = props.limits.maxComputeWorkGroupCount[2];
            ctx->max_ssbo_range = props.limits.maxStorageBufferRange;
            ctx->ssbo_offset_align =
                props.limits.minStorageBufferOffsetAlignment ? props.limits.minStorageBufferOffsetAlignment : 16u;
            ctx->ts_valid_bits = ts_bits;
            ctx->timestamp_period_ns = props.limits.timestampPeriod;
            ctx->has_timestamps = (ts_bits > 0 && props.limits.timestampPeriod > 0.0f);
        }
    }
    if (best_score < 0) {
        if (status && !forced_dev && !forced_qfam)
            *status = VXFEC_VK_UNAVAILABLE;
        goto fail;
    }

    VkPhysicalDeviceVulkan12Features f12_sup = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    };
    VkPhysicalDeviceFeatures2 f2_sup = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f12_sup,
    };
    vkGetPhysicalDeviceFeatures2(ctx->phys, &f2_sup);
    ctx->has_timeline_sem = (f12_sup.timelineSemaphore == VK_TRUE);

    uint32_t ext_cnt = 0;
    const char *enabled_exts[5];
    uint32_t n_enabled_exts = 0;
    if (vkEnumerateDeviceExtensionProperties(ctx->phys, NULL, &ext_cnt, NULL) == VK_SUCCESS && ext_cnt > 0) {
        VkExtensionProperties *exts = (VkExtensionProperties *)calloc(ext_cnt, sizeof(*exts));
        if (exts && vkEnumerateDeviceExtensionProperties(ctx->phys, NULL, &ext_cnt, exts) == VK_SUCCESS) {
            bool sup_ext_mem = has_dev_ext(exts, ext_cnt, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
            if (has_dev_ext(exts, ext_cnt, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) &&
                has_dev_ext(exts, ext_cnt, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)) {
                if (sup_ext_mem)
                    enabled_exts[n_enabled_exts++] = VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME;
                enabled_exts[n_enabled_exts++] = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
                enabled_exts[n_enabled_exts++] = VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME;
                ctx->has_ext_mem_fd = true;
                ctx->has_dmabuf = true;
                if (has_dev_ext(exts, ext_cnt, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
                    enabled_exts[n_enabled_exts++] = VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME;
                    ctx->has_foreign_qfam = true;
                }
            }
            if (has_dev_ext(exts, ext_cnt, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
                if (sup_ext_mem && !ctx->has_ext_mem_fd)
                    enabled_exts[n_enabled_exts++] = VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME;
                enabled_exts[n_enabled_exts++] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
                ctx->has_ext_host_mem = true;
            }
        }
        free(exts);
    }

    if (ctx->has_dmabuf) {
        VkPhysicalDeviceExternalBufferInfo ebi = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
            .flags = 0,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        };
        VkExternalBufferProperties ebp = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalBufferProperties(ctx->phys, &ebi, &ebp);
        VkExternalMemoryFeatureFlags feat = ebp.externalMemoryProperties.externalMemoryFeatures;
        ctx->has_dmabuf = (feat & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        ctx->has_dmabuf_export = (feat & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
        ctx->dmabuf_dedicated_only = (feat & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;
    }

    if (ctx->has_ext_host_mem) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
        };
        VkPhysicalDeviceProperties2 p2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &hp,
        };
        vkGetPhysicalDeviceProperties2(ctx->phys, &p2);
        ctx->host_import_align = hp.minImportedHostPointerAlignment ? hp.minImportedHostPointerAlignment : 4096u;
    }

    VkPhysicalDeviceVulkan12Features f12_en = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .timelineSemaphore = ctx->has_timeline_sem ? VK_TRUE : VK_FALSE,
    };

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo dqci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->qfam,
        .queueCount = 1,
        .pQueuePriorities = &qprio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = ctx->has_timeline_sem ? &f12_en : NULL,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &dqci,
        .enabledExtensionCount = n_enabled_exts,
        .ppEnabledExtensionNames = n_enabled_exts ? enabled_exts : NULL,
    };
    if (vkCreateDevice(ctx->phys, &dci, NULL, &ctx->dev) != VK_SUCCESS)
        goto fail;
    vkGetDeviceQueue(ctx->dev, ctx->qfam, 0, &ctx->queue);

    if (ctx->has_ext_mem_fd) {
        ctx->pfn_get_mem_fd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ctx->dev, "vkGetMemoryFdKHR");
        ctx->pfn_get_mem_fd_props = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(ctx->dev, "vkGetMemoryFdPropertiesKHR");
    }
    if (ctx->has_ext_host_mem) {
        ctx->pfn_get_host_ptr_props =
            (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(ctx->dev, "vkGetMemoryHostPointerPropertiesEXT");
    }

    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->qfam,
    };
    if (vkCreateCommandPool(ctx->dev, &cpci, NULL, &ctx->cmd_pool) != VK_SUCCESS)
        goto fail;

    VkCommandBuffer all_cmds[1u + VXFEC_VK_ASYNC_SLOTS];
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1u + VXFEC_VK_ASYNC_SLOTS,
    };
    if (vkAllocateCommandBuffers(ctx->dev, &cbai, all_cmds) != VK_SUCCESS)
        goto fail;
    ctx->cmd = all_cmds[0];
    for (uint32_t s = 0; s < VXFEC_VK_ASYNC_SLOTS; s++)
        ctx->async_cmds[s] = all_cmds[1u + s];

    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(ctx->dev, &fci, NULL, &ctx->fence) != VK_SUCCESS)
        goto fail;

    if (ctx->has_timeline_sem) {
        VkSemaphoreTypeCreateInfo stci = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = 1,
        };
        VkSemaphoreCreateInfo sci = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &stci,
        };
        if (vkCreateSemaphore(ctx->dev, &sci, NULL, &ctx->timeline_sem) != VK_SUCCESS) {
            ctx->has_timeline_sem = false;
        } else {
            ctx->timeline_submitted = 1u;
            ctx->timeline_next = 2u;
        }
    }

    if (ctx->has_timestamps) {
        VkQueryPoolCreateInfo qpci = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
        };
        if (vkCreateQueryPool(ctx->dev, &qpci, NULL, &ctx->query_pool) != VK_SUCCESS)
            ctx->has_timestamps = false;
    }

    VkDescriptorSetLayoutBinding run_bindings[2] = {
        {.binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo dsl_run_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = run_bindings,
    };
    if (vkCreateDescriptorSetLayout(ctx->dev, &dsl_run_ci, NULL, &ctx->dsl_run) != VK_SUCCESS)
        goto fail;

    VkDescriptorSetLayoutBinding mix_bindings[3] = {
        {.binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo dsl_mix_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = mix_bindings,
    };
    if (vkCreateDescriptorSetLayout(ctx->dev, &dsl_mix_ci, NULL, &ctx->dsl_mix) != VK_SUCCESS)
        goto fail;

    VkPushConstantRange pcr_run = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(vxfec_run_push_consts),
    };
    VkPipelineLayoutCreateInfo pl_run_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx->dsl_run,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr_run,
    };
    if (vkCreatePipelineLayout(ctx->dev, &pl_run_ci, NULL, &ctx->pl_run) != VK_SUCCESS)
        goto fail;

    VkPushConstantRange pcr_mix = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(vxfec_mix_push_consts),
    };
    VkPipelineLayoutCreateInfo pl_mix_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx->dsl_mix,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr_mix,
    };
    if (vkCreatePipelineLayout(ctx->dev, &pl_mix_ci, NULL, &ctx->pl_mix) != VK_SUCCESS)
        goto fail;

    uint32_t total_sets = 3u + 3u * VXFEC_VK_ASYNC_SLOTS;
    VkDescriptorPoolSize pool_sz = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = total_sets * 3u,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = total_sets,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_sz,
    };
    if (vkCreateDescriptorPool(ctx->dev, &dpci, NULL, &ctx->desc_pool) != VK_SUCCESS)
        goto fail;

    VkDescriptorSetLayout layouts[3u + 3u * VXFEC_VK_ASYNC_SLOTS];
    VkDescriptorSet sets[3u + 3u * VXFEC_VK_ASYNC_SLOTS];
    layouts[0] = ctx->dsl_run;
    layouts[1] = ctx->dsl_mix;
    layouts[2] = ctx->dsl_mix;
    for (uint32_t s = 0; s < VXFEC_VK_ASYNC_SLOTS; s++) {
        layouts[3u + 3u * s] = ctx->dsl_run;
        layouts[3u + 3u * s + 1u] = ctx->dsl_mix;
        layouts[3u + 3u * s + 2u] = ctx->dsl_mix;
    }
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = total_sets,
        .pSetLayouts = layouts,
    };
    if (vkAllocateDescriptorSets(ctx->dev, &dsai, sets) != VK_SUCCESS)
        goto fail;
    ctx->ds_run = sets[0];
    ctx->ds_mix = sets[1];
    ctx->ds_mix_aux = sets[2];
    for (uint32_t s = 0; s < VXFEC_VK_ASYNC_SLOTS; s++) {
        ctx->async_ds_run[s] = sets[3u + 3u * s];
        ctx->async_ds_mix[s] = sets[3u + 3u * s + 1u];
        ctx->async_ds_mix_aux[s] = sets[3u + 3u * s + 2u];
    }

    if (!build_all_pipelines(ctx))
        goto fail;

    ctx->instance = shared_inst;
    if (status)
        *status = VXFEC_VK_OK;
    return ctx;

fail:
    vxfec_vk_ctx_free(ctx);
    return NULL;
}

bool vxfec_vk_ctx_supports_dmabuf_fd(vxfec_vk_ctx *ctx, int dmabuf_fd)
{
#if defined(_WIN32)
    (void)ctx;
    (void)dmabuf_fd;
    return false;
#else
    if (!ctx || dmabuf_fd < 0 || !ctx->has_ext_mem_fd || !ctx->has_dmabuf || !ctx->pfn_get_mem_fd_props)
        return false;

    VkExternalMemoryBufferCreateInfo embci = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &embci,
        .size = 4096u,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer probe_buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(ctx->dev, &bci, NULL, &probe_buf) != VK_SUCCESS)
        return false;

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx->dev, probe_buf, &mr);
    vkDestroyBuffer(ctx->dev, probe_buf, NULL);

    VkMemoryFdPropertiesKHR fd_props = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
    };
    if (ctx->pfn_get_mem_fd_props(ctx->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dmabuf_fd, &fd_props) != VK_SUCCESS)
        return false;

    uint32_t mem_idx = 0;
    return find_mem_type(ctx->phys, mr.memoryTypeBits & fd_props.memoryTypeBits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mem_idx);
#endif
}

static vxfec_vk_ctx *ctx_new_select(int *status, int dmabuf_fd)
{
    bool forced_dev = false, forced_qfam = false;
    uint32_t forced_dev_idx = 0, forced_qfam_idx = 0;
    if (!parse_env_uint("VXFEC_VK_DEVICE", &forced_dev, &forced_dev_idx) ||
        !parse_env_uint("VXFEC_VK_QUEUE_FAMILY", &forced_qfam, &forced_qfam_idx)) {
        if (status)
            *status = VXFEC_VK_ERROR;
        return NULL;
    }

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vxfec",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance inst = VK_NULL_HANDLE;
    VkResult ir = vkCreateInstance(&ici, NULL, &inst);
    if (ir != VK_SUCCESS) {
        if (status)
            *status = (!forced_dev && !forced_qfam && ir == VK_ERROR_INCOMPATIBLE_DRIVER) ? VXFEC_VK_UNAVAILABLE : VXFEC_VK_ERROR;
        return NULL;
    }

    uint32_t dev_count = 0;
    VkResult er = vkEnumeratePhysicalDevices(inst, &dev_count, NULL);
    if (er != VK_SUCCESS || dev_count == 0) {
        vkDestroyInstance(inst, NULL);
        if (status)
            *status = (er == VK_SUCCESS && dev_count == 0 && !forced_dev && !forced_qfam) ? VXFEC_VK_UNAVAILABLE : VXFEC_VK_ERROR;
        return NULL;
    }

    VkPhysicalDevice *devs = (VkPhysicalDevice *)calloc(dev_count, sizeof(*devs));
    if (!devs || vkEnumeratePhysicalDevices(inst, &dev_count, devs) != VK_SUCCESS) {
        free(devs);
        vkDestroyInstance(inst, NULL);
        if (status)
            *status = VXFEC_VK_ERROR;
        return NULL;
    }

    int previous_score = INT_MAX, previous_index = -1;
    bool attempted = false;
    for (;;) {
        int current_status = VXFEC_VK_ERROR;
        int selected_score = INT_MAX, selected_index = -1;
        vxfec_vk_ctx *ctx = ctx_new_attempt(&current_status, inst, devs, dev_count, forced_dev, forced_dev_idx, forced_qfam,
                                            forced_qfam_idx, previous_score, previous_index, &selected_score, &selected_index);
        if (ctx) {
            if (dmabuf_fd < 0 || vxfec_vk_ctx_supports_dmabuf_fd(ctx, dmabuf_fd)) {
                free(devs);
                if (status)
                    *status = VXFEC_VK_OK;
                return ctx;
            }
            /* preserve shared VkInstance while probing remaining physical devices */
            ctx->instance = VK_NULL_HANDLE;
            vxfec_vk_ctx_free(ctx);
        }
        if (selected_index >= 0)
            attempted = true;
        if (forced_dev || selected_index < 0) {
            free(devs);
            vkDestroyInstance(inst, NULL);
            if (status)
                *status = (forced_dev || forced_qfam || attempted) ? VXFEC_VK_ERROR : current_status;
            return NULL;
        }
        previous_score = selected_score;
        previous_index = selected_index;
    }
}

vxfec_vk_ctx *vxfec_vk_ctx_new_status(int *status)
{
    return ctx_new_select(status, -1);
}

vxfec_vk_ctx *vxfec_vk_ctx_new_for_dmabuf(int dmabuf_fd, int *status)
{
#if defined(_WIN32)
    (void)dmabuf_fd;
    if (status)
        *status = VXFEC_VK_ERROR;
    return NULL;
#else
    if (dmabuf_fd < 0) {
        if (status)
            *status = VXFEC_VK_ERROR;
        return NULL;
    }
    return ctx_new_select(status, dmabuf_fd);
#endif
}

vxfec_vk_ctx *vxfec_vk_ctx_new(void)
{
    return vxfec_vk_ctx_new_status(NULL);
}

bool vxfec_vk_wait_idle(vxfec_vk_ctx *ctx)
{
    if (!ctx || !ctx->dev)
        return false;
    if (!vk_idle_confirmed(ctx->dev))
        return false;
    ctx->last_submit_unconfirmed = false;
    for (uint32_t i = 0; i < VXFEC_VK_ASYNC_SLOTS; i++)
        ctx->async_slot_tickets[i] = 0;
    if (ctx->pending_release_buf) {
        ctx->pending_release_buf->owned_by_vk = false;
        ctx->pending_release_buf = NULL;
    }
    reap_retired_bufs(ctx);
    return true;
}

void vxfec_vk_ctx_free(vxfec_vk_ctx *ctx)
{
    if (!ctx)
        return;
    if (ctx->dev) {
        if (!vk_idle_confirmed(ctx->dev))
            return;
        ctx->last_submit_unconfirmed = false;
        for (uint32_t i = 0; i < VXFEC_VK_ASYNC_SLOTS; i++)
            ctx->async_slot_tickets[i] = 0;
        if (ctx->pending_release_buf) {
            ctx->pending_release_buf->owned_by_vk = false;
            ctx->pending_release_buf = NULL;
        }
        reap_retired_bufs(ctx);
        if (ctx->io_stage_ptr && ctx->io_stage_mem)
            vkUnmapMemory(ctx->dev, ctx->io_stage_mem);
        if (ctx->io_stage_buf)
            vkDestroyBuffer(ctx->dev, ctx->io_stage_buf, NULL);
        if (ctx->io_stage_mem)
            vkFreeMemory(ctx->dev, ctx->io_stage_mem, NULL);
        vxfec_vk_buf_free(ctx, ctx->mix_tab_buf);
        vxfec_vk_buf_free(ctx, ctx->direct_tab_buf);
        for (uint32_t s = 0; s < VXFEC_VK_ASYNC_SLOTS; s++)
            vxfec_vk_buf_free(ctx, ctx->async_tab_bufs[s]);
        if (ctx->pipe_vec4_gf256)
            vkDestroyPipeline(ctx->dev, ctx->pipe_vec4_gf256, NULL);
        if (ctx->pipe_vec4_gf2)
            vkDestroyPipeline(ctx->dev, ctx->pipe_vec4_gf2, NULL);
        if (ctx->pipe_vram_gf256)
            vkDestroyPipeline(ctx->dev, ctx->pipe_vram_gf256, NULL);
        if (ctx->pipe_vram_gf2)
            vkDestroyPipeline(ctx->dev, ctx->pipe_vram_gf2, NULL);
        if (ctx->pipe_mix_vec4)
            vkDestroyPipeline(ctx->dev, ctx->pipe_mix_vec4, NULL);
        if (ctx->pipe_mix)
            vkDestroyPipeline(ctx->dev, ctx->pipe_mix, NULL);
        if (ctx->pipe_direct_gf2_vec4_16)
            vkDestroyPipeline(ctx->dev, ctx->pipe_direct_gf2_vec4_16, NULL);
        if (ctx->pipe_direct_gf2_vec4_32)
            vkDestroyPipeline(ctx->dev, ctx->pipe_direct_gf2_vec4_32, NULL);
        if (ctx->pipe_direct_gf2_16)
            vkDestroyPipeline(ctx->dev, ctx->pipe_direct_gf2_16, NULL);
        if (ctx->pipe_direct_gf2_32)
            vkDestroyPipeline(ctx->dev, ctx->pipe_direct_gf2_32, NULL);
        if (ctx->pipe_direct_gf256_nib_vec4_8)
            vkDestroyPipeline(ctx->dev, ctx->pipe_direct_gf256_nib_vec4_8, NULL);
        if (ctx->pipe_direct_gf256_nib_vec4_16)
            vkDestroyPipeline(ctx->dev, ctx->pipe_direct_gf256_nib_vec4_16, NULL);
        if (ctx->pipe_direct_gf256_nib_8)
            vkDestroyPipeline(ctx->dev, ctx->pipe_direct_gf256_nib_8, NULL);
        if (ctx->pipe_direct_gf256_nib_16)
            vkDestroyPipeline(ctx->dev, ctx->pipe_direct_gf256_nib_16, NULL);
        if (ctx->pipe_wire_gf2_vec4_16)
            vkDestroyPipeline(ctx->dev, ctx->pipe_wire_gf2_vec4_16, NULL);
        if (ctx->pipe_wire_gf2_vec4_32)
            vkDestroyPipeline(ctx->dev, ctx->pipe_wire_gf2_vec4_32, NULL);
        if (ctx->pipe_wire_gf256_nib_vec4_8)
            vkDestroyPipeline(ctx->dev, ctx->pipe_wire_gf256_nib_vec4_8, NULL);
        if (ctx->pipe_wire_gf256_nib_vec4_16)
            vkDestroyPipeline(ctx->dev, ctx->pipe_wire_gf256_nib_vec4_16, NULL);
        if (ctx->desc_pool)
            vkDestroyDescriptorPool(ctx->dev, ctx->desc_pool, NULL);
        if (ctx->pl_run)
            vkDestroyPipelineLayout(ctx->dev, ctx->pl_run, NULL);
        if (ctx->pl_mix)
            vkDestroyPipelineLayout(ctx->dev, ctx->pl_mix, NULL);
        if (ctx->dsl_run)
            vkDestroyDescriptorSetLayout(ctx->dev, ctx->dsl_run, NULL);
        if (ctx->dsl_mix)
            vkDestroyDescriptorSetLayout(ctx->dev, ctx->dsl_mix, NULL);
        if (ctx->query_pool)
            vkDestroyQueryPool(ctx->dev, ctx->query_pool, NULL);
        if (ctx->timeline_sem)
            vkDestroySemaphore(ctx->dev, ctx->timeline_sem, NULL);
        if (ctx->fence)
            vkDestroyFence(ctx->dev, ctx->fence, NULL);
        if (ctx->cmd_pool)
            vkDestroyCommandPool(ctx->dev, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->dev, NULL);
    }
    if (ctx->instance)
        vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
}

const char *vxfec_vk_device_name(const vxfec_vk_ctx *ctx)
{
    return ctx ? ctx->dev_name : "";
}

bool vxfec_vk_is_discrete(const vxfec_vk_ctx *ctx)
{
    return ctx ? ctx->is_discrete : false;
}

bool vxfec_vk_is_dedicated_compute(const vxfec_vk_ctx *ctx)
{
    return ctx ? ctx->dedicated_compute : false;
}

uint32_t vxfec_vk_queue_family(const vxfec_vk_ctx *ctx)
{
    return ctx ? ctx->qfam : 0;
}

uint32_t vxfec_vk_shared_mem_bytes(const vxfec_vk_ctx *ctx)
{
    return ctx ? ctx->max_shared_bytes : 0;
}

uint64_t vxfec_vk_max_storage_buffer_bytes(const vxfec_vk_ctx *ctx)
{
    return ctx ? (uint64_t)ctx->max_ssbo_range : 0;
}

bool vxfec_vk_has_dmabuf(const vxfec_vk_ctx *ctx)
{
    return ctx ? (ctx->has_ext_mem_fd && ctx->has_dmabuf) : false;
}

bool vxfec_vk_has_dmabuf_export(const vxfec_vk_ctx *ctx)
{
    return ctx ? (ctx->has_ext_mem_fd && ctx->has_dmabuf_export) : false;
}

bool vxfec_vk_has_ext_host_mem(const vxfec_vk_ctx *ctx)
{
    return ctx ? ctx->has_ext_host_mem : false;
}

bool vxfec_vk_has_foreign_queue(const vxfec_vk_ctx *ctx)
{
    return ctx ? ctx->has_foreign_qfam : false;
}

vxfec_vk_buf *vxfec_vk_buf_new(vxfec_vk_ctx *ctx, size_t bytes, bool with_staging)
{
    if (!ctx || bytes == 0)
        return NULL;
    vxfec_vk_buf *b = (vxfec_vk_buf *)calloc(1, sizeof(*b));
    if (!b)
        return NULL;
    b->bytes = bytes;
    b->ext_kind = VXFEC_VK_EXT_NONE;
    b->owned_by_vk = true;

    VkBufferUsageFlags dev_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    if (with_staging && !ctx->is_discrete) {
        VkMemoryPropertyFlags uma_req = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        if (create_raw_buffer(ctx, bytes, dev_usage, uma_req, 0, &b->dev_buf, &b->dev_mem)) {
            if (vkMapMemory(ctx->dev, b->dev_mem, 0, bytes, 0, &b->stage_ptr) == VK_SUCCESS) {
                b->zero_copy = true;
                return b;
            }
            vkFreeMemory(ctx->dev, b->dev_mem, NULL);
            vkDestroyBuffer(ctx->dev, b->dev_buf, NULL);
            b->dev_mem = VK_NULL_HANDLE;
            b->dev_buf = VK_NULL_HANDLE;
        }
    }

    if (!create_raw_buffer(ctx, bytes, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &b->dev_buf, &b->dev_mem)) {
        free(b);
        return NULL;
    }

    if (with_staging) {
        VkBufferUsageFlags st_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkMemoryPropertyFlags req = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if (!create_raw_buffer(ctx, bytes, st_usage, req, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &b->stage_buf, &b->stage_mem) ||
            vkMapMemory(ctx->dev, b->stage_mem, 0, bytes, 0, &b->stage_ptr) != VK_SUCCESS) {
            vxfec_vk_buf_free(ctx, b);
            return NULL;
        }
    }
    return b;
}

void vxfec_vk__release_staging(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf)
{
    if (!ctx || !buf)
        return;
    if (buf->imported_host) {
        buf->stage_ptr = NULL;
        return;
    }
    if (buf->zero_copy) {
        if (buf->stage_ptr) {
            vkUnmapMemory(ctx->dev, buf->dev_mem);
            buf->stage_ptr = NULL;
        }
        return;
    }
    if (buf->stage_ptr) {
        vkUnmapMemory(ctx->dev, buf->stage_mem);
        buf->stage_ptr = NULL;
    }
    if (buf->stage_buf) {
        vkDestroyBuffer(ctx->dev, buf->stage_buf, NULL);
        buf->stage_buf = VK_NULL_HANDLE;
    }
    if (buf->stage_mem) {
        vkFreeMemory(ctx->dev, buf->stage_mem, NULL);
        buf->stage_mem = VK_NULL_HANDLE;
    }
}

void vxfec_vk_buf_free(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf)
{
    if (!ctx || !buf)
        return;
    vxfec_vk__release_staging(ctx, buf);
    if (buf->dev_buf)
        vkDestroyBuffer(ctx->dev, buf->dev_buf, NULL);
    if (buf->dev_mem)
        vkFreeMemory(ctx->dev, buf->dev_mem, NULL);
    free(buf);
}

void vxfec_vk__retire_buf(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf)
{
    if (!ctx || !buf)
        return;
    buf->next_retired = ctx->retired_bufs;
    ctx->retired_bufs = buf;
}

void vxfec_vk__retire_or_free_buf(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf)
{
    if (!ctx || !buf)
        return;
    if (!ensure_sync_cmd_idle(ctx) || has_pending_async_slots(ctx)) {
        vxfec_vk__retire_buf(ctx, buf);
        return;
    }
    vxfec_vk_buf_free(ctx, buf);
}

void *vxfec_vk_buf_staging_ptr(vxfec_vk_buf *buf)
{
    return buf ? buf->stage_ptr : NULL;
}

bool vxfec_vk_buf_upload(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf, size_t bytes)
{
    if (!ctx || !buf || !buf->stage_ptr || bytes == 0 || bytes > buf->bytes)
        return false;
    if (buf->zero_copy)
        return ensure_sync_cmd_idle(ctx);
    if (!vxfec_vk__begin_cmd(ctx))
        return false;
    VkBufferCopy copy = {.srcOffset = 0, .dstOffset = buf->view_offset, .size = bytes};
    vkCmdCopyBuffer(ctx->cmd, buf->stage_buf, buf->dev_buf, 1, &copy);
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    return vxfec_vk__submit_and_wait(ctx);
}

void vxfec_vk__record_transfer_to_host_cmd(VkCommandBuffer cmd)
{
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
}

bool vxfec_vk_buf_download(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf, size_t bytes)
{
    if (!ctx || !buf || !buf->stage_ptr || bytes == 0 || bytes > buf->bytes)
        return false;
    if (buf->zero_copy)
        return ensure_sync_cmd_idle(ctx) && wait_async_idle(ctx);
    if (!vxfec_vk__begin_cmd(ctx))
        return false;
    VkBufferCopy copy = {.srcOffset = buf->view_offset, .dstOffset = 0, .size = bytes};
    vkCmdCopyBuffer(ctx->cmd, buf->dev_buf, buf->stage_buf, 1, &copy);
    vxfec_vk__record_transfer_to_host_cmd(ctx->cmd);
    return vxfec_vk__submit_and_wait(ctx);
}

uint32_t vxfec_vk__ext_queue_family(const vxfec_vk_ctx *ctx, const vxfec_vk_buf *buf)
{
    if (!ctx || !buf || buf->ext_kind == VXFEC_VK_EXT_NONE)
        return VK_QUEUE_FAMILY_IGNORED;
    if (buf->ext_kind == VXFEC_VK_EXT_DMABUF && ctx->has_foreign_qfam)
        return VK_QUEUE_FAMILY_FOREIGN_EXT;
    return VK_QUEUE_FAMILY_EXTERNAL;
}

void vxfec_vk__record_ext_acquire_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, vxfec_vk_buf *buf, VkPipelineStageFlags dst_stage,
                                      VkAccessFlags dst_access)
{
    uint32_t ext_qf = vxfec_vk__ext_queue_family(ctx, buf);
    if (ext_qf == VK_QUEUE_FAMILY_IGNORED)
        return;
    VkBufferMemoryBarrier bmb = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = dst_access,
        .srcQueueFamilyIndex = ext_qf,
        .dstQueueFamilyIndex = ctx->qfam,
        .buffer = buf->dev_buf,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stage, 0, 0, NULL, 1, &bmb, 0, NULL);
}

void vxfec_vk__record_ext_release_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, vxfec_vk_buf *buf, VkPipelineStageFlags src_stage,
                                      VkAccessFlags src_access)
{
    uint32_t ext_qf = vxfec_vk__ext_queue_family(ctx, buf);
    if (ext_qf == VK_QUEUE_FAMILY_IGNORED)
        return;
    VkBufferMemoryBarrier bmb = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = 0,
        .srcQueueFamilyIndex = ctx->qfam,
        .dstQueueFamilyIndex = ext_qf,
        .buffer = buf->dev_buf,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(cmd, src_stage, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 1, &bmb, 0, NULL);
}

bool vxfec_vk__ensure_owned_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, vxfec_vk_buf *buf, VkPipelineStageFlags dst_stage,
                                VkAccessFlags dst_access)
{
    if (!ctx || !buf)
        return false;
    if (buf->owned_by_vk || buf->ext_kind == VXFEC_VK_EXT_NONE)
        return true;
    for (uint32_t i = 0; i < ctx->pending_owned_cnt; i++) {
        if (ctx->pending_owned_bufs[i] == buf)
            return true;
    }
    if (ctx->pending_owned_cnt >= VXFEC_VK_MAX_PENDING_OWNED)
        return false;
    VkPipelineStageFlags full_stage = dst_stage | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkAccessFlags full_access = dst_access | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                VK_ACCESS_TRANSFER_WRITE_BIT;
    vxfec_vk__record_ext_acquire_cmd(ctx, cmd, buf, full_stage, full_access);
    ctx->pending_owned_bufs[ctx->pending_owned_cnt++] = buf;
    return true;
}

bool vxfec_vk_buf_acquire_external(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf)
{
    if (!ctx || !buf)
        return false;
    if (buf->ext_kind == VXFEC_VK_EXT_NONE || buf->owned_by_vk)
        return true;
    if (!vxfec_vk__begin_cmd(ctx))
        return false;
    if (!vxfec_vk__ensure_owned_cmd(ctx, ctx->cmd, buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                        VK_ACCESS_TRANSFER_WRITE_BIT)) {
        ctx->pending_owned_cnt = 0;
        return false;
    }
    return vxfec_vk__submit_and_wait(ctx);
}

bool vxfec_vk_buf_release_external(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf)
{
    if (!ctx || !buf)
        return false;
    if (buf->ext_kind == VXFEC_VK_EXT_NONE || !buf->owned_by_vk)
        return true;
    if (!vxfec_vk__begin_cmd(ctx))
        return false;
    vxfec_vk__record_ext_release_cmd(ctx, ctx->cmd, buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    ctx->pending_release_buf = buf;
    return vxfec_vk__submit_and_wait(ctx);
}

bool vxfec_vk_buf_copy_offset(vxfec_vk_ctx *ctx, vxfec_vk_buf *dst, size_t dst_off, const vxfec_vk_buf *src, size_t src_off,
                              size_t bytes)
{
    if (!ctx || !dst || !src || bytes == 0 || dst_off > dst->bytes || bytes > dst->bytes - dst_off || src_off > src->bytes ||
        bytes > src->bytes - src_off)
        return false;
    size_t real_dst = dst->view_offset + dst_off;
    size_t real_src = src->view_offset + src_off;
    if (dst->dev_buf == src->dev_buf && real_dst < real_src + bytes && real_src < real_dst + bytes)
        return false;
    if (!vxfec_vk__begin_cmd(ctx))
        return false;
    if (!vxfec_vk__ensure_owned_cmd(ctx, ctx->cmd, (vxfec_vk_buf *)src, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_ACCESS_TRANSFER_READ_BIT) ||
        !vxfec_vk__ensure_owned_cmd(ctx, ctx->cmd, dst, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT)) {
        ctx->pending_owned_cnt = 0;
        return false;
    }
    VkBufferCopy copy = {.srcOffset = real_src, .dstOffset = real_dst, .size = bytes};
    vkCmdCopyBuffer(ctx->cmd, src->dev_buf, dst->dev_buf, 1, &copy);
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &barrier, 0, NULL, 0, NULL);
    return vxfec_vk__submit_and_wait(ctx);
}

bool vxfec_vk_buf_copy(vxfec_vk_ctx *ctx, vxfec_vk_buf *dst, const vxfec_vk_buf *src, size_t bytes)
{
    return vxfec_vk_buf_copy_offset(ctx, dst, 0, src, 0, bytes);
}

vxfec_vk_buf *vxfec_vk_buf_new_exportable(vxfec_vk_ctx *ctx, size_t bytes, bool is_dmabuf, int *out_fd)
{
    if (!ctx || bytes == 0 || !is_dmabuf || !out_fd || !ctx->has_ext_mem_fd || !ctx->has_dmabuf_export || !ctx->pfn_get_mem_fd)
        return NULL;
    *out_fd = -1;

    VkExternalMemoryHandleTypeFlagBits htype = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkExternalMemoryBufferCreateInfo embci = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = htype,
    };
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &embci,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    vxfec_vk_buf *b = (vxfec_vk_buf *)calloc(1, sizeof(*b));
    if (!b)
        return NULL;
    b->bytes = bytes;
    b->ext_kind = VXFEC_VK_EXT_DMABUF;
    b->owned_by_vk = true;

    if (vkCreateBuffer(ctx->dev, &bci, NULL, &b->dev_buf) != VK_SUCCESS) {
        free(b);
        return NULL;
    }

    VkMemoryDedicatedRequirements ded_req = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
    };
    VkMemoryRequirements2 mr2 = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = &ded_req,
    };
    VkBufferMemoryRequirementsInfo2 bmri = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
        .buffer = b->dev_buf,
    };
    vkGetBufferMemoryRequirements2(ctx->dev, &bmri, &mr2);

    uint32_t mem_idx = 0;
    if (!find_mem_type(ctx->phys, mr2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &mem_idx)) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }

    bool use_ded = ctx->dmabuf_dedicated_only || ded_req.requiresDedicatedAllocation == VK_TRUE;
    VkMemoryDedicatedAllocateInfo ded_ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .buffer = b->dev_buf,
    };
    VkExportMemoryAllocateInfo emai = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .pNext = use_ded ? &ded_ai : NULL,
        .handleTypes = htype,
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &emai,
        .allocationSize = mr2.memoryRequirements.size,
        .memoryTypeIndex = mem_idx,
    };
    if (vkAllocateMemory(ctx->dev, &mai, NULL, &b->dev_mem) != VK_SUCCESS ||
        vkBindBufferMemory(ctx->dev, b->dev_buf, b->dev_mem, 0) != VK_SUCCESS) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }

    VkMemoryGetFdInfoKHR gmfi = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = b->dev_mem,
        .handleType = htype,
    };
    if (ctx->pfn_get_mem_fd(ctx->dev, &gmfi, out_fd) != VK_SUCCESS || *out_fd < 0) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }
    return b;
}

vxfec_vk_buf *vxfec_vk_buf_import_fd_range(vxfec_vk_ctx *ctx, int fd, size_t base_offset, size_t bytes, bool is_dmabuf)
{
#if defined(_WIN32)
    (void)ctx;
    (void)fd;
    (void)base_offset;
    (void)bytes;
    (void)is_dmabuf;
    return NULL;
#else
    /* generic OPAQUE_FD import requires exporter allocation metadata; only DMA-BUF is supported */
    if (!ctx || fd < 0 || bytes == 0 || base_offset > SIZE_MAX - bytes || !is_dmabuf || !ctx->has_ext_mem_fd || !ctx->has_dmabuf ||
        !ctx->pfn_get_mem_fd_props)
        return NULL;

    size_t span_bytes = base_offset + bytes;
    VkExternalMemoryHandleTypeFlagBits htype = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkExternalMemoryBufferCreateInfo embci = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = htype,
    };
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &embci,
        .size = span_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    vxfec_vk_buf *b = (vxfec_vk_buf *)calloc(1, sizeof(*b));
    if (!b)
        return NULL;
    b->bytes = bytes;
    b->view_offset = base_offset;
    b->ext_kind = VXFEC_VK_EXT_DMABUF;
    b->owned_by_vk = false;

    if (vkCreateBuffer(ctx->dev, &bci, NULL, &b->dev_buf) != VK_SUCCESS) {
        free(b);
        return NULL;
    }

    VkMemoryDedicatedRequirements ded_req = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
    };
    VkMemoryRequirements2 mr2 = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = &ded_req,
    };
    VkBufferMemoryRequirementsInfo2 bmri = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
        .buffer = b->dev_buf,
    };
    vkGetBufferMemoryRequirements2(ctx->dev, &bmri, &mr2);

    VkMemoryFdPropertiesKHR fd_props = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
    };
    if (ctx->pfn_get_mem_fd_props(ctx->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fd_props) != VK_SUCCESS) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }
    uint32_t type_bits = mr2.memoryRequirements.memoryTypeBits & fd_props.memoryTypeBits;

    uint32_t mem_idx = 0;
    if (!find_mem_type(ctx->phys, type_bits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &mem_idx)) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }

    int vk_fd = dup(fd);
    if (vk_fd < 0) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }

    bool use_ded = ctx->dmabuf_dedicated_only || ded_req.requiresDedicatedAllocation == VK_TRUE;
    VkMemoryDedicatedAllocateInfo ded_ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .buffer = b->dev_buf,
    };
    VkImportMemoryFdInfoKHR imfi = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .pNext = use_ded ? &ded_ai : NULL,
        .handleType = htype,
        .fd = vk_fd,
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &imfi,
        .allocationSize = mr2.memoryRequirements.size,
        .memoryTypeIndex = mem_idx,
    };
    if (vkAllocateMemory(ctx->dev, &mai, NULL, &b->dev_mem) != VK_SUCCESS) {
        close(vk_fd);
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }
    if (vkBindBufferMemory(ctx->dev, b->dev_buf, b->dev_mem, 0) != VK_SUCCESS) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }
    close(fd);
    return b;
#endif
}

vxfec_vk_buf *vxfec_vk_buf_import_fd(vxfec_vk_ctx *ctx, int fd, size_t bytes, bool is_dmabuf)
{
    return vxfec_vk_buf_import_fd_range(ctx, fd, 0u, bytes, is_dmabuf);
}

vxfec_vk_buf *vxfec_vk_buf_import_host_ptr(vxfec_vk_ctx *ctx, void *host_ptr, size_t bytes)
{
    if (!ctx || !host_ptr || bytes == 0 || !ctx->has_ext_host_mem || !ctx->pfn_get_host_ptr_props)
        return NULL;
    VkDeviceSize align = ctx->host_import_align ? ctx->host_import_align : 4096u;
    if (((uintptr_t)host_ptr & (align - 1u)) != 0 || (bytes & (align - 1u)) != 0)
        return NULL;

    VkExternalMemoryBufferCreateInfo embci = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
    };
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &embci,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    vxfec_vk_buf *b = (vxfec_vk_buf *)calloc(1, sizeof(*b));
    if (!b)
        return NULL;
    b->bytes = bytes;
    b->ext_kind = VXFEC_VK_EXT_NONE;
    b->owned_by_vk = true;
    b->zero_copy = true;
    b->imported_host = true;

    if (vkCreateBuffer(ctx->dev, &bci, NULL, &b->dev_buf) != VK_SUCCESS) {
        free(b);
        return NULL;
    }

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx->dev, b->dev_buf, &mr);
    if (mr.size > bytes) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }
    VkMemoryHostPointerPropertiesEXT hp_props = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
    };
    if (ctx->pfn_get_host_ptr_props(ctx->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host_ptr, &hp_props) !=
        VK_SUCCESS) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }

    uint32_t mem_idx = 0;
    if (!find_mem_type(ctx->phys, mr.memoryTypeBits & hp_props.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &mem_idx)) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }

    VkImportMemoryHostPointerInfoEXT imhpi = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = host_ptr,
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &imhpi,
        .allocationSize = mr.size,
        .memoryTypeIndex = mem_idx,
    };
    if (vkAllocateMemory(ctx->dev, &mai, NULL, &b->dev_mem) != VK_SUCCESS ||
        vkBindBufferMemory(ctx->dev, b->dev_buf, b->dev_mem, 0) != VK_SUCCESS) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }
    b->stage_ptr = host_ptr;
    return b;
}

static bool ensure_io_stage(vxfec_vk_ctx *ctx, size_t bytes)
{
    if (!ensure_sync_cmd_idle(ctx))
        return false;
    if (ctx->io_stage_buf && ctx->io_stage_cap >= bytes)
        return true;
    if (ctx->io_stage_ptr && ctx->io_stage_mem)
        vkUnmapMemory(ctx->dev, ctx->io_stage_mem);
    if (ctx->io_stage_buf)
        vkDestroyBuffer(ctx->dev, ctx->io_stage_buf, NULL);
    if (ctx->io_stage_mem)
        vkFreeMemory(ctx->dev, ctx->io_stage_mem, NULL);
    ctx->io_stage_buf = VK_NULL_HANDLE;
    ctx->io_stage_mem = VK_NULL_HANDLE;
    ctx->io_stage_ptr = NULL;
    ctx->io_stage_cap = 0;

    size_t alloc_sz = (bytes < 65536u) ? 65536u : bytes;
    VkBufferUsageFlags st_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags req = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!create_raw_buffer(ctx, alloc_sz, st_usage, req, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &ctx->io_stage_buf,
                           &ctx->io_stage_mem) ||
        vkMapMemory(ctx->dev, ctx->io_stage_mem, 0, alloc_sz, 0, &ctx->io_stage_ptr) != VK_SUCCESS) {
        if (ctx->io_stage_buf)
            vkDestroyBuffer(ctx->dev, ctx->io_stage_buf, NULL);
        if (ctx->io_stage_mem)
            vkFreeMemory(ctx->dev, ctx->io_stage_mem, NULL);
        ctx->io_stage_buf = VK_NULL_HANDLE;
        ctx->io_stage_mem = VK_NULL_HANDLE;
        ctx->io_stage_ptr = NULL;
        return false;
    }
    ctx->io_stage_cap = alloc_sz;
    return true;
}

vxfec_vk_buf *vxfec_vk__upload_const_buf(vxfec_vk_ctx *ctx, const void *data, size_t bytes)
{
    if (!ctx || !data || bytes == 0)
        return NULL;
    if (!ctx->is_discrete) {
        vxfec_vk_buf *ub = vxfec_vk_buf_new(ctx, bytes, true);
        if (!ub)
            return NULL;
        memcpy(vxfec_vk_buf_staging_ptr(ub), data, bytes);
        if (!vxfec_vk_buf_upload(ctx, ub, bytes)) {
            vxfec_vk__retire_or_free_buf(ctx, ub);
            return NULL;
        }
        vxfec_vk__release_staging(ctx, ub);
        return ub;
    }

    vxfec_vk_buf *b = vxfec_vk_buf_new(ctx, bytes, false);
    if (!b)
        return NULL;
    if (!ensure_io_stage(ctx, bytes) || !vxfec_vk__begin_cmd(ctx)) {
        vxfec_vk_buf_free(ctx, b);
        return NULL;
    }
    memcpy(ctx->io_stage_ptr, data, bytes);
    VkBufferCopy copy = {.srcOffset = 0, .dstOffset = 0, .size = bytes};
    vkCmdCopyBuffer(ctx->cmd, ctx->io_stage_buf, b->dev_buf, 1, &copy);
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0,
                         NULL);
    if (!vxfec_vk__submit_and_wait(ctx)) {
        vxfec_vk__retire_or_free_buf(ctx, b);
        return NULL;
    }
    return b;
}

bool vxfec_vk_buf_read_offset(vxfec_vk_ctx *ctx, vxfec_vk_buf *src_vram, size_t offset, void *dst_host, size_t bytes)
{
    if (!ctx || !src_vram || !dst_host || bytes == 0 || offset > src_vram->bytes || bytes > src_vram->bytes - offset)
        return false;
    if (src_vram->zero_copy && src_vram->stage_ptr) {
        if (!ensure_sync_cmd_idle(ctx) || !wait_async_idle(ctx))
            return false;
        memcpy(dst_host, (const uint8_t *)src_vram->stage_ptr + src_vram->view_offset + offset, bytes);
        return true;
    }
    if (!ensure_io_stage(ctx, bytes) || !vxfec_vk__begin_cmd(ctx))
        return false;
    bool transient_ext = (src_vram->ext_kind != VXFEC_VK_EXT_NONE && !src_vram->owned_by_vk);
    if (transient_ext) {
        if (ctx->pending_owned_cnt >= VXFEC_VK_MAX_PENDING_OWNED)
            return false;
        vxfec_vk__record_ext_acquire_cmd(ctx, ctx->cmd, src_vram, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        ctx->pending_owned_bufs[ctx->pending_owned_cnt++] = src_vram;
        ctx->pending_release_buf = src_vram;
    }
    VkBufferCopy copy = {.srcOffset = src_vram->view_offset + offset, .dstOffset = 0, .size = bytes};
    vkCmdCopyBuffer(ctx->cmd, src_vram->dev_buf, ctx->io_stage_buf, 1, &copy);
    vxfec_vk__record_transfer_to_host_cmd(ctx->cmd);
    if (transient_ext)
        vxfec_vk__record_ext_release_cmd(ctx, ctx->cmd, src_vram, VK_PIPELINE_STAGE_TRANSFER_BIT, 0);
    if (!vxfec_vk__submit_and_wait(ctx))
        return false;
    memcpy(dst_host, ctx->io_stage_ptr, bytes);
    return true;
}

bool vxfec_vk_buf_write_offset(vxfec_vk_ctx *ctx, vxfec_vk_buf *dst_vram, size_t offset, const void *src_host, size_t bytes)
{
    if (!ctx || !dst_vram || !src_host || bytes == 0 || offset > dst_vram->bytes || bytes > dst_vram->bytes - offset)
        return false;
    if (dst_vram->zero_copy && dst_vram->stage_ptr) {
        if (!ensure_sync_cmd_idle(ctx) || !wait_async_idle(ctx))
            return false;
        memcpy((uint8_t *)dst_vram->stage_ptr + dst_vram->view_offset + offset, src_host, bytes);
        return true;
    }
    if (!ensure_io_stage(ctx, bytes) || !vxfec_vk__begin_cmd(ctx))
        return false;
    memcpy(ctx->io_stage_ptr, src_host, bytes);
    bool transient_ext = (dst_vram->ext_kind != VXFEC_VK_EXT_NONE && !dst_vram->owned_by_vk);
    if (transient_ext) {
        if (ctx->pending_owned_cnt >= VXFEC_VK_MAX_PENDING_OWNED)
            return false;
        vxfec_vk__record_ext_acquire_cmd(ctx, ctx->cmd, dst_vram, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        ctx->pending_owned_bufs[ctx->pending_owned_cnt++] = dst_vram;
        ctx->pending_release_buf = dst_vram;
    }
    VkBufferCopy copy = {.srcOffset = 0, .dstOffset = dst_vram->view_offset + offset, .size = bytes};
    vkCmdCopyBuffer(ctx->cmd, ctx->io_stage_buf, dst_vram->dev_buf, 1, &copy);
    if (transient_ext) {
        vxfec_vk__record_ext_release_cmd(ctx, ctx->cmd, dst_vram, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    } else {
        VkMemoryBarrier xfer_to_dev = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &xfer_to_dev, 0, NULL, 0,
                             NULL);
    }
    return vxfec_vk__submit_and_wait(ctx);
}

uint64_t vxfec_vk__read_gpu_ns(vxfec_vk_ctx *ctx)
{
    if (!ctx->has_timestamps)
        return 0;
    uint64_t ts[2] = {0, 0};
    if (vkGetQueryPoolResults(ctx->dev, ctx->query_pool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
        return 0;
    uint64_t mask = (ctx->ts_valid_bits > 0 && ctx->ts_valid_bits < 64u) ? ((1ULL << ctx->ts_valid_bits) - 1ULL) : UINT64_MAX;
    uint64_t diff = (ts[1] - ts[0]) & mask;
    return (uint64_t)((double)diff * (double)ctx->timestamp_period_ns);
}

static size_t select_tile_cap(const vxfec_vk_ctx *ctx)
{
    if (!ctx->is_discrete)
        return 12ULL << 20;
    if (ctx->vendor_id == 0x1002u)
        return 96ULL << 20; /* AMD RDNA Infinity Cache (64-96 MiB) */
    if (ctx->vendor_id == 0x10deu)
        return 36ULL << 20; /* NVIDIA Hopper/Ada/Ampere L2 (40-96 MiB) */
    return 32ULL << 20;
}

uint32_t vxfec_vk__compute_tile_blocks(const vxfec_vk_ctx *ctx, uint32_t rows, uint32_t stride, uint32_t wg_x, uint32_t num_blocks)
{
    uint32_t tile_blks = num_blocks;
    uint32_t min_wgs = ctx->is_discrete ? 256u : 32u;
    if ((uint64_t)wg_x * num_blocks > min_wgs) {
        size_t tile_cap = select_tile_cap(ctx);
        size_t blk_bytes = (size_t)rows * stride;
        uint32_t max_tile = (blk_bytes < tile_cap) ? (uint32_t)(tile_cap / blk_bytes) : 1u;
        if (num_blocks > max_tile) {
            uint32_t n_tiles = (num_blocks + max_tile - 1u) / max_tile;
            tile_blks = (num_blocks + n_tiles - 1u) / n_tiles;
        }
    }
    if (tile_blks > ctx->max_wg_y)
        tile_blks = ctx->max_wg_y;
    return tile_blks;
}

bool vxfec_vk__ensure_staging_buf(vxfec_vk_ctx *ctx, vxfec_vk_buf **io_buf, size_t *io_cap, size_t need_bytes)
{
    bool is_sync_buf = (io_buf == &ctx->mix_tab_buf || io_buf == &ctx->direct_tab_buf);
    if (is_sync_buf && !ensure_sync_cmd_idle(ctx))
        return false;
    if (*io_buf && *io_cap >= need_bytes)
        return true;
    vxfec_vk_buf *nb = vxfec_vk_buf_new(ctx, need_bytes, true);
    if (!nb)
        return false;
    if (is_sync_buf)
        vxfec_vk_buf_free(ctx, *io_buf);
    else
        vxfec_vk__retire_or_free_buf(ctx, *io_buf);
    *io_buf = nb;
    *io_cap = need_bytes;
    return true;
}

bool vxfec_vk__select_direct_plan(const vxfec_vk_ctx *ctx, bool pure_gf2, uint32_t tile_e, size_t d_bytes, size_t out_bytes,
                                  uint32_t stride, uint32_t num_tiles, uint32_t num_blocks, VkPipeline *out_pipe,
                                  uint32_t *out_stride_units, uint32_t *out_wg_x)
{
    (void)d_bytes;
    (void)out_bytes;
    bool can_vec4 = ((stride & 15u) == 0);
    bool use_vec4 = false;
    uint32_t min_wgs = ctx->is_discrete ? 256u : 32u;

    if (can_vec4) {
        uint32_t wg_x_vec4 = ((stride / 16u) + 31u) / 32u;
        if ((uint64_t)wg_x_vec4 * num_tiles * num_blocks >= min_wgs)
            use_vec4 = true;
    }

    uint32_t stride_units = use_vec4 ? (stride / 16u) : (stride / 4u);
    uint32_t wg_x = (stride_units + 31u) / 32u;
    if (wg_x == 0 || wg_x > ctx->max_wg_x)
        return false;

    if (pure_gf2) {
        *out_pipe = use_vec4 ? ((tile_e == 16u) ? ctx->pipe_direct_gf2_vec4_16 : ctx->pipe_direct_gf2_vec4_32)
                             : ((tile_e == 16u) ? ctx->pipe_direct_gf2_16 : ctx->pipe_direct_gf2_32);
    } else {
        *out_pipe = use_vec4 ? ((tile_e == 16u) ? ctx->pipe_direct_gf256_nib_vec4_16 : ctx->pipe_direct_gf256_nib_vec4_8)
                             : ((tile_e == 16u) ? ctx->pipe_direct_gf256_nib_16 : ctx->pipe_direct_gf256_nib_8);
    }
    *out_stride_units = stride_units;
    *out_wg_x = wg_x;
    return true;
}

void vxfec_vk__record_mix_like_dispatches_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, VkDescriptorSet ds_mix, uint32_t rows,
                                              uint32_t stride_units, uint32_t wg_x, uint32_t num_esis, uint32_t y_count,
                                              uint32_t block_base, uint32_t num_blocks, VkPipeline pipe)
{
    vxfec_mix_push_consts mpc = {
        .rows = rows,
        .stride_units = stride_units,
        .num_esis = num_esis,
        .esi_offset = 0,
        .block_offset = 0,
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pl_mix, 0, 1, &ds_mix, 0, NULL);

    for (uint32_t b0 = 0; b0 < num_blocks;) {
        uint32_t b_chunk = (num_blocks - b0 > ctx->max_wg_z) ? ctx->max_wg_z : (num_blocks - b0);
        for (uint32_t y0 = 0; y0 < y_count;) {
            uint32_t y_chunk = (y_count - y0 > ctx->max_wg_y) ? ctx->max_wg_y : (y_count - y0);
            mpc.esi_offset = y0;
            mpc.block_offset = block_base + b0;
            vkCmdPushConstants(cmd, ctx->pl_mix, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mpc), &mpc);
            vkCmdDispatch(cmd, wg_x, y_chunk, b_chunk);
            y0 += y_chunk;
        }
        b0 += b_chunk;
    }
}

bool vxfec_vk__record_direct_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, VkDescriptorSet ds_mix, const vxfec_vk_direct_desc *gd,
                                 vxfec_vk_buf *d_buf, uint32_t stride, uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d,
                                 bool download_out, bool record_ts)
{
    size_t d_bytes = 0, out_bytes = 0;
    if (!vxfec_vk__mul3(num_blocks, gd->rows, stride, &d_bytes) || !vxfec_vk__mul3(num_blocks, gd->num_esis, stride, &out_bytes) ||
        d_bytes > d_buf->bytes || out_bytes > out_buf->bytes || d_bytes > ctx->max_ssbo_range || out_bytes > ctx->max_ssbo_range)
        return false;
    if (((d_buf->view_offset | out_buf->view_offset) & (size_t)(ctx->ssbo_offset_align - 1u)) != 0)
        return false;
    if ((upload_d && !d_buf->stage_ptr) || (download_out && !out_buf->stage_ptr))
        return false;

    VkPipeline pipe = VK_NULL_HANDLE;
    uint32_t stride_units = 0, wg_x = 0;
    if (!vxfec_vk__select_direct_plan(ctx, gd->pure_gf2, gd->tile_e, d_bytes, out_bytes, stride, gd->num_tiles, num_blocks, &pipe,
                                      &stride_units, &wg_x))
        return false;

    if (!vxfec_vk__ensure_owned_cmd(ctx, cmd, d_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT) ||
        !vxfec_vk__ensure_owned_cmd(ctx, cmd, out_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT))
        return false;

    VkDescriptorBufferInfo dbi[3] = {
        {.buffer = d_buf->dev_buf, .offset = d_buf->view_offset, .range = d_bytes},
        {.buffer = gd->tab_buf->dev_buf, .offset = gd->tab_buf->view_offset, .range = gd->tab_bytes},
        {.buffer = out_buf->dev_buf, .offset = out_buf->view_offset, .range = out_bytes},
    };
    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds_mix,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds_mix,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds_mix,
         .dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[2]},
    };
    vkUpdateDescriptorSets(ctx->dev, 3, writes, 0, NULL);

    if (record_ts && ctx->has_timestamps) {
        vkCmdResetQueryPool(cmd, ctx->query_pool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->query_pool, 0);
    }

    if (upload_d && !d_buf->zero_copy) {
        VkBufferCopy d_copy = {.srcOffset = 0, .dstOffset = d_buf->view_offset, .size = d_bytes};
        vkCmdCopyBuffer(cmd, d_buf->stage_buf, d_buf->dev_buf, 1, &d_copy);
        VkMemoryBarrier xfer_mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &xfer_mb, 0, NULL, 0,
                             NULL);
    }

    vxfec_vk__record_mix_like_dispatches_cmd(ctx, cmd, ds_mix, gd->rows, stride_units, wg_x, gd->num_esis, gd->num_tiles, 0u,
                                             num_blocks, pipe);

    if (download_out && !out_buf->zero_copy) {
        VkMemoryBarrier mix_to_xfer = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mix_to_xfer, 0, NULL, 0,
                             NULL);
        VkBufferCopy out_copy = {.srcOffset = out_buf->view_offset, .dstOffset = 0, .size = out_bytes};
        vkCmdCopyBuffer(cmd, out_buf->dev_buf, out_buf->stage_buf, 1, &out_copy);
        vxfec_vk__record_transfer_to_host_cmd(cmd);
    } else {
        VkMemoryBarrier post_mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
                             1, &post_mb, 0, NULL, 0, NULL);
    }

    if (record_ts && ctx->has_timestamps)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->query_pool, 1);
    return true;
}

bool vxfec_vk__direct_pipelined(vxfec_vk_ctx *ctx, const vxfec_vk_direct_desc *gd, vxfec_vk_buf *d_buf, uint32_t stride,
                                uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out, uint64_t *gpu_ns_out)
{
    if (!ctx || !gd || !gd->tab_buf || !d_buf || !out_buf || stride == 0 || (stride & 3u) != 0 || num_blocks == 0)
        return false;
    if (!vxfec_vk__begin_cmd(ctx))
        return false;
    if (!vxfec_vk__record_direct_cmd(ctx, ctx->cmd, ctx->ds_mix, gd, d_buf, stride, num_blocks, out_buf, upload_d, download_out,
                                     gpu_ns_out != NULL)) {
        ctx->pending_owned_cnt = 0;
        return false;
    }
    if (!vxfec_vk__submit_and_wait(ctx))
        return false;
    if (gpu_ns_out)
        *gpu_ns_out = vxfec_vk__read_gpu_ns(ctx);
    return true;
}

uint64_t vxfec_vk__direct_submit(vxfec_vk_ctx *ctx, const vxfec_vk_direct_desc *gd, vxfec_vk_buf *d_buf, uint32_t stride,
                                 uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out)
{
    if (!ctx || !gd || !gd->tab_buf || !d_buf || !out_buf || stride == 0 || (stride & 3u) != 0 || num_blocks == 0)
        return 0;
    if (!ctx->has_timeline_sem || !ctx->timeline_sem)
        return vxfec_vk__direct_pipelined(ctx, gd, d_buf, stride, num_blocks, out_buf, upload_d, download_out, NULL) ? 1u : 0u;

    uint32_t slot = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!vxfec_vk__acquire_async_slot(ctx, &slot, &cmd))
        return 0;
    if (!vxfec_vk__record_direct_cmd(ctx, cmd, ctx->async_ds_mix[slot], gd, d_buf, stride, num_blocks, out_buf, upload_d,
                                     download_out, false)) {
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }
    return vxfec_vk__submit_async_slot(ctx, slot, cmd);
}

uint64_t vxfec_vk__direct_submit_staged_words(vxfec_vk_ctx *ctx, const uint32_t *tab_words, size_t tab_bytes, uint32_t rows,
                                              uint32_t num_esis, uint32_t num_tiles, uint32_t tile_e, bool pure_gf2,
                                              vxfec_vk_buf *d_buf, uint32_t stride, uint32_t num_blocks, vxfec_vk_buf *out_buf,
                                              bool upload_d, bool download_out)
{
    if (!ctx || !tab_words || tab_bytes == 0 || tab_bytes > ctx->max_ssbo_range || rows == 0 || num_esis == 0 || num_tiles == 0 ||
        !d_buf || !out_buf || stride == 0 || (stride & 3u) != 0 || num_blocks == 0)
        return 0;

    if (!ctx->has_timeline_sem || !ctx->timeline_sem) {
        if (!vxfec_vk__ensure_staging_buf(ctx, &ctx->direct_tab_buf, &ctx->direct_tab_cap, tab_bytes))
            return 0;
        memcpy(vxfec_vk_buf_staging_ptr(ctx->direct_tab_buf), tab_words, tab_bytes);
        if (!vxfec_vk__begin_cmd(ctx))
            return 0;
        if (!ctx->direct_tab_buf->zero_copy) {
            VkBufferCopy t_copy = {.srcOffset = 0, .dstOffset = 0, .size = tab_bytes};
            vkCmdCopyBuffer(ctx->cmd, ctx->direct_tab_buf->stage_buf, ctx->direct_tab_buf->dev_buf, 1, &t_copy);
            VkMemoryBarrier xfer_mb = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &xfer_mb, 0,
                                 NULL, 0, NULL);
        }
        vxfec_vk_direct_desc desc = {
            .tab_buf = ctx->direct_tab_buf,
            .tab_bytes = tab_bytes,
            .rows = rows,
            .num_esis = num_esis,
            .num_tiles = num_tiles,
            .tile_e = tile_e,
            .pure_gf2 = pure_gf2,
        };
        if (!vxfec_vk__record_direct_cmd(ctx, ctx->cmd, ctx->ds_mix, &desc, d_buf, stride, num_blocks, out_buf, upload_d,
                                         download_out, false)) {
            ctx->pending_owned_cnt = 0;
            return 0;
        }
        return vxfec_vk__submit_and_wait(ctx) ? 1u : 0u;
    }

    uint32_t slot = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!vxfec_vk__acquire_async_slot(ctx, &slot, &cmd))
        return 0;
    if (!vxfec_vk__ensure_staging_buf(ctx, &ctx->async_tab_bufs[slot], &ctx->async_tab_caps[slot], tab_bytes)) {
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }
    vxfec_vk_buf *tbuf = ctx->async_tab_bufs[slot];
    memcpy(vxfec_vk_buf_staging_ptr(tbuf), tab_words, tab_bytes);
    if (!tbuf->zero_copy) {
        VkBufferCopy t_copy = {.srcOffset = 0, .dstOffset = 0, .size = tab_bytes};
        vkCmdCopyBuffer(cmd, tbuf->stage_buf, tbuf->dev_buf, 1, &t_copy);
        VkMemoryBarrier xfer_mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &xfer_mb, 0, NULL, 0,
                             NULL);
    }
    vxfec_vk_direct_desc desc = {
        .tab_buf = tbuf,
        .tab_bytes = tab_bytes,
        .rows = rows,
        .num_esis = num_esis,
        .num_tiles = num_tiles,
        .tile_e = tile_e,
        .pure_gf2 = pure_gf2,
    };
    if (!vxfec_vk__record_direct_cmd(ctx, cmd, ctx->async_ds_mix[slot], &desc, d_buf, stride, num_blocks, out_buf, upload_d,
                                     download_out, false)) {
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }
    return vxfec_vk__submit_async_slot(ctx, slot, cmd);
}

uint32_t vxfec_vk__choose_gf256_tile_e(uint32_t num_out)
{
    return (num_out <= 8u) ? 8u : 16u;
}

bool vxfec_vk__emit_direct_gf256_tab(VkDeviceSize max_ssbo_range, const uint8_t *Y, size_t m_stride, uint32_t rows,
                                     uint32_t num_out, uint32_t **out_words, size_t *out_bytes, uint32_t *out_tiles,
                                     uint32_t *out_tile_e)
{
    if (!Y || rows == 0 || num_out == 0 || num_out > UINT32_MAX - 16u)
        return false;
    uint32_t tile_e = vxfec_vk__choose_gf256_tile_e(num_out);
    uint32_t n_tiles = (num_out + tile_e - 1u) / tile_e;
    size_t words_per_cmd = (tile_e == 16u) ? 2u : 1u;

    size_t per_tile = 0, max_uvec4 = 0, alloc_bytes = 0;
    if (!vxfec_vk__mul3(words_per_cmd, rows, 1u, &per_tile) || per_tile > SIZE_MAX - 1u ||
        !vxfec_vk__mul3(n_tiles, per_tile + 1u, 1u, &max_uvec4))
        return false;
    if (max_uvec4 == 0)
        max_uvec4 = 1;
    if (max_uvec4 > UINT32_MAX || !vxfec_vk__mul3(max_uvec4, 16u, 1u, &alloc_bytes) ||
        (max_ssbo_range > 0 && alloc_bytes > (size_t)max_ssbo_range))
        return false;

    uint32_t *tab = (uint32_t *)malloc(alloc_bytes);
    if (!tab)
        return false;

    uint32_t cursor = n_tiles;
    for (uint32_t t = 0; t < n_tiles; t++) {
        uint32_t beg = cursor;
        uint32_t out_base = t * tile_e;
        uint32_t tile_len = (num_out - out_base > tile_e) ? tile_e : (num_out - out_base);
        for (uint32_t r = 0; r < rows; r++) {
            const uint8_t *yc = Y + (size_t)r * m_stride + out_base;
            uint32_t lo0_7 = 0u, hi0_7 = 0u;
            for (uint32_t b = 0; b < 8u && b < tile_len; b++) {
                uint8_t c = yc[b];
                if (c != 0u) {
                    lo0_7 |= ((uint32_t)(c & 0x0fu)) << (4u * b);
                    hi0_7 |= ((uint32_t)(c >> 4u)) << (4u * b);
                }
            }
            if (tile_e == 16u) {
                uint32_t lo8_15 = 0u, hi8_15 = 0u;
                for (uint32_t b = 8u; b < 16u && b < tile_len; b++) {
                    uint8_t c = yc[b];
                    if (c != 0u) {
                        uint32_t sh = 4u * (b - 8u);
                        lo8_15 |= ((uint32_t)(c & 0x0fu)) << sh;
                        hi8_15 |= ((uint32_t)(c >> 4u)) << sh;
                    }
                }
                if ((lo0_7 | lo8_15 | hi0_7 | hi8_15) != 0u) {
                    size_t base = (size_t)cursor * 4u;
                    tab[base + 0] = r;
                    tab[base + 1] = 0u;
                    tab[base + 2] = lo0_7;
                    tab[base + 3] = lo8_15;
                    tab[base + 4] = hi0_7;
                    tab[base + 5] = hi8_15;
                    tab[base + 6] = 0u;
                    tab[base + 7] = 0u;
                    cursor += 2u;
                }
            } else if ((lo0_7 | hi0_7) != 0u) {
                size_t base = (size_t)cursor * 4u;
                tab[base + 0] = r;
                tab[base + 1] = lo0_7;
                tab[base + 2] = hi0_7;
                tab[base + 3] = 0u;
                cursor++;
            }
        }
        size_t tbase = (size_t)t * 4u;
        tab[tbase + 0] = beg;
        tab[tbase + 1] = cursor;
        tab[tbase + 2] = 0u;
        tab[tbase + 3] = 0u;
    }

    *out_words = tab;
    *out_bytes = (size_t)(cursor ? cursor : 1u) * 16u;
    *out_tiles = n_tiles;
    *out_tile_e = tile_e;
    return true;
}

int vxfec_vk_poll_status(vxfec_vk_ctx *ctx, uint64_t ticket)
{
    if (!ctx)
        return -1;
    if (ticket == 0 || !ctx->has_timeline_sem || !ctx->timeline_sem)
        return 1;
    uint64_t val = 0;
    if (vkGetSemaphoreCounterValue(ctx->dev, ctx->timeline_sem, &val) != VK_SUCCESS)
        return -1;
    if (val < ticket)
        return 0;
    /* establish Vulkan device-to-host memory visibility on signaled ticket */
    VkSemaphoreWaitInfo swi = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &ctx->timeline_sem,
        .pValues = &ticket,
    };
    return (vkWaitSemaphores(ctx->dev, &swi, 0) == VK_SUCCESS) ? 1 : -1;
}

bool vxfec_vk_poll(vxfec_vk_ctx *ctx, uint64_t ticket)
{
    return vxfec_vk_poll_status(ctx, ticket) > 0;
}

bool vxfec_vk_wait(vxfec_vk_ctx *ctx, uint64_t ticket, uint64_t timeout_ns)
{
    if (!ctx)
        return false;
    if (ticket == 0 || !ctx->has_timeline_sem || !ctx->timeline_sem)
        return true;
    VkSemaphoreWaitInfo swi = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &ctx->timeline_sem,
        .pValues = &ticket,
    };
    return vkWaitSemaphores(ctx->dev, &swi, timeout_ns) == VK_SUCCESS;
}

static bool reap_and_find_async_slot(vxfec_vk_ctx *ctx, uint32_t *out_slot)
{
    uint64_t done_val = 0;
    bool any_pending = false;
    if (ctx->has_timeline_sem && ctx->timeline_sem) {
        if (vkGetSemaphoreCounterValue(ctx->dev, ctx->timeline_sem, &done_val) != VK_SUCCESS)
            return false;
        for (uint32_t i = 0; i < VXFEC_VK_ASYNC_SLOTS; i++) {
            if (ctx->async_slot_tickets[i] > 0) {
                if (ctx->async_slot_tickets[i] <= done_val)
                    ctx->async_slot_tickets[i] = 0;
                else
                    any_pending = true;
            }
        }
    }
    if (ctx->retired_bufs && !ctx->last_submit_unconfirmed && !any_pending)
        reap_retired_bufs(ctx);
    for (uint32_t i = 0; i < VXFEC_VK_ASYNC_SLOTS; i++) {
        uint32_t s = (ctx->async_cursor + i) % VXFEC_VK_ASYNC_SLOTS;
        if (ctx->async_slot_tickets[s] == 0) {
            if (out_slot)
                *out_slot = s;
            return true;
        }
    }
    return false;
}

bool vxfec_vk_async_can_submit(vxfec_vk_ctx *ctx)
{
    if (!ctx)
        return false;
    if (!ctx->has_timeline_sem || !ctx->timeline_sem)
        return true;
    return reap_and_find_async_slot(ctx, NULL);
}

bool vxfec_vk_has_timeline(const vxfec_vk_ctx *ctx)
{
    return ctx && ctx->has_timeline_sem && ctx->timeline_sem;
}

bool vxfec_vk__acquire_async_slot(vxfec_vk_ctx *ctx, uint32_t *out_slot, VkCommandBuffer *out_cmd)
{
    ctx->pending_owned_cnt = 0;
    uint32_t slot = 0;
    if (!reap_and_find_async_slot(ctx, &slot))
        return false;

    VkCommandBuffer cmd = ctx->async_cmds[slot];
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS)
        return false;
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS)
        return false;

    *out_slot = slot;
    *out_cmd = cmd;
    return true;
}

uint64_t vxfec_vk__submit_async_slot(vxfec_vk_ctx *ctx, uint32_t slot, VkCommandBuffer cmd)
{
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        ctx->pending_owned_cnt = 0;
        return 0;
    }

    uint64_t ticket = ctx->timeline_next;
    VkTimelineSemaphoreSubmitInfo tssi = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &ticket,
    };
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &tssi,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &ctx->timeline_sem,
    };
    if (vkQueueSubmit(ctx->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
        ctx->pending_owned_cnt = 0;
        return 0;
    }

    commit_pending_owned(ctx);
    ctx->timeline_submitted = ticket;
    ctx->timeline_next = ticket + 1u;
    ctx->async_slot_tickets[slot] = ticket;
    ctx->async_cursor = (slot + 1u) % VXFEC_VK_ASYNC_SLOTS;
    return ticket;
}

static VkPipeline select_wire_pipe(const vxfec_vk_ctx *ctx, bool pure_gf2, uint32_t tile_e)
{
    if (pure_gf2)
        return (tile_e == 16u) ? ctx->pipe_wire_gf2_vec4_16 : ctx->pipe_wire_gf2_vec4_32;
    return (tile_e == 16u) ? ctx->pipe_wire_gf256_nib_vec4_16 : ctx->pipe_wire_gf256_nib_vec4_8;
}

static void update_wire_ds(VkDevice dev, VkDescriptorSet ds, VkBuffer wire_buf, size_t wire_off, size_t wire_bytes,
                           VkBuffer tab_buf, size_t tab_off, size_t tab_bytes)
{
    VkDescriptorBufferInfo dbi[3] = {
        {.buffer = wire_buf, .offset = wire_off, .range = wire_bytes},
        {.buffer = tab_buf, .offset = tab_off, .range = tab_bytes},
        {.buffer = wire_buf, .offset = wire_off, .range = wire_bytes},
    };
    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds,
         .dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[2]},
    };
    vkUpdateDescriptorSets(dev, 3, writes, 0, NULL);
}

static bool record_wire_tab_copy_cmd(VkCommandBuffer cmd, vxfec_vk_buf *tab_buf, size_t tab_bytes, size_t hdr_bytes,
                                     size_t cmd_src_off)
{
    if (tab_buf->zero_copy)
        return false;
    VkBufferCopy copies[2];
    uint32_t n_copies = 0;
    if (cmd_src_off > 0 && hdr_bytes > 0 && hdr_bytes < cmd_src_off && cmd_src_off <= tab_bytes) {
        copies[n_copies++] = (VkBufferCopy){.srcOffset = 0, .dstOffset = 0, .size = hdr_bytes};
        if (tab_bytes > cmd_src_off) {
            copies[n_copies++] =
                (VkBufferCopy){.srcOffset = cmd_src_off, .dstOffset = cmd_src_off, .size = tab_bytes - cmd_src_off};
        }
    } else {
        copies[n_copies++] = (VkBufferCopy){.srcOffset = 0, .dstOffset = 0, .size = tab_bytes};
    }
    vkCmdCopyBuffer(cmd, tab_buf->stage_buf, tab_buf->dev_buf, n_copies, copies);
    return true;
}

static bool record_wire_single_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, VkDescriptorSet ds, vxfec_vk_buf *wire_vram,
                                   size_t wire_bytes, vxfec_vk_buf *tab_buf, size_t tab_bytes, size_t tab_hdr_bytes,
                                   size_t tab_cmd_src_off, bool upload_tab, bool pure_gf2, uint32_t tile_e, uint32_t Z,
                                   uint32_t stride, uint32_t hdr_stride_tiles, uint32_t active_tiles, uint32_t disp_blocks,
                                   bool uniform_tab, bool record_ts)
{
    if (!ctx || !wire_vram || !tab_buf || Z == 0 || stride == 0 || (stride & 15u) != 0 || hdr_stride_tiles == 0 ||
        hdr_stride_tiles > 0x7fffffffu)
        return false;
    if (((wire_vram->view_offset | tab_buf->view_offset) & (size_t)(ctx->ssbo_offset_align - 1u)) != 0)
        return false;
    if (wire_bytes == 0 || wire_bytes > wire_vram->bytes || wire_bytes > ctx->max_ssbo_range || tab_bytes == 0 ||
        tab_bytes > tab_buf->bytes || tab_bytes > ctx->max_ssbo_range)
        return false;
    if (upload_tab && !tab_buf->stage_ptr)
        return false;

    uint32_t stride_vecs = stride / 16u;
    uint32_t wg_x = (stride_vecs + 31u) / 32u;
    VkPipeline pipe = select_wire_pipe(ctx, pure_gf2, tile_e);
    if (wg_x == 0 || wg_x > ctx->max_wg_x || !pipe)
        return false;

    if (!vxfec_vk__ensure_owned_cmd(ctx, cmd, wire_vram, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT))
        return false;

    update_wire_ds(ctx->dev, ds, wire_vram->dev_buf, wire_vram->view_offset, wire_bytes, tab_buf->dev_buf, tab_buf->view_offset,
                   tab_bytes);

    if (record_ts && ctx->has_timestamps) {
        vkCmdResetQueryPool(cmd, ctx->query_pool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->query_pool, 0);
    }

    if (upload_tab && record_wire_tab_copy_cmd(cmd, tab_buf, tab_bytes, tab_hdr_bytes, tab_cmd_src_off)) {
        VkMemoryBarrier xfer_mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &xfer_mb, 0, NULL, 0,
                             NULL);
    }

    uint32_t flags_tiles = hdr_stride_tiles | (uniform_tab ? 0x80000000u : 0u);
    vxfec_vk__record_mix_like_dispatches_cmd(ctx, cmd, ds, Z, stride_vecs, wg_x, flags_tiles, active_tiles, 0u, disp_blocks, pipe);

    VkMemoryBarrier post_mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &post_mb, 0, NULL, 0, NULL);

    if (record_ts && ctx->has_timestamps)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->query_pool, 1);
    return true;
}

bool vxfec_vk__exec_wire_cmd(vxfec_vk_ctx *ctx, vxfec_vk_buf *wire_vram, size_t wire_bytes, vxfec_vk_buf *tab_buf, size_t tab_bytes,
                             size_t tab_hdr_bytes, size_t tab_cmd_src_off, bool upload_tab, bool pure_gf2, uint32_t tile_e,
                             uint32_t Z, uint32_t stride, uint32_t hdr_stride_tiles, uint32_t active_tiles, uint32_t disp_blocks,
                             bool uniform_tab, uint64_t *gpu_ns_out)
{
    if (active_tiles == 0 || disp_blocks == 0) {
        if (gpu_ns_out)
            *gpu_ns_out = 0;
        return true;
    }
    if (!ctx || !vxfec_vk__begin_cmd(ctx))
        return false;
    if (!record_wire_single_cmd(ctx, ctx->cmd, ctx->ds_mix, wire_vram, wire_bytes, tab_buf, tab_bytes, tab_hdr_bytes,
                                tab_cmd_src_off, upload_tab, pure_gf2, tile_e, Z, stride, hdr_stride_tiles, active_tiles,
                                disp_blocks, uniform_tab, gpu_ns_out != NULL)) {
        ctx->pending_owned_cnt = 0;
        return false;
    }
    if (!vxfec_vk__submit_and_wait(ctx))
        return false;
    if (gpu_ns_out)
        *gpu_ns_out = vxfec_vk__read_gpu_ns(ctx);
    return true;
}

uint64_t vxfec_vk__submit_wire_cmd(vxfec_vk_ctx *ctx, vxfec_vk_buf *wire_vram, size_t wire_bytes, vxfec_vk_buf *tab_buf,
                                   size_t tab_bytes, size_t tab_hdr_bytes, size_t tab_cmd_src_off, bool upload_tab, bool pure_gf2,
                                   uint32_t tile_e, uint32_t Z, uint32_t stride, uint32_t hdr_stride_tiles, uint32_t active_tiles,
                                   uint32_t disp_blocks, bool uniform_tab)
{
    if (!ctx)
        return 0;
    if (!ctx->has_timeline_sem || !ctx->timeline_sem) {
        return vxfec_vk__exec_wire_cmd(ctx, wire_vram, wire_bytes, tab_buf, tab_bytes, tab_hdr_bytes, tab_cmd_src_off, upload_tab,
                                       pure_gf2, tile_e, Z, stride, hdr_stride_tiles, active_tiles, disp_blocks, uniform_tab, NULL)
                   ? 1u
                   : 0u;
    }
    if (active_tiles == 0 || disp_blocks == 0)
        return 1u;

    uint32_t slot = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!vxfec_vk__acquire_async_slot(ctx, &slot, &cmd))
        return 0;
    if (!record_wire_single_cmd(ctx, cmd, ctx->async_ds_mix[slot], wire_vram, wire_bytes, tab_buf, tab_bytes, tab_hdr_bytes,
                                tab_cmd_src_off, upload_tab, pure_gf2, tile_e, Z, stride, hdr_stride_tiles, active_tiles,
                                disp_blocks, uniform_tab, false)) {
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }
    return vxfec_vk__submit_async_slot(ctx, slot, cmd);
}

static bool record_wire_dual_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, VkDescriptorSet ds_gf2, VkDescriptorSet ds_256,
                                 vxfec_vk_buf *wire_vram, size_t wire_bytes, uint32_t Z, uint32_t stride, vxfec_vk_buf *gf2_tab,
                                 size_t gf2_bytes, size_t gf2_hdr_bytes, size_t gf2_cmd_off, uint32_t gf2_tile_e,
                                 uint32_t gf2_hdr_stride, uint32_t gf2_active_tiles, uint32_t gf2_blocks, vxfec_vk_buf *gf256_tab,
                                 size_t gf256_bytes, size_t gf256_hdr_bytes, size_t gf256_cmd_off, uint32_t gf256_tile_e,
                                 uint32_t gf256_hdr_stride, uint32_t gf256_active_tiles, uint32_t gf256_blocks, bool record_ts)
{
    if (!ctx || !wire_vram || !gf2_tab || !gf256_tab || Z == 0 || stride == 0 || (stride & 15u) != 0 || gf2_hdr_stride == 0 ||
        gf2_hdr_stride > 0x7fffffffu || gf256_hdr_stride == 0 || gf256_hdr_stride > 0x7fffffffu)
        return false;
    if (((wire_vram->view_offset | gf2_tab->view_offset | gf256_tab->view_offset) & (size_t)(ctx->ssbo_offset_align - 1u)) != 0)
        return false;
    if (wire_bytes == 0 || wire_bytes > wire_vram->bytes || wire_bytes > ctx->max_ssbo_range || gf2_bytes == 0 ||
        gf2_bytes > gf2_tab->bytes || gf2_bytes > ctx->max_ssbo_range || gf256_bytes == 0 || gf256_bytes > gf256_tab->bytes ||
        gf256_bytes > ctx->max_ssbo_range || !gf2_tab->stage_ptr || !gf256_tab->stage_ptr)
        return false;

    uint32_t stride_vecs = stride / 16u;
    uint32_t wg_x = (stride_vecs + 31u) / 32u;
    VkPipeline pipe_gf2 = select_wire_pipe(ctx, true, gf2_tile_e);
    VkPipeline pipe_256 = select_wire_pipe(ctx, false, gf256_tile_e);
    if (wg_x == 0 || wg_x > ctx->max_wg_x || !pipe_gf2 || !pipe_256)
        return false;

    if (!vxfec_vk__ensure_owned_cmd(ctx, cmd, wire_vram, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT))
        return false;

    update_wire_ds(ctx->dev, ds_gf2, wire_vram->dev_buf, wire_vram->view_offset, wire_bytes, gf2_tab->dev_buf, gf2_tab->view_offset,
                   gf2_bytes);
    update_wire_ds(ctx->dev, ds_256, wire_vram->dev_buf, wire_vram->view_offset, wire_bytes, gf256_tab->dev_buf,
                   gf256_tab->view_offset, gf256_bytes);

    if (record_ts && ctx->has_timestamps) {
        vkCmdResetQueryPool(cmd, ctx->query_pool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->query_pool, 0);
    }

    bool did_xfer = false;
    if (record_wire_tab_copy_cmd(cmd, gf2_tab, gf2_bytes, gf2_hdr_bytes, gf2_cmd_off))
        did_xfer = true;
    if (record_wire_tab_copy_cmd(cmd, gf256_tab, gf256_bytes, gf256_hdr_bytes, gf256_cmd_off))
        did_xfer = true;
    if (did_xfer) {
        VkMemoryBarrier xfer_mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &xfer_mb, 0, NULL, 0,
                             NULL);
    }

    vxfec_vk__record_mix_like_dispatches_cmd(ctx, cmd, ds_gf2, Z, stride_vecs, wg_x, gf2_hdr_stride, gf2_active_tiles, 0u,
                                             gf2_blocks, pipe_gf2);
    vxfec_vk__record_mix_like_dispatches_cmd(ctx, cmd, ds_256, Z, stride_vecs, wg_x, gf256_hdr_stride, gf256_active_tiles, 0u,
                                             gf256_blocks, pipe_256);

    VkMemoryBarrier post_mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &post_mb, 0, NULL, 0, NULL);

    if (record_ts && ctx->has_timestamps)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->query_pool, 1);
    return true;
}

bool vxfec_vk__exec_wire_hybrid_cmd(vxfec_vk_ctx *ctx, vxfec_vk_buf *wire_vram, size_t wire_bytes, uint32_t Z, uint32_t stride,
                                    vxfec_vk_buf *gf2_tab, size_t gf2_bytes, size_t gf2_hdr_bytes, size_t gf2_cmd_off,
                                    uint32_t gf2_tile_e, uint32_t gf2_hdr_stride, uint32_t gf2_active_tiles, uint32_t gf2_blocks,
                                    vxfec_vk_buf *gf256_tab, size_t gf256_bytes, size_t gf256_hdr_bytes, size_t gf256_cmd_off,
                                    uint32_t gf256_tile_e, uint32_t gf256_hdr_stride, uint32_t gf256_active_tiles,
                                    uint32_t gf256_blocks, uint64_t *gpu_ns_out)
{
    bool run_gf2 = (gf2_blocks > 0 && gf2_active_tiles > 0);
    bool run_256 = (gf256_blocks > 0 && gf256_active_tiles > 0);
    if (!run_gf2 && !run_256) {
        if (gpu_ns_out)
            *gpu_ns_out = 0;
        return true;
    }
    if (!run_256) {
        return vxfec_vk__exec_wire_cmd(ctx, wire_vram, wire_bytes, gf2_tab, gf2_bytes, gf2_hdr_bytes, gf2_cmd_off, true, true,
                                       gf2_tile_e, Z, stride, gf2_hdr_stride, gf2_active_tiles, gf2_blocks, false, gpu_ns_out);
    }
    if (!run_gf2) {
        return vxfec_vk__exec_wire_cmd(ctx, wire_vram, wire_bytes, gf256_tab, gf256_bytes, gf256_hdr_bytes, gf256_cmd_off, true,
                                       false, gf256_tile_e, Z, stride, gf256_hdr_stride, gf256_active_tiles, gf256_blocks, false,
                                       gpu_ns_out);
    }
    if (!ctx || !vxfec_vk__begin_cmd(ctx))
        return false;
    if (!record_wire_dual_cmd(ctx, ctx->cmd, ctx->ds_mix, ctx->ds_mix_aux, wire_vram, wire_bytes, Z, stride, gf2_tab, gf2_bytes,
                              gf2_hdr_bytes, gf2_cmd_off, gf2_tile_e, gf2_hdr_stride, gf2_active_tiles, gf2_blocks, gf256_tab,
                              gf256_bytes, gf256_hdr_bytes, gf256_cmd_off, gf256_tile_e, gf256_hdr_stride, gf256_active_tiles,
                              gf256_blocks, gpu_ns_out != NULL)) {
        ctx->pending_owned_cnt = 0;
        return false;
    }
    if (!vxfec_vk__submit_and_wait(ctx))
        return false;
    if (gpu_ns_out)
        *gpu_ns_out = vxfec_vk__read_gpu_ns(ctx);
    return true;
}

uint64_t vxfec_vk__submit_wire_hybrid_cmd(vxfec_vk_ctx *ctx, vxfec_vk_buf *wire_vram, size_t wire_bytes, uint32_t Z,
                                          uint32_t stride, vxfec_vk_buf *gf2_tab, size_t gf2_bytes, size_t gf2_hdr_bytes,
                                          size_t gf2_cmd_off, uint32_t gf2_tile_e, uint32_t gf2_hdr_stride,
                                          uint32_t gf2_active_tiles, uint32_t gf2_blocks, vxfec_vk_buf *gf256_tab,
                                          size_t gf256_bytes, size_t gf256_hdr_bytes, size_t gf256_cmd_off, uint32_t gf256_tile_e,
                                          uint32_t gf256_hdr_stride, uint32_t gf256_active_tiles, uint32_t gf256_blocks)
{
    if (!ctx)
        return 0;
    if (!ctx->has_timeline_sem || !ctx->timeline_sem) {
        return vxfec_vk__exec_wire_hybrid_cmd(ctx, wire_vram, wire_bytes, Z, stride, gf2_tab, gf2_bytes, gf2_hdr_bytes, gf2_cmd_off,
                                              gf2_tile_e, gf2_hdr_stride, gf2_active_tiles, gf2_blocks, gf256_tab, gf256_bytes,
                                              gf256_hdr_bytes, gf256_cmd_off, gf256_tile_e, gf256_hdr_stride, gf256_active_tiles,
                                              gf256_blocks, NULL)
                   ? 1u
                   : 0u;
    }

    bool run_gf2 = (gf2_blocks > 0 && gf2_active_tiles > 0);
    bool run_256 = (gf256_blocks > 0 && gf256_active_tiles > 0);
    if (!run_gf2 && !run_256)
        return 1u;
    if (!run_256) {
        return vxfec_vk__submit_wire_cmd(ctx, wire_vram, wire_bytes, gf2_tab, gf2_bytes, gf2_hdr_bytes, gf2_cmd_off, true, true,
                                         gf2_tile_e, Z, stride, gf2_hdr_stride, gf2_active_tiles, gf2_blocks, false);
    }
    if (!run_gf2) {
        return vxfec_vk__submit_wire_cmd(ctx, wire_vram, wire_bytes, gf256_tab, gf256_bytes, gf256_hdr_bytes, gf256_cmd_off, true,
                                         false, gf256_tile_e, Z, stride, gf256_hdr_stride, gf256_active_tiles, gf256_blocks, false);
    }

    uint32_t slot = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!vxfec_vk__acquire_async_slot(ctx, &slot, &cmd))
        return 0;
    if (!record_wire_dual_cmd(ctx, cmd, ctx->async_ds_mix[slot], ctx->async_ds_mix_aux[slot], wire_vram, wire_bytes, Z, stride,
                              gf2_tab, gf2_bytes, gf2_hdr_bytes, gf2_cmd_off, gf2_tile_e, gf2_hdr_stride, gf2_active_tiles,
                              gf2_blocks, gf256_tab, gf256_bytes, gf256_hdr_bytes, gf256_cmd_off, gf256_tile_e, gf256_hdr_stride,
                              gf256_active_tiles, gf256_blocks, false)) {
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }
    return vxfec_vk__submit_async_slot(ctx, slot, cmd);
}
