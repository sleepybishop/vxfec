#ifndef VXFEC_VK_INTERNAL_H
#define VXFEC_VK_INTERNAL_H

#include "nanorq_core.h"
#include "nanorq_ops.h"
#include "vxfec_vk.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#define VXFEC_VK_ASYNC_SLOTS 32u
#define VXFEC_VK_MAX_PENDING_OWNED 8u

#ifndef VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME
#define VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME "VK_EXT_queue_family_foreign"
#endif
#ifndef VK_QUEUE_FAMILY_FOREIGN_EXT
#define VK_QUEUE_FAMILY_FOREIGN_EXT (~2U)
#endif

enum {
    VXFEC_VK_EXT_NONE = 0,
    VXFEC_VK_EXT_DMABUF = 1,
};

typedef struct {
    uint32_t j;
    uint32_t i0;
    uint32_t i1;
    uint32_t i2;
    uint32_t i3;
    uint32_t u_pack;
} vxfec_gpu_op4;

typedef struct {
    uint32_t rows;
    uint32_t stride_units;
    uint32_t cp0_op4;
    uint32_t cp1_op4;
    uint32_t fwd_op4;
    uint32_t total_op4;
    uint32_t block_offset;
    uint32_t hdpc_beg_op4;
    uint32_t hdpc_end_op4;
} vxfec_run_push_consts;

typedef struct {
    uint32_t rows;
    uint32_t stride_units;
    uint32_t num_esis;
    uint32_t esi_offset;
    uint32_t block_offset;
} vxfec_mix_push_consts;

struct vxfec_vk_buf {
    VkBuffer dev_buf;
    VkDeviceMemory dev_mem;
    VkBuffer stage_buf;
    VkDeviceMemory stage_mem;
    void *stage_ptr;
    size_t bytes;
    size_t view_offset;
    uint8_t ext_kind;
    bool owned_by_vk;
    bool zero_copy;
    bool imported_host;
    struct vxfec_vk_buf *next_retired;
};

/* shared GPU descriptor for any DirectTab (GF(2) or GF(256) nibble) operator */
typedef struct {
    vxfec_vk_buf *tab_buf;
    size_t tab_bytes;
    uint32_t rows;
    uint32_t num_esis;
    uint32_t num_tiles;
    uint32_t tile_e;
    bool pure_gf2;
} vxfec_vk_direct_desc;

struct vxfec_vk_ctx {
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;

    char dev_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint32_t vendor_id;
    bool is_discrete;
    bool dedicated_compute;
    uint32_t max_shared_bytes;
    uint32_t max_wg_x;
    uint32_t max_wg_y;
    uint32_t max_wg_z;
    VkDeviceSize max_ssbo_range;
    VkDeviceSize ssbo_offset_align;
    VkDeviceSize host_import_align;
    uint32_t ts_valid_bits;
    float timestamp_period_ns;
    bool has_timestamps;
    bool has_timeline_sem;
    bool has_ext_mem_fd;
    bool has_dmabuf;
    bool has_dmabuf_export;
    bool dmabuf_dedicated_only;
    bool has_ext_host_mem;
    bool has_foreign_qfam;

    PFN_vkGetMemoryFdKHR pfn_get_mem_fd;
    PFN_vkGetMemoryFdPropertiesKHR pfn_get_mem_fd_props;
    PFN_vkGetMemoryHostPointerPropertiesEXT pfn_get_host_ptr_props;

    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkFence fence;
    VkQueryPool query_pool;

    VkSemaphore timeline_sem;
    uint64_t timeline_next;
    uint64_t timeline_submitted;
    VkCommandBuffer async_cmds[VXFEC_VK_ASYNC_SLOTS];
    VkDescriptorSet async_ds_run[VXFEC_VK_ASYNC_SLOTS];
    VkDescriptorSet async_ds_mix[VXFEC_VK_ASYNC_SLOTS];
    VkDescriptorSet async_ds_mix_aux[VXFEC_VK_ASYNC_SLOTS];
    vxfec_vk_buf *async_tab_bufs[VXFEC_VK_ASYNC_SLOTS];
    size_t async_tab_caps[VXFEC_VK_ASYNC_SLOTS];
    uint64_t async_slot_tickets[VXFEC_VK_ASYNC_SLOTS];
    uint32_t async_cursor;

    /* pending external ownership transitions recorded into active cmd */
    vxfec_vk_buf *pending_owned_bufs[VXFEC_VK_MAX_PENDING_OWNED];
    uint32_t pending_owned_cnt;
    vxfec_vk_buf *pending_release_buf;
    vxfec_vk_buf *retired_bufs;
    bool last_submit_unconfirmed;

    VkDescriptorSetLayout dsl_run;
    VkDescriptorSetLayout dsl_mix;
    VkPipelineLayout pl_run;
    VkPipelineLayout pl_mix;
    VkDescriptorPool desc_pool;
    VkDescriptorSet ds_run;
    VkDescriptorSet ds_mix;
    VkDescriptorSet ds_mix_aux;

    VkPipeline pipe_vec4_gf256;
    VkPipeline pipe_vec4_gf2;
    VkPipeline pipe_vram_gf256;
    VkPipeline pipe_vram_gf2;
    VkPipeline pipe_mix_vec4;
    VkPipeline pipe_mix;
    VkPipeline pipe_direct_gf2_vec4_16;
    VkPipeline pipe_direct_gf2_vec4_32;
    VkPipeline pipe_direct_gf2_16;
    VkPipeline pipe_direct_gf2_32;
    VkPipeline pipe_direct_gf256_nib_vec4_8;
    VkPipeline pipe_direct_gf256_nib_vec4_16;
    VkPipeline pipe_direct_gf256_nib_8;
    VkPipeline pipe_direct_gf256_nib_16;
    VkPipeline pipe_wire_gf2_vec4_16;
    VkPipeline pipe_wire_gf2_vec4_32;
    VkPipeline pipe_wire_gf256_nib_vec4_8;
    VkPipeline pipe_wire_gf256_nib_vec4_16;

    vxfec_vk_buf *mix_tab_buf;
    size_t mix_tab_cap;
    vxfec_vk_buf *direct_tab_buf;
    size_t direct_tab_cap;

    VkBuffer io_stage_buf;
    VkDeviceMemory io_stage_mem;
    void *io_stage_ptr;
    size_t io_stage_cap;
};

static inline bool vxfec_vk__mul3(size_t a, size_t b, size_t c, size_t *out)
{
    size_t ab = 0;
    if (a != 0 && b > SIZE_MAX / a)
        return false;
    ab = a * b;
    if (ab != 0 && c > SIZE_MAX / ab)
        return false;
    *out = ab * c;
    return true;
}

bool vxfec_vk__begin_cmd(vxfec_vk_ctx *ctx);
bool vxfec_vk__submit_and_wait(vxfec_vk_ctx *ctx);
uint64_t vxfec_vk__read_gpu_ns(vxfec_vk_ctx *ctx);
void vxfec_vk__record_transfer_to_host_cmd(VkCommandBuffer cmd);
uint32_t vxfec_vk__ext_queue_family(const vxfec_vk_ctx *ctx, const vxfec_vk_buf *buf);
void vxfec_vk__record_ext_acquire_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, vxfec_vk_buf *buf, VkPipelineStageFlags dst_stage,
                                      VkAccessFlags dst_access);
void vxfec_vk__record_ext_release_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, vxfec_vk_buf *buf, VkPipelineStageFlags src_stage,
                                      VkAccessFlags src_access);
bool vxfec_vk__ensure_owned_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, vxfec_vk_buf *buf, VkPipelineStageFlags dst_stage,
                                VkAccessFlags dst_access);

bool vxfec_vk__acquire_async_slot(vxfec_vk_ctx *ctx, uint32_t *out_slot, VkCommandBuffer *out_cmd);
uint64_t vxfec_vk__submit_async_slot(vxfec_vk_ctx *ctx, uint32_t slot, VkCommandBuffer cmd);

bool vxfec_vk__ensure_staging_buf(vxfec_vk_ctx *ctx, vxfec_vk_buf **io_buf, size_t *io_cap, size_t need_bytes);
void vxfec_vk__release_staging(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf);
void vxfec_vk__retire_buf(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf);
void vxfec_vk__retire_or_free_buf(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf);
vxfec_vk_buf *vxfec_vk__upload_const_buf(vxfec_vk_ctx *ctx, const void *data, size_t bytes);

uint32_t vxfec_vk__compute_tile_blocks(const vxfec_vk_ctx *ctx, uint32_t rows, uint32_t stride, uint32_t wg_x, uint32_t num_blocks);

bool vxfec_vk__select_direct_plan(const vxfec_vk_ctx *ctx, bool pure_gf2, uint32_t tile_e, size_t d_bytes, size_t out_bytes,
                                  uint32_t stride, uint32_t num_tiles, uint32_t num_blocks, VkPipeline *out_pipe,
                                  uint32_t *out_stride_units, uint32_t *out_wg_x);

void vxfec_vk__record_mix_like_dispatches_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, VkDescriptorSet ds_mix, uint32_t rows,
                                              uint32_t stride_units, uint32_t wg_x, uint32_t num_esis, uint32_t y_count,
                                              uint32_t block_base, uint32_t num_blocks, VkPipeline pipe);

bool vxfec_vk__record_direct_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, VkDescriptorSet ds_mix, const vxfec_vk_direct_desc *gd,
                                 vxfec_vk_buf *d_buf, uint32_t stride, uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d,
                                 bool download_out, bool record_ts);

bool vxfec_vk__direct_pipelined(vxfec_vk_ctx *ctx, const vxfec_vk_direct_desc *gd, vxfec_vk_buf *d_buf, uint32_t stride,
                                uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out, uint64_t *gpu_ns_out);
uint64_t vxfec_vk__direct_submit(vxfec_vk_ctx *ctx, const vxfec_vk_direct_desc *gd, vxfec_vk_buf *d_buf, uint32_t stride,
                                 uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out);
uint64_t vxfec_vk__direct_submit_staged_words(vxfec_vk_ctx *ctx, const uint32_t *tab_words, size_t tab_bytes, uint32_t rows,
                                              uint32_t num_esis, uint32_t num_tiles, uint32_t tile_e, bool pure_gf2,
                                              vxfec_vk_buf *d_buf, uint32_t stride, uint32_t num_blocks, vxfec_vk_buf *out_buf,
                                              bool upload_d, bool download_out);

uint32_t vxfec_vk__choose_gf256_tile_e(uint32_t num_out);

/* pack a dense rows x num_out GF(256) matrix Y into an 8/16-output nibble DirectTab256 */
bool vxfec_vk__emit_direct_gf256_tab(VkDeviceSize max_ssbo_range, const uint8_t *Y, size_t m_stride, uint32_t rows,
                                     uint32_t num_out, uint32_t **out_words, size_t *out_bytes, uint32_t *out_tiles,
                                     uint32_t *out_tile_e);

bool vxfec_vk__exec_wire_cmd(vxfec_vk_ctx *ctx, vxfec_vk_buf *wire_vram, size_t wire_bytes, vxfec_vk_buf *tab_buf, size_t tab_bytes,
                             size_t tab_hdr_bytes, size_t tab_cmd_src_off, bool upload_tab, bool pure_gf2, uint32_t tile_e,
                             uint32_t Z, uint32_t stride, uint32_t hdr_stride_tiles, uint32_t active_tiles, uint32_t disp_blocks,
                             bool uniform_tab, uint64_t *gpu_ns_out);

bool vxfec_vk__exec_wire_hybrid_cmd(vxfec_vk_ctx *ctx, vxfec_vk_buf *wire_vram, size_t wire_bytes, uint32_t Z, uint32_t stride,
                                    vxfec_vk_buf *gf2_tab, size_t gf2_bytes, size_t gf2_hdr_bytes, size_t gf2_cmd_off,
                                    uint32_t gf2_tile_e, uint32_t gf2_hdr_stride, uint32_t gf2_active_tiles, uint32_t gf2_blocks,
                                    vxfec_vk_buf *gf256_tab, size_t gf256_bytes, size_t gf256_hdr_bytes, size_t gf256_cmd_off,
                                    uint32_t gf256_tile_e, uint32_t gf256_hdr_stride, uint32_t gf256_active_tiles,
                                    uint32_t gf256_blocks, uint64_t *gpu_ns_out);

uint64_t vxfec_vk__submit_wire_cmd(vxfec_vk_ctx *ctx, vxfec_vk_buf *wire_vram, size_t wire_bytes, vxfec_vk_buf *tab_buf,
                                   size_t tab_bytes, size_t tab_hdr_bytes, size_t tab_cmd_src_off, bool upload_tab, bool pure_gf2,
                                   uint32_t tile_e, uint32_t Z, uint32_t stride, uint32_t hdr_stride_tiles, uint32_t active_tiles,
                                   uint32_t disp_blocks, bool uniform_tab);

uint64_t vxfec_vk__submit_wire_hybrid_cmd(vxfec_vk_ctx *ctx, vxfec_vk_buf *wire_vram, size_t wire_bytes, uint32_t Z,
                                          uint32_t stride, vxfec_vk_buf *gf2_tab, size_t gf2_bytes, size_t gf2_hdr_bytes,
                                          size_t gf2_cmd_off, uint32_t gf2_tile_e, uint32_t gf2_hdr_stride,
                                          uint32_t gf2_active_tiles, uint32_t gf2_blocks, vxfec_vk_buf *gf256_tab,
                                          size_t gf256_bytes, size_t gf256_hdr_bytes, size_t gf256_cmd_off, uint32_t gf256_tile_e,
                                          uint32_t gf256_hdr_stride, uint32_t gf256_active_tiles, uint32_t gf256_blocks);

/* shared RQ schedule & adjoint helpers reused by vk_rq.c and wire_ring.c */
bool vxfec_rq__sched_is_pure_gf2(const schedule *S);
void vxfec_rq__adjoint_gf2(nanorq_core *rq, const sched_op *ops, uint32_t n_ops, uint32_t cp0, uint32_t cp1, uint32_t rows,
                           const uint32_t *esis, uint32_t num_esis, uint64_t *Y, size_t wpr);
void vxfec_rq__adjoint_gf256(nanorq_core *rq, const sched_op *ops, uint32_t n_ops, uint32_t cp0, uint32_t cp1, uint32_t rows,
                             const uint32_t *esis, uint32_t num_esis, uint8_t *Y, size_t m_stride);

bool vxfec_rs8__encode_matrix(uint32_t K, uint32_t M, const uint8_t *parity_m, uint32_t num_parity, uint8_t *Y, size_t m_stride);
bool vxfec_rs8__recovery_matrix(uint32_t K, uint32_t M, const uint8_t *erased_k, const uint8_t *parity_m, uint32_t e, uint8_t *Y,
                                size_t m_stride);

#endif /* VXFEC_VK_INTERNAL_H */
