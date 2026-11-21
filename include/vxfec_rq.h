#ifndef VXFEC_RQ_H
#define VXFEC_RQ_H

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vxfec_vk.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "nanorq_core.h"
#include "nanorq_ops.h"

typedef struct vxfec_rq_sched vxfec_rq_sched;
typedef struct vxfec_rq_cpu_sched vxfec_rq_cpu_sched;
typedef struct vxfec_rq_direct vxfec_rq_direct;

/* prepack RaptorQ schedule ops on CPU without holding a Vulkan context lock */
vxfec_rq_cpu_sched *vxfec_rq_cpu_sched_pack(nanorq_core *rq, const schedule *S);
void vxfec_rq_cpu_sched_free(vxfec_rq_cpu_sched *cs);
vxfec_rq_sched *vxfec_rq_sched_from_cpu_staged(vxfec_vk_ctx *ctx, vxfec_rq_cpu_sched **cs_inout);

/* compile and upload packed RQ schedule to VRAM (GPU stride must be 4B-aligned) */
vxfec_rq_sched *vxfec_rq_sched_new(vxfec_vk_ctx *ctx, nanorq_core *rq, const schedule *S);
/* stage packed RQ schedule without blocking upload */
vxfec_rq_sched *vxfec_rq_sched_new_staged(vxfec_vk_ctx *ctx, nanorq_core *rq, const schedule *S);
void vxfec_rq_sched_free(vxfec_vk_ctx *ctx, vxfec_rq_sched *gs);
bool vxfec_rq_sched_is_pure_gf2(const vxfec_rq_sched *gs);

/* compile schedule + output ESIs into a single-pass VRAM operator (out = G * D_in) */
vxfec_rq_direct *vxfec_rq_direct_new(vxfec_vk_ctx *ctx, nanorq_core *rq, const schedule *S, const uint32_t *esis,
                                     uint32_t num_esis);
void vxfec_rq_direct_free(vxfec_vk_ctx *ctx, vxfec_rq_direct *gd);
bool vxfec_rq_direct_is_pure_gf2(const vxfec_rq_direct *gd);

/* execute ops_run in-place across num_blocks matrices in VRAM (stride % 4 == 0) */
bool vxfec_rq_ops_run_batch(vxfec_vk_ctx *ctx, const vxfec_rq_sched *gs, vxfec_vk_buf *d_vram, uint32_t stride, uint32_t num_blocks,
                            uint64_t *gpu_ns_out);

/* mix GPU-solved matrices in physical pivot order (stride % 4 == 0) */
bool vxfec_rq_ops_mix_batch(vxfec_vk_ctx *ctx, nanorq_core *rq, vxfec_vk_buf *d_vram, uint32_t stride, uint32_t num_blocks,
                            const uint32_t *esis, uint32_t num_esis, vxfec_vk_buf *out_vram, uint64_t *gpu_ns_out);

/* execute precompiled direct operator across num_blocks matrices in VRAM */
bool vxfec_rq_direct_batch(vxfec_vk_ctx *ctx, const vxfec_rq_direct *gd, vxfec_vk_buf *d_vram, uint32_t stride, uint32_t num_blocks,
                           vxfec_vk_buf *out_vram, uint64_t *gpu_ns_out);

/* single-submit pipelined upload(D) -> solve+mix -> download(out) */
bool vxfec_rq_run_and_mix_pipelined(vxfec_vk_ctx *ctx, nanorq_core *rq, const vxfec_rq_sched *gs, vxfec_vk_buf *d_buf,
                                    uint32_t stride, uint32_t num_blocks, const uint32_t *esis, uint32_t num_esis,
                                    vxfec_vk_buf *out_buf, bool upload_d, bool download_out, uint64_t *gpu_ns_out);

/* non-blocking async timeline-semaphore submission; returns ticket > 0 or 0 on error */
uint64_t vxfec_rq_direct_submit(vxfec_vk_ctx *ctx, const vxfec_rq_direct *gd, vxfec_vk_buf *d_buf, uint32_t stride,
                                uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out);
uint64_t vxfec_rq_direct_submit_raw(vxfec_vk_ctx *ctx, nanorq_core *rq, const schedule *S, vxfec_vk_buf *d_buf, uint32_t stride,
                                    uint32_t num_blocks, const uint32_t *esis, uint32_t num_esis, vxfec_vk_buf *out_buf,
                                    bool upload_d, bool download_out);
uint64_t vxfec_rq_run_and_mix_submit(vxfec_vk_ctx *ctx, nanorq_core *rq, const vxfec_rq_sched *gs, vxfec_vk_buf *d_buf,
                                     uint32_t stride, uint32_t num_blocks, const uint32_t *esis, uint32_t num_esis,
                                     vxfec_vk_buf *out_buf, bool upload_d, bool download_out);

/* GPU-resident VRAM-to-VRAM RaptorQ encode and in-place VRAM erasure repair */
uint64_t vxfec_rq_encode_submit_vram(vxfec_vk_ctx *ctx, nanorq_core *rq, const vxfec_rq_sched *gs, vxfec_vk_buf *src_vram,
                                     size_t src_offset, size_t valid_bytes, uint32_t sym_T, vxfec_vk_buf *work_d_buf,
                                     uint32_t stride, const uint32_t *esis, uint32_t num_esis, vxfec_vk_buf *out_buf,
                                     bool download_out);
uint64_t vxfec_rq_repair_submit_vram(vxfec_vk_ctx *ctx, nanorq_core *rq, const vxfec_rq_sched *gs, const uint8_t *sys_bitmap,
                                     uint32_t K, uint32_t sym_T, vxfec_vk_buf *target_vram, size_t target_offset,
                                     size_t total_bytes, vxfec_vk_buf *work_d_buf, uint32_t stride, const uint32_t *missing_esis,
                                     uint32_t num_missing, uint32_t overhead, vxfec_vk_buf *out_buf);

#ifdef __cplusplus
}
#endif

#endif /* VXFEC_RQ_H */
