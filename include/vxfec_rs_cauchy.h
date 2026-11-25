#ifndef VXFEC_RS_CAUCHY_H
#define VXFEC_RS_CAUCHY_H

#include "vxfec_vk.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * strict MDS Cauchy RS8 over GF(256) (poly 0x11d), wire-compatible with nanors:
 *   VXFEC_RS8_GEN_CAUCHY     : A_{m,k} = 1 / ((M + k) ^ m),       K + M <= 256
 *   VXFEC_RS8_GEN_CAUCHY_EXP : A_{m,k} = 1 / (a^(254 - k) + a^m), K + M <= 255
 * GPU execution uses 4-bit nibble register-table shaders and requires stride % 4 == 0;
 * CPU execution accepts any byte stride >= 1.
 */
typedef enum {
    VXFEC_RS8_GEN_CAUCHY = 0,
    VXFEC_RS8_GEN_CAUCHY_EXP = 1,
} vxfec_rs8_gen_t;

typedef struct vxfec_rs8_plan vxfec_rs8_plan;

/* compile encode plan for parity_m[0..num_parity-1] (ctx may be NULL for CPU-only) */
vxfec_rs8_plan *vxfec_rs8_encode_plan_new(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, const uint8_t *parity_m, uint32_t num_parity);
vxfec_rs8_plan *vxfec_rs8_encode_plan_new_ex(vxfec_vk_ctx *ctx, vxfec_rs8_gen_t gen, uint32_t K, uint32_t M,
                                             const uint8_t *parity_m, uint32_t num_parity);

/* compile erasure-recovery plan for erased_k[0..num_erased-1] using parity_m */
vxfec_rs8_plan *vxfec_rs8_decode_plan_new(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, const uint8_t *erased_k,
                                          const uint8_t *parity_m, uint32_t num_erased);
vxfec_rs8_plan *vxfec_rs8_decode_plan_new_ex(vxfec_vk_ctx *ctx, vxfec_rs8_gen_t gen, uint32_t K, uint32_t M,
                                             const uint8_t *erased_k, const uint8_t *parity_m, uint32_t num_erased);

void vxfec_rs8_plan_free(vxfec_vk_ctx *ctx, vxfec_rs8_plan *plan);
uint32_t vxfec_rs8_plan_k(const vxfec_rs8_plan *plan);
uint32_t vxfec_rs8_plan_m(const vxfec_rs8_plan *plan);
uint32_t vxfec_rs8_plan_outputs(const vxfec_rs8_plan *plan);

/* execute plan on CPU using SIMD GF(256) axpy kernels */
bool vxfec_rs8_exec_cpu(const vxfec_rs8_plan *plan, const uint8_t *in_rows, uint32_t stride, uint32_t num_blocks,
                        uint8_t *out_rows);

/* one-shot CPU helpers */
bool vxfec_rs8_encode_cpu(uint32_t K, uint32_t M, const uint8_t *sys_rows, uint32_t stride, const uint8_t *parity_m,
                          uint32_t num_parity, uint8_t *parity_out);
bool vxfec_rs8_encode_cpu_ex(vxfec_rs8_gen_t gen, uint32_t K, uint32_t M, const uint8_t *sys_rows, uint32_t stride,
                             const uint8_t *parity_m, uint32_t num_parity, uint8_t *parity_out);
bool vxfec_rs8_decode_compact_cpu(uint32_t K, uint32_t M, uint8_t *compact_k_rows, uint32_t stride, const uint8_t *erased_k,
                                  const uint8_t *parity_m, uint32_t num_erased);
bool vxfec_rs8_decode_compact_cpu_ex(vxfec_rs8_gen_t gen, uint32_t K, uint32_t M, uint8_t *compact_k_rows, uint32_t stride,
                                     const uint8_t *erased_k, const uint8_t *parity_m, uint32_t num_erased);

/* execute compiled GPU plan across num_blocks matrices in VRAM (stride % 4 == 0) */
bool vxfec_rs8_exec_batch(vxfec_vk_ctx *ctx, const vxfec_rs8_plan *plan, vxfec_vk_buf *d_vram, uint32_t stride, uint32_t num_blocks,
                          vxfec_vk_buf *out_vram, uint64_t *gpu_ns_out);
bool vxfec_rs8_exec_pipelined(vxfec_vk_ctx *ctx, const vxfec_rs8_plan *plan, vxfec_vk_buf *d_buf, uint32_t stride,
                              uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out, uint64_t *gpu_ns_out);
uint64_t vxfec_rs8_exec_submit(vxfec_vk_ctx *ctx, const vxfec_rs8_plan *plan, vxfec_vk_buf *d_buf, uint32_t stride,
                               uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out);

/* zero-allocation non-blocking RS8 GPU submission staged into async timeline ring */
uint64_t vxfec_rs8_submit_raw(vxfec_vk_ctx *ctx, bool is_decode, uint32_t K, uint32_t M, const uint8_t *erased_k,
                              const uint8_t *parity_m, uint32_t num_out, vxfec_vk_buf *d_buf, uint32_t stride, uint32_t num_blocks,
                              vxfec_vk_buf *out_buf, bool upload_d, bool download_out);
uint64_t vxfec_rs8_submit_raw_ex(vxfec_vk_ctx *ctx, vxfec_rs8_gen_t gen, bool is_decode, uint32_t K, uint32_t M,
                                 const uint8_t *erased_k, const uint8_t *parity_m, uint32_t num_out, vxfec_vk_buf *d_buf,
                                 uint32_t stride, uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out);

/* GPU-resident VRAM-to-VRAM RS8 encode and in-place VRAM erasure repair (zero host payload bounce) */
uint64_t vxfec_rs8_encode_submit_vram(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, const uint8_t *parity_m, uint32_t num_parity,
                                      vxfec_vk_buf *src_vram, size_t src_offset, size_t valid_bytes, vxfec_vk_buf *work_d_buf,
                                      uint32_t stride, vxfec_vk_buf *out_buf, bool download_out);
uint64_t vxfec_rs8_repair_submit_vram(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, const uint8_t *sys_bitmap, const uint8_t *erased_k,
                                      const uint8_t *parity_m, uint32_t num_erased, vxfec_vk_buf *target_vram, size_t target_offset,
                                      size_t total_bytes, vxfec_vk_buf *work_d_buf, uint32_t stride, vxfec_vk_buf *out_buf);

#ifdef __cplusplus
}
#endif

#endif /* VXFEC_RS_CAUCHY_H */
