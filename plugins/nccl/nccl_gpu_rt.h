#ifndef VXFEC_NCCL_GPU_RT_H
#define VXFEC_NCCL_GPU_RT_H

#include "nanorq_core.h"
#include "nanorq_ops.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>

#define VXFEC_GPURT_MAX_POOL 8u
#define VXF_GPURT_CONTEXT_DESTROYED 709 /* hipErrorContextIsDestroyed / CUDA_ERROR_CONTEXT_IS_DESTROYED */

typedef enum {
    VXF_GPURT_NONE = 0,
    VXF_GPURT_HIP = 1,
    VXF_GPURT_CUDA = 2,
} vxf_gpurt_kind_t;

typedef struct {
    vxf_gpurt_kind_t kind;
    char backend_name[32];
    char arch_name[32];
    int device_ordinal;
    uint32_t total_units;
    uint32_t reserved_units;
    bool used_sm_partition;
    bool uses_high_priority_stream;
    bool has_dmabuf_export;
} vxf_gpurt_caps_t;

bool vxf_gpurt_should_auto_enable(void);
bool vxf_gpurt_init(vxf_gpurt_caps_t *out_caps);
void vxf_gpurt_shutdown(void);
bool vxf_gpurt_wait_idle(void);
bool vxf_gpurt_get_caps(vxf_gpurt_caps_t *out_caps);

/* HIP's pointer-attribute ABI on ROCm 6 and 7 uses hipMemoryTypeDevice = 2. */
static inline bool vxf_hip_device_ordinal(int (*pointer_attr)(void *, int, void *), const void *ptr, int *ordinal)
{
    unsigned int type = 0;
    return pointer_attr(&type, 2 /* HIP_POINTER_ATTRIBUTE_MEMORY_TYPE */, (void *)ptr) == 0 && type == 2u &&
           pointer_attr(ordinal, 9 /* HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL */, (void *)ptr) == 0;
}

bool vxf_gpurt_validate_dev_ptr(const void *ptr);
int vxf_gpurt_export_dmabuf(const void *dev_ptr, size_t size, uint64_t *out_offset);
bool vxf_gpurt_memcpy_dtoh(void *dst_host, const void *src_dev, size_t bytes);
bool vxf_gpurt_memcpy_htod(void *dst_dev, const void *src_host, size_t bytes);

/* A queued copy owns the caller's device memory and the shared bounce buffer
 * until its stream drains. A driver error can originate from earlier work. */
static inline bool vxf_gpurt_finish_copy(void *stream, int (*sync)(void *), int (*query)(void *))
{
    int status = sync(stream);
    if (status == 0)
        return true;
    while (status != VXF_GPURT_CONTEXT_DESTROYED) {
        status = query(stream);
        if (status != 600 /* hipErrorNotReady / CUDA_ERROR_NOT_READY */)
            break;
        usleep(1000);
        status = sync(stream);
        if (status == 0)
            break;
    }
    return false;
}

int vxf_gpurt_pool_acquire(const void *prefer_dev_ptr, size_t in_bytes, size_t out_bytes, void **out_in_stage,
                           void **out_out_stage);
void vxf_gpurt_pool_release(int slot);
uint64_t vxf_gpurt_pool_out_id(int slot);
int vxf_gpurt_pool_export_out_dmabuf(int slot, size_t min_bytes, uint64_t *out_offset, size_t *out_cap);

uint64_t vxf_gpurt_rs8_submit(int slot, bool is_decode, uint32_t K, uint32_t M, const uint8_t *erased_k, const uint8_t *parity_m,
                              uint32_t count, uint32_t sym_T);
uint64_t vxf_gpurt_rs8_submit_vram_enc(int slot, const void *src_dev_ptr, size_t src_offset, size_t valid_bytes, uint32_t K,
                                       uint32_t M, const uint8_t *parity_m, uint32_t count, uint32_t sym_T, bool use_vram_out);
uint64_t vxf_gpurt_rs8_submit_vram_rep(int slot, void *target_dev_ptr, size_t target_offset, size_t total_bytes, uint32_t K,
                                       uint32_t M, const uint8_t *sys_bitmap, const uint8_t *erased_k, const uint8_t *parity_m,
                                       uint32_t count, uint32_t sym_T);

uint64_t vxf_gpurt_rq_submit(int slot, nanorq_core *rq, const schedule *ops, uint32_t stride, const uint32_t *esis, uint32_t count);
uint64_t vxf_gpurt_rq_submit_vram_enc(int slot, const void *src_dev_ptr, size_t src_offset, size_t valid_bytes, nanorq_core *rq,
                                      const schedule *ops, uint32_t sym_T, uint32_t stride, const uint32_t *esis, uint32_t count,
                                      bool use_vram_out);
uint64_t vxf_gpurt_rq_submit_vram_rep(int slot, void *target_dev_ptr, size_t target_offset, size_t total_bytes, nanorq_core *rq,
                                      const schedule *ops, const uint8_t *sys_bitmap, uint32_t K, uint32_t sym_T, uint32_t stride,
                                      const uint32_t *missing_esis, uint32_t count, uint32_t overhead);

typedef enum {
    VXF_GPURT_POLL_UNSAFE = -3, /* runtime error with no proof of quiescence */
    VXF_GPURT_POLL_FAILED = -2, /* operation failed and its buffers are safe to reuse */
    VXF_GPURT_POLL_UNKNOWN = -1,
    VXF_GPURT_POLL_PENDING = 0,
    VXF_GPURT_POLL_COMPLETE = 1,
} vxf_gpurt_poll_status_t;

vxf_gpurt_poll_status_t vxf_gpurt_poll(uint64_t ticket, bool wait);

#endif /* VXFEC_NCCL_GPU_RT_H */
