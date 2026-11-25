#ifndef VXFEC_VK_H
#define VXFEC_VK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vxfec_vk_ctx vxfec_vk_ctx;
typedef struct vxfec_vk_buf vxfec_vk_buf;

/* contexts and buffers require external synchronization per context */
enum {
    VXFEC_VK_OK = 0,
    VXFEC_VK_UNAVAILABLE = 1,
    VXFEC_VK_ERROR = 2,
};

/*
 * initialize a Vulkan 1.2 compute context (prefers discrete GPU + dedicated
 * ACE compute queue). overrides: VXFEC_VK_DEVICE, VXFEC_VK_QUEUE_FAMILY.
 */
vxfec_vk_ctx *vxfec_vk_ctx_new(void);
vxfec_vk_ctx *vxfec_vk_ctx_new_status(int *status);
vxfec_vk_ctx *vxfec_vk_ctx_new_at(uint32_t device_index, int *status);
vxfec_vk_ctx *vxfec_vk_ctx_new_for_dmabuf(int dmabuf_fd, int *status);
bool vxfec_vk_ctx_supports_dmabuf_fd(vxfec_vk_ctx *ctx, int dmabuf_fd);
void vxfec_vk_ctx_free(vxfec_vk_ctx *ctx);

const char *vxfec_vk_device_name(const vxfec_vk_ctx *ctx);
bool vxfec_vk_device_uuid(const vxfec_vk_ctx *ctx, uint8_t out_uuid[16]);
bool vxfec_vk_device_pci_bus_info(const vxfec_vk_ctx *ctx, char *out_bdf, size_t cap);
bool vxfec_vk_is_discrete(const vxfec_vk_ctx *ctx);
bool vxfec_vk_is_dedicated_compute(const vxfec_vk_ctx *ctx);
uint32_t vxfec_vk_queue_family(const vxfec_vk_ctx *ctx);
uint32_t vxfec_vk_shared_mem_bytes(const vxfec_vk_ctx *ctx);
uint64_t vxfec_vk_max_storage_buffer_bytes(const vxfec_vk_ctx *ctx);
bool vxfec_vk_has_dmabuf(const vxfec_vk_ctx *ctx);
bool vxfec_vk_has_dmabuf_export(const vxfec_vk_ctx *ctx);
bool vxfec_vk_has_ext_host_mem(const vxfec_vk_ctx *ctx);
bool vxfec_vk_has_foreign_queue(const vxfec_vk_ctx *ctx);

/* allocate device-local VRAM buffer with optional mapped host staging */
vxfec_vk_buf *vxfec_vk_buf_new(vxfec_vk_ctx *ctx, size_t bytes, bool with_staging);
void vxfec_vk_buf_free(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf);
void *vxfec_vk_buf_staging_ptr(vxfec_vk_buf *buf);
bool vxfec_vk_buf_upload(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf, size_t bytes);
bool vxfec_vk_buf_download(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf, size_t bytes);
bool vxfec_vk_buf_copy(vxfec_vk_ctx *ctx, vxfec_vk_buf *dst, const vxfec_vk_buf *src, size_t bytes);
bool vxfec_vk_buf_copy_offset(vxfec_vk_ctx *ctx, vxfec_vk_buf *dst, size_t dst_off, const vxfec_vk_buf *src, size_t src_off,
                              size_t bytes);

/*
 * zero-copy external memory interop (DMA-BUF when is_dmabuf=true, or
 * page-aligned host memory; generic OPAQUE_FD with is_dmabuf=false is rejected).
 * on successful import_fd*, ownership of fd transfers; dup() if retaining fd.
 */
vxfec_vk_buf *vxfec_vk_buf_new_exportable(vxfec_vk_ctx *ctx, size_t bytes, bool is_dmabuf, int *out_fd);
vxfec_vk_buf *vxfec_vk_buf_import_fd(vxfec_vk_ctx *ctx, int fd, size_t bytes, bool is_dmabuf);
vxfec_vk_buf *vxfec_vk_buf_import_fd_range(vxfec_vk_ctx *ctx, int fd, size_t base_offset, size_t bytes, bool is_dmabuf);
vxfec_vk_buf *vxfec_vk_buf_import_opaque_fd(vxfec_vk_ctx *ctx, int opaque_fd, size_t alloc_bytes, size_t view_bytes,
                                            void (*on_free)(void *), void *user_ctx);
/* memory_type_index >= 0 pins the exporter's Vulkan memoryTypeIndex; < 0 auto-selects device-local (e.g. CUDA export) */
vxfec_vk_buf *vxfec_vk_buf_import_opaque_fd_ex(vxfec_vk_ctx *ctx, int opaque_fd, size_t alloc_bytes, size_t view_bytes,
                                               int32_t memory_type_index, void (*on_free)(void *), void *user_ctx);
vxfec_vk_buf *vxfec_vk_buf_import_host_ptr(vxfec_vk_ctx *ctx, void *host_ptr, size_t bytes);
bool vxfec_vk_buf_acquire_external(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf);
bool vxfec_vk_buf_release_external(vxfec_vk_ctx *ctx, vxfec_vk_buf *buf);
typedef struct {
    size_t offset;
    size_t bytes;
} vxfec_vk_span;

bool vxfec_vk_buf_read_offset(vxfec_vk_ctx *ctx, vxfec_vk_buf *src_vram, size_t offset, void *dst_host, size_t bytes);
bool vxfec_vk_buf_write_offset(vxfec_vk_ctx *ctx, vxfec_vk_buf *dst_vram, size_t offset, const void *src_host, size_t bytes);
bool vxfec_vk_buf_fill_offset(vxfec_vk_ctx *ctx, vxfec_vk_buf *dst_vram, size_t offset, size_t bytes, uint32_t pattern);
bool vxfec_vk_buf_fill_spans(vxfec_vk_ctx *ctx, vxfec_vk_buf *dst_vram, const vxfec_vk_span *spans, size_t span_count,
                             uint32_t pattern);
uint64_t vxfec_vk_buf_fill_spans_submit(vxfec_vk_ctx *ctx, vxfec_vk_buf *dst_vram, const vxfec_vk_span *spans, size_t span_count,
                                        uint32_t pattern);

/*
 * non-blocking timeline-semaphore completion polling and bounded waiting.
 * vxfec_vk_poll_status returns 1 if complete, 0 if pending, -1 on error.
 */
bool vxfec_vk_async_can_submit(vxfec_vk_ctx *ctx);
bool vxfec_vk_has_timeline(const vxfec_vk_ctx *ctx);
int vxfec_vk_poll_status(vxfec_vk_ctx *ctx, uint64_t ticket);
bool vxfec_vk_poll(vxfec_vk_ctx *ctx, uint64_t ticket);
bool vxfec_vk_wait(vxfec_vk_ctx *ctx, uint64_t ticket, uint64_t timeout_ns);
bool vxfec_vk_wait_idle(vxfec_vk_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* VXFEC_VK_H */
