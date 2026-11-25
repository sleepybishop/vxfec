#ifndef VXFEC_BENCH_COMMON_H
#define VXFEC_BENCH_COMMON_H

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static inline uint64_t vxfec_bench_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline bool vxfec_bench_send_all(int fd, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0u) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0 && errno == ENOTSOCK)
            n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            return false;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return true;
}

static inline bool vxfec_bench_recv_all(int fd, void *buf, size_t len)
{
    uint8_t *p = (uint8_t *)buf;
    while (len > 0u) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0 && errno == ENOTSOCK)
            n = read(fd, p, len);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (n == 0)
            return false;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return true;
}

static inline void vxfec_bench_parse_host_port(const char *spec, char *host_out, size_t host_cap, uint16_t default_port,
                                               uint16_t *port_out)
{
    if (!host_out || host_cap == 0u)
        return;
    snprintf(host_out, host_cap, "0.0.0.0");
    uint16_t port = default_port;
    if (spec && *spec) {
        const char *colon = strchr(spec, ':');
        if (colon) {
            size_t hlen = (size_t)(colon - spec);
            if (hlen >= host_cap)
                hlen = host_cap - 1u;
            memcpy(host_out, spec, hlen);
            host_out[hlen] = '\0';
            unsigned long parsed = strtoul(colon + 1, NULL, 10);
            if (parsed >= 1UL && parsed <= 65535UL)
                port = (uint16_t)parsed;
        } else {
            snprintf(host_out, host_cap, "%.*s", (int)(host_cap - 1u), spec);
        }
    }
    if (port_out)
        *port_out = port;
}

static inline int vxfec_bench_tcp_listen_accept_one(struct in_addr bind_ip, uint16_t port)
{
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0)
        return -1;
    int one = 1;
    (void)setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in baddr;
    memset(&baddr, 0, sizeof(baddr));
    baddr.sin_family = AF_INET;
    baddr.sin_addr = bind_ip;
    baddr.sin_port = htons(port);
    if (bind(lfd, (struct sockaddr *)&baddr, sizeof(baddr)) < 0 || listen(lfd, 4) < 0) {
        close(lfd);
        return -1;
    }
    struct pollfd pfd = {.fd = lfd, .events = POLLIN, .revents = 0};
    int prc;
    do {
        prc = poll(&pfd, 1, 45000);
    } while (prc < 0 && errno == EINTR);
    if (prc <= 0) {
        close(lfd);
        return -1;
    }
    int fd = accept(lfd, NULL, NULL);
    close(lfd);
    if (fd >= 0)
        (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

static inline int vxfec_bench_tcp_connect_retry(struct in_addr peer_ip, uint16_t port, int max_attempts, long retry_ns)
{
    int one = 1;
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0)
            return -1;
        struct sockaddr_in caddr;
        memset(&caddr, 0, sizeof(caddr));
        caddr.sin_family = AF_INET;
        caddr.sin_addr = peer_ip;
        caddr.sin_port = htons(port);
        if (connect(fd, (struct sockaddr *)&caddr, sizeof(caddr)) == 0) {
            (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            return fd;
        }
        close(fd);
        struct timespec ts = {0, retry_ns};
        nanosleep(&ts, NULL);
    }
    return -1;
}

#if defined(VXFEC_VK_H) && !defined(_WIN32)
#include <dlfcn.h>
#include <pthread.h>

/*
 * Optional benchmark-harness --gdr allocator fallback for displayless headless GPUs
 * whose Vulkan ICD lacks VK_EXT_external_memory_dma_buf export (e.g. headless SXM GPUs).
 * Core libvxfec_vk / libnccl-vxfec stay 100% pure Vulkan; this helper is confined to t/.
 */
typedef struct {
    int type;
    int requested_handle_types;
    struct {
        int type;
        int id;
    } location;
    void *win32_meta;
    struct {
        unsigned char compression_type;
        unsigned char gdr_capable;
        unsigned short usage;
        unsigned char reserved[4];
    } alloc_flags;
} vxfec_bench_cu_prop_t;

typedef struct {
    struct {
        int type;
        int id;
    } location;
    int flags;
} vxfec_bench_cu_access_t;

typedef struct {
    void *dl;
    bool ready;
    int (*init)(unsigned int);
    int (*dev_get_count)(int *);
    int (*dev_get)(int *, int);
    int (*dev_get_uuid)(uint8_t[16], int);
    int (*pctx_retain)(void **, int);
    int (*pctx_release)(int);
    int (*ctx_push)(void *);
    int (*ctx_pop)(void **);
    int (*mem_get_gran)(size_t *, const vxfec_bench_cu_prop_t *, int);
    int (*mem_create)(unsigned long long *, size_t, const vxfec_bench_cu_prop_t *, unsigned long long);
    int (*mem_addr_reserve)(unsigned long long *, size_t, size_t, unsigned long long, unsigned long long);
    int (*mem_map)(unsigned long long, size_t, size_t, unsigned long long, unsigned long long);
    int (*mem_set_access)(unsigned long long, size_t, const vxfec_bench_cu_access_t *, size_t);
    int (*mem_export_fd)(void *, unsigned long long, int, unsigned long long);
    int (*mem_get_range_handle)(void *, unsigned long long, size_t, int, unsigned long long);
    int (*mem_unmap)(unsigned long long, size_t);
    int (*mem_addr_free)(unsigned long long, size_t);
    int (*mem_release)(unsigned long long);
} vxfec_bench_cu_syms_t;

typedef struct {
    const vxfec_bench_cu_syms_t *syms;
    int dev;
    unsigned long long dptr;
    unsigned long long alloc_h;
    size_t aligned_sz;
} vxfec_bench_cu_alloc_t;

static vxfec_bench_cu_syms_t g_vxfec_bench_cu_s;
static pthread_once_t g_vxfec_bench_cu_once = PTHREAD_ONCE_INIT;

static void vxfec_bench_cu_syms_init_once(void)
{
    vxfec_bench_cu_syms_t *s = &g_vxfec_bench_cu_s;
    s->dl = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!s->dl)
        s->dl = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
    if (!s->dl)
        return;
    s->init = (int (*)(unsigned int))dlsym(s->dl, "cuInit");
    s->dev_get_count = (int (*)(int *))dlsym(s->dl, "cuDeviceGetCount");
    s->dev_get = (int (*)(int *, int))dlsym(s->dl, "cuDeviceGet");
    s->dev_get_uuid = (int (*)(uint8_t[16], int))dlsym(s->dl, "cuDeviceGetUuid_v2");
    if (!s->dev_get_uuid)
        s->dev_get_uuid = (int (*)(uint8_t[16], int))dlsym(s->dl, "cuDeviceGetUuid");
    s->pctx_retain = (int (*)(void **, int))dlsym(s->dl, "cuDevicePrimaryCtxRetain");
    s->pctx_release = (int (*)(int))dlsym(s->dl, "cuDevicePrimaryCtxRelease_v2");
    if (!s->pctx_release)
        s->pctx_release = (int (*)(int))dlsym(s->dl, "cuDevicePrimaryCtxRelease");
    s->ctx_push = (int (*)(void *))dlsym(s->dl, "cuCtxPushCurrent_v2");
    if (!s->ctx_push)
        s->ctx_push = (int (*)(void *))dlsym(s->dl, "cuCtxPushCurrent");
    s->ctx_pop = (int (*)(void **))dlsym(s->dl, "cuCtxPopCurrent_v2");
    if (!s->ctx_pop)
        s->ctx_pop = (int (*)(void **))dlsym(s->dl, "cuCtxPopCurrent");
    s->mem_get_gran = (int (*)(size_t *, const vxfec_bench_cu_prop_t *, int))dlsym(s->dl, "cuMemGetAllocationGranularity");
    s->mem_create =
        (int (*)(unsigned long long *, size_t, const vxfec_bench_cu_prop_t *, unsigned long long))dlsym(s->dl, "cuMemCreate");
    s->mem_addr_reserve =
        (int (*)(unsigned long long *, size_t, size_t, unsigned long long, unsigned long long))dlsym(s->dl, "cuMemAddressReserve");
    s->mem_map = (int (*)(unsigned long long, size_t, size_t, unsigned long long, unsigned long long))dlsym(s->dl, "cuMemMap");
    s->mem_set_access =
        (int (*)(unsigned long long, size_t, const vxfec_bench_cu_access_t *, size_t))dlsym(s->dl, "cuMemSetAccess");
    s->mem_export_fd = (int (*)(void *, unsigned long long, int, unsigned long long))dlsym(s->dl, "cuMemExportToShareableHandle");
    s->mem_get_range_handle =
        (int (*)(void *, unsigned long long, size_t, int, unsigned long long))dlsym(s->dl, "cuMemGetHandleForAddressRange");
    s->mem_unmap = (int (*)(unsigned long long, size_t))dlsym(s->dl, "cuMemUnmap");
    s->mem_addr_free = (int (*)(unsigned long long, size_t))dlsym(s->dl, "cuMemAddressFree");
    s->mem_release = (int (*)(unsigned long long))dlsym(s->dl, "cuMemRelease");
    if (s->init && s->dev_get_count && s->dev_get && s->dev_get_uuid && s->pctx_retain && s->pctx_release && s->ctx_push &&
        s->ctx_pop && s->mem_get_gran && s->mem_create && s->mem_addr_reserve && s->mem_map && s->mem_set_access &&
        s->mem_export_fd && s->mem_get_range_handle && s->mem_unmap && s->mem_addr_free && s->mem_release && s->init(0) == 0)
        s->ready = true;
}

static inline const vxfec_bench_cu_syms_t *vxfec_bench_cu_syms(void)
{
    pthread_once(&g_vxfec_bench_cu_once, vxfec_bench_cu_syms_init_once);
    return g_vxfec_bench_cu_s.ready ? &g_vxfec_bench_cu_s : NULL;
}

static inline void vxfec_bench_free_cu_alloc(void *arg)
{
    vxfec_bench_cu_alloc_t *ca = (vxfec_bench_cu_alloc_t *)arg;
    if (!ca)
        return;
    const vxfec_bench_cu_syms_t *s = ca->syms;
    if (s && s->ready) {
        void *pctx = NULL, *popped = NULL;
        bool retained = (s->pctx_retain(&pctx, ca->dev) == 0);
        bool pushed = retained && (s->ctx_push(pctx) == 0);
        if (ca->dptr && ca->aligned_sz) {
            (void)s->mem_unmap(ca->dptr, ca->aligned_sz);
            (void)s->mem_addr_free(ca->dptr, ca->aligned_sz);
        }
        if (ca->alloc_h)
            (void)s->mem_release(ca->alloc_h);
        if (pushed)
            (void)s->ctx_pop(&popped);
        if (retained)
            (void)s->pctx_release(ca->dev);
    }
    free(ca);
}

static inline bool vxfec_bench_can_alloc_gdr_buf(vxfec_vk_ctx *ctx)
{
    if (!ctx)
        return false;
    if (vxfec_vk_has_dmabuf_export(ctx))
        return true;
    uint8_t vk_uuid[16];
    const vxfec_bench_cu_syms_t *s = vxfec_bench_cu_syms();
    if (!s || !vxfec_vk_device_uuid(ctx, vk_uuid))
        return false;
    int ndev = 0;
    if (s->dev_get_count(&ndev) != 0 || ndev <= 0)
        return false;
    for (int i = 0; i < ndev; i++) {
        int d = 0;
        uint8_t u[16] = {0};
        if (s->dev_get(&d, i) == 0 && s->dev_get_uuid(u, d) == 0 && memcmp(u, vk_uuid, 16) == 0)
            return true;
    }
    return false;
}

static inline vxfec_vk_buf *vxfec_bench_alloc_gdr_buf(vxfec_vk_ctx *ctx, size_t bytes, int *out_dmabuf_fd)
{
    if (!ctx || !out_dmabuf_fd || bytes == 0u)
        return NULL;
    *out_dmabuf_fd = -1;

    /* 1. Prefer native Vulkan VK_EXT_external_memory_dma_buf when supported by the ICD */
    if (vxfec_vk_has_dmabuf_export(ctx)) {
        vxfec_vk_buf *vb = vxfec_vk_buf_new_exportable(ctx, bytes, true, out_dmabuf_fd);
        if (vb && *out_dmabuf_fd >= 0)
            return vb;
        if (vb)
            vxfec_vk_buf_free(ctx, vb);
    }

    /* 2. Optional benchmark fallback for headless GPUs without VK_EXT_external_memory_dma_buf */
    uint8_t vk_uuid[16];
    const vxfec_bench_cu_syms_t *s = vxfec_bench_cu_syms();
    if (!s || !vxfec_vk_device_uuid(ctx, vk_uuid))
        return NULL;

    int ndev = 0, cu_dev = -1;
    if (s->dev_get_count(&ndev) != 0 || ndev <= 0)
        return NULL;
    for (int i = 0; i < ndev; i++) {
        int d = 0;
        uint8_t u[16] = {0};
        if (s->dev_get(&d, i) == 0 && s->dev_get_uuid(u, d) == 0 && memcmp(u, vk_uuid, 16) == 0) {
            cu_dev = d;
            break;
        }
    }
    if (cu_dev < 0)
        return NULL;

    void *pctx = NULL, *popped = NULL;
    if (s->pctx_retain(&pctx, cu_dev) != 0)
        return NULL;
    if (s->ctx_push(pctx) != 0) {
        (void)s->pctx_release(cu_dev);
        return NULL;
    }

    vxfec_bench_cu_prop_t prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = 1;                   /* CU_MEM_ALLOCATION_TYPE_PINNED */
    prop.requested_handle_types = 1; /* CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR */
    prop.location.type = 1;          /* CU_MEM_LOCATION_TYPE_DEVICE */
    prop.location.id = cu_dev;
    prop.alloc_flags.gdr_capable = 1;

    size_t gran = 0;
    if (s->mem_get_gran(&gran, &prop, 0) != 0 || gran == 0u) {
        prop.alloc_flags.gdr_capable = 0;
        if (s->mem_get_gran(&gran, &prop, 0) != 0 || gran == 0u) {
            (void)s->ctx_pop(&popped);
            (void)s->pctx_release(cu_dev);
            return NULL;
        }
    }

    size_t aligned_sz = ((bytes + gran - 1u) / gran) * gran;
    unsigned long long alloc_h = 0, dptr = 0;
    if (s->mem_create(&alloc_h, aligned_sz, &prop, 0ULL) != 0 && prop.alloc_flags.gdr_capable) {
        prop.alloc_flags.gdr_capable = 0;
        (void)s->mem_create(&alloc_h, aligned_sz, &prop, 0ULL);
    }
    if (!alloc_h || s->mem_addr_reserve(&dptr, aligned_sz, gran, 0ULL, 0ULL) != 0) {
        if (alloc_h)
            (void)s->mem_release(alloc_h);
        (void)s->ctx_pop(&popped);
        (void)s->pctx_release(cu_dev);
        return NULL;
    }
    vxfec_bench_cu_access_t adesc = {.location = {.type = 1, .id = cu_dev}, .flags = 3};
    if (s->mem_map(dptr, aligned_sz, 0ULL, alloc_h, 0ULL) != 0 || s->mem_set_access(dptr, aligned_sz, &adesc, 1u) != 0) {
        (void)s->mem_unmap(dptr, aligned_sz);
        (void)s->mem_addr_free(dptr, aligned_sz);
        (void)s->mem_release(alloc_h);
        (void)s->ctx_pop(&popped);
        (void)s->pctx_release(cu_dev);
        return NULL;
    }

    int dmabuf_fd = -1, opaque_fd = -1;
    if (s->mem_get_range_handle(&dmabuf_fd, dptr, aligned_sz, 1, 0ULL) != 0 || dmabuf_fd < 0)
        (void)s->mem_get_range_handle(&dmabuf_fd, dptr, aligned_sz, 1, 1ULL);
    if (dmabuf_fd < 0 || s->mem_export_fd(&opaque_fd, alloc_h, 1, 0ULL) != 0 || opaque_fd < 0) {
        if (dmabuf_fd >= 0)
            close(dmabuf_fd);
        (void)s->mem_unmap(dptr, aligned_sz);
        (void)s->mem_addr_free(dptr, aligned_sz);
        (void)s->mem_release(alloc_h);
        (void)s->ctx_pop(&popped);
        (void)s->pctx_release(cu_dev);
        return NULL;
    }
    (void)s->ctx_pop(&popped);
    (void)s->pctx_release(cu_dev);

    vxfec_bench_cu_alloc_t *ca = (vxfec_bench_cu_alloc_t *)calloc(1, sizeof(*ca));
    if (!ca) {
        close(opaque_fd);
        close(dmabuf_fd);
        vxfec_bench_cu_alloc_t tmp = {.syms = s, .dev = cu_dev, .dptr = dptr, .alloc_h = alloc_h, .aligned_sz = aligned_sz};
        if (s->pctx_retain(&pctx, tmp.dev) == 0) {
            if (s->ctx_push(pctx) == 0) {
                (void)s->mem_unmap(tmp.dptr, tmp.aligned_sz);
                (void)s->mem_addr_free(tmp.dptr, tmp.aligned_sz);
                (void)s->mem_release(tmp.alloc_h);
                (void)s->ctx_pop(&popped);
            }
            (void)s->pctx_release(tmp.dev);
        }
        return NULL;
    }
    ca->syms = s;
    ca->dev = cu_dev;
    ca->dptr = dptr;
    ca->alloc_h = alloc_h;
    ca->aligned_sz = aligned_sz;

    vxfec_vk_buf *vb = vxfec_vk_buf_import_opaque_fd(ctx, opaque_fd, aligned_sz, bytes, vxfec_bench_free_cu_alloc, ca);
    if (!vb) {
        /* If vkAllocateMemory did not take ownership of opaque_fd, close may EBADF harmlessly or close it */
        close(opaque_fd);
        close(dmabuf_fd);
        vxfec_bench_free_cu_alloc(ca);
        return NULL;
    }
    *out_dmabuf_fd = dmabuf_fd;
    return vb;
}
#endif

#endif /* VXFEC_BENCH_COMMON_H */
