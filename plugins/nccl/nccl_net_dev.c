#define _GNU_SOURCE
#include "nccl_net_internal.h"
#include <dlfcn.h>

ncclDebugLogger_t vxf_logger = NULL;
vxfec_nic_dev_t vxf_devs[VXFEC_MAX_DEVS];
atomic_uint vxf_num_devs = 1u;
atomic_bool vxf_cfg_strict_pix_mr = false;
_Atomic uint64_t vxf_cfg_cwnd_kb = 0ULL;
_Atomic uint64_t vxf_cfg_max_rate_mbps = 0ULL;
_Atomic int64_t vxf_cfg_target_qdelay_us = -1LL;
atomic_uint vxf_leaked_ibv_eps = 0u;

atomic_uint vxf_cfg_ports = VXFEC_DEF_PORTS;
atomic_uint vxf_cfg_mtu_sym = 0u;
atomic_uint vxf_cfg_bind_ip_be = 0u;
atomic_uint vxf_cfg_busy_poll_us = 0u;
atomic_uint vxf_cfg_slice_min_bytes = VXFEC_DEF_SLICE_MIN_BYTES;
atomic_uint vxf_cfg_rail_workers = 1u;
atomic_uint vxf_granted_rcvbuf = (4u * 1024u * 1024u);

static pthread_mutex_t g_init_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_vk_mtx = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    int out_dmabuf_fd;
    uint64_t out_dmabuf_off;
    size_t out_dmabuf_cap;
    uint64_t out_allocation_id;
    vxfec_ibv_mr_t *out_ibv_mrs[VXFEC_MAX_DEVS];
    vxfec_ibv_dev_ref_t *out_ibv_refs[VXFEC_MAX_DEVS];
} vxf_gpu_out_export_t;

typedef struct {
    vxfec_vk_ctx *ctx;
    bool is_discrete;
    bool has_timeline;
    bool is_hw_gpu;
    uint64_t max_ssbo_bytes;
    struct {
        vxfec_vk_buf *in_buf;
        size_t in_cap;
        vxfec_vk_buf *out_buf;
        size_t out_cap;
        vxfec_vk_buf *out_dmabuf;
        vxf_gpu_out_export_t out_export;
        bool in_use;
    } pool[VXFEC_VK_ASYNC_POOL_SLOTS];
} vxf_vk_slot_t;

static vxf_vk_slot_t g_vk_devs[VXFEC_MAX_VK_DEVS];
static uint32_t g_num_vk_devs = 0u;
static bool g_vk_probed = false;
static atomic_bool g_vk_caps_ready = false;
static int g_vk_mode = 1;
static bool g_vk_force_all = false;
static atomic_bool g_gpurt_active = false;
static vxf_gpurt_caps_t g_gpurt_caps;
static vxf_gpu_out_export_t g_gpurt_ibv_pool[VXFEC_GPURT_MAX_POOL];
static unsigned g_gpu_active_jobs = 0u;
static vxfec_gpu_job_t *g_gpu_retired = NULL;

__attribute__((visibility("default"))) uint32_t vxfec_nccl_get_inproc_vk_devs(void)
{
    pthread_mutex_lock(&g_vk_mtx);
    uint32_t n = g_num_vk_devs;
    pthread_mutex_unlock(&g_vk_mtx);
    return n;
}

static bool is_cuda_mps_partitioned(void);

__attribute__((visibility("default"))) const char *vxfec_nccl_get_gpu_backend(void)
{
    pthread_mutex_lock(&g_vk_mtx);
    const char *res = "cpu";
    if (atomic_load_explicit(&g_gpurt_active, memory_order_acquire))
        res = (g_gpurt_caps.kind == VXF_GPURT_HIP) ? "hip" : ((g_gpurt_caps.kind == VXF_GPURT_CUDA) ? "cuda" : "cpu");
    else if (g_vk_mode != 0 && g_num_vk_devs > 0u)
        res = "vulkan";
    pthread_mutex_unlock(&g_vk_mtx);
    return res;
}

__attribute__((visibility("default"))) int vxfec_nccl_gpu_sm_partitioned(void)
{
    pthread_mutex_lock(&g_vk_mtx);
    vxf_gpurt_caps_t caps;
    bool part = atomic_load_explicit(&g_gpurt_active, memory_order_acquire) ? (vxf_gpurt_get_caps(&caps) && caps.used_sm_partition)
                                                                            : is_cuda_mps_partitioned();
    pthread_mutex_unlock(&g_vk_mtx);
    return part ? 1 : 0;
}

__attribute__((visibility("default"))) uint32_t vxfec_nccl_gpu_reserved_units(void)
{
    vxf_gpurt_caps_t caps;
    return vxf_gpurt_get_caps(&caps) ? caps.reserved_units : 0u;
}

__attribute__((visibility("default"))) int vxfec_nccl_gpu_high_priority_stream(void)
{
    vxf_gpurt_caps_t caps;
    return vxf_gpurt_get_caps(&caps) && caps.uses_high_priority_stream;
}

void vxf_ibv_ref_retain(vxfec_ibv_dev_ref_t *ref)
{
    if (ref)
        atomic_fetch_add_explicit(&ref->refcnt, 1u, memory_order_relaxed);
}

vxfec_ibv_dev_ref_t *vxf_acquire_dev_ibv_ref(int dev_idx, uint32_t *out_mtu)
{
    vxfec_ibv_dev_ref_t *iref = NULL;
    pthread_mutex_lock(&g_init_mtx);
    if (dev_idx >= 0 && (uint32_t)dev_idx < vxf_num_devs && vxf_devs[dev_idx].ibv_ref) {
        iref = vxf_devs[dev_idx].ibv_ref;
        vxf_ibv_ref_retain(iref);
        if (out_mtu)
            *out_mtu = vxf_devs[dev_idx].mtu;
    }
    pthread_mutex_unlock(&g_init_mtx);
    return iref;
}

void vxf_ibv_ref_release(vxfec_ibv_dev_ref_t *ref)
{
    if (!ref)
        return;
    if (atomic_fetch_sub_explicit(&ref->refcnt, 1u, memory_order_acq_rel) == 1u) {
        if (ref->ctx)
            vxfec_ibv_ctx_close(ref->ctx);
        free(ref);
    }
}

static void clear_gpu_out_export_locked(vxf_gpu_out_export_t *out)
{
    for (uint32_t d = 0; d < VXFEC_MAX_DEVS; d++) {
        if (out->out_ibv_mrs[d])
            vxfec_ibv_mr_dereg(out->out_ibv_mrs[d]);
        vxf_ibv_ref_release(out->out_ibv_refs[d]);
    }
    /* capacity is set only after we take ownership of an exported FD, including FD zero */
    if (out->out_dmabuf_cap != 0u)
        close(out->out_dmabuf_fd);
    memset(out, 0, sizeof(*out));
}

static vxfec_ibv_dev_ref_t *ibv_ref_create(uint32_t ibv_idx)
{
    vxfec_ibv_ctx_t *ictx = vxfec_ibv_ctx_open(ibv_idx);
    if (!ictx)
        return NULL;
    vxfec_ibv_dev_ref_t *ref = (vxfec_ibv_dev_ref_t *)calloc(1, sizeof(*ref));
    if (!ref) {
        vxfec_ibv_ctx_close(ictx);
        return NULL;
    }
    atomic_init(&ref->refcnt, 1u);
    ref->ctx = ictx;
    return ref;
}

static int register_vk_slot_locked(vxfec_vk_ctx *ctx)
{
    if (!ctx || g_num_vk_devs >= VXFEC_MAX_VK_DEVS)
        return -1;
    uint32_t idx = g_num_vk_devs++;
    vxf_vk_slot_t *s = &g_vk_devs[idx];
    memset(s, 0, sizeof(*s));
    s->ctx = ctx;
    const char *dname = vxfec_vk_device_name(ctx);
    s->is_hw_gpu = (dname != NULL && strstr(dname, "llvmpipe") == NULL && strstr(dname, "lavapipe") == NULL);
    s->is_discrete = vxfec_vk_is_discrete(ctx);
    s->has_timeline = vxfec_vk_has_timeline(ctx);
    s->max_ssbo_bytes = vxfec_vk_max_storage_buffer_bytes(ctx);
    return (int)idx;
}

static bool is_nvidia_nccl_resident(void)
{
    const char *sim = getenv("VXFEC_TEST_FAKE_NVIDIA_NCCL");
    if (sim && atoi(sim) > 0)
        return true;
    void *h = dlopen("libnccl.so.2", RTLD_LAZY | RTLD_NOLOAD);
    if (h) {
        dlclose(h);
        return true;
    }
    return false;
}

static bool is_cuda_mps_partitioned(void)
{
    const char *pct = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
    if (pct && *pct) {
        int v = atoi(pct);
        if (v > 0 && v < 100)
            return true;
    }
    const char *pipe_dir = getenv("CUDA_MPS_PIPE_DIRECTORY");
    return (pipe_dir && *pipe_dir && access(pipe_dir, F_OK) == 0);
}

static void gpu_reap_locked(void);

static void configure_gpu_backends_locked(void)
{
    const char *be = getenv("VXFEC_NCCL_GPU_BACKEND");
    const char *en = getenv("VXFEC_NCCL_VULKAN");
    const char *gf = getenv("VXFEC_NCCL_GPU_FORCE");
    bool explicit_vk_off = (en && *en && atoi(en) == 0);
    bool explicit_cpu = (be && (strcmp(be, "cpu") == 0 || strcmp(be, "none") == 0 || strcmp(be, "0") == 0));
    bool explicit_vk = (be && (strcmp(be, "vk") == 0 || strcmp(be, "vulkan") == 0));
    bool explicit_gpurt = (be && (strcmp(be, "hip") == 0 || strcmp(be, "cuda") == 0 || strcmp(be, "native") == 0));
    bool want_gpurt = !explicit_cpu && !explicit_vk && (explicit_gpurt || (!explicit_vk_off && vxf_gpurt_should_auto_enable()));

    g_vk_force_all = ((en && *en && atoi(en) >= 2) || (gf && *gf && atoi(gf) > 0));

    if (!want_gpurt && atomic_load_explicit(&g_gpurt_active, memory_order_acquire)) {
        if (!vxf_gpurt_wait_idle())
            return;
        gpu_reap_locked();
        if (g_gpu_active_jobs != 0u)
            return;
        for (uint32_t i = 0; i < VXFEC_GPURT_MAX_POOL; i++)
            clear_gpu_out_export_locked(&g_gpurt_ibv_pool[i]);
        vxf_gpurt_shutdown();
        atomic_store_explicit(&g_gpurt_active, false, memory_order_release);
    }

    if (explicit_cpu) {
        g_vk_mode = 0;
        g_vk_probed = true;
        atomic_store_explicit(&g_vk_caps_ready, true, memory_order_release);
        return;
    }

    if (want_gpurt) {
        if (!atomic_load_explicit(&g_gpurt_active, memory_order_acquire)) {
            bool ok = vxf_gpurt_init(&g_gpurt_caps);
            if (ok)
                vxf_register_gpu_atexit(true);
            atomic_store_explicit(&g_gpurt_active, ok, memory_order_release);
        }
        if (atomic_load_explicit(&g_gpurt_active, memory_order_acquire) || explicit_gpurt) {
            g_vk_mode = 0;
            g_vk_probed = true;
            atomic_store_explicit(&g_vk_caps_ready, true, memory_order_release);
            return;
        }
    }

    if (en && *en)
        g_vk_mode = atoi(en);
    else if (explicit_vk || !is_nvidia_nccl_resident() || is_cuda_mps_partitioned())
        g_vk_mode = 1;
    else
        g_vk_mode = 0;

    if (g_vk_mode != 0 && g_num_vk_devs == 0u) {
        vxfec_vk_ctx *ctx = vxfec_vk_ctx_new();
        if (ctx) {
            (void)register_vk_slot_locked(ctx);
            vxf_register_gpu_atexit(false);
        }
    }
    g_vk_probed = true;
    atomic_store_explicit(&g_vk_caps_ready, true, memory_order_release);
}

static int get_primary_vk_slot_locked(void)
{
    if (!g_vk_probed)
        configure_gpu_backends_locked();
    if (atomic_load_explicit(&g_gpurt_active, memory_order_acquire))
        return 0;
    return (g_vk_mode != 0 && g_num_vk_devs > 0u) ? 0 : -1;
}

static int find_vk_slot_by_ctx_locked(const vxfec_vk_ctx *ctx)
{
    if (!ctx)
        return -1;
    for (uint32_t i = 0; i < g_num_vk_devs; i++) {
        if (g_vk_devs[i].ctx == ctx)
            return (int)i;
    }
    return -1;
}

/* match or open the Vulkan physical device that owns a given DMA-BUF FD */
static vxfec_vk_ctx *get_vk_ctx_for_dmabuf_locked(int dmabuf_fd)
{
    (void)get_primary_vk_slot_locked();
    if (g_vk_mode == 0)
        return NULL;

    for (uint32_t i = 0; i < g_num_vk_devs; i++) {
        if (g_vk_devs[i].ctx && vxfec_vk_has_dmabuf(g_vk_devs[i].ctx) &&
            vxfec_vk_ctx_supports_dmabuf_fd(g_vk_devs[i].ctx, dmabuf_fd))
            return g_vk_devs[i].ctx;
    }
    if (g_num_vk_devs < VXFEC_MAX_VK_DEVS) {
        vxfec_vk_ctx *matched = vxfec_vk_ctx_new_for_dmabuf(dmabuf_fd, NULL);
        if (matched) {
            if (register_vk_slot_locked(matched) >= 0) {
                vxf_register_gpu_atexit(false);
                return matched;
            }
            vxfec_vk_ctx_free(matched);
        }
    }
    if (g_num_vk_devs > 0u && vxfec_vk_has_dmabuf(g_vk_devs[0].ctx))
        return g_vk_devs[0].ctx;
    return NULL;
}

static void ensure_vk_caps_probed(void)
{
    if (!atomic_load_explicit(&g_vk_caps_ready, memory_order_acquire)) {
        pthread_mutex_lock(&g_vk_mtx);
        (void)get_primary_vk_slot_locked();
        pthread_mutex_unlock(&g_vk_mtx);
    }
}

void vxf_mr_retain(vxfec_mr_t *mr)
{
    if (mr)
        atomic_fetch_add_explicit(&mr->refcnt, 1u, memory_order_relaxed);
}

void vxf_mr_release(vxfec_mr_t *mr)
{
    if (!mr)
        return;
    if (atomic_fetch_sub_explicit(&mr->refcnt, 1u, memory_order_acq_rel) == 1u) {
        for (uint32_t i = 0; i < VXFEC_MAX_DEVS; i++) {
            if (mr->ibv_mrs[i])
                vxfec_ibv_mr_dereg(mr->ibv_mrs[i]);
            if (mr->ibv_refs[i])
                vxf_ibv_ref_release(mr->ibv_refs[i]);
        }
        if (mr->vk_buf) {
            pthread_mutex_lock(&g_vk_mtx);
            if (mr->vk_ctx && find_vk_slot_by_ctx_locked(mr->vk_ctx) >= 0)
                vxfec_vk_buf_free(mr->vk_ctx, mr->vk_buf);
            pthread_mutex_unlock(&g_vk_mtx);
        }
        free(mr);
    }
}

bool vxf_mr_can_gpu_rw(const vxfec_mr_t *mr)
{
    if (!mr)
        return false;
    if (mr->gpu_dev_ptr != NULL && atomic_load_explicit(&g_gpurt_active, memory_order_acquire))
        return true;
    if (mr->vk_buf && mr->vk_ctx)
        return true;
    for (uint32_t d = 0; d < VXFEC_MAX_DEVS; d++) {
        if (mr->ibv_mrs[d] && mr->ibv_can_rwrite[d] && mr->ibv_refs[d] && mr->ibv_refs[d]->ctx)
            return true;
    }
    return false;
}

bool vxf_mr_vk_read(vxfec_mr_t *mr, size_t byte_offset, void *dst, size_t bytes)
{
    if (!mr)
        return false;
    if (mr->gpu_dev_ptr != NULL && atomic_load_explicit(&g_gpurt_active, memory_order_acquire) &&
        vxf_gpurt_memcpy_dtoh(dst, (const uint8_t *)mr->gpu_dev_ptr + byte_offset, bytes))
        return true;
    if (mr->vk_buf && mr->vk_ctx) {
        pthread_mutex_lock(&g_vk_mtx);
        bool ok = (find_vk_slot_by_ctx_locked(mr->vk_ctx) >= 0) &&
                  vxfec_vk_buf_read_offset(mr->vk_ctx, mr->vk_buf, byte_offset, dst, bytes);
        pthread_mutex_unlock(&g_vk_mtx);
        if (ok)
            return true;
    }
    for (uint32_t d = 0; d < VXFEC_MAX_DEVS; d++) {
        if (mr->ibv_mrs[d] && mr->ibv_refs[d] && mr->ibv_refs[d]->ctx) {
            if (vxfec_ibv_mr_read_to_host(mr->ibv_refs[d]->ctx, mr->ibv_mrs[d], byte_offset, dst, bytes))
                return true;
        }
    }
    return false;
}

bool vxf_mr_vk_write(vxfec_mr_t *mr, size_t byte_offset, const void *src, size_t bytes)
{
    if (!mr)
        return false;
    if (mr->gpu_dev_ptr != NULL && atomic_load_explicit(&g_gpurt_active, memory_order_acquire) &&
        vxf_gpurt_memcpy_htod((uint8_t *)mr->gpu_dev_ptr + byte_offset, src, bytes))
        return true;
    if (mr->vk_buf && mr->vk_ctx) {
        pthread_mutex_lock(&g_vk_mtx);
        bool ok = (find_vk_slot_by_ctx_locked(mr->vk_ctx) >= 0) &&
                  vxfec_vk_buf_write_offset(mr->vk_ctx, mr->vk_buf, byte_offset, src, bytes);
        pthread_mutex_unlock(&g_vk_mtx);
        if (ok)
            return true;
    }
    for (uint32_t d = 0; d < VXFEC_MAX_DEVS; d++) {
        if (mr->ibv_mrs[d] && mr->ibv_can_rwrite[d] && mr->ibv_refs[d] && mr->ibv_refs[d]->ctx) {
            if (vxfec_ibv_mr_write_from_host(mr->ibv_refs[d]->ctx, mr->ibv_mrs[d], byte_offset, src, bytes))
                return true;
        }
    }
    return false;
}

static bool job_finished_locked(vxfec_gpu_job_t *job, bool wait)
{
    if (!job->ticket)
        return true;
    if (job->gpurt_kind) {
        vxf_gpurt_poll_status_t status = vxf_gpurt_poll(job->ticket, wait);
        return status == VXF_GPURT_POLL_COMPLETE || status == VXF_GPURT_POLL_FAILED;
    }
    vxfec_vk_ctx *vk = (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) ? g_vk_devs[job->vk_idx].ctx : NULL;
    if (!vk)
        return !job->user_jobs;
    if (wait)
        return vxfec_vk_wait(vk, job->ticket, UINT64_MAX) || vxfec_vk_wait_idle(vk);
    int status = vxfec_vk_poll_status(vk, job->ticket);
    if (status < 0 && job->user_jobs)
        return vxfec_vk_wait_idle(vk);
    return status == 1 && vxfec_vk_wait(vk, job->ticket, 0u);
}

static void gpu_job_destroy_locked(vxfec_gpu_job_t *job)
{
    if (job->gpurt_kind) {
        if (job->pool_idx >= 0)
            vxf_gpurt_pool_release(job->pool_idx);
    } else {
        vxf_vk_slot_t *vs = (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) ? &g_vk_devs[job->vk_idx] : NULL;
        vxfec_vk_ctx *vk = vs ? vs->ctx : NULL;
        vxfec_rq_sched_free(vk, job->rq_sched);
        if (vs && job->pool_idx >= 0 && job->pool_idx < (int)VXFEC_VK_ASYNC_POOL_SLOTS) {
            vs->pool[job->pool_idx].in_use = false;
        } else if (vk) {
            vxfec_vk_buf_free(vk, job->in_buf);
            vxfec_vk_buf_free(vk, job->out_buf);
        }
    }
    free(job->rq_saved_ops);
    free(job->rs8_erased_k);
    free(job);
    if (g_gpu_active_jobs > 0u)
        g_gpu_active_jobs--;
}

static void gpu_reap_locked(void)
{
    vxfec_gpu_job_t **link = &g_gpu_retired;
    while (*link) {
        vxfec_gpu_job_t *job = *link;
        if (job_finished_locked(job, false)) {
            *link = job->next;
            gpu_job_destroy_locked(job);
        } else {
            link = &job->next;
        }
    }
}

/* The comm owns this list. A wire slot may disappear on DONE or ABORT while
 * its GPU submission still reads or writes the user buffer. */
void vxf_gpu_track_direct_job(vxfec_gpu_job_t *job, vxfec_gpu_job_t **head)
{
    job->user_jobs = head;
    job->user_next = *head;
    *head = job;
}

static void finish_direct_jobs_locked(vxfec_gpu_job_t **link, bool retired_only, bool wait)
{
    while (*link) {
        vxfec_gpu_job_t *job = *link;
        if ((!retired_only || job->retired) && job_finished_locked(job, wait)) {
            *link = job->user_next;
            job->ticket = 0u;
            job->user_jobs = NULL;
            job->user_next = NULL;
            if (job->retired)
                gpu_job_destroy_locked(job);
        } else {
            link = &job->user_next;
        }
    }
}

bool vxf_gpu_finish_direct_jobs(vxfec_gpu_job_t **head, bool wait)
{
    if (!*head)
        return true;
    pthread_mutex_lock(&g_vk_mtx);
    finish_direct_jobs_locked(head, false, wait);
    pthread_mutex_unlock(&g_vk_mtx);
    return !*head;
}

int vxf_gpu_job_poll(vxfec_gpu_job_t *job)
{
    if (!job)
        return -1;
    if (!job->ticket)
        return 1;
    if (job->gpurt_kind) {
        vxf_gpurt_poll_status_t status = vxf_gpurt_poll(job->ticket, false);
        atomic_fetch_add_explicit(&vxf_st_vk_async_polls, 1u, memory_order_relaxed);
        if (status == VXF_GPURT_POLL_COMPLETE || status == VXF_GPURT_POLL_FAILED)
            job->ticket = 0u;
        if (status == VXF_GPURT_POLL_COMPLETE)
            atomic_fetch_add_explicit(&vxf_st_vk_offload_calls, 1u, memory_order_relaxed);
        return status == VXF_GPURT_POLL_UNKNOWN ? VXF_GPURT_POLL_UNSAFE : status;
    }
    pthread_mutex_lock(&g_vk_mtx);
    vxfec_vk_ctx *vk = (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) ? g_vk_devs[job->vk_idx].ctx : NULL;
    int status = vk ? vxfec_vk_poll_status(vk, job->ticket) : -1;
    atomic_fetch_add_explicit(&vxf_st_vk_async_polls, 1u, memory_order_relaxed);
    if (status == 1) {
        if (vxfec_vk_wait(vk, job->ticket, 0u)) {
            job->ticket = 0u;
            atomic_fetch_add_explicit(&vxf_st_vk_offload_calls, 1u, memory_order_relaxed);
        } else {
            status = -1;
        }
    }
    pthread_mutex_unlock(&g_vk_mtx);
    return status < 0 && job->user_jobs ? VXF_GPURT_POLL_UNSAFE : status;
}

void vxf_gpu_job_free(vxfec_gpu_job_t *job)
{
    if (!job)
        return;
    pthread_mutex_lock(&g_vk_mtx);
    gpu_reap_locked();
    if (job->user_jobs) {
        job->retired = true;
        finish_direct_jobs_locked(job->user_jobs, true, false);
    } else if (job_finished_locked(job, false)) {
        gpu_job_destroy_locked(job);
    } else {
        job->next = g_gpu_retired;
        g_gpu_retired = job;
    }
    pthread_mutex_unlock(&g_vk_mtx);
}

static int acquire_vk_async_pool_locked(vxf_vk_slot_t *vs, size_t in_bytes, size_t out_bytes)
{
    if (!vs || !vs->ctx)
        return -1;
    for (uint32_t i = 0; i < VXFEC_VK_ASYNC_POOL_SLOTS; i++) {
        if (vs->pool[i].in_use)
            continue;
        if (in_bytes > vs->pool[i].in_cap) {
            size_t ncap = (in_bytes < 512u * 1024u) ? (512u * 1024u) : ((in_bytes + 65535u) & ~(size_t)65535u);
            vxfec_vk_buf *nb = vxfec_vk_buf_new(vs->ctx, ncap, true);
            if (!nb)
                return -1;
            vxfec_vk_buf_free(vs->ctx, vs->pool[i].in_buf);
            vs->pool[i].in_buf = nb;
            vs->pool[i].in_cap = ncap;
        }
        if (out_bytes > vs->pool[i].out_cap) {
            size_t ncap = (out_bytes < 256u * 1024u) ? (256u * 1024u) : ((out_bytes + 65535u) & ~(size_t)65535u);
            vxfec_vk_buf *nb = vxfec_vk_buf_new(vs->ctx, ncap, true);
            if (!nb)
                return -1;
            vxfec_vk_buf_free(vs->ctx, vs->pool[i].out_buf);
            vs->pool[i].out_buf = nb;
            vs->pool[i].out_cap = ncap;
        }
        vs->pool[i].in_use = true;
        return (int)i;
    }
    return -1;
}

static int resolve_job_vk_slot_locked(const vxfec_vk_ctx *prefer_vk)
{
    if (prefer_vk) {
        int idx = find_vk_slot_by_ctx_locked(prefer_vk);
        if (idx >= 0)
            return idx;
    }
    return get_primary_vk_slot_locked();
}

static bool vxf_mr_has_ibv(const vxfec_mr_t *mr)
{
    if (!mr)
        return false;
    for (uint32_t d = 0; d < VXFEC_MAX_DEVS; d++) {
        if (mr->ibv_mrs[d] != NULL)
            return true;
    }
    return false;
}

static int acquire_gpurt_slot_locked(const vxfec_mr_t *mr, size_t in_bytes, size_t out_bytes, vxfec_gpu_job_t *job)
{
    int slot = vxf_gpurt_pool_acquire(mr ? mr->gpu_dev_ptr : NULL, in_bytes, out_bytes, &job->in_stage_ptr, &job->out_stage_ptr);
    if (slot < 0)
        return slot;
    job->gpurt_kind = g_gpurt_caps.kind;
    if (g_gpurt_ibv_pool[slot].out_dmabuf_cap != 0u && g_gpurt_ibv_pool[slot].out_allocation_id != vxf_gpurt_pool_out_id(slot))
        clear_gpu_out_export_locked(&g_gpurt_ibv_pool[slot]);
    return slot;
}

vxfec_gpu_job_t *vxf_rq_gpu_job_new(const vxfec_mr_t *prefer_mr, nanorq_core *rq, const schedule *ops, size_t input_bytes,
                                    size_t output_bytes)
{
    ensure_vk_caps_probed();
    bool gpurt = atomic_load_explicit(&g_gpurt_active, memory_order_acquire);
    if (!gpurt && (g_vk_mode == 0 || g_num_vk_devs == 0u))
        return NULL;
    vxfec_vk_ctx *prefer_vk = prefer_mr ? prefer_mr->vk_ctx : NULL;
    bool has_mr = vxf_mr_has_gpu_buf(prefer_mr);
    bool has_ibv_mr = has_mr && vxf_mr_has_ibv(prefer_mr);

    if (gpurt) {
        bool size_ok = g_vk_force_all || has_ibv_mr || (input_bytes >= (512u * 1024u) && output_bytes >= (64u * 1024u));
        if (!rq || !ops || !ops->ops.a || ops->cpidx != 2 || ops->overflowed || !size_ok)
            return NULL;
        vxfec_gpu_job_t *job = (vxfec_gpu_job_t *)calloc(1, sizeof(*job));
        sched_op *sops = (job && ops->ops.n > 0) ? (sched_op *)malloc(ops->ops.n * sizeof(sched_op)) : NULL;
        if (!job || (ops->ops.n > 0 && !sops)) {
            free(sops);
            free(job);
            return NULL;
        }
        if (ops->ops.n > 0)
            memcpy(sops, ops->ops.a, ops->ops.n * sizeof(sched_op));
        job->kind = VXFEC_GPU_JOB_RQ;
        job->vk_idx = -1;
        job->pool_idx = -1;
        job->rq_saved_ops = sops;
        job->rq_saved_sched.ops.a = sops;
        job->rq_saved_sched.ops.n = ops->ops.n;
        job->rq_saved_sched.ops.m = ops->ops.n;
        job->rq_saved_sched.cp[0] = ops->cp[0];
        job->rq_saved_sched.cp[1] = ops->cp[1];
        job->rq_saved_sched.cpidx = 2;

        pthread_mutex_lock(&g_vk_mtx);
        gpu_reap_locked();
        if (atomic_load_explicit(&g_gpurt_active, memory_order_acquire) && g_gpu_active_jobs < VXFEC_RQ_GPU_MAX_JOBS) {
            int pidx = acquire_gpurt_slot_locked(prefer_mr, input_bytes, output_bytes, job);
            if (pidx >= 0) {
                g_gpu_active_jobs++;
                job->pool_idx = pidx;
                pthread_mutex_unlock(&g_vk_mtx);
                return job;
            }
        }
        pthread_mutex_unlock(&g_vk_mtx);
        free(sops);
        free(job);
        return NULL;
    }

    /* quick lock-free precheck against primary device before packing schedule */
    const vxf_vk_slot_t *vs0 = &g_vk_devs[0];
    size_t min0 = vs0->is_discrete ? VXFEC_RQ_GPU_MIN_DISCRETE_BYTES : VXFEC_RQ_GPU_MIN_OTHER_BYTES;
    if (!prefer_vk && (!vs0->has_timeline || !(g_vk_force_all || vs0->is_hw_gpu) || !(g_vk_force_all || input_bytes >= min0) ||
                       input_bytes > vs0->max_ssbo_bytes || output_bytes > vs0->max_ssbo_bytes))
        return NULL;

    vxfec_gpu_job_t *job = (vxfec_gpu_job_t *)calloc(1, sizeof(*job));
    vxfec_rq_cpu_sched *cpu_sched = job ? vxfec_rq_cpu_sched_pack(rq, ops) : NULL;
    if (!cpu_sched) {
        free(job);
        return NULL;
    }
    job->kind = VXFEC_GPU_JOB_RQ;
    job->vk_idx = -1;
    job->pool_idx = -1;

    pthread_mutex_lock(&g_vk_mtx);
    int vidx = resolve_job_vk_slot_locked(prefer_vk);
    gpu_reap_locked();
    if (vidx >= 0) {
        vxf_vk_slot_t *vs = &g_vk_devs[vidx];
        size_t min_bytes = vs->is_discrete ? VXFEC_RQ_GPU_MIN_DISCRETE_BYTES : VXFEC_RQ_GPU_MIN_OTHER_BYTES;
        bool caps_ok = vs->has_timeline && (g_vk_force_all || vs->is_hw_gpu) &&
                       (g_vk_force_all || prefer_vk != NULL || input_bytes >= min_bytes) && input_bytes <= vs->max_ssbo_bytes &&
                       output_bytes <= vs->max_ssbo_bytes;
        if (caps_ok && g_gpu_active_jobs < VXFEC_RQ_GPU_MAX_JOBS && vxfec_vk_async_can_submit(vs->ctx)) {
            g_gpu_active_jobs++;
            job->vk_idx = vidx;
            int pidx = acquire_vk_async_pool_locked(vs, input_bytes, output_bytes);
            if (pidx >= 0) {
                job->pool_idx = pidx;
                job->in_buf = vs->pool[pidx].in_buf;
                job->out_buf = vs->pool[pidx].out_buf;
            } else {
                job->in_buf = vxfec_vk_buf_new(vs->ctx, input_bytes, true);
                job->out_buf = vxfec_vk_buf_new(vs->ctx, output_bytes, true);
            }
            if (job->in_buf && job->out_buf)
                job->rq_sched = vxfec_rq_sched_from_cpu_staged(vs->ctx, &cpu_sched);
            if (!job->rq_sched) {
                gpu_job_destroy_locked(job);
                job = NULL;
            }
        } else {
            free(job);
            job = NULL;
        }
    } else {
        free(job);
        job = NULL;
    }
    pthread_mutex_unlock(&g_vk_mtx);
    vxfec_rq_cpu_sched_free(cpu_sched);
    return job;
}

static bool ensure_export_ibv_mr_locked(vxf_gpu_out_export_t *out, int dev_idx, vxfec_ibv_dev_ref_t *iref, uint64_t iova,
                                        vxfec_gpu_job_t *job)
{
    if (out->out_dmabuf_cap == 0u)
        return false;
    if (!out->out_ibv_mrs[dev_idx] || out->out_ibv_refs[dev_idx] != iref) {
        if (out->out_ibv_mrs[dev_idx]) {
            vxfec_ibv_mr_dereg(out->out_ibv_mrs[dev_idx]);
            out->out_ibv_mrs[dev_idx] = NULL;
        }
        if (out->out_ibv_refs[dev_idx]) {
            vxf_ibv_ref_release(out->out_ibv_refs[dev_idx]);
            out->out_ibv_refs[dev_idx] = NULL;
        }
        vxfec_ibv_mr_t *imr =
            vxfec_ibv_mr_reg_dmabuf(iref->ctx, out->out_dmabuf_off, out->out_dmabuf_cap, iova, out->out_dmabuf_fd, false);
        if (imr) {
            out->out_ibv_mrs[dev_idx] = imr;
            out->out_ibv_refs[dev_idx] = iref;
            vxf_ibv_ref_retain(iref);
        }
    }
    vxfec_ibv_mr_t *imr = out->out_ibv_mrs[dev_idx];
    if (!imr || vxfec_ibv_mr_lkey(imr) == 0u)
        return false;
    job->has_out_dmabuf = true;
    job->out_ibv_lkey = vxfec_ibv_mr_lkey(imr);
    job->out_ibv_iova = vxfec_ibv_mr_iova(imr);
    return true;
}

bool vxf_gpu_job_prepare_ibv_out(vxfec_gpu_job_t *job, int dev_idx, size_t out_bytes)
{
    if (!job || job->pool_idx < 0 || dev_idx < 0 || (uint32_t)dev_idx >= VXFEC_MAX_DEVS || out_bytes == 0u)
        return false;
    uint32_t max_slots = job->gpurt_kind ? VXFEC_GPURT_MAX_POOL : VXFEC_VK_ASYNC_POOL_SLOTS;
    if ((uint32_t)job->pool_idx >= max_slots)
        return false;

    pthread_mutex_lock(&g_init_mtx);
    vxfec_ibv_dev_ref_t *iref = ((uint32_t)dev_idx < vxf_num_devs) ? vxf_devs[dev_idx].ibv_ref : NULL;
    if (iref)
        vxf_ibv_ref_retain(iref);
    pthread_mutex_unlock(&g_init_mtx);
    if (!iref || !iref->ctx) {
        vxf_ibv_ref_release(iref);
        return false;
    }

    bool ok = false;
    pthread_mutex_lock(&g_vk_mtx);
    if (job->gpurt_kind) {
        uint32_t pidx = (uint32_t)job->pool_idx;
        uint64_t allocation_id = vxf_gpurt_pool_out_id(job->pool_idx);
        vxf_gpu_out_export_t *out = &g_gpurt_ibv_pool[pidx];
        if (atomic_load_explicit(&g_gpurt_active, memory_order_acquire) && allocation_id != 0u) {
            if (out->out_allocation_id != allocation_id || out_bytes > out->out_dmabuf_cap) {
                clear_gpu_out_export_locked(out);
                uint64_t noff = 0ULL;
                size_t ncap = 0u;
                int fd = vxf_gpurt_pool_export_out_dmabuf((int)pidx, out_bytes, &noff, &ncap);
                if (fd >= 0 && ncap >= out_bytes) {
                    out->out_dmabuf_fd = fd;
                    out->out_dmabuf_off = noff;
                    out->out_dmabuf_cap = ncap;
                    out->out_allocation_id = allocation_id;
                } else if (fd >= 0) {
                    close(fd);
                }
            }
            uint64_t iova = 0x730000000000ULL + (uint64_t)pidx * 0x40000000ULL;
            ok = ensure_export_ibv_mr_locked(out, dev_idx, iref, iova, job);
        }
    } else if (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) {
        vxf_vk_slot_t *vs = &g_vk_devs[job->vk_idx];
        uint32_t pidx = (uint32_t)job->pool_idx;
        vxf_gpu_out_export_t *out = &vs->pool[pidx].out_export;
        if (vs->ctx && vxfec_vk_has_dmabuf_export(vs->ctx)) {
            if (!vs->pool[pidx].out_dmabuf || out_bytes > out->out_dmabuf_cap) {
                clear_gpu_out_export_locked(out);
                vxfec_vk_buf_free(vs->ctx, vs->pool[pidx].out_dmabuf);
                vs->pool[pidx].out_dmabuf = NULL;

                size_t ncap = (out_bytes < 256u * 1024u) ? (256u * 1024u) : ((out_bytes + 65535u) & ~(size_t)65535u);
                int fd = -1;
                vxfec_vk_buf *nb = vxfec_vk_buf_new_exportable(vs->ctx, ncap, true, &fd);
                if (nb) {
                    if (vxfec_vk_buf_release_external(vs->ctx, nb)) {
                        vs->pool[pidx].out_dmabuf = nb;
                        out->out_dmabuf_cap = ncap;
                        out->out_dmabuf_fd = fd;
                    } else {
                        close(fd);
                        vxfec_vk_buf_free(vs->ctx, nb);
                    }
                }
            }
            if (vs->pool[pidx].out_dmabuf) {
                uint64_t iova = 0x700000000000ULL + ((uint64_t)job->vk_idx * VXFEC_VK_ASYNC_POOL_SLOTS + pidx) * 0x40000000ULL;
                ok = ensure_export_ibv_mr_locked(out, dev_idx, iref, iova, job);
                if (ok)
                    job->out_dmabuf = vs->pool[pidx].out_dmabuf;
            }
        }
    }
    pthread_mutex_unlock(&g_vk_mtx);
    vxf_ibv_ref_release(iref);
    return ok;
}

bool vxf_rq_gpu_job_submit(vxfec_gpu_job_t *job, nanorq_core *rq, uint32_t stride, const uint32_t *esis, uint32_t count)
{
    if (!job)
        return false;
    if (job->gpurt_kind) {
        job->ticket = vxf_gpurt_rq_submit(job->pool_idx, rq, &job->rq_saved_sched, stride, esis, count);
        return job->ticket != 0u;
    }
    pthread_mutex_lock(&g_vk_mtx);
    vxfec_vk_ctx *vk = (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) ? g_vk_devs[job->vk_idx].ctx : NULL;
    job->ticket =
        vk ? vxfec_rq_run_and_mix_submit(vk, rq, job->rq_sched, job->in_buf, stride, 1u, esis, count, job->out_buf, true, true)
           : 0u;
    pthread_mutex_unlock(&g_vk_mtx);
    return job->ticket != 0u;
}

bool vxf_rq_gpu_job_submit_vram_enc(vxfec_gpu_job_t *job, nanorq_core *rq, vxfec_mr_t *src_mr, size_t src_offset,
                                    size_t valid_bytes, uint32_t sym_T, uint32_t stride, const uint32_t *esis, uint32_t count,
                                    bool use_ibv_out)
{
    if (!job || !src_mr)
        return false;
    bool zc_out = use_ibv_out && job->has_out_dmabuf && job->out_ibv_lkey != 0u;
    if (job->gpurt_kind) {
        if (!src_mr->gpu_dev_ptr)
            return false;
        job->ticket = vxf_gpurt_rq_submit_vram_enc(job->pool_idx, src_mr->gpu_dev_ptr, src_offset, valid_bytes, rq,
                                                   &job->rq_saved_sched, sym_T, stride, esis, count, zc_out);
        if (job->ticket != 0u)
            job->used_vram_out = zc_out;
        return job->ticket != 0u;
    }
    if (!src_mr->vk_buf)
        return false;
    vxfec_vk_buf *target_out = zc_out ? job->out_dmabuf : job->out_buf;
    pthread_mutex_lock(&g_vk_mtx);
    vxfec_vk_ctx *vk = (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) ? g_vk_devs[job->vk_idx].ctx : NULL;
    job->ticket = vk ? vxfec_rq_encode_submit_vram(vk, rq, job->rq_sched, src_mr->vk_buf, src_offset, valid_bytes, sym_T,
                                                   job->in_buf, stride, esis, count, target_out, !zc_out)
                     : 0u;
    if (job->ticket != 0u)
        job->used_vram_out = zc_out;
    pthread_mutex_unlock(&g_vk_mtx);
    return job->ticket != 0u;
}

bool vxf_rq_gpu_job_submit_vram_rep(vxfec_gpu_job_t *job, nanorq_core *rq, const uint8_t *sys_bitmap, uint32_t K, uint32_t sym_T,
                                    vxfec_mr_t *target_mr, size_t target_offset, size_t total_bytes, uint32_t stride,
                                    const uint32_t *missing_esis, uint32_t count, uint32_t overhead)
{
    if (!job || !target_mr)
        return false;
    if (job->gpurt_kind) {
        if (!target_mr->gpu_dev_ptr)
            return false;
        job->ticket =
            vxf_gpurt_rq_submit_vram_rep(job->pool_idx, target_mr->gpu_dev_ptr, target_offset, total_bytes, rq,
                                         &job->rq_saved_sched, sys_bitmap, K, sym_T, stride, missing_esis, count, overhead);
        return job->ticket != 0u;
    }
    if (!target_mr->vk_buf)
        return false;
    pthread_mutex_lock(&g_vk_mtx);
    vxfec_vk_ctx *vk = (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) ? g_vk_devs[job->vk_idx].ctx : NULL;
    job->ticket = vk ? vxfec_rq_repair_submit_vram(vk, rq, job->rq_sched, sys_bitmap, K, sym_T, target_mr->vk_buf, target_offset,
                                                   total_bytes, job->in_buf, stride, missing_esis, count, overhead, job->out_buf)
                     : 0u;
    pthread_mutex_unlock(&g_vk_mtx);
    return job->ticket != 0u;
}

/* non-blocking RS8 job creation: zero VkBuffer allocations and zero vkWaitForFences stalls */
vxfec_gpu_job_t *vxf_rs8_gpu_job_new(const vxfec_mr_t *prefer_mr, bool is_decode, uint32_t K, uint32_t rs_m,
                                     const uint8_t *erased_k, const uint8_t *parity_m, uint32_t count, uint32_t sym_T)
{
    ensure_vk_caps_probed();
    bool gpurt = atomic_load_explicit(&g_gpurt_active, memory_order_acquire);
    bool has_mr = vxf_mr_has_gpu_buf(prefer_mr);
    bool vram_only = has_mr && (!gpurt || vxf_mr_has_ibv(prefer_mr));
    uint32_t min_k = (vram_only || g_vk_force_all) ? 16u : 32u;
    if ((!gpurt && (g_vk_mode == 0 || g_num_vk_devs == 0u)) || K < min_k || count == 0u || count > 255u || (sym_T & 3u) != 0u)
        return NULL;
    if (!g_vk_force_all) {
        if (vram_only) {
            if (sym_T < 64u)
                return NULL;
        } else {
            uint64_t min_work = 64ULL * 1024ULL * 1024ULL;
            if (sym_T < 256u || count < 4u || ((uint64_t)K * (uint64_t)count * (uint64_t)sym_T) < min_work)
                return NULL;
        }
    }

    size_t in_bytes = (size_t)K * sym_T;
    size_t out_bytes = (size_t)count * sym_T;
    vxfec_gpu_job_t *job = (vxfec_gpu_job_t *)calloc(1, sizeof(*job));
    uint8_t *meta = job ? (uint8_t *)malloc(512u) : NULL;
    if (!meta) {
        free(job);
        return NULL;
    }
    job->kind = VXFEC_GPU_JOB_RS8;
    job->vk_idx = -1;
    job->pool_idx = -1;
    job->rs8_is_decode = is_decode;
    job->rs8_K = K;
    job->rs8_M = rs_m;
    job->rs8_count = count;
    job->rs8_erased_k = meta;
    job->rs8_parity_m = meta + 256u;
    if (is_decode && erased_k)
        memcpy(job->rs8_erased_k, erased_k, count);
    if (parity_m)
        memcpy(job->rs8_parity_m, parity_m, count);

    pthread_mutex_lock(&g_vk_mtx);
    gpu_reap_locked();
    if (g_gpu_active_jobs >= VXFEC_RQ_GPU_MAX_JOBS) {
        pthread_mutex_unlock(&g_vk_mtx);
        free(job->rs8_erased_k);
        free(job);
        return NULL;
    }
    if (atomic_load_explicit(&g_gpurt_active, memory_order_acquire)) {
        int pidx = acquire_gpurt_slot_locked(prefer_mr, in_bytes, out_bytes, job);
        if (pidx >= 0) {
            g_gpu_active_jobs++;
            job->pool_idx = pidx;
            pthread_mutex_unlock(&g_vk_mtx);
            return job;
        }
        pthread_mutex_unlock(&g_vk_mtx);
        free(job->rs8_erased_k);
        free(job);
        return NULL;
    }

    int vidx = resolve_job_vk_slot_locked(prefer_mr ? prefer_mr->vk_ctx : NULL);
    if (vidx >= 0) {
        vxf_vk_slot_t *vs = &g_vk_devs[vidx];
        bool caps_ok = vs->has_timeline && (g_vk_force_all || vs->is_hw_gpu) && in_bytes <= vs->max_ssbo_bytes &&
                       out_bytes <= vs->max_ssbo_bytes;
        if (caps_ok && vxfec_vk_async_can_submit(vs->ctx)) {
            int pidx = acquire_vk_async_pool_locked(vs, in_bytes, out_bytes);
            if (pidx >= 0) {
                g_gpu_active_jobs++;
                job->vk_idx = vidx;
                job->pool_idx = pidx;
                job->in_buf = vs->pool[pidx].in_buf;
                job->out_buf = vs->pool[pidx].out_buf;
            } else {
                free(job->rs8_erased_k);
                free(job);
                job = NULL;
            }
        } else {
            free(job->rs8_erased_k);
            free(job);
            job = NULL;
        }
    } else {
        free(job->rs8_erased_k);
        free(job);
        job = NULL;
    }
    pthread_mutex_unlock(&g_vk_mtx);
    return job;
}

bool vxf_rs8_gpu_job_submit(vxfec_gpu_job_t *job, uint32_t sym_T)
{
    if (!job)
        return false;
    if (job->gpurt_kind) {
        job->ticket = vxf_gpurt_rs8_submit(job->pool_idx, job->rs8_is_decode, job->rs8_K, job->rs8_M,
                                           job->rs8_is_decode ? job->rs8_erased_k : NULL, job->rs8_parity_m, job->rs8_count, sym_T);
        return job->ticket != 0u;
    }
    pthread_mutex_lock(&g_vk_mtx);
    vxfec_vk_ctx *vk = (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) ? g_vk_devs[job->vk_idx].ctx : NULL;
    job->ticket =
        vk ? vxfec_rs8_submit_raw(vk, job->rs8_is_decode, job->rs8_K, job->rs8_M, job->rs8_is_decode ? job->rs8_erased_k : NULL,
                                  job->rs8_parity_m, job->rs8_count, job->in_buf, sym_T, 1u, job->out_buf, true, true)
           : 0u;
    pthread_mutex_unlock(&g_vk_mtx);
    return job->ticket != 0u;
}

bool vxf_rs8_gpu_job_submit_vram_enc(vxfec_gpu_job_t *job, vxfec_mr_t *src_mr, size_t src_offset, size_t valid_bytes,
                                     uint32_t sym_T, bool use_ibv_out)
{
    if (!job || job->rs8_is_decode || !src_mr)
        return false;
    bool zc_out = use_ibv_out && job->has_out_dmabuf && job->out_ibv_lkey != 0u;
    if (job->gpurt_kind) {
        if (!src_mr->gpu_dev_ptr)
            return false;
        job->ticket = vxf_gpurt_rs8_submit_vram_enc(job->pool_idx, src_mr->gpu_dev_ptr, src_offset, valid_bytes, job->rs8_K,
                                                    job->rs8_M, job->rs8_parity_m, job->rs8_count, sym_T, zc_out);
        if (job->ticket != 0u)
            job->used_vram_out = zc_out;
        return job->ticket != 0u;
    }
    if (!src_mr->vk_buf)
        return false;
    vxfec_vk_buf *target_out = zc_out ? job->out_dmabuf : job->out_buf;
    pthread_mutex_lock(&g_vk_mtx);
    vxfec_vk_ctx *vk = (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) ? g_vk_devs[job->vk_idx].ctx : NULL;
    job->ticket = vk ? vxfec_rs8_encode_submit_vram(vk, job->rs8_K, job->rs8_M, job->rs8_parity_m, job->rs8_count, src_mr->vk_buf,
                                                    src_offset, valid_bytes, job->in_buf, sym_T, target_out, !zc_out)
                     : 0u;
    if (job->ticket != 0u)
        job->used_vram_out = zc_out;
    pthread_mutex_unlock(&g_vk_mtx);
    return job->ticket != 0u;
}

bool vxf_rs8_gpu_job_submit_vram_rep(vxfec_gpu_job_t *job, const uint8_t *sys_bitmap, vxfec_mr_t *target_mr, size_t target_offset,
                                     size_t total_bytes, uint32_t sym_T)
{
    if (!job || !job->rs8_is_decode || !target_mr)
        return false;
    if (job->gpurt_kind) {
        if (!target_mr->gpu_dev_ptr)
            return false;
        job->ticket =
            vxf_gpurt_rs8_submit_vram_rep(job->pool_idx, target_mr->gpu_dev_ptr, target_offset, total_bytes, job->rs8_K, job->rs8_M,
                                          sys_bitmap, job->rs8_erased_k, job->rs8_parity_m, job->rs8_count, sym_T);
        return job->ticket != 0u;
    }
    if (!target_mr->vk_buf)
        return false;
    pthread_mutex_lock(&g_vk_mtx);
    vxfec_vk_ctx *vk = (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) ? g_vk_devs[job->vk_idx].ctx : NULL;
    job->ticket = vk ? vxfec_rs8_repair_submit_vram(vk, job->rs8_K, job->rs8_M, sys_bitmap, job->rs8_erased_k, job->rs8_parity_m,
                                                    job->rs8_count, target_mr->vk_buf, target_offset, total_bytes, job->in_buf,
                                                    sym_T, job->out_buf)
                     : 0u;
    pthread_mutex_unlock(&g_vk_mtx);
    return job->ticket != 0u;
}

static void query_iface_props(const char *ifname, uint32_t *out_mtu, uint64_t *out_speed_mbps, char out_pci[PATH_MAX])
{
    *out_mtu = 1500u;
    *out_speed_mbps = 400000ULL;
    snprintf(out_pci, PATH_MAX, "/sys/devices/virtual/net/%s", (ifname && *ifname) ? ifname : "lo");

    if (!ifname || !*ifname)
        return;
    if (strcmp(ifname, "lo") == 0) {
        *out_mtu = 65536u;
        return;
    }

    int sfd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (sfd >= 0) {
        struct ifreq ifr;
        memset(&ifr, 0, sizeof(ifr));
        snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "%.*s", (int)(sizeof(ifr.ifr_name) - 1u), ifname);
        if (ioctl(sfd, SIOCGIFMTU, &ifr) == 0 && ifr.ifr_mtu >= 576)
            *out_mtu = (uint32_t)ifr.ifr_mtu;
        close(sfd);
    }

    char sys_spd[PATH_MAX];
    int sn = snprintf(sys_spd, sizeof(sys_spd), "/sys/class/net/%s/speed", ifname);
    if (sn > 0 && (size_t)sn < sizeof(sys_spd)) {
        FILE *fp = fopen(sys_spd, "r");
        if (fp) {
            long spd = 0;
            if (fscanf(fp, "%ld", &spd) == 1 && spd > 0)
                *out_speed_mbps = (uint64_t)spd;
            fclose(fp);
        }
    }

    char sys_link[PATH_MAX];
    char resolved[PATH_MAX];
    int n = snprintf(sys_link, sizeof(sys_link), "/sys/class/net/%s/device", ifname);
    if (n > 0 && (size_t)n < sizeof(sys_link) && realpath(sys_link, resolved))
        snprintf(out_pci, PATH_MAX, "%.*s", (int)(PATH_MAX - 1u), resolved);
}

static bool ibv_dev_has_netdev(const char *ibv_dev, const char *ifname)
{
    if (!ibv_dev || !*ibv_dev || !ifname || !*ifname)
        return false;
    char sys_path[PATH_MAX];
    int n = snprintf(sys_path, sizeof(sys_path), "/sys/class/infiniband/%s/device/net/%s", ibv_dev, ifname);
    return n > 0 && (size_t)n < sizeof(sys_path) && access(sys_path, F_OK) == 0;
}

static uint32_t resolve_ibv_ctrl_ip_be(const vxfec_ibv_dev_info_t *info, const char *if_filter, uint32_t default_bind_ip_be,
                                       bool explicit_bind_ip)
{
    if (explicit_bind_ip && (!if_filter || !*if_filter))
        return default_bind_ip_be;
    if (info->ipv4_be != 0u && (ntohl(info->ipv4_be) >> 24u) != 127u && (!if_filter || !*if_filter))
        return info->ipv4_be;

    char parent_if[IFNAMSIZ] = {0};
    char parent_path[PATH_MAX];
    int pn = snprintf(parent_path, sizeof(parent_path), "/sys/class/infiniband/%s/parent", info->dev_name);
    if (pn > 0 && (size_t)pn < sizeof(parent_path)) {
        FILE *fp = fopen(parent_path, "r");
        if (fp) {
            if (fgets(parent_if, sizeof(parent_if), fp))
                parent_if[strcspn(parent_if, "\r\n")] = '\0';
            fclose(fp);
        }
    }

    uint32_t netdev_flt_ip = 0u, netdev_any_ip = 0u;
    uint32_t host_flt_ip = 0u, host_any_ip = 0u, explicit_lo_ip = 0u;
    struct ifaddrs *ifap = NULL;
    if (getifaddrs(&ifap) == 0) {
        for (struct ifaddrs *ifa = ifap; ifa != NULL; ifa = ifa->ifa_next) {
            if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET || !ifa->ifa_name || (ifa->ifa_flags & IFF_UP) == 0)
                continue;
            uint32_t ip_be = ((const struct sockaddr_in *)ifa->ifa_addr)->sin_addr.s_addr;
            if (ip_be == 0u)
                continue;
            bool is_lo = ((ifa->ifa_flags & IFF_LOOPBACK) != 0) || ((ntohl(ip_be) >> 24u) == 127u);
            bool match_flt = (if_filter && *if_filter) ? vxfec_match_ifname(ifa->ifa_name, if_filter) : !is_lo;
            if (is_lo) {
                if (if_filter && *if_filter && match_flt && explicit_lo_ip == 0u)
                    explicit_lo_ip = ip_be;
                continue;
            }
            bool is_assoc =
                (parent_if[0] && strcmp(ifa->ifa_name, parent_if) == 0) || ibv_dev_has_netdev(info->dev_name, ifa->ifa_name);
            if (is_assoc && match_flt && netdev_flt_ip == 0u)
                netdev_flt_ip = ip_be;
            if (is_assoc && netdev_any_ip == 0u)
                netdev_any_ip = ip_be;
            if (match_flt && host_flt_ip == 0u)
                host_flt_ip = ip_be;
            if (host_any_ip == 0u)
                host_any_ip = ip_be;
        }
        freeifaddrs(ifap);
    }

    if (if_filter && *if_filter) {
        if (netdev_flt_ip != 0u)
            return explicit_bind_ip ? default_bind_ip_be : netdev_flt_ip;
        if ((netdev_any_ip == 0u || !info->is_roce) && host_flt_ip != 0u)
            return explicit_bind_ip ? default_bind_ip_be : host_flt_ip;
        if (explicit_lo_ip != 0u)
            return explicit_bind_ip ? default_bind_ip_be : explicit_lo_ip;
        return 0u;
    }

    if (explicit_bind_ip && default_bind_ip_be != 0u)
        return default_bind_ip_be;
    if (netdev_any_ip != 0u)
        return netdev_any_ip;
    if (info->ipv4_be != 0u && (ntohl(info->ipv4_be) >> 24u) != 127u)
        return info->ipv4_be;
    if (host_any_ip != 0u)
        return host_any_ip;
    return info->ipv4_be ? info->ipv4_be : default_bind_ip_be;
}

static void apply_nic_slice_and_rails_locked(void)
{
    const char *slice_s = getenv("VXFEC_NCCL_DEV_SLICE");
    if (slice_s && *slice_s && vxf_num_devs > 1u) {
        int proc_idx = -1, num_procs = 0;
        char extra = '\0';
        if (sscanf(slice_s, "%d/%d%c", &proc_idx, &num_procs, &extra) == 2 && proc_idx >= 0 && num_procs > 1 &&
            proc_idx < num_procs) {
            bool any_match = false;
            for (uint32_t i = 0; i < vxf_num_devs; i++) {
                if ((int)(i % (uint32_t)num_procs) == proc_idx) {
                    any_match = true;
                    break;
                }
            }
            if (any_match) {
                uint32_t out_n = 0u;
                for (uint32_t i = 0; i < vxf_num_devs; i++) {
                    if ((int)(i % (uint32_t)num_procs) == proc_idx) {
                        if (out_n != i)
                            vxf_devs[out_n] = vxf_devs[i];
                        out_n++;
                    } else if (vxf_devs[i].ibv_ref) {
                        vxf_ibv_ref_release(vxf_devs[i].ibv_ref);
                        vxf_devs[i].ibv_ref = NULL;
                    }
                }
                if (out_n < vxf_num_devs)
                    memset(&vxf_devs[out_n], 0, (size_t)(vxf_num_devs - out_n) * sizeof(vxf_devs[0]));
                vxf_num_devs = out_n;
            }
        }
    }

    const char *rails_s = getenv("VXFEC_NCCL_RAILS");
    if (rails_s && *rails_s && vxf_num_devs >= 1u) {
        int want_rails = atoi(rails_s);
        if (want_rails > (int)VXFEC_MAX_DEVS)
            want_rails = (int)VXFEC_MAX_DEVS;
        uint32_t base_n = vxf_num_devs;
        while ((int)vxf_num_devs < want_rails) {
            uint32_t dst_idx = vxf_num_devs;
            vxfec_nic_dev_t src_copy = vxf_devs[dst_idx % base_n];
            vxfec_nic_dev_t *dst = &vxf_devs[dst_idx];
            *dst = src_copy;
            if (dst->ibv_ref)
                vxf_ibv_ref_retain(dst->ibv_ref);
            snprintf(dst->name, sizeof(dst->name), "%.112s-r%u", src_copy.name, dst_idx);
            dst->guid = src_copy.guid + 0x100ULL * (uint64_t)dst_idx;
            vxf_num_devs++;
        }
    }
}

static void populate_nic_devices_locked(uint32_t default_bind_ip_be, bool explicit_bind_ip)
{
    for (uint32_t i = 0; i < vxf_num_devs; i++) {
        if (vxf_devs[i].ibv_ref) {
            vxf_ibv_ref_release(vxf_devs[i].ibv_ref);
            vxf_devs[i].ibv_ref = NULL;
        }
    }
    memset(vxf_devs, 0, sizeof(vxf_devs));
    vxf_num_devs = 0u;

    const char *if_filter = getenv("VXFEC_NCCL_IFNAMES");
    if ((!if_filter || !*if_filter) && !explicit_bind_ip) {
        const char *nccl_if = getenv("NCCL_SOCKET_IFNAME");
        if (nccl_if && *nccl_if)
            if_filter = nccl_if;
    }

    const char *backend = getenv("VXFEC_NCCL_BACKEND");
    bool want_ibv = (backend && strcmp(backend, "ibv") == 0);

    if (want_ibv && vxfec_ibv_available()) {
        vxfec_ibv_dev_info_t ibv_infos[VXFEC_MAX_DEVS];
        uint32_t n_ibv = vxfec_ibv_enumerate(ibv_infos, VXFEC_MAX_DEVS);
        for (uint32_t i = 0; i < n_ibv && vxf_num_devs < VXFEC_MAX_DEVS; i++) {
            uint32_t ctrl_ip_be = resolve_ibv_ctrl_ip_be(&ibv_infos[i], if_filter, default_bind_ip_be, explicit_bind_ip);
            if (ctrl_ip_be == 0u)
                continue;
            vxfec_ibv_dev_ref_t *iref = ibv_ref_create(i);
            if (!iref)
                continue;
            vxfec_nic_dev_t *d = &vxf_devs[vxf_num_devs];
            uint8_t pnum = ibv_infos[i].port_num ? ibv_infos[i].port_num : 1u;
            snprintf(d->name, sizeof(d->name), "vxfec-ibv-%.44s:%u", ibv_infos[i].dev_name, (unsigned)pnum);
            snprintf(d->ifname, sizeof(d->ifname), "%.*s", (int)(sizeof(d->ifname) - 1u), ibv_infos[i].dev_name);
            if (ibv_infos[i].pci_path[0])
                snprintf(d->pci_path, sizeof(d->pci_path), "%.*s", (int)(sizeof(d->pci_path) - 1u), ibv_infos[i].pci_path);
            else
                snprintf(d->pci_path, sizeof(d->pci_path), "/sys/class/infiniband/%.48s", ibv_infos[i].dev_name);
            d->ip_be = ctrl_ip_be;
            d->mtu = ibv_infos[i].active_mtu ? ibv_infos[i].active_mtu : 4096u;
            d->speed_mbps = ibv_infos[i].speed_mbps ? ibv_infos[i].speed_mbps : 400000ULL;
            d->guid = ibv_infos[i].guid ? ibv_infos[i].guid : (0x5658464543303000ULL + vxf_num_devs);
            d->is_ibv = true;
            d->ibv_port = pnum;
            d->ibv_idx = i;
            d->ibv_ref = iref;
            vxf_num_devs++;
        }
        if (vxf_num_devs > 0u || (if_filter && *if_filter)) {
            apply_nic_slice_and_rails_locked();
            return;
        }
    }

    struct ifaddrs *ifap = NULL;
    if (getifaddrs(&ifap) == 0) {
        for (struct ifaddrs *ifa = ifap; ifa != NULL && vxf_num_devs < VXFEC_MAX_DEVS; ifa = ifa->ifa_next) {
            if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET || !ifa->ifa_name || (ifa->ifa_flags & IFF_UP) == 0)
                continue;
            const struct sockaddr_in *sin = (const struct sockaddr_in *)ifa->ifa_addr;
            if (if_filter && *if_filter) {
                if (!vxfec_match_ifname(ifa->ifa_name, if_filter))
                    continue;
            } else if (sin->sin_addr.s_addr != default_bind_ip_be) {
                continue;
            }

            bool dup = false;
            for (uint32_t j = 0; j < vxf_num_devs; j++) {
                if (vxf_devs[j].ip_be == sin->sin_addr.s_addr && strcmp(vxf_devs[j].ifname, ifa->ifa_name) == 0) {
                    dup = true;
                    break;
                }
            }
            if (dup)
                continue;

            vxfec_nic_dev_t *d = &vxf_devs[vxf_num_devs];
            snprintf(d->name, sizeof(d->name), "vxfec-ecmp-harq%u", vxf_num_devs);
            snprintf(d->ifname, sizeof(d->ifname), "%.*s", (int)(sizeof(d->ifname) - 1u), ifa->ifa_name);
            d->ip_be = sin->sin_addr.s_addr;
            query_iface_props(ifa->ifa_name, &d->mtu, &d->speed_mbps, d->pci_path);
            d->guid = 0x5658464543303038ULL + vxf_num_devs;
            vxf_num_devs++;
            if (!if_filter || !*if_filter)
                break;
        }
        freeifaddrs(ifap);
    }

    if (vxf_num_devs == 0u && (!if_filter || !*if_filter)) {
        vxfec_nic_dev_t *d = &vxf_devs[0];
        snprintf(d->name, sizeof(d->name), "vxfec-ecmp-harq0");
        snprintf(d->ifname, sizeof(d->ifname), "lo");
        snprintf(d->pci_path, sizeof(d->pci_path), "/sys/devices/virtual/net/lo");
        d->ip_be = default_bind_ip_be;
        d->mtu = (default_bind_ip_be == htonl(INADDR_LOOPBACK)) ? 65536u : 1500u;
        d->speed_mbps = 400000ULL;
        d->guid = 0x5658464543303038ULL;
        vxf_num_devs = 1u;
    }
    apply_nic_slice_and_rails_locked();
}

uint32_t vxf_get_dev_ip_be(int dev_idx)
{
    pthread_mutex_lock(&g_init_mtx);
    uint32_t ip =
        (dev_idx >= 0 && (uint32_t)dev_idx < vxf_num_devs && vxf_devs[dev_idx].ip_be != 0u) ? vxf_devs[dev_idx].ip_be : 0u;
    pthread_mutex_unlock(&g_init_mtx);
    if (ip != 0u)
        return ip;
    ip = atomic_load_explicit(&vxf_cfg_bind_ip_be, memory_order_relaxed);
    return ip ? ip : htonl(INADDR_LOOPBACK);
}

uint32_t vxf_parse_slice_min_env(uint32_t fallback)
{
    const char *slice_s = getenv("VXFEC_NCCL_SLICE_KB");
    if (!slice_s || !*slice_s)
        return fallback;
    long kb = atol(slice_s);
    if (kb <= 0)
        return UINT32_MAX;
    if ((uint64_t)kb * 1024ULL <= UINT32_MAX)
        return (uint32_t)((uint64_t)kb * 1024ULL);
    return UINT32_MAX;
}

ncclResult_t vxf_init(ncclDebugLogger_t logFunction)
{
    pthread_mutex_lock(&g_init_mtx);
    vxf_logger = logFunction;
    nanorq_core_init();

    const char *ports_s = getenv("VXFEC_NCCL_PORTS");
    uint32_t n_ports = VXFEC_DEF_PORTS;
    if (ports_s && *ports_s) {
        int v = atoi(ports_s);
        n_ports = (v < 1) ? 1u : ((v > (int)VXFEC_MAX_PORTS) ? VXFEC_MAX_PORTS : (uint32_t)v);
    }
    atomic_store(&vxf_cfg_ports, n_ports);

    const char *mtu_s = getenv("VXFEC_NCCL_MTU_SYM");
    uint32_t mtu_sym = 0u;
    if (mtu_s && *mtu_s) {
        int v = atoi(mtu_s);
        if (v >= 64 && v <= (int)VXFEC_MAX_SYM_T)
            mtu_sym = ((uint32_t)v + 15u) & ~15u;
    }
    atomic_store(&vxf_cfg_mtu_sym, mtu_sym);

    uint32_t bind_ip = htonl(INADDR_LOOPBACK);
    const char *ip_s = getenv("VXFEC_NCCL_BIND_IP");
    bool explicit_ip = false;
    struct in_addr addr;
    if (ip_s && *ip_s && inet_pton(AF_INET, ip_s, &addr) == 1) {
        bind_ip = addr.s_addr;
        explicit_ip = true;
    }
    atomic_store(&vxf_cfg_bind_ip_be, bind_ip);

    const char *bp_s = getenv("VXFEC_NCCL_BUSY_POLL_US");
    atomic_store(&vxf_cfg_busy_poll_us, (bp_s && *bp_s && atoi(bp_s) > 0) ? (uint32_t)atoi(bp_s) : 0u);
    atomic_store(&vxf_cfg_slice_min_bytes, vxf_parse_slice_min_env(VXFEC_DEF_SLICE_MIN_BYTES));
    const char *rw_s = getenv("VXFEC_NCCL_RAIL_WORKERS");
    uint32_t rw_mode = 1u;
    if (rw_s && *rw_s && strcmp(rw_s, "auto") != 0) {
        int rv = atoi(rw_s);
        rw_mode = (rv <= 0) ? 0u : 2u;
    }
    atomic_store(&vxf_cfg_rail_workers, rw_mode);

    const char *strict_pix = getenv("VXFEC_NCCL_STRICT_PIX_MR");
    atomic_store(&vxf_cfg_strict_pix_mr, strict_pix && atoi(strict_pix) > 0);
    const char *cwnd_s = getenv("VXFEC_NCCL_CWND_KB");
    long cwnd_kb = (cwnd_s && *cwnd_s) ? atol(cwnd_s) : 0L;
    atomic_store(&vxf_cfg_cwnd_kb, (cwnd_kb >= 16L) ? (uint64_t)cwnd_kb : 0ULL);
    const char *rate_s = getenv("VXFEC_NCCL_MAX_RATE_MBPS");
    long rate_mbps = (rate_s && *rate_s) ? atol(rate_s) : 0L;
    atomic_store(&vxf_cfg_max_rate_mbps, (rate_mbps > 0L) ? (uint64_t)rate_mbps : 0ULL);
    const char *qd_s = getenv("VXFEC_NCCL_TARGET_QDELAY_US");
    long qdelay_us = (qd_s && *qd_s) ? atol(qd_s) : 0L;
    atomic_store(&vxf_cfg_target_qdelay_us, (qd_s && *qd_s) ? (qdelay_us > 0L ? (int64_t)qdelay_us : 0LL) : -1LL);

    populate_nic_devices_locked(bind_ip, explicit_ip);

    int env_ovr = (int)vxf_parse_codec_mode(getenv("VXFEC_NCCL_CODEC"));
    atomic_store(&vxf_env_codec_ovr, env_ovr);
    atomic_store(&vxf_codec_override, env_ovr);
    vxf_sim_init_from_env();
    pthread_mutex_unlock(&g_init_mtx);

    pthread_mutex_lock(&g_vk_mtx);
    configure_gpu_backends_locked();
    pthread_mutex_unlock(&g_vk_mtx);
    return ncclSuccess;
}

ncclResult_t vxf_devices(int *ndev)
{
    if (!ndev)
        return ncclInvalidArgument;
    pthread_mutex_lock(&g_init_mtx);
    *ndev = (int)vxf_num_devs;
    pthread_mutex_unlock(&g_init_mtx);
    return ncclSuccess;
}

__attribute__((visibility("default"))) size_t vxfec_nccl_get_rail_stats(vxfec_nccl_rail_stats_t *out, size_t capacity)
{
    pthread_mutex_lock(&g_init_mtx);
    size_t count = vxf_num_devs;
    for (size_t i = 0; out && i < count && i < capacity; i++) {
        memcpy(out[i].name, vxf_devs[i].name, sizeof(out[i].name));
        out[i].name[sizeof(out[i].name) - 1u] = '\0';
        out[i].tx_sys_gpu_direct_pkts = atomic_load_explicit(&vxf_st_rail_tx_gpu_direct_pkts[i], memory_order_relaxed);
        out[i].rx_sys_gpu_direct_pkts = atomic_load_explicit(&vxf_st_rail_rx_gpu_direct_pkts[i], memory_order_relaxed);
    }
    pthread_mutex_unlock(&g_init_mtx);
    return count;
}

ncclResult_t vxf_get_properties(int dev, ncclNetProperties_v8_t *props)
{
    if (dev < 0 || !props)
        return ncclInvalidArgument;

    pthread_mutex_lock(&g_vk_mtx);
    bool gpurt = atomic_load_explicit(&g_gpurt_active, memory_order_acquire);
    int vidx = (!gpurt && g_vk_mode != 0) ? get_primary_vk_slot_locked() : -1;
    vxf_gpurt_caps_t rt_caps;
    bool gpu_ptr_cuda = gpurt && vxf_gpurt_get_caps(&rt_caps);
    bool vk_dmabuf = false;
    if (gpurt) {
        vk_dmabuf = gpu_ptr_cuda && rt_caps.has_dmabuf_export;
    } else if (g_vk_mode != 0 && vidx >= 0 && g_vk_devs[vidx].ctx) {
        vk_dmabuf = vxfec_vk_has_dmabuf(g_vk_devs[vidx].ctx);
    }
    pthread_mutex_unlock(&g_vk_mtx);

    pthread_mutex_lock(&g_init_mtx);
    if ((uint32_t)dev >= vxf_num_devs) {
        pthread_mutex_unlock(&g_init_mtx);
        return ncclInvalidArgument;
    }
    memset(props, 0, sizeof(*props));
    vxfec_nic_dev_t *d = &vxf_devs[dev];
    bool has_dmabuf = vk_dmabuf || (d->is_ibv && d->ibv_ref != NULL && vxfec_ibv_has_dmabuf());
    bool per_dev_mr = atomic_load_explicit(&vxf_cfg_strict_pix_mr, memory_order_relaxed);
    props->name = d->name;
    props->pciPath = d->pci_path;
    props->guid = d->guid;
    props->ptrSupport = NCCL_PTR_HOST | (gpu_ptr_cuda ? NCCL_PTR_CUDA : 0) | (has_dmabuf ? NCCL_PTR_DMABUF : 0);
    props->regIsGlobal = per_dev_mr ? 0 : 1;
    props->speed = (int)((d->speed_mbps > 0u && d->speed_mbps <= (uint64_t)INT_MAX) ? d->speed_mbps : 400000ULL);
    props->port = d->is_ibv ? (int)(d->ibv_port ? d->ibv_port : 1u) : dev;
    props->latency = d->is_ibv ? 1.2f : 2.0f;
    props->maxComms = 65536;
    props->maxRecvs = 1;
    props->netDeviceType = NCCL_NET_DEVICE_HOST;
    props->netDeviceVersion = 8;
    pthread_mutex_unlock(&g_init_mtx);
    return ncclSuccess;
}

static int resolve_strict_pix_dev(void *comm)
{
    if (atomic_load_explicit(&vxf_cfg_strict_pix_mr, memory_order_relaxed) && comm) {
        const vxfec_comm_hdr_t *ch = (const vxfec_comm_hdr_t *)comm;
        if (ch->comm_magic == VXFEC_COMM_MAGIC && ch->comm_dev_idx >= 0)
            return ch->comm_dev_idx;
    }
    return -1;
}

static bool mr_reg_ibv_dmabuf_devs(vxfec_mr_t *mr, int only_dev, uint64_t offset, size_t size, void *data, int fd,
                                   bool *out_had_ibv)
{
    vxfec_ibv_dev_ref_t *refs[VXFEC_MAX_DEVS] = {0};
    bool had_ibv = false;
    pthread_mutex_lock(&g_init_mtx);
    uint32_t ndev = vxf_num_devs;
    for (uint32_t d = 0; d < ndev; d++) {
        if (only_dev >= 0 && (int)d != only_dev)
            continue;
        if (vxf_devs[d].ibv_ref && vxf_devs[d].ibv_ref->ctx) {
            refs[d] = vxf_devs[d].ibv_ref;
            vxf_ibv_ref_retain(refs[d]);
            had_ibv = true;
        }
    }
    pthread_mutex_unlock(&g_init_mtx);

    if (out_had_ibv)
        *out_had_ibv = had_ibv;
    bool any_mr = false;
    for (uint32_t d = 0; d < ndev; d++) {
        if (!refs[d])
            continue;
        if (size > 0u) {
            mr->ibv_mrs[d] = vxfec_ibv_mr_reg_dmabuf(refs[d]->ctx, offset, size, (uint64_t)(uintptr_t)data, fd, true);
            if (mr->ibv_mrs[d]) {
                mr->ibv_can_rwrite[d] = true;
            } else {
                mr->ibv_mrs[d] = vxfec_ibv_mr_reg_dmabuf(refs[d]->ctx, offset, size, (uint64_t)(uintptr_t)data, fd, false);
            }
            if (mr->ibv_mrs[d]) {
                mr->ibv_refs[d] = refs[d];
                refs[d] = NULL;
                any_mr = true;
            }
        }
        vxf_ibv_ref_release(refs[d]);
    }
    return any_mr;
}

ncclResult_t vxf_reg_mr(void *comm, void *data, size_t size, int type, void **mhandle)
{
    if (!mhandle)
        return ncclInvalidArgument;
    ensure_vk_caps_probed();
    if (type == NCCL_PTR_CUDA) {
        if (!atomic_load_explicit(&g_gpurt_active, memory_order_acquire) || (size > 0u && !vxf_gpurt_validate_dev_ptr(data)))
            return ncclInvalidArgument;
        vxfec_mr_t *mr = (vxfec_mr_t *)calloc(1, sizeof(*mr));
        if (!mr)
            return ncclSystemError;
        atomic_init(&mr->refcnt, 1u);
        mr->base_ptr = data;
        mr->gpu_dev_ptr = data;
        mr->size = size;
        mr->ptr_type = NCCL_PTR_CUDA;

        /* auto-export DMA-BUF handle for GPUDirect RDMA_WRITE_WITH_IMM when verbs NICs are present */
        if (data && size > 0u) {
            int only_dev = resolve_strict_pix_dev(comm);
            bool need_ibv = false;
            pthread_mutex_lock(&g_init_mtx);
            for (uint32_t d = 0; d < vxf_num_devs; d++) {
                if ((only_dev < 0 || (int)d == only_dev) && vxf_devs[d].ibv_ref && vxf_devs[d].ibv_ref->ctx) {
                    need_ibv = true;
                    break;
                }
            }
            pthread_mutex_unlock(&g_init_mtx);

            if (need_ibv) {
                uint64_t dmabuf_off = 0ULL;
                int fd = vxf_gpurt_export_dmabuf(data, size, &dmabuf_off);
                if (fd >= 0) {
                    (void)mr_reg_ibv_dmabuf_devs(mr, only_dev, dmabuf_off, size, data, fd, NULL);
                    close(fd);
                }
            }
        }
        *mhandle = mr;
        return ncclSuccess;
    }

    if (type != NCCL_PTR_HOST)
        return ncclInvalidArgument;

    vxfec_mr_t *mr = (vxfec_mr_t *)calloc(1, sizeof(*mr));
    if (!mr)
        return ncclSystemError;
    atomic_init(&mr->refcnt, 1u);
    mr->base_ptr = data;
    mr->size = size;
    mr->ptr_type = type;
    if (data && size > 0u) {
        int only_dev = resolve_strict_pix_dev(comm);
        vxfec_ibv_dev_ref_t *refs[VXFEC_MAX_DEVS] = {0};
        pthread_mutex_lock(&g_init_mtx);
        uint32_t ndev = vxf_num_devs;
        for (uint32_t d = 0; d < ndev; d++) {
            if (only_dev >= 0 && (int)d != only_dev)
                continue;
            if (vxf_devs[d].ibv_ref && vxf_devs[d].ibv_ref->ctx) {
                refs[d] = vxf_devs[d].ibv_ref;
                vxf_ibv_ref_retain(refs[d]);
            }
        }
        pthread_mutex_unlock(&g_init_mtx);
        for (uint32_t d = 0; d < ndev; d++) {
            if (!refs[d])
                continue;
            mr->ibv_mrs[d] = vxfec_ibv_mr_reg_host(refs[d]->ctx, data, size, true);
            if (mr->ibv_mrs[d])
                mr->ibv_can_rwrite[d] = true;
            else
                mr->ibv_mrs[d] = vxfec_ibv_mr_reg_host(refs[d]->ctx, data, size, false);
            if (mr->ibv_mrs[d]) {
                mr->ibv_refs[d] = refs[d];
                refs[d] = NULL;
            }
            vxf_ibv_ref_release(refs[d]);
        }
    }
    *mhandle = mr;
    return ncclSuccess;
}

ncclResult_t vxf_reg_mr_dmabuf(void *comm, void *data, size_t size, int type, uint64_t offset, int fd, void **mhandle)
{
    if (!mhandle)
        return ncclInvalidArgument;
    if (type == NCCL_PTR_HOST)
        return vxf_reg_mr(comm, data, size, type, mhandle);
    if ((type != NCCL_PTR_CUDA && type != NCCL_PTR_DMABUF && type != (NCCL_PTR_CUDA | NCCL_PTR_DMABUF)) || fd < 0)
        return ncclInvalidArgument;
    if ((uint64_t)size > UINT64_MAX - offset)
        return ncclInvalidArgument;
    ensure_vk_caps_probed();

    vxfec_mr_t *mr = (vxfec_mr_t *)calloc(1, sizeof(*mr));
    if (!mr)
        return ncclSystemError;
    atomic_init(&mr->refcnt, 1u);
    mr->base_ptr = data;
    mr->size = size;
    mr->offset = offset;
    mr->ptr_type = NCCL_PTR_CUDA;

    int only_dev = resolve_strict_pix_dev(comm);
    bool had_ibv_dev = false;
    bool any_ibv_mr = mr_reg_ibv_dmabuf_devs(mr, only_dev, offset, size, data, fd, &had_ibv_dev);

    bool vk_supported = false;
    pthread_mutex_lock(&g_vk_mtx);
    if (atomic_load_explicit(&g_gpurt_active, memory_order_acquire)) {
        if (size == 0u || vxf_gpurt_validate_dev_ptr(data)) {
            vk_supported = true;
            mr->gpu_dev_ptr = data;
        }
    } else {
        vxfec_vk_ctx *vk = get_vk_ctx_for_dmabuf_locked(fd);
        if (vk && vxfec_vk_has_dmabuf(vk)) {
            vk_supported = true;
            if (size > 0u) {
                int dup_fd = dup(fd);
                if (dup_fd >= 0) {
                    mr->vk_buf = vxfec_vk_buf_import_fd_range(vk, dup_fd, offset, size, true);
                    if (!mr->vk_buf)
                        close(dup_fd);
                    else
                        mr->vk_ctx = vk;
                }
            } else {
                mr->vk_ctx = vk;
            }
        }
    }
    pthread_mutex_unlock(&g_vk_mtx);

    if (!vk_supported && !had_ibv_dev) {
        vxf_mr_release(mr);
        return ncclInvalidUsage;
    }
    if (size > 0u && !mr->gpu_dev_ptr && !mr->vk_buf && !any_ibv_mr) {
        vxf_mr_release(mr);
        return ncclSystemError;
    }

    *mhandle = mr;
    return ncclSuccess;
}

ncclResult_t vxf_dereg_mr(void *comm, void *mhandle)
{
    (void)comm;
    vxf_mr_release((vxfec_mr_t *)mhandle);
    return ncclSuccess;
}

ncclResult_t vxf_get_device_mr(void *comm, void *mhandle, void **dptr_mhandle)
{
    (void)comm;
    if (!dptr_mhandle)
        return ncclInvalidArgument;
    *dptr_mhandle = mhandle;
    return ncclSuccess;
}

void vxf_dev_shutdown(void)
{
    pthread_mutex_lock(&g_init_mtx);
    if (atomic_load_explicit(&vxf_leaked_ibv_eps, memory_order_relaxed) == 0u) {
        for (uint32_t i = 0; i < vxf_num_devs; i++) {
            if (vxf_devs[i].ibv_ref) {
                vxf_ibv_ref_release(vxf_devs[i].ibv_ref);
                vxf_devs[i].ibv_ref = NULL;
            }
        }
        vxf_num_devs = 0u;
    }
    pthread_mutex_unlock(&g_init_mtx);

    pthread_mutex_lock(&g_vk_mtx);
    bool all_idle = true;
    bool gpurt = atomic_load_explicit(&g_gpurt_active, memory_order_acquire);
    if (gpurt && !vxf_gpurt_wait_idle())
        all_idle = false;
    for (uint32_t d = 0; d < g_num_vk_devs; d++) {
        if (g_vk_devs[d].ctx && !vxfec_vk_wait_idle(g_vk_devs[d].ctx))
            all_idle = false;
    }
    gpu_reap_locked();
    if (all_idle) {
        while (g_gpu_retired) {
            vxfec_gpu_job_t *job = g_gpu_retired;
            g_gpu_retired = job->next;
            if (!job->gpurt_kind) {
                vxfec_vk_ctx *vk = (job->vk_idx >= 0 && (uint32_t)job->vk_idx < g_num_vk_devs) ? g_vk_devs[job->vk_idx].ctx : NULL;
                if (vk)
                    (void)vxfec_vk_wait(vk, job->ticket, 0u);
            }
            gpu_job_destroy_locked(job);
        }
    }
    /* A request or a busy verbs QP may still own an exported repair buffer
     * after the GPU has finished writing it. Keep its pool and backend alive. */
    if (all_idle && g_gpu_active_jobs == 0u) {
        for (uint32_t i = 0; i < VXFEC_GPURT_MAX_POOL; i++)
            clear_gpu_out_export_locked(&g_gpurt_ibv_pool[i]);
        if (gpurt) {
            vxf_gpurt_shutdown();
            atomic_store_explicit(&g_gpurt_active, false, memory_order_release);
        }
        for (uint32_t d = 0; d < g_num_vk_devs; d++) {
            vxf_vk_slot_t *vs = &g_vk_devs[d];
            if (!vs->ctx)
                continue;
            for (uint32_t i = 0; i < VXFEC_VK_ASYNC_POOL_SLOTS; i++) {
                clear_gpu_out_export_locked(&vs->pool[i].out_export);
                vxfec_vk_buf_free(vs->ctx, vs->pool[i].out_dmabuf);
                vxfec_vk_buf_free(vs->ctx, vs->pool[i].in_buf);
                vxfec_vk_buf_free(vs->ctx, vs->pool[i].out_buf);
            }
            vxfec_vk_ctx_free(vs->ctx);
            vs->ctx = NULL;
        }
        memset(g_vk_devs, 0, sizeof(g_vk_devs));
        g_num_vk_devs = 0u;
        g_vk_probed = false;
        atomic_store_explicit(&g_vk_caps_ready, false, memory_order_release);
    }
    pthread_mutex_unlock(&g_vk_mtx);
}
