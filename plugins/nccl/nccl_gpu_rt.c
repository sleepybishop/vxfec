#define _GNU_SOURCE
#include "nccl_gpu_rt.h"
#include "nccl_gpu_kernels.h"
#include "util.h"

/* shared GF(256) matrix builders from vk_rs_cauchy.o / vk_rq.o */
extern bool vxfec_rs8__encode_matrix(uint32_t K, uint32_t M, const uint8_t *parity_m, uint32_t num_parity, uint8_t *Y,
                                     size_t m_stride);
extern bool vxfec_rs8__recovery_matrix(uint32_t K, uint32_t M, const uint8_t *erased_k, const uint8_t *parity_m, uint32_t e,
                                       uint8_t *Y, size_t m_stride);
extern void vxfec_rq__adjoint_gf256(nanorq_core *rq, const sched_op *ops, uint32_t n_ops, uint32_t cp0, uint32_t cp1, uint32_t rows,
                                    const uint32_t *esis, uint32_t num_esis, uint8_t *Y, size_t m_stride);

/* green-context resource descriptors are defined by the CUDA SDK */
#if defined(__has_include) && __has_include(<cuda.h>)
#include <cuda.h>
#endif
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12040 && UINTPTR_MAX > UINT32_MAX
#define VXFEC_HAVE_CUDA_GREEN
#endif

#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef void *vxf_gpu_stream_t;
typedef void *vxf_gpu_event_t;
typedef void *vxf_gpu_mod_t;
typedef void *vxf_gpu_func_t;
typedef void *vxf_gpu_ctx_t;
typedef uint64_t vxf_cu_devptr_t;

typedef struct {
    void *in_stage;
    void *out_stage;
    size_t in_cap;
    size_t out_cap;
    void *d_in;
    void *d_out;
    size_t d_in_cap;
    size_t d_out_cap;
    uint64_t d_out_id;
    void *d_meta;
    size_t d_meta_cap;
    void *h_meta;
    size_t h_meta_cap;
    uint8_t *y_scratch;
    size_t y_scratch_cap;
    vxf_gpu_event_t ev;
    uint64_t ticket;
    bool event_recorded;
    bool failed;
} gpurt_slot_t;

/* HIP runtime + HIPRTC + HSA dynamic table */
typedef struct {
    void *hip_dl;
    void *rtc_dl;
    void *hsa_dl;
    int (*get_dev_cnt)(int *);
    int (*get_dev)(int *);
    int (*set_dev)(int);
    int (*dev_get_attr)(int *, int, int);
    int (*dev_get_pci)(char *, int, int);
    int (*stream_prio_range)(int *, int *);
    int (*stream_create_prio)(vxf_gpu_stream_t *, unsigned int, int);
    int (*stream_create_cumask)(vxf_gpu_stream_t *, uint32_t, const uint32_t *);
    int (*stream_get_cumask)(vxf_gpu_stream_t, uint32_t, uint32_t *);
    int (*ptr_attr)(void *, int, void *);
    int (*stream_sync)(vxf_gpu_stream_t);
    int (*stream_query)(vxf_gpu_stream_t);
    int (*stream_destroy)(vxf_gpu_stream_t);
    int (*event_create)(vxf_gpu_event_t *, unsigned int);
    int (*event_record)(vxf_gpu_event_t, vxf_gpu_stream_t);
    int (*event_query)(vxf_gpu_event_t);
    int (*event_destroy)(vxf_gpu_event_t);
    int (*host_malloc)(void **, size_t, unsigned int);
    int (*host_free)(void *);
    int (*dev_malloc)(void **, size_t);
    int (*dev_free)(void *);
    int (*memcpy_async)(void *, const void *, size_t, int, vxf_gpu_stream_t);
    int (*mod_load_data)(vxf_gpu_mod_t *, const void *);
    int (*mod_unload)(vxf_gpu_mod_t);
    int (*mod_get_func)(vxf_gpu_func_t *, vxf_gpu_mod_t, const char *);
    int (*launch_kernel)(vxf_gpu_func_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                         unsigned int, vxf_gpu_stream_t, void **, void **);
    int (*rtc_create)(void **, const char *, const char *, int, const char *const *, const char *const *);
    int (*rtc_compile)(void *, int, const char *const *);
    int (*rtc_code_sz)(void *, size_t *);
    int (*rtc_code)(void *, char *);
    int (*rtc_destroy)(void **);
    int (*hsa_export_dmabuf)(const void *, size_t, int *, uint64_t *);
} hip_fn_t;

/* CUDA Driver + NVRTC + CUDA 12.4 Green Context dynamic table */
typedef struct {
    void *cu_dl;
    void *rtc_dl;
    int (*init)(unsigned int);
    int (*dev_cnt)(int *);
    int (*dev_get)(int *, int);
    int (*dev_attr)(int *, int, int);
    int (*ctx_get_dev)(int *);
    int (*ctx_push)(vxf_gpu_ctx_t);
    int (*ctx_pop)(vxf_gpu_ctx_t *);
    int (*ptr_attrs)(unsigned int, int *, void **, vxf_cu_devptr_t);
    int (*primary_retain)(vxf_gpu_ctx_t *, int);
    int (*primary_release)(int);
    int (*prio_range)(int *, int *);
    int (*stream_create_prio)(vxf_gpu_stream_t *, unsigned int, int);
    int (*stream_sync)(vxf_gpu_stream_t);
    int (*stream_query)(vxf_gpu_stream_t);
    int (*stream_destroy)(vxf_gpu_stream_t);
#ifdef VXFEC_HAVE_CUDA_GREEN
    CUresult(CUDAAPI *dev_get_res)(CUdevice, CUdevResource *, CUdevResourceType);
    CUresult(CUDAAPI *sm_split)(CUdevResource *, unsigned int *, const CUdevResource *, CUdevResource *, unsigned int,
                                unsigned int);
    CUresult(CUDAAPI *res_gen_desc)(CUdevResourceDesc *, CUdevResource *, unsigned int);
    CUresult(CUDAAPI *green_create)(CUgreenCtx *, CUdevResourceDesc, CUdevice, unsigned int);
    CUresult(CUDAAPI *green_destroy)(CUgreenCtx);
    CUresult(CUDAAPI *green_get_res)(CUgreenCtx, CUdevResource *, CUdevResourceType);
    CUresult(CUDAAPI *green_get_ctx)(CUcontext *, CUgreenCtx);
    CUresult(CUDAAPI *stream_get_green)(CUstream, CUgreenCtx *);
#endif
    int (*event_create)(vxf_gpu_event_t *, unsigned int);
    int (*event_record)(vxf_gpu_event_t, vxf_gpu_stream_t);
    int (*event_query)(vxf_gpu_event_t);
    int (*event_destroy)(vxf_gpu_event_t);
    int (*host_alloc)(void **, size_t, unsigned int);
    int (*host_free)(void *);
    int (*dev_alloc)(vxf_cu_devptr_t *, size_t);
    int (*dev_free)(vxf_cu_devptr_t);
    int (*dev_alloc_async)(vxf_cu_devptr_t *, size_t, vxf_gpu_stream_t);
    int (*dev_free_async)(vxf_cu_devptr_t, vxf_gpu_stream_t);
    int (*mem_get_addr_range)(vxf_cu_devptr_t *, size_t *, vxf_cu_devptr_t);
    int (*memcpy_htod_async)(vxf_cu_devptr_t, const void *, size_t, vxf_gpu_stream_t);
    int (*memcpy_dtoh_async)(void *, vxf_cu_devptr_t, size_t, vxf_gpu_stream_t);
    int (*mem_get_dmabuf)(void *, vxf_cu_devptr_t, size_t, int, unsigned long long);
    int (*mod_load_data)(vxf_gpu_mod_t *, const void *);
    int (*mod_unload)(vxf_gpu_mod_t);
    int (*mod_get_func)(vxf_gpu_func_t *, vxf_gpu_mod_t, const char *);
    int (*launch_kernel)(vxf_gpu_func_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                         unsigned int, vxf_gpu_stream_t, void **, void **);
    int (*rtc_create)(void **, const char *, const char *, int, const char *const *, const char *const *);
    int (*rtc_compile)(void *, int, const char *const *);
    int (*rtc_ptx_sz)(void *, size_t *);
    int (*rtc_ptx)(void *, char *);
    int (*rtc_destroy)(void **);
} cuda_fn_t;

typedef struct gpurt_retired {
    void *ptr;
    bool is_host;
    struct gpurt_retired *next;
} gpurt_retired_t;

typedef struct {
    vxf_gpurt_caps_t caps;
    int cu_device;
    vxf_gpu_ctx_t cu_primary;
    vxf_gpu_ctx_t cu_ctx;
#ifdef VXFEC_HAVE_CUDA_GREEN
    CUgreenCtx cu_green;
#endif
    vxf_gpu_stream_t stream;
    vxf_gpu_stream_t copy_stream;
    vxf_gpu_mod_t mod;
    vxf_gpu_func_t fn_staged;
    vxf_gpu_func_t fn_vram_enc;
    vxf_gpu_func_t fn_vram_rep;
    void *bounce_host;
    size_t bounce_cap;
    gpurt_retired_t *retired_head;
    gpurt_slot_t pool[VXFEC_GPURT_MAX_POOL];
} gpurt_device_t;

#define VXFEC_GPURT_BIN_CACHE_MAX 8u
typedef struct {
    char arch[64];
    char *code;
    size_t size;
} gpurt_bin_cache_t;

/* the runtime mutex also keeps each CUDA green context current to at most one thread */
static pthread_mutex_t g_rt_mtx = PTHREAD_MUTEX_INITIALIZER;
static vxf_gpurt_kind_t g_kind;
static hip_fn_t g_hip;
static cuda_fn_t g_cu;
static gpurt_device_t *g_devices;
static int g_device_count;
static int g_default_device;
static gpurt_device_t *g_slot_owner[VXFEC_GPURT_MAX_POOL];
static gpurt_device_t *g_slot_last_device[VXFEC_GPURT_MAX_POOL];
static uint64_t g_next_ticket = 1ULL;
static uint64_t g_next_out_id = 1ULL;
static gpurt_bin_cache_t g_bin_cache[VXFEC_GPURT_BIN_CACHE_MAX];
static uint32_t g_bin_cache_cnt = 0u;
static pid_t g_init_pid = 0;

static const char *bin_cache_find_locked(const char *arch, size_t *out_sz)
{
    for (uint32_t i = 0u; i < g_bin_cache_cnt; i++) {
        if (strcmp(g_bin_cache[i].arch, arch) == 0) {
            if (out_sz)
                *out_sz = g_bin_cache[i].size;
            return g_bin_cache[i].code;
        }
    }
    return NULL;
}

static void bin_cache_put_locked(const char *arch, const char *code, size_t sz)
{
    if (!arch || !code || sz == 0u)
        return;
    uint32_t idx = 0u;
    while (idx < g_bin_cache_cnt && strcmp(g_bin_cache[idx].arch, arch) != 0)
        idx++;
    if (idx >= VXFEC_GPURT_BIN_CACHE_MAX)
        return;
    char *dup = (char *)malloc(sz);
    if (!dup)
        return;
    memcpy(dup, code, sz);
    gpurt_bin_cache_t *entry = &g_bin_cache[idx];
    free(entry->code);
    snprintf(entry->arch, sizeof(entry->arch), "%s", arch);
    entry->code = dup;
    entry->size = sz;
    if (idx == g_bin_cache_cnt)
        g_bin_cache_cnt++;
}

typedef struct {
    int hip_previous;
    bool restore;
} gpurt_binding_t;

static bool bind_device_locked(const gpurt_device_t *device, gpurt_binding_t *binding)
{
    if (g_kind == VXF_GPURT_HIP) {
        int ordinal = device->caps.device_ordinal;
        if (g_hip.get_dev(&binding->hip_previous) != 0)
            return false;
        if (binding->hip_previous != ordinal) {
            if (g_hip.set_dev(ordinal) != 0)
                return false;
            binding->restore = true;
        }
        return true;
    }
    if (g_cu.ctx_push(device->cu_ctx) != 0)
        return false;
    binding->restore = true;
    return true;
}

static void unbind_device_locked(gpurt_binding_t binding)
{
    if (!binding.restore)
        return;
    if (g_kind == VXF_GPURT_HIP) {
        (void)g_hip.set_dev(binding.hip_previous);
    } else {
        vxf_gpu_ctx_t context;
        (void)g_cu.ctx_pop(&context);
    }
}

static void unlock_device(gpurt_binding_t binding)
{
    unbind_device_locked(binding);
    pthread_mutex_unlock(&g_rt_mtx);
}

static bool is_any_lib_resident(const char *const *names)
{
    for (int i = 0; names[i] != NULL; i++) {
        void *h = dlopen(names[i], RTLD_LAZY | RTLD_NOLOAD);
        if (h) {
            dlclose(h);
            return true;
        }
    }
    return false;
}

static const char *const g_resident_hip_libs[] = {
    "librccl.so.1", "librccl.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", NULL,
};
static const char *const g_resident_cuda_libs[] = {
    "libnccl.so.2", "libnccl.so", "libcuda.so.1", "libcudart.so.13", "libcudart.so.12", "libcudart.so", NULL,
};

bool vxf_gpurt_should_auto_enable(void)
{
    const char *be = getenv("VXFEC_NCCL_GPU_BACKEND");
    if (be && *be) {
        if (strcmp(be, "hip") == 0 || strcmp(be, "cuda") == 0 || strcmp(be, "native") == 0)
            return true;
        if (strcmp(be, "auto") != 0)
            return false;
    }
    return is_any_lib_resident(g_resident_hip_libs) || is_any_lib_resident(g_resident_cuda_libs);
}

static void *dlopen_first(const char *const *names)
{
    for (int i = 0; names[i] != NULL; i++) {
        void *probe = dlopen(names[i], RTLD_LAZY | RTLD_NOLOAD);
        if (probe) {
            dlclose(probe);
            void *h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
            if (h)
                return h;
        }
    }
    for (int i = 0; names[i] != NULL; i++) {
        void *h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
        if (h)
            return h;
    }
    return NULL;
}

/* Query AMD KFD sysfs topology for exact gfx target string and physical CU count */
static void probe_kfd_gpu_props(int target_gpu_ord, const char *pci_bdf, char out_arch[32], uint32_t *out_phys_cus)
{
    out_arch[0] = '\0';
    if (out_phys_cus)
        *out_phys_cus = 0u;
    const char *env_arch = getenv("VXFEC_NCCL_HIP_ARCH");
    if (env_arch && *env_arch)
        snprintf(out_arch, 32u, "%.31s", env_arch);

    uint32_t want_dom = 0u, want_loc = 0u;
    bool have_pci = false;
    if (pci_bdf && *pci_bdf) {
        unsigned d = 0u, b = 0u, dev = 0u, fn = 0u;
        if (sscanf(pci_bdf, "%x:%x:%x.%x", &d, &b, &dev, &fn) == 4) {
            want_dom = d;
            want_loc = ((b & 0xffu) << 8u) | ((dev & 0x1fu) << 3u) | (fn & 0x7u);
            have_pci = true;
        }
    }

    int gpu_seen = 0;
    for (uint32_t nid = 0u; nid < 64u; nid++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/virtual/kfd/kfd/topology/nodes/%u/properties", nid);
        FILE *fp = fopen(path, "r");
        if (!fp)
            continue;
        uint32_t gfx_ver = 0u, simd_cnt = 0u, simd_per_cu = 0u, dom = 0u, loc = 0u;
        char line[128];
        while (fgets(line, sizeof(line), fp)) {
            unsigned val = 0u;
            if (sscanf(line, "gfx_target_version %u", &val) == 1)
                gfx_ver = val;
            else if (sscanf(line, "simd_count %u", &val) == 1)
                simd_cnt = val;
            else if (sscanf(line, "simd_per_cu %u", &val) == 1)
                simd_per_cu = val;
            else if (sscanf(line, "domain %u", &val) == 1)
                dom = val;
            else if (sscanf(line, "location_id %u", &val) == 1)
                loc = val;
        }
        fclose(fp);
        if (gfx_ver == 0u)
            continue;
        bool match = have_pci ? (dom == want_dom && loc == want_loc) : (gpu_seen == target_gpu_ord);
        gpu_seen++;
        if (match) {
            if (!out_arch[0]) {
                uint32_t maj = gfx_ver / 10000u;
                uint32_t min = (gfx_ver / 100u) % 100u;
                uint32_t step = gfx_ver % 100u;
                snprintf(out_arch, 32u, "gfx%u%x%x", maj, min, step);
            }
            if (out_phys_cus && simd_cnt > 0u && simd_per_cu > 0u)
                *out_phys_cus = simd_cnt / simd_per_cu;
            return;
        }
    }
}

static bool load_hip_locked(void)
{
    static const char *const hip_libs[] = {"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", NULL};
    static const char *const rtc_libs[] = {"libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so", NULL};
    static const char *const hsa_libs[] = {"libhsa-runtime64.so.1", "libhsa-runtime64.so", NULL};

    memset(&g_hip, 0, sizeof(g_hip));
    g_hip.hip_dl = dlopen_first(hip_libs);
    g_hip.rtc_dl = g_hip.hip_dl ? dlopen_first(rtc_libs) : NULL;
    if (!g_hip.hip_dl || !g_hip.rtc_dl)
        return false;
    g_hip.hsa_dl = dlopen_first(hsa_libs);

    g_hip.get_dev_cnt = dlsym(g_hip.hip_dl, "hipGetDeviceCount");
    g_hip.get_dev = dlsym(g_hip.hip_dl, "hipGetDevice");
    g_hip.set_dev = dlsym(g_hip.hip_dl, "hipSetDevice");
    g_hip.dev_get_attr = dlsym(g_hip.hip_dl, "hipDeviceGetAttribute");
    g_hip.dev_get_pci = dlsym(g_hip.hip_dl, "hipDeviceGetPCIBusId");
    g_hip.stream_prio_range = dlsym(g_hip.hip_dl, "hipDeviceGetStreamPriorityRange");
    g_hip.stream_create_prio = dlsym(g_hip.hip_dl, "hipStreamCreateWithPriority");
    g_hip.stream_create_cumask = dlsym(g_hip.hip_dl, "hipExtStreamCreateWithCUMask");
    g_hip.stream_get_cumask = dlsym(g_hip.hip_dl, "hipExtStreamGetCUMask");
    g_hip.ptr_attr = dlsym(g_hip.hip_dl, "hipPointerGetAttribute");
    g_hip.stream_sync = dlsym(g_hip.hip_dl, "hipStreamSynchronize");
    g_hip.stream_query = dlsym(g_hip.hip_dl, "hipStreamQuery");
    g_hip.stream_destroy = dlsym(g_hip.hip_dl, "hipStreamDestroy");
    g_hip.event_create = dlsym(g_hip.hip_dl, "hipEventCreateWithFlags");
    g_hip.event_record = dlsym(g_hip.hip_dl, "hipEventRecord");
    g_hip.event_query = dlsym(g_hip.hip_dl, "hipEventQuery");
    g_hip.event_destroy = dlsym(g_hip.hip_dl, "hipEventDestroy");
    g_hip.host_malloc = dlsym(g_hip.hip_dl, "hipHostMalloc");
    g_hip.host_free = dlsym(g_hip.hip_dl, "hipHostFree");
    g_hip.dev_malloc = dlsym(g_hip.hip_dl, "hipMalloc");
    g_hip.dev_free = dlsym(g_hip.hip_dl, "hipFree");
    g_hip.memcpy_async = dlsym(g_hip.hip_dl, "hipMemcpyAsync");
    g_hip.mod_load_data = dlsym(g_hip.hip_dl, "hipModuleLoadData");
    g_hip.mod_unload = dlsym(g_hip.hip_dl, "hipModuleUnload");
    g_hip.mod_get_func = dlsym(g_hip.hip_dl, "hipModuleGetFunction");
    g_hip.launch_kernel = dlsym(g_hip.hip_dl, "hipModuleLaunchKernel");

    g_hip.rtc_create = dlsym(g_hip.rtc_dl, "hiprtcCreateProgram");
    g_hip.rtc_compile = dlsym(g_hip.rtc_dl, "hiprtcCompileProgram");
    g_hip.rtc_code_sz = dlsym(g_hip.rtc_dl, "hiprtcGetCodeSize");
    g_hip.rtc_code = dlsym(g_hip.rtc_dl, "hiprtcGetCode");
    g_hip.rtc_destroy = dlsym(g_hip.rtc_dl, "hiprtcDestroyProgram");
    if (g_hip.hsa_dl)
        g_hip.hsa_export_dmabuf = dlsym(g_hip.hsa_dl, "hsa_amd_portable_export_dmabuf");

    if (!g_hip.get_dev_cnt || !g_hip.get_dev || !g_hip.set_dev || !g_hip.ptr_attr || !g_hip.stream_create_prio ||
        !g_hip.stream_sync || !g_hip.stream_query || !g_hip.stream_destroy || !g_hip.event_create || !g_hip.event_record ||
        !g_hip.event_query || !g_hip.event_destroy || !g_hip.host_malloc || !g_hip.host_free || !g_hip.dev_malloc ||
        !g_hip.dev_free || !g_hip.memcpy_async || !g_hip.mod_load_data || !g_hip.mod_unload || !g_hip.mod_get_func ||
        !g_hip.launch_kernel || !g_hip.rtc_create || !g_hip.rtc_compile || !g_hip.rtc_code_sz || !g_hip.rtc_code ||
        !g_hip.rtc_destroy)
        return false;

    return g_hip.get_dev_cnt(&g_device_count) == 0 && g_device_count > 0;
}

static bool init_hip_device_locked(gpurt_device_t *device)
{
    int dev = device->caps.device_ordinal;
    char pci_bdf[32] = {0};
    if (g_hip.dev_get_pci)
        (void)g_hip.dev_get_pci(pci_bdf, sizeof(pci_bdf), dev);

    char arch[32] = {0};
    uint32_t phys_cus = 0u;
    probe_kfd_gpu_props(dev, pci_bdf, arch, &phys_cus);

    int physical_units = 0;
    if (g_hip.dev_get_attr && g_hip.dev_get_attr(&physical_units, 63 /* hipDeviceAttributeMultiprocessorCount */, dev) == 0 &&
        physical_units > 0)
        phys_cus = (uint32_t)physical_units;

    const char *cached = arch[0] ? bin_cache_find_locked(arch, NULL) : NULL;
    bool mod_ok = false;
    if (cached)
        mod_ok = (g_hip.mod_load_data(&device->mod, cached) == 0);
    if (!mod_ok) {
        void *prog = NULL;
        if (g_hip.rtc_create(&prog, g_vxfec_gpu_kernels_src, "vxfec_fec.hip", 0, NULL, NULL) != 0 || !prog)
            return false;

        char arch_opt[64];
        const char *opts[2];
        int nopts = 0;
        if (arch[0]) {
            snprintf(arch_opt, sizeof(arch_opt), "--offload-arch=%s", arch);
            opts[nopts++] = arch_opt;
        }
        int comp_rc = g_hip.rtc_compile(prog, nopts, nopts ? opts : NULL);
        if (comp_rc != 0 && nopts > 0) {
            (void)g_hip.rtc_destroy(&prog);
            prog = NULL;
            nopts = 0;
            if (g_hip.rtc_create(&prog, g_vxfec_gpu_kernels_src, "vxfec_fec.hip", 0, NULL, NULL) == 0 && prog)
                comp_rc = g_hip.rtc_compile(prog, 0, NULL);
        }
        size_t code_sz = 0u;
        char *code = NULL;
        if (comp_rc == 0 && g_hip.rtc_code_sz(prog, &code_sz) == 0 && code_sz > 0u) {
            code = (char *)malloc(code_sz);
            if (code && g_hip.rtc_code(prog, code) == 0 && g_hip.mod_load_data(&device->mod, code) == 0) {
                mod_ok = true;
                if (nopts > 0)
                    bin_cache_put_locked(arch, code, code_sz);
            }
        }
        free(code);
        if (prog)
            (void)g_hip.rtc_destroy(&prog);
    }
    if (!mod_ok)
        return false;

    if (g_hip.mod_get_func(&device->fn_staged, device->mod, "vxfec_gf256_staged_kernel") != 0 ||
        g_hip.mod_get_func(&device->fn_vram_enc, device->mod, "vxfec_gf256_vram_enc_kernel") != 0 ||
        g_hip.mod_get_func(&device->fn_vram_rep, device->mod, "vxfec_gf256_vram_rep_kernel") != 0)
        return false;

    const char *cu_env = getenv("VXFEC_NCCL_CU_COUNT");
    int want_cus = (cu_env && *cu_env) ? atoi(cu_env) : 0;
    uint32_t nwords = (phys_cus + 31u) / 32u;
    if (g_hip.stream_create_cumask && g_hip.stream_get_cumask && want_cus > 0 && (uint32_t)want_cus < phys_cus && nwords <= 16u) {
        uint32_t mask[16] = {0}, actual[16] = {0};
        for (uint32_t c = phys_cus - (uint32_t)want_cus; c < phys_cus; c++)
            mask[c >> 5u] |= (1u << (c & 31u));
        if (g_hip.stream_create_cumask(&device->stream, nwords, mask) == 0 && device->stream &&
            g_hip.stream_get_cumask(device->stream, nwords, actual) == 0 && memcmp(mask, actual, nwords * sizeof(mask[0])) == 0) {
            device->caps.reserved_units = (uint32_t)want_cus;
            device->caps.used_sm_partition = true;
        } else if (device->stream) {
            (void)g_hip.stream_destroy(device->stream);
            device->stream = NULL;
        }
    }
    if (!device->stream) {
        int lo_p = 0, hi_p = 0;
        if (g_hip.stream_prio_range)
            (void)g_hip.stream_prio_range(&lo_p, &hi_p);
        if (g_hip.stream_create_prio(&device->stream, 0x01u /* hipStreamNonBlocking */, hi_p) != 0 || !device->stream)
            return false;
        device->caps.uses_high_priority_stream = (hi_p < lo_p);
    }
    (void)g_hip.stream_create_prio(&device->copy_stream, 0x01u /* hipStreamNonBlocking */, 0);

    snprintf(device->caps.backend_name, sizeof(device->caps.backend_name), "hip");
    snprintf(device->caps.arch_name, sizeof(device->caps.arch_name), "%s", arch[0] ? arch : "amdgcn");
    device->caps.device_ordinal = dev;
    device->caps.total_units = phys_cus;
    device->caps.has_dmabuf_export = (g_hip.hsa_export_dmabuf != NULL);
    return true;
}

static bool load_cuda_locked(void)
{
    static const char *const cu_libs[] = {"libcuda.so.1", "libcuda.so", NULL};
    static const char *const nvrtc_libs[] = {"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so", NULL};

    memset(&g_cu, 0, sizeof(g_cu));
    g_cu.cu_dl = dlopen_first(cu_libs);
    g_cu.rtc_dl = g_cu.cu_dl ? dlopen_first(nvrtc_libs) : NULL;
    if (!g_cu.cu_dl || !g_cu.rtc_dl)
        return false;

    g_cu.init = dlsym(g_cu.cu_dl, "cuInit");
    g_cu.dev_cnt = dlsym(g_cu.cu_dl, "cuDeviceGetCount");
    g_cu.dev_get = dlsym(g_cu.cu_dl, "cuDeviceGet");
    g_cu.dev_attr = dlsym(g_cu.cu_dl, "cuDeviceGetAttribute");
    g_cu.ctx_get_dev = dlsym(g_cu.cu_dl, "cuCtxGetDevice");
    g_cu.ctx_push = dlsym(g_cu.cu_dl, "cuCtxPushCurrent_v2");
    g_cu.ctx_pop = dlsym(g_cu.cu_dl, "cuCtxPopCurrent_v2");
    g_cu.ptr_attrs = dlsym(g_cu.cu_dl, "cuPointerGetAttributes");
    g_cu.primary_retain = dlsym(g_cu.cu_dl, "cuDevicePrimaryCtxRetain");
    g_cu.primary_release = dlsym(g_cu.cu_dl, "cuDevicePrimaryCtxRelease_v2");
    if (!g_cu.primary_release)
        g_cu.primary_release = dlsym(g_cu.cu_dl, "cuDevicePrimaryCtxRelease");
    g_cu.prio_range = dlsym(g_cu.cu_dl, "cuCtxGetStreamPriorityRange");
    g_cu.stream_create_prio = dlsym(g_cu.cu_dl, "cuStreamCreateWithPriority");
    g_cu.stream_sync = dlsym(g_cu.cu_dl, "cuStreamSynchronize");
    g_cu.stream_query = dlsym(g_cu.cu_dl, "cuStreamQuery");
    g_cu.stream_destroy = dlsym(g_cu.cu_dl, "cuStreamDestroy_v2");
    if (!g_cu.stream_destroy)
        g_cu.stream_destroy = dlsym(g_cu.cu_dl, "cuStreamDestroy");

#ifdef VXFEC_HAVE_CUDA_GREEN
    g_cu.dev_get_res = dlsym(g_cu.cu_dl, "cuDeviceGetDevResource");
    g_cu.sm_split = dlsym(g_cu.cu_dl, "cuDevSmResourceSplitByCount");
    g_cu.res_gen_desc = dlsym(g_cu.cu_dl, "cuDevResourceGenerateDesc");
    g_cu.green_create = dlsym(g_cu.cu_dl, "cuGreenCtxCreate");
    g_cu.green_destroy = dlsym(g_cu.cu_dl, "cuGreenCtxDestroy");
    g_cu.green_get_res = dlsym(g_cu.cu_dl, "cuGreenCtxGetDevResource");
    g_cu.green_get_ctx = dlsym(g_cu.cu_dl, "cuCtxFromGreenCtx");
    g_cu.stream_get_green = dlsym(g_cu.cu_dl, "cuStreamGetGreenCtx");
#endif

    g_cu.event_create = dlsym(g_cu.cu_dl, "cuEventCreate");
    g_cu.event_record = dlsym(g_cu.cu_dl, "cuEventRecord");
    g_cu.event_query = dlsym(g_cu.cu_dl, "cuEventQuery");
    g_cu.event_destroy = dlsym(g_cu.cu_dl, "cuEventDestroy_v2");
    if (!g_cu.event_destroy)
        g_cu.event_destroy = dlsym(g_cu.cu_dl, "cuEventDestroy");

    g_cu.host_alloc = dlsym(g_cu.cu_dl, "cuMemHostAlloc");
    g_cu.host_free = dlsym(g_cu.cu_dl, "cuMemFreeHost");
    g_cu.dev_alloc = dlsym(g_cu.cu_dl, "cuMemAlloc_v2");
    g_cu.dev_free = dlsym(g_cu.cu_dl, "cuMemFree_v2");
    g_cu.dev_alloc_async = dlsym(g_cu.cu_dl, "cuMemAllocAsync");
    g_cu.dev_free_async = dlsym(g_cu.cu_dl, "cuMemFreeAsync");
    g_cu.mem_get_addr_range = dlsym(g_cu.cu_dl, "cuMemGetAddressRange_v2");
    if (!g_cu.mem_get_addr_range)
        g_cu.mem_get_addr_range = dlsym(g_cu.cu_dl, "cuMemGetAddressRange");

    g_cu.memcpy_htod_async = dlsym(g_cu.cu_dl, "cuMemcpyHtoDAsync_v2");
    g_cu.memcpy_dtoh_async = dlsym(g_cu.cu_dl, "cuMemcpyDtoHAsync_v2");
    g_cu.mem_get_dmabuf = dlsym(g_cu.cu_dl, "cuMemGetHandleForAddressRange");
    g_cu.mod_load_data = dlsym(g_cu.cu_dl, "cuModuleLoadData");
    g_cu.mod_unload = dlsym(g_cu.cu_dl, "cuModuleUnload");
    g_cu.mod_get_func = dlsym(g_cu.cu_dl, "cuModuleGetFunction");
    g_cu.launch_kernel = dlsym(g_cu.cu_dl, "cuLaunchKernel");

    g_cu.rtc_create = dlsym(g_cu.rtc_dl, "nvrtcCreateProgram");
    g_cu.rtc_compile = dlsym(g_cu.rtc_dl, "nvrtcCompileProgram");
    g_cu.rtc_ptx_sz = dlsym(g_cu.rtc_dl, "nvrtcGetPTXSize");
    g_cu.rtc_ptx = dlsym(g_cu.rtc_dl, "nvrtcGetPTX");
    g_cu.rtc_destroy = dlsym(g_cu.rtc_dl, "nvrtcDestroyProgram");

    if (!g_cu.init || !g_cu.dev_cnt || !g_cu.dev_get || !g_cu.dev_attr || !g_cu.ctx_get_dev || !g_cu.ctx_push || !g_cu.ctx_pop ||
        !g_cu.ptr_attrs || !g_cu.primary_release || !g_cu.primary_retain || !g_cu.stream_create_prio || !g_cu.stream_sync ||
        !g_cu.stream_query || !g_cu.stream_destroy || !g_cu.event_create || !g_cu.event_record || !g_cu.event_query ||
        !g_cu.event_destroy || !g_cu.host_alloc || !g_cu.host_free || !g_cu.dev_alloc || !g_cu.dev_free ||
        !g_cu.memcpy_htod_async || !g_cu.memcpy_dtoh_async || !g_cu.mod_load_data || !g_cu.mod_unload || !g_cu.mod_get_func ||
        !g_cu.launch_kernel || !g_cu.rtc_create || !g_cu.rtc_compile || !g_cu.rtc_ptx_sz || !g_cu.rtc_ptx || !g_cu.rtc_destroy)
        return false;

    return g_cu.init(0u) == 0 && g_cu.dev_cnt(&g_device_count) == 0 && g_device_count > 0;
}

static bool setup_cuda_stream_locked(gpurt_device_t *device, const char *ptx, int priority)
{
    bool ok = g_cu.mod_load_data(&device->mod, ptx) == 0 &&
              g_cu.mod_get_func(&device->fn_staged, device->mod, "vxfec_gf256_staged_kernel") == 0 &&
              g_cu.mod_get_func(&device->fn_vram_enc, device->mod, "vxfec_gf256_vram_enc_kernel") == 0 &&
              g_cu.mod_get_func(&device->fn_vram_rep, device->mod, "vxfec_gf256_vram_rep_kernel") == 0 &&
              g_cu.stream_create_prio(&device->stream, 0x1u /* CU_STREAM_NON_BLOCKING */, priority) == 0 && device->stream;
    if (ok)
        (void)g_cu.stream_create_prio(&device->copy_stream, 0x1u /* CU_STREAM_NON_BLOCKING */, 0);
    return ok;
}

#ifdef VXFEC_HAVE_CUDA_GREEN
static bool setup_cuda_green_locked(gpurt_device_t *device, int dev, const char *ptx, int priority, unsigned int want_sms)
{
    CUdevResource all = {0}, fec = {0}, remaining = {0}, actual = {0};
    unsigned int groups = 1u;
    CUdevResourceDesc desc = NULL;
    CUcontext context = NULL;
    if (!g_cu.dev_get_res || !g_cu.sm_split || !g_cu.res_gen_desc || !g_cu.green_create || !g_cu.green_destroy ||
        !g_cu.green_get_res || !g_cu.green_get_ctx || !g_cu.stream_get_green ||
        g_cu.dev_get_res(dev, &all, CU_DEV_RESOURCE_TYPE_SM) != CUDA_SUCCESS ||
        g_cu.sm_split(&fec, &groups, &all, &remaining, 0u, want_sms) != CUDA_SUCCESS || groups != 1u ||
        g_cu.res_gen_desc(&desc, &fec, 1u) != CUDA_SUCCESS ||
        g_cu.green_create(&device->cu_green, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM) != CUDA_SUCCESS)
        return false;

    CUgreenCtx stream_green = NULL;
    if (g_cu.green_get_res(device->cu_green, &actual, CU_DEV_RESOURCE_TYPE_SM) == CUDA_SUCCESS &&
        actual.type == CU_DEV_RESOURCE_TYPE_SM && actual.sm.smCount >= want_sms && actual.sm.smCount < all.sm.smCount &&
        g_cu.green_get_ctx(&context, device->cu_green) == CUDA_SUCCESS && context && g_cu.ctx_push(context) == 0) {
        bool ok = setup_cuda_stream_locked(device, ptx, priority) &&
                  g_cu.stream_get_green((CUstream)device->stream, &stream_green) == CUDA_SUCCESS &&
                  stream_green == device->cu_green;
        if (!ok) {
            if (device->copy_stream)
                (void)g_cu.stream_destroy(device->copy_stream);
            if (device->stream)
                (void)g_cu.stream_destroy(device->stream);
            if (device->mod)
                (void)g_cu.mod_unload(device->mod);
            device->copy_stream = NULL;
            device->stream = NULL;
            device->mod = NULL;
        }
        vxf_gpu_ctx_t popped;
        (void)g_cu.ctx_pop(&popped);
        if (ok) {
            device->cu_ctx = context;
            device->caps.used_sm_partition = true;
            device->caps.reserved_units = actual.sm.smCount;
            return true;
        }
    }
    (void)g_cu.green_destroy(device->cu_green);
    device->cu_green = NULL;
    return false;
}
#endif

static bool init_cuda_device_locked(gpurt_device_t *device)
{
    int dev = device->cu_device;
    int cc_maj = 8, cc_min = 0, total_sms = 0, dmabuf_ok = 0;
    (void)g_cu.dev_attr(&cc_maj, 75 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */, dev);
    (void)g_cu.dev_attr(&cc_min, 76 /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR */, dev);
    (void)g_cu.dev_attr(&total_sms, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, dev);
    (void)g_cu.dev_attr(&dmabuf_ok, 124 /* CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED */, dev);

    char cache_key[32];
    snprintf(cache_key, sizeof(cache_key), "compute_%d%d", cc_maj, cc_min);
    size_t cached_sz = 0u;
    const char *cached_ptx = bin_cache_find_locked(cache_key, &cached_sz);
    char *ptx_owned = NULL;
    const char *ptx = cached_ptx;
    if (!ptx) {
        void *prog = NULL;
        if (g_cu.rtc_create(&prog, g_vxfec_gpu_kernels_src, "vxfec_fec.cu", 0, NULL, NULL) != 0 || !prog)
            return false;

        char arch_opt[64];
        snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=%s", cache_key);
        const char *opts[1] = {arch_opt};
        int comp_rc = g_cu.rtc_compile(prog, 1, opts);
        if (comp_rc != 0) {
            if (prog)
                (void)g_cu.rtc_destroy(&prog);
            prog = NULL;
            if (g_cu.rtc_create(&prog, g_vxfec_gpu_kernels_src, "vxfec_fec.cu", 0, NULL, NULL) == 0 && prog) {
                opts[0] = "--gpu-architecture=compute_80";
                comp_rc = g_cu.rtc_compile(prog, 1, opts);
            }
        }
        size_t ptx_sz = 0u;
        if (comp_rc == 0 && g_cu.rtc_ptx_sz(prog, &ptx_sz) == 0 && ptx_sz > 0u) {
            ptx_owned = (char *)malloc(ptx_sz);
            if (ptx_owned && g_cu.rtc_ptx(prog, ptx_owned) != 0) {
                free(ptx_owned);
                ptx_owned = NULL;
            } else if (ptx_owned) {
                bin_cache_put_locked(cache_key, ptx_owned, ptx_sz);
                ptx = ptx_owned;
            }
        }
        if (prog)
            (void)g_cu.rtc_destroy(&prog);
    }
    if (!ptx)
        return false;

    int lo_p = 0, hi_p = 0;
    if (g_cu.prio_range)
        (void)g_cu.prio_range(&lo_p, &hi_p);

    bool ok = false;
#ifdef VXFEC_HAVE_CUDA_GREEN
    const char *sm_env = getenv("VXFEC_NCCL_GREEN_SMS");
    int want_sms = (sm_env && *sm_env) ? atoi(sm_env) : 8;
    if (want_sms > 0 && want_sms < total_sms)
        ok = setup_cuda_green_locked(device, dev, ptx, hi_p, (unsigned int)want_sms);
#endif
    if (!ok)
        ok = setup_cuda_stream_locked(device, ptx, hi_p);
    free(ptx_owned);
    if (!ok)
        return false;

    snprintf(device->caps.backend_name, sizeof(device->caps.backend_name), "cuda");
    snprintf(device->caps.arch_name, sizeof(device->caps.arch_name), "sm_%d%d", cc_maj, cc_min);
    device->caps.total_units = (total_sms > 0) ? (uint32_t)total_sms : 0u;
    device->caps.uses_high_priority_stream = (hi_p < lo_p);
    device->caps.has_dmabuf_export = (dmabuf_ok != 0 && g_cu.mem_get_dmabuf != NULL);
    return true;
}

static void free_gpu_mem_locked(void *ptr, bool is_host)
{
    if (!ptr)
        return;
    if (g_kind == VXF_GPURT_HIP) {
        if (is_host)
            (void)g_hip.host_free(ptr);
        else
            (void)g_hip.dev_free(ptr);
    } else {
        if (is_host)
            (void)g_cu.host_free(ptr);
        else
            (void)g_cu.dev_free((vxf_cu_devptr_t)(uintptr_t)ptr);
    }
}

static void cleanup_device_resources_locked(gpurt_device_t *device)
{
    if (device->stream) {
        if (g_kind == VXF_GPURT_HIP)
            (void)g_hip.stream_sync(device->stream);
        else
            (void)g_cu.stream_sync(device->stream);
    }
    if (device->copy_stream) {
        if (g_kind == VXF_GPURT_HIP)
            (void)g_hip.stream_sync(device->copy_stream);
        else
            (void)g_cu.stream_sync(device->copy_stream);
    }

    for (uint32_t i = 0; i < VXFEC_GPURT_MAX_POOL; i++) {
        gpurt_slot_t *s = &device->pool[i];
        if (s->ev) {
            if (g_kind == VXF_GPURT_HIP)
                (void)g_hip.event_destroy(s->ev);
            else
                (void)g_cu.event_destroy(s->ev);
        }
        free_gpu_mem_locked(s->d_in, false);
        free_gpu_mem_locked(s->d_out, false);
        free_gpu_mem_locked(s->d_meta, false);
        free_gpu_mem_locked(s->in_stage, true);
        free_gpu_mem_locked(s->out_stage, true);
        free_gpu_mem_locked(s->h_meta, true);
        free(s->y_scratch);
    }
    free_gpu_mem_locked(device->bounce_host, true);
    while (device->retired_head) {
        gpurt_retired_t *r = device->retired_head;
        device->retired_head = r->next;
        free_gpu_mem_locked(r->ptr, r->is_host);
        free(r);
    }

    if (g_kind == VXF_GPURT_HIP) {
        if (device->copy_stream)
            (void)g_hip.stream_destroy(device->copy_stream);
        if (device->stream)
            (void)g_hip.stream_destroy(device->stream);
        if (device->mod)
            (void)g_hip.mod_unload(device->mod);
    } else {
        if (device->copy_stream)
            (void)g_cu.stream_destroy(device->copy_stream);
        if (device->stream)
            (void)g_cu.stream_destroy(device->stream);
        if (device->mod)
            (void)g_cu.mod_unload(device->mod);
    }
}

static void cleanup_device_locked(gpurt_device_t *device)
{
    gpurt_binding_t binding = {0};
    if (bind_device_locked(device, &binding)) {
        cleanup_device_resources_locked(device);
        unbind_device_locked(binding);
    }
#ifdef VXFEC_HAVE_CUDA_GREEN
    if (device->cu_green)
        (void)g_cu.green_destroy(device->cu_green);
#endif
    if (device->cu_primary)
        (void)g_cu.primary_release(device->cu_device);
    memset(device, 0, sizeof(*device));
}

static void gpurt_cleanup_locked(void)
{
    bool same_pid = (g_init_pid == 0 || getpid() == g_init_pid);
    if (g_devices) {
        if (same_pid) {
            for (int i = 0; i < g_device_count; i++) {
                if (g_devices[i].caps.kind != VXF_GPURT_NONE)
                    cleanup_device_locked(&g_devices[i]);
            }
        }
        free(g_devices);
        g_devices = NULL;
    }
    for (uint32_t i = 0u; i < g_bin_cache_cnt; i++)
        free(g_bin_cache[i].code);
    memset(g_bin_cache, 0, sizeof(g_bin_cache));
    g_bin_cache_cnt = 0u;
    if (g_hip.hsa_dl)
        dlclose(g_hip.hsa_dl);
    if (g_hip.rtc_dl)
        dlclose(g_hip.rtc_dl);
    if (g_hip.hip_dl)
        dlclose(g_hip.hip_dl);
    if (g_cu.rtc_dl)
        dlclose(g_cu.rtc_dl);
    if (g_cu.cu_dl)
        dlclose(g_cu.cu_dl);
    memset(&g_hip, 0, sizeof(g_hip));
    memset(&g_cu, 0, sizeof(g_cu));
    memset(g_slot_owner, 0, sizeof(g_slot_owner));
    memset(g_slot_last_device, 0, sizeof(g_slot_last_device));
    g_kind = VXF_GPURT_NONE;
    g_device_count = 0;
    g_default_device = 0;
    g_init_pid = 0;
}

static int current_ordinal_locked(void)
{
    int dev;
    if (g_kind == VXF_GPURT_HIP)
        return g_hip.get_dev(&dev) == 0 ? dev : g_default_device;
    if (g_cu.ctx_get_dev(&dev) != 0)
        return g_default_device;
    return dev;
}

static bool rt_realloc_host_locked(gpurt_device_t *device, void **ptr, size_t *cap, size_t need, bool allow_sync);
static bool rt_realloc_dev_locked(gpurt_device_t *device, void **ptr, size_t *cap, size_t need, bool allow_sync);

static void prewarm_device_pool_locked(gpurt_device_t *device)
{
    const size_t init_stage = 1024u * 1024u;
    const size_t init_meta = 256u * 1024u;
    const size_t init_bounce = 4u * 1024u * 1024u;
    (void)rt_realloc_host_locked(device, &device->bounce_host, &device->bounce_cap, init_bounce, true);
    for (uint32_t i = 0; i < VXFEC_GPURT_MAX_POOL; i++) {
        gpurt_slot_t *s = &device->pool[i];
        if (!s->ev) {
            int rc = (g_kind == VXF_GPURT_HIP) ? g_hip.event_create(&s->ev, 0x2u /* hipEventDisableTiming */)
                                               : g_cu.event_create(&s->ev, 0x2u /* CU_EVENT_DISABLE_TIMING */);
            if (rc != 0)
                s->ev = NULL;
        }
        (void)rt_realloc_host_locked(device, &s->in_stage, &s->in_cap, init_stage, true);
        (void)rt_realloc_host_locked(device, &s->out_stage, &s->out_cap, init_stage, true);
        (void)rt_realloc_dev_locked(device, &s->d_in, &s->d_in_cap, init_stage, true);
        if (rt_realloc_dev_locked(device, &s->d_out, &s->d_out_cap, init_stage, true) && s->d_out_id == 0u) {
            s->d_out_id = g_next_out_id++;
            if (g_next_out_id == 0u)
                g_next_out_id = 1u;
        }
        (void)rt_realloc_host_locked(device, &s->h_meta, &s->h_meta_cap, init_meta, true);
        (void)rt_realloc_dev_locked(device, &s->d_meta, &s->d_meta_cap, init_meta, true);
    }
}

static gpurt_device_t *get_device_locked(int ordinal, vxf_gpu_ctx_t owner)
{
    if (!g_devices || ordinal < 0 || ordinal >= g_device_count)
        return NULL;
    gpurt_device_t *device = &g_devices[ordinal];
    if (device->caps.kind != VXF_GPURT_NONE)
        return (!owner || device->cu_primary == owner) ? device : NULL;

    device->caps.device_ordinal = ordinal;
    device->caps.kind = g_kind;
    if (g_kind == VXF_GPURT_CUDA) {
        if (g_cu.dev_get(&device->cu_device, ordinal) != 0 || g_cu.primary_retain(&device->cu_primary, device->cu_device) != 0 ||
            !device->cu_primary) {
            memset(device, 0, sizeof(*device));
            return NULL;
        }
        device->cu_ctx = device->cu_primary;
        if (owner && device->cu_primary != owner) {
            (void)g_cu.primary_release(device->cu_device);
            memset(device, 0, sizeof(*device));
            return NULL;
        }
    }

    gpurt_binding_t binding = {0};
    bool bound = bind_device_locked(device, &binding);
    bool ok = bound && (g_kind == VXF_GPURT_HIP ? init_hip_device_locked(device) : init_cuda_device_locked(device));
    if (ok)
        prewarm_device_pool_locked(device);
    if (bound && !ok)
        cleanup_device_resources_locked(device);
    unbind_device_locked(binding);
    if (!ok) {
        if (device->cu_primary)
            (void)g_cu.primary_release(device->cu_device);
        memset(device, 0, sizeof(*device));
        return NULL;
    }
    return device;
}

static bool pointer_owner_locked(const void *ptr, int *ordinal, vxf_gpu_ctx_t *owner)
{
    unsigned int type = 0u;
    *ordinal = -1;
    *owner = NULL;
    if (g_kind == VXF_GPURT_HIP)
        return vxf_hip_device_ordinal(g_hip.ptr_attr, ptr, ordinal);
    /* async and virtual-memory allocations can have an ordinal without a context handle */
    int attrs[] = {1 /* CU_POINTER_ATTRIBUTE_CONTEXT */, 2 /* CU_POINTER_ATTRIBUTE_MEMORY_TYPE */,
                   9 /* CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL */};
    void *outputs[] = {owner, &type, ordinal};
    vxf_cu_devptr_t address = (vxf_cu_devptr_t)(uintptr_t)ptr;
    return g_cu.ptr_attrs(3u, attrs, outputs, address) == 0 && type == 2u /* CU_MEMORYTYPE_DEVICE */;
}

static gpurt_device_t *get_pointer_device_locked(const void *ptr)
{
    int ordinal;
    vxf_gpu_ctx_t owner;
    if (!g_devices || !pointer_owner_locked(ptr, &ordinal, &owner))
        return NULL;
    return get_device_locked(ordinal, owner);
}

static bool start_backend_locked(vxf_gpurt_kind_t kind)
{
    bool loaded = kind == VXF_GPURT_HIP ? load_hip_locked() : load_cuda_locked();
    if (loaded) {
        g_kind = kind;
        g_init_pid = getpid();
        g_devices = calloc((size_t)g_device_count, sizeof(*g_devices));
        g_default_device = current_ordinal_locked();
        if (get_device_locked(g_default_device, NULL))
            return true;
    }
    gpurt_cleanup_locked();
    return false;
}

bool vxf_gpurt_init(vxf_gpurt_caps_t *out_caps)
{
    pthread_mutex_lock(&g_rt_mtx);
    if (!g_devices) {
        nanorq_core_init();
        const char *be = getenv("VXFEC_NCCL_GPU_BACKEND");
        bool automatic = !be || !*be || strcmp(be, "auto") == 0 || strcmp(be, "native") == 0;
        bool try_hip = automatic || strcmp(be, "hip") == 0;
        bool try_cuda = automatic || strcmp(be, "cuda") == 0;
        bool prefer_cuda = automatic && is_any_lib_resident(g_resident_cuda_libs) && !is_any_lib_resident(g_resident_hip_libs);
        if (prefer_cuda) {
            if ((!try_cuda || !start_backend_locked(VXF_GPURT_CUDA)) && try_hip)
                (void)start_backend_locked(VXF_GPURT_HIP);
        } else if ((!try_hip || !start_backend_locked(VXF_GPURT_HIP)) && try_cuda) {
            (void)start_backend_locked(VXF_GPURT_CUDA);
        }
    }

    bool ok = g_devices != NULL;
    if (ok && out_caps)
        *out_caps = g_devices[g_default_device].caps;
    pthread_mutex_unlock(&g_rt_mtx);
    return ok;
}

void vxf_gpurt_shutdown(void)
{
    pthread_mutex_lock(&g_rt_mtx);
    gpurt_cleanup_locked();
    pthread_mutex_unlock(&g_rt_mtx);
}

bool vxf_gpurt_wait_idle(void)
{
    pthread_mutex_lock(&g_rt_mtx);
    if (g_init_pid != 0 && getpid() != g_init_pid) {
        pthread_mutex_unlock(&g_rt_mtx);
        return true;
    }
    bool ok = true;
    for (int i = 0; g_devices && i < g_device_count; i++) {
        gpurt_device_t *device = &g_devices[i];
        if (!device->stream)
            continue;
        gpurt_binding_t binding = {0};
        if (!bind_device_locked(device, &binding)) {
            ok = false;
            continue;
        }
        int rc = g_kind == VXF_GPURT_HIP ? g_hip.stream_sync(device->stream) : g_cu.stream_sync(device->stream);
        if (rc == 0 && device->copy_stream)
            rc = g_kind == VXF_GPURT_HIP ? g_hip.stream_sync(device->copy_stream) : g_cu.stream_sync(device->copy_stream);
        if (rc != 0) {
            ok = false;
        } else {
            while (device->retired_head) {
                gpurt_retired_t *r = device->retired_head;
                device->retired_head = r->next;
                free_gpu_mem_locked(r->ptr, r->is_host);
                free(r);
            }
        }
        unbind_device_locked(binding);
    }
    pthread_mutex_unlock(&g_rt_mtx);
    return ok;
}

bool vxf_gpurt_get_caps(vxf_gpurt_caps_t *out_caps)
{
    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = g_devices ? get_device_locked(current_ordinal_locked(), NULL) : NULL;
    if (device && out_caps)
        *out_caps = device->caps;
    pthread_mutex_unlock(&g_rt_mtx);
    return device != NULL;
}

bool vxf_gpurt_validate_dev_ptr(const void *ptr)
{
    if (!ptr)
        return false;
    pthread_mutex_lock(&g_rt_mtx);
    bool ok = get_pointer_device_locked(ptr) != NULL;
    pthread_mutex_unlock(&g_rt_mtx);
    return ok;
}

static int export_dmabuf_locked(const void *dev_ptr, size_t size, uint64_t *out_offset)
{
    if (out_offset)
        *out_offset = 0ULL;
    int fd = -1;
    if (g_kind == VXF_GPURT_HIP) {
        uint64_t off = 0ULL;
        if (g_hip.hsa_export_dmabuf(dev_ptr, size, &fd, &off) != 0)
            return -1;
        if (out_offset)
            *out_offset = off;
        return fd;
    }

    uint64_t addr = (uint64_t)(uintptr_t)dev_ptr;
    if (size > UINT64_MAX - addr)
        return -1;
    uint64_t end_req = addr + (uint64_t)size;
    vxf_cu_devptr_t alloc_base = 0u;
    size_t alloc_size = 0u;
    bool have_range = (g_cu.mem_get_addr_range && g_cu.mem_get_addr_range(&alloc_base, &alloc_size, (vxf_cu_devptr_t)addr) == 0 &&
                       alloc_size > 0u && addr >= (uint64_t)alloc_base && end_req <= (uint64_t)alloc_base + (uint64_t)alloc_size);

    long host_pg = sysconf(_SC_PAGESIZE);
    uint64_t pg = (host_pg > 0) ? (uint64_t)host_pg : 4096ULL;
    uint64_t aligns[2] = {(pg < 65536ULL) ? 65536ULL : pg, pg};
    for (int i = 0; i < 2; i++) {
        uint64_t al = aligns[i];
        if (i == 1 && al == aligns[0])
            continue;
        uint64_t start = addr & ~(al - 1ULL);
        if (have_range && start < (uint64_t)alloc_base)
            continue;
        uint64_t end = (end_req + al - 1ULL) & ~(al - 1ULL);
        if (end < end_req)
            continue;
        if (have_range && end > (uint64_t)alloc_base + (uint64_t)alloc_size)
            continue;
        if (g_cu.mem_get_dmabuf(&fd, (vxf_cu_devptr_t)start, (size_t)(end - start), 1 /* CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD */,
                                0ULL) == 0 &&
            fd >= 0) {
            if (out_offset)
                *out_offset = addr - start;
            return fd;
        }
    }
    if (g_cu.mem_get_dmabuf(&fd, (vxf_cu_devptr_t)addr, size, 1 /* CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD */, 0ULL) != 0)
        return -1;
    return fd;
}

int vxf_gpurt_export_dmabuf(const void *dev_ptr, size_t size, uint64_t *out_offset)
{
    if (!dev_ptr || size == 0u)
        return -1;
    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = get_pointer_device_locked(dev_ptr);
    gpurt_binding_t binding = {0};
    int fd = -1;
    if (device && device->caps.has_dmabuf_export && bind_device_locked(device, &binding))
        fd = export_dmabuf_locked(dev_ptr, size, out_offset);
    unlock_device(binding);
    return fd;
}

/*
 * Never call hipFree / hipHostFree / cuMemFree / cuMemFreeHost inline on the
 * proxy thread: both ROCm and CUDA perform an implicit device-wide stream
 * synchronization during unmap, which deadlocks if a persistent NCCL/RCCL
 * collective kernel is actively spinning on the GPU waiting for net->test().
 */
static void rt_retire_ptr_locked(gpurt_device_t *device, void *old_ptr, bool is_host)
{
    if (!old_ptr)
        return;
    gpurt_retired_t *r = (gpurt_retired_t *)malloc(sizeof(*r));
    if (r) {
        r->ptr = old_ptr;
        r->is_host = is_host;
        r->next = device->retired_head;
        device->retired_head = r;
    }
}

static size_t rt_next_cap(size_t cur_cap, size_t need)
{
    size_t ncap = (cur_cap >= 1024u * 1024u) ? (cur_cap * 2u) : (1024u * 1024u);
    if (ncap < need)
        ncap = (need + 262143u) & ~(size_t)262143u;
    return ncap;
}

static bool rt_realloc_host_locked(gpurt_device_t *device, void **ptr, size_t *cap, size_t need, bool allow_sync)
{
    if (need <= *cap && *ptr)
        return true;
    /* avoid synchronizing cuMemHostAlloc while a collective kernel may be active */
    if (g_kind == VXF_GPURT_CUDA && !allow_sync)
        return false;
    size_t ncap = rt_next_cap(*cap, need);
    void *np = NULL;
    if (g_kind == VXF_GPURT_HIP) {
        if (g_hip.host_malloc(&np, ncap, 0u) != 0 || !np)
            return false;
    } else {
        if (g_cu.host_alloc(&np, ncap, 0u) != 0 || !np)
            return false;
    }
    rt_retire_ptr_locked(device, *ptr, true);
    *ptr = np;
    *cap = ncap;
    return true;
}

static bool rt_memcpy_bounce(void *host_buf, void *dev_ptr, size_t bytes, bool is_dtoh)
{
    if (bytes == 0u)
        return true;
    if (!host_buf || !dev_ptr)
        return false;
    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = get_pointer_device_locked(dev_ptr);
    if (!device) {
        pthread_mutex_unlock(&g_rt_mtx);
        return false;
    }
    const size_t max_chunk = 4u * 1024u * 1024u;
    size_t need = (bytes < max_chunk) ? bytes : max_chunk;
    gpurt_binding_t binding = {0};
    if (!bind_device_locked(device, &binding) ||
        !rt_realloc_host_locked(device, &device->bounce_host, &device->bounce_cap, need, false)) {
        unlock_device(binding);
        return false;
    }
    vxf_gpu_stream_t cstream = device->copy_stream ? device->copy_stream : device->stream;
    /* keep the bounce buffer and stream alive, and serialize CUDA green context bindings */
    bool ok = true;
    size_t off = 0u;
    while (off < bytes) {
        size_t chunk = ((bytes - off) < device->bounce_cap) ? (bytes - off) : device->bounce_cap;
        uint8_t *dptr = (uint8_t *)dev_ptr + off;
        uint8_t *hptr = (uint8_t *)host_buf + off;
        if (!is_dtoh)
            memcpy(device->bounce_host, hptr, chunk);
        bool step_ok;
        if (g_kind == VXF_GPURT_HIP) {
            step_ok =
                (is_dtoh ? g_hip.memcpy_async(device->bounce_host, dptr, chunk, 2 /* hipMemcpyDeviceToHost */, cstream)
                         : g_hip.memcpy_async(dptr, device->bounce_host, chunk, 1 /* hipMemcpyHostToDevice */, cstream)) == 0 &&
                vxf_gpurt_finish_copy(cstream, g_hip.stream_sync, g_hip.stream_query);
        } else {
            step_ok =
                (is_dtoh ? g_cu.memcpy_dtoh_async(device->bounce_host, (vxf_cu_devptr_t)(uintptr_t)dptr, chunk, cstream)
                         : g_cu.memcpy_htod_async((vxf_cu_devptr_t)(uintptr_t)dptr, device->bounce_host, chunk, cstream)) == 0 &&
                vxf_gpurt_finish_copy(cstream, g_cu.stream_sync, g_cu.stream_query);
        }
        if (!step_ok) {
            ok = false;
            break;
        }
        if (is_dtoh)
            memcpy(hptr, device->bounce_host, chunk);
        off += chunk;
    }
    unlock_device(binding);
    return ok;
}

bool vxf_gpurt_memcpy_dtoh(void *dst_host, const void *src_dev, size_t bytes)
{
    return rt_memcpy_bounce(dst_host, (void *)(uintptr_t)src_dev, bytes, true);
}

bool vxf_gpurt_memcpy_htod(void *dst_dev, const void *src_host, size_t bytes)
{
    return rt_memcpy_bounce((void *)(uintptr_t)src_host, dst_dev, bytes, false);
}

static bool rt_realloc_dev_locked(gpurt_device_t *device, void **ptr, size_t *cap, size_t need, bool allow_sync)
{
    if (need <= *cap && *ptr)
        return true;
    size_t ncap = rt_next_cap(*cap, need);
    void *np = NULL;
    if (g_kind == VXF_GPURT_HIP) {
        if (g_hip.dev_malloc(&np, ncap) != 0 || !np)
            return false;
    } else {
        vxf_cu_devptr_t dp = 0u;
        if (!allow_sync) {
            /* stream-ordered allocation avoids deadlocking against an active collective kernel */
            if (!g_cu.dev_alloc_async || !device->stream || g_cu.dev_alloc_async(&dp, ncap, device->stream) != 0 || dp == 0u)
                return false;
        } else if (g_cu.dev_alloc(&dp, ncap) != 0 || dp == 0u) {
            return false;
        }
        np = (void *)(uintptr_t)dp;
    }
    rt_retire_ptr_locked(device, *ptr, false);
    *ptr = np;
    *cap = ncap;
    return true;
}

static bool ensure_out_cap_locked(gpurt_device_t *device, gpurt_slot_t *slot, size_t bytes)
{
    if (slot->d_out && bytes <= slot->d_out_cap)
        return true;
    if (!rt_realloc_dev_locked(device, &slot->d_out, &slot->d_out_cap, bytes, false))
        return false;
    slot->d_out_id = g_next_out_id++;
    if (g_next_out_id == 0u)
        g_next_out_id = 1u;
    return true;
}

int vxf_gpurt_pool_acquire(const void *prefer_dev_ptr, size_t in_bytes, size_t out_bytes, void **out_in_stage, void **out_out_stage)
{
    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = NULL;
    if (g_devices)
        device = prefer_dev_ptr ? get_pointer_device_locked(prefer_dev_ptr) : get_device_locked(current_ordinal_locked(), NULL);
    gpurt_binding_t binding = {0};
    if (!device || !bind_device_locked(device, &binding)) {
        unlock_device(binding);
        return -1;
    }

    size_t need_in = in_bytes ? in_bytes : 4096u;
    size_t need_out = out_bytes ? out_bytes : 4096u;
    /* prefer a free slot last used by this device to avoid DMA-BUF MR thrashing across GPUs */
    for (int pass = 0; pass < 2; pass++) {
        for (uint32_t i = 0; i < VXFEC_GPURT_MAX_POOL; i++) {
            if (g_slot_owner[i])
                continue;
            if (pass == 0 && g_slot_last_device[i] != device)
                continue;
            gpurt_slot_t *s = &device->pool[i];
            if (!s->ev) {
                int rc = (g_kind == VXF_GPURT_HIP) ? g_hip.event_create(&s->ev, 0x2u /* hipEventDisableTiming */)
                                                   : g_cu.event_create(&s->ev, 0x2u /* CU_EVENT_DISABLE_TIMING */);
                if (rc != 0 || !s->ev) {
                    unlock_device(binding);
                    return -1;
                }
            }
            if (!rt_realloc_host_locked(device, &s->in_stage, &s->in_cap, need_in, false) ||
                !rt_realloc_host_locked(device, &s->out_stage, &s->out_cap, need_out, false) ||
                !rt_realloc_dev_locked(device, &s->d_in, &s->d_in_cap, need_in, false) ||
                !rt_realloc_host_locked(device, &s->h_meta, &s->h_meta_cap, 256u * 1024u, false) ||
                !rt_realloc_dev_locked(device, &s->d_meta, &s->d_meta_cap, 256u * 1024u, false) ||
                !ensure_out_cap_locked(device, s, need_out)) {
                unlock_device(binding);
                return -1;
            }
            g_slot_owner[i] = device;
            g_slot_last_device[i] = device;
            s->ticket = 0ULL;
            if (out_in_stage)
                *out_in_stage = s->in_stage;
            if (out_out_stage)
                *out_out_stage = s->out_stage;
            unlock_device(binding);
            return (int)i;
        }
    }
    unlock_device(binding);
    return -1;
}

void vxf_gpurt_pool_release(int slot)
{
    if (slot < 0 || slot >= (int)VXFEC_GPURT_MAX_POOL)
        return;
    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = g_slot_owner[slot];
    if (device && device->pool[slot].ticket == 0ULL)
        g_slot_owner[slot] = NULL;
    pthread_mutex_unlock(&g_rt_mtx);
}

uint64_t vxf_gpurt_pool_out_id(int slot)
{
    if (slot < 0 || slot >= (int)VXFEC_GPURT_MAX_POOL)
        return 0u;
    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = g_slot_owner[slot];
    uint64_t id = device ? device->pool[slot].d_out_id : 0u;
    pthread_mutex_unlock(&g_rt_mtx);
    return id;
}

int vxf_gpurt_pool_export_out_dmabuf(int slot, size_t min_bytes, uint64_t *out_offset, size_t *out_cap)
{
    if (out_cap)
        *out_cap = 0u;
    if (slot < 0 || slot >= (int)VXFEC_GPURT_MAX_POOL || min_bytes == 0u)
        return -1;

    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = g_slot_owner[slot];
    gpurt_binding_t binding = {0};
    int fd = -1;
    /* export the existing allocation; the VRAM submit path must use that same allocation */
    if (device && device->caps.has_dmabuf_export && bind_device_locked(device, &binding)) {
        gpurt_slot_t *s = &device->pool[slot];
        if (s->d_out_cap >= min_bytes) {
            fd = export_dmabuf_locked(s->d_out, s->d_out_cap, out_offset);
            if (fd >= 0 && out_cap)
                *out_cap = s->d_out_cap;
        }
    }
    unlock_device(binding);
    return fd;
}

static uint8_t *slot_ensure_y_scratch(gpurt_slot_t *s, size_t need)
{
    if (need <= s->y_scratch_cap && s->y_scratch)
        return s->y_scratch;
    size_t ncap = (need < 65536u) ? 65536u : ((need + 65535u) & ~(size_t)65535u);
    uint8_t *np = (uint8_t *)malloc(ncap);
    if (!np)
        return NULL;
    free(s->y_scratch);
    s->y_scratch = np;
    s->y_scratch_cap = ncap;
    return np;
}

static bool build_rs8_encode_mat_transposed(gpurt_slot_t *s, uint32_t K, uint32_t M, const uint8_t *parity_m, uint32_t count,
                                            uint8_t *mat_out)
{
    if (K == 0u || K > 255u || M == 0u || M > 256u - K || count == 0u || count > M)
        return false;
    uint8_t *Y = slot_ensure_y_scratch(s, (size_t)K * count);
    if (!Y || !vxfec_rs8__encode_matrix(K, M, parity_m, count, Y, count))
        return false;
    for (uint32_t e = 0u; e < count; e++) {
        for (uint32_t k = 0u; k < K; k++)
            mat_out[(size_t)e * K + k] = Y[(size_t)k * count + e];
    }
    return true;
}

static uint8_t *build_rq_adjoint_scratch(gpurt_slot_t *s, nanorq_core *rq, const schedule *S, const uint32_t *esis, uint32_t count,
                                         uint32_t *out_rows, size_t *out_m_stride)
{
    if (!rq || !S || !S->ops.a || S->cpidx != 2 || S->overflowed || S->cp[0] > S->cp[1] || S->cp[1] > S->ops.n || !esis ||
        count == 0u)
        return NULL;
    uint32_t rows = nanorq_core_get_pc_rows(rq);
    size_t align = get_align_size();
    size_t m_stride = ((size_t)count + align - 1u) & ~(align - 1u);
    if (m_stride == 0u)
        m_stride = align;
    uint8_t *Y = slot_ensure_y_scratch(s, (size_t)rows * m_stride);
    if (!Y)
        return NULL;
    vxfec_rq__adjoint_gf256(rq, S->ops.a, (uint32_t)S->ops.n, (uint32_t)S->cp[0], (uint32_t)S->cp[1], rows, esis, count, Y,
                            m_stride);
    *out_rows = rows;
    *out_m_stride = m_stride;
    return Y;
}

static bool rt_htod_async_locked(gpurt_device_t *device, void *dst_dev, const void *src_host, size_t bytes)
{
    if (bytes == 0u)
        return true;
    if (g_kind == VXF_GPURT_HIP)
        return g_hip.memcpy_async(dst_dev, src_host, bytes, 1 /* HtoD */, device->stream) == 0;
    return g_cu.memcpy_htod_async((vxf_cu_devptr_t)(uintptr_t)dst_dev, src_host, bytes, device->stream) == 0;
}

static bool rt_dtoh_async_locked(gpurt_device_t *device, void *dst_host, const void *src_dev, size_t bytes)
{
    if (bytes == 0u)
        return true;
    if (g_kind == VXF_GPURT_HIP)
        return g_hip.memcpy_async(dst_host, src_dev, bytes, 2 /* DtoH */, device->stream) == 0;
    return g_cu.memcpy_dtoh_async(dst_host, (vxf_cu_devptr_t)(uintptr_t)src_dev, bytes, device->stream) == 0;
}

static bool rt_launch_locked(gpurt_device_t *device, vxf_gpu_func_t fn, uint32_t gx, uint32_t gy, void **args)
{
    if (gx == 0u || gy == 0u)
        return true;
    if (g_kind == VXF_GPURT_HIP)
        return g_hip.launch_kernel(fn, gx, gy, 1u, 256u, 1u, 1u, 0u, device->stream, args, NULL) == 0;
    return g_cu.launch_kernel(fn, gx, gy, 1u, 256u, 1u, 1u, 0u, device->stream, args, NULL) == 0;
}

static uint64_t rt_record_ticket_locked(gpurt_device_t *device, gpurt_slot_t *s, bool submit_ok)
{
    int rc = (g_kind == VXF_GPURT_HIP) ? g_hip.event_record(s->ev, device->stream) : g_cu.event_record(s->ev, device->stream);
    /* even a failed submission may have queued part of its work */
    s->event_recorded = rc == 0;
    s->failed = !submit_ok || !s->event_recorded;
    uint64_t t = g_next_ticket++;
    if (g_next_ticket == 0ULL)
        g_next_ticket = 1ULL;
    s->ticket = t;
    return t;
}

uint64_t vxf_gpurt_rs8_submit(int slot, bool is_decode, uint32_t K, uint32_t M, const uint8_t *erased_k, const uint8_t *parity_m,
                              uint32_t count, uint32_t sym_T)
{
    if (slot < 0 || slot >= (int)VXFEC_GPURT_MAX_POOL || K == 0u || K > 255u || count == 0u || sym_T == 0u || (sym_T & 3u) != 0u)
        return 0ULL;

    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = g_slot_owner[slot];
    gpurt_slot_t *s = device ? &device->pool[slot] : NULL;
    gpurt_binding_t binding = {0};
    if (!device || !bind_device_locked(device, &binding)) {
        unlock_device(binding);
        return 0ULL;
    }

    size_t mat_bytes = (size_t)count * K;
    size_t in_bytes = (size_t)K * sym_T;
    size_t out_bytes = (size_t)count * sym_T;
    if (in_bytes > s->in_cap || out_bytes > s->out_cap ||
        !rt_realloc_host_locked(device, &s->h_meta, &s->h_meta_cap, mat_bytes, false) ||
        !rt_realloc_dev_locked(device, &s->d_meta, &s->d_meta_cap, mat_bytes, false) ||
        !rt_realloc_dev_locked(device, &s->d_in, &s->d_in_cap, in_bytes, false) || !ensure_out_cap_locked(device, s, out_bytes)) {
        unlock_device(binding);
        return 0ULL;
    }

    uint8_t *h_mat = (uint8_t *)s->h_meta;
    bool mok = false;
    if (is_decode) {
        uint8_t *Y = slot_ensure_y_scratch(s, mat_bytes);
        if (Y && vxfec_rs8__recovery_matrix(K, M, erased_k, parity_m, count, Y, count)) {
            for (uint32_t e = 0u; e < count; e++) {
                for (uint32_t k = 0u; k < K; k++)
                    h_mat[(size_t)e * K + k] = Y[(size_t)k * count + e];
            }
            mok = true;
        }
    } else {
        mok = build_rs8_encode_mat_transposed(s, K, M, parity_m, count, h_mat);
    }
    if (!mok) {
        unlock_device(binding);
        return 0ULL;
    }

    uint32_t row_beg = 0u;
    uint32_t stride_u32 = sym_T >> 2u;
    uint32_t gx = (stride_u32 + 255u) / 256u;
    uint32_t gy = (count + 7u) / 8u;
    void *args[8] = {&s->d_in, &s->d_out, &s->d_meta, &row_beg, &K, &count, &stride_u32, &stride_u32};

    bool submit_ok = rt_htod_async_locked(device, s->d_in, s->in_stage, in_bytes) &&
                     rt_htod_async_locked(device, s->d_meta, h_mat, mat_bytes) &&
                     rt_launch_locked(device, device->fn_staged, gx, gy, args) &&
                     rt_dtoh_async_locked(device, s->out_stage, s->d_out, out_bytes);
    uint64_t t = rt_record_ticket_locked(device, s, submit_ok);
    unlock_device(binding);
    return t;
}

uint64_t vxf_gpurt_rs8_submit_vram_enc(int slot, const void *src_dev_ptr, size_t src_offset, size_t valid_bytes, uint32_t K,
                                       uint32_t M, const uint8_t *parity_m, uint32_t count, uint32_t sym_T, bool use_vram_out)
{
    if (slot < 0 || slot >= (int)VXFEC_GPURT_MAX_POOL || !src_dev_ptr || K == 0u || K > 255u || count == 0u || sym_T == 0u ||
        (sym_T & 3u) != 0u)
        return 0ULL;

    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = g_slot_owner[slot];
    gpurt_slot_t *s = device ? &device->pool[slot] : NULL;
    gpurt_binding_t binding = {0};
    if (!device || get_pointer_device_locked(src_dev_ptr) != device || !bind_device_locked(device, &binding)) {
        unlock_device(binding);
        return 0ULL;
    }

    size_t mat_bytes = (size_t)count * K;
    size_t out_bytes = (size_t)count * sym_T;
    if (!rt_realloc_host_locked(device, &s->h_meta, &s->h_meta_cap, mat_bytes, false) ||
        !rt_realloc_dev_locked(device, &s->d_meta, &s->d_meta_cap, mat_bytes, false) ||
        (use_vram_out ? out_bytes > s->d_out_cap : (out_bytes > s->out_cap || !ensure_out_cap_locked(device, s, out_bytes))) ||
        !build_rs8_encode_mat_transposed(s, K, M, parity_m, count, (uint8_t *)s->h_meta)) {
        unlock_device(binding);
        return 0ULL;
    }

    const void *src_ptr = (const uint8_t *)src_dev_ptr + src_offset;
    uint64_t vbytes = (uint64_t)valid_bytes;
    uint32_t out_u32 = sym_T >> 2u;
    uint32_t gx = (out_u32 + 255u) / 256u;
    uint32_t gy = (count + 7u) / 8u;
    void *args[8] = {&src_ptr, &vbytes, &s->d_out, &s->d_meta, &K, &count, &sym_T, &out_u32};

    bool submit_ok = rt_htod_async_locked(device, s->d_meta, s->h_meta, mat_bytes) &&
                     rt_launch_locked(device, device->fn_vram_enc, gx, gy, args) &&
                     (use_vram_out || rt_dtoh_async_locked(device, s->out_stage, s->d_out, out_bytes));
    uint64_t t = rt_record_ticket_locked(device, s, submit_ok);
    unlock_device(binding);
    return t;
}

uint64_t vxf_gpurt_rs8_submit_vram_rep(int slot, void *target_dev_ptr, size_t target_offset, size_t total_bytes, uint32_t K,
                                       uint32_t M, const uint8_t *sys_bitmap, const uint8_t *erased_k, const uint8_t *parity_m,
                                       uint32_t count, uint32_t sym_T)
{
    if (slot < 0 || slot >= (int)VXFEC_GPURT_MAX_POOL || !target_dev_ptr || !sys_bitmap || !erased_k || !parity_m || K == 0u ||
        K > 255u || count == 0u || sym_T == 0u || (sym_T & 3u) != 0u)
        return 0ULL;
    uint8_t seen[32] = {0};
    for (uint32_t r = 0u; r < count; r++) {
        uint8_t ek = erased_k[r];
        if ((uint32_t)ek >= K || (sys_bitmap[ek >> 3u] & (1u << (ek & 7u))) != 0u || (seen[ek >> 3u] & (1u << (ek & 7u))) != 0u)
            return 0ULL;
        seen[ek >> 3u] |= (uint8_t)(1u << (ek & 7u));
    }

    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = g_slot_owner[slot];
    gpurt_slot_t *s = device ? &device->pool[slot] : NULL;
    gpurt_binding_t binding = {0};
    if (!device || (size_t)K * sym_T > s->in_cap || get_pointer_device_locked(target_dev_ptr) != device ||
        !bind_device_locked(device, &binding)) {
        unlock_device(binding);
        return 0ULL;
    }

    uint8_t *Y = slot_ensure_y_scratch(s, (size_t)K * count);
    if (!Y || !vxfec_rs8__recovery_matrix(K, M, erased_k, parity_m, count, Y, count)) {
        unlock_device(binding);
        return 0ULL;
    }

    uint32_t mat_cols = K + count;
    size_t miss_bytes = ((size_t)count * sizeof(uint32_t) + 15u) & ~(size_t)15u;
    size_t mat_bytes = (size_t)count * mat_cols;
    size_t meta_total = miss_bytes + mat_bytes;
    size_t rep_bytes = (size_t)count * sym_T;

    if (!rt_realloc_host_locked(device, &s->h_meta, &s->h_meta_cap, meta_total, false) ||
        !rt_realloc_dev_locked(device, &s->d_meta, &s->d_meta_cap, meta_total, false) ||
        !rt_realloc_host_locked(device, &s->out_stage, &s->out_cap, rep_bytes, false) ||
        !rt_realloc_dev_locked(device, &s->d_in, &s->d_in_cap, rep_bytes, false)) {
        unlock_device(binding);
        return 0ULL;
    }

    uint32_t *h_miss = (uint32_t *)s->h_meta;
    uint8_t *h_mat = (uint8_t *)s->h_meta + miss_bytes;
    memset(h_mat, 0, mat_bytes);
    for (uint32_t b = 0u; b < count; b++) {
        h_miss[b] = erased_k[b];
        uint8_t *mrow = h_mat + (size_t)b * mat_cols;
        for (uint32_t k = 0u; k < K; k++) {
            if (sys_bitmap[k >> 3u] & (1u << (k & 7u)))
                mrow[k] = Y[(size_t)k * count + b];
        }
        for (uint32_t r = 0u; r < count; r++)
            mrow[K + r] = Y[(size_t)erased_k[r] * count + b];
    }

    /* pack only the count received parity symbols for compact HtoD upload */
    const uint8_t *in_st = (const uint8_t *)s->in_stage;
    uint8_t *rep_pack = (uint8_t *)s->out_stage;
    for (uint32_t r = 0u; r < count; r++)
        memcpy(rep_pack + (size_t)r * sym_T, in_st + (size_t)erased_k[r] * sym_T, sym_T);

    void *dst_ptr = (uint8_t *)target_dev_ptr + target_offset;
    uint64_t tot_b = (uint64_t)total_bytes;
    void *d_miss_ptr = s->d_meta;
    void *d_mat_ptr = (uint8_t *)s->d_meta + miss_bytes;
    uint32_t sym_u32 = sym_T >> 2u;
    uint32_t gx = (sym_u32 + 255u) / 256u;
    uint32_t gy = (count + 7u) / 8u;
    void *args[9] = {&dst_ptr, &tot_b, &s->d_in, &d_mat_ptr, &d_miss_ptr, &K, &count, &count, &sym_T};

    bool submit_ok = rt_htod_async_locked(device, s->d_meta, s->h_meta, meta_total) &&
                     rt_htod_async_locked(device, s->d_in, rep_pack, rep_bytes) &&
                     rt_launch_locked(device, device->fn_vram_rep, gx, gy, args);
    uint64_t t = rt_record_ticket_locked(device, s, submit_ok);
    unlock_device(binding);
    return t;
}

uint64_t vxf_gpurt_rq_submit(int slot, nanorq_core *rq, const schedule *ops, uint32_t stride, const uint32_t *esis, uint32_t count)
{
    if (slot < 0 || slot >= (int)VXFEC_GPURT_MAX_POOL || count == 0u || stride == 0u || (stride & 3u) != 0u)
        return 0ULL;

    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = g_slot_owner[slot];
    gpurt_slot_t *s = device ? &device->pool[slot] : NULL;
    gpurt_binding_t binding = {0};
    if (!device || !bind_device_locked(device, &binding)) {
        unlock_device(binding);
        return 0ULL;
    }

    uint32_t rows = 0u;
    size_t m_stride = 0u;
    uint8_t *Y = build_rq_adjoint_scratch(s, rq, ops, esis, count, &rows, &m_stride);
    if (!Y) {
        unlock_device(binding);
        return 0ULL;
    }

    size_t mat_bytes = (size_t)count * rows;
    size_t in_bytes = (size_t)rows * stride;
    size_t out_bytes = (size_t)count * stride;
    if (in_bytes > s->in_cap || out_bytes > s->out_cap ||
        !rt_realloc_host_locked(device, &s->h_meta, &s->h_meta_cap, mat_bytes, false) ||
        !rt_realloc_dev_locked(device, &s->d_meta, &s->d_meta_cap, mat_bytes, false) ||
        !rt_realloc_dev_locked(device, &s->d_in, &s->d_in_cap, in_bytes, false) || !ensure_out_cap_locked(device, s, out_bytes)) {
        unlock_device(binding);
        return 0ULL;
    }

    uint8_t *h_mat = (uint8_t *)s->h_meta;
    for (uint32_t e = 0u; e < count; e++) {
        for (uint32_t r = 0u; r < rows; r++)
            h_mat[(size_t)e * rows + r] = Y[(size_t)r * m_stride + e];
    }

    uint32_t sh = nanorq_core_get_pc_genc_offset(rq);
    uint32_t stride_u32 = stride >> 2u;
    uint32_t gx = (stride_u32 + 255u) / 256u;
    uint32_t gy = (count + 7u) / 8u;
    void *args[8] = {&s->d_in, &s->d_out, &s->d_meta, &sh, &rows, &count, &stride_u32, &stride_u32};

    bool submit_ok = rt_htod_async_locked(device, s->d_in, s->in_stage, in_bytes) &&
                     rt_htod_async_locked(device, s->d_meta, h_mat, mat_bytes) &&
                     rt_launch_locked(device, device->fn_staged, gx, gy, args) &&
                     rt_dtoh_async_locked(device, s->out_stage, s->d_out, out_bytes);
    uint64_t t = rt_record_ticket_locked(device, s, submit_ok);
    unlock_device(binding);
    return t;
}

uint64_t vxf_gpurt_rq_submit_vram_enc(int slot, const void *src_dev_ptr, size_t src_offset, size_t valid_bytes, nanorq_core *rq,
                                      const schedule *ops, uint32_t sym_T, uint32_t stride, const uint32_t *esis, uint32_t count,
                                      bool use_vram_out)
{
    if (slot < 0 || slot >= (int)VXFEC_GPURT_MAX_POOL || !src_dev_ptr || count == 0u || sym_T == 0u || (sym_T & 3u) != 0u ||
        stride < sym_T || (stride & 3u) != 0u)
        return 0ULL;

    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = g_slot_owner[slot];
    gpurt_slot_t *s = device ? &device->pool[slot] : NULL;
    gpurt_binding_t binding = {0};
    if (!device || get_pointer_device_locked(src_dev_ptr) != device || !bind_device_locked(device, &binding)) {
        unlock_device(binding);
        return 0ULL;
    }

    uint32_t rows = 0u;
    size_t m_stride = 0u;
    uint8_t *Y = build_rq_adjoint_scratch(s, rq, ops, esis, count, &rows, &m_stride);
    uint32_t sh = Y ? nanorq_core_get_pc_genc_offset(rq) : 0u;
    uint32_t K = Y ? rq->P.K : 0u;
    size_t mat_bytes = (size_t)count * K;
    size_t out_bytes = (size_t)count * stride;
    if (!Y || (uint64_t)sh + (uint64_t)K > (uint64_t)rows ||
        !rt_realloc_host_locked(device, &s->h_meta, &s->h_meta_cap, mat_bytes, false) ||
        !rt_realloc_dev_locked(device, &s->d_meta, &s->d_meta_cap, mat_bytes, false) ||
        (use_vram_out ? out_bytes > s->d_out_cap : (out_bytes > s->out_cap || !ensure_out_cap_locked(device, s, out_bytes)))) {
        unlock_device(binding);
        return 0ULL;
    }

    uint8_t *h_mat = (uint8_t *)s->h_meta;
    for (uint32_t e = 0u; e < count; e++) {
        for (uint32_t k = 0u; k < K; k++)
            h_mat[(size_t)e * K + k] = Y[(size_t)(sh + k) * m_stride + e];
    }

    const void *src_ptr = (const uint8_t *)src_dev_ptr + src_offset;
    uint64_t vbytes = (uint64_t)valid_bytes;
    uint32_t out_u32 = stride >> 2u;
    uint32_t gx = (out_u32 + 255u) / 256u;
    uint32_t gy = (count + 7u) / 8u;
    void *args[8] = {&src_ptr, &vbytes, &s->d_out, &s->d_meta, &K, &count, &sym_T, &out_u32};

    bool submit_ok = rt_htod_async_locked(device, s->d_meta, h_mat, mat_bytes) &&
                     rt_launch_locked(device, device->fn_vram_enc, gx, gy, args) &&
                     (use_vram_out || rt_dtoh_async_locked(device, s->out_stage, s->d_out, out_bytes));
    uint64_t t = rt_record_ticket_locked(device, s, submit_ok);
    unlock_device(binding);
    return t;
}

uint64_t vxf_gpurt_rq_submit_vram_rep(int slot, void *target_dev_ptr, size_t target_offset, size_t total_bytes, nanorq_core *rq,
                                      const schedule *ops, const uint8_t *sys_bitmap, uint32_t K, uint32_t sym_T, uint32_t stride,
                                      const uint32_t *missing_esis, uint32_t count, uint32_t overhead)
{
    if (slot < 0 || slot >= (int)VXFEC_GPURT_MAX_POOL || !target_dev_ptr || !sys_bitmap || !missing_esis || K == 0u ||
        count == 0u || overhead > UINT32_MAX - count || sym_T == 0u || (sym_T & 3u) != 0u || stride < sym_T)
        return 0ULL;

    /* validate missing_esis are in-range, erased in sys_bitmap, and unique */
    size_t seen_bytes = ((size_t)K + 7u) / 8u;
    uint8_t stack_seen[256];
    uint8_t *seen = (seen_bytes <= sizeof(stack_seen)) ? stack_seen : (uint8_t *)calloc(seen_bytes, 1u);
    if (!seen)
        return 0ULL;
    if (seen == stack_seen)
        memset(seen, 0, seen_bytes);
    bool esis_ok = true;
    for (uint32_t r = 0u; r < count; r++) {
        uint32_t m = missing_esis[r];
        if (m >= K || (sys_bitmap[m >> 3u] & (1u << (m & 7u))) != 0u || (seen[m >> 3u] & (1u << (m & 7u))) != 0u) {
            esis_ok = false;
            break;
        }
        seen[m >> 3u] |= (uint8_t)(1u << (m & 7u));
    }
    if (seen != stack_seen)
        free(seen);
    if (!esis_ok)
        return 0ULL;

    pthread_mutex_lock(&g_rt_mtx);
    gpurt_device_t *device = g_slot_owner[slot];
    gpurt_slot_t *s = device ? &device->pool[slot] : NULL;
    gpurt_binding_t binding = {0};
    if (!device || get_pointer_device_locked(target_dev_ptr) != device || !bind_device_locked(device, &binding)) {
        unlock_device(binding);
        return 0ULL;
    }

    uint32_t rows = 0u;
    size_t m_stride = 0u;
    uint8_t *Y = build_rq_adjoint_scratch(s, rq, ops, missing_esis, count, &rows, &m_stride);
    uint32_t sh = Y ? nanorq_core_get_pc_genc_offset(rq) : 0u;
    uint32_t num_rep = count + overhead;
    uint32_t mat_cols = K + num_rep;
    size_t miss_bytes = ((size_t)count * sizeof(uint32_t) + 15u) & ~(size_t)15u;
    size_t mat_bytes = (size_t)count * mat_cols;
    size_t meta_total = miss_bytes + mat_bytes;
    size_t rep_bytes = (size_t)num_rep * sym_T;

    if (!Y || (uint64_t)sh + (uint64_t)rq->P.Kprime + (uint64_t)overhead > (uint64_t)rows ||
        (uint64_t)sh + (uint64_t)K > (uint64_t)rows || (size_t)rows * stride > s->in_cap ||
        !rt_realloc_host_locked(device, &s->h_meta, &s->h_meta_cap, meta_total, false) ||
        !rt_realloc_dev_locked(device, &s->d_meta, &s->d_meta_cap, meta_total, false) ||
        !rt_realloc_host_locked(device, &s->out_stage, &s->out_cap, rep_bytes, false) ||
        !rt_realloc_dev_locked(device, &s->d_in, &s->d_in_cap, rep_bytes, false)) {
        unlock_device(binding);
        return 0ULL;
    }

    uint32_t *h_miss = (uint32_t *)s->h_meta;
    uint8_t *h_mat = (uint8_t *)s->h_meta + miss_bytes;
    memset(h_mat, 0, mat_bytes);
    for (uint32_t b = 0u; b < count; b++) {
        h_miss[b] = missing_esis[b];
        uint8_t *mrow = h_mat + (size_t)b * mat_cols;
        for (uint32_t k = 0u; k < K; k++) {
            if (sys_bitmap[k >> 3u] & (1u << (k & 7u)))
                mrow[k] = Y[(size_t)(sh + k) * m_stride + b];
        }
        for (uint32_t r = 0u; r < count; r++)
            mrow[K + r] = Y[(size_t)(sh + missing_esis[r]) * m_stride + b];
        for (uint32_t ov = 0u; ov < overhead; ov++)
            mrow[K + count + ov] = Y[(size_t)(sh + rq->P.Kprime + ov) * m_stride + b];
    }

    /* gather only the count + overhead received repair symbols for compact HtoD upload */
    const uint8_t *in_st = (const uint8_t *)s->in_stage;
    uint8_t *rep_pack = (uint8_t *)s->out_stage;
    for (uint32_t r = 0u; r < count; r++)
        memcpy(rep_pack + (size_t)r * sym_T, in_st + (size_t)(sh + missing_esis[r]) * stride, sym_T);
    for (uint32_t ov = 0u; ov < overhead; ov++)
        memcpy(rep_pack + (size_t)(count + ov) * sym_T, in_st + (size_t)(sh + rq->P.Kprime + ov) * stride, sym_T);

    void *dst_ptr = (uint8_t *)target_dev_ptr + target_offset;
    uint64_t tot_b = (uint64_t)total_bytes;
    void *d_miss_ptr = s->d_meta;
    void *d_mat_ptr = (uint8_t *)s->d_meta + miss_bytes;
    uint32_t sym_u32 = sym_T >> 2u;
    uint32_t gx = (sym_u32 + 255u) / 256u;
    uint32_t gy = (count + 7u) / 8u;
    void *args[9] = {&dst_ptr, &tot_b, &s->d_in, &d_mat_ptr, &d_miss_ptr, &K, &num_rep, &count, &sym_T};

    bool submit_ok = rt_htod_async_locked(device, s->d_meta, s->h_meta, meta_total) &&
                     rt_htod_async_locked(device, s->d_in, rep_pack, rep_bytes) &&
                     rt_launch_locked(device, device->fn_vram_rep, gx, gy, args);
    uint64_t t = rt_record_ticket_locked(device, s, submit_ok);
    unlock_device(binding);
    return t;
}

vxf_gpurt_poll_status_t vxf_gpurt_poll(uint64_t ticket, bool wait)
{
    if (ticket == 0ULL)
        return VXF_GPURT_POLL_COMPLETE;
    pthread_mutex_lock(&g_rt_mtx);
    vxf_gpurt_poll_status_t res = VXF_GPURT_POLL_UNKNOWN;
    for (uint32_t i = 0; i < VXFEC_GPURT_MAX_POOL; i++) {
        gpurt_device_t *device = g_slot_owner[i];
        if (!device)
            continue;
        gpurt_slot_t *s = &device->pool[i];
        if (s->ticket != ticket)
            continue;
        res = VXF_GPURT_POLL_PENDING;
        bool idle = false;
        bool need_stream_query = !s->event_recorded;
        if (s->event_recorded) {
            int q = g_kind == VXF_GPURT_HIP ? g_hip.event_query(s->ev) : g_cu.event_query(s->ev);
            idle = (q == 0);
            if (q != 0 && q != 600 /* hipErrorNotReady / CUDA_ERROR_NOT_READY */) {
                need_stream_query = true;
                s->failed = true;
            }
        }
        if (need_stream_query || (wait && !idle)) {
            gpurt_binding_t binding = {0};
            if (bind_device_locked(device, &binding)) {
                int q;
                if (wait)
                    q = g_kind == VXF_GPURT_HIP ? g_hip.stream_sync(device->stream) : g_cu.stream_sync(device->stream);
                else
                    q = g_kind == VXF_GPURT_HIP ? g_hip.stream_query(device->stream) : g_cu.stream_query(device->stream);
                /* a destroyed owner context cannot still access the job's buffers */
                idle = q == 0 || q == VXF_GPURT_CONTEXT_DESTROYED;
                if (q == VXF_GPURT_CONTEXT_DESTROYED)
                    s->failed = true;
                else if (q != 0 && q != 600 /* hipErrorNotReady / CUDA_ERROR_NOT_READY */)
                    res = VXF_GPURT_POLL_UNSAFE;
                unbind_device_locked(binding);
            } else {
                res = VXF_GPURT_POLL_UNSAFE;
            }
        }
        if (idle) {
            res = s->failed ? VXF_GPURT_POLL_FAILED : VXF_GPURT_POLL_COMPLETE;
            s->ticket = 0ULL;
        }
        break;
    }
    pthread_mutex_unlock(&g_rt_mtx);
    return res;
}
