#define _GNU_SOURCE
#include "../plugins/nccl/nccl_net_vxfec.h"
#include "../t/bench_common.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/*
 * end-to-end multi-node GPU AllReduce demo & A/B benchmark (`libnccl-vxfec.so` vs stock `NCCL_IB` RC):
 *   - zero-arg mode: forks 2 simulated nodes on localhost (`make demo-nccl`)
 *   - multi-host mode: `--rank <r> --nranks <N> --master <ip:port>` or `--node-rank <n> --nnodes <M> --gpus-per-node <G>`
 *     with `[--native-ib] [--ab-bench] [--iters <N>] [--gdr] [--backend udp|ibv] [--max-mb <mb>] [--csv <file>]`
 *   - 512 KiB profiles: `--b200-four-case` or `--h200-four-case` with `--seed <N> --csv <file>` (rank 0);
 *     run once with `--native-ib` and once with the plugin for each seed.
 *   - dynamically loads ROCm RCCL or NVIDIA NCCL and verifies bit-exact FP32 ncclAllReduce.
 */

#define DEMO_MAX_ITERS 512
#define DEMO_MAX_LOCAL_GPUS 16
#define DEMO_MAX_PHASES 20
#define DEMO_HCA_NAME_MAX 44
#define DEMO_FOUR_CASE_BYTES (512u * 1024u)
#define DEMO_FOUR_CASE_LOSS_PPM 20000u
#define DEMO_FOUR_CASE_BURST 6u

typedef struct {
    char internal[128];
} nccl_uid_t;

typedef void *nccl_comm_t;
typedef void *gpu_stream_t;

enum { NCCL_DT_FLOAT32 = 7, NCCL_OP_SUM = 0 };
enum { GPU_MEMCPY_H2D = 1, GPU_MEMCPY_D2H = 2 };

typedef struct {
    const char *backend_name;
    bool is_hip;
    void *gpu_dl;
    void *ccl_dl;
    int (*set_device)(int);
    int (*dev_by_pci)(int *, const char *);
    int (*malloc_dev)(void **, size_t);
    int (*free_dev)(void *);
    int (*memcpy_dev)(void *, const void *, size_t, int);
    int (*stream_create)(gpu_stream_t *);
    int (*stream_create_flags)(gpu_stream_t *, unsigned int);
    int (*stream_sync)(gpu_stream_t);
    int (*stream_destroy)(gpu_stream_t);
    int (*get_uid)(nccl_uid_t *);
    int (*comm_init)(nccl_comm_t *, int, nccl_uid_t, int);
    int (*comm_destroy)(nccl_comm_t);
    int (*all_reduce)(const void *, void *, size_t, int, int, nccl_comm_t, gpu_stream_t);
    int (*group_start)(void);
    int (*group_end)(void);
    int (*ccl_send)(const void *, size_t, int, int, nccl_comm_t, gpu_stream_t);
    int (*ccl_recv)(void *, size_t, int, int, nccl_comm_t, gpu_stream_t);
} gpu_ccl_api_t;

static uint64_t now_ns(void)
{
    return vxfec_bench_now_ns();
}

static int cmp_u64_asc(const void *a, const void *b)
{
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

static bool write_full(int fd, const void *buf, size_t len)
{
    return vxfec_bench_send_all(fd, buf, len);
}

static bool read_full(int fd, void *buf, size_t len)
{
    return vxfec_bench_recv_all(fd, buf, len);
}

/* Parse explicit local-rank maps before forking so every rank uses the same complete map. */
static bool parse_rank_map(const char *option, const char *list, int ranks, char out[][64])
{
    if (!list)
        return true;
    const char *p = list;
    for (int rank = 0; rank < ranks; rank++) {
        while (*p == ' ' || *p == '\t')
            p++;
        const char *comma = strchr(p, ',');
        size_t len = comma ? (size_t)(comma - p) : strlen(p);
        while (len && (p[len - 1u] == ' ' || p[len - 1u] == '\t'))
            len--;
        if (len == 0 || len >= sizeof(out[rank]) || (comma != NULL) != (rank + 1 < ranks)) {
            fprintf(stderr, "%s requires exactly %d nonempty device IDs of at most %zu characters\n", option, ranks,
                    sizeof(out[rank]) - 1);
            return false;
        }
        memcpy(out[rank], p, len);
        out[rank][len] = '\0';
        if (comma)
            p = comma + 1;
    }
    return true;
}

static bool probe_gpu_ccl_installed_inproc(bool cuda_only)
{
    const char *hip_libs[] = {"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", NULL};
    const char *rccl_libs[] = {"librccl.so.1", "librccl.so", NULL};
    for (int i = 0; !cuda_only && hip_libs[i]; i++) {
        void *g = dlopen(hip_libs[i], RTLD_LAZY | RTLD_LOCAL);
        if (g) {
            for (int j = 0; rccl_libs[j]; j++) {
                void *c = dlopen(rccl_libs[j], RTLD_LAZY | RTLD_LOCAL);
                if (c) {
                    dlclose(c);
                    dlclose(g);
                    return true;
                }
            }
            dlclose(g);
        }
    }
    const char *cuda_libs[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so", NULL};
    const char *nccl_libs[] = {"libnccl.so.2", "libnccl.so", NULL};
    for (int i = 0; cuda_libs[i]; i++) {
        void *g = dlopen(cuda_libs[i], RTLD_LAZY | RTLD_LOCAL);
        if (g) {
            for (int j = 0; nccl_libs[j]; j++) {
                void *c = dlopen(nccl_libs[j], RTLD_LAZY | RTLD_LOCAL);
                if (c) {
                    dlclose(c);
                    dlclose(g);
                    return true;
                }
            }
            dlclose(g);
        }
    }
    return false;
}

/* probe in a child process so parent never loads GPU/CCL libraries before fork() */
static bool probe_gpu_ccl_installed(bool cuda_only)
{
    pid_t pid = fork();
    if (pid < 0)
        return probe_gpu_ccl_installed_inproc(cuda_only);
    if (pid == 0)
        _exit(probe_gpu_ccl_installed_inproc(cuda_only) ? 0 : 1);
    int st = 0;
    while (waitpid(pid, &st, 0) < 0) {
        if (errno != EINTR)
            return false;
    }
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

static bool load_gpu_ccl(gpu_ccl_api_t *api, bool cuda_only)
{
    memset(api, 0, sizeof(*api));

    const char *hip_libs[] = {"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", NULL};
    for (int i = 0; !cuda_only && hip_libs[i] && !api->gpu_dl; i++)
        api->gpu_dl = dlopen(hip_libs[i], RTLD_NOW | RTLD_GLOBAL);
    if (api->gpu_dl) {
        api->ccl_dl = dlopen("librccl.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!api->ccl_dl)
            api->ccl_dl = dlopen("librccl.so", RTLD_NOW | RTLD_GLOBAL);
        if (api->ccl_dl) {
            api->backend_name = "ROCm HIP + RCCL";
            api->is_hip = true;
            api->set_device = dlsym(api->gpu_dl, "hipSetDevice");
            api->dev_by_pci = dlsym(api->gpu_dl, "hipDeviceGetByPCIBusId");
            api->malloc_dev = dlsym(api->gpu_dl, "hipMalloc");
            api->free_dev = dlsym(api->gpu_dl, "hipFree");
            api->memcpy_dev = dlsym(api->gpu_dl, "hipMemcpy");
            api->stream_create = dlsym(api->gpu_dl, "hipStreamCreate");
            api->stream_create_flags = dlsym(api->gpu_dl, "hipStreamCreateWithFlags");
            api->stream_sync = dlsym(api->gpu_dl, "hipStreamSynchronize");
            api->stream_destroy = dlsym(api->gpu_dl, "hipStreamDestroy");
        } else {
            dlclose(api->gpu_dl);
            api->gpu_dl = NULL;
        }
    }

    if (!api->ccl_dl) {
        const char *cuda_libs[] = {"libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so", NULL};
        for (int i = 0; cuda_libs[i] && !api->gpu_dl; i++)
            api->gpu_dl = dlopen(cuda_libs[i], RTLD_NOW | RTLD_GLOBAL);
        if (api->gpu_dl) {
            api->ccl_dl = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
            if (!api->ccl_dl)
                api->ccl_dl = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
            if (api->ccl_dl) {
                api->backend_name = "CUDA + NCCL";
                api->is_hip = false;
                api->set_device = dlsym(api->gpu_dl, "cudaSetDevice");
                api->dev_by_pci = dlsym(api->gpu_dl, "cudaDeviceGetByPCIBusId");
                api->malloc_dev = dlsym(api->gpu_dl, "cudaMalloc");
                api->free_dev = dlsym(api->gpu_dl, "cudaFree");
                api->memcpy_dev = dlsym(api->gpu_dl, "cudaMemcpy");
                api->stream_create = dlsym(api->gpu_dl, "cudaStreamCreate");
                api->stream_create_flags = dlsym(api->gpu_dl, "cudaStreamCreateWithFlags");
                api->stream_sync = dlsym(api->gpu_dl, "cudaStreamSynchronize");
                api->stream_destroy = dlsym(api->gpu_dl, "cudaStreamDestroy");
            } else {
                dlclose(api->gpu_dl);
                api->gpu_dl = NULL;
            }
        }
    }

    if (!api->ccl_dl)
        return false;

    api->get_uid = dlsym(api->ccl_dl, "ncclGetUniqueId");
    api->comm_init = dlsym(api->ccl_dl, "ncclCommInitRank");
    api->comm_destroy = dlsym(api->ccl_dl, "ncclCommDestroy");
    api->all_reduce = dlsym(api->ccl_dl, "ncclAllReduce");
    api->group_start = dlsym(api->ccl_dl, "ncclGroupStart");
    api->group_end = dlsym(api->ccl_dl, "ncclGroupEnd");
    api->ccl_send = dlsym(api->ccl_dl, "ncclSend");
    api->ccl_recv = dlsym(api->ccl_dl, "ncclRecv");

    return api->set_device && api->malloc_dev && api->free_dev && api->memcpy_dev && api->stream_create && api->stream_sync &&
           api->stream_destroy && api->get_uid && api->comm_init && api->comm_destroy && api->all_reduce;
}

/*
 * Self-contained RTC tiled FP32 SGEMM compute workload (64x64 output tile, 4x4 register micro-tile)
 * for measuring concurrent GPU compute + AllToAll/AllReduce training-step overlap without SDK headers.
 */
static const char g_demo_sgemm_src[] =
    "extern \"C\" __global__ void demo_sgemm_64(const float * __restrict__ A,\n"
    "                                           const float * __restrict__ B,\n"
    "                                           float * __restrict__ C,\n"
    "                                           int N, int K)\n"
    "{\n"
    "    __shared__ float sA[16][64];\n"
    "    __shared__ float sB[16][64];\n"
    "    int tx = (int)threadIdx.x;\n"
    "    int ty = (int)threadIdx.y;\n"
    "    int tid = ty * 16 + tx;\n"
    "    int row_base = (int)blockIdx.y * 64 + ty * 4;\n"
    "    int col_base = (int)blockIdx.x * 64 + tx * 4;\n"
    "    float acc[4][4];\n"
    "    for (int i = 0; i < 4; i++)\n"
    "        for (int j = 0; j < 4; j++)\n"
    "            acc[i][j] = 0.0f;\n"
    "    for (int kt = 0; kt < K; kt += 16) {\n"
    "        for (int ld = 0; ld < 4; ld++) {\n"
    "            int idx = tid + ld * 256;\n"
    "            int ar = (int)blockIdx.y * 64 + (idx & 63);\n"
    "            int ak = kt + (idx >> 6);\n"
    "            sA[idx >> 6][idx & 63] = (ar < N && ak < K) ? A[(long)ar * K + ak] : 0.0f;\n"
    "            int br = kt + (idx >> 6);\n"
    "            int bc = (int)blockIdx.x * 64 + (idx & 63);\n"
    "            sB[idx >> 6][idx & 63] = (br < K && bc < N) ? B[(long)br * N + bc] : 0.0f;\n"
    "        }\n"
    "        __syncthreads();\n"
    "        for (int kk = 0; kk < 16; kk++) {\n"
    "            float a0 = sA[kk][ty * 4 + 0], a1 = sA[kk][ty * 4 + 1];\n"
    "            float a2 = sA[kk][ty * 4 + 2], a3 = sA[kk][ty * 4 + 3];\n"
    "            float b0 = sB[kk][tx * 4 + 0], b1 = sB[kk][tx * 4 + 1];\n"
    "            float b2 = sB[kk][tx * 4 + 2], b3 = sB[kk][tx * 4 + 3];\n"
    "            acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;\n"
    "            acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;\n"
    "            acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;\n"
    "            acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    for (int i = 0; i < 4; i++) {\n"
    "        int r = row_base + i;\n"
    "        if (r < N) {\n"
    "            for (int j = 0; j < 4; j++) {\n"
    "                int c = col_base + j;\n"
    "                if (c < N)\n"
    "                    C[(long)r * N + c] = acc[i][j];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n";

typedef struct {
    bool is_hip;
    void *drv_dl;
    void *rtc_dl;
    void *mod;
    void *fn;
    int (*mod_unload)(void *);
    int (*launch)(void *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void *,
                  void **, void **);
    void *d_a;
    void *d_b;
    void *d_c;
    gpu_stream_t comp_stream;
    int dim_n;
    int dim_k;
} demo_sgemm_ctx_t;

static bool demo_sgemm_init(const gpu_ccl_api_t *api, demo_sgemm_ctx_t *sg, int dim_n, int dim_k)
{
    memset(sg, 0, sizeof(*sg));
    sg->is_hip = api->is_hip;
    sg->dim_n = dim_n;
    sg->dim_k = dim_k;

    const char *rtc_libs_hip[] = {"libhiprtc.so.7", "libhiprtc.so.6", "libhiprtc.so", NULL};
    const char *rtc_libs_cu[] = {"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so", NULL};
    const char *const *rtc_libs = api->is_hip ? rtc_libs_hip : rtc_libs_cu;
    for (int i = 0; rtc_libs[i] && !sg->rtc_dl; i++)
        sg->rtc_dl = dlopen(rtc_libs[i], RTLD_NOW | RTLD_LOCAL);
    if (!sg->rtc_dl)
        return false;

    if (api->is_hip) {
        sg->drv_dl = api->gpu_dl;
    } else {
        const char *cu_libs[] = {"libcuda.so.1", "libcuda.so", NULL};
        for (int i = 0; cu_libs[i] && !sg->drv_dl; i++)
            sg->drv_dl = dlopen(cu_libs[i], RTLD_NOW | RTLD_LOCAL);
        if (!sg->drv_dl) {
            dlclose(sg->rtc_dl);
            sg->rtc_dl = NULL;
            return false;
        }
    }

    int (*rtc_create)(void **, const char *, const char *, int, const char *const *, const char *const *) =
        dlsym(sg->rtc_dl, api->is_hip ? "hiprtcCreateProgram" : "nvrtcCreateProgram");
    int (*rtc_compile)(void *, int, const char *const *) =
        dlsym(sg->rtc_dl, api->is_hip ? "hiprtcCompileProgram" : "nvrtcCompileProgram");
    int (*rtc_sz)(void *, size_t *) = dlsym(sg->rtc_dl, api->is_hip ? "hiprtcGetCodeSize" : "nvrtcGetPTXSize");
    int (*rtc_code)(void *, char *) = dlsym(sg->rtc_dl, api->is_hip ? "hiprtcGetCode" : "nvrtcGetPTX");
    int (*rtc_destroy)(void **) = dlsym(sg->rtc_dl, api->is_hip ? "hiprtcDestroyProgram" : "nvrtcDestroyProgram");
    int (*mod_load)(void **, const void *) = dlsym(sg->drv_dl, api->is_hip ? "hipModuleLoadData" : "cuModuleLoadData");
    int (*mod_func)(void **, void *, const char *) =
        dlsym(sg->drv_dl, api->is_hip ? "hipModuleGetFunction" : "cuModuleGetFunction");
    sg->mod_unload = dlsym(sg->drv_dl, api->is_hip ? "hipModuleUnload" : "cuModuleUnload");
    sg->launch = dlsym(sg->drv_dl, api->is_hip ? "hipModuleLaunchKernel" : "cuLaunchKernel");

    if (!rtc_create || !rtc_compile || !rtc_sz || !rtc_code || !rtc_destroy || !mod_load || !mod_func || !sg->launch)
        return false;

    void *prog = NULL;
    if (rtc_create(&prog, g_demo_sgemm_src, api->is_hip ? "sgemm.hip" : "sgemm.cu", 0, NULL, NULL) != 0 || !prog)
        return false;

    bool compiled = (rtc_compile(prog, 0, NULL) == 0);
    size_t bin_sz = 0;
    char *bin = NULL;
    bool loaded = false;
    if (compiled && rtc_sz(prog, &bin_sz) == 0 && bin_sz > 0) {
        bin = (char *)malloc(bin_sz);
        if (bin && rtc_code(prog, bin) == 0 && mod_load(&sg->mod, bin) == 0 && mod_func(&sg->fn, sg->mod, "demo_sgemm_64") == 0) {
            loaded = true;
        }
    }
    free(bin);
    (void)rtc_destroy(&prog);
    if (!loaded)
        return false;

    size_t a_bytes = (size_t)dim_n * (size_t)dim_k * sizeof(float);
    size_t c_bytes = (size_t)dim_n * (size_t)dim_n * sizeof(float);
    int stream_rc =
        api->stream_create_flags ? api->stream_create_flags(&sg->comp_stream, 1u) : api->stream_create(&sg->comp_stream);
    if (stream_rc != 0 || api->malloc_dev(&sg->d_a, a_bytes) != 0 || api->malloc_dev(&sg->d_b, a_bytes) != 0 ||
        api->malloc_dev(&sg->d_c, c_bytes) != 0)
        return false;

    float *h_init = (float *)malloc(a_bytes > c_bytes ? a_bytes : c_bytes);
    if (!h_init)
        return false;
    for (size_t i = 0; i < (a_bytes / sizeof(float)); i++)
        h_init[i] = (float)((i % 17u) + 1u) * 0.0625f;
    bool init_ok = api->memcpy_dev(sg->d_a, h_init, a_bytes, GPU_MEMCPY_H2D) == 0 &&
                   api->memcpy_dev(sg->d_b, h_init, a_bytes, GPU_MEMCPY_H2D) == 0 && api->stream_sync(NULL) == 0;
    free(h_init);
    return init_ok;
}

static bool demo_sgemm_launch(demo_sgemm_ctx_t *sg)
{
    unsigned int gx = (unsigned int)((sg->dim_n + 63) / 64);
    unsigned int gy = (unsigned int)((sg->dim_n + 63) / 64);
    void *args[5] = {&sg->d_a, &sg->d_b, &sg->d_c, &sg->dim_n, &sg->dim_k};
    return sg->launch(sg->fn, gx, gy, 1u, 16u, 16u, 1u, 0u, sg->comp_stream, args, NULL) == 0;
}

static void demo_sgemm_free(const gpu_ccl_api_t *api, demo_sgemm_ctx_t *sg)
{
    if (sg->comp_stream && api->stream_destroy)
        (void)api->stream_destroy(sg->comp_stream);
    if (sg->d_c && api->free_dev)
        (void)api->free_dev(sg->d_c);
    if (sg->d_b && api->free_dev)
        (void)api->free_dev(sg->d_b);
    if (sg->d_a && api->free_dev)
        (void)api->free_dev(sg->d_a);
    if (sg->mod && sg->mod_unload)
        (void)sg->mod_unload(sg->mod);
    if (!sg->is_hip && sg->drv_dl)
        dlclose(sg->drv_dl);
    if (sg->rtc_dl)
        dlclose(sg->rtc_dl);
    memset(sg, 0, sizeof(*sg));
}

typedef void (*get_stats_fn)(vxfec_nccl_stats_t *);
typedef void (*get_stats_ex_fn)(vxfec_nccl_stats_t *, size_t);
typedef size_t (*get_rail_stats_fn)(vxfec_nccl_rail_stats_t *, size_t);
typedef void (*reset_stats_fn)(void);
typedef void (*set_fault_fn)(uint32_t, uint32_t, bool);
typedef bool (*set_paired_fault_fn)(uint32_t, uint32_t, uint64_t, uint32_t, uint32_t);
typedef void (*get_paired_stats_fn)(vxfec_nccl_paired_fault_stats_t *);
typedef void (*set_codec_fn)(const char *);

typedef struct {
    bool is_fork;
    int rank;
    int nranks;
    pid_t child_pid;
    int tx_fd;
    int rx_fd;
    int *peer_fds; /* rank 0 TCP sockets to ranks 1..N-1 */
    int master_fd; /* rank >0 TCP socket to rank 0 */
} ctrl_fabric_t;

typedef struct {
    char device[DEMO_HCA_NAME_MAX + 1];
    unsigned port;
    uint64_t tx_before, rx_before;
    uint64_t tx_units, rx_units;
} demo_rail_t;

typedef struct {
    uint32_t tx, rx;
} rail_coverage_t;

static bool parse_rail(const char *id, demo_rail_t *rail)
{
    size_t len = strspn(id, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.");
    if (len == 0 || len > DEMO_HCA_NAME_MAX || (id[len] && id[len] != ':') || (len == 1u && id[0] == '.') ||
        (len == 2u && id[0] == '.' && id[1] == '.'))
        return false;
    unsigned long port = 1;
    if (id[len]) {
        const char *port_text = id + len + 1u;
        char *end = NULL;
        port = strtoul(port_text, &end, 10);
        if (end == port_text || *end || port == 0 || port > UINT8_MAX)
            return false;
    }
    memcpy(rail->device, id, len);
    rail->device[len] = '\0';
    rail->port = (unsigned)port;
    return true;
}

static bool read_port_counter(const demo_rail_t *rail, const char *counter, uint64_t *value)
{
    char path[256];
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%u/counters/%s", rail->device, rail->port, counter);
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        return false;
    }
    bool ok = fscanf(fp, "%" SCNu64, value) == 1;
    fclose(fp);
    if (!ok)
        fprintf(stderr, "Invalid InfiniBand port counter %s\n", path);
    return ok;
}

static bool snapshot_native_rails(demo_rail_t *rails, int count, bool start)
{
    for (int i = 0; i < count; i++) {
        demo_rail_t *rail = &rails[i];
        uint64_t tx, rx;
        if (!read_port_counter(rail, "port_xmit_data", &tx) || !read_port_counter(rail, "port_rcv_data", &rx))
            return false;
        if (start) {
            rail->tx_before = tx;
            rail->rx_before = rx;
        } else {
            if (tx < rail->tx_before || rx < rail->rx_before) {
                fprintf(stderr, "InfiniBand port counters decreased on %s:%u\n", rail->device, rail->port);
                return false;
            }
            rail->tx_units += tx - rail->tx_before;
            rail->rx_units += rx - rail->rx_before;
        }
    }
    return true;
}

static bool snapshot_plugin_rails(get_rail_stats_fn get_rail_stats, demo_rail_t *rails, int count, bool accumulate)
{
    vxfec_nccl_rail_stats_t stats[DEMO_MAX_LOCAL_GPUS];
    if (get_rail_stats(stats, DEMO_MAX_LOCAL_GPUS) != (size_t)count)
        return false;
    for (int i = 0; i < count; i++) {
        char name[sizeof(stats[0].name)];
        snprintf(name, sizeof(name), "vxfec-ibv-%s:%u", rails[i].device, rails[i].port);
        int dev = 0;
        for (int j = 0; j < count; j++) {
            int cmp = strcmp(rails[j].device, rails[i].device);
            if (cmp < 0 || (cmp == 0 && rails[j].port < rails[i].port))
                dev++;
        }
        if (strcmp(stats[dev].name, name) != 0)
            return false;
        if (accumulate) {
            rails[i].tx_units += stats[dev].tx_sys_gpu_direct_pkts;
            rails[i].rx_units += stats[dev].rx_sys_gpu_direct_pkts;
        }
    }
    return true;
}

static bool ctrl_barrier(ctrl_fabric_t *cf)
{
    uint8_t tok = 0x5au;
    if (cf->is_fork)
        return write_full(cf->tx_fd, &tok, 1) && read_full(cf->rx_fd, &tok, 1);

    if (cf->rank == 0) {
        for (int r = 1; r < cf->nranks; r++) {
            if (!read_full(cf->peer_fds[r], &tok, 1))
                return false;
        }
        tok = 0xa5u;
        for (int r = 1; r < cf->nranks; r++) {
            if (!write_full(cf->peer_fds[r], &tok, 1))
                return false;
        }
        return true;
    }
    return write_full(cf->master_fd, &tok, 1) && read_full(cf->master_fd, &tok, 1);
}

static bool ctrl_bcast(ctrl_fabric_t *cf, void *buf, size_t bytes)
{
    if (cf->is_fork) {
        if (cf->rank == 0)
            return write_full(cf->tx_fd, buf, bytes);
        return read_full(cf->rx_fd, buf, bytes);
    }
    if (cf->rank == 0) {
        for (int r = 1; r < cf->nranks; r++) {
            if (!write_full(cf->peer_fds[r], buf, bytes))
                return false;
        }
        return true;
    }
    return read_full(cf->master_fd, buf, bytes);
}

static bool ctrl_all_ok(ctrl_fabric_t *cf, bool my_ok, bool *all_ok_out)
{
    uint8_t mine = my_ok ? 1u : 0u;
    if (cf->is_fork) {
        uint8_t peer = 0u;
        if (!write_full(cf->tx_fd, &mine, 1) || !read_full(cf->rx_fd, &peer, 1))
            return false;
        *all_ok_out = (mine != 0u && peer != 0u);
        return true;
    }
    if (cf->rank > 0) {
        uint8_t all = 0u;
        if (!write_full(cf->master_fd, &mine, 1) || !read_full(cf->master_fd, &all, 1))
            return false;
        *all_ok_out = (all != 0u);
        return true;
    }
    uint8_t all = mine;
    for (int r = 1; r < cf->nranks; r++) {
        uint8_t peer = 0u;
        if (!read_full(cf->peer_fds[r], &peer, 1))
            return false;
        if (peer == 0u)
            all = 0u;
    }
    for (int r = 1; r < cf->nranks; r++) {
        if (!write_full(cf->peer_fds[r], &all, 1))
            return false;
    }
    *all_ok_out = (all != 0u);
    return true;
}

static bool init_collective(ctrl_fabric_t *cf, const gpu_ccl_api_t *api, nccl_comm_t *comm, bool native_ib)
{
    nccl_uid_t uid = {0};
    bool all_ready = false;
    bool uid_ready = cf->rank != 0 || api->get_uid(&uid) == 0;
    if (!ctrl_all_ok(cf, uid_ready, &all_ready) || !all_ready || !ctrl_bcast(cf, &uid, sizeof(uid)))
        return false;
    bool comm_ready = api->comm_init(comm, cf->nranks, uid, cf->rank) == 0;
    if (!comm_ready)
        fprintf(stderr, "rank %d: ncclCommInitRank failed%s\n", cf->rank,
                native_ib ? " with the built-in IB transport required" : "");
    return ctrl_all_ok(cf, comm_ready, &all_ready) && all_ready;
}

static bool ctrl_check_rail_coverage(ctrl_fabric_t *cf, const demo_rail_t *rails, int count, bool native_ib)
{
    rail_coverage_t mine = {0};
    for (int i = 0; i < count; i++) {
        if (rails[i].tx_units > 0u)
            mine.tx |= 1u << (unsigned)i;
        if (rails[i].rx_units > 0u)
            mine.rx |= 1u << (unsigned)i;
    }
    bool covered = true;
    if (cf->rank > 0) {
        if (!write_full(cf->master_fd, &mine, sizeof(mine)))
            return false;
    } else {
        const uint32_t expected = (1u << (unsigned)count) - 1u;
        for (int start = 0; start < cf->nranks; start += count) {
            rail_coverage_t node = {0};
            for (int r = start; r < start + count; r++) {
                rail_coverage_t rank = mine;
                if (r > 0 && !read_full(cf->peer_fds[r], &rank, sizeof(rank)))
                    return false;
                node.tx |= rank.tx;
                node.rx |= rank.rx;
            }
            bool node_covered = node.tx == expected && node.rx == expected;
            fprintf(node_covered ? stdout : stderr,
                    "%snode %d timed sweep %s (in --ibv-devs order): TX 0x%04" PRIx32 ", RX 0x%04" PRIx32 ", expected 0x%04" PRIx32
                    "\n",
                    node_covered ? "" : "ERROR: ", start / count,
                    native_ib ? "host-wide IB port activity" : "vxfec GPUDirect rails", node.tx, node.rx, expected);
            covered = covered && node_covered;
        }
    }
    bool all_covered = false;
    return ctrl_all_ok(cf, covered, &all_covered) && all_covered;
}

static void accum_stats(vxfec_nccl_stats_t *dst, const vxfec_nccl_stats_t *src)
{
    dst->tx_messages += src->tx_messages;
    dst->rx_messages += src->rx_messages;
    dst->tx_sys_pkts += src->tx_sys_pkts;
    dst->sim_dropped_pkts += src->sim_dropped_pkts;
    dst->tx_sack_retx_pkts += src->tx_sack_retx_pkts;
    dst->tx_harq_ir_pkts += src->tx_harq_ir_pkts;
    dst->tx_proactive_rep_pkts += src->tx_proactive_rep_pkts;
    dst->rx_fec_decodes += src->rx_fec_decodes;
    dst->rx_rs8_decodes += src->rx_rs8_decodes;
    dst->rx_rq_decodes += src->rx_rq_decodes;
    dst->vk_offload_calls += src->vk_offload_calls;
    dst->ibv_direct_pkts += src->ibv_direct_pkts;
    dst->tx_sys_ibv_pkts += src->tx_sys_ibv_pkts;
    dst->tx_sys_gpu_direct_pkts += src->tx_sys_gpu_direct_pkts;
    dst->rx_sys_pkts += src->rx_sys_pkts;
    dst->rx_sys_gpu_direct_pkts += src->rx_sys_gpu_direct_pkts;
    dst->tx_udp_payload_pkts += src->tx_udp_payload_pkts;
    dst->rx_cuda_repair_completions += src->rx_cuda_repair_completions;
}

static bool ctrl_gather_stats(ctrl_fabric_t *cf, const vxfec_nccl_stats_t *my_st, vxfec_nccl_stats_t *sum_out)
{
    if (cf->is_fork) {
        if (cf->rank == 1)
            return write_full(cf->tx_fd, my_st, sizeof(*my_st)) && read_full(cf->rx_fd, sum_out, sizeof(*sum_out));
        vxfec_nccl_stats_t peer = {0};
        if (!read_full(cf->rx_fd, &peer, sizeof(peer)))
            return false;
        *sum_out = *my_st;
        accum_stats(sum_out, &peer);
        return write_full(cf->tx_fd, sum_out, sizeof(*sum_out));
    }
    if (cf->rank > 0)
        return write_full(cf->master_fd, my_st, sizeof(*my_st)) && read_full(cf->master_fd, sum_out, sizeof(*sum_out));

    *sum_out = *my_st;
    for (int r = 1; r < cf->nranks; r++) {
        vxfec_nccl_stats_t peer = {0};
        if (!read_full(cf->peer_fds[r], &peer, sizeof(peer)))
            return false;
        accum_stats(sum_out, &peer);
    }
    for (int r = 1; r < cf->nranks; r++) {
        if (!write_full(cf->peer_fds[r], sum_out, sizeof(*sum_out)))
            return false;
    }
    return true;
}

static bool ctrl_gather_max_lats(ctrl_fabric_t *cf, const uint64_t *my_lats, uint64_t *max_out, int iters)
{
    size_t nbytes = (size_t)iters * sizeof(uint64_t);
    if (cf->is_fork) {
        if (cf->rank == 1)
            return write_full(cf->tx_fd, my_lats, nbytes);
        uint64_t peer[DEMO_MAX_ITERS];
        if (!read_full(cf->rx_fd, peer, nbytes))
            return false;
        for (int i = 0; i < iters; i++)
            max_out[i] = (peer[i] > my_lats[i]) ? peer[i] : my_lats[i];
        return true;
    }
    if (cf->rank > 0)
        return write_full(cf->master_fd, my_lats, nbytes);

    memcpy(max_out, my_lats, nbytes);
    uint64_t peer[DEMO_MAX_ITERS];
    for (int r = 1; r < cf->nranks; r++) {
        if (!read_full(cf->peer_fds[r], peer, nbytes))
            return false;
        for (int i = 0; i < iters; i++) {
            if (peer[i] > max_out[i])
                max_out[i] = peer[i];
        }
    }
    return true;
}

/* The focused profile runs with two ranks. Rank zero validates the paired trace. */
static bool ctrl_gather_paired_stats(ctrl_fabric_t *cf, vxfec_nccl_paired_fault_stats_t *sum)
{
    if (cf->rank == 1)
        return write_full(cf->is_fork ? cf->tx_fd : cf->master_fd, sum, sizeof(*sum));
    vxfec_nccl_paired_fault_stats_t peer;
    if (!read_full(cf->is_fork ? cf->rx_fd : cf->peer_fds[1], &peer, sizeof(peer)))
        return false;
    sum->systematic_trials += peer.systematic_trials;
    sum->drop_fingerprint += peer.drop_fingerprint;
    return true;
}

static bool close_csv(FILE **fp, const char *path, bool durable)
{
    if (!*fp)
        return true;
    bool ok = fflush(*fp) == 0 && !ferror(*fp);
    if (ok && durable)
        ok = fsync(fileno(*fp)) == 0;
    if (fclose(*fp) != 0)
        ok = false;
    *fp = NULL;
    if (!ok)
        fprintf(stderr, "ERROR: failed writing %s\n", path);
    return ok;
}

static bool setup_tcp_ctrl(ctrl_fabric_t *cf, const char *master_spec)
{
    char host[128];
    uint16_t port = 29550u;
    vxfec_bench_parse_host_port(master_spec, host, sizeof(host), 29550u, &port);

    if (cf->rank == 0) {
        int lfd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (lfd < 0)
            return false;
        int one = 1;
        (void)setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        struct sockaddr_in sin;
        memset(&sin, 0, sizeof(sin));
        sin.sin_family = AF_INET;
        sin.sin_addr.s_addr = htonl(INADDR_ANY);
        sin.sin_port = htons(port);
        if (bind(lfd, (struct sockaddr *)&sin, sizeof(sin)) != 0 || listen(lfd, cf->nranks + 16) != 0) {
            close(lfd);
            return false;
        }
        cf->peer_fds = (int *)calloc((size_t)cf->nranks, sizeof(int));
        if (!cf->peer_fds) {
            close(lfd);
            return false;
        }
        for (int i = 1; i < cf->nranks; i++) {
            int cfd = accept(lfd, NULL, NULL);
            if (cfd < 0) {
                close(lfd);
                for (int r = 1; r < cf->nranks; r++) {
                    if (cf->peer_fds[r] > 0)
                        close(cf->peer_fds[r]);
                }
                free(cf->peer_fds);
                cf->peer_fds = NULL;
                return false;
            }
            (void)setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            int peer_rank = -1;
            if (!read_full(cfd, &peer_rank, sizeof(peer_rank)) || peer_rank <= 0 || peer_rank >= cf->nranks ||
                cf->peer_fds[peer_rank] != 0) {
                close(cfd);
                close(lfd);
                for (int r = 1; r < cf->nranks; r++) {
                    if (cf->peer_fds[r] > 0)
                        close(cf->peer_fds[r]);
                }
                free(cf->peer_fds);
                cf->peer_fds = NULL;
                return false;
            }
            cf->peer_fds[peer_rank] = cfd;
        }
        close(lfd);
        return true;
    }

    struct in_addr peer_ip;
    if (inet_pton(AF_INET, host, &peer_ip) != 1)
        return false;
    int fd = vxfec_bench_tcp_connect_retry(peer_ip, port, 200, 100000000L);
    if (fd < 0)
        return false;
    if (!write_full(fd, &cf->rank, sizeof(cf->rank))) {
        close(fd);
        return false;
    }
    cf->master_fd = fd;
    return true;
}

static void close_ctrl(ctrl_fabric_t *cf)
{
    if (cf->is_fork) {
        if (cf->tx_fd >= 0)
            close(cf->tx_fd);
        if (cf->rx_fd >= 0)
            close(cf->rx_fd);
        return;
    }
    if (cf->rank == 0 && cf->peer_fds) {
        for (int r = 1; r < cf->nranks; r++) {
            if (cf->peer_fds[r] > 0)
                close(cf->peer_fds[r]);
        }
        free(cf->peer_fds);
    } else if (cf->master_fd > 0) {
        close(cf->master_fd);
    }
}

typedef struct {
    char name[64];
    const char *codec_env;
    size_t elems;
    uint32_t loss_ppm;
    uint32_t burst;
    bool reorder;
    bool require_fec;
    int iters;
} demo_phase_t;

int main(int argc, char **argv)
{
    int arg_rank = -1;
    int arg_nranks = 2;
    int node_rank = -1;
    int nnodes = 0;
    int gpus_per_node = 0;
    int gpu_dev = 0;
    int ab_iters = 100;
    size_t max_mb = 4;
    bool enable_gdr = false;
    bool native_ib = false;
    bool ab_bench = false;
    bool train_step = false;
    bool clos_fabric = false;
    bool max_mb_set = false;
    bool nranks_set = false;
    enum demo_profile { PROFILE_STANDARD, PROFILE_B200, PROFILE_H200 } profile_id = PROFILE_STANDARD;
    bool have_seed = false;
    uint64_t seed = 0u;
    const char *master_spec = NULL;
    const char *ifname = NULL;
    const char *backend = NULL;
    const char *csv_path = NULL;
    const char *single_codec = NULL;
    const char *label_tag = NULL;
    const char *gpu_bdfs = NULL;
    const char *ibv_devs = NULL;
    const char *vk_devs = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc)
            arg_rank = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nranks") == 0 && i + 1 < argc) {
            arg_nranks = atoi(argv[++i]);
            nranks_set = true;
        } else if (strcmp(argv[i], "--node-rank") == 0 && i + 1 < argc)
            node_rank = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nnodes") == 0 && i + 1 < argc)
            nnodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpus-per-node") == 0 && i + 1 < argc)
            gpus_per_node = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc)
            gpu_dev = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu-bdfs") == 0 && i + 1 < argc)
            gpu_bdfs = argv[++i];
        else if (strcmp(argv[i], "--ibv-devs") == 0 && i + 1 < argc)
            ibv_devs = argv[++i];
        else if (strcmp(argv[i], "--vk-devs") == 0 && i + 1 < argc)
            vk_devs = argv[++i];
        else if (strcmp(argv[i], "--master") == 0 && i + 1 < argc)
            master_spec = argv[++i];
        else if (strcmp(argv[i], "--ifname") == 0 && i + 1 < argc)
            ifname = argv[++i];
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc)
            backend = argv[++i];
        else if (strcmp(argv[i], "--max-mb") == 0 && i + 1 < argc) {
            max_mb = (size_t)atol(argv[++i]);
            max_mb_set = true;
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            ab_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
            csv_path = argv[++i];
        else if (strcmp(argv[i], "--single-codec") == 0 && i + 1 < argc)
            single_codec = argv[++i];
        else if (strcmp(argv[i], "--label-tag") == 0 && i + 1 < argc)
            label_tag = argv[++i];
        else if (strcmp(argv[i], "--gdr") == 0)
            enable_gdr = true;
        else if (strcmp(argv[i], "--native-ib") == 0)
            native_ib = true;
        else if (strcmp(argv[i], "--ab-bench") == 0)
            ab_bench = true;
        else if (strcmp(argv[i], "--train-step") == 0)
            train_step = true;
        else if (strcmp(argv[i], "--clos-fabric") == 0)
            clos_fabric = true;
        else if (strcmp(argv[i], "--b200-four-case") == 0 || strcmp(argv[i], "--h200-four-case") == 0) {
            enum demo_profile requested = strcmp(argv[i], "--h200-four-case") == 0 ? PROFILE_H200 : PROFILE_B200;
            if (profile_id != PROFILE_STANDARD && profile_id != requested) {
                fprintf(stderr, "Select one four-case GPU profile\n");
                return 1;
            }
            profile_id = requested;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            const char *value = argv[++i];
            char *end = NULL;
            errno = 0;
            uintmax_t parsed = strtoumax(value, &end, 10);
            if (*value < '0' || *value > '9' || *end || errno || parsed > UINT64_MAX) {
                fprintf(stderr, "Invalid --seed '%s' (expected an unsigned 64-bit integer)\n", value);
                return 1;
            }
            seed = (uint64_t)parsed;
            have_seed = true;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            return 1;
        }
    }

    if (train_step && !nranks_set && !master_spec && gpus_per_node == 0 && nnodes == 0)
        arg_nranks = 4;
    if (train_step) {
        bool user_set_iters = false;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--iters") == 0)
                user_set_iters = true;
        }
        if (!user_set_iters)
            ab_iters = 40;
    }

    const bool four_case = profile_id != PROFILE_STANDARD;
    const char *model = profile_id == PROFILE_H200 ? "H200" : "B200";
    const char *four_case_profile = profile_id == PROFILE_H200 ? "h200-four-case" : "b200-four-case";
    if (train_step && four_case) {
        fprintf(stderr, "Select either --train-step or --%s\n", four_case_profile);
        return 1;
    }
    if (ab_iters < 1)
        ab_iters = 1;
    if (ab_iters > DEMO_MAX_ITERS)
        ab_iters = DEMO_MAX_ITERS;

    char cwd[PATH_MAX];
    char plugin_path[PATH_MAX + 64];
    const char *env_so = getenv("VXFEC_NCCL_SO_PATH");
    if (env_so && *env_so) {
        snprintf(plugin_path, sizeof(plugin_path), "%s", env_so);
    } else {
        if (!getcwd(cwd, sizeof(cwd)))
            return 1;
        snprintf(plugin_path, sizeof(plugin_path), "%s/libnccl-vxfec.so", cwd);
    }
    if (!native_ib && access(plugin_path, R_OK) != 0) {
        fprintf(stderr, "Missing %s (run `make` first)\n", plugin_path);
        return 1;
    }

    /* set the baseline before probing the libraries so automatic NET selection cannot fall back to sockets */
    if (native_ib && (setenv("NCCL_NET_PLUGIN", "none", 1) || setenv("NCCL_NET", "IB", 1) || setenv("NCCL_IB_DISABLE", "0", 1))) {
        perror("selecting built-in IB");
        return 1;
    }

    int local_rank = 0;
    bool multi_gpu_node = false;
    pid_t spawned_pids[DEMO_MAX_LOCAL_GPUS];
    int n_spawned = 0;
    memset(spawned_pids, 0, sizeof(spawned_pids));

    const bool node_launch = gpus_per_node != 0 || nnodes != 0 || node_rank != -1;
    if (node_launch) {
        if (!master_spec || gpus_per_node < 1 || gpus_per_node > DEMO_MAX_LOCAL_GPUS || nnodes < 1 ||
            nnodes > 16384 / gpus_per_node || nnodes * gpus_per_node < 2 || node_rank < 0 || node_rank >= nnodes) {
            fprintf(stderr, "Invalid --node-rank %d / --nnodes %d / --gpus-per-node %d\n", node_rank, nnodes, gpus_per_node);
            return 1;
        }
    }
    if (four_case) {
        int profile_ranks = node_launch ? nnodes * gpus_per_node : arg_nranks;
        bool output_rank = node_launch ? node_rank == 0 : (!master_spec || arg_rank == 0);
        if (profile_ranks != 2 || (node_launch && gpus_per_node != 1) || !have_seed || (output_rank && (!csv_path || !*csv_path)) ||
            !enable_gdr || (!native_ib && (!backend || strcmp(backend, "ibv") != 0))) {
            fprintf(stderr,
                    "--%s requires two ranks (one per node), --seed, --gdr, "
                    "--native-ib or --backend ibv, and --csv on rank zero\n",
                    four_case_profile);
            return 1;
        }
        if (setenv("NCCL_ALGO", "Ring", 1) || setenv("NCCL_MIN_NCHANNELS", "1", 1) || setenv("NCCL_MAX_NCHANNELS", "1", 1) ||
            setenv("NCCL_MIN_CTAS", "1", 1) || setenv("NCCL_MAX_CTAS", "1", 1)) {
            perror("selecting one NCCL Ring channel");
            return 1;
        }
    } else if (have_seed) {
        fprintf(stderr, "--seed requires --b200-four-case or --h200-four-case\n");
        return 1;
    }
    int local_ranks = node_launch ? gpus_per_node : 1;
    char bdf_map[DEMO_MAX_LOCAL_GPUS][64], ibv_map[DEMO_MAX_LOCAL_GPUS][64], vk_map[DEMO_MAX_LOCAL_GPUS][64];
    if (!parse_rank_map("--gpu-bdfs", gpu_bdfs, local_ranks, bdf_map) ||
        !parse_rank_map("--ibv-devs", ibv_devs, local_ranks, ibv_map) || !parse_rank_map("--vk-devs", vk_devs, local_ranks, vk_map))
        return 1;

    const bool require_ibv_gdr = !native_ib && enable_gdr && backend && strcmp(backend, "ibv") == 0;
    const bool check_rails =
        node_launch && nnodes > 1 && local_ranks > 1 && ibv_devs && enable_gdr && (native_ib || (ab_bench && require_ibv_gdr));
    demo_rail_t rails[DEMO_MAX_LOCAL_GPUS] = {0};
    char full_hca_list[1 + sizeof(ibv_map) + DEMO_MAX_LOCAL_GPUS] = "";
    size_t hca_list_len = 0;
    if (native_ib)
        full_hca_list[hca_list_len++] = '=';
    for (int i = 0; ibv_devs && i < local_ranks; i++) {
        if (check_rails) {
            if (!parse_rail(ibv_map[i], &rails[i])) {
                fprintf(stderr, "Invalid --ibv-devs port '%s'\n", ibv_map[i]);
                return 1;
            }
            for (int prev = 0; prev < i; prev++) {
                if (rails[i].port == rails[prev].port && strcmp(rails[i].device, rails[prev].device) == 0) {
                    fprintf(stderr, "Duplicate --ibv-devs port '%s'\n", ibv_map[i]);
                    return 1;
                }
            }
            snprintf(ibv_map[i], sizeof(ibv_map[i]), "%.*s:%u", DEMO_HCA_NAME_MAX, rails[i].device, rails[i].port);
        }
        if (i > 0)
            full_hca_list[hca_list_len++] = ',';
        size_t len = strlen(ibv_map[i]);
        memcpy(full_hca_list + hca_list_len, ibv_map[i], len + 1);
        hca_list_len += len;
    }
    if (ibv_devs && setenv(native_ib ? "NCCL_IB_HCA" : "VXFEC_IBV_DEVS", full_hca_list, 1) != 0) {
        perror("setting the node HCA inventory");
        return 1;
    }
    if (check_rails && !native_ib) {
        unsetenv("VXFEC_NCCL_DEV_SLICE");
        unsetenv("VXFEC_NCCL_RAILS");
        if (setenv("VXFEC_NCCL_STRICT_PIX_MR", "1", 0) != 0) {
            perror("setting VXFEC_NCCL_STRICT_PIX_MR=1");
            return 1;
        }
    }
    if (enable_gdr && (native_ib || require_ibv_gdr) && setenv("NCCL_PROTO", "Simple", 1) != 0) {
        perror("selecting NCCL_PROTO=Simple");
        return 1;
    }
    if (!probe_gpu_ccl_installed(four_case)) {
        if (four_case) {
            fprintf(stderr, "%s four-case profile requires a CUDA/NCCL runtime\n", model);
            return 1;
        }
        if (master_spec || csv_path || ab_bench || native_ib || require_ibv_gdr) {
            fprintf(stderr, "AllReduce benchmark requires a CUDA/NCCL or HIP/RCCL runtime\n");
            return 1;
        }
        puts("SKIP: Neither ROCm RCCL (`librccl.so.1`) nor NVIDIA NCCL (`libnccl.so.2`) is installed.");
        puts("      Run `./bench_nccl` for the standalone plugin benchmark.");
        return 0;
    }

    if (node_launch) {
        multi_gpu_node = (gpus_per_node > 1);
        arg_nranks = nnodes * gpus_per_node;
        for (int lr = 1; lr < gpus_per_node; lr++) {
            pid_t p = fork();
            if (p < 0) {
                for (int k = 0; k < n_spawned; k++)
                    (void)waitpid(spawned_pids[k], NULL, 0);
                return 1;
            }
            if (p == 0) {
                local_rank = lr;
                n_spawned = 0;
                break;
            }
            spawned_pids[n_spawned++] = p;
        }
        arg_rank = node_rank * gpus_per_node + local_rank;
        gpu_dev = local_rank;
    }

    ctrl_fabric_t cf;
    memset(&cf, 0, sizeof(cf));
    cf.tx_fd = -1;
    cf.rx_fd = -1;
    cf.master_fd = -1;

    if (master_spec && (arg_nranks < 2 || arg_nranks > 16384 || arg_rank < 0 || arg_rank >= arg_nranks)) {
        fprintf(stderr, "Invalid --rank %d / --nranks %d (supported multi-host range: 2..16384)\n", arg_rank, arg_nranks);
        return 1;
    }

    if (master_spec && arg_rank >= 0 && arg_nranks >= 2) {
        cf.is_fork = false;
        cf.rank = arg_rank;
        cf.nranks = arg_nranks;
        if (!setup_tcp_ctrl(&cf, master_spec)) {
            fprintf(stderr, "rank %d: failed TCP control bootstrap with %s\n", arg_rank, master_spec);
            for (int k = 0; k < n_spawned; k++)
                (void)waitpid(spawned_pids[k], NULL, 0);
            return 1;
        }
    } else if (arg_nranks > 2) {
        if (arg_nranks > DEMO_MAX_LOCAL_GPUS) {
            fprintf(stderr, "Local fork mode supports up to %d ranks (got %d)\n", DEMO_MAX_LOCAL_GPUS, arg_nranks);
            return 1;
        }
        cf.is_fork = false;
        cf.rank = 0;
        cf.nranks = arg_nranks;
        cf.peer_fds = (int *)calloc((size_t)arg_nranks, sizeof(int));
        if (!cf.peer_fds)
            return 1;
        for (int r = 1; r < arg_nranks; r++) {
            int sv[2] = {-1, -1};
            if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
                for (int k = 0; k < n_spawned; k++)
                    (void)waitpid(spawned_pids[k], NULL, 0);
                close_ctrl(&cf);
                return 1;
            }
            pid_t pid = fork();
            if (pid < 0) {
                close(sv[0]);
                close(sv[1]);
                for (int k = 0; k < n_spawned; k++)
                    (void)waitpid(spawned_pids[k], NULL, 0);
                close_ctrl(&cf);
                return 1;
            }
            if (pid == 0) {
                for (int prev = 1; prev < r; prev++) {
                    if (cf.peer_fds[prev] > 0)
                        close(cf.peer_fds[prev]);
                }
                free(cf.peer_fds);
                cf.peer_fds = NULL;
                close(sv[0]);
                cf.rank = r;
                cf.master_fd = sv[1];
                n_spawned = 0;
                break;
            }
            close(sv[1]);
            cf.peer_fds[r] = sv[0];
            spawned_pids[n_spawned++] = pid;
        }
    } else {
        int p0to1[2], p1to0[2];
        if (pipe(p0to1) != 0 || pipe(p1to0) != 0)
            return 1;
        pid_t pid = fork();
        if (pid < 0)
            return 1;
        cf.is_fork = true;
        cf.child_pid = pid;
        cf.rank = (pid == 0) ? 1 : 0;
        cf.nranks = 2;
        cf.tx_fd = (cf.rank == 0) ? p0to1[1] : p1to0[1];
        cf.rx_fd = (cf.rank == 0) ? p1to0[0] : p0to1[0];
        close((cf.rank == 0) ? p0to1[0] : p1to0[0]);
        close((cf.rank == 0) ? p1to0[1] : p0to1[1]);
    }

    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);

    char host_id[64];
    if (multi_gpu_node)
        snprintf(host_id, sizeof(host_id), "vxfec-host-%d", node_rank);
    else
        snprintf(host_id, sizeof(host_id), "vxfec-node-%d", cf.rank);
    setenv("NCCL_HOSTID", host_id, 1);

    if (!multi_gpu_node || (native_ib && nnodes == 1)) {
        setenv("NCCL_P2P_DISABLE", "1", 1);
        setenv("NCCL_SHM_DISABLE", "1", 1);
    }

    if (!native_ib) {
        setenv("NCCL_NET_PLUGIN", plugin_path, 1);
        setenv("NCCL_NET", "vxfec-ecmp-harq", 1);
        setenv("NCCL_IB_DISABLE", "1", 1);
        if (backend && *backend)
            setenv("VXFEC_NCCL_BACKEND", backend, 1);
    }

    if (enable_gdr) {
        setenv("NCCL_GDR_LEVEL", "5", 1);
        setenv("NCCL_NET_GDR_LEVEL", "5", 1);
    } else {
        setenv("NCCL_GDR_LEVEL", "0", 1);
    }
    if (ifname && *ifname) {
        setenv("NCCL_SOCKET_IFNAME", ifname, 0);
        if (!native_ib)
            setenv("VXFEC_NCCL_IFNAMES", ifname, 1);
    } else if (cf.is_fork || !master_spec) {
        setenv("NCCL_SOCKET_IFNAME", "lo", 1);
    }
    if (train_step)
        setenv("NCCL_PROTO", "Simple", 0);
    if (!getenv("NCCL_DEBUG"))
        setenv("NCCL_DEBUG", "WARN", 1);

    gpu_ccl_api_t api;
    memset(&api, 0, sizeof(api));
    nccl_comm_t comm = NULL;
    void *plug_dl = NULL;
    float *h_in = NULL, *h_out = NULL;
    void *d_in = NULL, *d_out = NULL;
    gpu_stream_t stream = NULL;
    FILE *csv_fp = NULL;
    FILE *samples_fp = NULL;
    char samples_path[PATH_MAX];
    size_t n_phases = 0;
    bool csv_regular = false;
    bool samples_regular = false;
    bool ok = false;

    /* Both ranks need the same settings even when launched independently. */
    struct {
        uint64_t id, native, seed, iters;
    } profile = {profile_id, native_ib, seed, (uint64_t)ab_iters};
    bool profile_ready = false;
    if (!ctrl_bcast(&cf, &profile, sizeof(profile)) ||
        !ctrl_all_ok(&cf,
                     profile.id == profile_id && (!four_case || (profile.native == native_ib && profile.seed == seed &&
                                                                 profile.iters == (uint64_t)ab_iters)),
                     &profile_ready) ||
        !profile_ready) {
        if (cf.rank == 0)
            fprintf(stderr, "ERROR: ranks disagree on GPU profile, transport, seed or iteration count\n");
        goto cleanup;
    }

    bool device_ok = true;
    if (!native_ib && vk_devs && setenv("VXFEC_VK_DEVICE", vk_map[local_rank], 1) != 0) {
        perror("setting the per-rank Vulkan device");
        device_ok = false;
    }
    if (device_ok && !load_gpu_ccl(&api, four_case)) {
        fprintf(stderr, "rank %d: failed to load GPU/CCL runtime\n", cf.rank);
        device_ok = false;
    }
    if (enable_gdr) {
        if (api.backend_name && strcmp(api.backend_name, "ROCm HIP + RCCL") == 0) {
            struct utsname uts;
            char cfg_path[sizeof(uts.release) + 32];
            if (uname(&uts) == 0 && snprintf(cfg_path, sizeof(cfg_path), "/boot/config-%s", uts.release) > 0 &&
                access(cfg_path, R_OK) == 0) {
                setenv("NCCL_DMABUF_ENABLE", "1", 0);
            }
        } else {
            setenv("NCCL_DMABUF_ENABLE", "1", 1);
        }
    }

    if (device_ok && gpu_bdfs && (!api.dev_by_pci || api.dev_by_pci(&gpu_dev, bdf_map[local_rank]) != 0 || gpu_dev < 0)) {
        fprintf(stderr, "rank %d: GPU PCI address '%s' did not resolve in %s\n", cf.rank, bdf_map[local_rank], api.backend_name);
        device_ok = false;
    }
    if (device_ok && api.set_device(gpu_dev) != 0) {
        fprintf(stderr, "rank %d: failed to set GPU device %d\n", cf.rank, gpu_dev);
        device_ok = false;
    }
    bool devices_ready = false;
    if (!ctrl_all_ok(&cf, device_ok, &devices_ready) || !devices_ready)
        goto cleanup;

    if (!init_collective(&cf, &api, &comm, native_ib))
        goto cleanup;

    get_stats_ex_fn get_stats_ex = NULL;
    get_stats_fn get_stats = NULL;
    get_rail_stats_fn get_rail_stats = NULL;
    reset_stats_fn reset_stats = NULL;
    set_fault_fn set_fault = NULL;
    set_paired_fault_fn set_paired_fault = NULL;
    get_paired_stats_fn get_paired_stats = NULL;
    set_codec_fn set_codec = NULL;

    bool plugin_ready = true;
    if (!native_ib) {
        plug_dl = dlopen(plugin_path, RTLD_NOLOAD | RTLD_NOW);
        get_stats_ex = plug_dl ? (get_stats_ex_fn)dlsym(plug_dl, "vxfec_nccl_get_stats_ex") : NULL;
        get_stats = plug_dl ? (get_stats_fn)dlsym(plug_dl, "vxfec_nccl_get_stats") : NULL;
        get_rail_stats = plug_dl ? (get_rail_stats_fn)dlsym(plug_dl, "vxfec_nccl_get_rail_stats") : NULL;
        reset_stats = plug_dl ? (reset_stats_fn)dlsym(plug_dl, "vxfec_nccl_reset_stats") : NULL;
        set_fault = plug_dl ? (set_fault_fn)dlsym(plug_dl, "vxfec_nccl_set_fault_injection") : NULL;
        set_paired_fault = plug_dl ? (set_paired_fault_fn)dlsym(plug_dl, "vxfec_nccl_set_paired_fault") : NULL;
        get_paired_stats = plug_dl ? (get_paired_stats_fn)dlsym(plug_dl, "vxfec_nccl_get_paired_fault_stats") : NULL;
        set_codec = plug_dl ? (set_codec_fn)dlsym(plug_dl, "vxfec_nccl_set_codec_override") : NULL;
        if (!plug_dl || (!get_stats_ex && !get_stats) || !reset_stats || !set_fault || !set_codec ||
            (require_ibv_gdr && !get_stats_ex) || (check_rails && !get_rail_stats)) {
            fprintf(stderr, "rank %d: vxfec plugin (%s) was not selected by NCCL/RCCL or lacks fault/codec hooks\n", cf.rank,
                    plugin_path);
            plugin_ready = false;
        } else if (four_case && (!set_paired_fault || !get_paired_stats || !set_paired_fault(0u, 1u, seed, cf.rank, 0u))) {
            fprintf(stderr, "rank %d: vxfec plugin lacks enabled paired fault hooks\n", cf.rank);
            plugin_ready = false;
        } else if (check_rails && !snapshot_plugin_rails(get_rail_stats, rails, local_ranks, false)) {
            fprintf(stderr, "rank %d: vxfec device IDs do not match the requested HCA inventory %s\n", cf.rank, full_hca_list);
            plugin_ready = false;
        }
    }
    if (!ctrl_all_ok(&cf, plugin_ready, &devices_ready) || !devices_ready)
        goto cleanup;

    if (ab_bench || native_ib || train_step) {
        if (max_mb < 8)
            max_mb = 8;
    } else if (max_mb < 4) {
        max_mb = 4;
    }
    const size_t large_elems = (four_case ? DEMO_FOUR_CASE_BYTES : max_mb * 1024ULL * 1024ULL) / sizeof(float);
    const size_t max_bytes = large_elems * sizeof(float);
    h_in = (float *)malloc(max_bytes);
    h_out = (float *)calloc(large_elems, sizeof(float));

    bool buffers_ready = h_in && h_out && api.malloc_dev(&d_in, max_bytes) == 0 && api.malloc_dev(&d_out, max_bytes) == 0 &&
                         api.stream_create(&stream) == 0;
    if (!buffers_ready)
        fprintf(stderr, "rank %d: GPU buffer allocation failed\n", cf.rank);
    if (!ctrl_all_ok(&cf, buffers_ready, &devices_ready) || !devices_ready)
        goto cleanup;

    if (train_step) {
        typedef void (*set_cong_fn)(uint32_t, uint32_t, uint64_t);
        typedef void (*set_delay_fn)(uint64_t);
        set_cong_fn set_cong = (!native_ib && plug_dl) ? (set_cong_fn)dlsym(plug_dl, "vxfec_nccl_set_congestion_sim") : NULL;
        set_delay_fn set_delay = (!native_ib && plug_dl) ? (set_delay_fn)dlsym(plug_dl, "vxfec_nccl_set_link_delay_ns") : NULL;
        bool has_p2p = api.group_start && api.group_end && api.ccl_send && api.ccl_recv;
        demo_sgemm_ctx_t sgemm = {0};
        int gemm_n = 1536;
        int gemm_k = 2048;
        bool gemm_ok = has_p2p && demo_sgemm_init(&api, &sgemm, gemm_n, gemm_k);
        if (!gemm_ok)
            fprintf(stderr, "rank %d: failed to initialize RTC SGEMM or NCCL/RCCL P2P AllToAll symbols\n", cf.rank);
        if (!ctrl_all_ok(&cf, gemm_ok, &devices_ready) || !devices_ready) {
            demo_sgemm_free(&api, &sgemm);
            goto cleanup;
        }

        /* 1) Measure solo GPU SGEMM compute baseline across all active ranks concurrently */
        for (int w = 0; w < 4; w++) {
            if (!demo_sgemm_launch(&sgemm) || api.stream_sync(sgemm.comp_stream) != 0) {
                demo_sgemm_free(&api, &sgemm);
                goto cleanup;
            }
        }
        if (!ctrl_barrier(&cf)) {
            demo_sgemm_free(&api, &sgemm);
            goto cleanup;
        }
        int solo_iters = (ab_iters < 30) ? ab_iters : 30;
        uint64_t solo_my[DEMO_MAX_ITERS] = {0}, solo_max[DEMO_MAX_ITERS] = {0};
        for (int it = 0; it < solo_iters; it++) {
            if (!ctrl_barrier(&cf)) {
                demo_sgemm_free(&api, &sgemm);
                goto cleanup;
            }
            uint64_t t0 = now_ns();
            if (!demo_sgemm_launch(&sgemm) || api.stream_sync(sgemm.comp_stream) != 0) {
                demo_sgemm_free(&api, &sgemm);
                goto cleanup;
            }
            solo_my[it] = now_ns() - t0;
        }
        if (!ctrl_gather_max_lats(&cf, solo_my, solo_max, solo_iters)) {
            demo_sgemm_free(&api, &sgemm);
            goto cleanup;
        }
        qsort(solo_max, (size_t)solo_iters, sizeof(uint64_t), cmp_u64_asc);
        double solo_p50_ms = (double)solo_max[solo_iters / 2] / 1e6;
        double flops_per_rank = 2.0 * (double)gemm_n * (double)gemm_n * (double)gemm_k;
        double agg_flops = flops_per_rank * (double)cf.nranks;
        double solo_tflops = agg_flops / ((solo_p50_ms * 1e6) * 1e3);

        typedef struct {
            char label[64];
            const char *codec;
            bool is_alltoall;
            size_t total_bytes;
            uint32_t loss_ppm;
            uint32_t burst;
            uint32_t ecn_ppm;
            uint32_t hot_mask;
            uint64_t one_way_delay_ns;
        } train_case_t;

        /* 1.0 ms one-way propagation delay = 2.0 ms cross-pod / metro RoCEv2 AI fabric RTT */
        const uint64_t pod_delay_ns = clos_fabric ? 0ULL : 1000000ULL;
        const size_t ar_mb = max_mb_set ? max_mb : 8u;
        const size_t a2a_mb = (ar_mb >= 4u) ? 2u : 1u;
        const size_t a2a_bytes = a2a_mb * 1024ULL * 1024ULL;
        const size_t ar_bytes = ar_mb * 1024ULL * 1024ULL;
        train_case_t tcases[8];
        size_t n_tcases = 0u;

        if (clos_fabric) {
            if (native_ib) {
                const char *tg = (label_tag && *label_tag) ? label_tag : "NCCL_IB RC";
                tcases[n_tcases] = (train_case_t){"", "rc", true, a2a_bytes, 0u, 1u, 0u, 0u, 0ULL};
                snprintf(tcases[n_tcases++].label, sizeof(tcases[0].label), "MoE AllToAll %zuMB | %s", a2a_mb, tg);
                tcases[n_tcases] = (train_case_t){"", "rc", false, ar_bytes, 0u, 1u, 0u, 0u, 0ULL};
                snprintf(tcases[n_tcases++].label, sizeof(tcases[0].label), "DP AllReduce %zuMB | %s", ar_mb, tg);
            } else if (single_codec && *single_codec) {
                const char *tg = (label_tag && *label_tag) ? label_tag : single_codec;
                tcases[n_tcases] = (train_case_t){"", single_codec, true, a2a_bytes, 0u, 1u, 0u, 0u, 0ULL};
                snprintf(tcases[n_tcases++].label, sizeof(tcases[0].label), "MoE AllToAll %zuMB | %s", a2a_mb, tg);
                tcases[n_tcases] = (train_case_t){"", single_codec, false, ar_bytes, 0u, 1u, 0u, 0u, 0ULL};
                snprintf(tcases[n_tcases++].label, sizeof(tcases[0].label), "DP AllReduce %zuMB | %s", ar_mb, tg);
            } else {
                tcases[n_tcases] = (train_case_t){"", "gbn", true, a2a_bytes, 0u, 1u, 0u, 0u, 0ULL};
                snprintf(tcases[n_tcases++].label, sizeof(tcases[0].label), "MoE AllToAll %zuMB | Clos (GBN)", a2a_mb);
                tcases[n_tcases] = (train_case_t){"", "sack", true, a2a_bytes, 0u, 1u, 0u, 0u, 0ULL};
                snprintf(tcases[n_tcases++].label, sizeof(tcases[0].label), "MoE AllToAll %zuMB | Clos (SACK)", a2a_mb);
                tcases[n_tcases] = (train_case_t){"", "auto", true, a2a_bytes, 0u, 1u, 0u, 0u, 0ULL};
                snprintf(tcases[n_tcases++].label, sizeof(tcases[0].label), "MoE AllToAll %zuMB | Clos (vxfec)", a2a_mb);
                tcases[n_tcases] = (train_case_t){"", "gbn", false, ar_bytes, 0u, 1u, 0u, 0u, 0ULL};
                snprintf(tcases[n_tcases++].label, sizeof(tcases[0].label), "DP AllReduce %zuMB | Clos (GBN)", ar_mb);
                tcases[n_tcases] = (train_case_t){"", "sack", false, ar_bytes, 0u, 1u, 0u, 0u, 0ULL};
                snprintf(tcases[n_tcases++].label, sizeof(tcases[0].label), "DP AllReduce %zuMB | Clos (SACK)", ar_mb);
                tcases[n_tcases] = (train_case_t){"", "auto", false, ar_bytes, 0u, 1u, 0u, 0u, 0ULL};
                snprintf(tcases[n_tcases++].label, sizeof(tcases[0].label), "DP AllReduce %zuMB | Clos (vxfec)", ar_mb);
            }
        } else {
            const train_case_t def_cases[] = {
                {"MoE AllToAll 2MB | 0% Loss (vxfec)", "auto", true, a2a_bytes, 0u, 1u, 0u, 0u, pod_delay_ns},
                {"MoE AllToAll 2MB | 2% Burst (GBN)", "gbn", true, a2a_bytes, 20000u, 6u, 0u, 0u, pod_delay_ns},
                {"MoE AllToAll 2MB | 2% Burst (SACK)", "sack", true, a2a_bytes, 20000u, 6u, 0u, 0u, pod_delay_ns},
                {"MoE AllToAll 2MB | 2% Burst (vxfec)", "auto", true, a2a_bytes, 20000u, 6u, 0u, 0u, pod_delay_ns},
                {"DP AllReduce 8MB | 0% Loss (vxfec)", "auto", false, ar_bytes, 0u, 1u, 0u, 0u, pod_delay_ns},
                {"DP AllReduce 8MB | 2% Burst (GBN)", "gbn", false, ar_bytes, 20000u, 6u, 0u, 0u, pod_delay_ns},
                {"DP AllReduce 8MB | 2% Burst (SACK)", "sack", false, ar_bytes, 20000u, 6u, 0u, 0u, pod_delay_ns},
                {"DP AllReduce 8MB | 2% Burst (vxfec)", "auto", false, ar_bytes, 20000u, 6u, 0u, 0u, pod_delay_ns},
            };
            memcpy(tcases, def_cases, sizeof(def_cases));
            n_tcases = sizeof(def_cases) / sizeof(def_cases[0]);
        }

        if (cf.rank == 0) {
            printf(
                "================================================================================================================="
                "==============\n");
            printf("vxfec Overlapped AI Training Step Benchmark (`%s`, %d Ranks, N=%d, SGEMM %dx%dx%d/rank)\n", api.backend_name,
                   cf.nranks, ab_iters, gemm_n, gemm_n, gemm_k);
            printf("  Solo GPU Compute Baseline : p50 = %.2f ms | Aggregate Compute = %.2f TFLOPS (100.0%%)\n", solo_p50_ms,
                   solo_tflops);
            printf(
                "================================================================================================================="
                "==============\n");
            printf("  Training Collective + SGEMM       |  p50 Step |  p99 Step | p99 Stall | Eff TFLOPS (Ret) | Drops | SACK | "
                   "HARQ-IR | RS8/RQ\n");
            printf(
                "------------------------------------+-----------+-----------+-----------+------------------+-------+------+------"
                "---+--------\n");
            fflush(stdout);
        }

        ok = true;
        for (size_t tc = 0; tc < n_tcases && ok; tc++) {
            if (native_ib && tcases[tc].loss_ppm > 0u)
                continue;
            size_t bytes = ((tcases[tc].total_bytes / (size_t)cf.nranks) / sizeof(float)) * sizeof(float) * (size_t)cf.nranks;
            size_t elems = bytes / sizeof(float);
            size_t peer_elems = elems / (size_t)cf.nranks;

            if (!native_ib) {
                set_codec(tcases[tc].codec);
                if (set_delay)
                    set_delay(tcases[tc].one_way_delay_ns);
                if (set_cong)
                    set_cong(tcases[tc].ecn_ppm, tcases[tc].hot_mask, 0ULL);
                set_fault(tcases[tc].loss_ppm, tcases[tc].burst, tcases[tc].loss_ppm > 0u);
            }
            if (!ctrl_barrier(&cf)) {
                ok = false;
                break;
            }

            /* warmup overlapped GEMM + collective */
            bool w_ok = demo_sgemm_launch(&sgemm);
            if (w_ok) {
                if (tcases[tc].is_alltoall) {
                    w_ok = (api.group_start() == 0);
                    for (int r = 0; r < cf.nranks && w_ok; r++) {
                        const float *sp = (const float *)d_in + (size_t)r * peer_elems;
                        float *rp = (float *)d_out + (size_t)r * peer_elems;
                        w_ok = api.ccl_send(sp, peer_elems, NCCL_DT_FLOAT32, r, comm, stream) == 0 &&
                               api.ccl_recv(rp, peer_elems, NCCL_DT_FLOAT32, r, comm, stream) == 0;
                    }
                    w_ok = w_ok && (api.group_end() == 0);
                } else {
                    w_ok = api.all_reduce(d_in, d_out, elems, NCCL_DT_FLOAT32, NCCL_OP_SUM, comm, stream) == 0;
                }
            }
            w_ok = w_ok && api.stream_sync(stream) == 0 && api.stream_sync(sgemm.comp_stream) == 0;
            if (!ctrl_all_ok(&cf, w_ok, &ok) || !ok)
                break;

            if (!native_ib)
                reset_stats();
            if (!ctrl_barrier(&cf)) {
                ok = false;
                break;
            }

            uint64_t step_my[DEMO_MAX_ITERS] = {0}, step_max[DEMO_MAX_ITERS] = {0};
            for (int it = 0; it < ab_iters; it++) {
                uint32_t tag_seed = (uint32_t)(tc * 1024u + (size_t)it + 1u);
                if (tcases[tc].is_alltoall) {
                    for (int dst = 0; dst < cf.nranks; dst++) {
                        for (size_t i = 0; i < peer_elems; i++) {
                            h_in[(size_t)dst * peer_elems + i] =
                                (float)((uint32_t)(cf.rank + 1) * 256u + (uint32_t)(dst + 1) * 16u + (tag_seed & 63u) +
                                        (uint32_t)(i % 31u));
                        }
                    }
                } else {
                    uint32_t rwt = (uint32_t)((cf.rank % 32) + 1) * 16u;
                    for (size_t i = 0; i < elems; i++)
                        h_in[i] = (float)(rwt + (tag_seed & 63u) + (uint32_t)(i % 31u));
                }
                memset(h_out, 0, bytes);
                bool prep_ok = api.memcpy_dev(d_in, h_in, bytes, GPU_MEMCPY_H2D) == 0 &&
                               api.memcpy_dev(d_out, h_out, bytes, GPU_MEMCPY_H2D) == 0 && api.stream_sync(NULL) == 0;
                if (!ctrl_all_ok(&cf, prep_ok, &ok) || !ok)
                    break;
                if (!ctrl_barrier(&cf)) {
                    ok = false;
                    break;
                }

                uint64_t t0 = now_ns();
                bool it_ok = demo_sgemm_launch(&sgemm);
                if (it_ok) {
                    if (tcases[tc].is_alltoall) {
                        it_ok = (api.group_start() == 0);
                        for (int r = 0; r < cf.nranks && it_ok; r++) {
                            const float *sp = (const float *)d_in + (size_t)r * peer_elems;
                            float *rp = (float *)d_out + (size_t)r * peer_elems;
                            it_ok = api.ccl_send(sp, peer_elems, NCCL_DT_FLOAT32, r, comm, stream) == 0 &&
                                    api.ccl_recv(rp, peer_elems, NCCL_DT_FLOAT32, r, comm, stream) == 0;
                        }
                        it_ok = it_ok && (api.group_end() == 0);
                    } else {
                        it_ok = api.all_reduce(d_in, d_out, elems, NCCL_DT_FLOAT32, NCCL_OP_SUM, comm, stream) == 0;
                    }
                }
                it_ok = it_ok && api.stream_sync(stream) == 0 && api.stream_sync(sgemm.comp_stream) == 0;
                step_my[it] = now_ns() - t0;

                it_ok = it_ok && api.memcpy_dev(h_out, d_out, bytes, GPU_MEMCPY_D2H) == 0;
                if (it_ok) {
                    if (tcases[tc].is_alltoall) {
                        for (int src = 0; src < cf.nranks && it_ok; src++) {
                            for (size_t i = 0; i < peer_elems; i++) {
                                float exp = (float)((uint32_t)(src + 1) * 256u + (uint32_t)(cf.rank + 1) * 16u + (tag_seed & 63u) +
                                                    (uint32_t)(i % 31u));
                                if (h_out[(size_t)src * peer_elems + i] != exp) {
                                    fprintf(stderr, "rank %d: AllToAll mismatch in '%s' iter %d from src %d\n", cf.rank,
                                            tcases[tc].label, it, src);
                                    it_ok = false;
                                    break;
                                }
                            }
                        }
                    } else {
                        uint32_t rsum = 0u;
                        for (int r = 0; r < cf.nranks; r++)
                            rsum += (uint32_t)((r % 32) + 1) * 16u;
                        for (size_t i = 0; i < elems; i++) {
                            float exp = (float)(rsum + (uint32_t)cf.nranks * ((tag_seed & 63u) + (uint32_t)(i % 31u)));
                            if (h_out[i] != exp) {
                                fprintf(stderr, "rank %d: AllReduce mismatch in '%s' iter %d\n", cf.rank, tcases[tc].label, it);
                                it_ok = false;
                                break;
                            }
                        }
                    }
                }
                if (!ctrl_all_ok(&cf, it_ok, &ok) || !ok)
                    break;
            }
            if (!ok)
                break;

            if (!ctrl_gather_max_lats(&cf, step_my, step_max, ab_iters)) {
                ok = false;
                break;
            }
            vxfec_nccl_stats_t my_st = {0}, tot = {0};
            if (!native_ib) {
                if (get_stats_ex)
                    get_stats_ex(&my_st, sizeof(my_st));
                else
                    get_stats(&my_st);
                if (!ctrl_gather_stats(&cf, &my_st, &tot)) {
                    ok = false;
                    break;
                }
            }

            if (cf.rank == 0) {
                qsort(step_max, (size_t)ab_iters, sizeof(uint64_t), cmp_u64_asc);
                double p50_ms = (double)step_max[ab_iters / 2] / 1e6;
                size_t p99_idx = ((size_t)ab_iters * 99u + 99u) / 100u - 1u;
                double p99_ms = (double)step_max[p99_idx] / 1e6;
                double stall_ms = (p99_ms > solo_p50_ms) ? (p99_ms - solo_p50_ms) : 0.0;
                double p50_ns = (double)(step_max[ab_iters / 2] ? step_max[ab_iters / 2] : 1ULL);
                double eff_tflops = agg_flops / (p50_ns * 1e3);
                double ret_pct = (solo_tflops > 0.0) ? (100.0 * eff_tflops / solo_tflops) : 0.0;
                if (ret_pct > 100.0)
                    ret_pct = 100.0;
                uint64_t ir = tot.tx_harq_ir_pkts + tot.tx_proactive_rep_pkts;
                printf("  %-33s | %6.2f ms | %6.2f ms | +%5.2f ms | %5.2f TF (%5.1f%%) | %5" PRIu64 " | %4" PRIu64 " | %7" PRIu64
                       " | %3" PRIu64 "/%-2" PRIu64 "\n",
                       tcases[tc].label, p50_ms, p99_ms, stall_ms, eff_tflops, ret_pct, tot.sim_dropped_pkts, tot.tx_sack_retx_pkts,
                       ir, tot.rx_rs8_decodes, tot.rx_rq_decodes);
                fflush(stdout);
            }
        }

        if (!native_ib) {
            if (set_delay)
                set_delay(0ULL);
            if (set_cong)
                set_cong(0u, 0u, 0ULL);
            set_fault(0u, 1u, false);
            set_codec(NULL);
        }
        demo_sgemm_free(&api, &sgemm);
        if (cf.rank == 0 && ok) {
            printf(
                "================================================================================================================="
                "==============\n");
            printf("  Verified 100%% bit-exact GPU FP32 `AllToAll` (3->1 Incast) + `AllReduce` overlapped with GPU `SGEMM`.\n");
            printf(
                "================================================================================================================="
                "==============\n");
            fflush(stdout);
        }
        n_phases = 0;
        goto finish_train_step;
    }

    demo_phase_t phases[DEMO_MAX_PHASES];

    if (four_case) {
        phases[n_phases++] = (demo_phase_t){"plugin_clean", "auto", large_elems, 0, 1, false, false, ab_iters};
        if (native_ib)
            strcpy(phases[0].name, "native_clean");
        else {
            demo_phase_t sack = {"plugin_sack_loss",   "sack", large_elems, DEMO_FOUR_CASE_LOSS_PPM,
                                 DEMO_FOUR_CASE_BURST, false,  false,       ab_iters};
            demo_phase_t fec = {"plugin_auto_loss",   "auto", large_elems, DEMO_FOUR_CASE_LOSS_PPM,
                                DEMO_FOUR_CASE_BURST, false,  true,        ab_iters};
            phases[n_phases++] = (seed & 1u) ? sack : fec;
            phases[n_phases++] = (seed & 1u) ? fec : sack;
        }
    } else if (ab_bench || native_ib) {
        const size_t e_512k = (512ULL * 1024ULL) / sizeof(float);
        const size_t e_8m = (8ULL * 1024ULL * 1024ULL) / sizeof(float);
        const size_t e_64m = (64ULL * 1024ULL * 1024ULL) / sizeof(float);
        const size_t e_256m = (256ULL * 1024ULL * 1024ULL) / sizeof(float);
        const char *tag_lossless = native_ib ? "NCCL_IB RC" : ((backend && strcmp(backend, "ibv") == 0) ? "vxfec UC" : "vxfec UDP");

        phases[n_phases++] = (demo_phase_t){"", "auto", e_512k, 0, 1, false, false, ab_iters};
        snprintf(phases[n_phases - 1].name, sizeof(phases[0].name), "512 KiB | 0.0%% Lossless (%s)", tag_lossless);

        phases[n_phases++] = (demo_phase_t){"", "auto", e_8m, 0, 1, false, false, ab_iters};
        snprintf(phases[n_phases - 1].name, sizeof(phases[0].name), "8.0 MiB | 0.0%% Lossless (%s)", tag_lossless);

        if (max_mb >= 64) {
            phases[n_phases++] = (demo_phase_t){"", "auto", e_64m, 0, 1, false, false, ab_iters};
            snprintf(phases[n_phases - 1].name, sizeof(phases[0].name), "64 MiB  | 0.0%% Lossless (%s)", tag_lossless);
        }
        if (max_mb >= 256) {
            phases[n_phases++] = (demo_phase_t){"", "auto", e_256m, 0, 1, false, false, ab_iters};
            snprintf(phases[n_phases - 1].name, sizeof(phases[0].name), "256 MiB | 0.0%% Lossless (%s)", tag_lossless);
        }
        if (!native_ib) {
            phases[n_phases++] = (demo_phase_t){"512 KiB | 0.4% OOO Loss (SACK)", "sack", e_512k, 4000, 1, true, false, ab_iters};
            phases[n_phases++] = (demo_phase_t){"512 KiB | 0.4% OOO Loss (vxfec)", "auto", e_512k, 4000, 1, true, false, ab_iters};
            phases[n_phases++] = (demo_phase_t){"512 KiB | 2.0% Burst=6 (SACK)", "sack", e_512k, 20000, 6, true, false, ab_iters};
            phases[n_phases++] = (demo_phase_t){"512 KiB | 2.0% Burst=6 (vxfec)", "auto", e_512k, 20000, 6, true, true, ab_iters};
            phases[n_phases++] = (demo_phase_t){"8.0 MiB | 2.0% Burst=6 (SACK)", "sack", e_8m, 20000, 6, true, false, ab_iters};
            phases[n_phases++] = (demo_phase_t){"8.0 MiB | 2.0% Burst=6 (vxfec)", "auto", e_8m, 20000, 6, true, true, ab_iters};
            if (max_mb >= 64) {
                phases[n_phases++] =
                    (demo_phase_t){"64 MiB  | 2.0% Burst=6 (SACK)", "sack", e_64m, 20000, 6, true, false, ab_iters};
                phases[n_phases++] =
                    (demo_phase_t){"64 MiB  | 2.0% Burst=6 (vxfec)", "auto", e_64m, 20000, 6, true, true, ab_iters};
            }
            if (max_mb >= 256) {
                phases[n_phases++] =
                    (demo_phase_t){"256 MiB | 2.0% Burst=6 (SACK)", "sack", e_256m, 20000, 6, true, false, ab_iters};
                phases[n_phases++] =
                    (demo_phase_t){"256 MiB | 2.0% Burst=6 (vxfec)", "auto", e_256m, 20000, 6, true, true, ab_iters};
            }
        }
    } else {
        const size_t e_1m = 256 * 1024;
        phases[n_phases++] = (demo_phase_t){"1. Lossless Fabric (Baseline GBN)", "gbn", e_1m, 0, 1, false, false, 12};
        phases[n_phases++] = (demo_phase_t){"2. Lossless Fabric (vxfec)", "auto", e_1m, 0, 1, false, false, 12};
        phases[n_phases++] = (demo_phase_t){"3. 0.4% OOO + Drops (Baseline GBN)", "gbn", e_1m, 4000, 1, true, false, 10};
        phases[n_phases++] = (demo_phase_t){"4. 0.4% OOO + Drops (vxfec)", "auto", e_1m, 4000, 1, true, false, 10};
        phases[n_phases++] = (demo_phase_t){"5. 3.0% Burst=6 (Baseline GBN)", "gbn", e_1m, 30000, 6, true, false, 8};
        phases[n_phases++] = (demo_phase_t){"6. 3.0% Burst=6 (SACK)", "sack", e_1m, 30000, 6, true, false, 8};
        phases[n_phases++] = (demo_phase_t){"7. 3.0% Burst=6 (vxfec MDS)", "auto", e_1m, 30000, 6, true, true, 8};
        phases[n_phases++] = (demo_phase_t){"", "gbn", large_elems, 30000, 6, true, false, 6};
        snprintf(phases[n_phases - 1].name, sizeof(phases[0].name), "8. 3.0%% Burst=6 %zuMB (Baseline GBN)", max_mb);
        phases[n_phases++] = (demo_phase_t){"", "sack", large_elems, 30000, 6, true, false, 6};
        snprintf(phases[n_phases - 1].name, sizeof(phases[0].name), "9. 3.0%% Burst=6 %zuMB (SACK)", max_mb);
        phases[n_phases++] = (demo_phase_t){"", "auto", large_elems, 30000, 6, true, true, 6};
        snprintf(phases[n_phases - 1].name, sizeof(phases[0].name), "10. 3.0%% Burst=6 %zuMB (vxfec)", max_mb);
    }

    const bool use_ab_fmt = (four_case || ab_bench || native_ib);
    const char *xport_tag = native_ib ? "nccl-ib-rc" : ((backend && strcmp(backend, "ibv") == 0) ? "vxfec-ibv-uc" : "vxfec-udp");

    if (cf.rank == 0 && csv_path && *csv_path) {
        csv_fp = fopen(csv_path, "w");
        if (csv_fp) {
            struct stat csv_st;
            csv_regular = fstat(fileno(csv_fp), &csv_st) == 0 && S_ISREG(csv_st.st_mode);
            if (use_ab_fmt) {
                fprintf(csv_fp,
                        "phase,transport,codec,nranks,tensor_mb,iters,p50_ms,p99_ms,avg_ms,alg_gbps,bus_gbps,drops,sack,harq_ir,"
                        "rs8_dec,rq_dec,vk_calls,ibv_pkts");
                if (four_case)
                    fputs(",profile,seed,impairment_scope,loss_ppm,burst,systematic_trials,drop_fingerprint,"
                          "cuda_rx_repairs,fallback_rx_repairs",
                          csv_fp);
                fputc('\n', csv_fp);
            } else {
                fprintf(csv_fp, "phase,codec,nranks,tensor_mb,avg_ms,alg_gbps,bus_gbps,drops,sack,harq_ir,rs8_dec,rq_dec,vk_calls,"
                                "ibv_pkts\n");
            }
        } else {
            perror(csv_path);
        }
    }
    bool csv_ready = cf.rank != 0 || !csv_path || !*csv_path || (csv_fp && fflush(csv_fp) == 0 && !ferror(csv_fp));
    if (four_case && cf.rank == 0 && csv_ready) {
        int path_len = snprintf(samples_path, sizeof(samples_path), "%s.samples.csv", csv_path);
        if (path_len > 0 && (size_t)path_len < sizeof(samples_path)) {
            samples_fp = fopen(samples_path, "w");
            if (samples_fp) {
                struct stat samples_st;
                samples_regular = fstat(fileno(samples_fp), &samples_st) == 0 && S_ISREG(samples_st.st_mode);
                fputs("profile,seed,phase,impairment_scope,iteration,worst_rank_ns\n", samples_fp);
            } else {
                perror(samples_path);
            }
        }
        csv_ready = csv_regular && samples_regular && fflush(samples_fp) == 0 && !ferror(samples_fp);
        if (!csv_ready)
            fprintf(stderr, "ERROR: %s summary and samples require writable regular CSV files\n", model);
    }
    bool all_csv_ready = false;
    if (!ctrl_all_ok(&cf, csv_ready, &all_csv_ready) || !all_csv_ready)
        goto cleanup;

    if (check_rails && local_rank == 0) {
        printf("Node %d HCA inventory %s\n", node_rank, full_hca_list);
        fflush(stdout);
    }
    if (cf.rank == 0) {
        const char *proto = getenv("NCCL_PROTO");
        printf("NCCL_PROTO=%s\n", proto && *proto ? proto : "auto");
        if (four_case)
            printf("%s four-case seed=%" PRIu64 ", NCCL Ring, one channel, 512 KiB, two ranks\n", model, seed);
        printf(
            "====================================================================================================================="
            "\n");
        if (use_ab_fmt) {
            printf("vxfec Head-to-Head GPU Collective Benchmark (`%s` via `%s`, %d Ranks, N=%d)\n", api.backend_name,
                   native_ib ? "Stock NCCL_IB RC" : "libnccl-vxfec.so", cf.nranks, ab_iters);
            printf(
                "================================================================================================================="
                "====\n");
            printf(
                "  Phase / Network Condition          | Tensor  |  p50 CCT |  p99 CCT | Alg / BusBW   | Drops | SACK | HARQ-IR | "
                "RS8/RQ\n");
            printf(
                "-------------------------------------+---------+----------+----------+---------------+-------+------+---------+--"
                "------\n");
        } else {
            printf("vxfec End-to-End GPU Collective Demo (`%s` -> `libnccl-vxfec.so`, %d %s)\n", api.backend_name, cf.nranks,
                   cf.is_fork ? "Simulated Nodes" : "Physical Nodes");
            printf(
                "================================================================================================================="
                "\n");
            printf("  Phase / Network Condition          | Tensor  | Avg Lat  | Alg / BusBW | Drops | SACK | HARQ-IR | RS8/RQ | "
                   "VK\n");
            printf("-------------------------------------+---------+----------+-------------+-------+------+---------+--------+--"
                   "--\n");
        }
        fflush(stdout);
    }

    enum { RANK_WEIGHT_PERIOD = 32, ELEMENT_PERIOD = 31 };
    /* The worst-case positive integer sum bounds every intermediate sum in any reduction order. */
    _Static_assert(16384u / RANK_WEIGHT_PERIOD * (RANK_WEIGHT_PERIOD * (RANK_WEIGHT_PERIOD + 1u) / 2u) * (DEMO_MAX_PHASES + 1u) +
                           16384u * (ELEMENT_PERIOD - 1u + DEMO_MAX_ITERS + DEMO_MAX_PHASES - 1u) <
                       (1u << 24u),
                   "AllReduce input must sum exactly in FP32 for every supported rank");
    uint32_t full_cycles = (uint32_t)cf.nranks / RANK_WEIGHT_PERIOD;
    uint32_t rem_ranks = (uint32_t)cf.nranks % RANK_WEIGHT_PERIOD;
    uint32_t rank_weight_sum =
        full_cycles * (RANK_WEIGHT_PERIOD * (RANK_WEIGHT_PERIOD + 1u) / 2u) + (rem_ranks * (rem_ranks + 1u)) / 2u;
    uint32_t my_rank_weight = (uint32_t)(cf.rank % RANK_WEIGHT_PERIOD) + 1u;
    vxfec_nccl_paired_fault_stats_t first_loss_trace = {0};
    uint64_t first_loss_drops = 0;
    ok = true;

    for (size_t p = 0; p < n_phases; p++) {
        if (four_case && !native_ib && p > 0u) {
            /* A new connection gives both arms a fresh pacer and loss estimate. */
            set_fault(0u, 1u, false);
            bool destroyed = api.comm_destroy(comm) == 0;
            comm = NULL;
            if (!ctrl_all_ok(&cf, destroyed, &ok) || !ok) {
                ok = false;
                break;
            }
            if (!init_collective(&cf, &api, &comm, false)) {
                ok = false;
                break;
            }
        }
        size_t bytes = phases[p].elems * sizeof(float);
        uint32_t phase_offset = (uint32_t)p;
        uint32_t p_scale = phase_offset + 2u;
        uint32_t rank_base = my_rank_weight * p_scale;
        uint32_t sum_base = rank_weight_sum * p_scale;
        for (size_t i = 0; i < phases[p].elems; i++)
            h_in[i] = (float)(rank_base + (uint32_t)(i % ELEMENT_PERIOD) + (uint32_t)(i & 1u) * phase_offset);
        memset(h_out, 0, bytes);
        bool phase_ready = api.memcpy_dev(d_in, h_in, bytes, GPU_MEMCPY_H2D) == 0 &&
                           api.memcpy_dev(d_out, h_out, bytes, GPU_MEMCPY_H2D) == 0 && api.stream_sync(NULL) == 0;
        if (!ctrl_all_ok(&cf, phase_ready, &ok) || !ok) {
            ok = false;
            break;
        }

        if (!native_ib)
            set_codec(phases[p].codec_env);
        if (!ctrl_barrier(&cf)) {
            ok = false;
            break;
        }
        if (!native_ib) {
            if (!four_case)
                set_fault(phases[p].loss_ppm, phases[p].burst, phases[p].reorder);
            reset_stats();
        }

        int n_warm = use_ab_fmt ? 4 : 1;
        bool warm_ok = true;
        for (int w = 0; w < n_warm; w++) {
            if (four_case && !native_ib) {
                /* High-bit iteration IDs keep warmup and timed masks disjoint. */
                bool armed = set_paired_fault(phases[p].loss_ppm, phases[p].burst, seed, cf.rank, 0x80000000u | (uint32_t)w);
                if (!ctrl_all_ok(&cf, armed, &warm_ok) || !warm_ok) {
                    warm_ok = false;
                    break;
                }
            }
            if (api.all_reduce(d_in, d_out, phases[p].elems, NCCL_DT_FLOAT32, NCCL_OP_SUM, comm, stream) != 0 ||
                api.stream_sync(stream) != 0) {
                warm_ok = false;
                break;
            }
        }
        if (warm_ok)
            warm_ok = api.memcpy_dev(d_out, h_out, bytes, GPU_MEMCPY_H2D) == 0 && api.stream_sync(NULL) == 0;
        if (!ctrl_all_ok(&cf, warm_ok, &ok) || !ok) {
            ok = false;
            break;
        }

        if (!native_ib && p == 0u) {
            vxfec_nccl_stats_t warm_my = {0}, warm_tot = {0};
            if (get_stats_ex)
                get_stats_ex(&warm_my, sizeof(warm_my));
            else
                get_stats(&warm_my);
            bool my_routed = multi_gpu_node ? true : (warm_my.tx_messages > 0u || warm_my.rx_messages > 0u);
            bool all_routed = false;
            if (!ctrl_gather_stats(&cf, &warm_my, &warm_tot) || !ctrl_all_ok(&cf, my_routed, &all_routed) || !all_routed ||
                warm_tot.tx_messages == 0u || warm_tot.rx_messages == 0u) {
                if (cf.rank == 0)
                    fprintf(stderr, "ERROR: NCCL/RCCL did not route AllReduce traffic through vxfec; aborting A/B run\n");
                ok = false;
                break;
            }
        }
        if (!native_ib)
            reset_stats();
        bool counters_ready = !check_rails || !native_ib || local_rank != 0 || snapshot_native_rails(rails, local_ranks, true);
        if (!ctrl_all_ok(&cf, counters_ready, &ok) || !ok) {
            ok = false;
            break;
        }

        uint64_t my_lats[DEMO_MAX_ITERS];
        uint64_t max_lats[DEMO_MAX_ITERS];
        memset(my_lats, 0, sizeof(my_lats));
        memset(max_lats, 0, sizeof(max_lats));

        uint64_t elapsed_ns = 0;
        for (int it = 0; it < phases[p].iters; it++) {
            /* The changing offset catches stale outputs. Odd elements also distinguish phases of the same size. */
            const uint32_t iteration_offset = (uint32_t)it + 1u;
            for (size_t i = 0; i < phases[p].elems; i++)
                h_in[i] =
                    (float)(rank_base + iteration_offset + (uint32_t)(i % ELEMENT_PERIOD) + (uint32_t)(i & 1u) * phase_offset);
            /* Pageable H2D copies can return before DMA finishes. Wait on the copy's default stream. */
            bool local_ok = api.memcpy_dev(d_in, h_in, bytes, GPU_MEMCPY_H2D) == 0 && api.stream_sync(NULL) == 0;
            if (local_ok && four_case && !native_ib)
                local_ok = set_paired_fault(phases[p].loss_ppm, phases[p].burst, seed, cf.rank, iteration_offset);
            if (!ctrl_all_ok(&cf, local_ok, &ok) || !ok) {
                ok = false;
                break;
            }
            uint64_t it_t0 = now_ns();
            local_ok = api.all_reduce(d_in, d_out, phases[p].elems, NCCL_DT_FLOAT32, NCCL_OP_SUM, comm, stream) == 0 &&
                       api.stream_sync(stream) == 0;
            my_lats[it] = now_ns() - it_t0;
            elapsed_ns += my_lats[it];

            local_ok = local_ok && api.memcpy_dev(h_out, d_out, bytes, GPU_MEMCPY_D2H) == 0;
            if (local_ok) {
                for (size_t i = 0; i < phases[p].elems; i++) {
                    uint32_t element_offset = (uint32_t)(i % ELEMENT_PERIOD) + (uint32_t)(i & 1u) * phase_offset;
                    float expected = (float)(sum_base + (uint32_t)cf.nranks * (iteration_offset + element_offset));
                    if (h_out[i] != expected) {
                        fprintf(stderr,
                                "rank %d: GPU FP32 AllReduce mismatch at phase %zu iteration %d idx %zu (got %.9g, exp %.9g)\n",
                                cf.rank, p, it + 1, i, h_out[i], expected);
                        local_ok = false;
                        break;
                    }
                }
            }
            if (!local_ok)
                fprintf(stderr, "rank %d: AllReduce iteration %d in phase '%s' failed\n", cf.rank, it + 1, phases[p].name);
            if (!ctrl_all_ok(&cf, local_ok, &ok) || !ok) {
                ok = false;
                break;
            }
        }
        if (!ok)
            break;

        if (check_rails) {
            bool rails_ok = native_ib ? (local_rank != 0 || snapshot_native_rails(rails, local_ranks, false))
                                      : snapshot_plugin_rails(get_rail_stats, rails, local_ranks, true);
            if (!rails_ok)
                fprintf(stderr, "rank %d: failed to read HCA activity for phase '%s'\n", cf.rank, phases[p].name);
            if (!ctrl_all_ok(&cf, rails_ok, &ok) || !ok) {
                ok = false;
                break;
            }
        }
        if (use_ab_fmt && !ctrl_gather_max_lats(&cf, my_lats, max_lats, phases[p].iters)) {
            ok = false;
            break;
        }

        vxfec_nccl_stats_t my_st = {0}, tot = {0};
        vxfec_nccl_paired_fault_stats_t loss_trace = {0};
        if (!native_ib) {
            if (get_stats_ex)
                get_stats_ex(&my_st, sizeof(my_st));
            else if (get_stats)
                get_stats(&my_st);
            if (!ctrl_gather_stats(&cf, &my_st, &tot)) {
                ok = false;
                break;
            }
            if (require_ibv_gdr) {
                /* NCCL may use only some local ranks to carry cross-node traffic. */
                bool my_routed = multi_gpu_node || (my_st.tx_sys_gpu_direct_pkts > 0u && my_st.rx_sys_gpu_direct_pkts > 0u);
                bool all_routed = false;
                if (!ctrl_all_ok(&cf, my_routed, &all_routed) || !all_routed || tot.tx_sys_gpu_direct_pkts == 0u ||
                    tot.rx_sys_gpu_direct_pkts == 0u || tot.tx_udp_payload_pkts != 0u ||
                    tot.tx_sys_ibv_pkts != tot.tx_sys_gpu_direct_pkts || tot.rx_sys_pkts != tot.rx_sys_gpu_direct_pkts) {
                    if (cf.rank == 0)
                        fprintf(stderr,
                                "ERROR: phase '%s' did not use the required vxfec IB GPUDirect payload path "
                                "(TX GPU/IB=%" PRIu64 "/%" PRIu64 ", RX GPU/total=%" PRIu64 "/%" PRIu64 ", UDP=%" PRIu64 ")\n",
                                phases[p].name, tot.tx_sys_gpu_direct_pkts, tot.tx_sys_ibv_pkts, tot.rx_sys_gpu_direct_pkts,
                                tot.rx_sys_pkts, tot.tx_udp_payload_pkts);
                    ok = false;
                    break;
                }
            }
            if (four_case) {
                get_paired_stats(&loss_trace);
                if (!ctrl_gather_paired_stats(&cf, &loss_trace)) {
                    ok = false;
                    break;
                }
                bool trace_ok = true;
                if (cf.rank == 0) {
                    const bool repair_counters_ok = tot.rx_cuda_repair_completions <= tot.rx_fec_decodes;
                    if (phases[p].loss_ppm == 0u) {
                        trace_ok = !tot.sim_dropped_pkts && !loss_trace.systematic_trials && !loss_trace.drop_fingerprint;
                    } else {
                        trace_ok = tot.sim_dropped_pkts > 0u && loss_trace.systematic_trials >= tot.sim_dropped_pkts &&
                                   loss_trace.drop_fingerprint != 0u;
                        if (p == 1u) {
                            first_loss_trace = loss_trace;
                            first_loss_drops = tot.sim_dropped_pkts;
                        } else {
                            trace_ok = trace_ok && first_loss_drops == tot.sim_dropped_pkts &&
                                       first_loss_trace.systematic_trials == loss_trace.systematic_trials &&
                                       first_loss_trace.drop_fingerprint == loss_trace.drop_fingerprint;
                        }
                    }
                    if (!trace_ok)
                        fprintf(stderr,
                                "ERROR: phase '%s' has missing or mismatched systematic impairment "
                                "(drops=%" PRIu64 ", trials=%" PRIu64 ", fingerprint=%" PRIu64 ")\n",
                                phases[p].name, tot.sim_dropped_pkts, loss_trace.systematic_trials, loss_trace.drop_fingerprint);
                    if (!repair_counters_ok) {
                        fprintf(stderr, "ERROR: phase '%s' has inconsistent CUDA and FEC repair counts\n", phases[p].name);
                        trace_ok = false;
                    }
                    if (phases[p].require_fec &&
                        (tot.rx_cuda_repair_completions == 0u || tot.rx_cuda_repair_completions != tot.rx_fec_decodes)) {
                        fprintf(stderr,
                                "ERROR: phase '%s' requires timed native CUDA RX repair without fallback "
                                "(CUDA=%" PRIu64 ", FEC=%" PRIu64 ")\n",
                                phases[p].name, tot.rx_cuda_repair_completions, tot.rx_fec_decodes);
                        trace_ok = false;
                    }
                    if (phases[p].loss_ppm > 0u && !phases[p].require_fec &&
                        (tot.tx_sack_retx_pkts == 0u || tot.rx_fec_decodes != 0u || tot.rx_cuda_repair_completions != 0u)) {
                        fprintf(stderr, "ERROR: phase '%s' requires timed SACK retransmissions without FEC repair\n",
                                phases[p].name);
                        trace_ok = false;
                    }
                }
                if (!ctrl_all_ok(&cf, trace_ok, &ok) || !ok) {
                    ok = false;
                    break;
                }
            } else if (phases[p].require_fec && (tot.rx_rs8_decodes + tot.rx_rq_decodes) == 0u && tot.rx_fec_decodes == 0u) {
                if (cf.rank == 0) {
                    fprintf(stderr, "ERROR: phase '%s' completed without any FEC decode (drops=%" PRIu64 ", sack=%" PRIu64 ")\n",
                            phases[p].name, tot.sim_dropped_pkts, tot.tx_sack_retx_pkts);
                }
                ok = false;
                break;
            }
        }

        if (cf.rank == 0) {
            double mb = (double)bytes / (1024.0 * 1024.0);
            uint64_t ir = tot.tx_harq_ir_pkts + tot.tx_proactive_rep_pkts;
            if (use_ab_fmt) {
                int nit = phases[p].iters;
                const char *scope = phases[p].loss_ppm > 0u ? "first-systematic" : "none";
                if (samples_fp) {
                    for (int i = 0; i < nit; i++)
                        fprintf(samples_fp, "%s,%" PRIu64 ",%s,%s,%d,%" PRIu64 "\n", four_case_profile, seed, phases[p].name, scope,
                                i + 1, max_lats[i]);
                }
                qsort(max_lats, (size_t)nit, sizeof(uint64_t), cmp_u64_asc);
                uint64_t sum_ns = 0ULL;
                for (int i = 0; i < nit; i++)
                    sum_ns += max_lats[i];
                if (sum_ns == 0ULL)
                    sum_ns = 1ULL;
                double p50_ms = (double)max_lats[nit / 2] / 1e6;
                size_t p99_idx = ((size_t)nit * 99u + 99u) / 100u - 1u;
                double p99_ms = (double)max_lats[p99_idx] / 1e6;
                double avg_ms = ((double)sum_ns / (double)nit) / 1e6;
                double alg_gbps = (8.0 * (double)bytes * (double)nit) / (double)sum_ns;
                double bus_gbps = alg_gbps * (2.0 * (double)(cf.nranks - 1) / (double)cf.nranks);

                printf("  %-34s | %4.1f MB | %5.2f ms | %5.2f ms%s| %6.2f/%-6.2f | %5" PRIu64 " | %4" PRIu64 " | %7" PRIu64
                       " | %3" PRIu64 "/%-2" PRIu64 "\n",
                       phases[p].name, mb, p50_ms, p99_ms, (nit < 100) ? "*" : " ", alg_gbps, bus_gbps, tot.sim_dropped_pkts,
                       tot.tx_sack_retx_pkts, ir, tot.rx_rs8_decodes, tot.rx_rq_decodes);
                if (csv_fp) {
                    fprintf(csv_fp,
                            "\"%s\",%s,%s,%d,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                            ",%" PRIu64 ",%" PRIu64 ",%" PRIu64,
                            phases[p].name, xport_tag, native_ib ? "rc" : phases[p].codec_env, cf.nranks, mb, nit, p50_ms, p99_ms,
                            avg_ms, alg_gbps, bus_gbps, tot.sim_dropped_pkts, tot.tx_sack_retx_pkts, ir, tot.rx_rs8_decodes,
                            tot.rx_rq_decodes, tot.vk_offload_calls, tot.ibv_direct_pkts);
                    if (four_case) {
                        uint64_t fallback = tot.rx_fec_decodes - tot.rx_cuda_repair_completions;
                        fprintf(csv_fp, ",%s,%" PRIu64 ",%s,%" PRIu32 ",%" PRIu32 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64,
                                four_case_profile, seed, scope, phases[p].loss_ppm, phases[p].burst, loss_trace.systematic_trials,
                                loss_trace.drop_fingerprint, tot.rx_cuda_repair_completions, fallback);
                    }
                    fputc('\n', csv_fp);
                    fflush(csv_fp);
                }
            } else {
                if (elapsed_ns == 0ULL)
                    elapsed_ns = 1ULL;
                double avg_ms = ((double)elapsed_ns / (double)phases[p].iters) / 1e6;
                double alg_gbps = (8.0 * (double)bytes * (double)phases[p].iters) / (double)elapsed_ns;
                double bus_gbps = alg_gbps * (2.0 * (double)(cf.nranks - 1) / (double)cf.nranks);

                printf("  %-34s | %4.1f MB | %5.2f ms | %5.2f/%-5.2f | %5" PRIu64 " | %4" PRIu64 " | %7" PRIu64 " | %3" PRIu64
                       "/%-2" PRIu64 " | %2" PRIu64 "\n",
                       phases[p].name, mb, avg_ms, alg_gbps, bus_gbps, tot.sim_dropped_pkts, tot.tx_sack_retx_pkts, ir,
                       tot.rx_rs8_decodes, tot.rx_rq_decodes, tot.vk_offload_calls);
                if (csv_fp) {
                    fprintf(csv_fp,
                            "\"%s\",%s,%d,%.1f,%.3f,%.3f,%.3f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                            ",%" PRIu64 "\n",
                            phases[p].name, phases[p].codec_env, cf.nranks, mb, avg_ms, alg_gbps, bus_gbps, tot.sim_dropped_pkts,
                            tot.tx_sack_retx_pkts, ir, tot.rx_rs8_decodes, tot.rx_rq_decodes, tot.vk_offload_calls,
                            tot.ibv_direct_pkts);
                    fflush(csv_fp);
                }
            }
            fflush(stdout);
        }
    }

    if (ok && check_rails)
        ok = ctrl_check_rail_coverage(&cf, rails, local_ranks, native_ib);

    if (ok) {
        if (!native_ib) {
            set_fault(0, 1, false);
            set_codec(NULL);
        }
        bool output_ok = close_csv(&csv_fp, csv_path, four_case);
        output_ok = close_csv(&samples_fp, samples_path, true) && output_ok;
        bool all_ok = false;
        ok = ctrl_all_ok(&cf, output_ok, &all_ok) && all_ok;
    }

finish_train_step:
cleanup:
    if (csv_fp)
        fclose(csv_fp);
    if (samples_fp)
        fclose(samples_fp);
    if (stream && api.stream_destroy)
        (void)api.stream_destroy(stream);
    if (d_out && api.free_dev)
        (void)api.free_dev(d_out);
    if (d_in && api.free_dev)
        (void)api.free_dev(d_in);
    free(h_out);
    free(h_in);
    if (comm && api.comm_destroy)
        (void)api.comm_destroy(comm);
    if (plug_dl)
        dlclose(plug_dl);
    if (api.ccl_dl)
        dlclose(api.ccl_dl);
    if (api.gpu_dl)
        dlclose(api.gpu_dl);
    close_ctrl(&cf);

    if (cf.is_fork && cf.rank == 0) {
        int st = 0;
        if (waitpid(cf.child_pid, &st, 0) < 0 || !WIFEXITED(st) || WEXITSTATUS(st) != 0)
            ok = false;
    }
    for (int k = 0; k < n_spawned; k++) {
        int st = 0;
        if (waitpid(spawned_pids[k], &st, 0) < 0 || !WIFEXITED(st) || WEXITSTATUS(st) != 0)
            ok = false;
    }
    if (!ok && csv_regular)
        (void)unlink(csv_path);
    if (!ok && samples_regular)
        (void)unlink(samples_path);
    if (cf.rank == 0 && ok && n_phases > 0) {
        printf(
            "====================================================================================================================="
            "\n");
        printf("  Verified 100%% bit-exact GPU FP32 `ncclAllReduce` across all %zu collective phases.\n", n_phases);
        if (four_case)
            printf("  Verified %s four-case timed profile\n", model);
        printf(
            "====================================================================================================================="
            "\n");
        fflush(stdout);
    }
    return ok ? 0 : 1;
}
