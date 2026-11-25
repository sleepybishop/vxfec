/* CPU stand-in used by test_nccl_e2e.py. */
#define _GNU_SOURCE
#include "nccl_net_vxfec.h"
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    char bytes[128];
} unique_id;
static int comm_rank, comm_size, calls, resets, phase;
static int issued_ids, comm_inits;
static bool comm_active;
static bool copy_pending;
static uint32_t paired_loss;
static uint32_t paired_iteration;
static uint64_t paired_seed;
static const char *codec;

int cudaSetDevice(int index)
{
    return index < 0;
}
int cudaMalloc(void **p, size_t n)
{
    *p = malloc(n);
    return !*p;
}
int cudaFree(void *p)
{
    free(p);
    return 0;
}
int cudaMemcpy(void *dst, const void *src, size_t n, int kind)
{
    if (getenv("TEST_COPY_DELAY"))
        usleep(15000);
    copy_pending = kind == 1;
    memcpy(dst, src, n);
    return 0;
}
int cudaStreamCreate(void **p)
{
    *p = (void *)1;
    return 0;
}
int cudaStreamDestroy(void *p)
{
    return 0;
}
int cudaStreamSynchronize(void *p)
{
    if (p == NULL)
        copy_pending = false;
    return 0;
}

extern __typeof__(cudaSetDevice) hipSetDevice __attribute__((alias("cudaSetDevice")));
extern __typeof__(cudaMalloc) hipMalloc __attribute__((alias("cudaMalloc")));
extern __typeof__(cudaFree) hipFree __attribute__((alias("cudaFree")));
extern __typeof__(cudaMemcpy) hipMemcpy __attribute__((alias("cudaMemcpy")));
extern __typeof__(cudaStreamCreate) hipStreamCreate __attribute__((alias("cudaStreamCreate")));
extern __typeof__(cudaStreamDestroy) hipStreamDestroy __attribute__((alias("cudaStreamDestroy")));
extern __typeof__(cudaStreamSynchronize) hipStreamSynchronize __attribute__((alias("cudaStreamSynchronize")));

int ncclGetUniqueId(unique_id *p)
{
    memset(p, 0, sizeof(*p));
    p->bytes[0] = ++issued_ids;
    return 0;
}
int ncclCommInitRank(void **p, int n, unique_id id, int rank)
{
    const char *net = getenv("NCCL_NET");
    if (getenv("TEST_SIMPLE") && strcmp(getenv("NCCL_PROTO") ?: "", "Simple"))
        return 1;
    if (getenv("TEST_FOUR_CASE") &&
        (n != 2 || strcmp(getenv("NCCL_ALGO") ?: "", "Ring") || strcmp(getenv("NCCL_MIN_NCHANNELS") ?: "", "1") ||
         strcmp(getenv("NCCL_MAX_NCHANNELS") ?: "", "1") || strcmp(getenv("NCCL_MIN_CTAS") ?: "", "1") ||
         strcmp(getenv("NCCL_MAX_CTAS") ?: "", "1") || comm_active || id.bytes[0] != comm_inits + 1))
        return 1;
    if (strcmp(net, "IB") && !dlopen(getenv("VXFEC_NCCL_SO_PATH"), RTLD_NOW | RTLD_GLOBAL))
        return 1;
    if (getenv("TEST_HCAS")) {
        const char *hcas = getenv(strcmp(net, "IB") ? "VXFEC_IBV_DEVS" : "NCCL_IB_HCA");
        if (strcmp(hcas ?: "", strcmp(net, "IB") ? "mlx5_b:1,mlx5_a:2" : "=mlx5_b:1,mlx5_a:2"))
            return 1;
        if (strcmp(getenv("NCCL_HOSTID") ?: "", rank / 2 ? "vxfec-host-1" : "vxfec-host-0"))
            return 1;
        if (strcmp(net, "IB") && (strcmp(getenv("VXFEC_VK_DEVICE") ?: "", rank % 2 ? "1" : "0") ||
                                  strcmp(getenv("VXFEC_NCCL_STRICT_PIX_MR") ?: "", "1")))
            return 1;
    }
    comm_rank = rank;
    comm_size = n;
    comm_inits++;
    comm_active = true;
    *p = (void *)1;
    return 0;
}
int ncclCommDestroy(void *p)
{
    comm_active = false;
    return 0;
}
int ncclAllReduce(const float *src, float *dst, size_t n, int dtype, int op, void *comm, void *stream)
{
    if (copy_pending)
        return 1;
    if (getenv("TEST_FOUR_CASE") && n * sizeof(float) != 512u * 1024u)
        return 1;
    if (getenv("TEST_FOUR_CASE") && codec && paired_iteration != (resets % 2 ? 0x80000000u | (uint32_t)calls : (uint32_t)calls + 1))
        return 1;
    phase = (int)(src[1] - src[0]) - 1;
    if (getenv("TEST_FOUR_CASE") && (!comm_active || phase + 1 != comm_inits))
        return 1;
    bool measured = resets % 2 == 0;
    if (measured && comm_rank == 1 && calls == 1 && getenv("TEST_STALE")) {
        fprintf(stderr, "injected stale measured iteration rank %d\n", comm_rank);
        calls++;
        return 0;
    }
    int rank_sum = 0;
    for (int i = 0; i < comm_size; i++)
        rank_sum += i % 32 + 1;
    int my_rank_weight = comm_rank % 32 + 1;
    int adjustment = (rank_sum - comm_size * my_rank_weight) * (phase + 2);
    for (size_t i = 0; i < n; i++)
        dst[i] = comm_size * src[i] + adjustment;
    if (measured && comm_rank == 1 && calls == 0 && getenv("TEST_CORRUPT")) {
        dst[0] = -999.0f;
        fprintf(stderr, "injected corrupt measured iteration rank %d\n", comm_rank);
    }
    calls++;
    return 0;
}
void vxfec_nccl_reset_stats(void)
{
    calls = 0;
    resets++;
}
void vxfec_nccl_set_fault_injection(uint32_t ppm, uint32_t burst, bool reorder)
{
}
void vxfec_nccl_set_codec_override(const char *name)
{
    codec = name;
}
bool vxfec_nccl_set_paired_fault(uint32_t ppm, uint32_t burst, uint64_t seed, uint32_t rank, uint32_t iteration)
{
    if (getenv("TEST_NO_PAIRED") || rank != (uint32_t)comm_rank || !((ppm == 0u && burst == 1u) || (ppm == 20000u && burst == 6u)))
        return false;
    paired_seed = seed;
    paired_loss = ppm;
    paired_iteration = iteration;
    return true;
}
void vxfec_nccl_get_paired_fault_stats(vxfec_nccl_paired_fault_stats_t *out)
{
    *out = (vxfec_nccl_paired_fault_stats_t){0};
    if (paired_loss && !getenv("TEST_NO_DROPS")) {
        out->systematic_trials = calls * 100u;
        out->drop_fingerprint = (paired_seed + comm_rank + 1u) * calls;
        if (comm_rank == 1 && getenv("TEST_BAD_TRACE") && strcmp(codec, "sack"))
            out->drop_fingerprint++;
    }
}
void vxfec_nccl_get_stats_ex(vxfec_nccl_stats_t *out, size_t len)
{
    vxfec_nccl_stats_t st = {0};
    int active_calls = getenv("TEST_HCAS") && (comm_rank % 2) && phase < 2 ? 0 : calls;
    st.tx_messages = st.rx_messages = st.rx_sys_pkts = st.tx_sys_pkts = active_calls;
    st.rx_fec_decodes = st.rx_rs8_decodes = active_calls;
    st.tx_sys_ibv_pkts = st.tx_sys_gpu_direct_pkts = st.rx_sys_gpu_direct_pkts = active_calls;
    if (getenv("TEST_FOUR_CASE")) {
        st.ibv_direct_pkts = active_calls;
        st.rx_fec_decodes = st.rx_rs8_decodes = 0;
        if (paired_loss && !getenv("TEST_NO_DROPS"))
            st.sim_dropped_pkts = calls * 2u;
        if (!paired_loss && getenv("TEST_DIRTY_CLEAN"))
            st.sim_dropped_pkts = 1u;
        if (paired_loss && strcmp(codec, "sack") == 0 && !getenv("TEST_NO_SACK"))
            st.tx_sack_retx_pkts = calls;
        if (paired_loss && strcmp(codec, "auto") == 0) {
            bool vulkan_only = getenv("TEST_VULKAN_ONLY") != NULL;
            bool fallback_only = vulkan_only || getenv("TEST_CPU_ONLY");
            st.rx_fec_decodes = st.rx_rs8_decodes = calls;
            st.vk_offload_calls = vulkan_only ? calls : 0u;
            if (!fallback_only && (!getenv("TEST_WARM_CUDA_ONLY") || resets % 2 != 0))
                st.rx_cuda_repair_completions = calls;
            if (getenv("TEST_MIXED_FALLBACK") && comm_rank == 1)
                st.rx_cuda_repair_completions = 0;
        }
    }
    memcpy(out, &st, len < sizeof st ? len : sizeof st);
}

size_t vxfec_nccl_get_rail_stats(vxfec_nccl_rail_stats_t *out, size_t cap)
{
    if (cap < 2)
        return 2;
    const char *first = getenv("TEST_WRONG_ORDER") && comm_rank == 1 ? "vxfec-ibv-mlx5_b:1" : "vxfec-ibv-mlx5_a:2";
    const char *second = strcmp(first, "vxfec-ibv-mlx5_b:1") ? "vxfec-ibv-mlx5_b:1" : "vxfec-ibv-mlx5_a:2";
    memset(out, 0, 2 * sizeof(*out));
    strcpy(out[0].name, first);
    strcpy(out[1].name, second);
    if (comm_rank % 2 == 0)
        out[0].tx_sys_gpu_direct_pkts = out[0].rx_sys_gpu_direct_pkts = calls;
    if (comm_rank % 2 == 1 && phase >= 2 && !getenv("TEST_MISSING_RAIL"))
        out[1].tx_sys_gpu_direct_pkts = out[1].rx_sys_gpu_direct_pkts = calls;
    return 2;
}
FILE *fopen(const char *path, const char *mode)
{
    if (getenv("TEST_NATIVE_COUNTERS") && !strncmp(path, "/sys/class/infiniband/", 22)) {
        static char data[64];
        int value = ((strstr(path, "mlx5_b") && getenv("TEST_MISSING_RAIL")) ? 0 : calls * 100000);
        snprintf(data, sizeof(data), "%d", value);
        return fmemopen(data, sizeof(data), "r");
    }
    FILE *(*next)(const char *, const char *) = dlsym(RTLD_NEXT, "fopen");
    return next(path, mode);
}
FILE *fopen64(const char *path, const char *mode)
{
    if (getenv("TEST_NATIVE_COUNTERS") && !strncmp(path, "/sys/class/infiniband/", 22))
        return fopen(path, mode);
    FILE *(*next)(const char *, const char *) = dlsym(RTLD_NEXT, "fopen64");
    return next(path, mode);
}
