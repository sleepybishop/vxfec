#define _GNU_SOURCE
#include "../plugins/nccl/nccl_net_vxfec.h"
#include "../t/bench_common.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/*
 * end-to-end multi-node GPU AllReduce demo over libnccl-vxfec.so:
 *   - zero-arg mode: forks 2 simulated nodes on localhost (`make demo-nccl`)
 *   - multi-host mode: `--rank <r> --nranks <N> --master <ip:port> [--ifname <nic>] [--gdr] [--backend udp|ibv] [--max-mb <mb>]
 * [--csv <file>]` dynamically loads ROCm RCCL or NVIDIA NCCL and verifies bit-exact FP32 ncclAllReduce.
 */

typedef struct {
    char internal[128];
} nccl_uid_t;

typedef void *nccl_comm_t;
typedef void *gpu_stream_t;

enum { NCCL_DT_FLOAT32 = 7, NCCL_OP_SUM = 0 };
enum { GPU_MEMCPY_H2D = 1, GPU_MEMCPY_D2H = 2 };

typedef struct {
    const char *backend_name;
    void *gpu_dl;
    void *ccl_dl;
    int (*set_device)(int);
    int (*malloc_dev)(void **, size_t);
    int (*free_dev)(void *);
    int (*memcpy_dev)(void *, const void *, size_t, int);
    int (*stream_create)(gpu_stream_t *);
    int (*stream_sync)(gpu_stream_t);
    int (*stream_destroy)(gpu_stream_t);
    int (*get_uid)(nccl_uid_t *);
    int (*comm_init)(nccl_comm_t *, int, nccl_uid_t, int);
    int (*comm_destroy)(nccl_comm_t);
    int (*all_reduce)(const void *, void *, size_t, int, int, nccl_comm_t, gpu_stream_t);
} gpu_ccl_api_t;

static uint64_t now_ns(void)
{
    return vxfec_bench_now_ns();
}

static bool write_full(int fd, const void *buf, size_t len)
{
    return vxfec_bench_send_all(fd, buf, len);
}

static bool read_full(int fd, void *buf, size_t len)
{
    return vxfec_bench_recv_all(fd, buf, len);
}

static bool load_gpu_ccl(gpu_ccl_api_t *api)
{
    memset(api, 0, sizeof(*api));

    const char *hip_libs[] = {"libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so", NULL};
    for (int i = 0; hip_libs[i] && !api->gpu_dl; i++)
        api->gpu_dl = dlopen(hip_libs[i], RTLD_NOW | RTLD_GLOBAL);
    if (api->gpu_dl) {
        api->ccl_dl = dlopen("librccl.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!api->ccl_dl)
            api->ccl_dl = dlopen("librccl.so", RTLD_NOW | RTLD_GLOBAL);
        if (api->ccl_dl) {
            api->backend_name = "ROCm HIP + RCCL";
            api->set_device = dlsym(api->gpu_dl, "hipSetDevice");
            api->malloc_dev = dlsym(api->gpu_dl, "hipMalloc");
            api->free_dev = dlsym(api->gpu_dl, "hipFree");
            api->memcpy_dev = dlsym(api->gpu_dl, "hipMemcpy");
            api->stream_create = dlsym(api->gpu_dl, "hipStreamCreate");
            api->stream_sync = dlsym(api->gpu_dl, "hipStreamSynchronize");
            api->stream_destroy = dlsym(api->gpu_dl, "hipStreamDestroy");
        } else {
            dlclose(api->gpu_dl);
            api->gpu_dl = NULL;
        }
    }

    if (!api->ccl_dl) {
        const char *cuda_libs[] = {"libcudart.so.12", "libcudart.so.11.0", "libcudart.so", NULL};
        for (int i = 0; cuda_libs[i] && !api->gpu_dl; i++)
            api->gpu_dl = dlopen(cuda_libs[i], RTLD_NOW | RTLD_GLOBAL);
        if (api->gpu_dl) {
            api->ccl_dl = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
            if (!api->ccl_dl)
                api->ccl_dl = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
            if (api->ccl_dl) {
                api->backend_name = "CUDA + NCCL";
                api->set_device = dlsym(api->gpu_dl, "cudaSetDevice");
                api->malloc_dev = dlsym(api->gpu_dl, "cudaMalloc");
                api->free_dev = dlsym(api->gpu_dl, "cudaFree");
                api->memcpy_dev = dlsym(api->gpu_dl, "cudaMemcpy");
                api->stream_create = dlsym(api->gpu_dl, "cudaStreamCreate");
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

    return api->set_device && api->malloc_dev && api->free_dev && api->memcpy_dev && api->stream_create && api->stream_sync &&
           api->stream_destroy && api->get_uid && api->comm_init && api->comm_destroy && api->all_reduce;
}

typedef void (*get_stats_fn)(vxfec_nccl_stats_t *);
typedef void (*get_stats_ex_fn)(vxfec_nccl_stats_t *, size_t);
typedef void (*reset_stats_fn)(void);
typedef void (*set_fault_fn)(uint32_t, uint32_t, bool);
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

static bool ctrl_bcast_uid(ctrl_fabric_t *cf, nccl_uid_t *uid)
{
    if (cf->is_fork) {
        if (cf->rank == 0)
            return write_full(cf->tx_fd, uid, sizeof(*uid));
        return read_full(cf->rx_fd, uid, sizeof(*uid));
    }
    if (cf->rank == 0) {
        for (int r = 1; r < cf->nranks; r++) {
            if (!write_full(cf->peer_fds[r], uid, sizeof(*uid)))
                return false;
        }
        return true;
    }
    return read_full(cf->master_fd, uid, sizeof(*uid));
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

static bool ctrl_gather_stats(ctrl_fabric_t *cf, const vxfec_nccl_stats_t *my_st, vxfec_nccl_stats_t *sum_out)
{
    if (cf->is_fork) {
        if (cf->rank == 1)
            return write_full(cf->tx_fd, my_st, sizeof(*my_st)) && read_full(cf->rx_fd, sum_out, sizeof(*sum_out));
        vxfec_nccl_stats_t peer = {0};
        if (!read_full(cf->rx_fd, &peer, sizeof(peer)))
            return false;
        *sum_out = *my_st;
        sum_out->tx_messages += peer.tx_messages;
        sum_out->rx_messages += peer.rx_messages;
        sum_out->sim_dropped_pkts += peer.sim_dropped_pkts;
        sum_out->tx_sack_retx_pkts += peer.tx_sack_retx_pkts;
        sum_out->tx_harq_ir_pkts += peer.tx_harq_ir_pkts;
        sum_out->tx_proactive_rep_pkts += peer.tx_proactive_rep_pkts;
        sum_out->rx_fec_decodes += peer.rx_fec_decodes;
        sum_out->rx_rs8_decodes += peer.rx_rs8_decodes;
        sum_out->rx_rq_decodes += peer.rx_rq_decodes;
        sum_out->vk_offload_calls += peer.vk_offload_calls;
        sum_out->ibv_direct_pkts += peer.ibv_direct_pkts;
        return write_full(cf->tx_fd, sum_out, sizeof(*sum_out));
    }
    if (cf->rank > 0)
        return write_full(cf->master_fd, my_st, sizeof(*my_st)) && read_full(cf->master_fd, sum_out, sizeof(*sum_out));

    *sum_out = *my_st;
    for (int r = 1; r < cf->nranks; r++) {
        vxfec_nccl_stats_t peer = {0};
        if (!read_full(cf->peer_fds[r], &peer, sizeof(peer)))
            return false;
        sum_out->tx_messages += peer.tx_messages;
        sum_out->rx_messages += peer.rx_messages;
        sum_out->sim_dropped_pkts += peer.sim_dropped_pkts;
        sum_out->tx_sack_retx_pkts += peer.tx_sack_retx_pkts;
        sum_out->tx_harq_ir_pkts += peer.tx_harq_ir_pkts;
        sum_out->tx_proactive_rep_pkts += peer.tx_proactive_rep_pkts;
        sum_out->rx_fec_decodes += peer.rx_fec_decodes;
        sum_out->rx_rs8_decodes += peer.rx_rs8_decodes;
        sum_out->rx_rq_decodes += peer.rx_rq_decodes;
        sum_out->vk_offload_calls += peer.vk_offload_calls;
        sum_out->ibv_direct_pkts += peer.ibv_direct_pkts;
    }
    for (int r = 1; r < cf->nranks; r++) {
        if (!write_full(cf->peer_fds[r], sum_out, sizeof(*sum_out)))
            return false;
    }
    return true;
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
        if (bind(lfd, (struct sockaddr *)&sin, sizeof(sin)) != 0 || listen(lfd, cf->nranks) != 0) {
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
                return false;
            }
            (void)setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            int peer_rank = -1;
            if (!read_full(cfd, &peer_rank, sizeof(peer_rank)) || peer_rank <= 0 || peer_rank >= cf->nranks ||
                cf->peer_fds[peer_rank] != 0) {
                close(cfd);
                close(lfd);
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
    int fd = vxfec_bench_tcp_connect_retry(peer_ip, port, 150, 100000000L);
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

int main(int argc, char **argv)
{
    int arg_rank = -1;
    int arg_nranks = 2;
    int gpu_dev = 0;
    size_t max_mb = 4;
    bool enable_gdr = false;
    const char *master_spec = NULL;
    const char *ifname = NULL;
    const char *backend = NULL;
    const char *csv_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc)
            arg_rank = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nranks") == 0 && i + 1 < argc)
            arg_nranks = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc)
            gpu_dev = atoi(argv[++i]);
        else if (strcmp(argv[i], "--master") == 0 && i + 1 < argc)
            master_spec = argv[++i];
        else if (strcmp(argv[i], "--ifname") == 0 && i + 1 < argc)
            ifname = argv[++i];
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc)
            backend = argv[++i];
        else if (strcmp(argv[i], "--max-mb") == 0 && i + 1 < argc)
            max_mb = (size_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
            csv_path = argv[++i];
        else if (strcmp(argv[i], "--gdr") == 0)
            enable_gdr = true;
    }

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
    if (access(plugin_path, R_OK) != 0) {
        fprintf(stderr, "Missing %s (run `make` first)\n", plugin_path);
        return 1;
    }

    {
        gpu_ccl_api_t probe;
        if (!load_gpu_ccl(&probe)) {
            printf("SKIP: Neither ROCm RCCL (`librccl.so.1`) nor NVIDIA NCCL (`libnccl.so.2`) is installed.\n");
            printf("      Run `./bench_nccl` for the standalone plugin benchmark.\n");
            return 0;
        }
        dlclose(probe.ccl_dl);
        dlclose(probe.gpu_dl);
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
            return 1;
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

    char host_id[64];
    snprintf(host_id, sizeof(host_id), "vxfec-node-%d", cf.rank);
    setenv("NCCL_NET_PLUGIN", plugin_path, 1);
    setenv("NCCL_NET", "vxfec-ecmp-harq", 1);
    setenv("NCCL_HOSTID", host_id, 1);
    setenv("NCCL_P2P_DISABLE", "1", 1);
    setenv("NCCL_SHM_DISABLE", "1", 1);
    if (enable_gdr) {
        setenv("NCCL_GDR_LEVEL", "5", 1);
        setenv("NCCL_NET_GDR_LEVEL", "5", 1);
        setenv("NCCL_DMABUF_ENABLE", "1", 1);
    } else {
        setenv("NCCL_GDR_LEVEL", "0", 1);
    }
    if (ifname && *ifname) {
        setenv("NCCL_SOCKET_IFNAME", ifname, 0);
        setenv("VXFEC_NCCL_IFNAMES", ifname, 1);
    } else if (cf.is_fork) {
        setenv("NCCL_SOCKET_IFNAME", "lo", 1);
    }
    if (backend && *backend)
        setenv("VXFEC_NCCL_BACKEND", backend, 1);
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
    bool ok = false;

    if (!load_gpu_ccl(&api) || api.set_device(gpu_dev) != 0) {
        fprintf(stderr, "rank %d: failed to initialize GPU/CCL runtime on gpu %d\n", cf.rank, gpu_dev);
        goto cleanup;
    }

    nccl_uid_t uid;
    memset(&uid, 0, sizeof(uid));
    if (cf.rank == 0 && api.get_uid(&uid) != 0)
        goto cleanup;
    if (!ctrl_bcast_uid(&cf, &uid))
        goto cleanup;

    if (api.comm_init(&comm, cf.nranks, uid, cf.rank) != 0) {
        fprintf(stderr, "rank %d: ncclCommInitRank failed\n", cf.rank);
        goto cleanup;
    }

    plug_dl = dlopen(plugin_path, RTLD_NOLOAD | RTLD_NOW);
    get_stats_ex_fn get_stats_ex = plug_dl ? (get_stats_ex_fn)dlsym(plug_dl, "vxfec_nccl_get_stats_ex") : NULL;
    get_stats_fn get_stats = plug_dl ? (get_stats_fn)dlsym(plug_dl, "vxfec_nccl_get_stats") : NULL;
    reset_stats_fn reset_stats = plug_dl ? (reset_stats_fn)dlsym(plug_dl, "vxfec_nccl_reset_stats") : NULL;
    set_fault_fn set_fault = plug_dl ? (set_fault_fn)dlsym(plug_dl, "vxfec_nccl_set_fault_injection") : NULL;
    set_codec_fn set_codec = plug_dl ? (set_codec_fn)dlsym(plug_dl, "vxfec_nccl_set_codec_override") : NULL;
    if (!plug_dl || (!get_stats_ex && !get_stats) || !reset_stats || !set_fault || !set_codec) {
        fprintf(stderr, "rank %d: vxfec plugin (%s) was not selected by NCCL/RCCL or lacks fault/codec hooks\n", cf.rank,
                plugin_path);
        goto cleanup;
    }

    if (max_mb < 4)
        max_mb = 4;
    const size_t large_elems = (max_mb * 1024ULL * 1024ULL) / sizeof(float);
    const size_t max_bytes = large_elems * sizeof(float);
    h_in = (float *)malloc(max_bytes);
    h_out = (float *)calloc(large_elems, sizeof(float));

    if (!h_in || !h_out || api.malloc_dev(&d_in, max_bytes) != 0 || api.malloc_dev(&d_out, max_bytes) != 0 ||
        api.stream_create(&stream) != 0) {
        fprintf(stderr, "rank %d: GPU buffer allocation failed\n", cf.rank);
        goto cleanup;
    }

    if (cf.rank == 0 && csv_path && *csv_path) {
        csv_fp = fopen(csv_path, "w");
        if (csv_fp)
            fprintf(csv_fp,
                    "phase,codec,nranks,tensor_mb,avg_ms,alg_gbps,bus_gbps,drops,sack,harq_ir,rs8_dec,rq_dec,vk_calls,ibv_pkts\n");
    }

    if (cf.rank == 0) {
        printf(
            "=================================================================================================================\n");
        printf("vxfec End-to-End GPU Collective Demo (`%s` -> `libnccl-vxfec.so`, %d %s)\n", api.backend_name, cf.nranks,
               cf.is_fork ? "Simulated Nodes" : "Physical Nodes");
        printf(
            "=================================================================================================================\n");
        printf("  Phase / Network Condition          | Tensor  | Avg Lat  | Alg / BusBW | Drops | SACK | HARQ-IR | RS8/RQ | VK\n");
        printf("-------------------------------------+---------+----------+-------------+-------+------+---------+--------+----\n");
    }

    char lbl_gbn_lg[64], lbl_sack_lg[64], lbl_vxf_lg[64];
    snprintf(lbl_gbn_lg, sizeof(lbl_gbn_lg), "8. 3.0%% Burst=6 %zuMB (Baseline GBN)", max_mb);
    snprintf(lbl_sack_lg, sizeof(lbl_sack_lg), "9. 3.0%% Burst=6 %zuMB (SACK)", max_mb);
    snprintf(lbl_vxf_lg, sizeof(lbl_vxf_lg), "10. 3.0%% Burst=6 %zuMB (vxfec)", max_mb);

    const struct {
        const char *name;
        const char *codec_env;
        size_t elems;
        uint32_t loss_ppm;
        uint32_t burst;
        bool reorder;
        bool require_fec;
        int iters;
    } phases[] = {
        {"1. Lossless Fabric (Baseline GBN)", "gbn", 256 * 1024, 0, 1, false, false, 12},
        {"2. Lossless Fabric (vxfec)", "auto", 256 * 1024, 0, 1, false, false, 12},
        {"3. 0.4% OOO + Drops (Baseline GBN)", "gbn", 256 * 1024, 4000, 1, true, false, 10},
        {"4. 0.4% OOO + Drops (vxfec)", "auto", 256 * 1024, 4000, 1, true, false, 10},
        {"5. 3.0% Burst=6 (Baseline GBN)", "gbn", 256 * 1024, 30000, 6, true, false, 8},
        {"6. 3.0% Burst=6 (SACK)", "sack", 256 * 1024, 30000, 6, true, false, 8},
        {"7. 3.0% Burst=6 (vxfec MDS)", "auto", 256 * 1024, 30000, 6, true, true, 8},
        {lbl_gbn_lg, "gbn", large_elems, 30000, 6, true, false, 6},
        {lbl_sack_lg, "sack", large_elems, 30000, 6, true, false, 6},
        {lbl_vxf_lg, "auto", large_elems, 30000, 6, true, true, 6},
    };

    /* periodic integer rank weights (1..64) keep every input and partial sum < 2^24 through 16,384 ranks */
    uint64_t full_cycles = (uint64_t)cf.nranks / 64ULL;
    uint64_t rem_ranks = (uint64_t)cf.nranks % 64ULL;
    uint64_t rank_weight_sum = full_cycles * 2080ULL + (rem_ranks * (rem_ranks + 1ULL)) / 2ULL;
    uint32_t my_rank_weight = (uint32_t)(cf.rank % 64) + 1u;
    ok = true;
    for (size_t p = 0; p < sizeof(phases) / sizeof(phases[0]); p++) {
        size_t bytes = phases[p].elems * sizeof(float);
        uint32_t p_scale = (uint32_t)p + 2u;
        for (size_t i = 0; i < phases[p].elems; i++)
            h_in[i] = (float)(my_rank_weight * p_scale + (uint32_t)(i % 31u));
        memset(h_out, 0, bytes);
        if (api.memcpy_dev(d_in, h_in, bytes, GPU_MEMCPY_H2D) != 0 || api.memcpy_dev(d_out, h_out, bytes, GPU_MEMCPY_H2D) != 0) {
            ok = false;
            break;
        }

        set_codec(phases[p].codec_env);
        if (!ctrl_barrier(&cf)) {
            ok = false;
            break;
        }
        set_fault(phases[p].loss_ppm, phases[p].burst, phases[p].reorder);
        reset_stats();

        if (api.all_reduce(d_in, d_out, phases[p].elems, NCCL_DT_FLOAT32, NCCL_OP_SUM, comm, stream) != 0 ||
            api.stream_sync(stream) != 0 || api.memcpy_dev(d_out, h_out, bytes, GPU_MEMCPY_H2D) != 0) {
            ok = false;
            break;
        }
        if (!ctrl_barrier(&cf)) {
            ok = false;
            break;
        }
        if (p == 0u) {
            vxfec_nccl_stats_t warm_my = {0}, warm_tot = {0};
            if (get_stats_ex)
                get_stats_ex(&warm_my, sizeof(warm_my));
            else
                get_stats(&warm_my);
            bool my_routed = (warm_my.tx_messages > 0u || warm_my.rx_messages > 0u);
            bool all_routed = false;
            if (!ctrl_gather_stats(&cf, &warm_my, &warm_tot) || !ctrl_all_ok(&cf, my_routed, &all_routed) || !all_routed ||
                (warm_tot.tx_messages == 0u && warm_tot.rx_messages == 0u)) {
                if (cf.rank == 0)
                    fprintf(stderr, "ERROR: NCCL/RCCL did not route AllReduce traffic through vxfec; aborting A/B run\n");
                ok = false;
                break;
            }
        }
        reset_stats();
        if (!ctrl_barrier(&cf)) {
            ok = false;
            break;
        }

        uint64_t t0 = now_ns();
        bool local_ok = true;
        for (int it = 0; it < phases[p].iters; it++) {
            if (api.all_reduce(d_in, d_out, phases[p].elems, NCCL_DT_FLOAT32, NCCL_OP_SUM, comm, stream) != 0 ||
                api.stream_sync(stream) != 0) {
                local_ok = false;
                break;
            }
        }
        uint64_t elapsed_ns = now_ns() - t0;

        if (local_ok && api.memcpy_dev(h_out, d_out, bytes, GPU_MEMCPY_D2H) != 0)
            local_ok = false;
        if (local_ok) {
            for (size_t i = 0; i < phases[p].elems; i++) {
                float exp = (float)(rank_weight_sum * (uint64_t)p_scale + (uint64_t)cf.nranks * (uint64_t)(i % 31u));
                if (!isfinite(h_out[i]) || h_out[i] != exp) {
                    fprintf(stderr, "rank %d: GPU FP32 AllReduce mismatch at phase %zu idx %zu (got %.9g, exp %.9g)\n", cf.rank, p,
                            i, h_out[i], exp);
                    local_ok = false;
                    break;
                }
            }
        }
        if (!ctrl_all_ok(&cf, local_ok, &ok) || !ok) {
            ok = false;
            break;
        }

        vxfec_nccl_stats_t my_st = {0}, tot = {0};
        if (get_stats_ex)
            get_stats_ex(&my_st, sizeof(my_st));
        else if (get_stats)
            get_stats(&my_st);
        if (!ctrl_gather_stats(&cf, &my_st, &tot)) {
            ok = false;
            break;
        }
        if (phases[p].require_fec && (tot.rx_rs8_decodes + tot.rx_rq_decodes) == 0u && tot.rx_fec_decodes == 0u) {
            if (cf.rank == 0) {
                fprintf(stderr, "ERROR: phase '%s' completed without any FEC decode (drops=%" PRIu64 ", sack=%" PRIu64 ")\n",
                        phases[p].name, tot.sim_dropped_pkts, tot.tx_sack_retx_pkts);
            }
            ok = false;
            break;
        }
        if (cf.rank == 0) {
            if (elapsed_ns == 0ULL)
                elapsed_ns = 1ULL;
            double avg_ms = ((double)elapsed_ns / (double)phases[p].iters) / 1e6;
            double alg_gbps = (8.0 * (double)bytes * (double)phases[p].iters) / (double)elapsed_ns;
            double bus_gbps = alg_gbps * (2.0 * (double)(cf.nranks - 1) / (double)cf.nranks);
            uint64_t ir = tot.tx_harq_ir_pkts + tot.tx_proactive_rep_pkts;
            double mb = (double)bytes / (1024.0 * 1024.0);

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
    }

    if (ok) {
        set_fault(0, 1, false);
        set_codec(NULL);
        (void)ctrl_barrier(&cf);
    }

    if (cf.rank == 0 && ok) {
        printf(
            "=================================================================================================================\n");
        printf("  Verified 100%% bit-exact GPU FP32 `ncclAllReduce` across all lossless, SACK, and burst-FEC phases.\n");
        printf(
            "=================================================================================================================\n");
    }

cleanup:
    if (csv_fp) {
        fclose(csv_fp);
        if (!ok && cf.rank == 0 && csv_path && *csv_path)
            (void)unlink(csv_path);
    }
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
        waitpid(cf.child_pid, &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
            ok = false;
    }
    return ok ? 0 : 1;
}
