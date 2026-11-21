#define _GNU_SOURCE
#include "../plugins/nccl/nccl_net_v8.h"
#include "../plugins/nccl/nccl_net_vxfec.h"
#include "bench_common.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define NUM_RANKS 4
#define INCAST_SENDERS (NUM_RANKS - 1)
#define MAX_ITERS 256

static uint64_t now_ns(void)
{
    return vxfec_bench_now_ns();
}

static const char *resolve_nccl_plugin_path(void)
{
    const char *env_so = getenv("VXFEC_NCCL_SO_PATH");
    return (env_so && *env_so) ? env_so : "./libnccl-vxfec.so";
}

static void silent_logger(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...)
{
    (void)level;
    (void)flags;
    (void)file;
    (void)line;
    (void)fmt;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

static inline uint32_t ring_seed_word(int rank, uint32_t iter_tag, size_t idx)
{
    return (uint32_t)(rank + 1) * 1009u + iter_tag * 53u + (uint32_t)(idx * 17u + 3u);
}

typedef struct {
    void *listen_comm[NUM_RANKS];
    void *send_comm[NUM_RANKS];
    void *recv_comm[NUM_RANKS];
} ring_comms_t;

typedef struct {
    void *sink_listen;
    void *send_comm[INCAST_SENDERS];
    void *recv_comm[INCAST_SENDERS];
} incast_comms_t;

static bool setup_ring(const ncclNet_v8_t *net, ring_comms_t *ring)
{
    memset(ring, 0, sizeof(*ring));
    uint8_t handles[NUM_RANKS][NCCL_NET_HANDLE_MAXSIZE];

    for (int r = 0; r < NUM_RANKS; r++) {
        if (net->listen(0, handles[r], &ring->listen_comm[r]) != ncclSuccess)
            return false;
    }
    for (int r = 0; r < NUM_RANKS; r++) {
        int dst = (r + 1) % NUM_RANKS;
        while (!ring->send_comm[r] || !ring->recv_comm[dst]) {
            if (!ring->send_comm[r] && net->connect(0, handles[dst], &ring->send_comm[r], NULL) != ncclSuccess)
                return false;
            if (!ring->recv_comm[dst] && net->accept(ring->listen_comm[dst], &ring->recv_comm[dst], NULL) != ncclSuccess)
                return false;
        }
    }
    return true;
}

static void teardown_ring(const ncclNet_v8_t *net, ring_comms_t *ring)
{
    for (int r = 0; r < NUM_RANKS; r++) {
        if (ring->send_comm[r])
            net->closeSend(ring->send_comm[r]);
        if (ring->recv_comm[r])
            net->closeRecv(ring->recv_comm[r]);
        if (ring->listen_comm[r])
            net->closeListen(ring->listen_comm[r]);
    }
}

static bool setup_incast(const ncclNet_v8_t *net, incast_comms_t *ic)
{
    memset(ic, 0, sizeof(*ic));
    uint8_t sink_handle[NCCL_NET_HANDLE_MAXSIZE];
    if (net->listen(0, sink_handle, &ic->sink_listen) != ncclSuccess)
        return false;

    for (int s = 0; s < INCAST_SENDERS; s++) {
        while (!ic->send_comm[s] || !ic->recv_comm[s]) {
            if (!ic->send_comm[s] && net->connect(0, sink_handle, &ic->send_comm[s], NULL) != ncclSuccess)
                return false;
            if (!ic->recv_comm[s] && net->accept(ic->sink_listen, &ic->recv_comm[s], NULL) != ncclSuccess)
                return false;
        }
    }
    return true;
}

static void teardown_incast(const ncclNet_v8_t *net, incast_comms_t *ic)
{
    for (int s = 0; s < INCAST_SENDERS; s++) {
        if (ic->send_comm[s])
            net->closeSend(ic->send_comm[s]);
        if (ic->recv_comm[s])
            net->closeRecv(ic->recv_comm[s]);
    }
    if (ic->sink_listen)
        net->closeListen(ic->sink_listen);
}

/* execute one 4-rank ring AllReduce (ReduceScatter + AllGather = 2*(R-1) steps) */
static bool run_ring_allreduce_once(const ncclNet_v8_t *net, ring_comms_t *ring, uint32_t **bufs, uint32_t **scratch, void **mr_buf,
                                    void **mr_scr, size_t chunk_elems)
{
    const size_t chunk_bytes = chunk_elems * sizeof(uint32_t);
    const int steps_per_phase = NUM_RANKS - 1;

    for (int phase = 0; phase < 2; phase++) {
        for (int step = 0; step < steps_per_phase; step++) {
            void *sreq[NUM_RANKS] = {0};
            void *rreq[NUM_RANKS] = {0};
            int sdone[NUM_RANKS] = {0};
            int rdone[NUM_RANKS] = {0};
            int tag = 500 + phase * 16 + step;

            for (int r = 0; r < NUM_RANKS; r++) {
                int send_idx = (phase == 0) ? ((r - step + NUM_RANKS) % NUM_RANKS) : ((r - step + 1 + NUM_RANKS) % NUM_RANKS);
                int recv_idx = (phase == 0) ? ((r - step - 1 + 2 * NUM_RANKS) % NUM_RANKS) : ((r - step + NUM_RANKS) % NUM_RANKS);

                void *sptr = bufs[r] + (size_t)send_idx * chunk_elems;
                void *rptr = (phase == 0) ? (void *)scratch[r] : (void *)(bufs[r] + (size_t)recv_idx * chunk_elems);
                void *rmr = (phase == 0) ? mr_scr[r] : mr_buf[r];
                int rsz = (int)chunk_bytes;

                while (!rreq[r]) {
                    if (net->irecv(ring->recv_comm[r], 1, &rptr, &rsz, &tag, &rmr, &rreq[r]) != ncclSuccess)
                        return false;
                }
                while (!sreq[r]) {
                    if (net->isend(ring->send_comm[r], sptr, (int)chunk_bytes, tag, mr_buf[r], &sreq[r]) != ncclSuccess)
                        return false;
                }
            }

            int completed = 0;
            while (completed < 2 * NUM_RANKS) {
                for (int r = 0; r < NUM_RANKS; r++) {
                    if (!sdone[r]) {
                        int sz = 0;
                        if (net->test(sreq[r], &sdone[r], &sz) != ncclSuccess)
                            return false;
                        if (sdone[r])
                            completed++;
                    }
                    if (!rdone[r]) {
                        int sz = 0;
                        if (net->test(rreq[r], &rdone[r], &sz) != ncclSuccess)
                            return false;
                        if (rdone[r]) {
                            if ((size_t)sz != chunk_bytes)
                                return false;
                            completed++;
                        }
                    }
                }
            }

            if (phase == 0) {
                for (int r = 0; r < NUM_RANKS; r++) {
                    int recv_idx = (r - step - 1 + 2 * NUM_RANKS) % NUM_RANKS;
                    uint32_t *dst = bufs[r] + (size_t)recv_idx * chunk_elems;
                    for (size_t i = 0; i < chunk_elems; i++)
                        dst[i] += scratch[r][i];
                }
            }
        }
    }
    return true;
}

static bool seed_and_expect_ring(uint32_t **bufs, uint32_t *expected, size_t total_elems, uint32_t iter_tag)
{
    for (int r = 0; r < NUM_RANKS; r++) {
        for (size_t i = 0; i < total_elems; i++)
            bufs[r][i] = ring_seed_word(r, iter_tag, i);
    }
    for (size_t i = 0; i < total_elems; i++) {
        uint32_t sum = 0u;
        for (int r = 0; r < NUM_RANKS; r++)
            sum += ring_seed_word(r, iter_tag, i);
        expected[i] = sum;
    }
    return true;
}

static bool verify_ring_outputs(uint32_t **bufs, const uint32_t *expected, size_t total_bytes)
{
    for (int r = 0; r < NUM_RANKS; r++) {
        if (memcmp(bufs[r], expected, total_bytes) != 0)
            return false;
    }
    return true;
}

/* run one 3->1 incast burst into a single sink rank */
static bool run_incast_once(const ncclNet_v8_t *net, incast_comms_t *ic, uint32_t **tx_bufs, uint32_t **rx_bufs, void **tx_mrs,
                            void **rx_mrs, size_t chunk_bytes, int tag_base)
{
    void *sreq[INCAST_SENDERS] = {0};
    void *rreq[INCAST_SENDERS] = {0};
    int sdone[INCAST_SENDERS] = {0};
    int rdone[INCAST_SENDERS] = {0};

    for (int s = 0; s < INCAST_SENDERS; s++) {
        void *rptr = rx_bufs[s];
        int rsz = (int)chunk_bytes;
        int tag = tag_base + s;
        if (net->irecv(ic->recv_comm[s], 1, &rptr, &rsz, &tag, &rx_mrs[s], &rreq[s]) != ncclSuccess ||
            net->isend(ic->send_comm[s], tx_bufs[s], (int)chunk_bytes, tag, tx_mrs[s], &sreq[s]) != ncclSuccess)
            return false;
    }

    int completed = 0;
    while (completed < 2 * INCAST_SENDERS) {
        for (int s = 0; s < INCAST_SENDERS; s++) {
            if (!sdone[s]) {
                int sz = 0;
                if (net->test(sreq[s], &sdone[s], &sz) != ncclSuccess)
                    return false;
                if (sdone[s])
                    completed++;
            }
            if (!rdone[s]) {
                int sz = 0;
                if (net->test(rreq[s], &rdone[s], &sz) != ncclSuccess)
                    return false;
                if (rdone[s]) {
                    if ((size_t)sz != chunk_bytes)
                        return false;
                    completed++;
                }
            }
        }
    }
    return true;
}

typedef void (*get_stats_fn)(vxfec_nccl_stats_t *);
typedef void (*get_stats_ex_fn)(vxfec_nccl_stats_t *, size_t);
typedef void (*reset_stats_fn)(void);
typedef void (*set_fault_fn)(uint32_t, uint32_t, bool);
typedef void (*set_cong_fn)(uint32_t, uint32_t, uint64_t);
typedef void (*set_delay_fn)(uint64_t);
typedef void (*set_codec_fn)(const char *);

static get_stats_ex_fn g_bench_get_stats_ex = NULL;
static get_stats_fn g_bench_get_stats_v1 = NULL;

static void bench_query_stats(vxfec_nccl_stats_t *out)
{
    memset(out, 0, sizeof(*out));
    if (g_bench_get_stats_ex)
        g_bench_get_stats_ex(out, sizeof(*out));
    else if (g_bench_get_stats_v1)
        g_bench_get_stats_v1(out);
}

static void print_stats_row(const char *label, uint64_t *lat_ns, int iters, double alg_bits_per_iter, double bus_factor,
                            double wire_bits_per_iter, get_stats_fn get_stats)
{
    qsort(lat_ns, (size_t)iters, sizeof(uint64_t), cmp_u64);
    uint64_t total_ns = 0;
    for (int i = 0; i < iters; i++)
        total_ns += lat_ns[i];
    if (total_ns == 0ULL)
        total_ns = 1ULL;

    double p50_ms = (double)lat_ns[iters / 2] / 1e6;
    /* nearest-rank p99: ceil(0.99 * N) - 1 so worst sample is included for N <= 100 */
    size_t p99_idx = ((size_t)iters * 99u + 99u) / 100u - 1u;
    double p99_ms = (double)lat_ns[p99_idx] / 1e6;
    double alg_gbps = (alg_bits_per_iter * (double)iters) / (double)total_ns;
    double bus_gbps = alg_gbps * bus_factor;
    double goodput_gbps = (wire_bits_per_iter * (double)iters) / (double)total_ns;

    vxfec_nccl_stats_t st;
    get_stats(&st);
    uint64_t extra_pkts = st.tx_proactive_rep_pkts + st.tx_sack_retx_pkts + st.tx_harq_ir_pkts;
    double wire_tax_pct = st.tx_sys_pkts ? (100.0 * (double)extra_pkts / (double)st.tx_sys_pkts) : 0.0;

    printf("  %-30s | %6.2f ms | %6.2f ms | %5.2f/%-5.2f | %7.2f Gbps | %6.2f%% | %5" PRIu64 " | %4" PRIu64 "/%-4" PRIu64
           " | %5" PRIu64 "\n",
           label, p50_ms, p99_ms, alg_gbps, bus_gbps, goodput_gbps, wire_tax_pct, st.sim_dropped_pkts, st.rx_rs8_decodes,
           st.rx_rq_decodes, st.tx_fec_clamped_pkts);
}

static bool run_scenario(const ncclNet_v8_t *net, get_stats_fn get_stats, reset_stats_fn reset_stats, set_fault_fn set_fault,
                         set_cong_fn set_cong, set_codec_fn set_codec, const char *label, const char *codec_mode,
                         size_t total_bytes, uint32_t loss_ppm, uint32_t burst, bool reorder, uint32_t ecn_ppm, uint32_t hot_mask,
                         int iters)
{
    set_codec(codec_mode);
    set_cong(ecn_ppm, hot_mask, 0);

    ring_comms_t ring;
    if (!setup_ring(net, &ring))
        return false;

    size_t chunk_bytes = total_bytes / NUM_RANKS;
    size_t chunk_elems = chunk_bytes / sizeof(uint32_t);
    size_t total_elems = chunk_elems * NUM_RANKS;

    uint32_t *bufs[NUM_RANKS] = {0};
    uint32_t *scratch[NUM_RANKS] = {0};
    uint32_t *expected = (uint32_t *)malloc(total_bytes);
    void *mr_buf[NUM_RANKS] = {0};
    void *mr_scr[NUM_RANKS] = {0};
    bool ok = (expected != NULL);

    for (int r = 0; r < NUM_RANKS && ok; r++) {
        if (posix_memalign((void **)&bufs[r], 4096, total_bytes ? total_bytes : 4096u) != 0 ||
            posix_memalign((void **)&scratch[r], 4096, chunk_bytes ? chunk_bytes : 4096u) != 0) {
            ok = false;
            break;
        }
        if (net->regMr(ring.send_comm[r], bufs[r], total_bytes, NCCL_PTR_HOST, &mr_buf[r]) != ncclSuccess ||
            net->regMr(ring.recv_comm[r], scratch[r], chunk_bytes, NCCL_PTR_HOST, &mr_scr[r]) != ncclSuccess) {
            ok = false;
            break;
        }
    }

    if (ok) {
        set_fault(loss_ppm, burst, reorder);
        for (int w = 0; w < 3 && ok; w++) {
            seed_and_expect_ring(bufs, expected, total_elems, 1000u + (uint32_t)w);
            ok = run_ring_allreduce_once(net, &ring, bufs, scratch, mr_buf, mr_scr, chunk_elems) &&
                 verify_ring_outputs(bufs, expected, total_bytes);
        }

        reset_stats();
        uint64_t lat_ns[MAX_ITERS];
        if (iters > MAX_ITERS)
            iters = MAX_ITERS;

        for (int it = 0; it < iters && ok; it++) {
            seed_and_expect_ring(bufs, expected, total_elems, (uint32_t)(it + 1));
            uint64_t t0 = now_ns();
            ok = run_ring_allreduce_once(net, &ring, bufs, scratch, mr_buf, mr_scr, chunk_elems);
            lat_ns[it] = now_ns() - t0;
            if (ok && !verify_ring_outputs(bufs, expected, total_bytes)) {
                fprintf(stderr, "FAIL: ring AllReduce verification mismatch in '%s' at iter=%d\n", label, it);
                ok = false;
            }
        }

        if (ok) {
            double alg_bits_per_iter = 8.0 * (double)total_bytes;
            double bus_factor = 2.0 * (double)(NUM_RANKS - 1) / (double)NUM_RANKS;
            double wire_bits_per_iter = 8.0 * 2.0 * (NUM_RANKS - 1) * (double)total_bytes;
            print_stats_row(label, lat_ns, iters, alg_bits_per_iter, bus_factor, wire_bits_per_iter, get_stats);
        }
    }

    for (int r = 0; r < NUM_RANKS; r++) {
        if (mr_scr[r])
            net->deregMr(ring.recv_comm[r], mr_scr[r]);
        if (mr_buf[r])
            net->deregMr(ring.send_comm[r], mr_buf[r]);
        free(scratch[r]);
        free(bufs[r]);
    }
    free(expected);
    teardown_ring(net, &ring);
    set_cong(0, 0, 0);
    return ok;
}

static bool run_incast_scenario(const ncclNet_v8_t *net, get_stats_fn get_stats, reset_stats_fn reset_stats, set_fault_fn set_fault,
                                set_cong_fn set_cong, set_codec_fn set_codec, const char *label, const char *codec_mode,
                                size_t chunk_bytes, uint32_t loss_ppm, uint32_t burst, uint32_t ecn_ppm, uint64_t rate_mbps,
                                int iters)
{
    set_codec(codec_mode);
    set_cong(ecn_ppm, 0x0003u, rate_mbps);

    incast_comms_t ic;
    if (!setup_incast(net, &ic))
        return false;

    uint32_t *tx_bufs[INCAST_SENDERS] = {0};
    uint32_t *rx_bufs[INCAST_SENDERS] = {0};
    void *tx_mrs[INCAST_SENDERS] = {0};
    void *rx_mrs[INCAST_SENDERS] = {0};
    size_t alloc_sz = chunk_bytes ? chunk_bytes : 4096u;
    size_t chunk_elems = chunk_bytes / sizeof(uint32_t);
    bool ok = true;

    for (int s = 0; s < INCAST_SENDERS; s++) {
        if (posix_memalign((void **)&tx_bufs[s], 4096, alloc_sz) != 0 ||
            posix_memalign((void **)&rx_bufs[s], 4096, alloc_sz) != 0) {
            ok = false;
            break;
        }
        if (net->regMr(ic.send_comm[s], tx_bufs[s], chunk_bytes, NCCL_PTR_HOST, &tx_mrs[s]) != ncclSuccess ||
            net->regMr(ic.recv_comm[s], rx_bufs[s], chunk_bytes, NCCL_PTR_HOST, &rx_mrs[s]) != ncclSuccess) {
            ok = false;
            break;
        }
    }

    if (ok) {
        set_fault(loss_ppm, burst, true);
        for (int w = 0; w < 3 && ok; w++) {
            for (int s = 0; s < INCAST_SENDERS; s++) {
                for (size_t i = 0; i < chunk_elems; i++)
                    tx_bufs[s][i] = ring_seed_word(s, 2000u + (uint32_t)w, i);
                memset(rx_bufs[s], 0, chunk_bytes);
            }
            ok = run_incast_once(net, &ic, tx_bufs, rx_bufs, tx_mrs, rx_mrs, chunk_bytes, 700 + w * 8);
            for (int s = 0; s < INCAST_SENDERS && ok; s++) {
                if (memcmp(rx_bufs[s], tx_bufs[s], chunk_bytes) != 0)
                    ok = false;
            }
        }

        reset_stats();
        uint64_t lat_ns[MAX_ITERS];
        if (iters > MAX_ITERS)
            iters = MAX_ITERS;

        for (int it = 0; it < iters && ok; it++) {
            for (int s = 0; s < INCAST_SENDERS; s++) {
                for (size_t i = 0; i < chunk_elems; i++)
                    tx_bufs[s][i] = ring_seed_word(s, (uint32_t)(it + 1), i);
                memset(rx_bufs[s], 0, chunk_bytes);
            }
            uint64_t t0 = now_ns();
            ok = run_incast_once(net, &ic, tx_bufs, rx_bufs, tx_mrs, rx_mrs, chunk_bytes, 800 + it * 8);
            lat_ns[it] = now_ns() - t0;
            for (int s = 0; s < INCAST_SENDERS && ok; s++) {
                if (memcmp(rx_bufs[s], tx_bufs[s], chunk_bytes) != 0) {
                    fprintf(stderr, "FAIL: incast verification mismatch in '%s' sender=%d iter=%d\n", label, s, it);
                    ok = false;
                }
            }
        }

        if (ok) {
            double wire_bits_per_iter = 8.0 * (double)INCAST_SENDERS * (double)chunk_bytes;
            print_stats_row(label, lat_ns, iters, wire_bits_per_iter, 1.0, wire_bits_per_iter, get_stats);
        }
    }

    for (int s = 0; s < INCAST_SENDERS; s++) {
        if (rx_mrs[s])
            net->deregMr(ic.recv_comm[s], rx_mrs[s]);
        if (tx_mrs[s])
            net->deregMr(ic.send_comm[s], tx_mrs[s]);
        free(rx_bufs[s]);
        free(tx_bufs[s]);
    }
    teardown_incast(net, &ic);
    set_cong(0, 0, 0);
    return ok;
}

static bool tcp_send_all(int fd, const void *buf, size_t len)
{
    return vxfec_bench_send_all(fd, buf, len);
}

static bool tcp_recv_all(int fd, void *buf, size_t len)
{
    return vxfec_bench_recv_all(fd, buf, len);
}

static bool tcp_barrier(int fd, uint32_t step)
{
    uint32_t peer_step = 0;
    return tcp_send_all(fd, &step, sizeof(step)) && tcp_recv_all(fd, &peer_step, sizeof(peer_step)) && (peer_step == step);
}

#define MAX_CHANS_2N 32

struct multinode_comms;

typedef struct {
    struct multinode_comms *mc;
    int chan_idx;
    pthread_t thr;
    bool started;
} chan_worker_t;

typedef struct multinode_comms {
    const ncclNet_v8_t *net;
    int ndev;
    int nchan;
    void *listen_comm[MAX_CHANS_2N];
    void *send_comm[MAX_CHANS_2N];
    void *recv_comm[MAX_CHANS_2N];
    chan_worker_t workers[MAX_CHANS_2N];

    /* per-phase lock-free parallel dispatch state */
    int cur_phase;
    int send_half;
    int recv_half;
    size_t seg_off[2];
    size_t seg_len[2];
    uint32_t *buf;
    uint32_t *scratch;
    void **mr_buf;
    void **mr_scr;
    atomic_uint work_epoch;
    atomic_uint done_workers;
    atomic_bool phase_failed;
    atomic_bool stop_workers;
} multinode_comms_t;

static inline void slice_channel_range(size_t seg_elems, int nchan, int c, size_t *off_out, size_t *cnt_out)
{
    size_t base = seg_elems / (size_t)nchan;
    size_t rem = seg_elems % (size_t)nchan;
    size_t idx = (size_t)c;
    *off_out = idx * base + ((idx < rem) ? idx : rem);
    *cnt_out = base + ((idx < rem) ? 1u : 0u);
}

static bool exec_2node_channel_phase(multinode_comms_t *mc, int c)
{
    const ncclNet_v8_t *net = mc->net;
    int nchan = mc->nchan;
    size_t s_off = 0u, s_cnt = 0u, r_off = 0u, r_cnt = 0u;
    slice_channel_range(mc->seg_len[mc->send_half], nchan, c, &s_off, &s_cnt);
    slice_channel_range(mc->seg_len[mc->recv_half], nchan, c, &r_off, &r_cnt);
    size_t s_bytes = s_cnt * sizeof(uint32_t);
    size_t r_bytes = r_cnt * sizeof(uint32_t);
    if (s_bytes == 0u && r_bytes == 0u)
        return true;
    if (s_bytes > (size_t)INT_MAX || r_bytes > (size_t)INT_MAX)
        return false;

    int phase = mc->cur_phase;
    int tag = 900 + phase * 64 + c;
    void *sreq = NULL, *rreq = NULL;

    if (r_bytes > 0u) {
        void *rptr = (phase == 0) ? (void *)(mc->scratch + r_off) : (void *)(mc->buf + mc->seg_off[mc->recv_half] + r_off);
        void *rmr = (phase == 0) ? mc->mr_scr[c] : mc->mr_buf[c];
        int rsz = (int)r_bytes;
        while (!rreq) {
            if (net->irecv(mc->recv_comm[c], 1, &rptr, &rsz, &tag, &rmr, &rreq) != ncclSuccess)
                return false;
        }
    }
    if (s_bytes > 0u) {
        void *sptr = mc->buf + mc->seg_off[mc->send_half] + s_off;
        while (!sreq) {
            if (net->isend(mc->send_comm[c], sptr, (int)s_bytes, tag, mc->mr_buf[c], &sreq) != ncclSuccess)
                return false;
        }
    }

    int sdone = (s_bytes == 0u) ? 1 : 0;
    int rdone = (r_bytes == 0u) ? 1 : 0;
    while (!sdone || !rdone) {
        if (!sdone) {
            int sz = 0;
            if (net->test(sreq, &sdone, &sz) != ncclSuccess)
                return false;
        }
        if (!rdone) {
            int sz = 0;
            if (net->test(rreq, &rdone, &sz) != ncclSuccess)
                return false;
            if (rdone && (size_t)sz != r_bytes)
                return false;
        }
    }
    return true;
}

static void *chan_worker_main(void *arg)
{
    chan_worker_t *w = (chan_worker_t *)arg;
    multinode_comms_t *mc = w->mc;
    uint32_t seen_epoch = 0u;

    for (;;) {
        uint32_t ep = atomic_load_explicit(&mc->work_epoch, memory_order_acquire);
        if (atomic_load_explicit(&mc->stop_workers, memory_order_acquire))
            break;
        if (ep == seen_epoch) {
            sched_yield();
            continue;
        }
        seen_epoch = ep;
        if (!exec_2node_channel_phase(mc, w->chan_idx))
            atomic_store_explicit(&mc->phase_failed, true, memory_order_release);
        atomic_fetch_add_explicit(&mc->done_workers, 1u, memory_order_release);
    }
    return NULL;
}

static int get_bench_2node_channels(int ndev)
{
    int def_ch = (ndev >= 4) ? ndev : 4;
    const char *env_ch = getenv("VXFEC_BENCH_CHANNELS");
    if (env_ch && *env_ch) {
        int v = atoi(env_ch);
        if (v >= 1 && v <= MAX_CHANS_2N)
            def_ch = v;
    }
    if (def_ch < ndev)
        def_ch = ndev;
    if (def_ch > MAX_CHANS_2N)
        def_ch = MAX_CHANS_2N;
    return def_ch;
}

static void teardown_multinode_2rank(const ncclNet_v8_t *net, multinode_comms_t *mc);

static bool setup_multinode_2rank(const ncclNet_v8_t *net, int ctrl_fd, int rank, multinode_comms_t *mc)
{
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < 65536u && rl.rlim_max > rl.rlim_cur) {
        rl.rlim_cur = (rl.rlim_max < 65536u) ? rl.rlim_max : 65536u;
        (void)setrlimit(RLIMIT_NOFILE, &rl);
    }

    memset(mc, 0, sizeof(*mc));
    mc->net = net;
    int ndev = 1;
    if (net->devices(&ndev) != ncclSuccess || ndev < 1)
        return false;
    if (ndev > MAX_CHANS_2N)
        ndev = MAX_CHANS_2N;
    mc->ndev = ndev;

    uint32_t my_nchan = (uint32_t)get_bench_2node_channels(ndev);
    uint32_t peer_nchan = 0u;
    if (rank == 0) {
        if (!tcp_send_all(ctrl_fd, &my_nchan, sizeof(my_nchan)) || !tcp_recv_all(ctrl_fd, &peer_nchan, sizeof(peer_nchan)))
            return false;
    } else {
        if (!tcp_recv_all(ctrl_fd, &peer_nchan, sizeof(peer_nchan)) || !tcp_send_all(ctrl_fd, &my_nchan, sizeof(my_nchan)))
            return false;
    }
    int nchan = (int)((my_nchan < peer_nchan) ? my_nchan : peer_nchan);
    if (nchan < 1)
        nchan = 1;
    if (nchan > MAX_CHANS_2N)
        nchan = MAX_CHANS_2N;
    mc->nchan = nchan;

    uint8_t my_handles[MAX_CHANS_2N][NCCL_NET_HANDLE_MAXSIZE];
    uint8_t peer_handles[MAX_CHANS_2N][NCCL_NET_HANDLE_MAXSIZE];
    memset(my_handles, 0, sizeof(my_handles));
    memset(peer_handles, 0, sizeof(peer_handles));

    for (int c = 0; c < nchan; c++) {
        int dev = c % ndev;
        if (net->listen(dev, my_handles[c], &mc->listen_comm[c]) != ncclSuccess) {
            teardown_multinode_2rank(net, mc);
            return false;
        }
    }

    if (rank == 0) {
        if (!tcp_send_all(ctrl_fd, my_handles, (size_t)nchan * NCCL_NET_HANDLE_MAXSIZE) ||
            !tcp_recv_all(ctrl_fd, peer_handles, (size_t)nchan * NCCL_NET_HANDLE_MAXSIZE)) {
            teardown_multinode_2rank(net, mc);
            return false;
        }
    } else {
        if (!tcp_recv_all(ctrl_fd, peer_handles, (size_t)nchan * NCCL_NET_HANDLE_MAXSIZE) ||
            !tcp_send_all(ctrl_fd, my_handles, (size_t)nchan * NCCL_NET_HANDLE_MAXSIZE)) {
            teardown_multinode_2rank(net, mc);
            return false;
        }
    }

    for (int c = 0; c < nchan; c++) {
        int dev = c % ndev;
        while (!mc->send_comm[c] || !mc->recv_comm[c]) {
            if (!mc->send_comm[c] && net->connect(dev, peer_handles[c], &mc->send_comm[c], NULL) != ncclSuccess) {
                teardown_multinode_2rank(net, mc);
                return false;
            }
            if (!mc->recv_comm[c] && net->accept(mc->listen_comm[c], &mc->recv_comm[c], NULL) != ncclSuccess) {
                teardown_multinode_2rank(net, mc);
                return false;
            }
        }
    }

    atomic_init(&mc->work_epoch, 0u);
    atomic_init(&mc->done_workers, 0u);
    atomic_init(&mc->phase_failed, false);
    atomic_init(&mc->stop_workers, false);

    for (int c = 1; c < nchan; c++) {
        mc->workers[c].mc = mc;
        mc->workers[c].chan_idx = c;
        if (pthread_create(&mc->workers[c].thr, NULL, chan_worker_main, &mc->workers[c]) != 0) {
            teardown_multinode_2rank(net, mc);
            return false;
        }
        mc->workers[c].started = true;
    }
    return true;
}

static void teardown_multinode_2rank(const ncclNet_v8_t *net, multinode_comms_t *mc)
{
    atomic_store_explicit(&mc->stop_workers, true, memory_order_release);
    atomic_fetch_add_explicit(&mc->work_epoch, 1u, memory_order_release);
    for (int c = 1; c < mc->nchan; c++) {
        if (mc->workers[c].started) {
            pthread_join(mc->workers[c].thr, NULL);
            mc->workers[c].started = false;
        }
    }
    for (int c = 0; c < mc->nchan; c++) {
        if (mc->send_comm[c])
            net->closeSend(mc->send_comm[c]);
        if (mc->recv_comm[c])
            net->closeRecv(mc->recv_comm[c]);
        if (mc->listen_comm[c])
            net->closeListen(mc->listen_comm[c]);
    }
}

/* 2-node multi-channel Ring AllReduce (ReduceScatter + AllGather striped across nchan lock-free channels & ndev NICs) */
static bool run_2node_allreduce_once(const ncclNet_v8_t *net, multinode_comms_t *mc, int rank, uint32_t *buf, uint32_t *scratch,
                                     void **mr_buf, void **mr_scr, size_t total_elems)
{
    (void)net;
    if (rank < 0 || rank > 1)
        return false;
    mc->seg_off[0] = 0u;
    mc->seg_len[0] = total_elems / 2u;
    mc->seg_off[1] = mc->seg_len[0];
    mc->seg_len[1] = total_elems - mc->seg_len[0];
    int nchan = mc->nchan;
    mc->buf = buf;
    mc->scratch = scratch;
    mc->mr_buf = mr_buf;
    mc->mr_scr = mr_scr;

    for (int phase = 0; phase < 2; phase++) {
        int send_half = (phase == 0) ? rank : (1 - rank);
        int recv_half = (phase == 0) ? (1 - rank) : rank;
        mc->cur_phase = phase;
        mc->send_half = send_half;
        mc->recv_half = recv_half;

        if (nchan > 1) {
            atomic_store_explicit(&mc->done_workers, 0u, memory_order_relaxed);
            atomic_store_explicit(&mc->phase_failed, false, memory_order_relaxed);
            atomic_fetch_add_explicit(&mc->work_epoch, 1u, memory_order_release);
        }

        bool ok0 = exec_2node_channel_phase(mc, 0);

        if (nchan > 1) {
            uint32_t exp_workers = (uint32_t)(nchan - 1);
            while (atomic_load_explicit(&mc->done_workers, memory_order_acquire) < exp_workers)
                sched_yield();
            if (atomic_load_explicit(&mc->phase_failed, memory_order_acquire))
                return false;
        }
        if (!ok0)
            return false;

        if (phase == 0) {
            uint32_t *dst = buf + mc->seg_off[recv_half];
            size_t recv_elems = mc->seg_len[recv_half];
            for (size_t i = 0; i < recv_elems; i++)
                dst[i] += scratch[i];
        }
    }
    return true;
}

static bool run_2node_scenario(const ncclNet_v8_t *net, int ctrl_fd, int rank, get_stats_fn get_stats, reset_stats_fn reset_stats,
                               set_fault_fn set_fault, set_codec_fn set_codec, const char *label, const char *codec_mode,
                               size_t total_bytes, uint32_t loss_ppm, uint32_t burst, bool reorder, int iters, FILE *csv_fp)
{
    static uint32_t s_step = 100u;
    if ((rank != 0 && rank != 1) || total_bytes < sizeof(uint32_t) || (total_bytes % sizeof(uint32_t)) != 0u) {
        fprintf(stderr, "Invalid 2-node benchmark args (rank=%d, bytes=%zu; bytes must be a non-zero multiple of %zu)\n", rank,
                total_bytes, sizeof(uint32_t));
        return false;
    }
    if (iters < 1)
        iters = 1;

    set_codec(codec_mode);
    set_fault(0, 1, false);
    if (!tcp_barrier(ctrl_fd, ++s_step))
        return false;

    multinode_comms_t mc;
    if (!setup_multinode_2rank(net, ctrl_fd, rank, &mc))
        return false;

    size_t total_elems = total_bytes / sizeof(uint32_t);
    size_t max_seg_elems = total_elems - (total_elems / 2u);
    size_t scratch_bytes = max_seg_elems * sizeof(uint32_t);
    uint32_t *buf = NULL, *scratch = NULL, *expected = (uint32_t *)malloc(total_bytes);
    void *mr_buf[MAX_CHANS_2N] = {0};
    void *mr_scr[MAX_CHANS_2N] = {0};
    ncclNetProperties_v8_t props;
    bool global_mr = (net->getProperties(0, &props) == ncclSuccess && props.regIsGlobal != 0);
    bool ok = (expected != NULL) && (posix_memalign((void **)&buf, 4096, total_bytes) == 0) &&
              (posix_memalign((void **)&scratch, 4096, scratch_bytes) == 0);

    if (ok && global_mr) {
        void *shared_mr_buf = NULL, *shared_mr_scr = NULL;
        if (net->regMr(mc.send_comm[0], buf, total_bytes, NCCL_PTR_HOST, &shared_mr_buf) != ncclSuccess ||
            net->regMr(mc.recv_comm[0], scratch, scratch_bytes, NCCL_PTR_HOST, &shared_mr_scr) != ncclSuccess) {
            if (shared_mr_scr)
                net->deregMr(mc.recv_comm[0], shared_mr_scr);
            if (shared_mr_buf)
                net->deregMr(mc.send_comm[0], shared_mr_buf);
            ok = false;
        } else {
            for (int c = 0; c < mc.nchan; c++) {
                mr_buf[c] = shared_mr_buf;
                mr_scr[c] = shared_mr_scr;
            }
        }
    } else {
        for (int c = 0; c < mc.nchan && ok; c++) {
            if (net->regMr(mc.send_comm[c], buf, total_bytes, NCCL_PTR_HOST, &mr_buf[c]) != ncclSuccess ||
                net->regMr(mc.recv_comm[c], scratch, scratch_bytes, NCCL_PTR_HOST, &mr_scr[c]) != ncclSuccess)
                ok = false;
        }
    }

    if (ok) {
        set_fault(loss_ppm, burst, reorder);
        if (!tcp_barrier(ctrl_fd, ++s_step))
            ok = false;

        int warmups = (loss_ppm == 0u) ? 8 : 3;
        for (int w = 0; w < warmups && ok; w++) {
            for (size_t i = 0; i < total_elems; i++)
                buf[i] = ring_seed_word(rank, 5000u + (uint32_t)w, i);
            ok = run_2node_allreduce_once(net, &mc, rank, buf, scratch, mr_buf, mr_scr, total_elems);
        }

        if (ok && !tcp_barrier(ctrl_fd, ++s_step))
            ok = false;
        reset_stats();

        uint64_t lat_ns[MAX_ITERS];
        uint64_t peer_lat_ns[MAX_ITERS];
        memset(lat_ns, 0, sizeof(lat_ns));
        memset(peer_lat_ns, 0, sizeof(peer_lat_ns));
        if (iters > MAX_ITERS)
            iters = MAX_ITERS;

        bool valid_ok = true;
        for (int it = 0; it < iters && ok; it++) {
            uint32_t tag = (uint32_t)(it + 1);
            for (size_t i = 0; i < total_elems; i++) {
                buf[i] = ring_seed_word(rank, tag, i);
                expected[i] = ring_seed_word(0, tag, i) + ring_seed_word(1, tag, i);
            }
            if (!tcp_barrier(ctrl_fd, ++s_step)) {
                ok = false;
                break;
            }
            uint64_t t0 = now_ns();
            ok = run_2node_allreduce_once(net, &mc, rank, buf, scratch, mr_buf, mr_scr, total_elems);
            lat_ns[it] = now_ns() - t0;
            if (ok && valid_ok && memcmp(buf, expected, total_bytes) != 0) {
                fprintf(stderr, "FAIL: 2-node AllReduce mismatch in '%s' at iter=%d\n", label, it);
                valid_ok = false;
            }
        }

        vxfec_nccl_stats_t my_st, peer_st;
        get_stats(&my_st);
        memset(&peer_st, 0, sizeof(peer_st));
        uint8_t my_ok = (ok && valid_ok) ? 1u : 0u;
        uint8_t peer_ok = 0u;
        if (ok) {
            if (rank == 0) {
                if (!tcp_recv_all(ctrl_fd, &peer_ok, sizeof(peer_ok)) || !tcp_send_all(ctrl_fd, &my_ok, sizeof(my_ok)) ||
                    !tcp_recv_all(ctrl_fd, &peer_st, sizeof(peer_st)) ||
                    !tcp_recv_all(ctrl_fd, peer_lat_ns, (size_t)iters * sizeof(uint64_t)))
                    ok = false;
            } else {
                if (!tcp_send_all(ctrl_fd, &my_ok, sizeof(my_ok)) || !tcp_recv_all(ctrl_fd, &peer_ok, sizeof(peer_ok)) ||
                    !tcp_send_all(ctrl_fd, &my_st, sizeof(my_st)) ||
                    !tcp_send_all(ctrl_fd, lat_ns, (size_t)iters * sizeof(uint64_t)))
                    ok = false;
            }
        }
        if (!valid_ok || peer_ok == 0u)
            ok = false;

        if (ok && rank == 0) {
            /* collective completion time (CCT) per iteration is the max across participating ranks */
            for (int i = 0; i < iters; i++) {
                if (peer_lat_ns[i] > lat_ns[i])
                    lat_ns[i] = peer_lat_ns[i];
            }
            qsort(lat_ns, (size_t)iters, sizeof(uint64_t), cmp_u64);
            uint64_t total_ns = 0;
            for (int i = 0; i < iters; i++)
                total_ns += lat_ns[i];
            double p50_ms = (double)lat_ns[iters / 2] / 1e6;
            size_t p99_idx = ((size_t)iters * 99u + 99u) / 100u - 1u;
            double p99_ms = (double)lat_ns[p99_idx] / 1e6;
            double max_ms = (double)lat_ns[iters - 1] / 1e6;
            /* NCCL AlgBW = S / t, 2-rank BusBW = AlgBW * 2*(2-1)/2 = AlgBW, BiDir wire goodput = 2 * S / t */
            if (total_ns == 0ULL)
                total_ns = 1ULL;
            double alg_gbps = (8.0 * (double)total_bytes * (double)iters) / (double)total_ns;
            double bus_gbps = alg_gbps;
            double wire_bits = 8.0 * 2.0 * (double)total_bytes;
            double goodput_gbps = (wire_bits * (double)iters) / (double)total_ns;

            uint64_t sys_pkts = my_st.tx_sys_pkts + peer_st.tx_sys_pkts;
            uint64_t extra_pkts = (my_st.tx_proactive_rep_pkts + peer_st.tx_proactive_rep_pkts) +
                                  (my_st.tx_sack_retx_pkts + peer_st.tx_sack_retx_pkts) +
                                  (my_st.tx_harq_ir_pkts + peer_st.tx_harq_ir_pkts);
            double wire_tax = sys_pkts ? (100.0 * (double)extra_pkts / (double)sys_pkts) : 0.0;
            uint64_t drops = my_st.sim_dropped_pkts + peer_st.sim_dropped_pkts;
            uint64_t rs8 = my_st.rx_rs8_decodes + peer_st.rx_rs8_decodes;
            uint64_t rq = my_st.rx_rq_decodes + peer_st.rx_rq_decodes;
            uint64_t ibv = my_st.ibv_direct_pkts + peer_st.ibv_direct_pkts;

            printf("  %-30s | %6.2f ms | %6.2f ms%s | %6.2f Gbps | %7.2f Gbps | %6.2f%% | %5" PRIu64 " | %4" PRIu64 "/%-4" PRIu64
                   " | %5" PRIu64 "\n",
                   label, p50_ms, p99_ms, (iters < 100) ? "*" : " ", bus_gbps, goodput_gbps, wire_tax, drops, rs8, rq, ibv);
            fflush(stdout);

            if (csv_fp) {
                fprintf(csv_fp,
                        "\"%s\",%s,%d,%zu,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.3f,%.3f\n",
                        label, codec_mode, mc.ndev, total_bytes, iters, p50_ms, p99_ms, max_ms, goodput_gbps, wire_tax, drops, rs8,
                        rq, ibv, alg_gbps, bus_gbps);
                fflush(csv_fp);
            }
        }
    }

    if (global_mr) {
        if (mr_scr[0])
            net->deregMr(mc.recv_comm[0], mr_scr[0]);
        if (mr_buf[0])
            net->deregMr(mc.send_comm[0], mr_buf[0]);
    } else {
        for (int c = 0; c < mc.nchan; c++) {
            if (mr_scr[c])
                net->deregMr(mc.recv_comm[c], mr_scr[c]);
            if (mr_buf[c])
                net->deregMr(mc.send_comm[c], mr_buf[c]);
        }
    }
    free(scratch);
    free(buf);
    free(expected);
    teardown_multinode_2rank(net, &mc);
    set_fault(0, 1, false);
    return ok;
}

static int run_multinode_main(const ncclNet_v8_t *net, get_stats_fn get_stats, reset_stats_fn reset_stats, set_fault_fn set_fault,
                              set_codec_fn set_codec, int rank, const char *master_spec, const char *csv_path,
                              const char *single_codec, const char *single_label, size_t single_bytes, uint32_t single_loss_ppm,
                              uint32_t single_burst, int iters)
{
    char host[128];
    uint16_t port = 29600u;
    vxfec_bench_parse_host_port(master_spec, host, sizeof(host), 29600u, &port);

    int ctrl_fd = -1;
    if (rank == 0) {
        struct in_addr any_ip = {.s_addr = htonl(INADDR_ANY)};
        ctrl_fd = vxfec_bench_tcp_listen_accept_one(any_ip, port);
    } else {
        struct in_addr peer_ip;
        if (inet_pton(AF_INET, host, &peer_ip) == 1)
            ctrl_fd = vxfec_bench_tcp_connect_retry(peer_ip, port, 150, 100000000L);
    }
    if (ctrl_fd < 0) {
        perror("ctrl connect/accept");
        return 1;
    }

    int ndev = 1;
    (void)net->devices(&ndev);
    ncclNetProperties_v8_t props;
    (void)net->getProperties(0, &props);

    FILE *csv_fp = NULL;
    if (rank == 0 && csv_path) {
        bool need_hdr = (access(csv_path, F_OK) != 0);
        csv_fp = fopen(csv_path, "a");
        if (csv_fp && need_hdr) {
            fprintf(csv_fp,
                    "scenario,codec,ndev,bytes,iters,p50_ms,p99_ms,max_ms,goodput_gbps,wire_tax_pct,drops,rs8_dec,rq_dec,ibv_pkts,"
                    "alg_gbps,bus_gbps\n");
            fflush(csv_fp);
        }
    }

    if (rank == 0 && !single_codec) {
        printf(
            "====================================================================================================================="
            "\n");
        printf("vxfec 2-Node Physical Fabric Benchmark (`%s`, ndev=%d, nchan=%d, speed=%d Mbps, N=%d; * = max when N<100)\n",
               net->name, ndev, get_bench_2node_channels(ndev), props.speed, iters);
        printf(
            "====================================================================================================================="
            "\n");
        printf(
            "  Scenario (2-Node AllReduce)    |   p50 CCT |  Tail CCT | BusBW (1x) | BiDir Goodput| WireTax | Drops |  RS8/RQ   | "
            "IBV  \n");
        printf(
            "---------------------------------+-----------+-----------+------------+-------------+---------+-------+-----------+--"
            "-----\n");
    }

    bool ok = true;
    if (single_codec) {
        ok = run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec,
                                single_label ? single_label : single_codec, single_codec, single_bytes, single_loss_ppm,
                                single_burst, single_loss_ppm > 0u, iters, csv_fp);
    } else {
        const size_t sz_512k = 512 * 1024;
        const size_t sz_8m = 8 * 1024 * 1024;
        const size_t sz_64m = 64ULL * 1024ULL * 1024ULL;
        int it_lg = (iters >= 20) ? (iters / 2) : iters;
        int it_xl = (iters >= 20) ? 10 : iters;
        ok = run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "512 KiB | 0.0% Lossless (GBN)",
                                "gbn", sz_512k, 0, 1, false, iters, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "512 KiB | 0.0% Lossless (vxfec)",
                                "auto", sz_512k, 0, 1, false, iters, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "8.0 MiB | 0.0% Lossless (GBN)",
                                "gbn", sz_8m, 0, 1, false, it_lg, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "8.0 MiB | 0.0% Lossless (vxfec)",
                                "auto", sz_8m, 0, 1, false, it_lg, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "64 MiB  | 0.0% Lossless (vxfec)",
                                "auto", sz_64m, 0, 1, false, it_xl, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "512 KiB | 0.4% OOO Loss (GBN)",
                                "gbn", sz_512k, 4000, 1, true, iters, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "512 KiB | 0.4% OOO Loss (SACK)",
                                "sack", sz_512k, 4000, 1, true, iters, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "512 KiB | 0.4% OOO Loss (vxfec)",
                                "auto", sz_512k, 4000, 1, true, iters, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "512 KiB | 2.0% Burst=6 (GBN)",
                                "gbn", sz_512k, 20000, 6, true, iters, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "512 KiB | 2.0% Burst=6 (SACK)",
                                "sack", sz_512k, 20000, 6, true, iters, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "512 KiB | 2.0% Burst=6 (MDS)",
                                "rs8", sz_512k, 20000, 6, true, iters, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec,
                                "512 KiB | 2.0% Burst=6 (Rateless)", "rq", sz_512k, 20000, 6, true, iters, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "8.0 MiB | 2.0% Burst=6 (GBN)",
                                "gbn", sz_8m, 20000, 6, true, it_lg, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "8.0 MiB | 2.0% Burst=6 (SACK)",
                                "sack", sz_8m, 20000, 6, true, it_lg, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "8.0 MiB | 2.0% Burst=6 (vxfec)",
                                "auto", sz_8m, 20000, 6, true, it_lg, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "64 MiB  | 2.0% Burst=6 (SACK)",
                                "sack", sz_64m, 20000, 6, true, it_xl, csv_fp) &&
             run_2node_scenario(net, ctrl_fd, rank, get_stats, reset_stats, set_fault, set_codec, "64 MiB  | 2.0% Burst=6 (vxfec)",
                                "auto", sz_64m, 20000, 6, true, it_xl, csv_fp);
    }

    if (rank == 0 && !single_codec)
        printf("===========================================================================================================\n");

    if (csv_fp)
        fclose(csv_fp);
    close(ctrl_fd);
    set_codec(NULL);
    return ok ? 0 : 1;
}

/* keep libvulkan mapped after dlclose so LSan can symbolize ICD suppressions */
static void pin_vulkan_for_lsan(void)
{
#if defined(__SANITIZE_ADDRESS__)
    (void)dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE);
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    (void)dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE);
#endif
#endif
}

static int run_2node_self_test(void)
{
    int sv[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return 1;

    setenv("VXFEC_NCCL_IFNAMES", "lo", 1);
    pid_t pid = fork();
    if (pid < 0) {
        close(sv[0]);
        close(sv[1]);
        return 1;
    }

    int my_rank = (pid == 0) ? 1 : 0;
    int ctrl_fd = (my_rank == 0) ? sv[0] : sv[1];
    close((my_rank == 0) ? sv[1] : sv[0]);

    const char *so_path = resolve_nccl_plugin_path();
    void *dl = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        close(ctrl_fd);
        if (my_rank == 1)
            _exit(1);
        (void)waitpid(pid, NULL, 0);
        return 1;
    }
    pin_vulkan_for_lsan();

    const ncclNet_v8_t *net = (const ncclNet_v8_t *)dlsym(dl, "ncclNetPlugin_v8");
    g_bench_get_stats_ex = (get_stats_ex_fn)dlsym(dl, "vxfec_nccl_get_stats_ex");
    g_bench_get_stats_v1 = (get_stats_fn)dlsym(dl, "vxfec_nccl_get_stats");
    get_stats_fn get_stats = (g_bench_get_stats_ex || g_bench_get_stats_v1) ? bench_query_stats : NULL;
    reset_stats_fn reset_stats = (reset_stats_fn)dlsym(dl, "vxfec_nccl_reset_stats");
    set_fault_fn set_fault = (set_fault_fn)dlsym(dl, "vxfec_nccl_set_fault_injection");
    set_codec_fn set_codec = (set_codec_fn)dlsym(dl, "vxfec_nccl_set_codec_override");

    bool ok = net && get_stats && reset_stats && set_fault && set_codec && (net->init(silent_logger) == ncclSuccess);
    /* verify 1-element, small odd-length, and multi-block odd-length tensors across channels */
    ok = ok &&
         run_2node_scenario(net, ctrl_fd, my_rank, get_stats, reset_stats, set_fault, set_codec, "2N SelfTest 1 elem (4 B)", "auto",
                            4u, 0u, 1u, false, 4, NULL) &&
         run_2node_scenario(net, ctrl_fd, my_rank, get_stats, reset_stats, set_fault, set_codec, "2N SelfTest 3 elems (12 B)",
                            "auto", 12u, 0u, 1u, false, 4, NULL) &&
         run_2node_scenario(net, ctrl_fd, my_rank, get_stats, reset_stats, set_fault, set_codec, "2N SelfTest 5 elems (20 B)",
                            "auto", 20u, 0u, 1u, false, 4, NULL) &&
         run_2node_scenario(net, ctrl_fd, my_rank, get_stats, reset_stats, set_fault, set_codec, "2N SelfTest 131073 elems Loss",
                            "auto", 131073u * sizeof(uint32_t), 20000u, 4u, true, 4, NULL);

    if (set_codec)
        set_codec(NULL);
    close(ctrl_fd);
    dlclose(dl);

    if (my_rank == 1)
        _exit(ok ? 0 : 1);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        ok = false;
    if (ok)
        printf("PASS: 2-rank odd-length AllReduce self-test (1, 3, 5, and 131073 elements)\n");
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    int rank = -1;
    bool self_test = false;
    const char *master_spec = NULL;
    const char *csv_path = NULL;
    const char *single_codec = NULL;
    const char *single_label = NULL;
    size_t single_bytes = 512 * 1024;
    uint32_t single_loss_ppm = 0u;
    uint32_t single_burst = 1u;
    int cli_iters = 50;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-test") == 0)
            self_test = true;
        else if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc)
            rank = atoi(argv[++i]);
        else if (strcmp(argv[i], "--master") == 0 && i + 1 < argc)
            master_spec = argv[++i];
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
            csv_path = argv[++i];
        else if (strcmp(argv[i], "--single-codec") == 0 && i + 1 < argc)
            single_codec = argv[++i];
        else if (strcmp(argv[i], "--label") == 0 && i + 1 < argc)
            single_label = argv[++i];
        else if (strcmp(argv[i], "--bytes") == 0 && i + 1 < argc)
            single_bytes = (size_t)strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--loss-ppm") == 0 && i + 1 < argc)
            single_loss_ppm = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--burst") == 0 && i + 1 < argc)
            single_burst = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            cli_iters = atoi(argv[++i]);
    }

    if (self_test)
        return run_2node_self_test();

    const char *so_path = resolve_nccl_plugin_path();
    void *dl = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", so_path, dlerror());
        return 1;
    }
    pin_vulkan_for_lsan();

    const ncclNet_v8_t *net = (const ncclNet_v8_t *)dlsym(dl, "ncclNetPlugin_v8");
    g_bench_get_stats_ex = (get_stats_ex_fn)dlsym(dl, "vxfec_nccl_get_stats_ex");
    g_bench_get_stats_v1 = (get_stats_fn)dlsym(dl, "vxfec_nccl_get_stats");
    get_stats_fn get_stats = (g_bench_get_stats_ex || g_bench_get_stats_v1) ? bench_query_stats : NULL;
    reset_stats_fn reset_stats = (reset_stats_fn)dlsym(dl, "vxfec_nccl_reset_stats");
    set_fault_fn set_fault = (set_fault_fn)dlsym(dl, "vxfec_nccl_set_fault_injection");
    set_cong_fn set_cong = (set_cong_fn)dlsym(dl, "vxfec_nccl_set_congestion_sim");
    set_delay_fn set_delay = (set_delay_fn)dlsym(dl, "vxfec_nccl_set_link_delay_ns");
    set_codec_fn set_codec = (set_codec_fn)dlsym(dl, "vxfec_nccl_set_codec_override");
    if (!net || !get_stats || !reset_stats || !set_fault || !set_cong || !set_codec || net->init(silent_logger) != ncclSuccess) {
        dlclose(dl);
        return 1;
    }

    if (rank >= 0 && master_spec) {
        int rc = run_multinode_main(net, get_stats, reset_stats, set_fault, set_codec, rank, master_spec, csv_path, single_codec,
                                    single_label, single_bytes, single_loss_ppm, single_burst ? single_burst : 1u, cli_iters);
        dlclose(dl);
        return rc;
    }

    int iters_small = 100;
    int iters_large = 50;
    const char *env_it = getenv("VXFEC_BENCH_ITERS");
    if (env_it && atoi(env_it) >= 10) {
        iters_small = atoi(env_it);
        iters_large = iters_small;
    }

    ncclNetProperties_v8_t props;
    (void)net->getProperties(0, &props);

    printf("======================================================================================================================="
           "\n");
    printf("vxfec NCCL Net v8 Plugin Benchmark (`%s`, Single-Host Loopback, N=%d/%d Samples)\n", net->name, iters_small,
           iters_large);
    printf("======================================================================================================================="
           "\n");
    printf("  Scenario (Topology / Policy)   |   p50 Lat |   p99 Lat | Alg / BusBW | Agg Goodput | WireTax | Drops |  RS8/RQ   | "
           "Clamp\n");
    printf("---------------------------------+-----------+-----------+-------------+-------------+---------+-------+-----------+---"
           "----\n");

    const size_t sz_small = 512 * 1024;             /* 512 KiB -> 128 KiB/chunk (K=103) */
    const size_t sz_large = 2 * 1024 * 1024;        /* 2 MiB   -> 512 KiB/chunk (K=128 @ T=4096) */
    const size_t sz_xl = 16ULL * 1024ULL * 1024ULL; /* 16 MiB  -> 4.0 MiB/chunk (4x 1 MiB RS8 windows) */
    int iters_xl = (iters_large > 12) ? 12 : iters_large;

    if (!run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 0.0% Lossless (GBN)", "gbn", sz_small,
                      0, 1, false, 0, 0, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 0.0% Lossless (vxfec)", "auto",
                      sz_small, 0, 1, false, 0, 0, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 0.2% OOO Loss (GBN)", "gbn", sz_small,
                      2000, 1, true, 0, 0, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 0.2% OOO Loss (SACK)", "sack",
                      sz_small, 2000, 1, true, 0, 0, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 0.2% OOO Loss (vxfec)", "auto",
                      sz_small, 2000, 1, true, 0, 0, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | ECMP Hotspot (GBN)", "gbn", sz_small,
                      5000, 2, true, 0, 0x0003u, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | ECMP Hotspot (SACK)", "sack",
                      sz_small, 5000, 2, true, 0, 0x0003u, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | ECMP Hotspot (vxfec)", "auto",
                      sz_small, 5000, 2, true, 0, 0x0003u, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 2.0% Burst=6 (GBN)", "gbn", sz_small,
                      20000, 6, true, 0, 0, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 2.0% Burst=6 (SACK)", "sack",
                      sz_small, 20000, 6, true, 0, 0, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 2.0% Burst=6 (MDS)", "rs8", sz_small,
                      20000, 6, true, 0, 0, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 2.0% Burst=6 (Rateless)", "rq",
                      sz_small, 20000, 6, true, 0, 0, iters_small) ||
        !run_incast_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "3->1 Incast 128K | ECN+2% (GBN)", "gbn",
                             sz_small / NUM_RANKS, 20000, 6, 400000, 2400, iters_small) ||
        !run_incast_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "3->1 Incast 128K | ECN+2% (SACK)",
                             "sack", sz_small / NUM_RANKS, 20000, 6, 400000, 2400, iters_small) ||
        !run_incast_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "3->1 Incast 128K | ECN+2% (vxfec)",
                             "auto", sz_small / NUM_RANKS, 20000, 6, 400000, 2400, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "2.0 MiB | 2.0% Burst=8 (GBN)", "gbn", sz_large,
                      20000, 8, true, 0, 0, iters_large) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "2.0 MiB | 2.0% Burst=8 (SACK)", "sack",
                      sz_large, 20000, 8, true, 0, 0, iters_large) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "2.0 MiB | 2.0% Burst=8 (vxfec)", "auto",
                      sz_large, 20000, 8, true, 0, 0, iters_large) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "16 MiB  | 0.0% Lossless (vxfec)", "auto", sz_xl,
                      0, 1, false, 0, 0, iters_xl) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "16 MiB  | 2.0% Burst=6 (GBN)", "gbn", sz_xl,
                      20000, 6, true, 0, 0, iters_xl) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "16 MiB  | 2.0% Burst=6 (SACK)", "sack", sz_xl,
                      20000, 6, true, 0, 0, iters_xl) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "16 MiB  | 2.0% Burst=6 (vxfec)", "auto", sz_xl,
                      20000, 6, true, 0, 0, iters_xl)) {
        if (set_delay)
            set_delay(0);
        set_codec(NULL);
        dlclose(dl);
        return 1;
    }

    if (set_delay) {
        /* 250 us one-way propagation delay (~500 us cross-spine / DCI fabric RTT) */
        set_delay(250000ULL);
        if (!run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 2% @ 500us RTT (GBN)", "gbn",
                          sz_small, 20000, 6, true, 0, 0, iters_large) ||
            !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 2% @ 500us RTT (SACK)", "sack",
                          sz_small, 20000, 6, true, 0, 0, iters_large) ||
            !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 2% @ 500us RTT (MDS)", "rs8",
                          sz_small, 20000, 6, true, 0, 0, iters_large) ||
            !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "16 MiB  | 2% @ 500us RTT (GBN)", "gbn",
                          sz_xl, 20000, 6, true, 0, 0, iters_xl) ||
            !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "16 MiB  | 2% @ 500us RTT (SACK)", "sack",
                          sz_xl, 20000, 6, true, 0, 0, iters_xl) ||
            !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "16 MiB  | 2% @ 500us RTT (vxfec)", "auto",
                          sz_xl, 20000, 6, true, 0, 0, iters_xl)) {
            set_delay(0);
            set_codec(NULL);
            dlclose(dl);
            return 1;
        }
        set_delay(0);
    }

    printf("===========================================================================================================\n");
    set_codec(NULL);
    dlclose(dl);
    return 0;
}
