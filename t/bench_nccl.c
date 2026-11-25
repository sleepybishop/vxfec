#define _GNU_SOURCE
#include "../plugins/nccl/nccl_net_v8.h"
#include "../plugins/nccl/nccl_net_vxfec.h"
#include "vxfec_vk.h"
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
#include <signal.h>
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

static bool poll_incast_request(const ncclNet_v8_t *net, void *req, int *done, int expected_bytes)
{
    if (!req || *done)
        return true;
    int size = 0;
    return net->test(req, done, &size) == ncclSuccess && (!*done || expected_bytes < 0 || size == expected_bytes);
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
        while (!rreq[s]) {
            if (net->irecv(ic->recv_comm[s], 1, &rptr, &rsz, &tag, &rx_mrs[s], &rreq[s]) != ncclSuccess)
                return false;
            if (!rreq[s]) {
                for (int j = 0; j < s; j++) {
                    if (!poll_incast_request(net, sreq[j], &sdone[j], -1) ||
                        !poll_incast_request(net, rreq[j], &rdone[j], (int)chunk_bytes))
                        return false;
                }
            }
        }
        while (!sreq[s]) {
            if (net->isend(ic->send_comm[s], tx_bufs[s], (int)chunk_bytes, tag, tx_mrs[s], &sreq[s]) != ncclSuccess)
                return false;
            if (!sreq[s]) {
                for (int j = 0; j <= s; j++) {
                    if (!poll_incast_request(net, sreq[j], &sdone[j], -1) ||
                        !poll_incast_request(net, rreq[j], &rdone[j], (int)chunk_bytes))
                        return false;
                }
            }
        }
    }

    int completed;
    do {
        completed = 0;
        for (int s = 0; s < INCAST_SENDERS; s++) {
            if (!poll_incast_request(net, sreq[s], &sdone[s], -1) ||
                !poll_incast_request(net, rreq[s], &rdone[s], (int)chunk_bytes))
                return false;
            completed += sdone[s] + rdone[s];
        }
    } while (completed < 2 * INCAST_SENDERS);
    return true;
}

typedef void (*get_stats_fn)(vxfec_nccl_stats_t *);
typedef void (*get_stats_ex_fn)(vxfec_nccl_stats_t *, size_t);
typedef void (*reset_stats_fn)(void);
typedef void (*set_fault_fn)(uint32_t, uint32_t, bool);
typedef void (*set_cong_fn)(uint32_t, uint32_t, uint64_t);
typedef void (*set_delay_fn)(uint64_t);
typedef void (*set_asym_fn)(uint32_t, uint64_t, uint64_t, uint32_t, uint32_t, uint32_t);
typedef void (*set_codec_fn)(const char *);

static get_stats_ex_fn g_bench_get_stats_ex = NULL;
static get_stats_fn g_bench_get_stats_v1 = NULL;
static size_t (*g_bench_get_rail_stats)(vxfec_nccl_rail_stats_t *, size_t);

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

    printf("  %-30s | %6.2f ms | %6.2f ms%s | %5.2f/%-5.2f | %7.2f Gbps | %6.2f%% | %5" PRIu64 " | %4" PRIu64 "/%-4" PRIu64
           " | %5" PRIu64 "\n",
           label, p50_ms, p99_ms, (iters < 100) ? "*" : " ", alg_gbps, bus_gbps, goodput_gbps, wire_tax_pct, st.sim_dropped_pkts,
           st.rx_rs8_decodes, st.rx_rq_decodes, st.tx_fec_clamped_pkts);
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

static bool run_leo_handoff_bench(const ncclNet_v8_t *net, get_stats_fn get_stats, reset_stats_fn reset_stats,
                                  set_fault_fn set_fault, set_cong_fn set_cong, set_asym_fn set_asym, set_delay_fn set_delay,
                                  set_codec_fn set_codec, int iters, bool standalone_hdr)
{
    if (!set_asym)
        return true;
    if (standalone_hdr) {
        printf("==================================================================================================================="
               "====\n");
        printf("vxfec Multi-Orbit LEO (35ms RTT) + Backup (85ms RTT) Beam-Handoff Benchmark (N=%d; * = max when N<100)\n", iters);
        printf("  Link Profile: 12x LEO subflows (17.5ms 1-way, 14-pkt beam-handoff blackout/96 + 1.5%% loss) + 4x Backup (42.5ms "
               "1-way)\n");
        printf("==================================================================================================================="
               "====\n");
        printf("  Scenario (Topology / Policy)   |   p50 Lat |   p99 Lat  | Alg / BusBW | Agg Goodput | WireTax | Drops |  RS8/RQ "
               "  | Clamp\n");
        printf("---------------------------------+-----------+------------+-------------+-------------+---------+-------+---------"
               "--+-------\n");
    }

    const size_t sz_telem = 512 * 1024;
    const struct {
        const char *label;
        const char *codec;
    } cases[] = {
        {"512 KiB | LEO+Backup Handoff (GBN)", "gbn"},
        {"512 KiB | LEO+Backup Handoff (SACK)", "sack"},
        {"512 KiB | LEO+Backup Handoff (MDS)", "rs8"},
        {"512 KiB | LEO+Backup Handoff (vxfec)", "rq-gf256"},
    };

    bool ok = true;
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]) && ok; c++) {
        /* 12 primary LEO subflows (0x0fff @ 17.5ms 1-way = 35ms RTT) + 4 terrestrial/MEO backup subflows (42.5ms 1-way = 85ms RTT)
         */
        set_asym(0x0fffu, 17500000ULL, 42500000ULL, 15000u, 96u, 14u);
        ok = run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, cases[c].label, cases[c].codec, sz_telem, 0u,
                          1u, true, 0u, 0u, iters);
    }
    set_asym(0u, 0ULL, 0ULL, 0u, 0u, 0u);
    if (set_delay)
        set_delay(0ULL);
    set_codec(NULL);
    if (standalone_hdr && ok)
        printf("===========================================================================================================\n");
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
#define MAX_GDR_RAILS 16

typedef struct {
    int proc_idx;
    int num_procs;
    int coord_fd;
    int local_fds[MAX_GDR_RAILS];
} mpi_node_ctx_t;

static void pin_proc_to_cpu_slice(int proc_idx, int num_procs)
{
    if (num_procs <= 1 || proc_idx < 0 || proc_idx >= num_procs)
        return;
    cpu_set_t base;
    CPU_ZERO(&base);
    if (sched_getaffinity(0, sizeof(base), &base) != 0)
        return;
    int cpus[CPU_SETSIZE];
    int ncpus = 0;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (CPU_ISSET(c, &base))
            cpus[ncpus++] = c;
    }
    if (ncpus < 2)
        return;
    cpu_set_t sub;
    CPU_ZERO(&sub);
    if (ncpus >= num_procs) {
        int lo = (proc_idx * ncpus) / num_procs;
        int hi = ((proc_idx + 1) * ncpus) / num_procs;
        if (hi <= lo)
            hi = lo + 1;
        for (int i = lo; i < hi && i < ncpus; i++)
            CPU_SET(cpus[i], &sub);
    } else {
        CPU_SET(cpus[proc_idx % ncpus], &sub);
    }
    (void)sched_setaffinity(0, sizeof(sub), &sub);
}

static bool local_mpi_barrier(const mpi_node_ctx_t *mpi, uint32_t step)
{
    if (!mpi || mpi->num_procs <= 1)
        return true;
    if (mpi->proc_idx == 0) {
        for (int p = 1; p < mpi->num_procs; p++) {
            uint32_t s = 0u;
            if (!tcp_recv_all(mpi->local_fds[p], &s, sizeof(s)) || s != step)
                return false;
        }
        for (int p = 1; p < mpi->num_procs; p++) {
            if (!tcp_send_all(mpi->local_fds[p], &step, sizeof(step)))
                return false;
        }
        return true;
    }
    uint32_t ack = 0u;
    return tcp_send_all(mpi->coord_fd, &step, sizeof(step)) && tcp_recv_all(mpi->coord_fd, &ack, sizeof(ack)) && (ack == step);
}

static bool cluster_barrier(int ctrl_fd, const mpi_node_ctx_t *mpi, uint32_t step)
{
    return local_mpi_barrier(mpi, step) && tcp_barrier(ctrl_fd, step);
}

typedef struct {
    int nr_rails;
    int nr_unique_gpus;
    int required_ibv_rails;
    vxfec_vk_ctx *unique_vk[MAX_GDR_RAILS];
    uint8_t unique_uuid[MAX_GDR_RAILS][16];
    int rail_gpu_slot[MAX_GDR_RAILS];
    vxfec_vk_ctx *rail_vk[MAX_GDR_RAILS];
} gdr_rail_pool_t;

static bool verify_gdr_transport(const vxfec_nccl_stats_t *stats, int required, int active, int rank)
{
    vxfec_nccl_rail_stats_t rails[MAX_GDR_RAILS] = {0};
    size_t count = g_bench_get_rail_stats ? g_bench_get_rail_stats(rails, MAX_GDR_RAILS) : 0;
    if (active != required || count != (size_t)required || stats->tx_udp_payload_pkts || !stats->tx_sys_gpu_direct_pkts ||
        stats->tx_sys_gpu_direct_pkts != stats->tx_sys_ibv_pkts || !stats->rx_sys_gpu_direct_pkts ||
        stats->rx_sys_gpu_direct_pkts != stats->rx_sys_pkts) {
        fprintf(stderr,
                "rank %d GDR exchange requires %d active IB rails (active=%d, reported=%zu), "
                "GPU-direct TX=%" PRIu64 "/%" PRIu64 " RX=%" PRIu64 "/%" PRIu64 " UDP payload=%" PRIu64 "\n",
                rank, required, active, count, stats->tx_sys_gpu_direct_pkts, stats->tx_sys_ibv_pkts, stats->rx_sys_gpu_direct_pkts,
                stats->rx_sys_pkts, stats->tx_udp_payload_pkts);
        return false;
    }
    for (size_t r = 0; r < count; r++) {
        /* Virtual rail replicas have a suffix after the port number and share an HCA. */
        const char *port = strrchr(rails[r].name, ':');
        if (strncmp(rails[r].name, "vxfec-ibv-", 10) != 0 || !port || !port[1] || port[1 + strspn(port + 1, "0123456789")] ||
            !rails[r].tx_sys_gpu_direct_pkts || !rails[r].rx_sys_gpu_direct_pkts) {
            fprintf(stderr, "rank %d GDR exchange requires measured GPU-direct TX/RX on physical IB port %s\n", rank,
                    rails[r].name);
            return false;
        }
    }
    return true;
}

static int bench_shared_pci_depth(const char *a, const char *b)
{
    int comps = 0;
    size_t i = 0;
    while (a[i] && b[i] && a[i] == b[i]) {
        if (a[i] == '/')
            comps++;
        i++;
    }
    if ((a[i] == '\0' || a[i] == '/') && (b[i] == '\0' || b[i] == '/'))
        return comps + 1;
    return comps;
}

static void gdr_rail_pool_free(gdr_rail_pool_t *gp)
{
    if (!gp)
        return;
    for (int i = 0; i < gp->nr_unique_gpus; i++) {
        if (gp->unique_vk[i])
            vxfec_vk_ctx_free(gp->unique_vk[i]);
    }
    memset(gp, 0, sizeof(*gp));
}

static int gdr_rail_pool_intern_vk(gdr_rail_pool_t *gp, vxfec_vk_ctx *vk)
{
    uint8_t uuid[16] = {0};
    bool has_u = vxfec_vk_device_uuid(vk, uuid);
    if (has_u) {
        for (int i = 0; i < gp->nr_unique_gpus; i++) {
            if (memcmp(gp->unique_uuid[i], uuid, 16) == 0) {
                vxfec_vk_ctx_free(vk);
                return i;
            }
        }
    }
    if (gp->nr_unique_gpus >= MAX_GDR_RAILS) {
        vxfec_vk_ctx_free(vk);
        return -1;
    }
    int slot = gp->nr_unique_gpus++;
    gp->unique_vk[slot] = vk;
    if (has_u)
        memcpy(gp->unique_uuid[slot], uuid, 16);
    return slot;
}

static bool gdr_rail_pool_init(const ncclNet_v8_t *net, int ndev, gdr_rail_pool_t *gp)
{
    memset(gp, 0, sizeof(*gp));
    if (ndev < 1)
        ndev = 1;
    if (ndev > MAX_GDR_RAILS)
        ndev = MAX_GDR_RAILS;
    gp->nr_rails = ndev;

    ncclNetProperties_v8_t p0 = {0};
    if (net->getProperties(0, &p0) != ncclSuccess || (p0.ptrSupport & NCCL_PTR_DMABUF) == 0)
        return false;

    const char *map_env = getenv("VXFEC_BENCH_GDR_VK_MAP");
    if (map_env && *map_env) {
        const char *cur = map_env;
        for (int r = 0; r < ndev; r++) {
            while (*cur == ',' || *cur == ' ')
                cur++;
            if (!*cur)
                cur = map_env;
            char *end = NULL;
            unsigned long vidx = strtoul(cur, &end, 10);
            vxfec_vk_ctx *vk = (end != cur) ? vxfec_vk_ctx_new_at((uint32_t)vidx, NULL) : vxfec_vk_ctx_new();
            if (!vk || !vxfec_bench_can_alloc_gdr_buf(vk)) {
                if (vk)
                    vxfec_vk_ctx_free(vk);
                gdr_rail_pool_free(gp);
                return false;
            }
            int slot = gdr_rail_pool_intern_vk(gp, vk);
            if (slot < 0) {
                gdr_rail_pool_free(gp);
                return false;
            }
            gp->rail_gpu_slot[r] = slot;
            gp->rail_vk[r] = gp->unique_vk[slot];
            cur = (*end == ',') ? (end + 1) : end;
        }
        return true;
    }

    /* Auto-discover PIX-local Vulkan GPUs matching each NIC's sysfs PCI path */
    const char *forced_vk = getenv("VXFEC_VK_DEVICE");
    if (!forced_vk || !*forced_vk) {
        char gpu_rp[MAX_GDR_RAILS][PATH_MAX];
        memset(gpu_rp, 0, sizeof(gpu_rp));
        for (uint32_t idx = 0; idx < 32u && gp->nr_unique_gpus < MAX_GDR_RAILS; idx++) {
            vxfec_vk_ctx *cand = vxfec_vk_ctx_new_at(idx, NULL);
            if (!cand)
                break;
            if (!vxfec_vk_is_discrete(cand) || !vxfec_bench_can_alloc_gdr_buf(cand)) {
                vxfec_vk_ctx_free(cand);
                continue;
            }
            int prev_cnt = gp->nr_unique_gpus;
            int slot = gdr_rail_pool_intern_vk(gp, cand);
            if (slot >= 0 && slot == prev_cnt) {
                char bdf[32] = {0}, sys_p[128];
                if (vxfec_vk_device_pci_bus_info(gp->unique_vk[slot], bdf, sizeof(bdf))) {
                    snprintf(sys_p, sizeof(sys_p), "/sys/bus/pci/devices/%.31s", bdf);
                    if (!realpath(sys_p, gpu_rp[slot]))
                        gpu_rp[slot][0] = '\0';
                }
            }
        }
        if (gp->nr_unique_gpus > 0) {
            for (int r = 0; r < ndev; r++) {
                ncclNetProperties_v8_t pr = {0};
                char nic_rp[PATH_MAX] = {0};
                if (net->getProperties(r, &pr) == ncclSuccess && pr.pciPath && pr.pciPath[0]) {
                    if (!realpath(pr.pciPath, nic_rp))
                        nic_rp[0] = '\0';
                }
                int best_slot = r % gp->nr_unique_gpus;
                int best_depth = -1;
                for (int s = 0; s < gp->nr_unique_gpus; s++) {
                    int d = (nic_rp[0] && gpu_rp[s][0]) ? bench_shared_pci_depth(nic_rp, gpu_rp[s]) : 0;
                    if (d > best_depth) {
                        best_depth = d;
                        best_slot = s;
                    }
                }
                gp->rail_gpu_slot[r] = best_slot;
                gp->rail_vk[r] = gp->unique_vk[best_slot];
            }
            return true;
        }
    }

    vxfec_vk_ctx *vk0 = vxfec_vk_ctx_new();
    if (!vk0 || !vxfec_bench_can_alloc_gdr_buf(vk0)) {
        if (vk0)
            vxfec_vk_ctx_free(vk0);
        return false;
    }
    int s0 = gdr_rail_pool_intern_vk(gp, vk0);
    if (s0 < 0)
        return false;
    for (int r = 0; r < ndev; r++) {
        gp->rail_gpu_slot[r] = s0;
        gp->rail_vk[r] = gp->unique_vk[s0];
    }
    return true;
}

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
    bool gdr_mode;
    size_t seg_off[2];
    size_t seg_len[2];
    uint32_t *buf;
    uint32_t *scratch;
    uint32_t **chan_buf;
    uint32_t **chan_scr;
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
    uint32_t *cbuf = mc->chan_buf ? mc->chan_buf[c] : mc->buf;
    uint32_t *cscr = mc->chan_scr ? mc->chan_scr[c] : mc->scratch;

    if (r_bytes > 0u) {
        void *rptr = mc->gdr_mode ? (void *)(cscr + mc->seg_off[mc->recv_half] + r_off)
                                  : ((phase == 0) ? (void *)(cscr + r_off) : (void *)(cbuf + mc->seg_off[mc->recv_half] + r_off));
        void *rmr = (mc->gdr_mode || phase == 0) ? mc->mr_scr[c] : mc->mr_buf[c];
        int rsz = (int)r_bytes;
        while (!rreq) {
            if (net->irecv(mc->recv_comm[c], 1, &rptr, &rsz, &tag, &rmr, &rreq) != ncclSuccess)
                return false;
        }
    }
    if (s_bytes > 0u) {
        void *sptr = cbuf + mc->seg_off[mc->send_half] + s_off;
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

/* host buffers run a two-rank ring sum; GDR buffers exchange the two halves of each rank's original payload */
static bool run_2node_exchange_once(multinode_comms_t *mc, int rank, uint32_t *buf, uint32_t *scratch, void **mr_buf, void **mr_scr,
                                    size_t total_elems, bool gdr_mode)
{
    if (rank < 0 || rank > 1)
        return false;
    mc->seg_off[0] = 0u;
    mc->seg_len[0] = total_elems / 2u;
    mc->seg_off[1] = mc->seg_len[0];
    mc->seg_len[1] = total_elems - mc->seg_len[0];
    int nchan = mc->nchan;
    mc->gdr_mode = gdr_mode;
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

        if (phase == 0 && !gdr_mode) {
            uint32_t *dst = buf + mc->seg_off[recv_half];
            size_t recv_elems = mc->seg_len[recv_half];
            for (size_t i = 0; i < recv_elems; i++)
                dst[i] += scratch[i];
        }
    }
    return true;
}

static bool run_2node_scenario(const ncclNet_v8_t *net, int ctrl_fd, int rank, const mpi_node_ctx_t *mpi, gdr_rail_pool_t *gdr_pool,
                               get_stats_fn get_stats, reset_stats_fn reset_stats, set_fault_fn set_fault, set_codec_fn set_codec,
                               const char *label, const char *codec_mode, size_t total_bytes, uint32_t loss_ppm, uint32_t burst,
                               bool reorder, int iters, FILE *csv_fp)
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
    if (!cluster_barrier(ctrl_fd, mpi, ++s_step))
        return false;

    multinode_comms_t mc;
    if (!setup_multinode_2rank(net, ctrl_fd, rank, &mc))
        return false;

    size_t page_sz = (size_t)sysconf(_SC_PAGESIZE);
    if (page_sz < 4096u)
        page_sz = 4096u;
    size_t total_elems = total_bytes / sizeof(uint32_t);
    size_t max_seg_elems = total_elems - (total_elems / 2u);
    size_t scratch_bytes = max_seg_elems * sizeof(uint32_t);
    bool gdr_mode = (gdr_pool != NULL && gdr_pool->nr_rails > 0);
    size_t gdr_aligned_bytes = (total_bytes + page_sz - 1u) & ~(page_sz - 1u);
    if (gdr_aligned_bytes < page_sz)
        gdr_aligned_bytes = page_sz;

    uint32_t *buf = NULL, *scratch = NULL, *expected = (uint32_t *)malloc(total_bytes);
    uint32_t *gdr_host = gdr_mode ? (uint32_t *)malloc(total_bytes) : NULL;
    uint32_t *gdr_rail_dl = (gdr_mode && gdr_pool->nr_rails > 1) ? (uint32_t *)malloc(total_bytes) : NULL;
    int active_gdr_rails = 0;
    vxfec_vk_buf *rail_vk_tx[MAX_GDR_RAILS] = {0};
    vxfec_vk_buf *rail_vk_rx[MAX_GDR_RAILS] = {0};
    int rail_tx_fd[MAX_GDR_RAILS];
    int rail_rx_fd[MAX_GDR_RAILS];
    for (int r = 0; r < MAX_GDR_RAILS; r++) {
        rail_tx_fd[r] = -1;
        rail_rx_fd[r] = -1;
    }
    void *rail_mr_buf[MAX_GDR_RAILS] = {0};
    void *rail_mr_scr[MAX_GDR_RAILS] = {0};
    uint32_t *chan_buf_ptrs[MAX_CHANS_2N] = {0};
    uint32_t *chan_scr_ptrs[MAX_CHANS_2N] = {0};
    void *mr_buf[MAX_CHANS_2N] = {0};
    void *mr_scr[MAX_CHANS_2N] = {0};
    ncclNetProperties_v8_t props = {0};
    bool global_mr = (net->getProperties(0, &props) == ncclSuccess && props.regIsGlobal != 0);
    bool ok = (expected != NULL);

    if (gdr_mode) {
        if (mc.ndev > MAX_GDR_RAILS)
            mc.ndev = MAX_GDR_RAILS;
        active_gdr_rails = (mc.nchan < gdr_pool->nr_rails) ? mc.nchan : gdr_pool->nr_rails;
        if (active_gdr_rails > mc.ndev)
            active_gdr_rails = mc.ndev;
        if (active_gdr_rails < 1)
            active_gdr_rails = 1;
        ok = ok && gdr_host && (active_gdr_rails == 1 || gdr_rail_dl != NULL) && ((props.ptrSupport & NCCL_PTR_DMABUF) != 0);
        if (ok) {
            for (size_t i = 0; i < total_elems; i++) {
                gdr_host[i] = ring_seed_word(rank, 1u, i);
                expected[i] = ring_seed_word(1 - rank, 1u, i);
            }
        }
        uint64_t rail_stride = ((uint64_t)gdr_aligned_bytes * 2ULL + 4ULL * page_sz + 0x3fffffffULL) & ~0x3fffffffULL;
        for (int r = 0; r < active_gdr_rails && ok; r++) {
            vxfec_vk_ctx *rvk = gdr_pool->rail_vk[r];
            rail_vk_tx[r] = vxfec_bench_alloc_gdr_buf(rvk, gdr_aligned_bytes, &rail_tx_fd[r]);
            rail_vk_rx[r] = vxfec_bench_alloc_gdr_buf(rvk, gdr_aligned_bytes, &rail_rx_fd[r]);
            if (!rail_vk_tx[r] || !rail_vk_rx[r] || rail_tx_fd[r] < 0 || rail_rx_fd[r] < 0) {
                ok = false;
                break;
            }
            ok = vxfec_vk_buf_write_offset(rvk, rail_vk_tx[r], 0u, gdr_host, total_bytes) &&
                 vxfec_vk_buf_release_external(rvk, rail_vk_tx[r]) &&
                 vxfec_vk_buf_fill_offset(rvk, rail_vk_rx[r], 0u, gdr_aligned_bytes, 0xa5a5a5a5u) &&
                 vxfec_vk_buf_release_external(rvk, rail_vk_rx[r]);
            if (!ok)
                break;
            uint32_t *rbuf = (uint32_t *)(uintptr_t)(0x710000000000ULL + (uint64_t)r * rail_stride);
            uint32_t *rscr = (uint32_t *)(uintptr_t)((uintptr_t)rbuf + gdr_aligned_bytes + page_sz);
            if (net->regMrDmaBuf(mc.send_comm[r], rbuf, gdr_aligned_bytes, NCCL_PTR_CUDA, 0u, rail_tx_fd[r], &rail_mr_buf[r]) !=
                    ncclSuccess ||
                net->regMrDmaBuf(mc.recv_comm[r], rscr, gdr_aligned_bytes, NCCL_PTR_CUDA, 0u, rail_rx_fd[r], &rail_mr_scr[r]) !=
                    ncclSuccess) {
                ok = false;
                break;
            }
            for (int c = r; c < mc.nchan; c += active_gdr_rails) {
                chan_buf_ptrs[c] = rbuf;
                chan_scr_ptrs[c] = rscr;
                mr_buf[c] = rail_mr_buf[r];
                mr_scr[c] = rail_mr_scr[r];
            }
        }
        buf = chan_buf_ptrs[0];
        scratch = chan_scr_ptrs[0];
        mc.chan_buf = chan_buf_ptrs;
        mc.chan_scr = chan_scr_ptrs;
    } else {
        ok = ok && (posix_memalign((void **)&buf, 4096, total_bytes) == 0) &&
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
        } else if (ok) {
            for (int c = 0; c < mc.nchan && ok; c++) {
                if (net->regMr(mc.send_comm[c], buf, total_bytes, NCCL_PTR_HOST, &mr_buf[c]) != ncclSuccess ||
                    net->regMr(mc.recv_comm[c], scratch, scratch_bytes, NCCL_PTR_HOST, &mr_scr[c]) != ncclSuccess)
                    ok = false;
            }
        }
    }

    if (ok) {
        set_fault(loss_ppm, burst, reorder);
        if (!cluster_barrier(ctrl_fd, mpi, ++s_step))
            ok = false;

        int warmups = (loss_ppm == 0u) ? 8 : 3;
        for (int w = 0; w < warmups && ok; w++) {
            if (!gdr_mode) {
                for (size_t i = 0; i < total_elems; i++)
                    buf[i] = ring_seed_word(rank, 5000u + (uint32_t)w, i);
            }
            ok = run_2node_exchange_once(&mc, rank, buf, scratch, mr_buf, mr_scr, total_elems, gdr_mode);
        }

        if (ok && !cluster_barrier(ctrl_fd, mpi, ++s_step))
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
            if (gdr_mode) {
                for (int r = 0; r < active_gdr_rails && ok; r++) {
                    vxfec_vk_ctx *rvk = gdr_pool->rail_vk[r];
                    if (!vxfec_vk_buf_fill_offset(rvk, rail_vk_rx[r], 0u, gdr_aligned_bytes, 0xa5a5a5a5u) ||
                        !vxfec_vk_buf_release_external(rvk, rail_vk_rx[r])) {
                        ok = false;
                        break;
                    }
                }
                if (!ok)
                    break;
            } else {
                uint32_t tag = (uint32_t)(it + 1);
                for (size_t i = 0; i < total_elems; i++) {
                    buf[i] = ring_seed_word(rank, tag, i);
                    expected[i] = ring_seed_word(0, tag, i) + ring_seed_word(1, tag, i);
                }
            }
            if (!cluster_barrier(ctrl_fd, mpi, ++s_step)) {
                ok = false;
                break;
            }
            uint64_t t0 = now_ns();
            ok = run_2node_exchange_once(&mc, rank, buf, scratch, mr_buf, mr_scr, total_elems, gdr_mode);
            lat_ns[it] = now_ns() - t0;
            if (ok && valid_ok) {
                if (gdr_mode) {
                    if (it + 1 == iters || iters <= 8 || total_bytes <= (1024u * 1024u)) {
                        memset(gdr_host, 0, total_bytes);
                        bool dl_ok = true;
                        if (active_gdr_rails == 1) {
                            dl_ok = vxfec_vk_buf_read_offset(gdr_pool->rail_vk[0], rail_vk_rx[0], 0u, gdr_host, total_bytes);
                        } else {
                            for (int r = 0; r < active_gdr_rails && dl_ok; r++) {
                                if (!vxfec_vk_buf_read_offset(gdr_pool->rail_vk[r], rail_vk_rx[r], 0u, gdr_rail_dl, total_bytes)) {
                                    dl_ok = false;
                                    break;
                                }
                                for (int h = 0; h < 2; h++) {
                                    for (int c = r; c < mc.nchan; c += active_gdr_rails) {
                                        size_t r_off = 0u, r_cnt = 0u;
                                        slice_channel_range(mc.seg_len[h], mc.nchan, c, &r_off, &r_cnt);
                                        if (r_cnt > 0u) {
                                            size_t off_el = mc.seg_off[h] + r_off;
                                            memcpy(gdr_host + off_el, gdr_rail_dl + off_el, r_cnt * sizeof(uint32_t));
                                        }
                                    }
                                }
                            }
                        }
                        if (!dl_ok || memcmp(gdr_host, expected, total_bytes) != 0) {
                            fprintf(stderr, "FAIL: 2-node GDR GPU-VRAM verification mismatch in '%s' at iter=%d\n", label, it);
                            valid_ok = false;
                        }
                    }
                } else if (memcmp(buf, expected, total_bytes) != 0) {
                    fprintf(stderr, "FAIL: 2-node AllReduce mismatch in '%s' at iter=%d\n", label, it);
                    valid_ok = false;
                }
            }
        }

        vxfec_nccl_stats_t my_st, peer_st;
        get_stats(&my_st);
        memset(&peer_st, 0, sizeof(peer_st));
        bool verify_ibv = gdr_mode && gdr_pool->required_ibv_rails > 0;
        if (verify_ibv && !verify_gdr_transport(&my_st, gdr_pool->required_ibv_rails, active_gdr_rails, rank))
            valid_ok = false;
        uint8_t my_ok = (ok && valid_ok) ? (verify_ibv ? 2u : 1u) : 0u;
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
        if (!valid_ok || peer_ok == 0u || peer_ok != my_ok)
            ok = false;

        int nprocs = (mpi && mpi->num_procs > 1) ? mpi->num_procs : 1;
        int pidx = mpi ? mpi->proc_idx : 0;

        if (rank == 0) {
            for (int i = 0; i < iters; i++) {
                if (peer_lat_ns[i] > lat_ns[i])
                    lat_ns[i] = peer_lat_ns[i];
            }
            uint64_t pair_sys = my_st.tx_sys_pkts + peer_st.tx_sys_pkts;
            uint64_t pair_extra = (my_st.tx_proactive_rep_pkts + peer_st.tx_proactive_rep_pkts) +
                                  (my_st.tx_sack_retx_pkts + peer_st.tx_sack_retx_pkts) +
                                  (my_st.tx_harq_ir_pkts + peer_st.tx_harq_ir_pkts);
            uint64_t pair_drops = my_st.sim_dropped_pkts + peer_st.sim_dropped_pkts;
            uint64_t pair_rs8 = my_st.rx_rs8_decodes + peer_st.rx_rs8_decodes;
            uint64_t pair_rq = my_st.rx_rq_decodes + peer_st.rx_rq_decodes;
            uint64_t pair_ibv = my_st.ibv_direct_pkts + peer_st.ibv_direct_pkts;

            if (nprocs > 1) {
                typedef struct {
                    uint8_t ok_flag;
                    uint64_t sys_pkts;
                    uint64_t extra_pkts;
                    uint64_t drops;
                    uint64_t rs8;
                    uint64_t rq;
                    uint64_t ibv;
                } mpi_pair_rep_t;
                mpi_pair_rep_t rep = {
                    .ok_flag = ok ? 1u : 0u,
                    .sys_pkts = pair_sys,
                    .extra_pkts = pair_extra,
                    .drops = pair_drops,
                    .rs8 = pair_rs8,
                    .rq = pair_rq,
                    .ibv = pair_ibv,
                };
                if (pidx > 0) {
                    if (!tcp_send_all(mpi->coord_fd, &rep, sizeof(rep)) ||
                        !tcp_send_all(mpi->coord_fd, lat_ns, (size_t)iters * sizeof(uint64_t)))
                        ok = false;
                } else {
                    for (int p = 1; p < nprocs; p++) {
                        mpi_pair_rep_t wrep;
                        uint64_t wlat[MAX_ITERS];
                        memset(&wrep, 0, sizeof(wrep));
                        memset(wlat, 0, sizeof(wlat));
                        if (!tcp_recv_all(mpi->local_fds[p], &wrep, sizeof(wrep)) ||
                            !tcp_recv_all(mpi->local_fds[p], wlat, (size_t)iters * sizeof(uint64_t)) || wrep.ok_flag == 0u) {
                            ok = false;
                        } else {
                            for (int i = 0; i < iters; i++) {
                                if (wlat[i] > lat_ns[i])
                                    lat_ns[i] = wlat[i];
                            }
                            pair_sys += wrep.sys_pkts;
                            pair_extra += wrep.extra_pkts;
                            pair_drops += wrep.drops;
                            pair_rs8 += wrep.rs8;
                            pair_rq += wrep.rq;
                            pair_ibv += wrep.ibv;
                        }
                    }
                }
            }

            if (ok && pidx == 0) {
                qsort(lat_ns, (size_t)iters, sizeof(uint64_t), cmp_u64);
                uint64_t total_ns = 0;
                for (int i = 0; i < iters; i++)
                    total_ns += lat_ns[i];
                double p50_ms = (double)lat_ns[iters / 2] / 1e6;
                size_t p99_idx = ((size_t)iters * 99u + 99u) / 100u - 1u;
                double p99_ms = (double)lat_ns[p99_idx] / 1e6;
                double max_ms = (double)lat_ns[iters - 1] / 1e6;
                if (total_ns == 0ULL)
                    total_ns = 1ULL;
                double eff_bytes = (double)total_bytes * (double)nprocs;
                double one_way_gbps = (8.0 * eff_bytes * (double)iters) / (double)total_ns;
                double goodput_gbps = 2.0 * one_way_gbps;
                double wire_tax = pair_sys ? (100.0 * (double)pair_extra / (double)pair_sys) : 0.0;

                printf("  %-30s | %6.2f ms | %6.2f ms%s | %6.2f Gbps | %7.2f Gbps | %6.2f%% | %5" PRIu64 " | %4" PRIu64
                       "/%-4" PRIu64 " | %5" PRIu64 "\n",
                       label, p50_ms, p99_ms, (iters < 100) ? "*" : " ", one_way_gbps, goodput_gbps, wire_tax, pair_drops, pair_rs8,
                       pair_rq, pair_ibv);
                fflush(stdout);

                if (csv_fp) {
                    fprintf(csv_fp, "\"%s%s\",%s,%d,%zu,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64,
                            verify_ibv ? "GDR exchange | verified IBV | " : (gdr_mode ? "GDR exchange | " : ""), label, codec_mode,
                            mc.ndev * nprocs, total_bytes * (size_t)nprocs, iters, p50_ms, p99_ms, max_ms, goodput_gbps, wire_tax,
                            pair_drops, pair_rs8, pair_rq, pair_ibv);
                    if (gdr_mode)
                        fputs(",,\n", csv_fp); /* alg_gbps and bus_gbps apply only to the host AllReduce */
                    else
                        fprintf(csv_fp, ",%.3f,%.3f\n", one_way_gbps, one_way_gbps);
                    if (fflush(csv_fp) != 0 || ferror(csv_fp)) {
                        perror("writing benchmark CSV");
                        ok = false;
                    }
                }
            }
        }
    }

    if (gdr_mode) {
        for (int r = 0; r < active_gdr_rails; r++) {
            if (rail_mr_scr[r])
                net->deregMr(mc.recv_comm[r], rail_mr_scr[r]);
            if (rail_mr_buf[r])
                net->deregMr(mc.send_comm[r], rail_mr_buf[r]);
        }
    } else if (global_mr) {
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
    teardown_multinode_2rank(net, &mc);
    if (gdr_mode) {
        for (int i = 0; i < gdr_pool->nr_unique_gpus; i++)
            (void)vxfec_vk_wait_idle(gdr_pool->unique_vk[i]);
        for (int r = 0; r < active_gdr_rails; r++) {
            if (rail_tx_fd[r] >= 0)
                close(rail_tx_fd[r]);
            if (rail_rx_fd[r] >= 0)
                close(rail_rx_fd[r]);
            vxfec_vk_buf_free(gdr_pool->rail_vk[r], rail_vk_rx[r]);
            vxfec_vk_buf_free(gdr_pool->rail_vk[r], rail_vk_tx[r]);
        }
        free(gdr_rail_dl);
        free(gdr_host);
    } else {
        free(scratch);
        free(buf);
    }
    free(expected);
    set_fault(0, 1, false);
    return ok;
}

static int run_multinode_main(const ncclNet_v8_t *net, get_stats_fn get_stats, reset_stats_fn reset_stats, set_fault_fn set_fault,
                              set_codec_fn set_codec, int rank, const mpi_node_ctx_t *mpi, const char *master_spec,
                              const char *csv_path, const char *single_codec, const char *single_label, size_t single_bytes,
                              uint32_t single_loss_ppm, uint32_t single_burst, int iters, bool use_gdr)
{
    char host[128];
    uint16_t base_port = 29600u;
    vxfec_bench_parse_host_port(master_spec, host, sizeof(host), 29600u, &base_port);
    int pidx = mpi ? mpi->proc_idx : 0;
    int nprocs = (mpi && mpi->num_procs > 1) ? mpi->num_procs : 1;
    uint16_t port = (uint16_t)(base_port + (uint16_t)pidx);

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
    ncclNetProperties_v8_t props = {0};
    (void)net->getProperties(0, &props);

    gdr_rail_pool_t gdr_pool;
    memset(&gdr_pool, 0, sizeof(gdr_pool));
    gdr_rail_pool_t *gdr_ptr = NULL;
    if (use_gdr) {
        if (!gdr_rail_pool_init(net, ndev, &gdr_pool)) {
            fprintf(stderr, "--gdr requires GPU DMA-BUF export and NCCL_PTR_DMABUF support on %s\n", net->name);
            close(ctrl_fd);
            return 1;
        }
        gdr_ptr = &gdr_pool;
        const char *backend = getenv("VXFEC_NCCL_BACKEND");
        const char *requested = getenv("VXFEC_BENCH_EXPECT_RAILS");
        if ((backend && strcmp(backend, "ibv") == 0) || requested) {
            char *end = NULL;
            long rails = requested ? strtol(requested, &end, 10) : ndev;
            if (rails < 1 || rails > MAX_GDR_RAILS || (requested && *end) || rails != ndev) {
                fprintf(stderr, "rank %d expected %s IB rails per process, found %d\n", rank, requested ? requested : "supported",
                        ndev);
                gdr_rail_pool_free(&gdr_pool);
                close(ctrl_fd);
                return 1;
            }
            gdr_pool.required_ibv_rails = (int)rails;
        }
    }

    FILE *csv_fp = NULL;
    if (rank == 0 && pidx == 0 && csv_path) {
        bool need_hdr = (access(csv_path, F_OK) != 0);
        csv_fp = fopen(csv_path, "a");
        if (!csv_fp) {
            perror(csv_path);
            gdr_rail_pool_free(&gdr_pool);
            close(ctrl_fd);
            return 1;
        }
        if (need_hdr) {
            fprintf(csv_fp,
                    "scenario,codec,ndev,bytes,iters,p50_ms,p99_ms,max_ms,goodput_gbps,wire_tax_pct,drops,rs8_dec,rq_dec,ibv_pkts,"
                    "alg_gbps,bus_gbps\n");
            fflush(csv_fp);
        }
    }

    char gdr_desc[96] = "";
    if (gdr_ptr)
        snprintf(gdr_desc, sizeof(gdr_desc), " --gdr VRAM (%d PIX GPUs)", gdr_ptr->nr_unique_gpus);
    if (nprocs > 1) {
        size_t glen = strlen(gdr_desc);
        snprintf(gdr_desc + glen, sizeof(gdr_desc) - glen, " [MPI %d procs/node]", nprocs);
    }

    bool print_header = rank == 0 && pidx == 0 && (!single_codec || use_gdr);
    if (print_header) {
        printf(
            "====================================================================================================================="
            "\n");
        printf("vxfec 2-Node Physical Fabric Benchmark (`%s`%s, ndev=%d/proc, nchan=%d/proc, speed=%d Mbps, N=%d; * = max when "
               "N<100)\n",
               net->name, gdr_desc, ndev, get_bench_2node_channels(ndev), props.speed, iters);
        printf(
            "====================================================================================================================="
            "\n");
        printf("  %-30s |    Median |      Tail | %10s | BiDir Goodput| WireTax | Drops |  RS8/RQ   | IBV\n",
               gdr_ptr ? "2-Node GDR Exchange" : "2-Node Host AllReduce", gdr_ptr ? "Each way" : "BusBW (1x)");
        printf(
            "---------------------------------+-----------+-----------+------------+-------------+---------+-------+-----------+--"
            "-----\n");
    }

    bool ok = true;
    if (single_codec) {
        ok = run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                                single_label ? single_label : single_codec, single_codec, single_bytes, single_loss_ppm,
                                single_burst, single_loss_ppm > 0u, iters, csv_fp);
    } else {
        const size_t sz_512k = 512 * 1024;
        const size_t sz_8m = 8 * 1024 * 1024;
        const size_t sz_64m = 64ULL * 1024ULL * 1024ULL;
        const size_t sz_256m = 256ULL * 1024ULL * 1024ULL;
        const char *env_max_mb = getenv("VXFEC_BENCH_MAX_MB");
        size_t max_mb = (env_max_mb && atoi(env_max_mb) >= 64) ? (size_t)atol(env_max_mb) : 64u;
        int it_lg = (iters >= 100) ? iters : ((iters >= 20) ? (iters / 2) : iters);
        int it_xl = (iters >= 100) ? iters : ((iters >= 20) ? 10 : iters);
        const char *env_xl = getenv("VXFEC_BENCH_XL_ITERS");
        if (env_xl && atoi(env_xl) >= 1)
            it_xl = atoi(env_xl);
        ok =
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "512 KiB | 0.0% Lossless (GBN)", "gbn", sz_512k, 0, 1, false, iters, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "512 KiB | 0.0% Lossless (vxfec)", "auto", sz_512k, 0, 1, false, iters, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "8.0 MiB | 0.0% Lossless (GBN)", "gbn", sz_8m, 0, 1, false, it_lg, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "8.0 MiB | 0.0% Lossless (vxfec)", "auto", sz_8m, 0, 1, false, it_lg, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "64 MiB  | 0.0% Lossless (vxfec)", "auto", sz_64m, 0, 1, false, it_xl, csv_fp) &&
            (max_mb < 256u || run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                                                 "256 MiB | 0.0% Lossless (vxfec)", "auto", sz_256m, 0, 1, false, it_xl, csv_fp)) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "512 KiB | 0.4% OOO Loss (GBN)", "gbn", sz_512k, 4000, 1, true, iters, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "512 KiB | 0.4% OOO Loss (SACK)", "sack", sz_512k, 4000, 1, true, iters, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "512 KiB | 0.4% OOO Loss (vxfec)", "auto", sz_512k, 4000, 1, true, iters, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "512 KiB | 2.0% Burst=6 (GBN)", "gbn", sz_512k, 20000, 6, true, iters, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "512 KiB | 2.0% Burst=6 (SACK)", "sack", sz_512k, 20000, 6, true, iters, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "512 KiB | 2.0% Burst=6 (MDS)", "rs8", sz_512k, 20000, 6, true, iters, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "512 KiB | 2.0% Burst=6 (RQ-GF256)", "rq-gf256", sz_512k, 20000, 6, true, iters, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "512 KiB | 2.0% Burst=6 (RQ-GF2)", "rq-gf2", sz_512k, 20000, 6, true, iters, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "8.0 MiB | 2.0% Burst=6 (GBN)", "gbn", sz_8m, 20000, 6, true, it_lg, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "8.0 MiB | 2.0% Burst=6 (SACK)", "sack", sz_8m, 20000, 6, true, it_lg, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "8.0 MiB | 2.0% Burst=6 (RQ-GF256)", "rq-gf256", sz_8m, 20000, 6, true, it_lg, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "8.0 MiB | 2.0% Burst=6 (RQ-GF2)", "rq-gf2", sz_8m, 20000, 6, true, it_lg, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "8.0 MiB | 2.0% Burst=6 (vxfec)", "auto", sz_8m, 20000, 6, true, it_lg, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "64 MiB  | 2.0% Burst=6 (SACK)", "sack", sz_64m, 20000, 6, true, it_xl, csv_fp) &&
            run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                               "64 MiB  | 2.0% Burst=6 (vxfec)", "auto", sz_64m, 20000, 6, true, it_xl, csv_fp) &&
            (max_mb < 256u ||
             (run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                                 "256 MiB | 2.0% Burst=6 (SACK)", "sack", sz_256m, 20000, 6, true, it_xl, csv_fp) &&
              run_2node_scenario(net, ctrl_fd, rank, mpi, gdr_ptr, get_stats, reset_stats, set_fault, set_codec,
                                 "256 MiB | 2.0% Burst=6 (vxfec)", "auto", sz_256m, 20000, 6, true, it_xl, csv_fp)));
    }

    if (print_header)
        printf("===========================================================================================================\n");

    gdr_rail_pool_free(&gdr_pool);

    if (csv_fp && fclose(csv_fp) != 0) {
        perror(csv_path);
        ok = false;
    }
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

static int run_2node_self_test(bool want_gdr)
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
         run_2node_scenario(net, ctrl_fd, my_rank, NULL, NULL, get_stats, reset_stats, set_fault, set_codec,
                            "2N SelfTest 1 elem (4 B)", "auto", 4u, 0u, 1u, false, 4, NULL) &&
         run_2node_scenario(net, ctrl_fd, my_rank, NULL, NULL, get_stats, reset_stats, set_fault, set_codec,
                            "2N SelfTest 3 elems (12 B)", "auto", 12u, 0u, 1u, false, 4, NULL) &&
         run_2node_scenario(net, ctrl_fd, my_rank, NULL, NULL, get_stats, reset_stats, set_fault, set_codec,
                            "2N SelfTest 5 elems (20 B)", "auto", 20u, 0u, 1u, false, 4, NULL) &&
         run_2node_scenario(net, ctrl_fd, my_rank, NULL, NULL, get_stats, reset_stats, set_fault, set_codec,
                            "2N SelfTest 131073 elems Loss", "auto", 131073u * sizeof(uint32_t), 20000u, 4u, true, 4, NULL);

    if (ok) {
        gdr_rail_pool_t gp_self;
        memset(&gp_self, 0, sizeof(gp_self));
        int ndev = 1;
        (void)net->devices(&ndev);
        uint8_t my_gdr = gdr_rail_pool_init(net, ndev, &gp_self) ? 1u : 0u;
        uint8_t peer_gdr = 0u;
        if (my_rank == 0)
            ok = tcp_send_all(ctrl_fd, &my_gdr, 1u) && tcp_recv_all(ctrl_fd, &peer_gdr, 1u);
        else
            ok = tcp_recv_all(ctrl_fd, &peer_gdr, 1u) && tcp_send_all(ctrl_fd, &my_gdr, 1u);
        if (ok && my_gdr && peer_gdr) {
            ok = run_2node_scenario(net, ctrl_fd, my_rank, NULL, &gp_self, get_stats, reset_stats, set_fault, set_codec,
                                    "2N SelfTest GDR VRAM Loss", "auto", 262144u, 20000u, 4u, true, 3, NULL);
            if (ok && my_rank == 0)
                printf("PASS: two-phase GDR DMA-BUF exchange verified peer payloads\n");
        } else if (want_gdr) {
            ok = false;
        }
        gdr_rail_pool_free(&gp_self);
    }

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
        printf("PASS: 2-rank odd-length host AllReduce self-test\n");
    return ok ? 0 : 1;
}

typedef struct {
    uint8_t ok;
    uint64_t drops;
    uint64_t rs8_dec;
    uint64_t rq_dec;
    uint64_t lat_ns[MAX_ITERS];
} scale_worker_report_t;

static int run_scale_proxy_pair_child(int pair_idx, int num_pairs, int rails_in_proc, int rw_mode, size_t bytes_per_rail,
                                      uint32_t loss_ppm, uint32_t burst, int iters, int sync_fd)
{
    int peer_sv[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, peer_sv) != 0)
        return 1;

    pid_t side_pid = fork();
    if (side_pid < 0) {
        close(peer_sv[0]);
        close(peer_sv[1]);
        return 1;
    }

    int side = (side_pid == 0) ? 1 : 0;
    int pfd = peer_sv[side];
    close(peer_sv[1 - side]);
    if (side == 1)
        close(sync_fd);

    /* pin side 0 and side 1 of each pair to distinct CPU slices so all rails progress concurrently */
    pin_proc_to_cpu_slice(pair_idx * 2 + side, num_pairs * 2);

    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < 65536u && rl.rlim_max > rl.rlim_cur) {
        rl.rlim_cur = (rl.rlim_max < 65536u) ? rl.rlim_max : 65536u;
        (void)setrlimit(RLIMIT_NOFILE, &rl);
    }

    char rails_str[16], rw_str[16];
    snprintf(rails_str, sizeof(rails_str), "%d", rails_in_proc);
    setenv("VXFEC_NCCL_IFNAMES", "lo", 1);
    setenv("VXFEC_NCCL_PORTS", "4", 1);
    setenv("VXFEC_NCCL_VULKAN", "0", 1);
    setenv("VXFEC_NCCL_RAILS", rails_str, 1);
    if (rw_mode < 0) {
        setenv("VXFEC_NCCL_RAIL_WORKERS", "auto", 1);
    } else {
        snprintf(rw_str, sizeof(rw_str), "%d", rw_mode);
        setenv("VXFEC_NCCL_RAIL_WORKERS", rw_str, 1);
    }

    void *dl = dlopen(resolve_nccl_plugin_path(), RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        close(pfd);
        if (side == 1)
            _exit(1);
        (void)waitpid(side_pid, NULL, 0);
        return 1;
    }
    pin_vulkan_for_lsan();

    const ncclNet_v8_t *net = (const ncclNet_v8_t *)dlsym(dl, "ncclNetPlugin_v8");
    g_bench_get_stats_ex = (get_stats_ex_fn)dlsym(dl, "vxfec_nccl_get_stats_ex");
    g_bench_get_stats_v1 = (get_stats_fn)dlsym(dl, "vxfec_nccl_get_stats");
    reset_stats_fn reset_stats = (reset_stats_fn)dlsym(dl, "vxfec_nccl_reset_stats");
    set_fault_fn set_fault = (set_fault_fn)dlsym(dl, "vxfec_nccl_set_fault_injection");

    int ndev = 0;
    bool ok = net && (g_bench_get_stats_ex || g_bench_get_stats_v1) && reset_stats && set_fault &&
              net->init(silent_logger) == ncclSuccess && net->devices(&ndev) == ncclSuccess && ndev >= rails_in_proc;

    void *lc[MAX_GDR_RAILS] = {0}, *sc[MAX_GDR_RAILS] = {0}, *rc[MAX_GDR_RAILS] = {0};
    void *tx_mr[MAX_GDR_RAILS] = {0}, *rx_mr[MAX_GDR_RAILS] = {0};
    uint32_t *tx_buf[MAX_GDR_RAILS] = {0}, *rx_buf[MAX_GDR_RAILS] = {0};
    uint8_t my_h[MAX_GDR_RAILS][NCCL_NET_HANDLE_MAXSIZE];
    uint8_t peer_h[MAX_GDR_RAILS][NCCL_NET_HANDLE_MAXSIZE];
    memset(my_h, 0, sizeof(my_h));
    memset(peer_h, 0, sizeof(peer_h));

    for (int r = 0; r < rails_in_proc && ok; r++) {
        if (net->listen(r, my_h[r], &lc[r]) != ncclSuccess)
            ok = false;
    }
    if (ok) {
        if (side == 0) {
            ok = tcp_send_all(pfd, my_h, (size_t)rails_in_proc * NCCL_NET_HANDLE_MAXSIZE) &&
                 tcp_recv_all(pfd, peer_h, (size_t)rails_in_proc * NCCL_NET_HANDLE_MAXSIZE);
        } else {
            ok = tcp_recv_all(pfd, peer_h, (size_t)rails_in_proc * NCCL_NET_HANDLE_MAXSIZE) &&
                 tcp_send_all(pfd, my_h, (size_t)rails_in_proc * NCCL_NET_HANDLE_MAXSIZE);
        }
    }
    for (int r = 0; r < rails_in_proc && ok; r++) {
        while (!sc[r] || !rc[r]) {
            if (!sc[r] && net->connect(r, peer_h[r], &sc[r], NULL) != ncclSuccess) {
                ok = false;
                break;
            }
            if (!rc[r] && net->accept(lc[r], &rc[r], NULL) != ncclSuccess) {
                ok = false;
                break;
            }
        }
        if (ok) {
            if (posix_memalign((void **)&tx_buf[r], 4096u, bytes_per_rail) != 0 ||
                posix_memalign((void **)&rx_buf[r], 4096u, bytes_per_rail) != 0 ||
                net->regMr(sc[r], tx_buf[r], bytes_per_rail, NCCL_PTR_HOST, &tx_mr[r]) != ncclSuccess ||
                net->regMr(rc[r], rx_buf[r], bytes_per_rail, NCCL_PTR_HOST, &rx_mr[r]) != ncclSuccess) {
                ok = false;
            } else {
                size_t nelem = bytes_per_rail / sizeof(uint32_t);
                for (size_t i = 0; i < nelem; i++)
                    tx_buf[r][i] = ring_seed_word(side * 32 + r, 7u, i);
                memset(rx_buf[r], 0, bytes_per_rail);
            }
        }
    }

    scale_worker_report_t rep;
    memset(&rep, 0, sizeof(rep));
    if (iters > MAX_ITERS)
        iters = MAX_ITERS;
    if (bytes_per_rail > (size_t)INT_MAX)
        ok = false;

    if (ok) {
        set_fault(loss_ppm, burst ? burst : 1u, loss_ppm > 0u);
        for (int it = -3; it < iters && ok; it++) {
            uint32_t step = (uint32_t)(it + 10);
            if (side == 0) {
                uint32_t go = 0u;
                if (!tcp_send_all(sync_fd, &step, sizeof(step)) || !tcp_recv_all(sync_fd, &go, sizeof(go)) || go != step) {
                    ok = false;
                    break;
                }
            }
            if (!tcp_barrier(pfd, step)) {
                ok = false;
                break;
            }
            if (it == 0)
                reset_stats();

            uint64_t t0 = now_ns();
            /* single NCCL-style proxy loop driving all rails_in_proc concurrently */
            void *sreq[MAX_GDR_RAILS] = {0}, *rreq[MAX_GDR_RAILS] = {0};
            int sdone[MAX_GDR_RAILS] = {0}, rdone[MAX_GDR_RAILS] = {0};
            for (int r = 0; r < rails_in_proc && ok; r++) {
                void *rptr = rx_buf[r];
                int rsz = (int)bytes_per_rail;
                int tag = 2000 + r;
                while (!rreq[r]) {
                    if (net->irecv(rc[r], 1, &rptr, &rsz, &tag, &rx_mr[r], &rreq[r]) != ncclSuccess) {
                        ok = false;
                        break;
                    }
                }
                while (ok && !sreq[r]) {
                    if (net->isend(sc[r], tx_buf[r], (int)bytes_per_rail, tag, tx_mr[r], &sreq[r]) != ncclSuccess) {
                        ok = false;
                        break;
                    }
                }
            }
            int done_cnt = 0;
            while (ok && done_cnt < 2 * rails_in_proc) {
                for (int r = 0; r < rails_in_proc; r++) {
                    if (!sdone[r]) {
                        int sz = 0;
                        if (net->test(sreq[r], &sdone[r], &sz) != ncclSuccess) {
                            ok = false;
                            break;
                        }
                        if (sdone[r])
                            done_cnt++;
                    }
                    if (!rdone[r]) {
                        int sz = 0;
                        if (net->test(rreq[r], &rdone[r], &sz) != ncclSuccess) {
                            ok = false;
                            break;
                        }
                        if (rdone[r]) {
                            if ((size_t)sz != bytes_per_rail) {
                                ok = false;
                                break;
                            }
                            done_cnt++;
                        }
                    }
                }
            }
            uint64_t dt = now_ns() - t0;
            if (it >= 0)
                rep.lat_ns[it] = dt;
        }

        if (ok) {
            size_t nelem = bytes_per_rail / sizeof(uint32_t);
            for (int r = 0; r < rails_in_proc && ok; r++) {
                for (size_t i = 0; i < nelem; i++) {
                    if (rx_buf[r][i] != ring_seed_word((1 - side) * 32 + r, 7u, i)) {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if (ok) {
            vxfec_nccl_stats_t my_st, peer_st;
            bench_query_stats(&my_st);
            memset(&peer_st, 0, sizeof(peer_st));
            uint64_t peer_lat[MAX_ITERS] = {0};
            uint8_t my_ok = ok ? 1u : 0u, peer_ok = 0u;
            if (side == 0) {
                if (!tcp_recv_all(pfd, &peer_ok, 1u) || !tcp_send_all(pfd, &my_ok, 1u) ||
                    !tcp_recv_all(pfd, &peer_st, sizeof(peer_st)) ||
                    !tcp_recv_all(pfd, peer_lat, (size_t)iters * sizeof(uint64_t)) || peer_ok == 0u)
                    ok = false;
                for (int i = 0; i < iters; i++) {
                    if (peer_lat[i] > rep.lat_ns[i])
                        rep.lat_ns[i] = peer_lat[i];
                }
                rep.drops = my_st.sim_dropped_pkts + peer_st.sim_dropped_pkts;
                rep.rs8_dec = my_st.rx_rs8_decodes + peer_st.rx_rs8_decodes;
                rep.rq_dec = my_st.rx_rq_decodes + peer_st.rx_rq_decodes;
            } else {
                if (!tcp_send_all(pfd, &my_ok, 1u) || !tcp_recv_all(pfd, &peer_ok, 1u) ||
                    !tcp_send_all(pfd, &my_st, sizeof(my_st)) || !tcp_send_all(pfd, rep.lat_ns, (size_t)iters * sizeof(uint64_t)) ||
                    peer_ok == 0u)
                    ok = false;
            }
        }
    }

    for (int r = 0; r < rails_in_proc; r++) {
        if (rx_mr[r])
            net->deregMr(rc[r], rx_mr[r]);
        if (tx_mr[r])
            net->deregMr(sc[r], tx_mr[r]);
        if (sc[r])
            net->closeSend(sc[r]);
        if (rc[r])
            net->closeRecv(rc[r]);
        if (lc[r])
            net->closeListen(lc[r]);
        free(tx_buf[r]);
        free(rx_buf[r]);
    }
    set_fault(0u, 1u, false);
    dlclose(dl);
    close(pfd);

    if (side == 1)
        _exit(ok ? 0 : 1);

    int st = 0;
    if (waitpid(side_pid, &st, 0) < 0 || !WIFEXITED(st) || WEXITSTATUS(st) != 0)
        ok = false;
    rep.ok = ok ? 1u : 0u;
    (void)tcp_send_all(sync_fd, &rep, sizeof(rep));
    close(sync_fd);
    return ok ? 0 : 1;
}

static bool run_scale_case(const char *arch, int total_rails, bool mpi_mode, int rw_mode, size_t bytes_per_rail, uint32_t loss_ppm,
                           uint32_t burst, int iters, double *io_base_gbps, FILE *csv_fp)
{
    if (iters > MAX_ITERS)
        iters = MAX_ITERS;
    int num_pairs = mpi_mode ? total_rails : 1;
    int rails_per_pair = mpi_mode ? 1 : total_rails;
    int sv[MAX_GDR_RAILS][2];
    pid_t pids[MAX_GDR_RAILS];
    for (int p = 0; p < num_pairs; p++) {
        sv[p][0] = sv[p][1] = -1;
        pids[p] = -1;
    }

    for (int p = 0; p < num_pairs; p++) {
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv[p]) != 0) {
            for (int k = 0; k < p; k++) {
                if (sv[k][0] >= 0)
                    close(sv[k][0]);
                if (pids[k] > 0) {
                    (void)kill(pids[k], SIGTERM);
                    (void)waitpid(pids[k], NULL, 0);
                }
            }
            return false;
        }
        pid_t pid = fork();
        if (pid < 0) {
            close(sv[p][0]);
            close(sv[p][1]);
            for (int k = 0; k < p; k++) {
                if (sv[k][0] >= 0)
                    close(sv[k][0]);
                if (pids[k] > 0) {
                    (void)kill(pids[k], SIGTERM);
                    (void)waitpid(pids[k], NULL, 0);
                }
            }
            return false;
        }
        if (pid == 0) {
            for (int k = 0; k < p; k++)
                close(sv[k][0]);
            close(sv[p][0]);
            _exit(run_scale_proxy_pair_child(p, num_pairs, rails_per_pair, rw_mode, bytes_per_rail, loss_ppm, burst, iters,
                                             sv[p][1]));
        }
        close(sv[p][1]);
        pids[p] = pid;
    }

    bool ok = (bytes_per_rail <= (size_t)INT_MAX);
    for (int it = -3; it < iters && ok; it++) {
        uint32_t exp = (uint32_t)(it + 10);
        for (int p = 0; p < num_pairs && ok; p++) {
            uint32_t got = 0u;
            if (!tcp_recv_all(sv[p][0], &got, sizeof(got)) || got != exp)
                ok = false;
        }
        for (int p = 0; p < num_pairs && ok; p++) {
            if (!tcp_send_all(sv[p][0], &exp, sizeof(exp)))
                ok = false;
        }
    }

    uint64_t cct_ns[MAX_ITERS] = {0};
    uint64_t drops = 0u, rs8 = 0u, rq = 0u;
    for (int p = 0; p < num_pairs; p++) {
        scale_worker_report_t rep;
        memset(&rep, 0, sizeof(rep));
        if (!tcp_recv_all(sv[p][0], &rep, sizeof(rep)) || rep.ok == 0u)
            ok = false;
        close(sv[p][0]);
        int st = 0;
        if (waitpid(pids[p], &st, 0) < 0 || !WIFEXITED(st) || WEXITSTATUS(st) != 0)
            ok = false;
        for (int i = 0; i < iters; i++) {
            if (rep.lat_ns[i] > cct_ns[i])
                cct_ns[i] = rep.lat_ns[i];
        }
        drops += rep.drops;
        rs8 += rep.rs8_dec;
        rq += rep.rq_dec;
    }
    if (!ok)
        return false;

    qsort(cct_ns, (size_t)iters, sizeof(uint64_t), cmp_u64);
    uint64_t total_ns = 0ULL;
    for (int i = 0; i < iters; i++)
        total_ns += cct_ns[i];
    if (total_ns == 0ULL)
        total_ns = 1ULL;
    double p50_ms = (double)cct_ns[iters / 2] / 1e6;
    size_t p99_idx = ((size_t)iters * 99u + 99u) / 100u - 1u;
    double p99_ms = (double)cct_ns[p99_idx] / 1e6;
    double max_ms = (double)cct_ns[iters - 1] / 1e6;
    double agg_bytes = (double)total_rails * (double)bytes_per_rail;
    double goodput_gbps = (8.0 * 2.0 * agg_bytes * (double)iters) / (double)total_ns;
    if (total_rails == 1 || *io_base_gbps <= 0.0)
        *io_base_gbps = goodput_gbps;
    double speedup = (*io_base_gbps > 0.0) ? (goodput_gbps / *io_base_gbps) : 1.0;

    printf("  %-22s | %5d | %6.1f MiB | %6.2f ms | %6.2f ms%s | %9.2f Gbps | %6.2fx | %5" PRIu64 " | %4" PRIu64 "/%-4" PRIu64 "\n",
           arch, total_rails, agg_bytes / (1024.0 * 1024.0), p50_ms, p99_ms, (iters < 100) ? "*" : " ", goodput_gbps, speedup,
           drops, rs8, rq);
    fflush(stdout);

    if (csv_fp) {
        fprintf(csv_fp, "%s,%d,%zu,%.0f,%u,%u,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", arch, total_rails,
                bytes_per_rail, agg_bytes, loss_ppm, burst, iters, p50_ms, p99_ms, max_ms, goodput_gbps, speedup, drops, rs8, rq);
        fflush(csv_fp);
    }
    return true;
}

static int run_scale_bench_main(int max_rails, size_t bytes_per_rail, uint32_t loss_ppm, uint32_t burst, int iters,
                                const char *csv_path)
{
    if (max_rails < 1)
        max_rails = 1;
    if (max_rails > MAX_GDR_RAILS)
        max_rails = MAX_GDR_RAILS;
    if (bytes_per_rail < 4096u)
        bytes_per_rail = 4096u;
    bytes_per_rail &= ~(size_t)3u;
    if (iters < 1)
        iters = 1;

    FILE *csv_fp = NULL;
    if (csv_path) {
        bool need_hdr = (access(csv_path, F_OK) != 0);
        csv_fp = fopen(csv_path, "a");
        if (csv_fp && need_hdr) {
            fprintf(csv_fp, "arch,rails,bytes_per_rail,agg_bytes,loss_ppm,burst,iters,p50_ms,p99_ms,max_ms,agg_goodput_gbps,"
                            "speedup_vs_1rail,drops,rs8_dec,rq_dec\n");
            fflush(csv_fp);
        }
    }

    printf("===============================================================================================================\n");
    printf("vxfec Multi-Rail Scaling Benchmark (max_rails=%d, %.2f MiB/rail, loss=%.2f%% burst=%u, N=%d)\n", max_rails,
           (double)bytes_per_rail / (1024.0 * 1024.0), (double)loss_ppm / 10000.0, burst ? burst : 1u, iters);
    printf("===============================================================================================================\n");
    printf("  Architecture           | Rails | Agg Tensor |   p50 CCT |  Tail CCT  | BiDir Goodput | Speedup | Drops |  RS8/RQ\n");
    printf("-------------------------+-------+------------+-----------+------------+---------------+---------+-------+---------\n");

    struct {
        const char *name;
        bool mpi;
        const char *rw_env;
    } modes[3] = {
        {"1-Proc Single-Proxy", false, "0"},
        {"1-Proc Rail-Workers", false, "auto"},
        {"MPI 1-Proc-Per-NIC", true, "auto"},
    };

    bool ok = true;
    for (int m = 0; m < 3 && ok; m++) {
        int rw_val = (strcmp(modes[m].rw_env, "0") == 0) ? 0 : -1;
        double base_gbps = 0.0;
        for (int r = 1; r <= max_rails && ok; r *= 2) {
            ok = run_scale_case(modes[m].name, r, modes[m].mpi, rw_val, bytes_per_rail, loss_ppm, burst ? burst : 1u, iters,
                                &base_gbps, csv_fp);
            if (r * 2 > max_rails && r != max_rails && ok) {
                ok = run_scale_case(modes[m].name, max_rails, modes[m].mpi, rw_val, bytes_per_rail, loss_ppm, burst ? burst : 1u,
                                    iters, &base_gbps, csv_fp);
            }
        }
        if (m < 2 && ok)
            printf("-------------------------+-------+------------+-----------+------------+---------------+---------+-------+-----"
                   "----\n");
    }
    printf("===============================================================================================================\n");
    if (csv_fp)
        fclose(csv_fp);
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    int rank = -1;
    int procs_per_node = 1;
    int scale_rails = 8;
    bool self_test = false;
    bool scale_bench = false;
    bool leo_bench = false;
    bool use_gdr = false;
    bool has_bytes_arg = false;
    const char *master_spec = NULL;
    const char *csv_path = NULL;
    const char *single_codec = NULL;
    const char *single_label = NULL;
    size_t single_bytes = 512 * 1024;
    uint32_t single_loss_ppm = 0u;
    uint32_t single_burst = 1u;
    int cli_iters = 100;
    bool has_iters_arg = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-test") == 0)
            self_test = true;
        else if (strcmp(argv[i], "--scale-bench") == 0)
            scale_bench = true;
        else if (strcmp(argv[i], "--leo-bench") == 0)
            leo_bench = true;
        else if (strcmp(argv[i], "--rails") == 0 && i + 1 < argc)
            scale_rails = atoi(argv[++i]);
        else if (strcmp(argv[i], "--procs-per-node") == 0 && i + 1 < argc)
            procs_per_node = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gdr") == 0)
            use_gdr = true;
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
        else if (strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            single_bytes = (size_t)strtoull(argv[++i], NULL, 10);
            has_bytes_arg = true;
        } else if (strcmp(argv[i], "--loss-ppm") == 0 && i + 1 < argc)
            single_loss_ppm = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--burst") == 0 && i + 1 < argc)
            single_burst = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            cli_iters = atoi(argv[++i]);
            has_iters_arg = true;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            return 1;
        }
    }

    if (self_test) {
        if (run_2node_self_test(use_gdr) != 0)
            return 1;
        if (run_scale_bench_main(2, 65536u, 20000u, 4u, 4, NULL) != 0)
            return 1;
        return 0;
    }

    if (scale_bench) {
        size_t b = has_bytes_arg ? single_bytes : (8ULL * 1024ULL * 1024ULL);
        return run_scale_bench_main(scale_rails, b, single_loss_ppm, single_burst, cli_iters, csv_path);
    }

    mpi_node_ctx_t mpi_ctx;
    memset(&mpi_ctx, 0, sizeof(mpi_ctx));
    mpi_ctx.proc_idx = 0;
    mpi_ctx.num_procs = 1;
    mpi_ctx.coord_fd = -1;
    for (int i = 0; i < MAX_GDR_RAILS; i++)
        mpi_ctx.local_fds[i] = -1;
    pid_t mpi_pids[MAX_GDR_RAILS];
    memset(mpi_pids, 0, sizeof(mpi_pids));

    if (rank >= 0 && master_spec && procs_per_node > 1) {
        if (procs_per_node > MAX_GDR_RAILS)
            procs_per_node = MAX_GDR_RAILS;
        mpi_ctx.num_procs = procs_per_node;
        for (int p = 1; p < procs_per_node; p++) {
            int sv[2] = {-1, -1};
            if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
                for (int k = 1; k < p; k++) {
                    if (mpi_ctx.local_fds[k] >= 0)
                        close(mpi_ctx.local_fds[k]);
                    if (mpi_pids[k] > 0) {
                        (void)kill(mpi_pids[k], SIGTERM);
                        (void)waitpid(mpi_pids[k], NULL, 0);
                    }
                }
                return 1;
            }
            pid_t pid = fork();
            if (pid < 0) {
                close(sv[0]);
                close(sv[1]);
                for (int k = 1; k < p; k++) {
                    if (mpi_ctx.local_fds[k] >= 0)
                        close(mpi_ctx.local_fds[k]);
                    if (mpi_pids[k] > 0) {
                        (void)kill(mpi_pids[k], SIGTERM);
                        (void)waitpid(mpi_pids[k], NULL, 0);
                    }
                }
                return 1;
            }
            if (pid == 0) {
                for (int k = 1; k < p; k++)
                    close(mpi_ctx.local_fds[k]);
                close(sv[0]);
                mpi_ctx.proc_idx = p;
                mpi_ctx.coord_fd = sv[1];
                break;
            }
            close(sv[1]);
            mpi_ctx.local_fds[p] = sv[0];
            mpi_pids[p] = pid;
        }
        pin_proc_to_cpu_slice(mpi_ctx.proc_idx, mpi_ctx.num_procs);
        char slice_env[32];
        snprintf(slice_env, sizeof(slice_env), "%d/%d", mpi_ctx.proc_idx, mpi_ctx.num_procs);
        setenv("VXFEC_NCCL_DEV_SLICE", slice_env, 1);
    }

    const char *so_path = resolve_nccl_plugin_path();
    void *dl = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", so_path, dlerror());
        if (mpi_ctx.num_procs > 1) {
            if (mpi_ctx.proc_idx > 0) {
                close(mpi_ctx.coord_fd);
                _exit(1);
            }
            for (int p = 1; p < mpi_ctx.num_procs; p++) {
                if (mpi_ctx.local_fds[p] >= 0)
                    close(mpi_ctx.local_fds[p]);
                if (mpi_pids[p] > 0) {
                    (void)kill(mpi_pids[p], SIGTERM);
                    (void)waitpid(mpi_pids[p], NULL, 0);
                }
            }
        }
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
    set_asym_fn set_asym = (set_asym_fn)dlsym(dl, "vxfec_nccl_set_asymmetric_link_sim");
    g_bench_get_rail_stats = dlsym(dl, "vxfec_nccl_get_rail_stats");
    set_codec_fn set_codec = (set_codec_fn)dlsym(dl, "vxfec_nccl_set_codec_override");
    if (!net || !get_stats || !reset_stats || !set_fault || !set_cong || !set_codec || net->init(silent_logger) != ncclSuccess) {
        dlclose(dl);
        if (mpi_ctx.num_procs > 1) {
            if (mpi_ctx.proc_idx > 0) {
                close(mpi_ctx.coord_fd);
                _exit(1);
            }
            for (int p = 1; p < mpi_ctx.num_procs; p++) {
                if (mpi_ctx.local_fds[p] >= 0)
                    close(mpi_ctx.local_fds[p]);
                if (mpi_pids[p] > 0) {
                    (void)kill(mpi_pids[p], SIGTERM);
                    (void)waitpid(mpi_pids[p], NULL, 0);
                }
            }
        }
        return 1;
    }

    if (rank >= 0 && master_spec) {
        int rc = run_multinode_main(net, get_stats, reset_stats, set_fault, set_codec, rank, &mpi_ctx, master_spec, csv_path,
                                    single_codec, single_label, single_bytes, single_loss_ppm, single_burst ? single_burst : 1u,
                                    cli_iters, use_gdr);
        dlclose(dl);
        if (mpi_ctx.num_procs > 1) {
            if (mpi_ctx.proc_idx > 0) {
                close(mpi_ctx.coord_fd);
                _exit(rc);
            }
            for (int p = 1; p < mpi_ctx.num_procs; p++) {
                close(mpi_ctx.local_fds[p]);
                int st = 0;
                if (waitpid(mpi_pids[p], &st, 0) < 0 || !WIFEXITED(st) || WEXITSTATUS(st) != 0)
                    rc = 1;
            }
        }
        return rc;
    }

    if (leo_bench) {
        int leo_iters = (cli_iters > 0) ? cli_iters : 30;
        bool lok = run_leo_handoff_bench(net, get_stats, reset_stats, set_fault, set_cong, set_asym, set_delay, set_codec,
                                         leo_iters, true);
        dlclose(dl);
        return lok ? 0 : 1;
    }

    int iters_small = (has_iters_arg && cli_iters >= 10) ? cli_iters : 100;
    int iters_large = iters_small;
    const char *env_it = getenv("VXFEC_BENCH_ITERS");
    if (!has_iters_arg && env_it && atoi(env_it) >= 10) {
        iters_small = atoi(env_it);
        iters_large = iters_small;
    }

    ncclNetProperties_v8_t props;
    (void)net->getProperties(0, &props);

    printf("======================================================================================================================="
           "\n");
    printf("vxfec NCCL Net v8 Plugin Benchmark (`%s`, Single-Host Loopback, N=%d/%d Samples; * = max when N<100)\n", net->name,
           iters_small, iters_large);
    printf("======================================================================================================================="
           "\n");
    printf("  Scenario (Topology / Policy)   |   p50 Lat |   p99 Lat  | Alg / BusBW | Agg Goodput | WireTax | Drops |  RS8/RQ   | "
           "Clamp\n");
    printf(
        "---------------------------------+-----------+------------+-------------+-------------+---------+-------+-----------+---"
        "----\n");

    const size_t sz_small = 512 * 1024;             /* 512 KiB -> 128 KiB/chunk (K=103) */
    const size_t sz_large = 2 * 1024 * 1024;        /* 2 MiB   -> 512 KiB/chunk (K=128 @ T=4096) */
    const size_t sz_xl = 16ULL * 1024ULL * 1024ULL; /* 16 MiB  -> 4.0 MiB/chunk (4x 1 MiB RS8 windows) */
    int iters_xl = iters_large;
    const char *env_xl = getenv("VXFEC_BENCH_XL_ITERS");
    if (env_xl && atoi(env_xl) >= 1)
        iters_xl = atoi(env_xl);

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
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 2.0% Burst=6 (RQ-GF256)", "rq-gf256",
                      sz_small, 20000, 6, true, 0, 0, iters_small) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "512 KiB | 2.0% Burst=6 (RQ-GF2)", "rq-gf2",
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
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "2.0 MiB | 2.0% Burst=8 (RQ-GF256)", "rq-gf256",
                      sz_large, 20000, 8, true, 0, 0, iters_large) ||
        !run_scenario(net, get_stats, reset_stats, set_fault, set_cong, set_codec, "2.0 MiB | 2.0% Burst=8 (RQ-GF2)", "rq-gf2",
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

    if (set_asym) {
        int leo_iters = (iters_xl < 25) ? iters_xl : 25;
        if (!run_leo_handoff_bench(net, get_stats, reset_stats, set_fault, set_cong, set_asym, set_delay, set_codec, leo_iters,
                                   false)) {
            dlclose(dl);
            return 1;
        }
    }

    printf("===========================================================================================================\n");
    set_codec(NULL);
    dlclose(dl);
    return 0;
}
