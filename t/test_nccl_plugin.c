#define _GNU_SOURCE
#include "../plugins/nccl/nccl_net_v8.h"
#include "../plugins/nccl/nccl_net_vxfec.h"
#include "../transport/harq.h"
#include "../transport/ibv_transport.h"
#include "nanorq_core.h"
#include "vxfec_rs_cauchy.h"
#include "vxfec_vk.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef VXFEC_HAVE_IBVERBS
#include <infiniband/verbs.h>
#endif
#include <vulkan/vulkan.h>

#define NUM_RANKS 4

/* simulate kernel PMTU discovery and enforcement while preserving sanitizer hooks */
typedef int (*test_getsockopt_fn)(int, int, int, void *, socklen_t *);
typedef int (*test_sendmmsg_fn)(int, struct mmsghdr *, unsigned int, int);

extern int __interceptor_getsockopt(int, int, int, void *, socklen_t *) __attribute__((weak));
extern int __interceptor_sendmmsg(int, struct mmsghdr *, unsigned int, int) __attribute__((weak));

static test_getsockopt_fn test_real_getsockopt_fn;
static test_sendmmsg_fn test_real_sendmmsg_fn;
static pthread_once_t test_sock_once = PTHREAD_ONCE_INIT;

static void load_test_sock_symbols(void)
{
    test_real_getsockopt_fn =
        __interceptor_getsockopt ? __interceptor_getsockopt : (test_getsockopt_fn)dlsym(RTLD_NEXT, "getsockopt");
    test_real_sendmmsg_fn = __interceptor_sendmmsg ? __interceptor_sendmmsg : (test_sendmmsg_fn)dlsym(RTLD_NEXT, "sendmmsg");
}

static int test_real_sendmmsg(int fd, struct mmsghdr *msgs, unsigned int count, int flags)
{
    pthread_once(&test_sock_once, load_test_sock_symbols);
    return test_real_sendmmsg_fn(fd, msgs, count, flags);
}

static int test_path_mtu;
static int test_wire_mtu;
static unsigned test_mtu_rejections;
static int test_rq_drop_tag;
static unsigned test_rq_drop_cnt;
static int test_rq_send_with_loss(int fd, struct mmsghdr *msgs, unsigned count, int flags);

__attribute__((visibility("default"))) int getsockopt(int fd, int level, int opt, void *value, socklen_t *len)
{
    pthread_once(&test_sock_once, load_test_sock_symbols);
    if (test_path_mtu && level == IPPROTO_IP && opt == IP_MTU && value && len && *len >= sizeof(int)) {
        memcpy(value, &test_path_mtu, sizeof(int));
        *len = sizeof(int);
        return 0;
    }
    return test_real_getsockopt_fn(fd, level, opt, value, len);
}

__attribute__((visibility("default"))) int sendmmsg(int fd, struct mmsghdr *msgs, unsigned count, int flags)
{
    if (test_rq_drop_tag)
        return test_rq_send_with_loss(fd, msgs, count, flags);
    int mtu_cap = test_wire_mtu ? test_wire_mtu : test_path_mtu;
    if (mtu_cap && count) {
        size_t bytes = 28u;
        for (size_t i = 0; i < msgs[0].msg_hdr.msg_iovlen; i++)
            bytes += msgs[0].msg_hdr.msg_iov[i].iov_len;
        if (bytes > (size_t)mtu_cap) {
            test_mtu_rejections++;
            errno = EMSGSIZE;
            return -1;
        }
    }
    return test_real_sendmmsg(fd, msgs, count, flags);
}

static void dummy_nccl_logger(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...)
{
    (void)level;
    (void)flags;
    (void)file;
    (void)line;
    (void)fmt;
}

static bool test_iid_parity_budget(void)
{
    /* reference budgets from binomial survival over K + P packets */
    static const struct {
        uint32_t K;
        double loss;
        uint32_t rank_slack;
        uint32_t cap;
        uint32_t expected;
    } cases[] = {
        {56403u, 0.0, 8u, UINT32_MAX, 0u},  {103u, 9.71e-6, 0u, 153u, 0u},
        {103u, 9.72e-6, 0u, 153u, 1u},      {103u, 0.002, 0u, 153u, 3u},       /* 2 packets give 0.1288% failure rate */
        {200u, 0.30, 0u, 56u, 56u},                                            /* RS8 clamps at 256 - K */
        {410u, 0.02, 2u, UINT32_MAX, 21u},  {410u, 0.10, 8u, UINT32_MAX, 78u}, /* rank overhead is also exposed to drops */
        {56403u, 5e-7, 0u, UINT32_MAX, 1u}, {56403u, 0.90, 8u, UINT32_MAX, 514697u},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint32_t got = harq_iid_parity(cases[i].K, cases[i].loss, cases[i].rank_slack, cases[i].cap);
        if (got != cases[i].expected) {
            fprintf(stderr, "FAIL: IID parity K=%u p=%g slack=%u: expected %u, got %u\n", cases[i].K, cases[i].loss,
                    cases[i].rank_slack, cases[i].expected, got);
            return false;
        }
    }

    uint32_t clamped = 0u;
    if (harq_paced_iid_parity(410u, 0.002, 2u, UINT32_MAX, 0.25, &clamped) != 4u || clamped != 3u ||
        harq_paced_iid_parity(410u, 0.002, 2u, UINT32_MAX, 0.45, &clamped) != 0u || clamped != 7u) {
        fprintf(stderr, "FAIL: IID parity congestion clamp\n");
        return false;
    }
    return true;
}

static bool test_harq_math(void)
{
    if (!test_iid_parity_budget())
        return false;

    double ewma = harq_ewma_update(0.0, 200, 4, 0.25);
    if (ewma < 0.019 || ewma > 0.021)
        return false;
    for (int i = 0; i < 64; i++)
        ewma = harq_ewma_update(ewma, 200, 0, 0.25);
    if (ewma != 0.0)
        return false;

    harq_decision_t d0 = harq_plan_feedback(0, 0, 2, 8);
    harq_decision_t d_sack = harq_plan_feedback(2, 0, 2, 0);
    harq_decision_t d_sack_partial = harq_plan_feedback(1, 2, 2, 8);
    harq_decision_t d_ir_rs8 = harq_plan_feedback(6, 2, 2, 0);
    harq_decision_t d_mds = harq_plan_feedback(6, 6, 2, 0);
    harq_decision_t d_rq = harq_plan_feedback(6, 14, 2, 8);
    if (d0.action != HARQ_COMPLETE || d_sack.action != HARQ_SACK_RETX || d_sack.request_count != 2u ||
        d_sack_partial.action != HARQ_SACK_RETX || d_sack_partial.request_count != 1u || d_ir_rs8.action != HARQ_IR_PARITY ||
        d_ir_rs8.request_count != 4u || d_mds.action != HARQ_DECODE_NOW || d_rq.action != HARQ_DECODE_NOW)
        return false;

    int32_t cur[4] = {0, 0, 0, 0};
    const int32_t weights[4] = {4, 2, 1, 1};
    uint32_t hits[4] = {0, 0, 0, 0};
    for (int i = 0; i < 16; i++) {
        uint32_t p = swrr_next(cur, weights, 4);
        if (p >= 4u)
            return false;
        hits[p]++;
    }
    if (hits[0] != 8u || hits[1] != 4u || hits[2] != 2u || hits[3] != 2u)
        return false;

    /* Test congestion-window pacer, ECN/delay discriminator, and port weight adaptation */
    harq_pacer_t pacer;
    harq_pacer_init(&pacer, 512 * 1024, 32 * 1024, 4 * 1024 * 1024, 10 * 1000 * 1000, 1000ULL * 1000 * 1000, 50000ULL, 1000ULL);

    /* Non-congestive ACK (flat 20 us RTT, 0 ECN): cwnd grows, cong_level stays 0 */
    if (harq_pacer_on_ack(&pacer, 128 * 1024, 20000ULL, 100, 0))
        return false;
    if (pacer.cwnd_bytes <= 512 * 1024 || pacer.cong_level != 0.0)
        return false;

    /* delay-only spike without ECN backs off cwnd/rate but must NOT spike cong_level or clamp P_0 */
    if (!harq_pacer_on_ack(&pacer, 128 * 1024, 150000ULL, 100, 0) || pacer.cong_level != 0.0)
        return false;

    /* heavy ECN congestion ACK: cwnd & rate back off, proactive parity gets clamped */
    uint64_t pre_cwnd = pacer.cwnd_bytes;
    uint64_t pre_rate = pacer.rate_Bps;
    for (int i = 0; i < 4; i++) {
        if (!harq_pacer_on_ack(&pacer, 128 * 1024, 120000ULL, 100, 80))
            return false;
    }
    if (pacer.cwnd_bytes >= pre_cwnd || pacer.rate_Bps >= pre_rate || pacer.cong_level < 0.45)
        return false;

    uint32_t clamped = 0;
    uint32_t p_clamped = harq_paced_iid_parity(256, 0.02, 2u, UINT32_MAX, pacer.cong_level, &clamped);
    if (p_clamped != 0u || clamped == 0u)
        return false;

    /* min_rtt_ns refreshes after HARQ_MIN_RTT_WINDOW_NS so stale spin-poll minimums expire */
    (void)harq_pacer_budget(&pacer, 0, 1000ULL + HARQ_MIN_RTT_WINDOW_NS + 5000ULL);
    (void)harq_pacer_on_ack(&pacer, 64 * 1024, 45000ULL, 64, 0);
    if (pacer.min_rtt_ns != 45000ULL)
        return false;

    /* sub-microsecond token refill must not lose fractional bytes */
    pacer.tokens_bytes = 0.0;
    pacer.rate_Bps = 100ULL * 1000 * 1000; /* 0.1 B/ns -> 10 B per 100 ns */
    pacer.last_refill_ns = 1000ULL;
    for (uint64_t step = 1; step <= 200; step++)
        (void)harq_pacer_budget(&pacer, 0, 1000ULL + step * 100ULL);
    if (harq_pacer_budget(&pacer, 0, 21000ULL) < 1990u)
        return false;
    /* requests must honor cwnd_bytes when inflight >= cwnd */
    if (harq_pacer_budget(&pacer, pacer.cwnd_bytes, 22000ULL) != 0u)
        return false;

    /* low configured rate caps (< 32 Mbps / < 8 Mbps) must not be inflated by default floors */
    harq_pacer_t low_pacer;
    harq_pacer_init(&low_pacer, 64 * 1024, 32 * 1024, 256 * 1024, 125000ULL, 500000ULL, 50000ULL, 1000ULL);
    if (low_pacer.max_rate_Bps != 500000ULL || low_pacer.min_rate_Bps != 125000ULL || low_pacer.rate_Bps != 500000ULL)
        return false;

    /* Per-port SWRR weight deflection and recovery */
    int32_t pw[4] = {16, 16, 16, 16};
    for (int i = 0; i < 4; i++) {
        if (!harq_update_port_weights(pw, 4, 0x1u, 16))
            return false;
    }
    if (pw[0] != 1 || pw[1] != 16)
        return false;
    bool nonunif = true;
    for (int i = 0; i < 16; i++)
        nonunif = harq_update_port_weights(pw, 4, 0x0u, 16);
    return (!nonunif && pw[0] == 16);
}

static bool test_ifname_filter_syntax(void)
{
    /* prefix include list */
    if (!vxfec_match_ifname("eth0", "eth0,eth1") || !vxfec_match_ifname("eth0:1", "eth0,eth1") ||
        !vxfec_match_ifname("eth1", "eth0,eth1") || vxfec_match_ifname("eth2", "eth0,eth1") ||
        vxfec_match_ifname("ens3", "eth0,eth1") || vxfec_match_ifname("lo", "eth0,eth1")) {
        fprintf(stderr, "FAIL: prefix include list matching\n");
        return false;
    }

    /* exact '=' match applies across the entire comma-separated list */
    if (!vxfec_match_ifname("eth", "=eth") || vxfec_match_ifname("eth0", "=eth") || !vxfec_match_ifname("eth0", "=eth0,eth1") ||
        !vxfec_match_ifname("eth1", "=eth0,eth1") || vxfec_match_ifname("eth01", "=eth0,eth1") ||
        vxfec_match_ifname("eth10", "=eth0,eth1") || !vxfec_match_ifname("eth1", "=eth0,=eth1")) {
        fprintf(stderr, "FAIL: exact '=' whole-list interface matching\n");
        return false;
    }

    /* NCCL leading '^' whole-list negation (excludes all listed entries plus loopback) */
    if (vxfec_match_ifname("docker0", "^docker0,virbr0") || vxfec_match_ifname("docker01", "^docker0,virbr0") ||
        vxfec_match_ifname("virbr0", "^docker0,virbr0") || vxfec_match_ifname("lo", "^docker0,virbr0") ||
        !vxfec_match_ifname("eth0", "^docker0,virbr0") || !vxfec_match_ifname("enp1s0", "^docker0,virbr0")) {
        fprintf(stderr, "FAIL: leading '^' negated list matching\n");
        return false;
    }

    /* leading '^=' / '=^' applies exact negated matching to every listed entry */
    if (vxfec_match_ifname("docker0", "^=docker0,virbr0") || vxfec_match_ifname("virbr0", "^=docker0,virbr0") ||
        vxfec_match_ifname("lo", "^=docker0,virbr0") || !vxfec_match_ifname("docker01", "^=docker0,virbr0") ||
        !vxfec_match_ifname("virbr01", "^=docker0,virbr0") || vxfec_match_ifname("virbr0", "=^docker0,virbr0") ||
        !vxfec_match_ifname("virbr01", "=^docker0,virbr0")) {
        fprintf(stderr, "FAIL: negated exact whole-list matching\n");
        return false;
    }

    /* positive prefix with specific per-token '^' exclusion */
    if (!vxfec_match_ifname("eth1", "eth,^=eth0") || !vxfec_match_ifname("eth01", "eth,^=eth0") ||
        vxfec_match_ifname("eth0", "eth,^=eth0") || vxfec_match_ifname("ens3", "eth,^=eth0")) {
        fprintf(stderr, "FAIL: mixed include/exclude interface matching\n");
        return false;
    }

    /* explicit loopback selection and empty/degenerate filters */
    if (!vxfec_match_ifname("lo", "lo") || !vxfec_match_ifname("lo", "=lo") || vxfec_match_ifname("eth0", "=lo") ||
        vxfec_match_ifname("eth0", "") || vxfec_match_ifname("eth0", " , ") || vxfec_match_ifname("eth0", "^") ||
        vxfec_match_ifname("eth0", "^=")) {
        fprintf(stderr, "FAIL: loopback/empty interface filter cases\n");
        return false;
    }

    return true;
}

typedef void (*test_get_stats_v1_fn)(vxfec_nccl_stats_t *);
typedef void (*test_get_stats_ex_fn)(vxfec_nccl_stats_t *, size_t);

static test_get_stats_ex_fn g_test_get_stats_ex = NULL;

static void test_query_stats(vxfec_nccl_stats_t *out)
{
    memset(out, 0, sizeof(*out));
    if (g_test_get_stats_ex)
        g_test_get_stats_ex(out, sizeof(*out));
}

static bool test_stats_abi_versioning(test_get_stats_v1_fn get_v1, test_get_stats_ex_fn get_ex, void (*reset_stats)(void))
{
    if (!get_v1 || !get_ex || !reset_stats)
        return false;
    reset_stats();

    uint8_t buf[sizeof(vxfec_nccl_stats_t) + 32u];
    memset(buf, 0xa5, sizeof(buf));
    get_v1((vxfec_nccl_stats_t *)buf);
    for (size_t i = 0; i < VXFEC_NCCL_STATS_V1_BYTES; i++) {
        if (buf[i] != 0u) {
            fprintf(stderr, "FAIL: v1 get_stats expected zeroed byte at %zu\n", i);
            return false;
        }
    }
    for (size_t i = VXFEC_NCCL_STATS_V1_BYTES; i < sizeof(buf); i++) {
        if (buf[i] != 0xa5u) {
            fprintf(stderr, "FAIL: v1 get_stats overflowed past %zu bytes at byte %zu\n", (size_t)VXFEC_NCCL_STATS_V1_BYTES, i);
            return false;
        }
    }

    memset(buf, 0x5a, sizeof(buf));
    get_ex((vxfec_nccl_stats_t *)buf, 3u * sizeof(uint64_t));
    for (size_t i = 0; i < 3u * sizeof(uint64_t); i++) {
        if (buf[i] != 0u)
            return false;
    }
    for (size_t i = 3u * sizeof(uint64_t); i < sizeof(buf); i++) {
        if (buf[i] != 0x5au)
            return false;
    }

    memset(buf, 0x3c, sizeof(buf));
    get_ex((vxfec_nccl_stats_t *)buf, sizeof(vxfec_nccl_stats_t));
    for (size_t i = 0; i < sizeof(vxfec_nccl_stats_t); i++) {
        if (buf[i] != 0u)
            return false;
    }
    for (size_t i = sizeof(vxfec_nccl_stats_t); i < sizeof(buf); i++) {
        if (buf[i] != 0x3cu)
            return false;
    }

    printf("PASS: Versioned stats ABI (v1 136-byte canary intact, v2 get_stats_ex size-bounded)\n");
    return true;
}

typedef struct {
    void *listen_comm[NUM_RANKS];
    void *send_comm[NUM_RANKS];
    void *recv_comm[NUM_RANKS];
} ring_comms_t;

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
            if (!ring->send_comm[r]) {
                if (net->connect(0, handles[dst], &ring->send_comm[r], NULL) != ncclSuccess)
                    return false;
            }
            if (!ring->recv_comm[dst]) {
                if (net->accept(ring->listen_comm[dst], &ring->recv_comm[dst], NULL) != ncclSuccess)
                    return false;
            }
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

static bool test_path_mtu_limits(const ncclNet_v8_t *net)
{
    const int bytes = 1024 * 1024;
    unsigned char *src = malloc((size_t)bytes), *dst = calloc(1, (size_t)bytes);
    ring_comms_t ring = {0};
    bool ok = false;
    if (!src || !dst)
        goto out;
    for (int i = 0; i < bytes; i++)
        src[i] = (unsigned char)(i * 37 + (i >> 12) + 1);

    test_path_mtu = 1500;
    test_mtu_rejections = 0;
    if (!setup_ring(net, &ring))
        goto out;

    void *sq = NULL, *rq = NULL, *rp = dst;
    int size = bytes, tag = 0, sdone = 0, rdone = 0, received = 0;
    if (net->irecv(ring.recv_comm[1], 1, &rp, &size, &tag, NULL, &rq) != ncclSuccess || !rq ||
        net->isend(ring.send_comm[0], src, bytes, tag, NULL, &sq) != ncclSuccess || !sq)
        goto out;

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (!sdone || !rdone) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec - start.tv_sec > 3 || (!sdone && net->test(sq, &sdone, NULL) != ncclSuccess) ||
            (!rdone && net->test(rq, &rdone, &received) != ncclSuccess))
            goto out;
    }
    if (test_mtu_rejections || received != bytes || memcmp(src, dst, (size_t)bytes) != 0)
        goto out;

    /* route or IP-option shrink after connect surfaces as error and steps sym_T down */
    test_wire_mtu = 1480;
    sq = NULL;
    sdone = 0;
    if (net->isend(ring.send_comm[0], src, bytes, tag + 1, NULL, &sq) != ncclSuccess || !sq)
        goto out;
    if (net->test(sq, &sdone, NULL) != ncclSystemError || test_mtu_rejections != 1)
        goto out;

    /* next transfer on another tag succeeds with the clamped symbol limit */
    memset(dst, 0, (size_t)bytes);
    sq = rq = NULL;
    sdone = rdone = received = 0;
    int tag2 = tag + 2;
    if (net->irecv(ring.recv_comm[1], 1, &rp, &size, &tag2, NULL, &rq) != ncclSuccess || !rq ||
        net->isend(ring.send_comm[0], src, bytes, tag2, NULL, &sq) != ncclSuccess || !sq)
        goto out;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (!sdone || !rdone) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec - start.tv_sec > 3 || (!sdone && net->test(sq, &sdone, NULL) != ncclSuccess) ||
            (!rdone && net->test(rq, &rdone, &received) != ncclSuccess))
            goto out;
    }
    ok = test_mtu_rejections == 1 && received == bytes && memcmp(src, dst, (size_t)bytes) == 0;

out:
    test_wire_mtu = 0;
    test_path_mtu = 0;
    teardown_ring(net, &ring);
    free(dst);
    free(src);
    if (ok)
        printf("PASS: 1 MiB transfer respects 1500-byte PMTU and reports a later shrink\n");
    else
        fprintf(stderr, "FAIL: PMTU sizing or EMSGSIZE propagation\n");
    return ok;
}

static bool run_host_ring_allreduce(const ncclNet_v8_t *net, ring_comms_t *ring, size_t elems_per_chunk, uint32_t seed_tag)
{
    size_t chunk_bytes = elems_per_chunk * sizeof(uint32_t);
    size_t total_elems = elems_per_chunk * NUM_RANKS;
    size_t total_bytes = total_elems * sizeof(uint32_t);
    size_t alloc_chunk = (chunk_bytes + 4095u) & ~(size_t)4095u;
    size_t alloc_total = (total_bytes + 4095u) & ~(size_t)4095u;

    uint32_t *bufs[NUM_RANKS] = {0};
    uint32_t *scratch[NUM_RANKS] = {0};
    void *mr_buf[NUM_RANKS] = {0};
    void *mr_scr[NUM_RANKS] = {0};
    bool ok = true;

    for (int r = 0; r < NUM_RANKS; r++) {
        if (posix_memalign((void **)&bufs[r], 4096, alloc_total) != 0 ||
            posix_memalign((void **)&scratch[r], 4096, alloc_chunk) != 0) {
            ok = false;
            break;
        }
        for (size_t i = 0; i < total_elems; i++)
            bufs[r][i] = (uint32_t)((r + 1) * 1000u + (i * 17u + seed_tag));
        if (net->regMr(ring->send_comm[r], bufs[r], alloc_total, NCCL_PTR_HOST, &mr_buf[r]) != ncclSuccess ||
            net->regMr(ring->recv_comm[r], scratch[r], alloc_chunk, NCCL_PTR_HOST, &mr_scr[r]) != ncclSuccess) {
            ok = false;
            break;
        }
    }

    for (int step = 0; step < NUM_RANKS - 1 && ok; step++) {
        void *tx_reqs[NUM_RANKS] = {0};
        void *rx_reqs[NUM_RANKS] = {0};
        int tag = (int)(seed_tag * 100u + (uint32_t)step + 1u);

        for (int r = 0; r < NUM_RANKS; r++) {
            int tx_chunk = (r - step + NUM_RANKS) % NUM_RANKS;
            void *tx_ptr = bufs[r] + (size_t)tx_chunk * elems_per_chunk;
            void *rx_ptr = scratch[r];
            int rx_sz = (int)chunk_bytes;
            int rx_tag = tag;

            if (net->irecv(ring->recv_comm[r], 1, &rx_ptr, &rx_sz, &rx_tag, &mr_scr[r], &rx_reqs[r]) != ncclSuccess ||
                net->isend(ring->send_comm[r], tx_ptr, (int)chunk_bytes, tag, mr_buf[r], &tx_reqs[r]) != ncclSuccess) {
                ok = false;
                break;
            }
        }

        bool all_done = false;
        while (ok && !all_done) {
            all_done = true;
            for (int r = 0; r < NUM_RANKS; r++) {
                if (tx_reqs[r]) {
                    int d = 0, sz = 0;
                    if (net->test(tx_reqs[r], &d, &sz) != ncclSuccess) {
                        ok = false;
                        break;
                    }
                    if (d)
                        tx_reqs[r] = NULL;
                    else
                        all_done = false;
                }
                if (rx_reqs[r]) {
                    int d = 0, sz = 0;
                    if (net->test(rx_reqs[r], &d, &sz) != ncclSuccess) {
                        ok = false;
                        break;
                    }
                    if (d) {
                        if ((size_t)sz != chunk_bytes) {
                            ok = false;
                            break;
                        }
                        rx_reqs[r] = NULL;
                    } else {
                        all_done = false;
                    }
                }
            }
        }

        if (ok) {
            for (int r = 0; r < NUM_RANKS; r++) {
                int rx_chunk = (r - step - 1 + 2 * NUM_RANKS) % NUM_RANKS;
                uint32_t *dst = bufs[r] + (size_t)rx_chunk * elems_per_chunk;
                for (size_t i = 0; i < elems_per_chunk; i++)
                    dst[i] += scratch[r][i];
            }
        }
    }

    for (int step = 0; step < NUM_RANKS - 1 && ok; step++) {
        void *tx_reqs[NUM_RANKS] = {0};
        void *rx_reqs[NUM_RANKS] = {0};
        int tag = (int)(seed_tag * 100u + 50u + (uint32_t)step);

        for (int r = 0; r < NUM_RANKS; r++) {
            int tx_chunk = (r + 1 - step + NUM_RANKS) % NUM_RANKS;
            int rx_chunk = (r - step + NUM_RANKS) % NUM_RANKS;
            void *tx_ptr = bufs[r] + (size_t)tx_chunk * elems_per_chunk;
            void *rx_ptr = bufs[r] + (size_t)rx_chunk * elems_per_chunk;
            int rx_sz = (int)chunk_bytes;
            int rx_tag = tag;

            if (net->irecv(ring->recv_comm[r], 1, &rx_ptr, &rx_sz, &rx_tag, &mr_buf[r], &rx_reqs[r]) != ncclSuccess ||
                net->isend(ring->send_comm[r], tx_ptr, (int)chunk_bytes, tag, mr_buf[r], &tx_reqs[r]) != ncclSuccess) {
                ok = false;
                break;
            }
        }

        bool all_done = false;
        while (ok && !all_done) {
            all_done = true;
            for (int r = 0; r < NUM_RANKS; r++) {
                if (tx_reqs[r]) {
                    int d = 0, sz = 0;
                    if (net->test(tx_reqs[r], &d, &sz) != ncclSuccess) {
                        ok = false;
                        break;
                    }
                    if (d)
                        tx_reqs[r] = NULL;
                    else
                        all_done = false;
                }
                if (rx_reqs[r]) {
                    int d = 0, sz = 0;
                    if (net->test(rx_reqs[r], &d, &sz) != ncclSuccess) {
                        ok = false;
                        break;
                    }
                    if (d)
                        rx_reqs[r] = NULL;
                    else
                        all_done = false;
                }
            }
        }
    }

    for (int r = 0; r < NUM_RANKS && ok; r++) {
        for (size_t i = 0; i < total_elems; i++) {
            uint32_t exp = 0;
            for (int k = 0; k < NUM_RANKS; k++)
                exp += (uint32_t)((k + 1) * 1000u + (i * 17u + seed_tag));
            if (bufs[r][i] != exp) {
                fprintf(stderr, "AllReduce mismatch rank=%d idx=%zu got=%u expected=%u\n", r, i, bufs[r][i], exp);
                ok = false;
                break;
            }
        }
    }

    for (int r = 0; r < NUM_RANKS; r++) {
        if (mr_scr[r])
            net->deregMr(ring->recv_comm[r], mr_scr[r]);
        if (mr_buf[r])
            net->deregMr(ring->send_comm[r], mr_buf[r]);
        free(scratch[r]);
        free(bufs[r]);
    }
    return ok;
}

static bool run_dmabuf_ring_allgather(const ncclNet_v8_t *net, const ncclNetProperties_v8_t *props, ring_comms_t *ring,
                                      size_t elems_per_chunk)
{
    vxfec_vk_ctx *app_vk = vxfec_vk_ctx_new();
    bool host_dmabuf_sup = app_vk && vxfec_vk_has_dmabuf(app_vk);
    if (!host_dmabuf_sup) {
        vxfec_vk_ctx_free(app_vk);
        return (props->ptrSupport & (NCCL_PTR_CUDA | NCCL_PTR_DMABUF)) == 0;
    }
    if (!vxfec_vk_has_dmabuf_export(app_vk)) {
        vxfec_vk_ctx_free(app_vk);
        return true;
    }

    /* mirror real NCCL GDR / DMA-BUF selection logic from topo.cc and net.cc */
    bool gdr_no_dmabuf = (props->ptrSupport & NCCL_PTR_CUDA) != 0;
    if (gdr_no_dmabuf) {
        void *tmp_mh = NULL;
        if (net->regMr(ring->send_comm[0], (void *)0x100000000ULL, 4096, NCCL_PTR_CUDA, &tmp_mh) != ncclSuccess) {
            fprintf(stderr, "FAIL: ptrSupport advertises NCCL_PTR_CUDA but regMr(NCCL_PTR_CUDA) fails\n");
            vxfec_vk_ctx_free(app_vk);
            return false;
        }
        net->deregMr(ring->send_comm[0], tmp_mh);
    }

    bool gdr_support = (props->ptrSupport & NCCL_PTR_CUDA) || (host_dmabuf_sup && (props->ptrSupport & NCCL_PTR_DMABUF));
    bool use_dmabuf = gdr_support && host_dmabuf_sup && (props->ptrSupport & NCCL_PTR_DMABUF);
    if (!gdr_support || !use_dmabuf || !net->regMrDmaBuf) {
        fprintf(stderr, "FAIL: real NCCL selection could not enable DMA-BUF (ptrSupport=0x%x)\n", props->ptrSupport);
        vxfec_vk_ctx_free(app_vk);
        return false;
    }

    size_t chunk_bytes = elems_per_chunk * sizeof(uint32_t);
    size_t total_elems = elems_per_chunk * NUM_RANKS;
    size_t total_bytes = total_elems * sizeof(uint32_t);

    vxfec_vk_buf *vram[NUM_RANKS] = {0};
    int fds[NUM_RANKS] = {-1, -1, -1, -1};
    void *dev_base[NUM_RANKS] = {0};
    void *mrs[NUM_RANKS] = {0};
    uint32_t *init_slice = (uint32_t *)malloc(chunk_bytes);
    uint32_t *verify_buf = (uint32_t *)malloc(total_bytes);
    bool ok = (init_slice != NULL && verify_buf != NULL);

    for (int r = 0; r < NUM_RANKS && ok; r++) {
        vram[r] = vxfec_vk_buf_new_exportable(app_vk, total_bytes, true, &fds[r]);
        if (!vram[r] || fds[r] < 0) {
            ok = false;
            break;
        }
        dev_base[r] = (void *)((uintptr_t)0x100000000ULL + (uintptr_t)r * total_bytes);

        for (size_t i = 0; i < elems_per_chunk; i++)
            init_slice[i] = (uint32_t)((r + 1) * 10000u + i * 31u + 7u);
        if (!vxfec_vk_buf_write_offset(app_vk, vram[r], (size_t)r * chunk_bytes, init_slice, chunk_bytes)) {
            ok = false;
            break;
        }
        ncclResult_t reg_res =
            use_dmabuf ? net->regMrDmaBuf(ring->send_comm[r], dev_base[r], total_bytes, NCCL_PTR_CUDA, 0, fds[r], &mrs[r])
                       : net->regMr(ring->send_comm[r], dev_base[r], total_bytes, NCCL_PTR_CUDA, &mrs[r]);
        if (reg_res != ncclSuccess) {
            ok = false;
            break;
        }
    }

    for (int step = 0; step < NUM_RANKS - 1 && ok; step++) {
        void *tx_reqs[NUM_RANKS] = {0};
        void *rx_reqs[NUM_RANKS] = {0};
        int tag = 900 + step;

        for (int r = 0; r < NUM_RANKS; r++) {
            int tx_chunk = (r - step + NUM_RANKS) % NUM_RANKS;
            int rx_chunk = (r - step - 1 + 2 * NUM_RANKS) % NUM_RANKS;
            void *tx_ptr = (uint8_t *)dev_base[r] + (size_t)tx_chunk * chunk_bytes;
            void *rx_ptr = (uint8_t *)dev_base[r] + (size_t)rx_chunk * chunk_bytes;
            int rx_sz = (int)chunk_bytes;
            int rx_tag = tag;

            if (net->irecv(ring->recv_comm[r], 1, &rx_ptr, &rx_sz, &rx_tag, &mrs[r], &rx_reqs[r]) != ncclSuccess ||
                net->isend(ring->send_comm[r], tx_ptr, (int)chunk_bytes, tag, mrs[r], &tx_reqs[r]) != ncclSuccess) {
                ok = false;
                break;
            }
        }

        bool all_done = false;
        while (ok && !all_done) {
            all_done = true;
            for (int r = 0; r < NUM_RANKS; r++) {
                if (tx_reqs[r]) {
                    int d = 0, sz = 0;
                    net->test(tx_reqs[r], &d, &sz);
                    if (d)
                        tx_reqs[r] = NULL;
                    else
                        all_done = false;
                }
                if (rx_reqs[r]) {
                    int d = 0, sz = 0;
                    net->test(rx_reqs[r], &d, &sz);
                    if (d) {
                        rx_reqs[r] = NULL;
                        void *rx_ptr = (uint8_t *)dev_base[r] + (size_t)((r - step - 1 + 2 * NUM_RANKS) % NUM_RANKS) * chunk_bytes;
                        int fsz = (int)chunk_bytes;
                        void *freq = NULL;
                        if (net->iflush(ring->recv_comm[r], 1, &rx_ptr, &fsz, &mrs[r], &freq) != ncclSuccess) {
                            ok = false;
                            break;
                        }
                        if (freq) {
                            int fd_done = 0;
                            while (!fd_done)
                                net->test(freq, &fd_done, NULL);
                        }
                    } else {
                        all_done = false;
                    }
                }
            }
        }
    }

    for (int r = 0; r < NUM_RANKS && ok; r++) {
        if (!vxfec_vk_buf_read_offset(app_vk, vram[r], 0, verify_buf, total_bytes)) {
            ok = false;
            break;
        }
        for (int c = 0; c < NUM_RANKS && ok; c++) {
            for (size_t i = 0; i < elems_per_chunk; i++) {
                uint32_t exp = (uint32_t)((c + 1) * 10000u + i * 31u + 7u);
                uint32_t got = verify_buf[(size_t)c * elems_per_chunk + i];
                if (got != exp) {
                    ok = false;
                    break;
                }
            }
        }
    }

    for (int r = 0; r < NUM_RANKS; r++) {
        if (mrs[r])
            net->deregMr(ring->send_comm[r], mrs[r]);
        if (fds[r] >= 0)
            close(fds[r]);
        vxfec_vk_buf_free(app_vk, vram[r]);
    }
    free(verify_buf);
    free(init_slice);
    vxfec_vk_ctx_free(app_vk);
    return ok;
}

static bool test_handshake_loss_recovery(const ncclNet_v8_t *net, void (*set_hs_fault)(uint32_t, uint32_t))
{
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
    void *lcomm = NULL, *scomm = NULL, *rcomm = NULL;
    if (net->listen(0, handle, &lcomm) != ncclSuccess)
        return false;

    /* drop initial 3 HELLO datagrams and initial 2 HELLO_ACK datagrams */
    set_hs_fault(3u, 2u);

    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    while (!scomm || !rcomm) {
        if (!scomm && net->connect(0, handle, &scomm, NULL) != ncclSuccess)
            goto fail;
        if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess)
            goto fail;
        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        uint64_t elapsed_ms =
            (uint64_t)(ts_now.tv_sec - ts_start.tv_sec) * 1000ULL + (uint64_t)(ts_now.tv_nsec - ts_start.tv_nsec) / 1000000ULL;
        if (elapsed_ms > 2000ULL) {
            fprintf(stderr, "FAIL: handshake timed out after dropped HELLO/HELLO_ACK\n");
            goto fail;
        }
    }
    set_hs_fault(0u, 0u);

    uint32_t tx_val = 0xdeadbeefu, rx_val = 0u;
    void *tx_mr = NULL, *rx_mr = NULL, *sreq = NULL, *rreq = NULL;
    void *rx_ptr = &rx_val;
    int rx_sz = (int)sizeof(rx_val), tag = 77;
    if (net->regMr(scomm, &tx_val, sizeof(tx_val), NCCL_PTR_HOST, &tx_mr) != ncclSuccess ||
        net->regMr(rcomm, &rx_val, sizeof(rx_val), NCCL_PTR_HOST, &rx_mr) != ncclSuccess ||
        net->irecv(rcomm, 1, &rx_ptr, &rx_sz, &tag, &rx_mr, &rreq) != ncclSuccess ||
        net->isend(scomm, &tx_val, (int)sizeof(tx_val), tag, tx_mr, &sreq) != ncclSuccess)
        goto fail_mr;

    int sdone = 0, rdone = 0, got_sz = 0;
    while (!sdone || !rdone) {
        if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
            goto fail_mr;
        if (!rdone && net->test(rreq, &rdone, &got_sz) != ncclSuccess)
            goto fail_mr;
    }

    net->deregMr(scomm, tx_mr);
    net->deregMr(rcomm, rx_mr);
    net->closeSend(scomm);
    net->closeRecv(rcomm);
    net->closeListen(lcomm);
    if (got_sz != (int)sizeof(rx_val) || rx_val != tx_val)
        return false;

    /* black-box wire relay test: discard actual first UDP HELLO datagram in flight */
    lcomm = scomm = rcomm = NULL;
    tx_mr = rx_mr = NULL;
    int relay_fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (relay_fd < 0)
        return false;
    struct sockaddr_in rsin;
    memset(&rsin, 0, sizeof(rsin));
    rsin.sin_family = AF_INET;
    rsin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(relay_fd, (struct sockaddr *)&rsin, sizeof(rsin)) != 0) {
        close(relay_fd);
        return false;
    }
    socklen_t rlen = sizeof(rsin);
    if (getsockname(relay_fd, (struct sockaddr *)&rsin, &rlen) != 0) {
        close(relay_fd);
        return false;
    }

    if (net->listen(0, handle, &lcomm) != ncclSuccess) {
        close(relay_fd);
        return false;
    }
    uint16_t real_ctrl_port_be = 0;
    memcpy(&real_ctrl_port_be, handle + 8, sizeof(uint16_t));
    memcpy(handle + 8, &rsin.sin_port, sizeof(uint16_t));

    struct sockaddr_in real_ctrl_addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = htonl(INADDR_LOOPBACK),
        .sin_port = real_ctrl_port_be,
    };
    struct sockaddr_in client_addr;
    memset(&client_addr, 0, sizeof(client_addr));
    uint32_t seen_hellos = 0;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    while (!scomm || !rcomm) {
        if (!scomm && net->connect(0, handle, &scomm, NULL) != ncclSuccess) {
            close(relay_fd);
            goto fail;
        }
        for (;;) {
            uint8_t pkt[256];
            struct sockaddr_in src;
            socklen_t slen = sizeof(src);
            ssize_t n = recvfrom(relay_fd, pkt, sizeof(pkt), MSG_DONTWAIT, (struct sockaddr *)&src, &slen);
            if (n <= 0)
                break;
            if (src.sin_port != real_ctrl_port_be) {
                client_addr = src;
                seen_hellos++;
                if (seen_hellos == 1u)
                    continue; /* discard first HELLO datagram */
                (void)sendto(relay_fd, pkt, (size_t)n, MSG_DONTWAIT, (struct sockaddr *)&real_ctrl_addr, sizeof(real_ctrl_addr));
            } else if (client_addr.sin_port != 0) {
                (void)sendto(relay_fd, pkt, (size_t)n, MSG_DONTWAIT, (struct sockaddr *)&client_addr, sizeof(client_addr));
            }
        }
        if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess) {
            close(relay_fd);
            goto fail;
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        uint64_t elapsed_ms =
            (uint64_t)(ts_now.tv_sec - ts_start.tv_sec) * 1000ULL + (uint64_t)(ts_now.tv_nsec - ts_start.tv_nsec) / 1000000ULL;
        if (elapsed_ms > 2000ULL) {
            fprintf(stderr, "FAIL: wire-relay handshake timed out after discarding first HELLO\n");
            close(relay_fd);
            goto fail;
        }
    }
    close(relay_fd);
    net->closeSend(scomm);
    net->closeRecv(rcomm);
    net->closeListen(lcomm);
    return seen_hellos >= 2u;

fail_mr:
    if (tx_mr)
        net->deregMr(scomm, tx_mr);
    if (rx_mr)
        net->deregMr(rcomm, rx_mr);
fail:
    set_hs_fault(0u, 0u);
    if (scomm)
        net->closeSend(scomm);
    if (rcomm)
        net->closeRecv(rcomm);
    if (lcomm)
        net->closeListen(lcomm);
    return false;
}

static bool test_recv_capacity_enforcement(const ncclNet_v8_t *net)
{
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
    void *lcomm = NULL, *scomm = NULL, *rcomm = NULL;
    if (net->listen(0, handle, &lcomm) != ncclSuccess)
        return false;
    while (!scomm || !rcomm) {
        if (!scomm && net->connect(0, handle, &scomm, NULL) != ncclSuccess)
            goto fail;
        if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess)
            goto fail;
    }

    uint8_t tx_buf[128];
    uint8_t rx_buf[256];
    for (size_t i = 0; i < sizeof(tx_buf); i++)
        tx_buf[i] = (uint8_t)(0x40u + i);

    void *tx_mr = NULL, *rx_mr = NULL;
    if (net->regMr(scomm, tx_buf, sizeof(tx_buf), NCCL_PTR_HOST, &tx_mr) != ncclSuccess ||
        net->regMr(rcomm, rx_buf, sizeof(rx_buf), NCCL_PTR_HOST, &rx_mr) != ncclSuccess)
        goto fail_mr;

    /* test both pre-posted irecv (direct placement path) and post-arrival irecv (staged path) */
    for (int mode = 0; mode < 2; mode++) {
        memset(rx_buf, 0xa5, sizeof(rx_buf));
        void *sreq = NULL, *rreq = NULL;
        void *rx_ptr = rx_buf;
        int rx_cap = 1;
        int tag = 300 + mode;

        if (mode == 0) {
            if (net->irecv(rcomm, 1, &rx_ptr, &rx_cap, &tag, &rx_mr, &rreq) != ncclSuccess ||
                net->isend(scomm, tx_buf, (int)sizeof(tx_buf), tag, tx_mr, &sreq) != ncclSuccess)
                goto fail_mr;
        } else {
            if (net->isend(scomm, tx_buf, (int)sizeof(tx_buf), tag, tx_mr, &sreq) != ncclSuccess)
                goto fail_mr;
            int sdone_tmp = 0;
            (void)net->test(sreq, &sdone_tmp, NULL);
            if (net->irecv(rcomm, 1, &rx_ptr, &rx_cap, &tag, &rx_mr, &rreq) != ncclSuccess)
                goto fail_mr;
        }

        bool rx_rejected = false;
        bool tx_finished = false;
        for (int spin = 0; spin < 200000 && (!rx_rejected || !tx_finished); spin++) {
            if (!tx_finished) {
                int sdone = 0;
                ncclResult_t sres = net->test(sreq, &sdone, NULL);
                if (sres != ncclSuccess || sdone)
                    tx_finished = true;
            }
            if (!rx_rejected) {
                int rdone = 0, got_sz = 0;
                ncclResult_t rres = net->test(rreq, &rdone, &got_sz);
                if (rres != ncclSuccess) {
                    if (rdone != 0)
                        goto fail_mr;
                    rx_rejected = true;
                } else if (rdone) {
                    fprintf(stderr, "FAIL: oversized 128B send into 1B irecv reported success (mode=%d)\n", mode);
                    goto fail_mr;
                }
            }
        }
        if (!rx_rejected) {
            fprintf(stderr, "FAIL: oversized receive did not fail cleanly (mode=%d)\n", mode);
            goto fail_mr;
        }
        for (size_t i = 1; i < sizeof(rx_buf); i++) {
            if (rx_buf[i] != 0xa5u) {
                fprintf(stderr, "FAIL: receive buffer overwritten past 1B capacity at byte %zu (mode=%d)\n", i, mode);
                goto fail_mr;
            }
        }
    }

    /* retire all later generations of rejected multi-gen sliced messages without leaking RX wire slots */
    const size_t multi_bad_sz = 8u * 1024u * 1024u; /* 8 gens per msg x 18 msgs = 144 gens > 128 RX wire slots */
    const size_t multi_ok_sz = 2u * 1024u * 1024u;
    uint8_t *big_tx = (uint8_t *)malloc(multi_bad_sz);
    uint8_t *big_rx = (uint8_t *)malloc(multi_ok_sz);
    void *big_tx_mr = NULL, *big_rx_mr = NULL;
    if (!big_tx || !big_rx || net->regMr(scomm, big_tx, multi_bad_sz, NCCL_PTR_HOST, &big_tx_mr) != ncclSuccess ||
        net->regMr(rcomm, big_rx, multi_ok_sz, NCCL_PTR_HOST, &big_rx_mr) != ncclSuccess) {
        if (big_tx_mr)
            net->deregMr(scomm, big_tx_mr);
        if (big_rx_mr)
            net->deregMr(rcomm, big_rx_mr);
        free(big_rx);
        free(big_tx);
        goto fail_mr;
    }
    for (size_t i = 0; i < multi_bad_sz; i++)
        big_tx[i] = (uint8_t)((i * 59u + 17u) & 0xffu);

    bool multi_ok = true;
    for (int it = 0; it < 18 && multi_ok; it++) {
        memset(rx_buf, 0xa5, sizeof(rx_buf));
        void *sreq = NULL, *rreq = NULL, *rx_ptr = rx_buf;
        int rx_cap = 1, tag = 320 + it;
        if (net->irecv(rcomm, 1, &rx_ptr, &rx_cap, &tag, &rx_mr, &rreq) != ncclSuccess ||
            net->isend(scomm, big_tx, (int)multi_bad_sz, tag, big_tx_mr, &sreq) != ncclSuccess) {
            multi_ok = false;
            break;
        }
        bool rx_rej = false, tx_fin = false;
        for (int spin = 0; spin < 200000 && (!rx_rej || !tx_fin); spin++) {
            if (!tx_fin) {
                int sdone = 0;
                ncclResult_t sres = net->test(sreq, &sdone, NULL);
                if (sres != ncclSuccess)
                    tx_fin = true;
                else if (sdone) {
                    multi_ok = false;
                    break;
                }
            }
            if (!rx_rej) {
                int rdone = 0, got_sz = 0;
                ncclResult_t rres = net->test(rreq, &rdone, &got_sz);
                if (rres != ncclSuccess && rdone == 0)
                    rx_rej = true;
                else if (rdone) {
                    multi_ok = false;
                    break;
                }
            }
        }
        if (!rx_rej || !tx_fin)
            multi_ok = false;
    }

    if (multi_ok) {
        memset(big_rx, 0, multi_ok_sz);
        void *sreq = NULL, *rreq = NULL, *rx_ptr = big_rx;
        int rx_cap = (int)multi_ok_sz, tag = 399, sdone = 0, rdone = 0, got_sz = 0;
        if (net->irecv(rcomm, 1, &rx_ptr, &rx_cap, &tag, &big_rx_mr, &rreq) != ncclSuccess ||
            net->isend(scomm, big_tx, (int)multi_ok_sz, tag, big_tx_mr, &sreq) != ncclSuccess) {
            multi_ok = false;
        } else {
            for (int spin = 0; spin < 200000 && (!sdone || !rdone); spin++) {
                if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess) {
                    multi_ok = false;
                    break;
                }
                if (!rdone && net->test(rreq, &rdone, &got_sz) != ncclSuccess) {
                    multi_ok = false;
                    break;
                }
            }
            if (!sdone || !rdone || (size_t)got_sz != multi_ok_sz || memcmp(big_tx, big_rx, multi_ok_sz) != 0)
                multi_ok = false;
        }
    }
    net->deregMr(scomm, big_tx_mr);
    net->deregMr(rcomm, big_rx_mr);
    free(big_rx);
    free(big_tx);
    if (!multi_ok) {
        fprintf(stderr, "FAIL: multi-generation rejected message leaked RX wire slots or hung sender\n");
        goto fail_mr;
    }

    net->deregMr(scomm, tx_mr);
    net->deregMr(rcomm, rx_mr);
    net->closeSend(scomm);
    net->closeRecv(rcomm);
    net->closeListen(lcomm);
    return true;

fail_mr:
    if (tx_mr)
        net->deregMr(scomm, tx_mr);
    if (rx_mr)
        net->deregMr(rcomm, rx_mr);
fail:
    if (scomm)
        net->closeSend(scomm);
    if (rcomm)
        net->closeRecv(rcomm);
    if (lcomm)
        net->closeListen(lcomm);
    return false;
}

static bool test_rs8_to_rq_promotion_integrity(const ncclNet_v8_t *net, void (*get_stats)(vxfec_nccl_stats_t *),
                                               void (*reset_stats)(void), void (*set_fault)(uint32_t, uint32_t, bool),
                                               void (*set_codec)(const char *))
{
    /*
     * force RS8 mode at K=240 (307,200 B at T=1280 -> only 16 RS8 parity slots 240..255)
     * and also test 81,920 B (K=64 at T=1280) under heavy burst loss so RS8 repair space
     * is exhausted after RS8 parity packets have already been buffered at RX.
     */
    const size_t sizes_to_test[2] = {240u * 1280u, 81920u};
    bool ok = false;
    set_codec("rs8");

    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
    void *lcomm = NULL, *scomm = NULL, *rcomm = NULL;
    if (net->listen(0, handle, &lcomm) != ncclSuccess)
        goto out;
    while (!scomm || !rcomm) {
        if (!scomm && net->connect(0, handle, &scomm, NULL) != ncclSuccess)
            goto out;
        if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess)
            goto out;
    }

    size_t max_sz = sizes_to_test[0];
    uint8_t *tx_buf = (uint8_t *)malloc(max_sz);
    uint8_t *rx_buf = (uint8_t *)malloc(max_sz);
    void *tx_mr = NULL, *rx_mr = NULL;
    if (!tx_buf || !rx_buf || net->regMr(scomm, tx_buf, max_sz, NCCL_PTR_HOST, &tx_mr) != ncclSuccess ||
        net->regMr(rcomm, rx_buf, max_sz, NCCL_PTR_HOST, &rx_mr) != ncclSuccess)
        goto out_bufs;

    reset_stats();
    for (int c = 0; c < 2; c++) {
        size_t sz = sizes_to_test[c];
        uint32_t loss_ppm = (c == 0) ? 140000u : 780000u;
        uint32_t burst = (c == 0) ? 22u : 16u;
        int iters = (c == 0) ? 8 : 6;
        set_fault(loss_ppm, burst, true);

        for (int it = 0; it < iters; it++) {
            for (size_t i = 0; i < sz; i++)
                tx_buf[i] = (uint8_t)((i * 131u + (size_t)it * 53u + (size_t)c * 19u + 7u) & 0xffu);
            memset(rx_buf, 0, sz);

            void *sreq = NULL, *rreq = NULL;
            void *rx_ptr = rx_buf;
            int rx_cap = (int)sz;
            int tag = 600 + c * 32 + it;
            if (net->irecv(rcomm, 1, &rx_ptr, &rx_cap, &tag, &rx_mr, &rreq) != ncclSuccess ||
                net->isend(scomm, tx_buf, (int)sz, tag, tx_mr, &sreq) != ncclSuccess)
                goto out_bufs;

            int sdone = 0, rdone = 0, got_sz = 0;
            while (!sdone || !rdone) {
                if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
                    goto out_bufs;
                if (!rdone && net->test(rreq, &rdone, &got_sz) != ncclSuccess)
                    goto out_bufs;
            }
            if ((size_t)got_sz != sz || memcmp(tx_buf, rx_buf, sz) != 0) {
                size_t bad = 0;
                for (size_t i = 0; i < sz; i++) {
                    if (tx_buf[i] != rx_buf[i])
                        bad++;
                }
                fprintf(stderr, "FAIL: RS8->RaptorQ promotion corrupted %zu / %zu bytes (case=%d iter=%d)\n", bad, sz, c, it);
                goto out_bufs;
            }
        }
    }

    vxfec_nccl_stats_t st;
    get_stats(&st);
    if (st.rx_rq_decodes == 0) {
        fprintf(stderr, "FAIL: RS8->RaptorQ promotion test did not trigger RaptorQ decode\n");
        goto out_bufs;
    }
    printf("PASS: RS8-to-RaptorQ mid-transfer promotion bit-exact (rs8_dec=%lu rq_dec=%lu)\n", (unsigned long)st.rx_rs8_decodes,
           (unsigned long)st.rx_rq_decodes);
    ok = true;

out_bufs:
    set_fault(0u, 1u, false);
    if (tx_mr)
        net->deregMr(scomm, tx_mr);
    if (rx_mr)
        net->deregMr(rcomm, rx_mr);
    free(rx_buf);
    free(tx_buf);
out:
    set_codec(NULL);
    if (scomm)
        net->closeSend(scomm);
    if (rcomm)
        net->closeRecv(rcomm);
    if (lcomm)
        net->closeListen(lcomm);
    return ok;
}

static bool test_large_512mib_transfer(const ncclNet_v8_t *net, void (*get_stats)(vxfec_nccl_stats_t *), void (*reset_stats)(void),
                                       void (*set_fault)(uint32_t, uint32_t, bool))
{
    const size_t sz = 512ULL * 1024ULL * 1024ULL; /* exact 512 MiB (65,536 * 8 KiB) */
    uint8_t *tx_buf = (uint8_t *)malloc(sz);
    uint8_t *rx_buf = (uint8_t *)malloc(sz);
    if (!tx_buf || !rx_buf) {
        free(rx_buf);
        free(tx_buf);
        return false;
    }

    /* fill 64-bit word pattern so every 8 KiB block across all 512 MiB is unique */
    uint64_t *tx_w = (uint64_t *)tx_buf;
    size_t n_words = sz / sizeof(uint64_t);
    for (size_t i = 0; i < n_words; i++)
        tx_w[i] = 0x9e3779b97f4a7c15ULL ^ (uint64_t)i * 0xbf58476d1ce4e5b9ULL;
    memset(rx_buf, 0x5a, sz);

    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
    void *lcomm = NULL, *scomm = NULL, *rcomm = NULL;
    void *tx_mr = NULL, *rx_mr = NULL;
    bool ok = false;

    /* disable generation slicing so all 512 MiB runs as a single unsliced K=65536 > K_max SACK message */
    if (setenv("VXFEC_NCCL_SLICE_KB", "0", 1) != 0)
        goto out;

    if (net->listen(0, handle, &lcomm) != ncclSuccess)
        goto out;
    while (!scomm || !rcomm) {
        if (!scomm && net->connect(0, handle, &scomm, NULL) != ncclSuccess)
            goto out;
        if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess)
            goto out;
    }

    if (net->regMr(scomm, tx_buf, sz, NCCL_PTR_HOST, &tx_mr) != ncclSuccess ||
        net->regMr(rcomm, rx_buf, sz, NCCL_PTR_HOST, &rx_mr) != ncclSuccess)
        goto out;

    /* inject sparse drops during 512 MiB transfer to verify SACK recovery at K=65536 */
    reset_stats();
    set_fault(400u, 2u, false);

    void *sreq = NULL, *rreq = NULL;
    void *rx_ptr = rx_buf;
    int rx_cap = (int)sz;
    int tag = 888;
    if (net->irecv(rcomm, 1, &rx_ptr, &rx_cap, &tag, &rx_mr, &rreq) != ncclSuccess ||
        net->isend(scomm, tx_buf, (int)sz, tag, tx_mr, &sreq) != ncclSuccess)
        goto out;

    int sdone = 0, rdone = 0, got_sz = 0;
    while (!sdone || !rdone) {
        if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
            goto out;
        if (!rdone && net->test(rreq, &rdone, &got_sz) != ncclSuccess)
            goto out;
    }

    vxfec_nccl_stats_t st;
    get_stats(&st);
    if ((size_t)got_sz != sz || memcmp(tx_buf, rx_buf, sz) != 0 || st.rx_fec_decodes != 0u || st.tx_sack_retx_pkts == 0u) {
        fprintf(stderr, "FAIL: 512 MiB unsliced K=65536 transfer failed (got_sz=%d fec=%lu sack=%lu)\n", got_sz,
                (unsigned long)st.rx_fec_decodes, (unsigned long)st.tx_sack_retx_pkts);
        goto out;
    }

    printf("PASS: 512 MiB unsliced (K=65536 > K_max) SACK transfer bit-exact (sack_retx=%lu)\n",
           (unsigned long)st.tx_sack_retx_pkts);
    ok = true;

out:
    unsetenv("VXFEC_NCCL_SLICE_KB");
    set_fault(0u, 1u, false);
    if (tx_mr)
        net->deregMr(scomm, tx_mr);
    if (rx_mr)
        net->deregMr(rcomm, rx_mr);
    if (scomm)
        net->closeSend(scomm);
    if (rcomm)
        net->closeRecv(rcomm);
    if (lcomm)
        net->closeListen(lcomm);
    free(rx_buf);
    free(tx_buf);
    return ok;
}

static bool test_paced_recovery_and_cwnd_bounds(const ncclNet_v8_t *net, void (*get_stats)(vxfec_nccl_stats_t *),
                                                void (*reset_stats)(void), void (*set_fault)(uint32_t, uint32_t, bool),
                                                void (*set_cong)(uint32_t, uint32_t, uint64_t), void (*set_codec)(const char *))
{
    const size_t sz = 2ULL * 1024ULL * 1024ULL; /* 2 MiB >> 64 KiB cwnd */
    uint8_t *tx_buf = (uint8_t *)malloc(sz);
    uint8_t *rx_buf = (uint8_t *)malloc(sz);
    if (!tx_buf || !rx_buf) {
        free(rx_buf);
        free(tx_buf);
        return false;
    }

    for (size_t i = 0; i < sz; i++)
        tx_buf[i] = (uint8_t)((i * 73u + (i >> 8) * 29u + 11u) & 0xffu);

    setenv("VXFEC_NCCL_CWND_KB", "64", 1);
    set_cong(500000u, 0u, 400u); /* 50% ECN + 400 Mbps rate cap */

    bool ok = false;
    const char *modes[2] = {"gbn", NULL};
    for (int m = 0; m < 2; m++) {
        set_codec(modes[m]);

        uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
        void *lcomm = NULL, *scomm = NULL, *rcomm = NULL;
        void *tx_mr = NULL, *rx_mr = NULL;
        if (net->listen(0, handle, &lcomm) != ncclSuccess)
            goto mode_fail;
        while (!scomm || !rcomm) {
            if (!scomm && net->connect(0, handle, &scomm, NULL) != ncclSuccess)
                goto mode_fail;
            if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess)
                goto mode_fail;
        }
        if (net->regMr(scomm, tx_buf, sz, NCCL_PTR_HOST, &tx_mr) != ncclSuccess ||
            net->regMr(rcomm, rx_buf, sz, NCCL_PTR_HOST, &rx_mr) != ncclSuccess)
            goto mode_fail;

        reset_stats();
        set_fault(40000u, 4u, true);
        vxfec_nccl_stats_t st;
        memset(&st, 0, sizeof(st));
        int got_sz = 0;
        for (int attempt = 0; attempt < 4; attempt++) {
            memset(rx_buf, 0, sz);
            void *sreq = NULL, *rreq = NULL;
            void *rx_ptr = rx_buf;
            int rx_cap = (int)sz;
            int tag = 940 + m * 10 + attempt;
            if (net->irecv(rcomm, 1, &rx_ptr, &rx_cap, &tag, &rx_mr, &rreq) != ncclSuccess ||
                net->isend(scomm, tx_buf, (int)sz, tag, tx_mr, &sreq) != ncclSuccess)
                goto mode_fail;

            int sdone = 0, rdone = 0;
            got_sz = 0;
            while (!sdone || !rdone) {
                if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
                    goto mode_fail;
                if (!rdone && net->test(rreq, &rdone, &got_sz) != ncclSuccess)
                    goto mode_fail;
            }
            get_stats(&st);
            if (st.tx_sack_retx_pkts > 0 || st.tx_harq_ir_pkts > 0)
                break;
        }
        bool mode_ok = ((size_t)got_sz == sz && memcmp(tx_buf, rx_buf, sz) == 0 && st.tx_paced_slices > 0 &&
                        (st.tx_sack_retx_pkts > 0 || st.tx_harq_ir_pkts > 0));

        net->deregMr(scomm, tx_mr);
        net->deregMr(rcomm, rx_mr);
        net->closeSend(scomm);
        net->closeRecv(rcomm);
        net->closeListen(lcomm);
        if (!mode_ok) {
            fprintf(stderr, "FAIL: paced recovery / cwnd bound test failed in mode=%s (slices=%lu retx=%lu ir=%lu)\n",
                    modes[m] ? modes[m] : "harq", (unsigned long)st.tx_paced_slices, (unsigned long)st.tx_sack_retx_pkts,
                    (unsigned long)st.tx_harq_ir_pkts);
            goto out;
        }
        continue;

    mode_fail:
        if (tx_mr)
            net->deregMr(scomm, tx_mr);
        if (rx_mr)
            net->deregMr(rcomm, rx_mr);
        if (scomm)
            net->closeSend(scomm);
        if (rcomm)
            net->closeRecv(rcomm);
        if (lcomm)
            net->closeListen(lcomm);
        goto out;
    }

    printf("PASS: Unified paced recovery (GBN + HARQ-IR/SACK) & strict 64 KiB cwnd streaming across 2 MiB\n");
    ok = true;

out:
    set_fault(0u, 1u, false);
    set_cong(0u, 0u, 0u);
    set_codec(NULL);
    unsetenv("VXFEC_NCCL_CWND_KB");
    free(rx_buf);
    free(tx_buf);
    return ok;
}

static bool test_gpu_rx_write_failure_propagation(const ncclNet_v8_t *net, const ncclNetProperties_v8_t *props,
                                                  void (*set_fault)(uint32_t, uint32_t, bool), void (*set_gpu_rx_fault)(uint32_t))
{
    if (!(props->ptrSupport & NCCL_PTR_DMABUF) || !net->regMrDmaBuf)
        return true;

    vxfec_vk_ctx *app_vk = vxfec_vk_ctx_new();
    if (!app_vk || !vxfec_vk_has_dmabuf(app_vk) || !vxfec_vk_has_dmabuf_export(app_vk)) {
        vxfec_vk_ctx_free(app_vk);
        return true;
    }

    const size_t sz = 16384u;
    int dmabuf_fd = -1;
    vxfec_vk_buf *rx_vram = vxfec_vk_buf_new_exportable(app_vk, sz, true, &dmabuf_fd);
    uint8_t *tx_host = (uint8_t *)malloc(sz);
    uint8_t *chk_host = (uint8_t *)malloc(sz);
    void *lcomm = NULL, *scomm = NULL, *rcomm = NULL;
    void *tx_mr = NULL, *rx_mr = NULL;
    bool ok = false;

    if (!rx_vram || dmabuf_fd < 0 || !tx_host || !chk_host)
        goto out;

    memset(chk_host, 0xaa, sz);
    if (!vxfec_vk_buf_write_offset(app_vk, rx_vram, 0, chk_host, sz) || !vxfec_vk_buf_release_external(app_vk, rx_vram))
        goto out;

    for (size_t i = 0; i < sz; i++)
        tx_host[i] = (uint8_t)((i * 43u + 9u) & 0xffu);

    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
    if (net->listen(0, handle, &lcomm) != ncclSuccess)
        goto out;
    while (!scomm || !rcomm) {
        if (!scomm && net->connect(0, handle, &scomm, NULL) != ncclSuccess)
            goto out;
        if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess)
            goto out;
    }

    void *dev_base = (void *)0x200000000ULL;
    if (net->regMr(scomm, tx_host, sz, NCCL_PTR_HOST, &tx_mr) != ncclSuccess ||
        net->regMrDmaBuf(rcomm, dev_base, sz, NCCL_PTR_CUDA, 0, dmabuf_fd, &rx_mr) != ncclSuccess)
        goto out;

    /* inject 1 failed GPU receive write: RX must fail and send PKT_ERR (not PKT_DONE) */
    set_fault(0u, 1u, false);
    set_gpu_rx_fault(1u);

    void *sreq = NULL, *rreq = NULL;
    void *rx_ptr = dev_base;
    int rx_cap = (int)sz, tag = 975;
    if (net->irecv(rcomm, 1, &rx_ptr, &rx_cap, &tag, &rx_mr, &rreq) != ncclSuccess ||
        net->isend(scomm, tx_host, (int)sz, tag, tx_mr, &sreq) != ncclSuccess)
        goto out;

    bool rx_errored = false, tx_errored = false;
    for (int spin = 0; spin < 200000 && (!rx_errored || !tx_errored); spin++) {
        if (!rx_errored) {
            int rdone = 0, got_sz = 0;
            ncclResult_t rres = net->test(rreq, &rdone, &got_sz);
            if (rres != ncclSuccess) {
                if (rdone != 0)
                    goto out;
                rx_errored = true;
            } else if (rdone) {
                fprintf(stderr, "FAIL: failed GPU receive write reported done=1 ncclSuccess\n");
                goto out;
            }
        }
        if (!tx_errored) {
            int sdone = 0;
            ncclResult_t sres = net->test(sreq, &sdone, NULL);
            if (sres != ncclSuccess) {
                if (sdone != 0)
                    goto out;
                tx_errored = true;
            } else if (sdone) {
                fprintf(stderr, "FAIL: sender received PKT_DONE despite failed GPU receive write\n");
                goto out;
            }
        }
    }
    if (!rx_errored || !tx_errored) {
        fprintf(stderr, "FAIL: GPU receive write error did not propagate (rx_err=%d tx_err=%d)\n", rx_errored ? 1 : 0,
                tx_errored ? 1 : 0);
        goto out;
    }

    /* verify destination VRAM still holds 0xaaaaaaaa and subsequent clean receive succeeds */
    memset(chk_host, 0, sz);
    if (!vxfec_vk_buf_read_offset(app_vk, rx_vram, 0, chk_host, sz))
        goto out;
    for (size_t i = 0; i < sz; i++) {
        if (chk_host[i] != 0xaau) {
            fprintf(stderr, "FAIL: destination VRAM mutated on failed write at byte %zu\n", i);
            goto out;
        }
    }

    tag = 976;
    sreq = rreq = NULL;
    if (net->irecv(rcomm, 1, &rx_ptr, &rx_cap, &tag, &rx_mr, &rreq) != ncclSuccess ||
        net->isend(scomm, tx_host, (int)sz, tag, tx_mr, &sreq) != ncclSuccess)
        goto out;
    int sdone = 0, rdone = 0, got_sz = 0;
    while (!sdone || !rdone) {
        if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
            goto out;
        if (!rdone && net->test(rreq, &rdone, &got_sz) != ncclSuccess)
            goto out;
    }
    if ((size_t)got_sz != sz || !vxfec_vk_buf_read_offset(app_vk, rx_vram, 0, chk_host, sz) || memcmp(chk_host, tx_host, sz) != 0)
        goto out;

    printf("PASS: GPU receive write failure propagates ncclSystemError + PKT_ERR without false PKT_DONE\n");
    ok = true;

out:
    set_gpu_rx_fault(0u);
    if (tx_mr)
        net->deregMr(scomm, tx_mr);
    if (rx_mr)
        net->deregMr(rcomm, rx_mr);
    if (scomm)
        net->closeSend(scomm);
    if (rcomm)
        net->closeRecv(rcomm);
    if (lcomm)
        net->closeListen(lcomm);
    if (dmabuf_fd >= 0)
        close(dmabuf_fd);
    vxfec_vk_buf_free(app_vk, rx_vram);
    vxfec_vk_ctx_free(app_vk);
    free(chk_host);
    free(tx_host);
    return ok;
}

/* [P1] regression: out-of-order multi-tag irecv/isend on a single shared comm */
static bool test_multi_tag_ooo_irecv(const ncclNet_v8_t *net)
{
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
    void *lcomm = NULL, *scomm = NULL, *rcomm = NULL;
    if (net->listen(0, handle, &lcomm) != ncclSuccess)
        return false;
    while (!scomm || !rcomm) {
        if (!scomm && net->connect(0, handle, &scomm, NULL) != ncclSuccess)
            goto fail;
        if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess)
            goto fail;
    }

    uint8_t tx_a[2048], tx_b[2048], rx_a[2048], rx_b[2048];
    for (size_t i = 0; i < sizeof(tx_a); i++) {
        tx_a[i] = (uint8_t)(0x11u + (i * 13u));
        tx_b[i] = (uint8_t)(0x83u + (i * 29u));
    }
    memset(rx_a, 0, sizeof(rx_a));
    memset(rx_b, 0, sizeof(rx_b));

    void *mr_ta = NULL, *mr_tb = NULL, *mr_ra = NULL, *mr_rb = NULL;
    if (net->regMr(scomm, tx_a, sizeof(tx_a), NCCL_PTR_HOST, &mr_ta) != ncclSuccess ||
        net->regMr(scomm, tx_b, sizeof(tx_b), NCCL_PTR_HOST, &mr_tb) != ncclSuccess ||
        net->regMr(rcomm, rx_a, sizeof(rx_a), NCCL_PTR_HOST, &mr_ra) != ncclSuccess ||
        net->regMr(rcomm, rx_b, sizeof(rx_b), NCCL_PTR_HOST, &mr_rb) != ncclSuccess)
        goto fail_mr;

    void *sreq_a = NULL, *sreq_b = NULL, *rreq_a = NULL, *rreq_b = NULL;
    int tag_a = 101, tag_b = 202;
    int cap_a = (int)sizeof(rx_a), cap_b = (int)sizeof(rx_b);
    void *ptr_a = rx_a, *ptr_b = rx_b;

    if (net->isend(scomm, tx_a, (int)sizeof(tx_a), tag_a, mr_ta, &sreq_a) != ncclSuccess ||
        net->isend(scomm, tx_b, (int)sizeof(tx_b), tag_b, mr_tb, &sreq_b) != ncclSuccess)
        goto fail_mr;

    /* post irecv for tag_b first and verify it completes before tag_a is posted */
    if (net->irecv(rcomm, 1, &ptr_b, &cap_b, &tag_b, &mr_rb, &rreq_b) != ncclSuccess)
        goto fail_mr;

    int sb_done = 0, rb_done = 0, rb_sz = 0;
    for (int spin = 0; spin < 200000 && (!sb_done || !rb_done); spin++) {
        if (!sb_done && net->test(sreq_b, &sb_done, NULL) != ncclSuccess)
            goto fail_mr;
        if (!rb_done && net->test(rreq_b, &rb_done, &rb_sz) != ncclSuccess)
            goto fail_mr;
    }
    if (!sb_done || !rb_done || rb_sz != (int)sizeof(tx_b) || memcmp(rx_b, tx_b, sizeof(tx_b)) != 0)
        goto fail_mr;

    if (net->irecv(rcomm, 1, &ptr_a, &cap_a, &tag_a, &mr_ra, &rreq_a) != ncclSuccess)
        goto fail_mr;
    int sa_done = 0, ra_done = 0, ra_sz = 0;
    for (int spin = 0; spin < 200000 && (!sa_done || !ra_done); spin++) {
        if (!sa_done && net->test(sreq_a, &sa_done, NULL) != ncclSuccess)
            goto fail_mr;
        if (!ra_done && net->test(rreq_a, &ra_done, &ra_sz) != ncclSuccess)
            goto fail_mr;
    }
    if (!sa_done || !ra_done || ra_sz != (int)sizeof(tx_a) || memcmp(rx_a, tx_a, sizeof(tx_a)) != 0)
        goto fail_mr;

    net->deregMr(rcomm, mr_rb);
    net->deregMr(rcomm, mr_ra);
    net->deregMr(scomm, mr_tb);
    net->deregMr(scomm, mr_ta);
    net->closeSend(scomm);
    net->closeRecv(rcomm);
    net->closeListen(lcomm);
    printf("PASS: Multi-tag out-of-order irecv/isend on shared comm completes without deadlock\n");
    return true;

fail_mr:
    if (mr_rb)
        net->deregMr(rcomm, mr_rb);
    if (mr_ra)
        net->deregMr(rcomm, mr_ra);
    if (mr_tb)
        net->deregMr(scomm, mr_tb);
    if (mr_ta)
        net->deregMr(scomm, mr_ta);
fail:
    if (scomm)
        net->closeSend(scomm);
    if (rcomm)
        net->closeRecv(rcomm);
    if (lcomm)
        net->closeListen(lcomm);
    return false;
}

typedef struct {
    uint32_t magic;
    uint32_t ip_be;
    uint16_t ctrl_port_be;
    uint16_t num_ports;
    uint32_t cookie;
    uint64_t reserved;
} test_handle_t;

typedef struct {
    uint32_t magic;
    uint8_t pkt_type;
    uint8_t has_ibv;
    uint16_t num_ports;
    uint32_t cookie;
    uint16_t tx_ctrl_port_be;
    uint8_t features;
    uint8_t pad0;
    uint64_t conn_id;
    uint16_t tx_ports_be[32];
} test_hello_t;

typedef struct {
    uint32_t magic;
    uint8_t pkt_type;
    uint8_t has_ibv;
    uint16_t num_ports;
    uint32_t cookie;
    uint16_t rx_ctrl_port_be;
    uint8_t features;
    uint8_t pad0;
    uint64_t conn_id;
    uint16_t rx_ports_be[32];
} test_ack_t;

typedef struct {
    uint32_t magic;
    uint32_t cookie;
    uint8_t pkt_type;
    uint8_t flags;
    uint16_t sym_len;
    uint32_t msg_seq;
    uint32_t tag_seq;
    uint32_t tag;
    uint32_t total_bytes;
    uint32_t K;
    uint16_t sym_T;
    uint16_t srtt_us;
    uint32_t esi;
} test_wire_hdr_t;

typedef struct {
    int ctrl_fd;
    int data_fd;
    struct sockaddr_in lc_ctrl;
    struct sockaddr_in rc_ctrl;
    struct sockaddr_in sc_ctrl;
    struct sockaddr_in rc_data[32];
    uint16_t num_ports;
    uint32_t drop_tag;
    uint32_t drop_esi_beg;
    uint32_t drop_esi_cnt;
    uint32_t dropped_wire;
    bool block_nacks;
    bool nack_seen;
} wire_fec_relay_t;

static void pump_wire_fec_relay(wire_fec_relay_t *rl, const struct sockaddr_in *ctrl_bind, const struct sockaddr_in *data_bind)
{
    uint8_t buf[9000];
    for (;;) {
        struct sockaddr_in src;
        socklen_t slen = sizeof(src);
        ssize_t n = recvfrom(rl->ctrl_fd, buf, sizeof(buf), MSG_DONTWAIT, (struct sockaddr *)&src, &slen);
        if (n <= 0)
            break;
        if (n == (ssize_t)sizeof(test_hello_t) && buf[4] == 1u) {
            test_hello_t hello;
            memcpy(&hello, buf, sizeof(hello));
            rl->sc_ctrl = src;
            rl->num_ports = hello.num_ports;
            hello.tx_ctrl_port_be = ctrl_bind->sin_port;
            for (uint16_t p = 0; p < hello.num_ports && p < 32u; p++)
                hello.tx_ports_be[p] = data_bind->sin_port;
            (void)sendto(rl->ctrl_fd, &hello, sizeof(hello), MSG_DONTWAIT, (const struct sockaddr *)&rl->lc_ctrl,
                         sizeof(rl->lc_ctrl));
        } else if (n == (ssize_t)sizeof(test_ack_t) && buf[4] == 2u) {
            test_ack_t ack;
            memcpy(&ack, buf, sizeof(ack));
            rl->rc_ctrl.sin_family = AF_INET;
            rl->rc_ctrl.sin_addr = rl->lc_ctrl.sin_addr;
            rl->rc_ctrl.sin_port = ack.rx_ctrl_port_be;
            for (uint16_t p = 0; p < ack.num_ports && p < 32u; p++) {
                rl->rc_data[p].sin_family = AF_INET;
                rl->rc_data[p].sin_addr = rl->lc_ctrl.sin_addr;
                rl->rc_data[p].sin_port = ack.rx_ports_be[p];
                ack.rx_ports_be[p] = data_bind->sin_port;
            }
            ack.rx_ctrl_port_be = ctrl_bind->sin_port;
            (void)sendto(rl->ctrl_fd, &ack, sizeof(ack), MSG_DONTWAIT, (const struct sockaddr *)&rl->sc_ctrl, sizeof(rl->sc_ctrl));
        } else if (n >= 12) {
            uint8_t ptype = buf[8];
            if (rl->block_nacks && (ptype == 5u || ptype == 6u || ptype == 8u)) {
                rl->nack_seen = true;
                continue;
            }
            (void)sendto(rl->ctrl_fd, buf, (size_t)n, MSG_DONTWAIT, (const struct sockaddr *)&rl->sc_ctrl, sizeof(rl->sc_ctrl));
        }
    }

    uint32_t rr = 0u;
    for (;;) {
        ssize_t n = recvfrom(rl->data_fd, buf, sizeof(buf), MSG_DONTWAIT, NULL, NULL);
        if (n <= 0)
            break;
        if ((size_t)n >= sizeof(test_wire_hdr_t)) {
            test_wire_hdr_t wh;
            memcpy(&wh, buf, sizeof(wh));
            if (rl->drop_tag != 0u && wh.pkt_type == 3u && wh.tag == rl->drop_tag && wh.esi >= rl->drop_esi_beg &&
                wh.esi < rl->drop_esi_beg + rl->drop_esi_cnt) {
                rl->dropped_wire++;
                continue;
            }
        }
        uint16_t p = rl->num_ports ? (uint16_t)(rr++ % rl->num_ports) : 0u;
        (void)sendto(rl->data_fd, buf, (size_t)n, MSG_DONTWAIT, (const struct sockaddr *)&rl->rc_data[p], sizeof(rl->rc_data[p]));
    }
}

/* verify 0-RTT one-way proactive FEC repair (RS8 and RaptorQ) while physically blocking reverse SACK/HARQ-IR/GBN NACKs */
static bool test_zero_rtt_proactive_fec_repair(const ncclNet_v8_t *net, void (*get_stats)(vxfec_nccl_stats_t *),
                                               void (*reset_stats)(void), void (*set_codec)(const char *))
{
    const size_t sz = 64u * 1280u; /* K=64 at 1280B MTU symbol */
    uint8_t *tx = (uint8_t *)malloc(sz);
    uint8_t *rx = (uint8_t *)malloc(sz);
    void *lcomm = NULL, *scomm = NULL, *rcomm = NULL;
    void *tmr = NULL, *rmr = NULL;
    wire_fec_relay_t rl;
    memset(&rl, 0, sizeof(rl));
    rl.ctrl_fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    rl.data_fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    bool ok = false;

    if (!tx || !rx || rl.ctrl_fd < 0 || rl.data_fd < 0 || setenv("VXFEC_NCCL_MTU_SYM", "1280", 1) != 0)
        goto out;
    int rcvbuf = 2 * 1024 * 1024;
    (void)setsockopt(rl.ctrl_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    (void)setsockopt(rl.data_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    struct sockaddr_in ctrl_bind = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    struct sockaddr_in data_bind = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    socklen_t blen = sizeof(ctrl_bind);
    if (bind(rl.ctrl_fd, (struct sockaddr *)&ctrl_bind, blen) != 0 ||
        getsockname(rl.ctrl_fd, (struct sockaddr *)&ctrl_bind, &blen) != 0 ||
        bind(rl.data_fd, (struct sockaddr *)&data_bind, blen) != 0 ||
        getsockname(rl.data_fd, (struct sockaddr *)&data_bind, &blen) != 0)
        goto out;

    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
    if (net->listen(0, handle, &lcomm) != ncclSuccess)
        goto out;
    test_handle_t h;
    memcpy(&h, handle, sizeof(h));
    rl.lc_ctrl = (struct sockaddr_in){.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be, .sin_port = h.ctrl_port_be};
    h.ctrl_port_be = ctrl_bind.sin_port;
    memcpy(handle, &h, sizeof(h));

    for (int spin = 0; spin < 100000 && (!scomm || !rcomm); spin++) {
        if (!scomm && net->connect(0, handle, &scomm, NULL) != ncclSuccess)
            goto out;
        pump_wire_fec_relay(&rl, &ctrl_bind, &data_bind);
        if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess)
            goto out;
        pump_wire_fec_relay(&rl, &ctrl_bind, &data_bind);
    }
    if (!scomm || !rcomm || net->regMr(scomm, tx, sz, NCCL_PTR_HOST, &tmr) != ncclSuccess ||
        net->regMr(rcomm, rx, sz, NCCL_PTR_HOST, &rmr) != ncclSuccess)
        goto out;

    /* pass 0: warm loss_ewma by dropping 12/64 systematic datagrams (feedback allowed);
     * pass 1: Cauchy RS8 0-RTT proactive repair of 3 wire drops with reverse NACKs blocked;
     * pass 2: RaptorQ 0-RTT proactive repair of 3 wire drops with reverse NACKs blocked */
    for (int pass = 0; pass < 3; pass++) {
        set_codec((pass == 2) ? "rq-gf256" : "rs8");
        reset_stats();
        for (size_t i = 0; i < sz; i++)
            tx[i] = (uint8_t)((i * 97u + (size_t)pass * 41u + 13u) & 0xffu);
        memset(rx, 0, sz);

        int tag = 980 + pass;
        rl.drop_tag = (uint32_t)tag;
        rl.drop_esi_beg = (pass == 0) ? 8u : 20u;
        rl.drop_esi_cnt = (pass == 0) ? 12u : 3u;
        rl.dropped_wire = 0u;
        rl.block_nacks = (pass > 0);
        rl.nack_seen = false;

        void *sreq = NULL, *rreq = NULL, *rptr = rx;
        int rcap = (int)sz, sdone = 0, rdone = 0, got = 0;
        if (net->irecv(rcomm, 1, &rptr, &rcap, &tag, &rmr, &rreq) != ncclSuccess || !rreq ||
            net->isend(scomm, tx, (int)sz, tag, tmr, &sreq) != ncclSuccess || !sreq)
            goto out;

        for (int spin = 0; spin < 200000 && (!sdone || !rdone); spin++) {
            pump_wire_fec_relay(&rl, &ctrl_bind, &data_bind);
            if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
                goto out;
            pump_wire_fec_relay(&rl, &ctrl_bind, &data_bind);
            if (!rdone && net->test(rreq, &rdone, &got) != ncclSuccess)
                goto out;
        }
        vxfec_nccl_stats_t st;
        get_stats(&st);
        if (!sdone || !rdone || (size_t)got != sz || memcmp(tx, rx, sz) != 0 || rl.dropped_wire != rl.drop_esi_cnt) {
            fprintf(stderr, "FAIL: 0-RTT proactive pass %d failed completion/memcmp (dropped=%u/%u)\n", pass, rl.dropped_wire,
                    rl.drop_esi_cnt);
            goto out;
        }
        if (pass == 1 && (rl.nack_seen || st.tx_proactive_rep_pkts < 3u || st.tx_harq_ir_pkts != 0u || st.tx_sack_retx_pkts != 0u ||
                          st.rx_rs8_decodes != 1u)) {
            fprintf(stderr, "FAIL: 0-RTT RS8 proactive repair used feedback or missed decode (nack=%d p0=%lu ir=%lu rs8=%lu)\n",
                    rl.nack_seen ? 1 : 0, (unsigned long)st.tx_proactive_rep_pkts, (unsigned long)st.tx_harq_ir_pkts,
                    (unsigned long)st.rx_rs8_decodes);
            goto out;
        }
        if (pass == 2 && (rl.nack_seen || st.tx_proactive_rep_pkts < 5u || st.tx_harq_ir_pkts != 0u || st.tx_sack_retx_pkts != 0u ||
                          st.rx_rq_decodes != 1u)) {
            fprintf(stderr, "FAIL: 0-RTT RaptorQ proactive repair used feedback or missed decode (nack=%d p0=%lu ir=%lu rq=%lu)\n",
                    rl.nack_seen ? 1 : 0, (unsigned long)st.tx_proactive_rep_pkts, (unsigned long)st.tx_harq_ir_pkts,
                    (unsigned long)st.rx_rq_decodes);
            goto out;
        }
    }

    printf("PASS: 0-RTT one-way proactive FEC repair (Cauchy RS8 + RaptorQ, reverse NACKs blocked, 0 retx/IR)\n");
    ok = true;

out:
    unsetenv("VXFEC_NCCL_MTU_SYM");
    set_codec(NULL);
    if (tmr)
        net->deregMr(scomm, tmr);
    if (rmr)
        net->deregMr(rcomm, rmr);
    if (scomm)
        net->closeSend(scomm);
    if (rcomm)
        net->closeRecv(rcomm);
    if (lcomm)
        net->closeListen(lcomm);
    if (rl.ctrl_fd >= 0)
        close(rl.ctrl_fd);
    if (rl.data_fd >= 0)
        close(rl.data_fd);
    free(rx);
    free(tx);
    return ok;
}

enum { RQ_TX = 1, RQ_RX = 2, RQ_BYTES = 512 * 1024, RQ_K = 410, RQ_SYM = 1280, RS8_K = 64, RS8_BYTES = RS8_K * RQ_SYM };

static uint32_t test_rq_expected_k = RQ_K;
static uint32_t test_rq_drop_base = 80u;
static uint8_t test_rq_expected_flag = 0u;
static uint64_t test_rq_drop_mask;
static unsigned test_rq_drop_budget; /* zero keeps the fault active for the entire transfer */
static unsigned test_rq_repair_packets;
static struct {
    int fd;
    size_t len;
    struct sockaddr_storage peer;
    socklen_t peer_len;
    uint8_t bytes[sizeof(test_wire_hdr_t) + RQ_SYM];
} test_rq_late;

static int test_rq_send_with_loss(int fd, struct mmsghdr *msgs, unsigned count, int flags)
{
    unsigned sent = 0;
    for (; sent < count; sent++) {
        struct mmsghdr *msg = &msgs[sent];
        struct msghdr *hdr = &msg->msg_hdr;
        if (hdr->msg_iovlen > 0 && hdr->msg_iov[0].iov_len == sizeof(test_wire_hdr_t)) {
            test_wire_hdr_t wire;
            memcpy(&wire, hdr->msg_iov[0].iov_base, sizeof(wire));
            if (wire.pkt_type == 3u && wire.tag == (uint32_t)test_rq_drop_tag && wire.K == test_rq_expected_k &&
                wire.sym_T == RQ_SYM && (wire.flags & 0x0fu) == test_rq_expected_flag) {
                if (wire.esi >= test_rq_drop_base && wire.esi < test_rq_drop_base + test_rq_drop_cnt) {
                    if (wire.esi == test_rq_drop_base && !test_rq_late.len && hdr->msg_iovlen == 2 &&
                        hdr->msg_iov[1].iov_len == RQ_SYM && hdr->msg_namelen <= sizeof(test_rq_late.peer)) {
                        memcpy(test_rq_late.bytes, &wire, sizeof(wire));
                        memcpy(test_rq_late.bytes + sizeof(wire), hdr->msg_iov[1].iov_base, RQ_SYM);
                        test_rq_late.fd = fd;
                        test_rq_late.len = sizeof(wire) + RQ_SYM;
                        test_rq_late.peer_len = hdr->msg_namelen;
                        if (hdr->msg_namelen)
                            memcpy(&test_rq_late.peer, hdr->msg_name, hdr->msg_namelen);
                    }
                    test_rq_drop_mask |= 1ULL << (wire.esi - test_rq_drop_base);
                    if (test_rq_drop_budget && --test_rq_drop_budget == 0u)
                        test_rq_drop_tag = 0;
                    msg->msg_len = sizeof(wire) + wire.sym_len;
                    continue;
                }
                if (wire.esi >= test_rq_expected_k)
                    test_rq_repair_packets++;
            }
        }
        if (test_real_sendmmsg(fd, msg, 1u, flags) != 1)
            return sent ? (int)sent : -1;
    }
    return (int)sent;
}

/* hide a submitted timeline value or inject semaphore poll errors while tracking GPU buffer lifetime */
static struct {
    unsigned side;
    unsigned hold_side;
    unsigned fail_mask;
    unsigned failed_mask;
    unsigned waits;
    unsigned post_dispatch_waits;
    unsigned pending_polls;
    unsigned poll_fail_after;
    unsigned quarantine_error_budget;
    unsigned rx_submits;
    unsigned rx_destroyed_buffers;
    unsigned total_destroyed_buffers;
    bool dispatched;
    bool poll_failed;
    bool force_device_lost_idle;
    bool fail_fence_wait;
    bool fail_device_idle;
    VkSemaphore semaphore;
    uint64_t ticket;
    VkSemaphore quarantine_sem;
    uint64_t quarantine_ticket;
} test_rq_vk;

static PFN_vkQueueSubmit test_vk_queue_submit;
static PFN_vkGetSemaphoreCounterValue test_vk_get_counter;
static PFN_vkWaitSemaphores test_vk_wait_semaphores;
static PFN_vkWaitForFences test_vk_wait_fences;
static PFN_vkDestroyBuffer test_vk_destroy_buffer;
static PFN_vkDeviceWaitIdle test_vk_device_wait_idle;
static pthread_once_t test_vk_once = PTHREAD_ONCE_INIT;

static void load_test_vk_symbols(void)
{
    test_vk_queue_submit = (PFN_vkQueueSubmit)dlsym(RTLD_NEXT, "vkQueueSubmit");
    test_vk_get_counter = (PFN_vkGetSemaphoreCounterValue)dlsym(RTLD_NEXT, "vkGetSemaphoreCounterValue");
    test_vk_wait_semaphores = (PFN_vkWaitSemaphores)dlsym(RTLD_NEXT, "vkWaitSemaphores");
    test_vk_wait_fences = (PFN_vkWaitForFences)dlsym(RTLD_NEXT, "vkWaitForFences");
    test_vk_destroy_buffer = (PFN_vkDestroyBuffer)dlsym(RTLD_NEXT, "vkDestroyBuffer");
    test_vk_device_wait_idle = (PFN_vkDeviceWaitIdle)dlsym(RTLD_NEXT, "vkDeviceWaitIdle");
}

__attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue queue, uint32_t count,
                                                                                    const VkSubmitInfo *submits, VkFence fence)
{
    pthread_once(&test_vk_once, load_test_vk_symbols);
    const VkTimelineSemaphoreSubmitInfo *timeline = NULL;
    if (test_rq_vk.side && count == 1 && submits[0].signalSemaphoreCount == 1) {
        for (const VkBaseInStructure *info = submits[0].pNext; info; info = info->pNext) {
            if (info->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
                timeline = (const VkTimelineSemaphoreSubmitInfo *)info;
                break;
            }
        }
    }
    if (timeline && timeline->signalSemaphoreValueCount != 1)
        timeline = NULL;
    if (timeline && (test_rq_vk.fail_mask & test_rq_vk.side)) {
        test_rq_vk.fail_mask &= ~test_rq_vk.side;
        test_rq_vk.failed_mask |= test_rq_vk.side;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    VkResult result = test_vk_queue_submit(queue, count, submits, fence);
    if (result == VK_SUCCESS && timeline) {
        if (test_rq_vk.side == RQ_RX) {
            test_rq_vk.rx_submits++;
            test_rq_vk.dispatched = true;
            test_rq_vk.post_dispatch_waits = 0;
        }
        if (test_rq_vk.side == test_rq_vk.hold_side && !test_rq_vk.ticket) {
            test_rq_vk.semaphore = submits[0].pSignalSemaphores[0];
            test_rq_vk.ticket = timeline->pSignalSemaphoreValues[0];
        }
    }
    return result;
}

__attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL vkGetSemaphoreCounterValue(VkDevice dev, VkSemaphore sem,
                                                                                                 uint64_t *value)
{
    pthread_once(&test_vk_once, load_test_vk_symbols);
    VkResult result = test_vk_get_counter(dev, sem, value);
    if (result == VK_SUCCESS && test_rq_vk.ticket && sem == test_rq_vk.semaphore) {
        test_rq_vk.pending_polls++;
        if (test_rq_vk.poll_fail_after && test_rq_vk.pending_polls >= test_rq_vk.poll_fail_after) {
            test_rq_vk.poll_failed = true;
            if (!test_rq_vk.quarantine_ticket) {
                test_rq_vk.quarantine_sem = sem;
                test_rq_vk.quarantine_ticket = test_rq_vk.ticket;
            }
            test_rq_vk.ticket = 0;
            test_rq_vk.poll_fail_after = 0;
            return VK_ERROR_DEVICE_LOST;
        }
        uint64_t clamp = test_rq_vk.ticket - 1;
        if (test_rq_vk.quarantine_ticket && sem == test_rq_vk.quarantine_sem && test_rq_vk.quarantine_ticket - 1 < clamp)
            clamp = test_rq_vk.quarantine_ticket - 1;
        if (*value > clamp)
            *value = clamp;
        return result;
    }
    if (test_rq_vk.quarantine_ticket && sem == test_rq_vk.quarantine_sem) {
        if (test_rq_vk.quarantine_error_budget > 0u) {
            if (test_rq_vk.quarantine_error_budget != UINT32_MAX)
                test_rq_vk.quarantine_error_budget--;
            return VK_ERROR_DEVICE_LOST;
        }
        if (result == VK_SUCCESS && *value >= test_rq_vk.quarantine_ticket)
            *value = test_rq_vk.quarantine_ticket - 1;
    }
    return result;
}

__attribute__((visibility("default"))) VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice dev, VkBuffer buffer,
                                                                                  const VkAllocationCallbacks *alloc)
{
    pthread_once(&test_vk_once, load_test_vk_symbols);
    if (buffer != VK_NULL_HANDLE) {
        test_rq_vk.total_destroyed_buffers++;
        if (test_rq_vk.side == RQ_RX)
            test_rq_vk.rx_destroyed_buffers++;
    }
    test_vk_destroy_buffer(dev, buffer, alloc);
}

__attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice dev)
{
    pthread_once(&test_vk_once, load_test_vk_symbols);
    VkResult res = test_vk_device_wait_idle(dev);
    if (test_rq_vk.fail_device_idle)
        return VK_ERROR_UNKNOWN;
    if (test_rq_vk.force_device_lost_idle && res == VK_SUCCESS)
        return VK_ERROR_DEVICE_LOST;
    return res;
}

__attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL vkWaitSemaphores(VkDevice dev,
                                                                                       const VkSemaphoreWaitInfo *info,
                                                                                       uint64_t timeout)
{
    pthread_once(&test_vk_once, load_test_vk_symbols);
    if (test_rq_vk.side && timeout > 0) {
        test_rq_vk.waits++;
        if (test_rq_vk.dispatched)
            test_rq_vk.post_dispatch_waits++;
    }
    return test_vk_wait_semaphores(dev, info, timeout);
}

__attribute__((visibility("default"))) VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice dev, uint32_t count,
                                                                                      const VkFence *fences, VkBool32 wait_all,
                                                                                      uint64_t timeout)
{
    pthread_once(&test_vk_once, load_test_vk_symbols);
    if (test_rq_vk.side && timeout > 0) {
        test_rq_vk.waits++;
        if (test_rq_vk.dispatched)
            test_rq_vk.post_dispatch_waits++;
    }
    VkResult res = test_vk_wait_fences(dev, count, fences, wait_all, timeout);
    if (test_rq_vk.fail_fence_wait)
        return VK_ERROR_UNKNOWN;
    return res;
}

static uint64_t test_rq_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

typedef struct {
    void *tx_req;
    void *rx_req;
    int tx_done;
    int rx_done;
    int got;
} rq_test_transfer;

static bool post_rq_test_transfer(const ncclNet_v8_t *net, void *sc, void *rc, const void *src, void *dst, int len, int tag,
                                  rq_test_transfer *xfer)
{
    memset(xfer, 0, sizeof(*xfer));
    test_rq_vk.side = RQ_RX;
    bool ok = net->irecv(rc, 1, &dst, &len, &tag, NULL, &xfer->rx_req) == ncclSuccess && xfer->rx_req;
    if (ok) {
        test_rq_vk.side = RQ_TX;
        ok = net->isend(sc, (void *)src, len, tag, NULL, &xfer->tx_req) == ncclSuccess && xfer->tx_req;
    }
    test_rq_vk.side = 0;
    return ok;
}

static bool progress_rq_test_transfer(const ncclNet_v8_t *net, rq_test_transfer *xfer)
{
    test_rq_vk.side = RQ_TX;
    bool ok = xfer->tx_done || net->test(xfer->tx_req, &xfer->tx_done, NULL) == ncclSuccess;
    if (ok) {
        test_rq_vk.side = RQ_RX;
        ok = xfer->rx_done || net->test(xfer->rx_req, &xfer->rx_done, &xfer->got) == ncclSuccess;
    }
    test_rq_vk.side = 0;
    return ok;
}

static bool test_cold_sack_recovery(const ncclNet_v8_t *net, ring_comms_t *ring, void (*get_stats)(vxfec_nccl_stats_t *),
                                    void (*reset_stats)(void))
{
    enum { k = 8, bytes = k * RQ_SYM };
    uint8_t src[bytes], dst[bytes];
    bool ok = true;
    test_rq_expected_k = k;
    test_rq_expected_flag = 0x08u; /* WIRE_FLAG_SACK_ONLY */
    test_rq_drop_base = 2u;
    test_rq_drop_cnt = 1u;
    /* keep clean DONE and window ACK samples absent so RTT must bootstrap from SACK */
    for (int gen = 0; gen < 4 && ok; gen++) {
        for (size_t i = 0; i < sizeof(src); i++)
            src[i] = (uint8_t)(i * 17u + (i >> 6u) + (unsigned)gen);
        memset(dst, 0, sizeof(dst));
        reset_stats();
        int tag = 860 + gen;
        bool subset_late = (gen == 0);
        unsigned drops = (gen == 0 || gen == 2) ? 2u : 1u; /* gen 1 shrinks SACK via late ESI; gen 3 loses first retx */
        uint64_t exp_mask = subset_late ? 3u : 1u;
        test_rq_drop_cnt = subset_late ? 2u : 1u;
        test_rq_drop_tag = tag;
        test_rq_drop_budget = drops;
        test_rq_drop_mask = 0;
        test_rq_late.len = 0u;
        rq_test_transfer xfer;
        uint64_t t0 = test_rq_now_ns();
        bool late_sent = !subset_late;
        ok = post_rq_test_transfer(net, ring->send_comm[0], ring->recv_comm[1], src, dst, bytes, tag, &xfer);
        uint64_t deadline = t0 + 2000000000ULL;
        while (ok && (!xfer.tx_done || !xfer.rx_done)) {
            uint64_t now = test_rq_now_ns();
            if (!late_sent && test_rq_late.len && now - t0 >= 400000ULL) {
                if (test_rq_late.peer_len)
                    (void)sendto(test_rq_late.fd, test_rq_late.bytes, test_rq_late.len, MSG_DONTWAIT,
                                 (const struct sockaddr *)&test_rq_late.peer, test_rq_late.peer_len);
                else
                    (void)send(test_rq_late.fd, test_rq_late.bytes, test_rq_late.len, MSG_DONTWAIT);
                test_rq_late.len = 0u;
                late_sent = true;
            }
            ok = now < deadline && progress_rq_test_transfer(net, &xfer);
        }
        vxfec_nccl_stats_t st;
        get_stats(&st);
        uint64_t max_retx = subset_late ? 2u : 3u;
        if (!ok || xfer.got != bytes || memcmp(src, dst, bytes) || test_rq_drop_mask != exp_mask || test_rq_drop_budget ||
            st.tx_sack_retx_pkts < drops || st.tx_sack_retx_pkts > max_retx) {
            fprintf(stderr, "FAIL: cold 10 ms RTT SACK generation %d (drops=%u/%u retx=%lu)\n", gen + 1,
                    drops - test_rq_drop_budget, drops, (unsigned long)st.tx_sack_retx_pkts);
            ok = false;
        }
    }
    test_rq_drop_tag = 0;
    test_rq_drop_cnt = 0u;
    test_rq_drop_budget = 0;
    test_rq_late.len = 0u;
    return ok;
}

static bool test_delayed_loss_recovery(const ncclNet_v8_t *net, void (*get_stats)(vxfec_nccl_stats_t *), void (*reset_stats)(void),
                                       void (*set_fault)(uint32_t, uint32_t, bool), void (*set_delay)(uint64_t),
                                       void (*set_codec)(const char *))
{
    const char *modes[] = {"sack", "rs8"};
    bool ok = true;
    set_delay(5000000ULL); /* 5 ms each way */
    for (unsigned m = 0; m < 2; m++) {
        ring_comms_t ring;
        set_codec(modes[m]);
        set_fault(0u, 1u, false);
        if (!setup_ring(net, &ring) || (m == 0 && !test_cold_sack_recovery(net, &ring, get_stats, reset_stats))) {
            teardown_ring(net, &ring);
            ok = false;
            continue;
        }
        reset_stats();
        bool transferred = run_host_ring_allreduce(net, &ring, 32768u, 99u);
        vxfec_nccl_stats_t st;
        get_stats(&st);
        uint64_t extra = st.tx_sack_retx_pkts + st.tx_harq_ir_pkts + st.tx_proactive_rep_pkts;
        if (!transferred || extra > st.tx_sys_pkts / 20u) {
            fprintf(stderr, "FAIL: 10 ms RTT %s lossless control (sys=%lu extra=%lu)\n", modes[m], (unsigned long)st.tx_sys_pkts,
                    (unsigned long)extra);
            ok = false;
        }

        set_fault(20000u, 6u, true);
        reset_stats();
        for (uint32_t i = 0; i < 3 && transferred; i++)
            transferred = run_host_ring_allreduce(net, &ring, 32768u, 100u + i);
        get_stats(&st);
        teardown_ring(net, &ring);

        extra = st.tx_sack_retx_pkts + st.tx_harq_ir_pkts + st.tx_proactive_rep_pkts;
        bool recovered = m == 0 ? st.tx_sack_retx_pkts > 0 : st.rx_rs8_decodes > 0;
        /* allow burst adaptation and parity while catching repeated requests for in-flight packets */
        if (!transferred || !st.sim_dropped_pkts || !recovered || extra > st.tx_sys_pkts / 4u) {
            fprintf(stderr, "FAIL: 10 ms RTT %s recovery (drops=%lu sys=%lu retx=%lu ir=%lu proactive=%lu rs8_dec=%lu)\n", modes[m],
                    (unsigned long)st.sim_dropped_pkts, (unsigned long)st.tx_sys_pkts, (unsigned long)st.tx_sack_retx_pkts,
                    (unsigned long)st.tx_harq_ir_pkts, (unsigned long)st.tx_proactive_rep_pkts, (unsigned long)st.rx_rs8_decodes);
            ok = false;
        }
    }
    set_fault(0u, 1u, false);
    set_delay(0u);
    set_codec(NULL);
    if (ok)
        printf("PASS: bounded SACK and RS8 recovery at 10 ms RTT under 2%% burst loss\n");
    return ok;
}

static bool test_rq_gpu_progress(const ncclNet_v8_t *net, unsigned drops, bool submit_failure)
{
    const int tag = 410;
    const uint64_t end = test_rq_now_ns() + 5000000000ULL;
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
    uint8_t *src = malloc(RQ_BYTES), *dst = calloc(1, RQ_BYTES);
    void *lc = NULL, *sc = NULL, *rc = NULL;
    bool ok = false;
    memset(&test_rq_vk, 0, sizeof(test_rq_vk));
    test_rq_expected_k = RQ_K;
    test_rq_drop_base = 80u;
#ifdef NANORQ_HAS_GF2_MODE
    test_rq_expected_flag = 1u;
#else
    test_rq_expected_flag = 0u;
#endif
    test_rq_drop_mask = 0;
    test_rq_repair_packets = 0;
    test_rq_late.len = 0;
    if (!src || !dst || net->listen(0, handle, &lc) != ncclSuccess)
        goto out;
    for (size_t i = 0; i < RQ_BYTES; i++)
        src[i] = (uint8_t)(i * 37u + (i >> 6) * 23u + 1u);
    while (!sc || !rc) {
        if (test_rq_now_ns() >= end || (!sc && net->connect(0, handle, &sc, NULL) != ncclSuccess) ||
            (!rc && net->accept(lc, &rc, NULL) != ncclSuccess))
            goto out;
    }

    rq_test_transfer big;
    test_rq_drop_cnt = drops;
    test_rq_drop_tag = tag;
    test_rq_vk.fail_mask = submit_failure ? (RQ_TX | RQ_RX) : 0;
    test_rq_vk.hold_side = submit_failure ? 0 : RQ_TX;
    if (!post_rq_test_transfer(net, sc, rc, src, dst, RQ_BYTES, tag, &big))
        goto out;

    if (!submit_failure) {
        for (unsigned side = RQ_TX; side <= RQ_RX; side++) {
            while (!test_rq_vk.ticket) {
                if (test_rq_now_ns() >= end || !progress_rq_test_transfer(net, &big) || big.tx_done || big.rx_done)
                    goto out;
            }
            if (side == RQ_RX &&
                (!test_rq_late.len || sendto(test_rq_late.fd, test_rq_late.bytes, test_rq_late.len, 0,
                                             test_rq_late.peer_len ? (const struct sockaddr *)&test_rq_late.peer : NULL,
                                             test_rq_late.peer_len) != (ssize_t)test_rq_late.len))
                goto out;
            /* deliver a lossless tag on the same comm while RQ job is pending */
            const uint32_t ping = 0xf0123456u + side;
            uint32_t pong = 0;
            rq_test_transfer small;
            unsigned polls = test_rq_vk.pending_polls;
            if (!post_rq_test_transfer(net, sc, rc, &ping, &pong, sizeof(ping), tag + (int)side, &small))
                goto out;
            while (!small.tx_done || !small.rx_done) {
                if (test_rq_now_ns() >= end || !progress_rq_test_transfer(net, &small) || !progress_rq_test_transfer(net, &big) ||
                    big.tx_done || big.rx_done)
                    goto out;
            }
            if (small.got != sizeof(ping) || pong != ping || test_rq_vk.pending_polls == polls ||
                (side == RQ_TX && test_rq_repair_packets != 0))
                goto out;
            test_rq_vk.ticket = 0;
            test_rq_vk.hold_side = (side == RQ_TX) ? RQ_RX : 0;
        }
    }

    while (!big.tx_done || !big.rx_done) {
        if (test_rq_now_ns() >= end || !progress_rq_test_transfer(net, &big))
            goto out;
    }
    ok = !test_rq_vk.waits && test_rq_drop_mask == ((1ULL << drops) - 1ULL) && test_rq_repair_packets >= drops &&
         (!submit_failure || test_rq_vk.failed_mask == (RQ_TX | RQ_RX)) && big.got == RQ_BYTES && memcmp(src, dst, RQ_BYTES) == 0;

out:
    if (!ok)
        fprintf(stderr, "FAIL: K=410 %s (drops=%u hold=%u ticket=%lu polls=%u waits=%u dropped=%#lx repairs=%u rejections=%u)\n",
                submit_failure ? "GPU submit fallback" : "async RQ progress", drops, test_rq_vk.hold_side,
                (unsigned long)test_rq_vk.ticket, test_rq_vk.pending_polls, test_rq_vk.waits, (unsigned long)test_rq_drop_mask,
                test_rq_repair_packets, test_rq_vk.failed_mask);
    test_rq_drop_tag = 0;
    test_rq_drop_cnt = 0;
    memset(&test_rq_vk, 0, sizeof(test_rq_vk));
    if (sc)
        net->closeSend(sc);
    if (rc)
        net->closeRecv(rc);
    if (lc)
        net->closeListen(lc);
    free(src);
    free(dst);
    return ok;
}

static bool test_rs8_async_gpu_progress(const ncclNet_v8_t *net, void (*set_codec)(const char *))
{
    const int tag = 510;
    const uint64_t end = test_rq_now_ns() + 5000000000ULL;
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
    uint8_t *src = malloc(RS8_BYTES), *dst = calloc(1, RS8_BYTES);
    void *lc = NULL, *sc = NULL, *rc = NULL;
    bool ok = false;
    memset(&test_rq_vk, 0, sizeof(test_rq_vk));
    set_codec("rs8");
    test_rq_expected_k = RS8_K;
    test_rq_drop_base = 16u;
    test_rq_expected_flag = 2u; /* WIRE_FLAG_RS_CAUCHY8 */
    test_rq_drop_mask = 0;
    test_rq_repair_packets = 0;
    test_rq_late.len = 0;
    if (!src || !dst || net->listen(0, handle, &lc) != ncclSuccess)
        goto out;
    for (size_t i = 0; i < RS8_BYTES; i++)
        src[i] = (uint8_t)(i * 43u + (i >> 5) * 19u + 7u);
    while (!sc || !rc) {
        if (test_rq_now_ns() >= end || (!sc && net->connect(0, handle, &sc, NULL) != ncclSuccess) ||
            (!rc && net->accept(lc, &rc, NULL) != ncclSuccess))
            goto out;
    }

    rq_test_transfer big;
    test_rq_drop_cnt = 4u;
    test_rq_drop_tag = tag;
    test_rq_vk.hold_side = RQ_TX;
    if (!post_rq_test_transfer(net, sc, rc, src, dst, RS8_BYTES, tag, &big))
        goto out;

    for (unsigned side = RQ_TX; side <= RQ_RX; side++) {
        while (!test_rq_vk.ticket) {
            if (test_rq_now_ns() >= end || !progress_rq_test_transfer(net, &big) || big.tx_done || big.rx_done)
                goto out;
        }
        /* lossless tag on the same comm completes while RS8 TX/RX GPU job is in flight */
        const uint32_t ping = 0xe1230000u + side;
        uint32_t pong = 0;
        rq_test_transfer small;
        unsigned polls = test_rq_vk.pending_polls;
        if (!post_rq_test_transfer(net, sc, rc, &ping, &pong, sizeof(ping), tag + (int)side, &small))
            goto out;
        while (!small.tx_done || !small.rx_done) {
            if (test_rq_now_ns() >= end || !progress_rq_test_transfer(net, &small) || !progress_rq_test_transfer(net, &big) ||
                big.tx_done || big.rx_done)
                goto out;
        }
        if (small.got != sizeof(ping) || pong != ping || test_rq_vk.pending_polls == polls ||
            (side == RQ_TX && test_rq_repair_packets != 0))
            goto out;
        test_rq_vk.ticket = 0;
        test_rq_vk.hold_side = (side == RQ_TX) ? RQ_RX : 0;
    }

    while (!big.tx_done || !big.rx_done) {
        if (test_rq_now_ns() >= end || !progress_rq_test_transfer(net, &big))
            goto out;
    }
    ok = !test_rq_vk.waits && test_rq_drop_mask == 0xfULL && test_rq_repair_packets >= 4u && big.got == RS8_BYTES &&
         memcmp(src, dst, RS8_BYTES) == 0;

out:
    if (!ok)
        fprintf(stderr, "FAIL: RS8 async TX/RX progress (waits=%u dropped=%#lx repairs=%u)\n", test_rq_vk.waits,
                (unsigned long)test_rq_drop_mask, test_rq_repair_packets);
    set_codec(NULL);
    test_rq_drop_tag = 0;
    test_rq_drop_cnt = 0;
    test_rq_expected_k = RQ_K;
    test_rq_drop_base = 80u;
    test_rq_expected_flag = 0u;
    memset(&test_rq_vk, 0, sizeof(test_rq_vk));
    if (sc)
        net->closeSend(sc);
    if (rc)
        net->closeRecv(rc);
    if (lc)
        net->closeListen(lc);
    free(src);
    free(dst);
    return ok;
}

static bool test_rs8_gpu_poll_error_retirement(const ncclNet_v8_t *net, void (*set_codec)(const char *))
{
    enum { POOL_SLOTS = 8 };
    const uint64_t end = test_rq_now_ns() + 10000000000ULL;
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
    uint8_t *src = malloc(RS8_BYTES), *dst = calloc(1, RS8_BYTES);
    void *lc = NULL, *sc = NULL, *rc = NULL;
    bool ok = false;
    memset(&test_rq_vk, 0, sizeof(test_rq_vk));
    set_codec("rs8");
    test_rq_expected_k = RS8_K;
    test_rq_drop_base = 16u;
    test_rq_expected_flag = 2u; /* WIRE_FLAG_RS_CAUCHY8 */
    if (!src || !dst || net->listen(0, handle, &lc) != ncclSuccess)
        goto out;
    while (!sc || !rc) {
        if (test_rq_now_ns() >= end || (!sc && net->connect(0, handle, &sc, NULL) != ncclSuccess) ||
            (!rc && net->accept(lc, &rc, NULL) != ncclSuccess))
            goto out;
    }

    /* passes 0..7: quarantine all 8 async pool slots via poll failure -> 0 RX buffer destructions;
     * pass 8: pool exhausted -> 0 new RX GPU submits, clean CPU fallback, still 0 RX buffer destructions;
     * pass 9: clear quarantine -> reaps all 8 retired plans (vkDestroyBuffer >= 9) and resumes GPU decode;
     * pass 10: permanent poll error -> stays quarantined across closeRecv until dlclose wait_idle */
    for (int pass = 0; pass < POOL_SLOTS + 3; pass++) {
        int tag = 520 + pass;
        for (size_t i = 0; i < RS8_BYTES; i++)
            src[i] = (uint8_t)((i * 59u + (size_t)pass * 97u + 13u) & 0xffu);
        memset(dst, 0, RS8_BYTES);

        test_rq_drop_mask = 0;
        test_rq_repair_packets = 0;
        test_rq_late.len = 0;
        test_rq_drop_cnt = 4u;
        test_rq_drop_tag = tag;

        test_rq_vk.waits = 0;
        test_rq_vk.post_dispatch_waits = 0;
        test_rq_vk.pending_polls = 0;
        test_rq_vk.dispatched = false;
        test_rq_vk.poll_failed = false;
        test_rq_vk.ticket = 0;
        unsigned prev_submits = test_rq_vk.rx_submits;
        unsigned prev_rx_destroyed = test_rq_vk.rx_destroyed_buffers;

        if (pass < POOL_SLOTS) {
            test_rq_vk.hold_side = RQ_RX;
            test_rq_vk.poll_fail_after = 2u;
            test_rq_vk.quarantine_error_budget = 1u;
        } else if (pass == POOL_SLOTS) {
            test_rq_vk.hold_side = 0;
            test_rq_vk.poll_fail_after = 0;
            test_rq_vk.quarantine_error_budget = 0;
        } else if (pass == POOL_SLOTS + 1) {
            test_rq_vk.quarantine_error_budget = 0;
            test_rq_vk.hold_side = 0;
            test_rq_vk.poll_fail_after = 0;
        } else {
            test_rq_vk.quarantine_ticket = 0;
            test_rq_vk.quarantine_sem = VK_NULL_HANDLE;
            test_rq_vk.hold_side = RQ_RX;
            test_rq_vk.poll_fail_after = 2u;
            test_rq_vk.quarantine_error_budget = 1u;
        }

        rq_test_transfer xfer;
        memset(&xfer, 0, sizeof(xfer));
        void *rx_dst = dst;
        int rx_len = RS8_BYTES;
        if (net->irecv(rc, 1, &rx_dst, &rx_len, &tag, NULL, &xfer.rx_req) != ncclSuccess || !xfer.rx_req)
            goto out;
        test_rq_vk.side = RQ_TX;
        test_rq_vk.fail_mask = RQ_TX;
        ncclResult_t sst = net->isend(sc, src, RS8_BYTES, tag, NULL, &xfer.tx_req);
        test_rq_vk.side = 0;
        test_rq_vk.fail_mask = 0;
        if (sst != ncclSuccess || !xfer.tx_req)
            goto out;
        while (!xfer.tx_done || !xfer.rx_done) {
            if (test_rq_now_ns() >= end)
                goto out;
            if (!xfer.tx_done) {
                test_rq_vk.side = RQ_TX;
                test_rq_vk.fail_mask = RQ_TX;
                ncclResult_t tst = net->test(xfer.tx_req, &xfer.tx_done, NULL);
                test_rq_vk.side = 0;
                test_rq_vk.fail_mask = 0;
                if (tst != ncclSuccess)
                    goto out;
            }
            if (!xfer.rx_done) {
                if (pass == POOL_SLOTS + 1 && test_rq_vk.quarantine_ticket != 0u) {
                    test_rq_vk.quarantine_ticket = 0;
                    test_rq_vk.quarantine_sem = VK_NULL_HANDLE;
                }
                test_rq_vk.side = RQ_RX;
                ncclResult_t rst = net->test(xfer.rx_req, &xfer.rx_done, &xfer.got);
                test_rq_vk.side = 0;
                if (rst != ncclSuccess)
                    goto out;
            }
        }
        if (test_rq_vk.waits != 0u || test_rq_vk.post_dispatch_waits != 0u || xfer.got != RS8_BYTES ||
            memcmp(src, dst, RS8_BYTES) != 0)
            goto out;

        if (pass < POOL_SLOTS) {
            if (!test_rq_vk.dispatched || !test_rq_vk.poll_failed || test_rq_vk.rx_submits != prev_submits + 1u ||
                test_rq_vk.rx_destroyed_buffers != 0u)
                goto out;
        } else if (pass == POOL_SLOTS) {
            if (test_rq_vk.dispatched || test_rq_vk.waits != 0u || test_rq_vk.rx_submits != prev_submits ||
                test_rq_vk.rx_destroyed_buffers != 0u)
                goto out;
        } else if (pass == POOL_SLOTS + 1) {
            if (!test_rq_vk.dispatched || test_rq_vk.rx_submits != prev_submits + 1u || test_rq_vk.rx_destroyed_buffers != 0u)
                goto out;
        } else {
            if (!test_rq_vk.dispatched || !test_rq_vk.poll_failed || test_rq_vk.rx_submits != prev_submits + 1u ||
                test_rq_vk.rx_destroyed_buffers != prev_rx_destroyed)
                goto out;
        }
    }

    /* also verify closeRecv retires an in-flight (poll_status == 0) RS8 GPU job without blocking */
    {
        int cancel_tag = 540;
        memset(dst, 0, RS8_BYTES);
        test_rq_drop_mask = 0;
        test_rq_repair_packets = 0;
        test_rq_late.len = 0;
        test_rq_drop_cnt = 4u;
        test_rq_drop_tag = cancel_tag;

        test_rq_vk.waits = 0;
        test_rq_vk.post_dispatch_waits = 0;
        test_rq_vk.pending_polls = 0;
        test_rq_vk.dispatched = false;
        test_rq_vk.poll_failed = false;
        test_rq_vk.ticket = 0;
        test_rq_vk.hold_side = RQ_RX;
        test_rq_vk.poll_fail_after = 0;

        rq_test_transfer inflight;
        memset(&inflight, 0, sizeof(inflight));
        void *rx_dst = dst;
        int rx_len = RS8_BYTES;
        if (net->irecv(rc, 1, &rx_dst, &rx_len, &cancel_tag, NULL, &inflight.rx_req) != ncclSuccess || !inflight.rx_req)
            goto out;
        test_rq_vk.side = RQ_TX;
        test_rq_vk.fail_mask = RQ_TX;
        ncclResult_t csst = net->isend(sc, src, RS8_BYTES, cancel_tag, NULL, &inflight.tx_req);
        test_rq_vk.side = 0;
        test_rq_vk.fail_mask = 0;
        if (csst != ncclSuccess || !inflight.tx_req)
            goto out;
        while (!test_rq_vk.dispatched || test_rq_vk.pending_polls == 0u) {
            if (test_rq_now_ns() >= end)
                goto out;
            if (!inflight.tx_done) {
                test_rq_vk.side = RQ_TX;
                test_rq_vk.fail_mask = RQ_TX;
                ncclResult_t ctst = net->test(inflight.tx_req, &inflight.tx_done, NULL);
                test_rq_vk.side = 0;
                test_rq_vk.fail_mask = 0;
                if (ctst != ncclSuccess)
                    goto out;
            }
            test_rq_vk.side = RQ_RX;
            ncclResult_t rst = net->test(inflight.rx_req, &inflight.rx_done, &inflight.got);
            test_rq_vk.side = 0;
            if (rst != ncclSuccess || inflight.rx_done)
                goto out;
        }
        if (test_rq_vk.post_dispatch_waits != 0u)
            goto out;
    }
    ok = true;

out:
    set_codec(NULL);
    test_rq_drop_tag = 0;
    test_rq_drop_cnt = 0;
    test_rq_expected_k = RQ_K;
    test_rq_drop_base = 80u;
    test_rq_expected_flag = 0u;
    if (sc)
        net->closeSend(sc);
    if (rc) {
        unsigned pre_close_rx_destroyed = test_rq_vk.rx_destroyed_buffers;
        test_rq_vk.post_dispatch_waits = 0;
        test_rq_vk.side = RQ_RX;
        net->closeRecv(rc);
        test_rq_vk.side = 0;
        if (ok && (test_rq_vk.post_dispatch_waits != 0u || test_rq_vk.rx_destroyed_buffers != pre_close_rx_destroyed))
            ok = false;
    }
    if (lc)
        net->closeListen(lc);
    if (!ok) {
        fprintf(stderr, "FAIL: RS8 async GPU poll-error retirement / lifetime check (submits=%u rx_destroyed=%u post_waits=%u)\n",
                test_rq_vk.rx_submits, test_rq_vk.rx_destroyed_buffers, test_rq_vk.post_dispatch_waits);
        memset(&test_rq_vk, 0, sizeof(test_rq_vk));
    } else {
        /* keep quarantine_ticket + held ticket active into dlclose() under simulated DEVICE_LOST wait_idle */
        test_rq_vk.side = 0;
        test_rq_vk.hold_side = 0;
        test_rq_vk.poll_fail_after = 0;
    }
    free(src);
    free(dst);
    return ok;
}

static bool test_vk_wait_failure_preserves_ownership(vxfec_vk_ctx *ctx)
{
    const uint8_t parity_m[4] = {0, 1, 2, 3};
    vxfec_rs8_plan *plan1 = vxfec_rs8_encode_plan_new(ctx, 16u, 16u, parity_m, 4u);
    vxfec_rs8_plan *plan2 = vxfec_rs8_encode_plan_new(ctx, 16u, 16u, parity_m, 4u);
    vxfec_rs8_plan *plan3 = vxfec_rs8_encode_plan_new(ctx, 16u, 16u, parity_m, 4u);
    vxfec_vk_buf *d_buf = vxfec_vk_buf_new(ctx, 16u * 64u, true);
    vxfec_vk_buf *out_buf = vxfec_vk_buf_new(ctx, 4u * 64u, true);
    if (!plan1 || !plan2 || !plan3 || !d_buf || !out_buf) {
        vxfec_vk_buf_free(ctx, out_buf);
        vxfec_vk_buf_free(ctx, d_buf);
        vxfec_rs8_plan_free(ctx, plan3);
        vxfec_rs8_plan_free(ctx, plan2);
        vxfec_rs8_plan_free(ctx, plan1);
        return false;
    }

    memset(vxfec_vk_buf_staging_ptr(d_buf), 0, 16u * 64u);
    unsigned pre_destroyed = test_rq_vk.total_destroyed_buffers;
    test_rq_vk.fail_fence_wait = true;
    test_rq_vk.fail_device_idle = true;

    bool enc_ok = vxfec_rs8_exec_pipelined(ctx, plan1, d_buf, 64u, 1u, out_buf, true, true, NULL);
    /* freeing plans while fence wait is unconfirmed must retire their GPU buffers instead of destroying them */
    vxfec_rs8_plan_free(ctx, plan1);
    vxfec_rs8_plan_free(ctx, plan2);
    unsigned during_unconfirmed = test_rq_vk.total_destroyed_buffers;

    /* while previous sync cmd wait is unconfirmed, new sync submissions must also stay blocked */
    bool second_ok = vxfec_rs8_exec_pipelined(ctx, plan3, d_buf, 64u, 1u, out_buf, true, true, NULL);

    test_rq_vk.fail_fence_wait = false;
    test_rq_vk.fail_device_idle = false;
    bool idle_ok = vxfec_vk_wait_idle(ctx);
    unsigned after_idle = test_rq_vk.total_destroyed_buffers;

    /* once idle is confirmed, sync submissions and external ownership transitions work cleanly */
    bool recov_ok = vxfec_rs8_exec_pipelined(ctx, plan3, d_buf, 64u, 1u, out_buf, true, true, NULL);
    vxfec_rs8_plan_free(ctx, plan3);
    vxfec_vk_buf_free(ctx, out_buf);
    vxfec_vk_buf_free(ctx, d_buf);

    if (enc_ok || second_ok || during_unconfirmed != pre_destroyed || !idle_ok || after_idle < pre_destroyed + 2u || !recov_ok) {
        fprintf(stderr, "FAIL: Vulkan wait failure destroyed in-flight buffers prematurely (pre=%u during=%u after=%u)\n",
                pre_destroyed, during_unconfirmed, after_idle);
        return false;
    }
    return true;
}

static int run_rq_gpu_progress_tests(void)
{
    if (setenv("VXFEC_NCCL_VULKAN", "2", 1) != 0 || setenv("VXFEC_NCCL_MTU_SYM", "1280", 1) != 0 ||
        setenv("VXFEC_NCCL_SLICE_KB", "0", 1) != 0 || setenv("VXFEC_NCCL_CODEC", "auto", 1) != 0)
        return 1;
    int vk_status = VXFEC_VK_ERROR;
    vxfec_vk_ctx *ctx = vxfec_vk_ctx_new_status(&vk_status);
    const char *req_env = getenv("VXFEC_VK_REQUIRED");
    if (!req_env || !*req_env)
        req_env = getenv("NANORQ_VK_REQUIRED");
    bool vk_req = req_env && *req_env && strcmp(req_env, "0") != 0;
    if (!ctx) {
        bool can_skip = (vk_status == VXFEC_VK_UNAVAILABLE && !vk_req);
        fprintf(stderr, "%s: K=410 async RaptorQ requires Vulkan compute\n", can_skip ? "SKIP" : "FAIL");
        return can_skip ? 0 : 1;
    }
    bool timeline = vxfec_vk_has_timeline(ctx);
    bool wait_own_ok = timeline ? test_vk_wait_failure_preserves_ownership(ctx) : true;
    vxfec_vk_ctx_free(ctx);
    if (!wait_own_ok)
        return 1;
    if (!timeline) {
        fprintf(stderr, "%s: K=410 async RaptorQ requires Vulkan timeline semaphores\n", vk_req ? "FAIL" : "SKIP");
        return vk_req ? 1 : 0;
    }

    const char *so_path = getenv("VXFEC_NCCL_SO_PATH");
    void *dl = dlopen(so_path && *so_path ? so_path : "./libnccl-vxfec.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        fprintf(stderr, "FAIL: %s\n", dlerror());
        return 1;
    }
    const ncclNet_v8_t *net = (const ncclNet_v8_t *)dlsym(dl, "ncclNetPlugin_v8");
    void (*set_codec)(const char *) = (void (*)(const char *))dlsym(dl, "vxfec_nccl_set_codec_override");
    bool ok = net && set_codec && net->init(dummy_nccl_logger) == ncclSuccess && test_rq_gpu_progress(net, 6u, false) &&
              test_rq_gpu_progress(net, 34u, false) && test_rq_gpu_progress(net, 6u, true) &&
              test_rs8_async_gpu_progress(net, set_codec) && test_rs8_gpu_poll_error_retirement(net, set_codec);
    unsigned pre_fini_destroyed = test_rq_vk.total_destroyed_buffers;
    test_rq_vk.force_device_lost_idle = true;
    dlclose(dl);
    if (ok && test_rq_vk.total_destroyed_buffers <= pre_fini_destroyed) {
        fprintf(stderr, "FAIL: vxf_plugin_fini did not reap quarantined Vulkan buffers after DEVICE_LOST wait_idle\n");
        ok = false;
    }
    memset(&test_rq_vk, 0, sizeof(test_rq_vk));
    if (ok)
        printf("PASS: K=410 RaptorQ async TX/RX, RS8 8-slot pool quarantine/reap lifetime, and CPU fallback are bit-exact\n");
    return !ok;
}

/* observe selected wire codec and slicing negotiation using a peer that controls ACK feature bits */
static bool test_rq_wire_selection(const ncclNet_v8_t *net, void (*set_codec)(const char *), const char *mode, bool allow_gf2,
                                   bool allow_slice)
{
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
    void *lc = NULL, *sc = NULL, *req = NULL;
    int fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    bool ok = false;
    static uint8_t payload[2 * 1024 * 1024]; /* 2 MiB: unsliced K=256 > 200, sliced K=128 (2 gens) */
    if (fd < 0 || net->listen(0, handle, &lc) != ncclSuccess)
        goto out;
    test_handle_t h;
    memcpy(&h, handle, sizeof(h));
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be};
    socklen_t alen = sizeof(addr);
    if (bind(fd, (struct sockaddr *)&addr, alen) || getsockname(fd, (struct sockaddr *)&addr, &alen))
        goto out;
    h.ctrl_port_be = addr.sin_port;
    memcpy(handle, &h, sizeof(h));

    uint8_t expected = 0;
    for (int spin = 0; spin < 50000 && !sc; spin++) {
        if (net->connect(0, handle, &sc, NULL) != ncclSuccess)
            goto out;
        test_hello_t hello;
        struct sockaddr_in peer;
        socklen_t plen = sizeof(peer);
        if (recvfrom(fd, &hello, sizeof(hello), MSG_DONTWAIT, (struct sockaddr *)&peer, &plen) != sizeof(hello))
            continue;
        if (hello.has_ibv != 0u || !(hello.features & 0x04u))
            goto out;
        /* verify truncated 88B HELLO_ACK claiming VXFEC_HELLO_IBV (0x02) is ignored */
        test_ack_t bad_ack = {.magic = h.magic,
                              .pkt_type = 2u,
                              .has_ibv = 0x02u,
                              .features = 0u,
                              .num_ports = h.num_ports,
                              .cookie = h.cookie,
                              .rx_ctrl_port_be = addr.sin_port,
                              .conn_id = hello.conn_id};
        if (sendto(fd, &bad_ack, sizeof(bad_ack), 0, (struct sockaddr *)&peer, plen) != sizeof(bad_ack))
            goto out;
        void *tmp_sc = NULL;
        if (net->connect(0, handle, &tmp_sc, NULL) != ncclSuccess || tmp_sc != NULL) {
            sc = tmp_sc;
            goto out;
        }
        /* emulate legacy byte-5 88-byte HELLO_ACK (carries GF2 | SLICE at byte 5 when features==0) */
        bool legacy_byte5_ack = (mode != NULL) || allow_slice;
        uint8_t feat_bits = (uint8_t)((allow_gf2 ? 1u : 0u) | (allow_slice ? 4u : 0u));
        test_ack_t ack = {.magic = h.magic,
                          .pkt_type = 2u,
                          .has_ibv = (uint8_t)(legacy_byte5_ack ? feat_bits : 0u),
                          .features = (uint8_t)(legacy_byte5_ack ? 0u : feat_bits),
                          .num_ports = h.num_ports,
                          .cookie = h.cookie,
                          .rx_ctrl_port_be = addr.sin_port,
                          .conn_id = hello.conn_id};
        for (unsigned p = 0; p < h.num_ports; p++)
            ack.rx_ports_be[p] = addr.sin_port;
        if (sendto(fd, &ack, sizeof(ack), 0, (struct sockaddr *)&peer, plen) != sizeof(ack))
            goto out;
#ifdef NANORQ_HAS_GF2_MODE
        if (!(hello.features & 1u))
            goto out;
        expected = (allow_gf2 && (!mode || strcmp(mode, "rq-gf2") == 0)) ? 1u : 0u;
#else
        if (hello.features & 1u)
            goto out;
#endif
    }
    set_codec(mode);
    if (!sc || net->isend(sc, payload, sizeof(payload), 1, NULL, &req) != ncclSuccess || !req)
        goto out;
    bool expect_sliced = allow_slice && (mode == NULL);
    for (int spin = 0; spin < 50000; spin++) {
        uint8_t pkt[64];
        ssize_t n = recv(fd, pkt, sizeof(pkt), MSG_DONTWAIT);
        if (n >= (ssize_t)sizeof(test_wire_hdr_t)) {
            test_wire_hdr_t hdr;
            memcpy(&hdr, pkt, sizeof(hdr));
            if (hdr.magic == h.magic && hdr.pkt_type == 3u) {
                if (expect_sliced) {
                    ok = (hdr.K == 128u) && ((hdr.flags & 0x40u) == 0x40u) && ((hdr.flags & 0x0fu) == 0x02u);
                } else {
                    ok = (hdr.K > 200u) && ((hdr.flags & 0x40u) == 0u) && ((hdr.flags & 0x0fu) == expected);
                }
                break;
            }
        }
        int done = 0;
        if (net->test(req, &done, NULL) != ncclSuccess || done)
            break;
    }
out:
    set_codec(NULL);
    if (sc)
        net->closeSend(sc);
    if (lc)
        net->closeListen(lc);
    if (fd >= 0)
        close(fd);
    if (!ok)
        fprintf(stderr, "FAIL: RQ/slice wire selection (mode=%s, peer GF2=%d, peer SLICE=%d)\n", mode ? mode : "auto", allow_gf2,
                allow_slice);
    return ok;
}

typedef struct {
    int fd;
    void *send_comm;
    struct sockaddr_in sender_ctrl;
    uint16_t sender_ports[2];
} raw_feedback_peer_t;

/* use two egress sockets so packet counts reveal SWRR weight changes */
static bool setup_raw_feedback_peer(const ncclNet_v8_t *net, raw_feedback_peer_t *peer)
{
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    socklen_t addr_len = sizeof(addr);
    void *listener = NULL;
    if (peer->fd < 0 || bind(peer->fd, (struct sockaddr *)&addr, addr_len) != 0 ||
        getsockname(peer->fd, (struct sockaddr *)&addr, &addr_len) != 0 || net->listen(0, handle, &listener) != ncclSuccess)
        return false;

    test_handle_t h;
    memcpy(&h, handle, sizeof(h));
    net->closeListen(listener);
    h.ip_be = addr.sin_addr.s_addr;
    h.ctrl_port_be = addr.sin_port;
    h.num_ports = 2u;
    memcpy(handle, &h, sizeof(h));

    test_hello_t hello;
    for (int i = 0; i < 50000 && !peer->send_comm; i++) {
        if (net->connect(0, handle, &peer->send_comm, NULL) != ncclSuccess)
            return false;
        socklen_t ctrl_len = sizeof(peer->sender_ctrl);
        if (recvfrom(peer->fd, &hello, sizeof(hello), MSG_DONTWAIT, (struct sockaddr *)&peer->sender_ctrl, &ctrl_len) !=
            sizeof(hello))
            continue;
        peer->sender_ports[0] = hello.tx_ports_be[0];
        peer->sender_ports[1] = hello.tx_ports_be[1];
        test_ack_t ack = {.magic = h.magic,
                          .pkt_type = 2u,
                          .num_ports = h.num_ports,
                          .cookie = h.cookie,
                          .rx_ctrl_port_be = addr.sin_port,
                          .conn_id = hello.conn_id,
                          .rx_ports_be = {addr.sin_port, addr.sin_port}};
        if (sendto(peer->fd, &ack, sizeof(ack), 0, (struct sockaddr *)&peer->sender_ctrl, ctrl_len) != sizeof(ack))
            return false;
    }
    return peer->send_comm != NULL;
}

static bool send_raw_feedback(const raw_feedback_peer_t *peer, const test_wire_hdr_t *hdr, bool done, uint32_t rx_pkts,
                              uint32_t ce_pkts, uint32_t port_mask)
{
    const struct {
        uint32_t magic, cookie;
        uint8_t pkt_type, flags;
        uint16_t sack_cnt;
        uint32_t msg_seq, tag, K, rx_pkts, ecn_ce_pkts, ecn_port_mask, esi, ir_count, dwell_us;
    } ctrl = {.magic = hdr->magic,
              .cookie = hdr->cookie,
              .pkt_type = done ? 7u : 10u,
              .msg_seq = hdr->msg_seq,
              .tag = hdr->tag,
              .K = hdr->K,
              .rx_pkts = rx_pkts,
              .ecn_ce_pkts = ce_pkts,
              .ecn_port_mask = port_mask,
              .esi = done ? 0u : rx_pkts};
    return sendto(peer->fd, &ctrl, sizeof(ctrl), 0, (const struct sockaddr *)&peer->sender_ctrl, sizeof(peer->sender_ctrl)) ==
           sizeof(ctrl);
}

static bool observe_raw_send(const ncclNet_v8_t *net, const raw_feedback_peer_t *peer, int tag, test_wire_hdr_t *wire,
                             uint32_t by_port[2])
{
    static uint8_t data[128 * 1024];
    void *req = NULL;
    if (net->isend(peer->send_comm, data, sizeof(data), tag, NULL, &req) != ncclSuccess || !req)
        return false;
    by_port[0] = by_port[1] = 0u;
    for (int i = 0; i < 50000; i++) {
        test_wire_hdr_t hdr;
        struct sockaddr_in from;
        socklen_t len = sizeof(from);
        ssize_t n = recvfrom(peer->fd, &hdr, sizeof(hdr), MSG_DONTWAIT, (struct sockaddr *)&from, &len);
        if (n == sizeof(hdr) && hdr.tag == (uint32_t)tag && hdr.pkt_type == 3u && hdr.esi < hdr.K) {
            if (from.sin_port == peer->sender_ports[0])
                by_port[0]++;
            else if (from.sin_port == peer->sender_ports[1])
                by_port[1]++;
            else
                return false;
            if (by_port[0] + by_port[1] == hdr.K) {
                *wire = hdr;
                return true;
            }
        } else if (n < 0 && (i & 63) == 0) {
            int done = 0;
            if (net->test(req, &done, NULL) != ncclSuccess || done)
                return false;
        }
    }
    return false;
}

static bool test_fresh_ecn_feedback(const ncclNet_v8_t *net)
{
    raw_feedback_peer_t peer = {.fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0)};
    int rcvbuf = 1024 * 1024;
    bool ok = false;
    uint32_t observed[2] = {0};
    test_wire_hdr_t original, probe;
    if (peer.fd >= 0)
        (void)setsockopt(peer.fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    if (!setup_raw_feedback_peer(net, &peer) || !observe_raw_send(net, &peer, 1, &original, observed) || original.K < 15u)
        goto out;

    /* two fresh marks on port 0 halve its weight twice; duplicate ACK adds no penalty */
    if (!send_raw_feedback(&peer, &original, false, 1u, 1u, 1u) || !send_raw_feedback(&peer, &original, false, 1u, 1u, 1u) ||
        !send_raw_feedback(&peer, &original, false, 2u, 2u, 1u) || !observe_raw_send(net, &peer, 2, &probe, observed) ||
        observed[0] < probe.K / 8u || observed[0] > probe.K / 4u + 1u)
        goto out;

    /* mask stays set on clean reports while unchanged CE count lets port 0 recover */
    for (uint32_t i = 3u; i <= 14u; i++) {
        if (!send_raw_feedback(&peer, &original, false, i, 2u, 1u))
            goto out;
    }
    if (!send_raw_feedback(&peer, &original, false, 1u, 1u, 1u) || !send_raw_feedback(&peer, &original, true, original.K, 2u, 1u) ||
        !observe_raw_send(net, &peer, 3, &probe, observed))
        goto out;
    ok = observed[0] == probe.K / 2u || observed[0] == (probe.K + 1u) / 2u;

out:
    if (peer.send_comm)
        net->closeSend(peer.send_comm);
    if (peer.fd >= 0)
        close(peer.fd);
    if (!ok)
        fprintf(stderr, "FAIL: fresh ECN feedback test (port counts %u/%u)\n", observed[0], observed[1]);
    else
        printf("PASS: only fresh ECN marks deflect paths; clean ACKs restore SWRR weights\n");
    return ok;
}

/* [P2] regression: honor low (< 32 Mbps) rate caps and refresh clock before tail-RTO check */
static bool test_low_rate_cap_and_tail_clock(const ncclNet_v8_t *net, void (*set_cong)(uint32_t, uint32_t, uint64_t))
{
    raw_feedback_peer_t peer = {.fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0)};
    int rcvbuf = 1024 * 1024;
    bool ok = false;
    void *ref_req = NULL, *paced_req = NULL;
    static uint8_t payload[65536];

    if (setenv("VXFEC_NCCL_CWND_KB", "32", 1) != 0 || setenv("VXFEC_NCCL_MTU_SYM", "1280", 1) != 0)
        goto out;
    set_cong(0u, 0u, 8u); /* 8 Mbps = 1,000,000 B/s (< 32 Mbps) */

    if (peer.fd >= 0)
        (void)setsockopt(peer.fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    if (!setup_raw_feedback_peer(net, &peer))
        goto out;

    /* measure baseline tail copies emitted by a single unpaced 1 KiB send */
    if (net->isend(peer.send_comm, payload, 1024, 901, NULL, &ref_req) != ncclSuccess || !ref_req)
        goto out;
    uint32_t ref_tails = 0u;
    test_wire_hdr_t ref_hdr = {0};
    for (int i = 0; i < 4000; i++) {
        test_wire_hdr_t hdr;
        ssize_t n = recv(peer.fd, &hdr, sizeof(hdr), MSG_DONTWAIT);
        if (n < 0)
            break;
        if (n >= (ssize_t)sizeof(hdr) && hdr.tag == 901u) {
            ref_hdr = hdr;
            if (hdr.pkt_type == 4u)
                ref_tails++;
        }
    }
    if (ref_tails == 0u || !send_raw_feedback(&peer, &ref_hdr, true, 1u, 0u, 0u))
        goto out;
    int rdone = 0;
    for (int spin = 0; spin < 10000 && !rdone; spin++) {
        if (net->test(ref_req, &rdone, NULL) != ncclSuccess)
            goto out;
    }
    if (!rdone)
        goto out;

    /* send 64 KiB (K=52) into ~32 KiB cwnd at 8 Mbps: initial burst drains ~25 pkts and exhausts burst tokens */
    if (net->isend(peer.send_comm, payload, 65536, 902, NULL, &paced_req) != ncclSuccess || !paced_req)
        goto out;
    uint32_t init_data = 0u, init_tails = 0u;
    test_wire_hdr_t paced_hdr = {0};
    for (int i = 0; i < 4000; i++) {
        test_wire_hdr_t hdr;
        ssize_t n = recv(peer.fd, &hdr, sizeof(hdr), MSG_DONTWAIT);
        if (n < 0)
            break;
        if (n >= (ssize_t)sizeof(hdr) && hdr.tag == 902u) {
            paced_hdr = hdr;
            if (hdr.pkt_type == 3u)
                init_data++;
            else if (hdr.pkt_type == 4u)
                init_tails++;
        }
    }
    if (init_data == 0u || init_data + 15u >= paced_hdr.K || init_tails != 0u)
        goto out;

    /* open cwnd via WIN_ACK and sleep 3 ms: at 8 Mbps only ~3 KiB refills (vs 75 KiB if inflated to 200 Mbps) */
    if (!send_raw_feedback(&peer, &paced_hdr, false, init_data, 0u, 0u))
        goto out;
    struct timespec ts_short = {.tv_sec = 0, .tv_nsec = 3000000L};
    nanosleep(&ts_short, NULL);

    int pdone = 0;
    if (net->test(paced_req, &pdone, NULL) != ncclSuccess || pdone)
        goto out;
    uint32_t mid_data = 0u, mid_tails = 0u;
    for (int i = 0; i < 4000; i++) {
        test_wire_hdr_t hdr;
        ssize_t n = recv(peer.fd, &hdr, sizeof(hdr), MSG_DONTWAIT);
        if (n < 0)
            break;
        if (n >= (ssize_t)sizeof(hdr) && hdr.tag == 902u) {
            if (hdr.pkt_type == 3u)
                mid_data++;
            else if (hdr.pkt_type == 4u)
                mid_tails++;
        }
    }
    if (init_data + mid_data >= paced_hdr.K || mid_tails != 0u) {
        fprintf(stderr, "FAIL: 8 Mbps rate cap was not honored (sent %u/%u pkts in 3 ms)\n", init_data + mid_data, paced_hdr.K);
        goto out;
    }

    /* sleep 45 ms so remaining ~30 KiB tokens refill, then verify single progress tick does NOT double-send PKT_TAIL */
    struct timespec ts_rest = {.tv_sec = 0, .tv_nsec = 45000000L};
    nanosleep(&ts_rest, NULL);
    if (net->test(paced_req, &pdone, NULL) != ncclSuccess || pdone)
        goto out;
    uint32_t fin_data = 0u, fin_tails = 0u;
    for (int i = 0; i < 4000; i++) {
        test_wire_hdr_t hdr;
        ssize_t n = recv(peer.fd, &hdr, sizeof(hdr), MSG_DONTWAIT);
        if (n < 0)
            break;
        if (n >= (ssize_t)sizeof(hdr) && hdr.tag == 902u) {
            if (hdr.pkt_type == 3u)
                fin_data++;
            else if (hdr.pkt_type == 4u)
                fin_tails++;
        }
    }
    if (init_data + mid_data + fin_data != paced_hdr.K || fin_tails != ref_tails) {
        fprintf(stderr, "FAIL: tail clock check failed (data=%u/%u fin_tails=%u ref_tails=%u)\n", init_data + mid_data + fin_data,
                paced_hdr.K, fin_tails, ref_tails);
        goto out;
    }
    if (!send_raw_feedback(&peer, &paced_hdr, true, paced_hdr.K, 0u, 0u))
        goto out;
    for (int spin = 0; spin < 10000 && !pdone; spin++) {
        if (net->test(paced_req, &pdone, NULL) != ncclSuccess)
            goto out;
    }
    ok = (pdone == 1);

out:
    unsetenv("VXFEC_NCCL_CWND_KB");
    unsetenv("VXFEC_NCCL_MTU_SYM");
    set_cong(0u, 0u, 0u);
    if (peer.send_comm)
        net->closeSend(peer.send_comm);
    if (peer.fd >= 0)
        close(peer.fd);
    if (ok)
        printf("PASS: 8 Mbps low pacing cap honored and fresh clock prevents spurious immediate tail retry\n");
    return ok;
}

static bool expect_raw_completion(const ncclNet_v8_t *net, int ctrl_fd, void *probe, const test_wire_hdr_t *wire,
                                  uint8_t expected_type)
{
    struct {
        uint32_t magic;
        uint32_t cookie;
        uint8_t pkt_type;
        uint8_t flags;
        uint16_t sack_cnt;
        uint32_t msg_seq;
        uint32_t tag;
    } reply;
    bool matched = false;
    for (int spin = 0; spin < 50000; spin++) {
        ssize_t n;
        while ((n = recv(ctrl_fd, &reply, sizeof(reply), MSG_DONTWAIT)) >= 0) {
            if (n != (ssize_t)sizeof(reply) || reply.magic != wire->magic || reply.cookie != wire->cookie ||
                reply.msg_seq != wire->msg_seq || reply.tag != wire->tag || (reply.pkt_type != 7u && reply.pkt_type != 9u))
                continue;
            if (reply.pkt_type != expected_type)
                return false;
            matched = true;
        }
        if (matched)
            return true;
        int done = 0;
        if (!probe || net->test(probe, &done, NULL) != ncclSuccess || done)
            return false;
    }
    return false;
}

static bool test_completion_replay(const ncclNet_v8_t *net, void *rcomm, int tx_fd, int ctrl_fd, const struct sockaddr_in *rx_addr,
                                   test_wire_hdr_t wire)
{
    uint8_t data = 0;
    uint8_t packet[sizeof(wire) + 1u];
    wire.total_bytes = 1u;
    wire.sym_len = 1u;
    void *data_ptr = &data, *gap = NULL;
    int tag = (int)wire.tag + 1;
    int cap = 1;
    if (net->irecv(rcomm, 1, &data_ptr, &cap, &tag, NULL, &gap) != ncclSuccess || !gap)
        return false;

    /* leave #2 unresolved while DONE #3 and ERR #4 are evicted from the 4096-slot ring */
    for (uint32_t seq = 3u; seq <= 4102u; seq++) {
        bool fail = seq == 4u;
        int next_tag = tag + 1;
        int next_cap = fail ? 0 : 1;
        void *req = NULL;
        wire.tag = (uint32_t)next_tag;
        wire.tag_seq = seq - 2u;
        wire.msg_seq = seq;
        if (net->irecv(rcomm, 1, &data_ptr, &next_cap, &next_tag, NULL, &req) != ncclSuccess || !req)
            return false;
        memcpy(packet, &wire, sizeof(wire));
        packet[sizeof(wire)] = 0xa5;
        if (sendto(tx_fd, packet, sizeof(packet), 0, (const struct sockaddr *)rx_addr, sizeof(*rx_addr)) != sizeof(packet))
            return false;
        int done = 0;
        ncclResult_t status = ncclSuccess;
        for (int spin = 0; spin < 50000 && !done && status == ncclSuccess; spin++)
            status = net->test(req, &done, NULL);
        if (fail ? (status != ncclInternalError || done) : (status != ncclSuccess || !done))
            return false;
        /* drain original replies to simulate loss at the sender */
        if (!expect_raw_completion(net, ctrl_fd, NULL, &wire, fail ? 9u : 7u))
            return false;
    }

    for (int pass = 0; pass < 2; pass++) {
        void *probe = gap;
        if (pass == 1) {
            wire.pkt_type = 3u; /* PKT_DATA */
            wire.msg_seq = 2u;
            wire.tag = (uint32_t)tag;
            wire.tag_seq = 1u;
            wire.sym_len = 1u;
            memcpy(packet, &wire, sizeof(wire));
            if (sendto(tx_fd, packet, sizeof(packet), 0, (const struct sockaddr *)rx_addr, sizeof(*rx_addr)) != sizeof(packet))
                return false;
            int done = 0;
            for (int spin = 0; spin < 50000 && !done; spin++) {
                if (net->test(gap, &done, NULL) != ncclSuccess)
                    return false;
            }
            int probe_tag = tag + 2;
            if (!done || net->irecv(rcomm, 1, &data_ptr, &cap, &probe_tag, NULL, &probe) != ncclSuccess || !probe)
                return false;
        }

        wire.pkt_type = 4u; /* PKT_TAIL */
        wire.sym_len = 0u;
        wire.tag = (uint32_t)(tag + 1);
        for (uint32_t seq = 3u; seq <= 6u; seq++) {
            wire.msg_seq = seq;
            wire.tag_seq = seq - 2u;
            if (sendto(tx_fd, &wire, sizeof(wire), 0, (const struct sockaddr *)rx_addr, sizeof(*rx_addr)) != sizeof(wire) ||
                !expect_raw_completion(net, ctrl_fd, probe, &wire, seq == 4u ? 9u : 7u)) {
                fprintf(stderr, "FAIL: completion replay for sequence %u with gap %s\n", seq, pass ? "closed" : "open");
                return false;
            }
        }
    }
    printf("PASS: Completion replay survives cache turnover with a receive gap and preserves receive errors\n");
    return true;
}

static bool test_truncated_sys_pkt_rejection(const ncclNet_v8_t *net)
{
    uint8_t raw_handle[NCCL_NET_HANDLE_MAXSIZE];
    void *lcomm = NULL, *rcomm = NULL, *rx_mr = NULL;
    int tx_fd = -1, ctrl_fd = -1;
    bool ok = false;

    if (net->listen(0, raw_handle, &lcomm) != ncclSuccess)
        return false;
    test_handle_t h;
    memcpy(&h, raw_handle, sizeof(h));

    tx_fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    ctrl_fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (tx_fd < 0 || ctrl_fd < 0)
        goto out;

    struct sockaddr_in sin_tx = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be, .sin_port = 0};
    struct sockaddr_in sin_ct = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be, .sin_port = 0};
    socklen_t slen = sizeof(sin_tx);
    if (bind(tx_fd, (struct sockaddr *)&sin_tx, sizeof(sin_tx)) != 0 ||
        getsockname(tx_fd, (struct sockaddr *)&sin_tx, &slen) != 0 ||
        bind(ctrl_fd, (struct sockaddr *)&sin_ct, sizeof(sin_ct)) != 0 ||
        getsockname(ctrl_fd, (struct sockaddr *)&sin_ct, &slen) != 0)
        goto out;

    test_hello_t hello;
    memset(&hello, 0, sizeof(hello));
    hello.magic = h.magic;
    hello.pkt_type = 1u; /* PKT_HELLO */
    hello.num_ports = h.num_ports;
    hello.cookie = h.cookie;
    hello.tx_ctrl_port_be = sin_ct.sin_port;
    hello.conn_id = 0x7777ULL;
    for (uint16_t i = 0; i < h.num_ports; i++)
        hello.tx_ports_be[i] = sin_tx.sin_port;

    struct sockaddr_in lc_addr = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be, .sin_port = h.ctrl_port_be};
    test_ack_t ack;
    memset(&ack, 0, sizeof(ack));
    bool got_ack = false;
    for (int spin = 0; spin < 50000 && (!rcomm || !got_ack); spin++) {
        if ((spin % 500) == 0)
            (void)sendto(ctrl_fd, &hello, sizeof(hello), MSG_DONTWAIT, (struct sockaddr *)&lc_addr, sizeof(lc_addr));
        if (!got_ack && recvfrom(ctrl_fd, &ack, sizeof(ack), MSG_DONTWAIT, NULL, NULL) == (ssize_t)sizeof(ack) &&
            ack.magic == h.magic)
            got_ack = true;
        if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess)
            goto out;
    }
    if (!rcomm || !got_ack || ack.features != 0u)
        goto out;

    uint8_t rx_buf[256], tx_ref[256];
    memset(rx_buf, 0xee, sizeof(rx_buf));
    for (size_t i = 0; i < sizeof(tx_ref); i++)
        tx_ref[i] = (uint8_t)(0x21u + i * 7u);

    void *rreq = NULL, *rx_ptr = rx_buf;
    int rx_cap = (int)sizeof(rx_buf), tag = 711;
    if (net->regMr(rcomm, rx_buf, sizeof(rx_buf), NCCL_PTR_HOST, &rx_mr) != ncclSuccess ||
        net->irecv(rcomm, 1, &rx_ptr, &rx_cap, &tag, &rx_mr, &rreq) != ncclSuccess)
        goto out;

    struct sockaddr_in rx_data0 = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be, .sin_port = ack.rx_ports_be[0]};
    uint8_t pkt[sizeof(test_wire_hdr_t) + 256];
    test_wire_hdr_t wh = {
        .magic = h.magic,
        .cookie = h.cookie,
        .pkt_type = 3u, /* PKT_DATA */
        .flags = 0x02u,
        .sym_len = 256u,
        .msg_seq = 1u,
        .tag_seq = 1u,
        .tag = (uint32_t)tag,
        .total_bytes = 256u,
        .K = 1u,
        .sym_T = 1280u,
        .srtt_us = 0u,
        .esi = 0u,
    };

    /* legacy HELLO negotiated no features; ignore even a complete GF(2) packet */
    wh.flags = 0x01u;
    memcpy(pkt, &wh, sizeof(wh));
    memset(pkt + sizeof(wh), 0xcc, sizeof(tx_ref));
    if (sendto(tx_fd, pkt, sizeof(pkt), MSG_DONTWAIT, (struct sockaddr *)&rx_data0, sizeof(rx_data0)) != sizeof(pkt))
        goto out;
    wh.flags = 0x02u;

    /* 1) header claims 256 B but datagram only carries 64 B of payload */
    memcpy(pkt, &wh, sizeof(wh));
    memcpy(pkt + sizeof(wh), tx_ref, 64u);
    (void)sendto(tx_fd, pkt, sizeof(wh) + 64u, MSG_DONTWAIT, (struct sockaddr *)&rx_data0, sizeof(rx_data0));

    /* 2) header claims sym_len = 64 B for a 256 B single-symbol message */
    wh.sym_len = 64u;
    memcpy(pkt, &wh, sizeof(wh));
    (void)sendto(tx_fd, pkt, sizeof(wh) + 64u, MSG_DONTWAIT, (struct sockaddr *)&rx_data0, sizeof(rx_data0));

    for (int spin = 0; spin < 500; spin++) {
        int rdone = 0, got_sz = 0;
        if (net->test(rreq, &rdone, &got_sz) != ncclSuccess || rdone != 0) {
            fprintf(stderr, "FAIL: truncated PKT_DATA prematurely completed irecv\n");
            goto out;
        }
    }
    for (size_t i = 0; i < sizeof(rx_buf); i++) {
        if (rx_buf[i] != 0xeeu) {
            fprintf(stderr, "FAIL: truncated PKT_DATA mutated rx_buf at byte %zu\n", i);
            goto out;
        }
    }

    /* 3) valid full 256 B PKT_DATA completes cleanly */
    wh.sym_len = 256u;
    memcpy(pkt, &wh, sizeof(wh));
    memcpy(pkt + sizeof(wh), tx_ref, 256u);
    (void)sendto(tx_fd, pkt, sizeof(wh) + 256u, MSG_DONTWAIT, (struct sockaddr *)&rx_data0, sizeof(rx_data0));

    int rdone = 0, got_sz = 0;
    for (int spin = 0; spin < 50000 && !rdone; spin++) {
        if (net->test(rreq, &rdone, &got_sz) != ncclSuccess)
            goto out;
    }
    if (!rdone || got_sz != 256 || memcmp(rx_buf, tx_ref, 256u) != 0)
        goto out;

    printf("PASS: Truncated systematic PKT_DATA rejected without stale-byte completion\n");
    if (!test_completion_replay(net, rcomm, tx_fd, ctrl_fd, &rx_data0, wh))
        goto out;

    /* verify existing fork senders (has_ibv=0, features=0) can still deliver multi-gen WIRE_FLAG_GEN_EXT */
    enum { FORK_SYM_T = 64, FORK_GEN0_B = 128 * FORK_SYM_T, FORK_TOTAL_B = FORK_GEN0_B + FORK_SYM_T };
    uint8_t fork_ref[FORK_TOTAL_B], fork_rx[FORK_TOTAL_B];
    for (size_t i = 0; i < sizeof(fork_ref); i++)
        fork_ref[i] = (uint8_t)((i * 43u + 17u) & 0xffu);
    memset(fork_rx, 0, sizeof(fork_rx));
    void *fork_req = NULL, *fork_ptr = fork_rx;
    int fork_cap = FORK_TOTAL_B, fork_tag = 799;
    if (net->irecv(rcomm, 1, &fork_ptr, &fork_cap, &fork_tag, NULL, &fork_req) != ncclSuccess || !fork_req)
        goto out;

    struct {
        test_wire_hdr_t base;
        uint32_t msg_total_bytes;
        uint16_t gen_idx;
        uint16_t num_gens;
        uint8_t pay[FORK_SYM_T];
    } fpkt;
    for (uint16_t g = 0; g < 2u; g++) {
        uint32_t gk = (g == 0u) ? 128u : 1u;
        uint32_t gb = gk * FORK_SYM_T;
        size_t base_off = (g == 0u) ? 0u : FORK_GEN0_B;
        for (uint32_t e = 0; e < gk; e++) {
            fpkt.base = (test_wire_hdr_t){
                .magic = h.magic,
                .cookie = h.cookie,
                .pkt_type = 3u,
                .flags = 0x42u, /* WIRE_FLAG_GEN_EXT | WIRE_FLAG_RS_CAUCHY8 */
                .sym_len = FORK_SYM_T,
                .msg_seq = 5000u + g,
                .tag_seq = 1u,
                .tag = (uint32_t)fork_tag,
                .total_bytes = gb,
                .K = gk,
                .sym_T = FORK_SYM_T,
                .srtt_us = 0u,
                .esi = e,
            };
            fpkt.msg_total_bytes = FORK_TOTAL_B;
            fpkt.gen_idx = g;
            fpkt.num_gens = 2u;
            memcpy(fpkt.pay, fork_ref + base_off + (size_t)e * FORK_SYM_T, FORK_SYM_T);
            if (sendto(tx_fd, &fpkt, sizeof(fpkt), 0, (struct sockaddr *)&rx_data0, sizeof(rx_data0)) != (ssize_t)sizeof(fpkt))
                goto out;
        }
    }
    int fdone = 0, fgot = 0;
    for (int spin = 0; spin < 50000 && !fdone; spin++) {
        if (net->test(fork_req, &fdone, &fgot) != ncclSuccess)
            goto out;
    }
    if (!fdone || fgot != FORK_TOTAL_B || memcmp(fork_rx, fork_ref, FORK_TOTAL_B) != 0) {
        fprintf(stderr, "FAIL: sliced WIRE_FLAG_GEN_EXT from legacy/fork sender rejected\n");
        goto out;
    }

    /* verify pre-merge main 88-byte HELLO (byte 5 = GF2, byte 6 = 0) is accepted and echoes 88-byte ACK */
    void *rcomm2 = NULL;
    test_hello_t main_hello = hello;
    main_hello.has_ibv = 1u; /* byte 5 held VXFEC_HELLO_GF2 in pre-merge main */
    main_hello.features = 0u;
    main_hello.conn_id = 0x7778ULL;
    test_ack_t main_ack;
    memset(&main_ack, 0, sizeof(main_ack));
    bool got_main_ack = false;
    for (int spin = 0; spin < 50000 && (!rcomm2 || !got_main_ack); spin++) {
        if ((spin % 500) == 0)
            (void)sendto(ctrl_fd, &main_hello, sizeof(main_hello), MSG_DONTWAIT, (struct sockaddr *)&lc_addr, sizeof(lc_addr));
        if (!got_main_ack &&
            recvfrom(ctrl_fd, &main_ack, sizeof(main_ack), MSG_DONTWAIT, NULL, NULL) == (ssize_t)sizeof(main_ack) &&
            main_ack.magic == h.magic && main_ack.conn_id == main_hello.conn_id)
            got_main_ack = true;
        if (!rcomm2 && net->accept(lcomm, &rcomm2, NULL) != ncclSuccess) {
            if (rcomm2)
                net->closeRecv(rcomm2);
            goto out;
        }
    }
#ifdef NANORQ_HAS_GF2_MODE
    uint8_t exp_main_gf2 = 1u;
#else
    uint8_t exp_main_gf2 = 0u;
#endif
    bool main_ok = rcomm2 && got_main_ack && main_ack.has_ibv == exp_main_gf2 && main_ack.features == exp_main_gf2;
    if (rcomm2)
        net->closeRecv(rcomm2);
    if (!main_ok) {
        fprintf(stderr, "FAIL: pre-merge main 88-byte HELLO (byte 5 = GF2) rejected or mis-negotiated\n");
        goto out;
    }

    /* verify truncated 88-byte HELLO claiming VXFEC_HELLO_IBV (0x02) is rejected, while full RDMA HELLO with
     * byte 5 = VXFEC_HELLO_IBV | VXFEC_HELLO_SLICE | GF2 (0235844 / 7e0b2c9 layout) negotiates features & UDP fallback */
    test_hello_t trunc_ibv = hello;
    trunc_ibv.has_ibv = 0x02u;
    trunc_ibv.features = 0u;
    trunc_ibv.conn_id = 0x7779ULL;
    (void)sendto(ctrl_fd, &trunc_ibv, sizeof(trunc_ibv), MSG_DONTWAIT, (struct sockaddr *)&lc_addr, sizeof(lc_addr));
    for (int spin = 0; spin < 500; spin++) {
        void *bad_rc = NULL;
        if (net->accept(lcomm, &bad_rc, NULL) != ncclSuccess || bad_rc != NULL) {
            if (bad_rc)
                net->closeRecv(bad_rc);
            fprintf(stderr, "FAIL: truncated 88-byte HELLO with VXFEC_HELLO_IBV was accepted\n");
            goto out;
        }
    }

    struct {
        test_hello_t base;
        vxfec_ibv_wire_info_t ibv_info;
    } rdma_hello;
    memset(&rdma_hello, 0, sizeof(rdma_hello));
    rdma_hello.base = hello;
    rdma_hello.base.has_ibv = (uint8_t)(0x02u | 0x04u | exp_main_gf2);
    rdma_hello.base.features = 0u;
    rdma_hello.base.conn_id = 0x777aULL;
    void *rcomm3 = NULL;
    test_ack_t rdma_ack;
    memset(&rdma_ack, 0, sizeof(rdma_ack));
    bool got_rdma_ack = false;
    for (int spin = 0; spin < 50000 && (!rcomm3 || !got_rdma_ack); spin++) {
        if ((spin % 500) == 0)
            (void)sendto(ctrl_fd, &rdma_hello, sizeof(rdma_hello), MSG_DONTWAIT, (struct sockaddr *)&lc_addr, sizeof(lc_addr));
        if (!got_rdma_ack &&
            recvfrom(ctrl_fd, &rdma_ack, sizeof(rdma_ack), MSG_DONTWAIT, NULL, NULL) == (ssize_t)sizeof(rdma_ack) &&
            rdma_ack.magic == h.magic && rdma_ack.conn_id == rdma_hello.base.conn_id)
            got_rdma_ack = true;
        if (!rcomm3 && net->accept(lcomm, &rcomm3, NULL) != ncclSuccess) {
            if (rcomm3)
                net->closeRecv(rcomm3);
            goto out;
        }
    }
    uint8_t exp_rdma_feat = (uint8_t)(0x04u | exp_main_gf2);
    bool rdma_ok = rcomm3 && got_rdma_ack && rdma_ack.has_ibv == exp_rdma_feat && rdma_ack.features == exp_rdma_feat;
    if (rcomm3)
        net->closeRecv(rcomm3);
    if (!rdma_ok) {
        fprintf(stderr, "FAIL: reconciled RDMA HELLO (byte 5 = IBV|SLICE|GF2) rejected or mis-negotiated\n");
        goto out;
    }

    printf("PASS: Fork, pre-merge main, and reconciled RDMA handshakes + sliced WIRE_FLAG_GEN_EXT accepted bit-exactly\n");
    ok = true;

out:
    if (rx_mr)
        net->deregMr(rcomm, rx_mr);
    if (rcomm)
        net->closeRecv(rcomm);
    if (lcomm)
        net->closeListen(lcomm);
    if (ctrl_fd >= 0)
        close(ctrl_fd);
    if (tx_fd >= 0)
        close(tx_fd);
    return ok;
}

/* [P1] regression: connect() must never write into opaque_handle and must allow shared handles */
static bool test_const_shared_connect_handle(const ncclNet_v8_t *net, void (*set_hs_fault)(uint32_t, uint32_t))
{
    long pg = sysconf(_SC_PAGESIZE);
    size_t map_sz = (pg > 0) ? (size_t)pg : 4096u;
    uint8_t *ro_page = (uint8_t *)mmap(NULL, map_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ro_page == MAP_FAILED)
        return false;

    void *lcomm = NULL, *scomm[2] = {NULL, NULL}, *rcomm[2] = {NULL, NULL};
    bool ok = false;
    if (net->listen(0, ro_page, &lcomm) != ncclSuccess)
        goto out;

    uint8_t snap[NCCL_NET_HANDLE_MAXSIZE];
    memcpy(snap, ro_page, sizeof(snap));
    if (mprotect(ro_page, map_sz, PROT_READ) != 0)
        goto out;

    set_hs_fault(2u, 0u);
    for (int spin = 0; spin < 200000 && (!scomm[0] || !scomm[1] || !rcomm[0] || !rcomm[1]); spin++) {
        for (int i = 0; i < 2; i++) {
            if (!scomm[i] && net->connect(0, ro_page, &scomm[i], NULL) != ncclSuccess)
                goto out;
            if (!rcomm[i] && net->accept(lcomm, &rcomm[i], NULL) != ncclSuccess)
                goto out;
        }
    }
    set_hs_fault(0u, 0u);
    if (!scomm[0] || !scomm[1] || !rcomm[0] || !rcomm[1] || memcmp(ro_page, snap, sizeof(snap)) != 0)
        goto out;

    printf("PASS: Read-only (PROT_READ) shared handle concurrent connect() succeeded without mutation\n");
    ok = true;

out:
    set_hs_fault(0u, 0u);
    for (int i = 0; i < 2; i++) {
        if (scomm[i])
            net->closeSend(scomm[i]);
        if (rcomm[i])
            net->closeRecv(rcomm[i]);
    }
    if (lcomm)
        net->closeListen(lcomm);
    munmap(ro_page, map_sz);
    return ok;
}

typedef struct {
    const ncclNet_v8_t *net;
    void *scomm;
    void *rcomm;
    int tid;
    bool ok;
} burst_race_worker_t;

static void *burst_race_main(void *arg)
{
    burst_race_worker_t *w = (burst_race_worker_t *)arg;
    const size_t sz = 16384u;
    uint8_t *tx = (uint8_t *)malloc(sz);
    uint8_t *rx = (uint8_t *)malloc(sz);
    void *tmr = NULL, *rmr = NULL;
    w->ok = false;
    if (!tx || !rx)
        goto done;
    if (w->net->regMr(w->scomm, tx, sz, NCCL_PTR_HOST, &tmr) != ncclSuccess ||
        w->net->regMr(w->rcomm, rx, sz, NCCL_PTR_HOST, &rmr) != ncclSuccess)
        goto done;

    for (int it = 0; it < 6; it++) {
        for (size_t i = 0; i < sz; i++)
            tx[i] = (uint8_t)((i * 37u + (size_t)w->tid * 19u + (size_t)it) & 0xffu);
        memset(rx, 0, sz);
        void *sreq = NULL, *rreq = NULL, *rptr = rx;
        int rcap = (int)sz, tag = 1200 + w->tid * 16 + it;
        if (w->net->irecv(w->rcomm, 1, &rptr, &rcap, &tag, &rmr, &rreq) != ncclSuccess ||
            w->net->isend(w->scomm, tx, (int)sz, tag, tmr, &sreq) != ncclSuccess)
            goto done;
        int sd = 0, rd = 0, gsz = 0;
        while (!sd || !rd) {
            if (!sd && w->net->test(sreq, &sd, NULL) != ncclSuccess)
                goto done;
            if (!rd && w->net->test(rreq, &rd, &gsz) != ncclSuccess)
                goto done;
        }
        if ((size_t)gsz != sz || memcmp(tx, rx, sz) != 0)
            goto done;
    }
    w->ok = true;

done:
    if (rmr)
        w->net->deregMr(w->rcomm, rmr);
    if (tmr)
        w->net->deregMr(w->scomm, tmr);
    free(rx);
    free(tx);
    return NULL;
}

/* [P1] regression: multi-threaded sim_should_drop_data burst counter must not underflow */
static bool test_multithreaded_burst_drop_safety(const ncclNet_v8_t *net, ring_comms_t *ring,
                                                 void (*set_fault)(uint32_t, uint32_t, bool))
{
    pthread_t thrs[NUM_RANKS];
    burst_race_worker_t args[NUM_RANKS];
    set_fault(120000u, 3u, true);

    int started = 0;
    for (int r = 0; r < NUM_RANKS; r++) {
        int dst = (r + 1) % NUM_RANKS;
        args[r].net = net;
        args[r].scomm = ring->send_comm[r];
        args[r].rcomm = ring->recv_comm[dst];
        args[r].tid = r;
        args[r].ok = false;
        if (pthread_create(&thrs[r], NULL, burst_race_main, &args[r]) != 0)
            break;
        started++;
    }
    bool all_ok = (started == NUM_RANKS);
    for (int r = 0; r < started; r++) {
        pthread_join(thrs[r], NULL);
        if (!args[r].ok)
            all_ok = false;
    }
    set_fault(0u, 1u, false);
    if (all_ok)
        printf("PASS: Multi-threaded burst fault injection CAS safety verified across %d concurrent threads\n", NUM_RANKS);
    return all_ok;
}

/* [P1] regression: per-tag sequence map must never evict across colliding or >64 tags */
static bool test_tag_collision_non_eviction(const ncclNet_v8_t *net)
{
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
    void *lcomm = NULL, *scomm = NULL, *rcomm = NULL;
    void *tx_mr = NULL, *rx_mr = NULL;
    bool ok = false;

    uint8_t tx_buf[128][64];
    uint8_t rx_buf[128][64];
    for (size_t i = 0; i < 128u; i++) {
        for (size_t j = 0; j < 64u; j++)
            tx_buf[i][j] = (uint8_t)((i * 31u + j * 13u + 7u) & 0xffu);
    }
    memset(rx_buf, 0, sizeof(rx_buf));

    if (net->listen(0, handle, &lcomm) != ncclSuccess)
        goto out;
    for (int spin = 0; spin < 100000 && (!scomm || !rcomm); spin++) {
        if (!scomm && net->connect(0, handle, &scomm, NULL) != ncclSuccess)
            goto out;
        if (!rcomm && net->accept(lcomm, &rcomm, NULL) != ncclSuccess)
            goto out;
    }
    if (!scomm || !rcomm)
        goto out;

    if (net->regMr(scomm, tx_buf, sizeof(tx_buf), NCCL_PTR_HOST, &tx_mr) != ncclSuccess ||
        net->regMr(rcomm, rx_buf, sizeof(rx_buf), NCCL_PTR_HOST, &rx_mr) != ncclSuccess)
        goto out;

    /* send tags 0, 64, ..., 512 ascending and post irecv descending */
    const int collide_tags[9] = {0, 64, 128, 192, 256, 320, 384, 448, 512};
    void *sreqs[9] = {0}, *rreqs[9] = {0};
    for (int i = 0; i < 9; i++) {
        if (net->isend(scomm, tx_buf[i], 64, collide_tags[i], tx_mr, &sreqs[i]) != ncclSuccess)
            goto out;
    }
    for (int i = 8; i >= 0; i--) {
        void *dst = rx_buf[i];
        int cap = 64, t = collide_tags[i];
        if (net->irecv(rcomm, 1, &dst, &cap, &t, &rx_mr, &rreqs[i]) != ncclSuccess)
            goto out;
    }

    int done_cnt = 0;
    for (int spin = 0; spin < 200000 && done_cnt < 18; spin++) {
        for (int i = 0; i < 9; i++) {
            if (sreqs[i]) {
                int sdone = 0;
                if (net->test(sreqs[i], &sdone, NULL) != ncclSuccess)
                    goto out;
                if (sdone) {
                    sreqs[i] = NULL;
                    done_cnt++;
                }
            }
            if (rreqs[i]) {
                int rdone = 0, got = 0;
                if (net->test(rreqs[i], &rdone, &got) != ncclSuccess)
                    goto out;
                if (rdone) {
                    if (got != 64 || memcmp(rx_buf[i], tx_buf[i], 64u) != 0)
                        goto out;
                    rreqs[i] = NULL;
                    done_cnt++;
                }
            }
        }
    }
    if (done_cnt != 18)
        goto out;

    /* grow past inline 64-tag capacity in opposite order, then reuse tag 0 and tag 512 */
    for (int i = 0; i < 96; i++) {
        int tag_tx = 1000 + i;
        int tag_rx = 1000 + (95 - i);
        void *sr = NULL, *rr = NULL;
        void *dst = rx_buf[95 - i];
        int cap = 64;
        memset(rx_buf[95 - i], 0, 64u);
        if (net->isend(scomm, tx_buf[i], 64, tag_tx, tx_mr, &sr) != ncclSuccess ||
            net->irecv(rcomm, 1, &dst, &cap, &tag_rx, &rx_mr, &rr) != ncclSuccess)
            goto out;
        sreqs[0] = sr;
        rreqs[0] = rr;
        /* complete in pairs by posting matching counterpart immediately */
        void *sr2 = NULL, *rr2 = NULL;
        void *dst2 = rx_buf[i];
        memset(rx_buf[i], 0, 64u);
        if (net->irecv(rcomm, 1, &dst2, &cap, &tag_tx, &rx_mr, &rr2) != ncclSuccess ||
            net->isend(scomm, tx_buf[95 - i], 64, tag_rx, tx_mr, &sr2) != ncclSuccess)
            goto out;
        int sub_done = 0;
        void *sub[4] = {sr, rr2, sr2, rr};
        for (int spin = 0; spin < 100000 && sub_done < 4; spin++) {
            for (int k = 0; k < 4; k++) {
                if (!sub[k])
                    continue;
                int d = 0, got = 0;
                if (net->test(sub[k], &d, (k & 1) ? &got : NULL) != ncclSuccess)
                    goto out;
                if (d) {
                    sub[k] = NULL;
                    sub_done++;
                }
            }
        }
        if (sub_done != 4 || memcmp(rx_buf[i], tx_buf[i], 64u) != 0 || memcmp(rx_buf[95 - i], tx_buf[95 - i], 64u) != 0)
            goto out;
    }

    const int reuse_tags[2] = {0, 512};
    for (int r = 0; r < 2; r++) {
        void *sr = NULL, *rr = NULL;
        void *dst = rx_buf[120 + r];
        int cap = 64, t = reuse_tags[r];
        memset(dst, 0xaa, 64u);
        if (net->isend(scomm, tx_buf[120 + r], 64, t, tx_mr, &sr) != ncclSuccess ||
            net->irecv(rcomm, 1, &dst, &cap, &t, &rx_mr, &rr) != ncclSuccess)
            goto out;
        int sdone = 0, rdone = 0, got = 0;
        for (int spin = 0; spin < 100000 && (!sdone || !rdone); spin++) {
            if (!sdone && net->test(sr, &sdone, NULL) != ncclSuccess)
                goto out;
            if (!rdone && net->test(rr, &rdone, &got) != ncclSuccess)
                goto out;
        }
        if (!sdone || !rdone || got != 64 || memcmp(dst, tx_buf[120 + r], 64u) != 0) {
            fprintf(stderr, "FAIL: reused tag %d stalled or mismatched after colliding/many-tag traffic\n", t);
            goto out;
        }
    }

    printf("PASS: Per-tag sequence non-eviction across colliding and >64 distinct tags\n");
    ok = true;

out:
    if (tx_mr)
        net->deregMr(scomm, tx_mr);
    if (rx_mr)
        net->deregMr(rcomm, rx_mr);
    if (scomm)
        net->closeSend(scomm);
    if (rcomm)
        net->closeRecv(rcomm);
    if (lcomm)
        net->closeListen(lcomm);
    return ok;
}

/* [P2] regression: lost HELLO_ACK must replay original recvComm even after >64 accepts */
static bool test_handshake_ack_loss_many_conns(const ncclNet_v8_t *net, void (*set_hs_fault)(uint32_t, uint32_t))
{
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < 4096u && rl.rlim_max > rl.rlim_cur) {
        rl.rlim_cur = (rl.rlim_max < 4096u) ? rl.rlim_max : 4096u;
        (void)setrlimit(RLIMIT_NOFILE, &rl);
    }

    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
    void *lcomm = NULL, *scomm0 = NULL, *rcomm0 = NULL;
    void *extra_rc[65] = {0};
    void *tx_mr = NULL, *rx_mr = NULL;
    bool ok = false;

    if (net->listen(0, handle, &lcomm) != ncclSuccess)
        goto out;

    /* drop connection 0's initial HELLO_ACK while accepting rcomm0 */
    set_hs_fault(0u, 1u);
    if (net->connect(0, handle, &scomm0, NULL) != ncclSuccess || scomm0 != NULL)
        goto out;
    for (int spin = 0; spin < 100000 && !rcomm0; spin++) {
        if (net->accept(lcomm, &rcomm0, NULL) != ncclSuccess)
            goto out;
    }
    set_hs_fault(0u, 0u);
    if (!rcomm0)
        goto out;

    /* accept 65 additional connections while connection 0's HELLO retry is still pending */
    for (int i = 0; i < 65; i++) {
        void *sc_i = NULL;
        for (int spin = 0; spin < 100000 && (!sc_i || !extra_rc[i]); spin++) {
            if (!sc_i && net->connect(0, handle, &sc_i, NULL) != ncclSuccess)
                goto out;
            if (!extra_rc[i] && net->accept(lcomm, &extra_rc[i], NULL) != ncclSuccess)
                goto out;
        }
        if (!sc_i || !extra_rc[i])
            goto out;
        net->closeSend(sc_i);
    }

    /* close half of extra recvComms before retry to exercise lc_reap_closed_locked */
    for (int i = 0; i < 32; i++) {
        net->closeRecv(extra_rc[i]);
        extra_rc[i] = NULL;
    }

    struct timespec ts = {.tv_sec = 0, .tv_nsec = 2000000L};
    nanosleep(&ts, NULL);

    void *spurious_rc = NULL;
    for (int spin = 0; spin < 100000 && !scomm0; spin++) {
        if (net->connect(0, handle, &scomm0, NULL) != ncclSuccess)
            goto out;
        if (!spurious_rc && net->accept(lcomm, &spurious_rc, NULL) != ncclSuccess)
            goto out;
    }
    if (!scomm0 || spurious_rc != NULL) {
        if (spurious_rc)
            net->closeRecv(spurious_rc);
        fprintf(stderr, "FAIL: retried HELLO after 65 accepts %s\n",
                spurious_rc ? "allocated duplicate recvComm" : "failed to connect");
        goto out;
    }

    /* close listener before closing remaining recvComms to exercise inverse teardown order */
    net->closeListen(lcomm);
    lcomm = NULL;

    uint8_t tx_payload[4096], rx_payload[4096];
    for (size_t i = 0; i < sizeof(tx_payload); i++)
        tx_payload[i] = (uint8_t)((i * 59u + 17u) & 0xffu);
    memset(rx_payload, 0, sizeof(rx_payload));

    if (net->regMr(scomm0, tx_payload, sizeof(tx_payload), NCCL_PTR_HOST, &tx_mr) != ncclSuccess ||
        net->regMr(rcomm0, rx_payload, sizeof(rx_payload), NCCL_PTR_HOST, &rx_mr) != ncclSuccess)
        goto out;

    void *sreq = NULL, *rreq = NULL, *dst = rx_payload;
    int cap = (int)sizeof(rx_payload), tag = 42;
    if (net->irecv(rcomm0, 1, &dst, &cap, &tag, &rx_mr, &rreq) != ncclSuccess ||
        net->isend(scomm0, tx_payload, (int)sizeof(tx_payload), tag, tx_mr, &sreq) != ncclSuccess)
        goto out;

    int sdone = 0, rdone = 0, got = 0;
    for (int spin = 0; spin < 100000 && (!sdone || !rdone); spin++) {
        if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
            goto out;
        if (!rdone && net->test(rreq, &rdone, &got) != ncclSuccess)
            goto out;
    }
    if (!sdone || !rdone || got != (int)sizeof(tx_payload) || memcmp(rx_payload, tx_payload, sizeof(tx_payload)) != 0)
        goto out;

    printf("PASS: Lost HELLO_ACK recovery across 66 concurrent connections and dual teardown order\n");
    ok = true;

out:
    set_hs_fault(0u, 0u);
    if (tx_mr)
        net->deregMr(scomm0, tx_mr);
    if (rx_mr)
        net->deregMr(rcomm0, rx_mr);
    if (scomm0)
        net->closeSend(scomm0);
    if (rcomm0)
        net->closeRecv(rcomm0);
    for (int i = 0; i < 65; i++) {
        if (extra_rc[i])
            net->closeRecv(extra_rc[i]);
    }
    if (lcomm)
        net->closeListen(lcomm);
    return ok;
}

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F')
        return 10 + (c - 'A');
    return -1;
}

static int run_sender_child_mode(const char *hex_handle, int child_idx)
{
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
    if (strlen(hex_handle) != NCCL_NET_HANDLE_MAXSIZE * 2u)
        return 2;
    for (size_t i = 0; i < NCCL_NET_HANDLE_MAXSIZE; i++) {
        int hi = hex_nibble(hex_handle[2u * i]);
        int lo = hex_nibble(hex_handle[2u * i + 1u]);
        if (hi < 0 || lo < 0)
            return 2;
        handle[i] = (uint8_t)((hi << 4) | lo);
    }

    const char *so_path = getenv("VXFEC_NCCL_SO_PATH");
    if (!so_path || !*so_path)
        so_path = "./libnccl-vxfec.so";
    void *dl = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!dl)
        return 3;

    const ncclNet_v8_t *net = (const ncclNet_v8_t *)dlsym(dl, "ncclNetPlugin_v8");
    if (!net || net->init(dummy_nccl_logger) != ncclSuccess) {
        dlclose(dl);
        return 4;
    }

    void *scomm = NULL, *tx_mr = NULL, *sreq = NULL;
    uint8_t tx_buf[1024];
    uint8_t seed = (child_idx == 0) ? 0x3cu : 0xc3u;
    for (size_t i = 0; i < sizeof(tx_buf); i++)
        tx_buf[i] = (uint8_t)((seed ^ (uint8_t)(i * 19u + 5u)) & 0xffu);

    int rc = 5;
    for (int spin = 0; spin < 200000 && !scomm; spin++) {
        if (net->connect(0, handle, &scomm, NULL) != ncclSuccess)
            goto out;
        if (!scomm) {
            struct timespec ts = {.tv_sec = 0, .tv_nsec = 100000L};
            nanosleep(&ts, NULL);
        }
    }
    if (!scomm)
        goto out;

    if (net->regMr(scomm, tx_buf, sizeof(tx_buf), NCCL_PTR_HOST, &tx_mr) != ncclSuccess)
        goto out;
    if (net->isend(scomm, tx_buf, (int)sizeof(tx_buf), 900, tx_mr, &sreq) != ncclSuccess)
        goto out;

    int sdone = 0;
    for (int spin = 0; spin < 200000 && !sdone; spin++) {
        if (net->test(sreq, &sdone, NULL) != ncclSuccess)
            goto out;
        if (!sdone) {
            struct timespec ts = {.tv_sec = 0, .tv_nsec = 100000L};
            nanosleep(&ts, NULL);
        }
    }
    if (sdone)
        rc = 0;

out:
    if (tx_mr)
        net->deregMr(scomm, tx_mr);
    if (scomm)
        net->closeSend(scomm);
    dlclose(dl);
    return rc;
}

/* [P1] regression: multi-process senders and identical synthetic conn_ids must not collide */
static bool test_cross_process_sender_connect(const ncclNet_v8_t *net, const char *self_exe)
{
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE];
    void *lcomm = NULL, *rcomm[2] = {NULL, NULL}, *rx_mr[2] = {NULL, NULL};
    pid_t pids[2] = {-1, -1};
    bool ok = false;

    if (net->listen(0, handle, &lcomm) != ncclSuccess)
        return false;

    char hex[NCCL_NET_HANDLE_MAXSIZE * 2u + 1u];
    for (size_t i = 0; i < NCCL_NET_HANDLE_MAXSIZE; i++)
        snprintf(&hex[2u * i], 3u, "%02x", handle[i]);

    for (int c = 0; c < 2; c++) {
        char idx_str[8];
        snprintf(idx_str, sizeof(idx_str), "%d", c);
        pid_t p = fork();
        if (p < 0)
            goto out_proc;
        if (p == 0) {
            execl(self_exe, self_exe, "--sender-child", hex, idx_str, (char *)NULL);
            _exit(127);
        }
        pids[c] = p;
    }

    for (int spin = 0; spin < 300000 && (!rcomm[0] || !rcomm[1]); spin++) {
        for (int c = 0; c < 2; c++) {
            if (!rcomm[c] && net->accept(lcomm, &rcomm[c], NULL) != ncclSuccess)
                goto out_proc;
        }
        if (!rcomm[0] || !rcomm[1]) {
            struct timespec ts = {.tv_sec = 0, .tv_nsec = 100000L};
            nanosleep(&ts, NULL);
        }
    }
    if (!rcomm[0] || !rcomm[1]) {
        fprintf(stderr, "FAIL: multi-process senders collided onto a single recvComm\n");
        goto out_proc;
    }

    uint8_t rx_buf[2][1024];
    memset(rx_buf, 0, sizeof(rx_buf));
    void *rreq[2] = {NULL, NULL};
    for (int c = 0; c < 2; c++) {
        void *dst = rx_buf[c];
        int cap = 1024, tag = 900;
        if (net->regMr(rcomm[c], rx_buf[c], 1024u, NCCL_PTR_HOST, &rx_mr[c]) != ncclSuccess ||
            net->irecv(rcomm[c], 1, &dst, &cap, &tag, &rx_mr[c], &rreq[c]) != ncclSuccess)
            goto out_proc;
    }

    int rdone[2] = {0, 0}, got[2] = {0, 0};
    for (int spin = 0; spin < 300000 && (!rdone[0] || !rdone[1]); spin++) {
        for (int c = 0; c < 2; c++) {
            if (!rdone[c] && net->test(rreq[c], &rdone[c], &got[c]) != ncclSuccess)
                goto out_proc;
        }
        if (!rdone[0] || !rdone[1]) {
            struct timespec ts = {.tv_sec = 0, .tv_nsec = 100000L};
            nanosleep(&ts, NULL);
        }
    }
    if (!rdone[0] || !rdone[1] || got[0] != 1024 || got[1] != 1024)
        goto out_proc;

    uint32_t seen_mask = 0u;
    for (int c = 0; c < 2; c++) {
        for (int pat = 0; pat < 2; pat++) {
            uint8_t seed = (pat == 0) ? 0x3cu : 0xc3u;
            bool match = true;
            for (size_t i = 0; i < 1024u; i++) {
                if (rx_buf[c][i] != (uint8_t)((seed ^ (uint8_t)(i * 19u + 5u)) & 0xffu)) {
                    match = false;
                    break;
                }
            }
            if (match)
                seen_mask |= (1u << pat);
        }
    }
    if (seen_mask != 0x3u)
        goto out_proc;

    ok = true;

out_proc:
    for (int c = 0; c < 2; c++) {
        if (pids[c] > 0) {
            int st = 0;
            if (waitpid(pids[c], &st, 0) != pids[c] || !WIFEXITED(st) || WEXITSTATUS(st) != 0)
                ok = false;
        }
        if (rx_mr[c])
            net->deregMr(rcomm[c], rx_mr[c]);
        if (rcomm[c])
            net->closeRecv(rcomm[c]);
    }
    if (lcomm)
        net->closeListen(lcomm);
    if (!ok)
        return false;

    /* also verify listener endpoint check rejects synthetic conn_id == 1 collisions */
    typedef struct {
        uint32_t magic;
        uint32_t ip_be;
        uint16_t ctrl_port_be;
        uint16_t num_ports;
        uint32_t cookie;
        uint64_t reserved;
    } raw_h_t;
    typedef struct {
        uint32_t magic;
        uint8_t pkt_type;
        uint8_t reserved;
        uint16_t num_ports;
        uint32_t cookie;
        uint16_t tx_ctrl_port_be;
        uint16_t pad0;
        uint64_t conn_id;
        uint16_t tx_ports_be[32];
    } raw_hello_t;
    typedef struct {
        uint32_t magic;
        uint8_t pkt_type;
        uint8_t reserved;
        uint16_t num_ports;
        uint32_t cookie;
        uint16_t rx_ctrl_port_be;
        uint16_t pad0;
        uint64_t conn_id;
        uint16_t rx_ports_be[32];
    } raw_ack_t;

    void *lc2 = NULL, *rc2[2] = {NULL, NULL};
    int cfd[2] = {-1, -1}, dfd[2] = {-1, -1};
    bool raw_ok = false;
    if (net->listen(0, handle, &lc2) != ncclSuccess)
        return false;
    raw_h_t rh;
    memcpy(&rh, handle, sizeof(rh));
    struct sockaddr_in lc_sin = {.sin_family = AF_INET, .sin_addr.s_addr = rh.ip_be, .sin_port = rh.ctrl_port_be};

    raw_hello_t hl[2];
    raw_ack_t ak[2];
    memset(hl, 0, sizeof(hl));
    memset(ak, 0, sizeof(ak));
    for (int s = 0; s < 2; s++) {
        cfd[s] = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        dfd[s] = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (cfd[s] < 0 || dfd[s] < 0)
            goto out_raw;
        struct sockaddr_in sc = {.sin_family = AF_INET, .sin_addr.s_addr = rh.ip_be, .sin_port = 0};
        struct sockaddr_in sd = sc;
        socklen_t sl = sizeof(sc);
        if (bind(cfd[s], (struct sockaddr *)&sc, sizeof(sc)) != 0 || getsockname(cfd[s], (struct sockaddr *)&sc, &sl) != 0 ||
            bind(dfd[s], (struct sockaddr *)&sd, sizeof(sd)) != 0 || getsockname(dfd[s], (struct sockaddr *)&sd, &sl) != 0)
            goto out_raw;
        hl[s].magic = rh.magic;
        hl[s].pkt_type = 1u;
        hl[s].num_ports = rh.num_ports;
        hl[s].cookie = rh.cookie;
        hl[s].tx_ctrl_port_be = sc.sin_port;
        hl[s].conn_id = 1ULL; /* identical conn_id across two distinct sender endpoints */
        for (uint16_t p = 0; p < rh.num_ports; p++)
            hl[s].tx_ports_be[p] = sd.sin_port;
    }

    bool got_ak[2] = {false, false};
    for (int spin = 0; spin < 100000 && (!rc2[0] || !rc2[1] || !got_ak[0] || !got_ak[1]); spin++) {
        if ((spin % 500) == 0) {
            for (int s = 0; s < 2; s++)
                (void)sendto(cfd[s], &hl[s], sizeof(hl[s]), MSG_DONTWAIT, (struct sockaddr *)&lc_sin, sizeof(lc_sin));
        }
        for (int s = 0; s < 2; s++) {
            if (!got_ak[s] && recvfrom(cfd[s], &ak[s], sizeof(ak[s]), MSG_DONTWAIT, NULL, NULL) == (ssize_t)sizeof(ak[s]) &&
                ak[s].magic == rh.magic)
                got_ak[s] = true;
            if (!rc2[s] && net->accept(lc2, &rc2[s], NULL) != ncclSuccess)
                goto out_raw;
        }
    }
    if (!rc2[0] || !rc2[1] || !got_ak[0] || !got_ak[1] || ak[0].rx_ports_be[0] == ak[1].rx_ports_be[0])
        goto out_raw;

    printf("PASS: Multi-process sender connect() + identical conn_id endpoint isolation\n");
    raw_ok = true;

out_raw:
    for (int s = 0; s < 2; s++) {
        if (rc2[s])
            net->closeRecv(rc2[s]);
        if (cfd[s] >= 0)
            close(cfd[s]);
        if (dfd[s] >= 0)
            close(dfd[s]);
    }
    if (lc2)
        net->closeListen(lc2);
    return raw_ok;
}

typedef struct {
    const ncclNet_v8_t *net;
    void *scomm;
    void *rcomm;
    int chan_id;
    size_t nbytes;
    bool ok;
} mchan_slice_arg_t;

static void *mchan_slice_worker(void *arg)
{
    mchan_slice_arg_t *a = (mchan_slice_arg_t *)arg;
    const ncclNet_v8_t *net = a->net;
    uint8_t *tx = (uint8_t *)malloc(a->nbytes);
    uint8_t *rx = (uint8_t *)calloc(1, a->nbytes);
    void *tmr = NULL, *rmr = NULL;
    bool ok = false;
    if (!tx || !rx)
        goto out;
    for (size_t i = 0; i < a->nbytes; i++)
        tx[i] = (uint8_t)((i * 131u + (size_t)a->chan_id * 37u + 19u) & 0xffu);

    if (net->regMr(a->scomm, tx, a->nbytes, NCCL_PTR_HOST, &tmr) != ncclSuccess ||
        net->regMr(a->rcomm, rx, a->nbytes, NCCL_PTR_HOST, &rmr) != ncclSuccess)
        goto out;

    int tag = 4200 + a->chan_id;
    int rsz = (int)a->nbytes;
    void *dst = rx;
    void *sreq = NULL, *rreq = NULL;
    while (!rreq) {
        if (net->irecv(a->rcomm, 1, &dst, &rsz, &tag, &rmr, &rreq) != ncclSuccess)
            goto out;
    }
    while (!sreq) {
        if (net->isend(a->scomm, tx, (int)a->nbytes, tag, tmr, &sreq) != ncclSuccess)
            goto out;
    }

    int sdone = 0, rdone = 0, got = 0;
    while (!sdone || !rdone) {
        if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
            goto out;
        if (!rdone && net->test(rreq, &rdone, &got) != ncclSuccess)
            goto out;
    }
    if ((size_t)got != a->nbytes)
        goto out;
    ok = (memcmp(tx, rx, a->nbytes) == 0);

out:
    if (tmr)
        net->deregMr(a->scomm, tmr);
    if (rmr)
        net->deregMr(a->rcomm, rmr);
    free(tx);
    free(rx);
    a->ok = ok;
    return NULL;
}

static bool test_rs8_subblock_slicing_and_multichannel(const ncclNet_v8_t *net, const ncclNetProperties_v8_t *props,
                                                       void (*get_stats)(vxfec_nccl_stats_t *), void (*reset_stats)(void),
                                                       void (*set_fault)(uint32_t, uint32_t, bool))
{
    ring_comms_t sring;
    if (!setup_ring(net, &sring))
        return false;

    /* 4 MiB per ring chunk (16 MiB tensor/rank, 64 MiB aggregate) under 2.0% burst=6 loss:
     * must slice into K <= 128 (1.0 MiB) Cauchy RS8 windows with 0 RaptorQ promotions */
    reset_stats();
    set_fault(20000, 6, true);
    if (!run_host_ring_allreduce(net, &sring, 1048576u, 77u)) {
        set_fault(0, 1, false);
        teardown_ring(net, &sring);
        fprintf(stderr, "FAIL: 64 MiB multi-gen RS8 Ring AllReduce failed\n");
        return false;
    }
    vxfec_nccl_stats_t st;
    get_stats(&st);
    if (st.rx_rs8_decodes == 0 || st.rx_rq_decodes != 0) {
        set_fault(0, 1, false);
        teardown_ring(net, &sring);
        fprintf(stderr, "FAIL: expected pure K<=128 RS8 slicing (rs8_dec=%lu, rq_dec=%lu)\n", (unsigned long)st.rx_rs8_decodes,
                (unsigned long)st.rx_rq_decodes);
        return false;
    }

    /* concurrent multi-channel striping across 4 pthreads x 4 MiB */
    pthread_t cthrs[NUM_RANKS];
    mchan_slice_arg_t cargs[NUM_RANKS];
    int cstarted = 0;
    for (int r = 0; r < NUM_RANKS; r++) {
        int dst = (r + 1) % NUM_RANKS;
        cargs[r].net = net;
        cargs[r].scomm = sring.send_comm[r];
        cargs[r].rcomm = sring.recv_comm[dst];
        cargs[r].chan_id = r;
        cargs[r].nbytes = 4u * 1024u * 1024u;
        cargs[r].ok = false;
        if (pthread_create(&cthrs[r], NULL, mchan_slice_worker, &cargs[r]) != 0)
            break;
        cstarted++;
    }
    bool chan_ok = (cstarted == NUM_RANKS);
    for (int r = 0; r < cstarted; r++) {
        pthread_join(cthrs[r], NULL);
        if (!cargs[r].ok)
            chan_ok = false;
    }
    vxfec_nccl_stats_t st_mc;
    get_stats(&st_mc);
    if (!chan_ok || st_mc.rx_rq_decodes != 0) {
        set_fault(0, 1, false);
        teardown_ring(net, &sring);
        fprintf(stderr, "FAIL: concurrent 4-channel 4 MiB RS8 slicing failed (ok=%d rq_dec=%lu)\n", chan_ok ? 1 : 0,
                (unsigned long)st_mc.rx_rq_decodes);
        return false;
    }

    /* multi-generation (4 MiB/chunk = 16 MiB/rank) GPU VRAM DMA-BUF Ring AllGather under 2.0% burst=6 */
    reset_stats();
    set_fault(20000, 6, true);
    if (!run_dmabuf_ring_allgather(net, props, &sring, 1048576u)) {
        set_fault(0, 1, false);
        teardown_ring(net, &sring);
        fprintf(stderr, "FAIL: multi-gen GPU VRAM DMA-BUF Ring AllGather failed\n");
        return false;
    }
    vxfec_nccl_stats_t st_gpu;
    get_stats(&st_gpu);
    set_fault(0, 1, false);
    teardown_ring(net, &sring);
    if ((props->ptrSupport & NCCL_PTR_DMABUF) && st_gpu.tx_messages > 0u &&
        (st_gpu.rx_fec_decodes == 0u || st_gpu.rx_rq_decodes != 0u)) {
        fprintf(stderr, "FAIL: GPU DMA-BUF multi-gen missed RS8 FEC repair or promoted to RQ (gpu_fec=%lu rq_dec=%lu)\n",
                (unsigned long)st_gpu.rx_fec_decodes, (unsigned long)st_gpu.rx_rq_decodes);
        return false;
    }

    printf("PASS: 64 MiB Sub-Block RS8 Slicing (K<=128, ir_slack=0) + 4-Channel Parallel + Multi-Gen DMA-BUF "
           "(rs8_dec=%lu rq_dec=%lu gpu_fec=%lu)\n",
           (unsigned long)st_mc.rx_rs8_decodes, (unsigned long)st_mc.rx_rq_decodes, (unsigned long)st_gpu.rx_fec_decodes);
    return true;
}

#ifdef VXFEC_HAVE_IBVERBS
/* in-process mock ibverbs provider for end-to-end 240B RDMA handshakes, zero-copy sends, and partial QP teardown */
enum { MOCK_CQ_RING = 8192, MOCK_RQ_RING = 1024, MOCK_MAX_QPS = 256 };

typedef struct {
    struct ibv_cq cq;
    struct ibv_wc wcs[MOCK_CQ_RING];
    uint32_t head;
    uint32_t cnt;
} mock_ibv_cq_t;

typedef struct {
    struct ibv_qp qp;
    uint32_t dest_qpn;
    enum ibv_qp_state state;
    uint64_t rq_wr_id[MOCK_RQ_RING];
    struct ibv_sge rq_sge[MOCK_RQ_RING];
    uint32_t rq_head;
    uint32_t rq_cnt;
} mock_ibv_qp_t;

typedef struct {
    bool used;
    uint32_t key;
    uint64_t base_addr;
    size_t length;
    vxfec_vk_buf *vk_buf;
} mock_ibv_mr_entry_t;

#define MOCK_MAX_MRS 128

static pthread_mutex_t g_mock_ibv_mtx = PTHREAD_MUTEX_INITIALIZER;
static struct ibv_device g_mock_ibv_dev;
static mock_ibv_qp_t *g_mock_qps[MOCK_MAX_QPS];
static mock_ibv_mr_entry_t g_mock_mrs[MOCK_MAX_MRS];
static struct {
    uint32_t next_qpn;
    uint32_t next_key;
    uint32_t post_sends;
    uint32_t max_send_bytes;
    uint32_t active_mrs;
    uint32_t busy_qpn;
    uint32_t busy_budget;
    uint32_t drop_tag;
    uint32_t drop_esi_beg;
    uint32_t drop_esi_cnt;
    uint32_t dropped_uc_pkts;
    uint32_t repair_uc_pkts;
    uint32_t rdma_write_imm_pkts;
    uint32_t drop_rdma_rem;
    uint32_t dropped_rdma_pkts;
    uint32_t recv_wqes_consumed;
    vxfec_vk_ctx *vk_ctx;
    struct ibv_cq *fail_poll_cq;
    bool inject_wc_err;
    bool ipv6_only_gid;
    bool rocev2_gid_matrix;
    bool dual_port;
    uint8_t port1_speed;
    uint8_t port2_speed;
} g_mock_ibv;

static const mock_ibv_mr_entry_t *mock_find_mr_locked(uint32_t key)
{
    if (key == 0u)
        return NULL;
    for (int i = 0; i < MOCK_MAX_MRS; i++) {
        if (g_mock_mrs[i].used && g_mock_mrs[i].key == key)
            return &g_mock_mrs[i];
    }
    return NULL;
}

static bool mock_mr_read_locked(uint32_t lkey, uint64_t addr, void *dst, size_t len)
{
    if (len == 0u)
        return true;
    const mock_ibv_mr_entry_t *e = mock_find_mr_locked(lkey);
    if (e && e->vk_buf && g_mock_ibv.vk_ctx) {
        if (addr < e->base_addr || (addr - e->base_addr) > e->length || len > e->length - (size_t)(addr - e->base_addr))
            return false;
        return vxfec_vk_buf_read_offset(g_mock_ibv.vk_ctx, e->vk_buf, (size_t)(addr - e->base_addr), dst, len);
    }
    memcpy(dst, (const void *)(uintptr_t)addr, len);
    return true;
}

static bool mock_mr_write_locked(uint32_t rkey, uint64_t addr, const void *src, size_t len)
{
    if (len == 0u)
        return true;
    const mock_ibv_mr_entry_t *e = mock_find_mr_locked(rkey);
    if (e && e->vk_buf && g_mock_ibv.vk_ctx) {
        if (addr < e->base_addr || (addr - e->base_addr) > e->length || len > e->length - (size_t)(addr - e->base_addr))
            return false;
        return vxfec_vk_buf_write_offset(g_mock_ibv.vk_ctx, e->vk_buf, (size_t)(addr - e->base_addr), src, len);
    }
    memcpy((void *)(uintptr_t)addr, src, len);
    return true;
}

static void mock_cq_push_locked(struct ibv_cq *raw_cq, struct ibv_wc wc)
{
    mock_ibv_cq_t *mcq = (mock_ibv_cq_t *)raw_cq;
    if (mcq && mcq->cnt < MOCK_CQ_RING) {
        uint32_t idx = (mcq->head + mcq->cnt++) % MOCK_CQ_RING;
        mcq->wcs[idx] = wc;
    }
}

static int mock_ibv_poll_cq(struct ibv_cq *raw_cq, int num_entries, struct ibv_wc *wc)
{
    mock_ibv_cq_t *mcq = (mock_ibv_cq_t *)raw_cq;
    if (!mcq || num_entries <= 0 || !wc)
        return 0;
    pthread_mutex_lock(&g_mock_ibv_mtx);
    if (g_mock_ibv.fail_poll_cq == raw_cq) {
        g_mock_ibv.fail_poll_cq = NULL;
        pthread_mutex_unlock(&g_mock_ibv_mtx);
        return -1;
    }
    int n = 0;
    while (n < num_entries && mcq->cnt > 0u) {
        wc[n++] = mcq->wcs[mcq->head];
        mcq->head = (mcq->head + 1u) % MOCK_CQ_RING;
        mcq->cnt--;
    }
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    return n;
}

static int mock_ibv_post_recv(struct ibv_qp *raw_qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr)
{
    mock_ibv_qp_t *mqp = (mock_ibv_qp_t *)raw_qp;
    pthread_mutex_lock(&g_mock_ibv_mtx);
    for (struct ibv_recv_wr *it = wr; it != NULL; it = it->next) {
        if (!mqp || mqp->rq_cnt >= MOCK_RQ_RING || (it->num_sge > 0 && !it->sg_list)) {
            if (bad_wr)
                *bad_wr = it;
            pthread_mutex_unlock(&g_mock_ibv_mtx);
            return ENOMEM;
        }
        uint32_t dst = (mqp->rq_head + mqp->rq_cnt++) % MOCK_RQ_RING;
        mqp->rq_wr_id[dst] = it->wr_id;
        memset(&mqp->rq_sge[dst], 0, sizeof(mqp->rq_sge[dst]));
        if (it->num_sge > 0)
            mqp->rq_sge[dst] = it->sg_list[0];
    }
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    return 0;
}

static int mock_ibv_post_send(struct ibv_qp *raw_qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr)
{
    mock_ibv_qp_t *mqp = (mock_ibv_qp_t *)raw_qp;
    pthread_mutex_lock(&g_mock_ibv_mtx);
    for (struct ibv_send_wr *it = wr; it != NULL; it = it->next) {
        if (!mqp || mqp->state != IBV_QPS_RTS) {
            if (bad_wr)
                *bad_wr = it;
            pthread_mutex_unlock(&g_mock_ibv_mtx);
            return EINVAL;
        }
        g_mock_ibv.post_sends++;
        if (g_mock_ibv.inject_wc_err) {
            g_mock_ibv.inject_wc_err = false;
            struct ibv_wc err_wc;
            memset(&err_wc, 0, sizeof(err_wc));
            err_wc.wr_id = it->wr_id;
            err_wc.status = IBV_WC_REM_ABORT_ERR;
            err_wc.opcode = IBV_WC_SEND;
            mock_cq_push_locked(raw_qp->send_cq, err_wc);
            continue;
        }

        uint32_t total_len = 0u;
        for (int s = 0; s < it->num_sge; s++)
            total_len += it->sg_list[s].length;
        if (total_len > g_mock_ibv.max_send_bytes)
            g_mock_ibv.max_send_bytes = total_len;

        if (mqp->dest_qpn != 0u && it->opcode == IBV_WR_SEND) {
            bool drop_uc = false;
            if (g_mock_ibv.drop_tag != 0u && it->num_sge > 0 && it->sg_list[0].length >= sizeof(test_wire_hdr_t)) {
                test_wire_hdr_t wh;
                memcpy(&wh, (const void *)(uintptr_t)it->sg_list[0].addr, sizeof(wh));
                if (wh.pkt_type == 3u && wh.tag == g_mock_ibv.drop_tag) {
                    if (wh.esi >= g_mock_ibv.drop_esi_beg && wh.esi < g_mock_ibv.drop_esi_beg + g_mock_ibv.drop_esi_cnt) {
                        g_mock_ibv.dropped_uc_pkts++;
                        drop_uc = true;
                    } else if (wh.esi >= wh.K) {
                        g_mock_ibv.repair_uc_pkts++;
                    }
                }
            }
            mock_ibv_qp_t *peer = NULL;
            for (int i = 0; !drop_uc && i < MOCK_MAX_QPS; i++) {
                if (g_mock_qps[i] && g_mock_qps[i]->qp.qp_num == mqp->dest_qpn) {
                    peer = g_mock_qps[i];
                    break;
                }
            }
            if (peer && peer->state >= IBV_QPS_RTR && peer->rq_cnt > 0u) {
                uint32_t ridx = peer->rq_head;
                peer->rq_head = (peer->rq_head + 1u) % MOCK_RQ_RING;
                peer->rq_cnt--;
                g_mock_ibv.recv_wqes_consumed++;
                struct ibv_sge rsge = peer->rq_sge[ridx];
                if (total_len <= rsge.length) {
                    uint8_t *dst = (uint8_t *)(uintptr_t)rsge.addr;
                    size_t off = 0u;
                    bool copy_ok = true;
                    for (int s = 0; s < it->num_sge; s++) {
                        if (it->sg_list[s].length > 0u) {
                            if (!mock_mr_read_locked(it->sg_list[s].lkey, it->sg_list[s].addr, dst + off, it->sg_list[s].length)) {
                                copy_ok = false;
                                break;
                            }
                            off += it->sg_list[s].length;
                        }
                    }
                    struct ibv_wc rwc;
                    memset(&rwc, 0, sizeof(rwc));
                    rwc.wr_id = peer->rq_wr_id[ridx];
                    rwc.status = copy_ok ? IBV_WC_SUCCESS : IBV_WC_LOC_PROT_ERR;
                    rwc.opcode = IBV_WC_RECV;
                    rwc.byte_len = total_len;
                    mock_cq_push_locked(peer->qp.recv_cq, rwc);
                }
            }
        } else if (mqp->dest_qpn != 0u && it->opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
            if (g_mock_ibv.drop_rdma_rem > 0u) {
                g_mock_ibv.drop_rdma_rem--;
                g_mock_ibv.dropped_rdma_pkts++;
            } else {
                mock_ibv_qp_t *peer = NULL;
                for (int i = 0; i < MOCK_MAX_QPS; i++) {
                    if (g_mock_qps[i] && g_mock_qps[i]->qp.qp_num == mqp->dest_qpn) {
                        peer = g_mock_qps[i];
                        break;
                    }
                }
                if (peer && peer->state >= IBV_QPS_RTR && peer->rq_cnt > 0u) {
                    uint32_t ridx = peer->rq_head;
                    peer->rq_head = (peer->rq_head + 1u) % MOCK_RQ_RING;
                    peer->rq_cnt--;
                    g_mock_ibv.recv_wqes_consumed++;
                    g_mock_ibv.rdma_write_imm_pkts++;

                    bool copy_ok = true;
                    if (total_len > 0u) {
                        uint8_t *tmp = (uint8_t *)malloc(total_len);
                        if (!tmp) {
                            copy_ok = false;
                        } else {
                            size_t off = 0u;
                            for (int s = 0; s < it->num_sge; s++) {
                                if (it->sg_list[s].length > 0u) {
                                    if (!mock_mr_read_locked(it->sg_list[s].lkey, it->sg_list[s].addr, tmp + off,
                                                             it->sg_list[s].length)) {
                                        copy_ok = false;
                                        break;
                                    }
                                    off += it->sg_list[s].length;
                                }
                            }
                            if (copy_ok)
                                copy_ok = mock_mr_write_locked(it->wr.rdma.rkey, it->wr.rdma.remote_addr, tmp, total_len);
                            free(tmp);
                        }
                    }
                    struct ibv_wc rwc;
                    memset(&rwc, 0, sizeof(rwc));
                    rwc.wr_id = peer->rq_wr_id[ridx];
                    rwc.status = copy_ok ? IBV_WC_SUCCESS : IBV_WC_REM_ACCESS_ERR;
                    rwc.opcode = IBV_WC_RECV_RDMA_WITH_IMM;
                    rwc.wc_flags = IBV_WC_WITH_IMM;
                    rwc.imm_data = it->imm_data;
                    rwc.byte_len = total_len;
                    mock_cq_push_locked(peer->qp.recv_cq, rwc);
                }
            }
        }

        if (it->send_flags & IBV_SEND_SIGNALED) {
            struct ibv_wc swc;
            memset(&swc, 0, sizeof(swc));
            swc.wr_id = it->wr_id;
            swc.status = IBV_WC_SUCCESS;
            swc.opcode = IBV_WC_SEND;
            swc.byte_len = total_len;
            mock_cq_push_locked(raw_qp->send_cq, swc);
        }
    }
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    return 0;
}

static int mock_ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr, size_t attr_sz)
{
    (void)context;
    (void)attr_sz;
    memset(port_attr, 0, sizeof(*port_attr));
    if (port_num != 1u && !(g_mock_ibv.dual_port && port_num == 2u)) {
        port_attr->state = IBV_PORT_DOWN;
        return 0;
    }
    port_attr->state = IBV_PORT_ACTIVE;
    port_attr->max_mtu = IBV_MTU_4096;
    port_attr->active_mtu = IBV_MTU_4096;
    port_attr->gid_tbl_len = g_mock_ibv.rocev2_gid_matrix ? 4 : 2;
    port_attr->lid = port_num;
    port_attr->active_width = 2u;
    if (port_num == 2u && g_mock_ibv.port2_speed != 0u)
        port_attr->active_speed = g_mock_ibv.port2_speed;
    else if (port_num == 1u && g_mock_ibv.port1_speed != 0u)
        port_attr->active_speed = g_mock_ibv.port1_speed;
    else
        port_attr->active_speed = 64u;
    port_attr->link_layer = IBV_LINK_LAYER_ETHERNET;
    return 0;
}

__attribute__((visibility("default"))) struct ibv_device **vxfec_test_ibv_get_device_list(int *num_devices)
{
    struct ibv_device **list = (struct ibv_device **)calloc(2, sizeof(*list));
    if (!list)
        return NULL;
    snprintf(g_mock_ibv_dev.name, sizeof(g_mock_ibv_dev.name), "mock_rxe0");
    list[0] = &g_mock_ibv_dev;
    if (num_devices)
        *num_devices = 1;
    return list;
}

__attribute__((visibility("default"))) void vxfec_test_ibv_free_device_list(struct ibv_device **list)
{
    free(list);
}

__attribute__((visibility("default"))) const char *vxfec_test_ibv_get_device_name(struct ibv_device *device)
{
    return device ? device->name : NULL;
}

__attribute__((visibility("default"))) uint64_t vxfec_test_ibv_get_device_guid(struct ibv_device *device)
{
    (void)device;
    return 0x56584d4f434b3031ULL;
}

__attribute__((visibility("default"))) struct ibv_context *vxfec_test_ibv_open_device(struct ibv_device *device)
{
    struct verbs_context *vctx = (struct verbs_context *)calloc(1, sizeof(*vctx));
    if (!vctx)
        return NULL;
    vctx->sz = sizeof(*vctx);
    vctx->query_port = mock_ibv_query_port;
    vctx->context.device = device;
    vctx->context.abi_compat = __VERBS_ABI_IS_EXTENDED;
    vctx->context.ops.poll_cq = mock_ibv_poll_cq;
    vctx->context.ops.post_send = mock_ibv_post_send;
    vctx->context.ops.post_recv = mock_ibv_post_recv;
    return &vctx->context;
}

__attribute__((visibility("default"))) int vxfec_test_ibv_close_device(struct ibv_context *context)
{
    if (!context)
        return 0;
    struct verbs_context *vctx = verbs_get_ctx(context);
    free(vctx);
    return 0;
}

__attribute__((visibility("default"))) int vxfec_test_ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index,
                                                                    union ibv_gid *gid)
{
    (void)context;
    memset(gid, 0, sizeof(*gid));
    int max_idx = g_mock_ibv.rocev2_gid_matrix ? 3 : 1;
    if ((port_num != 1u && !(g_mock_ibv.dual_port && port_num == 2u)) || index < 0 || index > max_idx)
        return -1;
    if (g_mock_ibv.rocev2_gid_matrix) {
        gid->raw[10] = 0xffu;
        gid->raw[11] = 0xffu;
        if (index == 0 || index == 1) {
            /* idx 0 = RoCE v1 10.20.30.40, idx 1 = RoCE v2 10.20.30.40 */
            gid->raw[12] = 10u;
            gid->raw[13] = 20u;
            gid->raw[14] = 30u;
            gid->raw[15] = 40u;
        } else {
            /* idx 2,3 = link-local 169.254.1.2 (must not overwrite routable idx 1) */
            gid->raw[12] = 169u;
            gid->raw[13] = 254u;
            gid->raw[14] = 1u;
            gid->raw[15] = 2u;
        }
        return 0;
    }
    if (g_mock_ibv.ipv6_only_gid) {
        gid->raw[0] = 0xfeu;
        gid->raw[1] = 0x80u;
        gid->raw[15] = (uint8_t)(index + 1);
        return 0;
    }
    gid->raw[10] = 0xffu;
    gid->raw[11] = 0xffu;
    gid->raw[12] = 127u;
    gid->raw[15] = 1u;
    return 0;
}

__attribute__((visibility("default"))) struct ibv_pd *vxfec_test_ibv_alloc_pd(struct ibv_context *context)
{
    struct ibv_pd *pd = (struct ibv_pd *)calloc(1, sizeof(*pd));
    if (pd)
        pd->context = context;
    return pd;
}

__attribute__((visibility("default"))) int vxfec_test_ibv_dealloc_pd(struct ibv_pd *pd)
{
    free(pd);
    return 0;
}

__attribute__((visibility("default"))) struct ibv_mr *vxfec_test_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
                                                                            int access)
{
    (void)access;
    struct ibv_mr *mr = (struct ibv_mr *)calloc(1, sizeof(*mr));
    if (!mr)
        return NULL;
    pthread_mutex_lock(&g_mock_ibv_mtx);
    uint32_t key = ++g_mock_ibv.next_key;
    g_mock_ibv.active_mrs++;
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    mr->context = pd->context;
    mr->pd = pd;
    mr->addr = addr;
    mr->length = length;
    mr->lkey = key;
    mr->rkey = key;
    return mr;
}

__attribute__((visibility("default"))) struct ibv_mr *vxfec_test_ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset,
                                                                                   size_t length, uint64_t iova, int fd, int access)
{
    if (fd < 0)
        return NULL;
    struct ibv_mr *mr = vxfec_test_ibv_reg_mr(pd, (void *)(uintptr_t)iova, length, access);
    if (!mr)
        return NULL;
    pthread_mutex_lock(&g_mock_ibv_mtx);
    if (g_mock_ibv.vk_ctx && length > 0u) {
        int dfd = dup(fd);
        vxfec_vk_buf *vb = (dfd >= 0) ? vxfec_vk_buf_import_fd_range(g_mock_ibv.vk_ctx, dfd, offset, length, true) : NULL;
        if (!vb && dfd >= 0)
            close(dfd);
        if (vb) {
            for (int i = 0; i < MOCK_MAX_MRS; i++) {
                if (!g_mock_mrs[i].used) {
                    g_mock_mrs[i] = (mock_ibv_mr_entry_t){
                        .used = true,
                        .key = mr->lkey,
                        .base_addr = iova,
                        .length = length,
                        .vk_buf = vb,
                    };
                    break;
                }
            }
        }
    }
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    return mr;
}

__attribute__((visibility("default"))) int vxfec_test_ibv_dereg_mr(struct ibv_mr *mr)
{
    if (!mr)
        return 0;
    pthread_mutex_lock(&g_mock_ibv_mtx);
    for (int i = 0; i < MOCK_MAX_MRS; i++) {
        if (g_mock_mrs[i].used && g_mock_mrs[i].key == mr->lkey) {
            if (g_mock_mrs[i].vk_buf && g_mock_ibv.vk_ctx)
                vxfec_vk_buf_free(g_mock_ibv.vk_ctx, g_mock_mrs[i].vk_buf);
            memset(&g_mock_mrs[i], 0, sizeof(g_mock_mrs[i]));
            break;
        }
    }
    if (g_mock_ibv.active_mrs > 0u)
        g_mock_ibv.active_mrs--;
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    free(mr);
    return 0;
}

__attribute__((visibility("default"))) struct ibv_cq *
vxfec_test_ibv_create_cq(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vec)
{
    (void)comp_vec;
    mock_ibv_cq_t *mcq = (mock_ibv_cq_t *)calloc(1, sizeof(*mcq));
    if (!mcq)
        return NULL;
    mcq->cq.context = context;
    mcq->cq.channel = channel;
    mcq->cq.cq_context = cq_context;
    mcq->cq.cqe = cqe;
    return &mcq->cq;
}

__attribute__((visibility("default"))) int vxfec_test_ibv_destroy_cq(struct ibv_cq *cq)
{
    free(cq);
    return 0;
}

__attribute__((visibility("default"))) struct ibv_qp *vxfec_test_ibv_create_qp(struct ibv_pd *pd,
                                                                               struct ibv_qp_init_attr *qp_init_attr)
{
    mock_ibv_qp_t *mqp = (mock_ibv_qp_t *)calloc(1, sizeof(*mqp));
    if (!mqp)
        return NULL;
    pthread_mutex_lock(&g_mock_ibv_mtx);
    int slot = -1;
    for (int i = 0; i < MOCK_MAX_QPS; i++) {
        if (!g_mock_qps[i]) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        pthread_mutex_unlock(&g_mock_ibv_mtx);
        free(mqp);
        return NULL;
    }
    mqp->qp.context = pd->context;
    mqp->qp.pd = pd;
    mqp->qp.send_cq = qp_init_attr->send_cq;
    mqp->qp.recv_cq = qp_init_attr->recv_cq;
    mqp->qp.qp_type = qp_init_attr->qp_type;
    mqp->qp.qp_num = ++g_mock_ibv.next_qpn;
    mqp->state = IBV_QPS_RESET;
    g_mock_qps[slot] = mqp;
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    return &mqp->qp;
}

__attribute__((visibility("default"))) int vxfec_test_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask)
{
    mock_ibv_qp_t *mqp = (mock_ibv_qp_t *)qp;
    if (!mqp || !attr)
        return EINVAL;
    pthread_mutex_lock(&g_mock_ibv_mtx);
    if (attr_mask & IBV_QP_STATE) {
        mqp->state = attr->qp_state;
        qp->state = attr->qp_state;
    }
    if (attr_mask & IBV_QP_DEST_QPN)
        mqp->dest_qpn = attr->dest_qp_num;
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    return 0;
}

__attribute__((visibility("default"))) int vxfec_test_ibv_destroy_qp(struct ibv_qp *qp)
{
    mock_ibv_qp_t *mqp = (mock_ibv_qp_t *)qp;
    if (!mqp)
        return 0;
    pthread_mutex_lock(&g_mock_ibv_mtx);
    if (g_mock_ibv.busy_qpn != 0u && mqp->qp.qp_num == g_mock_ibv.busy_qpn && g_mock_ibv.busy_budget > 0u) {
        g_mock_ibv.busy_budget--;
        pthread_mutex_unlock(&g_mock_ibv_mtx);
        return EBUSY;
    }
    for (int i = 0; i < MOCK_MAX_QPS; i++) {
        if (g_mock_qps[i] == mqp) {
            g_mock_qps[i] = NULL;
            break;
        }
    }
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    free(mqp);
    return 0;
}

static bool host_has_routable_ipv4(void)
{
    bool found = false;
    struct ifaddrs *ifap = NULL;
    if (getifaddrs(&ifap) == 0) {
        for (struct ifaddrs *ifa = ifap; ifa != NULL; ifa = ifa->ifa_next) {
            if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET || !ifa->ifa_name || (ifa->ifa_flags & IFF_UP) == 0)
                continue;
            uint32_t ip_be = ((const struct sockaddr_in *)ifa->ifa_addr)->sin_addr.s_addr;
            if (ip_be != 0u && (ifa->ifa_flags & IFF_LOOPBACK) == 0 && (ntohl(ip_be) >> 24u) != 127u) {
                found = true;
                break;
            }
        }
        freeifaddrs(ifap);
    }
    return found;
}

static bool test_ibv_transport_direct_contracts(void)
{
    /* 1) verify exact comma-delimited VXFEC_IBV_DEVS token matching ("mock_rxe" != "mock_rxe0") */
    vxfec_ibv_dev_info_t infos[4];
    if (setenv("VXFEC_IBV_DEVS", "mock_rxe,other0", 1) != 0 || vxfec_ibv_enumerate(infos, 4u) != 0u) {
        unsetenv("VXFEC_IBV_DEVS");
        fprintf(stderr, "FAIL: VXFEC_IBV_DEVS substring matched 'mock_rxe' against 'mock_rxe0'\n");
        return false;
    }
    if (setenv("VXFEC_IBV_DEVS", "other0, mock_rxe0:1", 1) != 0 || vxfec_ibv_enumerate(infos, 4u) != 1u) {
        unsetenv("VXFEC_IBV_DEVS");
        fprintf(stderr, "FAIL: VXFEC_IBV_DEVS exact token 'mock_rxe0:1' did not match\n");
        return false;
    }
    unsetenv("VXFEC_IBV_DEVS");

    /* 2) verify sysfs RoCEv2 GID selection prefers RoCE v2 over v1 and skips 169.254.x.x link-local */
    char tmpl[] = "/tmp/vxfec_gid_XXXXXX";
    if (!mkdtemp(tmpl))
        return false;
    char dirpath[256], fpath[280];
    snprintf(dirpath, sizeof(dirpath), "%s/mock_rxe0", tmpl);
    (void)mkdir(dirpath, 0700);
    snprintf(dirpath, sizeof(dirpath), "%s/mock_rxe0/ports", tmpl);
    (void)mkdir(dirpath, 0700);
    snprintf(dirpath, sizeof(dirpath), "%s/mock_rxe0/ports/1", tmpl);
    (void)mkdir(dirpath, 0700);
    snprintf(dirpath, sizeof(dirpath), "%s/mock_rxe0/ports/1/gid_attrs", tmpl);
    (void)mkdir(dirpath, 0700);
    snprintf(dirpath, sizeof(dirpath), "%s/mock_rxe0/ports/1/gid_attrs/types", tmpl);
    (void)mkdir(dirpath, 0700);
    const char *types[4] = {"IB/RoCE v1\n", "RoCE v2\n", "RoCE v2\n", "RoCE v2\n"};
    for (int i = 0; i < 4; i++) {
        snprintf(fpath, sizeof(fpath), "%s/%d", dirpath, i);
        FILE *fp = fopen(fpath, "w");
        if (fp) {
            fputs(types[i], fp);
            fclose(fp);
        }
    }
    g_mock_ibv.rocev2_gid_matrix = true;
    setenv("VXFEC_IBV_SYSFS_ROOT", tmpl, 1);
    uint32_t ng = vxfec_ibv_enumerate(infos, 4u);
    bool gid_ok = (ng == 1u && infos[0].gid_index == 1u && ntohl(infos[0].ipv4_be) == 0x0a141e28u);
    /* also verify VXFEC_IBV_GID_INDEX override */
    setenv("VXFEC_IBV_GID_INDEX", "2", 1);
    ng = vxfec_ibv_enumerate(infos, 4u);
    gid_ok = gid_ok && (ng == 1u && infos[0].gid_index == 2u);
    unsetenv("VXFEC_IBV_GID_INDEX");
    unsetenv("VXFEC_IBV_SYSFS_ROOT");
    g_mock_ibv.rocev2_gid_matrix = false;
    for (int i = 0; i < 4; i++) {
        snprintf(fpath, sizeof(fpath), "%s/%d", dirpath, i);
        unlink(fpath);
    }
    rmdir(dirpath);
    snprintf(dirpath, sizeof(dirpath), "%s/mock_rxe0/ports/1/gid_attrs", tmpl);
    rmdir(dirpath);
    snprintf(dirpath, sizeof(dirpath), "%s/mock_rxe0/ports/1", tmpl);
    rmdir(dirpath);
    snprintf(dirpath, sizeof(dirpath), "%s/mock_rxe0/ports", tmpl);
    rmdir(dirpath);
    snprintf(dirpath, sizeof(dirpath), "%s/mock_rxe0", tmpl);
    rmdir(dirpath);
    rmdir(tmpl);
    if (!gid_ok) {
        fprintf(stderr, "FAIL: sysfs RoCEv2 GID selection / link-local avoidance check failed\n");
        return false;
    }

    /* 3) verify header_only_tx + vxfec_ibv_ep_tx_done consumes 0 remote UC Receive WQEs */
    vxfec_ibv_ctx_t *ictx = vxfec_ibv_ctx_open(0u);
    if (!ictx)
        return false;
    vxfec_ibv_wire_info_t tx_wi, rx_wi;
    vxfec_ibv_ep_t *tx_ep = vxfec_ibv_ep_create_opts(ictx, 2u, 4096u, true, false, true, &tx_wi);
    vxfec_ibv_ep_t *rx_ep = vxfec_ibv_ep_create_opts(ictx, 2u, 4096u, false, true, false, &rx_wi);
    uint8_t zc_buf[256] = {0x42};
    vxfec_ibv_mr_t *zmr = vxfec_ibv_mr_reg_host(ictx, zc_buf, sizeof(zc_buf), false);
    bool ep_ok = tx_ep && rx_ep && zmr && vxfec_ibv_ep_tx_is_hdr_only(tx_ep) && vxfec_ibv_ep_connect(tx_ep, &rx_wi, 0u) &&
                 vxfec_ibv_ep_connect(rx_ep, &tx_wi, 0u);
    if (ep_ok) {
        uint8_t hdr[24] = {1};
        uint64_t target_seq[VXFEC_IBV_MAX_QPS] = {0};
        g_mock_ibv.recv_wqes_consumed = 0u;
        ep_ok = vxfec_ibv_ep_send_pkt(tx_ep, 0u, hdr, sizeof(hdr), zc_buf, 128u, vxfec_ibv_mr_lkey(zmr));
        vxfec_ibv_ep_record_qp_tail(tx_ep, 0u, target_seq);
        uint32_t rq_after_data = g_mock_ibv.recv_wqes_consumed;
        ep_ok = ep_ok && (rq_after_data == 1u) && vxfec_ibv_ep_tx_done(tx_ep, target_seq) &&
                (g_mock_ibv.recv_wqes_consumed == rq_after_data) &&
                !vxfec_ibv_ep_rdma_write_imm(tx_ep, 0u, zc_buf, 8192u, vxfec_ibv_mr_lkey(zmr), 0x1000ULL, 1u, 0u);
    }
    vxfec_ibv_mr_dereg(zmr);
    (void)vxfec_ibv_ep_destroy(rx_ep);
    (void)vxfec_ibv_ep_destroy(tx_ep);
    vxfec_ibv_ctx_close(ictx);
    if (!ep_ok) {
        fprintf(stderr, "FAIL: header_only_tx / 0-RQE UC fence / MTU bounds check failed\n");
        return false;
    }
    return true;
}

static int run_ibv_mock_tests(void)
{
    if (setenv("VXFEC_IBV_LIB_PATH", "self", 1) != 0 || setenv("VXFEC_NCCL_BACKEND", "ibv", 1) != 0 ||
        setenv("VXFEC_NCCL_PORTS", "4", 1) != 0 || setenv("VXFEC_NCCL_VULKAN", "0", 1) != 0)
        return 1;
    unsetenv("VXFEC_NCCL_BIND_IP");
    unsetenv("VXFEC_NCCL_IFNAMES");
    unsetenv("NCCL_SOCKET_IFNAME");

    memset(&g_mock_ibv, 0, sizeof(g_mock_ibv));
    g_mock_ibv.next_qpn = 100u;
    g_mock_ibv.next_key = 1000u;

    const char *so_path = getenv("VXFEC_NCCL_SO_PATH");
    void *dl = dlopen(so_path && *so_path ? so_path : "./libnccl-vxfec.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        fprintf(stderr, "FAIL: %s\n", dlerror());
        return 1;
    }
    const ncclNet_v8_t *net = (const ncclNet_v8_t *)dlsym(dl, "ncclNetPlugin_v8");
    test_get_stats_ex_fn get_stats_ex = (test_get_stats_ex_fn)dlsym(dl, "vxfec_nccl_get_stats_ex");
    void (*reset_stats)(void) = (void (*)(void))dlsym(dl, "vxfec_nccl_reset_stats");
    void (*set_codec)(const char *) = (void (*)(const char *))dlsym(dl, "vxfec_nccl_set_codec_override");
    void (*set_done_fault)(uint32_t) = (void (*)(uint32_t))dlsym(dl, "vxfec_nccl_set_done_fault");
    void (*set_abort_fault)(uint32_t) = (void (*)(uint32_t))dlsym(dl, "vxfec_nccl_set_abort_fault");

    bool ok = false;
    void *lc = NULL, *sc = NULL, *rc = NULL, *tmr = NULL, *rmr = NULL;
    void *rc_fork = NULL, *rc_reconciled = NULL, *sc_raw = NULL, *sc_udp_fb = NULL;
    int ctrl_fd = -1, raw_lc_fd = -1;
    uint8_t *tx = (uint8_t *)malloc(65536u), *rx = (uint8_t *)calloc(1, 65536u);
    if (!net || !get_stats_ex || !reset_stats || !set_codec || !set_done_fault || !set_abort_fault || !tx || !rx)
        goto out;

    if (!test_ibv_transport_direct_contracts())
        goto out;

    /* 0a) verify dual-port RDMA HCA gets distinct NCCL device names, GUIDs, ports, and exact 4-lane IBTA speeds */
    g_mock_ibv.dual_port = true;
    g_mock_ibv.port1_speed = 32u;  /* EDR: 4 x 25G = 100000 Mbps */
    g_mock_ibv.port2_speed = 128u; /* NDR: 4 x 100G = 400000 Mbps */
    if (net->init(dummy_nccl_logger) != ncclSuccess)
        goto out;
    int mock_ndev = 0;
    ncclNetProperties_v8_t mp0, mp1;
    if (net->devices(&mock_ndev) != ncclSuccess || mock_ndev != 2 || net->getProperties(0, &mp0) != ncclSuccess ||
        net->getProperties(1, &mp1) != ncclSuccess) {
        fprintf(stderr, "FAIL: dual-port mock HCA device enumeration failed (ndev=%d)\n", mock_ndev);
        goto out;
    }
    if (strcmp(mp0.name, mp1.name) == 0 || strcmp(mp0.name, "vxfec-ibv-mock_rxe0:1") != 0 ||
        strcmp(mp1.name, "vxfec-ibv-mock_rxe0:2") != 0 || mp0.guid == mp1.guid || mp0.port != 1 || mp1.port != 2 ||
        mp0.speed != 100000 || mp1.speed != 400000) {
        fprintf(stderr, "FAIL: dual-port props mismatch: dev0=(%s,guid=%#lx,port=%d,spd=%d) dev1=(%s,guid=%#lx,port=%d,spd=%d)\n",
                mp0.name, (unsigned long)mp0.guid, mp0.port, mp0.speed, mp1.name, (unsigned long)mp1.guid, mp1.port, mp1.speed);
        goto out;
    }
    /* unmatched explicit interface filter on RDMA must expose 0 devices rather than falling back */
    if (setenv("VXFEC_NCCL_IFNAMES", "=missingXX", 1) != 0 || net->init(dummy_nccl_logger) != ncclSuccess ||
        net->devices(&mock_ndev) != ncclSuccess || mock_ndev != 0) {
        (void)unsetenv("VXFEC_NCCL_IFNAMES");
        fprintf(stderr, "FAIL: RDMA unmatched VXFEC_NCCL_IFNAMES='=missingXX' exposed ndev=%d\n", mock_ndev);
        goto out;
    }
    (void)unsetenv("VXFEC_NCCL_IFNAMES");
    g_mock_ibv.dual_port = false;
    g_mock_ibv.port1_speed = 0u;
    g_mock_ibv.port2_speed = 0u;

    /* 0) verify non-IPv4 RDMA GID (IB / RoCEv1 / IPv6 link-local) + default 4xHDR (200000 Mbps) */
    g_mock_ibv.ipv6_only_gid = true;
    if (net->init(dummy_nccl_logger) != ncclSuccess || net->getProperties(0, &mp0) != ncclSuccess || mp0.speed != 200000) {
        fprintf(stderr, "FAIL: 4xHDR port speed expected 200000 Mbps, got %d\n", mp0.speed);
        goto out;
    }
    uint8_t routable_handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
    void *lc_routable = NULL;
    if (net->listen(0, routable_handle, &lc_routable) != ncclSuccess || !lc_routable)
        goto out;
    test_handle_t rh;
    memcpy(&rh, routable_handle, sizeof(rh));
    net->closeListen(lc_routable);
    bool expect_routable = host_has_routable_ipv4();
    if (expect_routable ? ((ntohl(rh.ip_be) >> 24u) == 127u || rh.ip_be == 0u) : (rh.ip_be != htonl(INADDR_LOOPBACK))) {
        fprintf(stderr, "FAIL: verbs control IP resolution mismatch (expect_routable=%d ip_be=%#x)\n", expect_routable,
                ntohl(rh.ip_be));
        goto out;
    }

    /* pin loopback control IP for self-contained in-process socket tests */
    g_mock_ibv.ipv6_only_gid = false;
    if (setenv("VXFEC_NCCL_BIND_IP", "127.0.0.1", 1) != 0 || net->init(dummy_nccl_logger) != ncclSuccess)
        goto out;

    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
    if (net->listen(0, handle, &lc) != ncclSuccess || !lc)
        goto out;
    test_handle_t h;
    memcpy(&h, handle, sizeof(h));
    /* verify handle advertises is_ibv == 1 (byte 17 of vxfec_net_handle_t) */
    if (((h.reserved >> 8u) & 0xffu) != 1u) {
        fprintf(stderr, "FAIL: mock IBV listen handle did not set is_ibv=1\n");
        goto out;
    }

    /* 1) full end-to-end 240B RDMA connect/accept + role-sized slabs + zero-copy 2-SGE send + fence completion */
    for (int spin = 0; spin < 50000 && (!sc || !rc); spin++) {
        if (!sc && net->connect(0, handle, &sc, NULL) != ncclSuccess)
            goto out;
        if (!rc && net->accept(lc, &rc, NULL) != ncclSuccess)
            goto out;
    }
    if (!sc || !rc || g_mock_ibv.active_mrs != 2u) {
        fprintf(stderr, "FAIL: RDMA connect/accept or role-sized MR count (active_mrs=%u, expected 2)\n", g_mock_ibv.active_mrs);
        goto out;
    }
    for (size_t i = 0; i < 65536u; i++)
        tx[i] = (uint8_t)((i * 73u + 19u) & 0xffu);
    if (net->regMr(sc, tx, 65536u, NCCL_PTR_HOST, &tmr) != ncclSuccess ||
        net->regMr(rc, rx, 65536u, NCCL_PTR_HOST, &rmr) != ncclSuccess)
        goto out;
    reset_stats();
    void *sreq = NULL, *rreq = NULL, *rptr = rx;
    int rcap = 65536, tag = 801, sdone = 0, rdone = 0, got = 0;
    if (net->irecv(rc, 1, &rptr, &rcap, &tag, &rmr, &rreq) != ncclSuccess || !rreq ||
        net->isend(sc, tx, 65536, tag, tmr, &sreq) != ncclSuccess || !sreq)
        goto out;
    for (int spin = 0; spin < 50000 && (!sdone || !rdone); spin++) {
        if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
            goto out;
        if (!rdone && net->test(rreq, &rdone, &got) != ncclSuccess)
            goto out;
    }
    vxfec_nccl_stats_t st;
    get_stats_ex(&st, sizeof(st));
    if (!sdone || !rdone || got != 65536 || memcmp(tx, rx, 65536u) != 0 || st.ibv_direct_pkts == 0u) {
        fprintf(stderr, "FAIL: zero-copy RDMA transfer failed (sdone=%d rdone=%d got=%d ibv_pkts=%lu)\n", sdone, rdone, got,
                (unsigned long)st.ibv_direct_pkts);
        goto out;
    }

    /* 1a) verify GPUDirect RDMA DMA-BUF registration & zero-copy send succeed with VXFEC_NCCL_VULKAN=0 */
    {
        ncclNetProperties_v8_t ibv_p;
        if (net->getProperties(0, &ibv_p) != ncclSuccess || (ibv_p.ptrSupport & NCCL_PTR_DMABUF) == 0) {
            fprintf(stderr, "FAIL: RDMA device omitted NCCL_PTR_DMABUF when VXFEC_NCCL_VULKAN=0\n");
            goto out;
        }
        int dfd = memfd_create("vxfec_ibv_dmabuf", MFD_CLOEXEC);
        if (dfd < 0 || ftruncate(dfd, 131072) != 0) {
            if (dfd >= 0)
                close(dfd);
            goto out;
        }
        void *ovf_mr = NULL;
        if (net->regMrDmaBuf(sc, tx, 8192u, NCCL_PTR_CUDA, UINT64_MAX - 4095u, dfd, &ovf_mr) != ncclInvalidArgument) {
            close(dfd);
            fprintf(stderr, "FAIL: regMrDmaBuf accepted overflowing offset + size\n");
            goto out;
        }
        void *dmr = NULL;
        if (net->regMrDmaBuf(sc, tx, 65536u, NCCL_PTR_CUDA, 4096u, dfd, &dmr) != ncclSuccess || !dmr) {
            close(dfd);
            fprintf(stderr, "FAIL: regMrDmaBuf failed on RDMA device with VXFEC_NCCL_VULKAN=0\n");
            goto out;
        }
        close(dfd);
        memset(rx, 0, 16384u);
        reset_stats();
        void *dsreq = NULL, *drreq = NULL;
        int dtag = 804, dsd = 0, drd = 0, dgot = 0;
        if (net->irecv(rc, 1, &rptr, &rcap, &dtag, &rmr, &drreq) != ncclSuccess || !drreq ||
            net->isend(sc, tx + 2048u, 16384, dtag, dmr, &dsreq) != ncclSuccess || !dsreq) {
            net->deregMr(sc, dmr);
            goto out;
        }
        for (int spin = 0; spin < 50000 && (!dsd || !drd); spin++) {
            if (!dsd && net->test(dsreq, &dsd, NULL) != ncclSuccess)
                break;
            if (!drd && net->test(drreq, &drd, &dgot) != ncclSuccess)
                break;
        }
        net->deregMr(sc, dmr);
        get_stats_ex(&st, sizeof(st));
        if (!dsd || !drd || dgot != 16384 || memcmp(tx + 2048u, rx, 16384u) != 0 || st.ibv_direct_pkts == 0u) {
            fprintf(stderr, "FAIL: GPUDirect RDMA DMA-BUF zero-copy send with VXFEC_NCCL_VULKAN=0 failed\n");
            goto out;
        }
    }

    /* 1b) verify partial tail systematic symbols send only valid_len bytes via zero-copy 2-SGE verbs */
    reset_stats();
    g_mock_ibv.max_send_bytes = 0u;
    memset(rx, 0, 1024u);
    sreq = NULL;
    rreq = NULL;
    sdone = 0;
    rdone = 0;
    got = 0;
    int tail_tag = 805;
    if (net->irecv(rc, 1, &rptr, &rcap, &tail_tag, &rmr, &rreq) != ncclSuccess || !rreq ||
        net->isend(sc, tx, 300, tail_tag, tmr, &sreq) != ncclSuccess || !sreq)
        goto out;
    for (int spin = 0; spin < 50000 && (!sdone || !rdone); spin++) {
        if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
            goto out;
        if (!rdone && net->test(rreq, &rdone, &got) != ncclSuccess)
            goto out;
    }
    get_stats_ex(&st, sizeof(st));
    if (!sdone || !rdone || got != 300 || memcmp(tx, rx, 300u) != 0 || st.ibv_direct_pkts != 1u ||
        g_mock_ibv.max_send_bytes != sizeof(test_wire_hdr_t) + 300u) {
        fprintf(stderr, "FAIL: tail symbol sent padded length (max_send=%u expected=%zu ibv_pkts=%lu)\n", g_mock_ibv.max_send_bytes,
                sizeof(test_wire_hdr_t) + 300u, (unsigned long)st.ibv_direct_pkts);
        goto out;
    }

    /* 1c) verify silent IBV_QPT_UC frame drops trigger verbs FEC repair (RS8 HARQ-IR, RS8 0-RTT proactive, and RaptorQ) */
    for (int fcase = 0; fcase < 3; fcase++) {
        set_codec((fcase == 2) ? "rq-gf256" : NULL);
        reset_stats();
        for (size_t i = 0; i < 65536u; i++)
            tx[i] = (uint8_t)((i * 61u + (size_t)fcase * 29u + 7u) & 0xffu);
        memset(rx, 0, 65536u);

        int ftag = 806 + fcase;
        uint32_t exp_drops = (fcase == 0) ? 4u : ((fcase == 1) ? 2u : 6u);
        g_mock_ibv.drop_tag = (uint32_t)ftag;
        g_mock_ibv.drop_esi_beg = 3u;
        g_mock_ibv.drop_esi_cnt = exp_drops;
        g_mock_ibv.dropped_uc_pkts = 0u;
        g_mock_ibv.repair_uc_pkts = 0u;

        sreq = NULL;
        rreq = NULL;
        sdone = 0;
        rdone = 0;
        got = 0;
        if (net->irecv(rc, 1, &rptr, &rcap, &ftag, &rmr, &rreq) != ncclSuccess || !rreq ||
            net->isend(sc, tx, 65536, ftag, tmr, &sreq) != ncclSuccess || !sreq)
            goto out;
        for (int spin = 0; spin < 100000 && (!sdone || !rdone); spin++) {
            if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
                goto out;
            if (!rdone && net->test(rreq, &rdone, &got) != ncclSuccess)
                goto out;
        }
        g_mock_ibv.drop_tag = 0u;
        get_stats_ex(&st, sizeof(st));
        if (!sdone || !rdone || got != 65536 || memcmp(tx, rx, 65536u) != 0 || g_mock_ibv.dropped_uc_pkts != exp_drops ||
            g_mock_ibv.repair_uc_pkts < exp_drops || st.tx_sack_retx_pkts != 0u) {
            fprintf(stderr, "FAIL: verbs UC FEC case %d failed (dropped=%u/%u rep_uc=%u sack=%lu)\n", fcase,
                    g_mock_ibv.dropped_uc_pkts, exp_drops, g_mock_ibv.repair_uc_pkts, (unsigned long)st.tx_sack_retx_pkts);
            goto out;
        }
        if (fcase == 0 && (st.rx_rs8_decodes != 1u || st.tx_harq_ir_pkts < 4u)) {
            fprintf(stderr, "FAIL: verbs UC RS8 HARQ-IR missed decode (rs8=%lu ir=%lu)\n", (unsigned long)st.rx_rs8_decodes,
                    (unsigned long)st.tx_harq_ir_pkts);
            goto out;
        }
        if (fcase == 1 && (st.rx_rs8_decodes != 1u || st.tx_proactive_rep_pkts < 2u || st.tx_harq_ir_pkts != 0u)) {
            fprintf(stderr, "FAIL: verbs UC 0-RTT proactive RS8 missed decode or needed IR (rs8=%lu p0=%lu ir=%lu)\n",
                    (unsigned long)st.rx_rs8_decodes, (unsigned long)st.tx_proactive_rep_pkts, (unsigned long)st.tx_harq_ir_pkts);
            goto out;
        }
        if (fcase == 2 && st.rx_rq_decodes != 1u) {
            fprintf(stderr, "FAIL: verbs UC RaptorQ missed decode (rq=%lu)\n", (unsigned long)st.rx_rq_decodes);
            goto out;
        }
    }
    set_codec(NULL);

    /* 1d) verify lost PKT_DONE on RDMA completes sreq even when caller stops polling rc after rreq finishes */
    set_done_fault(4u);
    sreq = NULL;
    rreq = NULL;
    sdone = 0;
    rdone = 0;
    got = 0;
    int dtag = 809;
    memset(rx, 0, 4096u);
    if (net->irecv(rc, 1, &rptr, &rcap, &dtag, &rmr, &rreq) != ncclSuccess || !rreq ||
        net->isend(sc, tx, 4096, dtag, tmr, &sreq) != ncclSuccess || !sreq)
        goto out;
    for (int spin = 0; spin < 50000 && !rdone; spin++) {
        if (net->test(rreq, &rdone, &got) != ncclSuccess)
            goto out;
        if (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)
            goto out;
    }
    if (!rdone || got != 4096 || memcmp(tx, rx, 4096u) != 0)
        goto out;
    /* caller no longer touches rc/rreq; only sreq is polled across RTO */
    uint64_t ibv_done_deadline = test_rq_now_ns() + 2000000000ULL;
    while (!sdone && test_rq_now_ns() < ibv_done_deadline) {
        if (net->test(sreq, &sdone, NULL) != ncclSuccess)
            goto out;
        if (!sdone)
            usleep(200);
    }
    set_done_fault(0u);
    if (!sdone) {
        fprintf(stderr, "FAIL: RDMA send hung after initial PKT_DONE datagrams were lost\n");
        goto out;
    }

    /* 2) verify full-frame 240B RDMA HELLO -> 240B RDMA HELLO_ACK for both fork (has_ibv=1) and reconciled (0x02|feat) */
    ctrl_fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    struct sockaddr_in csin = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be};
    socklen_t clen = sizeof(csin);
    if (ctrl_fd < 0 || bind(ctrl_fd, (struct sockaddr *)&csin, clen) != 0 ||
        getsockname(ctrl_fd, (struct sockaddr *)&csin, &clen) != 0)
        goto out;
    struct sockaddr_in lc_addr = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be, .sin_port = h.ctrl_port_be};

    struct {
        test_hello_t base;
        vxfec_ibv_wire_info_t ibv_info;
    } full_hello;
    struct {
        test_ack_t base;
        vxfec_ibv_wire_info_t ibv_info;
    } full_ack;
#ifdef NANORQ_HAS_GF2_MODE
    uint8_t exp_gf2 = 1u;
#else
    uint8_t exp_gf2 = 0u;
#endif

    for (int dialect = 0; dialect < 2; dialect++) {
        memset(&full_hello, 0, sizeof(full_hello));
        full_hello.base.magic = h.magic;
        full_hello.base.pkt_type = 1u;
        full_hello.base.has_ibv = (dialect == 0) ? 1u : (uint8_t)(0x02u | 0x04u | exp_gf2);
        full_hello.base.num_ports = h.num_ports;
        full_hello.base.cookie = h.cookie;
        full_hello.base.tx_ctrl_port_be = csin.sin_port;
        full_hello.base.conn_id = 0x8800ULL + (uint64_t)dialect;
        for (uint16_t p = 0; p < h.num_ports; p++)
            full_hello.base.tx_ports_be[p] = csin.sin_port;
        full_hello.ibv_info.gid[10] = 0xffu;
        full_hello.ibv_info.gid[11] = 0xffu;
        full_hello.ibv_info.gid[12] = 127u;
        full_hello.ibv_info.gid[15] = 1u;
        full_hello.ibv_info.lid = 1u;
        full_hello.ibv_info.port_num = 1u;
        full_hello.ibv_info.mtu_enum = IBV_MTU_4096;
        full_hello.ibv_info.num_qps = (uint8_t)h.num_ports;
        full_hello.ibv_info.is_roce = 1u;
        for (uint16_t q = 0; q < h.num_ports; q++)
            full_hello.ibv_info.qpn[q] = 9000u + q;

        memset(&full_ack, 0, sizeof(full_ack));
        bool got_full = false;
        void **rc_slot = (dialect == 0) ? &rc_fork : &rc_reconciled;
        for (int spin = 0; spin < 50000 && (!*rc_slot || !got_full); spin++) {
            if ((spin % 500) == 0)
                (void)sendto(ctrl_fd, &full_hello, sizeof(full_hello), MSG_DONTWAIT, (struct sockaddr *)&lc_addr, sizeof(lc_addr));
            if (!got_full &&
                recvfrom(ctrl_fd, &full_ack, sizeof(full_ack), MSG_DONTWAIT, NULL, NULL) == (ssize_t)sizeof(full_ack) &&
                full_ack.base.magic == h.magic && full_ack.base.conn_id == full_hello.base.conn_id)
                got_full = true;
            if (!*rc_slot && net->accept(lc, rc_slot, NULL) != ncclSuccess)
                goto out;
        }
        uint8_t exp_ack_b5 = (dialect == 0) ? 1u : (uint8_t)(0x02u | 0x04u | exp_gf2);
        uint8_t exp_ack_feat = (dialect == 0) ? 0u : (uint8_t)(0x04u | exp_gf2);
        if (!*rc_slot || !got_full || full_ack.base.has_ibv != exp_ack_b5 || full_ack.base.features != exp_ack_feat ||
            full_ack.ibv_info.num_qps != h.num_ports) {
            fprintf(stderr, "FAIL: full-frame 240B RDMA listener handshake dialect %d failed (got=%d b5=%#x feat=%#x qps=%u)\n",
                    dialect, got_full, full_ack.base.has_ibv, full_ack.base.features, full_ack.ibv_info.num_qps);
            goto out;
        }
    }

    /* 3) verify sender emits 240B RDMA HELLO (has_ibv = 0x02|features) and accepts 240B reconciled RDMA HELLO_ACK */
    raw_lc_fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    struct sockaddr_in rsin = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be};
    socklen_t rlen = sizeof(rsin);
    if (raw_lc_fd < 0 || bind(raw_lc_fd, (struct sockaddr *)&rsin, rlen) != 0 ||
        getsockname(raw_lc_fd, (struct sockaddr *)&rsin, &rlen) != 0)
        goto out;
    test_handle_t raw_h = h;
    raw_h.ctrl_port_be = rsin.sin_port;
    raw_h.cookie = 0x55aa33ccu;
    uint8_t raw_handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
    memcpy(raw_handle, &raw_h, sizeof(raw_h));

    for (int spin = 0; spin < 50000 && !sc_raw; spin++) {
        if (net->connect(0, raw_handle, &sc_raw, NULL) != ncclSuccess)
            goto out;
        struct sockaddr_in peer;
        socklen_t plen = sizeof(peer);
        if (recvfrom(raw_lc_fd, &full_hello, sizeof(full_hello), MSG_DONTWAIT, (struct sockaddr *)&peer, &plen) !=
            (ssize_t)sizeof(full_hello))
            continue;
        uint8_t exp_local = (uint8_t)(0x04u | exp_gf2);
        if (full_hello.base.has_ibv != (uint8_t)(0x02u | exp_local) || full_hello.base.features != exp_local ||
            full_hello.ibv_info.num_qps != h.num_ports) {
            fprintf(stderr, "FAIL: sender 240B RDMA HELLO unexpected flags (b5=%#x feat=%#x qps=%u)\n", full_hello.base.has_ibv,
                    full_hello.base.features, full_hello.ibv_info.num_qps);
            goto out;
        }
        memset(&full_ack, 0, sizeof(full_ack));
        full_ack.base.magic = raw_h.magic;
        full_ack.base.pkt_type = 2u;
        full_ack.base.has_ibv = (uint8_t)(0x02u | exp_local);
        full_ack.base.features = 0u; /* reconciled 0235844 / 7e0b2c9 RDMA HELLO_ACK */
        full_ack.base.num_ports = raw_h.num_ports;
        full_ack.base.cookie = raw_h.cookie;
        full_ack.base.rx_ctrl_port_be = rsin.sin_port;
        full_ack.base.conn_id = full_hello.base.conn_id;
        for (uint16_t p = 0; p < raw_h.num_ports; p++)
            full_ack.base.rx_ports_be[p] = rsin.sin_port;
        full_ack.ibv_info = full_hello.ibv_info;
        full_ack.ibv_info.mtu_enum = IBV_MTU_1024; /* peer advertises smaller 1024B verbs MTU */
        (void)sendto(raw_lc_fd, &full_ack, sizeof(full_ack), 0, (struct sockaddr *)&peer, plen);
    }
    if (!sc_raw) {
        fprintf(stderr, "FAIL: sender rejected full-frame 240B reconciled RDMA HELLO_ACK\n");
        goto out;
    }
    /* verify sender clamps verbs symbol size to peer's 1024B MTU (hdr + sym_T <= 1024B) */
    g_mock_ibv.max_send_bytes = 0u;
    void *mtu_req = NULL;
    if (net->isend(sc_raw, tx, 8192, 810, NULL, &mtu_req) != ncclSuccess || !mtu_req || g_mock_ibv.max_send_bytes == 0u ||
        g_mock_ibv.max_send_bytes > 1024u) {
        fprintf(stderr, "FAIL: sender ignored peer 1024B verbs MTU (max_send_bytes=%u)\n", g_mock_ibv.max_send_bytes);
        goto out;
    }

    /* verify UDP fallback on an RDMA device sizes symbols for UDP socket MTU rather than d->is_ibv */
    raw_h.cookie = 0x55aa33cdu;
    memcpy(raw_handle, &raw_h, sizeof(raw_h));
    for (int spin = 0; spin < 50000 && !sc_udp_fb; spin++) {
        if (net->connect(0, raw_handle, &sc_udp_fb, NULL) != ncclSuccess)
            goto out;
        struct sockaddr_in peer;
        socklen_t plen = sizeof(peer);
        if (recvfrom(raw_lc_fd, &full_hello, sizeof(full_hello), MSG_DONTWAIT, (struct sockaddr *)&peer, &plen) !=
            (ssize_t)sizeof(full_hello))
            continue;
        test_ack_t udp_ack;
        memset(&udp_ack, 0, sizeof(udp_ack));
        udp_ack.magic = raw_h.magic;
        udp_ack.pkt_type = 2u;
        udp_ack.features = full_hello.base.features;
        udp_ack.num_ports = raw_h.num_ports;
        udp_ack.cookie = raw_h.cookie;
        udp_ack.rx_ctrl_port_be = rsin.sin_port;
        udp_ack.conn_id = full_hello.base.conn_id;
        /* leave rx_ports_be[0] = 0 so connect(fds[0]) fails and UDP fallback uses safe 1280B default instead of 4016B IB MTU */
        for (uint16_t p = 1; p < raw_h.num_ports; p++)
            udp_ack.rx_ports_be[p] = rsin.sin_port;
        (void)sendto(raw_lc_fd, &udp_ack, sizeof(udp_ack), 0, (struct sockaddr *)&peer, plen);
    }
    if (!sc_udp_fb)
        goto out;
    uint32_t pre_ibv_posts = g_mock_ibv.post_sends;
    void *fb_req = NULL;
    if (net->isend(sc_udp_fb, tx, 4096, 811, NULL, &fb_req) != ncclSuccess || !fb_req || g_mock_ibv.post_sends != pre_ibv_posts)
        goto out;
    uint16_t fb_sym_t = 0u;
    for (int spin = 0; spin < 4000; spin++) {
        uint8_t pkt[2048];
        ssize_t n = recv(raw_lc_fd, pkt, sizeof(pkt), MSG_DONTWAIT);
        if (n >= (ssize_t)sizeof(test_wire_hdr_t)) {
            test_wire_hdr_t wh;
            memcpy(&wh, pkt, sizeof(wh));
            if (wh.magic == raw_h.magic && wh.cookie == raw_h.cookie && wh.tag == 811u && wh.pkt_type == 3u) {
                fb_sym_t = wh.sym_T;
                break;
            }
        }
    }
    if (fb_sym_t == 0u || fb_sym_t > 1424u) {
        fprintf(stderr, "FAIL: UDP fallback on verbs device used oversized sym_T=%u\n", fb_sym_t);
        goto out;
    }

    /* 3b) verify fatal TX CQ error notifies matching pending RDMA receive with ncclRemoteError */
    {
        void *lc_txe = NULL, *sc_txe = NULL, *rc_txe = NULL;
        uint8_t h_txe[NCCL_NET_HANDLE_MAXSIZE] = {0};
        if (net->listen(0, h_txe, &lc_txe) != ncclSuccess || !lc_txe)
            goto out;
        for (int spin = 0; spin < 50000 && (!sc_txe || !rc_txe); spin++) {
            if (!sc_txe && net->connect(0, h_txe, &sc_txe, NULL) != ncclSuccess)
                goto out;
            if (!rc_txe && net->accept(lc_txe, &rc_txe, NULL) != ncclSuccess)
                goto out;
        }
        net->closeListen(lc_txe);
        if (!sc_txe || !rc_txe)
            goto out;
        void *r_pend = NULL, *s_fail = NULL, *rp_txe = rx;
        int cap_txe = 65536, tag_txe = 820, rd_txe = 0, sd_txe = 0, got_txe = 0;
        if (net->irecv(rc_txe, 1, &rp_txe, &cap_txe, &tag_txe, &rmr, &r_pend) != ncclSuccess || !r_pend)
            goto out;
        pthread_mutex_lock(&g_mock_ibv_mtx);
        g_mock_ibv.inject_wc_err = true;
        pthread_mutex_unlock(&g_mock_ibv_mtx);
        set_abort_fault(3u);
        if (net->isend(sc_txe, tx, 4096, tag_txe, tmr, &s_fail) != ncclSuccess || !s_fail) {
            set_abort_fault(0u);
            goto out;
        }
        if (net->test(s_fail, &sd_txe, NULL) != ncclSystemError) {
            set_abort_fault(0u);
            fprintf(stderr, "FAIL: fatal TX CQ error did not fail sreq\n");
            goto out;
        }
        /* do not poll sc_txe again; background RTO must retransmit dropped PKT_ERR */
        ncclResult_t rx_abort_st = ncclSuccess;
        uint64_t ibv_abt_dl = test_rq_now_ns() + 2000000000ULL;
        while (rx_abort_st == ncclSuccess && !rd_txe && test_rq_now_ns() < ibv_abt_dl) {
            rx_abort_st = net->test(r_pend, &rd_txe, &got_txe);
            if (rx_abort_st == ncclSuccess && !rd_txe)
                usleep(200);
        }
        set_abort_fault(0u);
        net->closeSend(sc_txe);
        net->closeRecv(rc_txe);
        if (rx_abort_st != ncclRemoteError || rd_txe != 0) {
            fprintf(stderr, "FAIL: fatal TX CQ error left matching RDMA receive pending (st=%d done=%d)\n", (int)rx_abort_st,
                    rd_txe);
            goto out;
        }
    }

    /* 3c) verify fatal RX WC error surfaces ncclSystemError on recvComm and notifies sender with PKT_ERR */
    void *rx_err_req = NULL, *tx_err_req = NULL, *rx_err_ptr = rx;
    int rx_err_cap = 65536, rx_err_tag = 812, rx_err_done = 0, tx_err_done = 0, rx_err_got = 0;
    set_codec("sack");
    g_mock_ibv.drop_tag = (uint32_t)rx_err_tag;
    g_mock_ibv.drop_esi_beg = 0u;
    g_mock_ibv.drop_esi_cnt = 1u;
    if (net->irecv(rc, 1, &rx_err_ptr, &rx_err_cap, &rx_err_tag, &rmr, &rx_err_req) != ncclSuccess || !rx_err_req ||
        net->isend(sc, tx, 65536, rx_err_tag, tmr, &tx_err_req) != ncclSuccess || !tx_err_req)
        goto out;
    /* pump recvComm once so the non-dropped UC symbols create an active RX slot before the RQ fault */
    if (net->test(rx_err_req, &rx_err_done, &rx_err_got) != ncclSuccess || rx_err_done != 0)
        goto out;
    g_mock_ibv.drop_tag = 0u;
    set_codec(NULL);
    pthread_mutex_lock(&g_mock_ibv_mtx);
    struct ibv_cq *rc_cq = NULL;
    for (int i = 0; i < MOCK_MAX_QPS; i++) {
        if (g_mock_qps[i] && g_mock_qps[i]->rq_cnt > 0u) {
            rc_cq = g_mock_qps[i]->qp.recv_cq;
            break;
        }
    }
    if (rc_cq) {
        struct ibv_wc bad_rwc;
        memset(&bad_rwc, 0, sizeof(bad_rwc));
        bad_rwc.status = IBV_WC_LOC_LEN_ERR;
        bad_rwc.opcode = IBV_WC_RECV;
        mock_cq_push_locked(rc_cq, bad_rwc);
    }
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    if (!rc_cq || net->test(rx_err_req, &rx_err_done, &rx_err_got) != ncclSystemError || rx_err_done != 0) {
        fprintf(stderr, "FAIL: fatal RX WC error did not surface ncclSystemError\n");
        goto out;
    }
    ncclResult_t tx_err_res = ncclSuccess;
    for (int spin = 0; spin < 5000 && tx_err_res == ncclSuccess && !tx_err_done; spin++)
        tx_err_res = net->test(tx_err_req, &tx_err_done, NULL);
    if (tx_err_res != ncclRemoteError || tx_err_done != 0) {
        fprintf(stderr, "FAIL: broken RDMA recvComm did not notify sender via PKT_ERR (res=%d done=%d)\n", (int)tx_err_res,
                tx_err_done);
        goto out;
    }
    void *rx_blocked_req = NULL;
    if (net->irecv(rc, 1, &rx_err_ptr, &rx_err_cap, &rx_err_tag, &rmr, &rx_blocked_req) != ncclSystemError) {
        fprintf(stderr, "FAIL: broken RDMA recvComm accepted new irecv\n");
        goto out;
    }

    void *neg_cq_req = NULL;
    int neg_tag = 813;
    if (net->irecv(rc_fork, 1, &rx_err_ptr, &rx_err_cap, &neg_tag, NULL, &neg_cq_req) != ncclSuccess || !neg_cq_req)
        goto out;
    pthread_mutex_lock(&g_mock_ibv_mtx);
    struct ibv_cq *fork_cq = NULL;
    for (int i = 0; i < MOCK_MAX_QPS; i++) {
        if (g_mock_qps[i] && g_mock_qps[i]->rq_cnt > 0u) {
            fork_cq = g_mock_qps[i]->qp.recv_cq;
            break;
        }
    }
    g_mock_ibv.fail_poll_cq = fork_cq;
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    if (!fork_cq || net->test(neg_cq_req, &rx_err_done, &rx_err_got) != ncclSystemError || rx_err_done != 0) {
        fprintf(stderr, "FAIL: negative ibv_poll_cq on RX CQ did not surface ncclSystemError\n");
        goto out;
    }

    /* 4) verify fatal TX CQ error + partial destroy_qp (EBUSY) holds zero-copy req without crashing on nulled QPs */
    pthread_mutex_lock(&g_mock_ibv_mtx);
    uint32_t busy_qpn = g_mock_qps[1] ? g_mock_qps[1]->qp.qp_num : 0u;
    g_mock_ibv.busy_qpn = busy_qpn;
    g_mock_ibv.busy_budget = 2u;
    g_mock_ibv.inject_wc_err = true;
    pthread_mutex_unlock(&g_mock_ibv_mtx);
    if (busy_qpn == 0u)
        goto out;

    void *err_req = NULL;
    if (net->isend(sc, tx, 65536, 802, tmr, &err_req) != ncclSuccess || !err_req)
        goto out;
    /* first 2 polls: destroy_qp returns EBUSY on busy_qpn while sibling QPs are already NULL -> must stay pending */
    int edone = 0;
    if (net->test(err_req, &edone, NULL) != ncclSuccess || edone != 0) {
        fprintf(stderr, "FAIL: zero-copy RDMA send completed while QP teardown was still EBUSY\n");
        goto out;
    }
    void *blocked_req = NULL;
    if (net->isend(sc, tx, 1024, 803, tmr, &blocked_req) != ncclSystemError) {
        fprintf(stderr, "FAIL: broken RDMA sendComm accepted new isend\n");
        goto out;
    }
    /* budget exhausted on next poll -> final QP destroys cleanly and err_req surfaces ncclSystemError */
    if (net->test(err_req, &edone, NULL) != ncclSystemError) {
        fprintf(stderr, "FAIL: zero-copy RDMA send did not surface ncclSystemError once final QP destroyed\n");
        goto out;
    }

    printf("PASS: Full-frame 240B RDMA handshakes (fork, reconciled, current), zero-copy 2-SGE send, and EBUSY QP teardown\n");
    ok = true;

out:
    if (set_codec)
        set_codec(NULL);
    g_mock_ibv.drop_tag = 0u;
    g_mock_ibv.busy_budget = 0u;
    g_mock_ibv.busy_qpn = 0u;
    g_mock_ibv.fail_poll_cq = NULL;
    g_mock_ibv.inject_wc_err = false;
    g_mock_ibv.ipv6_only_gid = false;
    g_mock_ibv.dual_port = false;
    g_mock_ibv.port1_speed = 0u;
    g_mock_ibv.port2_speed = 0u;
    if (set_done_fault)
        set_done_fault(0u);
    unsetenv("VXFEC_NCCL_BIND_IP");
    if (tmr)
        net->deregMr(sc, tmr);
    if (rmr)
        net->deregMr(rc, rmr);
    if (sc_udp_fb)
        net->closeSend(sc_udp_fb);
    if (sc_raw)
        net->closeSend(sc_raw);
    if (rc_reconciled)
        net->closeRecv(rc_reconciled);
    if (rc_fork)
        net->closeRecv(rc_fork);
    if (sc)
        net->closeSend(sc);
    if (rc)
        net->closeRecv(rc);
    if (lc)
        net->closeListen(lc);
    if (raw_lc_fd >= 0)
        close(raw_lc_fd);
    if (ctrl_fd >= 0)
        close(ctrl_fd);
    free(tx);
    free(rx);

    /* 5) GPU VRAM DMA-BUF -> PKT_RDMA_CTS -> RDMA_WRITE_WITH_IMM + GPU-resident out_dmabuf parity + in-place VRAM decode */
    if (ok) {
        vxfec_vk_ctx *vk = vxfec_vk_ctx_new();
        if (vk && vxfec_vk_has_dmabuf(vk) && vxfec_vk_has_dmabuf_export(vk) && vxfec_vk_has_timeline(vk)) {
            pthread_mutex_lock(&g_mock_ibv_mtx);
            g_mock_ibv.vk_ctx = vk;
            g_mock_ibv.rdma_write_imm_pkts = 0u;
            g_mock_ibv.drop_rdma_rem = 0u;
            g_mock_ibv.dropped_rdma_pkts = 0u;
            pthread_mutex_unlock(&g_mock_ibv_mtx);

            setenv("VXFEC_NCCL_VULKAN", "2", 1);
            setenv("VXFEC_NCCL_BACKEND", "ibv", 1);
            setenv("VXFEC_NCCL_PORTS", "4", 1);
            if (net->init(dummy_nccl_logger) != ncclSuccess) {
                ok = false;
            } else {
                const size_t gsz = 262144u;
                int tfd = -1, rfd = -1;
                vxfec_vk_buf *tvb = vxfec_vk_buf_new_exportable(vk, gsz, true, &tfd);
                vxfec_vk_buf *rvb = vxfec_vk_buf_new_exportable(vk, gsz, true, &rfd);
                uint8_t *h_src = (uint8_t *)malloc(gsz);
                uint8_t *h_dst = (uint8_t *)calloc(1, gsz);
                uint8_t *h_poison = (uint8_t *)malloc(gsz);
                void *glc = NULL, *gsc = NULL, *grc = NULL, *gtmr = NULL, *grmr = NULL;
                uint8_t gh[NCCL_NET_HANDLE_MAXSIZE] = {0};

                if (!tvb || !rvb || tfd < 0 || rfd < 0 || !h_src || !h_dst || !h_poison ||
                    net->listen(0, gh, &glc) != ncclSuccess || !glc) {
                    ok = false;
                } else {
                    memset(h_poison, 0xa5, gsz);
                    for (int spin = 0; spin < 50000 && (!gsc || !grc); spin++) {
                        if (!gsc && net->connect(0, gh, &gsc, NULL) != ncclSuccess) {
                            ok = false;
                            break;
                        }
                        if (!grc && net->accept(glc, &grc, NULL) != ncclSuccess) {
                            ok = false;
                            break;
                        }
                    }
                    void *tx_iova = (void *)(uintptr_t)0x610000000000ULL;
                    void *rx_iova = (void *)(uintptr_t)0x620000000000ULL;
                    if (!ok || !gsc || !grc || net->regMrDmaBuf(gsc, tx_iova, gsz, NCCL_PTR_CUDA, 0u, tfd, &gtmr) != ncclSuccess ||
                        net->regMrDmaBuf(grc, rx_iova, gsz, NCCL_PTR_CUDA, 0u, rfd, &grmr) != ncclSuccess) {
                        ok = false;
                    } else {
                        /* 5a: Lossless RDMA_WRITE_WITH_IMM directly into RX GPU VRAM (0-copy), 5b: RS8 VRAM repair, 5c: RQ VRAM */
                        for (int sub = 0; sub < 3 && ok; sub++) {
                            const char *cmode = (sub == 0) ? NULL : ((sub == 1) ? "rs8" : "rq-gf2");
                            uint32_t want_drops = (sub == 0) ? 0u : ((sub == 1) ? 4u : 6u);
                            set_codec(cmode);
                            reset_stats();
                            for (size_t i = 0; i < gsz; i++)
                                h_src[i] = (uint8_t)((i * 73u + (size_t)sub * 41u + 19u) & 0xffu);
                            if (!vxfec_vk_buf_write_offset(vk, tvb, 0u, h_src, gsz) || !vxfec_vk_buf_release_external(vk, tvb) ||
                                !vxfec_vk_buf_write_offset(vk, rvb, 0u, h_poison, gsz) || !vxfec_vk_buf_release_external(vk, rvb)) {
                                ok = false;
                                break;
                            }
                            pthread_mutex_lock(&g_mock_ibv_mtx);
                            uint32_t pre_rdma = g_mock_ibv.rdma_write_imm_pkts;
                            g_mock_ibv.drop_rdma_rem = want_drops;
                            g_mock_ibv.dropped_rdma_pkts = 0u;
                            pthread_mutex_unlock(&g_mock_ibv_mtx);

                            void *gsreq = NULL, *grreq = NULL, *grptr = rx_iova;
                            int gcap = (int)gsz, gtag = 860 + sub, gsd = 0, grd = 0, ggot = 0;
                            if (net->irecv(grc, 1, &grptr, &gcap, &gtag, &grmr, &grreq) != ncclSuccess || !grreq) {
                                ok = false;
                                break;
                            }
                            /* let TX drain PKT_RDMA_CTS before spawning or during isend */
                            if (net->isend(gsc, tx_iova, (int)gsz, gtag, gtmr, &gsreq) != ncclSuccess || !gsreq) {
                                ok = false;
                                break;
                            }
                            for (int spin = 0; spin < 200000 && (!gsd || !grd); spin++) {
                                if (!gsd && net->test(gsreq, &gsd, NULL) != ncclSuccess) {
                                    ok = false;
                                    break;
                                }
                                if (!grd && net->test(grreq, &grd, &ggot) != ncclSuccess) {
                                    ok = false;
                                    break;
                                }
                            }
                            pthread_mutex_lock(&g_mock_ibv_mtx);
                            uint32_t post_rdma = g_mock_ibv.rdma_write_imm_pkts - pre_rdma;
                            uint32_t got_drops = g_mock_ibv.dropped_rdma_pkts;
                            g_mock_ibv.drop_rdma_rem = 0u;
                            pthread_mutex_unlock(&g_mock_ibv_mtx);

                            vxfec_nccl_stats_t gst;
                            get_stats_ex(&gst, sizeof(gst));
                            memset(h_dst, 0, gsz);
                            if (!ok || !gsd || !grd || ggot != (int)gsz || post_rdma == 0u || got_drops != want_drops ||
                                !vxfec_vk_buf_read_offset(vk, rvb, 0u, h_dst, gsz) || memcmp(h_src, h_dst, gsz) != 0) {
                                fprintf(stderr,
                                        "FAIL: GPUDirect RDMA VRAM subtest %d failed (sd=%d rd=%d got=%d rdma=%u drops=%u/%u)\n",
                                        sub, gsd, grd, ggot, post_rdma, got_drops, want_drops);
                                ok = false;
                                break;
                            }
                            if (sub > 0 && (gst.vk_offload_calls < 2u || gst.rx_fec_decodes == 0u)) {
                                fprintf(stderr,
                                        "FAIL: GPUDirect RDMA VRAM subtest %d missed GPU encode/decode (vk_calls=%lu fec=%lu)\n",
                                        sub, (unsigned long)gst.vk_offload_calls, (unsigned long)gst.rx_fec_decodes);
                                ok = false;
                                break;
                            }
                        }
                        set_codec(NULL);
                    }
                }
                if (gtmr)
                    net->deregMr(gsc, gtmr);
                if (grmr)
                    net->deregMr(grc, grmr);
                if (gsc)
                    net->closeSend(gsc);
                if (grc)
                    net->closeRecv(grc);
                if (glc)
                    net->closeListen(glc);
                if (tfd >= 0)
                    close(tfd);
                if (rfd >= 0)
                    close(rfd);
                vxfec_vk_buf_free(vk, tvb);
                vxfec_vk_buf_free(vk, rvb);
                free(h_poison);
                free(h_dst);
                free(h_src);
            }
            if (ok) {
                printf("PASS: GPUDirect RDMA_WRITE_WITH_IMM -> DMA-BUF VRAM + zero-copy out_dmabuf parity + in-place VRAM "
                       "RS8/RQ-GF2 decode\n");
            }
        } else if (vk) {
            vxfec_vk_ctx_free(vk);
        }
    }

    dlclose(dl);
    if (g_mock_ibv.vk_ctx) {
        vxfec_vk_ctx_free(g_mock_ibv.vk_ctx);
        g_mock_ibv.vk_ctx = NULL;
    }
    return ok ? 0 : 1;
}
#endif

static int count_self_threads(void)
{
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp)
        return -1;
    char line[256];
    int thr = -1;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "Threads:", 8) == 0) {
            thr = atoi(line + 8);
            break;
        }
    }
    fclose(fp);
    return thr;
}

static int count_self_fds(void)
{
    DIR *dp = opendir("/proc/self/fd");
    if (!dp)
        return -1;
    int cnt = 0;
    struct dirent *de;
    while ((de = readdir(dp)) != NULL) {
        if (de->d_name[0] != '.')
            cnt++;
    }
    closedir(dp);
    return cnt - 1; /* subtract dirfd opened by opendir */
}

static bool test_lost_done_and_send_abort_and_close_listen(const ncclNet_v8_t *net, void (*set_hs_fault)(uint32_t, uint32_t),
                                                           void (*set_done_fault)(uint32_t), void (*set_abort_fault)(uint32_t),
                                                           void (*set_delay)(uint64_t))
{
    /* 1) closeListen immediately after accept with dropped HELLO_ACK must connect within < 100 tight polls */
    {
        uint8_t handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
        void *lc = NULL, *sc = NULL, *rc = NULL;
        set_hs_fault(0u, 1u);
        if (net->listen(0, handle, &lc) != ncclSuccess || !lc) {
            set_hs_fault(0u, 0u);
            return false;
        }
        if (net->connect(0, handle, &sc, NULL) != ncclSuccess) {
            set_hs_fault(0u, 0u);
            net->closeListen(lc);
            return false;
        }
        uint64_t deadline = test_rq_now_ns() + 2000000000ULL;
        while (!rc && test_rq_now_ns() < deadline) {
            if (net->accept(lc, &rc, NULL) != ncclSuccess) {
                set_hs_fault(0u, 0u);
                net->closeListen(lc);
                return false;
            }
        }
        /* close listener right after accept while first HELLO_ACK was dropped */
        net->closeListen(lc);
        lc = NULL;
        if (!rc || sc != NULL) {
            set_hs_fault(0u, 0u);
            if (sc)
                net->closeSend(sc);
            if (rc)
                net->closeRecv(rc);
            fprintf(stderr, "FAIL: expected accepted rc and unconnected sc after dropped HELLO_ACK\n");
            return false;
        }
        int polls = 0;
        for (; polls < 100 && !sc; polls++) {
            if (net->connect(0, handle, &sc, NULL) != ncclSuccess) {
                set_hs_fault(0u, 0u);
                net->closeRecv(rc);
                return false;
            }
        }
        set_hs_fault(0u, 0u);
        if (!sc) {
            net->closeRecv(rc);
            fprintf(stderr, "FAIL: sender still unconnected after 100 polls post-closeListen with 1 dropped HELLO_ACK\n");
            return false;
        }

        /* 2) lost PKT_DONE replies: once rreq completes, polling ONLY sreq must still complete via RTO + RX epoll */
        uint8_t tx_buf[8192], rx_buf[8192] = {0};
        for (size_t i = 0; i < sizeof(tx_buf); i++)
            tx_buf[i] = (uint8_t)((i * 41u + 9u) & 0xffu);
        set_delay(5000000ULL);
        set_done_fault(4u);
        void *sreq = NULL, *rreq = NULL, *rptr = rx_buf;
        int cap = (int)sizeof(rx_buf), tag = 910, sdone = 0, rdone = 0, got = 0;
        if (net->irecv(rc, 1, &rptr, &cap, &tag, NULL, &rreq) != ncclSuccess || !rreq ||
            net->isend(sc, tx_buf, (int)sizeof(tx_buf), tag, NULL, &sreq) != ncclSuccess || !sreq) {
            set_done_fault(0u);
            net->closeSend(sc);
            net->closeRecv(rc);
            return false;
        }
        deadline = test_rq_now_ns() + 2000000000ULL;
        while (!rdone && test_rq_now_ns() < deadline) {
            if (net->test(rreq, &rdone, &got) != ncclSuccess || (!sdone && net->test(sreq, &sdone, NULL) != ncclSuccess)) {
                set_done_fault(0u);
                net->closeSend(sc);
                net->closeRecv(rc);
                return false;
            }
        }
        if (!rdone || got != (int)sizeof(tx_buf) || memcmp(tx_buf, rx_buf, sizeof(tx_buf)) != 0) {
            set_done_fault(0u);
            net->closeSend(sc);
            net->closeRecv(rc);
            fprintf(stderr, "FAIL: rreq did not complete before PKT_DONE loss check\n");
            return false;
        }
        /* stop polling rc completely; only sreq is tested from here on */
        deadline = test_rq_now_ns() + 2000000000ULL;
        while (!sdone && test_rq_now_ns() < deadline) {
            if (net->test(sreq, &sdone, NULL) != ncclSuccess) {
                set_done_fault(0u);
                net->closeSend(sc);
                net->closeRecv(rc);
                return false;
            }
            if (!sdone)
                usleep(200);
        }
        set_done_fault(0u);
        set_delay(0u);
        if (!sdone) {
            net->closeSend(sc);
            net->closeRecv(rc);
            fprintf(stderr, "FAIL: sreq hung when initial PKT_DONE replies were lost and rreq was already retired\n");
            return false;
        }

        /* 3) local TX send error (EMSGSIZE) must fail matching irecv even when initial PKT_ERR datagrams are lost */
        for (int order = 0; order < 2; order++) {
            int etag = 920 + order;
            rreq = NULL;
            sreq = NULL;
            rdone = 0;
            sdone = 0;
            got = 0;
            if (order == 0) {
                if (net->irecv(rc, 1, &rptr, &cap, &etag, NULL, &rreq) != ncclSuccess || !rreq) {
                    net->closeSend(sc);
                    net->closeRecv(rc);
                    return false;
                }
            }
            set_abort_fault(3u);
            test_wire_mtu = 500;
            if (net->isend(sc, tx_buf, (int)sizeof(tx_buf), etag, NULL, &sreq) != ncclSuccess || !sreq) {
                test_wire_mtu = 0;
                set_abort_fault(0u);
                net->closeSend(sc);
                net->closeRecv(rc);
                return false;
            }
            ncclResult_t tx_st = net->test(sreq, &sdone, NULL);
            test_wire_mtu = 0;
            if (tx_st != ncclSystemError) {
                set_abort_fault(0u);
                net->closeSend(sc);
                net->closeRecv(rc);
                fprintf(stderr, "FAIL: expected ncclSystemError on oversized TX send (got %d)\n", (int)tx_st);
                return false;
            }
            if (order == 1) {
                usleep(1000);
                if (net->irecv(rc, 1, &rptr, &cap, &etag, NULL, &rreq) != ncclSuccess || !rreq) {
                    set_abort_fault(0u);
                    net->closeSend(sc);
                    net->closeRecv(rc);
                    return false;
                }
            }
            /* stop polling sc completely; background RTO must retransmit dropped PKT_ERR */
            ncclResult_t rx_st = ncclSuccess;
            deadline = test_rq_now_ns() + 2000000000ULL;
            while (rx_st == ncclSuccess && !rdone && test_rq_now_ns() < deadline) {
                rx_st = net->test(rreq, &rdone, &got);
                if (rx_st == ncclSuccess && !rdone)
                    usleep(200);
            }
            set_abort_fault(0u);
            if (rx_st != ncclRemoteError || rdone != 0) {
                net->closeSend(sc);
                net->closeRecv(rc);
                fprintf(stderr, "FAIL: lost initial PKT_ERR left matching irecv pending (order=%d rx_st=%d rdone=%d)\n", order,
                        (int)rx_st, rdone);
                return false;
            }
        }

        net->closeSend(sc);
        net->closeRecv(rc);
    }

    /* 3b) delayed duplicate PKT_HELLO after transfer must NOT create a second receive connection */
    {
        int rfd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (rfd < 0)
            return false;
        struct sockaddr_in rsin = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
        socklen_t rlen = sizeof(rsin);
        if (bind(rfd, (struct sockaddr *)&rsin, rlen) != 0 || getsockname(rfd, (struct sockaddr *)&rsin, &rlen) != 0) {
            close(rfd);
            return false;
        }
        uint8_t handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
        void *lc = NULL, *sc = NULL, *rc = NULL;
        if (net->listen(0, handle, &lc) != ncclSuccess || !lc) {
            close(rfd);
            return false;
        }
        uint16_t real_port_be = 0;
        memcpy(&real_port_be, handle + 8, sizeof(uint16_t));
        memcpy(handle + 8, &rsin.sin_port, sizeof(uint16_t));
        struct sockaddr_in real_lc = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK), .sin_port = real_port_be};
        struct sockaddr_in cli_addr;
        memset(&cli_addr, 0, sizeof(cli_addr));
        uint8_t saved_hello[256];
        size_t saved_hello_len = 0u;

        uint64_t dl = test_rq_now_ns() + 2000000000ULL;
        while ((!sc || !rc) && test_rq_now_ns() < dl) {
            if (!sc && net->connect(0, handle, &sc, NULL) != ncclSuccess)
                break;
            for (;;) {
                uint8_t pkt[256];
                struct sockaddr_in src;
                socklen_t slen = sizeof(src);
                ssize_t n = recvfrom(rfd, pkt, sizeof(pkt), MSG_DONTWAIT, (struct sockaddr *)&src, &slen);
                if (n <= 0)
                    break;
                if (src.sin_port != real_port_be) {
                    cli_addr = src;
                    if (saved_hello_len == 0u && (size_t)n <= sizeof(saved_hello)) {
                        memcpy(saved_hello, pkt, (size_t)n);
                        saved_hello_len = (size_t)n;
                    }
                    (void)sendto(rfd, pkt, (size_t)n, MSG_DONTWAIT, (struct sockaddr *)&real_lc, sizeof(real_lc));
                } else if (cli_addr.sin_port != 0) {
                    (void)sendto(rfd, pkt, (size_t)n, MSG_DONTWAIT, (struct sockaddr *)&cli_addr, sizeof(cli_addr));
                }
            }
            if (!rc && net->accept(lc, &rc, NULL) != ncclSuccess)
                break;
        }
        if (!sc || !rc || saved_hello_len == 0u) {
            close(rfd);
            if (sc)
                net->closeSend(sc);
            if (rc)
                net->closeRecv(rc);
            net->closeListen(lc);
            return false;
        }
        /* complete a transfer so handshake_confirmed transitions to true while lc stays open */
        uint32_t tval = 0x11223344u, rval = 0u;
        void *sreq = NULL, *rreq = NULL, *rp = &rval;
        int sz = (int)sizeof(rval), tag = 930, sd = 0, rd = 0, got = 0;
        if (net->irecv(rc, 1, &rp, &sz, &tag, NULL, &rreq) != ncclSuccess ||
            net->isend(sc, &tval, sz, tag, NULL, &sreq) != ncclSuccess) {
            close(rfd);
            net->closeSend(sc);
            net->closeRecv(rc);
            net->closeListen(lc);
            return false;
        }
        dl = test_rq_now_ns() + 2000000000ULL;
        while ((!sd || !rd) && test_rq_now_ns() < dl) {
            if (!sd && net->test(sreq, &sd, NULL) != ncclSuccess)
                break;
            if (!rd && net->test(rreq, &rd, &got) != ncclSuccess)
                break;
        }
        /* inject delayed duplicate PKT_HELLO from the same sender address while rc is still open */
        (void)sendto(rfd, saved_hello, saved_hello_len, MSG_DONTWAIT, (struct sockaddr *)&real_lc, sizeof(real_lc));
        usleep(2000);
        void *dup_rc = NULL;
        bool dup_ok = (sd && rd && rval == tval && net->accept(lc, &dup_rc, NULL) == ncclSuccess && dup_rc == NULL);
        net->closeSend(sc);
        net->closeRecv(rc);
        /* also verify delayed duplicate PKT_HELLO after closeRecv still creates no second recvComm */
        if (dup_ok) {
            (void)sendto(rfd, saved_hello, saved_hello_len, MSG_DONTWAIT, (struct sockaddr *)&real_lc, sizeof(real_lc));
            usleep(2000);
            dup_ok = (net->accept(lc, &dup_rc, NULL) == ncclSuccess && dup_rc == NULL);
        }
        if (dup_rc)
            net->closeRecv(dup_rc);
        net->closeListen(lc);
        close(rfd);
        if (!dup_ok) {
            fprintf(stderr, "FAIL: delayed duplicate PKT_HELLO created a second receive connection\n");
            return false;
        }
    }

    /* 3c) 6 closed listeners must reap their threads and FDs as soon as their handshakes are confirmed */
    {
        int base_thr = count_self_threads();
        int base_fds = count_self_fds();
        void *scs[6] = {0}, *rcs[6] = {0};
        bool reap_ok = (base_thr > 0 && base_fds > 0);
        int one_pair_fds = 0;

        for (int c = 0; c < 6 && reap_ok; c++) {
            uint8_t h[NCCL_NET_HANDLE_MAXSIZE] = {0};
            void *lc = NULL;
            if (net->listen(0, h, &lc) != ncclSuccess || !lc) {
                reap_ok = false;
                break;
            }
            uint64_t dl = test_rq_now_ns() + 2000000000ULL;
            while ((!scs[c] || !rcs[c]) && test_rq_now_ns() < dl) {
                if (!scs[c] && net->connect(0, h, &scs[c], NULL) != ncclSuccess) {
                    reap_ok = false;
                    break;
                }
                if (!rcs[c] && net->accept(lc, &rcs[c], NULL) != ncclSuccess) {
                    reap_ok = false;
                    break;
                }
            }
            net->closeListen(lc);
            if (!scs[c] || !rcs[c]) {
                reap_ok = false;
                break;
            }
            uint32_t tv = 0x9000u + (uint32_t)c, rv = 0u;
            void *sq = NULL, *rq = NULL, *rp = &rv;
            int sz = (int)sizeof(rv), tag = 940 + c, sd = 0, rd = 0, got = 0;
            if (net->irecv(rcs[c], 1, &rp, &sz, &tag, NULL, &rq) != ncclSuccess ||
                net->isend(scs[c], &tv, sz, tag, NULL, &sq) != ncclSuccess) {
                reap_ok = false;
                break;
            }
            dl = test_rq_now_ns() + 2000000000ULL;
            while ((!sd || !rd) && test_rq_now_ns() < dl) {
                if (!sd && net->test(sq, &sd, NULL) != ncclSuccess)
                    break;
                if (!rd && net->test(rq, &rd, &got) != ncclSuccess)
                    break;
            }
            if (!sd || !rd || rv != tv)
                reap_ok = false;
            if (c == 0 && reap_ok) {
                usleep(2000);
                one_pair_fds = count_self_fds() - base_fds;
            }
        }
        usleep(5000);
        int post_thr = count_self_threads();
        int post_fds = count_self_fds();
        for (int c = 0; c < 6; c++) {
            if (scs[c])
                net->closeSend(scs[c]);
            if (rcs[c])
                net->closeRecv(rcs[c]);
        }
        if (!reap_ok || post_thr != base_thr || one_pair_fds <= 0 || post_fds != base_fds + 6 * one_pair_fds) {
            fprintf(stderr, "FAIL: retired listeners leaked threads/FDs (thr %d->%d, fds %d->%d, 1pair=%d)\n", base_thr, post_thr,
                    base_fds, post_fds, one_pair_fds);
            return false;
        }
    }

    /* 4) unmatched interface filter must expose 0 devices instead of falling back to loopback */
    {
        const char *old_if = getenv("VXFEC_NCCL_IFNAMES");
        char saved_if[256] = {0};
        bool had_if = (old_if != NULL);
        if (had_if)
            snprintf(saved_if, sizeof(saved_if), "%.*s", (int)(sizeof(saved_if) - 1u), old_if);

        if (setenv("VXFEC_NCCL_IFNAMES", "vxfec_nonexistent_if99", 1) != 0)
            return false;
        int unmatched_ndev = -1;
        bool filter_ok =
            (net->init(dummy_nccl_logger) == ncclSuccess && net->devices(&unmatched_ndev) == ncclSuccess && unmatched_ndev == 0);
        if (had_if)
            (void)setenv("VXFEC_NCCL_IFNAMES", saved_if, 1);
        else
            (void)unsetenv("VXFEC_NCCL_IFNAMES");
        if (net->init(dummy_nccl_logger) != ncclSuccess || !filter_ok) {
            fprintf(stderr, "FAIL: unmatched VXFEC_NCCL_IFNAMES exposed ndev=%d instead of 0\n", unmatched_ndev);
            return false;
        }
    }

    printf("PASS: closeListen ACK replay (<100 polls), lost PKT_DONE recovery, TX->RX error propagation, and strict IF filter\n");
    return true;
}

/* [P1] regression: reject sym_T==0 / misaligned sym_T without slot leaks, and prevent 4-tag sliced HoL starvation */
static bool test_phase3_wire_and_hol_regressions(const ncclNet_v8_t *net)
{
    uint8_t handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
    void *lc = NULL, *sc = NULL, *rc = NULL;
    int raw_fd = -1;
    bool ok = false;
    enum { N_TAGS = 4, SLICE_MSG_BYTES = 2 * 1024 * 1024 };
    uint8_t *tx_bufs[N_TAGS] = {0};
    uint8_t *rx_bufs[N_TAGS] = {0};

    if (net->listen(0, handle, &lc) != ncclSuccess || !lc)
        goto out;
    uint64_t dl = test_rq_now_ns() + 3000000000ULL;
    while ((!sc || !rc) && test_rq_now_ns() < dl) {
        if (!sc && net->connect(0, handle, &sc, NULL) != ncclSuccess)
            goto out;
        if (!rc && net->accept(lc, &rc, NULL) != ncclSuccess)
            goto out;
    }
    if (!sc || !rc)
        goto out;

    /* 1) flood 160 malformed PKT_DATA frames (sym_T == 0 and misaligned sym_T == 1282) > VXFEC_RX_WIRE_SLOTS (128):
     *    if any malformed packet wedged an RX slot, subsequent transfers would starve */
    test_handle_t h;
    memcpy(&h, handle, sizeof(h));
    raw_fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (raw_fd < 0)
        goto out;

    /* discover rc's ctrl_fd port by inspecting the connected peer address of sc's ctrl_fd? Even simpler:
     * send malformed frames via a raw HELLO handshake on lc so we have rc2's exact rx_ports_be[0]! */
    {
        struct sockaddr_in csin = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be};
        socklen_t clen = sizeof(csin);
        if (bind(raw_fd, (struct sockaddr *)&csin, clen) != 0 || getsockname(raw_fd, (struct sockaddr *)&csin, &clen) != 0)
            goto out;
        test_hello_t mh;
        memset(&mh, 0, sizeof(mh));
        mh.magic = h.magic;
        mh.pkt_type = 1u;
        mh.num_ports = h.num_ports;
        mh.cookie = h.cookie;
        mh.tx_ctrl_port_be = csin.sin_port;
        mh.conn_id = 0x99112233ULL;
        for (uint16_t p = 0; p < h.num_ports; p++)
            mh.tx_ports_be[p] = csin.sin_port;
        struct sockaddr_in lc_addr = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be, .sin_port = h.ctrl_port_be};
        test_ack_t mack;
        memset(&mack, 0, sizeof(mack));
        void *rc2 = NULL;
        bool got_mack = false;
        for (int spin = 0; spin < 20000 && (!rc2 || !got_mack); spin++) {
            if ((spin % 250) == 0)
                (void)sendto(raw_fd, &mh, sizeof(mh), MSG_DONTWAIT, (struct sockaddr *)&lc_addr, sizeof(lc_addr));
            if (!got_mack && recvfrom(raw_fd, &mack, sizeof(mack), MSG_DONTWAIT, NULL, NULL) == (ssize_t)sizeof(mack) &&
                mack.magic == h.magic && mack.conn_id == mh.conn_id)
                got_mack = true;
            if (!rc2 && net->accept(lc, &rc2, NULL) != ncclSuccess)
                goto out;
        }
        if (!rc2 || !got_mack) {
            if (rc2)
                net->closeRecv(rc2);
            goto out;
        }
        struct sockaddr_in rdata = {.sin_family = AF_INET, .sin_addr.s_addr = h.ip_be, .sin_port = mack.rx_ports_be[0]};
        uint8_t bad_pkt[sizeof(test_wire_hdr_t) + 64];
        memset(bad_pkt, 0x5a, sizeof(bad_pkt));
        for (uint32_t seq = 1; seq <= 160u; seq++) {
            test_wire_hdr_t bh = {
                .magic = h.magic,
                .cookie = h.cookie,
                .pkt_type = 3u,
                .flags = 0x02u,
                .sym_len = 64u,
                .msg_seq = seq,
                .tag_seq = seq,
                .tag = 777u + seq,
                .total_bytes = 64u,
                .K = 1u,
                .sym_T = (seq & 1u) ? 0u : 1282u,
                .esi = 0u,
            };
            memcpy(bad_pkt, &bh, sizeof(bh));
            (void)sendto(raw_fd, bad_pkt, sizeof(bad_pkt), MSG_DONTWAIT, (struct sockaddr *)&rdata, sizeof(rdata));
        }
        /* pump rc2 and verify a valid single-symbol packet on msg_seq=300 still allocates a slot and completes */
        uint8_t good_rx[64] = {0};
        void *grreq = NULL, *gptr = good_rx;
        int gcap = 64, gtag = 999, gdone = 0, ggot = 0;
        if (net->irecv(rc2, 1, &gptr, &gcap, &gtag, NULL, &grreq) != ncclSuccess || !grreq) {
            net->closeRecv(rc2);
            goto out;
        }
        test_wire_hdr_t gh = {
            .magic = h.magic,
            .cookie = h.cookie,
            .pkt_type = 3u,
            .flags = 0x02u,
            .sym_len = 64u,
            .msg_seq = 300u,
            .tag_seq = 1u,
            .tag = (uint32_t)gtag,
            .total_bytes = 64u,
            .K = 1u,
            .sym_T = 1280u,
            .esi = 0u,
        };
        memcpy(bad_pkt, &gh, sizeof(gh));
        (void)sendto(raw_fd, bad_pkt, sizeof(bad_pkt), MSG_DONTWAIT, (struct sockaddr *)&rdata, sizeof(rdata));
        for (int spin = 0; spin < 20000 && !gdone; spin++) {
            if (net->test(grreq, &gdone, &ggot) != ncclSuccess)
                break;
        }
        net->closeRecv(rc2);
        if (!gdone || ggot != 64) {
            fprintf(stderr, "FAIL: malformed sym_T=0/1282 packets wedged RX wire slots\n");
            goto out;
        }
    }

    /* 2) post 4 concurrent 2 MiB sliced isend() requests on tags 700..703, then drain irecv() in reverse order 703..700 */
    void *sreqs[N_TAGS] = {0};
    for (int t = 0; t < N_TAGS; t++) {
        tx_bufs[t] = (uint8_t *)malloc(SLICE_MSG_BYTES);
        rx_bufs[t] = (uint8_t *)calloc(1, SLICE_MSG_BYTES);
        if (!tx_bufs[t] || !rx_bufs[t])
            goto out;
        for (size_t i = 0; i < SLICE_MSG_BYTES; i++)
            tx_bufs[t][i] = (uint8_t)((i * 31u + (size_t)t * 53u + 7u) & 0xffu);
        int tag = 700 + t;
        if (net->isend(sc, tx_bufs[t], SLICE_MSG_BYTES, tag, NULL, &sreqs[t]) != ncclSuccess || !sreqs[t]) {
            fprintf(stderr, "FAIL: 4-tag concurrent sliced isend starved on tag %d (sreq=%p)\n", tag, sreqs[t]);
            goto out;
        }
    }

    int sdone[N_TAGS] = {0};
    for (int t = N_TAGS - 1; t >= 0; t--) {
        void *rreq = NULL, *rptr = rx_bufs[t];
        int rcap = SLICE_MSG_BYTES, tag = 700 + t, rdone = 0, got = 0;
        if (net->irecv(rc, 1, &rptr, &rcap, &tag, NULL, &rreq) != ncclSuccess || !rreq)
            goto out;
        uint64_t tdl = test_rq_now_ns() + 5000000000ULL;
        while ((!rdone || !sdone[t]) && test_rq_now_ns() < tdl) {
            for (int k = 0; k < N_TAGS; k++) {
                if (!sdone[k] && net->test(sreqs[k], &sdone[k], NULL) != ncclSuccess)
                    goto out;
            }
            if (!rdone && net->test(rreq, &rdone, &got) != ncclSuccess)
                goto out;
        }
        if (!rdone || !sdone[t] || got != SLICE_MSG_BYTES || memcmp(tx_bufs[t], rx_bufs[t], SLICE_MSG_BYTES) != 0) {
            fprintf(stderr, "FAIL: reverse-order 4-tag sliced transfer failed on tag %d (rd=%d sd=%d got=%d)\n", tag, rdone,
                    sdone[t], got);
            goto out;
        }
    }

    printf("PASS: sym_T==0/misaligned rejection and 4-tag reverse-order sliced isend HoL anti-deadlock verified\n");
    ok = true;

out:
    for (int t = 0; t < N_TAGS; t++) {
        free(tx_bufs[t]);
        free(rx_bufs[t]);
    }
    if (raw_fd >= 0)
        close(raw_fd);
    if (sc)
        net->closeSend(sc);
    if (rc)
        net->closeRecv(rc);
    if (lc)
        net->closeListen(lc);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc == 4 && strcmp(argv[1], "--sender-child") == 0)
        return run_sender_child_mode(argv[2], atoi(argv[3]));
    if (argc == 2 && strcmp(argv[1], "--rq-async") == 0)
        return run_rq_gpu_progress_tests();
#ifdef VXFEC_HAVE_IBVERBS
    if (argc == 2 && strcmp(argv[1], "--ibv-mock") == 0)
        return run_ibv_mock_tests();
#else
    if (argc == 2 && strcmp(argv[1], "--ibv-mock") == 0)
        return 0;
#endif

    if (!test_harq_math() || !test_ifname_filter_syntax())
        return 1;
    printf("PASS: IID parity budget, HARQ SACK/IR, SWRR math, and NCCL interface filter syntax\n");

    const char *so_path = getenv("VXFEC_NCCL_SO_PATH");
    if (!so_path || !*so_path)
        so_path = "./libnccl-vxfec.so";
    void *dl = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        fprintf(stderr, "FAIL: dlopen(%s): %s\n", so_path, dlerror());
        return 1;
    }

    const ncclNet_v8_t *net = (const ncclNet_v8_t *)dlsym(dl, "ncclNetPlugin_v8");
    test_get_stats_v1_fn get_stats_v1 = (test_get_stats_v1_fn)dlsym(dl, "vxfec_nccl_get_stats");
    g_test_get_stats_ex = (test_get_stats_ex_fn)dlsym(dl, "vxfec_nccl_get_stats_ex");
    void (*get_stats)(vxfec_nccl_stats_t *) = g_test_get_stats_ex ? test_query_stats : NULL;
    void (*reset_stats)(void) = (void (*)(void))dlsym(dl, "vxfec_nccl_reset_stats");
    void (*set_fault)(uint32_t, uint32_t, bool) = (void (*)(uint32_t, uint32_t, bool))dlsym(dl, "vxfec_nccl_set_fault_injection");
    void (*set_hs_fault)(uint32_t, uint32_t) = (void (*)(uint32_t, uint32_t))dlsym(dl, "vxfec_nccl_set_handshake_fault");
    void (*set_done_fault)(uint32_t) = (void (*)(uint32_t))dlsym(dl, "vxfec_nccl_set_done_fault");
    void (*set_abort_fault)(uint32_t) = (void (*)(uint32_t))dlsym(dl, "vxfec_nccl_set_abort_fault");
    void (*set_delay)(uint64_t) = (void (*)(uint64_t))dlsym(dl, "vxfec_nccl_set_link_delay_ns");
    void (*set_gpu_rx_fault)(uint32_t) = (void (*)(uint32_t))dlsym(dl, "vxfec_nccl_set_gpu_rx_fault");
    void (*set_cong)(uint32_t, uint32_t, uint64_t) =
        (void (*)(uint32_t, uint32_t, uint64_t))dlsym(dl, "vxfec_nccl_set_congestion_sim");
    void (*set_codec)(const char *) = (void (*)(const char *))dlsym(dl, "vxfec_nccl_set_codec_override");

    if (!net || !get_stats_v1 || !get_stats || !reset_stats || !set_fault || !set_hs_fault || !set_done_fault || !set_abort_fault ||
        !set_delay || !set_gpu_rx_fault || !set_cong || !set_codec) {
        fprintf(stderr, "FAIL: missing symbols in libnccl-vxfec.so\n");
        dlclose(dl);
        return 1;
    }

    if (!test_stats_abi_versioning(get_stats_v1, g_test_get_stats_ex, reset_stats)) {
        dlclose(dl);
        return 1;
    }

    int ndev = 0;
    ncclNetProperties_v8_t props;
    if (net->init(dummy_nccl_logger) != ncclSuccess || net->devices(&ndev) != ncclSuccess || ndev < 1 ||
        net->getProperties(0, &props) != ncclSuccess) {
        fprintf(stderr, "FAIL: plugin init/getProperties failed\n");
        dlclose(dl);
        return 1;
    }
    printf("Loaded plugin '%s' (dev='%s', speed=%d Mbps, ptrSupport=0x%x)\n", net->name, props.name, props.speed, props.ptrSupport);

    if (!test_path_mtu_limits(net) || !test_fresh_ecn_feedback(net) || !test_low_rate_cap_and_tail_clock(net, set_cong) ||
        !test_lost_done_and_send_abort_and_close_listen(net, set_hs_fault, set_done_fault, set_abort_fault, set_delay)) {
        set_delay(0u);
        dlclose(dl);
        return 1;
    }

    ring_comms_t ring;
    if (!setup_ring(net, &ring)) {
        fprintf(stderr, "FAIL: 4-rank ECMP ring setup failed\n");
        dlclose(dl);
        return 1;
    }

    /* Scenario A: 0% Lossless Fabric */
    reset_stats();
    set_fault(0, 1, false);
    if (!run_host_ring_allreduce(net, &ring, 32768, 1))
        goto fail;
    vxfec_nccl_stats_t st_a;
    get_stats(&st_a);
    if (st_a.tx_proactive_rep_pkts != 0 || st_a.rx_fec_decodes != 0 || st_a.rx_fast_completions != 24)
        goto fail;
    printf("PASS: 4-rank Ring AllReduce [0%% lossless]: 0 parity tax, %lu fast completions\n",
           (unsigned long)st_a.rx_fast_completions);

    /* Scenario B: Out-of-order ECMP + sparse 0.3% single-packet drops -> SACK */
    reset_stats();
    set_fault(3000, 1, true);
    if (!run_host_ring_allreduce(net, &ring, 32768, 2))
        goto fail;
    vxfec_nccl_stats_t st_b;
    get_stats(&st_b);
    printf("PASS: 4-rank Ring AllReduce [ECMP OOO + 0.3%% loss]: ooo=%lu dropped=%lu sack_retx=%lu ir=%lu\n",
           (unsigned long)st_b.rx_ooo_pkts, (unsigned long)st_b.sim_dropped_pkts, (unsigned long)st_b.tx_sack_retx_pkts,
           (unsigned long)st_b.tx_harq_ir_pkts);

    /* Scenario C1: 3.0% Microburst (burst=6, K=103) -> Cauchy RS8 MDS (`ir_slack = 0`) */
    reset_stats();
    set_fault(30000, 6, true);
    for (uint32_t iter = 0; iter < 3; iter++) {
        if (!run_host_ring_allreduce(net, &ring, 32768, 10u + iter))
            goto fail;
    }
    vxfec_nccl_stats_t st_c1;
    get_stats(&st_c1);
    if (st_c1.rx_rs8_decodes == 0) {
        fprintf(stderr, "FAIL: expected Cauchy RS8 MDS decodes at K=103\n");
        goto fail;
    }
    printf("PASS: 4-rank Ring AllReduce [Cauchy RS8 MDS ir_slack=0, 3%% burst=6]: dropped=%lu harq_ir=%lu "
           "proactive=%lu rs8_dec=%lu vk=%lu\n",
           (unsigned long)st_c1.sim_dropped_pkts, (unsigned long)st_c1.tx_harq_ir_pkts, (unsigned long)st_c1.tx_proactive_rep_pkts,
           (unsigned long)st_c1.rx_rs8_decodes, (unsigned long)st_c1.vk_offload_calls);

    /* Scenario C2: 3.0% Microburst (burst=6, K=205) -> Automatic promotion to RaptorQ */
    reset_stats();
    set_fault(30000, 6, true);
    for (uint32_t iter = 0; iter < 2; iter++) {
        if (!run_host_ring_allreduce(net, &ring, 65536, 20u + iter))
            goto fail;
    }
    vxfec_nccl_stats_t st_c2;
    get_stats(&st_c2);
    if (st_c2.rx_rq_decodes == 0) {
        fprintf(stderr, "FAIL: expected RaptorQ promotion decodes at K=205\n");
        goto fail;
    }
    printf("PASS: 4-rank Ring AllReduce [RaptorQ auto-promoted K=205, 3%% burst=6]: dropped=%lu rq_dec=%lu vk=%lu\n",
           (unsigned long)st_c2.sim_dropped_pkts, (unsigned long)st_c2.rx_rq_decodes, (unsigned long)st_c2.vk_offload_calls);

    /* scenario D: GPU VRAM DMA-BUF (`NCCL_PTR_DMABUF` + `regMrDmaBuf` + `iflush`) */
    reset_stats();
    set_fault(30000, 5, true);
    for (uint32_t iter = 0; iter < 3; iter++) {
        if (!run_dmabuf_ring_allgather(net, &props, &ring, 32768))
            goto fail;
    }
    if (props.ptrSupport & NCCL_PTR_DMABUF) {
        vxfec_nccl_stats_t st_d;
        get_stats(&st_d);
        if (st_d.tx_messages > 0u && st_d.rx_fec_decodes == 0u) {
            fprintf(stderr, "FAIL: GPU VRAM DMA-BUF Ring AllGather under 3%% burst=5 completed with 0 FEC decodes\n");
            goto fail;
        }
        printf("PASS: 4-rank GPU VRAM DMA-BUF Ring AllGather [3.0%% burst=5]: dropped=%lu fec_dec=%lu vk=%lu\n",
               (unsigned long)st_d.sim_dropped_pkts, (unsigned long)st_d.rx_fec_decodes, (unsigned long)st_d.vk_offload_calls);
    }

    /* Scenario E: ECN Congestion-Window Pacing + Hotspot Port Deflection + FEC Clamping */
    reset_stats();
    set_fault(20000, 4, true);
    set_cong(600000, 0x0003u, 1600);
    for (uint32_t iter = 0; iter < 4; iter++) {
        if (!run_host_ring_allreduce(net, &ring, 32768, 40u + iter))
            goto fail;
    }
    vxfec_nccl_stats_t st_e;
    get_stats(&st_e);
    set_cong(0, 0, 0);
    if (st_e.rx_ecn_ce_pkts == 0 || st_e.tx_cnp_events == 0 || st_e.tx_paced_slices == 0 || st_e.tx_fec_clamped_pkts == 0) {
        fprintf(stderr, "FAIL: expected ECN/CNP pacing & FEC clamping (ecn=%lu cnp=%lu slices=%lu clamped=%lu)\n",
                (unsigned long)st_e.rx_ecn_ce_pkts, (unsigned long)st_e.tx_cnp_events, (unsigned long)st_e.tx_paced_slices,
                (unsigned long)st_e.tx_fec_clamped_pkts);
        goto fail;
    }
    printf("PASS: 4-rank Ring AllReduce [ECN Pacing + SWRR Deflection + FEC Clamp]: ecn=%lu cnp=%lu paced_slices=%lu "
           "fec_clamped=%lu\n",
           (unsigned long)st_e.rx_ecn_ce_pkts, (unsigned long)st_e.tx_cnp_events, (unsigned long)st_e.tx_paced_slices,
           (unsigned long)st_e.tx_fec_clamped_pkts);

    if (!test_handshake_loss_recovery(net, set_hs_fault))
        goto fail;
    printf("PASS: Handshake retry recovery across dropped HELLO and HELLO_ACK datagrams\n");

    if (!test_recv_capacity_enforcement(net))
        goto fail;
    printf("PASS: Host receive capacity enforcement (oversized send rejected without overwrite)\n");

    if (!test_rs8_to_rq_promotion_integrity(net, get_stats, reset_stats, set_fault, set_codec))
        goto fail;

    if (!test_zero_rtt_proactive_fec_repair(net, get_stats, reset_stats, set_codec))
        goto fail;

    if (!test_paced_recovery_and_cwnd_bounds(net, get_stats, reset_stats, set_fault, set_cong, set_codec))
        goto fail;

    if (!test_gpu_rx_write_failure_propagation(net, &props, set_fault, set_gpu_rx_fault))
        goto fail;

    if (!test_multi_tag_ooo_irecv(net) || !test_phase3_wire_and_hol_regressions(net))
        goto fail;

    if (!test_rq_wire_selection(net, set_codec, NULL, false, false) || !test_rq_wire_selection(net, set_codec, NULL, true, false) ||
        !test_rq_wire_selection(net, set_codec, "rq-gf256", true, false) ||
        !test_rq_wire_selection(net, set_codec, "rq-gf2", false, false) ||
        !test_rq_wire_selection(net, set_codec, "rq-gf2", true, false) ||
        !test_rq_wire_selection(net, set_codec, NULL, true, true) ||
        !test_rq_wire_selection(net, set_codec, "rq-gf2", true, true) || !test_truncated_sys_pkt_rejection(net))
        goto fail;
    printf("PASS: 2 MiB RaptorQ wire selection and VXFEC_HELLO_SLICE feature negotiation verified\n");

    if (!test_const_shared_connect_handle(net, set_hs_fault))
        goto fail;

    if (!test_tag_collision_non_eviction(net))
        goto fail;

    if (!test_handshake_ack_loss_many_conns(net, set_hs_fault))
        goto fail;

    if (!test_cross_process_sender_connect(net, (argc > 0 && argv[0] && *argv[0]) ? argv[0] : "./t/test_nccl_plugin"))
        goto fail;

    if (!test_multithreaded_burst_drop_safety(net, &ring, set_fault))
        goto fail;

    if (!test_large_512mib_transfer(net, get_stats, reset_stats, set_fault))
        goto fail;

    if (!test_rs8_subblock_slicing_and_multichannel(net, &props, get_stats, reset_stats, set_fault))
        goto fail;

    if (!test_delayed_loss_recovery(net, get_stats, reset_stats, set_fault, set_delay, set_codec))
        goto fail;

    teardown_ring(net, &ring);
    dlclose(dl);
    printf("All vxfec NCCL v8 ECMP + Cauchy RS8 / RaptorQ HARQ + DMA-BUF tests passed!\n");
    return 0;

fail:
    set_cong(0, 0, 0);
    teardown_ring(net, &ring);
    dlclose(dl);
    return 1;
}
