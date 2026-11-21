#define _GNU_SOURCE
#include "bench_common.h"
#include "ibv_transport.h"
#include "vxfec_interleave.h"
#include "vxfec_vk.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define WIRE_NET_MAGIC 0x5658574Eu /* "VXWN" */
#define WIRE_NET_GDR_MAGIC 0x56584752u      /* "VXGR": DMA-BUF required on both peers */
#define WIRE_NET_PIPELINE_MAGIC 0x56584750u /* "VXGP": two RX DMA-BUF regions */
#define WIRE_NET_MAX_RAILS 128u
#define WIRE_NET_IBV_QPS 4u
#define WIRE_NET_BATCH 64u
#define WIRE_NET_FLAG_TAIL 0x0001u

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint16_t iter;
    uint16_t flags;
    uint64_t wire_seq;
} wire_pkt_hdr_t;

typedef struct {
    uint32_t magic;
    uint32_t codec;
    uint32_t Z;
    uint32_t K;
    uint32_t P;
    uint32_t T;
    uint32_t rails;
    uint32_t iters;
    uint32_t use_ibv;
    uint32_t wire_only;
    uint64_t remote_iova;
    uint32_t remote_rkey;
    uint32_t gdr_dmabuf;
    vxfec_ibv_wire_info_t ibv_info;
} wire_ctrl_handshake_t;

typedef struct {
    uint64_t iova;
    uint32_t rkey;
    uint32_t reserved;
} wire_rdma_region_t;

typedef struct {
    uint32_t iter;
    uint32_t ok;
    uint32_t fatal_err;
    uint64_t rx_frames;
    uint64_t wire_ns;
    uint64_t dma_ns;
    uint64_t recover_wall_ns;
    uint64_t recover_gpu_ns;
    uint32_t blocks_repaired;
    uint32_t unique_masks;
} wire_ctrl_result_t;

typedef struct {
    uint32_t iter;
    uint32_t ok;
    uint32_t fatal_err;
    uint64_t rx_bytes;
    uint64_t wire_ns;
    uint64_t wall_ns;
} wire_bidir_iter_t;

static uint64_t now_ns(void)
{
    return vxfec_bench_now_ns();
}

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static uint64_t count_planned_tx_frames(uint64_t total_frames, uint64_t drop_beg, uint64_t drop_end, uint32_t loss_ppm,
                                        uint32_t rails, uint32_t iter, bool use_ibv)
{
    uint64_t planned = 0u;
    if (use_ibv) {
        uint64_t rng = 0x9876543210fedcbaULL ^ (uint64_t)iter;
        for (uint64_t seq = 0; seq < total_frames; seq++) {
            if (seq >= drop_beg && seq < drop_end)
                continue;
            if (loss_ppm > 0u && (splitmix64(&rng) % 1000000ULL) < loss_ppm)
                continue;
            planned++;
        }
        return planned;
    }
    for (uint32_t r = 0; r < rails; r++) {
        uint64_t rng = 0x123456789abcdef0ULL ^ ((uint64_t)r << 32) ^ (uint16_t)iter;
        for (uint64_t seq = r; seq < total_frames; seq += rails) {
            if (seq >= drop_beg && seq < drop_end)
                continue;
            if (loss_ppm > 0u && (splitmix64(&rng) % 1000000ULL) < loss_ppm)
                continue;
            planned++;
        }
    }
    return planned;
}

static void fill_deterministic_superframe(uint8_t *buf, size_t sys_bytes)
{
    uint64_t *w = (uint64_t *)buf;
    size_t nwords = sys_bytes / 8u;
    for (size_t i = 0; i < nwords; i++) {
        uint64_t x = (uint64_t)i * 0x9e3779b97f4a7c15ULL + 0x517cc1b727220a95ULL;
        x ^= x >> 30;
        w[i] = x;
    }
    for (size_t i = nwords * 8u; i < sys_bytes; i++)
        buf[i] = (uint8_t)((i * 71u + 13u) & 0xffu);
}

static bool verify_deterministic_superframe(const uint8_t *buf, size_t sys_bytes)
{
    const uint64_t *w = (const uint64_t *)buf;
    size_t nwords = sys_bytes / 8u;
    for (size_t i = 0; i < nwords; i++) {
        uint64_t x = (uint64_t)i * 0x9e3779b97f4a7c15ULL + 0x517cc1b727220a95ULL;
        x ^= x >> 30;
        if (w[i] != x)
            return false;
    }
    for (size_t i = nwords * 8u; i < sys_bytes; i++) {
        if (buf[i] != (uint8_t)((i * 71u + 13u) & 0xffu))
            return false;
    }
    return true;
}

static bool write_all(int fd, const void *buf, size_t len)
{
    return vxfec_bench_send_all(fd, buf, len);
}

static bool read_all(int fd, void *buf, size_t len)
{
    return vxfec_bench_recv_all(fd, buf, len);
}

static void tune_udp_socket(int fd)
{
    int sz = 16 * 1024 * 1024;
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
#ifdef SO_BUSY_POLL
    int busy_us = 50;
    (void)setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &busy_us, sizeof(busy_us));
#endif
}

static uint32_t parse_ip_list(const char *csv, struct in_addr *out_ips, uint32_t max_ips)
{
    if (!csv || !*csv)
        return 0;
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", csv);
    uint32_t count = 0;
    char *save = NULL;
    for (char *tok = strtok_r(tmp, ",", &save); tok && count < max_ips; tok = strtok_r(NULL, ",", &save)) {
        if (inet_pton(AF_INET, tok, &out_ips[count]) == 1)
            count++;
    }
    return count;
}

typedef struct {
    int fd;
    struct sockaddr_in peer;
    const uint8_t *wire_src;
    uint32_t rail_idx;
    uint32_t num_rails;
    uint32_t T;
    uint16_t iter;
    uint64_t total_frames;
    uint64_t drop_beg;
    uint64_t drop_end;
    uint32_t loss_ppm;
    double rail_rate_gbps;
    _Atomic uint32_t *ready_workers;
    _Atomic bool *go_flag;
    uint64_t sent_frames;
    uint64_t my_first_ns;
    uint64_t my_last_ns;
    uint8_t pad[64];
} udp_tx_worker_t;

static void send_tail_beacons(const int *tx_fds, const struct sockaddr_in *peers, uint32_t rails, uint16_t iter)
{
    wire_pkt_hdr_t tail = {
        .magic = WIRE_NET_MAGIC,
        .iter = iter,
        .flags = WIRE_NET_FLAG_TAIL,
        .wire_seq = UINT64_MAX,
    };
    for (int rep = 0; rep < 2; rep++) {
        for (uint32_t r = 0; r < rails; r++)
            (void)sendto(tx_fds[r], &tail, sizeof(tail), 0, (const struct sockaddr *)&peers[r], sizeof(peers[r]));
    }
}

static void pin_worker_cpu(uint32_t rail_idx, bool is_rx)
{
    long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpus >= 4) {
        cpu_set_t set;
        CPU_ZERO(&set);
        uint32_t off = is_rx ? ((uint32_t)ncpus / 2u) : 0u;
        CPU_SET((int)((rail_idx + off) % (uint32_t)ncpus), &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
}

static void *udp_tx_worker_fn(void *arg)
{
    udp_tx_worker_t *w = (udp_tx_worker_t *)arg;
    pin_worker_cpu(w->rail_idx, false);
    wire_pkt_hdr_t hdrs[WIRE_NET_BATCH];
    struct iovec iovs[WIRE_NET_BATCH][2];
    struct mmsghdr msgs[WIRE_NET_BATCH];
    memset(msgs, 0, sizeof(msgs));

    uint64_t rng = 0x123456789abcdef0ULL ^ ((uint64_t)w->rail_idx << 32) ^ (uint64_t)w->iter;
    uint32_t batch_n = 0;
    uint64_t sent_total = 0;
    uint64_t first_ns = 0, last_ns = 0;
    double bits_per_pkt = 8.0 * (double)(28u + sizeof(wire_pkt_hdr_t) + w->T);

    if (w->ready_workers)
        atomic_fetch_add_explicit(w->ready_workers, 1u, memory_order_release);
    if (w->go_flag) {
        while (!atomic_load_explicit(w->go_flag, memory_order_acquire)) {
        }
    }
    uint64_t t_start = now_ns();

    for (uint64_t seq = w->rail_idx; seq < w->total_frames; seq += w->num_rails) {
        if (seq >= w->drop_beg && seq < w->drop_end)
            continue;
        if (w->loss_ppm > 0 && (splitmix64(&rng) % 1000000ULL) < w->loss_ppm)
            continue;

        hdrs[batch_n].magic = WIRE_NET_MAGIC;
        hdrs[batch_n].iter = w->iter;
        hdrs[batch_n].flags = 0u;
        hdrs[batch_n].wire_seq = seq;

        iovs[batch_n][0].iov_base = &hdrs[batch_n];
        iovs[batch_n][0].iov_len = sizeof(wire_pkt_hdr_t);
        iovs[batch_n][1].iov_base = (void *)(w->wire_src + seq * (size_t)w->T);
        iovs[batch_n][1].iov_len = w->T;

        msgs[batch_n].msg_hdr.msg_name = &w->peer;
        msgs[batch_n].msg_hdr.msg_namelen = sizeof(w->peer);
        msgs[batch_n].msg_hdr.msg_iov = iovs[batch_n];
        msgs[batch_n].msg_hdr.msg_iovlen = 2;
        batch_n++;

        if (batch_n == WIRE_NET_BATCH) {
            if (first_ns == 0)
                first_ns = now_ns();
            uint32_t off = 0;
            while (off < batch_n) {
                int rc = sendmmsg(w->fd, &msgs[off], batch_n - off, 0);
                if (rc < 0) {
                    if (errno == EINTR || errno == EAGAIN || errno == ENOBUFS)
                        continue;
                    break;
                }
                off += (uint32_t)rc;
                sent_total += (uint64_t)rc;
            }
            last_ns = now_ns();
            batch_n = 0;
            if (w->rail_rate_gbps > 0.0) {
                uint64_t target_ns = t_start + (uint64_t)(((double)sent_total * bits_per_pkt) / w->rail_rate_gbps);
                while (now_ns() < target_ns) {
                    /* spin-pace batch to avoid hypervisor token-bucket overflow */
                }
            }
        }
    }

    if (batch_n > 0) {
        if (first_ns == 0)
            first_ns = now_ns();
        uint32_t off = 0;
        while (off < batch_n) {
            int rc = sendmmsg(w->fd, &msgs[off], batch_n - off, 0);
            if (rc < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == ENOBUFS)
                    continue;
                break;
            }
            off += (uint32_t)rc;
            sent_total += (uint64_t)rc;
        }
        last_ns = now_ns();
    }

    w->sent_frames = sent_total;
    w->my_first_ns = first_ns;
    w->my_last_ns = last_ns;
    return NULL;
}

typedef struct {
    int fd;
    uint32_t rail_idx;
    uint8_t *wire_dst;
    _Atomic uint64_t *rx_bits_atomic;
    uint32_t Z;
    uint32_t K;
    uint32_t P;
    uint32_t T;
    size_t wpb;
    uint16_t iter;
    uint64_t target_frames;
    _Atomic uint32_t *ready_workers;
    _Atomic uint64_t *rx_count;
    _Atomic bool *stop_flag;
    uint64_t my_first_ns;
    uint64_t my_last_ns;
    uint8_t pad[64];
} udp_rx_worker_t;

static void *udp_rx_worker_fn(void *arg)
{
    udp_rx_worker_t *w = (udp_rx_worker_t *)arg;
    pin_worker_cpu(w->rail_idx, true);
    size_t pkt_cap = sizeof(wire_pkt_hdr_t) + (size_t)w->T;
    uint8_t *scratch = (uint8_t *)malloc((size_t)WIRE_NET_BATCH * pkt_cap);
    if (!scratch)
        return NULL;

    struct iovec iovs[WIRE_NET_BATCH];
    struct mmsghdr msgs[WIRE_NET_BATCH];
    memset(msgs, 0, sizeof(msgs));
    for (uint32_t i = 0; i < WIRE_NET_BATCH; i++) {
        iovs[i].iov_base = scratch + (size_t)i * pkt_cap;
        iovs[i].iov_len = pkt_cap;
        msgs[i].msg_hdr.msg_iov = &iovs[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
    }

    if (w->ready_workers)
        atomic_fetch_add_explicit(w->ready_workers, 1u, memory_order_release);

    uint64_t total_slots = (uint64_t)(w->K + w->P) * (uint64_t)w->Z;
    uint64_t tail_seen_ns = 0;
    uint64_t local_first = 0, local_last = 0;
    uint64_t wait_start_ns = now_ns();
    uint64_t last_prog_ns = wait_start_ns;

    while (!atomic_load_explicit(w->stop_flag, memory_order_relaxed)) {
        uint64_t t_pre = now_ns();
        int rc = recvmmsg(w->fd, msgs, WIRE_NET_BATCH, MSG_DONTWAIT, NULL);
        if (rc <= 0) {
            uint64_t cur = atomic_load_explicit(w->rx_count, memory_order_relaxed);
            if (cur >= w->target_frames) {
                atomic_store_explicit(w->stop_flag, true, memory_order_release);
                break;
            }
            if (tail_seen_ns != 0 && (t_pre - tail_seen_ns) > 50000000ULL) {
                atomic_store_explicit(w->stop_flag, true, memory_order_release);
                break;
            }
            /* bound idle wait if VM shaper/link drops both data frames and tail beacons */
            if (local_first != 0 && (t_pre - last_prog_ns) > 500000000ULL) {
                atomic_store_explicit(w->stop_flag, true, memory_order_release);
                break;
            }
            if (local_first == 0 && (t_pre - wait_start_ns) > 15000000000ULL) {
                atomic_store_explicit(w->stop_flag, true, memory_order_release);
                break;
            }
            if (local_first == 0) {
                struct timespec ts = {0, 20000}; /* 20 us idle poll before burst starts */
                nanosleep(&ts, NULL);
            } else {
                sched_yield();
            }
            continue;
        }
        last_prog_ns = t_pre;

        uint64_t batch_new = 0;

        for (int i = 0; i < rc; i++) {
            uint32_t len = msgs[i].msg_len;
            const uint8_t *pkt = scratch + (size_t)i * pkt_cap;
            if (len < sizeof(wire_pkt_hdr_t))
                continue;
            wire_pkt_hdr_t hdr;
            memcpy(&hdr, pkt, sizeof(hdr));
            if (hdr.magic != WIRE_NET_MAGIC || hdr.iter != w->iter)
                continue;
            if (hdr.flags & WIRE_NET_FLAG_TAIL) {
                if (tail_seen_ns == 0)
                    tail_seen_ns = t_pre;
                if (hdr.wire_seq == UINT64_MAX)
                    continue;
            }
            if (len < sizeof(wire_pkt_hdr_t) + w->T || hdr.wire_seq >= total_slots)
                continue;

            uint32_t z = (uint32_t)(hdr.wire_seq % (uint64_t)w->Z);
            uint32_t esi = (uint32_t)(hdr.wire_seq / (uint64_t)w->Z);
            size_t word_idx = (size_t)z * w->wpb + (esi >> 6u);
            uint64_t bit = 1ULL << (esi & 63u);

            uint64_t prev = atomic_fetch_or_explicit(&w->rx_bits_atomic[word_idx], bit, memory_order_relaxed);
            if (!(prev & bit)) {
                memcpy(w->wire_dst + hdr.wire_seq * (size_t)w->T, pkt + sizeof(wire_pkt_hdr_t), w->T);
                batch_new++;
            }
        }

        if (batch_new > 0) {
            uint64_t t_post = now_ns();
            if (local_first == 0)
                local_first = t_pre;
            local_last = (t_post > local_first) ? t_post : (local_first + 1ULL);
            uint64_t c = atomic_fetch_add_explicit(w->rx_count, batch_new, memory_order_relaxed) + batch_new;
            if (c >= w->target_frames)
                atomic_store_explicit(w->stop_flag, true, memory_order_release);
        }
    }

    w->my_first_ns = local_first;
    w->my_last_ns = local_last;
    free(scratch);
    return NULL;
}

/*
 * Self-contained loopback test (`--self-test`):
 * Encodes a Z x (K + P) x T superframe in Vulkan VRAM, streams over 4 local UDP
 * sockets with a 10% contiguous link blackout, imports/uploads into RX VRAM,
 * executes `vxfec_wire_recover_submit()`, and verifies 100% bit-exact recovery
 * for both RFC 6330 RaptorQ (`VXFEC_WIRE_RATELESS`) and Cauchy RS8 (`VXFEC_WIRE_MDS`).
 */
static bool run_loopback_self_test_codec(vxfec_vk_ctx *ctx, int codec, uint32_t Z, uint32_t K, uint32_t P, uint32_t T,
                                         bool *out_retain_vk)
{
    if (out_retain_vk)
        *out_retain_vk = false;
    const uint32_t rails = 4u;
    const size_t sys_bytes = (size_t)Z * K * T;
    const size_t wire_bytes = (size_t)Z * (K + P) * T;
    const size_t wpb = ((size_t)(K + P) + 63u) / 64u;
    const uint64_t total_frames = (uint64_t)(K + P) * Z;
    const uint64_t drop_frames = ((uint64_t)K * Z) / 10u; /* 10% contiguous link blackout */
    const uint64_t drop_beg = (uint64_t)Z * 2u + 5u;
    const uint64_t drop_end = drop_beg + drop_frames;

    bool ok = false;
    vxfec_wire_ring *tx_ring = vxfec_wire_ring_new_ex(ctx, codec, Z, K, P, T, 0u, true, false);
    vxfec_wire_ring *rx_ring = vxfec_wire_ring_new_ex(ctx, codec, Z, K, P, T, 0u, false, true);
    vxfec_vk_buf *tx_vram = vxfec_vk_buf_new(ctx, wire_bytes, false);
    vxfec_vk_buf *rx_vram = NULL;
    bool rx_ext_host = false;

    size_t page_sz = (size_t)sysconf(_SC_PAGESIZE);
    size_t aligned_wire = (wire_bytes + page_sz - 1u) & ~(page_sz - 1u);
    uint8_t *tx_host = (uint8_t *)malloc(wire_bytes);
    uint8_t *rx_host = NULL;
    uint8_t *rx_verify = (uint8_t *)malloc(sys_bytes);
    _Atomic uint64_t *rx_bits_atomic = (_Atomic uint64_t *)calloc((size_t)Z * wpb, sizeof(_Atomic uint64_t));
    uint64_t *rx_bits = (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t));

    int rx_fds[WIRE_NET_MAX_RAILS];
    int tx_fds[WIRE_NET_MAX_RAILS];
    struct sockaddr_in rx_addrs[WIRE_NET_MAX_RAILS];
    for (uint32_t i = 0; i < rails; i++) {
        rx_fds[i] = -1;
        tx_fds[i] = -1;
    }

    if (posix_memalign((void **)&rx_host, page_sz, aligned_wire) != 0)
        rx_host = NULL;
    if (!tx_ring || !rx_ring || !tx_vram || !tx_host || !rx_host || !rx_verify || !rx_bits_atomic || !rx_bits)
        goto cleanup;

    memset(rx_host, 0xa5, aligned_wire);
    if (vxfec_vk_has_ext_host_mem(ctx)) {
        rx_vram = vxfec_vk_buf_import_host_ptr(ctx, rx_host, aligned_wire);
        if (rx_vram)
            rx_ext_host = true;
    }
    if (!rx_vram) {
        rx_vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
        if (!rx_vram)
            goto cleanup;
    }

    fill_deterministic_superframe(tx_host, sys_bytes);
    if (!vxfec_vk_buf_write_offset(ctx, tx_vram, 0, tx_host, sys_bytes))
        goto cleanup;
    uint64_t enc_ticket = vxfec_wire_encode_submit(tx_ring, tx_vram);
    if (!enc_ticket || !vxfec_vk_wait(ctx, enc_ticket, 5000000000ULL))
        goto cleanup;
    if (!vxfec_vk_buf_read_offset(ctx, tx_vram, 0, tx_host, wire_bytes))
        goto cleanup;

    for (uint32_t r = 0; r < rails; r++) {
        rx_fds[r] = socket(AF_INET, SOCK_DGRAM, 0);
        tx_fds[r] = socket(AF_INET, SOCK_DGRAM, 0);
        if (rx_fds[r] < 0 || tx_fds[r] < 0)
            goto cleanup;
        tune_udp_socket(rx_fds[r]);
        tune_udp_socket(tx_fds[r]);

        struct sockaddr_in bind_addr;
        memset(&bind_addr, 0, sizeof(bind_addr));
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        bind_addr.sin_port = 0;
        if (bind(rx_fds[r], (struct sockaddr *)&bind_addr, sizeof(bind_addr)) < 0)
            goto cleanup;
        socklen_t alen = sizeof(rx_addrs[r]);
        if (getsockname(rx_fds[r], (struct sockaddr *)&rx_addrs[r], &alen) < 0)
            goto cleanup;
    }

    _Atomic uint64_t rx_count = 0;
    _Atomic bool stop_flag = false;
    pthread_t rx_th[WIRE_NET_MAX_RAILS];
    pthread_t tx_th[WIRE_NET_MAX_RAILS];
    udp_rx_worker_t rx_w[WIRE_NET_MAX_RAILS];
    udp_tx_worker_t tx_w[WIRE_NET_MAX_RAILS];

    uint32_t started_rx = 0, started_tx = 0;
    uint64_t expected_frames = total_frames - drop_frames;
    for (uint32_t r = 0; r < rails; r++) {
        rx_w[r] = (udp_rx_worker_t){
            .fd = rx_fds[r],
            .rail_idx = r,
            .wire_dst = rx_ext_host ? rx_host : (uint8_t *)vxfec_vk_buf_staging_ptr(rx_vram),
            .rx_bits_atomic = rx_bits_atomic,
            .Z = Z,
            .K = K,
            .P = P,
            .T = T,
            .wpb = wpb,
            .iter = 1u,
            .target_frames = expected_frames,
            .rx_count = &rx_count,
            .stop_flag = &stop_flag,
            .my_first_ns = 0u,
            .my_last_ns = 0u,
        };
        if (pthread_create(&rx_th[r], NULL, udp_rx_worker_fn, &rx_w[r]) != 0) {
            atomic_store_explicit(&stop_flag, true, memory_order_release);
            for (uint32_t j = 0; j < started_rx; j++)
                pthread_join(rx_th[j], NULL);
            goto cleanup;
        }
        started_rx++;
    }

    uint64_t tx_start_ns = now_ns();
    for (uint32_t r = 0; r < rails; r++) {
        tx_w[r] = (udp_tx_worker_t){
            .fd = tx_fds[r],
            .peer = rx_addrs[r],
            .wire_src = tx_host,
            .rail_idx = r,
            .num_rails = rails,
            .T = T,
            .iter = 1u,
            .total_frames = total_frames,
            .drop_beg = drop_beg,
            .drop_end = drop_end,
            .loss_ppm = 0u,
            .sent_frames = 0u,
        };
        if (pthread_create(&tx_th[r], NULL, udp_tx_worker_fn, &tx_w[r]) != 0) {
            for (uint32_t j = 0; j < started_tx; j++)
                pthread_join(tx_th[j], NULL);
            atomic_store_explicit(&stop_flag, true, memory_order_release);
            for (uint32_t j = 0; j < started_rx; j++)
                pthread_join(rx_th[j], NULL);
            goto cleanup;
        }
        started_tx++;
    }

    for (uint32_t r = 0; r < started_tx; r++)
        pthread_join(tx_th[r], NULL);
    send_tail_beacons(tx_fds, rx_addrs, rails, 1u);
    for (uint32_t r = 0; r < started_rx; r++)
        pthread_join(rx_th[r], NULL);

    for (size_t i = 0; i < (size_t)Z * wpb; i++)
        rx_bits[i] = atomic_load_explicit(&rx_bits_atomic[i], memory_order_relaxed);

    if (rx_ext_host) {
        if (!vxfec_vk_buf_acquire_external(ctx, rx_vram))
            goto cleanup;
    } else {
        if (!vxfec_vk_buf_upload(ctx, rx_vram, wire_bytes))
            goto cleanup;
    }

    vxfec_wire_stats st = {0};
    uint64_t rec_t0 = now_ns();
    uint64_t rec_ticket = vxfec_wire_recover_submit(rx_ring, rx_vram, rx_bits, &st);
    if (!rec_ticket || !vxfec_vk_wait(ctx, rec_ticket, 5000000000ULL)) {
        fprintf(stderr, "FAIL: %s loopback recovery %s (received %" PRIu64 "/%" PRIu64 " planned frames, %u failed blocks)\n",
                codec == VXFEC_WIRE_MDS ? "RS8" : "RaptorQ", rec_ticket ? "wait failed" : "submit failed", atomic_load(&rx_count),
                expected_frames, st.blocks_failed);
        goto cleanup;
    }
    uint64_t rec_wall_ns = now_ns() - rec_t0;

    if (rx_ext_host) {
        if (!vxfec_vk_buf_release_external(ctx, rx_vram))
            goto cleanup;
        memcpy(rx_verify, rx_host, sys_bytes);
    } else {
        if (!vxfec_vk_buf_read_offset(ctx, rx_vram, 0, rx_verify, sys_bytes))
            goto cleanup;
    }

    if (!verify_deterministic_superframe(rx_verify, sys_bytes)) {
        fprintf(stderr, "FAIL: bench_wire_net --self-test bit-exact check failed (codec=%d)\n", codec);
        goto cleanup;
    }

    uint64_t f_ns = tx_start_ns, l_ns = 0;
    for (uint32_t r = 0; r < rails; r++) {
        if (rx_w[r].my_first_ns > 0 && rx_w[r].my_first_ns < f_ns)
            f_ns = rx_w[r].my_first_ns;
        if (rx_w[r].my_last_ns > l_ns)
            l_ns = rx_w[r].my_last_ns;
    }
    uint64_t wire_ns = (l_ns > f_ns) ? (l_ns - f_ns) : 1ULL;
    double wire_gbps = (8.0 * (double)atomic_load(&rx_count) * (double)T) / (double)wire_ns;
    double rec_gbps = (8.0 * (double)sys_bytes) / (double)(rec_wall_ns ? rec_wall_ns : 1ULL);

    printf("  [PASS] %-12s | Z=%2u K=%3u P=%2u T=%4u B | RX=%" PRIu64 "/%" PRIu64
           " | Wire=%6.2f Gbps | GPU Rec=%7.2f Gbps (%u masks, ext_host=%d)\n",
           (codec == VXFEC_WIRE_MDS) ? "MDS (RS8)" : "Rateless(RQ)", Z, K, P, T, (uint64_t)atomic_load(&rx_count), total_frames,
           wire_gbps, rec_gbps, st.unique_masks, (int)rx_ext_host);
    ok = true;

cleanup:
    for (uint32_t r = 0; r < rails; r++) {
        if (rx_fds[r] >= 0)
            close(rx_fds[r]);
        if (tx_fds[r] >= 0)
            close(tx_fds[r]);
    }
    bool vk_quiesced = vxfec_vk_wait_idle(ctx);
    if (!vk_quiesced) {
        ok = false;
        if (out_retain_vk)
            *out_retain_vk = true;
    }
    if (vk_quiesced) {
        vxfec_wire_ring_free(rx_ring);
        vxfec_wire_ring_free(tx_ring);
        vxfec_vk_buf_free(ctx, rx_vram);
        vxfec_vk_buf_free(ctx, tx_vram);
        free(rx_host);
    }
    free(rx_bits);
    free(rx_bits_atomic);
    free(rx_verify);
    free(tx_host);
    return ok;
}

typedef struct {
    const char *mode;
    const char *bind_csv;
    const char *peers_csv;
    const char *csv_path;
    const char *bidir_sync;
    uint16_t base_port;
    uint32_t ports_per_ip;
    int codec;
    uint32_t Z;
    uint32_t K;
    uint32_t P;
    uint32_t T;
    uint32_t iters;
    double blackout_pct;
    double rate_gbps;
    uint32_t loss_ppm;
    bool use_ibv;
    bool gdr;
    bool pipeline;
    bool wire_only;
    bool bidir_leader;
    bool self_test;
} wire_cli_cfg_t;

static int open_bidir_sync_socket(const char *spec, bool leader)
{
    char host[128];
    uint16_t port = 31490u;
    vxfec_bench_parse_host_port(spec, host, sizeof(host), 31490u, &port);
    if (leader) {
        struct in_addr any_ip = {.s_addr = htonl(INADDR_ANY)};
        return vxfec_bench_tcp_listen_accept_one(any_ip, port);
    }
    struct in_addr peer_ip;
    if (inet_pton(AF_INET, host, &peer_ip) != 1)
        return -1;
    return vxfec_bench_tcp_connect_retry(peer_ip, port, 350, 100000000L);
}

static bool receive_ibv_frame(vxfec_ibv_ep_t *ibv_ep, int ctrl_fd, uint32_t Z, size_t wpb, uint64_t total_frames,
                              uint64_t expected_frames, bool require_tails, uint64_t *rx_bits, wire_ctrl_result_t *res)
{
    uint32_t it = res->iter;
    uint64_t rx_frames = 0;
    uint64_t f_ns = 0, l_ns = 0;
    uint64_t tail_ns = 0;
    uint32_t tail_qps = 0u;
    const uint32_t all_qps = (1u << WIRE_NET_IBV_QPS) - 1u;
    uint64_t wait_start_ns = now_ns();
    uint64_t last_prog_ns = wait_start_ns;
    uint32_t empty_polls = 0;
    vxfec_ibv_rx_wc_t wcs[64];
    /* With direct VRAM writes, wait for a completion on every QP before Vulkan can acquire the buffer. */
    while (require_tails ? tail_qps != all_qps : rx_frames < expected_frames) {
        uint64_t t_pre = now_ns();
        uint32_t n = vxfec_ibv_ep_poll_rx(ibv_ep, wcs, 64u);
        if (vxfec_ibv_ep_has_error(ibv_ep)) {
            fprintf(stderr, "ERROR: fatal IBV RX completion error at iter %u\n", it);
            return false;
        }
        if (n == 0) {
            if (!require_tails && tail_ns != 0 && (t_pre - tail_ns) > 50000000ULL)
                break;
            /* bound idle wait if unplanned UC drops also lost all 4 tail beacons */
            if ((f_ns != 0 || tail_ns != 0) && (t_pre - last_prog_ns) > 2000000000ULL)
                break;
            if (f_ns == 0 && (t_pre - wait_start_ns) > 15000000000ULL) {
                fprintf(stderr, "ERROR: timed out waiting for IBV RX burst at iter %u\n", it);
                return false;
            }
            if ((++empty_polls & 1023u) == 0u) {
                uint8_t probe = 0;
                ssize_t pr = recv(ctrl_fd, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
                if (pr == 0 || (pr < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
                    fprintf(stderr, "ERROR: peer disconnected ctrl channel during IBV RX at iter %u\n", it);
                    return false;
                }
            }
            continue;
        }
        empty_polls = 0;
        last_prog_ns = t_pre;
        uint32_t batch_data = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (wcs[i].has_imm) {
                uint32_t seq = wcs[i].imm_data;
                if (seq == UINT32_MAX) {
                    if (tail_ns == 0)
                        tail_ns = t_pre;
                    tail_qps |= 1u << vxfec_ibv_slot_qp(wcs[i].slot_id);
                } else if ((uint64_t)seq < total_frames) {
                    vxfec_wire_mark_seq(rx_bits, Z, wpb, (uint64_t)seq);
                    rx_frames++;
                    batch_data++;
                }
            }
            if (!vxfec_ibv_ep_repost_rx(ibv_ep, wcs[i].slot_id)) {
                fprintf(stderr, "ERROR: failed to repost IBV RX slot at iter %u\n", it);
                return false;
            }
        }
        if (batch_data > 0u) {
            uint64_t t_post = now_ns();
            if (f_ns == 0)
                f_ns = t_pre;
            l_ns = (t_post > f_ns) ? t_post : (f_ns + 1ULL);
        }
    }
    if (require_tails && tail_qps != all_qps) {
        fprintf(stderr, "ERROR: RX --gdr did not receive a tail completion from every QP at iter %u\n", it);
        return false;
    }
    res->rx_frames = rx_frames;
    res->wire_ns = (l_ns > f_ns) ? (l_ns - f_ns) : 1ULL;
    return true;
}

typedef struct {
    vxfec_vk_ctx *ctx;
    vxfec_wire_ring *ring;
    vxfec_vk_buf *vram;
    uint8_t *host;
    uint8_t *verify;
    const uint64_t *bits;
    uint32_t Z;
    uint32_t K;
    size_t wpb;
    size_t sys_bytes;
    size_t wire_bytes;
    bool gdr;
    bool ext_host;
    bool wire_only;
} wire_rx_recovery_t;

static void recover_rx_frame(const wire_rx_recovery_t *job, wire_ctrl_result_t *res)
{
    vxfec_vk_ctx *ctx = job->ctx;
    vxfec_vk_buf *rx_vram = job->vram;
    const uint64_t *rx_bits = job->bits;
    bool gdr_dmabuf = job->gdr;
    bool rx_ext_host = job->ext_host;
    uint32_t Z = job->Z, K = job->K;
    size_t wpb = job->wpb, sys_bytes = job->sys_bytes;
    uint64_t dma_ns = 0;
    uint64_t rec_wall_ns = 0;
    vxfec_wire_stats st = {0};
    bool verify_ok = false;
    bool fatal_err = false;

    bool enough_frames = true;
    for (uint32_t z = 0; z < Z; z++) {
        uint32_t cnt = 0;
        for (size_t w = 0; w < wpb; w++)
            cnt += (uint32_t)__builtin_popcountll(rx_bits[(size_t)z * wpb + w]);
        if (cnt < K) {
            enough_frames = false;
            break;
        }
    }

    if (job->wire_only) {
        /* verify each block received >= K frames for decodability without CPU llvmpipe simulation */
        verify_ok = enough_frames;
    } else {
        uint64_t dma_t0 = now_ns();
        bool prep_ok = (gdr_dmabuf || rx_ext_host) ? vxfec_vk_buf_acquire_external(ctx, rx_vram)
                                                   : vxfec_vk_buf_upload(ctx, rx_vram, job->wire_bytes);
        dma_ns = now_ns() - dma_t0;

        uint64_t rec_t0 = now_ns();
        bool rec_ok = prep_ok && vxfec_wire_recover_vram(job->ring, rx_vram, rx_bits, &st);
        rec_wall_ns = now_ns() - rec_t0;

        if (rec_ok) {
            if (rx_ext_host && !gdr_dmabuf) {
                (void)vxfec_vk_buf_release_external(ctx, rx_vram);
                verify_ok = verify_deterministic_superframe(job->host, sys_bytes);
            } else {
                if (vxfec_vk_buf_read_offset(ctx, rx_vram, 0, job->verify, sys_bytes))
                    verify_ok = verify_deterministic_superframe(job->verify, sys_bytes);
            }
            if (!verify_ok) {
                fprintf(stderr, "ERROR: RX payload verification mismatch at iter %u!\n", res->iter);
                fatal_err = true;
            }
        } else {
            if (rx_ext_host)
                (void)vxfec_vk_buf_release_external(ctx, rx_vram);
            if (!prep_ok) {
                fprintf(stderr, "ERROR: failed to acquire RX GPU buffer at iter %u\n", res->iter);
                fatal_err = true;
            } else if (enough_frames) {
                fprintf(stderr, "ERROR: RX FEC decode failed despite >= K frames/block at iter %u!\n", res->iter);
                fatal_err = true;
            }
        }
        if (gdr_dmabuf && !vxfec_vk_buf_release_external(ctx, rx_vram)) {
            fprintf(stderr, "ERROR: failed to release RX GPU buffer at iter %u\n", res->iter);
            fatal_err = true;
            verify_ok = false;
        }
    }

    res->ok = verify_ok;
    res->fatal_err = fatal_err;
    res->dma_ns = dma_ns;
    res->recover_wall_ns = rec_wall_ns;
    res->recover_gpu_ns = st.gpu_exec_ns;
    res->blocks_repaired = st.blocks_repaired;
    res->unique_masks = st.unique_masks;
}

static bool prepare_gdr_buffer(vxfec_vk_ctx *ctx, vxfec_vk_buf *vram, uint8_t *scratch, size_t sys_bytes)
{
    memset(scratch, 0xa5, sys_bytes);
    return vxfec_vk_buf_write_offset(ctx, vram, 0, scratch, sys_bytes) && vxfec_vk_buf_release_external(ctx, vram);
}

typedef struct {
    uint64_t *latencies_ns;
    uint64_t start_ns;
    uint64_t end_ns;
    uint32_t passed;
} wire_delivery_stats_t;

static void record_delivery(wire_delivery_stats_t *stats, const wire_ctrl_result_t *res, uint64_t ready_ns, uint64_t done_ns)
{
    if (stats->start_ns == 0u)
        stats->start_ns = ready_ns;
    stats->end_ns = done_ns;
    if (res->ok)
        stats->latencies_ns[stats->passed++] = done_ns - ready_ns;
}

static int compare_duration(const void *a, const void *b)
{
    uint64_t left = *(const uint64_t *)a;
    uint64_t right = *(const uint64_t *)b;
    return (left > right) - (left < right);
}

static void report_delivery(wire_delivery_stats_t *stats, uint32_t iters, size_t sys_bytes)
{
    printf("  RX verified %u/%u superframes", stats->passed, iters);
    if (stats->passed > 0u) {
        uint64_t span_ns = stats->end_ns - stats->start_ns;
        qsort(stats->latencies_ns, stats->passed, sizeof(*stats->latencies_ns), compare_duration);
        uint32_t p99 = (99u * stats->passed + 99u) / 100u - 1u;
        double goodput = (8.0 * (double)sys_bytes * stats->passed) / (double)(span_ns ? span_ns : 1u);
        printf(" | Verified goodput=%.2f Gbps | RX ready-to-verified p99=%.3f ms", goodput, (double)stats->latencies_ns[p99] / 1e6);
    }
    putchar('\n');
}

typedef struct {
    sem_t work;
    sem_t done;
    bool stop;
    bool reuse;
    wire_rx_recovery_t job;
    wire_ctrl_result_t result;
    uint64_t completed_ns;
} wire_recovery_worker_t;

static void wait_for_work_signal(sem_t *signal)
{
    while (sem_wait(signal) != 0 && errno == EINTR) {
    }
}

static void *recover_wire_frames(void *arg)
{
    wire_recovery_worker_t *worker = arg;
    for (;;) {
        wait_for_work_signal(&worker->work);
        if (worker->stop)
            return NULL;
        recover_rx_frame(&worker->job, &worker->result);
        worker->completed_ns = now_ns();
        if (worker->reuse && !worker->result.fatal_err &&
            !prepare_gdr_buffer(worker->job.ctx, worker->job.vram, worker->job.verify, worker->job.sys_bytes)) {
            fprintf(stderr, "ERROR: failed to prepare pipelined RX buffer after iter %u\n", worker->result.iter);
            worker->result.fatal_err = true;
        }
        sem_post(&worker->done);
    }
}

static bool receive_pipelined_frames(int ctrl_fd, vxfec_ibv_ep_t *ep, const wire_ctrl_handshake_t *hs, wire_rx_recovery_t job,
                                     vxfec_vk_buf *second_vram, uint64_t *masks, wire_delivery_stats_t *stats)
{
    wire_recovery_worker_t worker = {0};
    if (sem_init(&worker.work, 0, 0) != 0) {
        perror("RX pipeline work semaphore");
        return false;
    }
    if (sem_init(&worker.done, 0, 0) != 0) {
        perror("RX pipeline completion semaphore");
        sem_destroy(&worker.work);
        return false;
    }
    pthread_t thread;
    int err = pthread_create(&thread, NULL, recover_wire_frames, &worker);
    if (err != 0) {
        fprintf(stderr, "ERROR: could not start RX recovery worker: %s\n", strerror(err));
        sem_destroy(&worker.done);
        sem_destroy(&worker.work);
        return false;
    }

    bool ok = true;
    uint64_t pending_ready_ns = 0u;
    size_t mask_words = (size_t)job.Z * job.wpb;
    uint64_t total_frames = (uint64_t)(hs->K + hs->P) * hs->Z;
    for (uint32_t it = 1; it <= hs->iters; it++) {
        uint64_t expected_frames;
        uint64_t *bits = masks + ((it - 1u) % 2u) * mask_words;
        memset(bits, 0, mask_words * sizeof(*bits));
        wire_ctrl_result_t received = {.iter = it};
        if (!read_all(ctrl_fd, &expected_frames, sizeof(expected_frames))) {
            ok = false;
            break;
        }
        uint64_t ready_ns = now_ns();
        /* The ACK grants access to this slot. Its previous recovery and reinitialization already finished. */
        if (!write_all(ctrl_fd, &it, sizeof(it)) ||
            !receive_ibv_frame(ep, ctrl_fd, job.Z, job.wpb, total_frames, expected_frames, true, bits, &received)) {
            ok = false;
            break;
        }
        if (pending_ready_ns != 0u) {
            /* The sender reads the previous result after posting this frame's writes. */
            wait_for_work_signal(&worker.done);
            record_delivery(stats, &worker.result, pending_ready_ns, worker.completed_ns);
            pending_ready_ns = 0u;
            if (!write_all(ctrl_fd, &worker.result, sizeof(worker.result)) || worker.result.fatal_err) {
                ok = false;
                break;
            }
        }
        worker.job = job;
        worker.job.vram = (it % 2u == 0u) ? second_vram : job.vram;
        worker.job.bits = bits;
        worker.result = received;
        worker.reuse = it + 2u <= hs->iters;
        pending_ready_ns = ready_ns;
        sem_post(&worker.work);
    }
    if (pending_ready_ns != 0u) {
        wait_for_work_signal(&worker.done);
        record_delivery(stats, &worker.result, pending_ready_ns, worker.completed_ns);
        if (ok)
            ok = write_all(ctrl_fd, &worker.result, sizeof(worker.result)) && !worker.result.fatal_err;
    }

    worker.stop = true;
    sem_post(&worker.work);
    pthread_join(thread, NULL);
    sem_destroy(&worker.done);
    sem_destroy(&worker.work);
    return ok;
}

static uint32_t wire_handshake_magic(const wire_cli_cfg_t *cfg)
{
    if (cfg->pipeline)
        return WIRE_NET_PIPELINE_MAGIC;
    return cfg->gdr ? WIRE_NET_GDR_MAGIC : WIRE_NET_MAGIC;
}

static int run_rx_server(vxfec_vk_ctx *ctx, const wire_cli_cfg_t *cfg, bool *out_retain_vk)
{
    if (out_retain_vk)
        *out_retain_vk = false;
    struct in_addr ips[WIRE_NET_MAX_RAILS];
    uint32_t n_ips = parse_ip_list(cfg->bind_csv, ips, WIRE_NET_MAX_RAILS);
    if (n_ips == 0) {
        ips[0].s_addr = htonl(INADDR_ANY);
        n_ips = 1u;
    }

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return 1;
    }
    int one = 1;
    (void)setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in ctrl_addr;
    memset(&ctrl_addr, 0, sizeof(ctrl_addr));
    ctrl_addr.sin_family = AF_INET;
    ctrl_addr.sin_addr = ips[0];
    ctrl_addr.sin_port = htons((uint16_t)(cfg->base_port - 1u));
    if (bind(listen_fd, (struct sockaddr *)&ctrl_addr, sizeof(ctrl_addr)) < 0 || listen(listen_fd, 4) < 0) {
        perror("bind/listen ctrl");
        close(listen_fd);
        return 1;
    }

    printf("[bench_wire_net RX] Listening for TX peer on port %u (GPU: %s)...\n", (unsigned)(cfg->base_port - 1u),
           vxfec_vk_device_name(ctx));
    fflush(stdout);

    int ctrl_fd = accept(listen_fd, NULL, NULL);
    if (ctrl_fd < 0) {
        perror("accept");
        close(listen_fd);
        return 1;
    }
    (void)setsockopt(ctrl_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    wire_ctrl_handshake_t tx_hs;
    uint32_t expected_magic = wire_handshake_magic(cfg);
    if (!read_all(ctrl_fd, &tx_hs, sizeof(tx_hs)) || tx_hs.magic != expected_magic ||
        (cfg->gdr && (!tx_hs.use_ibv || tx_hs.wire_only))) {
        fprintf(stderr, "Invalid handshake from TX peer (TX and RX need matching --gdr and --pipeline options)\n");
        close(ctrl_fd);
        close(listen_fd);
        return 1;
    }

    size_t page_sz = (size_t)sysconf(_SC_PAGESIZE);
    uint32_t Z = tx_hs.Z, K = tx_hs.K, P = tx_hs.P, T = tx_hs.T;
    if ((tx_hs.codec != (uint32_t)VXFEC_WIRE_RATELESS && tx_hs.codec != (uint32_t)VXFEC_WIRE_MDS &&
         tx_hs.codec != (uint32_t)VXFEC_WIRE_RATELESS_GF2) ||
        Z == 0u || Z > 4096u || K == 0u || K > 56403u || P == 0u || P > 65535u || T < 64u || T > (1u << 20) || (T & 15u) != 0u ||
        tx_hs.iters == 0u || tx_hs.iters > 100000u || (uint64_t)(K + P) > (SIZE_MAX - page_sz) / ((size_t)Z * (size_t)T)) {
        fprintf(stderr, "Invalid superframe dimensions in TX handshake (Z=%u K=%u P=%u T=%u iters=%u)\n", Z, K, P, T, tx_hs.iters);
        close(ctrl_fd);
        close(listen_fd);
        return 1;
    }
    size_t sys_bytes = (size_t)Z * K * T;
    size_t wire_bytes = (size_t)Z * (K + P) * T;
    size_t wpb = ((size_t)(K + P) + 63u) / 64u;
    size_t aligned_wire = (wire_bytes + page_sz - 1u) & ~(page_sz - 1u);

    vxfec_wire_ring *rx_ring = tx_hs.wire_only ? NULL : vxfec_wire_ring_new_ex(ctx, (int)tx_hs.codec, Z, K, P, T, 0u, false, true);
    uint8_t *rx_host = NULL;
    if (!cfg->gdr && posix_memalign((void **)&rx_host, page_sz, aligned_wire) != 0)
        rx_host = NULL;
    uint8_t *rx_verify = (uint8_t *)malloc(sys_bytes);
    _Atomic uint64_t *rx_bits_atomic = (_Atomic uint64_t *)calloc((size_t)Z * wpb, sizeof(_Atomic uint64_t));
    uint64_t *rx_bits = (uint64_t *)calloc((size_t)Z * wpb * (cfg->pipeline ? 2u : 1u), sizeof(uint64_t));
    wire_delivery_stats_t stats = {0};
    if (cfg->gdr)
        stats.latencies_ns = (uint64_t *)calloc(tx_hs.iters, sizeof(*stats.latencies_ns));

    vxfec_vk_buf *rx_vram = NULL;
    vxfec_vk_buf *second_vram = NULL;
    bool rx_ext_host = false;
    bool gdr_dmabuf = false;
    bool rx_aborted = true;
    int dmabuf_fd = -1;
    int second_fd = -1;
    int rx_fds[WIRE_NET_MAX_RAILS];
    uint32_t rails = 0;

    vxfec_ibv_ctx_t *ibv_ctx = NULL;
    vxfec_ibv_mr_t *ibv_mr = NULL;
    vxfec_ibv_mr_t *second_mr = NULL;
    vxfec_ibv_ep_t *ibv_ep = NULL;
    wire_ctrl_handshake_t rx_hs = tx_hs;

    if ((!cfg->gdr && !rx_host) || !rx_verify || !rx_bits_atomic || !rx_bits || (!tx_hs.wire_only && !rx_ring) ||
        (cfg->gdr && !stats.latencies_ns)) {
        fprintf(stderr, "Failed to allocate RX superframe buffers/ring\n");
        goto cleanup;
    }

    if (tx_hs.use_ibv) {
        ibv_ctx = vxfec_ibv_ctx_open(0);
        if (!ibv_ctx) {
            fprintf(stderr, "RX requested --ibv but no active RDMA device found\n");
            goto cleanup;
        }
        if (vxfec_vk_has_dmabuf_export(ctx)) {
            rx_vram = vxfec_vk_buf_new_exportable(ctx, aligned_wire, true, &dmabuf_fd);
            if (rx_vram && dmabuf_fd >= 0) {
                /* A DMA-BUF MR uses a page-aligned IOVA, independent of any host mapping. */
                uint64_t iova = cfg->gdr ? page_sz : (uint64_t)(uintptr_t)rx_host;
                ibv_mr = vxfec_ibv_mr_reg_dmabuf(ibv_ctx, 0, aligned_wire, iova, dmabuf_fd, true);
                if (ibv_mr)
                    gdr_dmabuf = true;
                else {
                    close(dmabuf_fd);
                    dmabuf_fd = -1;
                    vxfec_vk_buf_free(ctx, rx_vram);
                    rx_vram = NULL;
                }
            } else if (rx_vram) {
                vxfec_vk_buf_free(ctx, rx_vram);
                rx_vram = NULL;
            }
        }
        if (cfg->gdr && !gdr_dmabuf) {
            fprintf(stderr, "RX --gdr failed to export or register a GPU DMA-BUF\n");
            goto cleanup;
        }
        if (!ibv_mr) {
            if (vxfec_vk_has_ext_host_mem(ctx)) {
                rx_vram = vxfec_vk_buf_import_host_ptr(ctx, rx_host, aligned_wire);
                if (rx_vram)
                    rx_ext_host = true;
            }
            if (!rx_vram)
                rx_vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
            void *mr_ptr = rx_ext_host ? (void *)rx_host : vxfec_vk_buf_staging_ptr(rx_vram);
            if (mr_ptr)
                ibv_mr = vxfec_ibv_mr_reg_host(ibv_ctx, mr_ptr, wire_bytes, true);
        }
        ibv_ep = vxfec_ibv_ep_create_ex(ibv_ctx, WIRE_NET_IBV_QPS, 256u, false, true, &rx_hs.ibv_info);
        if (!ibv_mr || !ibv_ep || !vxfec_ibv_ep_connect(ibv_ep, &tx_hs.ibv_info, 0)) {
            fprintf(stderr, "Failed to setup RX ibverbs endpoint/MR\n");
            goto cleanup;
        }
        rx_hs.remote_iova = vxfec_ibv_mr_iova(ibv_mr);
        rx_hs.remote_rkey = vxfec_ibv_mr_rkey(ibv_mr);
        rx_hs.gdr_dmabuf = gdr_dmabuf ? 1u : 0u;
    } else if (tx_hs.wire_only) {
        rx_ext_host = true;
    } else {
        if (vxfec_vk_has_ext_host_mem(ctx)) {
            rx_vram = vxfec_vk_buf_import_host_ptr(ctx, rx_host, aligned_wire);
            if (rx_vram)
                rx_ext_host = true;
        }
        if (!rx_vram)
            rx_vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
        if (!rx_vram) {
            fprintf(stderr, "Failed to allocate RX Vulkan buffer\n");
            goto cleanup;
        }
    }

    if (!tx_hs.use_ibv) {
        for (uint32_t ip_i = 0; ip_i < n_ips && rails < WIRE_NET_MAX_RAILS; ip_i++) {
            for (uint32_t q = 0; q < cfg->ports_per_ip && rails < WIRE_NET_MAX_RAILS; q++) {
                int fd = socket(AF_INET, SOCK_DGRAM, 0);
                if (fd < 0) {
                    perror("socket udp rx");
                    goto cleanup;
                }
                tune_udp_socket(fd);
                struct sockaddr_in baddr;
                memset(&baddr, 0, sizeof(baddr));
                baddr.sin_family = AF_INET;
                baddr.sin_addr = ips[ip_i];
                baddr.sin_port = htons((uint16_t)(cfg->base_port + rails));
                if (bind(fd, (struct sockaddr *)&baddr, sizeof(baddr)) < 0) {
                    perror("bind udp rx");
                    close(fd);
                    goto cleanup;
                }
                rx_fds[rails++] = fd;
            }
        }
        rx_hs.rails = rails;
    }

    wire_rdma_region_t second_region = {0};
    if (cfg->pipeline) {
        second_vram = vxfec_vk_buf_new_exportable(ctx, aligned_wire, true, &second_fd);
        if (second_vram)
            second_mr = vxfec_ibv_mr_reg_dmabuf(ibv_ctx, 0, aligned_wire, (uint64_t)page_sz + aligned_wire, second_fd, true);
        if (!second_mr || !prepare_gdr_buffer(ctx, rx_vram, rx_verify, sys_bytes) ||
            !prepare_gdr_buffer(ctx, second_vram, rx_verify, sys_bytes)) {
            fprintf(stderr, "ERROR: could not register and prepare both pipelined RX GPU buffers\n");
            goto cleanup;
        }
        second_region.iova = vxfec_ibv_mr_iova(second_mr);
        second_region.rkey = vxfec_ibv_mr_rkey(second_mr);
    }

    if (!write_all(ctrl_fd, &rx_hs, sizeof(rx_hs)) || (cfg->pipeline && !write_all(ctrl_fd, &second_region, sizeof(second_region))))
        goto cleanup;

    wire_rx_recovery_t job = {
        .ctx = ctx,
        .ring = rx_ring,
        .vram = rx_vram,
        .host = rx_host,
        .verify = rx_verify,
        .bits = rx_bits,
        .Z = Z,
        .K = K,
        .wpb = wpb,
        .sys_bytes = sys_bytes,
        .wire_bytes = wire_bytes,
        .gdr = gdr_dmabuf,
        .ext_host = rx_ext_host,
        .wire_only = tx_hs.wire_only,
    };
    if (cfg->pipeline) {
        rx_aborted = !receive_pipelined_frames(ctrl_fd, ibv_ep, &tx_hs, job, second_vram, rx_bits, &stats);
        goto cleanup;
    }

    rx_aborted = false;
    uint64_t total_frames = (uint64_t)(K + P) * Z;
    for (uint32_t it = 1; it <= tx_hs.iters; it++) {
        uint64_t expected_frames = 0;
        if (!read_all(ctrl_fd, &expected_frames, sizeof(expected_frames))) {
            rx_aborted = true;
            break;
        }

        for (size_t i = 0; i < (size_t)Z * wpb; i++)
            atomic_store_explicit(&rx_bits_atomic[i], 0ULL, memory_order_relaxed);
        memset(rx_bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
        if (!tx_hs.wire_only) {
            /* poison systematic buffer before each iteration so stale bytes cannot pass verification */
            if (gdr_dmabuf) {
                if (!prepare_gdr_buffer(ctx, rx_vram, rx_verify, sys_bytes)) {
                    fprintf(stderr, "ERROR: failed to prepare RX GPU buffer at iter %u\n", it);
                    rx_aborted = true;
                    break;
                }
            } else if (rx_ext_host) {
                memset(rx_host, 0xa5, sys_bytes);
            } else {
                void *sptr = vxfec_vk_buf_staging_ptr(rx_vram);
                if (sptr)
                    memset(sptr, 0xa5, sys_bytes);
            }
        }

        uint64_t rx_frames = 0;
        uint64_t wire_ns = 1ULL;
        uint64_t ready_ns = 0u;

        if (tx_hs.use_ibv) {
            uint32_t ready_ack = it;
            if (cfg->gdr)
                ready_ns = now_ns();
            if (!write_all(ctrl_fd, &ready_ack, sizeof(ready_ack))) {
                rx_aborted = true;
                break;
            }
            wire_ctrl_result_t received = {.iter = it};
            if (!receive_ibv_frame(ibv_ep, ctrl_fd, Z, wpb, total_frames, expected_frames, cfg->gdr, rx_bits, &received)) {
                rx_aborted = true;
                break;
            }
            rx_frames = received.rx_frames;
            wire_ns = received.wire_ns;
        } else {
            _Atomic uint32_t ready_workers = 0;
            _Atomic uint64_t rx_cnt = 0;
            _Atomic bool stop_flag = false;
            pthread_t rx_th[WIRE_NET_MAX_RAILS];
            udp_rx_worker_t rx_w[WIRE_NET_MAX_RAILS];
            uint8_t *dst_ptr = rx_ext_host ? rx_host : (uint8_t *)vxfec_vk_buf_staging_ptr(rx_vram);

            for (uint32_t r = 0; r < rails; r++) {
                uint8_t drain;
                while (recv(rx_fds[r], &drain, 1, MSG_DONTWAIT) >= 0) {
                }
            }

            uint32_t started_rx = 0;
            for (uint32_t r = 0; r < rails; r++) {
                rx_w[r] = (udp_rx_worker_t){
                    .fd = rx_fds[r],
                    .rail_idx = r,
                    .wire_dst = dst_ptr,
                    .rx_bits_atomic = rx_bits_atomic,
                    .Z = Z,
                    .K = K,
                    .P = P,
                    .T = T,
                    .wpb = wpb,
                    .iter = (uint16_t)it,
                    .target_frames = expected_frames,
                    .ready_workers = &ready_workers,
                    .rx_count = &rx_cnt,
                    .stop_flag = &stop_flag,
                    .my_first_ns = 0u,
                    .my_last_ns = 0u,
                };
                if (pthread_create(&rx_th[r], NULL, udp_rx_worker_fn, &rx_w[r]) != 0) {
                    atomic_store_explicit(&stop_flag, true, memory_order_release);
                    for (uint32_t j = 0; j < started_rx; j++)
                        pthread_join(rx_th[j], NULL);
                    rx_aborted = true;
                    break;
                }
                started_rx++;
            }
            if (rx_aborted)
                break;
            while (atomic_load_explicit(&ready_workers, memory_order_acquire) < started_rx)
                sched_yield();

            uint32_t ready_ack = it;
            if (!write_all(ctrl_fd, &ready_ack, sizeof(ready_ack))) {
                atomic_store_explicit(&stop_flag, true, memory_order_release);
                for (uint32_t r = 0; r < started_rx; r++)
                    pthread_join(rx_th[r], NULL);
                rx_aborted = true;
                break;
            }
            for (uint32_t r = 0; r < started_rx; r++)
                pthread_join(rx_th[r], NULL);

            for (size_t i = 0; i < (size_t)Z * wpb; i++)
                rx_bits[i] = atomic_load_explicit(&rx_bits_atomic[i], memory_order_relaxed);
            rx_frames = atomic_load(&rx_cnt);
            uint64_t fn = UINT64_MAX, ln = 0;
            for (uint32_t r = 0; r < rails; r++) {
                if (rx_w[r].my_first_ns > 0 && rx_w[r].my_first_ns < fn)
                    fn = rx_w[r].my_first_ns;
                if (rx_w[r].my_last_ns > ln)
                    ln = rx_w[r].my_last_ns;
            }
            wire_ns = (fn != UINT64_MAX && ln > fn) ? (ln - fn) : 1ULL;
        }

        wire_ctrl_result_t res = {.iter = it, .rx_frames = rx_frames, .wire_ns = wire_ns};
        recover_rx_frame(&job, &res);
        if (cfg->gdr)
            record_delivery(&stats, &res, ready_ns, now_ns());
        if (!write_all(ctrl_fd, &res, sizeof(res)) || res.fatal_err) {
            rx_aborted = true;
            break;
        }
    }

cleanup:
    if (stats.start_ns != 0u)
        report_delivery(&stats, tx_hs.iters, sys_bytes);
    for (uint32_t r = 0; r < rails; r++)
        close(rx_fds[r]);
    bool ibv_quiesced = !ibv_ep || vxfec_ibv_ep_destroy(ibv_ep);
    bool vk_quiesced = vxfec_vk_wait_idle(ctx);
    if (!ibv_quiesced || !vk_quiesced) {
        rx_aborted = true;
        if (out_retain_vk)
            *out_retain_vk = true;
    }
    if (ibv_quiesced) {
        if (second_mr)
            vxfec_ibv_mr_dereg(second_mr);
        if (ibv_mr)
            vxfec_ibv_mr_dereg(ibv_mr);
        if (ibv_ctx)
            vxfec_ibv_ctx_close(ibv_ctx);
    }
    if (ibv_quiesced && vk_quiesced) {
        vxfec_wire_ring_free(rx_ring);
        vxfec_vk_buf_free(ctx, rx_vram);
        vxfec_vk_buf_free(ctx, second_vram);
        if (second_fd >= 0)
            close(second_fd);
        if (dmabuf_fd >= 0)
            close(dmabuf_fd);
        free(rx_host);
    }
    free(rx_bits);
    free(rx_bits_atomic);
    free(rx_verify);
    free(stats.latencies_ns);
    close(ctrl_fd);
    close(listen_fd);
    return rx_aborted ? 1 : 0;
}

static int run_tx_client(vxfec_vk_ctx *ctx, const wire_cli_cfg_t *cfg, bool *out_retain_vk)
{
    if (out_retain_vk)
        *out_retain_vk = false;
    struct in_addr peers[WIRE_NET_MAX_RAILS];
    uint32_t n_peers = parse_ip_list(cfg->peers_csv, peers, WIRE_NET_MAX_RAILS);
    if (n_peers == 0) {
        fprintf(stderr, "TX mode requires --peers <ip1,ip2,...>\n");
        return 1;
    }

    int ctrl_fd = vxfec_bench_tcp_connect_retry(peers[0], (uint16_t)(cfg->base_port - 1u), 100, 100000000L);
    if (ctrl_fd < 0) {
        perror("connect ctrl");
        return 1;
    }

    uint32_t Z = cfg->Z, K = cfg->K, P = cfg->P, T = cfg->T;
    size_t sys_bytes = (size_t)Z * K * T;
    size_t wire_bytes = (size_t)Z * (K + P) * T;
    uint64_t total_frames = (uint64_t)(K + P) * Z;

    size_t dmabuf_bytes = 0u;
    size_t page_sz = 0u;
    if (cfg->gdr) {
        page_sz = (size_t)sysconf(_SC_PAGESIZE);
        if (wire_bytes > SIZE_MAX - page_sz) {
            fprintf(stderr, "TX superframe is too large for page-aligned DMA-BUF registration\n");
            close(ctrl_fd);
            return 1;
        }
        dmabuf_bytes = (wire_bytes + page_sz - 1u) & ~(page_sz - 1u);
    }
    uint8_t *tx_host = (uint8_t *)malloc(cfg->gdr ? sys_bytes : wire_bytes);
    if (!tx_host) {
        close(ctrl_fd);
        return 1;
    }

    fill_deterministic_superframe(tx_host, sys_bytes);
    vxfec_wire_ring *tx_ring = NULL;
    vxfec_vk_buf *tx_vram = NULL;
    int dmabuf_fd = -1;
    vxfec_ibv_ctx_t *ibv_ctx = NULL;
    vxfec_ibv_mr_t *ibv_mr = NULL;
    vxfec_ibv_ep_t *ibv_ep = NULL;
    int tx_fds[WIRE_NET_MAX_RAILS];
    int bidir_fd = -1;
    uint32_t open_tx_fds = 0;
    bool bench_ok = false;

    if (cfg->wire_only) {
        fill_deterministic_superframe(tx_host + sys_bytes, wire_bytes - sys_bytes);
    } else {
        tx_ring = vxfec_wire_ring_new_ex(ctx, cfg->codec, Z, K, P, T, 0u, true, false);
        tx_vram =
            cfg->gdr ? vxfec_vk_buf_new_exportable(ctx, dmabuf_bytes, true, &dmabuf_fd) : vxfec_vk_buf_new(ctx, wire_bytes, false);
        if (cfg->gdr && !tx_vram) {
            fprintf(stderr, "TX --gdr failed to allocate/export a GPU DMA-BUF\n");
            goto tx_cleanup;
        }
        if (!tx_ring || !tx_vram || !vxfec_vk_buf_write_offset(ctx, tx_vram, 0, tx_host, sys_bytes) ||
            !vxfec_wire_encode_vram(tx_ring, tx_vram, NULL) ||
            (cfg->gdr ? !vxfec_vk_buf_release_external(ctx, tx_vram)
                      : !vxfec_vk_buf_read_offset(ctx, tx_vram, 0, tx_host, wire_bytes))) {
            fprintf(stderr, "TX superframe GPU encode failed\n");
            goto tx_cleanup;
        }
        if (cfg->gdr) {
            free(tx_host);
            tx_host = NULL;
        }
    }

    wire_ctrl_handshake_t hs;
    memset(&hs, 0, sizeof(hs));
    hs.magic = wire_handshake_magic(cfg);
    hs.codec = (uint32_t)cfg->codec;
    hs.Z = Z;
    hs.K = K;
    hs.P = P;
    hs.T = T;
    hs.rails = n_peers * cfg->ports_per_ip;
    hs.iters = cfg->iters;
    hs.use_ibv = cfg->use_ibv ? 1u : 0u;
    hs.wire_only = cfg->wire_only ? 1u : 0u;

    if (cfg->use_ibv) {
        ibv_ctx = vxfec_ibv_ctx_open(0);
        if (!ibv_ctx) {
            fprintf(stderr, "TX requested --ibv but no active RDMA device found\n");
            goto tx_cleanup;
        }
        ibv_mr = cfg->gdr ? vxfec_ibv_mr_reg_dmabuf(ibv_ctx, 0, dmabuf_bytes, page_sz, dmabuf_fd, false)
                          : vxfec_ibv_mr_reg_host(ibv_ctx, tx_host, wire_bytes, false);
        if (cfg->gdr && !ibv_mr) {
            fprintf(stderr, "TX --gdr failed to register the GPU DMA-BUF with ibverbs\n");
            goto tx_cleanup;
        }
        ibv_ep = vxfec_ibv_ep_create_ex(ibv_ctx, WIRE_NET_IBV_QPS, 256u, true, false, &hs.ibv_info);
        if (!ibv_mr || !ibv_ep) {
            fprintf(stderr, "TX failed to create ibverbs MR/EP\n");
            goto tx_cleanup;
        }
    }

    wire_ctrl_handshake_t rx_hs;
    if (!write_all(ctrl_fd, &hs, sizeof(hs)) || !read_all(ctrl_fd, &rx_hs, sizeof(rx_hs)) || rx_hs.magic != hs.magic) {
        fprintf(stderr, "TX/RX handshake failed\n");
        goto tx_cleanup;
    }
    if (cfg->gdr && !rx_hs.gdr_dmabuf) {
        fprintf(stderr, "TX --gdr requires a GPU DMA-BUF MR from RX\n");
        goto tx_cleanup;
    }
    wire_rdma_region_t regions[2] = {{.iova = rx_hs.remote_iova, .rkey = rx_hs.remote_rkey}, {0}};
    if (cfg->pipeline && (!read_all(ctrl_fd, &regions[1], sizeof(regions[1])) || regions[1].iova == 0u)) {
        fprintf(stderr, "TX --pipeline requires a second registered RX GPU buffer\n");
        goto tx_cleanup;
    }
    if (cfg->use_ibv && !vxfec_ibv_ep_connect(ibv_ep, &rx_hs.ibv_info, 0)) {
        fprintf(stderr, "TX failed to connect ibverbs QPs\n");
        goto tx_cleanup;
    }

    if (cfg->use_ibv && T > vxfec_ibv_ep_path_mtu(ibv_ep)) {
        fprintf(stderr, "TX --sym-t %u exceeds the RDMA path MTU of %u bytes\n", T, vxfec_ibv_ep_path_mtu(ibv_ep));
        goto tx_cleanup;
    }

    struct sockaddr_in peer_addrs[WIRE_NET_MAX_RAILS];
    uint32_t rails = cfg->use_ibv ? WIRE_NET_IBV_QPS : rx_hs.rails;
    if (!cfg->use_ibv) {
        uint32_t idx = 0;
        for (uint32_t ip_i = 0; ip_i < n_peers && idx < rails && idx < WIRE_NET_MAX_RAILS; ip_i++) {
            for (uint32_t q = 0; q < cfg->ports_per_ip && idx < rails && idx < WIRE_NET_MAX_RAILS; q++) {
                int fd = socket(AF_INET, SOCK_DGRAM, 0);
                if (fd < 0) {
                    perror("socket udp tx");
                    goto tx_cleanup;
                }
                tx_fds[idx] = fd;
                open_tx_fds = idx + 1u;
                tune_udp_socket(tx_fds[idx]);
                memset(&peer_addrs[idx], 0, sizeof(peer_addrs[idx]));
                peer_addrs[idx].sin_family = AF_INET;
                peer_addrs[idx].sin_addr = peers[ip_i];
                peer_addrs[idx].sin_port = htons((uint16_t)(cfg->base_port + idx));
                idx++;
            }
        }
        rails = idx;
    }

    if (cfg->bidir_sync) {
        bidir_fd = open_bidir_sync_socket(cfg->bidir_sync, cfg->bidir_leader);
        if (bidir_fd < 0) {
            fprintf(stderr, "Failed to establish --bidir-sync barrier on %s\n", cfg->bidir_sync);
            goto tx_cleanup;
        }
    }

    uint64_t drop_frames = (uint64_t)((double)(K * Z) * (cfg->blackout_pct / 100.0));
    uint64_t drop_beg = (uint64_t)Z * 2u + 7u;
    uint64_t drop_end = drop_beg + drop_frames;

    double best_wire_gbps = 0.0, best_e2e_gbps = 0.0, best_gpu_gbps = 0.0;
    double best_bidir_wire_gbps = 0.0, best_bidir_wall_gbps = 0.0;
    uint64_t best_planned_frames = 0, best_rx_frames = 0;
    uint32_t last_rep = 0, last_masks = 0, pass_count = 0;

    printf("===========================================================================================================\n");
    const char *transport = "Multi-NIC UDP";
    if (cfg->pipeline)
        transport = "IBV UC GPU-to-GPU DMA-BUF, two RX buffers";
    else if (cfg->gdr)
        transport = "IBV UC GPU-to-GPU DMA-BUF";
    else if (cfg->use_ibv)
        transport = rx_hs.gdr_dmabuf ? "IBV UC host-to-GPU DMA-BUF" : "IBV Zero-Bounce UC";
    printf("vxfec Multi-Rail Wire -> VRAM Superframe Benchmark (%s%s, Z=%u K=%u P=%u T=%u B, %.1f MiB)\n", transport,
           cfg->wire_only ? ", wire-only" : "", Z, K, P, T, (double)sys_bytes / (1024.0 * 1024.0));
    printf("===========================================================================================================\n");

    struct wire_tx_sample {
        uint64_t planned_frames;
        uint64_t started_ns;
        uint64_t wire_ns;
    } samples[2];
    /* In pipeline mode, send the next frame before waiting for the previous recovery result. */
    const uint32_t result_lag = cfg->pipeline ? 1u : 0u;
    bool tx_aborted = false;
    for (uint32_t send_it = 1; send_it <= cfg->iters + result_lag; send_it++) {
        if (send_it <= cfg->iters) {
            uint32_t it = send_it;
            uint64_t planned_frames =
                count_planned_tx_frames(total_frames, drop_beg, drop_end, cfg->loss_ppm, rails, it, cfg->use_ibv);
            if (bidir_fd >= 0) {
                uint32_t peer_it = 0;
                if (!write_all(bidir_fd, &it, sizeof(it)) || !read_all(bidir_fd, &peer_it, sizeof(peer_it)) || peer_it != it) {
                    tx_aborted = true;
                    break;
                }
            }
            uint64_t iter_wall_t0 = now_ns();
            uint32_t ready_ack = 0;
            if (!write_all(ctrl_fd, &planned_frames, sizeof(planned_frames)) || !read_all(ctrl_fd, &ready_ack, sizeof(ready_ack)) ||
                ready_ack != it) {
                tx_aborted = true;
                break;
            }
            uint64_t tx_wire_ns = 1ULL;

            if (cfg->use_ibv) {
                uint64_t tx_wire_t0 = now_ns();
                uint64_t rng = 0x9876543210fedcbaULL ^ it;
                uint32_t lkey = vxfec_ibv_mr_lkey(ibv_mr);
                uint64_t src_iova = vxfec_ibv_mr_iova(ibv_mr);
                const wire_rdma_region_t *region = &regions[(it - 1u) % (result_lag + 1u)];
                for (uint64_t seq = 0; seq < total_frames; seq++) {
                    if (seq >= drop_beg && seq < drop_end)
                        continue;
                    if (cfg->loss_ppm > 0 && (splitmix64(&rng) % 1000000ULL) < cfg->loss_ppm)
                        continue;
                    uint64_t offset = seq * (uint64_t)T;
                    const void *src = (const void *)(uintptr_t)(src_iova + offset);
                    while (!vxfec_ibv_ep_rdma_write_imm(ibv_ep, (uint32_t)(seq % WIRE_NET_IBV_QPS), src, T, lkey,
                                                        region->iova + offset, region->rkey, (uint32_t)seq)) {
                        vxfec_ibv_ep_drain_tx(ibv_ep);
                        if (vxfec_ibv_ep_has_error(ibv_ep))
                            break;
                    }
                    if (vxfec_ibv_ep_has_error(ibv_ep))
                        break;
                }
                if (!vxfec_ibv_ep_has_error(ibv_ep)) {
                    for (uint32_t q = 0; q < WIRE_NET_IBV_QPS; q++) {
                        while (!vxfec_ibv_ep_rdma_write_imm(ibv_ep, q, NULL, 0u, lkey, region->iova, region->rkey, UINT32_MAX)) {
                            vxfec_ibv_ep_drain_tx(ibv_ep);
                            if (vxfec_ibv_ep_has_error(ibv_ep))
                                break;
                        }
                        if (vxfec_ibv_ep_has_error(ibv_ep))
                            break;
                    }
                    vxfec_ibv_ep_drain_tx(ibv_ep);
                }
                if (vxfec_ibv_ep_has_error(ibv_ep)) {
                    fprintf(stderr, "ERROR: TX ibverbs fatal CQ/send failure at iter %u\n", it);
                    tx_aborted = true;
                    shutdown(ctrl_fd, SHUT_RDWR);
                    break;
                }
                tx_wire_ns = now_ns() - tx_wire_t0;
            } else {
                _Atomic uint32_t ready_tx = 0;
                _Atomic bool go_tx = false;
                pthread_t tx_th[WIRE_NET_MAX_RAILS];
                udp_tx_worker_t tx_w[WIRE_NET_MAX_RAILS];
                uint32_t started_tx = 0;
                double rail_rate = (cfg->rate_gbps > 0.0 && rails > 0u) ? (cfg->rate_gbps / (double)rails) : 0.0;
                for (uint32_t r = 0; r < rails; r++) {
                    tx_w[r] = (udp_tx_worker_t){
                        .fd = tx_fds[r],
                        .peer = peer_addrs[r],
                        .wire_src = tx_host,
                        .rail_idx = r,
                        .num_rails = rails,
                        .T = T,
                        .iter = (uint16_t)it,
                        .total_frames = total_frames,
                        .drop_beg = drop_beg,
                        .drop_end = drop_end,
                        .loss_ppm = cfg->loss_ppm,
                        .rail_rate_gbps = rail_rate,
                        .ready_workers = &ready_tx,
                        .go_flag = &go_tx,
                        .sent_frames = 0u,
                        .my_first_ns = 0u,
                        .my_last_ns = 0u,
                    };
                    if (pthread_create(&tx_th[r], NULL, udp_tx_worker_fn, &tx_w[r]) != 0) {
                        tx_aborted = true;
                        break;
                    }
                    started_tx++;
                }
                if (!tx_aborted) {
                    while (atomic_load_explicit(&ready_tx, memory_order_acquire) < started_tx)
                        sched_yield();
                }
                atomic_store_explicit(&go_tx, true, memory_order_release);
                for (uint32_t r = 0; r < started_tx; r++)
                    pthread_join(tx_th[r], NULL);
                send_tail_beacons(tx_fds, peer_addrs, rails, (uint16_t)it);
                if (tx_aborted) {
                    shutdown(ctrl_fd, SHUT_RDWR);
                    break;
                }
                uint64_t tfn = UINT64_MAX, tln = 0;
                for (uint32_t r = 0; r < started_tx; r++) {
                    if (tx_w[r].my_first_ns > 0 && tx_w[r].my_first_ns < tfn)
                        tfn = tx_w[r].my_first_ns;
                    if (tx_w[r].my_last_ns > tln)
                        tln = tx_w[r].my_last_ns;
                }
                tx_wire_ns = (tfn != UINT64_MAX && tln > tfn) ? (tln - tfn) : 1ULL;
            }
            struct wire_tx_sample *sample = &samples[(it - 1u) % 2u];
            *sample = (struct wire_tx_sample){
                .planned_frames = planned_frames,
                .started_ns = iter_wall_t0,
                .wire_ns = tx_wire_ns ? tx_wire_ns : 1u,
            };
        }
        if (send_it <= result_lag)
            continue;

        uint32_t it = send_it - result_lag;
        const struct wire_tx_sample *sample = &samples[(it - 1u) % 2u];
        uint64_t planned_frames = sample->planned_frames;
        wire_ctrl_result_t res;
        if (!read_all(ctrl_fd, &res, sizeof(res)) || res.iter != it) {
            tx_aborted = true;
            break;
        }
        if (res.wire_ns < sample->wire_ns)
            res.wire_ns = sample->wire_ns;
        uint64_t iter_wall_ns = now_ns() - sample->started_ns;

        double wire_gbps = (8.0 * (double)res.rx_frames * (double)T) / (double)(res.wire_ns ? res.wire_ns : 1ULL);
        uint64_t e2e_ns = res.wire_ns + res.dma_ns + res.recover_wall_ns;
        double e2e_gbps = cfg->wire_only ? 0.0 : ((8.0 * (double)sys_bytes) / (double)(e2e_ns ? e2e_ns : 1ULL));
        uint64_t rec_ns = res.recover_gpu_ns ? res.recover_gpu_ns : res.recover_wall_ns;
        double gpu_gbps = cfg->wire_only ? 0.0 : ((8.0 * (double)sys_bytes) / (double)(rec_ns ? rec_ns : 1ULL));

        double bidir_wire_gbps = 0.0, bidir_wall_gbps = 0.0;
        bool peer_fatal = false;
        if (bidir_fd >= 0) {
            wire_bidir_iter_t my_bi = {
                .iter = it,
                .ok = res.ok,
                .fatal_err = res.fatal_err,
                .rx_bytes = res.rx_frames * (uint64_t)T,
                .wire_ns = res.wire_ns,
                .wall_ns = iter_wall_ns,
            };
            wire_bidir_iter_t peer_bi = {0};
            if (!write_all(bidir_fd, &my_bi, sizeof(my_bi)) || !read_all(bidir_fd, &peer_bi, sizeof(peer_bi))) {
                tx_aborted = true;
                break;
            }
            peer_fatal = (peer_bi.fatal_err != 0u);
            if (res.ok && !res.fatal_err && peer_bi.ok && !peer_fatal) {
                uint64_t common_wire = (res.wire_ns > peer_bi.wire_ns) ? res.wire_ns : peer_bi.wire_ns;
                uint64_t common_wall = (iter_wall_ns > peer_bi.wall_ns) ? iter_wall_ns : peer_bi.wall_ns;
                double tot_bits = 8.0 * (double)(my_bi.rx_bytes + peer_bi.rx_bytes);
                bidir_wire_gbps = tot_bits / (double)(common_wire ? common_wire : 1ULL);
                bidir_wall_gbps = tot_bits / (double)(common_wall ? common_wall : 1ULL);
                if (bidir_wire_gbps > best_bidir_wire_gbps) {
                    best_bidir_wire_gbps = bidir_wire_gbps;
                    best_bidir_wall_gbps = bidir_wall_gbps;
                }
            }
        }

        if (res.fatal_err || peer_fatal) {
            fprintf(stderr, "ERROR: Fatal payload/decode verification failure on iter %u; aborting benchmark!\n", it);
            tx_aborted = true;
            break;
        }

        if (res.ok) {
            pass_count++;
            /* capture all CSV metrics coherently from the single best-wire passing iteration */
            if (pass_count == 1u || wire_gbps > best_wire_gbps) {
                best_wire_gbps = wire_gbps;
                best_planned_frames = planned_frames;
                best_rx_frames = res.rx_frames;
                if (!cfg->wire_only) {
                    best_e2e_gbps = e2e_gbps;
                    best_gpu_gbps = gpu_gbps;
                    last_rep = res.blocks_repaired;
                    last_masks = res.unique_masks;
                }
            }
        }

        if (cfg->wire_only) {
            if (bidir_fd >= 0 && cfg->bidir_leader && bidir_wire_gbps > 0.0) {
                printf("  Iter %2u/%2u | RX=%6" PRIu64 "/%" PRIu64 " (of %" PRIu64 ") | DirWire=%7.2f Gbps | "
                       "BiDirWire=%7.2f Gbps (Wall=%7.2f Gbps) | %s\n",
                       it, cfg->iters, res.rx_frames, planned_frames, total_frames, wire_gbps, bidir_wire_gbps, bidir_wall_gbps,
                       res.ok ? "PASS" : "SHAPER-DROP");
            } else {
                printf("  Iter %2u/%2u | RX=%6" PRIu64 "/%" PRIu64 " (of %" PRIu64
                       ") | Wire=%7.2f Gbps | E2E=    n/a | GPU Rec=    n/a | %s\n",
                       it, cfg->iters, res.rx_frames, planned_frames, total_frames, wire_gbps, res.ok ? "PASS" : "SHAPER-DROP");
            }
        } else {
            printf("  Iter %2u/%2u | RX=%6" PRIu64 "/%" PRIu64 " (of %" PRIu64
                   ") | Wire=%7.2f Gbps | E2E=%7.2f Gbps | GPU Rec=%7.2f Gbps | %s\n",
                   it, cfg->iters, res.rx_frames, planned_frames, total_frames, wire_gbps, e2e_gbps, gpu_gbps,
                   res.ok ? "PASS" : "SHAPER-DROP");
        }
    }
    printf("===========================================================================================================\n");
    bench_ok = !tx_aborted && (cfg->gdr ? pass_count == cfg->iters : pass_count > 0);
    if (!bench_ok)
        fprintf(stderr, "ERROR: Wire benchmark aborted or failed RX recovery!\n");

    if (bench_ok && cfg->csv_path) {
        bool need_hdr = (access(cfg->csv_path, F_OK) != 0);
        FILE *fp = fopen(cfg->csv_path, "a");
        if (fp) {
            if (need_hdr) {
                fprintf(fp,
                        "transport,codec,wire_only,Z,K,P,T,rails,blackout_pct,loss_ppm,planned_frames,rx_frames,payload_mib,"
                        "wire_gbps,bidir_wire_gbps,bidir_wall_gbps,e2e_gbps,gpu_gbps,blocks_repaired,unique_masks,gdr_dmabuf\n");
            }
            const char *cname =
                (cfg->codec == VXFEC_WIRE_MDS) ? "mds" : ((cfg->codec == VXFEC_WIRE_RATELESS_GF2) ? "rateless-gf2" : "rateless");
            fprintf(fp, "%s,%s,%u,%u,%u,%u,%u,%u,%.2f,%u,%" PRIu64 ",%" PRIu64 ",%.2f,%.3f,",
                    cfg->pipeline ? "ibv-gdr-pipeline" : (cfg->gdr ? "ibv-gdr" : (cfg->use_ibv ? "ibv" : "udp")), cname,
                    cfg->wire_only ? 1u : 0u, Z, K, P, T, rails, cfg->blackout_pct, cfg->loss_ppm, best_planned_frames,
                    best_rx_frames, (double)sys_bytes / (1024.0 * 1024.0), best_wire_gbps);
            if (bidir_fd >= 0 && best_bidir_wire_gbps > 0.0)
                fprintf(fp, "%.3f,%.3f,", best_bidir_wire_gbps, best_bidir_wall_gbps);
            else
                fprintf(fp, ",,");
            if (cfg->wire_only)
                fprintf(fp, ",,,,%u\n", rx_hs.gdr_dmabuf);
            else
                fprintf(fp, "%.3f,%.3f,%u,%u,%u\n", best_e2e_gbps, best_gpu_gbps, last_rep, last_masks, rx_hs.gdr_dmabuf);
            fclose(fp);
        }
    }

tx_cleanup:
    if (bidir_fd >= 0)
        close(bidir_fd);
    for (uint32_t r = 0; r < open_tx_fds; r++)
        close(tx_fds[r]);
    bool ibv_quiesced = !ibv_ep || vxfec_ibv_ep_destroy(ibv_ep);
    bool vk_quiesced = vxfec_vk_wait_idle(ctx);
    if (!ibv_quiesced || !vk_quiesced) {
        bench_ok = false;
        if (out_retain_vk)
            *out_retain_vk = true;
    }
    if (ibv_quiesced) {
        if (ibv_mr)
            vxfec_ibv_mr_dereg(ibv_mr);
        if (ibv_ctx)
            vxfec_ibv_ctx_close(ibv_ctx);
    }
    if (ibv_quiesced && vk_quiesced) {
        vxfec_wire_ring_free(tx_ring);
        vxfec_vk_buf_free(ctx, tx_vram);
        if (dmabuf_fd >= 0)
            close(dmabuf_fd);
    }
    if (ibv_quiesced)
        free(tx_host);
    close(ctrl_fd);
    return bench_ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    wire_cli_cfg_t cfg = {
        .mode = NULL,
        .bind_csv = "0.0.0.0",
        .peers_csv = NULL,
        .csv_path = NULL,
        .base_port = 31500u,
        .ports_per_ip = 4u,
        .codec = VXFEC_WIRE_RATELESS,
        .Z = 64u,
        .K = 200u,
        .P = 48u,
        .T = 8192u,
        .iters = 5u,
        .blackout_pct = 10.0,
        .rate_gbps = 0.0,
        .loss_ppm = 0u,
        .use_ibv = false,
        .wire_only = false,
        .self_test = false,
    };
    bool sym_t_set = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-test") == 0) {
            cfg.self_test = true;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            cfg.mode = argv[++i];
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            cfg.bind_csv = argv[++i];
        } else if (strcmp(argv[i], "--peers") == 0 && i + 1 < argc) {
            cfg.peers_csv = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            cfg.base_port = (uint16_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--ports-per-ip") == 0 && i + 1 < argc) {
            cfg.ports_per_ip = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            const char *c = argv[++i];
            if (strcmp(c, "mds") == 0 || strcmp(c, "rs8") == 0)
                cfg.codec = VXFEC_WIRE_MDS;
            else if (strcmp(c, "rateless-gf2") == 0 || strcmp(c, "gf2") == 0)
                cfg.codec = VXFEC_WIRE_RATELESS_GF2;
            else
                cfg.codec = VXFEC_WIRE_RATELESS;
        } else if (strcmp(argv[i], "--z") == 0 && i + 1 < argc) {
            cfg.Z = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            cfg.K = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--p") == 0 && i + 1 < argc) {
            cfg.P = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--sym-t") == 0 && i + 1 < argc) {
            cfg.T = (uint32_t)strtoul(argv[++i], NULL, 10);
            sym_t_set = true;
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            cfg.iters = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--blackout-pct") == 0 && i + 1 < argc) {
            cfg.blackout_pct = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--rate-gbps") == 0 && i + 1 < argc) {
            cfg.rate_gbps = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--loss-ppm") == 0 && i + 1 < argc) {
            cfg.loss_ppm = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--ibv") == 0) {
            cfg.use_ibv = true;
        } else if (strcmp(argv[i], "--gdr") == 0) {
            cfg.gdr = true;
            cfg.use_ibv = true;
        } else if (strcmp(argv[i], "--pipeline") == 0) {
            cfg.pipeline = true;
        } else if (strcmp(argv[i], "--wire-only") == 0) {
            cfg.wire_only = true;
        } else if (strcmp(argv[i], "--bidir-sync") == 0 && i + 1 < argc) {
            cfg.bidir_sync = argv[++i];
        } else if (strcmp(argv[i], "--bidir-leader") == 0) {
            cfg.bidir_leader = true;
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            cfg.csv_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage:\n");
            printf("  %s --self-test\n", argv[0]);
            printf("  %s --mode rx [--bind ip1,ip2] [--port 31500] [--ports-per-ip 4] [--gdr [--pipeline]]\n", argv[0]);
            printf("  %s --mode tx --peers ip1,ip2 [--port 31500] [--ports-per-ip 4] [--codec rateless|mds] "
                   "[--z 64] [--k 200] [--p 48] [--sym-t bytes] [--blackout-pct 10] [--ibv | --gdr [--pipeline]] [--csv out.csv]\n",
                   argv[0]);
            printf("  --gdr        Require GPU DMA-BUF RDMA on both peers; use on TX and RX.\n");
            printf("  --pipeline   With --gdr on both peers, use two RX buffers to overlap transfer and recovery.\n");
            printf("  --sym-t      Symbol size (default 4096 with --gdr, 8192 otherwise).\n");
            return 0;
        }
    }

    if (cfg.pipeline && !cfg.gdr) {
        fprintf(stderr, "--pipeline requires --gdr on both endpoints\n");
        return 1;
    }

    if (cfg.gdr) {
        if (cfg.self_test || !cfg.mode || (strcmp(cfg.mode, "tx") != 0 && strcmp(cfg.mode, "rx") != 0) || cfg.wire_only ||
            cfg.bidir_sync || cfg.bidir_leader || cfg.rate_gbps != 0.0) {
            fprintf(stderr, "--gdr requires --mode tx or rx and cannot use --self-test, --wire-only, --bidir-*, or --rate-gbps\n");
            return 1;
        }
        if (!sym_t_set)
            cfg.T = 4096u;
    }

    if (!cfg.self_test && cfg.mode) {
        if (cfg.base_port < 2u || (uint32_t)cfg.base_port + WIRE_NET_MAX_RAILS > 65535u || cfg.ports_per_ip == 0u ||
            cfg.ports_per_ip > WIRE_NET_MAX_RAILS) {
            fprintf(stderr, "Invalid --port %u / --ports-per-ip %u (port must allow ctrl port-1 and <= %u rails)\n",
                    (unsigned)cfg.base_port, cfg.ports_per_ip, WIRE_NET_MAX_RAILS);
            return 1;
        }
        if (strcmp(cfg.mode, "tx") == 0) {
            if (cfg.Z == 0u || cfg.Z > 4096u || cfg.K == 0u || cfg.K > 56403u || cfg.P == 0u || cfg.P > 65535u || cfg.T < 64u ||
                cfg.T > (1u << 20) || (cfg.T & 15u) != 0u || cfg.iters == 0u || cfg.iters > 100000u || cfg.blackout_pct < 0.0 ||
                cfg.blackout_pct > 100.0 || (uint64_t)(cfg.K + cfg.P) > SIZE_MAX / ((size_t)cfg.Z * (size_t)cfg.T)) {
                fprintf(stderr, "Invalid TX superframe parameters (Z=%u K=%u P=%u T=%u iters=%u blackout=%.2f)\n", cfg.Z, cfg.K,
                        cfg.P, cfg.T, cfg.iters, cfg.blackout_pct);
                return 1;
            }
        }
    }

    int vk_status = VXFEC_VK_ERROR;
    vxfec_vk_ctx *ctx = vxfec_vk_ctx_new_status(&vk_status);
    if (!ctx) {
        const char *req_env = getenv("VXFEC_VK_REQUIRED");
        if (!req_env || !*req_env)
            req_env = getenv("NANORQ_VK_REQUIRED");
        bool vk_req = req_env && *req_env && strcmp(req_env, "0") != 0;
        bool can_skip = (vk_status == VXFEC_VK_UNAVAILABLE && !vk_req && !cfg.gdr);
        fprintf(stderr, "%s: Vulkan 1.2 compute device unavailable (status=%d)\n", can_skip ? "SKIP" : "FAIL", vk_status);
        return can_skip ? 0 : 1;
    }

    if (cfg.gdr && (!vxfec_vk_has_dmabuf_export(ctx) || !vxfec_ibv_has_dmabuf())) {
        fprintf(stderr, "--gdr requires Vulkan GPU DMA-BUF export and ibverbs DMA-BUF registration (device: %s)\n",
                vxfec_vk_device_name(ctx));
        vxfec_vk_ctx_free(ctx);
        return 1;
    }

    if (cfg.self_test || !cfg.mode) {
        printf("=== vxfec Wire -> VRAM Superframe Loopback Self-Test (%s) ===\n", vxfec_vk_device_name(ctx));
        bool retain_rq = false, retain_rs = false;
        bool ok_rq = run_loopback_self_test_codec(ctx, VXFEC_WIRE_RATELESS, 16u, 128u, 36u, 4096u, &retain_rq);
        bool ok_rs = !retain_rq && run_loopback_self_test_codec(ctx, VXFEC_WIRE_MDS, 16u, 128u, 24u, 4096u, &retain_rs);
        if (!retain_rq && !retain_rs)
            vxfec_vk_ctx_free(ctx);
        return (ok_rq && ok_rs) ? 0 : 1;
    }

    bool retain_vk = false;
    int rc = 1;
    if (strcmp(cfg.mode, "rx") == 0)
        rc = run_rx_server(ctx, &cfg, &retain_vk);
    else if (strcmp(cfg.mode, "tx") == 0)
        rc = run_tx_client(ctx, &cfg, &retain_vk);
    else
        fprintf(stderr, "Unknown --mode '%s' (expected 'rx' or 'tx')\n", cfg.mode);

    if (!retain_vk)
        vxfec_vk_ctx_free(ctx);
    return rc;
}
