/*
 * clos_switch.c - Self-contained Spine Switch MMU Emulator, Netlink TBF Shaper,
 *                 and ECMP Elephant Cross-Traffic Generator for 2-Leaf/2-Spine Clos tests.
 *
 * Forwards real L2 Ethernet / RoCEv2 (UDP 4791) / UDP frames between two Leaf uplinks
 * using AF_PACKET + PACKET_VNET_HDR (preserving veth CHECKSUM_PARTIAL offloads) with:
 *   - Token-bucket line-rate shaping (Mbps)
 *   - Shared egress MMU buffer (bytes) with hard tail-drop overflow
 *   - Hardware-style K_min ECN marking (rewrites IPv4 ECT -> CE + RFC 1624 header csum)
 *   - Configurable 1-way wire propagation delay (us)
 *   - Optional native NETLINK_ROUTE RTM_NEWQDISC sch_tbf kernel qdisc installer
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/ip.h>
#include <linux/virtio_net.h>
#include <net/if.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#ifndef PACKET_IGNORE_OUTGOING
#define PACKET_IGNORE_OUTGOING 23
#endif

#ifndef IPTOS_ECN_MASK
#define IPTOS_ECN_MASK 0x03u
#define IPTOS_ECN_ECT0 0x02u
#define IPTOS_ECN_CE 0x03u
#endif

#define MAX_FRAME_BUF 16384u
#define RING_CAP 2048u

static volatile sig_atomic_t g_stop = 0;
static volatile sig_atomic_t g_flush = 0;

static void on_sig(int sig)
{
    if (sig == SIGUSR1)
        g_flush = 1;
    else
        g_stop = 1;
}

static inline uint64_t mono_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

typedef struct {
    uint64_t arrival_ns;
    uint32_t len;      /* includes sizeof(struct virtio_net_hdr) */
    uint32_t wire_len; /* Ethernet frame bytes on wire */
    bool is_elephant;
    _Alignas(8) uint8_t data[MAX_FRAME_BUF];
} switch_pkt_t;

typedef struct {
    switch_pkt_t *ring;
    uint32_t head;
    uint32_t cnt;
    uint64_t q_bytes;
    uint64_t peak_q_bytes;
    uint64_t mmu_cap_bytes;
    uint64_t ecn_min_bytes;
    uint64_t delay_ns;
    double rate_Bps;
    double burst_cap_bytes;
    double tokens_bytes;
    uint64_t last_ns;

    uint64_t rx_pkts;
    uint64_t tx_pkts;
    uint64_t tx_bytes;
    uint64_t ecn_marks;
    uint64_t tail_drops;
    uint64_t app_tx_bytes;
    uint64_t app_ecn_marks;
    uint64_t app_tail_drops;
    uint64_t rx_truncated;
    uint64_t rx_gso;
    uint64_t rx_errors;
    uint64_t socket_drops;
    uint64_t tx_errors;
} switch_port_q_t;

static bool is_elephant_frame(const uint8_t *frame, size_t frame_len)
{
    if (frame_len < sizeof(struct ethhdr) + sizeof(struct iphdr) + 8u)
        return false;
    const struct ethhdr *eth = (const struct ethhdr *)frame;
    if (ntohs(eth->h_proto) != ETH_P_IP)
        return false;
    const struct iphdr *iph = (const struct iphdr *)(frame + sizeof(struct ethhdr));
    if (iph->version != 4u || iph->ihl < 5u || iph->protocol != IPPROTO_UDP)
        return false;
    size_t l4_off = sizeof(struct ethhdr) + (size_t)iph->ihl * 4u;
    if (frame_len < l4_off + 4u)
        return false;
    uint16_t dport_be;
    memcpy(&dport_be, frame + l4_off + 2u, sizeof(dport_be));
    return ntohs(dport_be) == 52001u;
}

static void maybe_mark_ipv4_ecn_ce(uint8_t *frame, size_t frame_len, uint64_t *ecn_counter)
{
    if (frame_len < sizeof(struct ethhdr) + sizeof(struct iphdr))
        return;
    const struct ethhdr *eth = (const struct ethhdr *)frame;
    uint16_t proto = ntohs(eth->h_proto);
    size_t l3_off = sizeof(struct ethhdr);
    if (proto == ETH_P_8021Q && frame_len >= l3_off + 4u + sizeof(struct iphdr)) {
        uint16_t inner;
        memcpy(&inner, frame + l3_off + 2u, sizeof(inner));
        proto = ntohs(inner);
        l3_off += 4u;
    }
    if (proto != ETH_P_IP)
        return;

    struct iphdr *iph = (struct iphdr *)(frame + l3_off);
    if (iph->version != 4u || iph->ihl < 5u || frame_len < l3_off + (size_t)iph->ihl * 4u)
        return;

    uint8_t ecn = iph->tos & IPTOS_ECN_MASK;
    if (ecn == 0x01u || ecn == 0x02u) {
        uint16_t old_word = ((uint16_t)iph->version << 12u) | ((uint16_t)iph->ihl << 8u) | iph->tos;
        iph->tos = (uint8_t)((iph->tos & ~IPTOS_ECN_MASK) | IPTOS_ECN_CE);
        uint16_t new_word = ((uint16_t)iph->version << 12u) | ((uint16_t)iph->ihl << 8u) | iph->tos;

        /* RFC 1624 incremental IPv4 header checksum update */
        uint32_t sum = (uint16_t)~ntohs(iph->check);
        sum += (uint16_t)~old_word;
        sum += new_word;
        while (sum >> 16u)
            sum = (sum & 0xffffu) + (sum >> 16u);
        iph->check = htons((uint16_t)~sum);
        (*ecn_counter)++;
    }
}

static int open_raw_port(const char *ifname, int *out_ifindex)
{
    int fd = socket(AF_PACKET, SOCK_RAW | SOCK_NONBLOCK | SOCK_CLOEXEC, htons(ETH_P_ALL));
    if (fd < 0) {
        perror("socket(AF_PACKET)");
        return -1;
    }

    int ifidx = (int)if_nametoindex(ifname);
    if (ifidx == 0) {
        perror(ifname);
        close(fd);
        return -1;
    }

    int one = 1;
    (void)setsockopt(fd, SOL_PACKET, PACKET_IGNORE_OUTGOING, &one, sizeof(one));
    if (setsockopt(fd, SOL_PACKET, PACKET_VNET_HDR, &one, sizeof(one)) != 0) {
        perror("setsockopt(PACKET_VNET_HDR)");
        close(fd);
        return -1;
    }

    int buf_sz = 16 * 1024 * 1024;
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf_sz, sizeof(buf_sz));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf_sz, sizeof(buf_sz));

    struct sockaddr_ll sll;
    memset(&sll, 0, sizeof(sll));
    sll.sll_family = AF_PACKET;
    sll.sll_protocol = htons(ETH_P_ALL);
    sll.sll_ifindex = ifidx;
    if (bind(fd, (const struct sockaddr *)&sll, sizeof(sll)) != 0) {
        perror("bind(AF_PACKET)");
        close(fd);
        return -1;
    }

    struct packet_mreq mreq;
    memset(&mreq, 0, sizeof(mreq));
    mreq.mr_ifindex = ifidx;
    mreq.mr_type = PACKET_MR_PROMISC;
    if (setsockopt(fd, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) != 0) {
        perror("PACKET_ADD_MEMBERSHIP(PROMISC)");
        close(fd);
        return -1;
    }

    *out_ifindex = ifidx;
    return fd;
}

static void init_port_q(switch_port_q_t *q, double rate_mbps, uint64_t mmu_kb, uint64_t ecn_kb, uint64_t delay_us, uint64_t now)
{
    memset(q, 0, sizeof(*q));
    q->ring = (switch_pkt_t *)calloc(RING_CAP, sizeof(switch_pkt_t));
    if (!q->ring) {
        perror("calloc(RING_CAP)");
        exit(1);
    }
    q->mmu_cap_bytes = mmu_kb * 1024ULL;
    q->ecn_min_bytes = ecn_kb * 1024ULL;
    q->delay_ns = delay_us * 1000ULL;
    q->rate_Bps = rate_mbps * 125000.0;
    q->burst_cap_bytes = 65536.0;
    q->tokens_bytes = q->burst_cap_bytes;
    q->last_ns = now;
}

static bool rx_into_queue(int in_fd, switch_port_q_t *q, uint64_t now)
{
    for (int batch = 0; batch < 64; batch++) {
        _Alignas(8) uint8_t tmp[MAX_FRAME_BUF];
        struct sockaddr_ll sll = {0};
        struct iovec iov = {.iov_base = tmp, .iov_len = sizeof(tmp)};
        struct msghdr msg = {.msg_name = &sll, .msg_namelen = sizeof(sll), .msg_iov = &iov, .msg_iovlen = 1};
        ssize_t n = recvmsg(in_fd, &msg, MSG_DONTWAIT | MSG_TRUNC);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                break;
            q->rx_errors++;
            perror("switch recvmsg");
            return false;
        }
        if (sll.sll_pkttype == PACKET_OUTGOING)
            continue;
        if ((msg.msg_flags & MSG_TRUNC) || (size_t)n > sizeof(tmp)) {
            q->rx_truncated++;
            fprintf(stderr, "switch: truncated frame (%zd bytes); disable link offloads before benchmarking\n", n);
            return false;
        }
        if ((size_t)n < sizeof(struct virtio_net_hdr) + sizeof(struct ethhdr)) {
            q->rx_errors++;
            fprintf(stderr, "switch: incomplete virtio/Ethernet header\n");
            return false;
        }
        struct virtio_net_hdr vnet;
        memcpy(&vnet, tmp, sizeof(vnet));
        if (vnet.gso_type != VIRTIO_NET_HDR_GSO_NONE) {
            q->rx_gso++;
            fprintf(stderr, "switch: GSO aggregate type %u is not a wire packet; disable link offloads\n", vnet.gso_type);
            return false;
        }

        uint32_t wire_len = (uint32_t)((size_t)n - sizeof(struct virtio_net_hdr));
        bool is_ele = is_elephant_frame(tmp + sizeof(struct virtio_net_hdr), wire_len);
        q->rx_pkts++;

        if (q->cnt >= RING_CAP || q->q_bytes + wire_len > q->mmu_cap_bytes) {
            q->tail_drops++;
            if (!is_ele)
                q->app_tail_drops++;
            continue;
        }

        uint32_t tail = (q->head + q->cnt) & (RING_CAP - 1u);
        switch_pkt_t *slot = &q->ring[tail];
        slot->arrival_ns = now;
        slot->len = (uint32_t)n;
        slot->wire_len = wire_len;
        slot->is_elephant = is_ele;
        memcpy(slot->data, tmp, (size_t)n);

        q->cnt++;
        q->q_bytes += wire_len;
        if (q->q_bytes > q->peak_q_bytes)
            q->peak_q_bytes = q->q_bytes;

        if (q->ecn_min_bytes > 0u && q->q_bytes >= q->ecn_min_bytes) {
            uint64_t pre_ecn = q->ecn_marks;
            maybe_mark_ipv4_ecn_ce(slot->data + sizeof(struct virtio_net_hdr), wire_len, &q->ecn_marks);
            if (!is_ele && q->ecn_marks > pre_ecn)
                q->app_ecn_marks++;
        }
    }
    return true;
}

static bool tx_from_queue(int out_fd, int out_ifidx, switch_port_q_t *q, uint64_t now)
{
    if (now > q->last_ns) {
        double dt = (double)(now - q->last_ns) * 1e-9;
        q->last_ns = now;
        q->tokens_bytes += q->rate_Bps * dt;
        if (q->tokens_bytes > q->burst_cap_bytes)
            q->tokens_bytes = q->burst_cap_bytes;
    }

    struct sockaddr_ll dst;
    memset(&dst, 0, sizeof(dst));
    dst.sll_family = AF_PACKET;
    dst.sll_ifindex = out_ifidx;
    dst.sll_protocol = htons(ETH_P_ALL);

    while (q->cnt > 0u) {
        switch_pkt_t *front = &q->ring[q->head];
        if (now < front->arrival_ns + q->delay_ns)
            break;
        if (q->tokens_bytes < (double)front->wire_len)
            break;

        ssize_t s = sendto(out_fd, front->data, front->len, MSG_DONTWAIT, (const struct sockaddr *)&dst, sizeof(dst));
        if (s < 0 && errno == EINTR)
            continue;
        if (s < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == ENOBUFS))
            break;
        if (s != (ssize_t)front->len) {
            q->tx_errors++;
            if (s < 0)
                perror("switch sendto");
            else
                fprintf(stderr, "switch: short frame send (%zd/%u bytes)\n", s, front->len);
            return false;
        }

        q->tokens_bytes -= (double)front->wire_len;
        q->q_bytes = (front->wire_len <= q->q_bytes) ? (q->q_bytes - front->wire_len) : 0u;
        q->head = (q->head + 1u) & (RING_CAP - 1u);
        q->cnt--;
        if (s > 0) {
            q->tx_pkts++;
            q->tx_bytes += front->wire_len;
            if (!front->is_elephant)
                q->app_tx_bytes += front->wire_len;
        }
    }
    return true;
}

static bool sample_socket_drops(int fd, switch_port_q_t *q)
{
    struct tpacket_stats stats;
    socklen_t size = sizeof(stats);
    if (getsockopt(fd, SOL_PACKET, PACKET_STATISTICS, &stats, &size) != 0 || size != sizeof(stats)) {
        q->rx_errors++;
        perror("PACKET_STATISTICS");
        return false;
    }
    /* PACKET_STATISTICS resets on read; accumulate every snapshot. */
    q->socket_drops += stats.tp_drops;
    if (stats.tp_drops) {
        fprintf(stderr, "switch: packet socket dropped %u frames before MMU accounting\n", stats.tp_drops);
        return false;
    }
    return true;
}

static uint64_t invalid_events(const switch_port_q_t *q)
{
    return q->rx_truncated + q->rx_gso + q->rx_errors + q->socket_drops + q->tx_errors;
}

static void write_stats_file(const char *path, const switch_port_q_t *ab, const switch_port_q_t *ba)
{
    if (!path || !*path)
        return;
    static uint64_t stat_seq;
    char tmp_path[512];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    FILE *fp = fopen(tmp_path, "w");
    if (!fp)
        return;
    fprintf(fp,
            "ab_tx_pkts=%" PRIu64 " ab_tx_bytes=%" PRIu64 " ab_ecn_marks=%" PRIu64 " ab_tail_drops=%" PRIu64
            " ab_peak_q_bytes=%" PRIu64 "\n"
            "ba_tx_pkts=%" PRIu64 " ba_tx_bytes=%" PRIu64 " ba_ecn_marks=%" PRIu64 " ba_tail_drops=%" PRIu64
            " ba_peak_q_bytes=%" PRIu64 "\n"
            "tot_tx_pkts=%" PRIu64 " tot_tx_bytes=%" PRIu64 " tot_ecn_marks=%" PRIu64 " tot_tail_drops=%" PRIu64
            " tot_app_tx_bytes=%" PRIu64 " tot_app_ecn_marks=%" PRIu64 " tot_app_tail_drops=%" PRIu64 "\n",
            ab->tx_pkts, ab->tx_bytes, ab->ecn_marks, ab->tail_drops, ab->peak_q_bytes, ba->tx_pkts, ba->tx_bytes, ba->ecn_marks,
            ba->tail_drops, ba->peak_q_bytes, ab->tx_pkts + ba->tx_pkts, ab->tx_bytes + ba->tx_bytes, ab->ecn_marks + ba->ecn_marks,
            ab->tail_drops + ba->tail_drops, ab->app_tx_bytes + ba->app_tx_bytes, ab->app_ecn_marks + ba->app_ecn_marks,
            ab->app_tail_drops + ba->app_tail_drops);
    fprintf(fp,
            "stat_seq=%" PRIu64 " valid=%u ab_rx_truncated=%" PRIu64 " ba_rx_truncated=%" PRIu64 " ab_rx_gso=%" PRIu64
            " ba_rx_gso=%" PRIu64 " ab_rx_errors=%" PRIu64 " ba_rx_errors=%" PRIu64 " ab_socket_drops=%" PRIu64
            " ba_socket_drops=%" PRIu64 " ab_tx_errors=%" PRIu64 " ba_tx_errors=%" PRIu64 " tot_invalid_events=%" PRIu64 "\n",
            ++stat_seq, (unsigned)(invalid_events(ab) == 0u && invalid_events(ba) == 0u), ab->rx_truncated, ba->rx_truncated,
            ab->rx_gso, ba->rx_gso, ab->rx_errors, ba->rx_errors, ab->socket_drops, ba->socket_drops, ab->tx_errors, ba->tx_errors,
            invalid_events(ab) + invalid_events(ba));
    fclose(fp);
    (void)rename(tmp_path, path);
}

static int cmd_switch(int argc, char **argv)
{
    if (argc < 9) {
        fprintf(stderr, "Usage: %s switch <if_a> <if_b> <rate_mbps> <mmu_kb> <ecn_kb> <delay_us> <stats_file>\n", argv[0]);
        return 1;
    }
    const char *if_a = argv[2];
    const char *if_b = argv[3];
    double rate_mbps = atof(argv[4]);
    uint64_t mmu_kb = (uint64_t)strtoull(argv[5], NULL, 10);
    uint64_t ecn_kb = (uint64_t)strtoull(argv[6], NULL, 10);
    uint64_t delay_us = (uint64_t)strtoull(argv[7], NULL, 10);
    const char *stats_path = argv[8];

    if (rate_mbps < 1.0)
        rate_mbps = 100.0;
    if (mmu_kb < 16u)
        mmu_kb = 16u;

    int idx_a = 0, idx_b = 0;
    int fd_a = open_raw_port(if_a, &idx_a);
    int fd_b = open_raw_port(if_b, &idx_b);
    if (fd_a < 0 || fd_b < 0)
        return 1;

    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);
    signal(SIGUSR1, on_sig);

    uint64_t now = mono_ns();
    switch_port_q_t q_ab, q_ba;
    init_port_q(&q_ab, rate_mbps, mmu_kb, ecn_kb, delay_us, now);
    init_port_q(&q_ba, rate_mbps, mmu_kb, ecn_kb, delay_us, now);
    write_stats_file(stats_path, &q_ab, &q_ba);

    uint64_t last_stat_ns = now;
    bool ok = true;
    while (!g_stop) {
        now = mono_ns();
        if (!rx_into_queue(fd_a, &q_ab, now) || !rx_into_queue(fd_b, &q_ba, now) || !tx_from_queue(fd_b, idx_b, &q_ab, now) ||
            !tx_from_queue(fd_a, idx_a, &q_ba, now)) {
            ok = false;
            break;
        }

        if (g_flush || now - last_stat_ns >= 15000000ULL) {
            g_flush = 0;
            if (!sample_socket_drops(fd_a, &q_ab) || !sample_socket_drops(fd_b, &q_ba)) {
                ok = false;
                break;
            }
            write_stats_file(stats_path, &q_ab, &q_ba);
            last_stat_ns = now;
        }

        if (q_ab.cnt == 0u && q_ba.cnt == 0u) {
            struct pollfd pfds[2] = {
                {.fd = fd_a, .events = POLLIN},
                {.fd = fd_b, .events = POLLIN},
            };
            (void)poll(pfds, 2, 1);
        }
    }

    bool input_ok_a = sample_socket_drops(fd_a, &q_ab);
    bool input_ok_b = sample_socket_drops(fd_b, &q_ba);
    ok = ok && input_ok_a && input_ok_b;
    write_stats_file(stats_path, &q_ab, &q_ba);
    free(q_ab.ring);
    free(q_ba.ring);
    close(fd_a);
    close(fd_b);
    return ok ? 0 : 1;
}

static int cmd_elephant_rx(int argc, char **argv)
{
    if (argc < 4)
        return 1;
    const char *bind_ip = argv[2];
    uint16_t port = (uint16_t)atoi(argv[3]);

    int fd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return 1;
    int rcv = 8 * 1024 * 1024;
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcv, sizeof(rcv));

    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_port = htons(port);
    if (inet_pton(AF_INET, bind_ip, &sin.sin_addr) != 1 || bind(fd, (const struct sockaddr *)&sin, sizeof(sin)) != 0) {
        perror("elephant_rx bind");
        close(fd);
        return 1;
    }

    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);
    uint8_t buf[9216];
    struct pollfd pfd = {.fd = fd, .events = POLLIN};
    while (!g_stop) {
        if (poll(&pfd, 1, 50) > 0) {
            while (recv(fd, buf, sizeof(buf), MSG_DONTWAIT) > 0) {
            }
        }
    }
    close(fd);
    return 0;
}

static int cmd_elephant_tx(int argc, char **argv)
{
    if (argc < 8)
        return 1;
    const char *src_ip = argv[2];
    uint16_t src_port = (uint16_t)atoi(argv[3]);
    const char *dst_ip = argv[4];
    uint16_t dst_port = (uint16_t)atoi(argv[5]);
    double rate_mbps = atof(argv[6]);
    size_t pkt_bytes = (size_t)strtoul(argv[7], NULL, 10);
    if (pkt_bytes < 64u)
        pkt_bytes = 64u;
    if (pkt_bytes > 8192u)
        pkt_bytes = 8192u;

    int fd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return 1;
    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    int tos = IPTOS_ECN_ECT0;
    (void)setsockopt(fd, IPPROTO_IP, IP_TOS, &tos, sizeof(tos));

    struct sockaddr_in s_in, d_in;
    memset(&s_in, 0, sizeof(s_in));
    s_in.sin_family = AF_INET;
    s_in.sin_port = htons(src_port);
    if (inet_pton(AF_INET, src_ip, &s_in.sin_addr) != 1 || bind(fd, (const struct sockaddr *)&s_in, sizeof(s_in)) != 0) {
        perror("elephant_tx bind");
        close(fd);
        return 1;
    }

    memset(&d_in, 0, sizeof(d_in));
    d_in.sin_family = AF_INET;
    d_in.sin_port = htons(dst_port);
    if (inet_pton(AF_INET, dst_ip, &d_in.sin_addr) != 1 || connect(fd, (const struct sockaddr *)&d_in, sizeof(d_in)) != 0) {
        perror("elephant_tx connect");
        close(fd);
        return 1;
    }

    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);

    uint8_t payload[8192];
    memset(payload, 0x5au, pkt_bytes);
    double rate_Bps = rate_mbps * 125000.0;
    double tokens = (double)(pkt_bytes * 4u);
    uint64_t last_ns = mono_ns();

    while (!g_stop) {
        uint64_t now = mono_ns();
        if (now > last_ns) {
            tokens += rate_Bps * (double)(now - last_ns) * 1e-9;
            last_ns = now;
            if (tokens > 65536.0)
                tokens = 65536.0;
        }
        if (tokens >= (double)pkt_bytes) {
            (void)send(fd, payload, pkt_bytes, MSG_DONTWAIT);
            tokens -= (double)pkt_bytes;
        } else {
            struct timespec req = {.tv_sec = 0, .tv_nsec = 50000L};
            nanosleep(&req, NULL);
        }
    }
    close(fd);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <switch|elephant-rx|elephant-tx> ...\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "switch") == 0)
        return cmd_switch(argc, argv);
    if (strcmp(argv[1], "elephant-rx") == 0)
        return cmd_elephant_rx(argc, argv);
    if (strcmp(argv[1], "elephant-tx") == 0)
        return cmd_elephant_tx(argc, argv);
    fprintf(stderr, "Unknown subcommand: %s\n", argv[1]);
    return 1;
}
