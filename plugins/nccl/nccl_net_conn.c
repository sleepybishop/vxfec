#define _GNU_SOURCE
#include "nccl_net_internal.h"

typedef struct vxfec_pending_conn {
    int dev_idx;
    void **send_comm_slot;
    uint32_t ip_be;
    uint16_t ctrl_port_be;
    uint32_t cookie;
    uint64_t created_ns;
    vxfec_send_comm_t *sc;
    struct vxfec_pending_conn *next;
} vxfec_pending_conn_t;

atomic_uint vxf_retired_lc_cnt = 0u;
atomic_bool vxf_retired_lc_reap_needed = false;

static _Atomic uint64_t g_conn_id_seq = 1u;
static pthread_mutex_t g_conn_mtx = PTHREAD_MUTEX_INITIALIZER;
static vxfec_pending_conn_t *g_pending_conns = NULL;
static pthread_mutex_t g_retired_lc_mtx = PTHREAD_MUTEX_INITIALIZER;
static vxfec_listen_comm_t *g_retired_listeners = NULL;

static pthread_mutex_t g_rx_bg_mtx = PTHREAD_MUTEX_INITIALIZER;
static vxfec_recv_comm_t *g_rx_bg_head = NULL;
static vxfec_send_comm_t *g_tx_bg_head = NULL;
static int g_rx_epoll_fd = -1;
static int g_rx_wake_fds[2] = {-1, -1};
static pthread_t g_rx_bg_thr;
static bool g_rx_bg_started = false;
static atomic_bool g_rx_bg_stop = false;
#if VXFEC_ENABLE_SIM
static atomic_bool g_rx_delay_changed = false;
#endif

uint64_t vxf_next_unique_u64(void)
{
    uint64_t rnd = 0u;
    (void)getrandom(&rnd, sizeof(rnd), GRND_NONBLOCK);
    uint64_t seq = atomic_fetch_add_explicit(&g_conn_id_seq, 1u, memory_order_relaxed);
    uint64_t z = seq * 0x9E3779B97F4A7C15ULL ^ ((uint64_t)(uint32_t)getpid() << 32) ^ mono_ns();
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= (z >> 31) ^ rnd;
    return z ? z : 1u;
}

uint32_t vxf_next_unique_u32(void)
{
    uint64_t v = vxf_next_unique_u64();
    uint32_t u = (uint32_t)(v ^ (v >> 32));
    return u ? u : 1u;
}

void vxf_accept_node_release(vxfec_accept_node_t *node)
{
    if (!node)
        return;
    if (atomic_fetch_sub_explicit(&node->refcnt, 1u, memory_order_acq_rel) == 1u)
        free(node);
}

static int open_nonblock_udp(uint32_t ip_be, uint16_t *out_port_be)
{
    int fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return -1;

    int buf_sz = 32 * 1024 * 1024;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf_sz, sizeof(buf_sz)) != 0) {
        buf_sz = 16 * 1024 * 1024;
        (void)setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf_sz, sizeof(buf_sz));
    }
    int got_rcv = 0;
    socklen_t got_len = sizeof(got_rcv);
    if (getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &got_rcv, &got_len) == 0 && got_rcv > 0)
        atomic_store_explicit(&vxf_granted_rcvbuf, (uint32_t)got_rcv, memory_order_relaxed);
    buf_sz = 32 * 1024 * 1024;
    if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf_sz, sizeof(buf_sz)) != 0) {
        buf_sz = 16 * 1024 * 1024;
        (void)setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf_sz, sizeof(buf_sz));
    }

#ifdef SO_BUSY_POLL
    uint32_t bpoll = atomic_load_explicit(&vxf_cfg_busy_poll_us, memory_order_relaxed);
    if (bpoll > 0u) {
        int bp_val = (int)bpoll;
        (void)setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &bp_val, sizeof(bp_val));
    }
#endif

    int tos_ect0 = IPTOS_ECN_ECT0;
    int recv_tos = 1;
    (void)setsockopt(fd, IPPROTO_IP, IP_TOS, &tos_ect0, sizeof(tos_ect0));
#ifdef IP_RECVTOS
    (void)setsockopt(fd, IPPROTO_IP, IP_RECVTOS, &recv_tos, sizeof(recv_tos));
#else
    (void)recv_tos;
#endif

    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = ip_be;
    sin.sin_port = 0;
    if (bind(fd, (struct sockaddr *)&sin, sizeof(sin)) != 0) {
        close(fd);
        return -1;
    }
    if (out_port_be) {
        socklen_t slen = sizeof(sin);
        if (getsockname(fd, (struct sockaddr *)&sin, &slen) != 0) {
            close(fd);
            return -1;
        }
        *out_port_be = sin.sin_port;
    }
    return fd;
}

void vxf_rx_bg_wake(void)
{
    uint8_t b = 1u;
    if (g_rx_wake_fds[1] >= 0)
        (void)write(g_rx_wake_fds[1], &b, 1u);
}

#if VXFEC_ENABLE_SIM
void vxf_rx_bg_schedule_delay(void)
{
    if (!atomic_exchange_explicit(&g_rx_delay_changed, true, memory_order_acq_rel))
        vxf_rx_bg_wake();
}
#endif

static bool ensure_bg_epoll_started_locked(void)
{
    if (g_rx_bg_started)
        return true;
    atomic_store_explicit(&g_rx_bg_stop, false, memory_order_relaxed);
    g_rx_epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (g_rx_epoll_fd >= 0 && pipe2(g_rx_wake_fds, O_NONBLOCK | O_CLOEXEC) == 0) {
        struct epoll_event wev = {.events = EPOLLIN, .data.ptr = NULL};
        if (epoll_ctl(g_rx_epoll_fd, EPOLL_CTL_ADD, g_rx_wake_fds[0], &wev) == 0)
            return true;
        close(g_rx_wake_fds[0]);
        close(g_rx_wake_fds[1]);
        g_rx_wake_fds[0] = g_rx_wake_fds[1] = -1;
    }
    if (g_rx_epoll_fd >= 0) {
        close(g_rx_epoll_fd);
        g_rx_epoll_fd = -1;
    }
    return false;
}

static void *rx_bg_main(void *arg)
{
    (void)arg;
    struct epoll_event evs[64];
#if VXFEC_ENABLE_SIM
    uint64_t next_header_ns = 0;
#endif
    while (!atomic_load_explicit(&g_rx_bg_stop, memory_order_acquire)) {
        pthread_mutex_lock(&g_rx_bg_mtx);
        bool has_retired = atomic_load_explicit(&vxf_retired_lc_cnt, memory_order_acquire) > 0u;
        int timeout_ms = (g_tx_bg_head != NULL) ? 2 : (has_retired ? 100 : -1);
#if VXFEC_ENABLE_SIM
        /* deliver delayed headers even after receiver polling stops */
        if ((atomic_load_explicit(&g_rx_delay_changed, memory_order_relaxed) &&
             atomic_exchange_explicit(&g_rx_delay_changed, false, memory_order_acq_rel)) ||
            (next_header_ns && mono_ns() >= next_header_ns)) {
            next_header_ns = 0;
            for (vxfec_recv_comm_t *rc = g_rx_bg_head; rc; rc = rc->bg_next) {
                pthread_mutex_lock(&rc->mtx);
                uint64_t due = vxf_rx_deliver_due_headers(rc, mono_ns());
                pthread_mutex_unlock(&rc->mtx);
                if (due && (!next_header_ns || due < next_header_ns))
                    next_header_ns = due;
            }
        }
        if (next_header_ns) {
            uint64_t now = mono_ns();
            uint64_t wait_ms = next_header_ns > now ? 1u + (next_header_ns - now - 1u) / 1000000u : 0u;
            int delay_timeout_ms = wait_ms > INT_MAX ? INT_MAX : (int)wait_ms;
            if (timeout_ms < 0 || delay_timeout_ms < timeout_ms)
                timeout_ms = delay_timeout_ms;
        }
#endif
        pthread_mutex_unlock(&g_rx_bg_mtx);

        int nfds = epoll_wait(g_rx_epoll_fd, evs, 64, timeout_ms);
        if (nfds < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        if (atomic_load_explicit(&g_rx_bg_stop, memory_order_acquire))
            break;

        pthread_mutex_lock(&g_rx_bg_mtx);
        for (int i = 0; i < nfds; i++) {
            if (evs[i].data.ptr == NULL) {
                uint8_t drain[32];
                while (read(g_rx_wake_fds[0], drain, sizeof(drain)) > 0) {
                }
                continue;
            }
            vxfec_recv_comm_t *target = (vxfec_recv_comm_t *)evs[i].data.ptr;
            bool still_live = false;
            for (vxfec_recv_comm_t *it = g_rx_bg_head; it != NULL; it = it->bg_next) {
                if (it == target) {
                    still_live = true;
                    break;
                }
            }
            if (!still_live)
                continue;

            pthread_mutex_lock(&target->mtx);
            vxf_rx_drain_ctrl_wire(target, mono_ns());
            vxf_rx_bg_peek_data_ports(target);
            pthread_mutex_unlock(&target->mtx);
        }

        for (vxfec_send_comm_t **pp = &g_tx_bg_head; *pp != NULL;) {
            vxfec_send_comm_t *sc = *pp;
            pthread_mutex_lock(&sc->mtx);
            vxf_progress_send_comm(sc);
            bool done_aborts = (sc->unacked_aborts == 0u);
            if (done_aborts)
                sc->tx_bg_registered = false;
            pthread_mutex_unlock(&sc->mtx);
            if (done_aborts) {
                if (g_rx_epoll_fd >= 0 && sc->ctrl_fd >= 0)
                    (void)epoll_ctl(g_rx_epoll_fd, EPOLL_CTL_DEL, sc->ctrl_fd, NULL);
                *pp = sc->tx_bg_next;
                sc->tx_bg_next = NULL;
            } else {
                pp = &sc->tx_bg_next;
            }
        }
        pthread_mutex_unlock(&g_rx_bg_mtx);

        if (atomic_exchange_explicit(&vxf_retired_lc_reap_needed, false, memory_order_acq_rel) ||
            atomic_load_explicit(&vxf_retired_lc_cnt, memory_order_acquire) > 0u)
            vxf_reap_retired_listeners(false);
    }
    return NULL;
}

void vxf_rx_bg_register(vxfec_recv_comm_t *rc)
{
    if (!rc || !rc->mtx_ready)
        return;
    pthread_mutex_lock(&g_rx_bg_mtx);
    if (!g_rx_bg_started && ensure_bg_epoll_started_locked()) {
        if (pthread_create(&g_rx_bg_thr, NULL, rx_bg_main, NULL) == 0) {
            g_rx_bg_started = true;
        } else {
            close(g_rx_wake_fds[0]);
            close(g_rx_wake_fds[1]);
            g_rx_wake_fds[0] = g_rx_wake_fds[1] = -1;
            close(g_rx_epoll_fd);
            g_rx_epoll_fd = -1;
        }
    }
    if (g_rx_bg_started && !rc->bg_registered) {
        if (rc->ctrl_fd >= 0) {
            struct epoll_event cev = {.events = EPOLLIN, .data.ptr = rc};
            (void)epoll_ctl(g_rx_epoll_fd, EPOLL_CTL_ADD, rc->ctrl_fd, &cev);
        }
        for (uint32_t p = 0; p < rc->num_ports; p++) {
            if (rc->fds[p] >= 0) {
                struct epoll_event dev = {.events = EPOLLIN | EPOLLET, .data.ptr = rc};
                (void)epoll_ctl(g_rx_epoll_fd, EPOLL_CTL_ADD, rc->fds[p], &dev);
            }
        }
        rc->bg_next = g_rx_bg_head;
        g_rx_bg_head = rc;
        rc->bg_registered = true;
    }
    pthread_mutex_unlock(&g_rx_bg_mtx);
}

void vxf_tx_bg_register(vxfec_send_comm_t *sc)
{
    if (!sc || !sc->mtx_ready)
        return;
    pthread_mutex_lock(&g_rx_bg_mtx);
    if (!g_rx_bg_started && ensure_bg_epoll_started_locked()) {
        if (pthread_create(&g_rx_bg_thr, NULL, rx_bg_main, NULL) == 0) {
            g_rx_bg_started = true;
        } else {
            close(g_rx_wake_fds[0]);
            close(g_rx_wake_fds[1]);
            g_rx_wake_fds[0] = g_rx_wake_fds[1] = -1;
            close(g_rx_epoll_fd);
            g_rx_epoll_fd = -1;
        }
    }
    if (g_rx_bg_started) {
        pthread_mutex_lock(&sc->mtx);
        bool should_add = (sc->unacked_aborts > 0u && !sc->tx_bg_registered);
        if (should_add)
            sc->tx_bg_registered = true;
        pthread_mutex_unlock(&sc->mtx);
        if (should_add) {
            if (sc->ctrl_fd >= 0) {
                struct epoll_event cev = {.events = EPOLLIN, .data.ptr = sc};
                (void)epoll_ctl(g_rx_epoll_fd, EPOLL_CTL_ADD, sc->ctrl_fd, &cev);
            }
            sc->tx_bg_next = g_tx_bg_head;
            g_tx_bg_head = sc;
            vxf_rx_bg_wake();
        }
    }
    pthread_mutex_unlock(&g_rx_bg_mtx);
}

void vxf_tx_bg_unregister(vxfec_send_comm_t *sc)
{
    if (!sc)
        return;
    pthread_mutex_lock(&g_rx_bg_mtx);
    bool was_reg = false;
    for (vxfec_send_comm_t **pp = &g_tx_bg_head; *pp != NULL; pp = &(*pp)->tx_bg_next) {
        if (*pp == sc) {
            *pp = sc->tx_bg_next;
            sc->tx_bg_next = NULL;
            was_reg = true;
            break;
        }
    }
    if (was_reg && g_rx_epoll_fd >= 0 && sc->ctrl_fd >= 0)
        (void)epoll_ctl(g_rx_epoll_fd, EPOLL_CTL_DEL, sc->ctrl_fd, NULL);
    if (sc->mtx_ready) {
        pthread_mutex_lock(&sc->mtx);
        sc->tx_bg_registered = false;
        pthread_mutex_unlock(&sc->mtx);
    }
    pthread_mutex_unlock(&g_rx_bg_mtx);
}

void vxf_rx_bg_unregister(vxfec_recv_comm_t *rc)
{
    if (!rc)
        return;
    pthread_mutex_lock(&g_rx_bg_mtx);
    if (rc->bg_registered) {
        if (g_rx_epoll_fd >= 0) {
            if (rc->ctrl_fd >= 0)
                (void)epoll_ctl(g_rx_epoll_fd, EPOLL_CTL_DEL, rc->ctrl_fd, NULL);
            for (uint32_t p = 0; p < rc->num_ports; p++) {
                if (rc->fds[p] >= 0)
                    (void)epoll_ctl(g_rx_epoll_fd, EPOLL_CTL_DEL, rc->fds[p], NULL);
            }
        }
        for (vxfec_recv_comm_t **pp = &g_rx_bg_head; *pp != NULL; pp = &(*pp)->bg_next) {
            if (*pp == rc) {
                *pp = rc->bg_next;
                break;
            }
        }
        rc->bg_next = NULL;
        rc->bg_registered = false;
    }
    pthread_mutex_unlock(&g_rx_bg_mtx);
}

static void rx_bg_shutdown(void)
{
    pthread_t thr;
    bool need_join = false;
    pthread_mutex_lock(&g_rx_bg_mtx);
    if (g_rx_bg_started) {
        atomic_store_explicit(&g_rx_bg_stop, true, memory_order_release);
        vxf_rx_bg_wake();
        thr = g_rx_bg_thr;
        need_join = true;
        g_rx_bg_started = false;
    }
    pthread_mutex_unlock(&g_rx_bg_mtx);

    if (need_join)
        pthread_join(thr, NULL);

    pthread_mutex_lock(&g_rx_bg_mtx);
    if (g_rx_wake_fds[0] >= 0)
        close(g_rx_wake_fds[0]);
    if (g_rx_wake_fds[1] >= 0)
        close(g_rx_wake_fds[1]);
    g_rx_wake_fds[0] = g_rx_wake_fds[1] = -1;
    if (g_rx_epoll_fd >= 0)
        close(g_rx_epoll_fd);
    g_rx_epoll_fd = -1;
    g_rx_bg_head = NULL;
    g_tx_bg_head = NULL;
    pthread_mutex_unlock(&g_rx_bg_mtx);
}

static vxfec_recv_comm_t *alloc_recv_comm_for_hello(const vxfec_listen_comm_t *lc, const vxfec_hello_payload_t *hello,
                                                    const struct sockaddr_in *peer, vxfec_ibv_wire_info_t *out_ibv_info,
                                                    bool *out_has_ibv)
{
    if (out_has_ibv)
        *out_has_ibv = false;
    vxfec_recv_comm_t *rc = (vxfec_recv_comm_t *)calloc(1, sizeof(*rc));
    if (!rc)
        return NULL;
    if (pthread_mutex_init(&rc->mtx, NULL) != 0) {
        free(rc);
        return NULL;
    }
    rc->mtx_ready = true;
    rc->dev_idx = lc->dev_idx;
    rc->ctrl_fd = -1;
    for (uint32_t i = 0; i < VXFEC_MAX_PORTS; i++)
        rc->fds[i] = -1;
    for (uint32_t i = 0; i < VXFEC_RX_HASH_SLOTS; i++)
        rc->msg_seq_hash[i] = -1;
    rc->num_ports = lc->num_ports;
    rc->cookie = lc->cookie;

    rc->ctrl_fd = open_nonblock_udp(lc->ip_be, &rc->ctrl_port_be);
    if (rc->ctrl_fd < 0) {
        (void)vxf_close_recv(rc);
        return NULL;
    }
    if (hello->tx_ctrl_port_be != 0u) {
        rc->peer_ctrl.sin_family = AF_INET;
        rc->peer_ctrl.sin_addr = peer->sin_addr;
        rc->peer_ctrl.sin_port = hello->tx_ctrl_port_be;
        if (connect(rc->ctrl_fd, (const struct sockaddr *)&rc->peer_ctrl, sizeof(rc->peer_ctrl)) == 0)
            rc->ctrl_connected = true;
    }

    if (hello->has_ibv && lc->dev_idx >= 0 && (uint32_t)lc->dev_idx < vxf_num_devs && vxf_devs[lc->dev_idx].ibv_ref &&
        out_ibv_info) {
        vxfec_ibv_dev_ref_t *iref = vxf_devs[lc->dev_idx].ibv_ref;
        uint32_t nqps = hello->ibv_info.num_qps ? hello->ibv_info.num_qps : 8u;
        uint32_t max_pkt = (uint32_t)sizeof(vxfec_wire_hdr_ext_t) + vxf_devs[lc->dev_idx].mtu;
        rc->ibv_ep = vxfec_ibv_ep_create_ex(iref->ctx, nqps, max_pkt, false, true, out_ibv_info);
        if (rc->ibv_ep) {
            if (vxfec_ibv_ep_connect(rc->ibv_ep, &hello->ibv_info, IPTOS_ECN_ECT0)) {
                rc->ibv_ref = iref;
                vxf_ibv_ref_retain(iref);
                if (out_has_ibv)
                    *out_has_ibv = true;
            } else {
                VXF_LOG_WARN("vxfec: failed to connect RX verbs QPs on dev %d; falling back to UDP", lc->dev_idx);
                if (!vxfec_ibv_ep_destroy(rc->ibv_ep))
                    atomic_fetch_add_explicit(&vxf_leaked_ibv_eps, 1u, memory_order_relaxed);
                rc->ibv_ep = NULL;
            }
        } else {
            VXF_LOG_WARN("vxfec: failed to create RX verbs endpoint on dev %d; falling back to UDP", lc->dev_idx);
        }
    }

    /* skip opening 16 unused UDP data sockets on pure RDMA comms unless peer offered UDP fallback ports */
    bool need_udp_data = (rc->ibv_ep == NULL);
    if (!need_udp_data) {
        for (uint32_t i = 0; i < lc->num_ports; i++) {
            if (hello->tx_ports_be[i] != 0u) {
                need_udp_data = true;
                break;
            }
        }
    }
    if (need_udp_data) {
        for (uint32_t i = 0; i < lc->num_ports; i++) {
            rc->fds[i] = open_nonblock_udp(lc->ip_be, &rc->rx_ports_be[i]);
            if (rc->fds[i] < 0) {
                (void)vxf_close_recv(rc);
                return NULL;
            }
            rc->peer_data[i].sin_family = AF_INET;
            rc->peer_data[i].sin_addr = peer->sin_addr;
            rc->peer_data[i].sin_port = hello->tx_ports_be[i];
            if (hello->tx_ports_be[i] != 0u)
                (void)connect(rc->fds[i], (const struct sockaddr *)&rc->peer_data[i], sizeof(rc->peer_data[i]));
        }
    }

    return rc;
}

static void lc_reap_closed_locked(vxfec_listen_comm_t *lc)
{
    bool retired = atomic_load_explicit(&lc->closed_by_user, memory_order_acquire);
    bool retired_expired = false;
    if (retired && lc->retired_ns > 0u) {
        uint64_t now_ns = mono_ns();
        retired_expired = (now_ns > lc->retired_ns && (now_ns - lc->retired_ns) > VXFEC_CONN_TIMEOUT_NS);
    }

    uint32_t closed_cnt = 0u;
    if (!retired) {
        for (vxfec_accept_node_t *it = lc->head; it != NULL; it = it->next) {
            if (it->rc == NULL && atomic_load_explicit(&it->rc_closed, memory_order_acquire))
                closed_cnt++;
        }
    }

    vxfec_accept_node_t *prev = NULL;
    vxfec_accept_node_t **pp = &lc->head;
    while (*pp) {
        vxfec_accept_node_t *it = *pp;
        bool is_closed = (it->rc == NULL) && atomic_load_explicit(&it->rc_closed, memory_order_acquire);
        bool is_conf = (it->rc == NULL) && atomic_load_explicit(&it->hs_confirmed, memory_order_acquire);
        bool drop = retired ? (is_closed || is_conf || (it->rc == NULL && retired_expired))
                            : (is_closed && closed_cnt > VXFEC_LISTEN_TOMBSTONES);
        if (drop) {
            if (!retired && closed_cnt > 0u)
                closed_cnt--;
            *pp = it->next;
            vxf_accept_node_release(it);
        } else {
            prev = it;
            pp = &it->next;
        }
    }
    lc->tail = prev;
}

static void lc_drain_ctrl_locked(vxfec_listen_comm_t *lc)
{
    lc_reap_closed_locked(lc);
    for (;;) {
        struct sockaddr_in peer;
        socklen_t slen = sizeof(peer);
        vxfec_hello_payload_t hello;
        memset(&hello, 0, sizeof(hello));
        ssize_t n = recvfrom(lc->ctrl_fd, &hello, sizeof(hello), MSG_DONTWAIT, (struct sockaddr *)&peer, &slen);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            break;
        bool legacy_byte5_hello = false;
        if ((size_t)n == sizeof(hello) && hello.has_ibv != 0u) {
            if (hello.features == 0u && (hello.has_ibv & VXFEC_HELLO_IBV) != 0u) {
                legacy_byte5_hello = true;
                hello.features = hello.has_ibv & VXFEC_LOCAL_FEATURES;
            }
            hello.has_ibv = 1u;
        } else if ((size_t)n == VXFEC_HELLO_BASE_BYTES) {
            if ((hello.has_ibv & VXFEC_HELLO_IBV) != 0u)
                continue;
            if (hello.has_ibv != 0u) {
                if (hello.features != 0u)
                    continue;
                legacy_byte5_hello = true;
                hello.features = hello.has_ibv & VXFEC_LOCAL_FEATURES;
                hello.has_ibv = 0u;
            }
        } else {
            continue;
        }
        if (hello.magic != VXFEC_NET_MAGIC || hello.pkt_type != PKT_HELLO || hello.cookie != lc->cookie ||
            hello.num_ports != lc->num_ports || hello.conn_id == 0u)
            continue;

        vxfec_accept_node_t *existing = NULL;
        for (vxfec_accept_node_t *it = lc->head; it != NULL; it = it->next) {
            if (it->peer_ip_be == peer.sin_addr.s_addr && it->peer_port_be == peer.sin_port && it->hello.conn_id == hello.conn_id &&
                it->hello.tx_ctrl_port_be == hello.tx_ctrl_port_be &&
                memcmp(it->hello.tx_ports_be, hello.tx_ports_be, (size_t)lc->num_ports * sizeof(uint16_t)) == 0) {
                existing = it;
                break;
            }
        }
        if (existing) {
            if (!atomic_load_explicit(&existing->rc_closed, memory_order_acquire) && !vxf_sim_should_drop_acks())
                (void)sendto(lc->ctrl_fd, &existing->ack, existing->ack_len, MSG_DONTWAIT, (struct sockaddr *)&peer, slen);
            continue;
        }

        if (atomic_load_explicit(&lc->closed_by_user, memory_order_acquire) || lc->backlog_cnt >= VXFEC_MAX_LISTEN_BACKLOG)
            continue;

        vxfec_ibv_wire_info_t ibv_ack_info;
        memset(&ibv_ack_info, 0, sizeof(ibv_ack_info));
        bool has_ibv_ack = false;
        vxfec_recv_comm_t *rc = alloc_recv_comm_for_hello(lc, &hello, &peer, &ibv_ack_info, &has_ibv_ack);
        if (!rc)
            continue;

        vxfec_accept_node_t *node = (vxfec_accept_node_t *)calloc(1, sizeof(*node));
        if (!node) {
            (void)vxf_close_recv(rc);
            continue;
        }
        atomic_init(&node->refcnt, 2u);
        atomic_init(&node->rc_closed, false);
        atomic_init(&node->hs_confirmed, false);
        rc->accept_node = node;
        rc->features = hello.features & VXFEC_LOCAL_FEATURES;
        node->peer_ip_be = peer.sin_addr.s_addr;
        node->peer_port_be = peer.sin_port;
        node->hello = hello;
        node->rc = rc;
        node->ack.magic = VXFEC_NET_MAGIC;
        node->ack.pkt_type = PKT_HELLO_ACK;
        node->ack.features = rc->features;
        node->ack.num_ports = (uint16_t)lc->num_ports;
        node->ack.cookie = lc->cookie;
        node->ack.rx_ctrl_port_be = rc->ctrl_port_be;
        node->ack.conn_id = hello.conn_id;
        for (uint32_t i = 0; i < lc->num_ports; i++)
            node->ack.rx_ports_be[i] = rc->rx_ports_be[i];
        if (has_ibv_ack) {
            node->ack.has_ibv = (!legacy_byte5_hello && hello.features == 0u) ? 1u : (uint8_t)(VXFEC_HELLO_IBV | rc->features);
            node->ack.ibv_info = ibv_ack_info;
            node->ack_len = sizeof(node->ack);
        } else {
            node->ack.has_ibv = legacy_byte5_hello ? rc->features : 0u;
            node->ack_len = VXFEC_HELLO_ACK_BASE_BYTES;
        }

        if (lc->tail)
            lc->tail->next = node;
        else
            lc->head = node;
        lc->tail = node;
        lc->backlog_cnt++;

        if (!vxf_sim_should_drop_acks())
            (void)sendto(lc->ctrl_fd, &node->ack, node->ack_len, MSG_DONTWAIT, (struct sockaddr *)&peer, slen);
    }
}

static void *lc_bg_main(void *arg)
{
    vxfec_listen_comm_t *lc = (vxfec_listen_comm_t *)arg;
    struct pollfd pfds[2] = {
        {.fd = lc->ctrl_fd, .events = POLLIN},
        {.fd = lc->wake_fds[0], .events = POLLIN},
    };
    while (!atomic_load_explicit(&lc->stop, memory_order_acquire)) {
        int r = poll(pfds, 2, -1);
        if (r <= 0) {
            if (r < 0 && errno == EINTR)
                continue;
            break;
        }
        if (pfds[1].revents & POLLIN) {
            uint8_t drain[16];
            while (read(lc->wake_fds[0], drain, sizeof(drain)) > 0) {
            }
        }
        if (atomic_load_explicit(&lc->stop, memory_order_acquire))
            break;
        if (pfds[0].revents & POLLIN) {
            pthread_mutex_lock(&lc->mtx);
            lc_drain_ctrl_locked(lc);
            pthread_mutex_unlock(&lc->mtx);
        }
    }
    return NULL;
}

static void destroy_listen_comm_now(vxfec_listen_comm_t *lc)
{
    if (!lc)
        return;
    if (lc->bg_started) {
        atomic_store_explicit(&lc->stop, true, memory_order_release);
        uint8_t b = 1u;
        if (lc->wake_fds[1] >= 0)
            (void)write(lc->wake_fds[1], &b, 1u);
        pthread_join(lc->bg_thr, NULL);
        lc->bg_started = false;
    }
    vxfec_accept_node_t *it = lc->head;
    lc->head = NULL;
    lc->tail = NULL;
    while (it) {
        vxfec_accept_node_t *nxt = it->next;
        if (it->rc) {
            (void)vxf_close_recv(it->rc);
            it->rc = NULL;
        }
        vxf_accept_node_release(it);
        it = nxt;
    }
    if (lc->ctrl_fd >= 0)
        close(lc->ctrl_fd);
    if (lc->wake_fds[0] >= 0)
        close(lc->wake_fds[0]);
    if (lc->wake_fds[1] >= 0)
        close(lc->wake_fds[1]);
    if (lc->mtx_ready)
        pthread_mutex_destroy(&lc->mtx);
    free(lc);
}

void vxf_reap_retired_listeners(bool force_all)
{
    vxfec_listen_comm_t *to_free = NULL;
    pthread_mutex_lock(&g_retired_lc_mtx);
    vxfec_listen_comm_t **pp = &g_retired_listeners;
    while (*pp) {
        vxfec_listen_comm_t *lc = *pp;
        bool can_reap = force_all;
        if (!can_reap) {
            pthread_mutex_lock(&lc->mtx);
            lc_reap_closed_locked(lc);
            can_reap = (lc->head == NULL);
            pthread_mutex_unlock(&lc->mtx);
        }
        if (can_reap) {
            *pp = lc->retired_next;
            lc->retired_next = to_free;
            to_free = lc;
            atomic_fetch_sub_explicit(&vxf_retired_lc_cnt, 1u, memory_order_acq_rel);
        } else {
            pp = &lc->retired_next;
        }
    }
    pthread_mutex_unlock(&g_retired_lc_mtx);

    while (to_free) {
        vxfec_listen_comm_t *nxt = to_free->retired_next;
        to_free->retired_next = NULL;
        destroy_listen_comm_now(to_free);
        to_free = nxt;
    }
}

ncclResult_t vxf_listen(int dev, void *opaque_handle, void **listenComm)
{
    if (dev < 0 || (uint32_t)dev >= vxf_num_devs || !opaque_handle || !listenComm)
        return ncclInvalidArgument;
    vxf_reap_retired_listeners(false);

    vxfec_listen_comm_t *lc = (vxfec_listen_comm_t *)calloc(1, sizeof(*lc));
    if (!lc)
        return ncclSystemError;
    lc->dev_idx = dev;
    lc->ctrl_fd = -1;
    lc->wake_fds[0] = -1;
    lc->wake_fds[1] = -1;

    if (pthread_mutex_init(&lc->mtx, NULL) != 0)
        goto fail;
    lc->mtx_ready = true;
    if (pipe2(lc->wake_fds, O_NONBLOCK | O_CLOEXEC) != 0)
        goto fail;

    lc->ip_be = vxf_get_dev_ip_be(dev);
    lc->num_ports = atomic_load_explicit(&vxf_cfg_ports, memory_order_relaxed);
    lc->cookie = vxf_next_unique_u32();

    lc->ctrl_fd = open_nonblock_udp(lc->ip_be, &lc->ctrl_port_be);
    if (lc->ctrl_fd < 0)
        goto fail;

    if (pthread_create(&lc->bg_thr, NULL, lc_bg_main, lc) != 0)
        goto fail;
    lc->bg_started = true;

    vxfec_net_handle_t h;
    memset(&h, 0, sizeof(h));
    h.magic = VXFEC_NET_MAGIC;
    h.ip_be = lc->ip_be;
    h.ctrl_port_be = lc->ctrl_port_be;
    h.num_ports = (uint16_t)lc->num_ports;
    h.cookie = lc->cookie;
    h.dev_idx = (uint8_t)dev;
    h.is_ibv = vxf_devs[dev].is_ibv ? 1u : 0u;
    memcpy(opaque_handle, &h, sizeof(h));

    *listenComm = lc;
    return ncclSuccess;

fail:
    if (lc->ctrl_fd >= 0)
        close(lc->ctrl_fd);
    if (lc->wake_fds[0] >= 0)
        close(lc->wake_fds[0]);
    if (lc->wake_fds[1] >= 0)
        close(lc->wake_fds[1]);
    if (lc->mtx_ready)
        pthread_mutex_destroy(&lc->mtx);
    free(lc);
    return ncclSystemError;
}

static bool sc_ensure_udp_data_fds(vxfec_send_comm_t *sc)
{
    for (uint32_t i = 0; i < sc->num_ports; i++) {
        if (sc->fds[i] >= 0)
            continue;
        sc->fds[i] = open_nonblock_udp(sc->bind_ip_be, &sc->hello.tx_ports_be[i]);
        if (sc->fds[i] < 0)
            return false;
    }
    return true;
}

static bool sc_poll_hello_ack(vxfec_send_comm_t *sc)
{
    for (;;) {
        vxfec_hello_ack_t ack;
        memset(&ack, 0, sizeof(ack));
        ssize_t n = recvfrom(sc->ctrl_fd, &ack, sizeof(ack), MSG_DONTWAIT, NULL, NULL);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            return false;
        bool ack_has_ibv = false;
        uint8_t ack_features = 0u;
        if ((size_t)n == sizeof(ack) && ack.has_ibv != 0u) {
            ack_has_ibv = true;
            if (ack.features != 0u)
                ack_features = ack.features;
            else if ((ack.has_ibv & VXFEC_HELLO_IBV) != 0u)
                ack_features = ack.has_ibv & VXFEC_LOCAL_FEATURES;
        } else if ((size_t)n == VXFEC_HELLO_ACK_BASE_BYTES && (ack.has_ibv & VXFEC_HELLO_IBV) == 0u) {
            ack_has_ibv = false;
            ack_features = ack.features ? ack.features : (ack.has_ibv & VXFEC_LOCAL_FEATURES);
        } else {
            continue;
        }
        if (ack.magic != VXFEC_NET_MAGIC || ack.pkt_type != PKT_HELLO_ACK || ack.cookie != sc->cookie ||
            ack.conn_id != sc->hello.conn_id || ack.num_ports != sc->num_ports)
            continue;

        if (sc->ibv_ep) {
            if (!ack_has_ibv || !vxfec_ibv_ep_connect(sc->ibv_ep, &ack.ibv_info, IPTOS_ECN_ECT0)) {
                VXF_LOG_WARN("vxfec: %s on dev %d; falling back to UDP",
                             !ack_has_ibv ? "peer declined verbs handshake" : "failed to connect TX verbs QPs", sc->dev_idx);
                if (!vxfec_ibv_ep_destroy(sc->ibv_ep))
                    atomic_fetch_add_explicit(&vxf_leaked_ibv_eps, 1u, memory_order_relaxed);
                sc->ibv_ep = NULL;
                if (sc->ibv_ref) {
                    vxf_ibv_ref_release(sc->ibv_ref);
                    sc->ibv_ref = NULL;
                }
            }
        }

        if (sc->ibv_ep == NULL || vxfec_ibv_ep_tx_is_hdr_only(sc->ibv_ep)) {
            if (!sc_ensure_udp_data_fds(sc))
                return false;
            bool all_data_ok = true;
            for (uint32_t i = 0; i < sc->num_ports; i++) {
                sc->peer_data[i].sin_port = ack.rx_ports_be[i];
                if (connect(sc->fds[i], (const struct sockaddr *)&sc->peer_data[i], sizeof(sc->peer_data[i])) != 0)
                    all_data_ok = false;
            }
            sc->data_connected = all_data_ok;
        }

        if (ack.rx_ctrl_port_be != 0u) {
            sc->peer_rx_ctrl.sin_family = AF_INET;
            sc->peer_rx_ctrl.sin_addr = sc->peer_ctrl.sin_addr;
            sc->peer_rx_ctrl.sin_port = ack.rx_ctrl_port_be;
            if (connect(sc->ctrl_fd, (const struct sockaddr *)&sc->peer_rx_ctrl, sizeof(sc->peer_rx_ctrl)) == 0)
                sc->ctrl_connected = true;
        }

        sc->features = ack_features & sc->hello.features & VXFEC_LOCAL_FEATURES;
        sc->connected = true;
        sc->path_sym_limit = vxf_tx_path_sym_limit(sc);
        return true;
    }
}

static vxfec_send_comm_t *alloc_send_comm_for_handle(int dev, const vxfec_net_handle_t *h)
{
    vxfec_send_comm_t *sc = (vxfec_send_comm_t *)calloc(1, sizeof(*sc));
    if (!sc)
        return NULL;
    if (pthread_mutex_init(&sc->mtx, NULL) != 0) {
        free(sc);
        return NULL;
    }
    sc->mtx_ready = true;
    sc->dev_idx = dev;
    sc->ctrl_fd = -1;
    for (uint32_t i = 0; i < VXFEC_MAX_PORTS; i++)
        sc->fds[i] = -1;
    for (uint32_t i = 0; i < VXFEC_TX_HASH_SLOTS; i++)
        sc->tx_seq_hash[i] = -1;

    sc->num_ports = h->num_ports;
    sc->cookie = h->cookie;
    sc->next_msg_seq = 1u;
    sc->slice_min_bytes = vxf_parse_slice_min_env(atomic_load_explicit(&vxf_cfg_slice_min_bytes, memory_order_relaxed));
    vxf_init_send_comm_pacer(sc, mono_ns());

    struct msghdr tmp_mh;
    memset(&tmp_mh, 0, sizeof(tmp_mh));
    tmp_mh.msg_control = sc->tx_cmsg_ce;
    tmp_mh.msg_controllen = sizeof(sc->tx_cmsg_ce);
    struct cmsghdr *cm = CMSG_FIRSTHDR(&tmp_mh);
    if (cm) {
        cm->cmsg_level = IPPROTO_IP;
        cm->cmsg_type = IP_TOS;
        cm->cmsg_len = CMSG_LEN(sizeof(uint8_t));
        *CMSG_DATA(cm) = IPTOS_ECN_CE;
        sc->tx_cmsg_ce_len = CMSG_SPACE(sizeof(uint8_t));
    }

    uint32_t bind_ip = vxf_get_dev_ip_be(dev);
    sc->bind_ip_be = bind_ip;
    sc->hello.magic = VXFEC_NET_MAGIC;
    sc->hello.pkt_type = PKT_HELLO;
    sc->hello.has_ibv = 0u;
    sc->hello.features = VXFEC_LOCAL_FEATURES;
    sc->hello.num_ports = h->num_ports;
    sc->hello.cookie = h->cookie;
    sc->hello.conn_id = vxf_next_unique_u64();

    if (dev >= 0 && (uint32_t)dev < vxf_num_devs && vxf_devs[dev].ibv_ref && h->is_ibv) {
        vxfec_ibv_dev_ref_t *iref = vxf_devs[dev].ibv_ref;
        uint32_t nqps = (h->num_ports <= VXFEC_IBV_MAX_QPS) ? h->num_ports : 8u;
        uint32_t max_pkt = (uint32_t)sizeof(vxfec_wire_hdr_ext_t) + vxf_devs[dev].mtu;
        sc->ibv_ep = vxfec_ibv_ep_create_ex(iref->ctx, nqps, max_pkt, true, false, &sc->hello.ibv_info);
        if (sc->ibv_ep) {
            sc->ibv_ref = iref;
            vxf_ibv_ref_retain(iref);
            sc->hello.has_ibv = (uint8_t)(VXFEC_HELLO_IBV | sc->hello.features);
            if (vxfec_ibv_ep_tx_is_hdr_only(sc->ibv_ep)) {
                VXF_LOG_WARN("vxfec: verbs TX payload slab registration failed on dev %d (check RLIMIT_MEMLOCK); "
                             "zero-copy RDMA active, unregistered/repair sends will fall back to UDP",
                             dev);
            }
        } else {
            VXF_LOG_WARN("vxfec: failed to create TX verbs endpoint on dev %d; falling back to UDP", dev);
        }
    }

    sc->ctrl_fd = open_nonblock_udp(bind_ip, &sc->hello.tx_ctrl_port_be);
    if (sc->ctrl_fd < 0) {
        (void)vxf_close_send(sc);
        return NULL;
    }

    for (uint32_t i = 0; i < sc->num_ports; i++) {
        sc->peer_data[i].sin_family = AF_INET;
        sc->peer_data[i].sin_addr.s_addr = h->ip_be;
        sc->peer_data[i].sin_port = 0;
    }
    /* open UDP data sockets immediately only when IBV is absent or header-only TX fallback requires UDP */
    if (!sc->ibv_ep || vxfec_ibv_ep_tx_is_hdr_only(sc->ibv_ep)) {
        if (!sc_ensure_udp_data_fds(sc)) {
            (void)vxf_close_send(sc);
            return NULL;
        }
    }

    sc->path_sym_limit = vxf_tx_path_sym_limit(sc);
    sc->peer_ctrl.sin_family = AF_INET;
    sc->peer_ctrl.sin_addr.s_addr = h->ip_be;
    sc->peer_ctrl.sin_port = h->ctrl_port_be;
    return sc;
}

static void expire_stale_pending_conns_locked(uint64_t now_ns)
{
    vxfec_pending_conn_t **pp = &g_pending_conns;
    while (*pp) {
        vxfec_pending_conn_t *it = *pp;
        if (now_ns > it->created_ns && (now_ns - it->created_ns) > VXFEC_CONN_TIMEOUT_NS) {
            *pp = it->next;
            if (it->sc)
                (void)vxf_close_send(it->sc);
            free(it);
        } else {
            pp = &it->next;
        }
    }
}

ncclResult_t vxf_connect(int dev, void *opaque_handle, void **sendComm, ncclNetDeviceHandle_v8_t **sendDevComm)
{
    if (dev < 0 || (uint32_t)dev >= vxf_num_devs || !opaque_handle || !sendComm)
        return ncclInvalidArgument;
    if (sendDevComm)
        *sendDevComm = NULL;
    *sendComm = NULL;

    vxfec_net_handle_t h;
    memcpy(&h, opaque_handle, sizeof(h));
    if (h.magic != VXFEC_NET_MAGIC || h.num_ports == 0u || h.num_ports > VXFEC_MAX_PORTS)
        return ncclInvalidArgument;

    uint64_t now_ns = mono_ns();
    pthread_mutex_lock(&g_conn_mtx);
    expire_stale_pending_conns_locked(now_ns);

    vxfec_pending_conn_t *pc = NULL;
    vxfec_pending_conn_t **prev_link = &g_pending_conns;
    for (vxfec_pending_conn_t **pp = &g_pending_conns; *pp != NULL; pp = &(*pp)->next) {
        vxfec_pending_conn_t *it = *pp;
        if (it->dev_idx == dev && it->send_comm_slot == sendComm && it->ip_be == h.ip_be && it->ctrl_port_be == h.ctrl_port_be &&
            it->cookie == h.cookie) {
            pc = it;
            prev_link = pp;
            break;
        }
    }

    if (!pc) {
        vxfec_send_comm_t *sc = alloc_send_comm_for_handle(dev, &h);
        if (!sc) {
            pthread_mutex_unlock(&g_conn_mtx);
            return ncclSystemError;
        }
        pc = (vxfec_pending_conn_t *)calloc(1, sizeof(*pc));
        if (!pc) {
            pthread_mutex_unlock(&g_conn_mtx);
            (void)vxf_close_send(sc);
            return ncclSystemError;
        }
        pc->dev_idx = dev;
        pc->send_comm_slot = sendComm;
        pc->ip_be = h.ip_be;
        pc->ctrl_port_be = h.ctrl_port_be;
        pc->cookie = h.cookie;
        pc->created_ns = now_ns;
        pc->sc = sc;
        pc->next = g_pending_conns;
        g_pending_conns = pc;
        prev_link = &g_pending_conns;
    }

    vxfec_send_comm_t *sc = pc->sc;
    if (!sc_poll_hello_ack(sc)) {
        sc->empty_hello_polls++;
        if (sc->last_hello_ns == 0u || sc->empty_hello_polls >= 4u || (now_ns - sc->last_hello_ns) >= VXFEC_HELLO_RETRY_NS) {
            sc->empty_hello_polls = 0u;
            if (vxf_sim_should_drop_hellos()) {
                sc->last_hello_ns = now_ns;
            } else {
                size_t hello_len = sc->hello.has_ibv ? sizeof(sc->hello) : VXFEC_HELLO_BASE_BYTES;
                ssize_t sent = sendto(sc->ctrl_fd, &sc->hello, hello_len, MSG_DONTWAIT, (struct sockaddr *)&sc->peer_ctrl,
                                      sizeof(sc->peer_ctrl));
                sc->last_hello_ns = (sent == (ssize_t)hello_len) ? now_ns : 0u;
                if (sent == (ssize_t)hello_len) {
                    struct pollfd pfd = {.fd = sc->ctrl_fd, .events = POLLIN};
                    struct timespec pts = {.tv_sec = 0, .tv_nsec = 100000L};
                    (void)ppoll(&pfd, 1, &pts, NULL);
                }
            }
            (void)sc_poll_hello_ack(sc);
        }
    }

    if (sc->connected) {
        *prev_link = pc->next;
        free(pc);
        *sendComm = sc;
    }
    pthread_mutex_unlock(&g_conn_mtx);
    return ncclSuccess;
}

ncclResult_t vxf_accept(void *listenComm, void **recvComm, ncclNetDeviceHandle_v8_t **recvDevComm)
{
    if (!listenComm || !recvComm)
        return ncclInvalidArgument;
    if (recvDevComm)
        *recvDevComm = NULL;
    *recvComm = NULL;

    vxfec_listen_comm_t *lc = (vxfec_listen_comm_t *)listenComm;
    pthread_mutex_lock(&lc->mtx);
    lc_drain_ctrl_locked(lc);
    vxfec_recv_comm_t *rc = NULL;
    for (vxfec_accept_node_t *it = lc->head; it != NULL; it = it->next) {
        if (it->rc) {
            rc = it->rc;
            it->rc = NULL;
            if (lc->backlog_cnt > 0u)
                lc->backlog_cnt--;
            break;
        }
    }
    lc_reap_closed_locked(lc);
    pthread_mutex_unlock(&lc->mtx);

    if (rc)
        vxf_rx_bg_register(rc);
    *recvComm = rc;
    return ncclSuccess;
}

ncclResult_t vxf_close_send(void *sendComm)
{
    if (!sendComm)
        return ncclSuccess;
    vxfec_send_comm_t *sc = (vxfec_send_comm_t *)sendComm;
    vxf_tx_bg_unregister(sc);

    bool ibv_quiesced = true;
    if (sc->ibv_ep) {
        ibv_quiesced = vxfec_ibv_ep_destroy(sc->ibv_ep);
        if (!ibv_quiesced)
            atomic_fetch_add_explicit(&vxf_leaked_ibv_eps, 1u, memory_order_relaxed);
        sc->ibv_ep = NULL;
    }
    if (sc->ibv_ref) {
        if (ibv_quiesced)
            vxf_ibv_ref_release(sc->ibv_ref);
        sc->ibv_ref = NULL;
    }

    uint64_t m = sc->active_tx_mask;
    while (m) {
        uint32_t s = (uint32_t)__builtin_ctzll(m);
        m &= m - 1ULL;
        if (sc->tx_pool[s].in_use) {
            if (!ibv_quiesced && sc->tx_pool[s].src_ibv_lkey != 0u)
                sc->tx_pool[s].mr = NULL;
            vxf_free_tx_req_resources(&sc->tx_pool[s]);
            sc->tx_pool[s].in_use = false;
            sc->tx_pool[s].wire_active = false;
        }
    }
    sc->active_tx_mask = 0ULL;

    for (int i = 0; i < NCCL_NET_MAX_REQUESTS; i++) {
        if (sc->user_tx_pool[i].in_use) {
            if (ibv_quiesced || !sc->user_tx_pool[i].ibv_zc_pending) {
                free(sc->user_tx_pool[i].bounce_buf);
                vxf_mr_release(sc->user_tx_pool[i].mr);
            }
            sc->user_tx_pool[i].bounce_buf = NULL;
            sc->user_tx_pool[i].mr = NULL;
            sc->user_tx_pool[i].in_use = false;
        }
    }

    vxf_free_send_comm_rq_cache(sc);
    free(sc->tx_udp);
#if VXFEC_ENABLE_SIM
    free(sc->ctrl_delay_q);
#endif
    vxf_tag_map_free(&sc->tag_map);
    if (sc->ctrl_fd >= 0)
        close(sc->ctrl_fd);
    for (uint32_t i = 0; i < sc->num_ports; i++) {
        if (sc->fds[i] >= 0)
            close(sc->fds[i]);
    }
    if (sc->mtx_ready)
        pthread_mutex_destroy(&sc->mtx);
    free(sc);
    if (atomic_exchange_explicit(&vxf_retired_lc_reap_needed, false, memory_order_acq_rel))
        vxf_reap_retired_listeners(false);
    return ncclSuccess;
}

ncclResult_t vxf_close_recv(void *recvComm)
{
    if (!recvComm)
        return ncclSuccess;
    vxfec_recv_comm_t *rc = (vxfec_recv_comm_t *)recvComm;
    vxf_rx_bg_unregister(rc);

    bool ibv_quiesced = true;
    if (rc->ibv_ep) {
        ibv_quiesced = vxfec_ibv_ep_destroy(rc->ibv_ep);
        if (!ibv_quiesced)
            atomic_fetch_add_explicit(&vxf_leaked_ibv_eps, 1u, memory_order_relaxed);
        rc->ibv_ep = NULL;
    }
    if (rc->ibv_ref) {
        if (ibv_quiesced)
            vxf_ibv_ref_release(rc->ibv_ref);
        rc->ibv_ref = NULL;
    }

    for (uint32_t w = 0; w < VXFEC_RX_WIRE_SLOTS / 64u; w++) {
        uint64_t m = rc->active_slot_mask[w];
        while (m) {
            uint32_t b = (uint32_t)__builtin_ctzll(m);
            m &= m - 1ULL;
            vxf_reset_rx_slot(rc, &rc->slots[w * 64u + b]);
        }
    }

    for (int i = 0; i < NCCL_NET_MAX_REQUESTS; i++) {
        if (rc->rx_pool[i].in_use) {
            free(rc->rx_pool[i].heap_gen_bits);
            rc->rx_pool[i].heap_gen_bits = NULL;
            vxf_mr_release(rc->rx_pool[i].mr);
            rc->rx_pool[i].mr = NULL;
            rc->rx_pool[i].in_use = false;
        }
    }

    if (rc->accept_node) {
        atomic_store_explicit(&rc->accept_node->rc_closed, true, memory_order_release);
        vxf_accept_node_release(rc->accept_node);
        rc->accept_node = NULL;
    }
    if (rc->rx_udp) {
        free(rc->rx_udp->pkt_bufs);
        free(rc->rx_udp);
    }
#if VXFEC_ENABLE_SIM
    free(rc->delay_q);
    free(rc->header_delay_q);
#endif
    vxf_tag_map_free(&rc->tag_map);
    vxf_rx_done_range_discard(&rc->pending_done, UINT64_MAX);
    vxf_rx_done_range_discard(&rc->done_errors, UINT64_MAX);
    for (uint32_t i = 0; i < VXFEC_STAGE_POOL_SLOTS; i++)
        free(rc->stage_pool[i].buf);
    if (rc->ctrl_fd >= 0)
        close(rc->ctrl_fd);
    for (uint32_t i = 0; i < rc->num_ports; i++) {
        if (rc->fds[i] >= 0)
            close(rc->fds[i]);
    }
    if (rc->mtx_ready)
        pthread_mutex_destroy(&rc->mtx);
    free(rc);
    vxf_reap_retired_listeners(false);
    return ncclSuccess;
}

ncclResult_t vxf_close_listen(void *listenComm)
{
    if (!listenComm)
        return ncclSuccess;
    vxfec_listen_comm_t *lc = (vxfec_listen_comm_t *)listenComm;
    vxf_reap_retired_listeners(false);

    vxfec_recv_comm_t *unaccepted[VXFEC_MAX_LISTEN_BACKLOG];
    uint32_t n_unacc = 0u;
    vxfec_accept_node_t *overflow_head = NULL;

    pthread_mutex_lock(&lc->mtx);
    lc->retired_ns = mono_ns();
    atomic_store_explicit(&lc->closed_by_user, true, memory_order_release);
    lc_drain_ctrl_locked(lc);
    for (vxfec_accept_node_t *it = lc->head; it != NULL; it = it->next) {
        if (it->rc) {
            if (n_unacc < VXFEC_MAX_LISTEN_BACKLOG) {
                unaccepted[n_unacc++] = it->rc;
            } else {
                vxfec_accept_node_t *tmp = (vxfec_accept_node_t *)calloc(1, sizeof(*tmp));
                if (tmp) {
                    tmp->rc = it->rc;
                    tmp->next = overflow_head;
                    overflow_head = tmp;
                }
            }
            it->rc = NULL;
            atomic_store_explicit(&it->rc_closed, true, memory_order_release);
        }
    }
    lc->backlog_cnt = 0u;
    lc_reap_closed_locked(lc);
    pthread_mutex_unlock(&lc->mtx);

    for (uint32_t i = 0; i < n_unacc; i++)
        (void)vxf_close_recv(unaccepted[i]);
    while (overflow_head) {
        vxfec_accept_node_t *nxt = overflow_head->next;
        if (overflow_head->rc)
            (void)vxf_close_recv(overflow_head->rc);
        free(overflow_head);
        overflow_head = nxt;
    }

    pthread_mutex_lock(&lc->mtx);
    lc_reap_closed_locked(lc);
    bool keep_for_replay = (lc->bg_started && lc->head != NULL);
    pthread_mutex_unlock(&lc->mtx);

    if (keep_for_replay) {
        pthread_mutex_lock(&g_retired_lc_mtx);
        lc->retired_next = g_retired_listeners;
        g_retired_listeners = lc;
        atomic_fetch_add_explicit(&vxf_retired_lc_cnt, 1u, memory_order_acq_rel);
        pthread_mutex_unlock(&g_retired_lc_mtx);
        vxf_rx_bg_wake();
        vxf_reap_retired_listeners(false);
        return ncclSuccess;
    }

    destroy_listen_comm_now(lc);
    return ncclSuccess;
}

void vxf_conn_shutdown(void)
{
    rx_bg_shutdown();
    vxf_reap_retired_listeners(true);

    pthread_mutex_lock(&g_conn_mtx);
    vxfec_pending_conn_t *pc = g_pending_conns;
    g_pending_conns = NULL;
    pthread_mutex_unlock(&g_conn_mtx);
    while (pc) {
        vxfec_pending_conn_t *nxt = pc->next;
        if (pc->sc)
            (void)vxf_close_send(pc->sc);
        free(pc);
        pc = nxt;
    }
}
