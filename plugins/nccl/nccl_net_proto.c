#define _GNU_SOURCE
#include "nccl_net_internal.h"

static const vxfec_req_kind_t g_flush_sentinel = VXFEC_REQ_FLUSH;

void vxf_tag_map_free(vxfec_tag_map_t *map)
{
    if (map->slots && map->slots != map->inline_slots)
        free(map->slots);
    map->slots = NULL;
    map->cap = 0u;
    map->count = 0u;
}

uint32_t vxf_next_tag_seq(vxfec_tag_map_t *map, uint32_t tag)
{
    if (!map->slots) {
        map->slots = map->inline_slots;
        map->cap = VXFEC_TAG_INLINE_CAP;
        map->count = 0u;
    }

    uint32_t mask = map->cap - 1u;
    uint32_t idx = (tag * 0x9E3779B1u) & mask;
    while (map->slots[idx].used) {
        if (map->slots[idx].tag == tag)
            return next_non_zero_seq(&map->slots[idx].next_seq);
        idx = (idx + 1u) & mask;
    }

    if ((uint64_t)(map->count + 1u) * 4u > (uint64_t)map->cap * 3u) {
        if (map->cap > (UINT32_MAX / 2u))
            return 0u;
        uint32_t new_cap = map->cap * 2u;
        vxfec_tag_slot_t *new_slots = (vxfec_tag_slot_t *)calloc(new_cap, sizeof(*new_slots));
        if (!new_slots)
            return 0u;
        uint32_t new_mask = new_cap - 1u;
        for (uint32_t i = 0; i < map->cap; i++) {
            if (!map->slots[i].used)
                continue;
            uint32_t dst = (map->slots[i].tag * 0x9E3779B1u) & new_mask;
            while (new_slots[dst].used)
                dst = (dst + 1u) & new_mask;
            new_slots[dst] = map->slots[i];
        }
        if (map->slots != map->inline_slots)
            free(map->slots);
        map->slots = new_slots;
        map->cap = new_cap;
        mask = new_mask;
        idx = (tag * 0x9E3779B1u) & mask;
        while (map->slots[idx].used)
            idx = (idx + 1u) & mask;
    }

    map->slots[idx].used = true;
    map->slots[idx].tag = tag;
    map->slots[idx].next_seq = 2u;
    map->count++;
    return 1u;
}

uint32_t vxf_peek_tag_seq(const vxfec_tag_map_t *map, uint32_t tag)
{
    if (!map->slots || map->cap == 0u)
        return 1u;
    uint32_t mask = map->cap - 1u;
    uint32_t idx = (tag * 0x9E3779B1u) & mask;
    while (map->slots[idx].used) {
        if (map->slots[idx].tag == tag)
            return map->slots[idx].next_seq ? map->slots[idx].next_seq : 1u;
        idx = (idx + 1u) & mask;
    }
    return 1u;
}

/* O(1) open-addressed msg_seq -> slot index hash helpers (Knuth Algorithm R deletion) */
static void tx_hash_insert(vxfec_send_comm_t *comm, uint32_t slot_idx, uint32_t msg_seq)
{
    const uint32_t mask = VXFEC_TX_HASH_SLOTS - 1u;
    uint32_t h = (msg_seq * 0x9E3779B1u) & mask;
    while (comm->tx_seq_hash[h] >= 0) {
        if ((uint32_t)comm->tx_seq_hash[h] == slot_idx)
            return;
        h = (h + 1u) & mask;
    }
    comm->tx_seq_hash[h] = (int16_t)slot_idx;
}

static vxfec_tx_slot_t *tx_hash_lookup(vxfec_send_comm_t *comm, uint32_t msg_seq)
{
    if (msg_seq == 0u || comm->active_tx_mask == 0ULL)
        return NULL;
    const uint32_t mask = VXFEC_TX_HASH_SLOTS - 1u;
    uint32_t h = (msg_seq * 0x9E3779B1u) & mask;
    for (uint32_t step = 0; step <= mask; step++) {
        int16_t s = comm->tx_seq_hash[h];
        if (s < 0)
            return NULL;
        if (comm->tx_pool[s].wire_active && comm->tx_pool[s].msg_seq == msg_seq)
            return &comm->tx_pool[s];
        h = (h + 1u) & mask;
    }
    return NULL;
}

static void tx_hash_remove(vxfec_send_comm_t *comm, uint32_t slot_idx, uint32_t msg_seq)
{
    const uint32_t mask = VXFEC_TX_HASH_SLOTS - 1u;
    uint32_t i = (msg_seq * 0x9E3779B1u) & mask;
    for (uint32_t step = 0; step <= mask; step++) {
        if (comm->tx_seq_hash[i] < 0)
            return;
        if ((uint32_t)comm->tx_seq_hash[i] == slot_idx)
            break;
        i = (i + 1u) & mask;
    }
    if (comm->tx_seq_hash[i] < 0 || (uint32_t)comm->tx_seq_hash[i] != slot_idx)
        return;

    uint32_t j = i;
    for (;;) {
        j = (j + 1u) & mask;
        int16_t sj = comm->tx_seq_hash[j];
        if (sj < 0)
            break;
        uint32_t k = (comm->tx_pool[sj].msg_seq * 0x9E3779B1u) & mask;
        bool in_cyclic = (i <= j) ? (i < k && k <= j) : (i < k || k <= j);
        if (!in_cyclic) {
            comm->tx_seq_hash[i] = sj;
            i = j;
        }
    }
    comm->tx_seq_hash[i] = -1;
}

static void rx_hash_insert(vxfec_recv_comm_t *comm, uint32_t slot_idx, uint32_t msg_seq)
{
    const uint32_t mask = VXFEC_RX_HASH_SLOTS - 1u;
    uint32_t h = (msg_seq * 0x9E3779B1u) & mask;
    while (comm->msg_seq_hash[h] >= 0) {
        if ((uint32_t)comm->msg_seq_hash[h] == slot_idx)
            return;
        h = (h + 1u) & mask;
    }
    comm->msg_seq_hash[h] = (int16_t)slot_idx;
}

static vxfec_rx_slot_t *rx_hash_lookup(vxfec_recv_comm_t *comm, uint32_t msg_seq)
{
    if (msg_seq == 0u || (comm->active_slot_mask[0] | comm->active_slot_mask[1]) == 0ULL)
        return NULL;
    const uint32_t mask = VXFEC_RX_HASH_SLOTS - 1u;
    uint32_t h = (msg_seq * 0x9E3779B1u) & mask;
    for (uint32_t step = 0; step <= mask; step++) {
        int16_t s = comm->msg_seq_hash[h];
        if (s < 0)
            return NULL;
        if (comm->slots[s].active && comm->slots[s].msg_seq == msg_seq)
            return &comm->slots[s];
        h = (h + 1u) & mask;
    }
    return NULL;
}

static void rx_hash_remove(vxfec_recv_comm_t *comm, uint32_t slot_idx, uint32_t msg_seq)
{
    const uint32_t mask = VXFEC_RX_HASH_SLOTS - 1u;
    uint32_t i = (msg_seq * 0x9E3779B1u) & mask;
    for (uint32_t step = 0; step <= mask; step++) {
        if (comm->msg_seq_hash[i] < 0)
            return;
        if ((uint32_t)comm->msg_seq_hash[i] == slot_idx)
            break;
        i = (i + 1u) & mask;
    }
    if (comm->msg_seq_hash[i] < 0 || (uint32_t)comm->msg_seq_hash[i] != slot_idx)
        return;

    uint32_t j = i;
    for (;;) {
        j = (j + 1u) & mask;
        int16_t sj = comm->msg_seq_hash[j];
        if (sj < 0)
            break;
        uint32_t k = (comm->slots[sj].msg_seq * 0x9E3779B1u) & mask;
        bool in_cyclic = (i <= j) ? (i < k && k <= j) : (i < k || k <= j);
        if (!in_cyclic) {
            comm->msg_seq_hash[i] = sj;
            i = j;
        }
    }
    comm->msg_seq_hash[i] = -1;
}

uint16_t vxf_tx_path_sym_limit(const vxfec_send_comm_t *comm)
{
    const vxfec_nic_dev_t *d =
        (comm->dev_idx >= 0 && (uint32_t)comm->dev_idx < vxf_num_devs) ? &vxf_devs[comm->dev_idx] : &vxf_devs[0];
    uint32_t limit = VXFEC_MAX_SYM_T;
    uint8_t active_feat = comm->connected ? comm->features : comm->hello.features;
    size_t wire_hdr_bytes = (active_feat & VXFEC_HELLO_SLICE) ? sizeof(vxfec_wire_hdr_ext_t) : sizeof(vxfec_wire_hdr_t);

    if (comm->ibv_ep && !comm->ibv_failed) {
        uint32_t imtu = vxfec_ibv_ep_path_mtu(comm->ibv_ep);
        if (imtu == 0u)
            imtu = d->mtu ? d->mtu : 4096u;
        uint32_t hdr_rsv = (wire_hdr_bytes > 80u) ? (uint32_t)wire_hdr_bytes : 80u;
        if (imtu <= hdr_rsv)
            return 0u;
        uint32_t ibv_cap = (imtu - hdr_rsv) & ~15u;
        if (ibv_cap < limit)
            limit = ibv_cap;
        if (!vxfec_ibv_ep_tx_is_hdr_only(comm->ibv_ep))
            return (uint16_t)(limit & ~15u);
    }

    const int headers = (int)(sizeof(struct iphdr) + sizeof(struct udphdr) + wire_hdr_bytes);
    if (!d->is_ibv && d->mtu > 0u && d->mtu < 9000u) {
        if (d->mtu <= (uint32_t)headers)
            return 0u;
        uint32_t dev_cap = d->mtu - (uint32_t)headers;
        if (dev_cap < limit)
            limit = dev_cap;
    }
    for (uint32_t i = 0; i < comm->num_ports; i++) {
        int mtu = 0;
        socklen_t len = sizeof(mtu);
        uint32_t cap = VXFEC_DEF_SYM_T;
        if (comm->fds[i] >= 0 && getsockopt(comm->fds[i], IPPROTO_IP, IP_MTU, &mtu, &len) == 0) {
            if (mtu <= headers)
                return 0u;
            cap = (uint32_t)(mtu - headers);
        }
        if (cap < limit)
            limit = cap;
    }
    return (uint16_t)(limit & ~15u);
}

static uint16_t get_cfg_sym_t(uint32_t total_bytes, uint16_t path_limit)
{
    uint32_t cfg = atomic_load_explicit(&vxf_cfg_mtu_sym, memory_order_relaxed);
    if (path_limit == 0u)
        return 0u;
    if (cfg)
        return (uint16_t)((cfg < path_limit) ? cfg : path_limit);

    uint32_t t = VXFEC_DEF_SYM_T;
    if (total_bytes >= 1024u * 1024u)
        t = 8192u;
    else if (total_bytes >= 320u * 1024u)
        t = 4096u;
    if (t > path_limit)
        t = path_limit;
    while (t < path_limit && (((uint64_t)total_bytes + t - 1u) / t) > 56000u)
        t = (t * 2u < path_limit) ? (t * 2u) : path_limit;
    return (uint16_t)t;
}

static bool is_rq_wire_flag(uint8_t flags)
{
    return flags == WIRE_FLAG_RQ_GF256 || flags == WIRE_FLAG_RQ_GF2;
}

static vxfec_codec_t choose_tx_codec(uint32_t K, uint8_t features, uint8_t *out_wire_flags)
{
    vxfec_codec_ovr_t ovr = (vxfec_codec_ovr_t)atomic_load_explicit(&vxf_codec_override, memory_order_acquire);
    if (ovr == VXFEC_CODEC_OVR_GBN) {
        *out_wire_flags = WIRE_FLAG_NOFEC_GBN;
        return VXFEC_CODEC_RS_CAUCHY8;
    }
    if (ovr == VXFEC_CODEC_OVR_SACK || K > K_max) {
        *out_wire_flags = WIRE_FLAG_SACK_ONLY;
        return VXFEC_CODEC_RQ_GF256;
    }
    vxfec_codec_t c = vxfec_select_codec(K, 56u, false);
    if (c == VXFEC_CODEC_RQ_GF2)
        c = VXFEC_CODEC_RQ_GF256;
    if (ovr == VXFEC_CODEC_OVR_RS8 && K <= 240u)
        c = VXFEC_CODEC_RS_CAUCHY8;
    else if (ovr == VXFEC_CODEC_OVR_RQ_GF256)
        c = VXFEC_CODEC_RQ_GF256;
    else if (ovr == VXFEC_CODEC_OVR_RQ_GF2)
        c = (features & VXFEC_HELLO_GF2) ? VXFEC_CODEC_RQ_GF2 : VXFEC_CODEC_RQ_GF256;

    if (c == VXFEC_CODEC_RS_CAUCHY8)
        *out_wire_flags = WIRE_FLAG_RS_CAUCHY8;
    else if (c == VXFEC_CODEC_RQ_GF2)
        *out_wire_flags = WIRE_FLAG_RQ_GF2;
    else
        *out_wire_flags = WIRE_FLAG_RQ_GF256;
    return c;
}

static uint32_t wire_flags_ir_slack(uint8_t flags, uint32_t K)
{
    if (flags == WIRE_FLAG_RS_CAUCHY8 || flags == WIRE_FLAG_NOFEC_GBN || flags == WIRE_FLAG_SACK_ONLY)
        return vxfec_codec_ir_slack(VXFEC_CODEC_RS_CAUCHY8, K);
    if (flags == WIRE_FLAG_RQ_GF2)
        return vxfec_codec_ir_slack(VXFEC_CODEC_RQ_GF2, K);
    return vxfec_codec_ir_slack(VXFEC_CODEC_RQ_GF256, K);
}

void vxf_init_send_comm_pacer(vxfec_send_comm_t *sc, uint64_t now_ns)
{
    uint32_t rmem = atomic_load_explicit(&vxf_granted_rcvbuf, memory_order_relaxed);
    uint64_t init_cwnd = (rmem >= 16u * 1024u * 1024u) ? (8ULL * 1024ULL * 1024ULL) : (2ULL * 1024ULL * 1024ULL);
    uint64_t max_cwnd = (rmem >= 16u * 1024u * 1024u) ? (32ULL * 1024ULL * 1024ULL) : (16ULL * 1024ULL * 1024ULL);
    const char *cwnd_s = getenv("VXFEC_NCCL_CWND_KB");
    if (cwnd_s && *cwnd_s) {
        long kb = atol(cwnd_s);
        if (kb >= 16) {
            init_cwnd = (uint64_t)kb * 1024ULL;
            max_cwnd = (init_cwnd > 8ULL * 1024ULL * 1024ULL) ? init_cwnd : (8ULL * 1024ULL * 1024ULL);
        }
    }

    uint64_t max_rate_mbps = vxf_sim_rate_cap_mbps();
    if (max_rate_mbps == 0u) {
        const char *rate_s = getenv("VXFEC_NCCL_MAX_RATE_MBPS");
        max_rate_mbps = (rate_s && *rate_s && atol(rate_s) > 0) ? (uint64_t)atol(rate_s) : 400000ULL;
    }
    uint64_t max_rate_Bps = max_rate_mbps * 125000ULL;
    uint64_t min_rate_Bps = 25ULL * 1000ULL * 1000ULL;
    if (min_rate_Bps > max_rate_Bps / 4u)
        min_rate_Bps = (max_rate_Bps >= 4u) ? (max_rate_Bps / 4u) : 1u;

    uint64_t target_qdelay_ns = 2500000ULL;
    const char *qd_s = getenv("VXFEC_NCCL_TARGET_QDELAY_US");
    if (qd_s && *qd_s) {
        long us = atol(qd_s);
        target_qdelay_ns = (us > 0) ? ((uint64_t)us * 1000ULL) : 0ULL;
    }

    harq_pacer_init(&sc->pacer, init_cwnd, 32ULL * 1024ULL, max_cwnd, min_rate_Bps, max_rate_Bps, target_qdelay_ns, now_ns);
    for (uint32_t i = 0; i < VXFEC_MAX_PORTS; i++)
        sc->port_weights[i] = VXFEC_PORT_WEIGHT_MAX;
    sc->has_nonuniform_weights = false;
    sc->cfg_epoch = vxf_sim_epoch();
}

static bool ensure_tx_host_ptr(vxfec_tx_slot_t *req)
{
    if (req->src_host || req->total_bytes == 0u)
        return true;
    if (!req->mr || req->mr->ptr_type != NCCL_PTR_CUDA || !req->mr->vk_buf)
        return false;
    req->bounce_buf = (uint8_t *)malloc(req->total_bytes);
    if (!req->bounce_buf)
        return false;
    size_t sub_off = (size_t)((uintptr_t)req->src_ibv_ptr - (uintptr_t)req->mr->base_ptr);
    if (!vxf_mr_vk_read(req->mr, sub_off, req->bounce_buf, req->total_bytes)) {
        free(req->bounce_buf);
        req->bounce_buf = NULL;
        return false;
    }
    req->src_host = req->bounce_buf;
    return true;
}

static const uint8_t *get_tx_rs8_input(vxfec_tx_slot_t *req)
{
    if (!ensure_tx_host_ptr(req))
        return NULL;
    size_t tot = (size_t)req->K * req->sym_T;
    if (req->total_bytes == tot && req->src_host)
        return req->src_host;
    if (!req->rs_sys_padded) {
        req->rs_sys_padded = (uint8_t *)calloc(1, tot ? tot : 16u);
        if (!req->rs_sys_padded)
            return NULL;
        if (req->total_bytes > 0u && req->src_host)
            memcpy(req->rs_sys_padded, req->src_host, req->total_bytes);
    }
    return req->rs_sys_padded;
}

void vxf_free_send_comm_rq_cache(vxfec_send_comm_t *sc)
{
    if (!sc || !sc->rq_cache_valid)
        return;
    free(sc->rq_cache_sched.ops.a);
    sc->rq_cache_sched.ops.a = NULL;
    free(sc->rq_cache_work);
    sc->rq_cache_work = NULL;
    free(sc->rq_cache_prep);
    sc->rq_cache_prep = NULL;
    sc->rq_cache_valid = false;
}

static bool build_tx_rq_schedule(uint32_t K, vxfec_codec_t codec, nanorq_core *out_enc, uint8_t **out_prep, uint8_t **out_work,
                                 schedule *out_sched, uint32_t *out_rows)
{
    memset(out_enc, 0, sizeof(*out_enc));
#ifdef NANORQ_HAS_GF2_MODE
    bool ok_init =
        (codec == VXFEC_CODEC_RQ_GF2) ? nanorq_core_encoder_new_gf2(K, 0, out_enc) : nanorq_core_encoder_new(K, 0, out_enc);
#else
    (void)codec;
    bool ok_init = nanorq_core_encoder_new(K, 0, out_enc);
#endif
    if (!ok_init)
        return false;

    size_t prep_len = nanorq_core_calculate_prepare_memory(out_enc);
    uint8_t *prep_b = (uint8_t *)malloc(prep_len);
    if (!prep_b || !nanorq_core_prepare(out_enc, prep_b, prep_len)) {
        free(prep_b);
        return false;
    }

    size_t work_len = nanorq_core_calculate_prepared_work_memory(out_enc);
    size_t sched_len = ops_estimate_schedule_bytes(K);
    uint8_t *work_b = work_len ? (uint8_t *)malloc(work_len) : NULL;
    void *sched_b = malloc(sched_len);
    memset(out_sched, 0, sizeof(*out_sched));
    if (!work_b || !schedule_init(out_sched, sched_b, sched_len)) {
        free(sched_b);
        free(work_b);
        free(prep_b);
        return false;
    }

    nanorq_core_set_op_callback(out_enc, out_sched, ops_push);
    if (!nanorq_core_precalculate(out_enc, work_b, work_len) || out_sched->overflowed || out_sched->cpidx != 2) {
        free(out_sched->ops.a);
        out_sched->ops.a = NULL;
        free(work_b);
        free(prep_b);
        return false;
    }

    *out_prep = prep_b;
    *out_work = work_b;
    *out_rows = nanorq_core_get_pc_rows(out_enc);
    return true;
}

static bool ensure_tx_rq_encoder(vxfec_tx_slot_t *req)
{
    if (req->rq_ready)
        return true;
    if (!ensure_tx_host_ptr(req))
        return false;

    vxfec_send_comm_t *sc = req->comm;
    bool use_comm_cache = (sc != NULL);
    if (sc && sc->rq_cache_valid && (sc->rq_cache_K != req->K || sc->rq_cache_codec != req->codec)) {
        uint64_t m = sc->active_tx_mask;
        while (m) {
            uint32_t s = (uint32_t)__builtin_ctzll(m);
            m &= m - 1ULL;
            if (&sc->tx_pool[s] != req && sc->tx_pool[s].in_use && sc->tx_pool[s].enc_borrowed) {
                use_comm_cache = false;
                break;
            }
        }
    }

    if (use_comm_cache) {
        if (!sc->rq_cache_valid || sc->rq_cache_K != req->K || sc->rq_cache_codec != req->codec) {
            vxf_free_send_comm_rq_cache(sc);
            if (!build_tx_rq_schedule(req->K, req->codec, &sc->rq_cache_enc, &sc->rq_cache_prep, &sc->rq_cache_work,
                                      &sc->rq_cache_sched, &sc->rq_cache_rows))
                return false;
            sc->rq_cache_K = req->K;
            sc->rq_cache_codec = req->codec;
            sc->rq_cache_valid = true;
        }
        req->enc_ptr = &sc->rq_cache_enc;
        req->enc_sched = sc->rq_cache_sched;
        req->enc_rows = sc->rq_cache_rows;
        req->enc_borrowed = true;
    } else {
        req->enc_owned = (nanorq_core *)calloc(1, sizeof(*req->enc_owned));
        if (!req->enc_owned || !build_tx_rq_schedule(req->K, req->codec, req->enc_owned, &req->enc_prep, &req->enc_work,
                                                     &req->enc_sched, &req->enc_rows)) {
            free(req->enc_owned);
            req->enc_owned = NULL;
            return false;
        }
        req->enc_ptr = req->enc_owned;
        req->enc_borrowed = false;
    }

    req->enc_stride = nanorq_core_recommended_stride(req->sym_T);
    req->rq_ready = true;
    return true;
}

static void init_tx_rq_rows(vxfec_tx_slot_t *req, uint8_t *rows)
{
    uint32_t sh = nanorq_core_get_pc_genc_offset(req->enc_ptr);
    memset(rows, 0, (size_t)req->enc_rows * req->enc_stride);
    for (uint32_t e = 0; e < req->K; e++) {
        size_t off = (size_t)e * req->sym_T;
        size_t len = (off < req->total_bytes) ? (req->total_bytes - off) : 0u;
        if (len > req->sym_T)
            len = req->sym_T;
        if (len > 0u)
            memcpy(rows + (size_t)(sh + e) * req->enc_stride, req->src_host + off, len);
    }
}

static uint16_t tx_srtt_us(const vxfec_send_comm_t *comm)
{
    if (!comm || comm->pacer.srtt_ns == 0u)
        return 0u;
    uint64_t us = (comm->pacer.srtt_ns + 999ULL) / 1000ULL;
    return (us > 65535ULL) ? 65535u : (uint16_t)us;
}

static uint64_t clamp_u64(uint64_t v, uint64_t lo, uint64_t hi)
{
    if (v < lo)
        return lo;
    return (v > hi) ? hi : v;
}

static uint64_t tx_tail_rto_ns(const vxfec_send_comm_t *comm)
{
    uint64_t srtt = (comm && comm->pacer.srtt_ns > 0u) ? comm->pacer.srtt_ns : 250000ULL;
    return clamp_u64(2ULL * srtt + 250000ULL, 500000ULL, 50000000ULL);
}

static uint64_t rx_reorder_grace_ns(const vxfec_recv_comm_t *comm)
{
    uint64_t srtt = (comm && comm->peer_srtt_ns > 0u) ? comm->peer_srtt_ns : 200000ULL;
    return clamp_u64(srtt / 2ULL, 50000ULL, 2000000ULL);
}

static uint64_t rx_fb_cooldown_ns(const vxfec_recv_comm_t *comm)
{
    uint64_t srtt = (comm && comm->peer_srtt_ns > 0u) ? comm->peer_srtt_ns : 200000ULL;
    return clamp_u64(srtt + (srtt >> 1u), 200000ULL, 20000000ULL);
}

static uint64_t rx_idle_tail_ns(const vxfec_recv_comm_t *comm)
{
    uint64_t srtt = (comm && comm->peer_srtt_ns > 0u) ? comm->peer_srtt_ns : 200000ULL;
    return clamp_u64(2ULL * srtt + 200000ULL, 400000ULL, 25000000ULL);
}

/* match SACK feedback floor (200 us) for fair Go-Back-N baseline comparisons */
static uint64_t rx_gbn_cooldown_ns(const vxfec_recv_comm_t *comm)
{
    uint64_t srtt = (comm && comm->peer_srtt_ns > 0u) ? comm->peer_srtt_ns : 200000ULL;
    return clamp_u64(srtt + (srtt >> 1u), 200000ULL, 20000000ULL);
}

static vxfec_tx_udp_batch_t *ensure_tx_udp_batch(vxfec_send_comm_t *comm)
{
    if (!comm->tx_udp)
        comm->tx_udp = (vxfec_tx_udp_batch_t *)calloc(1, sizeof(*comm->tx_udp));
    return comm->tx_udp;
}

static bool tx_send_batches(vxfec_tx_slot_t *req, vxfec_tx_udp_batch_t *ub, const uint8_t *pkt_ports, const uint8_t *pkt_niov,
                            const bool *pkt_ecn, uint32_t n_valid, const uint32_t *port_cnt)
{
    if (n_valid == 0u)
        return true;
    vxfec_send_comm_t *comm = req->comm;
    uint32_t port_off[VXFEC_MAX_PORTS] = {0};
    uint32_t port_cur[VXFEC_MAX_PORTS] = {0};
    uint32_t acc = 0u;
    for (uint32_t p = 0; p < comm->num_ports; p++) {
        port_off[p] = acc;
        port_cur[p] = acc;
        acc += port_cnt[p];
    }
    for (uint32_t i = 0; i < n_valid; i++) {
        uint32_t p = pkt_ports[i];
        uint32_t dst = port_cur[p]++;
        struct mmsghdr *mm = &ub->msgs[dst];
        memset(mm, 0, sizeof(*mm));
        if (!comm->data_connected) {
            mm->msg_hdr.msg_name = &comm->peer_data[p];
            mm->msg_hdr.msg_namelen = sizeof(comm->peer_data[p]);
        }
        if (pkt_ecn[i] && comm->tx_cmsg_ce_len > 0u) {
            mm->msg_hdr.msg_control = comm->tx_cmsg_ce;
            mm->msg_hdr.msg_controllen = comm->tx_cmsg_ce_len;
        }
        mm->msg_hdr.msg_iov = ub->iovs[i];
        mm->msg_hdr.msg_iovlen = pkt_niov[i];
    }

    for (uint32_t p = 0; p < comm->num_ports; p++) {
        if (port_cnt[p] == 0u)
            continue;
        if (comm->fds[p] < 0)
            return false;
        uint32_t off = 0u;
        while (off < port_cnt[p]) {
            int sent = sendmmsg(comm->fds[p], ub->msgs + port_off[p] + off, port_cnt[p] - off, MSG_DONTWAIT);
            if (sent > 0) {
                off += (uint32_t)sent;
                continue;
            }
            if (sent < 0 && errno == EINTR)
                continue;
            req->ack_probe_tx_ns = 0u;
            if (sent < 0 && errno == EMSGSIZE) {
                uint16_t refreshed = vxf_tx_path_sym_limit(comm);
                if (refreshed >= req->sym_T)
                    refreshed = (req->sym_T > 64u) ? (uint16_t)((req->sym_T - 64u) & ~15u) : 0u;
                comm->path_sym_limit = refreshed;
                return false;
            }
            break;
        }
    }
    return true;
}

static void tx_send_udp_hdr_port(vxfec_send_comm_t *comm, uint32_t p, const void *hdr, size_t hdr_len)
{
    if (p >= comm->num_ports || comm->fds[p] < 0)
        return;
    if (!comm->data_connected && comm->peer_data[p].sin_port == 0)
        return;
    struct iovec iov = {.iov_base = (void *)hdr, .iov_len = hdr_len};
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    if (!comm->data_connected) {
        msg.msg_name = &comm->peer_data[p];
        msg.msg_namelen = sizeof(comm->peer_data[p]);
    }
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    (void)sendmsg(comm->fds[p], &msg, MSG_DONTWAIT);
}

static void tx_send_pkt_single(vxfec_send_comm_t *comm, const void *hdr, size_t hdr_len, const void *payload, size_t pay_len)
{
    const int32_t *w = comm->has_nonuniform_weights ? comm->port_weights : NULL;
    uint32_t p = swrr_next(comm->swrr_cur, w, comm->num_ports);
    if (comm->ibv_ep && !comm->ibv_failed && pay_len == 0u) {
        (void)vxfec_ibv_ep_send_pkt(comm->ibv_ep, p, hdr, (uint32_t)hdr_len, NULL, 0u, 0u);
        return;
    }
    if (comm->fds[p] < 0)
        return;
    struct iovec iov[2] = {
        {.iov_base = (void *)hdr, .iov_len = hdr_len},
        {.iov_base = (void *)payload, .iov_len = pay_len},
    };
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    if (!comm->data_connected) {
        msg.msg_name = &comm->peer_data[p];
        msg.msg_namelen = sizeof(comm->peer_data[p]);
    }
    msg.msg_iov = iov;
    msg.msg_iovlen = pay_len ? 2 : 1;
    (void)sendmsg(comm->fds[p], &msg, MSG_DONTWAIT);
}

static bool tx_spray_sys_range(vxfec_tx_slot_t *req, uint32_t base_esi, const uint32_t *esis, uint32_t count, bool is_retx,
                               bool need_ack_last, uint64_t now_ns)
{
    vxfec_send_comm_t *comm = req->comm;
    bool reorder = vxf_sim_reorder_enabled();
    const int32_t *w = comm->has_nonuniform_weights ? comm->port_weights : NULL;
    uint16_t srtt_us = tx_srtt_us(comm);
    const bool has_ext = (req->num_gens > 1u);
    const uint8_t base_flags = req->wire_flags | (has_ext ? WIRE_FLAG_GEN_EXT : 0u);
    const size_t hdr_len = has_ext ? sizeof(vxfec_wire_hdr_ext_t) : sizeof(vxfec_wire_hdr_t);
    uint32_t idx = 0;

    while (idx < count) {
        uint32_t batch = (count - idx > VXFEC_UDP_TX_BATCH) ? VXFEC_UDP_TX_BATCH : (count - idx);
        bool is_last_batch = (idx + batch == count);
        uint32_t port_cnt[VXFEC_MAX_PORTS] = {0};
        uint32_t order[VXFEC_UDP_TX_BATCH];
        uint8_t pkt_ports[VXFEC_UDP_TX_BATCH];
        uint8_t pkt_niov[VXFEC_UDP_TX_BATCH];
        bool pkt_ecn[VXFEC_UDP_TX_BATCH];
        vxfec_tx_udp_batch_t *ub = NULL;
        uint32_t n_valid = 0;

        for (uint32_t b = 0; b < batch; b++)
            order[b] = b;
        if (reorder && batch > 1u) {
            for (uint32_t b = batch - 1u; b > 0u; b--) {
                uint32_t swap_idx = vxf_fast_rand32() % (b + 1u);
                uint32_t tmp = order[b];
                order[b] = order[swap_idx];
                order[swap_idx] = tmp;
            }
        }

        for (uint32_t b = 0; b < batch; b++) {
            uint32_t e = esis ? esis[idx + order[b]] : (base_esi + idx + order[b]);
            if (vxf_sim_should_drop_data()) {
                req->ack_probe_tx_ns = 0u;
                continue;
            }

            size_t off = (size_t)e * req->sym_T;
            size_t rem = (off < req->total_bytes) ? (req->total_bytes - off) : 0u;
            uint16_t valid_len = (uint16_t)((rem > req->sym_T) ? req->sym_T : rem);

            vxfec_wire_hdr_ext_t whdr;
            whdr.base = (vxfec_wire_hdr_t){
                .magic = VXFEC_NET_MAGIC,
                .cookie = comm->cookie,
                .pkt_type = PKT_DATA,
                .flags = base_flags,
                .sym_len = valid_len,
                .msg_seq = req->msg_seq,
                .tag_seq = req->tag_seq,
                .tag = req->tag,
                .total_bytes = req->total_bytes,
                .K = req->K,
                .sym_T = req->sym_T,
                .srtt_us = srtt_us,
                .esi = e,
            };
            if (has_ext) {
                whdr.ext = (vxfec_gen_ext_t){
                    .msg_total_bytes = req->msg_total_bytes,
                    .gen_idx = req->gen_idx,
                    .num_gens = req->num_gens,
                };
            }

            uint32_t p = swrr_next(comm->swrr_cur, w, comm->num_ports);
            if (comm->ibv_ep) {
                if (vxfec_ibv_ep_has_error(comm->ibv_ep)) {
                    req->ack_probe_tx_ns = 0u;
                    return false;
                }
                vxfec_wire_hdr_ext_t ihdr = whdr;
                bool is_final_pkt = is_last_batch && (b + 1u == batch);
                if (need_ack_last && is_final_pkt) {
                    ihdr.base.flags |= WIRE_FLAG_NEED_ACK;
                    if (!is_retx && !req->retx_or_ir_sent && req->ack_probe_tx_ns == 0u)
                        req->ack_probe_tx_ns = now_ns;
                }
                if (req->src_ibv_lkey != 0u) {
                    const void *pay_ptr = (valid_len > 0u) ? (const void *)(req->src_ibv_ptr + off) : NULL;
                    if (vxfec_ibv_ep_send_pkt_ex(comm->ibv_ep, p, &ihdr, (uint32_t)hdr_len, pay_ptr, valid_len, req->src_ibv_lkey,
                                                 is_final_pkt)) {
                        if (req->parent && valid_len > 0u) {
                            req->parent->ibv_zc_pending = true;
                            vxfec_ibv_ep_record_qp_tail(comm->ibv_ep, p, req->parent->ibv_target_seq);
                        }
                        atomic_fetch_add_explicit(&vxf_st_ibv_direct_pkts, 1, memory_order_relaxed);
                        continue;
                    }
                } else if (ensure_tx_host_ptr(req)) {
                    const void *pay_ptr = (valid_len > 0u) ? (const void *)(req->src_host + off) : NULL;
                    if (vxfec_ibv_ep_send_pkt(comm->ibv_ep, p, &ihdr, (uint32_t)hdr_len, pay_ptr, valid_len, 0u)) {
                        atomic_fetch_add_explicit(&vxf_st_ibv_direct_pkts, 1, memory_order_relaxed);
                        continue;
                    }
                }
                if (vxfec_ibv_ep_has_error(comm->ibv_ep)) {
                    req->ack_probe_tx_ns = 0u;
                    return false;
                }
            }
            if (!ensure_tx_host_ptr(req) || !(ub = ensure_tx_udp_batch(comm))) {
                req->ack_probe_tx_ns = 0u;
                return false;
            }
            ub->hdrs[n_valid] = whdr;
            ub->iovs[n_valid][0].iov_base = &ub->hdrs[n_valid];
            ub->iovs[n_valid][0].iov_len = hdr_len;
            uint8_t niov = 1u;
            if (valid_len > 0u) {
                ub->iovs[n_valid][1].iov_base = (void *)(req->src_host + off);
                ub->iovs[n_valid][1].iov_len = valid_len;
                niov = 2u;
            }
            pkt_ports[n_valid] = (uint8_t)p;
            pkt_niov[n_valid] = niov;
            pkt_ecn[n_valid] = vxf_sim_should_mark_ecn(p);
            port_cnt[p]++;
            n_valid++;
        }

        if (need_ack_last && is_last_batch && n_valid > 0u) {
            ub->hdrs[n_valid - 1u].base.flags |= WIRE_FLAG_NEED_ACK;
            if (!is_retx && !req->retx_or_ir_sent && req->ack_probe_tx_ns == 0u)
                req->ack_probe_tx_ns = now_ns;
        }

        if (!tx_send_batches(req, ub, pkt_ports, pkt_niov, pkt_ecn, n_valid, port_cnt))
            return false;

        if (is_retx)
            atomic_fetch_add_explicit(&vxf_st_tx_sack_retx_pkts, batch, memory_order_relaxed);
        else
            atomic_fetch_add_explicit(&vxf_st_tx_sys_pkts, batch, memory_order_relaxed);
        idx += batch;
    }
    return true;
}

static bool tx_spray_repair_batch(vxfec_tx_slot_t *req, uint8_t flags, const uint32_t *esis, uint32_t start_esi,
                                  const uint8_t *rep_buf, size_t rep_stride, uint32_t count, bool need_ack_last, uint64_t now_ns)
{
    vxfec_send_comm_t *comm = req->comm;
    const int32_t *w = comm->has_nonuniform_weights ? comm->port_weights : NULL;
    uint16_t srtt_us = tx_srtt_us(comm);
    const bool has_ext = (req->num_gens > 1u);
    const uint8_t base_flags = flags | (has_ext ? WIRE_FLAG_GEN_EXT : 0u);
    const size_t hdr_len = has_ext ? sizeof(vxfec_wire_hdr_ext_t) : sizeof(vxfec_wire_hdr_t);
    uint32_t idx = 0;

    while (idx < count) {
        uint32_t batch = (count - idx > VXFEC_UDP_TX_BATCH) ? VXFEC_UDP_TX_BATCH : (count - idx);
        bool is_last_batch = (idx + batch == count);
        uint32_t port_cnt[VXFEC_MAX_PORTS] = {0};
        uint8_t pkt_ports[VXFEC_UDP_TX_BATCH];
        uint8_t pkt_niov[VXFEC_UDP_TX_BATCH];
        bool pkt_ecn[VXFEC_UDP_TX_BATCH];
        vxfec_tx_udp_batch_t *ub = NULL;
        uint32_t n_valid = 0;

        for (uint32_t b = 0; b < batch; b++) {
            if (vxf_sim_should_drop_data()) {
                req->ack_probe_tx_ns = 0u;
                continue;
            }
            uint32_t i = idx + b;
            vxfec_wire_hdr_ext_t whdr;
            whdr.base = (vxfec_wire_hdr_t){
                .magic = VXFEC_NET_MAGIC,
                .cookie = comm->cookie,
                .pkt_type = PKT_DATA,
                .flags = base_flags,
                .sym_len = req->sym_T,
                .msg_seq = req->msg_seq,
                .tag_seq = req->tag_seq,
                .tag = req->tag,
                .total_bytes = req->total_bytes,
                .K = req->K,
                .sym_T = req->sym_T,
                .srtt_us = srtt_us,
                .esi = esis ? esis[i] : (start_esi + i),
            };
            if (has_ext) {
                whdr.ext = (vxfec_gen_ext_t){
                    .msg_total_bytes = req->msg_total_bytes,
                    .gen_idx = req->gen_idx,
                    .num_gens = req->num_gens,
                };
            }
            uint32_t p = swrr_next(comm->swrr_cur, w, comm->num_ports);
            if (comm->ibv_ep) {
                if (vxfec_ibv_ep_has_error(comm->ibv_ep)) {
                    req->ack_probe_tx_ns = 0u;
                    return false;
                }
                vxfec_wire_hdr_ext_t ihdr = whdr;
                if (need_ack_last && is_last_batch && (b + 1u == batch)) {
                    ihdr.base.flags |= WIRE_FLAG_NEED_ACK;
                    if (!req->retx_or_ir_sent && req->ack_probe_tx_ns == 0u)
                        req->ack_probe_tx_ns = now_ns;
                }
                if (vxfec_ibv_ep_send_pkt(comm->ibv_ep, p, &ihdr, (uint32_t)hdr_len, rep_buf + (size_t)i * rep_stride, req->sym_T,
                                          0u)) {
                    atomic_fetch_add_explicit(&vxf_st_ibv_direct_pkts, 1, memory_order_relaxed);
                    continue;
                }
                if (vxfec_ibv_ep_has_error(comm->ibv_ep)) {
                    req->ack_probe_tx_ns = 0u;
                    return false;
                }
            }
            if (!(ub = ensure_tx_udp_batch(comm))) {
                req->ack_probe_tx_ns = 0u;
                return false;
            }
            ub->hdrs[n_valid] = whdr;
            ub->iovs[n_valid][0].iov_base = &ub->hdrs[n_valid];
            ub->iovs[n_valid][0].iov_len = hdr_len;
            ub->iovs[n_valid][1].iov_base = (void *)(rep_buf + (size_t)i * rep_stride);
            ub->iovs[n_valid][1].iov_len = req->sym_T;
            pkt_ports[n_valid] = (uint8_t)p;
            pkt_niov[n_valid] = 2u;
            pkt_ecn[n_valid] = vxf_sim_should_mark_ecn(p);
            port_cnt[p]++;
            n_valid++;
        }

        if (need_ack_last && is_last_batch && n_valid > 0u) {
            ub->hdrs[n_valid - 1u].base.flags |= WIRE_FLAG_NEED_ACK;
            if (!req->retx_or_ir_sent && req->ack_probe_tx_ns == 0u)
                req->ack_probe_tx_ns = now_ns;
        }

        if (!tx_send_batches(req, ub, pkt_ports, pkt_niov, pkt_ecn, n_valid, port_cnt))
            return false;
        idx += batch;
    }
    return true;
}

typedef enum {
    TX_REPAIR_FAILED,
    TX_REPAIR_SENT,
    TX_REPAIR_GPU_QUEUED,
} tx_repair_result_t;

static bool tx_spray_coded_repairs(vxfec_tx_slot_t *req, uint8_t flags, uint32_t start_esi, uint32_t count, const uint8_t *rep_buf,
                                   size_t rep_stride, bool proactive, bool need_ack_last, uint64_t now_ns)
{
    if (!tx_spray_repair_batch(req, flags, NULL, start_esi, rep_buf, rep_stride, count, need_ack_last, now_ns))
        return false;
    if (start_esi + count > req->next_rep_esi)
        req->next_rep_esi = start_esi + count;
    atomic_fetch_add_explicit(proactive ? &vxf_st_tx_proactive_rep_pkts : &vxf_st_tx_harq_ir_pkts, count, memory_order_relaxed);
    return true;
}

static tx_repair_result_t tx_generate_and_spray_repairs_rs8(vxfec_tx_slot_t *req, uint32_t start_esi, uint32_t count,
                                                            bool proactive, bool need_ack_last, uint64_t now_ns)
{
    if (start_esi < req->K || start_esi >= 256u)
        return TX_REPAIR_FAILED;
    if (start_esi + count > 256u)
        count = 256u - start_esi;
    if (count == 0u)
        return TX_REPAIR_SENT;

    const uint8_t *sys_in = get_tx_rs8_input(req);
    if (!sys_in)
        return TX_REPAIR_FAILED;

    uint8_t parity_m[256];
    for (uint32_t i = 0; i < count; i++)
        parity_m[i] = (uint8_t)((start_esi - req->K) + i);

    size_t in_bytes = (size_t)req->K * req->sym_T;
    size_t out_bytes = (size_t)count * req->sym_T;
    uint32_t rs_m = 256u - req->K;
    vxfec_vk_ctx *prefer_vk = req->mr ? req->mr->vk_ctx : NULL;

    vxfec_gpu_job_t *job =
        req->tx_gpu_failed ? NULL : vxf_rs8_gpu_job_new(prefer_vk, false, req->K, rs_m, NULL, parity_m, count, req->sym_T);
    if (job) {
        memcpy(vxfec_vk_buf_staging_ptr(job->in_buf), sys_in, in_bytes);
        if (vxf_rs8_gpu_job_submit(job, req->sym_T)) {
            req->tx_gpu = job;
            req->tx_gpu_esi = start_esi;
            req->tx_gpu_count = count;
            req->tx_gpu_offset = 0u;
            req->tx_gpu_stride = req->sym_T;
            req->tx_gpu_flags = WIRE_FLAG_RS_CAUCHY8;
            req->tx_gpu_proactive = proactive;
            req->tx_gpu_ready = false;
            return TX_REPAIR_GPU_QUEUED;
        }
        vxf_gpu_job_free(job);
    }

    uint8_t *rep_buf = (uint8_t *)calloc(1, out_bytes);
    if (!rep_buf)
        return TX_REPAIR_FAILED;
    bool ok =
        vxfec_rs8_encode_cpu(req->K, rs_m, sys_in, req->sym_T, parity_m, count, rep_buf) &&
        tx_spray_coded_repairs(req, WIRE_FLAG_RS_CAUCHY8, start_esi, count, rep_buf, req->sym_T, proactive, need_ack_last, now_ns);
    free(rep_buf);
    return ok ? TX_REPAIR_SENT : TX_REPAIR_FAILED;
}

/* queue GPU repair batch or generate and spray immediately on CPU using pivoted SIMD ops */
static tx_repair_result_t tx_generate_and_spray_repairs(vxfec_tx_slot_t *req, uint32_t start_esi, uint32_t count, bool proactive,
                                                        bool need_ack_last, uint64_t now_ns)
{
    if (count == 0u)
        return TX_REPAIR_SENT;
    if (req->codec == VXFEC_CODEC_RS_CAUCHY8) {
        if (start_esi + count <= 256u)
            return tx_generate_and_spray_repairs_rs8(req, start_esi, count, proactive, need_ack_last, now_ns);
        req->codec = VXFEC_CODEC_RQ_GF256;
        req->wire_flags = WIRE_FLAG_RQ_GF256;
        if (start_esi < 256u)
            start_esi = 256u;
        req->ir_start_esi = start_esi;
    }
    if (!ensure_tx_rq_encoder(req))
        return TX_REPAIR_FAILED;

    size_t d_bytes = (size_t)req->enc_rows * req->enc_stride;
    size_t out_bytes = (size_t)count * req->enc_stride;
    vxfec_vk_ctx *prefer_vk = req->mr ? req->mr->vk_ctx : NULL;
    vxfec_gpu_job_t *job = (req->tx_gpu_failed || count > VXFEC_BATCH_PKTS)
                               ? NULL
                               : vxf_rq_gpu_job_new(prefer_vk, req->enc_ptr, &req->enc_sched, d_bytes, out_bytes);
    if (job) {
        uint32_t esis[VXFEC_BATCH_PKTS];
        for (uint32_t i = 0; i < count; i++)
            esis[i] = start_esi + i;
        init_tx_rq_rows(req, vxfec_vk_buf_staging_ptr(job->in_buf));
        if (vxf_rq_gpu_job_submit(job, req->enc_ptr, req->enc_stride, esis, count)) {
            req->tx_gpu = job;
            req->tx_gpu_esi = start_esi;
            req->tx_gpu_count = count;
            req->tx_gpu_offset = 0u;
            req->tx_gpu_stride = req->enc_stride;
            req->tx_gpu_flags = req->wire_flags;
            req->tx_gpu_proactive = proactive;
            req->tx_gpu_ready = false;
            return TX_REPAIR_GPU_QUEUED;
        }
        vxf_gpu_job_free(job);
    }

    if (!req->enc_d_cpu) {
        req->enc_d_cpu = (uint8_t *)malloc(d_bytes);
        if (!req->enc_d_cpu)
            return TX_REPAIR_FAILED;
        init_tx_rq_rows(req, req->enc_d_cpu);
        if (!ops_run_pivoted(req->enc_d_cpu, req->enc_stride, &req->enc_sched)) {
            free(req->enc_d_cpu);
            req->enc_d_cpu = NULL;
            return TX_REPAIR_FAILED;
        }
    }

    uint8_t *rep_buf = (uint8_t *)malloc(out_bytes);
    if (!rep_buf)
        return TX_REPAIR_FAILED;
    for (uint32_t i = 0; i < count; i++)
        ops_mix_pivoted(req->enc_ptr, req->enc_d_cpu, req->enc_stride, start_esi + i, rep_buf + (size_t)i * req->enc_stride);
    bool sent =
        tx_spray_coded_repairs(req, req->wire_flags, start_esi, count, rep_buf, req->enc_stride, proactive, need_ack_last, now_ns);
    free(rep_buf);
    return sent ? TX_REPAIR_SENT : TX_REPAIR_FAILED;
}

static bool tx_send_ctrl_wire_hdr(vxfec_send_comm_t *comm, const void *hdr, size_t hdr_len)
{
    if (comm->ctrl_fd < 0)
        return false;
    if (comm->ctrl_connected) {
        if (send(comm->ctrl_fd, hdr, hdr_len, MSG_DONTWAIT) < 0 && errno == ECONNREFUSED)
            return false;
        return true;
    }
    if (comm->peer_rx_ctrl.sin_port != 0) {
        (void)sendto(comm->ctrl_fd, hdr, hdr_len, MSG_DONTWAIT, (const struct sockaddr *)&comm->peer_rx_ctrl,
                     sizeof(comm->peer_rx_ctrl));
        return true;
    }
    return false;
}

static void tx_send_tail_ex(vxfec_tx_slot_t *req, bool wake_ctrl)
{
    const bool has_ext = (req->num_gens > 1u);
    vxfec_wire_hdr_ext_t thdr;
    thdr.base = (vxfec_wire_hdr_t){
        .magic = VXFEC_NET_MAGIC,
        .cookie = req->comm->cookie,
        .pkt_type = PKT_TAIL,
        .flags = req->wire_flags | (has_ext ? WIRE_FLAG_GEN_EXT : 0u),
        .sym_len = 0,
        .msg_seq = req->msg_seq,
        .tag_seq = req->tag_seq,
        .tag = req->tag,
        .total_bytes = req->total_bytes,
        .K = req->K,
        .sym_T = req->sym_T,
        .srtt_us = tx_srtt_us(req->comm),
        .esi = req->next_rep_esi,
    };
    if (has_ext) {
        thdr.ext = (vxfec_gen_ext_t){
            .msg_total_bytes = req->msg_total_bytes,
            .gen_idx = req->gen_idx,
            .num_gens = req->num_gens,
        };
    }
    size_t hlen = has_ext ? sizeof(thdr) : sizeof(thdr.base);
    tx_send_pkt_single(req->comm, &thdr, hlen, NULL, 0);
    if (wake_ctrl) {
        (void)tx_send_ctrl_wire_hdr(req->comm, &thdr, hlen);
        if (req->comm->ibv_ep)
            tx_send_udp_hdr_port(req->comm, 0u, &thdr, hlen);
    }
    req->last_tail_ns = mono_ns();
}

static void tx_send_tail(vxfec_tx_slot_t *req)
{
    tx_send_tail_ex(req, false);
}

static bool tx_emit_abort_pkt(vxfec_send_comm_t *comm, uint32_t msg_seq, uint32_t tag_seq, uint32_t tag, int copies)
{
    if (msg_seq == 0u || tag_seq == 0u)
        return true;
    if (vxf_sim_should_drop_aborts())
        return true;
    vxfec_wire_hdr_t ehdr = {
        .magic = VXFEC_NET_MAGIC,
        .cookie = comm->cookie,
        .pkt_type = PKT_ERR,
        .flags = 0u,
        .sym_len = 0u,
        .msg_seq = msg_seq,
        .tag_seq = tag_seq,
        .tag = tag,
        .total_bytes = 0u,
        .K = 1u,
        .sym_T = VXFEC_DEF_SYM_T,
    };
    bool peer_alive = true;
    for (int c = 0; c < copies; c++) {
        uint32_t p = swrr_next(comm->swrr_cur, NULL, comm->num_ports);
        if (comm->ibv_ep && !comm->ibv_failed)
            (void)vxfec_ibv_ep_send_pkt(comm->ibv_ep, p, &ehdr, (uint32_t)sizeof(ehdr), NULL, 0u, 0u);
        tx_send_udp_hdr_port(comm, p, &ehdr, sizeof(ehdr));
        if (!tx_send_ctrl_wire_hdr(comm, &ehdr, sizeof(ehdr)) && comm->ctrl_connected)
            peer_alive = false;
    }
    return peer_alive;
}

static void tx_record_and_send_abort(vxfec_send_comm_t *comm, uint32_t msg_seq, uint32_t tag_seq, uint32_t tag)
{
    if (tag_seq == 0u)
        return;
    uint64_t now_ns = mono_ns();
    for (uint32_t i = 0; i < VXFEC_ABORT_RING; i++) {
        if (comm->tx_aborts[i].tag_seq == tag_seq && comm->tx_aborts[i].tag == tag) {
            if (msg_seq != 0u && comm->tx_aborts[i].msg_seq == 0u)
                comm->tx_aborts[i].msg_seq = msg_seq;
            comm->tx_aborts[i].last_tx_ns = now_ns;
            if (!comm->tx_aborts[i].acked)
                (void)tx_emit_abort_pkt(comm, comm->tx_aborts[i].msg_seq, tag_seq, tag, 2);
            return;
        }
    }
    if (msg_seq == 0u)
        msg_seq = next_non_zero_seq(&comm->next_msg_seq);
    uint32_t idx = comm->tx_abort_cursor++ & (VXFEC_ABORT_RING - 1u);
    if (comm->tx_aborts[idx].tag_seq != 0u && !comm->tx_aborts[idx].acked && comm->unacked_aborts > 0u)
        comm->unacked_aborts--;
    bool alive = tx_emit_abort_pkt(comm, msg_seq, tag_seq, tag, 2);
    comm->tx_aborts[idx] = (vxfec_abort_entry_t){
        .msg_seq = msg_seq,
        .tag_seq = tag_seq,
        .tag = tag,
        .last_tx_ns = now_ns,
        .tx_tries = 1u,
        .acked = !alive,
    };
    if (alive)
        comm->unacked_aborts++;
}

static void tx_ack_abort_locked(vxfec_send_comm_t *comm, uint32_t msg_seq, uint32_t tag)
{
    if (comm->unacked_aborts == 0u)
        return;
    for (uint32_t i = 0; i < VXFEC_ABORT_RING; i++) {
        vxfec_abort_entry_t *ae = &comm->tx_aborts[i];
        if (ae->tag_seq != 0u && !ae->acked && ae->tag == tag && (msg_seq == 0u || ae->msg_seq == msg_seq)) {
            ae->acked = true;
            if (comm->unacked_aborts > 0u)
                comm->unacked_aborts--;
        }
    }
}

static void tx_retransmit_aborts_locked(vxfec_send_comm_t *comm, uint64_t now_ns)
{
    if (comm->unacked_aborts == 0u)
        return;
    for (uint32_t i = 0; i < VXFEC_ABORT_RING; i++) {
        vxfec_abort_entry_t *ae = &comm->tx_aborts[i];
        if (ae->tag_seq == 0u || ae->acked)
            continue;
        uint64_t rto = (ae->tx_tries < 32u) ? 2000000ULL : 20000000ULL;
        if (now_ns < ae->last_tx_ns || (now_ns - ae->last_tx_ns) < rto)
            continue;
        if (ae->tx_tries >= 128u || !tx_emit_abort_pkt(comm, ae->msg_seq, ae->tag_seq, ae->tag, 1)) {
            ae->acked = true;
            if (comm->unacked_aborts > 0u)
                comm->unacked_aborts--;
            continue;
        }
        ae->last_tx_ns = now_ns;
        ae->tx_tries++;
    }
}

void vxf_free_tx_req_resources(vxfec_tx_slot_t *req)
{
    if (!req)
        return;
    vxf_gpu_job_free(req->tx_gpu);
    req->tx_gpu = NULL;
    free(req->sack_queue);
    req->sack_queue = NULL;
    free(req->rs_sys_padded);
    req->rs_sys_padded = NULL;
    free(req->enc_d_cpu);
    req->enc_d_cpu = NULL;
    if (!req->enc_borrowed) {
        free(req->enc_sched.ops.a);
        free(req->enc_work);
        free(req->enc_prep);
        free(req->enc_owned);
    }
    req->enc_sched.ops.a = NULL;
    req->enc_work = NULL;
    req->enc_prep = NULL;
    req->enc_owned = NULL;
    req->enc_borrowed = false;
    req->enc_ptr = NULL;
    free(req->bounce_buf);
    req->bounce_buf = NULL;
    if (req->mr) {
        vxf_mr_release(req->mr);
        req->mr = NULL;
    }
    req->rq_ready = false;
}

static void tx_deactivate_slot(vxfec_send_comm_t *comm, vxfec_tx_slot_t *req, bool release_resources)
{
    uint32_t s = req->slot_idx;
    if (req->wire_active && req->msg_seq != 0u)
        tx_hash_remove(comm, s, req->msg_seq);
    req->wire_active = false;
    req->done = true;
    if (release_resources) {
        vxf_free_tx_req_resources(req);
        req->in_use = false;
        comm->active_tx_mask &= ~(1ULL << s);
    }
}

static void release_rx_slot_vk(vxfec_rx_slot_t *s)
{
    vxf_gpu_job_free(s->rs8_gpu_job);
    s->rs8_gpu_job = NULL;
    free(s->vk_compact);
    s->vk_compact = NULL;
}

void vxf_reset_rx_slot(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *s)
{
    if (s->active && comm) {
        uint32_t idx = s->slot_idx;
        if (s->msg_seq != 0u)
            rx_hash_remove(comm, idx, s->msg_seq);
        comm->active_slot_mask[idx >> 6u] &= ~(1ULL << (idx & 63u));
    }
    vxf_gpu_job_free(s->rq_gpu_job);
    free(s->rq_gpu_missing);
    s->rq_gpu_job = NULL;
    s->rq_gpu_missing = NULL;
    release_rx_slot_vk(s);
    free(s->sys_bitmap);
    if (s->stage_pool_idx > 0 && comm && (uint32_t)s->stage_pool_idx <= VXFEC_STAGE_POOL_SLOTS)
        comm->stage_pool[s->stage_pool_idx - 1].in_use = false;
    else
        free(s->sys_stage);
    free(s->rep_esis);
    free(s->rep_data);
    memset(s, 0, sizeof(*s));
}

static void rx_send_ctrl(vxfec_recv_comm_t *comm, const vxfec_ctrl_pkt_t *ctrl, const void *payload, size_t pay_len, int copies)
{
    if (ctrl->pkt_type == PKT_DONE) {
        int live_copies = 0;
        for (int c = 0; c < copies; c++) {
            if (!vxf_sim_should_drop_dones())
                live_copies++;
        }
        if (live_copies == 0)
            return;
        copies = live_copies;
    }

    struct iovec iov[2] = {
        {.iov_base = (void *)ctrl, .iov_len = sizeof(*ctrl)},
        {.iov_base = (void *)payload, .iov_len = pay_len},
    };
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = iov;
    msg.msg_iovlen = pay_len ? 2 : 1;

    if (comm->ctrl_fd >= 0 && (comm->ctrl_connected || comm->peer_ctrl.sin_port != 0)) {
        if (!comm->ctrl_connected) {
            msg.msg_name = &comm->peer_ctrl;
            msg.msg_namelen = sizeof(comm->peer_ctrl);
        }
        for (int c = 0; c < copies; c++)
            (void)sendmsg(comm->ctrl_fd, &msg, MSG_DONTWAIT);
        return;
    }

    for (int c = 0; c < copies; c++) {
        uint32_t p = swrr_next(comm->swrr_cur, NULL, comm->num_ports);
        if (comm->fds[p] < 0)
            continue;
        msg.msg_name = &comm->peer_data[p];
        msg.msg_namelen = sizeof(comm->peer_data[p]);
        (void)sendmsg(comm->fds[p], &msg, MSG_DONTWAIT);
    }
}

static uint64_t rx_done_seq(const vxfec_recv_comm_t *comm, uint32_t msg_seq)
{
    if (msg_seq == 0u || comm->completed_watermark_seq == 0u)
        return msg_seq;
    int32_t delta = (int32_t)(msg_seq - (uint32_t)comm->completed_watermark_seq);
    int64_t seq = (int64_t)comm->completed_watermark_seq + delta;
    return seq > 0 ? (uint64_t)seq : 0u;
}

static uint64_t rx_done_range_end(const vxfec_done_range_t *range, uint64_t seq)
{
    while (range && range->last < seq)
        range = range->next;
    return (range && range->first <= seq) ? range->last : 0u;
}

static bool rx_done_range_add(vxfec_done_range_t **head, uint64_t seq)
{
    while (*head && (*head)->last + 1u < seq)
        head = &(*head)->next;

    vxfec_done_range_t *range = *head;
    if (!range || seq + 1u < range->first) {
        range = (vxfec_done_range_t *)malloc(sizeof(*range));
        if (!range)
            return false;
        *range = (vxfec_done_range_t){.first = seq, .last = seq, .next = *head};
        *head = range;
    } else if (seq < range->first) {
        range->first = seq;
    } else if (seq > range->last) {
        range->last = seq;
        if (range->next && range->last + 1u == range->next->first) {
            vxfec_done_range_t *next = range->next;
            range->last = next->last;
            range->next = next->next;
            free(next);
        }
    }
    return true;
}

void vxf_rx_done_range_discard(vxfec_done_range_t **head, uint64_t through)
{
    while (*head && (*head)->last <= through) {
        vxfec_done_range_t *range = *head;
        *head = range->next;
        free(range);
    }
    if (*head && (*head)->first <= through)
        (*head)->first = through + 1u;
}

static bool rx_prepare_done(vxfec_recv_comm_t *comm, uint32_t msg_seq)
{
    uint32_t idx = msg_seq & (VXFEC_DONE_RING - 1u);
    uint64_t prev = comm->done_seqs[idx];
    if (prev == 0u)
        return true;
    if (comm->done_err[idx]) {
        uint64_t watermark = comm->completed_watermark_seq;
        if (prev > watermark || watermark - prev <= INT32_MAX)
            return rx_done_range_add(&comm->done_errors, prev);
    } else if (prev > comm->completed_watermark_seq) {
        return rx_done_range_add(&comm->pending_done, prev);
    }
    return true;
}

static bool rx_comm_was_done(vxfec_recv_comm_t *comm, uint32_t msg_seq, uint32_t *out_lost, uint32_t *out_k, uint32_t *out_ecn_ce,
                             uint32_t *out_rx_pkts, uint32_t *out_ecn_mask, bool *out_err)
{
    uint64_t seq = rx_done_seq(comm, msg_seq);
    if (seq == 0u || seq > comm->highest_done_seq)
        return false;
    uint32_t idx = msg_seq & (VXFEC_DONE_RING - 1u);
    if (comm->done_seqs[idx] == seq) {
        if (out_lost)
            *out_lost = comm->done_lost[idx];
        if (out_k)
            *out_k = comm->done_k[idx];
        if (out_ecn_ce)
            *out_ecn_ce = comm->done_ecn_ce[idx];
        if (out_rx_pkts)
            *out_rx_pkts = comm->done_rx_pkts[idx];
        if (out_ecn_mask)
            *out_ecn_mask = comm->done_ecn_mask[idx];
        if (out_err)
            *out_err = comm->done_err[idx];
        return true;
    }
    bool is_err = rx_done_range_end(comm->done_errors, seq) != 0u;
    if (is_err || seq <= comm->completed_watermark_seq || rx_done_range_end(comm->pending_done, seq) != 0u) {
        if (out_lost)
            *out_lost = 0u;
        if (out_k)
            *out_k = 1u;
        if (out_ecn_ce)
            *out_ecn_ce = 0u;
        if (out_rx_pkts)
            *out_rx_pkts = 1u;
        if (out_ecn_mask)
            *out_ecn_mask = 0u;
        if (out_err)
            *out_err = is_err;
        return true;
    }
    return false;
}

static void rx_record_done(vxfec_recv_comm_t *comm, uint32_t msg_seq, uint32_t K, uint32_t lost_sys, uint32_t ecn_ce,
                           uint32_t rx_pkts, uint32_t ecn_mask, bool is_err)
{
    if (msg_seq == 0u)
        return;
    uint32_t idx = msg_seq & (VXFEC_DONE_RING - 1u);
    uint64_t seq = rx_done_seq(comm, msg_seq);
    comm->done_seqs[idx] = seq;
    comm->done_k[idx] = K;
    comm->done_lost[idx] = lost_sys;
    comm->done_ecn_ce[idx] = ecn_ce;
    comm->done_rx_pkts[idx] = rx_pkts;
    comm->done_ecn_mask[idx] = ecn_mask;
    comm->done_err[idx] = is_err;
    if (seq > comm->highest_done_seq)
        comm->highest_done_seq = seq;

    uint64_t watermark = comm->completed_watermark_seq;
    for (;;) {
        uint64_t next = watermark + 1u;
        if ((uint32_t)next == 0u)
            next++;
        if (next > comm->highest_done_seq)
            break;
        if (comm->done_seqs[next & (VXFEC_DONE_RING - 1u)] == next) {
            watermark = next;
            continue;
        }
        uint64_t end = rx_done_range_end(comm->pending_done, next);
        if (end == 0u)
            end = rx_done_range_end(comm->done_errors, next);
        if (end == 0u)
            break;
        watermark = end;
    }
    if (watermark != comm->completed_watermark_seq) {
        comm->completed_watermark_seq = watermark;
        vxf_rx_done_range_discard(&comm->pending_done, watermark);
        if (watermark > INT32_MAX)
            vxf_rx_done_range_discard(&comm->done_errors, watermark - INT32_MAX - 1u);
    }
}

static bool rx_req_test_gen_bit(const vxfec_rx_req_t *req, uint16_t gen_idx)
{
    uint64_t bit = 1ULL << (gen_idx & 63u);
    if (req->expected_num_gens <= 64u)
        return (req->inline_gen_bits & bit) != 0u;
    return req->heap_gen_bits && ((req->heap_gen_bits[gen_idx >> 6u] & bit) != 0u);
}

static bool rx_req_set_gen_bit(vxfec_rx_req_t *req, uint16_t gen_idx)
{
    uint64_t bit = 1ULL << (gen_idx & 63u);
    if (req->expected_num_gens <= 64u) {
        bool was_set = (req->inline_gen_bits & bit) != 0u;
        req->inline_gen_bits |= bit;
        return !was_set;
    }
    if (!req->heap_gen_bits)
        return false;
    bool was_set = (req->heap_gen_bits[gen_idx >> 6u] & bit) != 0u;
    req->heap_gen_bits[gen_idx >> 6u] |= bit;
    return !was_set;
}

static bool rx_bind_slot_to_req(vxfec_rx_slot_t *s, vxfec_rx_req_t *req)
{
    if (req->expected_num_gens == 0u) {
        if (s->num_gens > 64u) {
            req->heap_gen_bits = (uint64_t *)calloc(((size_t)s->num_gens + 63u) / 64u, sizeof(uint64_t));
            if (!req->heap_gen_bits)
                return false;
        }
        req->expected_num_gens = s->num_gens;
        req->msg_total_bytes = s->msg_total_bytes;
        req->bound_sym_T = s->sym_T;
    } else if (req->expected_num_gens != s->num_gens || req->msg_total_bytes != s->msg_total_bytes ||
               (s->num_gens > 1u && req->bound_sym_T != s->sym_T)) {
        return false;
    }
    if (rx_req_test_gen_bit(req, s->gen_idx))
        return false;
    s->bound_req = req;
    return true;
}

static bool rx_slot_overflows_req(const vxfec_rx_slot_t *slot)
{
    if (!slot->bound_req)
        return false;
    if (slot->bound_req->max_size < 0)
        return true;
    size_t max_sz = (size_t)slot->bound_req->max_size;
    return (size_t)slot->msg_total_bytes > max_sz || slot->gen_offset > max_sz ||
           (size_t)slot->total_bytes > (max_sz - slot->gen_offset);
}

static void rx_promote_slot_to_rq(vxfec_rx_slot_t *slot, uint8_t rq_flags)
{
    slot->flags = rq_flags;
    slot->ir_slack = wire_flags_ir_slack(rq_flags, slot->K);
    slot->rx_rep_cnt = 0u;
    if (slot->max_rep_esi < 255u)
        slot->max_rep_esi = 255u;
}

static uint8_t *rx_slot_sys_base(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *slot)
{
    if (rx_slot_overflows_req(slot))
        return NULL;
    if (slot->bound_req && (!slot->bound_req->mr || slot->bound_req->mr->ptr_type == NCCL_PTR_HOST) && !slot->sys_stage)
        return (uint8_t *)slot->bound_req->dst_ptr + slot->gen_offset;

    if (!slot->sys_stage) {
        size_t alloc_sz = (size_t)slot->K * slot->sym_T;
        if (alloc_sz == 0u)
            alloc_sz = 1u;
        if (comm) {
            for (uint32_t p = 0; p < VXFEC_STAGE_POOL_SLOTS; p++) {
                vxfec_stage_slot_t *ps = &comm->stage_pool[p];
                if (ps->in_use)
                    continue;
                if (alloc_sz > ps->cap) {
                    uint8_t *nb = (uint8_t *)realloc(ps->buf, alloc_sz);
                    if (!nb)
                        break;
                    ps->buf = nb;
                    ps->cap = alloc_sz;
                }
                ps->in_use = true;
                slot->sys_stage = ps->buf;
                slot->stage_pool_idx = (int8_t)(p + 1u);
                if (alloc_sz > slot->total_bytes)
                    memset(slot->sys_stage + slot->total_bytes, 0, alloc_sz - slot->total_bytes);
                return slot->sys_stage;
            }
        }
        slot->sys_stage = (uint8_t *)calloc(1, alloc_sz);
        slot->stage_pool_idx = 0;
    }
    return slot->sys_stage;
}

static bool rx_tag_seq_is_retired(const vxfec_recv_comm_t *comm, uint32_t tag, uint32_t tag_seq)
{
    uint32_t next_s = vxf_peek_tag_seq(&comm->tag_map, tag);
    if (!seq_before(tag_seq, next_s))
        return false;
    for (uint32_t r = 0; r < NCCL_NET_MAX_REQUESTS; r++) {
        const vxfec_rx_req_t *req = &comm->rx_pool[r];
        if (req->in_use && !req->done && req->tag == tag && req->expected_tag_seq == tag_seq)
            return false;
    }
    return true;
}

static void rx_fail_slot(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *slot, ncclResult_t err_status)
{
    vxfec_rx_req_t *req = slot->bound_req;
    uint32_t fail_tag = slot->tag;
    uint32_t fail_tag_seq = slot->tag_seq;
    (void)rx_prepare_done(comm, slot->msg_seq);
    rx_record_done(comm, slot->msg_seq, slot->K, 0u, 0u, 0u, 0u, true);
    vxfec_ctrl_pkt_t ehdr = {
        .magic = VXFEC_NET_MAGIC,
        .cookie = comm->cookie,
        .pkt_type = PKT_ERR,
        .msg_seq = slot->msg_seq,
        .tag = slot->tag,
    };
    rx_send_ctrl(comm, &ehdr, NULL, 0, 2);
    vxf_reset_rx_slot(comm, slot);

    for (uint32_t w = 0; w < VXFEC_RX_WIRE_SLOTS / 64u; w++) {
        uint64_t m = comm->active_slot_mask[w];
        while (m) {
            uint32_t i = w * 64u + (uint32_t)__builtin_ctzll(m);
            m &= m - 1ULL;
            if (comm->slots[i].active && ((req && comm->slots[i].bound_req == req) ||
                                          (comm->slots[i].tag == fail_tag && comm->slots[i].tag_seq == fail_tag_seq))) {
                (void)rx_prepare_done(comm, comm->slots[i].msg_seq);
                rx_record_done(comm, comm->slots[i].msg_seq, comm->slots[i].K, 0u, 0u, 0u, 0u, true);
                ehdr.msg_seq = comm->slots[i].msg_seq;
                ehdr.tag = comm->slots[i].tag;
                rx_send_ctrl(comm, &ehdr, NULL, 0, 1);
                vxf_reset_rx_slot(comm, &comm->slots[i]);
            }
        }
    }
    if (req) {
        req->status = err_status;
        req->done = true;
    }
}

static void rx_finish_slot(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *slot, bool used_fec)
{
    vxfec_rx_req_t *req = slot->bound_req;
    if (!req)
        return;

    bool write_ok = !slot->stage_failed;
    if (write_ok && slot->total_bytes > 0u && slot->sys_stage) {
        size_t max_rem = ((size_t)req->max_size > slot->gen_offset) ? ((size_t)req->max_size - slot->gen_offset) : 0u;
        uint32_t copy_bytes = (slot->total_bytes <= max_rem) ? slot->total_bytes : (uint32_t)max_rem;
        if (req->mr && req->mr->ptr_type == NCCL_PTR_CUDA) {
            if (!req->mr->vk_buf || vxf_sim_should_fail_gpu_rx()) {
                write_ok = false;
            } else {
                size_t sub_off = (size_t)((uintptr_t)req->dst_ptr - (uintptr_t)req->mr->base_ptr) + slot->gen_offset;
                write_ok = vxf_mr_vk_write(req->mr, sub_off, slot->sys_stage, copy_bytes);
            }
        } else if (req->dst_ptr) {
            memcpy((uint8_t *)req->dst_ptr + slot->gen_offset, slot->sys_stage, copy_bytes);
        } else {
            write_ok = false;
        }
    } else if (write_ok && slot->total_bytes > 0u && req->mr && req->mr->ptr_type == NCCL_PTR_CUDA) {
        write_ok = false;
    }

    if (!write_ok) {
        rx_fail_slot(comm, slot, ncclSystemError);
        return;
    }

    if (req->expected_num_gens == 0u && !rx_bind_slot_to_req(slot, req)) {
        rx_fail_slot(comm, slot, ncclSystemError);
        return;
    }
    if (req->expected_num_gens != slot->num_gens || req->msg_total_bytes != slot->msg_total_bytes) {
        rx_fail_slot(comm, slot, ncclInternalError);
        return;
    }

    uint32_t lost = slot->initial_missing_sys;
    uint32_t ecn_ce = slot->rx_ecn_ce_cnt;
    uint32_t rx_pkts = slot->rx_pkt_cnt ? slot->rx_pkt_cnt : slot->K;
    uint64_t done_ns = mono_ns();
    uint64_t ref_rx_ns = slot->tail_seen ? slot->tail_seen_ns : slot->last_pkt_ns;
    uint64_t dwell_us64 = (ref_rx_ns > 0u && done_ns > ref_rx_ns) ? ((done_ns - ref_rx_ns) / 1000ULL) : 0u;
    uint32_t dwell_us = (dwell_us64 > UINT32_MAX) ? UINT32_MAX : (uint32_t)dwell_us64;

    rx_record_done(comm, slot->msg_seq, slot->K, lost, ecn_ce, rx_pkts, slot->ecn_port_mask, false);

    vxfec_ctrl_pkt_t dhdr = {
        .magic = VXFEC_NET_MAGIC,
        .cookie = comm->cookie,
        .pkt_type = PKT_DONE,
        .msg_seq = slot->msg_seq,
        .tag = slot->tag,
        .K = slot->K,
        .rx_pkts = rx_pkts,
        .ecn_ce_pkts = ecn_ce,
        .ecn_port_mask = slot->ecn_port_mask,
        .esi = lost,
        .dwell_us = dwell_us,
    };
    rx_send_ctrl(comm, &dhdr, NULL, 0, (slot->num_gens > 1u) ? 1 : 2);

    if (used_fec) {
        req->any_gen_used_fec = true;
        atomic_fetch_add_explicit(&vxf_st_rx_fec_decodes, 1, memory_order_relaxed);
    }

    if (rx_req_set_gen_bit(req, slot->gen_idx))
        req->gens_completed++;

    vxf_reset_rx_slot(comm, slot);

    if (req->gens_completed >= req->expected_num_gens) {
        req->actual_size = (int)req->msg_total_bytes;
        atomic_fetch_add_explicit(&vxf_st_rx_messages, 1, memory_order_relaxed);
        if (!req->any_gen_used_fec)
            atomic_fetch_add_explicit(&vxf_st_rx_fast_completions, 1, memory_order_relaxed);
        req->done = true;
    }
}

static int rx_try_rs8_decode(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *slot)
{
    uint32_t K = slot->K;
    uint32_t missing = K - slot->rx_sys_cnt;
    if (missing == 0u)
        return 1;
    if (K > 255u || slot->rx_rep_cnt < missing)
        return -1;

    uint8_t erased_k[256];
    uint8_t parity_m[256];
    uint32_t m_idx = 0;
    for (uint32_t e = 0; e < K && m_idx < missing; e++) {
        if (!(slot->sys_bitmap[e >> 3u] & (1u << (e & 7u))))
            erased_k[m_idx++] = (uint8_t)e;
    }

    uint32_t p_idx = 0;
    for (uint32_t r = 0; r < slot->rx_rep_cnt && p_idx < missing; r++) {
        uint32_t esi = slot->rep_esis[r];
        if (esi >= K && esi < 256u)
            parity_m[p_idx++] = (uint8_t)(esi - K);
    }
    if (p_idx < missing)
        return -1;

    uint8_t *sys_base = rx_slot_sys_base(comm, slot);
    if (!sys_base)
        return -1;

    size_t k_bytes = (size_t)K * slot->sym_T;
    uint8_t *compact = (uint8_t *)calloc(1, k_bytes);
    if (!compact)
        return -1;

    for (uint32_t e = 0; e < K; e++) {
        if (slot->sys_bitmap[e >> 3u] & (1u << (e & 7u))) {
            size_t off = (size_t)e * slot->sym_T;
            size_t rem = (off < slot->total_bytes) ? (slot->total_bytes - off) : 0u;
            size_t clen = (rem > slot->sym_T) ? slot->sym_T : rem;
            if (clen > 0u)
                memcpy(compact + (size_t)e * slot->sym_T, sys_base + off, clen);
        }
    }

    uint32_t rep_cursor = 0;
    for (uint32_t r = 0; r < slot->rx_rep_cnt && rep_cursor < missing; r++) {
        uint32_t esi = slot->rep_esis[r];
        if (esi >= K && esi < 256u) {
            memcpy(compact + (size_t)erased_k[rep_cursor] * slot->sym_T, slot->rep_data + (size_t)r * slot->sym_T, slot->sym_T);
            rep_cursor++;
        }
    }

    uint32_t rs_m = 256u - K;
    vxfec_vk_ctx *prefer_vk = (slot->bound_req && slot->bound_req->mr) ? slot->bound_req->mr->vk_ctx : NULL;
    vxfec_gpu_job_t *job = vxf_rs8_gpu_job_new(prefer_vk, true, K, rs_m, erased_k, parity_m, missing, slot->sym_T);
    if (job) {
        memcpy(vxfec_vk_buf_staging_ptr(job->in_buf), compact, k_bytes);
        if (vxf_rs8_gpu_job_submit(job, slot->sym_T)) {
            slot->rs8_gpu_job = job;
            slot->vk_missing = missing;
            slot->vk_stride = slot->sym_T;
            slot->vk_compact = compact;
            return 0;
        }
        vxf_gpu_job_free(job);
    }

    bool decoded = vxfec_rs8_decode_compact_cpu(K, rs_m, compact, slot->sym_T, erased_k, parity_m, missing);
    if (decoded) {
        for (uint32_t i = 0; i < missing; i++) {
            uint32_t e = erased_k[i];
            uint8_t mask = (uint8_t)(1u << (e & 7u));
            if (slot->sys_bitmap[e >> 3u] & mask)
                continue;
            size_t off = (size_t)e * slot->sym_T;
            size_t rem = (off < slot->total_bytes) ? (slot->total_bytes - off) : 0u;
            size_t clen = (rem > slot->sym_T) ? slot->sym_T : rem;
            if (clen > 0u)
                memcpy(sys_base + off, compact + (size_t)e * slot->sym_T, clen);
            slot->sys_bitmap[e >> 3u] |= mask;
            slot->rx_sys_cnt++;
        }
        slot->fec_repaired = true;
        atomic_fetch_add_explicit(&vxf_st_rx_rs8_decodes, 1, memory_order_relaxed);
    }

    free(compact);
    return decoded ? 1 : -1;
}

static void rx_rq_fill_matrix(uint8_t *matrix, uint32_t stride, nanorq_core *dec, size_t bytes, const vxfec_rx_slot_t *slot,
                              const uint8_t *sys_base, const uint32_t *missing_esis, uint32_t missing, uint32_t rep_offset,
                              uint32_t overhead)
{
    uint32_t sh = nanorq_core_get_pc_genc_offset(dec);
    memset(matrix, 0, bytes);
    for (uint32_t e = 0; e < slot->K; e++) {
        if (slot->sys_bitmap[e >> 3u] & (1u << (e & 7u))) {
            size_t off = (size_t)e * slot->sym_T;
            size_t rem = (off < slot->total_bytes) ? slot->total_bytes - off : 0u;
            size_t clen = rem > slot->sym_T ? slot->sym_T : rem;
            if (clen > 0u)
                memcpy(matrix + (size_t)(sh + e) * stride, sys_base + off, clen);
        }
    }
    for (uint32_t r = 0; r < missing; r++)
        memcpy(matrix + (size_t)(sh + missing_esis[r]) * stride, slot->rep_data + (size_t)(rep_offset + r) * slot->sym_T,
               slot->sym_T);
    for (uint32_t r = 0; r < overhead; r++)
        memcpy(matrix + (size_t)(sh + dec->P.Kprime + r) * stride,
               slot->rep_data + (size_t)(rep_offset + missing + r) * slot->sym_T, slot->sym_T);
}

static void rx_rq_apply_recovery(vxfec_rx_slot_t *slot, uint8_t *sys_base, const uint32_t *missing_esis, uint32_t missing,
                                 uint32_t stride, const uint8_t *recovered)
{
    for (uint32_t r = 0; r < missing; r++) {
        uint32_t e = missing_esis[r];
        uint8_t mask = (uint8_t)(1u << (e & 7u));
        if (slot->sys_bitmap[e >> 3u] & mask)
            continue;
        size_t off = (size_t)e * slot->sym_T;
        size_t rem = (off < slot->total_bytes) ? (slot->total_bytes - off) : 0u;
        size_t clen = rem > slot->sym_T ? slot->sym_T : rem;
        if (clen > 0u)
            memcpy(sys_base + off, recovered + (size_t)r * stride, clen);
        slot->sys_bitmap[e >> 3u] |= mask;
        slot->rx_sys_cnt++;
    }
    slot->fec_repaired = true;
    atomic_fetch_add_explicit(&vxf_st_rx_rq_decodes, 1, memory_order_relaxed);
}

static bool rx_poll_rq_recovery(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *slot)
{
    if (slot->rx_sys_cnt != slot->K) {
        int status = vxf_gpu_job_poll(slot->rq_gpu_job);
        if (status == 0)
            return false;
        if (status > 0) {
            uint8_t *sys_base = rx_slot_sys_base(comm, slot);
            if (sys_base)
                rx_rq_apply_recovery(slot, sys_base, slot->rq_gpu_missing, slot->rq_gpu_count, slot->rq_gpu_stride,
                                     vxfec_vk_buf_staging_ptr(slot->rq_gpu_job->out_buf));
            else
                slot->stage_failed = true;
        } else {
            slot->rq_gpu_disabled = true;
        }
    }
    vxf_gpu_job_free(slot->rq_gpu_job);
    free(slot->rq_gpu_missing);
    slot->rq_gpu_job = NULL;
    slot->rq_gpu_missing = NULL;
    return true;
}

static bool rx_solve_rq_window(const vxfec_rx_slot_t *slot, const uint32_t *miss_esis, uint32_t missing, uint32_t rep_offset,
                               uint32_t overhead, nanorq_core *dec, schedule *S_dec, uint8_t **out_prep, uint8_t **out_work,
                               void **out_sched_buf)
{
    *out_prep = NULL;
    *out_work = NULL;
    *out_sched_buf = NULL;
    memset(dec, 0, sizeof(*dec));
    memset(S_dec, 0, sizeof(*S_dec));

#ifdef NANORQ_HAS_GF2_MODE
    bool ok_init = (slot->flags == WIRE_FLAG_RQ_GF2) ? nanorq_core_encoder_new_gf2(slot->K, overhead, dec)
                                                     : nanorq_core_encoder_new(slot->K, overhead, dec);
    if (!ok_init)
        return false;
#else
    if (!nanorq_core_encoder_new(slot->K, overhead, dec))
        return false;
#endif

    size_t prep_len = nanorq_core_calculate_prepare_memory(dec);
    size_t sched_len = ops_estimate_schedule_bytes(slot->K + overhead);
    uint8_t *prep_mem = prep_len ? (uint8_t *)malloc(prep_len) : NULL;
    void *sched_buf = sched_len ? malloc(sched_len) : NULL;
    uint8_t *work_mem = NULL;
    if (!prep_mem || !sched_buf || !schedule_init(S_dec, sched_buf, sched_len) || !nanorq_core_generate(dec, prep_mem, prep_len))
        goto fail;

    for (uint32_t m = 0; m < missing; m++)
        nanorq_core_replace_symbol(dec, miss_esis[m], slot->rep_esis[rep_offset + m]);
    for (uint32_t i = 0; i < overhead; i++)
        nanorq_core_replace_symbol(dec, dec->P.Kprime + i, slot->rep_esis[rep_offset + missing + i]);

    if (!nanorq_core_patch_matrix(dec))
        goto fail;
    size_t work_len = nanorq_core_calculate_prepared_work_memory(dec);
    if (work_len == 0u || !(work_mem = (uint8_t *)malloc(work_len)))
        goto fail;
    nanorq_core_set_op_callback(dec, S_dec, ops_push);
    bool pre_ok = nanorq_core_precalculate(dec, work_mem, work_len);
    if (pre_ok && S_dec->overflowed) {
        size_t dense_ops = (size_t)dec->P.L * ((size_t)dec->P.L + overhead) + 2048u;
        if (dense_ops <= SIZE_MAX / sizeof(sched_op)) {
            size_t big_len = dense_ops * sizeof(sched_op);
            void *big_buf = malloc(big_len);
            if (big_buf && nanorq_core_patch_matrix(dec)) {
                size_t wlen2 = nanorq_core_calculate_prepared_work_memory(dec);
                if (wlen2 > 0u && wlen2 <= work_len && schedule_init(S_dec, big_buf, big_len)) {
                    free(sched_buf);
                    sched_buf = big_buf;
                    big_buf = NULL;
                    nanorq_core_set_op_callback(dec, S_dec, ops_push);
                    pre_ok = nanorq_core_precalculate(dec, work_mem, wlen2);
                }
            }
            free(big_buf);
        }
    }
    if (!pre_ok || S_dec->overflowed || S_dec->cpidx != 2)
        goto fail;

    *out_prep = prep_mem;
    *out_work = work_mem;
    *out_sched_buf = sched_buf;
    return true;

fail:
    free(work_mem);
    free(sched_buf);
    free(prep_mem);
    return false;
}

static bool rx_try_rq_decode(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *slot)
{
    uint32_t K = slot->K;
    uint32_t missing = K - slot->rx_sys_cnt;
    if (missing == 0u)
        return true;
    if (slot->rx_rep_cnt < missing)
        return false;

    uint32_t *miss_esis = (uint32_t *)malloc((size_t)missing * sizeof(*miss_esis));
    if (!miss_esis)
        return false;
    uint32_t m_idx = 0;
    for (uint32_t e = 0; e < K; e++) {
        if (!(slot->sys_bitmap[e >> 3u] & (1u << (e & 7u))))
            miss_esis[m_idx++] = e;
    }

    uint32_t max_slack = slot->rx_rep_cnt - missing;
    uint32_t rep_offset = 0u;
    uint32_t overhead = max_slack;
    nanorq_core dec = {0};
    schedule S_dec = {0};
    uint8_t *prep_mem = NULL;
    uint8_t *work_mem = NULL;
    void *sched_buf = NULL;
    bool accepted = false;

    bool solved = rx_solve_rq_window(slot, miss_esis, missing, 0u, max_slack, &dec, &S_dec, &prep_mem, &work_mem, &sched_buf);
    if (!solved) {
        for (uint32_t off = 1u; off <= max_slack && !solved; off++) {
            uint32_t ov = slot->rx_rep_cnt - off - missing;
            if (rx_solve_rq_window(slot, miss_esis, missing, off, ov, &dec, &S_dec, &prep_mem, &work_mem, &sched_buf)) {
                rep_offset = off;
                overhead = ov;
                solved = true;
            }
        }
    }
    if (!solved)
        goto out;

    uint32_t rows = nanorq_core_get_pc_rows(&dec);
    uint32_t stride = nanorq_core_recommended_stride(slot->sym_T);
    size_t d_bytes = (size_t)rows * stride;
    size_t rec_bytes = (size_t)missing * stride;

    uint8_t *sys_base = rx_slot_sys_base(comm, slot);
    if (!sys_base)
        goto out;

    vxfec_vk_ctx *prefer_vk = (slot->bound_req && slot->bound_req->mr) ? slot->bound_req->mr->vk_ctx : NULL;
    vxfec_gpu_job_t *job = slot->rq_gpu_disabled ? NULL : vxf_rq_gpu_job_new(prefer_vk, &dec, &S_dec, d_bytes, rec_bytes);
    if (job) {
        rx_rq_fill_matrix(vxfec_vk_buf_staging_ptr(job->in_buf), stride, &dec, d_bytes, slot, sys_base, miss_esis, missing,
                          rep_offset, overhead);
        if (vxf_rq_gpu_job_submit(job, &dec, stride, miss_esis, missing)) {
            slot->rq_gpu_job = job;
            slot->rq_gpu_missing = miss_esis;
            slot->rq_gpu_count = missing;
            slot->rq_gpu_stride = stride;
            miss_esis = NULL;
            accepted = true;
            goto out;
        }
        vxf_gpu_job_free(job);
    }

    uint8_t *d_cpu = (uint8_t *)malloc(d_bytes);
    uint8_t *recovered = (uint8_t *)malloc(rec_bytes);
    if (d_cpu && recovered) {
        rx_rq_fill_matrix(d_cpu, stride, &dec, d_bytes, slot, sys_base, miss_esis, missing, rep_offset, overhead);
        if (ops_run_pivoted(d_cpu, stride, &S_dec)) {
            for (uint32_t r = 0; r < missing; r++)
                ops_mix_pivoted(&dec, d_cpu, stride, miss_esis[r], recovered + (size_t)r * stride);
            rx_rq_apply_recovery(slot, sys_base, miss_esis, missing, stride, recovered);
            accepted = true;
        }
    }
    free(d_cpu);
    free(recovered);

out:
    free(miss_esis);
    free(sched_buf);
    free(work_mem);
    free(prep_mem);
    return accepted;
}

static void rx_send_win_ack(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *slot)
{
    slot->last_acked_pkts = slot->rx_pkt_cnt;
    uint32_t dwell_us = 0u;
    if (slot->probe_rx_ns > 0u) {
        uint64_t ack_ns = mono_ns();
        uint64_t dw = (ack_ns > slot->probe_rx_ns) ? ((ack_ns - slot->probe_rx_ns) / 1000ULL) : 0u;
        dwell_us = (dw > UINT32_MAX) ? UINT32_MAX : (uint32_t)dw;
        slot->probe_rx_ns = 0u;
    }
    vxfec_ctrl_pkt_t whdr = {
        .magic = VXFEC_NET_MAGIC,
        .cookie = comm->cookie,
        .pkt_type = PKT_WIN_ACK,
        .msg_seq = slot->msg_seq,
        .tag = slot->tag,
        .K = slot->K,
        .rx_pkts = slot->rx_pkt_cnt,
        .ecn_ce_pkts = slot->rx_ecn_ce_cnt,
        .ecn_port_mask = slot->ecn_port_mask,
        .esi = slot->max_init_esi_plus1,
        .dwell_us = dwell_us,
    };
    rx_send_ctrl(comm, &whdr, NULL, 0, 1);
}

static void rx_send_sack_missing(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *slot)
{
    uint32_t sack_list[VXFEC_SACK_MAX];
    uint32_t cnt = 0;
    uint32_t scanned = 0;
    uint32_t cur = (slot->sack_cursor < slot->K) ? slot->sack_cursor : 0u;
    while (scanned < slot->K && cnt < VXFEC_SACK_MAX) {
        if (!(slot->sys_bitmap[cur >> 3u] & (1u << (cur & 7u))))
            sack_list[cnt++] = cur;
        cur++;
        if (cur >= slot->K)
            cur = 0u;
        scanned++;
    }
    slot->sack_cursor = cur;
    if (cnt == 0u)
        return;

    vxfec_ctrl_pkt_t shdr = {
        .magic = VXFEC_NET_MAGIC,
        .cookie = comm->cookie,
        .pkt_type = PKT_SACK,
        .sack_cnt = (uint16_t)cnt,
        .msg_seq = slot->msg_seq,
        .tag = slot->tag,
        .K = slot->K,
    };
    rx_send_ctrl(comm, &shdr, sack_list, cnt * sizeof(uint32_t), 1);
}

static int rx_poll_vk_decode(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *slot)
{
    vxfec_gpu_job_t *job = slot->rs8_gpu_job;
    if (!job || !job->rs8_erased_k || !job->rs8_parity_m)
        return -1;
    int pst = vxf_gpu_job_poll(job);
    if (pst == 0)
        return 0;

    if (pst == 1) {
        uint8_t *sys_base = rx_slot_sys_base(comm, slot);
        const uint8_t *rec = job->out_buf ? (const uint8_t *)vxfec_vk_buf_staging_ptr(job->out_buf) : NULL;
        if (!sys_base || !rec) {
            slot->stage_failed = true;
            release_rx_slot_vk(slot);
            return -1;
        }
        for (uint32_t i = 0; i < slot->vk_missing; i++) {
            uint32_t e = (uint32_t)job->rs8_erased_k[i];
            uint8_t mask = (uint8_t)(1u << (e & 7u));
            if (slot->sys_bitmap[e >> 3u] & mask)
                continue;
            size_t off = (size_t)e * slot->sym_T;
            size_t rem = (off < slot->total_bytes) ? (slot->total_bytes - off) : 0u;
            size_t clen = (rem > slot->sym_T) ? slot->sym_T : rem;
            if (clen > 0u)
                memcpy(sys_base + off, rec + (size_t)i * slot->vk_stride, clen);
            slot->sys_bitmap[e >> 3u] |= mask;
            slot->rx_sys_cnt++;
        }
        slot->fec_repaired = true;
        atomic_fetch_add_explicit(&vxf_st_rx_rs8_decodes, 1, memory_order_relaxed);
        release_rx_slot_vk(slot);
        return 1;
    }

    bool cpu_ok = false;
    if (slot->vk_compact) {
        uint32_t rs_m = 256u - slot->K;
        if (vxfec_rs8_decode_compact_cpu(slot->K, rs_m, slot->vk_compact, slot->sym_T, job->rs8_erased_k, job->rs8_parity_m,
                                         slot->vk_missing)) {
            uint8_t *sys_base = rx_slot_sys_base(comm, slot);
            if (sys_base) {
                for (uint32_t i = 0; i < slot->vk_missing; i++) {
                    uint32_t e = job->rs8_erased_k[i];
                    uint8_t mask = (uint8_t)(1u << (e & 7u));
                    if (slot->sys_bitmap[e >> 3u] & mask)
                        continue;
                    size_t off = (size_t)e * slot->sym_T;
                    size_t rem = (off < slot->total_bytes) ? (slot->total_bytes - off) : 0u;
                    size_t clen = (rem > slot->sym_T) ? slot->sym_T : rem;
                    if (clen > 0u)
                        memcpy(sys_base + off, slot->vk_compact + (size_t)e * slot->sym_T, clen);
                    slot->sys_bitmap[e >> 3u] |= mask;
                    slot->rx_sys_cnt++;
                }
                slot->fec_repaired = true;
                atomic_fetch_add_explicit(&vxf_st_rx_rs8_decodes, 1, memory_order_relaxed);
                cpu_ok = true;
            }
        }
    }
    release_rx_slot_vk(slot);
    return cpu_ok ? 1 : -1;
}

static void rx_evaluate_slot(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *slot, uint64_t now_ns)
{
    if (!slot->active || !slot->bound_req)
        return;

    if (rx_slot_overflows_req(slot)) {
        if (rx_prepare_done(comm, slot->msg_seq))
            rx_fail_slot(comm, slot, ncclInternalError);
        return;
    }
    if (slot->rq_gpu_job && !slot->stage_failed && !rx_poll_rq_recovery(comm, slot))
        return;
    if (slot->rs8_gpu_job && !slot->stage_failed && slot->rx_sys_cnt != slot->K) {
        int pst = rx_poll_vk_decode(comm, slot);
        if (pst == 0)
            return;
    }
    if (slot->stage_failed) {
        if (rx_prepare_done(comm, slot->msg_seq))
            rx_fail_slot(comm, slot, ncclSystemError);
        return;
    }

    uint32_t missing = slot->K - slot->rx_sys_cnt;
    if (missing == 0u) {
        if (rx_prepare_done(comm, slot->msg_seq))
            rx_finish_slot(comm, slot, slot->fec_repaired);
        return;
    }

    uint64_t grace_ns = rx_reorder_grace_ns(comm);
    uint64_t fb_cool_ns = rx_fb_cooldown_ns(comm);
    uint64_t idle_tail_ns = rx_idle_tail_ns(comm);

    if (slot->flags == WIRE_FLAG_NOFEC_GBN) {
        uint64_t gbn_cool_ns = rx_gbn_cooldown_ns(comm);
        if (!slot->tail_seen && (now_ns - slot->last_pkt_ns) < idle_tail_ns)
            return;
        if (slot->tail_seen && slot->last_fb_ns == 0u && (now_ns - slot->tail_seen_ns) < grace_ns)
            return;
        if (slot->initial_missing_sys == 0u)
            slot->initial_missing_sys = missing;
        if (slot->last_fb_ns != 0u && (now_ns - slot->last_fb_ns) < gbn_cool_ns)
            return;
        slot->last_fb_ns = now_ns;

        uint32_t first_miss = 0u;
        for (uint32_t e = 0; e < slot->K; e++) {
            if (!(slot->sys_bitmap[e >> 3u] & (1u << (e & 7u)))) {
                first_miss = e;
                break;
            }
        }
        vxfec_ctrl_pkt_t nhdr = {
            .magic = VXFEC_NET_MAGIC,
            .cookie = comm->cookie,
            .pkt_type = PKT_GBN_NACK,
            .flags = slot->flags,
            .msg_seq = slot->msg_seq,
            .tag = slot->tag,
            .K = slot->K,
            .esi = first_miss,
        };
        rx_send_ctrl(comm, &nhdr, NULL, 0, 1);
        return;
    }

    if (slot->flags == WIRE_FLAG_SACK_ONLY) {
        if (!slot->tail_seen && (now_ns - slot->last_pkt_ns) < idle_tail_ns)
            return;
        if (slot->tail_seen && slot->last_fb_ns == 0u && (now_ns - slot->tail_seen_ns) < grace_ns)
            return;
        if (slot->initial_missing_sys == 0u)
            slot->initial_missing_sys = missing;
        if (slot->last_fb_ns != 0u && (now_ns - slot->last_fb_ns) < fb_cool_ns)
            return;
        slot->last_fb_ns = now_ns;
        rx_send_sack_missing(comm, slot);
        return;
    }

    harq_decision_t dec = harq_plan_feedback(missing, slot->rx_rep_cnt, 2u, slot->ir_slack);
    if (dec.action == HARQ_DECODE_NOW) {
        if (!rx_prepare_done(comm, slot->msg_seq))
            return;
        if (slot->initial_missing_sys == 0u)
            slot->initial_missing_sys = missing;
        if (slot->flags == WIRE_FLAG_RS_CAUCHY8) {
            int dst = rx_try_rs8_decode(comm, slot);
            if (dst == 0)
                return;
            if (dst == 1) {
                rx_finish_slot(comm, slot, true);
                return;
            }
            rx_promote_slot_to_rq(slot, WIRE_FLAG_RQ_GF256);
        } else {
            if (rx_try_rq_decode(comm, slot)) {
                if (!slot->rq_gpu_job)
                    rx_finish_slot(comm, slot, true);
                return;
            }
            slot->ir_slack += 4u;
        }
        dec = harq_plan_feedback(missing, slot->rx_rep_cnt, 2u, slot->ir_slack);
    }

    if (!slot->tail_seen && (now_ns - slot->last_pkt_ns) < idle_tail_ns)
        return;
    if (slot->tail_seen && slot->last_fb_ns == 0u && (now_ns - slot->tail_seen_ns) < grace_ns)
        return;
    if (slot->initial_missing_sys == 0u)
        slot->initial_missing_sys = missing;
    if (slot->last_fb_ns != 0u && (now_ns - slot->last_fb_ns) < fb_cool_ns)
        return;

    if (dec.action == HARQ_SACK_RETX) {
        slot->last_fb_ns = now_ns;
        rx_send_sack_missing(comm, slot);
    } else if (dec.action == HARQ_IR_PARITY) {
        uint32_t start_esi = (slot->max_rep_esi >= slot->K) ? (slot->max_rep_esi + 1u) : slot->K;
        if (slot->flags == WIRE_FLAG_RS_CAUCHY8 && start_esi + dec.request_count > 256u) {
            if (slot->tail_seen && slot->last_fb_ns == 0u && (now_ns - slot->tail_seen_ns) < 2500000ULL)
                return;
            rx_promote_slot_to_rq(slot, WIRE_FLAG_RQ_GF256);
            dec = harq_plan_feedback(missing, slot->rx_rep_cnt, 2u, slot->ir_slack);
            start_esi = (slot->max_rep_esi >= slot->K) ? (slot->max_rep_esi + 1u) : slot->K;
        }
        slot->last_fb_ns = now_ns;
        vxfec_ctrl_pkt_t ihdr = {
            .magic = VXFEC_NET_MAGIC,
            .cookie = comm->cookie,
            .pkt_type = PKT_HARQ_IR,
            .flags = slot->flags,
            .msg_seq = slot->msg_seq,
            .tag = slot->tag,
            .K = slot->K,
            .esi = start_esi,
            .ir_count = dec.request_count,
        };
        rx_send_ctrl(comm, &ihdr, NULL, 0, 1);
    }
}

static vxfec_rx_slot_t *rx_find_or_create_slot(vxfec_recv_comm_t *comm, const vxfec_wire_hdr_t *hdr, const vxfec_gen_ext_t *gext)
{
    if (hdr->msg_seq == 0u || hdr->tag_seq == 0u || gext->num_gens == 0u || gext->gen_idx >= gext->num_gens)
        return NULL;

    vxfec_rx_slot_t *found = rx_hash_lookup(comm, hdr->msg_seq);
    if (found)
        return found;

    /* reject zero, misaligned, or oversized symbol lengths before allocating an RX wire slot */
    if (hdr->sym_T == 0u || (hdr->sym_T & 3u) != 0u || hdr->sym_T > VXFEC_MAX_SYM_T)
        return NULL;
    uint16_t sym_T = hdr->sym_T;

    size_t gen_off = 0u;
    if (gext->num_gens > 1u) {
        gen_off = (size_t)gext->gen_idx * VXFEC_RS8_WIN_K * sym_T;
        if (gen_off >= (size_t)gext->msg_total_bytes)
            return NULL;
        uint32_t exp_gen_b = (gext->gen_idx + 1u < gext->num_gens) ? ((uint32_t)VXFEC_RS8_WIN_K * sym_T)
                                                                   : (gext->msg_total_bytes - (uint32_t)gen_off);
        if (hdr->total_bytes != exp_gen_b)
            return NULL;
    } else if (gext->msg_total_bytes != hdr->total_bytes) {
        return NULL;
    }

    uint32_t exp_k = (hdr->total_bytes == 0u) ? 1u : (uint32_t)(((uint64_t)hdr->total_bytes + sym_T - 1u) / sym_T);
    if (hdr->K != exp_k || exp_k > K_max * 4u)
        return NULL;

    int free_idx = -1;
    for (uint32_t w = 0; w < VXFEC_RX_WIRE_SLOTS / 64u; w++) {
        uint64_t inv = ~comm->active_slot_mask[w];
        if (inv) {
            free_idx = (int)(w * 64u + (uint32_t)__builtin_ctzll(inv));
            break;
        }
    }
    if (free_idx < 0)
        return NULL;

    uint32_t i = (uint32_t)free_idx;
    vxfec_rx_slot_t *s = &comm->slots[i];
    memset(s, 0, sizeof(*s));
    s->sys_bitmap = (uint8_t *)calloc(((size_t)exp_k + 7u) / 8u, 1u);
    if (!s->sys_bitmap)
        return NULL;
    s->slot_idx = (uint8_t)i;
    s->active = true;
    s->msg_seq = hdr->msg_seq;
    s->tag_seq = hdr->tag_seq;
    s->tag = hdr->tag;
    s->msg_total_bytes = gext->msg_total_bytes;
    s->total_bytes = hdr->total_bytes;
    s->gen_offset = gen_off;
    s->K = exp_k;
    s->sym_T = sym_T;
    s->gen_idx = gext->gen_idx;
    s->num_gens = gext->num_gens;
    s->flags = hdr->flags & WIRE_CODEC_MASK;
    s->ir_slack = wire_flags_ir_slack(s->flags, s->K);

    comm->active_slot_mask[i >> 6u] |= (1ULL << (i & 63u));
    rx_hash_insert(comm, i, hdr->msg_seq);

    for (uint32_t r = 0; r < NCCL_NET_MAX_REQUESTS; r++) {
        vxfec_rx_req_t *req = &comm->rx_pool[r];
        if (req->in_use && !req->done && req->tag == s->tag && req->expected_tag_seq == s->tag_seq) {
            if (!rx_bind_slot_to_req(s, req)) {
                s->bound_req = req;
                if (rx_prepare_done(comm, s->msg_seq))
                    rx_fail_slot(comm, s, ncclSystemError);
                else
                    vxf_reset_rx_slot(comm, s);
                return NULL;
            }
            break;
        }
    }
    return s;
}

static bool extract_rx_ecn_ce(struct msghdr *mhdr)
{
    for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(mhdr); cmsg != NULL; cmsg = CMSG_NXTHDR(mhdr, cmsg)) {
        if (cmsg->cmsg_level == IPPROTO_IP &&
            (cmsg->cmsg_type == IP_TOS
#ifdef IP_RECVTOS
             || cmsg->cmsg_type == IP_RECVTOS
#endif
             ) &&
            cmsg->cmsg_len >= CMSG_LEN(sizeof(uint8_t))) {
            uint8_t tos = *CMSG_DATA(cmsg);
            if ((tos & IPTOS_ECN_MASK) == IPTOS_ECN_CE)
                return true;
        }
    }
    return false;
}

static void rx_confirm_handshake(vxfec_recv_comm_t *comm)
{
    if (comm->accept_node && !atomic_exchange_explicit(&comm->accept_node->hs_confirmed, true, memory_order_acq_rel)) {
        if (atomic_load_explicit(&vxf_retired_lc_cnt, memory_order_acquire) > 0u) {
            atomic_store_explicit(&vxf_retired_lc_reap_needed, true, memory_order_release);
            vxf_rx_bg_wake();
        }
    }
}

static void rx_process_wire_pkt(vxfec_recv_comm_t *comm, uint32_t p, const uint8_t *pkt_buf, size_t len, bool is_ecn_ce,
                                uint64_t now_ns)
{
    if (len < sizeof(vxfec_wire_hdr_t))
        return;
    vxfec_wire_hdr_t hdr;
    memcpy(&hdr, pkt_buf, sizeof(hdr));
    if (hdr.magic != VXFEC_NET_MAGIC || hdr.cookie != comm->cookie)
        return;

    rx_confirm_handshake(comm);

    if (hdr.pkt_type == PKT_ERR) {
        if (hdr.msg_seq == 0u || hdr.tag_seq == 0u)
            return;
        if (rx_prepare_done(comm, hdr.msg_seq))
            rx_record_done(comm, hdr.msg_seq, 1u, 0u, 0u, 0u, 0u, true);
        vxfec_ctrl_pkt_t ehdr = {
            .magic = VXFEC_NET_MAGIC,
            .cookie = comm->cookie,
            .pkt_type = PKT_ERR,
            .msg_seq = hdr.msg_seq,
            .tag = hdr.tag,
        };
        rx_send_ctrl(comm, &ehdr, NULL, 0, 1);

        for (uint32_t w = 0; w < VXFEC_RX_WIRE_SLOTS / 64u; w++) {
            uint64_t m = comm->active_slot_mask[w];
            while (m) {
                uint32_t i = w * 64u + (uint32_t)__builtin_ctzll(m);
                m &= m - 1ULL;
                if (comm->slots[i].active && comm->slots[i].tag == hdr.tag && comm->slots[i].tag_seq == hdr.tag_seq) {
                    if (rx_prepare_done(comm, comm->slots[i].msg_seq))
                        rx_record_done(comm, comm->slots[i].msg_seq, comm->slots[i].K, 0u, 0u, 0u, 0u, true);
                    vxf_reset_rx_slot(comm, &comm->slots[i]);
                }
            }
        }
        bool matched_req = false;
        for (uint32_t r = 0; r < NCCL_NET_MAX_REQUESTS; r++) {
            vxfec_rx_req_t *req = &comm->rx_pool[r];
            if (req->in_use && !req->done && req->tag == hdr.tag && req->expected_tag_seq == hdr.tag_seq) {
                req->status = ncclRemoteError;
                req->done = true;
                matched_req = true;
                break;
            }
        }
        if (!matched_req && !rx_tag_seq_is_retired(comm, hdr.tag, hdr.tag_seq)) {
            bool dup = false;
            for (uint32_t i = 0; i < VXFEC_ABORT_RING; i++) {
                if (comm->peer_aborts[i].tag_seq == hdr.tag_seq && comm->peer_aborts[i].tag == hdr.tag) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                uint32_t idx = comm->peer_abort_cursor++ & (VXFEC_ABORT_RING - 1u);
                comm->peer_aborts[idx] = (vxfec_abort_entry_t){
                    .msg_seq = hdr.msg_seq,
                    .tag_seq = hdr.tag_seq,
                    .tag = hdr.tag,
                };
            }
        }
        return;
    }

    const bool has_gen_ext = (hdr.flags & WIRE_FLAG_GEN_EXT) != 0u;
    const size_t hdr_bytes = has_gen_ext ? sizeof(vxfec_wire_hdr_ext_t) : sizeof(vxfec_wire_hdr_t);
    if (len < hdr_bytes)
        return;

    vxfec_gen_ext_t gext = {
        .msg_total_bytes = hdr.total_bytes,
        .gen_idx = 0u,
        .num_gens = 1u,
    };
    if (has_gen_ext) {
        memcpy(&gext, pkt_buf + sizeof(vxfec_wire_hdr_t), sizeof(gext));
        if (gext.num_gens < 2u || gext.gen_idx >= gext.num_gens || gext.msg_total_bytes < hdr.total_bytes)
            return;
    }

    if (hdr.srtt_us > 0u && (hdr.pkt_type == PKT_DATA || hdr.pkt_type == PKT_TAIL)) {
        uint64_t srtt_sample = (uint64_t)hdr.srtt_us * 1000ULL;
        comm->peer_srtt_ns = (comm->peer_srtt_ns == 0u) ? srtt_sample : ((7u * comm->peer_srtt_ns + srtt_sample) / 8u);
    }

    uint8_t pkt_codec_flags = hdr.flags & WIRE_CODEC_MASK;
    if (pkt_codec_flags == WIRE_FLAG_RQ_GF2 && !(comm->features & VXFEC_HELLO_GF2))
        return;
    bool need_win_ack = (hdr.flags & WIRE_FLAG_NEED_ACK) != 0u;

    uint32_t prev_lost = 0u, prev_ecn_mask = 0u, prev_k = 0u, prev_rx_pkts = 0u, prev_ecn_ce = 0u;
    bool prev_err = false;
    if (rx_comm_was_done(comm, hdr.msg_seq, &prev_lost, &prev_k, &prev_ecn_ce, &prev_rx_pkts, &prev_ecn_mask, &prev_err)) {
        if (hdr.pkt_type == PKT_TAIL || hdr.pkt_type == PKT_DATA) {
            vxfec_ctrl_pkt_t dhdr = {
                .magic = VXFEC_NET_MAGIC,
                .cookie = comm->cookie,
                .pkt_type = prev_err ? PKT_ERR : PKT_DONE,
                .msg_seq = hdr.msg_seq,
                .tag = hdr.tag,
                .K = prev_k,
                .rx_pkts = prev_rx_pkts,
                .ecn_ce_pkts = prev_ecn_ce,
                .ecn_port_mask = prev_ecn_mask,
                .esi = prev_lost,
            };
            rx_send_ctrl(comm, &dhdr, NULL, 0, 1);
        }
        return;
    }

    if (rx_tag_seq_is_retired(comm, hdr.tag, hdr.tag_seq)) {
        if (hdr.pkt_type == PKT_TAIL || hdr.pkt_type == PKT_DATA) {
            if (rx_prepare_done(comm, hdr.msg_seq))
                rx_record_done(comm, hdr.msg_seq, hdr.K, 0u, 0u, 0u, 0u, true);
            vxfec_ctrl_pkt_t ehdr = {
                .magic = VXFEC_NET_MAGIC,
                .cookie = comm->cookie,
                .pkt_type = PKT_ERR,
                .msg_seq = hdr.msg_seq,
                .tag = hdr.tag,
            };
            rx_send_ctrl(comm, &ehdr, NULL, 0, 1);
        }
        return;
    }

    vxfec_rx_slot_t *slot = rx_find_or_create_slot(comm, &hdr, &gext);
    if (!slot || !slot->sys_bitmap)
        return;
    if (hdr.tag != slot->tag || hdr.tag_seq != slot->tag_seq || hdr.total_bytes != slot->total_bytes || hdr.K != slot->K ||
        hdr.sym_T != slot->sym_T || gext.gen_idx != slot->gen_idx || gext.num_gens != slot->num_gens ||
        gext.msg_total_bytes != slot->msg_total_bytes)
        return;

    if (hdr.pkt_type == PKT_TAIL) {
        slot->last_pkt_ns = now_ns;
        if (is_ecn_ce) {
            slot->rx_ecn_ce_cnt++;
            if (p < 32u)
                slot->ecn_port_mask |= (1u << p);
            atomic_fetch_add_explicit(&vxf_st_rx_ecn_ce_pkts, 1, memory_order_relaxed);
        }
        if (slot->flags == WIRE_FLAG_RS_CAUCHY8 && is_rq_wire_flag(pkt_codec_flags))
            rx_promote_slot_to_rq(slot, pkt_codec_flags);
        if (!slot->tail_seen) {
            slot->tail_seen = true;
            slot->tail_seen_ns = now_ns;
        }
        return;
    }
    if (hdr.pkt_type != PKT_DATA)
        return;

    const uint8_t *pay = pkt_buf + hdr_bytes;
    size_t pay_avail = len - hdr_bytes;

    if (hdr.esi < slot->K) {
        uint32_t e = hdr.esi;
        size_t off = (size_t)e * slot->sym_T;
        uint16_t exp_len = 0u;
        if (slot->total_bytes > 0u) {
            if (off >= slot->total_bytes)
                return;
            size_t rem = slot->total_bytes - off;
            exp_len = (uint16_t)((rem > slot->sym_T) ? slot->sym_T : rem);
        }
        if (hdr.sym_len != exp_len || pay_avail < (size_t)exp_len)
            return;

        slot->last_pkt_ns = now_ns;
        slot->rx_pkt_cnt++;
        if (need_win_ack && slot->probe_rx_ns == 0u)
            slot->probe_rx_ns = now_ns;
        if (!slot->tail_seen && hdr.esi < UINT32_MAX && hdr.esi + 1u > slot->max_init_esi_plus1)
            slot->max_init_esi_plus1 = hdr.esi + 1u;
        if (is_ecn_ce) {
            slot->rx_ecn_ce_cnt++;
            if (p < 32u)
                slot->ecn_port_mask |= (1u << p);
            atomic_fetch_add_explicit(&vxf_st_rx_ecn_ce_pkts, 1, memory_order_relaxed);
        }

        if (!(slot->sys_bitmap[e >> 3u] & (1u << (e & 7u)))) {
            uint8_t *dst_base = NULL;
            if (slot->total_bytes > 0u && !rx_slot_overflows_req(slot)) {
                dst_base = rx_slot_sys_base(comm, slot);
                if (!dst_base) {
                    slot->stage_failed = true;
                    return;
                }
            }
            if (e != slot->rx_sys_cnt)
                atomic_fetch_add_explicit(&vxf_st_rx_ooo_pkts, 1, memory_order_relaxed);
            slot->sys_bitmap[e >> 3u] |= (uint8_t)(1u << (e & 7u));
            slot->rx_sys_cnt++;

            if (dst_base && exp_len > 0u)
                memcpy(dst_base + off, pay, exp_len);
        }
    } else {
        if (hdr.sym_len != slot->sym_T || pay_avail < slot->sym_T)
            return;

        slot->last_pkt_ns = now_ns;
        slot->rx_pkt_cnt++;
        if (need_win_ack && slot->probe_rx_ns == 0u)
            slot->probe_rx_ns = now_ns;
        if (!slot->tail_seen && hdr.esi < UINT32_MAX && hdr.esi + 1u > slot->max_init_esi_plus1)
            slot->max_init_esi_plus1 = hdr.esi + 1u;
        if (is_ecn_ce) {
            slot->rx_ecn_ce_cnt++;
            if (p < 32u)
                slot->ecn_port_mask |= (1u << p);
            atomic_fetch_add_explicit(&vxf_st_rx_ecn_ce_pkts, 1, memory_order_relaxed);
        }

        if (slot->flags == WIRE_FLAG_RS_CAUCHY8 && is_rq_wire_flag(pkt_codec_flags))
            rx_promote_slot_to_rq(slot, pkt_codec_flags);
        if (pkt_codec_flags != slot->flags || slot->flags == WIRE_FLAG_NOFEC_GBN || slot->flags == WIRE_FLAG_SACK_ONLY) {
            if (need_win_ack && slot->rx_sys_cnt < slot->K)
                rx_send_win_ack(comm, slot);
            return;
        }
        if (slot->flags == WIRE_FLAG_RS_CAUCHY8 && hdr.esi >= 256u) {
            if (need_win_ack && slot->rx_sys_cnt < slot->K)
                rx_send_win_ack(comm, slot);
            return;
        }

        /* cap stored repair equations per slot to prevent unbounded allocation under packet flood */
        uint32_t max_rep_cap = (slot->K > UINT32_MAX - 64u) ? UINT32_MAX : (slot->K + 64u);
        if (max_rep_cap < 256u)
            max_rep_cap = 256u;
        if (slot->rx_rep_cnt >= max_rep_cap) {
            if (need_win_ack && slot->rx_sys_cnt < slot->K)
                rx_send_win_ack(comm, slot);
            return;
        }

        bool dup = false;
        for (uint32_t r = 0; r < slot->rx_rep_cnt; r++) {
            if (slot->rep_esis[r] == hdr.esi) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            if (slot->rx_rep_cnt == slot->rep_cap) {
                uint32_t ncap = slot->rep_cap ? (slot->rep_cap * 2u) : 32u;
                if (ncap > max_rep_cap)
                    ncap = max_rep_cap;
                uint32_t *ne = (uint32_t *)malloc(ncap * sizeof(uint32_t));
                uint8_t *nd = (uint8_t *)malloc((size_t)ncap * slot->sym_T);
                if (!ne || !nd) {
                    free(nd);
                    free(ne);
                    return;
                }
                if (slot->rx_rep_cnt > 0u) {
                    memcpy(ne, slot->rep_esis, slot->rx_rep_cnt * sizeof(uint32_t));
                    memcpy(nd, slot->rep_data, (size_t)slot->rx_rep_cnt * slot->sym_T);
                }
                free(slot->rep_esis);
                free(slot->rep_data);
                slot->rep_esis = ne;
                slot->rep_data = nd;
                slot->rep_cap = ncap;
            }
            uint32_t idx = slot->rx_rep_cnt++;
            slot->rep_esis[idx] = hdr.esi;
            memcpy(slot->rep_data + (size_t)idx * slot->sym_T, pay, slot->sym_T);
            if (hdr.esi > slot->max_rep_esi)
                slot->max_rep_esi = hdr.esi;
        }
    }

    uint32_t ack_cadence = (slot->num_gens > 1u) ? VXFEC_SLICE_WIN_ACK_PKTS : VXFEC_WIN_ACK_PKTS;
    if (slot->rx_sys_cnt < slot->K && (need_win_ack || (slot->rx_pkt_cnt - slot->last_acked_pkts) >= ack_cadence))
        rx_send_win_ack(comm, slot);
}

static void rx_notify_broken_pkt(vxfec_recv_comm_t *comm, const uint8_t *pkt, size_t len)
{
    if (len < sizeof(vxfec_wire_hdr_t))
        return;
    vxfec_wire_hdr_t hdr;
    memcpy(&hdr, pkt, sizeof(hdr));
    if (hdr.magic != VXFEC_NET_MAGIC || hdr.cookie != comm->cookie || (hdr.pkt_type != PKT_DATA && hdr.pkt_type != PKT_TAIL))
        return;
    if (rx_prepare_done(comm, hdr.msg_seq))
        rx_record_done(comm, hdr.msg_seq, hdr.K, 0u, 0u, 0u, 0u, true);
    vxfec_ctrl_pkt_t ehdr = {
        .magic = VXFEC_NET_MAGIC,
        .cookie = comm->cookie,
        .pkt_type = PKT_ERR,
        .msg_seq = hdr.msg_seq,
        .tag = hdr.tag,
    };
    rx_send_ctrl(comm, &ehdr, NULL, 0, 1);
}

static void rx_quiesce_broken_ibv(vxfec_recv_comm_t *comm)
{
    if ((!comm->ibv_ep || !vxfec_ibv_ep_has_error(comm->ibv_ep)) && !comm->ibv_failed)
        return;
    comm->ibv_failed = true;
#if VXFEC_ENABLE_SIM
    while (comm->delay_q && comm->delay_cnt > 0u) {
        const vxfec_delay_data_t *dd = &comm->delay_q[comm->delay_head];
        rx_notify_broken_pkt(comm, dd->data, dd->len);
        comm->delay_head = (comm->delay_head + 1u) & (VXFEC_DELAY_RING - 1u);
        comm->delay_cnt--;
    }
    while (comm->header_delay_q && comm->header_delay_cnt > 0u) {
        const vxfec_delay_ctrl_t *hd = &comm->header_delay_q[comm->header_delay_head];
        rx_notify_broken_pkt(comm, hd->data, hd->len);
        comm->header_delay_head = (comm->header_delay_head + 1u) & (VXFEC_DELAY_RING - 1u);
        comm->header_delay_cnt--;
    }
#endif
    for (uint32_t w = 0; w < VXFEC_RX_WIRE_SLOTS / 64u; w++) {
        uint64_t m = comm->active_slot_mask[w];
        while (m) {
            uint32_t i = w * 64u + (uint32_t)__builtin_ctzll(m);
            m &= m - 1ULL;
            if (comm->slots[i].active)
                rx_fail_slot(comm, &comm->slots[i], ncclSystemError);
        }
    }
    for (uint32_t p = 0; p < comm->num_ports; p++) {
        if (comm->fds[p] < 0)
            continue;
        uint8_t hdr_buf[sizeof(vxfec_wire_hdr_t)];
        for (uint32_t b = 0; b < VXFEC_RX_BATCH; b++) {
            ssize_t n = recv(comm->fds[p], hdr_buf, sizeof(hdr_buf), MSG_DONTWAIT);
            if (n <= 0)
                break;
            rx_notify_broken_pkt(comm, hdr_buf, (size_t)n);
        }
    }
    for (uint32_t r = 0; r < NCCL_NET_MAX_REQUESTS; r++) {
        vxfec_rx_req_t *req = &comm->rx_pool[r];
        if (req->in_use && !req->done) {
            req->status = ncclSystemError;
            req->done = true;
        }
    }
    if (comm->ibv_ep && vxfec_ibv_ep_destroy(comm->ibv_ep)) {
        comm->ibv_ep = NULL;
        if (comm->ibv_ref) {
            vxf_ibv_ref_release(comm->ibv_ref);
            comm->ibv_ref = NULL;
        }
    }
}

static vxfec_rx_udp_batch_t *ensure_rx_udp_batch(vxfec_recv_comm_t *comm)
{
    if (comm->rx_udp)
        return comm->rx_udp;

    uint32_t max_wire = (uint32_t)(sizeof(vxfec_wire_hdr_ext_t) + VXFEC_MAX_SYM_T);
    uint32_t stride = max_wire;
    const vxfec_nic_dev_t *d = (comm->dev_idx >= 0 && (uint32_t)comm->dev_idx < vxf_num_devs) ? &vxf_devs[comm->dev_idx] : NULL;
    if (d && !d->is_ibv && d->mtu > 0u && d->mtu < 9000u) {
        uint32_t eff_mtu = d->mtu;
        for (uint32_t i = 0; i < comm->num_ports; i++) {
            int smtu = 0;
            socklen_t slen = sizeof(smtu);
            if (comm->fds[i] >= 0 && getsockopt(comm->fds[i], IPPROTO_IP, IP_MTU, &smtu, &slen) == 0 && smtu > (int)eff_mtu)
                eff_mtu = (uint32_t)smtu;
        }
        if (eff_mtu < 9000u && eff_mtu > sizeof(struct iphdr) + sizeof(struct udphdr)) {
            uint32_t udp_cap = eff_mtu - (uint32_t)(sizeof(struct iphdr) + sizeof(struct udphdr));
            if (udp_cap < stride)
                stride = udp_cap;
        }
    }
    stride = (stride + 63u) & ~63u;
    if (stride > max_wire)
        stride = max_wire;

    vxfec_rx_udp_batch_t *rb = (vxfec_rx_udp_batch_t *)calloc(1, sizeof(*rb));
    uint8_t *bufs = rb ? (uint8_t *)malloc((size_t)VXFEC_RX_BATCH * stride) : NULL;
    if (!rb || !bufs) {
        free(bufs);
        free(rb);
        return NULL;
    }
    rb->pkt_stride = stride;
    rb->pkt_bufs = bufs;
    for (uint32_t b = 0; b < VXFEC_RX_BATCH; b++) {
        rb->iovs[b].iov_base = bufs + (size_t)b * stride;
        rb->iovs[b].iov_len = stride;
        rb->msgs[b].msg_hdr.msg_iov = &rb->iovs[b];
        rb->msgs[b].msg_hdr.msg_iovlen = 1;
        rb->msgs[b].msg_hdr.msg_control = rb->ctrl_bufs[b];
        rb->msgs[b].msg_hdr.msg_controllen = sizeof(rb->ctrl_bufs[b]);
    }
    comm->rx_udp = rb;
    return rb;
}

#if VXFEC_ENABLE_SIM
uint64_t vxf_rx_deliver_due_headers(vxfec_recv_comm_t *comm, uint64_t now_ns)
{
    while (comm->header_delay_cnt) {
        vxfec_delay_ctrl_t *pkt = &comm->header_delay_q[comm->header_delay_head];
        if (pkt->deliver_ns > now_ns)
            return pkt->deliver_ns;
        rx_process_wire_pkt(comm, 0u, pkt->data, pkt->len, false, now_ns);
        comm->header_delay_head = (comm->header_delay_head + 1u) & (VXFEC_DELAY_RING - 1u);
        comm->header_delay_cnt--;
    }
    return 0u;
}
#endif

static void rx_receive_header(vxfec_recv_comm_t *comm, const uint8_t *buf, size_t len, uint64_t now_ns)
{
#if VXFEC_ENABLE_SIM
    uint64_t delay_ns = vxf_sim_delay_ns();
    if (delay_ns || comm->header_delay_cnt) {
        if (!comm->header_delay_q)
            comm->header_delay_q = calloc(VXFEC_DELAY_RING, sizeof(*comm->header_delay_q));
        if (!comm->header_delay_q || comm->header_delay_cnt == VXFEC_DELAY_RING || len > sizeof(comm->header_delay_q[0].data)) {
            atomic_fetch_add_explicit(&vxf_st_sim_dropped_pkts, 1, memory_order_relaxed);
            return;
        }
        uint64_t deliver_ns = now_ns + delay_ns;
        if (comm->header_delay_cnt > 0u) {
            uint32_t prev = (comm->header_delay_head + comm->header_delay_cnt - 1u) & (VXFEC_DELAY_RING - 1u);
            if (deliver_ns < comm->header_delay_q[prev].deliver_ns)
                deliver_ns = comm->header_delay_q[prev].deliver_ns;
        }
        uint32_t tail = (comm->header_delay_head + comm->header_delay_cnt) & (VXFEC_DELAY_RING - 1u);
        vxfec_delay_ctrl_t *pkt = &comm->header_delay_q[tail];
        pkt->deliver_ns = deliver_ns;
        pkt->len = (uint16_t)len;
        memcpy(pkt->data, buf, len);
        if (++comm->header_delay_cnt == 1u)
            vxf_rx_bg_schedule_delay();
        return;
    }
#endif
    rx_process_wire_pkt(comm, 0u, buf, len, false, now_ns);
}

void vxf_rx_drain_ctrl_wire(vxfec_recv_comm_t *comm, uint64_t now_ns)
{
#if VXFEC_ENABLE_SIM
    (void)vxf_rx_deliver_due_headers(comm, now_ns);
#endif
    if (comm->ctrl_fd < 0)
        return;
    uint8_t cbuf[sizeof(vxfec_wire_hdr_ext_t)];
    for (;;) {
#if VXFEC_ENABLE_SIM
        if (comm->header_delay_cnt == VXFEC_DELAY_RING)
            break;
#endif
        ssize_t n = recv(comm->ctrl_fd, cbuf, sizeof(cbuf), MSG_DONTWAIT);
        if (n <= 0)
            break;
        rx_receive_header(comm, cbuf, (size_t)n, now_ns);
    }
}

void vxf_rx_bg_peek_data_ports(vxfec_recv_comm_t *comm)
{
    uint8_t hbuf[sizeof(vxfec_wire_hdr_ext_t)];
    uint8_t sink;
    for (uint32_t p = 0; p < comm->num_ports; p++) {
        if (comm->fds[p] < 0)
            continue;
        for (;;) {
#if VXFEC_ENABLE_SIM
            if (comm->header_delay_cnt == VXFEC_DELAY_RING)
                return;
#endif
            ssize_t n = recv(comm->fds[p], hbuf, sizeof(hbuf), MSG_PEEK | MSG_DONTWAIT);
            if (n < 0)
                break;
            if ((size_t)n < sizeof(vxfec_wire_hdr_t)) {
                (void)recv(comm->fds[p], &sink, 1u, MSG_DONTWAIT);
                continue;
            }
            vxfec_wire_hdr_t hdr;
            memcpy(&hdr, hbuf, sizeof(hdr));
            if (hdr.magic != VXFEC_NET_MAGIC || hdr.cookie != comm->cookie) {
                (void)recv(comm->fds[p], &sink, 1u, MSG_DONTWAIT);
                continue;
            }
            rx_confirm_handshake(comm);

            if (hdr.pkt_type == PKT_ERR || rx_comm_was_done(comm, hdr.msg_seq, NULL, NULL, NULL, NULL, NULL, NULL) ||
                rx_tag_seq_is_retired(comm, hdr.tag, hdr.tag_seq)) {
                (void)recv(comm->fds[p], &sink, 1u, MSG_DONTWAIT);
                rx_receive_header(comm, hbuf, (size_t)n, mono_ns());
                continue;
            }
            break;
        }
    }
}

static uint32_t rx_batch_capacity(vxfec_recv_comm_t *comm, uint64_t delay_ns, uint32_t max)
{
#if VXFEC_ENABLE_SIM
    if ((delay_ns || comm->delay_cnt) && VXFEC_DELAY_RING - comm->delay_cnt < max)
        return VXFEC_DELAY_RING - comm->delay_cnt;
#endif
    return max;
}

static void rx_receive_data(vxfec_recv_comm_t *comm, uint32_t port, const uint8_t *pkt, size_t len, bool ecn, uint64_t now_ns,
                            uint64_t delay_ns)
{
#if VXFEC_ENABLE_SIM
    if (delay_ns || comm->delay_cnt) {
        if (!comm->delay_q)
            comm->delay_q = calloc(VXFEC_DELAY_RING, sizeof(*comm->delay_q));
        if (!comm->delay_q || comm->delay_cnt == VXFEC_DELAY_RING || len > sizeof(comm->delay_q[0].data)) {
            atomic_fetch_add_explicit(&vxf_st_sim_dropped_pkts, 1, memory_order_relaxed);
            return;
        }
        uint64_t deliver_ns = now_ns + delay_ns;
        if (comm->delay_cnt > 0u) {
            uint32_t prev = (comm->delay_head + comm->delay_cnt - 1u) & (VXFEC_DELAY_RING - 1u);
            if (deliver_ns < comm->delay_q[prev].deliver_ns)
                deliver_ns = comm->delay_q[prev].deliver_ns;
        }
        uint32_t tail = (comm->delay_head + comm->delay_cnt++) & (VXFEC_DELAY_RING - 1u);
        vxfec_delay_data_t *entry = &comm->delay_q[tail];
        entry->deliver_ns = deliver_ns;
        entry->port_idx = port;
        entry->len = (uint16_t)len;
        entry->is_ecn_ce = ecn;
        memcpy(entry->data, pkt, len);
        return;
    }
#endif
    rx_process_wire_pkt(comm, port, pkt, len, ecn, now_ns);
}

void vxf_progress_recv_comm(vxfec_recv_comm_t *comm)
{
    if (comm->ibv_failed || (comm->ibv_ep && vxfec_ibv_ep_has_error(comm->ibv_ep))) {
        rx_quiesce_broken_ibv(comm);
        return;
    }

    uint64_t now_ns = mono_ns();
    uint64_t delay_ns = vxf_sim_delay_ns();
    vxf_rx_drain_ctrl_wire(comm, now_ns);

    if (comm->ibv_ep) {
        vxfec_ibv_rx_wc_t wcs[64];
        for (;;) {
            uint32_t capacity = rx_batch_capacity(comm, delay_ns, 64u);
            if (!capacity)
                break;
            uint32_t nw = vxfec_ibv_ep_poll_rx(comm->ibv_ep, wcs, capacity);
            for (uint32_t i = 0; i < nw; i++) {
                uint32_t qp_idx = vxfec_ibv_slot_qp(wcs[i].slot_id);
                rx_receive_data(comm, qp_idx, wcs[i].data, wcs[i].byte_len, false, now_ns, delay_ns);
                if (!vxfec_ibv_ep_repost_rx(comm->ibv_ep, wcs[i].slot_id))
                    break;
            }
            if (vxfec_ibv_ep_has_error(comm->ibv_ep)) {
                rx_quiesce_broken_ibv(comm);
                return;
            }
            if (nw == 0u)
                break;
        }
    }

    for (uint32_t p = 0; p < comm->num_ports; p++) {
        if (comm->fds[p] < 0)
            continue;
        if (!comm->rx_udp && comm->ibv_ep) {
            uint8_t peek = 0;
            if (recv(comm->fds[p], &peek, 1, MSG_PEEK | MSG_DONTWAIT) < 0)
                continue;
        }
        vxfec_rx_udp_batch_t *rb = ensure_rx_udp_batch(comm);
        if (!rb)
            break;
        for (;;) {
            uint32_t capacity = rx_batch_capacity(comm, delay_ns, VXFEC_RX_BATCH);
            if (!capacity)
                break;
            int n = recvmmsg(comm->fds[p], rb->msgs, capacity, MSG_DONTWAIT, NULL);
            if (n <= 0)
                break;

            for (int i = 0; i < n; i++) {
                bool is_ecn_ce = extract_rx_ecn_ce(&rb->msgs[i].msg_hdr);
                rb->msgs[i].msg_hdr.msg_control = rb->ctrl_bufs[i];
                rb->msgs[i].msg_hdr.msg_controllen = sizeof(rb->ctrl_bufs[i]);
                size_t len = rb->msgs[i].msg_len;
                const uint8_t *pkt = rb->pkt_bufs + (size_t)i * rb->pkt_stride;

                rx_receive_data(comm, p, pkt, len, is_ecn_ce, now_ns, delay_ns);
            }
        }
    }

#if VXFEC_ENABLE_SIM
    while (comm->delay_q && comm->delay_cnt > 0u) {
        vxfec_delay_data_t *front = &comm->delay_q[comm->delay_head];
        if (now_ns < front->deliver_ns)
            break;
        rx_process_wire_pkt(comm, front->port_idx, front->data, front->len, front->is_ecn_ce, now_ns);
        comm->delay_head = (comm->delay_head + 1u) & (VXFEC_DELAY_RING - 1u);
        comm->delay_cnt--;
    }
#endif

    for (uint32_t w = 0; w < VXFEC_RX_WIRE_SLOTS / 64u; w++) {
        uint64_t m = comm->active_slot_mask[w];
        while (m) {
            uint32_t i = w * 64u + (uint32_t)__builtin_ctzll(m);
            m &= m - 1ULL;
            if (comm->slots[i].active)
                rx_evaluate_slot(comm, &comm->slots[i], now_ns);
        }
    }
}

static bool tx_req_has_recovery_data(const vxfec_tx_slot_t *r)
{
    return r->sack_cnt > 0u || r->ir_rem > 0u || r->gbn_rem > 0u;
}

static bool tx_req_has_paced_work(const vxfec_tx_slot_t *r)
{
    return r->tx_gpu != NULL || !r->initial_tail_sent || tx_req_has_recovery_data(r) || r->need_recovery_tail;
}

static void tx_release_req_inflight(vxfec_send_comm_t *comm, vxfec_tx_slot_t *req)
{
    comm->inflight_bytes = (req->queued_bytes >= comm->inflight_bytes) ? 0u : (comm->inflight_bytes - req->queued_bytes);
    req->queued_bytes = 0u;
}

static void tx_fail_req(vxfec_send_comm_t *comm, vxfec_tx_slot_t *req, ncclResult_t st)
{
    vxfec_tx_user_req_t *ureq = req->parent;
    uint32_t fail_msg_seq = req->msg_seq;
    uint32_t fail_tag_seq = req->tag_seq;
    uint32_t fail_tag = req->tag;
    tx_release_req_inflight(comm, req);
    req->status = st;
    tx_deactivate_slot(comm, req, true);

    if (st != ncclRemoteError)
        tx_record_and_send_abort(comm, fail_msg_seq, fail_tag_seq, fail_tag);

    if (ureq) {
        uint64_t m = comm->active_tx_mask;
        while (m) {
            uint32_t s = (uint32_t)__builtin_ctzll(m);
            m &= m - 1ULL;
            if (comm->tx_pool[s].in_use && comm->tx_pool[s].parent == ureq) {
                tx_release_req_inflight(comm, &comm->tx_pool[s]);
                tx_deactivate_slot(comm, &comm->tx_pool[s], true);
            }
        }
        ureq->active_gens = 0u;
        if (!ureq->failed) {
            ureq->failed = true;
            ureq->status = st;
        }
        ureq->completion_ready = true;
        if (!ureq->ibv_zc_pending ||
            (!comm->ibv_failed && (!comm->ibv_ep || vxfec_ibv_ep_tx_done(comm->ibv_ep, ureq->ibv_target_seq)))) {
            ureq->ibv_zc_pending = false;
            ureq->done = true;
        }
    }
}

/* round-robin generation spawner with reserved per-tag wire headroom (HoL anti-deadlock) */
static void tx_spawn_ready_generations(vxfec_send_comm_t *comm)
{
    uint32_t active_user_reqs = 0u;
    for (uint32_t u = 0; u < NCCL_NET_MAX_REQUESTS; u++) {
        const vxfec_tx_user_req_t *ureq = &comm->user_tx_pool[u];
        if (ureq->in_use && !ureq->failed && !ureq->completion_ready && !ureq->done)
            active_user_reqs++;
    }
    if (active_user_reqs == 0u)
        return;

    uint32_t reserved_idle = (NCCL_NET_MAX_REQUESTS > active_user_reqs) ? (NCCL_NET_MAX_REQUESTS - active_user_reqs) : 0u;
    uint32_t pool_budget = (VXFEC_TX_WIRE_SLOTS > reserved_idle) ? (VXFEC_TX_WIRE_SLOTS - reserved_idle) : active_user_reqs;
    uint32_t per_req_cap = pool_budget / active_user_reqs;
    if (per_req_cap == 0u)
        per_req_cap = 1u;
    if (per_req_cap > VXFEC_GEN_PIPE_DEPTH)
        per_req_cap = VXFEC_GEN_PIPE_DEPTH;

    for (;;) {
        uint64_t free_mask = ~comm->active_tx_mask;
        if (free_mask == 0ULL)
            return;

        bool spawned_any = false;
        for (uint32_t step = 0; step < NCCL_NET_MAX_REQUESTS; step++) {
            uint32_t u = (comm->spawn_rr_cursor + step) % NCCL_NET_MAX_REQUESTS;
            vxfec_tx_user_req_t *ureq = &comm->user_tx_pool[u];
            if (!ureq->in_use || ureq->failed || ureq->completion_ready || ureq->done)
                continue;
            if (ureq->next_gen_to_spawn >= ureq->num_gens || ureq->active_gens >= per_req_cap)
                continue;

            free_mask = ~comm->active_tx_mask;
            if (free_mask == 0ULL)
                return;
            uint32_t slot_idx = (uint32_t)__builtin_ctzll(free_mask);

            uint16_t g = ureq->next_gen_to_spawn;
            size_t gen_stride = (size_t)VXFEC_RS8_WIN_K * ureq->sym_T;
            size_t gen_off = (ureq->num_gens > 1u) ? ((size_t)g * gen_stride) : 0u;
            uint32_t gen_bytes = (ureq->num_gens == 1u) ? ureq->msg_total_bytes
                                                        : ((g + 1u < ureq->num_gens) ? (uint32_t)gen_stride
                                                                                     : (ureq->msg_total_bytes - (uint32_t)gen_off));
            uint32_t K = (gen_bytes == 0u) ? 1u : (uint32_t)(((uint64_t)gen_bytes + ureq->sym_T - 1u) / ureq->sym_T);

            vxfec_tx_slot_t *req = &comm->tx_pool[slot_idx];
            memset(req, 0, sizeof(*req));
            req->slot_idx = (uint8_t)slot_idx;
            req->in_use = true;
            req->wire_active = true;
            req->comm = comm;
            req->parent = ureq;
            req->msg_seq = next_non_zero_seq(&comm->next_msg_seq);
            req->tag = ureq->tag;
            req->tag_seq = ureq->tag_seq;
            req->msg_total_bytes = ureq->msg_total_bytes;
            req->total_bytes = gen_bytes;
            req->K = K;
            req->sym_T = ureq->sym_T;
            req->gen_idx = g;
            req->num_gens = ureq->num_gens;
            req->codec = choose_tx_codec(K, comm->features, &req->wire_flags);
            req->next_rep_esi = K;
            req->mr = ureq->mr;
            vxf_mr_retain(ureq->mr);
            req->src_host = (ureq->src_host && gen_bytes > 0u) ? (ureq->src_host + gen_off) : ureq->src_host;
            req->src_ibv_ptr = (ureq->src_ibv_ptr && gen_bytes > 0u) ? (ureq->src_ibv_ptr + gen_off) : ureq->src_ibv_ptr;
            req->src_ibv_lkey = ureq->src_ibv_lkey;
            req->src_ibv_dmabuf = ureq->src_ibv_dmabuf;

            if (gen_bytes > 0u && req->wire_flags != WIRE_FLAG_NOFEC_GBN && req->wire_flags != WIRE_FLAG_SACK_ONLY) {
                uint32_t slack = wire_flags_ir_slack(req->wire_flags, K);
                uint32_t max_parity = (req->codec == VXFEC_CODEC_RS_CAUCHY8) ? (256u - K) : UINT32_MAX;
                uint32_t clamped = 0u;
                req->proactive_target =
                    harq_paced_iid_parity(K, comm->loss_ewma, slack, max_parity, comm->pacer.cong_level, &clamped);
                if (clamped > 0u)
                    atomic_fetch_add_explicit(&vxf_st_tx_fec_clamped_pkts, clamped, memory_order_relaxed);
            }

            comm->active_tx_mask |= (1ULL << slot_idx);
            tx_hash_insert(comm, slot_idx, req->msg_seq);

            ureq->next_gen_to_spawn++;
            ureq->active_gens++;
            comm->spawn_rr_cursor = (u + 1u) % NCCL_NET_MAX_REQUESTS;
            spawned_any = true;
        }
        if (!spawned_any)
            break;
    }
}

static bool tx_poll_gpu_repair(vxfec_tx_slot_t *req)
{
    int status = vxf_gpu_job_poll(req->tx_gpu);
    if (status == 0)
        return false;
    if (status > 0) {
        req->tx_gpu_ready = true;
    } else {
        vxf_gpu_job_free(req->tx_gpu);
        req->tx_gpu = NULL;
        req->tx_gpu_failed = true;
    }
    return true;
}

static void tx_account_repair_slice(vxfec_tx_slot_t *req, uint32_t count, bool proactive, uint64_t now_ns)
{
    vxfec_send_comm_t *comm = req->comm;
    uint64_t bytes = (uint64_t)count * req->sym_T;
    if (proactive) {
        req->tx_proactive_cursor += count;
    } else {
        req->ir_start_esi += count;
        req->ir_rem -= count;
        req->last_ir_tx_ns = now_ns;
    }
    harq_pacer_consume(&comm->pacer, bytes);
    comm->inflight_bytes += bytes;
    req->queued_bytes += bytes;
    req->last_send_ns = now_ns;
}

static void tx_drain_paced(vxfec_send_comm_t *comm, uint64_t now_ns)
{
    uint64_t waiting_mask = 0ULL;
    for (;;) {
        vxfec_tx_slot_t *req = NULL;
        uint32_t selected = 0u;
        uint32_t best_seq = 0u;
        bool best_is_rec = false;

        uint64_t cand = comm->active_tx_mask & ~waiting_mask;
        while (cand) {
            uint32_t s = (uint32_t)__builtin_ctzll(cand);
            cand &= cand - 1ULL;
            vxfec_tx_slot_t *r = &comm->tx_pool[s];
            if (!r->wire_active || r->done || !tx_req_has_paced_work(r))
                continue;
            if (r->tx_gpu && !r->tx_gpu_ready && !tx_poll_gpu_repair(r)) {
                waiting_mask |= (1ULL << s);
                continue;
            }
            bool is_rec = r->initial_tail_sent;
            if (!req || (is_rec && !best_is_rec) || (is_rec == best_is_rec && seq_before(r->msg_seq, best_seq))) {
                req = r;
                selected = s;
                best_seq = r->msg_seq;
                best_is_rec = is_rec;
            }
        }
        if (!req)
            break;

        if (!req->initial_tail_sent && req->total_bytes == 0u) {
            if (!tx_spray_sys_range(req, 0u, NULL, 1u, false, false, now_ns)) {
                tx_fail_req(comm, req, ncclSystemError);
                continue;
            }
            req->tx_sys_cursor = req->K;
            req->burst_end_tx_ns = now_ns;
            req->last_send_ns = now_ns;
            tx_send_tail(req);
            req->initial_tail_sent = true;
            continue;
        }

        if (req->need_recovery_tail && !req->tx_gpu && !tx_req_has_recovery_data(req)) {
            tx_send_tail(req);
            req->need_recovery_tail = false;
            continue;
        }

        uint64_t budget = harq_pacer_budget(&comm->pacer, comm->inflight_bytes, now_ns);
        if (comm->inflight_bytes == 0u && comm->pacer.cwnd_bytes < req->sym_T && comm->pacer.tokens_bytes >= (double)req->sym_T)
            budget = req->sym_T;

        uint32_t max_pkts = (uint32_t)(budget / req->sym_T);
        if (max_pkts == 0u) {
            if (comm->inflight_bytes + req->sym_T > comm->pacer.cwnd_bytes) {
                uint64_t rto_ns = tx_tail_rto_ns(comm);
                bool released_any = false;
                uint64_t im = comm->active_tx_mask;
                while (im) {
                    uint32_t s = (uint32_t)__builtin_ctzll(im);
                    im &= im - 1ULL;
                    vxfec_tx_slot_t *infl = &comm->tx_pool[s];
                    if (infl->wire_active && !infl->done && infl->queued_bytes > 0u && infl->last_send_ns > 0u &&
                        (now_ns - infl->last_send_ns) >= rto_ns) {
                        infl->ack_probe_tx_ns = 0u;
                        tx_release_req_inflight(comm, infl);
                        released_any = true;
                    }
                }
                if (released_any)
                    continue;
            }
            break;
        }

        if (req->tx_gpu) {
            uint32_t remaining = req->tx_gpu_count - req->tx_gpu_offset;
            uint32_t n = (remaining > max_pkts) ? max_pkts : remaining;
            uint64_t bytes = (uint64_t)n * req->sym_T;
            bool proactive = req->tx_gpu_proactive;
            bool more_gens = (req->parent && (req->gen_idx + 1u < req->num_gens));
            bool need_ack = (comm->inflight_bytes + bytes + req->sym_T > comm->pacer.cwnd_bytes) &&
                            (!proactive || req->tx_proactive_cursor + n < req->proactive_target || more_gens);
            const uint8_t *out = vxfec_vk_buf_staging_ptr(req->tx_gpu->out_buf);
            if (!tx_spray_coded_repairs(req, req->tx_gpu_flags, req->tx_gpu_esi + req->tx_gpu_offset, n,
                                        out + (size_t)req->tx_gpu_offset * req->tx_gpu_stride, req->tx_gpu_stride, proactive,
                                        need_ack, now_ns)) {
                tx_fail_req(comm, req, ncclSystemError);
                continue;
            }
            tx_account_repair_slice(req, n, proactive, now_ns);
            req->tx_gpu_offset += n;
            max_pkts -= n;
            if (req->tx_gpu_offset == req->tx_gpu_count) {
                vxf_gpu_job_free(req->tx_gpu);
                req->tx_gpu = NULL;
            }
            if (n < remaining || (!proactive && req->ir_rem > 0u))
                atomic_fetch_add_explicit(&vxf_st_tx_paced_slices, 1, memory_order_relaxed);
        }

        if (req->sack_cnt > 0u && max_pkts > 0u) {
            uint32_t n_sack = (req->sack_cnt > max_pkts) ? max_pkts : req->sack_cnt;
            uint64_t sent_b = (uint64_t)n_sack * req->sym_T;
            bool need_ack = (comm->inflight_bytes + sent_b + req->sym_T > comm->pacer.cwnd_bytes);
            if (!tx_spray_sys_range(req, 0u, req->sack_queue + req->sack_head, n_sack, true, need_ack, now_ns)) {
                tx_fail_req(comm, req, ncclSystemError);
                continue;
            }
            req->sack_head += n_sack;
            req->sack_cnt -= n_sack;
            req->last_sack_tx_ns = now_ns;
            harq_pacer_consume(&comm->pacer, sent_b);
            comm->inflight_bytes += sent_b;
            req->queued_bytes += sent_b;
            req->last_send_ns = now_ns;
            max_pkts -= n_sack;
            if (req->sack_cnt > 0u)
                atomic_fetch_add_explicit(&vxf_st_tx_paced_slices, 1, memory_order_relaxed);
        }

        if (req->ir_rem > 0u && max_pkts > 0u) {
            uint32_t n_ir = (req->ir_rem > max_pkts) ? max_pkts : req->ir_rem;
            if (n_ir > VXFEC_BATCH_PKTS)
                n_ir = VXFEC_BATCH_PKTS;
            uint64_t sent_b = (uint64_t)n_ir * req->sym_T;
            bool need_ack = (comm->inflight_bytes + sent_b + req->sym_T > comm->pacer.cwnd_bytes);
            tx_repair_result_t result = tx_generate_and_spray_repairs(req, req->ir_start_esi, n_ir, false, need_ack, now_ns);
            if (result == TX_REPAIR_GPU_QUEUED) {
                waiting_mask |= (1ULL << selected);
                continue;
            }
            if (result == TX_REPAIR_FAILED) {
                tx_fail_req(comm, req, ncclSystemError);
                continue;
            }
            tx_account_repair_slice(req, n_ir, false, now_ns);
            max_pkts -= n_ir;
            if (req->ir_rem > 0u)
                atomic_fetch_add_explicit(&vxf_st_tx_paced_slices, 1, memory_order_relaxed);
        }

        if (req->gbn_rem > 0u && max_pkts > 0u) {
            uint32_t n_gbn = (req->gbn_rem > max_pkts) ? max_pkts : req->gbn_rem;
            uint64_t sent_b = (uint64_t)n_gbn * req->sym_T;
            bool need_ack = (comm->inflight_bytes + sent_b + req->sym_T > comm->pacer.cwnd_bytes);
            if (!tx_spray_sys_range(req, req->gbn_cursor, NULL, n_gbn, true, need_ack, now_ns)) {
                tx_fail_req(comm, req, ncclSystemError);
                continue;
            }
            req->gbn_cursor += n_gbn;
            req->gbn_rem -= n_gbn;
            harq_pacer_consume(&comm->pacer, sent_b);
            comm->inflight_bytes += sent_b;
            req->queued_bytes += sent_b;
            req->last_send_ns = now_ns;
            max_pkts -= n_gbn;
            if (req->gbn_rem > 0u)
                atomic_fetch_add_explicit(&vxf_st_tx_paced_slices, 1, memory_order_relaxed);
        }

        if (req->need_recovery_tail && !req->tx_gpu && !tx_req_has_recovery_data(req)) {
            tx_send_tail(req);
            req->need_recovery_tail = false;
            now_ns = req->last_tail_ns;
            continue;
        }

        if (!req->initial_tail_sent) {
            bool was_partial = (req->tx_sys_cursor > 0u || req->tx_proactive_cursor > 0u);

            if (req->tx_sys_cursor < req->K && max_pkts > 0u) {
                uint32_t rem_sys = req->K - req->tx_sys_cursor;
                uint32_t n_sys = (rem_sys > max_pkts) ? max_pkts : rem_sys;
                uint64_t sent_b = (uint64_t)n_sys * req->sym_T;
                bool more_gens = (req->parent && (req->gen_idx + 1u < req->num_gens));
                bool need_ack = (comm->inflight_bytes + sent_b + req->sym_T > comm->pacer.cwnd_bytes) &&
                                (req->tx_sys_cursor + n_sys < req->K || req->proactive_target > 0u || more_gens);
                if (req->tx_sys_cursor == 0u && comm->pacer.srtt_ns == 0u)
                    req->first_sys_tx_ns = mono_ns();
                if (!tx_spray_sys_range(req, req->tx_sys_cursor, NULL, n_sys, false, need_ack, now_ns)) {
                    tx_fail_req(comm, req, ncclSystemError);
                    continue;
                }
                req->tx_sys_cursor += n_sys;
                harq_pacer_consume(&comm->pacer, sent_b);
                comm->inflight_bytes += sent_b;
                req->queued_bytes += sent_b;
                req->last_send_ns = now_ns;
                max_pkts -= n_sys;
                if (was_partial) {
                    atomic_fetch_add_explicit(&vxf_st_tx_paced_slices, 1, memory_order_relaxed);
                    was_partial = false;
                }
            }

            if (req->tx_sys_cursor == req->K && req->tx_proactive_cursor < req->proactive_target && max_pkts > 0u) {
                uint32_t rem_rep = req->proactive_target - req->tx_proactive_cursor;
                uint32_t n_rep = (rem_rep > max_pkts) ? max_pkts : rem_rep;
                if (n_rep > VXFEC_BATCH_PKTS)
                    n_rep = VXFEC_BATCH_PKTS;
                uint64_t sent_b = (uint64_t)n_rep * req->sym_T;
                bool more_gens = (req->parent && (req->gen_idx + 1u < req->num_gens));
                bool need_ack = (comm->inflight_bytes + sent_b + req->sym_T > comm->pacer.cwnd_bytes) &&
                                (req->tx_proactive_cursor + n_rep < req->proactive_target || more_gens);
                uint32_t start_esi = req->K + req->tx_proactive_cursor;
                tx_repair_result_t result = tx_generate_and_spray_repairs(req, start_esi, n_rep, true, need_ack, now_ns);
                if (result == TX_REPAIR_GPU_QUEUED) {
                    waiting_mask |= (1ULL << selected);
                    continue;
                }
                if (result == TX_REPAIR_FAILED) {
                    tx_fail_req(comm, req, ncclSystemError);
                    continue;
                }
                tx_account_repair_slice(req, n_rep, true, now_ns);
                max_pkts -= n_rep;
                if (was_partial)
                    atomic_fetch_add_explicit(&vxf_st_tx_paced_slices, 1, memory_order_relaxed);
            }

            if (req->tx_sys_cursor == req->K && req->tx_proactive_cursor == req->proactive_target) {
                req->burst_end_tx_ns = mono_ns();
                tx_send_tail(req);
                req->initial_tail_sent = true;
                now_ns = req->burst_end_tx_ns;
                continue;
            }
        }

        if (max_pkts == 0u)
            break;
    }
}

static void tx_update_ecn_port_weights(vxfec_tx_slot_t *req, uint32_t reported_mask, uint32_t delta_ce)
{
    vxfec_send_comm_t *comm = req->comm;
    uint32_t hot_ports = 0u;
    if (delta_ce > 0u) {
        hot_ports = reported_mask & ~req->win_ecn_port_mask;
        req->win_ecn_port_mask |= reported_mask;
        if (hot_ports == 0u && (reported_mask & (reported_mask - 1u)) == 0u)
            hot_ports = reported_mask;
        if (hot_ports == 0u)
            return;
    }
    if (hot_ports != 0u || comm->has_nonuniform_weights)
        comm->has_nonuniform_weights =
            harq_update_port_weights(comm->port_weights, comm->num_ports, hot_ports, VXFEC_PORT_WEIGHT_MAX);
}

/* seed retry RTT from first cold-flight feedback while leaving congestion control to probe/DONE samples */
static void tx_bootstrap_srtt(vxfec_tx_slot_t *req, uint64_t now_ns)
{
    uint64_t ref_ns = req->burst_end_tx_ns ? req->burst_end_tx_ns : req->first_sys_tx_ns;
    if (req->comm->pacer.srtt_ns == 0u && !req->retx_or_ir_sent && ref_ns && now_ns > ref_ns)
        req->comm->pacer.srtt_ns = now_ns - ref_ns;
}

/* suppress duplicate feedback while repairs covering the same holes are still queued or in flight */
static bool tx_recovery_in_flight(const vxfec_send_comm_t *comm, uint32_t pending, uint64_t last_tx_ns, uint64_t now_ns)
{
    uint64_t window_ns = clamp_u64(comm->pacer.srtt_ns, 300000ULL, 20000000ULL);
    return pending || (last_tx_ns && now_ns - last_tx_ns < window_ns);
}

static bool tx_sack_subset_in_queue(const uint32_t *esis, uint32_t valid, const uint32_t *queued, uint32_t queued_total)
{
    if (valid > queued_total)
        return false;
    for (uint32_t i = 0; i < valid; i++) {
        bool found = false;
        for (uint32_t j = 0; j < queued_total; j++) {
            if (esis[i] == queued[j]) {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }
    return true;
}

static void tx_queue_sack(vxfec_tx_slot_t *req, uint32_t count, const uint8_t *payload, size_t len, uint64_t now_ns)
{
    uint32_t esis[VXFEC_SACK_MAX];
    if (count > VXFEC_SACK_MAX)
        count = VXFEC_SACK_MAX;
    if (len < count * sizeof(*esis))
        return;
    memcpy(esis, payload, count * sizeof(*esis));
    uint32_t valid = 0u;
    for (uint32_t i = 0; i < count; i++) {
        if (esis[i] < req->K)
            esis[valid++] = esis[i];
    }
    if (!valid)
        return;
    if (!req->sack_queue) {
        req->sack_queue = malloc(sizeof(esis));
        if (!req->sack_queue)
            return;
    } else if (tx_sack_subset_in_queue(esis, valid, req->sack_queue, req->sack_head + req->sack_cnt) &&
               tx_recovery_in_flight(req->comm, req->sack_cnt, req->last_sack_tx_ns, now_ns)) {
        return;
    }
    tx_bootstrap_srtt(req, now_ns);
    req->ack_probe_tx_ns = 0u;
    tx_release_req_inflight(req->comm, req);
    memcpy(req->sack_queue, esis, valid * sizeof(*esis));
    req->sack_head = 0u;
    req->sack_cnt = valid;
    req->need_recovery_tail = true;
    req->retx_or_ir_sent = true;
}

static void tx_quiesce_broken_ibv(vxfec_send_comm_t *comm)
{
    if (!comm->ibv_ep || !vxfec_ibv_ep_has_error(comm->ibv_ep))
        return;

    if (!comm->ibv_failed) {
        VXF_LOG_WARN("vxfec: fatal verbs TX CQ error on dev %d; tearing down QPs and failing active sends", comm->dev_idx);
        comm->ibv_failed = true;
    }
    bool qps_gone = vxfec_ibv_ep_destroy(comm->ibv_ep);
    if (qps_gone) {
        comm->ibv_ep = NULL;
        if (comm->ibv_ref) {
            vxf_ibv_ref_release(comm->ibv_ref);
            comm->ibv_ref = NULL;
        }
    }

    uint64_t m = comm->active_tx_mask;
    while (m) {
        uint32_t s = (uint32_t)__builtin_ctzll(m);
        m &= m - 1ULL;
        if (comm->tx_pool[s].in_use) {
            if (qps_gone || comm->tx_pool[s].src_ibv_lkey == 0u) {
                tx_fail_req(comm, &comm->tx_pool[s], ncclSystemError);
            } else {
                tx_record_and_send_abort(comm, comm->tx_pool[s].msg_seq, comm->tx_pool[s].tag_seq, comm->tx_pool[s].tag);
                tx_release_req_inflight(comm, &comm->tx_pool[s]);
                tx_deactivate_slot(comm, &comm->tx_pool[s], false);
                if (comm->tx_pool[s].parent && !comm->tx_pool[s].parent->failed) {
                    comm->tx_pool[s].parent->failed = true;
                    comm->tx_pool[s].parent->status = ncclSystemError;
                }
            }
        }
    }
    for (uint32_t u = 0; u < NCCL_NET_MAX_REQUESTS; u++) {
        vxfec_tx_user_req_t *ureq = &comm->user_tx_pool[u];
        if (ureq->in_use && !ureq->done) {
            tx_record_and_send_abort(comm, 0u, ureq->tag_seq, ureq->tag);
            ureq->active_gens = 0u;
            if (!ureq->failed) {
                ureq->failed = true;
                ureq->status = ncclSystemError;
            }
            ureq->completion_ready = true;
            if (qps_gone || !ureq->ibv_zc_pending) {
                ureq->ibv_zc_pending = false;
                ureq->done = true;
            }
        }
    }
}

void vxf_progress_send_comm(vxfec_send_comm_t *comm)
{
    uint8_t ctrl_buf[sizeof(vxfec_ctrl_pkt_t) + VXFEC_SACK_MAX * sizeof(uint32_t)];
    uint64_t now_ns = mono_ns();
    uint64_t delay_ns = vxf_sim_delay_ns();

    if (comm->ibv_ep) {
        vxfec_ibv_ep_drain_tx(comm->ibv_ep);
        tx_quiesce_broken_ibv(comm);
    }
    if (comm->ibv_failed) {
        while (comm->ctrl_fd >= 0 && comm->unacked_aborts > 0u) {
            ssize_t r = recv(comm->ctrl_fd, ctrl_buf, sizeof(ctrl_buf), MSG_DONTWAIT);
            if (r <= 0)
                break;
            if ((size_t)r < sizeof(vxfec_ctrl_pkt_t))
                continue;
            vxfec_ctrl_pkt_t ctrl;
            memcpy(&ctrl, ctrl_buf, sizeof(ctrl));
            if (ctrl.magic == VXFEC_NET_MAGIC && ctrl.cookie == comm->cookie && ctrl.pkt_type == PKT_ERR)
                tx_ack_abort_locked(comm, ctrl.msg_seq, ctrl.tag);
        }
        tx_retransmit_aborts_locked(comm, now_ns);
        return;
    }

    uint32_t cur_epoch = vxf_sim_epoch();
    if (comm->cfg_epoch != cur_epoch)
        vxf_init_send_comm_pacer(comm, now_ns);

    tx_spawn_ready_generations(comm);

    for (;;) {
        ssize_t n = -1;
        bool can_recv = comm->ctrl_fd >= 0;
#if VXFEC_ENABLE_SIM
        can_recv = can_recv && comm->ctrl_delay_cnt < VXFEC_DELAY_RING;
#endif
        if (can_recv) {
            ssize_t r = recv(comm->ctrl_fd, ctrl_buf, sizeof(ctrl_buf), MSG_DONTWAIT);
#if VXFEC_ENABLE_SIM
            if (r > 0 && (delay_ns > 0u || comm->ctrl_delay_cnt > 0u)) {
                if (!comm->ctrl_delay_q)
                    comm->ctrl_delay_q = (vxfec_delay_ctrl_t *)calloc(VXFEC_DELAY_RING, sizeof(*comm->ctrl_delay_q));
                if (comm->ctrl_delay_q && (size_t)r <= sizeof(comm->ctrl_delay_q[0].data)) {
                    uint64_t deliver_ns = now_ns + delay_ns;
                    if (comm->ctrl_delay_cnt > 0u) {
                        uint32_t prev = (comm->ctrl_delay_head + comm->ctrl_delay_cnt - 1u) & (VXFEC_DELAY_RING - 1u);
                        if (deliver_ns < comm->ctrl_delay_q[prev].deliver_ns)
                            deliver_ns = comm->ctrl_delay_q[prev].deliver_ns;
                    }
                    uint32_t dst = (comm->ctrl_delay_head + comm->ctrl_delay_cnt++) & (VXFEC_DELAY_RING - 1u);
                    comm->ctrl_delay_q[dst].deliver_ns = deliver_ns;
                    comm->ctrl_delay_q[dst].len = (uint16_t)r;
                    memcpy(comm->ctrl_delay_q[dst].data, ctrl_buf, (size_t)r);
                } else {
                    atomic_fetch_add_explicit(&vxf_st_sim_dropped_pkts, 1, memory_order_relaxed);
                }
                continue;
            }
#else
            (void)delay_ns;
#endif
            n = r;
        }
        if (n <= 0) {
#if VXFEC_ENABLE_SIM
            if (comm->ctrl_delay_q && comm->ctrl_delay_cnt > 0u && now_ns >= comm->ctrl_delay_q[comm->ctrl_delay_head].deliver_ns) {
                vxfec_delay_ctrl_t *front = &comm->ctrl_delay_q[comm->ctrl_delay_head];
                n = (ssize_t)front->len;
                memcpy(ctrl_buf, front->data, front->len);
                comm->ctrl_delay_head = (comm->ctrl_delay_head + 1u) & (VXFEC_DELAY_RING - 1u);
                comm->ctrl_delay_cnt--;
            } else
#endif
            {
                break;
            }
        }
        if ((size_t)n < sizeof(vxfec_ctrl_pkt_t))
            continue;
        vxfec_ctrl_pkt_t ctrl;
        memcpy(&ctrl, ctrl_buf, sizeof(ctrl));
        if (ctrl.magic != VXFEC_NET_MAGIC || ctrl.cookie != comm->cookie || ctrl.pkt_type == PKT_HELLO_ACK)
            continue;

        vxfec_tx_slot_t *req = tx_hash_lookup(comm, ctrl.msg_seq);
        if (!req || req->done) {
            if (ctrl.pkt_type == PKT_ERR) {
                tx_ack_abort_locked(comm, ctrl.msg_seq, ctrl.tag);
            } else if (ctrl.pkt_type != PKT_DONE) {
                for (uint32_t a = 0; a < VXFEC_ABORT_RING; a++) {
                    if (comm->tx_aborts[a].msg_seq == ctrl.msg_seq && comm->tx_aborts[a].tag == ctrl.tag &&
                        comm->tx_aborts[a].tag_seq != 0u) {
                        (void)tx_emit_abort_pkt(comm, ctrl.msg_seq, comm->tx_aborts[a].tag_seq, ctrl.tag, 1);
                        break;
                    }
                }
            }
            continue;
        }

        if (ctrl.pkt_type == PKT_WIN_ACK) {
            uint32_t init_sent = req->tx_sys_cursor + req->tx_proactive_cursor;
            uint32_t init_clear = (ctrl.esi < init_sent) ? ctrl.esi : init_sent;
            uint32_t newly_cleared = (init_clear > req->win_init_cleared) ? (init_clear - req->win_init_cleared) : 0u;
            if (newly_cleared > 0u)
                req->win_init_cleared = init_clear;

            uint32_t newly_rx = (ctrl.rx_pkts > req->win_rx_pkts) ? (ctrl.rx_pkts - req->win_rx_pkts) : 0u;
            uint32_t credit_pkts = req->retx_or_ir_sent ? newly_rx : ((newly_cleared > newly_rx) ? newly_cleared : newly_rx);
            if (credit_pkts > 0u) {
                uint64_t credit_b = (uint64_t)credit_pkts * req->sym_T;
                if (credit_b > req->queued_bytes)
                    credit_b = req->queued_bytes;
                req->queued_bytes -= credit_b;
                comm->inflight_bytes = (credit_b >= comm->inflight_bytes) ? 0u : (comm->inflight_bytes - credit_b);
            }

            if (newly_rx > 0u) {
                uint32_t delta_ce = (ctrl.ecn_ce_pkts > req->win_ecn_ce) ? (ctrl.ecn_ce_pkts - req->win_ecn_ce) : 0u;
                req->win_rx_pkts = ctrl.rx_pkts;
                req->win_ecn_ce += delta_ce;

                uint64_t acked_b = (uint64_t)newly_rx * req->sym_T;
                if (req->win_acked_bytes + acked_b > req->total_bytes)
                    acked_b = (req->total_bytes > req->win_acked_bytes) ? (req->total_bytes - req->win_acked_bytes) : 0u;
                req->win_acked_bytes += acked_b;

                uint64_t rtt_ns = 0u;
                if (!req->retx_or_ir_sent && req->ack_probe_tx_ns > 0u && now_ns > req->ack_probe_tx_ns) {
                    uint64_t raw_rtt = now_ns - req->ack_probe_tx_ns;
                    uint64_t dwell_ns = (uint64_t)ctrl.dwell_us * 1000ULL;
                    rtt_ns = (raw_rtt > dwell_ns) ? (raw_rtt - dwell_ns) : 1000ULL;
                    req->ack_probe_tx_ns = 0u;
                }
                if (harq_pacer_on_ack(&comm->pacer, acked_b, rtt_ns, newly_rx, delta_ce))
                    atomic_fetch_add_explicit(&vxf_st_tx_cnp_events, 1, memory_order_relaxed);
                tx_bootstrap_srtt(req, now_ns);
                tx_update_ecn_port_weights(req, ctrl.ecn_port_mask, delta_ce);
            }
        } else if (ctrl.pkt_type == PKT_DONE) {
            tx_release_req_inflight(comm, req);

            uint64_t rtt_ns = 0u;
            if (!req->retx_or_ir_sent && req->burst_end_tx_ns > 0u && now_ns > req->burst_end_tx_ns) {
                uint64_t raw_rtt = now_ns - req->burst_end_tx_ns;
                uint64_t dwell_ns = (uint64_t)ctrl.dwell_us * 1000ULL;
                rtt_ns = (raw_rtt > dwell_ns) ? (raw_rtt - dwell_ns) : 1000ULL;
            }

            uint32_t delta_ce = (ctrl.ecn_ce_pkts > req->win_ecn_ce) ? (ctrl.ecn_ce_pkts - req->win_ecn_ce) : 0u;
            uint32_t raw_rx = ctrl.rx_pkts ? ctrl.rx_pkts : ctrl.K;
            uint32_t delta_rx = (raw_rx > req->win_rx_pkts) ? (raw_rx - req->win_rx_pkts) : 0u;
            uint64_t rem_acked_b = (req->total_bytes > req->win_acked_bytes) ? (req->total_bytes - req->win_acked_bytes) : 0u;
            if (rem_acked_b > 0u || delta_ce > 0u || rtt_ns > 0u || req->win_rx_pkts == 0u) {
                if (harq_pacer_on_ack(&comm->pacer, rem_acked_b, rtt_ns, delta_rx ? delta_rx : 1u, delta_ce))
                    atomic_fetch_add_explicit(&vxf_st_tx_cnp_events, 1, memory_order_relaxed);
            }
            if (delta_rx > 0u || delta_ce > 0u)
                tx_update_ecn_port_weights(req, ctrl.ecn_port_mask, delta_ce);

            comm->loss_ewma = harq_ewma_update(comm->loss_ewma, ctrl.K, ctrl.esi, 0.20);

            vxfec_tx_user_req_t *ureq = req->parent;
            tx_deactivate_slot(comm, req, true);

            if (ureq && !ureq->failed) {
                if (ureq->active_gens > 0u)
                    ureq->active_gens--;
                ureq->gens_completed++;
                if (ureq->gens_completed >= ureq->num_gens) {
                    ureq->status = ncclSuccess;
                    ureq->completion_ready = true;
                    if (!ureq->ibv_zc_pending || !comm->ibv_ep || vxfec_ibv_ep_tx_done(comm->ibv_ep, ureq->ibv_target_seq)) {
                        ureq->ibv_zc_pending = false;
                        ureq->done = true;
                    }
                } else {
                    tx_spawn_ready_generations(comm);
                }
            }
        } else if (ctrl.pkt_type == PKT_ERR) {
            tx_fail_req(comm, req, ncclRemoteError);
        } else if (ctrl.pkt_type == PKT_SACK) {
            tx_queue_sack(req, ctrl.sack_cnt, ctrl_buf + sizeof(ctrl), (size_t)n - sizeof(ctrl), now_ns);
        } else if (ctrl.pkt_type == PKT_HARQ_IR) {
            uint32_t count = ctrl.ir_count ? ctrl.ir_count : 1u;
            uint8_t codec_flag = ctrl.flags & WIRE_CODEC_MASK;
            bool is_dup = ctrl.esi == req->last_ir_ctrl_esi && count <= req->last_ir_ctrl_cnt &&
                          codec_flag == req->last_ir_ctrl_codec &&
                          tx_recovery_in_flight(comm, req->ir_rem, req->last_ir_tx_ns, now_ns);
            if (!is_dup) {
                tx_bootstrap_srtt(req, now_ns);
                uint32_t start = (ctrl.esi >= req->next_rep_esi) ? ctrl.esi : req->next_rep_esi;
                uint32_t proactive_end = req->K + req->proactive_target;
                if (start < proactive_end)
                    start = proactive_end;
                uint32_t send_cnt = count;
                if (codec_flag == WIRE_FLAG_RS_CAUCHY8 && (req->codec != VXFEC_CODEC_RS_CAUCHY8 || start + send_cnt > 256u))
                    send_cnt += wire_flags_ir_slack(WIRE_FLAG_RQ_GF256, req->K);
                req->ack_probe_tx_ns = 0u;
                tx_release_req_inflight(comm, req);
                if (req->ir_rem == 0u) {
                    req->ir_start_esi = start;
                    req->ir_rem = send_cnt;
                } else if (send_cnt > req->ir_rem) {
                    req->ir_rem = send_cnt;
                }
                req->last_ir_ctrl_esi = ctrl.esi;
                req->last_ir_ctrl_cnt = count;
                req->last_ir_ctrl_codec = codec_flag;
                req->need_recovery_tail = true;
                req->retx_or_ir_sent = true;
            }
        } else if (ctrl.pkt_type == PKT_GBN_NACK) {
            tx_bootstrap_srtt(req, now_ns);
            uint32_t first_miss = (ctrl.esi < req->K) ? ctrl.esi : 0u;
            req->ack_probe_tx_ns = 0u;
            tx_release_req_inflight(comm, req);
            req->gbn_cursor = first_miss;
            req->gbn_rem = req->K - first_miss;
            req->need_recovery_tail = true;
            req->retx_or_ir_sent = true;
        }
    }

    tx_drain_paced(comm, now_ns);
    now_ns = mono_ns();

    uint64_t rto_ns = tx_tail_rto_ns(comm);
    uint64_t rm = comm->active_tx_mask;
    while (rm) {
        uint32_t s = (uint32_t)__builtin_ctzll(rm);
        rm &= rm - 1ULL;
        vxfec_tx_slot_t *req = &comm->tx_pool[s];
        if (req->wire_active && !req->done && req->initial_tail_sent && !req->tx_gpu && !tx_req_has_recovery_data(req) &&
            !req->need_recovery_tail && now_ns > req->last_tail_ns && (now_ns - req->last_tail_ns) > rto_ns) {
            req->ack_probe_tx_ns = 0u;
            tx_send_tail_ex(req, true);
        }
    }
    tx_retransmit_aborts_locked(comm, now_ns);

    if (comm->ibv_ep) {
        if (!comm->ibv_failed && !vxfec_ibv_ep_has_error(comm->ibv_ep)) {
            for (uint32_t u = 0; u < NCCL_NET_MAX_REQUESTS; u++) {
                vxfec_tx_user_req_t *ureq = &comm->user_tx_pool[u];
                if (ureq->in_use && ureq->completion_ready && !ureq->done) {
                    if (!ureq->ibv_zc_pending || vxfec_ibv_ep_tx_done(comm->ibv_ep, ureq->ibv_target_seq)) {
                        ureq->ibv_zc_pending = false;
                        ureq->done = true;
                    }
                }
            }
        }
        tx_quiesce_broken_ibv(comm);
    }
}

ncclResult_t vxf_isend(void *sendComm, void *data, int size, int tag, void *mhandle, void **request)
{
    if (!sendComm || size < 0 || (size > 0 && !data) || !request)
        return ncclInvalidArgument;
    *request = NULL;

    vxfec_mr_t *mr = (vxfec_mr_t *)mhandle;
    if (mr && mr->base_ptr && size > 0) {
        uintptr_t d_addr = (uintptr_t)data;
        uintptr_t b_addr = (uintptr_t)mr->base_ptr;
        if (d_addr < b_addr || (size_t)(d_addr - b_addr) > mr->size || (size_t)size > mr->size - (size_t)(d_addr - b_addr))
            return ncclInvalidUsage;
    }

    vxfec_send_comm_t *comm = (vxfec_send_comm_t *)sendComm;
    pthread_mutex_lock(&comm->mtx);
    vxf_progress_send_comm(comm);
    if (comm->ibv_failed) {
        bool need_bg = (comm->unacked_aborts > 0u && !comm->tx_bg_registered);
        pthread_mutex_unlock(&comm->mtx);
        if (need_bg)
            vxf_tx_bg_register(comm);
        return ncclSystemError;
    }

    int uslot = -1;
    for (int i = 0; i < NCCL_NET_MAX_REQUESTS; i++) {
        if (!comm->user_tx_pool[i].in_use) {
            uslot = i;
            break;
        }
    }
    if (uslot < 0 || ~comm->active_tx_mask == 0ULL) {
        pthread_mutex_unlock(&comm->mtx);
        return ncclSuccess;
    }

    uint16_t sym_T = get_cfg_sym_t((uint32_t)size, comm->path_sym_limit);
    if (sym_T == 0u) {
        pthread_mutex_unlock(&comm->mtx);
        return ncclSystemError;
    }

    vxfec_tx_user_req_t *ureq = &comm->user_tx_pool[uslot];
    memset(ureq, 0, sizeof(*ureq));

    vxfec_ibv_mr_t *ibv_mr =
        (comm->ibv_ep && mr && comm->dev_idx >= 0 && (uint32_t)comm->dev_idx < vxf_num_devs) ? mr->ibv_mrs[comm->dev_idx] : NULL;

    if (ibv_mr) {
        ureq->src_ibv_ptr = (const uint8_t *)data;
        ureq->src_ibv_lkey = vxfec_ibv_mr_lkey(ibv_mr);
        ureq->src_ibv_dmabuf = vxfec_ibv_mr_is_dmabuf(ibv_mr);
    }

    if (size > 0 && mr && mr->ptr_type == NCCL_PTR_CUDA) {
        if (ibv_mr && ureq->src_ibv_dmabuf) {
            ureq->src_host = NULL;
        } else if (mr->vk_buf) {
            ureq->bounce_buf = (uint8_t *)malloc((size_t)size);
            if (!ureq->bounce_buf) {
                pthread_mutex_unlock(&comm->mtx);
                return ncclSystemError;
            }
            size_t sub_off = (size_t)((uintptr_t)data - (uintptr_t)mr->base_ptr);
            if (!vxf_mr_vk_read(mr, sub_off, ureq->bounce_buf, (size_t)size)) {
                free(ureq->bounce_buf);
                ureq->bounce_buf = NULL;
                pthread_mutex_unlock(&comm->mtx);
                return ncclSystemError;
            }
            ureq->src_host = ureq->bounce_buf;
        } else {
            pthread_mutex_unlock(&comm->mtx);
            return ncclSystemError;
        }
    } else {
        ureq->src_host = (const uint8_t *)data;
    }

    uint32_t tag_seq = vxf_next_tag_seq(&comm->tag_map, (uint32_t)tag);
    if (tag_seq == 0u) {
        free(ureq->bounce_buf);
        memset(ureq, 0, sizeof(*ureq));
        pthread_mutex_unlock(&comm->mtx);
        return ncclSystemError;
    }

    uint32_t raw_K = (size == 0) ? 1u : (uint32_t)(((uint64_t)(uint32_t)size + sym_T - 1u) / sym_T);
    uint32_t slice_min = comm->slice_min_bytes;
    vxfec_codec_ovr_t ovr = (vxfec_codec_ovr_t)atomic_load_explicit(&vxf_codec_override, memory_order_acquire);
    bool slice_allowed = ((comm->features & VXFEC_HELLO_SLICE) != 0u) && (slice_min < UINT32_MAX) &&
                         (ovr != VXFEC_CODEC_OVR_RQ_GF256) && (ovr != VXFEC_CODEC_OVR_RQ_GF2);
    size_t gen_stride = (size_t)VXFEC_RS8_WIN_K * sym_T;
    size_t raw_gens = (size == 0) ? 1u : (((size_t)size + gen_stride - 1u) / gen_stride);
    uint16_t num_gens =
        (slice_allowed && (uint32_t)size >= slice_min && raw_K > VXFEC_RS8_WIN_K && raw_gens >= 2u && raw_gens <= UINT16_MAX)
            ? (uint16_t)raw_gens
            : 1u;

    ureq->kind = VXFEC_REQ_SEND;
    ureq->in_use = true;
    ureq->done = false;
    ureq->comm = comm;
    ureq->tag = (uint32_t)tag;
    ureq->tag_seq = tag_seq;
    ureq->msg_total_bytes = (uint32_t)size;
    ureq->sym_T = sym_T;
    ureq->num_gens = num_gens;
    ureq->status = ncclSuccess;
    ureq->mr = mr;
    vxf_mr_retain(mr);

    atomic_fetch_add_explicit(&vxf_st_tx_messages, 1, memory_order_relaxed);

    tx_spawn_ready_generations(comm);
    tx_drain_paced(comm, mono_ns());
    tx_quiesce_broken_ibv(comm);
    bool need_bg = (comm->unacked_aborts > 0u && !comm->tx_bg_registered);
    pthread_mutex_unlock(&comm->mtx);
    if (need_bg)
        vxf_tx_bg_register(comm);
    if (atomic_exchange_explicit(&vxf_retired_lc_reap_needed, false, memory_order_acq_rel))
        vxf_reap_retired_listeners(false);

    *request = ureq;
    return ncclSuccess;
}

ncclResult_t vxf_irecv(void *recvComm, int n, void **data, int *sizes, int *tags, void **mhandles, void **request)
{
    if (!recvComm || n != 1 || !data || !sizes || sizes[0] < 0 || (sizes[0] > 0 && !data[0]) || !tags || !request)
        return ncclInvalidArgument;
    *request = NULL;

    vxfec_mr_t *mr = mhandles ? (vxfec_mr_t *)mhandles[0] : NULL;
    if (mr && mr->base_ptr && sizes[0] > 0) {
        uintptr_t d_addr = (uintptr_t)data[0];
        uintptr_t b_addr = (uintptr_t)mr->base_ptr;
        if (d_addr < b_addr || (size_t)(d_addr - b_addr) > mr->size || (size_t)sizes[0] > mr->size - (size_t)(d_addr - b_addr))
            return ncclInvalidUsage;
    }

    vxfec_recv_comm_t *comm = (vxfec_recv_comm_t *)recvComm;
    pthread_mutex_lock(&comm->mtx);
    vxf_progress_recv_comm(comm);
    if (comm->ibv_failed) {
        pthread_mutex_unlock(&comm->mtx);
        return ncclSystemError;
    }

    int rslot = -1;
    for (int i = 0; i < NCCL_NET_MAX_REQUESTS; i++) {
        if (!comm->rx_pool[i].in_use) {
            rslot = i;
            break;
        }
    }
    if (rslot < 0) {
        pthread_mutex_unlock(&comm->mtx);
        return ncclSuccess;
    }

    uint32_t tag_seq = vxf_next_tag_seq(&comm->tag_map, (uint32_t)tags[0]);
    if (tag_seq == 0u) {
        pthread_mutex_unlock(&comm->mtx);
        return ncclSystemError;
    }

    vxfec_rx_req_t *req = &comm->rx_pool[rslot];
    memset(req, 0, sizeof(*req));
    req->kind = VXFEC_REQ_RECV;
    req->in_use = true;
    req->done = false;
    req->comm = comm;
    req->tag = (uint32_t)tags[0];
    req->expected_tag_seq = tag_seq;
    req->dst_ptr = data[0];
    req->max_size = sizes[0];
    req->status = ncclSuccess;
    req->mr = mr;
    vxf_mr_retain(mr);

    for (uint32_t a = 0; a < VXFEC_ABORT_RING; a++) {
        if (comm->peer_aborts[a].tag_seq == req->expected_tag_seq && comm->peer_aborts[a].tag == req->tag) {
            comm->peer_aborts[a].tag_seq = 0u;
            req->status = ncclRemoteError;
            req->done = true;
            break;
        }
    }

    if (!req->done) {
        for (uint32_t w = 0; w < VXFEC_RX_WIRE_SLOTS / 64u; w++) {
            uint64_t m = comm->active_slot_mask[w];
            bool failed = false;
            while (m) {
                uint32_t i = w * 64u + (uint32_t)__builtin_ctzll(m);
                m &= m - 1ULL;
                if (comm->slots[i].active && !comm->slots[i].bound_req && comm->slots[i].tag == req->tag &&
                    comm->slots[i].tag_seq == req->expected_tag_seq) {
                    if (!rx_bind_slot_to_req(&comm->slots[i], req)) {
                        comm->slots[i].bound_req = req;
                        if (rx_prepare_done(comm, comm->slots[i].msg_seq))
                            rx_fail_slot(comm, &comm->slots[i], ncclSystemError);
                        else
                            vxf_reset_rx_slot(comm, &comm->slots[i]);
                        failed = true;
                        break;
                    }
                }
            }
            if (failed)
                break;
        }
    }

    if (!req->done)
        vxf_progress_recv_comm(comm);
    pthread_mutex_unlock(&comm->mtx);
    if (atomic_exchange_explicit(&vxf_retired_lc_reap_needed, false, memory_order_acq_rel))
        vxf_reap_retired_listeners(false);

    *request = req;
    return ncclSuccess;
}

ncclResult_t vxf_iflush(void *recvComm, int n, void **data, int *sizes, void **mhandles, void **request)
{
    (void)recvComm;
    (void)n;
    (void)data;
    (void)sizes;
    (void)mhandles;
    if (!request)
        return ncclInvalidArgument;
    *request = (void *)&g_flush_sentinel;
    return ncclSuccess;
}

ncclResult_t vxf_test(void *request, int *done, int *sizes)
{
    if (!request || !done)
        return ncclInvalidArgument;
    *done = 0;

    if (request == (const void *)&g_flush_sentinel) {
        *done = 1;
        return ncclSuccess;
    }

    vxfec_req_kind_t kind = *(const vxfec_req_kind_t *)request;
    if (kind == VXFEC_REQ_SEND) {
        vxfec_tx_user_req_t *ureq = (vxfec_tx_user_req_t *)request;
        vxfec_send_comm_t *sc = ureq->comm;
        if (!sc)
            return ncclInvalidUsage;
        pthread_mutex_lock(&sc->mtx);
        if (!ureq->in_use || ureq->comm != sc) {
            pthread_mutex_unlock(&sc->mtx);
            return ncclInvalidUsage;
        }
        if (!ureq->done)
            vxf_progress_send_comm(sc);
        bool is_done = ureq->done;
        ncclResult_t st = ncclSuccess;
        int total = 0;
        vxfec_mr_t *mr = NULL;
        uint8_t *bbuf = NULL;
        if (is_done) {
            st = ureq->status;
            total = (int)ureq->msg_total_bytes;
            bbuf = ureq->bounce_buf;
            ureq->bounce_buf = NULL;
            mr = ureq->mr;
            ureq->mr = NULL;
            ureq->in_use = false;
        }
        bool need_bg = (sc->unacked_aborts > 0u && !sc->tx_bg_registered);
        pthread_mutex_unlock(&sc->mtx);
        if (need_bg)
            vxf_tx_bg_register(sc);
        if (atomic_exchange_explicit(&vxf_retired_lc_reap_needed, false, memory_order_acq_rel))
            vxf_reap_retired_listeners(false);
        if (is_done) {
            free(bbuf);
            vxf_mr_release(mr);
            if (st != ncclSuccess)
                return st;
            *done = 1;
            if (sizes)
                sizes[0] = total;
        }
        return ncclSuccess;
    }

    if (kind == VXFEC_REQ_RECV) {
        vxfec_rx_req_t *req = (vxfec_rx_req_t *)request;
        vxfec_recv_comm_t *rc = req->comm;
        if (!rc)
            return ncclInvalidUsage;
        pthread_mutex_lock(&rc->mtx);
        if (!req->in_use || req->comm != rc) {
            pthread_mutex_unlock(&rc->mtx);
            return ncclInvalidUsage;
        }
        vxf_progress_recv_comm(rc);
        if (req->done) {
            ncclResult_t st = req->status;
            int actual = req->actual_size;
            free(req->heap_gen_bits);
            req->heap_gen_bits = NULL;
            vxfec_mr_t *mr = req->mr;
            req->mr = NULL;
            req->in_use = false;
            pthread_mutex_unlock(&rc->mtx);
            if (atomic_exchange_explicit(&vxf_retired_lc_reap_needed, false, memory_order_acq_rel))
                vxf_reap_retired_listeners(false);
            vxf_mr_release(mr);
            if (st != ncclSuccess)
                return st;
            *done = 1;
            if (sizes)
                sizes[0] = actual;
            return ncclSuccess;
        }
        pthread_mutex_unlock(&rc->mtx);
        if (atomic_exchange_explicit(&vxf_retired_lc_reap_needed, false, memory_order_acq_rel))
            vxf_reap_retired_listeners(false);
        return ncclSuccess;
    }

    return ncclInvalidArgument;
}

ncclResult_t vxf_irecv_consumed(void *recvComm, int n, void *request)
{
    (void)recvComm;
    (void)n;
    (void)request;
    return ncclSuccess;
}
