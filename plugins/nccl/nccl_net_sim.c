#define _GNU_SOURCE
#include "nccl_net_internal.h"

_Atomic uint64_t vxf_st_tx_messages = 0;
_Atomic uint64_t vxf_st_tx_sys_pkts = 0;
_Atomic uint64_t vxf_st_tx_proactive_rep_pkts = 0;
_Atomic uint64_t vxf_st_tx_sack_retx_pkts = 0;
_Atomic uint64_t vxf_st_tx_harq_ir_pkts = 0;
_Atomic uint64_t vxf_st_rx_messages = 0;
_Atomic uint64_t vxf_st_rx_fast_completions = 0;
_Atomic uint64_t vxf_st_rx_fec_decodes = 0;
_Atomic uint64_t vxf_st_rx_rs8_decodes = 0;
_Atomic uint64_t vxf_st_rx_rq_decodes = 0;
_Atomic uint64_t vxf_st_rx_ooo_pkts = 0;
_Atomic uint64_t vxf_st_vk_offload_calls = 0;
_Atomic uint64_t vxf_st_sim_dropped_pkts = 0;
_Atomic uint64_t vxf_st_rx_ecn_ce_pkts = 0;
_Atomic uint64_t vxf_st_tx_cnp_events = 0;
_Atomic uint64_t vxf_st_tx_paced_slices = 0;
_Atomic uint64_t vxf_st_tx_fec_clamped_pkts = 0;
_Atomic uint64_t vxf_st_ibv_direct_pkts = 0;
_Atomic uint64_t vxf_st_vk_async_polls = 0;
_Atomic uint64_t vxf_st_tx_sys_ibv_pkts = 0;
_Atomic uint64_t vxf_st_tx_sys_gpu_direct_pkts = 0;
_Atomic uint64_t vxf_st_rx_sys_pkts = 0;
_Atomic uint64_t vxf_st_rx_sys_gpu_direct_pkts = 0;
_Atomic uint64_t vxf_st_rail_tx_gpu_direct_pkts[VXFEC_MAX_DEVS];
_Atomic uint64_t vxf_st_rail_rx_gpu_direct_pkts[VXFEC_MAX_DEVS];
_Atomic uint64_t vxf_st_tx_udp_payload_pkts = 0;
_Atomic uint64_t vxf_st_rx_cuda_repair_completions = 0;

atomic_int vxf_env_codec_ovr = VXFEC_CODEC_OVR_AUTO;
atomic_int vxf_codec_override = VXFEC_CODEC_OVR_AUTO;

_Static_assert(offsetof(vxfec_nccl_stats_t, ibv_direct_pkts) == VXFEC_NCCL_STATS_V1_BYTES,
               "vxfec_nccl_stats_t v1 prefix size mismatch");
_Static_assert(offsetof(vxfec_nccl_stats_t, tx_sys_ibv_pkts) == VXFEC_NCCL_STATS_V2_BYTES,
               "vxfec_nccl_stats_t v2 prefix size mismatch");
_Static_assert(offsetof(vxfec_nccl_stats_t, rx_cuda_repair_completions) == VXFEC_NCCL_STATS_V3_BYTES,
               "vxfec_nccl_stats_t v3 prefix size mismatch");

vxfec_codec_ovr_t vxf_parse_codec_mode(const char *mode)
{
    if (!mode || !*mode || strcmp(mode, "auto") == 0)
        return VXFEC_CODEC_OVR_AUTO;
    if (strcmp(mode, "gbn") == 0 || strcmp(mode, "baseline") == 0)
        return VXFEC_CODEC_OVR_GBN;
    if (strcmp(mode, "sack") == 0)
        return VXFEC_CODEC_OVR_SACK;
    if (strcmp(mode, "rs8") == 0)
        return VXFEC_CODEC_OVR_RS8;
    if (strcmp(mode, "rq-gf256") == 0)
        return VXFEC_CODEC_OVR_RQ_GF256;
    if (strcmp(mode, "rq-gf2") == 0 || strcmp(mode, "rq") == 0)
        return VXFEC_CODEC_OVR_RQ_GF2;
    return VXFEC_CODEC_OVR_AUTO;
}

static void snapshot_stats(vxfec_nccl_stats_t *snap)
{
    snap->tx_messages = atomic_load_explicit(&vxf_st_tx_messages, memory_order_relaxed);
    snap->tx_sys_pkts = atomic_load_explicit(&vxf_st_tx_sys_pkts, memory_order_relaxed);
    snap->tx_proactive_rep_pkts = atomic_load_explicit(&vxf_st_tx_proactive_rep_pkts, memory_order_relaxed);
    snap->tx_sack_retx_pkts = atomic_load_explicit(&vxf_st_tx_sack_retx_pkts, memory_order_relaxed);
    snap->tx_harq_ir_pkts = atomic_load_explicit(&vxf_st_tx_harq_ir_pkts, memory_order_relaxed);
    snap->rx_messages = atomic_load_explicit(&vxf_st_rx_messages, memory_order_relaxed);
    snap->rx_fast_completions = atomic_load_explicit(&vxf_st_rx_fast_completions, memory_order_relaxed);
    snap->rx_fec_decodes = atomic_load_explicit(&vxf_st_rx_fec_decodes, memory_order_relaxed);
    snap->rx_rs8_decodes = atomic_load_explicit(&vxf_st_rx_rs8_decodes, memory_order_relaxed);
    snap->rx_rq_decodes = atomic_load_explicit(&vxf_st_rx_rq_decodes, memory_order_relaxed);
    snap->rx_ooo_pkts = atomic_load_explicit(&vxf_st_rx_ooo_pkts, memory_order_relaxed);
    snap->vk_offload_calls = atomic_load_explicit(&vxf_st_vk_offload_calls, memory_order_relaxed);
    snap->sim_dropped_pkts = atomic_load_explicit(&vxf_st_sim_dropped_pkts, memory_order_relaxed);
    snap->rx_ecn_ce_pkts = atomic_load_explicit(&vxf_st_rx_ecn_ce_pkts, memory_order_relaxed);
    snap->tx_cnp_events = atomic_load_explicit(&vxf_st_tx_cnp_events, memory_order_relaxed);
    snap->tx_paced_slices = atomic_load_explicit(&vxf_st_tx_paced_slices, memory_order_relaxed);
    snap->tx_fec_clamped_pkts = atomic_load_explicit(&vxf_st_tx_fec_clamped_pkts, memory_order_relaxed);
    snap->ibv_direct_pkts = atomic_load_explicit(&vxf_st_ibv_direct_pkts, memory_order_relaxed);
    snap->vk_async_polls = atomic_load_explicit(&vxf_st_vk_async_polls, memory_order_relaxed);
    snap->tx_sys_ibv_pkts = atomic_load_explicit(&vxf_st_tx_sys_ibv_pkts, memory_order_relaxed);
    snap->tx_sys_gpu_direct_pkts = atomic_load_explicit(&vxf_st_tx_sys_gpu_direct_pkts, memory_order_relaxed);
    snap->rx_sys_pkts = atomic_load_explicit(&vxf_st_rx_sys_pkts, memory_order_relaxed);
    snap->rx_sys_gpu_direct_pkts = atomic_load_explicit(&vxf_st_rx_sys_gpu_direct_pkts, memory_order_relaxed);
    snap->tx_udp_payload_pkts = atomic_load_explicit(&vxf_st_tx_udp_payload_pkts, memory_order_relaxed);
    snap->rx_cuda_repair_completions = atomic_load_explicit(&vxf_st_rx_cuda_repair_completions, memory_order_relaxed);
}

__attribute__((visibility("default"))) void vxfec_nccl_get_stats_ex(vxfec_nccl_stats_t *out, size_t out_bytes)
{
    if (!out || out_bytes == 0u)
        return;
    vxfec_nccl_stats_t snap;
    snapshot_stats(&snap);
    memset(out, 0, out_bytes);
    size_t copy_bytes = (out_bytes < sizeof(snap)) ? out_bytes : sizeof(snap);
    memcpy(out, &snap, copy_bytes);
}

/* legacy v1 ABI: write only the original 17 uint64_t fields */
__attribute__((visibility("default"))) void vxfec_nccl_get_stats(vxfec_nccl_stats_t *out)
{
    vxfec_nccl_get_stats_ex(out, VXFEC_NCCL_STATS_V1_BYTES);
}

/* Odd epochs enable collection. Recheck the epoch under the histogram lock
 * so a sender delayed across two windows cannot add a request to the next. */
_Atomic uint64_t vxf_send_histogram_epoch = 0;
static pthread_mutex_t send_histogram_lock = PTHREAD_MUTEX_INITIALIZER;
static vxfec_nccl_send_size_bin_t send_bins[VXFEC_NCCL_SEND_HISTOGRAM_CAPACITY];
static size_t send_bin_count;
static bool send_histogram_overflow;

__attribute__((visibility("default"))) int vxfec_nccl_send_histogram_begin(void)
{
    pthread_mutex_lock(&send_histogram_lock);
    uint64_t epoch = atomic_load_explicit(&vxf_send_histogram_epoch, memory_order_relaxed);
    if (epoch & 1u) {
        pthread_mutex_unlock(&send_histogram_lock);
        return -EBUSY;
    }
    send_bin_count = 0;
    send_histogram_overflow = false;
    atomic_store_explicit(&vxf_send_histogram_epoch, epoch + 1u, memory_order_relaxed);
    pthread_mutex_unlock(&send_histogram_lock);
    return 0;
}

void vxf_record_send_size(uint64_t epoch, uint32_t size)
{
    pthread_mutex_lock(&send_histogram_lock);
    if (epoch == atomic_load_explicit(&vxf_send_histogram_epoch, memory_order_relaxed) && !send_histogram_overflow) {
        size_t i = 0;
        while (i < send_bin_count && send_bins[i].size_bytes != size)
            i++;
        if (i == VXFEC_NCCL_SEND_HISTOGRAM_CAPACITY) {
            send_histogram_overflow = true;
        } else if (i == send_bin_count) {
            send_bins[send_bin_count++] = (vxfec_nccl_send_size_bin_t){.size_bytes = size, .requests = 1u, .bytes = size};
        } else if (send_bins[i].requests == UINT64_MAX || send_bins[i].bytes > UINT64_MAX - size) {
            send_histogram_overflow = true;
        } else {
            send_bins[i].requests++;
            send_bins[i].bytes += size;
        }
    }
    pthread_mutex_unlock(&send_histogram_lock);
}

__attribute__((visibility("default"))) int vxfec_nccl_send_histogram_end(vxfec_nccl_send_size_bin_t *bins, size_t capacity)
{
    pthread_mutex_lock(&send_histogram_lock);
    uint64_t epoch = atomic_load_explicit(&vxf_send_histogram_epoch, memory_order_relaxed);
    int result = -EINVAL;
    if (epoch & 1u) {
        atomic_store_explicit(&vxf_send_histogram_epoch, epoch + 1u, memory_order_relaxed);
        if (send_histogram_overflow)
            result = -EOVERFLOW;
        else if (capacity < send_bin_count)
            result = -ENOSPC;
        else if (bins || !send_bin_count) {
            if (send_bin_count)
                memcpy(bins, send_bins, send_bin_count * sizeof(*bins));
            result = (int)send_bin_count;
        }
    }
    pthread_mutex_unlock(&send_histogram_lock);
    return result;
}

static void reset_common_stats(void)
{
    atomic_store(&vxf_st_tx_messages, 0);
    atomic_store(&vxf_st_tx_sys_pkts, 0);
    atomic_store(&vxf_st_tx_proactive_rep_pkts, 0);
    atomic_store(&vxf_st_tx_sack_retx_pkts, 0);
    atomic_store(&vxf_st_tx_harq_ir_pkts, 0);
    atomic_store(&vxf_st_rx_messages, 0);
    atomic_store(&vxf_st_rx_fast_completions, 0);
    atomic_store(&vxf_st_rx_fec_decodes, 0);
    atomic_store(&vxf_st_rx_rs8_decodes, 0);
    atomic_store(&vxf_st_rx_rq_decodes, 0);
    atomic_store(&vxf_st_rx_ooo_pkts, 0);
    atomic_store(&vxf_st_vk_offload_calls, 0);
    atomic_store(&vxf_st_sim_dropped_pkts, 0);
    atomic_store(&vxf_st_rx_ecn_ce_pkts, 0);
    atomic_store(&vxf_st_tx_cnp_events, 0);
    atomic_store(&vxf_st_tx_paced_slices, 0);
    atomic_store(&vxf_st_tx_fec_clamped_pkts, 0);
    atomic_store(&vxf_st_ibv_direct_pkts, 0);
    atomic_store(&vxf_st_vk_async_polls, 0);
    atomic_store(&vxf_st_tx_sys_ibv_pkts, 0);
    atomic_store(&vxf_st_tx_sys_gpu_direct_pkts, 0);
    atomic_store(&vxf_st_rx_sys_pkts, 0);
    atomic_store(&vxf_st_rx_sys_gpu_direct_pkts, 0);
    for (size_t i = 0; i < VXFEC_MAX_DEVS; i++) {
        atomic_store(&vxf_st_rail_tx_gpu_direct_pkts[i], 0);
        atomic_store(&vxf_st_rail_rx_gpu_direct_pkts[i], 0);
    }
    atomic_store(&vxf_st_tx_udp_payload_pkts, 0);
    atomic_store(&vxf_st_rx_cuda_repair_completions, 0);
}

__attribute__((visibility("default"))) void vxfec_nccl_set_codec_override(const char *mode)
{
    int ovr = mode ? (int)vxf_parse_codec_mode(mode) : atomic_load_explicit(&vxf_env_codec_ovr, memory_order_relaxed);
    atomic_store_explicit(&vxf_codec_override, ovr, memory_order_release);
}

#if VXFEC_ENABLE_SIM

atomic_bool vxf_sim_active = false;
atomic_bool vxf_sim_asym_active = false;
atomic_uint vxf_sim_cfg_epoch = 1u;
atomic_bool vxf_sim_reorder = false;
_Atomic uint64_t vxf_sim_max_rate_mbps = 0u;
_Atomic uint64_t vxf_sim_link_delay_ns = 0u;
atomic_uint vxf_sim_drop_hellos = 0u;
atomic_uint vxf_sim_drop_acks = 0u;
atomic_uint vxf_sim_drop_dones = 0u;
atomic_uint vxf_sim_drop_aborts = 0u;
atomic_uint vxf_sim_fail_gpu_rx_writes = 0u;

static atomic_uint g_sim_loss_ppm = 0u;
static atomic_uint g_sim_burst_len = 1u;
static atomic_uint g_sim_burst_rem = 0u;
static atomic_uint g_sim_ecn_ppm = 0u;
static atomic_uint g_sim_hot_port_mask = 0u;
static atomic_uint g_sim_asym_primary_mask = 0u;
static _Atomic uint64_t g_sim_asym_primary_delay_ns = 0u;
static _Atomic uint64_t g_sim_asym_backup_delay_ns = 0u;
static atomic_uint g_sim_asym_primary_loss_ppm = 0u;
static atomic_uint g_sim_asym_blackout_period = 0u;
static atomic_uint g_sim_asym_blackout_len = 0u;
static atomic_uint g_sim_asym_pkt_seq = 0u;
static atomic_uint g_rng_state = 0x9E3779B9u;
static pthread_mutex_t g_sim_mtx = PTHREAD_MUTEX_INITIALIZER;
static atomic_uint g_paired_epoch;
static uint32_t g_paired_next_epoch;
static uint64_t g_paired_key;
static atomic_uint g_paired_burst;
static atomic_uint g_paired_start_ppm;
static _Atomic uint64_t g_paired_trials;
static _Atomic uint64_t g_paired_fingerprint;

static uint64_t mix64(uint64_t x)
{
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

void vxf_sim_bind_paired_fault(vxfec_tx_user_req_t *req)
{
    if (req->msg_total_bytes == 0u || atomic_load_explicit(&g_paired_epoch, memory_order_relaxed) == 0u)
        return;
    pthread_mutex_lock(&g_sim_mtx);
    uint32_t epoch = atomic_load_explicit(&g_paired_epoch, memory_order_relaxed);
    if (epoch != 0u) {
        vxfec_send_comm_t *comm = req->comm;
        if (comm->paired_fault_epoch != epoch) {
            comm->paired_fault_epoch = epoch;
            comm->paired_fault_request_index = 0u;
        }
        uint64_t shape = ((uint64_t)req->msg_total_bytes << 16) | req->sym_T;
        req->paired_fault_key = mix64(g_paired_key + mix64(shape) + comm->paired_fault_request_index++);
        req->paired_fault_epoch = epoch;
    }
    pthread_mutex_unlock(&g_sim_mtx);
}

bool vxf_sim_should_drop_first_sys(const vxfec_tx_slot_t *req, uint32_t esi)
{
    if (req->parent->paired_fault_epoch != atomic_load_explicit(&g_paired_epoch, memory_order_acquire))
        return false;
    uint32_t burst = atomic_load_explicit(&g_paired_burst, memory_order_relaxed);
    uint32_t start_ppm = atomic_load_explicit(&g_paired_start_ppm, memory_order_relaxed);
    uint32_t pos = (uint32_t)req->gen_idx * VXFEC_RS8_WIN_K + esi;
    uint32_t begin = (pos >= burst) ? pos - burst + 1u : 0u;
    uint64_t key = req->parent->paired_fault_key;
    atomic_fetch_add_explicit(&g_paired_trials, 1, memory_order_relaxed);
    for (uint32_t start = begin; start <= pos; start++) {
        if (mix64(key + start) % 1000000u < start_ppm) {
            uint64_t fingerprint = mix64((key ^ UINT64_C(0xa0761d6478bd642f)) + pos);
            atomic_fetch_add_explicit(&g_paired_fingerprint, fingerprint, memory_order_relaxed);
            atomic_fetch_add_explicit(&vxf_st_sim_dropped_pkts, 1, memory_order_relaxed);
            return true;
        }
    }
    return false;
}

__attribute__((visibility("default"))) void vxfec_nccl_get_paired_fault_stats(vxfec_nccl_paired_fault_stats_t *out)
{
    if (!out)
        return;
    out->systematic_trials = atomic_load_explicit(&g_paired_trials, memory_order_relaxed);
    out->drop_fingerprint = atomic_load_explicit(&g_paired_fingerprint, memory_order_relaxed);
}

static void refresh_sim_active(void)
{
    /* serialize the snapshot and its publication with other refreshes */
    pthread_mutex_lock(&g_sim_mtx);
    bool asym = atomic_load_explicit(&g_sim_asym_primary_delay_ns, memory_order_relaxed) != 0u ||
                atomic_load_explicit(&g_sim_asym_backup_delay_ns, memory_order_relaxed) != 0u ||
                atomic_load_explicit(&g_sim_asym_primary_loss_ppm, memory_order_relaxed) != 0u ||
                atomic_load_explicit(&g_sim_asym_blackout_len, memory_order_relaxed) != 0u;
    atomic_store_explicit(&vxf_sim_asym_active, asym, memory_order_release);
    bool active = asym || atomic_load_explicit(&g_sim_loss_ppm, memory_order_relaxed) != 0u ||
                  atomic_load_explicit(&g_sim_burst_rem, memory_order_relaxed) != 0u ||
                  atomic_load_explicit(&vxf_sim_reorder, memory_order_relaxed) ||
                  atomic_load_explicit(&vxf_sim_drop_hellos, memory_order_relaxed) != 0u ||
                  atomic_load_explicit(&vxf_sim_drop_acks, memory_order_relaxed) != 0u ||
                  atomic_load_explicit(&vxf_sim_drop_dones, memory_order_relaxed) != 0u ||
                  atomic_load_explicit(&vxf_sim_drop_aborts, memory_order_relaxed) != 0u ||
                  atomic_load_explicit(&vxf_sim_fail_gpu_rx_writes, memory_order_relaxed) != 0u ||
                  atomic_load_explicit(&g_sim_ecn_ppm, memory_order_relaxed) != 0u ||
                  atomic_load_explicit(&g_sim_hot_port_mask, memory_order_relaxed) != 0u ||
                  atomic_load_explicit(&vxf_sim_max_rate_mbps, memory_order_relaxed) != 0u ||
                  atomic_load_explicit(&vxf_sim_link_delay_ns, memory_order_relaxed) != 0u;
    atomic_store_explicit(&vxf_sim_active, active, memory_order_release);
    pthread_mutex_unlock(&g_sim_mtx);
}

void vxf_sim_init_from_env(void)
{
    const char *env_ppm = getenv("VXFEC_NCCL_SIM_LOSS_PPM");
    atomic_store(&g_sim_loss_ppm, (env_ppm && *env_ppm && atoi(env_ppm) > 0) ? (uint32_t)atoi(env_ppm) : 0u);

    const char *env_burst = getenv("VXFEC_NCCL_SIM_BURST");
    int b = (env_burst && *env_burst) ? atoi(env_burst) : 1;
    atomic_store(&g_sim_burst_len, (b > 1) ? (uint32_t)b : 1u);

    const char *env_reord = getenv("VXFEC_NCCL_SIM_REORDER");
    atomic_store(&vxf_sim_reorder, (env_reord && atoi(env_reord) != 0));
    refresh_sim_active();
}

uint32_t vxf_fast_rand32(void)
{
    uint32_t x = atomic_fetch_add_explicit(&g_rng_state, 0x9E3779B9u, memory_order_relaxed);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}

bool vxf_sim_should_drop_data_slow(void)
{
    uint32_t ppm = atomic_load_explicit(&g_sim_loss_ppm, memory_order_relaxed);
    if (ppm == 0u)
        return false;
    uint32_t rem = atomic_load_explicit(&g_sim_burst_rem, memory_order_relaxed);
    while (rem > 0u) {
        if (atomic_compare_exchange_weak_explicit(&g_sim_burst_rem, &rem, rem - 1u, memory_order_acq_rel, memory_order_relaxed)) {
            atomic_fetch_add_explicit(&vxf_st_sim_dropped_pkts, 1, memory_order_relaxed);
            return true;
        }
    }
    uint32_t burst = atomic_load_explicit(&g_sim_burst_len, memory_order_relaxed);
    uint32_t start_ppm = (burst > 1u) ? ((ppm + burst - 1u) / burst) : ppm;
    if ((vxf_fast_rand32() % 1000000u) < start_ppm) {
        if (burst > 1u) {
            uint32_t exp = 0u;
            (void)atomic_compare_exchange_strong_explicit(&g_sim_burst_rem, &exp, burst - 1u, memory_order_acq_rel,
                                                          memory_order_relaxed);
        }
        atomic_fetch_add_explicit(&vxf_st_sim_dropped_pkts, 1, memory_order_relaxed);
        return true;
    }
    return false;
}

bool vxf_sim_should_drop_counter_slow(atomic_uint *counter)
{
    uint32_t cur = atomic_load_explicit(counter, memory_order_relaxed);
    while (cur > 0u) {
        if (atomic_compare_exchange_weak_explicit(counter, &cur, cur - 1u, memory_order_acq_rel, memory_order_relaxed)) {
            if (counter != &vxf_sim_fail_gpu_rx_writes)
                atomic_fetch_add_explicit(&vxf_st_sim_dropped_pkts, 1, memory_order_relaxed);
            if (cur == 1u)
                refresh_sim_active();
            return true;
        }
    }
    return false;
}

bool vxf_sim_should_mark_ecn_slow(uint32_t port_idx)
{
    uint32_t hot_mask = atomic_load_explicit(&g_sim_hot_port_mask, memory_order_relaxed);
    if (hot_mask != 0u && port_idx < 32u && ((hot_mask >> port_idx) & 1u))
        return true;
    uint32_t ppm = atomic_load_explicit(&g_sim_ecn_ppm, memory_order_relaxed);
    if (ppm == 0u)
        return false;
    return (vxf_fast_rand32() % 1000000u) < ppm;
}

uint64_t vxf_sim_port_delay_ns_slow(uint32_t port_idx)
{
    uint32_t pmask = atomic_load_explicit(&g_sim_asym_primary_mask, memory_order_relaxed);
    bool is_primary = (port_idx < 32u) && (((pmask >> port_idx) & 1u) != 0u);
    return is_primary ? atomic_load_explicit(&g_sim_asym_primary_delay_ns, memory_order_relaxed)
                      : atomic_load_explicit(&g_sim_asym_backup_delay_ns, memory_order_relaxed);
}

bool vxf_sim_should_drop_port_data_slow(uint32_t port_idx)
{
    uint32_t pmask = atomic_load_explicit(&g_sim_asym_primary_mask, memory_order_relaxed);
    bool is_primary = (port_idx < 32u) && (((pmask >> port_idx) & 1u) != 0u);
    if (is_primary) {
        uint32_t period = atomic_load_explicit(&g_sim_asym_blackout_period, memory_order_relaxed);
        uint32_t blen = atomic_load_explicit(&g_sim_asym_blackout_len, memory_order_relaxed);
        if (period > 0u && blen > 0u) {
            uint32_t seq = atomic_fetch_add_explicit(&g_sim_asym_pkt_seq, 1u, memory_order_relaxed) % period;
            if (seq < blen) {
                atomic_fetch_add_explicit(&vxf_st_sim_dropped_pkts, 1, memory_order_relaxed);
                return true;
            }
        }
        uint32_t p_ppm = atomic_load_explicit(&g_sim_asym_primary_loss_ppm, memory_order_relaxed);
        if (p_ppm > 0u && (vxf_fast_rand32() % 1000000u) < p_ppm) {
            atomic_fetch_add_explicit(&vxf_st_sim_dropped_pkts, 1, memory_order_relaxed);
            return true;
        }
    }
    return vxf_sim_should_drop_data_slow();
}

__attribute__((visibility("default"))) void vxfec_nccl_reset_stats(void)
{
    reset_common_stats();
    atomic_store(&g_paired_trials, 0);
    atomic_store(&g_paired_fingerprint, 0);
    atomic_store(&g_sim_burst_rem, 0);
    atomic_store(&g_sim_asym_pkt_seq, 0);
    refresh_sim_active();
}

__attribute__((visibility("default"))) void vxfec_nccl_set_fault_injection(uint32_t loss_ppm, uint32_t burst_len, bool reorder)
{
    atomic_store(&g_paired_epoch, 0u);
    atomic_store(&g_sim_loss_ppm, loss_ppm);
    atomic_store(&g_sim_burst_len, burst_len ? burst_len : 1u);
    atomic_store(&vxf_sim_reorder, reorder);
    atomic_store(&g_sim_burst_rem, 0u);
    refresh_sim_active();
}

__attribute__((visibility("default"))) bool vxfec_nccl_set_paired_fault(uint32_t loss_ppm, uint32_t burst_len, uint64_t seed,
                                                                        uint32_t rank, uint32_t iteration)
{
    if (loss_ppm > 1000000u || burst_len == 0u || burst_len > 256u)
        return false;
    vxfec_nccl_set_fault_injection(0u, 1u, false);
    if (loss_ppm == 0u)
        return true;
    pthread_mutex_lock(&g_sim_mtx);
    g_paired_key = mix64(seed + mix64(((uint64_t)rank << 32) | iteration));
    atomic_store_explicit(&g_paired_burst, burst_len, memory_order_relaxed);
    atomic_store_explicit(&g_paired_start_ppm, (loss_ppm + burst_len - 1u) / burst_len, memory_order_relaxed);
    if (++g_paired_next_epoch == 0u)
        ++g_paired_next_epoch;
    atomic_store_explicit(&g_paired_epoch, g_paired_next_epoch, memory_order_release);
    pthread_mutex_unlock(&g_sim_mtx);
    return true;
}

__attribute__((visibility("default"))) void vxfec_nccl_set_handshake_fault(uint32_t drop_hellos, uint32_t drop_acks)
{
    atomic_store(&vxf_sim_drop_hellos, drop_hellos);
    atomic_store(&vxf_sim_drop_acks, drop_acks);
    refresh_sim_active();
}

__attribute__((visibility("default"))) void vxfec_nccl_set_done_fault(uint32_t drop_dones)
{
    atomic_store(&vxf_sim_drop_dones, drop_dones);
    refresh_sim_active();
}

__attribute__((visibility("default"))) void vxfec_nccl_set_abort_fault(uint32_t drop_aborts)
{
    atomic_store(&vxf_sim_drop_aborts, drop_aborts);
    refresh_sim_active();
}

__attribute__((visibility("default"))) void vxfec_nccl_set_gpu_rx_fault(uint32_t fail_writes)
{
    atomic_store(&vxf_sim_fail_gpu_rx_writes, fail_writes);
    refresh_sim_active();
}

__attribute__((visibility("default"))) void vxfec_nccl_set_congestion_sim(uint32_t ecn_ppm, uint32_t hot_port_mask,
                                                                          uint64_t max_rate_mbps)
{
    atomic_store(&g_sim_ecn_ppm, ecn_ppm);
    atomic_store(&g_sim_hot_port_mask, hot_port_mask);
    atomic_store(&vxf_sim_max_rate_mbps, max_rate_mbps);
    atomic_fetch_add(&vxf_sim_cfg_epoch, 1u);
    refresh_sim_active();
}

__attribute__((visibility("default"))) void vxfec_nccl_set_link_delay_ns(uint64_t one_way_delay_ns)
{
    atomic_store_explicit(&vxf_sim_link_delay_ns, one_way_delay_ns, memory_order_release);
    atomic_fetch_add(&vxf_sim_cfg_epoch, 1u);
    refresh_sim_active();
}

__attribute__((visibility("default"))) void
vxfec_nccl_set_asymmetric_link_sim(uint32_t primary_port_mask, uint64_t primary_delay_ns, uint64_t backup_delay_ns,
                                   uint32_t primary_loss_ppm, uint32_t blackout_period_pkts, uint32_t blackout_len_pkts)
{
    atomic_store_explicit(&g_sim_asym_primary_mask, primary_port_mask, memory_order_relaxed);
    atomic_store_explicit(&g_sim_asym_primary_delay_ns, primary_delay_ns, memory_order_relaxed);
    atomic_store_explicit(&g_sim_asym_backup_delay_ns, backup_delay_ns, memory_order_relaxed);
    atomic_store_explicit(&g_sim_asym_primary_loss_ppm, primary_loss_ppm, memory_order_relaxed);
    atomic_store_explicit(&g_sim_asym_blackout_period, blackout_period_pkts, memory_order_relaxed);
    atomic_store_explicit(&g_sim_asym_blackout_len, blackout_len_pkts, memory_order_relaxed);
    atomic_store_explicit(&g_sim_asym_pkt_seq, 0u, memory_order_relaxed);
    /* set control-plane delay to the fast primary link one-way delay when asymmetric mode is active */
    atomic_store_explicit(&vxf_sim_link_delay_ns, primary_delay_ns, memory_order_release);
    atomic_fetch_add(&vxf_sim_cfg_epoch, 1u);
    refresh_sim_active();
}

#else /* !VXFEC_ENABLE_SIM */

__attribute__((visibility("default"))) bool vxfec_nccl_set_paired_fault(uint32_t loss_ppm, uint32_t burst_len, uint64_t seed,
                                                                        uint32_t rank, uint32_t iteration)
{
    (void)loss_ppm;
    (void)burst_len;
    (void)seed;
    (void)rank;
    (void)iteration;
    return false;
}

__attribute__((visibility("default"))) void vxfec_nccl_get_paired_fault_stats(vxfec_nccl_paired_fault_stats_t *out)
{
    if (out)
        *out = (vxfec_nccl_paired_fault_stats_t){0};
}

void vxf_sim_init_from_env(void)
{
}

__attribute__((visibility("default"))) void vxfec_nccl_reset_stats(void)
{
    reset_common_stats();
}

__attribute__((visibility("default"))) void vxfec_nccl_set_fault_injection(uint32_t loss_ppm, uint32_t burst_len, bool reorder)
{
    (void)loss_ppm;
    (void)burst_len;
    (void)reorder;
}

__attribute__((visibility("default"))) void vxfec_nccl_set_handshake_fault(uint32_t drop_hellos, uint32_t drop_acks)
{
    (void)drop_hellos;
    (void)drop_acks;
}

__attribute__((visibility("default"))) void vxfec_nccl_set_done_fault(uint32_t drop_dones)
{
    (void)drop_dones;
}

__attribute__((visibility("default"))) void vxfec_nccl_set_abort_fault(uint32_t drop_aborts)
{
    (void)drop_aborts;
}

__attribute__((visibility("default"))) void vxfec_nccl_set_gpu_rx_fault(uint32_t fail_writes)
{
    (void)fail_writes;
}

__attribute__((visibility("default"))) void vxfec_nccl_set_congestion_sim(uint32_t ecn_ppm, uint32_t hot_port_mask,
                                                                          uint64_t max_rate_mbps)
{
    (void)ecn_ppm;
    (void)hot_port_mask;
    (void)max_rate_mbps;
}

__attribute__((visibility("default"))) void vxfec_nccl_set_link_delay_ns(uint64_t one_way_delay_ns)
{
    (void)one_way_delay_ns;
}

__attribute__((visibility("default"))) void
vxfec_nccl_set_asymmetric_link_sim(uint32_t primary_port_mask, uint64_t primary_delay_ns, uint64_t backup_delay_ns,
                                   uint32_t primary_loss_ppm, uint32_t blackout_period_pkts, uint32_t blackout_len_pkts)
{
    (void)primary_port_mask;
    (void)primary_delay_ns;
    (void)backup_delay_ns;
    (void)primary_loss_ppm;
    (void)blackout_period_pkts;
    (void)blackout_len_pkts;
}

#endif
