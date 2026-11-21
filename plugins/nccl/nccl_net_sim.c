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

atomic_int vxf_env_codec_ovr = VXFEC_CODEC_OVR_AUTO;
atomic_int vxf_codec_override = VXFEC_CODEC_OVR_AUTO;

_Static_assert(offsetof(vxfec_nccl_stats_t, ibv_direct_pkts) == VXFEC_NCCL_STATS_V1_BYTES,
               "vxfec_nccl_stats_t v1 prefix size mismatch");

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

__attribute__((visibility("default"))) void vxfec_nccl_set_codec_override(const char *mode)
{
    int ovr = mode ? (int)vxf_parse_codec_mode(mode) : atomic_load_explicit(&vxf_env_codec_ovr, memory_order_relaxed);
    atomic_store_explicit(&vxf_codec_override, ovr, memory_order_release);
}

#if VXFEC_ENABLE_SIM

atomic_bool vxf_sim_active = false;
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
static atomic_uint g_rng_state = 0x9E3779B9u;

static void refresh_sim_active(void)
{
    bool active = atomic_load_explicit(&g_sim_loss_ppm, memory_order_relaxed) != 0u ||
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
            atomic_fetch_add_explicit(&vxf_st_sim_dropped_pkts, 1, memory_order_relaxed);
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

__attribute__((visibility("default"))) void vxfec_nccl_reset_stats(void)
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
    atomic_store(&g_sim_burst_rem, 0);
    refresh_sim_active();
}

__attribute__((visibility("default"))) void vxfec_nccl_set_fault_injection(uint32_t loss_ppm, uint32_t burst_len, bool reorder)
{
    atomic_store(&g_sim_loss_ppm, loss_ppm);
    atomic_store(&g_sim_burst_len, burst_len ? burst_len : 1u);
    atomic_store(&vxf_sim_reorder, reorder);
    atomic_store(&g_sim_burst_rem, 0u);
    refresh_sim_active();
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

#else /* !VXFEC_ENABLE_SIM */

void vxf_sim_init_from_env(void)
{
}

__attribute__((visibility("default"))) void vxfec_nccl_reset_stats(void)
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

#endif
