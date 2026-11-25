#ifndef VXFEC_NCCL_NET_H
#define VXFEC_NCCL_NET_H

#include "nccl_net_v8.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VXFEC_NCCL_STATS_V1_BYTES (17u * sizeof(uint64_t))
#define VXFEC_NCCL_STATS_V2_BYTES (19u * sizeof(uint64_t))
#define VXFEC_NCCL_STATS_V3_BYTES (24u * sizeof(uint64_t))
#define VXFEC_NCCL_STATS_ABI_VER 4u

typedef struct {
    uint64_t tx_messages;
    uint64_t tx_sys_pkts;
    uint64_t tx_proactive_rep_pkts;
    uint64_t tx_sack_retx_pkts;
    uint64_t tx_harq_ir_pkts;
    uint64_t rx_messages;
    uint64_t rx_fast_completions;
    uint64_t rx_fec_decodes;
    uint64_t rx_rs8_decodes;
    uint64_t rx_rq_decodes;
    uint64_t rx_ooo_pkts;
    uint64_t vk_offload_calls;
    uint64_t sim_dropped_pkts;
    uint64_t rx_ecn_ce_pkts;
    uint64_t tx_cnp_events;
    uint64_t tx_paced_slices;
    uint64_t tx_fec_clamped_pkts;
    uint64_t ibv_direct_pkts;
    uint64_t vk_async_polls;
    /* Nonempty systematic packets, including retransmits. */
    uint64_t tx_sys_ibv_pkts;
    uint64_t tx_sys_gpu_direct_pkts;
    uint64_t rx_sys_pkts;
    uint64_t rx_sys_gpu_direct_pkts;
    /* Systematic and repair payloads accepted by the UDP socket. */
    uint64_t tx_udp_payload_pkts;
    /* Completed RX FEC generations that consumed a native CUDA repair; subset of rx_fec_decodes. */
    uint64_t rx_cuda_repair_completions;
} vxfec_nccl_stats_t;

/* v1 ABI writes only the original 17 fields; use get_stats_ex for v2+ fields */
void vxfec_nccl_get_stats(vxfec_nccl_stats_t *out);
void vxfec_nccl_get_stats_ex(vxfec_nccl_stats_t *out, size_t out_bytes);
void vxfec_nccl_reset_stats(void);

#define VXFEC_NCCL_SEND_HISTOGRAM_CAPACITY 256u

typedef struct {
    uint64_t size_bytes;
    uint64_t requests;
    uint64_t bytes;
} vxfec_nccl_send_size_bin_t;

/* A window counts accepted isend requests across all communicators in this process.
 * A successful begin opens an empty window. A second begin returns -EBUSY.
 * End closes the window and returns the number of exact-size bins copied, in
 * unspecified order. Errors leave the output untouched: -EINVAL for no open
 * window or invalid output, -EOVERFLOW for more than CAPACITY distinct sizes
 * or a counter overflow, and -ENOSPC for insufficient output space. */
int vxfec_nccl_send_histogram_begin(void);
int vxfec_nccl_send_histogram_end(vxfec_nccl_send_size_bin_t *bins, size_t capacity);

typedef struct {
    char name[128];
    uint64_t tx_sys_gpu_direct_pkts;
    uint64_t rx_sys_gpu_direct_pkts;
} vxfec_nccl_rail_stats_t;

/* Returns the total number of devices; copies up to capacity in NCCL device order. */
size_t vxfec_nccl_get_rail_stats(vxfec_nccl_rail_stats_t *out, size_t capacity);

/* match interface name against NCCL_SOCKET_IFNAME / VXFEC_NCCL_IFNAMES syntax */
static inline bool vxfec_match_ifname(const char *ifname, const char *filter)
{
    if (!ifname || !*ifname || !filter)
        return false;

    const char *p = filter;
    while (*p == ',' || *p == ' ' || *p == '\t')
        p++;
    if (!*p)
        return false;

    bool negate_list = false;
    bool exact_list = false;
    while (*p == '^' || *p == '=') {
        if (*p == '^')
            negate_list = true;
        else
            exact_list = true;
        p++;
    }
    if (negate_list && strcmp(ifname, "lo") == 0)
        return false;

    bool any_token = false;
    bool matched_include = false;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t')
            p++;
        if (!*p)
            break;
        const char *end = p;
        while (*end && *end != ',' && *end != ' ' && *end != '\t')
            end++;

        const char *pat = p;
        size_t pat_len = (size_t)(end - p);
        bool tok_exclude = negate_list;
        bool exact = exact_list;
        while (pat_len > 0u && (*pat == '^' || *pat == '=')) {
            if (*pat == '^')
                tok_exclude = true;
            else
                exact = true;
            pat++;
            pat_len--;
        }
        if (pat_len > 0u) {
            any_token = true;
            bool hit =
                exact ? (strlen(ifname) == pat_len && memcmp(ifname, pat, pat_len) == 0) : (strncmp(ifname, pat, pat_len) == 0);
            if (hit) {
                if (tok_exclude)
                    return false;
                matched_include = true;
            }
        }
        p = end;
    }

    if (!any_token)
        return false;
    return negate_list ? true : matched_include;
}
void vxfec_nccl_set_fault_injection(uint32_t loss_ppm, uint32_t burst_len, bool reorder);
typedef struct {
    uint64_t systematic_trials;
    uint64_t drop_fingerprint;
} vxfec_nccl_paired_fault_stats_t;

/* Configure before issuing an operation, with older operations drained. A zero rate disables loss.
 * A paired loss schedule affects only first transmissions of nonempty systematic payloads.
 * Counters cover all paired operations since vxfec_nccl_reset_stats(). */
bool vxfec_nccl_set_paired_fault(uint32_t loss_ppm, uint32_t burst_len, uint64_t seed, uint32_t rank, uint32_t iteration);
void vxfec_nccl_get_paired_fault_stats(vxfec_nccl_paired_fault_stats_t *out);
void vxfec_nccl_set_handshake_fault(uint32_t drop_hellos, uint32_t drop_acks);
void vxfec_nccl_set_done_fault(uint32_t drop_dones);
void vxfec_nccl_set_abort_fault(uint32_t drop_aborts);
void vxfec_nccl_set_gpu_rx_fault(uint32_t fail_writes);
void vxfec_nccl_set_congestion_sim(uint32_t ecn_ppm, uint32_t hot_port_mask, uint64_t max_rate_mbps);
void vxfec_nccl_set_link_delay_ns(uint64_t one_way_delay_ns);
void vxfec_nccl_set_codec_override(const char *mode);

#ifdef __cplusplus
}
#endif

#endif /* VXFEC_NCCL_NET_H */
