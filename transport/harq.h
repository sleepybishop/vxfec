#ifndef HARQ_H
#define HARQ_H

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * codec-agnostic adaptive HARQ + multi-path ECMP spraying primitives:
 *   1. independent-loss proactive parity (P_0 = 0 on lossless links)
 *   2. selective repeat / SACK (RFC 2018) for tiny packet drops
 *   3. type-II HARQ incremental redundancy for burst losses
 *   4. Bresenham / SWRR multi-port spraying with ECN weight deflection
 * header-only, zero-malloc, freestanding C11 (no -lm required).
 */

#define HARQ_IID_FAILURE_TARGET 0.001

typedef enum {
    HARQ_COMPLETE = 0,   /* all K systematic symbols arrived */
    HARQ_DECODE_NOW = 1, /* enough parity symbols on hand to decode */
    HARQ_SACK_RETX = 2,  /* retransmit missing systematic symbols */
    HARQ_IR_PARITY = 3   /* request fresh IR parity symbols */
} harq_action_t;

typedef struct {
    harq_action_t action;
    uint32_t request_count; /* SACK symbol count or IR parity count */
} harq_decision_t;

/*
 * parity budget for >= K + rank_slack arrivals with probability 0.999,
 * capped by max_parity and p <= 0.90; sums negative-binomial mass around
 * the mode to avoid (1 - p)^(K + rank_slack) underflow.
 */
static inline uint32_t harq_iid_parity(uint32_t K, double p, uint32_t rank_slack, uint32_t max_parity)
{
    if (K == 0u || !(p > 0.0) || max_parity == 0u)
        return 0u;
    if (p > 0.90)
        p = 0.90;
    double q = 1.0 - p;
    const double target = HARQ_IID_FAILURE_TARGET * (1.0 - 1e-9); /* guard for tail-sum rounding */

    /* no rank slack needed if originals alone meet failure target */
    if ((double)K * p <= target)
        return 0u;
    double no_loss = 1.0, power = q;
    for (uint32_t n = K; n != 0u; n >>= 1u) {
        if (n & 1u)
            no_loss *= power;
        power *= power;
    }
    if (1.0 - no_loss <= target)
        return 0u;

    const double last = (double)K + rank_slack - 1.0;
    uint64_t mode = (uint64_t)(last * p / q);
    double lower = 1.0, upper = 0.0, term = 1.0;
    for (uint64_t r = mode; r > 0u; r--) {
        double ratio = (double)r / (p * (last + (double)r));
        term *= ratio;
        lower += term;
        if (term <= 1e-13 * (1.0 - ratio))
            break;
    }
    term = 1.0;
    double remainder;
    for (uint64_t r = mode + 1u;; r++) {
        double ratio = p * (last + (double)r) / (double)r;
        term *= ratio;
        upper += term;
        if (term <= 1e-13 * (1.0 - ratio)) {
            remainder = term * ratio / (1.0 - ratio);
            break;
        }
    }

    const double threshold = target * (lower + upper);
    double tail = upper + remainder;
    uint64_t losses = mode;
    term = 1.0;
    if (tail > threshold) {
        while (tail > threshold && losses + rank_slack < max_parity) {
            losses++;
            term *= p * (last + (double)losses) / (double)losses;
            tail -= term;
        }
    } else {
        while (losses > 0u && tail + term <= threshold) {
            tail += term;
            term *= (double)losses / (p * (last + (double)losses));
            losses--;
        }
    }
    uint64_t repairs = losses + rank_slack;
    return repairs < max_parity ? (uint32_t)repairs : max_parity;
}

/* update EWMA packet loss estimate; snaps to 0.0 below 2e-6 on clean links */
static inline double harq_ewma_update(double prev_ewma, uint32_t tx_pkts, uint32_t lost_pkts, double alpha)
{
    if (tx_pkts == 0u)
        return prev_ewma;
    if (alpha <= 0.0 || alpha > 1.0)
        alpha = 0.125;
    double sample = (lost_pkts >= tx_pkts) ? 1.0 : ((double)lost_pkts / (double)tx_pkts);
    double next = (prev_ewma <= 0.0) ? sample : ((1.0 - alpha) * prev_ewma + alpha * sample);
    if (lost_pkts == 0u && next < 2e-6)
        return 0.0;
    return next;
}

/* select immediate completion, FEC decode, SACK retransmit, or HARQ-IR */
static inline harq_decision_t harq_plan_feedback(uint32_t missing_sys, uint32_t rx_repairs, uint32_t sack_thresh, uint32_t ir_slack)
{
    harq_decision_t d = {HARQ_COMPLETE, 0u};
    if (missing_sys == 0u)
        return d;

    uint32_t target_repairs = (UINT32_MAX - missing_sys < ir_slack) ? UINT32_MAX : (missing_sys + ir_slack);
    if (rx_repairs >= target_repairs) {
        d.action = HARQ_DECODE_NOW;
        return d;
    }

    uint32_t ir_deficit = target_repairs - rx_repairs;
    if (missing_sys <= sack_thresh && (rx_repairs == 0u || missing_sys < ir_deficit)) {
        d.action = HARQ_SACK_RETX;
        d.request_count = missing_sys;
        return d;
    }

    d.action = HARQ_IR_PARITY;
    d.request_count = ir_deficit;
    return d;
}

/* Bresenham / SWRR multi-port ECMP selector */
static inline uint32_t swrr_next(int32_t *cur, const int32_t *weights, uint32_t n)
{
    if (n <= 1u || !cur)
        return 0u;
    if (!weights) {
        uint32_t idx = (uint32_t)(*cur) % n;
        *cur = (int32_t)((idx + 1u) % n);
        return idx;
    }

    int32_t total = 0;
    uint32_t best = 0u;
    int32_t best_val = INT32_MIN;
    for (uint32_t i = 0; i < n; i++) {
        int32_t w = (weights[i] > 0) ? weights[i] : 1;
        cur[i] += w;
        total += w;
        if (cur[i] > best_val) {
            best_val = cur[i];
            best = i;
        }
    }
    cur[best] -= total;
    return best;
}

/*
 * congestion-window + token-bucket pacer with ECN / queue-delay control:
 *   - non-congestive drops keep cwnd and pacing rate untouched
 *   - ECN or sustained queue delay paces cwnd/rate
 *   - proactive parity clamping is gated strictly on verified ECN (cong_level)
 */
#define HARQ_MIN_RTT_WINDOW_NS 2000000000ULL

typedef struct {
    uint64_t cwnd_bytes;
    uint64_t min_cwnd_bytes;
    uint64_t max_cwnd_bytes;
    uint64_t rate_Bps;
    uint64_t min_rate_Bps;
    uint64_t max_rate_Bps;
    uint64_t max_burst_bytes;
    double tokens_bytes;
    uint64_t last_refill_ns;
    uint64_t min_rtt_ns;
    uint64_t min_rtt_stamp_ns;
    uint64_t srtt_ns;
    uint64_t target_qdelay_ns;
    double ecn_alpha;
    double cong_level;
} harq_pacer_t;

static inline void harq_pacer_init(harq_pacer_t *p, uint64_t init_cwnd, uint64_t min_cwnd, uint64_t max_cwnd, uint64_t min_rate_Bps,
                                   uint64_t max_rate_Bps, uint64_t target_qdelay_ns, uint64_t now_ns)
{
    if (!p)
        return;
    if (min_cwnd < 4096u)
        min_cwnd = 4096u;
    if (max_cwnd < min_cwnd)
        max_cwnd = min_cwnd;
    if (init_cwnd < min_cwnd)
        init_cwnd = min_cwnd;
    if (init_cwnd > max_cwnd)
        init_cwnd = max_cwnd;
    if (min_rate_Bps == 0u)
        min_rate_Bps = 1000000ULL;
    if (max_rate_Bps == 0u)
        max_rate_Bps = min_rate_Bps;
    else if (min_rate_Bps > max_rate_Bps)
        min_rate_Bps = max_rate_Bps;

    p->cwnd_bytes = init_cwnd;
    p->min_cwnd_bytes = min_cwnd;
    p->max_cwnd_bytes = max_cwnd;
    p->rate_Bps = max_rate_Bps;
    p->min_rate_Bps = min_rate_Bps;
    p->max_rate_Bps = max_rate_Bps;
    p->max_burst_bytes = 1024ULL * 1024ULL;
    uint64_t init_tok = (init_cwnd < p->max_burst_bytes) ? init_cwnd : p->max_burst_bytes;
    p->tokens_bytes = (double)init_tok;
    p->last_refill_ns = now_ns;
    p->min_rtt_ns = 0u;
    p->min_rtt_stamp_ns = now_ns;
    p->srtt_ns = 0u;
    p->target_qdelay_ns = target_qdelay_ns;
    p->ecn_alpha = 0.0;
    p->cong_level = 0.0;
}

static inline uint64_t harq_pacer_budget(harq_pacer_t *p, uint64_t inflight_bytes, uint64_t now_ns)
{
    if (!p)
        return 0u;
    if (now_ns > p->last_refill_ns) {
        uint64_t dt_ns = now_ns - p->last_refill_ns;
        p->last_refill_ns = now_ns;
        p->tokens_bytes += (double)p->rate_Bps * (double)dt_ns * 1e-9;
    }
    uint64_t burst_cap = (p->cwnd_bytes < p->max_burst_bytes) ? p->cwnd_bytes : p->max_burst_bytes;
    if (burst_cap < p->min_cwnd_bytes)
        burst_cap = p->min_cwnd_bytes;
    if (p->tokens_bytes > (double)burst_cap)
        p->tokens_bytes = (double)burst_cap;

    uint64_t tok = (p->tokens_bytes > 0.0) ? (uint64_t)p->tokens_bytes : 0u;
    if (inflight_bytes >= p->cwnd_bytes)
        return 0u;
    uint64_t win_avail = p->cwnd_bytes - inflight_bytes;
    return (win_avail < tok) ? win_avail : tok;
}

static inline void harq_pacer_consume(harq_pacer_t *p, uint64_t sent_bytes)
{
    if (!p)
        return;
    double sb = (double)sent_bytes;
    p->tokens_bytes = (sb >= p->tokens_bytes) ? 0.0 : (p->tokens_bytes - sb);
}

static inline bool harq_pacer_on_ack(harq_pacer_t *p, uint64_t acked_bytes, uint64_t rtt_ns, uint32_t rx_pkts, uint32_t ecn_ce_pkts)
{
    if (!p)
        return false;

    if (rtt_ns > 0u) {
        bool expired = (p->min_rtt_stamp_ns > 0u && p->last_refill_ns > p->min_rtt_stamp_ns &&
                        (p->last_refill_ns - p->min_rtt_stamp_ns) >= HARQ_MIN_RTT_WINDOW_NS);
        if (p->min_rtt_ns == 0u || rtt_ns <= p->min_rtt_ns || expired) {
            p->min_rtt_ns = rtt_ns;
            p->min_rtt_stamp_ns = p->last_refill_ns;
        }
        p->srtt_ns = (p->srtt_ns == 0u) ? rtt_ns : ((7u * p->srtt_ns + rtt_ns) / 8u);
    }

    double ecn_frac = 0.0;
    if (rx_pkts > 0u && ecn_ce_pkts > 0u)
        ecn_frac = (ecn_ce_pkts >= rx_pkts) ? 1.0 : ((double)ecn_ce_pkts / (double)rx_pkts);

    p->ecn_alpha = 0.875 * p->ecn_alpha + 0.125 * ecn_frac;
    if (ecn_ce_pkts == 0u && p->ecn_alpha < 1e-4)
        p->ecn_alpha = 0.0;

    if (ecn_ce_pkts > 0u) {
        double sev_ecn = (ecn_frac > p->ecn_alpha) ? ecn_frac : p->ecn_alpha;
        if (sev_ecn < 0.05)
            sev_ecn = 0.05;
        if (sev_ecn > 0.80)
            sev_ecn = 0.80;
        p->cong_level = 0.60 * p->cong_level + 0.40 * sev_ecn;
        if (p->cong_level > 1.0)
            p->cong_level = 1.0;
    } else {
        p->cong_level *= 0.75;
        if (p->cong_level < 1e-3)
            p->cong_level = 0.0;
    }

    uint64_t qdelay_ns = (rtt_ns > p->min_rtt_ns && p->min_rtt_ns > 0u) ? (rtt_ns - p->min_rtt_ns) : 0u;
    bool delay_cong = (p->target_qdelay_ns > 0u && qdelay_ns > p->target_qdelay_ns);

    if (ecn_ce_pkts > 0u || delay_cong) {
        double sev_ecn = (ecn_ce_pkts > 0u) ? ((ecn_frac > p->ecn_alpha) ? ecn_frac : p->ecn_alpha) : 0.0;
        double sev_del = (delay_cong && rtt_ns > 0u) ? ((double)(qdelay_ns - p->target_qdelay_ns) / (double)rtt_ns) : 0.0;
        double sev = (sev_ecn > sev_del) ? sev_ecn : sev_del;
        if (sev < 0.05)
            sev = 0.05;
        if (sev > 0.80)
            sev = 0.80;

        double factor = 1.0 - 0.5 * sev;
        uint64_t next_cwnd = (uint64_t)((double)p->cwnd_bytes * factor);
        if (next_cwnd < p->min_cwnd_bytes)
            next_cwnd = p->min_cwnd_bytes;
        p->cwnd_bytes = next_cwnd;

        uint64_t next_rate = (uint64_t)((double)p->rate_Bps * factor);
        if (next_rate < p->min_rate_Bps)
            next_rate = p->min_rate_Bps;
        p->rate_Bps = next_rate;

        if (p->tokens_bytes > (double)p->cwnd_bytes)
            p->tokens_bytes = (double)p->cwnd_bytes;
        return true;
    }

    uint64_t cwnd_inc = acked_bytes / 8u;
    if (cwnd_inc < 4096u)
        cwnd_inc = 4096u;
    p->cwnd_bytes = (p->max_cwnd_bytes - p->cwnd_bytes > cwnd_inc) ? (p->cwnd_bytes + cwnd_inc) : p->max_cwnd_bytes;

    uint64_t rate_inc = p->rate_Bps / 16u + p->min_rate_Bps / 16u;
    p->rate_Bps = (p->max_rate_Bps - p->rate_Bps > rate_inc) ? (p->rate_Bps + rate_inc) : p->max_rate_Bps;
    return false;
}

static inline uint32_t harq_clamp_proactive_parity(uint32_t raw_p0, double cong_level)
{
    if (raw_p0 == 0u || cong_level <= 0.05)
        return raw_p0;
    if (cong_level >= 0.45)
        return 0u;
    double scale = 1.0 - ((cong_level - 0.05) / 0.40);
    return (uint32_t)((double)raw_p0 * scale);
}

/* preserve rank margin on any repairs sent after ECN clamping */
static inline uint32_t harq_paced_iid_parity(uint32_t K, double loss_ewma, uint32_t rank_slack, uint32_t max_parity,
                                             double cong_level, uint32_t *out_clamped)
{
    uint32_t raw = harq_iid_parity(K, loss_ewma, rank_slack, max_parity);
    uint32_t margin = (raw > rank_slack) ? rank_slack : 0u;
    uint32_t adj = harq_clamp_proactive_parity(raw - margin, cong_level);
    if (adj > 0u)
        adj += margin;
    if (out_clamped)
        *out_clamped = raw - adj;
    return adj;
}

static inline bool harq_update_port_weights(int32_t *weights, uint32_t num_ports, uint32_t hot_port_mask, int32_t max_w)
{
    if (!weights || num_ports == 0u || max_w <= 1)
        return false;
    bool nonuniform = false;
    for (uint32_t p = 0; p < num_ports; p++) {
        bool hot = (p < 32u) && (((hot_port_mask >> p) & 1u) != 0u);
        if (hot) {
            weights[p] = (weights[p] > 2) ? (weights[p] / 2) : 1;
        } else if (weights[p] < max_w) {
            weights[p]++;
        }
        if (weights[p] != max_w)
            nonuniform = true;
    }
    return nonuniform;
}

#endif
