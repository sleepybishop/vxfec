/*
 * wire-strided VRAM superframe benchmark:
 *   - continuous TX parity encode and RX decode across 8 poisoned masks
 *   - paced 10.00 ms arrival-cadence SLA check and VRAM copy contention
 *   - codec-isolated uncapped fade sweep with 100% output verification
 */

#include "vxfec_interleave.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEADLINE_MS 10.0
#define DEADLINE_NS 10000000ULL
#define DRAIN_TIMEOUT_NS (30ULL * 1000000000ULL)
#define PIPE_ITERS 8
#define SEED_CHUNK_BYTES (1031ULL * 4096ULL) /* prime-multiple chunk prevents row/block aliasing */

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

typedef struct {
    vxfec_vk_ctx *ctx;
    vxfec_vk_buf *buf_a;
    vxfec_vk_buf *buf_b;
    size_t copy_bytes;
    double target_rw_gbs;
    atomic_int running;
    atomic_ullong rw_bytes;
} vram_copy_stress_ctx;

static void *vram_copy_stress_main(void *arg)
{
    vram_copy_stress_ctx *d = (vram_copy_stress_ctx *)arg;
    bool flip = false;
    uint64_t start_t = now_ns();
    uint64_t issued_rw = 0;

    while (atomic_load_explicit(&d->running, memory_order_relaxed)) {
        vxfec_vk_buf *src = flip ? d->buf_b : d->buf_a;
        vxfec_vk_buf *dst = flip ? d->buf_a : d->buf_b;
        if (!vxfec_vk_buf_copy(d->ctx, dst, src, d->copy_bytes))
            break;
        uint64_t rw = (uint64_t)d->copy_bytes * 2ULL;
        issued_rw += rw;
        atomic_fetch_add_explicit(&d->rw_bytes, (unsigned long long)rw, memory_order_relaxed);
        flip = !flip;

        if (d->target_rw_gbs > 0.0) {
            uint64_t ideal_ns = (uint64_t)((double)issued_rw / d->target_rw_gbs);
            while (atomic_load_explicit(&d->running, memory_order_relaxed) && (now_ns() - start_t) < ideal_ns) {
            }
        }
    }
    return NULL;
}

static bool vram_copy_stress_start(vram_copy_stress_ctx *d, double target_rw_gbs, pthread_t *out_thr)
{
    memset(d, 0, sizeof(*d));
    const char *prev_qf = getenv("VXFEC_VK_QUEUE_FAMILY");
    char saved_qf[32] = {0};
    bool had_prev_qf = (prev_qf != NULL);
    if (had_prev_qf)
        snprintf(saved_qf, sizeof(saved_qf), "%s", prev_qf);

    /* queue family 0 (graphics+transfer) runs concurrently with dedicated ACE compute */
    setenv("VXFEC_VK_QUEUE_FAMILY", "0", 1);
    d->ctx = vxfec_vk_ctx_new();
    if (had_prev_qf)
        setenv("VXFEC_VK_QUEUE_FAMILY", saved_qf, 1);
    else
        unsetenv("VXFEC_VK_QUEUE_FAMILY");

    if (!d->ctx)
        d->ctx = vxfec_vk_ctx_new();
    if (!d->ctx)
        return false;

    d->copy_bytes = vxfec_vk_is_discrete(d->ctx) ? (128ULL * 1024ULL * 1024ULL) : (16ULL * 1024ULL * 1024ULL);
    d->target_rw_gbs = target_rw_gbs;
    d->buf_a = vxfec_vk_buf_new(d->ctx, d->copy_bytes, false);
    d->buf_b = vxfec_vk_buf_new(d->ctx, d->copy_bytes, false);
    if (!d->buf_a || !d->buf_b) {
        vxfec_vk_buf_free(d->ctx, d->buf_b);
        vxfec_vk_buf_free(d->ctx, d->buf_a);
        vxfec_vk_ctx_free(d->ctx);
        return false;
    }

    uint8_t seed[4096];
    for (size_t i = 0; i < sizeof(seed); i++)
        seed[i] = (uint8_t)(i * 37u + 13u);
    (void)vxfec_vk_buf_write_offset(d->ctx, d->buf_a, 0, seed, sizeof(seed));

    atomic_store_explicit(&d->running, 1, memory_order_relaxed);
    atomic_store_explicit(&d->rw_bytes, 0ULL, memory_order_relaxed);
    if (pthread_create(out_thr, NULL, vram_copy_stress_main, d) != 0) {
        vxfec_vk_buf_free(d->ctx, d->buf_b);
        vxfec_vk_buf_free(d->ctx, d->buf_a);
        vxfec_vk_ctx_free(d->ctx);
        return false;
    }
    return true;
}

static void vram_copy_stress_stop(vram_copy_stress_ctx *d, pthread_t thr)
{
    atomic_store_explicit(&d->running, 0, memory_order_relaxed);
    pthread_join(thr, NULL);
    vxfec_vk_buf_free(d->ctx, d->buf_b);
    vxfec_vk_buf_free(d->ctx, d->buf_a);
    vxfec_vk_ctx_free(d->ctx);
}

static void init_seed_chunk(uint8_t *seed, size_t len)
{
    for (size_t i = 0; i < len; i++)
        seed[i] = (uint8_t)((i * 131u + (i >> 11) * 67u + 29u) & 0xffu);
}

static bool fill_vram_pattern(vxfec_vk_ctx *ctx, vxfec_vk_buf *vram, size_t total_bytes, const uint8_t *seed, size_t seed_len)
{
    size_t first = (total_bytes < seed_len) ? total_bytes : seed_len;
    if (!vxfec_vk_buf_write_offset(ctx, vram, 0, seed, first))
        return false;

    size_t filled = first;
    while (filled < total_bytes) {
        size_t step = (total_bytes - filled < filled) ? (total_bytes - filled) : filled;
        if (!vxfec_vk_buf_copy_offset(ctx, vram, filled, vram, 0, step))
            return false;
        filled += step;
    }
    return true;
}

typedef struct {
    uint64_t tickets[2];
    uint64_t arrivals[2];
    uint64_t lat_sum;
    uint64_t max_lat_ns;
    uint32_t misses;
} sla_progress_t;

static bool poll_sla_slots(vxfec_vk_ctx *ctx, sla_progress_t *sla)
{
    for (int s = 0; s < 2; s++) {
        if (sla->tickets[s] == 0u)
            continue;
        int pst = vxfec_vk_poll_status(ctx, sla->tickets[s]);
        if (pst < 0)
            return false;
        uint64_t elapsed = now_ns() - sla->arrivals[s];
        if (pst > 0) {
            sla->lat_sum += elapsed;
            if (elapsed > sla->max_lat_ns)
                sla->max_lat_ns = elapsed;
            if (elapsed > DEADLINE_NS)
                sla->misses++;
            sla->tickets[s] = 0u;
        } else if (elapsed > DRAIN_TIMEOUT_NS) {
            return false;
        }
    }
    return true;
}

/* build coalesced VRAM byte spans for all erased systematic wire slots */
static bool build_erased_sys_spans(const uint64_t *bits, uint32_t Z, uint32_t K, size_t wpb, uint32_t stride,
                                   vxfec_vk_span **out_spans, size_t *out_count)
{
    *out_spans = NULL;
    *out_count = 0u;
    uint64_t sys_frames = (uint64_t)K * Z;
    size_t cap = 256u, cnt = 0u;
    vxfec_vk_span *spans = (vxfec_vk_span *)malloc(cap * sizeof(*spans));
    if (!spans)
        return false;

    uint64_t seq = 0;
    while (seq < sys_frames) {
        uint32_t z = 0, esi = 0;
        vxfec_interleave_unwire_seq(Z, seq, &z, &esi);
        if (((bits[(size_t)z * wpb + (esi >> 6u)] >> (esi & 63u)) & 1ULL) != 0ULL) {
            seq++;
            continue;
        }
        uint64_t run_beg = seq++;
        while (seq < sys_frames) {
            vxfec_interleave_unwire_seq(Z, seq, &z, &esi);
            if (((bits[(size_t)z * wpb + (esi >> 6u)] >> (esi & 63u)) & 1ULL) != 0ULL)
                break;
            seq++;
        }
        if (cnt == cap) {
            size_t ncap = cap * 2u;
            vxfec_vk_span *tmp = (vxfec_vk_span *)realloc(spans, ncap * sizeof(*spans));
            if (!tmp) {
                free(spans);
                return false;
            }
            spans = tmp;
            cap = ncap;
        }
        spans[cnt++] = (vxfec_vk_span){
            .offset = (size_t)run_beg * stride,
            .bytes = (size_t)(seq - run_beg) * stride,
        };
    }
    *out_spans = spans;
    *out_count = cnt;
    return true;
}

/* poison all erased systematic wire slots in wire_vram in one batched command buffer */
static bool poison_erased_sys_vram(vxfec_vk_ctx *ctx, vxfec_vk_buf *vram, const uint64_t *bits, uint32_t Z, uint32_t K, size_t wpb,
                                   uint32_t stride)
{
    vxfec_vk_span *spans = NULL;
    size_t cnt = 0u;
    if (!build_erased_sys_spans(bits, Z, K, wpb, stride, &spans, &cnt))
        return false;
    bool ok = vxfec_vk_buf_fill_spans(ctx, vram, spans, cnt, 0xa5a5a5a5u);
    free(spans);
    return ok;
}

/* verify 100% of sys_bytes in wire_vram bit-for-bit against the reference seed */
static bool verify_full_superframe(vxfec_vk_ctx *ctx, vxfec_vk_buf *vram, size_t sys_bytes, const uint8_t *seed, size_t seed_len,
                                   uint8_t *read_scratch)
{
    size_t off = 0;
    while (off < sys_bytes) {
        size_t chunk = (sys_bytes - off < seed_len) ? (sys_bytes - off) : seed_len;
        if (!vxfec_vk_buf_read_offset(ctx, vram, off, read_scratch, chunk))
            return false;
        if (memcmp(read_scratch, seed, chunk) != 0)
            return false;
        off += chunk;
    }
    return true;
}

typedef enum {
    FADE_INTERIOR = 0,
    FADE_PARITY_TAIL = 1,
    FADE_BOUNDARY_WRAP = 2,
} fade_pos_t;

typedef struct {
    double wire_loss_pct;
    double avg_sys_e;
    uint32_t min_sys_e;
    uint32_t max_sys_e;
    uint32_t min_surviving;
    uint32_t unique_masks;
} channel_summary_t;

/* uncapped wire-level burst + Bernoulli random erasure generator */
static void build_uncapped_wire_channel(uint64_t *bits, uint32_t Z, uint32_t K, uint32_t P, size_t wpb, uint32_t burst_pct_wire,
                                        uint32_t rand_ppm, fade_pos_t pos, uint32_t frame_salt, channel_summary_t *out)
{
    memset(bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
    uint64_t total_wire = (uint64_t)(K + P) * Z;
    uint64_t sys_wire = (uint64_t)K * Z;
    uint64_t burst_len = (total_wire * (uint64_t)burst_pct_wire) / 100ULL;
    if (burst_len > total_wire)
        burst_len = total_wire;

    uint64_t burst_beg = 0;
    if (pos == FADE_INTERIOR) {
        uint64_t span = (total_wire > burst_len) ? (total_wire - burst_len) : 1ULL;
        burst_beg = ((span * (11ULL + (uint64_t)(frame_salt * 13u) % 67ULL)) / 100ULL) + (frame_salt * 7ULL) % (Z ? Z : 1u);
        if (burst_beg + burst_len > total_wire)
            burst_beg = total_wire - burst_len;
    } else if (pos == FADE_PARITY_TAIL) {
        uint64_t half = burst_len / 2ULL;
        uint64_t jitter = (uint64_t)(frame_salt * 17u) % (Z ? Z : 1u);
        burst_beg = (sys_wire > half + jitter) ? (sys_wire - half - jitter) : 0ULL;
        if (burst_beg + burst_len > total_wire)
            burst_beg = total_wire - burst_len;
    } else {
        uint64_t tail_part = (burst_len * (35ULL + (frame_salt * 19ULL) % 35ULL)) / 100ULL;
        burst_beg = (total_wire > tail_part) ? (total_wire - tail_part) : 0ULL;
    }
    uint64_t burst_end = burst_beg + burst_len;

    uint32_t rng = 0x7f4a7c15u ^ (frame_salt * 0x9e3779b9u) ^ ((uint32_t)pos * 0x85ebca6bu);
    uint64_t dropped_wire = 0;

    for (uint64_t seq = 0; seq < total_wire; seq++) {
        bool in_burst = (burst_end <= total_wire) ? (seq >= burst_beg && seq < burst_end)
                                                  : (seq >= burst_beg || seq < (burst_end - total_wire));
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        bool in_rand = !in_burst && ((rng % 1000000u) < rand_ppm);
        if (in_burst || in_rand) {
            dropped_wire++;
        } else {
            vxfec_wire_mark_seq(bits, Z, wpb, seq);
        }
    }

    if (out) {
        uint32_t min_e = K, max_e = 0, sum_e = 0;
        uint32_t min_surv = K + P;
        uint32_t unique = 0;
        for (uint32_t z = 0; z < Z; z++) {
            const uint64_t *bz = bits + (size_t)z * wpb;
            uint32_t sys_ok = 0, tot_ok = 0;
            for (uint32_t e = 0; e < K + P; e++) {
                if ((bz[e >> 6u] >> (e & 63u)) & 1ULL) {
                    tot_ok++;
                    if (e < K)
                        sys_ok++;
                }
            }
            uint32_t e_cnt = K - sys_ok;
            if (e_cnt < min_e)
                min_e = e_cnt;
            if (e_cnt > max_e)
                max_e = e_cnt;
            if (tot_ok < min_surv)
                min_surv = tot_ok;
            sum_e += e_cnt;

            bool dup = false;
            for (uint32_t j = 0; j < z; j++) {
                if (memcmp(bz, bits + (size_t)j * wpb, wpb * sizeof(uint64_t)) == 0) {
                    dup = true;
                    break;
                }
            }
            if (!dup)
                unique++;
        }
        out->wire_loss_pct = (100.0 * (double)dropped_wire) / (double)total_wire;
        out->avg_sys_e = (double)sum_e / (double)Z;
        out->min_sys_e = min_e;
        out->max_sys_e = max_e;
        out->min_surviving = min_surv;
        out->unique_masks = unique;
    }
}

/* codec-isolated fade sweep: re-encode per codec, poison erased slots, and verify 100% of sys_bytes */
static bool run_fade_boundary_sweep(vxfec_vk_ctx *ctx, vxfec_wire_ring *wring_gf2, vxfec_wire_ring *wring_rq,
                                    vxfec_wire_ring *wring_mds, vxfec_vk_buf *vram_ref, vxfec_vk_buf *vram_work, uint32_t Z,
                                    uint32_t K, uint32_t P, size_t wpb, uint32_t stride, size_t sys_bytes, size_t wire_bytes,
                                    const uint8_t *seed_buf, size_t seed_len, uint8_t *verify_buf)
{
    static const struct {
        uint32_t burst_pct;
        uint32_t rand_ppm;
        fade_pos_t pos;
        const char *pos_lbl;
    } cases[] = {
        {14u, 20000u, FADE_INTERIOR, "Interior       "},      {16u, 20000u, FADE_PARITY_TAIL, "Parity-Tail    "},
        {16u, 20000u, FADE_BOUNDARY_WRAP, "Boundary-Wrap  "}, {19u, 20000u, FADE_PARITY_TAIL, "Near-Cap Tail  "},
        {22u, 20000u, FADE_BOUNDARY_WRAP, "Over-Cap Wrap  "},
    };
    enum { NUM_CASES = sizeof(cases) / sizeof(cases[0]), TRIALS = 6 };

    uint64_t *bits = (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t));
    if (!bits)
        return false;

    vxfec_wire_ring *rings[3] = {wring_gf2, wring_rq, wring_mds};
    uint32_t ok_cnt[3][NUM_CASES] = {{0}};
    uint64_t ok_ns[3][NUM_CASES] = {{0}};
    double loss_avg[NUM_CASES] = {0};

    for (int ci = 0; ci < 3; ci++) {
        if (!fill_vram_pattern(ctx, vram_ref, sys_bytes, seed_buf, seed_len) ||
            !vxfec_wire_encode_vram(rings[ci], vram_ref, NULL)) {
            free(bits);
            return false;
        }

        for (size_t c = 0; c < NUM_CASES; c++) {
            double loss_sum = 0.0;
            for (int t = 0; t < TRIALS; t++) {
                channel_summary_t cs = {0};
                build_uncapped_wire_channel(bits, Z, K, P, wpb, cases[c].burst_pct, cases[c].rand_ppm, cases[c].pos,
                                            (uint32_t)(c * 31u + (size_t)t + 100u), &cs);
                loss_sum += cs.wire_loss_pct;

                if (!vxfec_vk_buf_copy(ctx, vram_work, vram_ref, wire_bytes) ||
                    !poison_erased_sys_vram(ctx, vram_work, bits, Z, K, wpb, stride)) {
                    free(bits);
                    return false;
                }

                vxfec_wire_stats st = {0};
                if (vxfec_wire_recover_vram(rings[ci], vram_work, bits, &st)) {
                    if (!verify_full_superframe(ctx, vram_work, sys_bytes, seed_buf, seed_len, verify_buf)) {
                        fprintf(stderr, "FAIL: fade sweep codec=%d case=%zu trial=%d returned true with corrupted payload\n", ci, c,
                                t);
                        free(bits);
                        return false;
                    }
                    ok_cnt[ci][c]++;
                    ok_ns[ci][c] += st.total_wall_ns;
                }
            }
            if (ci == 0)
                loss_avg[c] = loss_sum / (double)TRIALS;
        }
    }

    printf("\n  Uncapped Fade & Superframe-Boundary Sweep (%d poisoned+verified trials/point, wire parity = %.2f%%):\n", TRIALS,
           (100.0 * (double)P) / (double)(K + P));
    printf("  Fade Profile   | Fade (ms) | Burst+Rand | Realized Loss | RQ-GF2 OK (Avg ms) | Rateless OK (Avg ms) | Strict MDS OK "
           "(Avg ms)\n");
    printf("  ---------------+-----------+------------+---------------+--------------------+----------------------+----------------"
           "-------\n");

    for (size_t c = 0; c < NUM_CASES; c++) {
        char gf2_buf[24], rq_buf[24], mds_buf[24];
        if (ok_cnt[0][c] > 0)
            snprintf(gf2_buf, sizeof(gf2_buf), "%u/%d (%5.2f ms)", ok_cnt[0][c], TRIALS,
                     (double)(ok_ns[0][c] / ok_cnt[0][c]) / 1e6);
        else
            snprintf(gf2_buf, sizeof(gf2_buf), "0/%d (   n/a  )", TRIALS);
        if (ok_cnt[1][c] > 0)
            snprintf(rq_buf, sizeof(rq_buf), "%u/%d (%5.2f ms)", ok_cnt[1][c], TRIALS, (double)(ok_ns[1][c] / ok_cnt[1][c]) / 1e6);
        else
            snprintf(rq_buf, sizeof(rq_buf), "0/%d (   n/a  )", TRIALS);
        if (ok_cnt[2][c] > 0)
            snprintf(mds_buf, sizeof(mds_buf), "%u/%d (%5.2f ms)", ok_cnt[2][c], TRIALS,
                     (double)(ok_ns[2][c] / ok_cnt[2][c]) / 1e6);
        else
            snprintf(mds_buf, sizeof(mds_buf), "0/%d (   n/a  )", TRIALS);

        double fade_ms = (DEADLINE_MS * (double)cases[c].burst_pct) / 100.0;
        printf("  %s |  %4.2f ms  | %2u%% + %3.1f%% |     %5.2f%%    | %-18s | %-20s | %-21s\n", cases[c].pos_lbl, fade_ms,
               cases[c].burst_pct, (double)cases[c].rand_ppm / 1e4, loss_avg[c], gf2_buf, rq_buf, mds_buf);
    }

    free(bits);
    return true;
}

/* error-aware nonblocking poll with bounded timeout */
static bool poll_inflight_slots(vxfec_vk_ctx *ctx, uint64_t slot_tkt[2], const uint64_t slot_sub_ns[2], uint64_t *io_max_lat_ns)
{
    uint64_t t_now = now_ns();
    for (int s = 0; s < 2; s++) {
        if (slot_tkt[s] == 0u)
            continue;
        int pst = vxfec_vk_poll_status(ctx, slot_tkt[s]);
        if (pst < 0)
            return false;
        if (pst > 0) {
            uint64_t lat = (t_now > slot_sub_ns[s]) ? (t_now - slot_sub_ns[s]) : 0u;
            if (lat > *io_max_lat_ns)
                *io_max_lat_ns = lat;
            slot_tkt[s] = 0u;
        } else if (t_now > slot_sub_ns[s] && (t_now - slot_sub_ns[s]) > DRAIN_TIMEOUT_NS) {
            return false;
        }
    }
    return true;
}

static bool prep_poisoned_pair(vxfec_vk_ctx *ctx, vxfec_vk_buf *wire_vram[2], const vxfec_vk_buf *wire_ref,
                               vxfec_vk_span *const rx_spans[PIPE_ITERS], const size_t rx_span_cnt[PIPE_ITERS], size_t wire_bytes)
{
    return vxfec_vk_buf_copy(ctx, wire_vram[0], wire_ref, wire_bytes) &&
           vxfec_vk_buf_copy(ctx, wire_vram[1], wire_ref, wire_bytes) &&
           vxfec_vk_buf_fill_spans(ctx, wire_vram[0], rx_spans[0], rx_span_cnt[0], 0xa5a5a5a5u) &&
           vxfec_vk_buf_fill_spans(ctx, wire_vram[1], rx_spans[1], rx_span_cnt[1], 0xa5a5a5a5u);
}

static bool run_pipeline_suite(vxfec_vk_ctx *ctx, double target_tbps, uint32_t K, uint32_t P, uint32_t stride, bool run_sweep)
{
    double payload_bytes_want = (target_tbps * 1e12 * (DEADLINE_MS * 1e-3)) / 8.0;
    size_t blk_sys_bytes = (size_t)K * stride;
    size_t blk_wire_bytes = (size_t)(K + P) * stride;
    uint64_t max_ssbo = vxfec_vk_max_storage_buffer_bytes(ctx);
    size_t max_z = (size_t)(max_ssbo / blk_wire_bytes);
    if (max_z == 0u)
        return false;

    uint32_t Z = (uint32_t)((payload_bytes_want + (double)blk_sys_bytes - 1.0) / (double)blk_sys_bytes);
    if (Z > (uint32_t)max_z)
        Z = (uint32_t)max_z;
    if (Z < 16u)
        Z = 16u;

    size_t sys_bytes = (size_t)Z * blk_sys_bytes;
    size_t wire_bytes = (size_t)Z * blk_wire_bytes;
    double actual_target_tbps = ((double)sys_bytes * 8.0) / (DEADLINE_MS * 1e-3 * 1e12);
    double actual_line_tbps = ((double)wire_bytes * 8.0) / (DEADLINE_MS * 1e-3 * 1e12);
    double paced_vram_rw_gbs = (double)(wire_bytes + sys_bytes) / (DEADLINE_MS * 1e-3 * 1e9);
    double max_mds_fade_ms = DEADLINE_MS * ((double)P / (double)(K + P));

    printf(
        "\n==================================================================================================================\n");
    printf("Superframe Target: %.2f Tbps Payload (%.2f Tbps Wire Rate @ K=%u, P=%u [%.2f%% wire parity], %.2f ms Frame)\n",
           actual_target_tbps, actual_line_tbps, K, P, (100.0 * (double)P) / (double)(K + P), DEADLINE_MS);
    printf("Wire Ring Layout : Z=%u blocks x %u slots x %u B = %.2f GiB/buf (2-buffer decode budget = %.2f ms/frame)\n", Z, K + P,
           stride, (double)wire_bytes / (1024.0 * 1024.0 * 1024.0), DEADLINE_MS);
    printf("Outage Envelope  : 1.40 ms timed blackout (ideal MDS limit = %.2f ms per %.1f ms frame; ~2.03 ms w/ 2%% random loss)\n",
           max_mds_fade_ms, DEADLINE_MS);

    size_t seed_len = (sys_bytes < SEED_CHUNK_BYTES) ? sys_bytes : SEED_CHUNK_BYTES;
    uint8_t *seed_buf = (uint8_t *)malloc(seed_len);
    uint8_t *verify_buf = (uint8_t *)malloc(seed_len);
    if (!seed_buf || !verify_buf) {
        free(verify_buf);
        free(seed_buf);
        return false;
    }
    init_seed_chunk(seed_buf, seed_len);

    vxfec_vk_buf *wire_vram[2] = {NULL, NULL};
    vxfec_vk_buf *wire_ref = vxfec_vk_buf_new(ctx, wire_bytes, false);
    wire_vram[0] = vxfec_vk_buf_new(ctx, wire_bytes, false);
    wire_vram[1] = vxfec_vk_buf_new(ctx, wire_bytes, false);
    if (!wire_ref || !wire_vram[0] || !wire_vram[1]) {
        fprintf(stderr, "FAIL: unable to allocate %.2f GiB wire_vram buffers\n", (double)wire_bytes / (1024.0 * 1024.0 * 1024.0));
        vxfec_vk_buf_free(ctx, wire_vram[1]);
        vxfec_vk_buf_free(ctx, wire_vram[0]);
        vxfec_vk_buf_free(ctx, wire_ref);
        free(verify_buf);
        free(seed_buf);
        return false;
    }

    size_t wpb = ((size_t)(K + P) + 63u) / 64u;
    uint64_t *rx_bits[PIPE_ITERS] = {0};
    vxfec_vk_span *rx_spans[PIPE_ITERS] = {0};
    size_t rx_span_cnt[PIPE_ITERS] = {0};
    bool bits_ok = true;
    double loss_pct_sum = 0.0;
    channel_summary_t cs0 = {0};
    for (int f = 0; f < PIPE_ITERS; f++) {
        rx_bits[f] = (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t));
        if (!rx_bits[f]) {
            bits_ok = false;
            break;
        }
        fade_pos_t pos = (f % 3 == 0) ? FADE_INTERIOR : ((f % 3 == 1) ? FADE_PARITY_TAIL : FADE_BOUNDARY_WRAP);
        channel_summary_t csf = {0};
        build_uncapped_wire_channel(rx_bits[f], Z, K, P, wpb, 14u, 20000u, pos, (uint32_t)f, &csf);
        if (!build_erased_sys_spans(rx_bits[f], Z, K, wpb, stride, &rx_spans[f], &rx_span_cnt[f])) {
            bits_ok = false;
            break;
        }
        loss_pct_sum += csf.wire_loss_pct;
        if (f == 0)
            cs0 = csf;
    }
    if (!bits_ok) {
        for (int f = 0; f < PIPE_ITERS; f++) {
            free(rx_spans[f]);
            free(rx_bits[f]);
        }
        vxfec_vk_buf_free(ctx, wire_vram[1]);
        vxfec_vk_buf_free(ctx, wire_vram[0]);
        vxfec_vk_buf_free(ctx, wire_ref);
        free(verify_buf);
        free(seed_buf);
        return false;
    }

    printf("Uncapped Channel : 14%% Wire Blackout + 2.0%% Random Drops (%.2f%% realized wire loss across %d fresh superframes)\n",
           loss_pct_sum / (double)PIPE_ITERS, PIPE_ITERS);
    printf("Frame 0 Stats    : %.1f avg sys erasures/blk [%u..%u], min surviving=%u/%u, %u/%u unique block masks\n", cs0.avg_sys_e,
           cs0.min_sys_e, cs0.max_sys_e, cs0.min_surviving, K + P, cs0.unique_masks, Z);
    printf("RX pipeline wall time includes erasure writes when reusing buffers. Paced arrivals stay 10 ms apart.\n");
    printf("------------------------------------------------------------------------------------------------------------------\n");
    printf("  Codec    | Pipeline Stage / Contention   | CPU Comp | Tab DMA | Avg Step | Max Lat   | Max<=10 | Useful Rate\n");
    printf("  ---------+-------------------------------+----------+---------+----------+-----------+---------+------------\n");

    const int codecs[3] = {VXFEC_WIRE_RATELESS_GF2, VXFEC_WIRE_RATELESS, VXFEC_WIRE_MDS};
    const char *cnames[3] = {"RQ-GF2  ", "Rateless", "MDS     "};
    vxfec_wire_ring *rings[3] = {NULL, NULL, NULL};
    bool all_ok = true;

    for (int ci = 0; ci < 3; ci++) {
        vxfec_wire_ring *wring = vxfec_wire_ring_new(ctx, codecs[ci], Z, K, P, stride, 0u);
        rings[ci] = wring;
        if (!wring) {
            fprintf(stderr, "FAIL: vxfec_wire_ring_new (%s)\n", cnames[ci]);
            all_ok = false;
            break;
        }

        if (!fill_vram_pattern(ctx, wire_ref, sys_bytes, seed_buf, seed_len) || !vxfec_wire_encode_vram(wring, wire_ref, NULL)) {
            all_ok = false;
            break;
        }

        /* verify all 8 fresh masks independently from freshly poisoned inputs */
        for (int f = 0; f < PIPE_ITERS; f++) {
            vxfec_vk_buf *dst = wire_vram[f & 1];
            if (!vxfec_vk_buf_copy(ctx, dst, wire_ref, wire_bytes) ||
                !vxfec_vk_buf_fill_spans(ctx, dst, rx_spans[f], rx_span_cnt[f], 0xa5a5a5a5u) ||
                !vxfec_wire_recover_vram(wring, dst, rx_bits[f], NULL) ||
                !verify_full_superframe(ctx, dst, sys_bytes, seed_buf, seed_len, verify_buf)) {
                fprintf(stderr, "FAIL: full-superframe verification failed on fresh mask %d (%s)\n", f, cnames[ci]);
                all_ok = false;
                break;
            }
        }
        if (!all_ok)
            break;

        /* 1. continuous TX forward parity encode */
        uint64_t tx_max_ns = 0;
        uint64_t tx_t0 = now_ns();
        for (int it = 0; it < PIPE_ITERS; it++) {
            uint64_t f0 = now_ns();
            if (!vxfec_wire_encode_vram(wring, wire_vram[it & 1], NULL)) {
                all_ok = false;
                break;
            }
            uint64_t f_dt = now_ns() - f0;
            if (f_dt > tx_max_ns)
                tx_max_ns = f_dt;
        }
        if (!all_ok)
            break;
        uint64_t tx_wall_avg = (now_ns() - tx_t0) / (uint64_t)PIPE_ITERS;
        double tx_avg_ms = (double)tx_wall_avg / 1e6;
        double tx_max_ms = (double)tx_max_ns / 1e6;
        double tx_tbps = ((double)sys_bytes * 8.0) / ((double)tx_wall_avg * 1e3);
        printf("  %s | TX Parity (+%u%%, %s)       |  0.00 ms |  0.0 MB | %5.2f ms |  %5.2f ms |  %s   |  %5.2f Tbps\n", cnames[ci],
               (P * 100u) / K, vxfec_wire_ring_tx_is_pure_gf2(wring) ? "GF2 " : "GF8 ", tx_avg_ms, tx_max_ms,
               (tx_max_ms <= DEADLINE_MS) ? "PASS" : "MISS", tx_tbps);

        /* 2. RX 1-shot serial latency across 4 freshly poisoned masks */
        uint64_t rx_cpu_sum = 0, rx_wall_sum = 0, rx_ser_max = 0;
        size_t rx_tab_bytes = 0;
        uint32_t rx_gf2_blks = 0, rx_256_blks = 0;
        for (int it = 0; it < 4; it++) {
            vxfec_vk_buf *dst = wire_vram[it & 1];
            if (!vxfec_vk_buf_copy(ctx, dst, wire_ref, wire_bytes) ||
                !vxfec_vk_buf_fill_spans(ctx, dst, rx_spans[it], rx_span_cnt[it], 0xa5a5a5a5u)) {
                all_ok = false;
                break;
            }
            vxfec_wire_stats st = {0};
            if (!vxfec_wire_recover_vram(wring, dst, rx_bits[it], &st) ||
                !verify_full_superframe(ctx, dst, sys_bytes, seed_buf, seed_len, verify_buf)) {
                all_ok = false;
                break;
            }
            rx_cpu_sum += st.cpu_compile_ns;
            rx_wall_sum += st.total_wall_ns;
            if (st.total_wall_ns > rx_ser_max)
                rx_ser_max = st.total_wall_ns;
            rx_tab_bytes = st.tab_bytes;
            rx_gf2_blks = st.gf2_blocks;
            rx_256_blks = st.gf256_blocks;
        }
        if (!all_ok)
            break;
        const char *fmode = (rx_256_blks == 0u) ? "GF2 " : ((rx_gf2_blks > 0u) ? "HYB " : "GF8 ");
        double rx_cpu_ms = (double)(rx_cpu_sum / 4ULL) / 1e6;
        double rx_ser_ms = (double)(rx_wall_sum / 4ULL) / 1e6;
        double rx_ser_max_ms = (double)rx_ser_max / 1e6;
        double rx_ser_tbps = ((double)sys_bytes * 8.0) / ((double)(rx_wall_sum / 4ULL) * 1e3);
        printf("  %s | RX 1-Shot Serial (%s)       | %5.2f ms | %4.1f MB | %5.2f ms |  %5.2f ms |  %s   |  %5.2f Tbps\n",
               cnames[ci], fmode, rx_cpu_ms, (double)rx_tab_bytes / (1024.0 * 1024.0), rx_ser_ms, rx_ser_max_ms,
               (rx_ser_max_ms <= DEADLINE_MS) ? "PASS" : "MISS", rx_ser_tbps);

        /* 3. RX double-buffered saturation throughput */
        if (!prep_poisoned_pair(ctx, wire_vram, wire_ref, rx_spans, rx_span_cnt, wire_bytes)) {
            all_ok = false;
            break;
        }
        uint64_t pipe_cpu_sum = 0, pipe_max_lat_ns = 0;
        uint64_t slot_tkt[2] = {0, 0};
        uint64_t slot_t0[2] = {0, 0};
        uint64_t pipe_t0 = now_ns();
        for (int it = 0; it < PIPE_ITERS; it++) {
            int sl = it & 1;
            if (!poll_inflight_slots(ctx, slot_tkt, slot_t0, &pipe_max_lat_ns)) {
                all_ok = false;
                break;
            }
            if (slot_tkt[sl] > 0u) {
                if (!vxfec_vk_wait(ctx, slot_tkt[sl], DRAIN_TIMEOUT_NS)) {
                    all_ok = false;
                    break;
                }
                uint64_t lat = now_ns() - slot_t0[sl];
                if (lat > pipe_max_lat_ns)
                    pipe_max_lat_ns = lat;
                slot_tkt[sl] = 0u;
            }
            if (it >= 2) {
                if (vxfec_vk_buf_fill_spans_submit(ctx, wire_vram[sl], rx_spans[it], rx_span_cnt[it], 0xa5a5a5a5u) == 0u) {
                    all_ok = false;
                    break;
                }
            }
            slot_t0[sl] = now_ns();
            vxfec_wire_stats st = {0};
            slot_tkt[sl] = vxfec_wire_recover_submit(wring, wire_vram[sl], rx_bits[it], &st);
            if (slot_tkt[sl] == 0u) {
                all_ok = false;
                break;
            }
            pipe_cpu_sum += st.cpu_compile_ns;
            if (!poll_inflight_slots(ctx, slot_tkt, slot_t0, &pipe_max_lat_ns)) {
                all_ok = false;
                break;
            }
        }
        while (all_ok && (slot_tkt[0] > 0u || slot_tkt[1] > 0u)) {
            if (!poll_inflight_slots(ctx, slot_tkt, slot_t0, &pipe_max_lat_ns))
                all_ok = false;
        }
        uint64_t pipe_elapsed = now_ns() - pipe_t0;
        if (!all_ok || !verify_full_superframe(ctx, wire_vram[0], sys_bytes, seed_buf, seed_len, verify_buf) ||
            !verify_full_superframe(ctx, wire_vram[1], sys_bytes, seed_buf, seed_len, verify_buf)) {
            fprintf(stderr, "FAIL: RX pipelined stage failed or corrupted payload (%s)\n", cnames[ci]);
            all_ok = false;
            break;
        }

        uint64_t pipe_wall_avg = pipe_elapsed / (uint64_t)PIPE_ITERS;
        double pipe_avg_ms = (double)pipe_wall_avg / 1e6;
        double pipe_max_ms = (double)pipe_max_lat_ns / 1e6;
        double pipe_tbps = ((double)sys_bytes * 8.0) / ((double)pipe_wall_avg * 1e3);
        printf("  %s | RX Pipelined (8 Masks)        | %5.2f ms | %4.1f MB | %5.2f ms |  %5.2f ms |  %s   |  %5.2f Tbps\n",
               cnames[ci], (double)(pipe_cpu_sum / (uint64_t)PIPE_ITERS) / 1e6, (double)rx_tab_bytes / (1024.0 * 1024.0),
               pipe_avg_ms, pipe_max_ms, (pipe_max_ms <= DEADLINE_MS) ? "PASS" : "MISS", pipe_tbps);

        /* 4. paced 10.00 ms arrival-cadence SLA check */
        if (!prep_poisoned_pair(ctx, wire_vram, wire_ref, rx_spans, rx_span_cnt, wire_bytes)) {
            all_ok = false;
            break;
        }
        uint64_t sla_cpu_sum = 0;
        sla_progress_t sla = {0};
        const uint64_t cadence_t0 = now_ns();
        for (int it = 0; it < PIPE_ITERS; it++) {
            int sl = it & 1;
            uint64_t target_arr = cadence_t0 + (uint64_t)it * DEADLINE_NS;
            /* prepare the next frame as soon as its buffer becomes available */
            while (all_ok && (sla.tickets[sl] > 0u || (it >= 2 && !vxfec_vk_is_discrete(ctx) && sla.tickets[1 - sl] > 0u)))
                all_ok = poll_sla_slots(ctx, &sla);
            if (all_ok && it >= 2)
                all_ok = vxfec_vk_buf_fill_spans_submit(ctx, wire_vram[sl], rx_spans[it], rx_span_cnt[it], 0xa5a5a5a5u) != 0u;
            while (all_ok && now_ns() < target_arr)
                all_ok = poll_sla_slots(ctx, &sla);
            if (!all_ok)
                break;

            sla.arrivals[sl] = target_arr;
            vxfec_wire_stats st = {0};
            sla.tickets[sl] = vxfec_wire_recover_submit(wring, wire_vram[sl], rx_bits[it], &st);
            if (sla.tickets[sl] == 0u) {
                all_ok = false;
                break;
            }
            sla_cpu_sum += st.cpu_compile_ns;
        }
        while (all_ok && (sla.tickets[0] > 0u || sla.tickets[1] > 0u))
            all_ok = poll_sla_slots(ctx, &sla);
        uint64_t sla_sched_span = now_ns() - cadence_t0;
        if (!all_ok || !verify_full_superframe(ctx, wire_vram[0], sys_bytes, seed_buf, seed_len, verify_buf) ||
            !verify_full_superframe(ctx, wire_vram[1], sys_bytes, seed_buf, seed_len, verify_buf)) {
            fprintf(stderr, "FAIL: RX paced SLA stage failed or corrupted payload (%s)\n", cnames[ci]);
            all_ok = false;
            break;
        }

        /* include frame 0's 10.00 ms ingress interval so rate reflects 8 frames over >= 80 ms */
        uint64_t sla_total_span = sla_sched_span + DEADLINE_NS;
        double sla_avg_lat_ms = (double)(sla.lat_sum / (uint64_t)PIPE_ITERS) / 1e6;
        double sla_max_lat_ms = (double)sla.max_lat_ns / 1e6;
        double sla_deliv_tbps = ((double)sys_bytes * (double)PIPE_ITERS * 8.0) / ((double)sla_total_span * 1e3);
        bool sla_pass = (sla.misses == 0u && sla_max_lat_ms <= DEADLINE_MS);
        char sla_lbl[40];
        snprintf(sla_lbl, sizeof(sla_lbl), "RX Paced 10ms (miss=%u/%d)", sla.misses, PIPE_ITERS);
        printf("  %s | %-29s | %5.2f ms | %4.1f MB | %5.2f ms |  %5.2f ms |  %s   |  %5.2f Tbps\n", cnames[ci], sla_lbl,
               (double)(sla_cpu_sum / (uint64_t)PIPE_ITERS) / 1e6, (double)rx_tab_bytes / (1024.0 * 1024.0), sla_avg_lat_ms,
               sla_max_lat_ms, sla_pass ? "PASS" : "MISS", sla_deliv_tbps);

        /* 5. RX pipelined decode + concurrent device-local VRAM copy contention */
        if (!prep_poisoned_pair(ctx, wire_vram, wire_ref, rx_spans, rx_span_cnt, wire_bytes)) {
            all_ok = false;
            break;
        }

        vram_copy_stress_ctx vcopy;
        pthread_t vcopy_thr;
        if (vram_copy_stress_start(&vcopy, paced_vram_rw_gbs, &vcopy_thr)) {
            uint64_t spin_t0 = now_ns();
            while ((now_ns() - spin_t0) < 12000000ULL) {
            }

            uint64_t cv_cpu = 0, cv_max_lat_ns = 0;
            uint64_t cv_tkt[2] = {0, 0};
            uint64_t cv_sub_t0[2] = {0, 0};
            uint64_t cv_t0 = now_ns();
            unsigned long long rw_b0 = atomic_load_explicit(&vcopy.rw_bytes, memory_order_relaxed);
            for (int it = 0; it < PIPE_ITERS; it++) {
                int sl = it & 1;
                if (!poll_inflight_slots(ctx, cv_tkt, cv_sub_t0, &cv_max_lat_ns)) {
                    all_ok = false;
                    break;
                }
                if (cv_tkt[sl] > 0u) {
                    if (!vxfec_vk_wait(ctx, cv_tkt[sl], DRAIN_TIMEOUT_NS)) {
                        all_ok = false;
                        break;
                    }
                    uint64_t lat = now_ns() - cv_sub_t0[sl];
                    if (lat > cv_max_lat_ns)
                        cv_max_lat_ns = lat;
                    cv_tkt[sl] = 0u;
                }
                if (it >= 2) {
                    if (vxfec_vk_buf_fill_spans_submit(ctx, wire_vram[sl], rx_spans[it], rx_span_cnt[it], 0xa5a5a5a5u) == 0u) {
                        all_ok = false;
                        break;
                    }
                }
                cv_sub_t0[sl] = now_ns();
                vxfec_wire_stats st = {0};
                cv_tkt[sl] = vxfec_wire_recover_submit(wring, wire_vram[sl], rx_bits[it], &st);
                if (cv_tkt[sl] == 0u) {
                    all_ok = false;
                    break;
                }
                cv_cpu += st.cpu_compile_ns;
                if (!poll_inflight_slots(ctx, cv_tkt, cv_sub_t0, &cv_max_lat_ns)) {
                    all_ok = false;
                    break;
                }
            }
            while (all_ok && (cv_tkt[0] > 0u || cv_tkt[1] > 0u)) {
                if (!poll_inflight_slots(ctx, cv_tkt, cv_sub_t0, &cv_max_lat_ns))
                    all_ok = false;
            }

            uint64_t cv_elapsed = now_ns() - cv_t0;
            unsigned long long rw_b1 = atomic_load_explicit(&vcopy.rw_bytes, memory_order_relaxed);

            vram_copy_stress_stop(&vcopy, vcopy_thr);

            if (!all_ok || !verify_full_superframe(ctx, wire_vram[0], sys_bytes, seed_buf, seed_len, verify_buf) ||
                !verify_full_superframe(ctx, wire_vram[1], sys_bytes, seed_buf, seed_len, verify_buf)) {
                fprintf(stderr, "FAIL: contended pipelined full-superframe verification failed (%s)\n", cnames[ci]);
                all_ok = false;
                break;
            }

            double meas_rw_gbs = (double)(rw_b1 - rw_b0) / (double)cv_elapsed;
            uint64_t cv_wall_avg = cv_elapsed / (uint64_t)PIPE_ITERS;
            double cv_avg_ms = (double)cv_wall_avg / 1e6;
            double cv_max_ms = (double)cv_max_lat_ns / 1e6;
            double cv_tbps = ((double)sys_bytes * 8.0) / ((double)cv_wall_avg * 1e3);
            char lbl[32];
            snprintf(lbl, sizeof(lbl), "RX Pipe + %3.0f GB/s VRAM Copy", meas_rw_gbs);
            printf("  %s | %-29s | %5.2f ms | %4.1f MB | %5.2f ms |  %5.2f ms |  %s   |  %5.2f Tbps\n", cnames[ci], lbl,
                   (double)(cv_cpu / (uint64_t)PIPE_ITERS) / 1e6, (double)rx_tab_bytes / (1024.0 * 1024.0), cv_avg_ms, cv_max_ms,
                   (cv_max_ms <= DEADLINE_MS) ? "PASS" : "MISS", cv_tbps);
        }

        if (ci + 1 < 3)
            printf(
                "  ---------+-------------------------------+----------+---------+----------+-----------+---------+------------\n");
    }

    if (all_ok && run_sweep && rings[0] && rings[1] && rings[2]) {
        if (!run_fade_boundary_sweep(ctx, rings[0], rings[1], rings[2], wire_ref, wire_vram[0], Z, K, P, wpb, stride, sys_bytes,
                                     wire_bytes, seed_buf, seed_len, verify_buf)) {
            all_ok = false;
        }
    }

    vxfec_wire_ring_free(rings[2]);
    vxfec_wire_ring_free(rings[1]);
    vxfec_wire_ring_free(rings[0]);
    for (int f = 0; f < PIPE_ITERS; f++) {
        free(rx_spans[f]);
        free(rx_bits[f]);
    }
    vxfec_vk_buf_free(ctx, wire_vram[1]);
    vxfec_vk_buf_free(ctx, wire_vram[0]);
    vxfec_vk_buf_free(ctx, wire_ref);
    free(verify_buf);
    free(seed_buf);
    return all_ok;
}

int main(void)
{
    int status = VXFEC_VK_ERROR;
    vxfec_vk_ctx *ctx = vxfec_vk_ctx_new_status(&status);
    if (!ctx) {
        if (status == VXFEC_VK_UNAVAILABLE) {
            fprintf(stderr, "SKIP: no Vulkan 1.2 device available\n");
            return 0;
        }
        fprintf(stderr, "FAIL: Vulkan context init\n");
        return 1;
    }

    bool discrete = vxfec_vk_is_discrete(ctx);
    double t_lo = discrete ? 1.00 : 0.05;
    double t_hi = discrete ? 1.50 : 0.10;
    const char *env_full = getenv("VXFEC_BENCH_FULL");
    if (env_full && atoi(env_full) != 0) {
        t_lo = 1.00;
        t_hi = 1.50;
    }

    printf("==================================================================================================================\n");
    printf("vxfec Wire-Strided Superframe Pipeline Benchmark (8 Poisoned Masks / Paced 10ms SLA / Verified Fade Sweep)\n");
    printf("Device : %s (%s, dedicated_ace=%d, qfam=%u, max_ssbo=%.2f GiB)\n", vxfec_vk_device_name(ctx),
           discrete ? "Discrete GPU" : "Integrated/Software", vxfec_vk_is_dedicated_compute(ctx) ? 1 : 0,
           vxfec_vk_queue_family(ctx), (double)vxfec_vk_max_storage_buffer_bytes(ctx) / (1024.0 * 1024.0 * 1024.0));

    bool ok1 = run_pipeline_suite(ctx, t_lo, 200u, 56u, 4096u, true);
    bool ok2 = run_pipeline_suite(ctx, t_hi, 200u, 56u, 4096u, false);

    vxfec_vk_ctx_free(ctx);
    return (ok1 && ok2) ? 0 : 1;
}
