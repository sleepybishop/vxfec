#include "vxfec_interleave.h"
#include "nanorq_core.h"
#include "nanorq_ops.h"
#include "util.h"
#include "vk_internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define WIRE_RQ_FAST_OVERHEAD 16u
#define WIRE_RQ_MAX_OVERHEAD 32u

typedef struct {
    vxfec_wire_ring *owner;
    uint32_t tid;
    pthread_t thread;
    bool started;

    /* pre-allocated per-worker scratch (zero heap allocations on hot path) */
    uint8_t *rq_prep;
    uint8_t *rq_retry_prep;
    uint8_t *rq_prep_tmpl16;
    uint8_t *rq_prep_tmpl32;
    size_t rq_prep_cap;
    size_t rq_tmpl16_bytes;
    size_t rq_tmpl32_bytes;
    nanorq_core rq_tmpl16_dec;
    nanorq_core rq_tmpl32_dec;
    bool rq_tmpl16_valid;
    bool rq_tmpl32_valid;
    uint8_t *rq_work;
    size_t rq_work_cap;
    void *rq_sched_mem;
    void *rq_retry_sched;
    size_t rq_sched_cap;

    uint64_t *y_gf2;
    uint8_t *y_256;
    uint32_t *miss_esis;
    uint32_t *rep_ps;
    uint32_t *active_rows;
    uint16_t *in_slots;
    uint16_t *out_slots;
    uint32_t *hdr_scratch; /* uvec4 words */
    uint32_t *cmd_scratch; /* uvec4 words */
} wire_worker_ctx;

struct vxfec_wire_ring {
    vxfec_vk_ctx *ctx;
    int codec;
    uint32_t Z;
    uint32_t K;
    uint32_t P;
    uint32_t stride;
    size_t wire_bytes;
    size_t words_per_blk;
    bool enable_tx;
    bool enable_rx;

    /* precompiled uniform TX forward-parity table */
    vxfec_vk_buf *tx_tab_buf;
    size_t tx_tab_bytes;
    uint32_t tx_tiles;
    uint32_t tx_tile_e;
    bool tx_pure_gf2;

    /* persistent double-buffered RX heterogeneous tables (CPU || GPU overlap) */
    uint32_t rx_max_overhead;
    uint32_t rx_gf2_tile_e;
    uint32_t rx_gf2_hdr_words;
    uint32_t rx_gf2_hdr_stride;
    size_t rx_gf2_max_uvec4;
    vxfec_vk_buf *rx_tab_gf2_buf[2];

    uint32_t rx_256_tile_e;
    uint32_t rx_256_hdr_words;
    uint32_t rx_256_hdr_stride;
    bool rx_256_tile_forced;
    size_t rx_256_max_uvec4;
    vxfec_vk_buf *rx_tab_256_buf[2];
    uint64_t rx_slot_tickets[2];
    uint32_t rx_slot_cursor;

    /* pre-allocated intra-superframe bitmap deduplication tables */
    uint32_t uniq_hash_cap;
    uint32_t *uniq_hash;
    uint32_t *uniq_rep_z;
    uint32_t *uniq_head;
    uint32_t *uniq_count;
    uint32_t *blk_next;
    uint64_t *uniq_bits;
    uint32_t job_num_unique;

    /* worker pool */
    uint32_t num_workers;
    wire_worker_ctx *workers;
    pthread_mutex_t mu;
    pthread_cond_t cv_start;
    pthread_cond_t cv_done;
    bool sync_ready;
    bool shutdown;
    uint64_t job_epoch;
    uint32_t active_bg_workers;

    /* active RX job parameters */
    uint32_t job_256_tile_e;
    uint32_t job_256_hdr_words;
    uint32_t job_256_hdr_stride;
    uint32_t *job_gf2_stage;
    uint32_t *job_256_stage;
    atomic_uint job_next_u;
    atomic_uint job_gf2_blocks;
    atomic_uint job_gf2_cmd_cursor;
    atomic_uint job_gf2_active_tiles;
    atomic_uint job_256_blocks;
    atomic_uint job_256_cmd_cursor;
    atomic_uint job_256_active_tiles;
    atomic_uint job_blocks_rep;
    atomic_uint job_blocks_failed;
    atomic_uint job_syms_rep;
};

static uint64_t wire_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint32_t choose_wire_256_tile_e(uint32_t num_out, bool *out_forced)
{
    if (out_forced)
        *out_forced = false;
    const char *env = getenv("VXFEC_WIRE_TILE_E");
    if (env && env[0] != '\0') {
        if (strcmp(env, "8") == 0) {
            if (out_forced)
                *out_forced = true;
            return 8u;
        }
        if (strcmp(env, "16") == 0) {
            if (out_forced)
                *out_forced = true;
            return 16u;
        }
    }
    return (num_out <= 8u) ? 8u : 16u;
}

static uint32_t wire_256_hdr_words(uint32_t tile_e)
{
    return (tile_e == 16u) ? 3u : 2u;
}

static size_t wire_256_words_per_cmd(uint32_t tile_e)
{
    return (tile_e == 16u) ? 2u : 1u;
}

static bool init_rq_core(bool gf2_h0_mode, uint32_t K, uint32_t overhead, nanorq_core *rq, bool *out_gf2_mode)
{
    if (gf2_h0_mode) {
#ifdef NANORQ_HAS_GF2_MODE
        if (out_gf2_mode)
            *out_gf2_mode = true;
        return nanorq_core_encoder_new_gf2(K, overhead, rq);
#else
        (void)K;
        (void)overhead;
        (void)rq;
        if (out_gf2_mode)
            *out_gf2_mode = false;
        return false;
#endif
    }
    if (out_gf2_mode)
        *out_gf2_mode = false;
    return nanorq_core_encoder_new(K, overhead, rq);
}

static void pack_hdr_out_slots(uint32_t *hdr, uint32_t hdr_words, const uint16_t *out_slots, uint32_t tile_len, uint32_t block_z)
{
    memset(hdr, 0, (size_t)hdr_words * 4u * sizeof(uint32_t));
    hdr[2] = tile_len;
    hdr[3] = block_z;
    for (uint32_t i = 0; i < tile_len; i++) {
        uint32_t w_idx = 4u + (i >> 1u);
        uint32_t shift = (i & 1u) * 16u;
        hdr[w_idx] |= ((uint32_t)out_slots[i] & 0xffffu) << shift;
    }
}

static void emit_gf2_tiles_local(const uint64_t *Y, size_t wpr, const uint32_t *active_rows, const uint16_t *in_slots,
                                 uint32_t n_inputs, const uint16_t *out_slots, uint32_t num_out, uint32_t block_z, uint32_t tile_e,
                                 uint32_t hdr_words, uint32_t hdr_stride, uint32_t *hdr_dst, uint32_t *cmd_scratch,
                                 uint32_t *out_n_tiles, uint32_t *out_cmd_uvec4, uint32_t tile_spans[][2])
{
    uint32_t n_tiles = (num_out + tile_e - 1u) / tile_e;
    uint64_t tile_mask = (tile_e == 16u) ? 0xffffULL : 0xffffffffULL;
    uint32_t cmd_cnt = 0;

    for (uint32_t t = 0; t < hdr_stride; t++) {
        uint32_t *hdr = hdr_dst + (size_t)t * hdr_words * 4u;
        if (t >= n_tiles) {
            memset(hdr, 0, (size_t)hdr_words * 4u * sizeof(uint32_t));
            tile_spans[t][0] = 0;
            tile_spans[t][1] = 0;
            continue;
        }
        uint32_t out_base = t * tile_e;
        uint32_t tile_len = (num_out - out_base > tile_e) ? tile_e : (num_out - out_base);
        pack_hdr_out_slots(hdr, hdr_words, out_slots + out_base, tile_len, block_z);

        size_t ew = (size_t)out_base >> 6u;
        uint32_t shift = out_base & 63u;
        uint32_t beg = cmd_cnt;
        bool have_half = false;

        for (uint32_t i = 0; i < n_inputs; i++) {
            uint32_t r = active_rows[i];
            uint32_t mask = (uint32_t)((Y[(size_t)r * wpr + ew] >> shift) & tile_mask);
            if (mask == 0u)
                continue;
            if (!have_half) {
                uint32_t *c = cmd_scratch + (size_t)cmd_cnt * 4u;
                c[0] = in_slots[i];
                c[1] = mask;
                c[2] = 0u;
                c[3] = 0u;
                have_half = true;
            } else {
                uint32_t *c = cmd_scratch + (size_t)cmd_cnt * 4u;
                c[2] = in_slots[i];
                c[3] = mask;
                cmd_cnt++;
                have_half = false;
            }
        }
        if (have_half)
            cmd_cnt++;

        tile_spans[t][0] = beg;
        tile_spans[t][1] = cmd_cnt;
    }

    *out_n_tiles = n_tiles;
    *out_cmd_uvec4 = cmd_cnt;
}

static inline void pack_nib_octet(const uint8_t *yc, uint32_t rem, uint32_t *out_lo, uint32_t *out_hi)
{
    uint32_t lo = 0u, hi = 0u;
    uint32_t lim = (rem > 8u) ? 8u : rem;
    for (uint32_t b = 0; b < lim; b++) {
        uint8_t c = yc[b];
        if (c != 0u) {
            uint32_t sh = 4u * b;
            lo |= ((uint32_t)(c & 0x0fu)) << sh;
            hi |= ((uint32_t)(c >> 4u)) << sh;
        }
    }
    *out_lo = lo;
    *out_hi = hi;
}

static void emit_gf256_tiles_local(const uint8_t *Y, size_t m_stride, const uint32_t *active_rows, const uint16_t *in_slots,
                                   uint32_t n_inputs, const uint16_t *out_slots, uint32_t num_out, uint32_t block_z,
                                   uint32_t tile_e, uint32_t hdr_words, uint32_t hdr_stride, uint32_t *hdr_dst,
                                   uint32_t *cmd_scratch, uint32_t *out_n_tiles, uint32_t *out_cmd_uvec4, uint32_t tile_spans[][2])
{
    uint32_t n_tiles = (num_out + tile_e - 1u) / tile_e;
    uint32_t cmd_cnt = 0;

    for (uint32_t t = 0; t < hdr_stride; t++) {
        uint32_t *hdr = hdr_dst + (size_t)t * hdr_words * 4u;
        if (t >= n_tiles) {
            memset(hdr, 0, (size_t)hdr_words * 4u * sizeof(uint32_t));
            tile_spans[t][0] = 0;
            tile_spans[t][1] = 0;
            continue;
        }
        uint32_t out_base = t * tile_e;
        uint32_t tile_len = (num_out - out_base > tile_e) ? tile_e : (num_out - out_base);
        pack_hdr_out_slots(hdr, hdr_words, out_slots + out_base, tile_len, block_z);

        uint32_t beg = cmd_cnt;
        for (uint32_t i = 0; i < n_inputs; i++) {
            uint32_t r = active_rows[i];
            const uint8_t *yc = Y + (size_t)r * m_stride + out_base;
            uint32_t lo0 = 0u, hi0 = 0u;
            pack_nib_octet(yc, tile_len, &lo0, &hi0);

            if (tile_e == 16u) {
                uint32_t lo1 = 0u, hi1 = 0u;
                if (tile_len > 8u)
                    pack_nib_octet(yc + 8u, tile_len - 8u, &lo1, &hi1);
                if ((lo0 | lo1 | hi0 | hi1) != 0u) {
                    uint32_t *w = cmd_scratch + (size_t)cmd_cnt * 4u;
                    w[0] = in_slots[i];
                    w[1] = 0u;
                    w[2] = lo0;
                    w[3] = lo1;
                    w[4] = hi0;
                    w[5] = hi1;
                    w[6] = 0u;
                    w[7] = 0u;
                    cmd_cnt += 2u;
                }
            } else if ((lo0 | hi0) != 0u) {
                uint32_t *w = cmd_scratch + (size_t)cmd_cnt * 4u;
                w[0] = in_slots[i];
                w[1] = lo0;
                w[2] = hi0;
                w[3] = 0u;
                cmd_cnt += 1u;
            }
        }
        tile_spans[t][0] = beg;
        tile_spans[t][1] = cmd_cnt;
    }

    *out_n_tiles = n_tiles;
    *out_cmd_uvec4 = cmd_cnt;
}

static bool upload_tx_wire_tab(vxfec_wire_ring *wring, uint32_t tile_e, uint32_t hdr_words, uint32_t n_tiles, bool pure_gf2,
                               uint32_t *tab, const uint32_t *cmd_tmp, uint32_t cmd_cnt, uint32_t spans[][2])
{
    uint32_t hdr_uvec4 = n_tiles * hdr_words;
    memcpy(tab + (size_t)hdr_uvec4 * 4u, cmd_tmp, (size_t)cmd_cnt * 16u);
    for (uint32_t t = 0; t < n_tiles; t++) {
        uint32_t *hdr = tab + (size_t)t * hdr_words * 4u;
        hdr[0] = hdr_uvec4 + spans[t][0];
        hdr[1] = hdr_uvec4 + spans[t][1];
    }
    size_t bytes = (size_t)(hdr_uvec4 + cmd_cnt) * 16u;
    if (bytes > wring->ctx->max_ssbo_range)
        return false;
    wring->tx_tab_buf = vxfec_vk__upload_const_buf(wring->ctx, tab, bytes);
    if (!wring->tx_tab_buf)
        return false;
    wring->tx_tab_bytes = bytes;
    wring->tx_tiles = n_tiles;
    wring->tx_tile_e = tile_e;
    wring->tx_pure_gf2 = pure_gf2;
    return true;
}

static bool build_tx_table(vxfec_wire_ring *wring)
{
    uint32_t K = wring->K, P = wring->P;
    uint32_t *active_rows = (uint32_t *)malloc((size_t)K * sizeof(uint32_t));
    uint16_t *in_slots = (uint16_t *)malloc((size_t)K * sizeof(uint16_t));
    uint16_t *out_slots = (uint16_t *)malloc((size_t)P * sizeof(uint16_t));
    if (!active_rows || !in_slots || !out_slots) {
        free(active_rows);
        free(in_slots);
        free(out_slots);
        return false;
    }
    for (uint32_t p = 0; p < P; p++)
        out_slots[p] = (uint16_t)(K + p);

    bool ok = false;
    if (wring->codec == VXFEC_WIRE_MDS) {
        for (uint32_t k = 0; k < K; k++) {
            active_rows[k] = k;
            in_slots[k] = (uint16_t)k;
        }
        size_t m_stride = ((size_t)P + 31u) & ~(size_t)31u;
        uint8_t *Y = (uint8_t *)calloc((size_t)K * m_stride, 1u);
        if (Y && vxfec_rs8__encode_matrix(K, P, NULL, P, Y, m_stride)) {
            uint32_t tile_e = choose_wire_256_tile_e(P, NULL);
            uint32_t hdr_words = wire_256_hdr_words(tile_e);
            uint32_t n_tiles = (P + tile_e - 1u) / tile_e;
            size_t wpc = wire_256_words_per_cmd(tile_e);
            size_t max_uvec4 = (size_t)n_tiles * (hdr_words + wpc * K);
            uint32_t *tab = (uint32_t *)calloc(max_uvec4 * 4u, sizeof(uint32_t));
            uint32_t *cmd_tmp = (uint32_t *)calloc(max_uvec4 * 4u, sizeof(uint32_t));
            uint32_t (*spans)[2] = (uint32_t (*)[2])calloc(n_tiles, sizeof(*spans));
            if (tab && cmd_tmp && spans) {
                uint32_t out_nt = 0, cmd_cnt = 0;
                emit_gf256_tiles_local(Y, m_stride, active_rows, in_slots, K, out_slots, P, 0u, tile_e, hdr_words, n_tiles, tab,
                                       cmd_tmp, &out_nt, &cmd_cnt, spans);
                ok = upload_tx_wire_tab(wring, tile_e, hdr_words, n_tiles, false, tab, cmd_tmp, cmd_cnt, spans);
            }
            free(spans);
            free(cmd_tmp);
            free(tab);
        }
        free(Y);
    } else {
        nanorq_core enc = {0};
        bool gf2_mode = false;
        if (init_rq_core(wring->codec == VXFEC_WIRE_RATELESS_GF2, K, 0u, &enc, &gf2_mode)) {
            size_t plen = nanorq_core_calculate_prepare_memory(&enc);
            size_t wlen = nanorq_core_calculate_work_memory(&enc);
            size_t slen = ops_estimate_schedule_bytes(K);
            uint8_t *prep = (uint8_t *)malloc(plen);
            uint8_t *work = (uint8_t *)malloc(wlen);
            void *smem = malloc(slen);
            uint32_t *esis = (uint32_t *)malloc((size_t)P * sizeof(uint32_t));
            schedule S = {0};
            if (prep && work && smem && esis && schedule_init(&S, smem, slen) && nanorq_core_prepare(&enc, prep, plen)) {
                nanorq_core_set_op_callback(&enc, &S, ops_push);
                if (nanorq_core_precalculate(&enc, work, wlen)) {
                    uint32_t rows = nanorq_core_get_pc_rows(&enc);
                    uint32_t sh = nanorq_core_get_pc_genc_offset(&enc);
                    for (uint32_t k = 0; k < K; k++) {
                        active_rows[k] = sh + k;
                        in_slots[k] = (uint16_t)k;
                    }
                    for (uint32_t p = 0; p < P; p++)
                        esis[p] = K + p;

                    bool pure_gf2 = gf2_mode || vxfec_rq__sched_is_pure_gf2(&S);
                    if (pure_gf2) {
                        uint32_t tile_e = (P <= 16u) ? 16u : 32u;
                        uint32_t hdr_words = (tile_e == 32u) ? 5u : 3u;
                        uint32_t n_tiles = (P + tile_e - 1u) / tile_e;
                        size_t wpr = ((size_t)P + 63u) / 64u;
                        uint64_t *Y = (uint64_t *)calloc((size_t)rows * wpr, sizeof(uint64_t));
                        size_t max_uvec4 = (size_t)n_tiles * (hdr_words + (K + 1u) / 2u);
                        uint32_t *tab = (uint32_t *)calloc(max_uvec4 * 4u, sizeof(uint32_t));
                        uint32_t *cmd_tmp = (uint32_t *)calloc(max_uvec4 * 4u, sizeof(uint32_t));
                        uint32_t (*spans)[2] = (uint32_t (*)[2])calloc(n_tiles, sizeof(*spans));
                        if (Y && tab && cmd_tmp && spans) {
                            vxfec_rq__adjoint_gf2(&enc, S.ops.a, (uint32_t)S.ops.n, S.cp[0], S.cp[1], rows, esis, P, Y, wpr);
                            uint32_t out_nt = 0, cmd_cnt = 0;
                            emit_gf2_tiles_local(Y, wpr, active_rows, in_slots, K, out_slots, P, 0u, tile_e, hdr_words, n_tiles,
                                                 tab, cmd_tmp, &out_nt, &cmd_cnt, spans);
                            ok = upload_tx_wire_tab(wring, tile_e, hdr_words, n_tiles, true, tab, cmd_tmp, cmd_cnt, spans);
                        }
                        free(spans);
                        free(cmd_tmp);
                        free(tab);
                        free(Y);
                    } else {
                        uint32_t tile_e = choose_wire_256_tile_e(P, NULL);
                        uint32_t hdr_words = wire_256_hdr_words(tile_e);
                        uint32_t n_tiles = (P + tile_e - 1u) / tile_e;
                        size_t align = get_align_size();
                        size_t m_stride = ((size_t)P + align - 1u) & ~(align - 1u);
                        uint8_t *Y = (uint8_t *)calloc((size_t)rows * m_stride, 1u);
                        size_t wpc = wire_256_words_per_cmd(tile_e);
                        size_t max_uvec4 = (size_t)n_tiles * (hdr_words + wpc * K);
                        uint32_t *tab = (uint32_t *)calloc(max_uvec4 * 4u, sizeof(uint32_t));
                        uint32_t *cmd_tmp = (uint32_t *)calloc(max_uvec4 * 4u, sizeof(uint32_t));
                        uint32_t (*spans)[2] = (uint32_t (*)[2])calloc(n_tiles, sizeof(*spans));
                        if (Y && tab && cmd_tmp && spans) {
                            vxfec_rq__adjoint_gf256(&enc, S.ops.a, (uint32_t)S.ops.n, S.cp[0], S.cp[1], rows, esis, P, Y, m_stride);
                            uint32_t out_nt = 0, cmd_cnt = 0;
                            emit_gf256_tiles_local(Y, m_stride, active_rows, in_slots, K, out_slots, P, 0u, tile_e, hdr_words,
                                                   n_tiles, tab, cmd_tmp, &out_nt, &cmd_cnt, spans);
                            ok = upload_tx_wire_tab(wring, tile_e, hdr_words, n_tiles, false, tab, cmd_tmp, cmd_cnt, spans);
                        }
                        free(spans);
                        free(cmd_tmp);
                        free(tab);
                        free(Y);
                    }
                }
            }
            free(esis);
            free(smem);
            free(work);
            free(prep);
        }
    }

    free(out_slots);
    free(in_slots);
    free(active_rows);
    return ok;
}

static void publish_unique_mask_to_stage(const vxfec_wire_ring *wring, uint32_t u, uint32_t *stage_words, atomic_uint *blk_counter,
                                         atomic_uint *cmd_cursor, uint32_t hdr_words, uint32_t hdr_stride, uint32_t n_tiles,
                                         uint32_t *hdr_local, const uint32_t *cmd_local, uint32_t cmd_cnt, uint32_t spans[][2])
{
    uint32_t chain_cnt = wring->uniq_count[u];
    uint32_t disp_base = atomic_fetch_add_explicit(blk_counter, chain_cnt, memory_order_relaxed);
    uint32_t base_uvec4 = 0u;
    if (cmd_cnt > 0u) {
        base_uvec4 = atomic_fetch_add_explicit(cmd_cursor, cmd_cnt, memory_order_relaxed);
        memcpy(stage_words + (size_t)base_uvec4 * 4u, cmd_local, (size_t)cmd_cnt * 16u);
    }
    for (uint32_t t = 0; t < n_tiles; t++) {
        uint32_t *hdr = hdr_local + (size_t)t * hdr_words * 4u;
        hdr[0] = base_uvec4 + spans[t][0];
        hdr[1] = base_uvec4 + spans[t][1];
    }

    size_t blk_hdr_words = (size_t)hdr_stride * hdr_words * 4u;
    uint32_t idx = 0;
    for (uint32_t z = wring->uniq_head[u]; z != UINT32_MAX; z = wring->blk_next[z]) {
        for (uint32_t t = 0; t < n_tiles; t++)
            hdr_local[(size_t)t * hdr_words * 4u + 3u] = z;
        memcpy(stage_words + (size_t)(disp_base + idx) * blk_hdr_words, hdr_local, blk_hdr_words * sizeof(uint32_t));
        idx++;
    }
}

static bool compile_rx_unique_mds(vxfec_wire_ring *wring, wire_worker_ctx *w, uint32_t u, const uint64_t *blk_bits,
                                  uint32_t *out_256_tiles, uint32_t *out_missing)
{
    uint32_t K = wring->K, P = wring->P;
    uint8_t erased_k[256], parity_m[256];
    uint8_t is_erased[256] = {0};
    uint8_t era_pos[256] = {0};

    uint32_t missing = 0;
    for (uint32_t k = 0; k < K; k++) {
        if (!((blk_bits[k >> 6u] >> (k & 63u)) & 1ULL)) {
            if (missing >= P)
                return false;
            erased_k[missing] = (uint8_t)k;
            w->out_slots[missing] = (uint16_t)k;
            is_erased[k] = 1u;
            era_pos[k] = (uint8_t)missing;
            missing++;
        }
    }

    uint32_t found_p = 0;
    for (uint32_t p = 0; p < P && found_p < missing; p++) {
        uint32_t esi = K + p;
        if ((blk_bits[esi >> 6u] >> (esi & 63u)) & 1ULL)
            parity_m[found_p++] = (uint8_t)p;
    }
    if (found_p < missing)
        return false;

    for (uint32_t k = 0; k < K; k++) {
        w->active_rows[k] = k;
        w->in_slots[k] = is_erased[k] ? (uint16_t)(K + parity_m[era_pos[k]]) : (uint16_t)k;
    }

    size_t m_stride = ((size_t)missing + 31u) & ~(size_t)31u;
    if (!vxfec_rs8__recovery_matrix(K, P, erased_k, parity_m, missing, w->y_256, m_stride))
        return false;

    uint32_t spans[32][2];
    uint32_t n_tiles = 0, cmd_cnt = 0;
    uint32_t tile_e = wring->job_256_tile_e;
    uint32_t hdr_words = wring->job_256_hdr_words;
    uint32_t hdr_stride = wring->job_256_hdr_stride;
    emit_gf256_tiles_local(w->y_256, m_stride, w->active_rows, w->in_slots, K, w->out_slots, missing, wring->uniq_rep_z[u], tile_e,
                           hdr_words, hdr_stride, w->hdr_scratch, w->cmd_scratch, &n_tiles, &cmd_cnt, spans);
    publish_unique_mask_to_stage(wring, u, wring->job_256_stage, &wring->job_256_blocks, &wring->job_256_cmd_cursor, hdr_words,
                                 hdr_stride, n_tiles, w->hdr_scratch, w->cmd_scratch, cmd_cnt, spans);

    *out_256_tiles = n_tiles;
    *out_missing = missing;
    return true;
}

static bool precalc_rq_schedule_bufs(nanorq_core *dec, uint32_t overhead, void *sched_buf, size_t sched_cap, uint8_t *work_buf,
                                     size_t work_cap, schedule *S_dec, void **out_heap_sched)
{
    *out_heap_sched = NULL;
    if (!schedule_init(S_dec, sched_buf, sched_cap))
        return false;

    size_t wlen = nanorq_core_calculate_prepared_work_memory(dec);
    if (wlen == 0)
        return false;
    uint8_t *dyn_work = (wlen > work_cap) ? (uint8_t *)malloc(wlen) : NULL;
    uint8_t *wptr = dyn_work ? dyn_work : work_buf;
    if (!wptr)
        return false;

    nanorq_core_set_op_callback(dec, S_dec, ops_push);
    bool ok = nanorq_core_precalculate(dec, wptr, wlen);
    if (ok && !S_dec->overflowed && S_dec->cpidx == 2) {
        free(dyn_work);
        return true;
    }
    if (!S_dec->overflowed) {
        free(dyn_work);
        return false;
    }

    /* retry with dense-bound schedule buffer if inactivation stalled */
    if (!nanorq_core_patch_matrix(dec)) {
        free(dyn_work);
        return false;
    }
    size_t wlen2 = nanorq_core_calculate_prepared_work_memory(dec);
    if (wlen2 == 0) {
        free(dyn_work);
        return false;
    }
    if (wlen2 > (dyn_work ? wlen : work_cap)) {
        uint8_t *nw = (uint8_t *)realloc(dyn_work, wlen2);
        if (!nw) {
            free(dyn_work);
            return false;
        }
        dyn_work = nw;
        wptr = dyn_work;
    }

    size_t dense_ops = (size_t)dec->P.L * ((size_t)dec->P.L + overhead) + 2048u;
    if (dense_ops > SIZE_MAX / sizeof(sched_op)) {
        free(dyn_work);
        return false;
    }
    size_t heap_bytes = dense_ops * sizeof(sched_op);
    void *heap_mem = malloc(heap_bytes);
    if (!heap_mem) {
        free(dyn_work);
        return false;
    }
    if (!schedule_init(S_dec, heap_mem, heap_bytes)) {
        free(heap_mem);
        free(dyn_work);
        return false;
    }
    nanorq_core_set_op_callback(dec, S_dec, ops_push);
    if (!nanorq_core_precalculate(dec, wptr, wlen2) || S_dec->overflowed || S_dec->cpidx != 2) {
        free(heap_mem);
        free(dyn_work);
        return false;
    }
    free(dyn_work);
    *out_heap_sched = heap_mem;
    return true;
}

static bool solve_rq_block_ex(wire_worker_ctx *w, uint32_t K, uint32_t missing, uint32_t avail_rep, uint32_t rep_offset,
                              uint32_t overhead_cap, bool allow_tmpl, uint8_t *prep_buf, size_t prep_cap, void *sched_buf,
                              size_t sched_cap, uint8_t *work_buf, size_t work_cap, nanorq_core *dec, schedule *S_dec,
                              uint32_t *out_overhead, void **out_heap_sched)
{
    *out_heap_sched = NULL;
    if (avail_rep < rep_offset + missing)
        return false;

    uint32_t overhead = avail_rep - rep_offset - missing;
    if (overhead > overhead_cap)
        overhead = overhead_cap;
    if (overhead > w->owner->rx_max_overhead)
        overhead = w->owner->rx_max_overhead;

    if (allow_tmpl && prep_buf == w->rq_prep && w->rq_tmpl16_valid && overhead == WIRE_RQ_FAST_OVERHEAD) {
        *dec = w->rq_tmpl16_dec;
        memcpy(w->rq_prep, w->rq_prep_tmpl16, w->rq_tmpl16_bytes);
    } else if (allow_tmpl && prep_buf == w->rq_prep && w->rq_tmpl32_valid && overhead == WIRE_RQ_MAX_OVERHEAD) {
        *dec = w->rq_tmpl32_dec;
        memcpy(w->rq_prep, w->rq_prep_tmpl32, w->rq_tmpl32_bytes);
    } else {
        bool gf2_mode = false;
        if (!init_rq_core(w->owner->codec == VXFEC_WIRE_RATELESS_GF2, K, overhead, dec, &gf2_mode))
            return false;

        size_t plen = nanorq_core_calculate_prepare_memory(dec);
        if (plen == 0 || plen > prep_cap)
            return false;
        if (!nanorq_core_generate(dec, prep_buf, plen))
            return false;
    }

    const uint32_t *reps = w->rep_ps + rep_offset;
    for (uint32_t i = 0; i < missing; i++)
        nanorq_core_replace_symbol(dec, w->miss_esis[i], K + reps[i]);
    for (uint32_t i = 0; i < overhead; i++)
        nanorq_core_replace_symbol(dec, dec->P.Kprime + i, K + reps[missing + i]);

    if (!nanorq_core_patch_matrix(dec))
        return false;
    if (!precalc_rq_schedule_bufs(dec, overhead, sched_buf, sched_cap, work_buf, work_cap, S_dec, out_heap_sched))
        return false;

    *out_overhead = overhead;
    return true;
}

static bool solve_rq_block(wire_worker_ctx *w, uint32_t K, uint32_t missing, uint32_t avail_rep, uint32_t rep_offset,
                           uint32_t overhead_cap, nanorq_core *dec, schedule *S_dec, uint32_t *out_overhead, void **out_heap_sched)
{
    return solve_rq_block_ex(w, K, missing, avail_rep, rep_offset, overhead_cap, true, w->rq_prep, w->rq_prep_cap, w->rq_sched_mem,
                             w->rq_sched_cap, w->rq_work, w->rq_work_cap, dec, S_dec, out_overhead, out_heap_sched);
}

static bool compile_rx_unique_rateless(vxfec_wire_ring *wring, wire_worker_ctx *w, uint32_t u, const uint64_t *blk_bits,
                                       uint32_t *out_gf2_tiles, uint32_t *out_256_tiles, uint32_t *out_missing)
{
    uint32_t K = wring->K, P = wring->P;
    uint32_t missing = 0;
    for (uint32_t k = 0; k < K; k++) {
        if (!((blk_bits[k >> 6u] >> (k & 63u)) & 1ULL)) {
            if (missing >= P)
                return false;
            w->miss_esis[missing] = k;
            w->out_slots[missing] = (uint16_t)k;
            missing++;
        }
    }

    uint32_t avail_rep = 0;
    for (uint32_t p = 0; p < P; p++) {
        uint32_t esi = K + p;
        if ((blk_bits[esi >> 6u] >> (esi & 63u)) & 1ULL)
            w->rep_ps[avail_rep++] = p;
    }
    if (avail_rep < missing)
        return false;

    nanorq_core dec = {0};
    schedule S_dec = {0};
    void *heap_sched = NULL;
    uint32_t overhead = 0;
    uint32_t rep_offset = 0;
    uint32_t max_slack = avail_rep - missing;
    uint32_t full_cap = wring->rx_max_overhead;

    /*
     * try WIRE_RQ_FAST_OVERHEAD first when sufficient for GF(2), then
     * WIRE_RQ_MAX_OVERHEAD, then uncapped full slack or sliding repair windows
     * so early dependent repairs or large erasure counts never fail recovery.
     */
    bool solved = solve_rq_block(w, K, missing, avail_rep, 0u, WIRE_RQ_FAST_OVERHEAD, &dec, &S_dec, &overhead, &heap_sched);
    if (!solved && max_slack > WIRE_RQ_FAST_OVERHEAD)
        solved = solve_rq_block(w, K, missing, avail_rep, 0u, WIRE_RQ_MAX_OVERHEAD, &dec, &S_dec, &overhead, &heap_sched);
    if (!solved && max_slack > WIRE_RQ_MAX_OVERHEAD && full_cap > WIRE_RQ_MAX_OVERHEAD)
        solved = solve_rq_block(w, K, missing, avail_rep, 0u, full_cap, &dec, &S_dec, &overhead, &heap_sched);
    if (!solved && max_slack > 2u)
        solved = solve_rq_block(w, K, missing, avail_rep, 0u, 2u, &dec, &S_dec, &overhead, &heap_sched);
    if (!solved) {
        for (uint32_t off = 1u; off <= max_slack && !solved; off++) {
            if (solve_rq_block(w, K, missing, avail_rep, off, full_cap, &dec, &S_dec, &overhead, &heap_sched)) {
                rep_offset = off;
                solved = true;
            }
        }
    }
    if (!solved)
        return false;

    if (!vxfec_rq__sched_is_pure_gf2(&S_dec) && max_slack > overhead) {
        uint32_t retry_cap =
            (max_slack > WIRE_RQ_MAX_OVERHEAD && full_cap > WIRE_RQ_MAX_OVERHEAD) ? full_cap : WIRE_RQ_MAX_OVERHEAD;
        void *retry_heap = NULL;
        nanorq_core retry_dec = {0};
        schedule retry_S = {0};
        uint32_t retry_ov = 0;
        /* run retry in pre-allocated isolated worker buffers so a failed retry never corrupts dec or S_dec */
        if (w->rq_retry_prep && w->rq_retry_sched &&
            solve_rq_block_ex(w, K, missing, avail_rep, rep_offset, retry_cap, false, w->rq_retry_prep, w->rq_prep_cap,
                              w->rq_retry_sched, w->rq_sched_cap, w->rq_work, w->rq_work_cap, &retry_dec, &retry_S, &retry_ov,
                              &retry_heap) &&
            vxfec_rq__sched_is_pure_gf2(&retry_S)) {
            free(heap_sched);
            heap_sched = retry_heap;
            dec = retry_dec;
            S_dec = retry_S;
            overhead = retry_ov;
        } else {
            free(retry_heap);
        }
    }

    bool pure_gf2 = vxfec_rq__sched_is_pure_gf2(&S_dec);
    uint32_t rows = nanorq_core_get_pc_rows(&dec);
    uint32_t sh = nanorq_core_get_pc_genc_offset(&dec);
    const uint32_t *used_reps = w->rep_ps + rep_offset;
    uint32_t m_cur = 0;
    for (uint32_t k = 0; k < K; k++) {
        w->active_rows[k] = sh + k;
        if (m_cur < missing && w->miss_esis[m_cur] == k) {
            w->in_slots[k] = (uint16_t)(K + used_reps[m_cur]);
            m_cur++;
        } else {
            w->in_slots[k] = (uint16_t)k;
        }
    }
    for (uint32_t i = 0; i < overhead; i++) {
        w->active_rows[K + i] = sh + dec.P.Kprime + i;
        w->in_slots[K + i] = (uint16_t)(K + used_reps[missing + i]);
    }
    uint32_t n_inputs = K + overhead;

    uint32_t spans[128][2];
    uint32_t n_tiles = 0, cmd_cnt = 0;
    uint32_t rep_z = wring->uniq_rep_z[u];
    if (pure_gf2) {
        size_t wpr = ((size_t)missing + 63u) / 64u;
        vxfec_rq__adjoint_gf2(&dec, S_dec.ops.a, (uint32_t)S_dec.ops.n, S_dec.cp[0], S_dec.cp[1], rows, w->miss_esis, missing,
                              w->y_gf2, wpr);
        emit_gf2_tiles_local(w->y_gf2, wpr, w->active_rows, w->in_slots, n_inputs, w->out_slots, missing, rep_z,
                             wring->rx_gf2_tile_e, wring->rx_gf2_hdr_words, wring->rx_gf2_hdr_stride, w->hdr_scratch,
                             w->cmd_scratch, &n_tiles, &cmd_cnt, spans);
        publish_unique_mask_to_stage(wring, u, wring->job_gf2_stage, &wring->job_gf2_blocks, &wring->job_gf2_cmd_cursor,
                                     wring->rx_gf2_hdr_words, wring->rx_gf2_hdr_stride, n_tiles, w->hdr_scratch, w->cmd_scratch,
                                     cmd_cnt, spans);
        *out_gf2_tiles = n_tiles;
        *out_256_tiles = 0u;
    } else {
        size_t align = get_align_size();
        size_t m_stride = ((size_t)missing + align - 1u) & ~(align - 1u);
        uint32_t tile_e = wring->job_256_tile_e;
        uint32_t hdr_words = wring->job_256_hdr_words;
        uint32_t hdr_stride = wring->job_256_hdr_stride;
        vxfec_rq__adjoint_gf256(&dec, S_dec.ops.a, (uint32_t)S_dec.ops.n, S_dec.cp[0], S_dec.cp[1], rows, w->miss_esis, missing,
                                w->y_256, m_stride);
        emit_gf256_tiles_local(w->y_256, m_stride, w->active_rows, w->in_slots, n_inputs, w->out_slots, missing, rep_z, tile_e,
                               hdr_words, hdr_stride, w->hdr_scratch, w->cmd_scratch, &n_tiles, &cmd_cnt, spans);
        publish_unique_mask_to_stage(wring, u, wring->job_256_stage, &wring->job_256_blocks, &wring->job_256_cmd_cursor, hdr_words,
                                     hdr_stride, n_tiles, w->hdr_scratch, w->cmd_scratch, cmd_cnt, spans);
        *out_gf2_tiles = 0u;
        *out_256_tiles = n_tiles;
    }

    free(heap_sched);
    *out_missing = missing;
    return true;
}

static void atomic_max_u32(atomic_uint *target, uint32_t val)
{
    if (val == 0u)
        return;
    uint32_t cur = atomic_load_explicit(target, memory_order_relaxed);
    while (val > cur && !atomic_compare_exchange_weak_explicit(target, &cur, val, memory_order_relaxed, memory_order_relaxed)) {
    }
}

static void wire_worker_exec_job(vxfec_wire_ring *wring, wire_worker_ctx *w)
{
    uint32_t n_uniq = wring->job_num_unique;
    size_t wpb = wring->words_per_blk;
    uint32_t max_gf2_t = 0, max_256_t = 0;
    uint32_t local_blks = 0, local_failed = 0, local_syms = 0;

    for (;;) {
        uint32_t u0 = atomic_fetch_add_explicit(&wring->job_next_u, 4u, memory_order_relaxed);
        if (u0 >= n_uniq)
            break;
        uint32_t u_end = (u0 + 4u < n_uniq) ? (u0 + 4u) : n_uniq;

        for (uint32_t u = u0; u < u_end; u++) {
            const uint64_t *blk_bits = wring->uniq_bits + (size_t)u * wpb;
            uint32_t copies = wring->uniq_count[u];
            uint32_t t_gf2 = 0, t_256 = 0, miss = 0;
            bool ok = (wring->codec == VXFEC_WIRE_MDS) ? compile_rx_unique_mds(wring, w, u, blk_bits, &t_256, &miss)
                                                       : compile_rx_unique_rateless(wring, w, u, blk_bits, &t_gf2, &t_256, &miss);
            if (!ok) {
                local_failed += copies;
                continue;
            }
            if (t_gf2 > max_gf2_t)
                max_gf2_t = t_gf2;
            if (t_256 > max_256_t)
                max_256_t = t_256;
            local_blks += copies;
            local_syms += miss * copies;
        }
    }

    if (local_blks > 0u)
        atomic_fetch_add_explicit(&wring->job_blocks_rep, local_blks, memory_order_relaxed);
    if (local_failed > 0u)
        atomic_fetch_add_explicit(&wring->job_blocks_failed, local_failed, memory_order_relaxed);
    if (local_syms > 0u)
        atomic_fetch_add_explicit(&wring->job_syms_rep, local_syms, memory_order_relaxed);
    atomic_max_u32(&wring->job_gf2_active_tiles, max_gf2_t);
    atomic_max_u32(&wring->job_256_active_tiles, max_256_t);
}

static void *wire_worker_thread_main(void *arg)
{
    wire_worker_ctx *w = (wire_worker_ctx *)arg;
    vxfec_wire_ring *wring = w->owner;
    uint64_t seen_epoch = 0;

    pthread_mutex_lock(&wring->mu);
    for (;;) {
        while (!wring->shutdown && wring->job_epoch == seen_epoch)
            pthread_cond_wait(&wring->cv_start, &wring->mu);
        if (wring->shutdown)
            break;
        seen_epoch = wring->job_epoch;
        pthread_mutex_unlock(&wring->mu);

        wire_worker_exec_job(wring, w);

        pthread_mutex_lock(&wring->mu);
        if (--wring->active_bg_workers == 0u)
            pthread_cond_signal(&wring->cv_done);
    }
    pthread_mutex_unlock(&wring->mu);
    return NULL;
}

static uint32_t count_sys_erasures(const uint64_t *blk_bits, uint32_t K)
{
    uint32_t missing = 0;
    uint32_t full_words = K >> 6u;
    for (uint32_t w = 0; w < full_words; w++) {
        uint64_t inv = ~blk_bits[w];
        if (inv != 0ULL)
            missing += (uint32_t)__builtin_popcountll(inv);
    }
    uint32_t rem = K & 63u;
    if (rem > 0u) {
        uint64_t need = (1ULL << rem) - 1ULL;
        uint64_t inv = (~blk_bits[full_words]) & need;
        if (inv != 0ULL)
            missing += (uint32_t)__builtin_popcountll(inv);
    }
    return missing;
}

/* clear received parity bits beyond keep_par so blocks differing only in unused tail repairs deduplicate */
static void mask_surplus_parities(uint64_t *cand, size_t wpb, uint32_t K, uint32_t keep_par)
{
    size_t w_idx = K >> 6u;
    uint32_t bit_off = K & 63u;
    uint32_t rem = keep_par;

    for (size_t w = w_idx; w < wpb; w++) {
        uint64_t sys_mask = (w == w_idx && bit_off > 0u) ? ((1ULL << bit_off) - 1ULL) : 0ULL;
        uint64_t par_bits = cand[w] & ~sys_mask;
        if (rem == 0u) {
            cand[w] = cand[w] & sys_mask;
            continue;
        }
        uint32_t cnt = (uint32_t)__builtin_popcountll(par_bits);
        if (cnt <= rem) {
            rem -= cnt;
            continue;
        }
        uint64_t kept = 0ULL;
        while (rem > 0u) {
            uint64_t lsb = par_bits & (~par_bits + 1ULL);
            kept |= lsb;
            par_bits ^= lsb;
            rem--;
        }
        cand[w] = (cand[w] & sys_mask) | kept;
    }
}

static uint64_t hash_block_bits(const uint64_t *blk_bits, size_t wpb)
{
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (size_t w = 0; w < wpb; w++) {
        h ^= blk_bits[w];
        h = (h ^ (h >> 30u)) * 0xbf58476d1ce4e5b9ULL;
    }
    h = (h ^ (h >> 27u)) * 0x94d049bb133111ebULL;
    return h ^ (h >> 31u);
}

static uint32_t dedup_erased_blocks(vxfec_wire_ring *wring, const uint64_t *rx_block_bits, uint32_t *out_max_missing)
{
    uint32_t Z = wring->Z, K = wring->K, P = wring->P;
    size_t wpb = wring->words_per_blk;
    uint32_t total_syms = K + P;
    uint32_t last_rem = total_syms & 63u;
    uint64_t valid_last_mask = (last_rem == 0u) ? ~0ULL : ((1ULL << last_rem) - 1ULL);
    uint32_t hmask = wring->uniq_hash_cap - 1u;

    memset(wring->uniq_hash, 0xff, (size_t)wring->uniq_hash_cap * sizeof(uint32_t));
    uint32_t n_uniq = 0;
    uint32_t max_missing = 0;

    for (uint32_t z = 0; z < Z; z++) {
        const uint64_t *blk_bits = rx_block_bits + (size_t)z * wpb;
        uint32_t missing = count_sys_erasures(blk_bits, K);
        if (missing == 0u)
            continue;

        uint64_t *cand = wring->uniq_bits + (size_t)n_uniq * wpb;
        memcpy(cand, blk_bits, wpb * sizeof(uint64_t));
        cand[wpb - 1u] &= valid_last_mask;

        if (missing <= P) {
            if (missing > max_missing)
                max_missing = missing;
            uint32_t keep_par = (wring->codec == VXFEC_WIRE_MDS) ? missing : (missing + WIRE_RQ_MAX_OVERHEAD);
            if (keep_par < P)
                mask_surplus_parities(cand, wpb, K, keep_par);
        }

        uint32_t slot = (uint32_t)hash_block_bits(cand, wpb) & hmask;
        for (;;) {
            uint32_t u = wring->uniq_hash[slot];
            if (u == UINT32_MAX) {
                u = n_uniq++;
                wring->uniq_hash[slot] = u;
                wring->uniq_rep_z[u] = z;
                wring->uniq_head[u] = z;
                wring->uniq_count[u] = 1u;
                wring->blk_next[z] = UINT32_MAX;
                break;
            }
            const uint64_t *rep_bits = wring->uniq_bits + (size_t)u * wpb;
            if (memcmp(cand, rep_bits, wpb * sizeof(uint64_t)) == 0) {
                wring->blk_next[z] = wring->uniq_head[u];
                wring->uniq_head[u] = z;
                wring->uniq_count[u]++;
                break;
            }
            slot = (slot + 1u) & hmask;
        }
    }
    if (out_max_missing)
        *out_max_missing = max_missing;
    return n_uniq;
}

vxfec_wire_ring *vxfec_wire_ring_new_ex(vxfec_vk_ctx *ctx, int codec, uint32_t Z, uint32_t K, uint32_t P, uint32_t stride,
                                        uint32_t num_threads, bool enable_tx, bool enable_rx)
{
    bool is_rateless = (codec == VXFEC_WIRE_RATELESS || codec == VXFEC_WIRE_RATELESS_GF2);
    bool gf2_h0_mode = (codec == VXFEC_WIRE_RATELESS_GF2);
    if (!ctx || (!enable_tx && !enable_rx) || (!is_rateless && codec != VXFEC_WIRE_MDS) || Z == 0 || Z > (1u << 28u) || K == 0 ||
        P == 0 || K + P > 65535u || stride < 16u || (stride & 15u) != 0)
        return NULL;
    if (codec == VXFEC_WIRE_MDS && (K > 255u || P > 256u - K || P > 256u))
        return NULL;
    if (is_rateless && P > 1024u)
        return NULL;

    size_t wire_bytes = 0;
    if (!vxfec_vk__mul3((size_t)(K + P), Z, stride, &wire_bytes) || wire_bytes > ctx->max_ssbo_range)
        return NULL;

    if (is_rateless)
        nanorq_core_init();

    vxfec_wire_ring *wring = (vxfec_wire_ring *)calloc(1, sizeof(*wring));
    if (!wring)
        return NULL;

    wring->ctx = ctx;
    wring->codec = codec;
    wring->Z = Z;
    wring->K = K;
    wring->P = P;
    wring->stride = stride;
    wring->wire_bytes = wire_bytes;
    wring->words_per_blk = ((size_t)(K + P) + 63u) / 64u;
    wring->enable_tx = enable_tx;
    wring->enable_rx = enable_rx;

    if (enable_tx && !build_tx_table(wring))
        goto fail;

    if (enable_rx) {
        if (num_threads == 0u) {
            long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
            num_threads = (ncpu > 1) ? (uint32_t)ncpu : 1u;
            if (num_threads > 24u)
                num_threads = 24u;
        }
        if (num_threads > Z)
            num_threads = Z;
        wring->num_workers = num_threads;

        uint32_t hcap = 64u;
        while (hcap < Z * 2u) {
            if (hcap > (UINT32_MAX >> 1u))
                goto fail;
            hcap <<= 1u;
        }
        wring->uniq_hash_cap = hcap;
        wring->uniq_hash = (uint32_t *)malloc((size_t)hcap * sizeof(uint32_t));
        wring->uniq_rep_z = (uint32_t *)malloc((size_t)Z * sizeof(uint32_t));
        wring->uniq_head = (uint32_t *)malloc((size_t)Z * sizeof(uint32_t));
        wring->uniq_count = (uint32_t *)malloc((size_t)Z * sizeof(uint32_t));
        wring->blk_next = (uint32_t *)malloc((size_t)Z * sizeof(uint32_t));
        wring->uniq_bits = (uint64_t *)calloc((size_t)Z * wring->words_per_blk, sizeof(uint64_t));
        if (!wring->uniq_hash || !wring->uniq_rep_z || !wring->uniq_head || !wring->uniq_count || !wring->blk_next ||
            !wring->uniq_bits)
            goto fail;

        wring->rx_gf2_tile_e = (P <= 16u) ? 16u : 32u;
        wring->rx_gf2_hdr_words = (wring->rx_gf2_tile_e == 32u) ? 5u : 3u;
        wring->rx_gf2_hdr_stride = (P + wring->rx_gf2_tile_e - 1u) / wring->rx_gf2_tile_e;

        wring->rx_256_tile_e = choose_wire_256_tile_e(P, &wring->rx_256_tile_forced);
        wring->rx_256_hdr_words = wire_256_hdr_words(wring->rx_256_tile_e);
        wring->rx_256_hdr_stride = (P + wring->rx_256_tile_e - 1u) / wring->rx_256_tile_e;
        size_t wpc_256 = wire_256_words_per_cmd(wring->rx_256_tile_e);

        uint32_t max_ov = is_rateless ? ((P > WIRE_RQ_MAX_OVERHEAD) ? P : WIRE_RQ_MAX_OVERHEAD) : 0u;
        if (is_rateless && max_ov > WIRE_RQ_MAX_OVERHEAD) {
            size_t full_in = (size_t)K + max_ov;
            size_t full_blk = (size_t)wring->rx_256_hdr_words + wpc_256 * full_in;
            size_t full_uvec4 = 0, full_bytes = 0;
            if (!vxfec_vk__mul3(Z, wring->rx_256_hdr_stride, full_blk, &full_uvec4) ||
                !vxfec_vk__mul3(full_uvec4, 16u, 1u, &full_bytes) || full_bytes > ctx->max_ssbo_range)
                max_ov = WIRE_RQ_MAX_OVERHEAD;
        }
        wring->rx_max_overhead = max_ov;

        size_t max_in_rq = (size_t)K + max_ov;
        size_t gf2_per_blk = (size_t)wring->rx_gf2_hdr_words + (max_in_rq + 1u) / 2u;
        size_t gf2_tab_bytes = 0;
        if (!vxfec_vk__mul3(Z, wring->rx_gf2_hdr_stride, gf2_per_blk, &wring->rx_gf2_max_uvec4) ||
            !vxfec_vk__mul3(wring->rx_gf2_max_uvec4, 16u, 1u, &gf2_tab_bytes))
            goto fail;

        size_t max_in_256 = (codec == VXFEC_WIRE_MDS) ? (size_t)K : max_in_rq;
        size_t m256_per_blk = (size_t)wring->rx_256_hdr_words + wpc_256 * max_in_256;
        size_t m256_tab_bytes = 0;
        if (!vxfec_vk__mul3(Z, wring->rx_256_hdr_stride, m256_per_blk, &wring->rx_256_max_uvec4) ||
            !vxfec_vk__mul3(wring->rx_256_max_uvec4, 16u, 1u, &m256_tab_bytes))
            goto fail;

        if (m256_tab_bytes > ctx->max_ssbo_range)
            goto fail;
        for (int s = 0; s < 2; s++) {
            wring->rx_tab_256_buf[s] = vxfec_vk_buf_new(ctx, m256_tab_bytes, true);
            if (!wring->rx_tab_256_buf[s])
                goto fail;
        }

        if (is_rateless) {
            if (gf2_tab_bytes > ctx->max_ssbo_range)
                goto fail;
            for (int s = 0; s < 2; s++) {
                wring->rx_tab_gf2_buf[s] = vxfec_vk_buf_new(ctx, gf2_tab_bytes, true);
                if (!wring->rx_tab_gf2_buf[s])
                    goto fail;
            }
        }

        size_t rq_prep_cap = 0, rq_work_cap = 0, rq_sched_cap = 0, max_rows = K;
        if (is_rateless) {
            nanorq_core probe = {0};
            if (!init_rq_core(gf2_h0_mode, K, max_ov, &probe, NULL))
                goto fail;
            rq_prep_cap = nanorq_core_calculate_prepare_memory(&probe);
            rq_work_cap = nanorq_core_calculate_work_memory(&probe);
            rq_sched_cap = ops_estimate_schedule_bytes(K + max_ov);
            max_rows = nanorq_core_get_pc_rows(&probe);
        }

        size_t max_wpr = ((size_t)P + 63u) / 64u;
        size_t max_m_stride = ((size_t)P + 63u) & ~(size_t)63u;
        size_t hdr_scratch_uvec4 = ((size_t)wring->rx_gf2_hdr_stride + wring->rx_256_hdr_stride) * 5u + 8u;
        size_t cmd_scratch_uvec4 = (size_t)wring->rx_256_hdr_stride * 3u * max_in_rq + 16u;

        wring->workers = (wire_worker_ctx *)calloc(num_threads, sizeof(*wring->workers));
        if (!wring->workers)
            goto fail;

        for (uint32_t i = 0; i < num_threads; i++) {
            wire_worker_ctx *w = &wring->workers[i];
            w->owner = wring;
            w->tid = i;
            w->active_rows = (uint32_t *)malloc(max_in_rq * sizeof(uint32_t));
            w->in_slots = (uint16_t *)malloc(max_in_rq * sizeof(uint16_t));
            w->out_slots = (uint16_t *)malloc((size_t)P * sizeof(uint16_t));
            w->y_256 = (uint8_t *)malloc(max_rows * max_m_stride);
            w->hdr_scratch = (uint32_t *)malloc(hdr_scratch_uvec4 * 16u);
            w->cmd_scratch = (uint32_t *)malloc(cmd_scratch_uvec4 * 16u);
            if (!w->active_rows || !w->in_slots || !w->out_slots || !w->y_256 || !w->hdr_scratch || !w->cmd_scratch)
                goto fail;
            if (is_rateless) {
                w->rq_prep_cap = rq_prep_cap;
                w->rq_work_cap = rq_work_cap;
                w->rq_sched_cap = rq_sched_cap;
                w->rq_prep = (uint8_t *)malloc(rq_prep_cap);
                w->rq_retry_prep = (uint8_t *)malloc(rq_prep_cap);
                w->rq_prep_tmpl16 = (uint8_t *)malloc(rq_prep_cap);
                w->rq_prep_tmpl32 = (uint8_t *)malloc(rq_prep_cap);
                w->rq_work = (uint8_t *)malloc(rq_work_cap);
                w->rq_sched_mem = malloc(rq_sched_cap);
                w->rq_retry_sched = malloc(rq_sched_cap);
                w->y_gf2 = (uint64_t *)malloc(max_rows * max_wpr * sizeof(uint64_t));
                w->miss_esis = (uint32_t *)malloc((size_t)P * sizeof(uint32_t));
                w->rep_ps = (uint32_t *)malloc((size_t)P * sizeof(uint32_t));
                if (!w->rq_prep || !w->rq_retry_prep || !w->rq_prep_tmpl16 || !w->rq_prep_tmpl32 || !w->rq_work ||
                    !w->rq_sched_mem || !w->rq_retry_sched || !w->y_gf2 || !w->miss_esis || !w->rep_ps)
                    goto fail;
                if (init_rq_core(gf2_h0_mode, K, WIRE_RQ_FAST_OVERHEAD, &w->rq_tmpl16_dec, NULL)) {
                    size_t tplen = nanorq_core_calculate_prepare_memory(&w->rq_tmpl16_dec);
                    if (tplen > 0 && tplen <= rq_prep_cap && nanorq_core_generate(&w->rq_tmpl16_dec, w->rq_prep, tplen)) {
                        w->rq_tmpl16_bytes = (size_t)(w->rq_tmpl16_dec.W.prep_mem.beg - w->rq_prep);
                        memcpy(w->rq_prep_tmpl16, w->rq_prep, w->rq_tmpl16_bytes);
                        w->rq_tmpl16_valid = true;
                    }
                }
                if (init_rq_core(gf2_h0_mode, K, WIRE_RQ_MAX_OVERHEAD, &w->rq_tmpl32_dec, NULL)) {
                    size_t tplen = nanorq_core_calculate_prepare_memory(&w->rq_tmpl32_dec);
                    if (tplen > 0 && tplen <= rq_prep_cap && nanorq_core_generate(&w->rq_tmpl32_dec, w->rq_prep, tplen)) {
                        w->rq_tmpl32_bytes = (size_t)(w->rq_tmpl32_dec.W.prep_mem.beg - w->rq_prep);
                        memcpy(w->rq_prep_tmpl32, w->rq_prep, w->rq_tmpl32_bytes);
                        w->rq_tmpl32_valid = true;
                    }
                }
            }
        }

        if (num_threads > 1u) {
            if (pthread_mutex_init(&wring->mu, NULL) != 0)
                goto fail;
            if (pthread_cond_init(&wring->cv_start, NULL) != 0) {
                pthread_mutex_destroy(&wring->mu);
                goto fail;
            }
            if (pthread_cond_init(&wring->cv_done, NULL) != 0) {
                pthread_cond_destroy(&wring->cv_start);
                pthread_mutex_destroy(&wring->mu);
                goto fail;
            }
            wring->sync_ready = true;
            for (uint32_t i = 1u; i < num_threads; i++) {
                if (pthread_create(&wring->workers[i].thread, NULL, wire_worker_thread_main, &wring->workers[i]) != 0)
                    goto fail;
                wring->workers[i].started = true;
            }
        }
    }

    return wring;

fail:
    vxfec_wire_ring_free(wring);
    return NULL;
}

vxfec_wire_ring *vxfec_wire_ring_new(vxfec_vk_ctx *ctx, int codec, uint32_t Z, uint32_t K, uint32_t P, uint32_t stride,
                                     uint32_t num_threads)
{
    return vxfec_wire_ring_new_ex(ctx, codec, Z, K, P, stride, num_threads, true, true);
}

void vxfec_wire_ring_free(vxfec_wire_ring *wring)
{
    if (!wring)
        return;
    bool vk_idle = !wring->ctx || vxfec_vk_wait_idle(wring->ctx);
    if (wring->sync_ready) {
        pthread_mutex_lock(&wring->mu);
        wring->shutdown = true;
        pthread_cond_broadcast(&wring->cv_start);
        pthread_mutex_unlock(&wring->mu);
        for (uint32_t i = 1u; i < wring->num_workers; i++) {
            if (wring->workers && wring->workers[i].started)
                pthread_join(wring->workers[i].thread, NULL);
        }
        pthread_cond_destroy(&wring->cv_done);
        pthread_cond_destroy(&wring->cv_start);
        pthread_mutex_destroy(&wring->mu);
    }
    if (wring->workers) {
        for (uint32_t i = 0; i < wring->num_workers; i++) {
            wire_worker_ctx *w = &wring->workers[i];
            free(w->rq_prep);
            free(w->rq_retry_prep);
            free(w->rq_prep_tmpl16);
            free(w->rq_prep_tmpl32);
            free(w->rq_work);
            free(w->rq_sched_mem);
            free(w->rq_retry_sched);
            free(w->y_gf2);
            free(w->y_256);
            free(w->miss_esis);
            free(w->rep_ps);
            free(w->active_rows);
            free(w->in_slots);
            free(w->out_slots);
            free(w->hdr_scratch);
            free(w->cmd_scratch);
        }
        free(wring->workers);
    }
    free(wring->uniq_hash);
    free(wring->uniq_rep_z);
    free(wring->uniq_head);
    free(wring->uniq_count);
    free(wring->blk_next);
    free(wring->uniq_bits);
    if (vk_idle) {
        vxfec_vk_buf_free(wring->ctx, wring->tx_tab_buf);
        for (int s = 0; s < 2; s++) {
            vxfec_vk_buf_free(wring->ctx, wring->rx_tab_gf2_buf[s]);
            vxfec_vk_buf_free(wring->ctx, wring->rx_tab_256_buf[s]);
        }
    } else {
        /* preserve GPU table buffer ownership on ctx until device idle is confirmed */
        vxfec_vk__retire_buf(wring->ctx, wring->tx_tab_buf);
        for (int s = 0; s < 2; s++) {
            vxfec_vk__retire_buf(wring->ctx, wring->rx_tab_gf2_buf[s]);
            vxfec_vk__retire_buf(wring->ctx, wring->rx_tab_256_buf[s]);
        }
    }
    free(wring);
}

size_t vxfec_wire_ring_words_per_block(const vxfec_wire_ring *wring)
{
    return wring ? wring->words_per_blk : 0;
}

uint32_t vxfec_wire_ring_workers(const vxfec_wire_ring *wring)
{
    return wring ? wring->num_workers : 0;
}

bool vxfec_wire_ring_tx_is_pure_gf2(const vxfec_wire_ring *wring)
{
    return wring ? wring->tx_pure_gf2 : false;
}

bool vxfec_wire_encode_vram(vxfec_wire_ring *wring, vxfec_vk_buf *wire_vram, uint64_t *gpu_ns_out)
{
    if (!wring || !wring->enable_tx || !wire_vram || wire_vram->bytes < wring->wire_bytes)
        return false;
    return vxfec_vk__exec_wire_cmd(wring->ctx, wire_vram, wring->wire_bytes, wring->tx_tab_buf, wring->tx_tab_bytes, 0u, 0u, false,
                                   wring->tx_pure_gf2, wring->tx_tile_e, wring->Z, wring->stride, wring->tx_tiles, wring->tx_tiles,
                                   wring->Z, true, gpu_ns_out);
}

uint64_t vxfec_wire_encode_submit(vxfec_wire_ring *wring, vxfec_vk_buf *wire_vram)
{
    if (!wring || !wring->enable_tx || !wire_vram || wire_vram->bytes < wring->wire_bytes)
        return 0;
    return vxfec_vk__submit_wire_cmd(wring->ctx, wire_vram, wring->wire_bytes, wring->tx_tab_buf, wring->tx_tab_bytes, 0u, 0u,
                                     false, wring->tx_pure_gf2, wring->tx_tile_e, wring->Z, wring->stride, wring->tx_tiles,
                                     wring->tx_tiles, wring->Z, true);
}

static bool acquire_rx_superframe_slot(vxfec_wire_ring *wring, bool wait_slot, uint32_t *out_slot)
{
    for (uint32_t i = 0; i < 2u; i++) {
        uint32_t s = (wring->rx_slot_cursor + i) & 1u;
        if (wring->rx_slot_tickets[s] == 0u) {
            *out_slot = s;
            return true;
        }
        int st = vxfec_vk_poll_status(wring->ctx, wring->rx_slot_tickets[s]);
        if (st < 0)
            return false;
        if (st > 0) {
            wring->rx_slot_tickets[s] = 0u;
            *out_slot = s;
            return true;
        }
    }
    if (!wait_slot)
        return false;
    uint32_t s = wring->rx_slot_cursor & 1u;
    if (!vxfec_vk_wait(wring->ctx, wring->rx_slot_tickets[s], UINT64_MAX))
        return false;
    wring->rx_slot_tickets[s] = 0u;
    *out_slot = s;
    return true;
}

bool vxfec_wire_can_submit(vxfec_wire_ring *wring)
{
    if (!wring || !wring->enable_rx || !vxfec_vk_async_can_submit(wring->ctx))
        return false;
    uint32_t slot = 0;
    return acquire_rx_superframe_slot(wring, false, &slot);
}

typedef struct {
    uint64_t cpu_ns;
    uint32_t unique_masks;
    uint32_t blocks_failed;
    uint32_t gf2_blks;
    uint32_t gf2_tiles;
    size_t gf2_bytes;
    size_t gf2_hdr_bytes;
    size_t gf2_cmd_off;
    uint32_t mds_blks;
    uint32_t mds_tiles;
    uint32_t mds_tile_e;
    uint32_t mds_hdr_stride;
    size_t mds_bytes;
    size_t mds_hdr_bytes;
    size_t mds_cmd_off;
    size_t xfer_bytes;
} wire_rx_compile_res;

static void compile_rx_superframe_slot(vxfec_wire_ring *wring, uint32_t slot, const uint64_t *rx_block_bits,
                                       wire_rx_compile_res *out)
{
    memset(out, 0, sizeof(*out));
    uint64_t t0 = wire_now_ns();

    uint32_t max_missing = 0;
    uint32_t n_uniq = dedup_erased_blocks(wring, rx_block_bits, &max_missing);
    out->unique_masks = n_uniq;
    out->mds_tile_e = wring->rx_256_tile_e;
    out->mds_hdr_stride = wring->rx_256_hdr_stride;
    if (n_uniq == 0u) {
        atomic_store_explicit(&wring->job_blocks_rep, 0u, memory_order_relaxed);
        atomic_store_explicit(&wring->job_blocks_failed, 0u, memory_order_relaxed);
        atomic_store_explicit(&wring->job_syms_rep, 0u, memory_order_relaxed);
        out->cpu_ns = wire_now_ns() - t0;
        return;
    }

    /* dynamically downshift GF(256) tile width to 8 when max_missing <= 8 */
    uint32_t cur_256_tile_e = (!wring->rx_256_tile_forced && max_missing > 0u && max_missing <= 8u) ? 8u : wring->rx_256_tile_e;
    uint32_t cur_256_hdr_words = wire_256_hdr_words(cur_256_tile_e);
    uint32_t cur_256_hdr_stride = (cur_256_tile_e == 8u && wring->rx_256_tile_e != 8u) ? 1u : wring->rx_256_hdr_stride;
    wring->job_256_tile_e = cur_256_tile_e;
    wring->job_256_hdr_words = cur_256_hdr_words;
    wring->job_256_hdr_stride = cur_256_hdr_stride;
    out->mds_tile_e = cur_256_tile_e;
    out->mds_hdr_stride = cur_256_hdr_stride;

    uint32_t gf2_hdr_uvec4 = wring->Z * wring->rx_gf2_hdr_stride * wring->rx_gf2_hdr_words;
    uint32_t mds_hdr_uvec4 = wring->Z * cur_256_hdr_stride * cur_256_hdr_words;

    wring->job_num_unique = n_uniq;
    wring->job_gf2_stage = wring->rx_tab_gf2_buf[slot] ? (uint32_t *)vxfec_vk_buf_staging_ptr(wring->rx_tab_gf2_buf[slot]) : NULL;
    wring->job_256_stage = (uint32_t *)vxfec_vk_buf_staging_ptr(wring->rx_tab_256_buf[slot]);
    atomic_store_explicit(&wring->job_next_u, 0u, memory_order_relaxed);
    atomic_store_explicit(&wring->job_gf2_blocks, 0u, memory_order_relaxed);
    atomic_store_explicit(&wring->job_gf2_cmd_cursor, gf2_hdr_uvec4, memory_order_relaxed);
    atomic_store_explicit(&wring->job_gf2_active_tiles, 0u, memory_order_relaxed);
    atomic_store_explicit(&wring->job_256_blocks, 0u, memory_order_relaxed);
    atomic_store_explicit(&wring->job_256_cmd_cursor, mds_hdr_uvec4, memory_order_relaxed);
    atomic_store_explicit(&wring->job_256_active_tiles, 0u, memory_order_relaxed);
    atomic_store_explicit(&wring->job_blocks_rep, 0u, memory_order_relaxed);
    atomic_store_explicit(&wring->job_blocks_failed, 0u, memory_order_relaxed);
    atomic_store_explicit(&wring->job_syms_rep, 0u, memory_order_relaxed);

    bool use_pool = (wring->num_workers > 1u && n_uniq > 2u);
    if (use_pool) {
        pthread_mutex_lock(&wring->mu);
        wring->active_bg_workers = wring->num_workers - 1u;
        wring->job_epoch++;
        pthread_cond_broadcast(&wring->cv_start);
        pthread_mutex_unlock(&wring->mu);
    }

    wire_worker_exec_job(wring, &wring->workers[0]);

    if (use_pool) {
        pthread_mutex_lock(&wring->mu);
        while (wring->active_bg_workers > 0u)
            pthread_cond_wait(&wring->cv_done, &wring->mu);
        pthread_mutex_unlock(&wring->mu);
    }

    out->cpu_ns = wire_now_ns() - t0;
    out->blocks_failed = atomic_load_explicit(&wring->job_blocks_failed, memory_order_relaxed);

    uint32_t gf2_blks = atomic_load_explicit(&wring->job_gf2_blocks, memory_order_relaxed);
    uint32_t gf2_cur = atomic_load_explicit(&wring->job_gf2_cmd_cursor, memory_order_relaxed);
    out->gf2_blks = gf2_blks;
    out->gf2_tiles = atomic_load_explicit(&wring->job_gf2_active_tiles, memory_order_relaxed);
    if (gf2_blks > 0u) {
        out->gf2_bytes = (size_t)gf2_cur * 16u;
        out->gf2_hdr_bytes = (size_t)gf2_blks * wring->rx_gf2_hdr_stride * wring->rx_gf2_hdr_words * 16u;
        out->gf2_cmd_off = (size_t)gf2_hdr_uvec4 * 16u;
        out->xfer_bytes += out->gf2_hdr_bytes + (out->gf2_bytes - out->gf2_cmd_off);
    }

    uint32_t mds_blks = atomic_load_explicit(&wring->job_256_blocks, memory_order_relaxed);
    uint32_t mds_cur = atomic_load_explicit(&wring->job_256_cmd_cursor, memory_order_relaxed);
    out->mds_blks = mds_blks;
    out->mds_tiles = atomic_load_explicit(&wring->job_256_active_tiles, memory_order_relaxed);
    if (mds_blks > 0u) {
        out->mds_bytes = (size_t)mds_cur * 16u;
        out->mds_hdr_bytes = (size_t)mds_blks * cur_256_hdr_stride * cur_256_hdr_words * 16u;
        out->mds_cmd_off = (size_t)mds_hdr_uvec4 * 16u;
        out->xfer_bytes += out->mds_hdr_bytes + (out->mds_bytes - out->mds_cmd_off);
    }
}

bool vxfec_wire_recover_vram(vxfec_wire_ring *wring, vxfec_vk_buf *wire_vram, const uint64_t *rx_block_bits,
                             vxfec_wire_stats *out_stats)
{
    if (!wring || !wring->enable_rx || !wire_vram || !rx_block_bits || wire_vram->bytes < wring->wire_bytes)
        return false;

    uint64_t t0 = wire_now_ns();
    uint32_t slot = 0;
    uint64_t gpu_ns = 0;
    wire_rx_compile_res cr;

    if (!acquire_rx_superframe_slot(wring, true, &slot))
        return false;
    compile_rx_superframe_slot(wring, slot, rx_block_bits, &cr);

    if (!vxfec_vk__exec_wire_hybrid_cmd(wring->ctx, wire_vram, wring->wire_bytes, wring->Z, wring->stride,
                                        wring->rx_tab_gf2_buf[slot], cr.gf2_bytes, cr.gf2_hdr_bytes, cr.gf2_cmd_off,
                                        wring->rx_gf2_tile_e, wring->rx_gf2_hdr_stride, cr.gf2_tiles, cr.gf2_blks,
                                        wring->rx_tab_256_buf[slot], cr.mds_bytes, cr.mds_hdr_bytes, cr.mds_cmd_off, cr.mds_tile_e,
                                        cr.mds_hdr_stride, cr.mds_tiles, cr.mds_blks, &gpu_ns))
        return false;

    uint64_t t_done = wire_now_ns();
    if (out_stats) {
        out_stats->cpu_compile_ns = cr.cpu_ns;
        out_stats->gpu_exec_ns = gpu_ns;
        out_stats->total_wall_ns = t_done - t0;
        out_stats->tab_bytes = cr.xfer_bytes;
        out_stats->blocks_repaired = atomic_load_explicit(&wring->job_blocks_rep, memory_order_relaxed);
        out_stats->blocks_failed = cr.blocks_failed;
        out_stats->unique_masks = cr.unique_masks;
        out_stats->gf2_blocks = cr.gf2_blks;
        out_stats->gf256_blocks = cr.mds_blks;
        out_stats->symbols_repaired = atomic_load_explicit(&wring->job_syms_rep, memory_order_relaxed);
        out_stats->workers_used = (cr.unique_masks > 2u) ? wring->num_workers : 1u;
        out_stats->pure_gf2 = (cr.gf2_blks > 0u && cr.mds_blks == 0u);
    }
    return cr.blocks_failed == 0u;
}

uint64_t vxfec_wire_recover_submit(vxfec_wire_ring *wring, vxfec_vk_buf *wire_vram, const uint64_t *rx_block_bits,
                                   vxfec_wire_stats *out_stats)
{
    if (!wring || !wring->enable_rx || !wire_vram || !rx_block_bits || wire_vram->bytes < wring->wire_bytes ||
        !vxfec_vk_async_can_submit(wring->ctx))
        return 0;

    uint64_t t0 = wire_now_ns();
    uint32_t slot = 0;
    wire_rx_compile_res cr;

    if (!acquire_rx_superframe_slot(wring, false, &slot))
        return 0;
    compile_rx_superframe_slot(wring, slot, rx_block_bits, &cr);
    if (cr.blocks_failed > 0u) {
        if (out_stats) {
            memset(out_stats, 0, sizeof(*out_stats));
            out_stats->cpu_compile_ns = cr.cpu_ns;
            out_stats->total_wall_ns = wire_now_ns() - t0;
            out_stats->blocks_failed = cr.blocks_failed;
            out_stats->unique_masks = cr.unique_masks;
        }
        return 0;
    }

    uint64_t ticket = vxfec_vk__submit_wire_hybrid_cmd(wring->ctx, wire_vram, wring->wire_bytes, wring->Z, wring->stride,
                                                       wring->rx_tab_gf2_buf[slot], cr.gf2_bytes, cr.gf2_hdr_bytes, cr.gf2_cmd_off,
                                                       wring->rx_gf2_tile_e, wring->rx_gf2_hdr_stride, cr.gf2_tiles, cr.gf2_blks,
                                                       wring->rx_tab_256_buf[slot], cr.mds_bytes, cr.mds_hdr_bytes, cr.mds_cmd_off,
                                                       cr.mds_tile_e, cr.mds_hdr_stride, cr.mds_tiles, cr.mds_blks);
    if (ticket == 0)
        return 0;

    bool has_gpu_work = (cr.gf2_blks > 0u && cr.gf2_tiles > 0u) || (cr.mds_blks > 0u && cr.mds_tiles > 0u);
    wring->rx_slot_tickets[slot] = has_gpu_work ? ticket : 0u;
    wring->rx_slot_cursor = slot + 1u;

    if (out_stats) {
        out_stats->cpu_compile_ns = cr.cpu_ns;
        out_stats->gpu_exec_ns = 0;
        out_stats->total_wall_ns = wire_now_ns() - t0;
        out_stats->tab_bytes = cr.xfer_bytes;
        out_stats->blocks_repaired = atomic_load_explicit(&wring->job_blocks_rep, memory_order_relaxed);
        out_stats->blocks_failed = 0u;
        out_stats->unique_masks = cr.unique_masks;
        out_stats->gf2_blocks = cr.gf2_blks;
        out_stats->gf256_blocks = cr.mds_blks;
        out_stats->symbols_repaired = atomic_load_explicit(&wring->job_syms_rep, memory_order_relaxed);
        out_stats->workers_used = (cr.unique_masks > 2u) ? wring->num_workers : 1u;
        out_stats->pure_gf2 = (cr.gf2_blks > 0u && cr.mds_blks == 0u);
    }
    return ticket;
}
