#include "vxfec_rq.h"
#include "util.h"
#include "vk_internal.h"

#include <stdlib.h>
#include <string.h>

struct vxfec_rq_sched {
    vxfec_vk_buf *ops_buf;
    sched_op *raw_ops;
    uint32_t n_raw_ops;
    uint32_t raw_cp0;
    uint32_t raw_cp1;
    uint32_t rows;
    uint32_t cp0_op4;
    uint32_t cp1_op4;
    uint32_t fwd_op4;
    uint32_t total_op4;
    uint32_t hdpc_beg_op4;
    uint32_t hdpc_end_op4;
    bool pure_gf2;
    bool ops_uploaded;
};

struct vxfec_rq_direct {
    vxfec_vk_direct_desc desc;
};

bool vxfec_rq__sched_is_pure_gf2(const schedule *S)
{
    if (!S)
        return false;
    for (size_t i = 0; i < S->ops.n; i++) {
        sched_op op = S->ops.a[i];
        if (op.u == 0) {
            if ((op.j & 0xffu) != 1u)
                return false;
        } else if (op.u != 1u) {
            return false;
        }
    }
    return true;
}

static size_t pack_ops_span(const sched_op *ops, uint32_t beg, uint32_t end, bool rev, bool is_pure_gf2, vxfec_gpu_op4 *out)
{
    size_t n = 0;
    uint32_t rem = (end > beg) ? (end - beg) : 0u;
    uint32_t pos = 0;
    uint32_t max_fanout = is_pure_gf2 ? 5u : 4u;

    while (pos < rem) {
        uint32_t cur_i = rev ? (end - 1u - pos) : (beg + pos);
        sched_op first = ops[cur_i];
        if (first.u == 0) {
            uint32_t scale = first.j & 0xffu;
            if (scale != 1u) {
                out[n++] = (vxfec_gpu_op4){
                    .j = scale,
                    .i0 = first.i,
                    .i1 = 0xffffffffu,
                    .i2 = 0xffffffffu,
                    .i3 = 0xffffffffu,
                    .u_pack = 0u,
                };
            }
            pos++;
            continue;
        }

        uint32_t j = first.j;
        uint32_t targets[5] = {first.i, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
        uint32_t coeffs[4] = {first.u, 1u, 1u, 1u};

        uint32_t cnt = 1;
        if (first.i != j) {
            while (cnt < max_fanout && pos + cnt < rem) {
                uint32_t cand_i = rev ? (end - 1u - (pos + cnt)) : (beg + pos + cnt);
                sched_op cand = ops[cand_i];
                if (cand.u == 0 || cand.j != j || cand.i == j)
                    break;
                bool dup = false;
                for (uint32_t m = 0; m < cnt; m++) {
                    if (targets[m] == cand.i) {
                        dup = true;
                        break;
                    }
                }
                if (dup)
                    break;
                targets[cnt] = cand.i;
                if (cnt < 4)
                    coeffs[cnt] = cand.u;
                cnt++;
            }
        }

        out[n++] = (vxfec_gpu_op4){
            .j = j,
            .i0 = targets[0],
            .i1 = targets[1],
            .i2 = targets[2],
            .i3 = targets[3],
            .u_pack = is_pure_gf2 ? targets[4] : (coeffs[0] | (coeffs[1] << 8) | (coeffs[2] << 16) | (coeffs[3] << 24)),
        };
        pos += cnt;
    }
    return n;
}

static bool detect_and_pack_hdpc(const sched_op *ops, uint32_t cp0, uint32_t cp1, uint32_t rows, vxfec_gpu_op4 *out, size_t *io_n,
                                 uint32_t *out_hbeg, uint32_t *out_hend)
{
    uint32_t raw_hbeg = cp1;
    for (uint32_t idx = cp0; idx < cp1; idx++) {
        if (ops[idx].u == 0)
            return false;
        if (ops[idx].u > 1) {
            raw_hbeg = idx;
            break;
        }
    }
    if (raw_hbeg >= cp1 || cp1 - raw_hbeg < 64u || rows == 0)
        return false;

    size_t words = ((size_t)rows + 31u) / 32u;
    uint32_t *bits = (uint32_t *)calloc(words * 2u, sizeof(uint32_t));
    if (!bits)
        return false;
    uint32_t *in_hr = bits;
    uint32_t *seen_j = bits + words;

    uint32_t hr[16] = {0};
    uint32_t hcnt = 0;
    uint32_t raw_hend = raw_hbeg;

    for (uint32_t idx = raw_hbeg; idx < cp1; idx++) {
        sched_op op = ops[idx];
        if (op.u == 0 || op.i >= rows || op.j >= rows || op.i == op.j)
            break;
        if ((in_hr[op.j >> 5] >> (op.j & 31u)) & 1u)
            break;
        if (!((in_hr[op.i >> 5] >> (op.i & 31u)) & 1u)) {
            if (hcnt >= 16u || ((seen_j[op.i >> 5] >> (op.i & 31u)) & 1u))
                break;
            hr[hcnt++] = op.i;
            in_hr[op.i >> 5] |= (1u << (op.i & 31u));
        }
        seen_j[op.j >> 5] |= (1u << (op.j & 31u));
        raw_hend = idx + 1u;
    }
    free(bits);

    if (hcnt == 0 || raw_hend - raw_hbeg < 64u)
        return false;

    uint8_t *slot = (uint8_t *)malloc(rows);
    if (!slot)
        return false;
    memset(slot, 0xff, rows);
    for (uint32_t m = 0; m < hcnt; m++)
        slot[hr[m]] = (uint8_t)m;
    for (uint32_t m = hcnt; m < 16u; m++)
        hr[m] = hr[0];

    size_t n = *io_n;
    n += pack_ops_span(ops, cp0, raw_hbeg, false, false, out + n);

    uint32_t hdpc_first = (uint32_t)n;
    for (uint32_t c_base = 0; c_base < hcnt; c_base += 8u) {
        uint32_t c_len = (hcnt - c_base > 8u) ? 8u : (hcnt - c_base);
        uint32_t chr[8];
        for (uint32_t m = 0; m < 8u; m++)
            chr[m] = (m < c_len) ? hr[c_base + m] : hr[c_base];

        uint32_t hdr_pos = (uint32_t)n;
        out[n++] = (vxfec_gpu_op4){.j = chr[0], .i0 = chr[1], .i1 = chr[2], .i2 = chr[3], .i3 = chr[4], .u_pack = chr[5]};
        out[n++] = (vxfec_gpu_op4){.j = chr[6], .i0 = chr[7], .i1 = c_len, .i2 = 0u, .i3 = 0u, .u_pack = 0u};

        uint32_t idx = raw_hbeg;
        while (idx < raw_hend) {
            uint32_t j = ops[idx].j;
            uint8_t coeff[8] = {0};
            while (idx < raw_hend && ops[idx].j == j) {
                uint8_t sl = slot[ops[idx].i];
                if (sl >= c_base && sl < c_base + c_len)
                    coeff[sl - c_base] ^= ops[idx].u;
                idx++;
            }

            uint32_t lc[8];
            uint32_t mask = 0;
            for (uint32_t m = 0; m < 8u; m++) {
                if (m < c_len && coeff[m] != 0) {
                    lc[m] = (uint32_t)OCT_LOG[coeff[m]];
                    mask |= (1u << m);
                    if (lc[m] != 0u)
                        mask |= 0x100u;
                } else {
                    lc[m] = 0xffu;
                }
            }
            if (mask != 0) {
                out[n++] = (vxfec_gpu_op4){
                    .j = j,
                    .i0 = lc[0] | (lc[1] << 8) | (lc[2] << 16) | (lc[3] << 24),
                    .i1 = lc[4] | (lc[5] << 8) | (lc[6] << 16) | (lc[7] << 24),
                    .i2 = 0u,
                    .i3 = 0u,
                    .u_pack = mask,
                };
            }
        }

        if (n == (size_t)hdr_pos + 2u) {
            n = hdr_pos;
        } else {
            out[hdr_pos + 1u].i2 = (uint32_t)n;
        }
    }
    free(slot);

    *out_hbeg = hdpc_first;
    *out_hend = (uint32_t)n;

    n += pack_ops_span(ops, raw_hend, cp1, false, false, out + n);
    *io_n = n;
    return true;
}

struct vxfec_rq_cpu_sched {
    vxfec_gpu_op4 *packed;
    size_t upload_bytes;
    sched_op *raw_ops;
    uint32_t n_raw_ops;
    uint32_t raw_cp0;
    uint32_t raw_cp1;
    uint32_t rows;
    uint32_t cp0_op4;
    uint32_t cp1_op4;
    uint32_t fwd_op4;
    uint32_t total_op4;
    uint32_t hdpc_beg_op4;
    uint32_t hdpc_end_op4;
    bool pure_gf2;
};

vxfec_rq_cpu_sched *vxfec_rq_cpu_sched_pack(nanorq_core *rq, const schedule *S)
{
    if (!rq || !S || !S->ops.a || S->cpidx != 2 || S->overflowed || S->ops.n > UINT32_MAX / 2u)
        return NULL;
    nanorq_core_init();

    bool pure_gf2 = vxfec_rq__sched_is_pure_gf2(S);
    uint32_t rows = nanorq_core_get_pc_rows(rq);
    size_t max_packed = S->ops.n + S->cp[0] + 8u;
    size_t packed_alloc = 0;
    if (!vxfec_vk__mul3(max_packed, sizeof(vxfec_gpu_op4), 1u, &packed_alloc))
        return NULL;

    vxfec_rq_cpu_sched *cs = (vxfec_rq_cpu_sched *)calloc(1, sizeof(*cs));
    vxfec_gpu_op4 *packed = cs ? (vxfec_gpu_op4 *)calloc(1, packed_alloc) : NULL;
    if (!cs || !packed) {
        free(packed);
        free(cs);
        return NULL;
    }

    size_t c0 = pack_ops_span(S->ops.a, 0, S->cp[0], false, pure_gf2, packed);
    size_t c1 = c0;
    uint32_t hbeg = 0, hend = 0;
    if (!pure_gf2 && detect_and_pack_hdpc(S->ops.a, S->cp[0], S->cp[1], rows, packed, &c1, &hbeg, &hend)) {
        /* packed [cp0, cp1) with register-accumulated HDPC span */
    } else {
        c1 += pack_ops_span(S->ops.a, S->cp[0], S->cp[1], false, pure_gf2, packed + c1);
        hbeg = (uint32_t)c1;
        hend = (uint32_t)c1;
    }
    size_t cfwd = c1 + pack_ops_span(S->ops.a, S->cp[1], (uint32_t)S->ops.n, false, pure_gf2, packed + c1);
    size_t ctot = cfwd + pack_ops_span(S->ops.a, 0, S->cp[0], true, pure_gf2, packed + cfwd);

    if (S->ops.n > 0) {
        cs->raw_ops = (sched_op *)malloc(S->ops.n * sizeof(sched_op));
        if (!cs->raw_ops) {
            free(packed);
            free(cs);
            return NULL;
        }
        memcpy(cs->raw_ops, S->ops.a, S->ops.n * sizeof(sched_op));
    }

    cs->packed = packed;
    cs->upload_bytes = (ctot ? ctot : 1u) * sizeof(vxfec_gpu_op4);
    cs->n_raw_ops = (uint32_t)S->ops.n;
    cs->raw_cp0 = S->cp[0];
    cs->raw_cp1 = S->cp[1];
    cs->rows = rows;
    cs->cp0_op4 = (uint32_t)c0;
    cs->cp1_op4 = (uint32_t)c1;
    cs->fwd_op4 = (uint32_t)cfwd;
    cs->total_op4 = (uint32_t)ctot;
    cs->hdpc_beg_op4 = hbeg;
    cs->hdpc_end_op4 = hend;
    cs->pure_gf2 = pure_gf2;
    return cs;
}

void vxfec_rq_cpu_sched_free(vxfec_rq_cpu_sched *cs)
{
    if (!cs)
        return;
    free(cs->packed);
    free(cs->raw_ops);
    free(cs);
}

static vxfec_rq_sched *rq_sched_upload_cpu(vxfec_vk_ctx *ctx, vxfec_rq_cpu_sched **cs_inout, bool staged)
{
    if (!ctx || !cs_inout || !*cs_inout)
        return NULL;
    vxfec_rq_cpu_sched *cs = *cs_inout;
    vxfec_vk_buf *ops_buf;
    if (staged) {
        ops_buf = vxfec_vk_buf_new(ctx, cs->upload_bytes, true);
        if (ops_buf)
            memcpy(vxfec_vk_buf_staging_ptr(ops_buf), cs->packed, cs->upload_bytes);
    } else {
        ops_buf = vxfec_vk__upload_const_buf(ctx, cs->packed, cs->upload_bytes);
    }
    if (!ops_buf)
        return NULL;

    vxfec_rq_sched *gs = (vxfec_rq_sched *)calloc(1, sizeof(*gs));
    if (!gs) {
        vxfec_vk_buf_free(ctx, ops_buf);
        return NULL;
    }
    gs->ops_buf = ops_buf;
    gs->raw_ops = cs->raw_ops;
    cs->raw_ops = NULL;
    gs->n_raw_ops = cs->n_raw_ops;
    gs->raw_cp0 = cs->raw_cp0;
    gs->raw_cp1 = cs->raw_cp1;
    gs->rows = cs->rows;
    gs->cp0_op4 = cs->cp0_op4;
    gs->cp1_op4 = cs->cp1_op4;
    gs->fwd_op4 = cs->fwd_op4;
    gs->total_op4 = cs->total_op4;
    gs->hdpc_beg_op4 = cs->hdpc_beg_op4;
    gs->hdpc_end_op4 = cs->hdpc_end_op4;
    gs->pure_gf2 = cs->pure_gf2;
    gs->ops_uploaded = !staged || ops_buf->zero_copy || !ops_buf->stage_buf;
    vxfec_rq_cpu_sched_free(cs);
    *cs_inout = NULL;
    return gs;
}

vxfec_rq_sched *vxfec_rq_sched_from_cpu_staged(vxfec_vk_ctx *ctx, vxfec_rq_cpu_sched **cs_inout)
{
    return rq_sched_upload_cpu(ctx, cs_inout, true);
}

static vxfec_rq_sched *rq_sched_new(vxfec_vk_ctx *ctx, nanorq_core *rq, const schedule *S, bool staged)
{
    if (!ctx)
        return NULL;
    vxfec_rq_cpu_sched *cs = vxfec_rq_cpu_sched_pack(rq, S);
    if (!cs)
        return NULL;
    vxfec_rq_sched *gs = rq_sched_upload_cpu(ctx, &cs, staged);
    vxfec_rq_cpu_sched_free(cs);
    return gs;
}

vxfec_rq_sched *vxfec_rq_sched_new(vxfec_vk_ctx *ctx, nanorq_core *rq, const schedule *S)
{
    return rq_sched_new(ctx, rq, S, false);
}

vxfec_rq_sched *vxfec_rq_sched_new_staged(vxfec_vk_ctx *ctx, nanorq_core *rq, const schedule *S)
{
    return rq_sched_new(ctx, rq, S, true);
}

void vxfec_rq_sched_free(vxfec_vk_ctx *ctx, vxfec_rq_sched *gs)
{
    if (!gs)
        return;
    if (ctx)
        vxfec_vk__retire_or_free_buf(ctx, gs->ops_buf);
    free(gs->raw_ops);
    free(gs);
}

bool vxfec_rq_sched_is_pure_gf2(const vxfec_rq_sched *gs)
{
    return gs ? gs->pure_gf2 : false;
}

void vxfec_rq__adjoint_gf2(nanorq_core *rq, const sched_op *ops, uint32_t n_ops, uint32_t cp0, uint32_t cp1, uint32_t rows,
                           const uint32_t *esis, uint32_t num_esis, uint64_t *Y, size_t wpr)
{
    memset(Y, 0, (size_t)rows * wpr * sizeof(uint64_t));
    bool has_pv = (rq->W.d.a != NULL && rq->W.ci.a != NULL);
    for (uint32_t e = 0; e < num_esis; e++) {
        uint32_t mix[GENC_MAX];
        uint32_t deg = nanorq_core_get_packet_mix(rq, esis[e], mix, GENC_MAX);
        size_t ew = e >> 6;
        uint64_t bit = 1ULL << (e & 63u);
        for (uint32_t m = 0; m < deg; m++) {
            uint32_t r = has_pv ? uv_A(rq->W.d, uv_A(rq->W.ci, mix[m])) : mix[m];
            if (r < rows)
                Y[(size_t)r * wpr + ew] ^= bit;
        }
    }

    if (wpr == 1u) {
        for (uint32_t idx = cp0; idx-- > 0;) {
            sched_op op = ops[idx];
            if (op.u != 0 && op.i < rows && op.j < rows) {
                if (op.i == op.j)
                    Y[op.j] = 0;
                else
                    Y[op.j] ^= Y[op.i];
            }
        }
        for (uint32_t idx = n_ops; idx-- > cp1;) {
            sched_op op = ops[idx];
            if (op.u != 0 && op.i < rows && op.j < rows) {
                if (op.i == op.j)
                    Y[op.j] = 0;
                else
                    Y[op.j] ^= Y[op.i];
            }
        }
        for (uint32_t idx = 0; idx < cp0; idx++) {
            sched_op op = ops[idx];
            if (op.u != 0 && op.i < rows && op.j < rows) {
                if (op.i == op.j)
                    Y[op.j] = 0;
                else
                    Y[op.j] ^= Y[op.i];
            }
        }
        for (uint32_t idx = cp1; idx-- > 0;) {
            sched_op op = ops[idx];
            if (op.u != 0 && op.i < rows && op.j < rows) {
                if (op.i == op.j)
                    Y[op.j] = 0;
                else
                    Y[op.j] ^= Y[op.i];
            }
        }
    } else {
        for (uint32_t pass = 0; pass < 4u; pass++) {
            uint32_t len = (pass == 0u || pass == 2u) ? cp0 : ((pass == 1u) ? (n_ops - cp1) : cp1);
            for (uint32_t step = 0; step < len; step++) {
                uint32_t idx = (pass == 0u) ? (cp0 - 1u - step)
                                            : ((pass == 1u) ? (n_ops - 1u - step) : ((pass == 2u) ? step : (cp1 - 1u - step)));
                sched_op op = ops[idx];
                if (op.u != 0 && op.i < rows && op.j < rows) {
                    uint64_t *dst = Y + (size_t)op.j * wpr;
                    const uint64_t *src = Y + (size_t)op.i * wpr;
                    if (op.i == op.j) {
                        memset(dst, 0, wpr * sizeof(uint64_t));
                    } else {
                        for (size_t w = 0; w < wpr; w++)
                            dst[w] ^= src[w];
                    }
                }
            }
        }
    }
}

static inline void adj_step_gf256(uint8_t *Y, size_t m_stride, uint32_t rows, sched_op op)
{
    if (op.i >= rows)
        return;
    if (op.u == 0) {
        uint8_t scale = (uint8_t)(op.j & 0xffu);
        if (scale != 1u)
            nanorq_oblas.scal(Y + (size_t)op.i * m_stride, scale, (unsigned)m_stride);
    } else if (op.i == op.j) {
        uint8_t comb = (uint8_t)(1u ^ op.u);
        if (comb == 0u)
            memset(Y + (size_t)op.i * m_stride, 0, m_stride);
        else if (comb != 1u)
            nanorq_oblas.scal(Y + (size_t)op.i * m_stride, comb, (unsigned)m_stride);
    } else if (op.j < rows) {
        nanorq_oblas.axpy(Y + (size_t)op.j * m_stride, Y + (size_t)op.i * m_stride, op.u, (unsigned)m_stride);
    }
}

void vxfec_rq__adjoint_gf256(nanorq_core *rq, const sched_op *ops, uint32_t n_ops, uint32_t cp0, uint32_t cp1, uint32_t rows,
                             const uint32_t *esis, uint32_t num_esis, uint8_t *Y, size_t m_stride)
{
    nanorq_core_init();
    memset(Y, 0, (size_t)rows * m_stride);
    bool has_pv = (rq->W.d.a != NULL && rq->W.ci.a != NULL);
    for (uint32_t e = 0; e < num_esis; e++) {
        uint32_t mix[GENC_MAX];
        uint32_t deg = nanorq_core_get_packet_mix(rq, esis[e], mix, GENC_MAX);
        for (uint32_t m = 0; m < deg; m++) {
            uint32_t r = has_pv ? uv_A(rq->W.d, uv_A(rq->W.ci, mix[m])) : mix[m];
            if (r < rows)
                Y[(size_t)r * m_stride + e] ^= 1u;
        }
    }

    for (uint32_t idx = cp0; idx-- > 0;)
        adj_step_gf256(Y, m_stride, rows, ops[idx]);
    for (uint32_t idx = n_ops; idx-- > cp1;)
        adj_step_gf256(Y, m_stride, rows, ops[idx]);
    for (uint32_t idx = 0; idx < cp0; idx++)
        adj_step_gf256(Y, m_stride, rows, ops[idx]);
    for (uint32_t idx = cp1; idx-- > 0;)
        adj_step_gf256(Y, m_stride, rows, ops[idx]);
}

static bool compile_direct_gf2_tab(VkDeviceSize max_ssbo_range, nanorq_core *rq, const sched_op *ops, uint32_t n_ops, uint32_t cp0,
                                   uint32_t cp1, uint32_t rows, const uint32_t *esis, uint32_t num_esis, uint32_t **out_words,
                                   size_t *out_bytes, uint32_t *out_tiles, uint32_t *out_tile_e)
{
    if (rows == 0 || num_esis == 0 || num_esis > UINT32_MAX - 32u)
        return false;
    uint32_t tile_e = (num_esis <= 16u) ? 16u : 32u;
    uint32_t n_tiles = (num_esis + tile_e - 1u) / tile_e;
    size_t wpr = ((size_t)num_esis + 63u) / 64u;
    size_t y_words = 0, max_uvec2 = 0, alloc_bytes = 0;
    if (!vxfec_vk__mul3(rows, wpr, 1u, &y_words) || !vxfec_vk__mul3(n_tiles, (size_t)rows + 1u, 1u, &max_uvec2))
        return false;
    if (max_uvec2 == 0)
        max_uvec2 = 1;
    if (max_uvec2 > UINT32_MAX || !vxfec_vk__mul3(max_uvec2, 8u, 1u, &alloc_bytes) ||
        (max_ssbo_range > 0 && alloc_bytes > (size_t)max_ssbo_range))
        return false;

    uint64_t stack_y[512];
    uint64_t *Y = (y_words <= 512u) ? stack_y : (uint64_t *)calloc(y_words, sizeof(uint64_t));
    if (!Y)
        return false;

    vxfec_rq__adjoint_gf2(rq, ops, n_ops, cp0, cp1, rows, esis, num_esis, Y, wpr);

    uint32_t *tab = (uint32_t *)malloc(alloc_bytes);
    if (!tab) {
        if (Y != stack_y)
            free(Y);
        return false;
    }

    uint64_t tile_mask = (tile_e == 16u) ? 0xffffULL : 0xffffffffULL;
    uint32_t cursor = n_tiles;
    for (uint32_t t = 0; t < n_tiles; t++) {
        uint32_t beg = cursor;
        size_t bit_pos = (size_t)t * tile_e;
        size_t ew = bit_pos >> 6;
        uint32_t shift = (uint32_t)(bit_pos & 63u);
        for (uint32_t r = 0; r < rows; r++) {
            uint32_t mask = (uint32_t)((Y[(size_t)r * wpr + ew] >> shift) & tile_mask);
            if (mask != 0) {
                tab[(size_t)cursor * 2u] = r;
                tab[(size_t)cursor * 2u + 1u] = mask;
                cursor++;
            }
        }
        tab[(size_t)t * 2u] = beg;
        tab[(size_t)t * 2u + 1u] = cursor;
    }

    if (Y != stack_y)
        free(Y);

    *out_words = tab;
    *out_bytes = (size_t)(cursor ? cursor : 1u) * 8u;
    *out_tiles = n_tiles;
    *out_tile_e = tile_e;
    return true;
}

static bool compile_direct_gf256_tab(VkDeviceSize max_ssbo_range, nanorq_core *rq, const sched_op *ops, uint32_t n_ops,
                                     uint32_t cp0, uint32_t cp1, uint32_t rows, const uint32_t *esis, uint32_t num_esis,
                                     uint32_t **out_words, size_t *out_bytes, uint32_t *out_tiles, uint32_t *out_tile_e)
{
    if (rows == 0 || num_esis == 0 || num_esis > UINT32_MAX - 64u)
        return false;
    nanorq_core_init();
    size_t align = get_align_size();
    size_t m_stride = ((size_t)num_esis + align - 1u) & ~(align - 1u);
    if (m_stride == 0)
        m_stride = align;

    size_t y_bytes = 0;
    if (!vxfec_vk__mul3(rows, m_stride, 1u, &y_bytes))
        return false;

    uint8_t *Y = (uint8_t *)calloc(1, y_bytes ? y_bytes : 32u);
    if (!Y)
        return false;

    vxfec_rq__adjoint_gf256(rq, ops, n_ops, cp0, cp1, rows, esis, num_esis, Y, m_stride);

    bool ok =
        vxfec_vk__emit_direct_gf256_tab(max_ssbo_range, Y, m_stride, rows, num_esis, out_words, out_bytes, out_tiles, out_tile_e);
    free(Y);
    return ok;
}

vxfec_rq_direct *vxfec_rq_direct_new(vxfec_vk_ctx *ctx, nanorq_core *rq, const schedule *S, const uint32_t *esis, uint32_t num_esis)
{
    if (!ctx || !rq || !S || !S->ops.a || S->cpidx != 2 || S->overflowed || S->ops.n > UINT32_MAX / 2u || !esis || num_esis == 0)
        return NULL;
    nanorq_core_init();

    bool pure_gf2 = vxfec_rq__sched_is_pure_gf2(S);
    uint32_t rows = nanorq_core_get_pc_rows(rq);
    uint32_t *tab = NULL;
    size_t tab_bytes = 0;
    uint32_t n_tiles = 0;
    uint32_t tile_e = 8u;
    bool ok = pure_gf2 ? compile_direct_gf2_tab(ctx->max_ssbo_range, rq, S->ops.a, (uint32_t)S->ops.n, S->cp[0], S->cp[1], rows,
                                                esis, num_esis, &tab, &tab_bytes, &n_tiles, &tile_e)
                       : compile_direct_gf256_tab(ctx->max_ssbo_range, rq, S->ops.a, (uint32_t)S->ops.n, S->cp[0], S->cp[1], rows,
                                                  esis, num_esis, &tab, &tab_bytes, &n_tiles, &tile_e);
    if (!ok) {
        free(tab);
        return NULL;
    }

    vxfec_vk_buf *tab_buf = vxfec_vk__upload_const_buf(ctx, tab, tab_bytes);
    free(tab);
    if (!tab_buf)
        return NULL;

    vxfec_rq_direct *gd = (vxfec_rq_direct *)calloc(1, sizeof(*gd));
    if (!gd) {
        vxfec_vk_buf_free(ctx, tab_buf);
        return NULL;
    }
    gd->desc.tab_buf = tab_buf;
    gd->desc.tab_bytes = tab_bytes;
    gd->desc.rows = rows;
    gd->desc.num_esis = num_esis;
    gd->desc.num_tiles = n_tiles;
    gd->desc.tile_e = tile_e;
    gd->desc.pure_gf2 = pure_gf2;
    return gd;
}

void vxfec_rq_direct_free(vxfec_vk_ctx *ctx, vxfec_rq_direct *gd)
{
    if (!gd)
        return;
    if (ctx)
        vxfec_vk__retire_or_free_buf(ctx, gd->desc.tab_buf);
    free(gd);
}

bool vxfec_rq_direct_is_pure_gf2(const vxfec_rq_direct *gd)
{
    return gd ? gd->desc.pure_gf2 : false;
}

static bool select_run_plan(const vxfec_vk_ctx *ctx, const vxfec_rq_sched *gs, uint32_t stride, uint32_t num_blocks,
                            VkPipeline *out_pipe, uint32_t *out_stride_units, uint32_t *out_wg_x)
{
    bool can_vec4 = ((stride & 15u) == 0);
    bool use_vec4 = false;
    uint32_t min_wgs = ctx->is_discrete ? 256u : 32u;

    if (can_vec4) {
        uint32_t wg_x_vec4 = ((stride / 16u) + 31u) / 32u;
        if ((uint64_t)wg_x_vec4 * num_blocks >= min_wgs)
            use_vec4 = true;
    }

    uint32_t stride_units = use_vec4 ? (stride / 16u) : (stride / 4u);
    uint32_t wg_x = (stride_units + 31u) / 32u;
    if (wg_x == 0 || wg_x > ctx->max_wg_x)
        return false;

    *out_pipe = use_vec4 ? (gs->pure_gf2 ? ctx->pipe_vec4_gf2 : ctx->pipe_vec4_gf256)
                         : (gs->pure_gf2 ? ctx->pipe_vram_gf2 : ctx->pipe_vram_gf256);
    *out_stride_units = stride_units;
    *out_wg_x = wg_x;
    return true;
}

static void record_ops_run_chunk_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, VkDescriptorSet ds_run, const vxfec_rq_sched *gs,
                                     VkPipeline pipe, uint32_t stride_units, uint32_t wg_x, uint32_t block_offset, uint32_t chunk)
{
    vxfec_run_push_consts pc = {
        .rows = gs->rows,
        .stride_units = stride_units,
        .cp0_op4 = gs->cp0_op4,
        .cp1_op4 = gs->cp1_op4,
        .fwd_op4 = gs->fwd_op4,
        .total_op4 = gs->total_op4,
        .block_offset = block_offset,
        .hdpc_beg_op4 = gs->hdpc_beg_op4,
        .hdpc_end_op4 = gs->hdpc_end_op4,
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pl_run, 0, 1, &ds_run, 0, NULL);
    vkCmdPushConstants(cmd, ctx->pl_run, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, wg_x, chunk, 1);
}

static void record_ops_run_dispatches_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, VkDescriptorSet ds_run, const vxfec_rq_sched *gs,
                                          uint32_t stride, uint32_t num_blocks, VkPipeline pipe, uint32_t stride_units,
                                          uint32_t wg_x)
{
    uint32_t tile_blks = vxfec_vk__compute_tile_blocks(ctx, gs->rows, stride, wg_x, num_blocks);
    for (uint32_t b0 = 0; b0 < num_blocks;) {
        uint32_t chunk = (num_blocks - b0 > tile_blks) ? tile_blks : (num_blocks - b0);
        record_ops_run_chunk_cmd(ctx, cmd, ds_run, gs, pipe, stride_units, wg_x, b0, chunk);
        b0 += chunk;
        if (b0 < num_blocks) {
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0,
                                 NULL, 0, NULL);
        }
    }
}

static bool select_mix_plan(const vxfec_vk_ctx *ctx, uint32_t stride, uint32_t num_blocks, uint32_t num_esis, VkPipeline *out_pipe,
                            uint32_t *out_stride_units, uint32_t *out_wg_x)
{
    if (num_esis > UINT32_MAX / 64u)
        return false;

    bool can_vec4 = ((stride & 15u) == 0);
    bool use_vec4 = false;
    uint32_t min_wgs = ctx->is_discrete ? 256u : 32u;

    if (can_vec4) {
        uint32_t wg_x_vec4 = ((stride / 16u) + 31u) / 32u;
        if ((uint64_t)wg_x_vec4 * num_esis * num_blocks >= min_wgs)
            use_vec4 = true;
    }

    uint32_t stride_units = use_vec4 ? (stride / 16u) : (stride / 4u);
    uint32_t wg_x = (stride_units + 31u) / 32u;
    if (wg_x == 0 || wg_x > ctx->max_wg_x)
        return false;

    *out_pipe = use_vec4 ? ctx->pipe_mix_vec4 : ctx->pipe_mix;
    *out_stride_units = stride_units;
    *out_wg_x = wg_x;
    return true;
}

static void fill_mix_table(uint32_t *tab, nanorq_core *rq, const uint32_t *esis, uint32_t num_esis, bool virtual_layout)
{
    for (uint32_t e = 0; e < num_esis; e++) {
        uint32_t mix[GENC_MAX];
        uint32_t deg = nanorq_core_get_packet_mix(rq, esis[e], mix, GENC_MAX);
        if (virtual_layout && rq->W.d.a && rq->W.ci.a) {
            for (uint32_t m = 0; m < deg; m++)
                mix[m] = uv_A(rq->W.d, uv_A(rq->W.ci, mix[m]));
        }
        uint32_t *dst = tab + (size_t)e * 64u;
        dst[0] = deg;
        for (uint32_t m = 0; m < deg; m++)
            dst[1 + m] = mix[m];
        for (uint32_t m = deg; m < 63u; m++)
            dst[1 + m] = 0;
    }
}

static bool should_use_auto_direct(const vxfec_rq_sched *gs, uint32_t num_esis)
{
    if (!gs->raw_ops && gs->n_raw_ops > 0)
        return false;
    if (gs->pure_gf2) {
        uint32_t tile_e = (num_esis <= 16u) ? 16u : 32u;
        uint32_t n_tiles = (num_esis + tile_e - 1u) / tile_e;
        uint64_t direct_ops = (uint64_t)n_tiles * gs->rows + num_esis;
        uint64_t two_stage_ops = 4ULL * ((uint64_t)gs->total_op4 + gs->cp0_op4) + 10ULL * num_esis;
        return (num_esis <= 512u && gs->n_raw_ops <= 250000u && direct_ops < two_stage_ops);
    }
    return (num_esis <= 32u && gs->rows <= 1024u);
}

static bool record_staged_ops_copy(VkCommandBuffer cmd, const vxfec_rq_sched *gs)
{
    if (gs->ops_uploaded)
        return false;
    const vxfec_vk_buf *ops = gs->ops_buf;
    if (!ops->stage_buf)
        return false;
    VkBufferCopy copy = {.size = ops->bytes};
    vkCmdCopyBuffer(cmd, ops->stage_buf, ops->dev_buf, 1, &copy);
    return true;
}

bool vxfec_rq_ops_run_batch(vxfec_vk_ctx *ctx, const vxfec_rq_sched *gs, vxfec_vk_buf *d_vram, uint32_t stride, uint32_t num_blocks,
                            uint64_t *gpu_ns_out)
{
    if (!ctx || !gs || !d_vram || stride == 0 || (stride & 3u) != 0 || num_blocks == 0)
        return false;
    if ((d_vram->view_offset & (size_t)(ctx->ssbo_offset_align - 1u)) != 0)
        return false;

    size_t need_bytes = 0;
    if (!vxfec_vk__mul3(num_blocks, gs->rows, stride, &need_bytes) || need_bytes > d_vram->bytes ||
        need_bytes > ctx->max_ssbo_range || gs->ops_buf->bytes > ctx->max_ssbo_range)
        return false;

    VkPipeline pipe = VK_NULL_HANDLE;
    uint32_t stride_units = 0, wg_x = 0;
    if (!select_run_plan(ctx, gs, stride, num_blocks, &pipe, &stride_units, &wg_x))
        return false;

    if (!vxfec_vk__begin_cmd(ctx))
        return false;
    if (!vxfec_vk__ensure_owned_cmd(ctx, ctx->cmd, d_vram, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)) {
        ctx->pending_owned_cnt = 0;
        return false;
    }

    VkDescriptorBufferInfo dbi[2] = {
        {.buffer = d_vram->dev_buf, .offset = d_vram->view_offset, .range = need_bytes},
        {.buffer = gs->ops_buf->dev_buf, .offset = 0, .range = gs->ops_buf->bytes},
    };
    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ctx->ds_run,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ctx->ds_run,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[1]},
    };
    vkUpdateDescriptorSets(ctx->dev, 2, writes, 0, NULL);

    bool copied_ops = record_staged_ops_copy(ctx->cmd, gs);
    if (copied_ops) {
        VkMemoryBarrier uploaded = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &uploaded, 0,
                             NULL, 0, NULL);
    }

    if (ctx->has_timestamps && gpu_ns_out) {
        vkCmdResetQueryPool(ctx->cmd, ctx->query_pool, 0, 2);
        vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->query_pool, 0);
    }

    record_ops_run_dispatches_cmd(ctx, ctx->cmd, ctx->ds_run, gs, stride, num_blocks, pipe, stride_units, wg_x);

    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &mb, 0, NULL, 0, NULL);

    if (ctx->has_timestamps && gpu_ns_out)
        vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->query_pool, 1);

    if (!vxfec_vk__submit_and_wait(ctx))
        return false;
    if (copied_ops)
        ((vxfec_rq_sched *)gs)->ops_uploaded = true;
    if (gpu_ns_out)
        *gpu_ns_out = vxfec_vk__read_gpu_ns(ctx);
    return true;
}

bool vxfec_rq_ops_mix_batch(vxfec_vk_ctx *ctx, nanorq_core *rq, vxfec_vk_buf *d_vram, uint32_t stride, uint32_t num_blocks,
                            const uint32_t *esis, uint32_t num_esis, vxfec_vk_buf *out_vram, uint64_t *gpu_ns_out)
{
    if (!ctx || !rq || !d_vram || !out_vram || !esis || stride == 0 || (stride & 3u) != 0 || num_blocks == 0 || num_esis == 0)
        return false;
    if (((d_vram->view_offset | out_vram->view_offset) & (size_t)(ctx->ssbo_offset_align - 1u)) != 0)
        return false;

    uint32_t rows = nanorq_core_get_pc_rows(rq);
    size_t d_bytes = 0, out_bytes = 0, tab_bytes = 0;
    if (!vxfec_vk__mul3(num_blocks, rows, stride, &d_bytes) || !vxfec_vk__mul3(num_blocks, num_esis, stride, &out_bytes) ||
        !vxfec_vk__mul3(num_esis, 64u, sizeof(uint32_t), &tab_bytes) || d_bytes > d_vram->bytes || out_bytes > out_vram->bytes ||
        d_bytes > ctx->max_ssbo_range || out_bytes > ctx->max_ssbo_range || tab_bytes > ctx->max_ssbo_range)
        return false;

    VkPipeline mix_pipe = VK_NULL_HANDLE;
    uint32_t stride_units = 0, wg_x = 0;
    if (!select_mix_plan(ctx, stride, num_blocks, num_esis, &mix_pipe, &stride_units, &wg_x))
        return false;

    if (!vxfec_vk__ensure_staging_buf(ctx, &ctx->mix_tab_buf, &ctx->mix_tab_cap, tab_bytes))
        return false;
    if (!vxfec_vk__begin_cmd(ctx))
        return false;
    if (!vxfec_vk__ensure_owned_cmd(ctx, ctx->cmd, d_vram, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT) ||
        !vxfec_vk__ensure_owned_cmd(ctx, ctx->cmd, out_vram, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT)) {
        ctx->pending_owned_cnt = 0;
        return false;
    }

    fill_mix_table((uint32_t *)vxfec_vk_buf_staging_ptr(ctx->mix_tab_buf), rq, esis, num_esis, true);

    VkDescriptorBufferInfo dbi[3] = {
        {.buffer = d_vram->dev_buf, .offset = d_vram->view_offset, .range = d_bytes},
        {.buffer = ctx->mix_tab_buf->dev_buf, .offset = 0, .range = tab_bytes},
        {.buffer = out_vram->dev_buf, .offset = out_vram->view_offset, .range = out_bytes},
    };
    VkWriteDescriptorSet writes[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ctx->ds_mix,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ctx->ds_mix,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ctx->ds_mix,
         .dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi[2]},
    };
    vkUpdateDescriptorSets(ctx->dev, 3, writes, 0, NULL);

    if (!ctx->mix_tab_buf->zero_copy) {
        VkBufferCopy tab_copy = {.srcOffset = 0, .dstOffset = 0, .size = tab_bytes};
        vkCmdCopyBuffer(ctx->cmd, ctx->mix_tab_buf->stage_buf, ctx->mix_tab_buf->dev_buf, 1, &tab_copy);
        VkMemoryBarrier pre_mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &pre_mb, 0, NULL,
                             0, NULL);
    }

    if (ctx->has_timestamps && gpu_ns_out) {
        vkCmdResetQueryPool(ctx->cmd, ctx->query_pool, 0, 2);
        vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->query_pool, 0);
    }

    vxfec_vk__record_mix_like_dispatches_cmd(ctx, ctx->cmd, ctx->ds_mix, rows, stride_units, wg_x, num_esis, num_esis, 0u,
                                             num_blocks, mix_pipe);

    VkMemoryBarrier post_mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &post_mb, 0, NULL, 0, NULL);

    if (ctx->has_timestamps && gpu_ns_out)
        vkCmdWriteTimestamp(ctx->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->query_pool, 1);

    if (!vxfec_vk__submit_and_wait(ctx))
        return false;
    if (gpu_ns_out)
        *gpu_ns_out = vxfec_vk__read_gpu_ns(ctx);
    return true;
}

bool vxfec_rq_direct_batch(vxfec_vk_ctx *ctx, const vxfec_rq_direct *gd, vxfec_vk_buf *d_vram, uint32_t stride, uint32_t num_blocks,
                           vxfec_vk_buf *out_vram, uint64_t *gpu_ns_out)
{
    if (!gd)
        return false;
    return vxfec_vk__direct_pipelined(ctx, &gd->desc, d_vram, stride, num_blocks, out_vram, false, false, gpu_ns_out);
}

uint64_t vxfec_rq_direct_submit(vxfec_vk_ctx *ctx, const vxfec_rq_direct *gd, vxfec_vk_buf *d_buf, uint32_t stride,
                                uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out)
{
    if (!gd)
        return 0;
    return vxfec_vk__direct_submit(ctx, &gd->desc, d_buf, stride, num_blocks, out_buf, upload_d, download_out);
}

uint64_t vxfec_rq_direct_submit_raw(vxfec_vk_ctx *ctx, nanorq_core *rq, const schedule *S, vxfec_vk_buf *d_buf, uint32_t stride,
                                    uint32_t num_blocks, const uint32_t *esis, uint32_t num_esis, vxfec_vk_buf *out_buf,
                                    bool upload_d, bool download_out)
{
    if (!ctx || !rq || !S || !S->ops.a || S->cpidx != 2 || S->overflowed || S->ops.n > UINT32_MAX / 2u || !esis || num_esis == 0)
        return 0;
    nanorq_core_init();

    bool pure_gf2 = vxfec_rq__sched_is_pure_gf2(S);
    uint32_t rows = nanorq_core_get_pc_rows(rq);
    uint32_t *tab = NULL;
    size_t tab_bytes = 0;
    uint32_t n_tiles = 0;
    uint32_t tile_e = 8u;
    bool ok = pure_gf2 ? compile_direct_gf2_tab(ctx->max_ssbo_range, rq, S->ops.a, (uint32_t)S->ops.n, S->cp[0], S->cp[1], rows,
                                                esis, num_esis, &tab, &tab_bytes, &n_tiles, &tile_e)
                       : compile_direct_gf256_tab(ctx->max_ssbo_range, rq, S->ops.a, (uint32_t)S->ops.n, S->cp[0], S->cp[1], rows,
                                                  esis, num_esis, &tab, &tab_bytes, &n_tiles, &tile_e);
    if (!ok) {
        free(tab);
        return 0;
    }

    uint64_t ticket = vxfec_vk__direct_submit_staged_words(ctx, tab, tab_bytes, rows, num_esis, n_tiles, tile_e, pure_gf2, d_buf,
                                                           stride, num_blocks, out_buf, upload_d, download_out);
    free(tab);
    return ticket;
}

static bool record_run_and_mix_cmd(vxfec_vk_ctx *ctx, VkCommandBuffer cmd, VkDescriptorSet ds_run, VkDescriptorSet ds_mix,
                                   vxfec_vk_buf **slot_tab_buf, size_t *slot_tab_cap, bool use_sync_cache, nanorq_core *rq,
                                   const vxfec_rq_sched *gs, vxfec_vk_buf *d_buf, uint32_t stride, uint32_t num_blocks,
                                   const uint32_t *esis, uint32_t num_esis, vxfec_vk_buf *out_buf, bool upload_d, bool download_out,
                                   bool record_ts, bool *out_copied_ops)
{
    if (out_copied_ops)
        *out_copied_ops = false;

    uint32_t rows = gs->rows;
    size_t d_bytes = 0, out_bytes = 0;
    if (!vxfec_vk__mul3(num_blocks, rows, stride, &d_bytes) || !vxfec_vk__mul3(num_blocks, num_esis, stride, &out_bytes) ||
        d_bytes > d_buf->bytes || out_bytes > out_buf->bytes || d_bytes > ctx->max_ssbo_range || out_bytes > ctx->max_ssbo_range)
        return false;
    if (((d_buf->view_offset | out_buf->view_offset) & (size_t)(ctx->ssbo_offset_align - 1u)) != 0)
        return false;
    if ((upload_d && !d_buf->stage_ptr) || (download_out && !out_buf->stage_ptr))
        return false;

    if (should_use_auto_direct(gs, num_esis)) {
        vxfec_vk_buf **dir_buf = use_sync_cache ? &ctx->direct_tab_buf : slot_tab_buf;
        size_t *dir_cap = use_sync_cache ? &ctx->direct_tab_cap : slot_tab_cap;
        uint32_t *tab = NULL;
        size_t tbytes = 0;
        uint32_t n_tiles = 0;
        uint32_t tile_e = 8u;
        bool comp_ok = gs->pure_gf2 ? compile_direct_gf2_tab(ctx->max_ssbo_range, rq, gs->raw_ops, gs->n_raw_ops, gs->raw_cp0,
                                                             gs->raw_cp1, rows, esis, num_esis, &tab, &tbytes, &n_tiles, &tile_e)
                                    : compile_direct_gf256_tab(ctx->max_ssbo_range, rq, gs->raw_ops, gs->n_raw_ops, gs->raw_cp0,
                                                               gs->raw_cp1, rows, esis, num_esis, &tab, &tbytes, &n_tiles, &tile_e);
        bool staged_ok = comp_ok && vxfec_vk__ensure_staging_buf(ctx, dir_buf, dir_cap, tbytes);
        if (staged_ok)
            memcpy(vxfec_vk_buf_staging_ptr(*dir_buf), tab, tbytes);
        free(tab);

        if (staged_ok) {
            vxfec_vk_buf *tbuf = *dir_buf;
            vxfec_vk_direct_desc desc = {
                .tab_buf = tbuf,
                .tab_bytes = tbytes,
                .rows = rows,
                .num_esis = num_esis,
                .num_tiles = n_tiles,
                .tile_e = tile_e,
                .pure_gf2 = gs->pure_gf2,
            };
            if (!tbuf->zero_copy) {
                VkBufferCopy t_copy = {.srcOffset = 0, .dstOffset = 0, .size = tbytes};
                vkCmdCopyBuffer(cmd, tbuf->stage_buf, tbuf->dev_buf, 1, &t_copy);
                VkMemoryBarrier xfer_mb = {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                };
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &xfer_mb, 0,
                                     NULL, 0, NULL);
            }
            if (vxfec_vk__record_direct_cmd(ctx, cmd, ds_mix, &desc, d_buf, stride, num_blocks, out_buf, upload_d, download_out,
                                            record_ts)) {
                return true;
            }
        }
    }

    size_t tab_bytes = 0;
    if (!vxfec_vk__mul3(num_esis, 64u, sizeof(uint32_t), &tab_bytes) || tab_bytes > ctx->max_ssbo_range ||
        gs->ops_buf->bytes > ctx->max_ssbo_range)
        return false;

    VkPipeline run_pipe = VK_NULL_HANDLE, mix_pipe = VK_NULL_HANDLE;
    uint32_t run_units = 0, run_wg_x = 0;
    uint32_t mix_units = 0, mix_wg_x = 0;
    if (!select_run_plan(ctx, gs, stride, num_blocks, &run_pipe, &run_units, &run_wg_x) ||
        !select_mix_plan(ctx, stride, num_blocks, num_esis, &mix_pipe, &mix_units, &mix_wg_x))
        return false;

    vxfec_vk_buf **target_buf = use_sync_cache ? &ctx->mix_tab_buf : slot_tab_buf;
    size_t *target_cap = use_sync_cache ? &ctx->mix_tab_cap : slot_tab_cap;
    if (!vxfec_vk__ensure_staging_buf(ctx, target_buf, target_cap, tab_bytes))
        return false;

    if (!vxfec_vk__ensure_owned_cmd(ctx, cmd, d_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT) ||
        !vxfec_vk__ensure_owned_cmd(ctx, cmd, out_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT))
        return false;

    vxfec_vk_buf *mtbuf = *target_buf;
    fill_mix_table((uint32_t *)vxfec_vk_buf_staging_ptr(mtbuf), rq, esis, num_esis, true);

    VkDescriptorBufferInfo dbi_run[2] = {
        {.buffer = d_buf->dev_buf, .offset = d_buf->view_offset, .range = d_bytes},
        {.buffer = gs->ops_buf->dev_buf, .offset = 0, .range = gs->ops_buf->bytes},
    };
    VkDescriptorBufferInfo dbi_mix[3] = {
        {.buffer = d_buf->dev_buf, .offset = d_buf->view_offset, .range = d_bytes},
        {.buffer = mtbuf->dev_buf, .offset = 0, .range = tab_bytes},
        {.buffer = out_buf->dev_buf, .offset = out_buf->view_offset, .range = out_bytes},
    };
    VkWriteDescriptorSet writes[5] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds_run,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi_run[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds_run,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi_run[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds_mix,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi_mix[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds_mix,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi_mix[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds_mix,
         .dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &dbi_mix[2]},
    };
    vkUpdateDescriptorSets(ctx->dev, 5, writes, 0, NULL);

    if (record_ts && ctx->has_timestamps) {
        vkCmdResetQueryPool(cmd, ctx->query_pool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx->query_pool, 0);
    }

    bool copied_ops = record_staged_ops_copy(cmd, gs);
    bool did_xfer = copied_ops;
    if (upload_d && !d_buf->zero_copy) {
        VkBufferCopy d_copy = {.srcOffset = 0, .dstOffset = d_buf->view_offset, .size = d_bytes};
        vkCmdCopyBuffer(cmd, d_buf->stage_buf, d_buf->dev_buf, 1, &d_copy);
        did_xfer = true;
    }
    if (!mtbuf->zero_copy) {
        VkBufferCopy tab_copy = {.srcOffset = 0, .dstOffset = 0, .size = tab_bytes};
        vkCmdCopyBuffer(cmd, mtbuf->stage_buf, mtbuf->dev_buf, 1, &tab_copy);
        did_xfer = true;
    }
    if (did_xfer) {
        VkMemoryBarrier xfer_to_comp = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &xfer_to_comp, 0,
                             NULL, 0, NULL);
    }

    uint32_t tile_blks = vxfec_vk__compute_tile_blocks(ctx, rows, stride, run_wg_x, num_blocks);
    VkMemoryBarrier run_to_mix = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    for (uint32_t b0 = 0; b0 < num_blocks;) {
        uint32_t chunk = (num_blocks - b0 > tile_blks) ? tile_blks : (num_blocks - b0);
        record_ops_run_chunk_cmd(ctx, cmd, ds_run, gs, run_pipe, run_units, run_wg_x, b0, chunk);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &run_to_mix, 0,
                             NULL, 0, NULL);
        vxfec_vk__record_mix_like_dispatches_cmd(ctx, cmd, ds_mix, rows, mix_units, mix_wg_x, num_esis, num_esis, b0, chunk,
                                                 mix_pipe);
        b0 += chunk;
        if (b0 < num_blocks) {
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0,
                                 NULL, 0, NULL);
        }
    }

    if (download_out && !out_buf->zero_copy) {
        VkMemoryBarrier mix_to_xfer = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mix_to_xfer, 0, NULL, 0,
                             NULL);
        VkBufferCopy out_copy = {.srcOffset = out_buf->view_offset, .dstOffset = 0, .size = out_bytes};
        vkCmdCopyBuffer(cmd, out_buf->dev_buf, out_buf->stage_buf, 1, &out_copy);
        vxfec_vk__record_transfer_to_host_cmd(cmd);
    } else {
        VkMemoryBarrier post_mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
                             1, &post_mb, 0, NULL, 0, NULL);
    }

    if (record_ts && ctx->has_timestamps)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx->query_pool, 1);
    if (out_copied_ops)
        *out_copied_ops = copied_ops;
    return true;
}

bool vxfec_rq_run_and_mix_pipelined(vxfec_vk_ctx *ctx, nanorq_core *rq, const vxfec_rq_sched *gs, vxfec_vk_buf *d_buf,
                                    uint32_t stride, uint32_t num_blocks, const uint32_t *esis, uint32_t num_esis,
                                    vxfec_vk_buf *out_buf, bool upload_d, bool download_out, uint64_t *gpu_ns_out)
{
    if (!ctx || !rq || !gs || !d_buf || !out_buf || !esis || stride == 0 || (stride & 3u) != 0 || num_blocks == 0 || num_esis == 0)
        return false;
    if (!vxfec_vk__begin_cmd(ctx))
        return false;
    bool copied_ops = false;
    if (!record_run_and_mix_cmd(ctx, ctx->cmd, ctx->ds_run, ctx->ds_mix, NULL, NULL, true, rq, gs, d_buf, stride, num_blocks, esis,
                                num_esis, out_buf, upload_d, download_out, gpu_ns_out != NULL, &copied_ops)) {
        ctx->pending_owned_cnt = 0;
        return false;
    }
    if (!vxfec_vk__submit_and_wait(ctx))
        return false;
    if (copied_ops)
        ((vxfec_rq_sched *)gs)->ops_uploaded = true;
    if (gpu_ns_out)
        *gpu_ns_out = vxfec_vk__read_gpu_ns(ctx);
    return true;
}

uint64_t vxfec_rq_run_and_mix_submit(vxfec_vk_ctx *ctx, nanorq_core *rq, const vxfec_rq_sched *gs, vxfec_vk_buf *d_buf,
                                     uint32_t stride, uint32_t num_blocks, const uint32_t *esis, uint32_t num_esis,
                                     vxfec_vk_buf *out_buf, bool upload_d, bool download_out)
{
    if (!ctx || !rq || !gs || !d_buf || !out_buf || !esis || stride == 0 || (stride & 3u) != 0 || num_blocks == 0 || num_esis == 0)
        return 0;

    if (!ctx->has_timeline_sem || !ctx->timeline_sem) {
        return vxfec_rq_run_and_mix_pipelined(ctx, rq, gs, d_buf, stride, num_blocks, esis, num_esis, out_buf, upload_d,
                                              download_out, NULL)
                   ? 1u
                   : 0u;
    }

    uint32_t slot = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!vxfec_vk__acquire_async_slot(ctx, &slot, &cmd))
        return 0;

    bool copied_ops = false;
    if (!record_run_and_mix_cmd(ctx, cmd, ctx->async_ds_run[slot], ctx->async_ds_mix[slot], &ctx->async_tab_bufs[slot],
                                &ctx->async_tab_caps[slot], false, rq, gs, d_buf, stride, num_blocks, esis, num_esis, out_buf,
                                upload_d, download_out, false, &copied_ops)) {
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }
    uint64_t ticket = vxfec_vk__submit_async_slot(ctx, slot, cmd);
    if (ticket != 0 && copied_ops)
        ((vxfec_rq_sched *)gs)->ops_uploaded = true;
    return ticket;
}
