#include "vxfec_rs_cauchy.h"
#include "nanorq_core.h"
#include "util.h"
#include "vk_internal.h"

#include <stdlib.h>
#include <string.h>

struct vxfec_rs8_plan {
    uint32_t K;
    uint32_t M;
    uint32_t num_out;
    size_t m_stride;
    uint8_t *Y; /* K x m_stride: Y[r * m_stride + out_idx] */
    bool has_gpu;
    vxfec_vk_direct_desc gpu;
};

static void rs8_ensure_oblas(void)
{
    nanorq_core_init();
}

static inline uint8_t gf256_inv_nz(uint8_t a)
{
    return GF2_8_EXP[255u - (uint32_t)GF2_8_LOG[a]];
}

static vxfec_rs8_plan *finalize_rs8_plan(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, uint32_t num_out, size_t m_stride, uint8_t *Y)
{
    vxfec_rs8_plan *plan = (vxfec_rs8_plan *)calloc(1, sizeof(*plan));
    if (!plan) {
        free(Y);
        return NULL;
    }
    plan->K = K;
    plan->M = M;
    plan->num_out = num_out;
    plan->m_stride = m_stride;
    plan->Y = Y;

    if (ctx) {
        uint32_t *tab = NULL;
        size_t tab_bytes = 0;
        uint32_t n_tiles = 0;
        uint32_t tile_e = 8u;
        if (!vxfec_vk__emit_direct_gf256_tab(ctx->max_ssbo_range, Y, m_stride, K, num_out, &tab, &tab_bytes, &n_tiles, &tile_e)) {
            free(tab);
            free(Y);
            free(plan);
            return NULL;
        }
        vxfec_vk_buf *tab_buf = vxfec_vk__upload_const_buf(ctx, tab, tab_bytes);
        free(tab);
        if (!tab_buf) {
            free(Y);
            free(plan);
            return NULL;
        }
        plan->has_gpu = true;
        plan->gpu.tab_buf = tab_buf;
        plan->gpu.tab_bytes = tab_bytes;
        plan->gpu.rows = K;
        plan->gpu.num_esis = num_out;
        plan->gpu.num_tiles = n_tiles;
        plan->gpu.tile_e = tile_e;
        plan->gpu.pure_gf2 = false;
    }
    return plan;
}

bool vxfec_rs8__encode_matrix(uint32_t K, uint32_t M, const uint8_t *parity_m, uint32_t num_parity, uint8_t *Y, size_t m_stride)
{
    if (K == 0 || K > 255u || M == 0 || M > 256u - K || num_parity == 0 || num_parity > M || !Y || m_stride < num_parity)
        return false;
    rs8_ensure_oblas();

    if (parity_m) {
        for (uint32_t p = 0; p < num_parity; p++) {
            if ((uint32_t)parity_m[p] >= M)
                return false;
        }
    }

    memset(Y, 0, (size_t)K * m_stride);
    /* nanors Cauchy node assignment: y_k = M + k, x_m = m */
    for (uint32_t k = 0; k < K; k++) {
        uint8_t yk = (uint8_t)(M + k);
        uint8_t *row = Y + (size_t)k * m_stride;
        for (uint32_t p = 0; p < num_parity; p++) {
            uint8_t xm = parity_m ? parity_m[p] : (uint8_t)p;
            row[p] = gf256_inv_nz((uint8_t)(yk ^ xm));
        }
    }
    return true;
}

vxfec_rs8_plan *vxfec_rs8_encode_plan_new(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, const uint8_t *parity_m, uint32_t num_parity)
{
    if (K == 0 || K > 255u || M == 0 || M > 256u - K || num_parity == 0 || num_parity > M)
        return NULL;

    size_t m_stride = ((size_t)num_parity + 31u) & ~(size_t)31u;
    uint8_t *Y = (uint8_t *)calloc((size_t)K * m_stride, 1u);
    if (!Y)
        return NULL;

    if (!vxfec_rs8__encode_matrix(K, M, parity_m, num_parity, Y, m_stride)) {
        free(Y);
        return NULL;
    }

    return finalize_rs8_plan(ctx, K, M, num_parity, m_stride, Y);
}

bool vxfec_rs8__recovery_matrix(uint32_t K, uint32_t M, const uint8_t *erased_k, const uint8_t *parity_m, uint32_t e, uint8_t *Y,
                                size_t m_stride)
{
    if (K == 0 || K > 255u || M == 0 || M > 256u - K || !erased_k || !parity_m || e == 0 || e > K || e > M || !Y || m_stride < e)
        return false;
    rs8_ensure_oblas();
    memset(Y, 0, (size_t)K * m_stride);

    uint8_t seen_k[32] = {0};
    uint8_t seen_m[32] = {0};
    uint8_t x[256], y[256];

    for (uint32_t i = 0; i < e; i++) {
        uint8_t k = erased_k[i];
        uint8_t m = parity_m[i];
        if ((uint32_t)k >= K || (uint32_t)m >= M)
            return false;
        if ((seen_k[k >> 3] >> (k & 7u)) & 1u)
            return false;
        if ((seen_m[m >> 3] >> (m & 7u)) & 1u)
            return false;
        seen_k[k >> 3] |= (uint8_t)(1u << (k & 7u));
        seen_m[m >> 3] |= (uint8_t)(1u << (m & 7u));
        y[i] = (uint8_t)(M + (uint32_t)k);
        x[i] = m;
    }

    /* closed-form O(e^2) Cauchy inverse: (C_E^-1)[b, a] = (u[a] * v[b]) / (x[a] ^ y[b]) */
    uint32_t lu[256], lv[256];
    for (uint32_t a = 0; a < e; a++) {
        uint32_t nlog = 0u, dlog = 0u;
        for (uint32_t j = 0; j < e; j++) {
            nlog += (uint32_t)GF2_8_LOG[x[a] ^ y[j]];
            if (j != a)
                dlog += (uint32_t)GF2_8_LOG[x[a] ^ x[j]];
        }
        lu[a] = (nlog + 255u * e - dlog) % 255u;
    }
    for (uint32_t b = 0; b < e; b++) {
        uint32_t nlog = 0u, dlog = 0u;
        for (uint32_t j = 0; j < e; j++) {
            nlog += (uint32_t)GF2_8_LOG[y[b] ^ x[j]];
            if (j != b)
                dlog += (uint32_t)GF2_8_LOG[y[b] ^ y[j]];
        }
        lv[b] = (nlog + 255u * e - dlog) % 255u;
    }

    /* place (C_E^-1)[b, a] at compact slot r = erased_k[a] */
    for (uint32_t a = 0; a < e; a++) {
        uint8_t *dst_row = Y + (size_t)erased_k[a] * m_stride;
        uint32_t lua = lu[a];
        for (uint32_t b = 0; b < e; b++) {
            uint32_t uv = lua + lv[b];
            if (uv >= 255u)
                uv -= 255u;
            dst_row[b] = GF2_8_EXP[uv + 255u - (uint32_t)GF2_8_LOG[x[a] ^ y[b]]];
        }
    }

    /* closed-form O(K * e) Lagrange residue for surviving systematic slots s not in E */
    for (uint32_t s = 0; s < K; s++) {
        if ((seen_k[s >> 3] >> (s & 7u)) & 1u)
            continue;
        uint8_t ys = (uint8_t)(M + s);
        uint32_t nlog = 0u, dlog = 0u;
        for (uint32_t j = 0; j < e; j++) {
            nlog += (uint32_t)GF2_8_LOG[ys ^ y[j]];
            dlog += (uint32_t)GF2_8_LOG[ys ^ x[j]];
        }
        uint32_t lw = (nlog + 255u * e - dlog) % 255u;
        uint8_t *dst_row = Y + (size_t)s * m_stride;
        for (uint32_t b = 0; b < e; b++) {
            uint32_t wv = lw + lv[b];
            if (wv >= 255u)
                wv -= 255u;
            dst_row[b] = GF2_8_EXP[wv + 255u - (uint32_t)GF2_8_LOG[y[b] ^ ys]];
        }
    }
    return true;
}

vxfec_rs8_plan *vxfec_rs8_decode_plan_new(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, const uint8_t *erased_k,
                                          const uint8_t *parity_m, uint32_t num_erased)
{
    if (K == 0 || K > 255u || M == 0 || M > 256u - K || !erased_k || !parity_m || num_erased == 0 || num_erased > K ||
        num_erased > M)
        return NULL;

    size_t m_stride = ((size_t)num_erased + 31u) & ~(size_t)31u;
    uint8_t *Y = (uint8_t *)calloc((size_t)K * m_stride, 1u);
    if (!Y)
        return NULL;

    if (!vxfec_rs8__recovery_matrix(K, M, erased_k, parity_m, num_erased, Y, m_stride)) {
        free(Y);
        return NULL;
    }
    return finalize_rs8_plan(ctx, K, M, num_erased, m_stride, Y);
}

void vxfec_rs8_plan_free(vxfec_vk_ctx *ctx, vxfec_rs8_plan *plan)
{
    if (!plan)
        return;
    if (plan->has_gpu && ctx)
        vxfec_vk__retire_or_free_buf(ctx, plan->gpu.tab_buf);
    free(plan->Y);
    free(plan);
}

uint32_t vxfec_rs8_plan_k(const vxfec_rs8_plan *plan)
{
    return plan ? plan->K : 0;
}

uint32_t vxfec_rs8_plan_m(const vxfec_rs8_plan *plan)
{
    return plan ? plan->M : 0;
}

uint32_t vxfec_rs8_plan_outputs(const vxfec_rs8_plan *plan)
{
    return plan ? plan->num_out : 0;
}

bool vxfec_rs8_exec_cpu(const vxfec_rs8_plan *plan, const uint8_t *in_rows, uint32_t stride, uint32_t num_blocks, uint8_t *out_rows)
{
    if (!plan || !in_rows || !out_rows || stride == 0 || num_blocks == 0)
        return false;
    rs8_ensure_oblas();

    size_t in_blk_bytes = 0, out_blk_bytes = 0, tot_out = 0;
    if (!vxfec_vk__mul3(plan->K, stride, 1u, &in_blk_bytes) || !vxfec_vk__mul3(plan->num_out, stride, 1u, &out_blk_bytes) ||
        !vxfec_vk__mul3(num_blocks, out_blk_bytes, 1u, &tot_out))
        return false;

    memset(out_rows, 0, tot_out);
    for (uint32_t b = 0; b < num_blocks; b++) {
        const uint8_t *blk_in = in_rows + (size_t)b * in_blk_bytes;
        uint8_t *blk_out = out_rows + (size_t)b * out_blk_bytes;
        for (uint32_t r = 0; r < plan->K; r++) {
            const uint8_t *src = blk_in + (size_t)r * stride;
            const uint8_t *yc = plan->Y + (size_t)r * plan->m_stride;
            for (uint32_t o = 0; o < plan->num_out; o++) {
                uint8_t coeff = yc[o];
                if (coeff != 0)
                    nanorq_oblas.axpy(blk_out + (size_t)o * stride, (uint8_t *)src, coeff, stride);
            }
        }
    }
    return true;
}

bool vxfec_rs8_encode_cpu(uint32_t K, uint32_t M, const uint8_t *sys_rows, uint32_t stride, const uint8_t *parity_m,
                          uint32_t num_parity, uint8_t *parity_out)
{
    vxfec_rs8_plan *plan = vxfec_rs8_encode_plan_new(NULL, K, M, parity_m, num_parity);
    if (!plan)
        return false;
    bool ok = vxfec_rs8_exec_cpu(plan, sys_rows, stride, 1u, parity_out);
    vxfec_rs8_plan_free(NULL, plan);
    return ok;
}

bool vxfec_rs8_decode_compact_cpu(uint32_t K, uint32_t M, uint8_t *compact_k_rows, uint32_t stride, const uint8_t *erased_k,
                                  const uint8_t *parity_m, uint32_t num_erased)
{
    if (!compact_k_rows || !erased_k || !parity_m || stride == 0 || num_erased == 0)
        return false;
    size_t tmp_bytes = 0;
    if (!vxfec_vk__mul3(num_erased, stride, 1u, &tmp_bytes))
        return false;

    uint8_t *recovered = (uint8_t *)malloc(tmp_bytes);
    if (!recovered)
        return false;

    vxfec_rs8_plan *plan = vxfec_rs8_decode_plan_new(NULL, K, M, erased_k, parity_m, num_erased);
    if (!plan) {
        free(recovered);
        return false;
    }
    bool ok = vxfec_rs8_exec_cpu(plan, compact_k_rows, stride, 1u, recovered);
    vxfec_rs8_plan_free(NULL, plan);

    if (ok) {
        for (uint32_t i = 0; i < num_erased; i++)
            memcpy(compact_k_rows + (size_t)erased_k[i] * stride, recovered + (size_t)i * stride, stride);
    }
    free(recovered);
    return ok;
}

bool vxfec_rs8_exec_pipelined(vxfec_vk_ctx *ctx, const vxfec_rs8_plan *plan, vxfec_vk_buf *d_buf, uint32_t stride,
                              uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out, uint64_t *gpu_ns_out)
{
    if (!plan || !plan->has_gpu)
        return false;
    return vxfec_vk__direct_pipelined(ctx, &plan->gpu, d_buf, stride, num_blocks, out_buf, upload_d, download_out, gpu_ns_out);
}

bool vxfec_rs8_exec_batch(vxfec_vk_ctx *ctx, const vxfec_rs8_plan *plan, vxfec_vk_buf *d_vram, uint32_t stride, uint32_t num_blocks,
                          vxfec_vk_buf *out_vram, uint64_t *gpu_ns_out)
{
    return vxfec_rs8_exec_pipelined(ctx, plan, d_vram, stride, num_blocks, out_vram, false, false, gpu_ns_out);
}

uint64_t vxfec_rs8_exec_submit(vxfec_vk_ctx *ctx, const vxfec_rs8_plan *plan, vxfec_vk_buf *d_buf, uint32_t stride,
                               uint32_t num_blocks, vxfec_vk_buf *out_buf, bool upload_d, bool download_out)
{
    if (!plan || !plan->has_gpu)
        return 0;
    return vxfec_vk__direct_submit(ctx, &plan->gpu, d_buf, stride, num_blocks, out_buf, upload_d, download_out);
}

uint64_t vxfec_rs8_submit_raw(vxfec_vk_ctx *ctx, bool is_decode, uint32_t K, uint32_t M, const uint8_t *erased_k,
                              const uint8_t *parity_m, uint32_t num_out, vxfec_vk_buf *d_buf, uint32_t stride, uint32_t num_blocks,
                              vxfec_vk_buf *out_buf, bool upload_d, bool download_out)
{
    if (!ctx || K == 0 || K > 255u || M == 0 || M > 256u - K || num_out == 0 || num_out > M)
        return 0;

    size_t m_stride = ((size_t)num_out + 31u) & ~(size_t)31u;
    uint8_t Y[256 * 256];
    bool ok = is_decode ? vxfec_rs8__recovery_matrix(K, M, erased_k, parity_m, num_out, Y, m_stride)
                        : vxfec_rs8__encode_matrix(K, M, parity_m, num_out, Y, m_stride);
    if (!ok)
        return 0;

    uint32_t *tab = NULL;
    size_t tab_bytes = 0;
    uint32_t n_tiles = 0;
    uint32_t tile_e = 8u;
    if (!vxfec_vk__emit_direct_gf256_tab(ctx->max_ssbo_range, Y, m_stride, K, num_out, &tab, &tab_bytes, &n_tiles, &tile_e)) {
        free(tab);
        return 0;
    }

    uint64_t ticket = vxfec_vk__direct_submit_staged_words(ctx, tab, tab_bytes, K, num_out, n_tiles, tile_e, false, d_buf, stride,
                                                           num_blocks, out_buf, upload_d, download_out);
    free(tab);
    return ticket;
}

uint64_t vxfec_rs8_encode_submit_vram(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, const uint8_t *parity_m, uint32_t num_parity,
                                      vxfec_vk_buf *src_vram, size_t src_offset, size_t valid_bytes, vxfec_vk_buf *work_d_buf,
                                      uint32_t stride, vxfec_vk_buf *out_buf, bool download_out)
{
    if (!ctx || !ctx->has_timeline_sem || !ctx->timeline_sem || !parity_m || !src_vram || !work_d_buf || !out_buf || K == 0 ||
        K > 255u || M == 0 || M > 256u - K || num_parity == 0 || num_parity > M || stride == 0 || (stride & 3u) != 0u)
        return 0;

    size_t d_bytes = (size_t)K * stride;
    size_t m_stride = ((size_t)num_parity + 31u) & ~(size_t)31u;
    uint8_t Y[256 * 256];
    if (!vxfec_rs8__encode_matrix(K, M, parity_m, num_parity, Y, m_stride))
        return 0;

    uint32_t *tab = NULL;
    size_t tab_bytes = 0;
    uint32_t n_tiles = 0, tile_e = 8u;
    if (!vxfec_vk__emit_direct_gf256_tab(ctx->max_ssbo_range, Y, m_stride, K, num_parity, &tab, &tab_bytes, &n_tiles, &tile_e)) {
        free(tab);
        return 0;
    }

    uint32_t slot = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!vxfec_vk__acquire_async_slot(ctx, &slot, &cmd)) {
        free(tab);
        return 0;
    }
    if (!vxfec_vk__ensure_staging_buf(ctx, &ctx->async_tab_bufs[slot], &ctx->async_tab_caps[slot], tab_bytes)) {
        free(tab);
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }
    vxfec_vk_buf *tbuf = ctx->async_tab_bufs[slot];
    memcpy(vxfec_vk_buf_staging_ptr(tbuf), tab, tab_bytes);
    free(tab);

    if (!tbuf->zero_copy) {
        VkBufferCopy t_copy = {.srcOffset = 0, .dstOffset = 0, .size = tab_bytes};
        vkCmdCopyBuffer(cmd, tbuf->stage_buf, tbuf->dev_buf, 1, &t_copy);
        VkMemoryBarrier xfer_mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &xfer_mb, 0, NULL, 0,
                             NULL);
    }

    /* bind src_vram directly as SSBO 0 when full generation & aligned, else gather+pad into work_d_buf */
    size_t real_src_off = src_vram->view_offset + src_offset;
    bool direct_src = (valid_bytes == d_bytes) && (src_vram->dev_buf != out_buf->dev_buf) && (src_offset <= src_vram->bytes) &&
                      (d_bytes <= src_vram->bytes - src_offset) && ((real_src_off & (size_t)(ctx->ssbo_offset_align - 1u)) == 0u);
    vxfec_vk_buf src_view = *src_vram;
    bool ext_src = false;
    vxfec_vk_buf *d_bind = work_d_buf;
    if (direct_src) {
        src_view.view_offset = real_src_off;
        src_view.bytes = d_bytes;
        src_view.owned_by_vk = true;
        ext_src = (src_vram->ext_kind != VXFEC_VK_EXT_NONE && !src_vram->owned_by_vk);
        if (ext_src)
            vxfec_vk__record_ext_acquire_cmd(ctx, cmd, src_vram, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        d_bind = &src_view;
    } else if (!vxfec_vk__record_vram_encode_input_cmd(ctx, cmd, work_d_buf, d_bytes, src_vram, src_offset, valid_bytes, 0u, K,
                                                       stride, stride)) {
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }

    bool ext_out = (!download_out && out_buf->ext_kind != VXFEC_VK_EXT_NONE && !out_buf->owned_by_vk);
    if (ext_out) {
        vxfec_vk__record_ext_acquire_cmd(ctx, cmd, out_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        out_buf->owned_by_vk = true;
    }

    vxfec_vk_direct_desc desc = {
        .tab_buf = tbuf,
        .tab_bytes = tab_bytes,
        .rows = K,
        .num_esis = num_parity,
        .num_tiles = n_tiles,
        .tile_e = tile_e,
        .pure_gf2 = false,
    };
    bool rec_ok = vxfec_vk__record_direct_cmd(ctx, cmd, ctx->async_ds_mix[slot], &desc, d_bind, stride, 1u, out_buf, false,
                                              download_out, false);
    if (ext_out)
        out_buf->owned_by_vk = false;
    if (!rec_ok) {
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }
    if (ext_src)
        vxfec_vk__record_ext_release_cmd(ctx, cmd, src_vram, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0);
    if (ext_out)
        vxfec_vk__record_ext_release_cmd(ctx, cmd, out_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);

    return vxfec_vk__submit_async_slot(ctx, slot, cmd);
}

uint64_t vxfec_rs8_repair_submit_vram(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, const uint8_t *sys_bitmap, const uint8_t *erased_k,
                                      const uint8_t *parity_m, uint32_t num_erased, vxfec_vk_buf *target_vram, size_t target_offset,
                                      size_t total_bytes, vxfec_vk_buf *work_d_buf, uint32_t stride, vxfec_vk_buf *out_buf)
{
    if (!ctx || !ctx->has_timeline_sem || !ctx->timeline_sem || !sys_bitmap || !erased_k || !parity_m || !target_vram ||
        !work_d_buf || !out_buf || K == 0 || K > 255u || M == 0 || M > 256u - K || num_erased == 0 || num_erased > M ||
        stride == 0 || (stride & 3u) != 0u)
        return 0;

    size_t d_bytes = (size_t)K * stride;
    size_t m_stride = ((size_t)num_erased + 31u) & ~(size_t)31u;
    uint8_t Y[256 * 256];
    if (!vxfec_rs8__recovery_matrix(K, M, erased_k, parity_m, num_erased, Y, m_stride))
        return 0;

    uint32_t *tab = NULL;
    size_t tab_bytes = 0;
    uint32_t n_tiles = 0, tile_e = 8u;
    if (!vxfec_vk__emit_direct_gf256_tab(ctx->max_ssbo_range, Y, m_stride, K, num_erased, &tab, &tab_bytes, &n_tiles, &tile_e)) {
        free(tab);
        return 0;
    }

    uint32_t slot = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (!vxfec_vk__acquire_async_slot(ctx, &slot, &cmd)) {
        free(tab);
        return 0;
    }
    if (!vxfec_vk__ensure_staging_buf(ctx, &ctx->async_tab_bufs[slot], &ctx->async_tab_caps[slot], tab_bytes)) {
        free(tab);
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }
    vxfec_vk_buf *tbuf = ctx->async_tab_bufs[slot];
    memcpy(vxfec_vk_buf_staging_ptr(tbuf), tab, tab_bytes);
    free(tab);

    if (!tbuf->zero_copy) {
        VkBufferCopy t_copy = {.srcOffset = 0, .dstOffset = 0, .size = tab_bytes};
        vkCmdCopyBuffer(cmd, tbuf->stage_buf, tbuf->dev_buf, 1, &t_copy);
    }

    uint32_t staged_rows[256];
    uint32_t miss_esis[256];
    for (uint32_t i = 0; i < num_erased; i++) {
        staged_rows[i] = erased_k[i];
        miss_esis[i] = erased_k[i];
    }

    bool ext_tgt = false;
    if (!vxfec_vk__record_vram_repair_gather_cmd(ctx, cmd, work_d_buf, d_bytes, sys_bitmap, K, stride, target_vram, target_offset,
                                                 total_bytes, 0u, stride, staged_rows, num_erased, &ext_tgt)) {
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }

    bool saved_out_owned = out_buf->owned_by_vk;
    if (out_buf->ext_kind != VXFEC_VK_EXT_NONE && !out_buf->owned_by_vk) {
        vxfec_vk__record_ext_acquire_cmd(ctx, cmd, out_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        out_buf->owned_by_vk = true;
    }

    vxfec_vk_direct_desc desc = {
        .tab_buf = tbuf,
        .tab_bytes = tab_bytes,
        .rows = K,
        .num_esis = num_erased,
        .num_tiles = n_tiles,
        .tile_e = tile_e,
        .pure_gf2 = false,
    };
    bool rec_ok =
        vxfec_vk__record_direct_cmd(ctx, cmd, ctx->async_ds_mix[slot], &desc, work_d_buf, stride, 1u, out_buf, false, false, false);
    if (rec_ok) {
        rec_ok = vxfec_vk__record_vram_repair_scatter_cmd(ctx, cmd, out_buf, stride, target_vram, target_offset, total_bytes,
                                                          stride, miss_esis, num_erased, ext_tgt);
    }
    if (!saved_out_owned && out_buf->ext_kind != VXFEC_VK_EXT_NONE) {
        out_buf->owned_by_vk = false;
        if (rec_ok)
            vxfec_vk__record_ext_release_cmd(ctx, cmd, out_buf, VK_PIPELINE_STAGE_TRANSFER_BIT, 0);
    }
    if (!rec_ok) {
        ctx->pending_owned_cnt = 0;
        (void)vkEndCommandBuffer(cmd);
        return 0;
    }
    return vxfec_vk__submit_async_slot(ctx, slot, cmd);
}
