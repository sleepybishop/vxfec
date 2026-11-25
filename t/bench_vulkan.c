#include "bench_batch.h"
#include "bench_common.h"
#include "vxfec_interleave.h"
#include "vxfec_rq.h"
#include "vxfec_rs_cauchy.h"
#include "vxfec_vk.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t mono_ns(void)
{
    return vxfec_bench_now_ns();
}

static bool fill_staging_batch(vxfec_vk_buf *buf, const uint8_t *initial, size_t block_bytes, uint32_t blocks)
{
    uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(buf);
    if (!stage || !initial || blocks == 0u || block_bytes > SIZE_MAX / blocks)
        return false;
    for (uint32_t b = 0; b < blocks; b++)
        memcpy(stage + (size_t)b * block_bytes, initial, block_bytes);
    return true;
}

/* populate Z identical blocks in device-local VRAM using O(log Z) VRAM-to-VRAM copies */
static bool populate_vram_blocks(vxfec_vk_ctx *ctx, vxfec_vk_buf *vram_buf, const uint8_t *blk0, size_t blk_bytes, uint32_t Z)
{
    if (!Z || !vxfec_vk_buf_write_offset(ctx, vram_buf, 0, blk0, blk_bytes))
        return false;
    uint32_t filled = 1u;
    while (filled < Z) {
        uint32_t chunk = (Z - filled < filled) ? (Z - filled) : filled;
        if (!vxfec_vk_buf_copy_offset(ctx, vram_buf, (size_t)filled * blk_bytes, vram_buf, 0, (size_t)chunk * blk_bytes))
            return false;
        filled += chunk;
    }
    return true;
}

static bool measure_pcie_dma(vxfec_vk_ctx *ctx, size_t xfer_bytes, double *out_h2d_gbps, double *out_d2h_gbps)
{
    vxfec_vk_buf *buf = vxfec_vk_buf_new(ctx, xfer_bytes, true);
    if (!buf)
        return false;

    uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(buf);
    for (size_t i = 0; i < xfer_bytes; i += 64u)
        stage[i] = (uint8_t)((i * 31u + 7u) & 0xffu);

    if (!vxfec_vk_buf_upload(ctx, buf, xfer_bytes) || !vxfec_vk_buf_download(ctx, buf, xfer_bytes)) {
        vxfec_vk_buf_free(ctx, buf);
        return false;
    }

    const int iters = 6;
    uint64_t t0 = mono_ns();
    for (int i = 0; i < iters; i++) {
        if (!vxfec_vk_buf_upload(ctx, buf, xfer_bytes)) {
            vxfec_vk_buf_free(ctx, buf);
            return false;
        }
    }
    uint64_t h2d_ns = mono_ns() - t0;

    t0 = mono_ns();
    for (int i = 0; i < iters; i++) {
        if (!vxfec_vk_buf_download(ctx, buf, xfer_bytes)) {
            vxfec_vk_buf_free(ctx, buf);
            return false;
        }
    }
    uint64_t d2h_ns = mono_ns() - t0;

    vxfec_vk_buf_free(ctx, buf);
    *out_h2d_gbps = (8.0 * (double)xfer_bytes * iters) / (double)(h2d_ns ? h2d_ns : 1ULL);
    *out_d2h_gbps = (8.0 * (double)xfer_bytes * iters) / (double)(d2h_ns ? d2h_ns : 1ULL);
    return true;
}

static bool run_rs8_row(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, uint32_t T, size_t target_batch_bytes)
{
    uint32_t stride = (T + 15u) & ~15u;
    size_t blk_bytes = (size_t)K * stride;
    size_t out_blk_bytes = (size_t)M * stride;
    uint64_t max_range = vxfec_vk_max_storage_buffer_bytes(ctx);
    size_t range = max_range > SIZE_MAX ? SIZE_MAX : (size_t)max_range;
    uint32_t Z = vxfec_bench_blocks(target_batch_bytes, range, blk_bytes, out_blk_bytes, 1024);
    if (!Z)
        return false;

    uint8_t parity_m[256];
    for (uint32_t m = 0; m < M; m++)
        parity_m[m] = (uint8_t)m;

    size_t d_bytes = (size_t)Z * blk_bytes;
    size_t o_bytes = (size_t)Z * out_blk_bytes;
    double batch_bits = 8.0 * (double)Z * (double)K * (double)T;

    vxfec_rs8_plan *plan = vxfec_rs8_encode_plan_new(ctx, K, M, parity_m, M);
    vxfec_vk_buf *d_buf = vxfec_vk_buf_new(ctx, d_bytes, true);
    vxfec_vk_buf *out_buf = vxfec_vk_buf_new(ctx, o_bytes, true);
    uint8_t *d_init = (uint8_t *)calloc(K, stride);
    uint8_t *cpu_out = (uint8_t *)calloc(M, stride);
    bool ok = false;
    if (!plan || !d_buf || !out_buf || !d_init || !cpu_out)
        goto done;

    for (size_t i = 0; i < blk_bytes; i++)
        d_init[i] = (uint8_t)((i * 37u + 13u) & 0xffu);

    if (!vxfec_rs8_exec_cpu(plan, d_init, stride, 1u, cpu_out) || !fill_staging_batch(d_buf, d_init, blk_bytes, Z) ||
        !vxfec_rs8_exec_pipelined(ctx, plan, d_buf, stride, Z, out_buf, true, true, NULL))
        goto done;

    const uint8_t *stage_out = (const uint8_t *)vxfec_vk_buf_staging_ptr(out_buf);
    if (memcmp(cpu_out, stage_out, out_blk_bytes) != 0 ||
        memcmp(cpu_out, stage_out + (size_t)(Z - 1u) * out_blk_bytes, out_blk_bytes) != 0) {
        fprintf(stderr, "RS8 verification mismatch at K=%u M=%u!\n", K, M);
        goto done;
    }

    int cpu_iters = 64;
    uint64_t t0_cpu = mono_ns();
    for (int it = 0; it < cpu_iters; it++)
        (void)vxfec_rs8_exec_cpu(plan, d_init, stride, 1u, cpu_out);
    uint64_t cpu_ns = mono_ns() - t0_cpu;
    double cpu_gbps = (8.0 * (double)cpu_iters * K * T) / (double)(cpu_ns ? cpu_ns : 1ULL);

    uint64_t warm_t0 = mono_ns();
    do {
        if (!vxfec_rs8_exec_batch(ctx, plan, d_buf, stride, Z, out_buf, NULL))
            goto done;
    } while ((mono_ns() - warm_t0) < 15000000ULL);

    int gpu_iters = 8;
    uint64_t gpu_ns_sum = 0;
    uint64_t wall_ns = 0;
    for (int it = 0; it < gpu_iters; it++) {
        uint64_t ns = 0;
        uint64_t t0 = mono_ns();
        if (!vxfec_rs8_exec_batch(ctx, plan, d_buf, stride, Z, out_buf, &ns))
            goto done;
        wall_ns += mono_ns() - t0;
        gpu_ns_sum += ns;
    }
    uint64_t eff_ns = (gpu_ns_sum > 0u) ? gpu_ns_sum : wall_ns;
    double vram_gbps = (batch_bits * gpu_iters) / (double)(eff_ns ? eff_ns : 1ULL);

    uint64_t e2e_ns = 0;
    for (int it = 0; it < gpu_iters; it++) {
        uint64_t t0 = mono_ns();
        if (!vxfec_rs8_exec_pipelined(ctx, plan, d_buf, stride, Z, out_buf, true, true, NULL))
            goto done;
        e2e_ns += mono_ns() - t0;
    }
    double e2e_gbps = (batch_bits * gpu_iters) / (double)(e2e_ns ? e2e_ns : 1ULL);

    printf("%7u %5u %5u | %10.2f | %15.2f | %13.2f\n", K, M, Z, cpu_gbps, vram_gbps, e2e_gbps);
    ok = true;

done:
    free(cpu_out);
    free(d_init);
    vxfec_vk_buf_free(ctx, out_buf);
    vxfec_vk_buf_free(ctx, d_buf);
    vxfec_rs8_plan_free(ctx, plan);
    return ok;
}

static bool run_encode_row(vxfec_vk_ctx *ctx, uint32_t K, uint32_t T, size_t target_batch_bytes, bool gf2_mode)
{
    nanorq_core rq = {0};
#ifdef NANORQ_HAS_GF2_MODE
    bool ok = gf2_mode ? nanorq_core_encoder_new_gf2(K, 0, &rq) : nanorq_core_encoder_new(K, 0, &rq);
#else
    (void)gf2_mode;
    bool ok = nanorq_core_encoder_new(K, 0, &rq);
#endif
    if (!ok) {
        fprintf(stderr, "encoder initialization failed at K=%u\n", K);
        return false;
    }

    size_t prep_len = nanorq_core_calculate_prepare_memory(&rq);
    size_t work_len = nanorq_core_calculate_work_memory(&rq);
    size_t sched_len = ops_estimate_schedule_bytes(K);
    uint8_t *prep_mem = (uint8_t *)malloc(prep_len);
    uint8_t *work_mem = (uint8_t *)malloc(work_len);
    void *sched_mem = malloc(sched_len);
    schedule S = {0};
    bool success = false;
    if (!prep_mem || !work_mem || !schedule_init(&S, sched_mem, sched_len))
        goto cleanup_early;

    if (!nanorq_core_prepare(&rq, prep_mem, prep_len))
        goto cleanup_early;
    nanorq_core_set_op_callback(&rq, &S, ops_push);
    if (!nanorq_core_precalculate(&rq, work_mem, work_len))
        goto cleanup_early;

    uint32_t rows = nanorq_core_get_pc_rows(&rq);
    uint32_t sh = nanorq_core_get_pc_genc_offset(&rq);
    uint32_t stride = nanorq_core_recommended_stride(T);
    if (!stride || rows > SIZE_MAX / stride)
        goto cleanup_early;
    size_t block_bytes = (size_t)rows * stride;
    uint32_t num_rep = (K + 9u) / 10u;
    if ((size_t)num_rep > SIZE_MAX / stride)
        goto cleanup_early;
    uint64_t max_range = vxfec_vk_max_storage_buffer_bytes(ctx);
    size_t range = max_range > SIZE_MAX ? SIZE_MAX : (size_t)max_range;
    uint32_t Z = vxfec_bench_blocks(target_batch_bytes, range, block_bytes, (size_t)num_rep * stride, 512);
    if (!Z)
        goto cleanup_early;
    size_t batch_d_bytes = (size_t)Z * block_bytes;
    size_t batch_out_bytes = (size_t)Z * num_rep * stride;
    double batch_payload_bits = 8.0 * (double)Z * (double)K * (double)T;

    vxfec_rq_sched *gs = vxfec_rq_sched_new(ctx, &rq, &S);
    vxfec_vk_buf *d_buf = vxfec_vk_buf_new(ctx, batch_d_bytes, true);
    vxfec_vk_buf *d_ref = vxfec_vk_buf_new(ctx, batch_d_bytes, false);
    vxfec_vk_buf *out_buf = vxfec_vk_buf_new(ctx, batch_out_bytes, true);
    uint8_t *d_cpu = (uint8_t *)calloc(rows, stride);
    uint8_t *d_init = (uint8_t *)calloc(1, block_bytes);
    uint32_t *esis = (uint32_t *)malloc(num_rep * sizeof(*esis));
    uint8_t *rep_cpu = (uint8_t *)calloc(num_rep, stride);

    if (!gs || !d_buf || !d_ref || !out_buf || !d_cpu || !d_init || !esis || !rep_cpu)
        goto cleanup;

    for (uint32_t e = 0; e < num_rep; e++)
        esis[e] = K + e;

    for (uint32_t esi = 0; esi < K; esi++) {
        uint8_t *row = d_init + (size_t)(sh + esi) * stride;
        for (uint32_t c = 0; c < T; c++)
            row[c] = (uint8_t)((esi * 31u + c * 17u + 7u) & 0xffu);
    }

    if (!fill_staging_batch(d_buf, d_init, block_bytes, Z) || !vxfec_vk_buf_upload(ctx, d_buf, batch_d_bytes) ||
        !vxfec_vk_buf_copy(ctx, d_ref, d_buf, batch_d_bytes))
        goto cleanup;

    memcpy(d_cpu, d_init, block_bytes);
    if (!ops_run_pivoted(d_cpu, stride, &S))
        goto cleanup;
    for (uint32_t e = 0; e < num_rep; e++)
        ops_mix_pivoted(&rq, d_cpu, stride, esis[e], rep_cpu + (size_t)e * stride);

    if (!vxfec_rq_run_and_mix_pipelined(ctx, &rq, gs, d_buf, stride, Z, esis, num_rep, out_buf, true, true, NULL))
        goto cleanup;
    const uint8_t *stage_out = (const uint8_t *)vxfec_vk_buf_staging_ptr(out_buf);
    uint32_t chk_blks[3] = {0u, Z / 2u, Z - 1u};
    for (int cb = 0; cb < 3; cb++) {
        uint32_t b = chk_blks[cb];
        for (uint32_t e = 0; e < num_rep; e++) {
            const uint8_t *got = stage_out + ((size_t)b * num_rep + e) * stride;
            if (memcmp(rep_cpu + (size_t)e * stride, got, T) != 0)
                goto cleanup;
        }
    }

    /* include paired ops_run_pivoted and 10% ops_mix_pivoted in CPU baseline */
    int cpu_iters = (K >= 10000u) ? 2 : ((K >= 1000u) ? 8 : 32);
    uint64_t cpu_ns = 0;
    for (int it = 0; it < cpu_iters; it++) {
        memcpy(d_cpu, d_init, block_bytes);
        uint64_t t0 = mono_ns();
        if (!ops_run_pivoted(d_cpu, stride, &S))
            goto cleanup;
        for (uint32_t e = 0; e < num_rep; e++)
            ops_mix_pivoted(&rq, d_cpu, stride, esis[e], rep_cpu + (size_t)e * stride);
        cpu_ns += mono_ns() - t0;
    }
    double cpu_gbps = (8.0 * (double)cpu_iters * K * T) / (double)(cpu_ns ? cpu_ns : 1ULL);

    int gpu_iters = (K >= 50000u) ? 2 : 6;
    uint64_t warm_t0 = mono_ns();
    do {
        if (!vxfec_vk_buf_copy(ctx, d_buf, d_ref, batch_d_bytes) || !vxfec_rq_ops_run_batch(ctx, gs, d_buf, stride, Z, NULL))
            goto cleanup;
    } while (K < 10000u && (mono_ns() - warm_t0) < 20000000ULL);

    uint64_t gpu_run_ns_sum = 0;
    uint64_t wall_run_ns = 0;
    for (int it = 0; it < gpu_iters; it++) {
        uint64_t ns = 0;
        if (!vxfec_vk_buf_copy(ctx, d_buf, d_ref, batch_d_bytes))
            goto cleanup;
        uint64_t t0 = mono_ns();
        if (!vxfec_rq_ops_run_batch(ctx, gs, d_buf, stride, Z, &ns))
            goto cleanup;
        wall_run_ns += mono_ns() - t0;
        gpu_run_ns_sum += ns;
    }
    uint64_t eff_run_ns = (gpu_run_ns_sum > 0u) ? gpu_run_ns_sum : wall_run_ns;
    double vram_run_gbps = (batch_payload_bits * gpu_iters) / (double)(eff_run_ns ? eff_run_ns : 1ULL);

    uint64_t gpu_rm_ns_sum = 0;
    uint64_t wall_rm_ns = 0;
    for (int it = 0; it < gpu_iters; it++) {
        uint64_t ns = 0;
        if (!vxfec_vk_buf_copy(ctx, d_buf, d_ref, batch_d_bytes))
            goto cleanup;
        uint64_t t0 = mono_ns();
        if (!vxfec_rq_run_and_mix_pipelined(ctx, &rq, gs, d_buf, stride, Z, esis, num_rep, out_buf, false, false, &ns))
            goto cleanup;
        wall_rm_ns += mono_ns() - t0;
        gpu_rm_ns_sum += ns;
    }
    uint64_t eff_rm_ns = (gpu_rm_ns_sum > 0u) ? gpu_rm_ns_sum : wall_rm_ns;
    double vram_rm_gbps = (batch_payload_bits * gpu_iters) / (double)(eff_rm_ns ? eff_rm_ns : 1ULL);

    bool discrete = vxfec_vk_is_discrete(ctx);
    uint64_t wall_e2e_ns = 0;
    for (int it = 0; it < gpu_iters; it++) {
        if (!discrete && !vxfec_vk_buf_copy(ctx, d_buf, d_ref, batch_d_bytes))
            goto cleanup;
        uint64_t t0 = mono_ns();
        if (!vxfec_rq_run_and_mix_pipelined(ctx, &rq, gs, d_buf, stride, Z, esis, num_rep, out_buf, true, true, NULL))
            goto cleanup;
        wall_e2e_ns += mono_ns() - t0;
    }
    double pcie_e2e_gbps = (batch_payload_bits * gpu_iters) / (double)(wall_e2e_ns ? wall_e2e_ns : 1ULL);

    printf("%7u %5u | %10.2f | %13.2f | %15.2f | %11.2f\n", K, Z, cpu_gbps, vram_run_gbps, vram_rm_gbps, pcie_e2e_gbps);
    success = true;

cleanup:
    free(rep_cpu);
    free(esis);
    free(d_init);
    free(d_cpu);
    vxfec_vk_buf_free(ctx, out_buf);
    vxfec_vk_buf_free(ctx, d_ref);
    vxfec_vk_buf_free(ctx, d_buf);
    vxfec_rq_sched_free(ctx, gs);
cleanup_early:
    free(sched_mem);
    free(work_mem);
    free(prep_mem);
    return success;
}

static bool run_interleave_row(vxfec_vk_ctx *ctx, uint32_t K, uint32_t T, size_t target_ring_bytes)
{
    uint32_t drops = K / 10u;
    if (drops < 1u)
        drops = 1u;
    params p_tmp = params_init(K);
    uint32_t overhead = (p_tmp.Kprime - K) + p_tmp.H + 8u;
    uint32_t stride = nanorq_core_recommended_stride(T);
    if (!stride)
        return false;

    nanorq_core enc = {0}, dec = {0};
    uint8_t *enc_prep = NULL, *enc_work = NULL, *prep_mem = NULL, *work_mem = NULL;
    void *enc_sched = NULL, *sched_mem = NULL;
    schedule S_enc = {0}, S = {0};
    uint8_t *src_orig = NULL, *d_enc = NULL, *repairs = NULL;
    uint32_t *drop_esis = NULL;
    vxfec_rq_sched *gs = NULL;
    vxfec_rq_direct *gd = NULL;
    vxfec_vk_buf *d_buf = NULL, *d_ref = NULL, *out_buf = NULL;
    uint8_t *d_cpu = NULL, *d_init = NULL, *out_cpu = NULL;
    bool success = false;

    /* encode a valid K-symbol RaptorQ block first */
    if (!nanorq_core_encoder_new(K, 0, &enc))
        goto cleanup;
    size_t ep_len = nanorq_core_calculate_prepare_memory(&enc);
    size_t ew_len = nanorq_core_calculate_work_memory(&enc);
    size_t es_len = ops_estimate_schedule_bytes(K);
    enc_prep = (uint8_t *)malloc(ep_len);
    enc_work = (uint8_t *)malloc(ew_len);
    enc_sched = malloc(es_len);
    if (!enc_prep || !enc_work || !schedule_init(&S_enc, enc_sched, es_len) || !nanorq_core_prepare(&enc, enc_prep, ep_len))
        goto cleanup;
    nanorq_core_set_op_callback(&enc, &S_enc, ops_push);
    if (!nanorq_core_precalculate(&enc, ew_len ? enc_work : NULL, ew_len))
        goto cleanup;

    uint32_t rows_enc = nanorq_core_get_pc_rows(&enc);
    uint32_t sh_enc = nanorq_core_get_pc_genc_offset(&enc);
    uint32_t total_rep = drops + overhead;
    src_orig = (uint8_t *)calloc(K, stride);
    d_enc = (uint8_t *)calloc(rows_enc, stride);
    repairs = (uint8_t *)calloc(total_rep, stride);
    if (!src_orig || !d_enc || !repairs)
        goto cleanup;
    for (uint32_t k = 0; k < K; k++) {
        for (uint32_t b = 0; b < T; b++)
            src_orig[(size_t)k * stride + b] = (uint8_t)((k * 53u + b * 23u + 9u) & 0xffu);
        memcpy(d_enc + (size_t)(sh_enc + k) * stride, src_orig + (size_t)k * stride, T);
    }
    if (!ops_run_pivoted(d_enc, stride, &S_enc))
        goto cleanup;
    for (uint32_t r = 0; r < total_rep; r++)
        ops_mix_pivoted(&enc, d_enc, stride, K + r, repairs + (size_t)r * stride);

    /* build valid decoder matrix at dec.P.Kprime + i */
    if (!nanorq_core_encoder_new(K, overhead, &dec))
        goto cleanup;
    size_t prep_len = nanorq_core_calculate_prepare_memory(&dec);
    size_t work_len = nanorq_core_calculate_work_memory(&dec);
    size_t sched_len = ops_estimate_schedule_bytes(K + overhead);
    prep_mem = (uint8_t *)malloc(prep_len);
    work_mem = (uint8_t *)malloc(work_len);
    sched_mem = malloc(sched_len);
    if (!prep_mem || !work_mem || !schedule_init(&S, sched_mem, sched_len) || !nanorq_core_generate(&dec, prep_mem, prep_len))
        goto cleanup;
    for (uint32_t r = 0; r < drops; r++)
        nanorq_core_replace_symbol(&dec, r, K + overhead + r);
    for (uint32_t i = 0; i < overhead; i++)
        nanorq_core_replace_symbol(&dec, dec.P.Kprime + i, K + i);
    if (!nanorq_core_patch_matrix(&dec))
        goto cleanup;
    nanorq_core_set_op_callback(&dec, &S, ops_push);
    if (!nanorq_core_precalculate(&dec, work_mem, work_len))
        goto cleanup;

    uint32_t rows = nanorq_core_get_pc_rows(&dec);
    uint32_t sh = nanorq_core_get_pc_genc_offset(&dec);
    if (rows > SIZE_MAX / stride)
        goto cleanup;
    size_t block_bytes = (size_t)rows * stride;
    uint64_t max_range = vxfec_vk_max_storage_buffer_bytes(ctx);
    size_t range = max_range > SIZE_MAX ? SIZE_MAX : (size_t)max_range;
    uint32_t Z = vxfec_bench_blocks(target_ring_bytes, range, block_bytes, (size_t)drops * stride, 1024);
    if (!Z)
        goto cleanup;
    size_t ring_bytes = (size_t)Z * block_bytes;
    size_t out_bytes = (size_t)Z * drops * stride;
    double batch_bits = 8.0 * (double)Z * (double)K * (double)T;

    drop_esis = (uint32_t *)malloc(drops * sizeof(*drop_esis));
    if (!drop_esis)
        goto cleanup;
    for (uint32_t r = 0; r < drops; r++)
        drop_esis[r] = r;

    gs = vxfec_rq_sched_new(ctx, &dec, &S);
    gd = vxfec_rq_direct_new(ctx, &dec, &S, drop_esis, drops);
    d_buf = vxfec_vk_buf_new(ctx, ring_bytes, true);
    d_ref = vxfec_vk_buf_new(ctx, ring_bytes, false);
    out_buf = vxfec_vk_buf_new(ctx, out_bytes, true);
    d_cpu = (uint8_t *)calloc(rows, stride);
    d_init = (uint8_t *)calloc(rows, stride);
    out_cpu = (uint8_t *)calloc(drops, stride);
    if (!gs || !gd || !d_buf || !d_ref || !out_buf || !d_cpu || !d_init || !out_cpu)
        goto cleanup;

    /* populate d_init with surviving systematic + replacement/overhead repairs */
    for (uint32_t k = drops; k < K; k++)
        memcpy(d_init + (size_t)(sh + k) * stride, src_orig + (size_t)k * stride, T);
    for (uint32_t r = 0; r < drops; r++)
        memcpy(d_init + (size_t)(sh + r) * stride, repairs + (size_t)(overhead + r) * stride, T);
    for (uint32_t i = 0; i < overhead; i++)
        memcpy(d_init + (size_t)(sh + dec.P.Kprime + i) * stride, repairs + (size_t)i * stride, T);

    if (!fill_staging_batch(d_buf, d_init, block_bytes, Z) || !vxfec_vk_buf_upload(ctx, d_buf, ring_bytes) ||
        !vxfec_vk_buf_copy(ctx, d_ref, d_buf, ring_bytes) || !vxfec_rq_direct_batch(ctx, gd, d_buf, stride, Z, out_buf, NULL) ||
        !vxfec_vk_buf_download(ctx, out_buf, out_bytes))
        goto cleanup;

    const uint8_t *stage_out = (const uint8_t *)vxfec_vk_buf_staging_ptr(out_buf);
    uint32_t check_blocks[3] = {0u, Z / 2u, Z - 1u};
    for (int i = 0; i < 3; i++) {
        uint32_t b = check_blocks[i];
        for (uint32_t r = 0; r < drops; r++) {
            if (memcmp(src_orig + (size_t)r * stride, stage_out + ((size_t)b * drops + r) * stride, T) != 0)
                goto cleanup;
        }
    }

    int cpu_iters = 16;
    uint64_t cpu_ns = 0;
    for (int it = 0; it < cpu_iters; it++) {
        memcpy(d_cpu, d_init, block_bytes);
        uint64_t t0 = mono_ns();
        if (!ops_run_pivoted(d_cpu, stride, &S))
            goto cleanup;
        for (uint32_t r = 0; r < drops; r++)
            ops_mix_pivoted(&dec, d_cpu, stride, drop_esis[r], out_cpu + (size_t)r * stride);
        cpu_ns += mono_ns() - t0;
    }
    double cpu_gbps = (8.0 * (double)cpu_iters * K * T) / (double)(cpu_ns ? cpu_ns : 1ULL);

    uint64_t warm_t0 = mono_ns();
    do {
        if (!vxfec_vk_buf_copy(ctx, d_buf, d_ref, ring_bytes) || !vxfec_rq_ops_run_batch(ctx, gs, d_buf, stride, Z, NULL))
            goto cleanup;
    } while ((mono_ns() - warm_t0) < 20000000ULL);
    int gpu_iters = 8;
    uint64_t gpu_ns_sum = 0;
    uint64_t wall_run_ns = 0;
    for (int it = 0; it < gpu_iters; it++) {
        uint64_t ns = 0;
        if (!vxfec_vk_buf_copy(ctx, d_buf, d_ref, ring_bytes))
            goto cleanup;
        uint64_t t0 = mono_ns();
        if (!vxfec_rq_run_and_mix_pipelined(ctx, &dec, gs, d_buf, stride, Z, drop_esis, drops, out_buf, false, false, &ns))
            goto cleanup;
        wall_run_ns += mono_ns() - t0;
        gpu_ns_sum += ns;
    }
    uint64_t eff_run_ns = (gpu_ns_sum > 0u) ? gpu_ns_sum : wall_run_ns;
    double vram_gbps = (batch_bits * gpu_iters) / (double)(eff_run_ns ? eff_run_ns : 1ULL);

    if (!vxfec_vk_buf_copy(ctx, d_buf, d_ref, ring_bytes))
        goto cleanup;
    uint64_t dir_warm_t0 = mono_ns();
    do {
        if (!vxfec_rq_direct_batch(ctx, gd, d_buf, stride, Z, out_buf, NULL))
            goto cleanup;
    } while ((mono_ns() - dir_warm_t0) < 20000000ULL);
    uint64_t dir_ns_sum = 0;
    uint64_t t0 = mono_ns();
    for (int it = 0; it < gpu_iters; it++) {
        uint64_t ns = 0;
        if (!vxfec_rq_direct_batch(ctx, gd, d_buf, stride, Z, out_buf, &ns))
            goto cleanup;
        dir_ns_sum += ns;
    }
    uint64_t wall_dir_ns = mono_ns() - t0;
    uint64_t eff_dir_ns = (dir_ns_sum > 0u) ? dir_ns_sum : wall_dir_ns;
    double dir_gpu_gbps = (batch_bits * gpu_iters) / (double)(eff_dir_ns ? eff_dir_ns : 1ULL);
    double dir_wall_gbps = (batch_bits * gpu_iters) / (double)(wall_dir_ns ? wall_dir_ns : 1ULL);

    printf("%6u %6u %5u %5u | %4s | %11.2f | %12.2f | %10.2f | %11.2f | %11.2f\n", K, T, Z, drops,
           vxfec_rq_sched_is_pure_gf2(gs) ? "GF2" : "GF8", cpu_gbps, vram_gbps, dir_gpu_gbps, dir_wall_gbps,
           dir_gpu_gbps * drops / K);
    success = true;

cleanup:
    free(out_cpu);
    free(d_init);
    free(d_cpu);
    vxfec_vk_buf_free(ctx, out_buf);
    vxfec_vk_buf_free(ctx, d_ref);
    vxfec_vk_buf_free(ctx, d_buf);
    vxfec_rq_direct_free(ctx, gd);
    vxfec_rq_sched_free(ctx, gs);
    free(drop_esis);
    free(sched_mem);
    free(work_mem);
    free(prep_mem);
    free(repairs);
    free(d_enc);
    free(src_orig);
    free(enc_sched);
    free(enc_work);
    free(enc_prep);
    return success;
}

static bool run_rq_superframe_row(vxfec_vk_ctx *ctx, uint32_t K, uint32_t T, uint32_t loss_pct, uint32_t extra_slack,
                                  size_t target_vram_bytes)
{
    uint32_t drops = (K * loss_pct + 99u) / 100u;
    if (drops < 1u)
        drops = 1u;
    if (drops > K)
        drops = K;

    params p_tmp = params_init(K);
    uint32_t overhead = (p_tmp.Kprime - K) + extra_slack;
    uint32_t stride = nanorq_core_recommended_stride(T);
    if (!stride)
        return false;

    bool ok = false;
    nanorq_core enc = {0}, dec = {0};
    uint8_t *enc_prep = NULL, *enc_work = NULL, *dec_prep = NULL, *dec_work = NULL;
    void *enc_sched = NULL, *dec_sched = NULL;
    uint8_t *src_orig = NULL, *d_enc = NULL, *repairs = NULL, *d_dec = NULL, *rec_chk = NULL;
    uint32_t *drop_esis = NULL;
    vxfec_rq_direct *gd = NULL;
    vxfec_vk_buf *d_vram = NULL, *out_vram = NULL;
    schedule S_enc = {0}, S_dec = {0};

    if (!nanorq_core_encoder_new(K, 0, &enc))
        goto done;
    size_t ep_len = nanorq_core_calculate_prepare_memory(&enc);
    size_t ew_len = nanorq_core_calculate_work_memory(&enc);
    size_t es_len = ops_estimate_schedule_bytes(K);
    enc_prep = (uint8_t *)malloc(ep_len);
    enc_work = (uint8_t *)malloc(ew_len);
    enc_sched = malloc(es_len);
    if (!enc_prep || !enc_work || !schedule_init(&S_enc, enc_sched, es_len) || !nanorq_core_prepare(&enc, enc_prep, ep_len))
        goto done;
    nanorq_core_set_op_callback(&enc, &S_enc, ops_push);
    if (!nanorq_core_precalculate(&enc, enc_work, ew_len))
        goto done;

    uint32_t rows_enc = nanorq_core_get_pc_rows(&enc);
    uint32_t sh = nanorq_core_get_pc_genc_offset(&enc);
    uint32_t total_rep = drops + overhead;
    src_orig = (uint8_t *)calloc(K, stride);
    d_enc = (uint8_t *)calloc(rows_enc, stride);
    repairs = (uint8_t *)calloc(total_rep, stride);
    if (!src_orig || !d_enc || !repairs)
        goto done;

    for (uint32_t k = 0; k < K; k++) {
        for (uint32_t b = 0; b < T; b++)
            src_orig[(size_t)k * stride + b] = (uint8_t)((k * 59u + b * 29u + (b >> 6u) * 17u + 11u) & 0xffu);
        memcpy(d_enc + (size_t)(sh + k) * stride, src_orig + (size_t)k * stride, T);
    }
    if (!ops_run_pivoted(d_enc, stride, &S_enc))
        goto done;
    for (uint32_t r = 0; r < total_rep; r++)
        ops_mix_pivoted(&enc, d_enc, stride, K + r, repairs + (size_t)r * stride);

    drop_esis = (uint32_t *)malloc(drops * sizeof(*drop_esis));
    if (!drop_esis)
        goto done;
    for (uint32_t r = 0; r < drops; r++)
        drop_esis[r] = r;

    uint64_t comp_t0 = mono_ns();
    if (!nanorq_core_encoder_new(K, overhead, &dec))
        goto done;
    size_t dp_len = nanorq_core_calculate_prepare_memory(&dec);
    size_t dw_len = nanorq_core_calculate_work_memory(&dec);
    size_t ds_len = ops_estimate_schedule_bytes(K + overhead);
    dec_prep = (uint8_t *)malloc(dp_len);
    dec_work = (uint8_t *)malloc(dw_len);
    dec_sched = malloc(ds_len);
    if (!dec_prep || !dec_work || !schedule_init(&S_dec, dec_sched, ds_len) || !nanorq_core_generate(&dec, dec_prep, dp_len))
        goto done;
    for (uint32_t r = 0; r < drops; r++)
        nanorq_core_replace_symbol(&dec, r, K + overhead + r);
    for (uint32_t i = 0; i < overhead; i++)
        nanorq_core_replace_symbol(&dec, dec.P.Kprime + i, K + i);
    if (!nanorq_core_patch_matrix(&dec))
        goto done;
    nanorq_core_set_op_callback(&dec, &S_dec, ops_push);
    if (!nanorq_core_precalculate(&dec, dec_work, dw_len))
        goto done;

    gd = vxfec_rq_direct_new(ctx, &dec, &S_dec, drop_esis, drops);
    if (!gd)
        goto done;
    uint64_t comp_ns = mono_ns() - comp_t0;

    uint32_t rows_dec = nanorq_core_get_pc_rows(&dec);
    size_t blk_bytes = (size_t)rows_dec * stride;
    size_t out_blk_bytes = (size_t)drops * stride;
    uint64_t max_ssbo = vxfec_vk_max_storage_buffer_bytes(ctx);
    size_t range = max_ssbo > SIZE_MAX ? SIZE_MAX : (size_t)max_ssbo;
    uint32_t Z = vxfec_bench_blocks(target_vram_bytes, range, blk_bytes, out_blk_bytes, 2048);
    if (!Z)
        goto done;

    size_t d_bytes = (size_t)Z * blk_bytes;
    size_t o_bytes = (size_t)Z * out_blk_bytes;
    d_dec = (uint8_t *)calloc(rows_dec, stride);
    rec_chk = (uint8_t *)calloc(drops, stride);
    d_vram = vxfec_vk_buf_new(ctx, d_bytes, false);
    out_vram = vxfec_vk_buf_new(ctx, o_bytes, false);
    if (!d_dec || !rec_chk || !d_vram || !out_vram)
        goto done;

    for (uint32_t k = drops; k < K; k++)
        memcpy(d_dec + (size_t)(sh + k) * stride, src_orig + (size_t)k * stride, T);
    for (uint32_t r = 0; r < drops; r++)
        memcpy(d_dec + (size_t)(sh + r) * stride, repairs + (size_t)(overhead + r) * stride, T);
    for (uint32_t i = 0; i < overhead; i++)
        memcpy(d_dec + (size_t)(sh + dec.P.Kprime + i) * stride, repairs + (size_t)i * stride, T);

    if (!populate_vram_blocks(ctx, d_vram, d_dec, blk_bytes, Z))
        goto done;

    uint64_t warm_t0 = mono_ns();
    do {
        if (!vxfec_rq_direct_batch(ctx, gd, d_vram, stride, Z, out_vram, NULL))
            goto done;
    } while ((mono_ns() - warm_t0) < 20000000ULL);

    uint32_t probe_z[3] = {0u, Z / 2u, Z - 1u};
    for (int p = 0; p < 3; p++) {
        if (!vxfec_vk_buf_read_offset(ctx, out_vram, (size_t)probe_z[p] * out_blk_bytes, rec_chk, out_blk_bytes))
            goto done;
        for (uint32_t r = 0; r < drops; r++) {
            if (memcmp(rec_chk + (size_t)r * stride, src_orig + (size_t)r * stride, T) != 0) {
                fprintf(stderr, "FAIL: RaptorQ superframe verify mismatch at loss=%u%%\n", loss_pct);
                goto done;
            }
        }
    }

    const int iters = 10;
    uint64_t gpu_ns_sum = 0;
    uint64_t wall_t0 = mono_ns();
    for (int it = 0; it < iters; it++) {
        uint64_t ns = 0;
        if (!vxfec_rq_direct_batch(ctx, gd, d_vram, stride, Z, out_vram, &ns))
            goto done;
        gpu_ns_sum += ns;
    }
    uint64_t wall_ns = mono_ns() - wall_t0;
    uint64_t eff_ns = (gpu_ns_sum > 0u) ? gpu_ns_sum : wall_ns;

    double payload_bits = 8.0 * (double)Z * (double)K * (double)T * iters;
    double repaired_bits = 8.0 * (double)Z * (double)drops * (double)T * iters;
    double min_rw_bytes = (double)Z * (double)(K + overhead + drops) * (double)stride * iters;

    double gpu_gbps = payload_bits / (double)(eff_ns ? eff_ns : 1ULL);
    double wall_gbps = payload_bits / (double)(wall_ns ? wall_ns : 1ULL);
    double rep_gbps = repaired_bits / (double)(eff_ns ? eff_ns : 1ULL);
    double mem_gbs = min_rw_bytes / (double)(eff_ns ? eff_ns : 1ULL);

    printf("  %3u%% (+%2u ov) | %4s | %6.1f MiB (%4u) | %6.2f ms | %8.1f Gbps | %8.1f Gbps | %7.1f Gbps | %6.1f GB/s\n", loss_pct,
           overhead, vxfec_rq_direct_is_pure_gf2(gd) ? "GF2" : "GF8", (double)d_bytes / (1024.0 * 1024.0), Z, (double)comp_ns / 1e6,
           gpu_gbps, wall_gbps, rep_gbps, mem_gbs);
    ok = true;

done:
    vxfec_vk_buf_free(ctx, out_vram);
    vxfec_vk_buf_free(ctx, d_vram);
    vxfec_rq_direct_free(ctx, gd);
    free(rec_chk);
    free(d_dec);
    free(drop_esis);
    free(dec_sched);
    free(dec_work);
    free(dec_prep);
    free(repairs);
    free(d_enc);
    free(src_orig);
    free(enc_sched);
    free(enc_work);
    free(enc_prep);
    return ok;
}

static bool run_rs8_superframe_row(vxfec_vk_ctx *ctx, uint32_t K, uint32_t T, uint32_t loss_pct, size_t target_vram_bytes)
{
    uint32_t stride = (T + 15u) & ~15u;
    uint32_t drops = (K * loss_pct + 99u) / 100u;
    if (drops < 1u)
        drops = 1u;
    uint32_t M = 255u - K;
    if (drops > M)
        return false;

    size_t blk_bytes = (size_t)K * stride;
    size_t out_blk_bytes = (size_t)drops * stride;
    uint64_t max_ssbo = vxfec_vk_max_storage_buffer_bytes(ctx);
    size_t range = max_ssbo > SIZE_MAX ? SIZE_MAX : (size_t)max_ssbo;
    uint32_t Z = vxfec_bench_blocks(target_vram_bytes, range, blk_bytes, out_blk_bytes, 2048);
    if (!Z)
        return false;

    uint8_t erased_k[256], parity_m[256];
    for (uint32_t i = 0; i < drops; i++) {
        erased_k[i] = (uint8_t)i;
        parity_m[i] = (uint8_t)i;
    }

    bool ok = false;
    uint8_t *sys_orig = (uint8_t *)calloc(K, stride);
    uint8_t *par_blk = (uint8_t *)calloc(drops, stride);
    uint8_t *compact = (uint8_t *)calloc(K, stride);
    uint8_t *rec_chk = (uint8_t *)calloc(drops, stride);
    vxfec_rs8_plan *dec_plan = NULL;
    vxfec_vk_buf *d_vram = NULL, *out_vram = NULL;
    if (!sys_orig || !par_blk || !compact || !rec_chk)
        goto done;

    for (size_t i = 0; i < blk_bytes; i++)
        sys_orig[i] = (uint8_t)((i * 43u + (i >> 7u) * 19u + 5u) & 0xffu);
    if (!vxfec_rs8_encode_cpu(K, M, sys_orig, stride, parity_m, drops, par_blk))
        goto done;

    memcpy(compact, sys_orig, blk_bytes);
    for (uint32_t i = 0; i < drops; i++)
        memcpy(compact + (size_t)erased_k[i] * stride, par_blk + (size_t)i * stride, stride);

    uint64_t comp_t0 = mono_ns();
    vxfec_rs8_plan *cpu_plan = vxfec_rs8_decode_plan_new(NULL, K, M, erased_k, parity_m, drops);
    uint64_t comp_ns = mono_ns() - comp_t0;
    vxfec_rs8_plan_free(NULL, cpu_plan);

    dec_plan = vxfec_rs8_decode_plan_new(ctx, K, M, erased_k, parity_m, drops);
    if (!dec_plan)
        goto done;

    size_t d_bytes = (size_t)Z * blk_bytes;
    size_t o_bytes = (size_t)Z * out_blk_bytes;
    d_vram = vxfec_vk_buf_new(ctx, d_bytes, false);
    out_vram = vxfec_vk_buf_new(ctx, o_bytes, false);
    if (!d_vram || !out_vram || !populate_vram_blocks(ctx, d_vram, compact, blk_bytes, Z))
        goto done;

    uint64_t warm_t0 = mono_ns();
    do {
        if (!vxfec_rs8_exec_batch(ctx, dec_plan, d_vram, stride, Z, out_vram, NULL))
            goto done;
    } while ((mono_ns() - warm_t0) < 15000000ULL);

    uint32_t probe_z[3] = {0u, Z / 2u, Z - 1u};
    for (int p = 0; p < 3; p++) {
        if (!vxfec_vk_buf_read_offset(ctx, out_vram, (size_t)probe_z[p] * out_blk_bytes, rec_chk, out_blk_bytes))
            goto done;
        for (uint32_t i = 0; i < drops; i++) {
            if (memcmp(rec_chk + (size_t)i * stride, sys_orig + (size_t)erased_k[i] * stride, T) != 0) {
                fprintf(stderr, "FAIL: Cauchy RS8 superframe verify mismatch at loss=%u%%\n", loss_pct);
                goto done;
            }
        }
    }

    const int iters = 10;
    uint64_t gpu_ns_sum = 0;
    uint64_t wall_t0 = mono_ns();
    for (int it = 0; it < iters; it++) {
        uint64_t ns = 0;
        if (!vxfec_rs8_exec_batch(ctx, dec_plan, d_vram, stride, Z, out_vram, &ns))
            goto done;
        gpu_ns_sum += ns;
    }
    uint64_t wall_ns = mono_ns() - wall_t0;
    uint64_t eff_ns = (gpu_ns_sum > 0u) ? gpu_ns_sum : wall_ns;

    double payload_bits = 8.0 * (double)Z * (double)K * (double)T * iters;
    double repaired_bits = 8.0 * (double)Z * (double)drops * (double)T * iters;
    double gpu_gbps = payload_bits / (double)(eff_ns ? eff_ns : 1ULL);
    double wall_gbps = payload_bits / (double)(wall_ns ? wall_ns : 1ULL);
    double rep_gbps = repaired_bits / (double)(eff_ns ? eff_ns : 1ULL);

    printf("  %3u%% (%3u/%3u) | RS8  | %6.1f MiB (%4u) | %6.2f ms | %8.1f Gbps | %8.1f Gbps | %7.1f Gbps\n", loss_pct, drops, K,
           (double)d_bytes / (1024.0 * 1024.0), Z, (double)comp_ns / 1e6, gpu_gbps, wall_gbps, rep_gbps);
    ok = true;

done:
    vxfec_vk_buf_free(ctx, out_vram);
    vxfec_vk_buf_free(ctx, d_vram);
    vxfec_rs8_plan_free(ctx, dec_plan);
    free(rec_chk);
    free(compact);
    free(par_blk);
    free(sys_orig);
    return ok;
}

static bool run_e2e_burst_check(vxfec_vk_ctx *ctx, bool discrete, uint32_t T_wire)
{
    const uint32_t Z = discrete ? 128u : 16u;
    const uint32_t K = 200u;
    const uint32_t P = 48u;
    const uint64_t total_frames = (uint64_t)(K + P) * Z;
    const size_t sys_bytes = (size_t)Z * K * T_wire;
    const size_t wire_bytes = (size_t)Z * (K + P) * T_wire;
    const size_t wpb = ((size_t)(K + P) + 63u) / 64u;

    vxfec_wire_ring *ring = vxfec_wire_ring_new(ctx, VXFEC_WIRE_RATELESS, Z, K, P, T_wire, 0u);
    vxfec_vk_buf *wire_vram = vxfec_vk_buf_new(ctx, wire_bytes, false);
    uint8_t *sys_orig = (uint8_t *)malloc(sys_bytes);
    uint8_t *rx_out = (uint8_t *)calloc(1, sys_bytes);
    uint8_t *poison = (uint8_t *)malloc(T_wire);
    uint64_t *rx_bits = (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t));
    bool ok = false;
    if (!ring || !wire_vram || !sys_orig || !rx_out || !poison || !rx_bits)
        goto done;

    for (size_t i = 0; i < sys_bytes; i++)
        sys_orig[i] = (uint8_t)((i * 71u + (i >> 8u) * 37u + 13u) & 0xffu);
    memset(poison, 0xa5, T_wire);

    if (!vxfec_vk_buf_write_offset(ctx, wire_vram, 0, sys_orig, sys_bytes))
        goto done;
    uint64_t enc_gpu_ns = 0;
    if (!vxfec_wire_encode_vram(ring, wire_vram, &enc_gpu_ns))
        goto done;

    const uint64_t drop_frames = ((uint64_t)K * Z) / 10u;
    const uint64_t drop_beg = (uint64_t)Z * 3u + 17u;
    const uint64_t drop_end = drop_beg + drop_frames;

    for (uint64_t seq = 0; seq < total_frames; seq++) {
        if (seq >= drop_beg && seq < drop_end) {
            if (seq < (uint64_t)K * Z) {
                if (!vxfec_vk_buf_write_offset(ctx, wire_vram, (size_t)seq * T_wire, poison, T_wire))
                    goto done;
            }
            continue;
        }
        vxfec_wire_mark_seq(rx_bits, Z, wpb, seq);
    }

    vxfec_wire_stats st = {0};
    if (!vxfec_wire_recover_vram(ring, wire_vram, rx_bits, &st))
        goto done;
    if (!vxfec_vk_buf_read_offset(ctx, wire_vram, 0, rx_out, sys_bytes))
        goto done;

    if (st.blocks_repaired != Z || st.unique_masks > 3u || memcmp(rx_out, sys_orig, sys_bytes) != 0)
        goto done;

    double sys_bits = 8.0 * (double)sys_bytes;
    double gpu_gbps = (st.gpu_exec_ns > 0u) ? (sys_bits / (double)st.gpu_exec_ns) : 0.0;
    double e2e_gbps = (st.total_wall_ns > 0u) ? (sys_bits / (double)st.total_wall_ns) : 0.0;

    printf("  Superframe    : Z=%u x (K=%u + P=%u) x T=%u B (%.1f MiB payload)\n", Z, K, P, T_wire,
           (double)sys_bytes / (1024.0 * 1024.0));
    printf("  Burst Drop    : %" PRIu64 " contiguous wire frames erased and poisoned\n", drop_frames);
    printf("  Mask Dedup    : %u/%u blocks solved via %u unique block masks (100%% bit-exact)\n", st.blocks_repaired, Z,
           st.unique_masks);
    printf("  GPU Compute   : %.2f Gbps | CPU Compile + Sparse Tab DMA + GPU E2E: %.2f Gbps\n", gpu_gbps, e2e_gbps);
    ok = true;

done:
    free(rx_bits);
    free(poison);
    free(rx_out);
    free(sys_orig);
    vxfec_vk_buf_free(ctx, wire_vram);
    vxfec_wire_ring_free(ring);
    return ok;
}

int main(int argc, char **argv)
{
    char *end = NULL;
    errno = 0;
    bool custom_t = (argc > 1);
    unsigned long parsed = custom_t ? strtoul(argv[1], &end, 10) : 1280u;
    if (argc > 2 || (custom_t && (errno || end == argv[1] || *end)) || parsed > UINT32_MAX || parsed < 16 || (parsed & 3u) != 0 ||
        !nanorq_core_recommended_stride((uint32_t)parsed)) {
        fprintf(stderr, "T must be >= 16 and a multiple of 4\n");
        return 1;
    }
    uint32_t T = (uint32_t)parsed;

    int vk_status = VXFEC_VK_ERROR;
    vxfec_vk_ctx *ctx = vxfec_vk_ctx_new_status(&vk_status);
    if (!ctx) {
        fprintf(stderr, "vulkan compute %s\n", vk_status == VXFEC_VK_UNAVAILABLE ? "device unavailable" : "initialization failed");
        return 1;
    }

    bool discrete = vxfec_vk_is_discrete(ctx);
    size_t rfc_batch = discrete ? (custom_t ? (64ULL << 20) : (128ULL << 20)) : (16ULL << 20);
    size_t interleave_ring = discrete ? (custom_t ? (128ULL << 20) : (256ULL << 20)) : (32ULL << 20);
    size_t vram_ring = discrete ? (custom_t ? (256ULL << 20) : (1536ULL << 20)) : (32ULL << 20);
    size_t dma_bytes = discrete ? (custom_t ? (128ULL << 20) : (512ULL << 20)) : (16ULL << 20);
    uint32_t T_rq_super = custom_t ? T : 8192u;
    uint32_t T_rs_super = custom_t ? T : 4096u;
    uint32_t T_wire = (T_rs_super < 64u) ? 64u : ((T_rs_super + 15u) & ~15u);
    params p256 = params_init(256u);

    printf("=======================================================================================================\n");
    printf("vxfec Vulkan Compute & Resident-VRAM Superframe Benchmark\n");
    printf("Device : %s (%s, ACE=%d, LDS=%u B, maxSSBO=%.0f MiB, DMA-BUF=%d)\n", vxfec_vk_device_name(ctx),
           discrete ? "Discrete GPU" : "Integrated/CPU", (int)vxfec_vk_is_dedicated_compute(ctx), vxfec_vk_shared_mem_bytes(ctx),
           (double)vxfec_vk_max_storage_buffer_bytes(ctx) / (1024.0 * 1024.0), (int)vxfec_vk_has_dmabuf(ctx));
    printf("=======================================================================================================\n\n");

    printf("--- 1. Cauchy RS8 MDS (`ir_slack = 0`, Cached Plan, T = %u B) ---\n", T);
    printf("      K     M     Z | CPU 1-Core |  VK VRAM Direct | VK PCIe E2E\n");
    printf("                    |   (Gbps)   |     (Gbps)      |   (Gbps)   \n");
    printf("--------------------+------------+-----------------+-------------\n");
    if (!run_rs8_row(ctx, 64, 8, T, rfc_batch) || !run_rs8_row(ctx, 128, 16, T, rfc_batch) ||
        !run_rs8_row(ctx, 200, 20, T, rfc_batch))
        goto failed;

    printf("\n--- 2. RaptorQ RFC 6330 GF(256) 2-Stage Encode + 10%% Mix (T = %u B) ---\n", T);
    printf("      K     Z | CPU Run+Mx |  VK VRAM Run  | VK VRAM Run+Mix | VK PCIe E2E\n");
    printf("--------------+------------+---------------+-----------------+-------------\n");
    uint32_t ks[] = {100, 256, 500, 1000, 5000, 10000, 50000};
    size_t n_ks = (discrete && !custom_t) ? 7 : 6;
    for (size_t i = 0; i < n_ks; i++) {
        if (!run_encode_row(ctx, ks[i], T, rfc_batch, false))
            goto failed;
    }

#ifdef NANORQ_HAS_GF2_MODE
    printf("\n--- 3. RaptorQ H=0 Pure-GF(2) 2-Stage Encode + 10%% Mix (T = %u B) ---\n", T);
    printf("      K     Z | CPU Run+Mx |  VK VRAM Run  | VK VRAM Run+Mix | VK PCIe E2E\n");
    printf("--------------+------------+---------------+-----------------+-------------\n");
    for (size_t i = 0; i < n_ks; i++) {
        if (!run_encode_row(ctx, ks[i], T, rfc_batch, true))
            goto failed;
    }
#endif

    printf("\n--- 4. Uniform-Mask Batch Recovery (~10%% Burst, Cached DirectTab vs 2-Stage) ---\n");
    printf("     K      T     Z     N | Mode | CPU Run+Mix | VK 2-Stg Run | Direct GPU | Direct Wall | Rebuilt Out\n");
    printf("--------------------------+------+-------------+--------------+------------+-------------+------------\n");
    if (!run_interleave_row(ctx, 128, custom_t ? T : 4096, interleave_ring) ||
        !run_interleave_row(ctx, 256, custom_t ? T : 4096, interleave_ring) ||
        !run_interleave_row(ctx, 256, custom_t ? T : 8192, interleave_ring) ||
        !run_interleave_row(ctx, 512, custom_t ? T : 8192, interleave_ring))
        goto failed;

    printf("\n--- 5. Resident-VRAM Loss Sweep: RFC 6330 RaptorQ (K = 256, T = %u B) & Cauchy RS8 (K = 128, T = %u B) ---\n",
           T_rq_super, T_rs_super);
    printf("    Burst Loss    | Mode |    VRAM Ring (Z) | CPU Comp |  GPU Stream  |  Wall-Clock  | Rebuilt Out | Min RW B/W\n");
    printf("------------------+------+------------------+----------+--------------+--------------+-------------+------------\n");
    const uint32_t rq_losses[] = {5, 10, 20, 30, 50};
    for (size_t i = 0; i < sizeof(rq_losses) / sizeof(rq_losses[0]); i++) {
        if (!run_rq_superframe_row(ctx, 256, T_rq_super, rq_losses[i], 2u, vram_ring))
            goto failed;
    }
    for (size_t i = 0; i < sizeof(rq_losses) / sizeof(rq_losses[0]); i++) {
        if (!run_rq_superframe_row(ctx, 256, T_rq_super, rq_losses[i], p256.H + 8u, vram_ring))
            goto failed;
    }
    const uint32_t rs_losses[] = {5, 10, 20, 30};
    for (size_t i = 0; i < sizeof(rs_losses) / sizeof(rs_losses[0]); i++) {
        if (!run_rs8_superframe_row(ctx, 128, T_rs_super, rs_losses[i], vram_ring))
            goto failed;
    }

    double h2d_gbps = 0.0, d2h_gbps = 0.0;
    if (!measure_pcie_dma(ctx, dma_bytes, &h2d_gbps, &d2h_gbps))
        goto failed;

    printf("\n--- 6. End-to-End 2D Wire-Strided Burst Pipeline (`vxfec_wire_ring`) & Host Staging DMA ---\n");
    if (!run_e2e_burst_check(ctx, discrete, T_wire))
        goto failed;
    printf("  Staging DMA   : Host -> VRAM = %.1f Gbps (%.2f GB/s) | VRAM -> Host = %.1f Gbps (%.2f GB/s)\n", h2d_gbps,
           h2d_gbps / 8.0, d2h_gbps, d2h_gbps / 8.0);
    printf("=======================================================================================================\n");

    vxfec_vk_ctx_free(ctx);
    return 0;

failed:
    vxfec_vk_ctx_free(ctx);
    return 1;
}
