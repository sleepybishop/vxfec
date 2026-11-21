#include "vxfec_fec.h"
#include "bench_batch.h"
#include "oblas_lite.h"
#include "vxfec_rq.h"
#include "vxfec_rs_cauchy.h"
#include "vxfec_vk.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif

typedef struct {
    atomic_int *ready_cnt;
    atomic_bool *go;
    atomic_bool *abort_run;
    int tid;
    bool ok;
} init_race_arg_t;

static void *init_race_worker(void *arg)
{
    init_race_arg_t *a = (init_race_arg_t *)arg;
    atomic_fetch_add_explicit(a->ready_cnt, 1, memory_order_acq_rel);
    while (!atomic_load_explicit(a->go, memory_order_acquire)) {
    }
    if (atomic_load_explicit(a->abort_run, memory_order_acquire))
        return NULL;

    if ((a->tid & 1) == 0) {
        uint8_t sys[16 * 64], par[4 * 64];
        for (size_t i = 0; i < sizeof(sys); i++)
            sys[i] = (uint8_t)((i * 37u + (size_t)a->tid * 11u + 5u) & 0xffu);
        uint8_t erased_k[2] = {1u, 7u};
        uint8_t parity_m[2] = {0u, 2u};
        if (!vxfec_rs8_encode_cpu(16u, 4u, sys, 64u, NULL, 4u, par)) {
            a->ok = false;
            return NULL;
        }
        uint8_t compact[16 * 64];
        memcpy(compact, sys, sizeof(sys));
        memcpy(compact + 1u * 64u, par + 0u * 64u, 64u);
        memcpy(compact + 7u * 64u, par + 2u * 64u, 64u);
        a->ok =
            vxfec_rs8_decode_compact_cpu(16u, 4u, compact, 64u, erased_k, parity_m, 2u) && memcmp(compact, sys, sizeof(sys)) == 0;
    } else {
        nanorq_core rq = {0};
        if (!nanorq_core_encoder_new(16u, 0u, &rq)) {
            a->ok = false;
            return NULL;
        }
        size_t plen = nanorq_core_calculate_prepare_memory(&rq);
        size_t wlen = nanorq_core_calculate_work_memory(&rq);
        size_t slen = ops_estimate_schedule_bytes(16u);
        uint8_t *pm = (uint8_t *)malloc(plen);
        uint8_t *wm = (uint8_t *)malloc(wlen);
        void *sm = malloc(slen);
        schedule S = {0};
        uint32_t rows = nanorq_core_get_pc_rows(&rq);
        uint32_t sh = nanorq_core_get_pc_genc_offset(&rq);
        uint8_t *d = (uint8_t *)calloc(rows, 64u);
        uint8_t out[64] = {0};
        bool loc_ok = false;
        if (pm && wm && sm && d && schedule_init(&S, sm, slen) && nanorq_core_prepare(&rq, pm, plen)) {
            nanorq_core_set_op_callback(&rq, &S, ops_push);
            if (nanorq_core_precalculate(&rq, wm, wlen)) {
                for (uint32_t e = 0; e < 16u; e++) {
                    for (uint32_t c = 0; c < 64u; c++)
                        d[(size_t)(sh + e) * 64u + c] = (uint8_t)((e * 29u + c + 3u) & 0xffu);
                }
                ops_run(&rq, d, 64u, &S);
                ops_mix(&rq, d, 64u, 16u, out);
                loc_ok = true;
            }
        }
        free(d);
        free(sm);
        free(wm);
        free(pm);
        a->ok = loc_ok;
    }
    return NULL;
}

static bool test_concurrent_first_use_init(void)
{
    pthread_t thrs[4];
    init_race_arg_t args[4];
    atomic_int ready_cnt = 0;
    atomic_bool go = false;
    atomic_bool abort_run = false;

    for (int i = 0; i < 4; i++) {
        args[i].ready_cnt = &ready_cnt;
        args[i].go = &go;
        args[i].abort_run = &abort_run;
        args[i].tid = i;
        args[i].ok = false;
        if (pthread_create(&thrs[i], NULL, init_race_worker, &args[i]) != 0) {
            /* release and join already-started workers before stack state expires */
            atomic_store_explicit(&abort_run, true, memory_order_release);
            atomic_store_explicit(&go, true, memory_order_release);
            for (int j = 0; j < i; j++)
                pthread_join(thrs[j], NULL);
            return false;
        }
    }
    while (atomic_load_explicit(&ready_cnt, memory_order_acquire) < 4) {
    }
    atomic_store_explicit(&go, true, memory_order_release);

    bool all_ok = true;
    for (int i = 0; i < 4; i++) {
        pthread_join(thrs[i], NULL);
        if (!args[i].ok)
            all_ok = false;
    }
    return all_ok;
}

static bool test_benchmark_batch_planner(void)
{
    static const struct {
        const char *name;
        size_t target, range, matrix, output;
        uint32_t cap, expected;
    } cases[] = {
        {"target caps batch", 9000, 50000, 1000, 100, 512, 9},
        {"matrix range caps batch", 32u << 20, 16u << 20, 1u << 20, 1u << 18, 1024, 16},
        {"output range caps batch", 20000, 10000, 1000, 3000, 512, 3},
        {"configured cap", 20000, 20000, 1000, 1000, 4, 4},
        {"target below one block", 1, 5000, 1000, 100, 512, 1},
        {"one block fits", 10000, 1999, 1000, 100, 512, 1},
        {"two blocks fit", 10000, 2999, 1000, 100, 512, 2},
        {"three blocks fit", 10000, 3999, 1000, 100, 512, 3},
        {"matrix block too large", 10000, 999, 1000, 100, 512, 0},
        {"output block too large", 10000, 999, 100, 1000, 512, 0},
        {"zero target", 0, 10000, 1000, 100, 512, 1},
        {"zero range", 10000, 0, 1000, 100, 512, 0},
        {"zero matrix", 10000, 10000, 0, 100, 512, 0},
        {"zero output", 10000, 10000, 1000, 0, 512, 0},
        {"zero cap", 10000, 10000, 1000, 100, 0, 0},
        {"size narrowing", SIZE_MAX, SIZE_MAX, 1, 1, UINT32_MAX, UINT32_MAX},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint32_t got = vxfec_bench_blocks(cases[i].target, cases[i].range, cases[i].matrix, cases[i].output, cases[i].cap);
        if (got != cases[i].expected) {
            fprintf(stderr, "FAIL: batch planner %s: expected %u, got %u\n", cases[i].name, cases[i].expected, got);
            return false;
        }
    }
    return true;
}

static void set_vk_device_env(const char *val)
{
#ifdef _WIN32
    _putenv_s("VXFEC_VK_DEVICE", val ? val : "");
#else
    if (val)
        setenv("VXFEC_VK_DEVICE", val, 1);
    else
        unsetenv("VXFEC_VK_DEVICE");
#endif
}

static bool test_rq_run_host_buf(vxfec_vk_ctx *ctx, nanorq_core *rq, const vxfec_rq_sched *gs, uint8_t *d_host, uint32_t stride,
                                 uint32_t num_blocks)
{
    uint32_t rows = nanorq_core_get_pc_rows(rq);
    size_t bytes = (size_t)num_blocks * rows * stride;
    vxfec_vk_buf *buf = vxfec_vk_buf_new(ctx, bytes, true);
    if (!buf)
        return false;
    memcpy(vxfec_vk_buf_staging_ptr(buf), d_host, bytes);
    bool ok = vxfec_vk_buf_upload(ctx, buf, bytes) && vxfec_rq_ops_run_batch(ctx, gs, buf, stride, num_blocks, NULL) &&
              vxfec_vk_buf_download(ctx, buf, bytes);
    if (ok)
        memcpy(d_host, vxfec_vk_buf_staging_ptr(buf), bytes);
    vxfec_vk_buf_free(ctx, buf);
    return ok;
}

static bool test_rq_mix_host_buf(vxfec_vk_ctx *ctx, nanorq_core *rq, const uint8_t *d_host, uint32_t stride, uint32_t num_blocks,
                                 const uint32_t *esis, uint32_t num_esis, uint8_t *out_host)
{
    uint32_t rows = nanorq_core_get_pc_rows(rq);
    size_t d_bytes = (size_t)num_blocks * rows * stride;
    size_t out_bytes = (size_t)num_blocks * num_esis * stride;
    vxfec_vk_buf *d_buf = vxfec_vk_buf_new(ctx, d_bytes, true);
    vxfec_vk_buf *out_buf = vxfec_vk_buf_new(ctx, out_bytes, true);
    if (!d_buf || !out_buf) {
        vxfec_vk_buf_free(ctx, out_buf);
        vxfec_vk_buf_free(ctx, d_buf);
        return false;
    }
    memcpy(vxfec_vk_buf_staging_ptr(d_buf), d_host, d_bytes);
    bool ok = vxfec_vk_buf_upload(ctx, d_buf, d_bytes) &&
              vxfec_rq_ops_mix_batch(ctx, rq, d_buf, stride, num_blocks, esis, num_esis, out_buf, NULL) &&
              vxfec_vk_buf_download(ctx, out_buf, out_bytes);
    if (ok)
        memcpy(out_host, vxfec_vk_buf_staging_ptr(out_buf), out_bytes);
    vxfec_vk_buf_free(ctx, out_buf);
    vxfec_vk_buf_free(ctx, d_buf);
    return ok;
}

static bool test_rq_direct_host_buf(vxfec_vk_ctx *ctx, uint32_t rows, const vxfec_rq_direct *gd, const uint8_t *d_host,
                                    uint32_t stride, uint32_t num_blocks, uint32_t num_esis, uint8_t *out_host)
{
    size_t d_bytes = (size_t)num_blocks * rows * stride;
    size_t out_bytes = (size_t)num_blocks * num_esis * stride;
    vxfec_vk_buf *d_buf = vxfec_vk_buf_new(ctx, d_bytes, true);
    vxfec_vk_buf *out_buf = vxfec_vk_buf_new(ctx, out_bytes, true);
    if (!d_buf || !out_buf) {
        vxfec_vk_buf_free(ctx, out_buf);
        vxfec_vk_buf_free(ctx, d_buf);
        return false;
    }
    memcpy(vxfec_vk_buf_staging_ptr(d_buf), d_host, d_bytes);
    bool ok = vxfec_vk_buf_upload(ctx, d_buf, d_bytes) &&
              vxfec_rq_direct_batch(ctx, gd, d_buf, stride, num_blocks, out_buf, NULL) &&
              vxfec_vk_buf_download(ctx, out_buf, out_bytes);
    if (ok)
        memcpy(out_host, vxfec_vk_buf_staging_ptr(out_buf), out_bytes);
    vxfec_vk_buf_free(ctx, out_buf);
    vxfec_vk_buf_free(ctx, d_buf);
    return ok;
}

static bool test_encode_case(vxfec_vk_ctx *ctx, uint32_t K, uint32_t T, uint32_t num_blocks, uint32_t num_repairs)
{
    nanorq_core rq = {0};
    if (!nanorq_core_encoder_new(K, 0, &rq))
        return false;

    bool ok = false;
    size_t prep_len = nanorq_core_calculate_prepare_memory(&rq);
    size_t work_len = nanorq_core_calculate_work_memory(&rq);
    size_t sched_len = ops_estimate_schedule_bytes(K);
    uint8_t *prep_mem = (uint8_t *)malloc(prep_len);
    uint8_t *work_mem = (uint8_t *)malloc(work_len);
    void *sched_mem = malloc(sched_len);
    uint8_t *d_cpu = NULL, *d_gpu = NULL, *d_orig = NULL;
    uint8_t *rep_cpu = NULL, *rep_gpu = NULL;
    uint32_t *esis = NULL;
    vxfec_rq_sched *gs = NULL;
    vxfec_rq_direct *gd = NULL;
    vxfec_vk_buf *vb_d = NULL, *vb_o = NULL;
    schedule S = {0};
    if (!prep_mem || !work_mem || !schedule_init(&S, sched_mem, sched_len))
        goto cleanup;

    if (!nanorq_core_prepare(&rq, prep_mem, prep_len))
        goto cleanup;
    nanorq_core_set_op_callback(&rq, &S, ops_push);
    if (!nanorq_core_precalculate(&rq, work_mem, work_len))
        goto cleanup;

    uint32_t rows = nanorq_core_get_pc_rows(&rq);
    uint32_t sh = nanorq_core_get_pc_genc_offset(&rq);
    uint32_t stride = (T == 20u) ? 20u : nanorq_core_recommended_stride(T);
    size_t block_bytes = (size_t)rows * stride;
    size_t total_bytes = (size_t)num_blocks * block_bytes;

    d_cpu = (uint8_t *)calloc(1, total_bytes);
    d_gpu = (uint8_t *)calloc(1, total_bytes);
    d_orig = (uint8_t *)malloc(total_bytes);
    if (!d_cpu || !d_gpu || !d_orig)
        goto cleanup;
    for (uint32_t b = 0; b < num_blocks; b++) {
        uint8_t *base = d_cpu + (size_t)b * block_bytes;
        for (uint32_t esi = 0; esi < K; esi++) {
            uint8_t *row = base + (size_t)(sh + esi) * stride;
            for (uint32_t col = 0; col < T; col++) {
                row[col] = (uint8_t)((b * 131u + esi * 37u + col * 17u + 5u) & 0xffu);
            }
        }
    }
    memcpy(d_orig, d_cpu, total_bytes);
    memcpy(d_gpu, d_cpu, total_bytes);

    for (uint32_t b = 0; b < num_blocks; b++) {
        if (!ops_run_pivoted(d_cpu + (size_t)b * block_bytes, stride, &S))
            goto cleanup;
    }

    gs = vxfec_rq_sched_new(ctx, &rq, &S);
    if (!gs || !test_rq_run_host_buf(ctx, &rq, gs, d_gpu, stride, num_blocks))
        goto cleanup;

    uint32_t sys_checks = K < 32u ? K : 32u;
    uint32_t total_esis = sys_checks + num_repairs;
    esis = (uint32_t *)malloc(total_esis * sizeof(*esis));
    size_t mix_bytes = (size_t)num_blocks * total_esis * stride;
    rep_cpu = (uint8_t *)calloc(1, mix_bytes);
    rep_gpu = (uint8_t *)calloc(1, mix_bytes);
    if (!esis || !rep_cpu || !rep_gpu)
        goto cleanup;
    for (uint32_t s = 0; s < sys_checks; s++)
        esis[s] = (uint32_t)(((uint64_t)s * K) / sys_checks);
    for (uint32_t e = 0; e < num_repairs; e++)
        esis[sys_checks + e] = K + e;

    for (uint32_t b = 0; b < num_blocks; b++) {
        for (uint32_t e = 0; e < total_esis; e++) {
            ops_mix_pivoted(&rq, d_cpu + (size_t)b * block_bytes, stride, esis[e], rep_cpu + ((size_t)b * total_esis + e) * stride);
        }
    }

    /* ops_run_pivoted shares identical virtual row layout with GPU ops_mix_batch */
    if (!test_rq_mix_host_buf(ctx, &rq, d_cpu, stride, num_blocks, esis, total_esis, rep_gpu))
        goto cleanup;
    for (uint32_t b = 0; b < num_blocks; b++) {
        for (uint32_t e = 0; e < total_esis; e++) {
            const uint8_t *mc = rep_cpu + ((size_t)b * total_esis + e) * stride;
            const uint8_t *mg = rep_gpu + ((size_t)b * total_esis + e) * stride;
            if (memcmp(mc, mg, T) != 0) {
                fprintf(stderr, "CPU-pivoted Vulkan-mix mismatch at K=%u T=%u block=%u esi=%u\n", K, T, b, esis[e]);
                goto cleanup;
            }
        }
    }
    memset(rep_gpu, 0, mix_bytes);

    if (!test_rq_mix_host_buf(ctx, &rq, d_gpu, stride, num_blocks, esis, total_esis, rep_gpu))
        goto cleanup;

    for (uint32_t b = 0; b < num_blocks; b++) {
        for (uint32_t e = 0; e < total_esis; e++) {
            const uint8_t *mc = rep_cpu + ((size_t)b * total_esis + e) * stride;
            const uint8_t *mg = rep_gpu + ((size_t)b * total_esis + e) * stride;
            if (memcmp(mc, mg, T) != 0) {
                fprintf(stderr, "ops_mix mismatch at K=%u T=%u block=%u esi=%u\n", K, T, b, esis[e]);
                goto cleanup;
            }
        }
    }

    /* verify precompiled direct GF(256) operator against rep_cpu */
    gd = vxfec_rq_direct_new(ctx, &rq, &S, esis, total_esis);
    memset(rep_gpu, 0, mix_bytes);
    if (!gd || vxfec_rq_direct_is_pure_gf2(gd) ||
        !test_rq_direct_host_buf(ctx, rows, gd, d_orig, stride, num_blocks, total_esis, rep_gpu)) {
        fprintf(stderr, "direct_batch failed at K=%u T=%u\n", K, T);
        goto cleanup;
    }
    for (uint32_t b = 0; b < num_blocks; b++) {
        for (uint32_t e = 0; e < total_esis; e++) {
            const uint8_t *mc = rep_cpu + ((size_t)b * total_esis + e) * stride;
            const uint8_t *mg = rep_gpu + ((size_t)b * total_esis + e) * stride;
            if (memcmp(mc, mg, T) != 0) {
                fprintf(stderr, "direct GF(256) mismatch at K=%u T=%u block=%u esi=%u\n", K, T, b, esis[e]);
                goto cleanup;
            }
        }
    }

    /* verify pipelined run_and_mix (exercises both auto-direct and tiled paths)
     */
    vb_d = vxfec_vk_buf_new(ctx, total_bytes, true);
    vb_o = vxfec_vk_buf_new(ctx, mix_bytes, true);
    if (!vb_d || !vb_o)
        goto cleanup;
    memcpy(vxfec_vk_buf_staging_ptr(vb_d), d_orig, total_bytes);
    if (!vxfec_rq_run_and_mix_pipelined(ctx, &rq, gs, vb_d, stride, num_blocks, esis, total_esis, vb_o, true, true, NULL))
        goto cleanup;
    const uint8_t *pipe_out = (const uint8_t *)vxfec_vk_buf_staging_ptr(vb_o);
    for (uint32_t b = 0; b < num_blocks; b++) {
        for (uint32_t e = 0; e < total_esis; e++) {
            const uint8_t *mc = rep_cpu + ((size_t)b * total_esis + e) * stride;
            const uint8_t *mg = pipe_out + ((size_t)b * total_esis + e) * stride;
            if (memcmp(mc, mg, T) != 0) {
                fprintf(stderr, "pipelined mismatch at K=%u T=%u block=%u esi=%u\n", K, T, b, esis[e]);
                goto cleanup;
            }
        }
    }
    ok = true;

cleanup:
    vxfec_vk_buf_free(ctx, vb_o);
    vxfec_vk_buf_free(ctx, vb_d);
    vxfec_rq_direct_free(ctx, gd);
    vxfec_rq_sched_free(ctx, gs);
    free(rep_gpu);
    free(rep_cpu);
    free(esis);
    free(d_orig);
    free(d_gpu);
    free(d_cpu);
    free(sched_mem);
    free(work_mem);
    free(prep_mem);
    return ok;
}

static bool test_decode_gf2_case(vxfec_vk_ctx *ctx, uint32_t K, uint32_t T)
{
    uint32_t drops = K / 10;
    if (drops == 0)
        drops = 1;
    uint32_t overhead = K / 20;
    if (overhead < 16)
        overhead = 16; /* 5% slack so solve_gf2 produces a pure GF(2) schedule */

    nanorq_core enc = {0};
    if (!nanorq_core_encoder_new(K, 0, &enc))
        return false;

    bool ok = false;
    size_t prep_len = nanorq_core_calculate_prepare_memory(&enc);
    size_t work_len = nanorq_core_calculate_work_memory(&enc);
    size_t sched_len = ops_estimate_schedule_bytes(K);
    uint8_t *prep_mem = (uint8_t *)malloc(prep_len);
    uint8_t *work_mem = (uint8_t *)malloc(work_len);
    void *sched_mem = malloc(sched_len);
    uint8_t *src_orig = NULL, *d_enc = NULL, *repairs = NULL;
    uint8_t *dec_prep = NULL, *dec_work = NULL, *d_dec = NULL, *recovered = NULL;
    void *dec_sched = NULL;
    uint32_t *drop_esis = NULL;
    vxfec_rq_direct *gd = NULL;
    vxfec_rq_sched *gs = NULL;
    schedule S_enc = {0}, S_dec = {0};
    if (!prep_mem || !work_mem || !schedule_init(&S_enc, sched_mem, sched_len))
        goto cleanup;
    if (!nanorq_core_prepare(&enc, prep_mem, prep_len))
        goto cleanup;
    nanorq_core_set_op_callback(&enc, &S_enc, ops_push);
    if (!nanorq_core_precalculate(&enc, work_mem, work_len))
        goto cleanup;

    uint32_t rows_enc = nanorq_core_get_pc_rows(&enc);
    uint32_t sh = nanorq_core_get_pc_genc_offset(&enc);
    uint32_t stride = (T == 20u) ? 20u : nanorq_core_recommended_stride(T);
    src_orig = (uint8_t *)calloc(K, stride);
    d_enc = (uint8_t *)calloc(rows_enc, stride);
    uint32_t total_rep = drops + overhead;
    repairs = (uint8_t *)calloc(total_rep, stride);
    if (!src_orig || !d_enc || !repairs)
        goto cleanup;
    for (uint32_t esi = 0; esi < K; esi++) {
        for (uint32_t col = 0; col < T; col++) {
            src_orig[(size_t)esi * stride + col] = (uint8_t)((esi * 53u + col * 29u + 11u) & 0xffu);
        }
        memcpy(d_enc + (size_t)(sh + esi) * stride, src_orig + (size_t)esi * stride, T);
    }
    ops_run(&enc, d_enc, stride, &S_enc);

    for (uint32_t r = 0; r < total_rep; r++) {
        ops_mix(&enc, d_enc, stride, K + r, repairs + (size_t)r * stride);
    }

    nanorq_core dec = {0};
    if (!nanorq_core_encoder_new(K, overhead, &dec))
        goto cleanup;
    size_t dec_prep_len = nanorq_core_calculate_prepare_memory(&dec);
    size_t dec_work_len = nanorq_core_calculate_work_memory(&dec);
    size_t dec_sched_len = ops_estimate_schedule_bytes(K + overhead);
    dec_prep = (uint8_t *)malloc(dec_prep_len);
    dec_work = (uint8_t *)malloc(dec_work_len);
    dec_sched = malloc(dec_sched_len);
    if (!dec_prep || !dec_work || !schedule_init(&S_dec, dec_sched, dec_sched_len))
        goto cleanup;

    if (!nanorq_core_prepare(&dec, dec_prep, dec_prep_len))
        goto cleanup;
    for (uint32_t r = 0; r < drops; r++)
        nanorq_core_replace_symbol(&dec, r, K + overhead + r);
    for (uint32_t i = 0; i < overhead; i++)
        nanorq_core_replace_symbol(&dec, K + i, K + i);
    if (!nanorq_core_patch_matrix(&dec))
        goto cleanup;

    nanorq_core_set_op_callback(&dec, &S_dec, ops_push);
    if (!nanorq_core_precalculate(&dec, dec_work, dec_work_len))
        goto cleanup;

    uint32_t rows_dec = nanorq_core_get_pc_rows(&dec);
    d_dec = (uint8_t *)calloc(rows_dec, stride);
    drop_esis = (uint32_t *)malloc(drops * sizeof(*drop_esis));
    recovered = (uint8_t *)calloc(drops, stride);
    if (!d_dec || !drop_esis || !recovered)
        goto cleanup;
    for (uint32_t esi = drops; esi < K; esi++) {
        memcpy(d_dec + (size_t)(sh + esi) * stride, src_orig + (size_t)esi * stride, T);
    }
    for (uint32_t r = 0; r < drops; r++) {
        memcpy(d_dec + (size_t)(sh + r) * stride, repairs + (size_t)(overhead + r) * stride, T);
    }
    for (uint32_t i = 0; i < overhead; i++) {
        memcpy(d_dec + (size_t)(sh + K + i) * stride, repairs + (size_t)i * stride, T);
    }

    for (uint32_t r = 0; r < drops; r++)
        drop_esis[r] = r;

    /* verify precompiled direct pure-GF(2) operator before mutating d_dec */
    gd = vxfec_rq_direct_new(ctx, &dec, &S_dec, drop_esis, drops);
    if (!gd || !vxfec_rq_direct_is_pure_gf2(gd) ||
        !test_rq_direct_host_buf(ctx, rows_dec, gd, d_dec, stride, 1, drops, recovered)) {
        fprintf(stderr, "direct GF(2) batch failed for K=%u\n", K);
        goto cleanup;
    }
    for (uint32_t r = 0; r < drops; r++) {
        if (memcmp(recovered + (size_t)r * stride, src_orig + (size_t)r * stride, T) != 0) {
            fprintf(stderr, "direct GF(2) decode mismatch at K=%u drop=%u\n", K, r);
            goto cleanup;
        }
    }

    gs = vxfec_rq_sched_new(ctx, &dec, &S_dec);
    if (!gs || !vxfec_rq_sched_is_pure_gf2(gs)) {
        fprintf(stderr, "expected pure GF(2) schedule for K=%u oh=%u\n", K, overhead);
        goto cleanup;
    }
    if (!test_rq_run_host_buf(ctx, &dec, gs, d_dec, stride, 1))
        goto cleanup;

    memset(recovered, 0, (size_t)drops * stride);
    if (!test_rq_mix_host_buf(ctx, &dec, d_dec, stride, 1, drop_esis, drops, recovered))
        goto cleanup;

    for (uint32_t r = 0; r < drops; r++) {
        if (memcmp(recovered + (size_t)r * stride, src_orig + (size_t)r * stride, T) != 0) {
            fprintf(stderr, "GF(2) decode mismatch at K=%u drop=%u\n", K, r);
            goto cleanup;
        }
    }
    ok = true;

cleanup:
    vxfec_rq_sched_free(ctx, gs);
    vxfec_rq_direct_free(ctx, gd);
    free(recovered);
    free(drop_esis);
    free(d_dec);
    free(dec_sched);
    free(dec_work);
    free(dec_prep);
    free(repairs);
    free(d_enc);
    free(src_orig);
    free(sched_mem);
    free(work_mem);
    free(prep_mem);
    return ok;
}

#ifdef NANORQ_HAS_GF2_MODE
static bool test_h0_gf2_roundtrip(vxfec_vk_ctx *ctx, uint32_t K, uint32_t T)
{
    uint32_t drops = K / 10;
    if (drops == 0)
        drops = 1;
    uint32_t overhead = K / 20;
    if (overhead < 16)
        overhead = 16;

    nanorq_core enc = {0};
    if (!nanorq_core_encoder_new_gf2(K, 0, &enc) || enc.P.H != 0)
        return false;

    bool ok = false;
    size_t prep_len = nanorq_core_calculate_prepare_memory(&enc);
    size_t work_len = nanorq_core_calculate_work_memory(&enc);
    size_t sched_len = ops_estimate_schedule_bytes(K);
    uint8_t *prep_mem = (uint8_t *)malloc(prep_len);
    uint8_t *work_mem = (uint8_t *)malloc(work_len);
    void *sched_mem = malloc(sched_len);
    uint8_t *src_orig = NULL, *d_cpu = NULL, *d_gpu = NULL;
    uint8_t *rep_cpu = NULL, *rep_gpu = NULL;
    uint8_t *dec_prep = NULL, *dec_work = NULL, *d_dec = NULL, *recovered = NULL;
    void *dec_sched = NULL;
    uint32_t *rep_esis = NULL, *drop_esis = NULL;
    vxfec_rq_sched *gs_enc = NULL, *gs_dec = NULL;
    schedule S_enc = {0}, S_dec = {0};
    if (!prep_mem || !work_mem || !schedule_init(&S_enc, sched_mem, sched_len))
        goto cleanup;

    if (!nanorq_core_prepare(&enc, prep_mem, prep_len))
        goto cleanup;
    nanorq_core_set_op_callback(&enc, &S_enc, ops_push);
    if (!nanorq_core_precalculate(&enc, work_mem, work_len))
        goto cleanup;

    /* H=0 encode schedule must be 100% pure GF(2) XORs */
    gs_enc = vxfec_rq_sched_new(ctx, &enc, &S_enc);
    if (!gs_enc || !vxfec_rq_sched_is_pure_gf2(gs_enc)) {
        fprintf(stderr, "expected pure GF(2) H=0 encode schedule for K=%u\n", K);
        goto cleanup;
    }

    uint32_t rows_enc = nanorq_core_get_pc_rows(&enc);
    uint32_t sh = nanorq_core_get_pc_genc_offset(&enc);
    uint32_t stride = (T == 20u) ? 20u : nanorq_core_recommended_stride(T);
    src_orig = (uint8_t *)calloc(K, stride);
    d_cpu = (uint8_t *)calloc(rows_enc, stride);
    d_gpu = (uint8_t *)calloc(rows_enc, stride);
    if (!src_orig || !d_cpu || !d_gpu)
        goto cleanup;
    for (uint32_t esi = 0; esi < K; esi++) {
        for (uint32_t col = 0; col < T; col++)
            src_orig[(size_t)esi * stride + col] = (uint8_t)((esi * 61u + col * 19u + 13u) & 0xffu);
        memcpy(d_cpu + (size_t)(sh + esi) * stride, src_orig + (size_t)esi * stride, T);
    }
    memcpy(d_gpu, d_cpu, (size_t)rows_enc * stride);

    ops_run(&enc, d_cpu, stride, &S_enc);
    if (!test_rq_run_host_buf(ctx, &enc, gs_enc, d_gpu, stride, 1))
        goto cleanup;

    uint32_t total_rep = drops + overhead;
    rep_esis = (uint32_t *)malloc(total_rep * sizeof(*rep_esis));
    rep_cpu = (uint8_t *)calloc(total_rep, stride);
    rep_gpu = (uint8_t *)calloc(total_rep, stride);
    if (!rep_esis || !rep_cpu || !rep_gpu)
        goto cleanup;
    for (uint32_t r = 0; r < total_rep; r++)
        rep_esis[r] = K + r;

    for (uint32_t r = 0; r < total_rep; r++)
        ops_mix(&enc, d_cpu, stride, rep_esis[r], rep_cpu + (size_t)r * stride);
    if (!test_rq_mix_host_buf(ctx, &enc, d_gpu, stride, 1, rep_esis, total_rep, rep_gpu))
        goto cleanup;

    for (uint32_t r = 0; r < total_rep; r++) {
        if (memcmp(rep_cpu + (size_t)r * stride, rep_gpu + (size_t)r * stride, T) != 0) {
            fprintf(stderr, "H=0 encode mix mismatch at K=%u r=%u\n", K, r);
            goto cleanup;
        }
    }

    /* decode on Vulkan using H=0 pure-GF(2) decoder */
    nanorq_core dec = {0};
    if (!nanorq_core_encoder_new_gf2(K, overhead, &dec))
        goto cleanup;
    size_t dec_prep_len = nanorq_core_calculate_prepare_memory(&dec);
    size_t dec_work_len = nanorq_core_calculate_work_memory(&dec);
    size_t dec_sched_len = ops_estimate_schedule_bytes(K + overhead);
    dec_prep = (uint8_t *)malloc(dec_prep_len);
    dec_work = (uint8_t *)malloc(dec_work_len);
    dec_sched = malloc(dec_sched_len);
    if (!dec_prep || !dec_work || !schedule_init(&S_dec, dec_sched, dec_sched_len))
        goto cleanup;

    if (!nanorq_core_generate(&dec, dec_prep, dec_prep_len))
        goto cleanup;
    for (uint32_t r = 0; r < drops; r++)
        nanorq_core_replace_symbol(&dec, r, K + overhead + r);
    for (uint32_t i = 0; i < overhead; i++)
        nanorq_core_replace_symbol(&dec, dec.P.Kprime + i, K + i);
    if (!nanorq_core_patch_matrix(&dec))
        goto cleanup;
    nanorq_core_set_op_callback(&dec, &S_dec, ops_push);
    if (!nanorq_core_precalculate(&dec, dec_work, dec_work_len))
        goto cleanup;

    gs_dec = vxfec_rq_sched_new(ctx, &dec, &S_dec);
    if (!gs_dec || !vxfec_rq_sched_is_pure_gf2(gs_dec))
        goto cleanup;

    uint32_t rows_dec = nanorq_core_get_pc_rows(&dec);
    d_dec = (uint8_t *)calloc(rows_dec, stride);
    drop_esis = (uint32_t *)malloc(drops * sizeof(*drop_esis));
    recovered = (uint8_t *)calloc(drops, stride);
    if (!d_dec || !drop_esis || !recovered)
        goto cleanup;
    for (uint32_t esi = drops; esi < K; esi++)
        memcpy(d_dec + (size_t)(sh + esi) * stride, src_orig + (size_t)esi * stride, T);
    for (uint32_t r = 0; r < drops; r++)
        memcpy(d_dec + (size_t)(sh + r) * stride, rep_gpu + (size_t)(overhead + r) * stride, T);
    for (uint32_t i = 0; i < overhead; i++)
        memcpy(d_dec + (size_t)(sh + dec.P.Kprime + i) * stride, rep_gpu + (size_t)i * stride, T);

    if (!test_rq_run_host_buf(ctx, &dec, gs_dec, d_dec, stride, 1))
        goto cleanup;

    for (uint32_t r = 0; r < drops; r++)
        drop_esis[r] = r;
    if (!test_rq_mix_host_buf(ctx, &dec, d_dec, stride, 1, drop_esis, drops, recovered))
        goto cleanup;

    for (uint32_t r = 0; r < drops; r++) {
        if (memcmp(recovered + (size_t)r * stride, src_orig + (size_t)r * stride, T) != 0) {
            fprintf(stderr, "H=0 GF(2) decode mismatch at K=%u drop=%u\n", K, r);
            goto cleanup;
        }
    }
    ok = true;

cleanup:
    vxfec_rq_sched_free(ctx, gs_dec);
    vxfec_rq_sched_free(ctx, gs_enc);
    free(recovered);
    free(drop_esis);
    free(d_dec);
    free(dec_sched);
    free(dec_work);
    free(dec_prep);
    free(rep_gpu);
    free(rep_cpu);
    free(rep_esis);
    free(d_gpu);
    free(d_cpu);
    free(src_orig);
    free(sched_mem);
    free(work_mem);
    free(prep_mem);
    return ok;
}
#endif

static bool test_exact_k_gf256_decode(vxfec_vk_ctx *ctx, uint32_t K, uint32_t T, uint32_t stride, uint32_t drops)
{
    nanorq_core enc = {0};
    if (!nanorq_core_encoder_new(K, 0, &enc))
        return false;
    bool ok = false;
    size_t prep_len = nanorq_core_calculate_prepare_memory(&enc);
    size_t work_len = nanorq_core_calculate_work_memory(&enc);
    size_t sched_len = ops_estimate_schedule_bytes(K);
    uint8_t *prep_mem = (uint8_t *)malloc(prep_len);
    uint8_t *work_mem = (uint8_t *)malloc(work_len);
    void *sched_mem = malloc(sched_len);
    uint8_t *src_orig = NULL, *d_enc = NULL, *repairs = NULL;
    uint32_t *rep_esis = NULL;
    vxfec_rq_sched *gs_enc = NULL;
    schedule S_enc = {0};
    if (!prep_mem || !work_mem || !schedule_init(&S_enc, sched_mem, sched_len))
        goto cleanup;
    if (!nanorq_core_prepare(&enc, prep_mem, prep_len))
        goto cleanup;
    nanorq_core_set_op_callback(&enc, &S_enc, ops_push);
    if (!nanorq_core_precalculate(&enc, work_mem, work_len))
        goto cleanup;

    uint32_t rows = nanorq_core_get_pc_rows(&enc);
    uint32_t sh = nanorq_core_get_pc_genc_offset(&enc);
    src_orig = (uint8_t *)calloc(K, stride);
    d_enc = (uint8_t *)calloc(rows, stride);
    rep_esis = (uint32_t *)malloc(drops * sizeof(uint32_t));
    repairs = (uint8_t *)calloc(drops, stride);
    if (!src_orig || !d_enc || !rep_esis || !repairs)
        goto cleanup;
    for (uint32_t esi = 0; esi < K; esi++) {
        for (uint32_t c = 0; c < T; c++)
            src_orig[(size_t)esi * stride + c] = (uint8_t)((esi * 73u + c * 29u + 9u) & 0xffu);
        memcpy(d_enc + (size_t)(sh + esi) * stride, src_orig + (size_t)esi * stride, T);
    }
    gs_enc = vxfec_rq_sched_new(ctx, &enc, &S_enc);
    if (!gs_enc || !test_rq_run_host_buf(ctx, &enc, gs_enc, d_enc, stride, 1))
        goto cleanup;

    for (uint32_t attempt = 0; attempt < 8 && !ok; attempt++) {
        for (uint32_t r = 0; r < drops; r++)
            rep_esis[r] = K + attempt * drops + r;
        if (!test_rq_mix_host_buf(ctx, &enc, d_enc, stride, 1, rep_esis, drops, repairs))
            break;

        nanorq_core dec = {0};
        if (!nanorq_core_encoder_new(K, 0, &dec))
            break;
        uint8_t *dec_prep = (uint8_t *)malloc(prep_len);
        uint8_t *dec_work = (uint8_t *)malloc(work_len);
        void *dec_sched = malloc(sched_len);
        schedule S_dec = {0};
        if (!dec_prep || !dec_work || !schedule_init(&S_dec, dec_sched, sched_len) ||
            !nanorq_core_prepare(&dec, dec_prep, prep_len)) {
            free(dec_sched);
            free(dec_work);
            free(dec_prep);
            break;
        }
        for (uint32_t r = 0; r < drops; r++)
            nanorq_core_replace_symbol(&dec, r, rep_esis[r]);
        if (!nanorq_core_patch_matrix(&dec)) {
            free(dec_sched);
            free(dec_work);
            free(dec_prep);
            continue;
        }
        nanorq_core_set_op_callback(&dec, &S_dec, ops_push);
        if (!nanorq_core_precalculate(&dec, dec_work, work_len)) {
            free(dec_sched);
            free(dec_work);
            free(dec_prep);
            continue;
        }

        uint8_t *d_dec = (uint8_t *)calloc(rows, stride);
        uint32_t *drop_esis = (uint32_t *)malloc(drops * sizeof(uint32_t));
        uint8_t *rec = (uint8_t *)calloc(drops, stride);
        vxfec_rq_sched *gs_dec = vxfec_rq_sched_new(ctx, &dec, &S_dec);
        if (d_dec && drop_esis && rec && gs_dec) {
            for (uint32_t esi = drops; esi < K; esi++)
                memcpy(d_dec + (size_t)(sh + esi) * stride, src_orig + (size_t)esi * stride, T);
            for (uint32_t r = 0; r < drops; r++) {
                memcpy(d_dec + (size_t)(sh + r) * stride, repairs + (size_t)r * stride, T);
                drop_esis[r] = r;
            }
            if (test_rq_run_host_buf(ctx, &dec, gs_dec, d_dec, stride, 1) &&
                test_rq_mix_host_buf(ctx, &dec, d_dec, stride, 1, drop_esis, drops, rec)) {
                ok = true;
                for (uint32_t r = 0; r < drops; r++) {
                    if (memcmp(rec + (size_t)r * stride, src_orig + (size_t)r * stride, T) != 0)
                        ok = false;
                }
            }
        }
        free(rec);
        free(drop_esis);
        vxfec_rq_sched_free(ctx, gs_dec);
        free(d_dec);
        free(dec_sched);
        free(dec_work);
        free(dec_prep);
    }

cleanup:
    free(repairs);
    free(rep_esis);
    vxfec_rq_sched_free(ctx, gs_enc);
    free(d_enc);
    free(src_orig);
    free(sched_mem);
    free(work_mem);
    free(prep_mem);
    return ok;
}

static bool test_edge_cases(vxfec_vk_ctx *ctx)
{
    nanorq_core rq = {0};
    if (!nanorq_core_encoder_new(16, 0, &rq))
        return false;
    bool ok = false;
    size_t prep_len = nanorq_core_calculate_prepare_memory(&rq);
    size_t work_len = nanorq_core_calculate_work_memory(&rq);
    size_t sched_len = ops_estimate_schedule_bytes(16);
    uint8_t *prep_mem = (uint8_t *)malloc(prep_len);
    uint8_t *work_mem = (uint8_t *)malloc(work_len);
    void *sched_mem = malloc(sched_len);
    uint8_t *d_cpu = NULL, *d_gpu = NULL, *edge_dir = NULL;
    vxfec_rq_sched *gs = NULL;
    vxfec_rq_direct *gd_edge = NULL;
    schedule S = {0};
    if (!prep_mem || !work_mem || !schedule_init(&S, sched_mem, sched_len) || !nanorq_core_prepare(&rq, prep_mem, prep_len))
        goto cleanup;
    nanorq_core_set_op_callback(&rq, &S, ops_push);
    if (!nanorq_core_precalculate(&rq, work_mem, work_len))
        goto cleanup;

    uint32_t rows = nanorq_core_get_pc_rows(&rq);
    uint32_t sh = nanorq_core_get_pc_genc_offset(&rq);
    uint32_t stride = 32;

    /* 1. large num_esis > 65535 chunked dispatch + bit-exact check */
    d_cpu = (uint8_t *)calloc(rows, stride);
    d_gpu = (uint8_t *)calloc(rows, stride);
    if (!d_cpu || !d_gpu)
        goto cleanup;
    for (uint32_t e = 0; e < 16; e++)
        for (uint32_t c = 0; c < stride; c++)
            d_cpu[(size_t)(sh + e) * stride + c] = (uint8_t)(e * 13u + c + 3u);
    memcpy(d_gpu, d_cpu, (size_t)rows * stride);

    ops_run_pivoted(d_cpu, stride, &S);
    gs = vxfec_rq_sched_new(ctx, &rq, &S);
    if (!gs || !test_rq_run_host_buf(ctx, &rq, gs, d_gpu, stride, 1))
        goto cleanup;
    vxfec_rq_sched_free(ctx, gs);
    gs = NULL;

    uint32_t num_esis = 70000;
    uint32_t *esis = (uint32_t *)malloc(num_esis * sizeof(uint32_t));
    uint8_t *out = (uint8_t *)calloc(num_esis, stride);
    if (!esis || !out) {
        free(out);
        free(esis);
        goto cleanup;
    }
    for (uint32_t i = 0; i < num_esis; i++)
        esis[i] = i;
    bool big_ok = test_rq_mix_host_buf(ctx, &rq, d_gpu, stride, 1, esis, num_esis, out);
    uint8_t m_cpu[32] = {0}, m_gpu[32] = {0};
    const uint32_t chk_esis[3] = {0u, 65535u, 69999u};
    for (int ce = 0; ce < 3 && big_ok; ce++) {
        uint32_t target_esi = chk_esis[ce];
        ops_mix_pivoted(&rq, d_cpu, stride, target_esi, m_cpu);
        if (memcmp(m_cpu, out + (size_t)target_esi * stride, stride) != 0)
            big_ok = false;
    }
    free(out);
    free(esis);
    if (!big_ok)
        goto cleanup;

    /* 2. hybrid GPU ops_run_batch -> CPU ops_mix_pivoted (zero row permutation) */
    ops_mix_pivoted(&rq, d_cpu, stride, 21, m_cpu);
    ops_mix_pivoted(&rq, d_gpu, stride, 21, m_gpu);
    if (memcmp(m_cpu, m_gpu, stride) != 0)
        goto cleanup;

    /* 3. schedule packing edge cases: u!=1 before cp[0], scal, self-op i==j */
    for (uint32_t i = 0; i < rq.W.rows; i++)
        uv_A(rq.W.d, i) = uv_A(rq.W.di, i) = i;
    for (uint32_t i = 0; i < rq.W.cols; i++)
        uv_A(rq.W.c, i) = uv_A(rq.W.ci, i) = i;

    sched_op op_buf[16];
    schedule S_edge = {0};
    schedule_init(&S_edge, op_buf, sizeof(op_buf));
    ops_push(&S_edge, 2, 1, 7);
    ops_push(&S_edge, 5, 3, 0);
    ops_push(&S_edge, 0, 0, 0);
    ops_push(&S_edge, 1, 1, 1);
    ops_push(&S_edge, 3, 1, 5);
    ops_push(&S_edge, 0, 0, 0);
    ops_push(&S_edge, 4, 3, 1);

    for (size_t i = 0; i < (size_t)rows * stride; i++)
        d_cpu[i] = d_gpu[i] = (uint8_t)((i * 29u + 7u) & 0xffu);

    uint32_t edge_esis[70];
    for (uint32_t e = 0; e < 70u; e++)
        edge_esis[e] = e;
    edge_dir = (uint8_t *)calloc(70u, stride);
    gd_edge = vxfec_rq_direct_new(ctx, &rq, &S_edge, edge_esis, 70u);
    if (!edge_dir || !gd_edge || !test_rq_direct_host_buf(ctx, rows, gd_edge, d_gpu, stride, 1, 70u, edge_dir))
        goto cleanup;
    vxfec_rq_direct_free(ctx, gd_edge);
    gd_edge = NULL;

    ops_run(&rq, d_cpu, stride, &S_edge);
    gs = vxfec_rq_sched_new(ctx, &rq, &S_edge);
    if (!gs || !test_rq_run_host_buf(ctx, &rq, gs, d_gpu, stride, 1) || memcmp(d_cpu, d_gpu, (size_t)rows * stride) != 0)
        goto cleanup;
    for (uint32_t e = 0; e < 70u; e++) {
        ops_mix(&rq, d_cpu, stride, e, m_cpu);
        if (memcmp(m_cpu, edge_dir + (size_t)e * stride, stride) != 0)
            goto cleanup;
    }
    vxfec_rq_sched_free(ctx, gs);
    gs = NULL;

    /* pure-GF(2) self-op i==j with >64 ESIs (wpr=2) in direct compiler + raw submit */
    sched_op gf2_edge_buf[8];
    schedule S_gf2_edge = {0};
    schedule_init(&S_gf2_edge, gf2_edge_buf, sizeof(gf2_edge_buf));
    ops_push(&S_gf2_edge, 2, 1, 1);
    ops_push(&S_gf2_edge, 0, 0, 0);
    ops_push(&S_gf2_edge, 1, 1, 1);
    ops_push(&S_gf2_edge, 3, 2, 1);
    ops_push(&S_gf2_edge, 0, 0, 0);
    ops_push(&S_gf2_edge, 4, 3, 1);

    for (size_t i = 0; i < (size_t)rows * stride; i++)
        d_cpu[i] = d_gpu[i] = (uint8_t)((i * 41u + 11u) & 0xffu);
    gd_edge = vxfec_rq_direct_new(ctx, &rq, &S_gf2_edge, edge_esis, 70u);
    memset(edge_dir, 0, 70u * stride);
    if (!gd_edge || !vxfec_rq_direct_is_pure_gf2(gd_edge) ||
        !test_rq_direct_host_buf(ctx, rows, gd_edge, d_gpu, stride, 1, 70u, edge_dir))
        goto cleanup;
    vxfec_rq_direct_free(ctx, gd_edge);
    gd_edge = NULL;

    /* also verify vxfec_rq_direct_submit_raw produces identical output */
    vxfec_vk_buf *raw_d = vxfec_vk_buf_new(ctx, (size_t)rows * stride, true);
    vxfec_vk_buf *raw_out = vxfec_vk_buf_new(ctx, 70u * stride, true);
    bool raw_ok = false;
    if (raw_d && raw_out) {
        memcpy(vxfec_vk_buf_staging_ptr(raw_d), d_gpu, (size_t)rows * stride);
        uint64_t rtk = vxfec_rq_direct_submit_raw(ctx, &rq, &S_gf2_edge, raw_d, stride, 1, edge_esis, 70u, raw_out, true, true);
        raw_ok = (rtk != 0) && vxfec_vk_wait(ctx, rtk, UINT64_MAX) &&
                 memcmp(vxfec_vk_buf_staging_ptr(raw_out), edge_dir, 70u * stride) == 0;
    }
    vxfec_vk_buf_free(ctx, raw_out);
    vxfec_vk_buf_free(ctx, raw_d);
    if (!raw_ok)
        goto cleanup;
    ops_run(&rq, d_cpu, stride, &S_gf2_edge);
    for (uint32_t e = 0; e < 70u; e++) {
        ops_mix(&rq, d_cpu, stride, e, m_cpu);
        if (memcmp(m_cpu, edge_dir + (size_t)e * stride, stride) != 0)
            goto cleanup;
    }

    /* 4. device-to-device buffer copy and bounds check */
    vxfec_vk_buf *b_src = vxfec_vk_buf_new(ctx, 256, true);
    vxfec_vk_buf *b_mid = vxfec_vk_buf_new(ctx, 256, false);
    vxfec_vk_buf *b_dst = vxfec_vk_buf_new(ctx, 256, true);
    bool copy_ok = false;
    if (b_src && b_mid && b_dst) {
        uint8_t *p_src = (uint8_t *)vxfec_vk_buf_staging_ptr(b_src);
        uint8_t *p_dst = (uint8_t *)vxfec_vk_buf_staging_ptr(b_dst);
        for (size_t i = 0; i < 256; i++) {
            p_src[i] = (uint8_t)(i ^ 0x5au);
            p_dst[i] = 0;
        }
        copy_ok = vxfec_vk_buf_upload(ctx, b_src, 256) && vxfec_vk_buf_copy(ctx, b_mid, b_src, 256) &&
                  vxfec_vk_buf_copy(ctx, b_dst, b_mid, 256) && vxfec_vk_buf_download(ctx, b_dst, 256) &&
                  memcmp(p_src, p_dst, 256) == 0 && !vxfec_vk_buf_copy(ctx, b_dst, b_src, 512);
    }
    vxfec_vk_buf_free(ctx, b_dst);
    vxfec_vk_buf_free(ctx, b_mid);
    vxfec_vk_buf_free(ctx, b_src);
    if (!copy_ok)
        goto cleanup;

    /* 5. VXFEC_VK_DEVICE parsing and forced-index error reporting */
    const char *orig_env = getenv("VXFEC_VK_DEVICE");
    char saved_env[64] = {0};
    bool had_env = orig_env != NULL;
    if (had_env) {
        strncpy(saved_env, orig_env, sizeof(saved_env) - 1);
    }
    const char *bad_vals[] = {"bad", "-1", "0x", "999999"};
    bool env_ok = true;
    for (size_t i = 0; i < sizeof(bad_vals) / sizeof(bad_vals[0]); i++) {
        int st = VXFEC_VK_OK;
        set_vk_device_env(bad_vals[i]);
        vxfec_vk_ctx *bad_ctx = vxfec_vk_ctx_new_status(&st);
        if (bad_ctx || st != VXFEC_VK_ERROR) {
            vxfec_vk_ctx_free(bad_ctx);
            env_ok = false;
            break;
        }
    }
    if (env_ok) {
        int st = VXFEC_VK_ERROR;
        set_vk_device_env("");
        vxfec_vk_ctx *empty_ctx = vxfec_vk_ctx_new_status(&st);
        if (!empty_ctx || st != VXFEC_VK_OK)
            env_ok = false;
        vxfec_vk_ctx_free(empty_ctx);
    }
    set_vk_device_env(had_env ? saved_env : NULL);
    if (!env_ok)
        goto cleanup;

#if !defined(_WIN32)
    const char *orig_qfam = getenv("VXFEC_VK_QUEUE_FAMILY");
    char saved_qfam[64] = {0};
    bool had_qfam = orig_qfam != NULL;
    if (had_qfam)
        strncpy(saved_qfam, orig_qfam, sizeof(saved_qfam) - 1);
    int qst = VXFEC_VK_OK;
    setenv("VXFEC_VK_QUEUE_FAMILY", "9999", 1);
    vxfec_vk_ctx *bad_qctx = vxfec_vk_ctx_new_status(&qst);
    if (bad_qctx || qst != VXFEC_VK_ERROR) {
        vxfec_vk_ctx_free(bad_qctx);
        env_ok = false;
    }
    if (had_qfam)
        setenv("VXFEC_VK_QUEUE_FAMILY", saved_qfam, 1);
    else
        unsetenv("VXFEC_VK_QUEUE_FAMILY");
    if (!env_ok)
        goto cleanup;
#endif
    ok = true;

cleanup:
    vxfec_rq_direct_free(ctx, gd_edge);
    vxfec_rq_sched_free(ctx, gs);
    free(edge_dir);
    free(d_gpu);
    free(d_cpu);
    free(sched_mem);
    free(work_mem);
    free(prep_mem);
    return ok;
}

static bool test_async_timeline_ring(vxfec_vk_ctx *ctx)
{
    const uint32_t K = 128, T = 1280, num_jobs = 12, num_rep = 20;
    uint32_t stride = nanorq_core_recommended_stride(T);

    nanorq_core rq_gf256 = {0};
    if (!nanorq_core_encoder_new(K, 0, &rq_gf256))
        return false;
    size_t prep_len = nanorq_core_calculate_prepare_memory(&rq_gf256);
    size_t work_len = nanorq_core_calculate_work_memory(&rq_gf256);
    size_t sched_len = ops_estimate_schedule_bytes(K);
    uint8_t *prep256 = (uint8_t *)malloc(prep_len);
    uint8_t *work256 = (uint8_t *)malloc(work_len);
    void *sched256_mem = malloc(sched_len);
    schedule S256 = {0};
    if (!prep256 || !work256 || !schedule_init(&S256, sched256_mem, sched_len) ||
        !nanorq_core_prepare(&rq_gf256, prep256, prep_len)) {
        free(sched256_mem);
        free(work256);
        free(prep256);
        return false;
    }
    nanorq_core_set_op_callback(&rq_gf256, &S256, ops_push);
    if (!nanorq_core_precalculate(&rq_gf256, work256, work_len)) {
        free(sched256_mem);
        free(work256);
        free(prep256);
        return false;
    }
    vxfec_rq_sched *gs256 = vxfec_rq_sched_new(ctx, &rq_gf256, &S256);
    if (!gs256) {
        free(sched256_mem);
        free(work256);
        free(prep256);
        return false;
    }

    /* Pure-GF(2) slack schedule (exercises TILE_E=32 async direct path too) */
    uint32_t oh_gf2 = 16;
    nanorq_core rq_gf2 = {0};
    if (!nanorq_core_encoder_new(K, oh_gf2, &rq_gf2)) {
        vxfec_rq_sched_free(ctx, gs256);
        free(sched256_mem);
        free(work256);
        free(prep256);
        return false;
    }
    size_t prep2_len = nanorq_core_calculate_prepare_memory(&rq_gf2);
    size_t work2_len = nanorq_core_calculate_work_memory(&rq_gf2);
    size_t sched2_len = ops_estimate_schedule_bytes(K + oh_gf2);
    uint8_t *prep2 = (uint8_t *)malloc(prep2_len);
    uint8_t *work2 = (uint8_t *)malloc(work2_len);
    void *sched2_mem = malloc(sched2_len);
    schedule S2 = {0};
    vxfec_rq_sched *gs2 = NULL;
    vxfec_rq_direct *gd_pre = NULL;
    bool ok = true;

    if (!prep2 || !work2 || !schedule_init(&S2, sched2_mem, sched2_len) || !nanorq_core_generate(&rq_gf2, prep2, prep2_len)) {
        ok = false;
    } else {
        for (uint32_t i = 0; i < oh_gf2; i++)
            nanorq_core_replace_symbol(&rq_gf2, rq_gf2.P.Kprime + i, K + i);
        if (!nanorq_core_patch_matrix(&rq_gf2)) {
            ok = false;
        } else {
            nanorq_core_set_op_callback(&rq_gf2, &S2, ops_push);
            if (!nanorq_core_precalculate(&rq_gf2, work2, work2_len))
                ok = false;
            else
                gs2 = vxfec_rq_sched_new(ctx, &rq_gf2, &S2);
        }
    }
    if (!gs2 || !vxfec_rq_sched_is_pure_gf2(gs2))
        ok = false;

    uint32_t pre_esis[20];
    for (uint32_t r = 0; r < num_rep; r++)
        pre_esis[r] = K + 500u + r;
    if (ok) {
        gd_pre = vxfec_rq_direct_new(ctx, &rq_gf2, &S2, pre_esis, num_rep);
        if (!gd_pre)
            ok = false;
    }

    vxfec_vk_buf *d_bufs[12] = {0};
    vxfec_vk_buf *out_bufs[12] = {0};
    uint8_t *exp_outs[12] = {0};
    uint64_t tickets[12] = {0};

    for (uint32_t j = 0; j < num_jobs && ok; j++) {
        bool use_gf2 = ((j & 1u) == 0);
        nanorq_core *rq = use_gf2 ? &rq_gf2 : &rq_gf256;
        schedule *S = use_gf2 ? &S2 : &S256;
        vxfec_rq_sched *gs = use_gf2 ? gs2 : gs256;

        uint32_t rows = nanorq_core_get_pc_rows(rq);
        uint32_t sh = nanorq_core_get_pc_genc_offset(rq);
        size_t d_bytes = (size_t)rows * stride;
        size_t out_bytes = (size_t)num_rep * stride;

        d_bufs[j] = vxfec_vk_buf_new(ctx, d_bytes, true);
        out_bufs[j] = vxfec_vk_buf_new(ctx, out_bytes, true);
        exp_outs[j] = (uint8_t *)calloc(num_rep, stride);
        uint8_t *d_cpu = (uint8_t *)calloc(rows, stride);
        if (!d_bufs[j] || !out_bufs[j] || !exp_outs[j] || !d_cpu) {
            free(d_cpu);
            ok = false;
            break;
        }

        uint8_t *d_stage = (uint8_t *)vxfec_vk_buf_staging_ptr(d_bufs[j]);
        memset(d_stage, 0, d_bytes);
        for (uint32_t e = 0; e < K; e++) {
            for (uint32_t c = 0; c < T; c++) {
                uint8_t v = (uint8_t)((j * 97u + e * 41u + c * 13u + 7u) & 0xffu);
                d_cpu[(size_t)(sh + e) * stride + c] = v;
                d_stage[(size_t)(sh + e) * stride + c] = v;
            }
        }
        if (use_gf2) {
            for (uint32_t i = 0; i < oh_gf2; i++) {
                for (uint32_t c = 0; c < T; c++) {
                    uint8_t v = (uint8_t)((j * 53u + i * 29u + c * 17u + 3u) & 0xffu);
                    d_cpu[(size_t)(sh + rq->P.Kprime + i) * stride + c] = v;
                    d_stage[(size_t)(sh + rq->P.Kprime + i) * stride + c] = v;
                }
            }
        }

        uint32_t esis[20];
        for (uint32_t r = 0; r < num_rep; r++)
            esis[r] = (j == 2u) ? pre_esis[r] : (K + 100u + j * num_rep + r);

        ops_run(rq, d_cpu, stride, S);
        for (uint32_t r = 0; r < num_rep; r++)
            ops_mix(rq, d_cpu, stride, esis[r], exp_outs[j] + (size_t)r * stride);
        free(d_cpu);

        if (!vxfec_vk_async_can_submit(ctx)) {
            ok = false;
            break;
        }
        if (j == 2u) {
            tickets[j] = vxfec_rq_direct_submit(ctx, gd_pre, d_bufs[j], stride, 1, out_bufs[j], true, true);
        } else {
            tickets[j] = vxfec_rq_run_and_mix_submit(ctx, rq, gs, d_bufs[j], stride, 1, esis, num_rep, out_bufs[j], true, true);
        }
        if (tickets[j] == 0) {
            ok = false;
            break;
        }
        (void)vxfec_vk_poll(ctx, tickets[j]);
    }

    /* verify non-blocking saturation behavior across a 40-submission burst */
    if (ok) {
        uint64_t burst_last = 0;
        uint32_t accepted = 0, backpressured = 0;
        for (uint32_t b = 0; b < 40u; b++) {
            uint64_t tk = vxfec_rq_direct_submit(ctx, gd_pre, d_bufs[2], stride, 1, out_bufs[2], false, false);
            if (tk != 0) {
                burst_last = tk;
                accepted++;
            } else {
                backpressured++;
            }
        }
        if (accepted == 0 || (burst_last != 0 && !vxfec_vk_wait(ctx, burst_last, UINT64_MAX)) || !vxfec_vk_async_can_submit(ctx)) {
            fprintf(stderr, "async non-blocking ring burst failed (accepted=%u, backpressured=%u)\n", accepted, backpressured);
            ok = false;
        }
    }

    if (ok && (vxfec_vk_poll_status(NULL, 1u) != -1 || vxfec_vk_poll_status(ctx, 0u) != 1))
        ok = false;

    for (uint32_t j = 0; j < num_jobs && ok; j++) {
        if (!vxfec_vk_wait(ctx, tickets[j], UINT64_MAX) || vxfec_vk_poll_status(ctx, tickets[j]) != 1 ||
            !vxfec_vk_poll(ctx, tickets[j])) {
            ok = false;
            break;
        }
        const uint8_t *got = (const uint8_t *)vxfec_vk_buf_staging_ptr(out_bufs[j]);
        for (uint32_t r = 0; r < num_rep; r++) {
            if (memcmp(got + (size_t)r * stride, exp_outs[j] + (size_t)r * stride, T) != 0) {
                fprintf(stderr, "async ring mismatch at job=%u rep=%u\n", j, r);
                ok = false;
                break;
            }
        }
    }

    /* verify staged schedule across ops_run_batch and 2-stage (>32 ESI) async submit */
    if (ok) {
        const uint32_t n_two_stage = 40u;
        uint32_t rows = nanorq_core_get_pc_rows(&rq_gf256);
        uint32_t sh = nanorq_core_get_pc_genc_offset(&rq_gf256);
        size_t d_bytes = (size_t)rows * stride;
        size_t out_bytes = (size_t)n_two_stage * stride;
        vxfec_rq_sched *gs_st1 = vxfec_rq_sched_new_staged(ctx, &rq_gf256, &S256);
        vxfec_rq_sched *gs_st2 = vxfec_rq_sched_new_staged(ctx, &rq_gf256, &S256);
        vxfec_vk_buf *d_st1 = vxfec_vk_buf_new(ctx, d_bytes, true);
        vxfec_vk_buf *d_st2 = vxfec_vk_buf_new(ctx, d_bytes, true);
        vxfec_vk_buf *out_st1 = vxfec_vk_buf_new(ctx, out_bytes, true);
        vxfec_vk_buf *out_st2 = vxfec_vk_buf_new(ctx, out_bytes, true);
        uint8_t *d_cpu = (uint8_t *)calloc(rows, stride);
        uint8_t *exp = (uint8_t *)calloc(n_two_stage, stride);
        uint32_t esis40[40];
        for (uint32_t r = 0; r < n_two_stage; r++)
            esis40[r] = K + 900u + r;

        if (!gs_st1 || !gs_st2 || !d_st1 || !d_st2 || !out_st1 || !out_st2 || !d_cpu || !exp) {
            ok = false;
        } else {
            uint8_t *s1 = (uint8_t *)vxfec_vk_buf_staging_ptr(d_st1);
            uint8_t *s2 = (uint8_t *)vxfec_vk_buf_staging_ptr(d_st2);
            memset(s1, 0, d_bytes);
            memset(s2, 0, d_bytes);
            for (uint32_t e = 0; e < K; e++) {
                for (uint32_t c = 0; c < T; c++) {
                    uint8_t v = (uint8_t)((e * 67u + c * 19u + 11u) & 0xffu);
                    d_cpu[(size_t)(sh + e) * stride + c] = v;
                    s1[(size_t)(sh + e) * stride + c] = v;
                    s2[(size_t)(sh + e) * stride + c] = v;
                }
            }
            ops_run(&rq_gf256, d_cpu, stride, &S256);
            for (uint32_t r = 0; r < n_two_stage; r++)
                ops_mix(&rq_gf256, d_cpu, stride, esis40[r], exp + (size_t)r * stride);

            uint64_t tk = 0;
            if (!vxfec_vk_buf_upload(ctx, d_st1, d_bytes) || !vxfec_rq_ops_run_batch(ctx, gs_st1, d_st1, stride, 1u, NULL) ||
                !vxfec_rq_ops_mix_batch(ctx, &rq_gf256, d_st1, stride, 1u, esis40, n_two_stage, out_st1, NULL) ||
                !vxfec_vk_buf_download(ctx, out_st1, out_bytes) ||
                (tk = vxfec_rq_run_and_mix_submit(ctx, &rq_gf256, gs_st2, d_st2, stride, 1u, esis40, n_two_stage, out_st2, true,
                                                  true)) == 0 ||
                !vxfec_vk_wait(ctx, tk, UINT64_MAX)) {
                ok = false;
            } else {
                const uint8_t *got1 = (const uint8_t *)vxfec_vk_buf_staging_ptr(out_st1);
                const uint8_t *got2 = (const uint8_t *)vxfec_vk_buf_staging_ptr(out_st2);
                for (uint32_t r = 0; r < n_two_stage; r++) {
                    if (memcmp(got1 + (size_t)r * stride, exp + (size_t)r * stride, T) != 0 ||
                        memcmp(got2 + (size_t)r * stride, exp + (size_t)r * stride, T) != 0) {
                        ok = false;
                        break;
                    }
                }
            }
        }
        free(exp);
        free(d_cpu);
        vxfec_vk_buf_free(ctx, out_st2);
        vxfec_vk_buf_free(ctx, out_st1);
        vxfec_vk_buf_free(ctx, d_st2);
        vxfec_vk_buf_free(ctx, d_st1);
        vxfec_rq_sched_free(ctx, gs_st2);
        vxfec_rq_sched_free(ctx, gs_st1);
    }

    for (uint32_t j = 0; j < num_jobs; j++) {
        vxfec_vk_buf_free(ctx, d_bufs[j]);
        vxfec_vk_buf_free(ctx, out_bufs[j]);
        free(exp_outs[j]);
    }
    vxfec_rq_direct_free(ctx, gd_pre);
    vxfec_rq_sched_free(ctx, gs2);
    free(sched2_mem);
    free(work2);
    free(prep2);
    vxfec_rq_sched_free(ctx, gs256);
    free(sched256_mem);
    free(work256);
    free(prep256);
    return ok;
}

#if !defined(_WIN32)
static bool test_ext_memory_interop(vxfec_vk_ctx *ctx)
{
    const uint32_t K = 64, T = 1280, num_rep = 20;
    uint32_t stride = nanorq_core_recommended_stride(T);

    nanorq_core rq = {0};
#ifdef NANORQ_HAS_GF2_MODE
    if (!nanorq_core_encoder_new_gf2(K, 0, &rq))
        return false;
#else
    if (!nanorq_core_encoder_new(K, 0, &rq))
        return false;
#endif
    size_t prep_len = nanorq_core_calculate_prepare_memory(&rq);
    size_t work_len = nanorq_core_calculate_work_memory(&rq);
    size_t sched_len = ops_estimate_schedule_bytes(K);
    uint8_t *prep_mem = (uint8_t *)malloc(prep_len);
    uint8_t *work_mem = (uint8_t *)malloc(work_len);
    void *sched_mem = malloc(sched_len);
    schedule S = {0};
    if (!prep_mem || !work_mem || !schedule_init(&S, sched_mem, sched_len) || !nanorq_core_prepare(&rq, prep_mem, prep_len)) {
        free(sched_mem);
        free(work_mem);
        free(prep_mem);
        return false;
    }
    nanorq_core_set_op_callback(&rq, &S, ops_push);
    if (!nanorq_core_precalculate(&rq, work_mem, work_len)) {
        free(sched_mem);
        free(work_mem);
        free(prep_mem);
        return false;
    }

    vxfec_rq_sched *gs = vxfec_rq_sched_new(ctx, &rq, &S);
    if (!gs) {
        free(sched_mem);
        free(work_mem);
        free(prep_mem);
        return false;
    }

    uint32_t rows = nanorq_core_get_pc_rows(&rq);
    uint32_t sh = nanorq_core_get_pc_genc_offset(&rq);
    size_t d_bytes = (size_t)rows * stride;
    size_t out_bytes = (size_t)num_rep * stride;

    uint8_t *d_cpu = (uint8_t *)calloc(rows, stride);
    uint8_t *d_orig = (uint8_t *)malloc(d_bytes);
    uint8_t *exp_rep = (uint8_t *)calloc(num_rep, stride);
    if (!d_cpu || !d_orig || !exp_rep) {
        free(exp_rep);
        free(d_orig);
        free(d_cpu);
        vxfec_rq_sched_free(ctx, gs);
        free(sched_mem);
        free(work_mem);
        free(prep_mem);
        return false;
    }

    uint32_t rep_esis[20];
    for (uint32_t r = 0; r < num_rep; r++)
        rep_esis[r] = K + r;

    for (uint32_t e = 1; e < K; e++) {
        for (uint32_t c = 0; c < T; c++) {
            d_cpu[(size_t)(sh + e) * stride + c] = (uint8_t)((e * 59u + c * 23u + 17u) & 0xffu);
        }
    }
    memcpy(d_orig, d_cpu, d_bytes);
    ops_run(&rq, d_cpu, stride, &S);
    for (uint32_t r = 0; r < num_rep; r++)
        ops_mix(&rq, d_cpu, stride, rep_esis[r], exp_rep + (size_t)r * stride);

    bool ok = true;

    /* 1. Zero-copy VK_EXT_external_memory_host import */
    if (vxfec_vk_has_ext_host_mem(ctx)) {
        size_t d_aligned = (d_bytes + 65535u) & ~(size_t)65535u;
        size_t out_aligned = (out_bytes + 65535u) & ~(size_t)65535u;
        void *h_d = NULL, *h_out = NULL;
        if (posix_memalign(&h_d, 65536u, d_aligned) != 0 || posix_memalign(&h_out, 65536u, out_aligned) != 0) {
            ok = false;
        } else {
            memset(h_d, 0, d_aligned);
            memset(h_out, 0, out_aligned);
            memcpy(h_d, d_orig, d_bytes);

            vxfec_vk_buf *imp_d = vxfec_vk_buf_import_host_ptr(ctx, h_d, d_aligned);
            vxfec_vk_buf *imp_out = vxfec_vk_buf_import_host_ptr(ctx, h_out, out_aligned);
            if (!imp_d || !imp_out) {
                fprintf(stderr, "FAIL: vxfec_vk_buf_import_host_ptr returned NULL despite has_ext_host_mem=1\n");
                ok = false;
            } else {
                uint64_t tk = vxfec_rq_run_and_mix_submit(ctx, &rq, gs, imp_d, stride, 1, rep_esis, num_rep, imp_out, true, true);
                if (tk == 0 || !vxfec_vk_wait(ctx, tk, UINT64_MAX)) {
                    ok = false;
                } else {
                    for (uint32_t r = 0; r < num_rep; r++) {
                        if (memcmp((const uint8_t *)h_out + (size_t)r * stride, exp_rep + (size_t)r * stride, T) != 0) {
                            fprintf(stderr, "host-ptr zero-copy mismatch at r=%u\n", r);
                            ok = false;
                            break;
                        }
                    }
                }
            }
            vxfec_vk_buf_free(ctx, imp_out);
            vxfec_vk_buf_free(ctx, imp_d);
        }
        free(h_out);
        free(h_d);
    }

    /* 2. Zero-copy VK_EXT_external_memory_dma_buf export -> device-matched ctx + sub-range import */
    if (ok && vxfec_vk_has_dmabuf(ctx) && vxfec_vk_has_dmabuf_export(ctx)) {
        size_t sub_off = 4096u;
        size_t exp_total = sub_off + d_bytes;
        int dmabuf_fd = -1;
        vxfec_vk_buf *d_exp = vxfec_vk_buf_new_exportable(ctx, exp_total, true, &dmabuf_fd);
        if (!d_exp || dmabuf_fd < 0) {
            ok = false;
        } else {
            bool seeded = vxfec_vk_buf_write_offset(ctx, d_exp, sub_off, d_orig, d_bytes) &&
                          vxfec_vk_buf_release_external(ctx, d_exp) && vxfec_vk_ctx_supports_dmabuf_fd(ctx, dmabuf_fd);
            int dmabuf_st = VXFEC_VK_ERROR;
            vxfec_vk_ctx *ctx2 = vxfec_vk_ctx_new_for_dmabuf(dmabuf_fd, &dmabuf_st);
            if (!seeded || !ctx2 || dmabuf_st != VXFEC_VK_OK) {
                close(dmabuf_fd);
                vxfec_vk_ctx_free(ctx2);
                ok = false;
            } else {
                vxfec_vk_buf *d_imp2 = vxfec_vk_buf_import_fd_range(ctx2, dmabuf_fd, sub_off, d_bytes, true);
                if (!d_imp2) {
                    close(dmabuf_fd);
                    ok = false;
                } else {
                    uint8_t slice_chk[64] = {0};
                    size_t chk_off = (size_t)(sh + 1u) * stride;
                    /* transient acquire+copy+release read on sub-range view_offset while unowned */
                    if (!vxfec_vk_buf_read_offset(ctx2, d_imp2, chk_off, slice_chk, sizeof(slice_chk)) ||
                        memcmp(slice_chk, d_orig + chk_off, sizeof(slice_chk)) != 0 ||
                        !vxfec_vk_buf_acquire_external(ctx2, d_imp2)) {
                        ok = false;
                    }
                    vxfec_rq_sched *gs2 = vxfec_rq_sched_new(ctx2, &rq, &S);
                    vxfec_vk_buf *out2 = vxfec_vk_buf_new(ctx2, out_bytes, true);
                    if (!ok || !gs2 || !out2 ||
                        !vxfec_rq_run_and_mix_pipelined(ctx2, &rq, gs2, d_imp2, stride, 1, rep_esis, num_rep, out2, false, true,
                                                        NULL) ||
                        !vxfec_vk_buf_release_external(ctx2, d_imp2) || !vxfec_vk_buf_acquire_external(ctx, d_exp)) {
                        ok = false;
                    } else {
                        const uint8_t *got = (const uint8_t *)vxfec_vk_buf_staging_ptr(out2);
                        for (uint32_t r = 0; r < num_rep; r++) {
                            if (memcmp(got + (size_t)r * stride, exp_rep + (size_t)r * stride, T) != 0) {
                                fprintf(stderr, "DMA-BUF sub-range cross-ctx mismatch at r=%u\n", r);
                                ok = false;
                                break;
                            }
                        }
                    }
                    vxfec_vk_buf_free(ctx2, out2);
                    vxfec_rq_sched_free(ctx2, gs2);
                    vxfec_vk_buf_free(ctx2, d_imp2);
                }
                vxfec_vk_ctx_free(ctx2);
            }
            vxfec_vk_buf_free(ctx, d_exp);
        }
    }

    /* 3. generic OPAQUE_FD (is_dmabuf=false) must be rejected without closing caller fd */
    if (ok) {
        int opq_fd = -1;
        int pfd[2] = {-1, -1};
        if (pipe(pfd) != 0) {
            ok = false;
        } else {
            vxfec_vk_buf *bad_exp = vxfec_vk_buf_new_exportable(ctx, d_bytes, false, &opq_fd);
            vxfec_vk_buf *bad_imp = vxfec_vk_buf_import_fd(ctx, pfd[0], d_bytes, false);
            if (bad_exp != NULL || opq_fd != -1 || bad_imp != NULL || close(pfd[0]) != 0) {
                fprintf(stderr, "FAIL: generic OPAQUE_FD interop was not rejected cleanly\n");
                vxfec_vk_buf_free(ctx, bad_exp);
                vxfec_vk_buf_free(ctx, bad_imp);
                ok = false;
            }
            close(pfd[1]);
        }
    }

    vxfec_rq_sched_free(ctx, gs);
    free(exp_rep);
    free(d_orig);
    free(d_cpu);
    free(sched_mem);
    free(work_mem);
    free(prep_mem);
    return ok;
}
#endif

static bool test_cauchy_rs8_case(vxfec_vk_ctx *ctx, uint32_t K, uint32_t M, uint32_t T, uint32_t Z, uint32_t erasures)
{
    uint8_t parity_m[256];
    for (uint32_t m = 0; m < M; m++)
        parity_m[m] = (uint8_t)m;

    size_t sys_bytes = (size_t)Z * K * T;
    size_t par_bytes = (size_t)Z * M * T;
    uint8_t *sys_orig = (uint8_t *)malloc(sys_bytes);
    uint8_t *par_cpu = (uint8_t *)calloc(1, par_bytes);
    uint8_t *par_gpu = (uint8_t *)calloc(1, par_bytes);
    uint8_t *compact = (uint8_t *)calloc(1, sys_bytes);
    uint8_t *rec_cpu = (uint8_t *)calloc(1, (size_t)Z * erasures * T);
    uint8_t *rec_gpu = (uint8_t *)calloc(1, (size_t)Z * erasures * T);
    vxfec_rs8_plan *enc_plan = NULL, *dec_plan = NULL;
    vxfec_vk_buf *d_buf = NULL, *out_buf = NULL;
    bool ok = false;

    if (!sys_orig || !par_cpu || !par_gpu || !compact || !rec_cpu || !rec_gpu)
        goto cleanup;

    for (size_t i = 0; i < sys_bytes; i++)
        sys_orig[i] = (uint8_t)((i * 59u + (i >> 5) * 19u + 3u) & 0xffu);

    enc_plan = vxfec_rs8_encode_plan_new(ctx, K, M, NULL, M);
    if (!enc_plan || vxfec_rs8_plan_k(enc_plan) != K || vxfec_rs8_plan_m(enc_plan) != M || vxfec_rs8_plan_outputs(enc_plan) != M)
        goto cleanup;

    d_buf = vxfec_vk_buf_new(ctx, sys_bytes, true);
    out_buf = vxfec_vk_buf_new(ctx, par_bytes, true);
    if (!d_buf || !out_buf)
        goto cleanup;
    memcpy(vxfec_vk_buf_staging_ptr(d_buf), sys_orig, sys_bytes);
    if (!vxfec_rs8_exec_cpu(enc_plan, sys_orig, T, Z, par_cpu) ||
        !vxfec_rs8_exec_pipelined(ctx, enc_plan, d_buf, T, Z, out_buf, true, true, NULL))
        goto cleanup;
    memcpy(par_gpu, vxfec_vk_buf_staging_ptr(out_buf), par_bytes);

    if (memcmp(par_cpu, par_gpu, par_bytes) != 0) {
        fprintf(stderr, "FAIL: Cauchy RS8 encode CPU vs GPU mismatch K=%u M=%u\n", K, M);
        goto cleanup;
    }

    /* verify non-blocking vxfec_rs8_submit_raw encode matches CPU */
    memset(vxfec_vk_buf_staging_ptr(out_buf), 0, par_bytes);
    uint64_t raw_enc_tk = vxfec_rs8_submit_raw(ctx, false, K, M, NULL, NULL, M, d_buf, T, Z, out_buf, false, true);
    if (raw_enc_tk == 0 || !vxfec_vk_wait(ctx, raw_enc_tk, UINT64_MAX) ||
        memcmp(par_cpu, vxfec_vk_buf_staging_ptr(out_buf), par_bytes) != 0) {
        fprintf(stderr, "FAIL: vxfec_rs8_submit_raw encode mismatch K=%u M=%u\n", K, M);
        goto cleanup;
    }
    vxfec_vk_buf_free(ctx, out_buf);
    out_buf = NULL;
    vxfec_vk_buf_free(ctx, d_buf);
    d_buf = NULL;

    /* Verify 100% bit-for-bit wire compatibility with nanors / GameStream: C[m, k] = inv((M + k) ^ m). */
    if (K + M <= 255u) {
        uint8_t *ref_par = (uint8_t *)calloc(M, T);
        if (!ref_par)
            goto cleanup;
        for (uint32_t m = 0; m < M; m++) {
            uint8_t *dst = ref_par + (size_t)m * T;
            for (uint32_t k = 0; k < K; k++) {
                uint8_t coef = GF2_8_INV[(M + k) ^ m];
                uint32_t lc = GF2_8_LOG[coef];
                const uint8_t *src = sys_orig + (size_t)k * T;
                for (uint32_t b = 0; b < T; b++) {
                    if (src[b] != 0)
                        dst[b] ^= GF2_8_EXP[(uint32_t)GF2_8_LOG[src[b]] + lc];
                }
            }
        }
        if (memcmp(ref_par, par_gpu, (size_t)M * T) != 0) {
            fprintf(stderr, "FAIL: GameStream / nanors reference vs Vulkan GPU parity mismatch K=%u M=%u\n", K, M);
            free(ref_par);
            goto cleanup;
        }
        free(ref_par);
    }

    /* Pick arbitrary non-contiguous erased systematic indices & parity indices */
    uint8_t erased_k[256];
    uint8_t used_m[256];
    for (uint32_t i = 0; i < erasures; i++) {
        erased_k[i] = (uint8_t)(((uint64_t)i * K) / erasures);
        used_m[i] = parity_m[M - 1u - i];
    }

    memcpy(compact, sys_orig, sys_bytes);
    for (uint32_t b = 0; b < Z; b++) {
        uint8_t *blk_c = compact + (size_t)b * K * T;
        const uint8_t *blk_p = par_gpu + (size_t)b * M * T;
        for (uint32_t i = 0; i < erasures; i++) {
            memcpy(blk_c + (size_t)erased_k[i] * T, blk_p + (size_t)used_m[i] * T, T);
        }
    }

    /* Test one-shot CPU compact decode on block 0 */
    uint8_t *blk0_copy = (uint8_t *)malloc((size_t)K * T);
    if (!blk0_copy)
        goto cleanup;
    memcpy(blk0_copy, compact, (size_t)K * T);
    if (!vxfec_rs8_decode_compact_cpu(K, M, blk0_copy, T, erased_k, used_m, erasures) ||
        memcmp(blk0_copy, sys_orig, (size_t)K * T) != 0) {
        free(blk0_copy);
        fprintf(stderr, "FAIL: Cauchy RS8 one-shot CPU decode mismatch K=%u\n", K);
        goto cleanup;
    }
    free(blk0_copy);

    /* Test batched GPU + async timeline ring decode across Z blocks */
    dec_plan = vxfec_rs8_decode_plan_new(ctx, K, M, erased_k, used_m, erasures);
    d_buf = vxfec_vk_buf_new(ctx, sys_bytes, true);
    out_buf = vxfec_vk_buf_new(ctx, (size_t)Z * erasures * T, true);
    if (!dec_plan || !d_buf || !out_buf)
        goto cleanup;

    if (!vxfec_rs8_exec_cpu(dec_plan, compact, T, Z, rec_cpu))
        goto cleanup;

    memcpy(vxfec_vk_buf_staging_ptr(d_buf), compact, sys_bytes);
    uint64_t tk = vxfec_rs8_exec_submit(ctx, dec_plan, d_buf, T, Z, out_buf, true, true);
    if (tk == 0 || !vxfec_vk_wait(ctx, tk, UINT64_MAX))
        goto cleanup;
    memcpy(rec_gpu, vxfec_vk_buf_staging_ptr(out_buf), (size_t)Z * erasures * T);

    if (memcmp(rec_cpu, rec_gpu, (size_t)Z * erasures * T) != 0) {
        fprintf(stderr, "FAIL: Cauchy RS8 decode CPU vs GPU mismatch K=%u\n", K);
        goto cleanup;
    }

    /* verify non-blocking vxfec_rs8_submit_raw decode matches CPU */
    memset(vxfec_vk_buf_staging_ptr(out_buf), 0, (size_t)Z * erasures * T);
    uint64_t raw_dec_tk = vxfec_rs8_submit_raw(ctx, true, K, M, erased_k, used_m, erasures, d_buf, T, Z, out_buf, false, true);
    if (raw_dec_tk == 0 || !vxfec_vk_wait(ctx, raw_dec_tk, UINT64_MAX) ||
        memcmp(rec_cpu, vxfec_vk_buf_staging_ptr(out_buf), (size_t)Z * erasures * T) != 0) {
        fprintf(stderr, "FAIL: vxfec_rs8_submit_raw decode mismatch K=%u\n", K);
        goto cleanup;
    }
    for (uint32_t b = 0; b < Z; b++) {
        const uint8_t *orig_b = sys_orig + (size_t)b * K * T;
        const uint8_t *rec_b = rec_gpu + (size_t)b * erasures * T;
        for (uint32_t i = 0; i < erasures; i++) {
            if (memcmp(orig_b + (size_t)erased_k[i] * T, rec_b + (size_t)i * T, T) != 0) {
                fprintf(stderr, "FAIL: Cauchy RS8 recovered symbol mismatch K=%u\n", K);
                goto cleanup;
            }
        }
    }

    /* Verify codec regime selector */
    if (vxfec_select_codec(128, 32, false) != VXFEC_CODEC_RS_CAUCHY8 || vxfec_codec_ir_slack(VXFEC_CODEC_RS_CAUCHY8, 128) != 0u) {
        fprintf(stderr, "FAIL: vxfec_select_codec RS8 check\n");
        goto cleanup;
    }

    ok = true;

cleanup:
    vxfec_vk_buf_free(ctx, out_buf);
    vxfec_vk_buf_free(ctx, d_buf);
    vxfec_rs8_plan_free(ctx, dec_plan);
    vxfec_rs8_plan_free(ctx, enc_plan);
    free(rec_gpu);
    free(rec_cpu);
    free(compact);
    free(par_gpu);
    free(par_cpu);
    free(sys_orig);
    return ok;
}

int main(void)
{
    if (!test_concurrent_first_use_init()) {
        fprintf(stderr, "FAIL: concurrent first-use RS8 + RaptorQ initialization\n");
        return 1;
    }
    printf("PASS: concurrent first-use RS8 + RaptorQ initialization\n");

    if (!test_benchmark_batch_planner())
        return 1;
    printf("PASS: benchmark batch planner\n");
    int status = VXFEC_VK_ERROR;
    vxfec_codec_t exp_rq = VXFEC_CODEC_RQ_GF256;
#ifdef NANORQ_HAS_GF2_MODE
    exp_rq = VXFEC_CODEC_RQ_GF2;
#endif
    if (K_max == 0 || vxfec_select_codec(256u, 56u, false) != exp_rq ||
        vxfec_select_codec(128u, 32u, false) != VXFEC_CODEC_RS_CAUCHY8 ||
        vxfec_select_codec(128u, UINT32_MAX, false) == VXFEC_CODEC_RS_CAUCHY8 ||
        vxfec_select_codec(16u, 250u, false) != VXFEC_CODEC_RQ_GF256) {
        fprintf(stderr, "FAIL: self-contained vxfec_fec.h codec selection check\n");
        return 1;
    }

    vxfec_vk_ctx *ctx = vxfec_vk_ctx_new_status(&status);
    if (!ctx) {
        const char *req_env = getenv("VXFEC_VK_REQUIRED");
        if (!req_env || !*req_env)
            req_env = getenv("NANORQ_VK_REQUIRED");
        bool vk_req = req_env && *req_env && strcmp(req_env, "0") != 0;
        if (status == VXFEC_VK_UNAVAILABLE && !vk_req) {
            fprintf(stderr, "SKIP: no compatible vulkan compute device available\n");
            return 0;
        }
        fprintf(stderr, "FAIL: vulkan context initialization %s\n", status == VXFEC_VK_UNAVAILABLE ? "unavailable" : "failed");
        return 1;
    }
    if (!vxfec_vk_max_storage_buffer_bytes(ctx) || vxfec_vk_max_storage_buffer_bytes(NULL) != 0) {
        fprintf(stderr, "FAIL: Vulkan storage-buffer range accessor\n");
        vxfec_vk_ctx_free(ctx);
        return 1;
    }
    printf("Vulkan device: %s (discrete=%d, qfam=%u, ace=%d, dmabuf=%d, dmabuf_export=%d, "
           "foreign_qfam=%d, host_import=%d, maxShared=%u B)\n",
           vxfec_vk_device_name(ctx), vxfec_vk_is_discrete(ctx) ? 1 : 0, vxfec_vk_queue_family(ctx),
           vxfec_vk_is_dedicated_compute(ctx) ? 1 : 0, vxfec_vk_has_dmabuf(ctx) ? 1 : 0, vxfec_vk_has_dmabuf_export(ctx) ? 1 : 0,
           vxfec_vk_has_foreign_queue(ctx) ? 1 : 0, vxfec_vk_has_ext_host_mem(ctx) ? 1 : 0, vxfec_vk_shared_mem_bytes(ctx));

    struct {
        uint32_t K;
        uint32_t T;
        uint32_t blocks;
        uint32_t repairs;
    } enc_cases[] = {
        {10, 20, 3, 4},      /* unaligned 4-byte uint fallback */
        {100, 1280, 8, 10},  /* batched small block uvec4 */
        {256, 1280, 4, 16},  /* 256-symbol block uvec4 */
        {500, 1280, 2, 20},  /* medium block uvec4 */
        {1000, 1280, 2, 25}, /* 1k block uvec4 */
        {5000, 1280, 1, 20}, /* large block uvec4 */
    };

    for (size_t i = 0; i < sizeof(enc_cases) / sizeof(enc_cases[0]); i++) {
        if (!test_encode_case(ctx, enc_cases[i].K, enc_cases[i].T, enc_cases[i].blocks, enc_cases[i].repairs)) {
            fprintf(stderr, "FAIL: encode case %zu (K=%u)\n", i, enc_cases[i].K);
            vxfec_vk_ctx_free(ctx);
            return 1;
        }
        printf("PASS: GF(256) encode + mix bit-exact (K=%u, T=%u, Z=%u)\n", enc_cases[i].K, enc_cases[i].T, enc_cases[i].blocks);
    }

    if (!test_decode_gf2_case(ctx, 10, 20)) {
        fprintf(stderr, "FAIL: Pure-GF(2) 4-byte decode case K=10 T=20\n");
        vxfec_vk_ctx_free(ctx);
        return 1;
    }
    printf("PASS: Pure-GF(2) decode + mix 4-byte fallback (K=10, T=20)\n");

    uint32_t gf2_ks[] = {100, 256, 1000, 5000};
    for (size_t i = 0; i < sizeof(gf2_ks) / sizeof(gf2_ks[0]); i++) {
        if (!test_decode_gf2_case(ctx, gf2_ks[i], 1280)) {
            fprintf(stderr, "FAIL: GF(2) decode case K=%u\n", gf2_ks[i]);
            vxfec_vk_ctx_free(ctx);
            return 1;
        }
        printf("PASS: Pure-GF(2) decode + mix bit-exact (K=%u, T=1280)\n", gf2_ks[i]);
    }

    struct {
        uint32_t K, T, stride, drops;
    } gf256_dec_cases[] = {
        {1, 4, 4, 1},
        {32, 20, 20, 16},
        {100, 1280, 1312, 100},
        {500, 1280, 1312, 125},
    };
    for (size_t i = 0; i < sizeof(gf256_dec_cases) / sizeof(gf256_dec_cases[0]); i++) {
        if (!test_exact_k_gf256_decode(ctx, gf256_dec_cases[i].K, gf256_dec_cases[i].T, gf256_dec_cases[i].stride,
                                       gf256_dec_cases[i].drops)) {
            fprintf(stderr, "FAIL: exact-K GF(256) decode K=%u\n", gf256_dec_cases[i].K);
            vxfec_vk_ctx_free(ctx);
            return 1;
        }
        printf("PASS: Exact-K GF(256) decode bit-exact (K=%u, T=%u, drops=%u)\n", gf256_dec_cases[i].K, gf256_dec_cases[i].T,
               gf256_dec_cases[i].drops);
    }

    if (!test_edge_cases(ctx)) {
        fprintf(stderr, "FAIL: Vulkan edge/interop test cases\n");
        vxfec_vk_ctx_free(ctx);
        return 1;
    }
    printf("PASS: Hybrid permute, >65k ESI chunking, and schedule edge cases\n");

#ifdef NANORQ_HAS_GF2_MODE
    uint32_t h0_ks[] = {10, 100, 256, 1000, 5000};
    for (size_t i = 0; i < sizeof(h0_ks) / sizeof(h0_ks[0]); i++) {
        if (!test_h0_gf2_roundtrip(ctx, h0_ks[i], 1280)) {
            fprintf(stderr, "FAIL: H=0 GF(2) roundtrip K=%u\n", h0_ks[i]);
            vxfec_vk_ctx_free(ctx);
            return 1;
        }
        printf("PASS: Opt-in H=0 Pure-GF(2) encode + decode roundtrip (K=%u)\n", h0_ks[i]);
    }
#endif

    if (!test_async_timeline_ring(ctx)) {
        fprintf(stderr, "FAIL: Async timeline semaphore ring test\n");
        vxfec_vk_ctx_free(ctx);
        return 1;
    }
    printf("PASS: Async timeline-semaphore 12-deep ring submit/poll/wait\n");

#if !defined(_WIN32)
    if (!test_ext_memory_interop(ctx)) {
        fprintf(stderr, "FAIL: External memory DMA-BUF / host-ptr interop test\n");
        vxfec_vk_ctx_free(ctx);
        return 1;
    }
    printf("PASS: Zero-copy DMA-BUF cross-ctx + external host-ptr interop\n");
#endif

    struct {
        uint32_t K, M, T, Z, erasures;
    } rs8_cases[] = {
        {1, 1, 4, 1, 1}, {10, 6, 20, 3, 5}, {64, 16, 1280, 8, 12}, {200, 55, 1280, 4, 24}, {200, 56, 1280, 4, 24},
    };
    for (size_t i = 0; i < sizeof(rs8_cases) / sizeof(rs8_cases[0]); i++) {
        if (!test_cauchy_rs8_case(ctx, rs8_cases[i].K, rs8_cases[i].M, rs8_cases[i].T, rs8_cases[i].Z, rs8_cases[i].erasures)) {
            fprintf(stderr, "FAIL: Cauchy RS8 case K=%u M=%u\n", rs8_cases[i].K, rs8_cases[i].M);
            vxfec_vk_ctx_free(ctx);
            return 1;
        }
        printf("PASS: Cauchy RS8 MDS encode + zero-slack decode (K=%u, M=%u, T=%u, "
               "e=%u)\n",
               rs8_cases[i].K, rs8_cases[i].M, rs8_cases[i].T, rs8_cases[i].erasures);
    }

    vxfec_vk_ctx_free(ctx);
    printf("All vxfec Vulkan codec tests passed!\n");
    return 0;
}
