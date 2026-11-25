#include "vxfec_interleave.h"
#include "nanorq_core.h"
#include "nanorq_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool test_wire_ring_codec(vxfec_vk_ctx *ctx, int codec, const char *label, bool tight_zero_slack)
{
    const uint32_t Z = 24u;
    const uint32_t K = 96u;
    const uint32_t P = 32u;
    const uint32_t stride = 1024u;
    const uint64_t total_seq = (uint64_t)(K + P) * Z;
    const size_t sys_bytes = (size_t)K * Z * stride;
    const size_t wire_bytes = (size_t)(K + P) * Z * stride;

    vxfec_wire_ring *wring = vxfec_wire_ring_new(ctx, codec, Z, K, P, stride, 4u);
    vxfec_vk_buf *wire_vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
    uint8_t *orig_wire = (uint8_t *)malloc(wire_bytes);
    size_t wpb = vxfec_wire_ring_words_per_block(wring);
    uint64_t *bits = wring ? (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t)) : NULL;
    bool ok = false;

    if (!wring || !wire_vram || !orig_wire || !bits) {
        fprintf(stderr, "FAIL: wire_ring alloc (%s)\n", label);
        goto out;
    }

    /* bounds-checked sequence marking contract */
    if (vxfec_wire_mark_seq_checked(bits, 0u, K, P, wpb, 0u) || vxfec_wire_mark_seq_checked(bits, Z, K, P, wpb, total_seq) ||
        !vxfec_wire_mark_seq_checked(bits, Z, K, P, wpb, total_seq - 1u)) {
        fprintf(stderr, "FAIL: vxfec_wire_mark_seq_checked bounds check (%s)\n", label);
        goto out;
    }

    uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(wire_vram);
    memset(orig_wire, 0, wire_bytes);
    for (size_t i = 0; i < sys_bytes; i++)
        orig_wire[i] = (uint8_t)((i * 97u + (i >> 8) * 53u + 7u) & 0xffu);

    memcpy(stage, orig_wire, wire_bytes);
    if (!vxfec_vk_buf_upload(ctx, wire_vram, wire_bytes))
        goto out;

    uint64_t tx_ns = 0;
    if (!vxfec_wire_encode_vram(wring, wire_vram, &tx_ns) || !vxfec_vk_buf_download(ctx, wire_vram, wire_bytes)) {
        fprintf(stderr, "FAIL: vxfec_wire_encode_vram (%s)\n", label);
        goto out;
    }
    memcpy(orig_wire, stage, wire_bytes);

    /* subtest A: lossless fast-path */
    memset(bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
    for (uint64_t seq = 0; seq < total_seq; seq++)
        vxfec_wire_mark_seq(bits, Z, wpb, seq);
    vxfec_wire_stats st = {0};
    if (!vxfec_wire_recover_vram(wring, wire_vram, bits, &st) || st.blocks_repaired != 0u || st.symbols_repaired != 0u) {
        fprintf(stderr, "FAIL: wire_ring lossless fast-path (%s)\n", label);
        goto out;
    }

    /* subtest B: ~20% blackout + independent per-block random erasures, slots poisoned to 0xa5 */
    memcpy(stage, orig_wire, wire_bytes);
    memset(bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
    uint32_t rng = 0x9e3779b9u ^ (uint32_t)codec;
    for (uint32_t z = 0; z < Z; z++) {
        uint32_t dropped_sys = 0;
        for (uint32_t esi = 0; esi < K; esi++) {
            uint64_t seq = vxfec_interleave_wire_seq(Z, z, esi);
            bool drop = (seq >= 211ULL && seq < 667ULL);
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            if (!drop && dropped_sys < P - 4u && (rng % 100u) < 5u)
                drop = true;
            if (drop) {
                memset(stage + (size_t)seq * stride, 0xa5, stride);
                dropped_sys++;
            } else {
                vxfec_wire_mark_seq(bits, Z, wpb, seq);
            }
        }
        uint32_t keep_rep = tight_zero_slack ? dropped_sys : P;
        uint32_t kept = 0;
        for (uint32_t p = 0; p < P; p++) {
            uint64_t seq = vxfec_interleave_wire_seq(Z, z, K + p);
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            bool drop = (!tight_zero_slack && (rng % 100u) < 5u && (P - p) > (dropped_sys - kept + 4u));
            if (!drop && kept < keep_rep) {
                vxfec_wire_mark_seq(bits, Z, wpb, seq);
                kept++;
            } else {
                memset(stage + (size_t)seq * stride, 0x5a, stride);
            }
        }
    }

    if (!vxfec_vk_buf_upload(ctx, wire_vram, wire_bytes))
        goto out;

    memset(&st, 0, sizeof(st));
    if (!vxfec_wire_recover_vram(wring, wire_vram, bits, &st)) {
        fprintf(stderr, "FAIL: vxfec_wire_recover_vram (%s)\n", label);
        goto out;
    }
    if (!vxfec_vk_buf_download(ctx, wire_vram, sys_bytes))
        goto out;
    if (memcmp(stage, orig_wire, sys_bytes) != 0) {
        fprintf(stderr, "FAIL: wire_ring in-place payload mismatch (%s)\n", label);
        goto out;
    }

    /* subtest C: back-to-back WAW async timeline-semaphore submissions */
    memset(stage, 0x3c, sys_bytes);
    for (uint64_t seq = 0; seq < (uint64_t)K * Z; seq++) {
        uint32_t z = 0, esi = 0;
        vxfec_interleave_unwire_seq(Z, seq, &z, &esi);
        if ((bits[(size_t)z * wpb + (esi >> 6u)] >> (esi & 63u)) & 1ULL)
            memcpy(stage + (size_t)seq * stride, orig_wire + (size_t)seq * stride, stride);
    }
    if (!vxfec_vk_buf_upload(ctx, wire_vram, sys_bytes))
        goto out;
    uint64_t t1 = vxfec_wire_recover_submit(wring, wire_vram, bits, NULL);
    uint64_t t2 = vxfec_wire_recover_submit(wring, wire_vram, bits, NULL);
    if (t1 == 0u || t2 == 0u || !vxfec_vk_wait(ctx, t2, UINT64_MAX) || vxfec_vk_poll_status(ctx, t2) != 1 ||
        !vxfec_vk_buf_download(ctx, wire_vram, sys_bytes) || memcmp(stage, orig_wire, sys_bytes) != 0) {
        fprintf(stderr, "FAIL: wire_ring async pipelined submit mismatch (%s)\n", label);
        goto out;
    }

    /* subtest D: > P erasures cleanly rejected */
    memset(bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
    if (vxfec_wire_recover_vram(wring, wire_vram, bits, NULL)) {
        fprintf(stderr, "FAIL: wire_ring accepted unrecoverable all-zero mask (%s)\n", label);
        goto out;
    }

    printf("PASS: wire_ring %s in-place TX+RX (repaired=%u/%u blks, %u syms, masks=%u, gf2=%d, wall=%.2f ms)\n", label,
           st.blocks_repaired, Z, st.symbols_repaired, st.unique_masks, st.pure_gf2 ? 1 : 0, (double)st.total_wall_ns / 1e6);
    ok = true;

out:
    free(bits);
    free(orig_wire);
    vxfec_vk_buf_free(ctx, wire_vram);
    vxfec_wire_ring_free(wring);
    return ok;
}

/* test opportunistic GF(2) retry fallback when slack > 16 fails pure-GF(2) or falls back to GF(256) */
static bool test_opportunistic_gf2_fallback_safety(vxfec_vk_ctx *ctx)
{
    const uint32_t Z = 8u;
    const uint32_t K = 96u;
    const uint32_t P = 32u;
    const uint32_t stride = 512u;
    const size_t sys_bytes = (size_t)K * Z * stride;
    const size_t wire_bytes = (size_t)(K + P) * Z * stride;

    vxfec_wire_ring *wring = vxfec_wire_ring_new(ctx, VXFEC_WIRE_RATELESS, Z, K, P, stride, 2u);
    vxfec_vk_buf *vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
    uint8_t *orig_wire = (uint8_t *)malloc(wire_bytes);
    size_t wpb = vxfec_wire_ring_words_per_block(wring);
    uint64_t *bits = wring ? (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t)) : NULL;
    bool ok = false;

    if (!wring || !vram || !orig_wire || !bits)
        goto out;

    uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(vram);
    memset(orig_wire, 0, wire_bytes);
    for (size_t i = 0; i < sys_bytes; i++)
        orig_wire[i] = (uint8_t)((i * 113u + (i >> 7) * 29u + 17u) & 0xffu);
    memcpy(stage, orig_wire, wire_bytes);
    if (!vxfec_vk_buf_upload(ctx, vram, wire_bytes) || !vxfec_wire_encode_vram(wring, vram, NULL) ||
        !vxfec_vk_buf_download(ctx, vram, wire_bytes))
        goto out;
    memcpy(orig_wire, stage, wire_bytes);

    /* mix high-slack blocks (slack=18..20 -> overhead=32 path) and tight-slack blocks (slack=2..17) in one superframe */
    memcpy(stage, orig_wire, wire_bytes);
    memset(bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
    for (uint32_t z = 0; z < Z; z++) {
        uint32_t drop_sys = 12u + z; /* 12..19 missing systematic symbols */
        uint32_t keep_rep = (z & 1u) ? (drop_sys + 17u) : (drop_sys + 2u);
        if (keep_rep > P)
            keep_rep = P;
        for (uint32_t esi = 0; esi < K; esi++) {
            uint64_t seq = vxfec_interleave_wire_seq(Z, z, esi);
            if (esi < drop_sys)
                memset(stage + (size_t)seq * stride, 0xa5, stride);
            else
                vxfec_wire_mark_seq(bits, Z, wpb, seq);
        }
        for (uint32_t p = 0; p < keep_rep; p++) {
            uint64_t seq = vxfec_interleave_wire_seq(Z, z, K + p);
            vxfec_wire_mark_seq(bits, Z, wpb, seq);
        }
    }

    vxfec_wire_stats st = {0};
    if (!vxfec_vk_buf_upload(ctx, vram, wire_bytes) || !vxfec_wire_recover_vram(wring, vram, bits, &st) ||
        !vxfec_vk_buf_download(ctx, vram, sys_bytes) || memcmp(stage, orig_wire, sys_bytes) != 0) {
        fprintf(stderr, "FAIL: opportunistic GF(2) / GF(256) hybrid retry corrupted payload\n");
        goto out;
    }

    printf("PASS: opportunistic GF(2) + GF(256) hybrid fallback bit-exact (gf2=%u, gf256=%u)\n", st.gf2_blocks, st.gf256_blocks);
    ok = true;

out:
    free(bits);
    free(orig_wire);
    vxfec_vk_buf_free(ctx, vram);
    vxfec_wire_ring_free(wring);
    return ok;
}

static bool test_explicit_rfc6330_vs_gf2_mode(vxfec_vk_ctx *ctx)
{
    const uint32_t Z = 2u, K = 32u, P = 4u, stride = 256u;
    const size_t sys_bytes = (size_t)K * Z * stride;
    const size_t wire_bytes = (size_t)(K + P) * Z * stride;

#ifndef NANORQ_HAS_GF2_MODE
    /* without NANORQ_HAS_GF2_MODE, explicit H=0 mode must reject rather than silently emit RFC 6330 */
    vxfec_wire_ring *bad_w = vxfec_wire_ring_new(ctx, VXFEC_WIRE_RATELESS_GF2, Z, K, P, stride, 1u);
    if (bad_w) {
        vxfec_wire_ring_free(bad_w);
        fprintf(stderr, "FAIL: explicit GF2 H=0 mode succeeded without NANORQ_HAS_GF2_MODE\n");
        return false;
    }
#endif

    /* verify VXFEC_WIRE_RATELESS (0) always produces exact RFC 6330 parity matching nanorq_core_encoder_new */
    vxfec_wire_ring *wring = vxfec_wire_ring_new(ctx, VXFEC_WIRE_RATELESS, Z, K, P, stride, 1u);
    vxfec_vk_buf *vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
    if (!wring || !vram) {
        vxfec_vk_buf_free(ctx, vram);
        vxfec_wire_ring_free(wring);
        return false;
    }

    uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(vram);
    memset(stage, 0, wire_bytes);
    for (size_t i = 0; i < sys_bytes; i++)
        stage[i] = (uint8_t)((i * 61u + 13u) & 0xffu);

    bool ok = vxfec_vk_buf_upload(ctx, vram, wire_bytes) && vxfec_wire_encode_vram(wring, vram, NULL) &&
              vxfec_vk_buf_download(ctx, vram, wire_bytes);

    if (ok) {
        nanorq_core rq = {0};
        ok = nanorq_core_encoder_new(K, 0u, &rq);
        size_t plen = ok ? nanorq_core_calculate_prepare_memory(&rq) : 0u;
        size_t wlen = ok ? nanorq_core_calculate_work_memory(&rq) : 0u;
        size_t slen = ok ? ops_estimate_schedule_bytes(K) : 0u;
        uint8_t *pm = ok ? (uint8_t *)malloc(plen) : NULL;
        uint8_t *wm = ok ? (uint8_t *)malloc(wlen) : NULL;
        void *sm = ok ? malloc(slen) : NULL;
        schedule S = {0};
        uint32_t rows = ok ? nanorq_core_get_pc_rows(&rq) : 0u;
        uint32_t sh = ok ? nanorq_core_get_pc_genc_offset(&rq) : 0u;
        uint32_t cpu_stride = ok ? nanorq_core_recommended_stride(stride) : 0u;
        uint8_t *d_cpu = (ok && cpu_stride) ? (uint8_t *)calloc(rows, cpu_stride) : NULL;
        uint8_t *ref_par = (uint8_t *)calloc(1, cpu_stride ? cpu_stride : stride);

        if (!pm || !wm || !sm || !d_cpu || !ref_par || !schedule_init(&S, sm, slen) || !nanorq_core_prepare(&rq, pm, plen)) {
            ok = false;
        } else {
            nanorq_core_set_op_callback(&rq, &S, ops_push);
            if (!nanorq_core_precalculate(&rq, wm, wlen)) {
                ok = false;
            } else {
                for (uint32_t k = 0; k < K; k++) {
                    uint64_t seq = vxfec_interleave_wire_seq(Z, 0u, k);
                    memcpy(d_cpu + (size_t)(sh + k) * cpu_stride, stage + (size_t)seq * stride, stride);
                }
                ops_run(&rq, d_cpu, cpu_stride, &S);
                for (uint32_t p = 0; p < P && ok; p++) {
                    ops_mix(&rq, d_cpu, cpu_stride, K + p, ref_par);
                    uint64_t pseq = vxfec_interleave_wire_seq(Z, 0u, K + p);
                    if (memcmp(ref_par, stage + (size_t)pseq * stride, stride) != 0) {
                        fprintf(stderr, "FAIL: VXFEC_WIRE_RATELESS diverged from RFC 6330 parity at p=%u\n", p);
                        ok = false;
                    }
                }
            }
        }
        free(ref_par);
        free(d_cpu);
        free(sm);
        free(wm);
        free(pm);
    }

#ifdef NANORQ_HAS_GF2_MODE
    if (ok && !test_wire_ring_codec(ctx, VXFEC_WIRE_RATELESS_GF2, "Rateless H=0 (explicit GF2 mode)", false))
        ok = false;
#endif

    vxfec_vk_buf_free(ctx, vram);
    vxfec_wire_ring_free(wring);
    if (ok)
        printf("PASS: explicit RFC 6330 vs H=0 GF(2) wire mode contract verified\n");
    return ok;
}

static bool vk_is_required(void)
{
    const char *v = getenv("VXFEC_VK_REQUIRED");
    if (!v || !*v)
        v = getenv("NANORQ_VK_REQUIRED");
    return v && *v && strcmp(v, "0") != 0;
}

static bool test_fresh_lossless_async_recover(void)
{
    int status = VXFEC_VK_ERROR;
    vxfec_vk_ctx *fresh_ctx = vxfec_vk_ctx_new_status(&status);
    if (!fresh_ctx)
        return status == VXFEC_VK_UNAVAILABLE && !vk_is_required();

    const uint32_t Z = 8u;
    const uint32_t K = 32u;
    const uint32_t P = 8u;
    const uint32_t stride = 256u;
    const uint64_t sys_seq = (uint64_t)K * Z;
    const size_t wire_bytes = (size_t)(K + P) * Z * stride;

    vxfec_wire_ring *wring = vxfec_wire_ring_new(fresh_ctx, VXFEC_WIRE_RATELESS, Z, K, P, stride, 2u);
    vxfec_vk_buf *wire_vram = vxfec_vk_buf_new(fresh_ctx, wire_bytes, true);
    size_t wpb = vxfec_wire_ring_words_per_block(wring);
    uint64_t *bits = wring ? (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t)) : NULL;
    bool ok = false;

    if (!wring || !wire_vram || !bits)
        goto out;

    for (uint64_t seq = 0; seq < sys_seq; seq++)
        vxfec_wire_mark_seq(bits, Z, wpb, seq);

    /* submit 6 lossless async recoveries on a fresh timeline to exercise 0-work ticket signaling */
    for (int i = 0; i < 6; i++) {
        vxfec_wire_stats st = {0};
        uint64_t tk = vxfec_wire_recover_submit(wring, wire_vram, bits, &st);
        if (st.blocks_repaired != 0u || vxfec_vk_poll_status(fresh_ctx, tk) != 1 || !vxfec_vk_wait(fresh_ctx, tk, 500000000ULL)) {
            fprintf(stderr, "FAIL: fresh lossless async recover hung or failed at iter=%d\n", i);
            goto out;
        }
    }

    printf("PASS: wire_ring fresh timeline lossless async recover (6 submissions, 0 hangs)\n");
    ok = true;

out:
    free(bits);
    vxfec_vk_buf_free(fresh_ctx, wire_vram);
    vxfec_wire_ring_free(wring);
    vxfec_vk_ctx_free(fresh_ctx);
    return ok;
}

static bool test_directional_wire_rings(vxfec_vk_ctx *ctx)
{
    const uint32_t Z = 4u, K = 32u, P = 8u, stride = 256u;
    const size_t sys_bytes = (size_t)Z * K * stride;
    const size_t wire_bytes = (size_t)Z * (K + P) * stride;

    if (vxfec_wire_ring_new_ex(ctx, VXFEC_WIRE_MDS, Z, K, P, stride, 2u, false, false) != NULL)
        return false;

    vxfec_wire_ring *tx_only = vxfec_wire_ring_new_ex(ctx, VXFEC_WIRE_MDS, Z, K, P, stride, 4u, true, false);
    vxfec_wire_ring *rx_only = vxfec_wire_ring_new_ex(ctx, VXFEC_WIRE_MDS, Z, K, P, stride, 2u, false, true);
    vxfec_vk_buf *vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
    uint8_t *ref = (uint8_t *)malloc(sys_bytes);
    size_t wpb = vxfec_wire_ring_words_per_block(rx_only);
    uint64_t *bits = wpb ? (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t)) : NULL;
    bool ok = false;

    if (!tx_only || !rx_only || !vram || !ref || !bits)
        goto out;
    if (vxfec_wire_ring_workers(tx_only) != 0u || vxfec_wire_ring_workers(rx_only) != 2u || vxfec_wire_can_submit(tx_only))
        goto out;

    uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(vram);
    memset(stage, 0, wire_bytes);
    for (size_t i = 0; i < sys_bytes; i++)
        ref[i] = stage[i] = (uint8_t)((i * 53u + 17u) & 0xffu);

    /* TX-only ring encodes parity and rejects RX recovery; RX-only ring rejects TX encode and recovers erasures */
    if (!vxfec_vk_buf_upload(ctx, vram, wire_bytes) || !vxfec_wire_encode_vram(tx_only, vram, NULL) ||
        vxfec_wire_recover_vram(tx_only, vram, bits, NULL) || vxfec_wire_encode_vram(rx_only, vram, NULL) ||
        vxfec_wire_encode_submit(rx_only, vram) != 0u || !vxfec_vk_buf_download(ctx, vram, wire_bytes))
        goto out;

    uint64_t total_seq = (uint64_t)(K + P) * Z;
    for (uint64_t seq = 0; seq < total_seq; seq++) {
        if (seq >= 2u * Z && seq < 5u * Z) {
            memset(stage + (size_t)seq * stride, 0xcc, stride);
            continue;
        }
        vxfec_wire_mark_seq(bits, Z, wpb, seq);
    }
    if (!vxfec_vk_buf_upload(ctx, vram, wire_bytes) || !vxfec_wire_recover_vram(rx_only, vram, bits, NULL) ||
        !vxfec_vk_buf_download(ctx, vram, sys_bytes) || memcmp(stage, ref, sys_bytes) != 0)
        goto out;

    printf("PASS: directional wire_ring_new_ex (TX-only 0 RX workers, RX-only 0 TX table) bit-exact\n");
    ok = true;

out:
    free(bits);
    free(ref);
    vxfec_vk_buf_free(ctx, vram);
    vxfec_wire_ring_free(rx_only);
    vxfec_wire_ring_free(tx_only);
    return ok;
}

static bool test_rq_101_missing_and_exact_p(vxfec_vk_ctx *ctx)
{
    /* test 1: 101 missing source symbols with P=160 (> 101+32) and after discarding 25 early repairs */
    {
        const uint32_t Z = 4u, K = 200u, P = 160u, stride = 256u;
        const size_t sys_bytes = (size_t)Z * K * stride;
        const size_t wire_bytes = (size_t)Z * (K + P) * stride;
        vxfec_wire_ring *wring = vxfec_wire_ring_new(ctx, VXFEC_WIRE_RATELESS, Z, K, P, stride, 2u);
        vxfec_vk_buf *vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
        uint8_t *orig = (uint8_t *)malloc(wire_bytes);
        size_t wpb = vxfec_wire_ring_words_per_block(wring);
        uint64_t *bits = wring ? (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t)) : NULL;
        bool sub_ok = false;
        if (!wring || !vram || !orig || !bits)
            goto sub1_out;

        uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(vram);
        memset(orig, 0, wire_bytes);
        for (size_t i = 0; i < sys_bytes; i++)
            orig[i] = (uint8_t)((i * 89u + (i >> 6) * 31u + 5u) & 0xffu);
        memcpy(stage, orig, wire_bytes);
        if (!vxfec_vk_buf_upload(ctx, vram, wire_bytes) || !vxfec_wire_encode_vram(wring, vram, NULL) ||
            !vxfec_vk_buf_download(ctx, vram, wire_bytes))
            goto sub1_out;
        memcpy(orig, stage, wire_bytes);

        for (int pass = 0; pass < 2; pass++) {
            memcpy(stage, orig, wire_bytes);
            memset(bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
            for (uint32_t z = 0; z < Z; z++) {
                uint32_t m = 0, rng = 0x9e3779b9u * (z + 1u) + 101u;
                for (uint32_t k = 0; k < K; k++) {
                    bool drop = false;
                    if (m < 101u) {
                        rng ^= rng << 13;
                        rng ^= rng >> 17;
                        rng ^= rng << 5;
                        if ((K - k) == (101u - m) || (rng % (K - k)) < (101u - m))
                            drop = true;
                    }
                    uint64_t seq = vxfec_interleave_wire_seq(Z, z, k);
                    if (drop) {
                        memset(stage + (size_t)seq * stride, 0xa5, stride);
                        m++;
                    } else {
                        vxfec_wire_mark_seq(bits, Z, wpb, seq);
                    }
                }
                uint32_t p_start = (pass == 0) ? 0u : 25u;
                for (uint32_t p = p_start; p < P; p++)
                    vxfec_wire_mark_seq(bits, Z, wpb, vxfec_interleave_wire_seq(Z, z, K + p));
            }
            if (!vxfec_vk_buf_upload(ctx, vram, wire_bytes) || !vxfec_wire_recover_vram(wring, vram, bits, NULL) ||
                !vxfec_vk_buf_download(ctx, vram, sys_bytes) || memcmp(stage, orig, sys_bytes) != 0) {
                fprintf(stderr, "FAIL: RaptorQ 101-missing recovery failed on pass=%d\n", pass);
                goto sub1_out;
            }
        }
        sub_ok = true;
    sub1_out:
        free(bits);
        free(orig);
        vxfec_vk_buf_free(ctx, vram);
        vxfec_wire_ring_free(wring);
        if (!sub_ok)
            return false;
    }

    /* test 2: exact boundary missing == P == 101 (0-slack exact P parity symbols) */
    {
        const uint32_t Z = 2u, K = 128u, P = 101u, stride = 256u;
        const size_t sys_bytes = (size_t)Z * K * stride;
        const size_t wire_bytes = (size_t)Z * (K + P) * stride;
        const int codecs[2] = {VXFEC_WIRE_RATELESS, VXFEC_WIRE_MDS};
        for (int ci = 0; ci < 2; ci++) {
            vxfec_wire_ring *wring = vxfec_wire_ring_new(ctx, codecs[ci], Z, K, P, stride, 1u);
            vxfec_vk_buf *vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
            uint8_t *orig = (uint8_t *)malloc(wire_bytes);
            size_t wpb = vxfec_wire_ring_words_per_block(wring);
            uint64_t *bits = wring ? (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t)) : NULL;
            bool sub_ok = false;
            if (!wring || !vram || !orig || !bits)
                goto sub2_out;

            uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(vram);
            memset(orig, 0, wire_bytes);
            for (size_t i = 0; i < sys_bytes; i++)
                orig[i] = (uint8_t)((i * 67u + 23u) & 0xffu);
            memcpy(stage, orig, wire_bytes);
            if (!vxfec_vk_buf_upload(ctx, vram, wire_bytes) || !vxfec_wire_encode_vram(wring, vram, NULL) ||
                !vxfec_vk_buf_download(ctx, vram, wire_bytes))
                goto sub2_out;
            memcpy(orig, stage, wire_bytes);

            memset(bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
            for (uint32_t z = 0; z < Z; z++) {
                for (uint32_t k = 0; k < K; k++) {
                    uint64_t seq = vxfec_interleave_wire_seq(Z, z, k);
                    if (k < 101u)
                        memset(stage + (size_t)seq * stride, 0x5a, stride);
                    else
                        vxfec_wire_mark_seq(bits, Z, wpb, seq);
                }
                for (uint32_t p = 0; p < P; p++)
                    vxfec_wire_mark_seq(bits, Z, wpb, vxfec_interleave_wire_seq(Z, z, K + p));
            }
            sub_ok = vxfec_vk_buf_upload(ctx, vram, wire_bytes) && vxfec_wire_recover_vram(wring, vram, bits, NULL) &&
                     vxfec_vk_buf_download(ctx, vram, sys_bytes) && memcmp(stage, orig, sys_bytes) == 0;
        sub2_out:
            free(bits);
            free(orig);
            vxfec_vk_buf_free(ctx, vram);
            vxfec_wire_ring_free(wring);
            if (!sub_ok) {
                fprintf(stderr, "FAIL: exact missing==P==101 recovery failed for codec=%d\n", codecs[ci]);
                return false;
            }
        }
    }

    /*
     * subtest 3: non-GF(2) initial decode with surplus repairs (max_slack > 16)
     * exercising the pure-GF(2) retry path without corrupting dec or S_dec when
     * the retry schedule remains over GF(256) or fails.
     */
    {
        const uint32_t Z = 4u, K_sm = 10u, P_sm = 40u, stride = 512u;
        const size_t sys_sm = (size_t)Z * K_sm * stride;
        const size_t wire_sm = (size_t)Z * (K_sm + P_sm) * stride;
        vxfec_wire_ring *wring = vxfec_wire_ring_new(ctx, VXFEC_WIRE_RATELESS, Z, K_sm, P_sm, stride, 2u);
        vxfec_vk_buf *vram = vxfec_vk_buf_new(ctx, wire_sm, true);
        uint8_t *orig = (uint8_t *)malloc(wire_sm);
        size_t wpb = vxfec_wire_ring_words_per_block(wring);
        uint64_t *bits = wring ? (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t)) : NULL;
        bool sub_ok = false;
        if (!wring || !vram || !orig || !bits)
            goto sub3_out;

        uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(vram);
        memset(orig, 0, wire_sm);
        for (size_t i = 0; i < sys_sm; i++)
            orig[i] = (uint8_t)((i * 53u + 11u) & 0xffu);
        memcpy(stage, orig, wire_sm);
        if (!vxfec_vk_buf_upload(ctx, vram, wire_sm) || !vxfec_wire_encode_vram(wring, vram, NULL) ||
            !vxfec_vk_buf_download(ctx, vram, wire_sm))
            goto sub3_out;
        memcpy(orig, stage, wire_sm);

        memset(bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
        for (uint32_t z = 0; z < Z; z++) {
            for (uint32_t k = 0; k < K_sm; k++) {
                uint64_t seq = vxfec_interleave_wire_seq(Z, z, k);
                if (k < 4u)
                    memset(stage + (size_t)seq * stride, 0x3cu, stride);
                else
                    vxfec_wire_mark_seq(bits, Z, wpb, seq);
            }
            /* 26 repairs for 4 missing -> max_slack = 22 > WIRE_RQ_FAST_OVERHEAD (16) */
            for (uint32_t p = 0; p < 26u; p++)
                vxfec_wire_mark_seq(bits, Z, wpb, vxfec_interleave_wire_seq(Z, z, K_sm + p));
        }
        sub_ok = vxfec_vk_buf_upload(ctx, vram, wire_sm) && vxfec_wire_recover_vram(wring, vram, bits, NULL) &&
                 vxfec_vk_buf_download(ctx, vram, sys_sm) && memcmp(stage, orig, sys_sm) == 0;
    sub3_out:
        free(bits);
        free(orig);
        vxfec_vk_buf_free(ctx, vram);
        vxfec_wire_ring_free(wring);
        if (!sub_ok) {
            fprintf(stderr, "FAIL: non-GF(2) surplus-repair retry corrupted decode\n");
            return false;
        }
    }

    printf("PASS: 101-missing RaptorQ recovery (all repairs + drop-25 + exact missing==P + retry safety) bit-exact\n");
    return true;
}

static bool test_partial_recovery_and_dynamic_tile_dedup(vxfec_vk_ctx *ctx)
{
    const uint32_t Z = 8u, K = 64u, P = 24u, stride = 256u;
    const size_t sys_bytes = (size_t)Z * K * stride;
    const size_t wire_bytes = (size_t)Z * (K + P) * stride;
    vxfec_wire_ring *wring = vxfec_wire_ring_new(ctx, VXFEC_WIRE_MDS, Z, K, P, stride, 2u);
    vxfec_vk_buf *vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
    uint8_t *orig = (uint8_t *)malloc(wire_bytes);
    size_t wpb = vxfec_wire_ring_words_per_block(wring);
    uint64_t *bits = wring ? (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t)) : NULL;
    bool ok = false;

    if (!wring || !vram || !orig || !bits)
        goto out;

    uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(vram);
    memset(orig, 0, wire_bytes);
    for (size_t i = 0; i < sys_bytes; i++)
        orig[i] = (uint8_t)((i * 71u + 19u) & 0xffu);
    memcpy(stage, orig, wire_bytes);
    if (!vxfec_vk_buf_upload(ctx, vram, wire_bytes) || !vxfec_wire_encode_vram(wring, vram, NULL) ||
        !vxfec_vk_buf_download(ctx, vram, wire_bytes))
        goto out;
    memcpy(orig, stage, wire_bytes);

    /*
     * 1. surplus parity mask deduplication + dynamic tile_e=8 downshift (max_missing=4 <= 8 while P=24):
     * all Z blocks erase [0..3), and each block randomly drops different unused tail parities in [4..24).
     */
    memcpy(stage, orig, wire_bytes);
    memset(bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
    for (uint32_t z = 0; z < Z; z++) {
        for (uint32_t k = 0; k < K; k++) {
            uint64_t seq = vxfec_interleave_wire_seq(Z, z, k);
            if (k < 4u)
                memset(stage + (size_t)seq * stride, 0xaa, stride);
            else
                vxfec_wire_mark_seq(bits, Z, wpb, seq);
        }
        for (uint32_t p = 0; p < P; p++) {
            uint64_t seq = vxfec_interleave_wire_seq(Z, z, K + p);
            if (p >= 4u && ((p + z) % 3u) == 0u)
                memset(stage + (size_t)seq * stride, 0x55, stride);
            else
                vxfec_wire_mark_seq(bits, Z, wpb, seq);
        }
    }
    vxfec_wire_stats st = {0};
    if (!vxfec_vk_buf_upload(ctx, vram, wire_bytes) || !vxfec_wire_recover_vram(wring, vram, bits, &st) ||
        !vxfec_vk_buf_download(ctx, vram, sys_bytes) || st.unique_masks != 1u || st.blocks_repaired != Z ||
        st.blocks_failed != 0u || memcmp(stage, orig, sys_bytes) != 0) {
        fprintf(stderr, "FAIL: surplus parity dedup / tile_e=8 downshift (masks=%u rep=%u)\n", st.unique_masks, st.blocks_repaired);
        goto out;
    }

    /*
     * 2. partial superframe recovery: block 0 loses P+1 symbols (unrecoverable),
     * while blocks 1..Z-1 lose 12 symbols (tile_e=16 path) and must still be repaired!
     */
    memcpy(stage, orig, wire_bytes);
    memset(bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
    for (uint32_t z = 0; z < Z; z++) {
        uint32_t drop_cnt = (z == 0u) ? (P + 1u) : 12u;
        for (uint32_t k = 0; k < K; k++) {
            uint64_t seq = vxfec_interleave_wire_seq(Z, z, k);
            if (k < drop_cnt)
                memset(stage + (size_t)seq * stride, 0xcc, stride);
            else
                vxfec_wire_mark_seq(bits, Z, wpb, seq);
        }
        for (uint32_t p = 0; p < P; p++)
            vxfec_wire_mark_seq(bits, Z, wpb, vxfec_interleave_wire_seq(Z, z, K + p));
    }
    memset(&st, 0, sizeof(st));
    if (!vxfec_vk_buf_upload(ctx, vram, wire_bytes) || vxfec_wire_recover_vram(wring, vram, bits, &st) || st.blocks_failed != 1u ||
        st.blocks_repaired != Z - 1u || !vxfec_vk_buf_download(ctx, vram, sys_bytes)) {
        fprintf(stderr, "FAIL: partial superframe recovery stats (rep=%u failed=%u)\n", st.blocks_repaired, st.blocks_failed);
        goto out;
    }
    for (uint32_t z = 1u; z < Z; z++) {
        for (uint32_t k = 0; k < K; k++) {
            uint64_t seq = vxfec_interleave_wire_seq(Z, z, k);
            if (memcmp(stage + (size_t)seq * stride, orig + (size_t)seq * stride, stride) != 0) {
                fprintf(stderr, "FAIL: solvable sibling block z=%u was not repaired\n", z);
                goto out;
            }
        }
    }

    /* 3. 0-erasure async recover_submit returns 1u instead of coupling to prior ticket */
    memset(bits, 0xff, (size_t)Z * wpb * sizeof(uint64_t));
    uint64_t lossless_tk = vxfec_wire_recover_submit(wring, vram, bits, &st);
    if (vxfec_vk_has_timeline(ctx) && lossless_tk != 1u) {
        fprintf(stderr, "FAIL: expected 0-work wire_recover_submit ticket=1, got %llu\n", (unsigned long long)lossless_tk);
        goto out;
    }

    printf("PASS: surplus-parity dedup, dynamic tile_e=8/16, partial recovery & 0-work ticket=1\n");
    ok = true;

out:
    free(bits);
    free(orig);
    vxfec_vk_buf_free(ctx, vram);
    vxfec_wire_ring_free(wring);
    return ok;
}

static bool test_wire_rq_tile_widths(vxfec_vk_ctx *ctx)
{
    static const struct {
        uint32_t K;
        uint32_t P;
    } shapes[2] = {{96u, 36u}, {256u, 40u}};
    static const char *tiles[3] = {"8", "16", "32"};
    const uint32_t Z = 4u, stride = 256u;

    for (size_t si = 0; si < 2u; si++) {
        uint32_t K = shapes[si].K, P = shapes[si].P;
        size_t sys_bytes = (size_t)Z * K * stride;
        size_t wire_bytes = (size_t)Z * (K + P) * stride;

        unsetenv("VXFEC_WIRE_TILE_E");
        vxfec_wire_ring *auto_ring = vxfec_wire_ring_new(ctx, VXFEC_WIRE_RATELESS, Z, K, P, stride, 1u);
        vxfec_vk_buf *vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
        uint8_t *ref_wire = (uint8_t *)malloc(wire_bytes);
        if (!auto_ring || !vram || !ref_wire) {
            free(ref_wire);
            vxfec_vk_buf_free(ctx, vram);
            vxfec_wire_ring_free(auto_ring);
            return false;
        }

        uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(vram);
        memset(stage, 0, wire_bytes);
        for (size_t i = 0; i < sys_bytes; i++)
            stage[i] = (uint8_t)((i * 107u + (i >> 7) * 43u + 29u) & 0xffu);
        bool ok = vxfec_vk_buf_upload(ctx, vram, wire_bytes) && vxfec_wire_encode_vram(auto_ring, vram, NULL) &&
                  vxfec_vk_buf_download(ctx, vram, wire_bytes);
        memcpy(ref_wire, stage, wire_bytes);

        /* also exercise async vxfec_wire_encode_submit on the default tile path */
        if (ok) {
            memset(stage + sys_bytes, 0xa5, wire_bytes - sys_bytes);
            uint64_t tk = vxfec_vk_buf_upload(ctx, vram, wire_bytes) ? vxfec_wire_encode_submit(auto_ring, vram) : 0u;
            ok = tk != 0u && vxfec_vk_wait(ctx, tk, UINT64_MAX) && vxfec_vk_buf_download(ctx, vram, wire_bytes) &&
                 memcmp(stage, ref_wire, wire_bytes) == 0;
        }
        vxfec_wire_ring_free(auto_ring);

        for (size_t ti = 0; ti < 3u && ok; ti++) {
            setenv("VXFEC_WIRE_TILE_E", tiles[ti], 1);
            vxfec_wire_ring *dir_ring = vxfec_wire_ring_new(ctx, VXFEC_WIRE_RATELESS, Z, K, P, stride, 1u);
            memset(stage + sys_bytes, 0x5a, wire_bytes - sys_bytes);
            ok = dir_ring && vxfec_vk_buf_upload(ctx, vram, wire_bytes) && vxfec_wire_encode_vram(dir_ring, vram, NULL) &&
                 vxfec_vk_buf_download(ctx, vram, wire_bytes) && memcmp(stage, ref_wire, wire_bytes) == 0;
            if (ok) {
                /* Poison parity again so async encoding must produce every output. */
                memset(stage + sys_bytes, 0xa5, wire_bytes - sys_bytes);
                uint64_t tk = vxfec_vk_buf_upload(ctx, vram, wire_bytes) ? vxfec_wire_encode_submit(dir_ring, vram) : 0u;
                ok = tk != 0u && vxfec_vk_wait(ctx, tk, UINT64_MAX) && vxfec_vk_buf_download(ctx, vram, wire_bytes) &&
                     memcmp(stage, ref_wire, wire_bytes) == 0;
            }
            vxfec_wire_ring_free(dir_ring);
        }
        unsetenv("VXFEC_WIRE_TILE_E");
        free(ref_wire);
        vxfec_vk_buf_free(ctx, vram);
        if (!ok) {
            fprintf(stderr, "FAIL: wire_rq_lds vs tile_e=8/16/32 mismatch at K=%u P=%u\n", K, P);
            return false;
        }
    }

    printf("PASS: wire_ring tile_e={8,16,32} sync+async bit-exact across K=96 and K=256\n");
    return true;
}

int main(void)
{
    int status = VXFEC_VK_ERROR;
    vxfec_vk_ctx *ctx = vxfec_vk_ctx_new_status(&status);
    if (!ctx) {
        if (status == VXFEC_VK_UNAVAILABLE && !vk_is_required()) {
            fprintf(stderr, "SKIP: no Vulkan 1.2 device available\n");
            return 0;
        }
        fprintf(stderr, "FAIL: Vulkan context init (%s)\n", status == VXFEC_VK_UNAVAILABLE ? "unavailable" : "failed");
        return 1;
    }

    const uint32_t Z = 32u;
    const uint32_t K = 128u;
    const uint32_t P = 48u;
    const uint32_t T = 1024u;
    const uint64_t total_frames = (uint64_t)(K + P) * Z;
    const size_t sys_bytes = (size_t)Z * K * T;
    const size_t wire_bytes = (size_t)Z * (K + P) * T;

    vxfec_wire_ring *ring = vxfec_wire_ring_new(ctx, VXFEC_WIRE_RATELESS, Z, K, P, T, 4u);
    vxfec_vk_buf *wire_vram = vxfec_vk_buf_new(ctx, wire_bytes, true);
    uint8_t *orig_wire = (uint8_t *)calloc(1, wire_bytes);
    size_t wpb = vxfec_wire_ring_words_per_block(ring);
    uint64_t *rx_bits = ring ? (uint64_t *)calloc((size_t)Z * wpb, sizeof(uint64_t)) : NULL;
    int rc = 1;

    if (!ring || !wire_vram || !orig_wire || !rx_bits) {
        fprintf(stderr, "FAIL: wire_ring superframe allocation\n");
        goto done;
    }

    uint8_t *stage = (uint8_t *)vxfec_vk_buf_staging_ptr(wire_vram);
    for (size_t i = 0; i < sys_bytes; i++)
        orig_wire[i] = (uint8_t)((i * 73u + (i >> 9) * 41u + 19u) & 0xffu);
    memcpy(stage, orig_wire, wire_bytes);
    if (!vxfec_vk_buf_upload(ctx, wire_vram, wire_bytes) || !vxfec_wire_encode_vram(ring, wire_vram, NULL) ||
        !vxfec_vk_buf_download(ctx, wire_vram, wire_bytes)) {
        fprintf(stderr, "FAIL: vxfec_wire_encode_vram\n");
        goto done;
    }
    memcpy(orig_wire, stage, wire_bytes);

    /* case 1: 0% lossless superframe -> 0 blocks repaired, 0 unique masks */
    memset(rx_bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
    for (uint64_t seq = 0; seq < (uint64_t)K * Z; seq++)
        vxfec_wire_mark_seq(rx_bits, Z, wpb, seq);
    vxfec_wire_stats st = {0};
    if (!vxfec_wire_recover_vram(ring, wire_vram, rx_bits, &st) || st.blocks_repaired != 0u || st.unique_masks != 0u) {
        fprintf(stderr, "FAIL: lossless superframe (repaired=%u masks=%u)\n", st.blocks_repaired, st.unique_masks);
        goto done;
    }
    printf("PASS: 2D wire_ring lossless fast-path (Z=%u, K=%u, T=%u, 0 FEC decodes)\n", Z, K, T);

    /*
     * case 2: contiguous ~26% blackout erasing 1,070 consecutive systematic
     * wire frames across [307 .. 1377). Row-major interleaving spreads the
     * blackout to 33..34 erasures per block and deduplicates all 32 blocks
     * into <= 3 unique masks.
     */
    memcpy(stage, orig_wire, wire_bytes);
    memset(rx_bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
    const uint64_t burst_beg = 307ULL;
    const uint64_t burst_end = 1377ULL;
    for (uint64_t seq = 0; seq < total_frames; seq++) {
        if (seq >= burst_beg && seq < burst_end) {
            memset(stage + (size_t)seq * T, 0xa5, T);
            continue;
        }
        vxfec_wire_mark_seq(rx_bits, Z, wpb, seq);
    }
    memset(&st, 0, sizeof(st));
    if (!vxfec_vk_buf_upload(ctx, wire_vram, wire_bytes) || !vxfec_wire_recover_vram(ring, wire_vram, rx_bits, &st) ||
        !vxfec_vk_buf_download(ctx, wire_vram, sys_bytes) || st.blocks_repaired != Z || st.unique_masks > 3u ||
        memcmp(stage, orig_wire, sys_bytes) != 0) {
        fprintf(stderr, "FAIL: contiguous burst recovery (repaired=%u masks=%u)\n", st.blocks_repaired, st.unique_masks);
        goto done;
    }
    printf("PASS: 2D wire_ring 1070-frame contiguous blackout recovery (repaired=%u/%u blocks via %u unique masks)\n",
           st.blocks_repaired, Z, st.unique_masks);

    /* case 3: contiguous blackout + random uncorrelated erasures across both sys & parity */
    memcpy(stage, orig_wire, wire_bytes);
    memset(rx_bits, 0, (size_t)Z * wpb * sizeof(uint64_t));
    uint32_t rng = 0x12345678u;
    uint32_t dropped_total = 0;
    for (uint64_t seq = 0; seq < total_frames; seq++) {
        bool drop = (seq >= 600ULL && seq < 1240ULL);
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        if (!drop && (rng % 100u) < 4u)
            drop = true;
        if (drop) {
            memset(stage + (size_t)seq * T, 0x5a, T);
            dropped_total++;
        } else {
            vxfec_wire_mark_seq(rx_bits, Z, wpb, seq);
        }
    }
    memset(&st, 0, sizeof(st));
    if (!vxfec_vk_buf_upload(ctx, wire_vram, wire_bytes) || !vxfec_wire_recover_vram(ring, wire_vram, rx_bits, &st) ||
        !vxfec_vk_buf_download(ctx, wire_vram, sys_bytes) || memcmp(stage, orig_wire, sys_bytes) != 0) {
        fprintf(stderr, "FAIL: mixed burst + random loss recovery\n");
        goto done;
    }
    printf("PASS: 2D wire_ring mixed burst + random erasures (dropped=%u frames, repaired=%u blocks, masks=%u)\n", dropped_total,
           st.blocks_repaired, st.unique_masks);

    if (!test_opportunistic_gf2_fallback_safety(ctx) || !test_explicit_rfc6330_vs_gf2_mode(ctx) ||
        !test_fresh_lossless_async_recover() || !test_directional_wire_rings(ctx) || !test_rq_101_missing_and_exact_p(ctx) ||
        !test_partial_recovery_and_dynamic_tile_dedup(ctx) || !test_wire_rq_tile_widths(ctx) ||
        !test_wire_ring_codec(ctx, VXFEC_WIRE_RATELESS, "Rateless (GF2 fast-path)", false) ||
        !test_wire_ring_codec(ctx, VXFEC_WIRE_RATELESS, "Rateless (0-slack GF256 fallback)", true) ||
        !test_wire_ring_codec(ctx, VXFEC_WIRE_MDS, "Strict MDS (GF256 nibble)", false))
        goto done;

    printf("All 2D VRAM wire-interleaver tests passed!\n");
    rc = 0;

done:
    free(rx_bits);
    free(orig_wire);
    vxfec_vk_buf_free(ctx, wire_vram);
    vxfec_wire_ring_free(ring);
    vxfec_vk_ctx_free(ctx);
    return rc;
}
