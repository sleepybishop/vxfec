#ifndef VXFEC_INTERLEAVE_H
#define VXFEC_INTERLEAVE_H

#include "vxfec_vk.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * zero-copy wire-strided VRAM superframe pipeline (vxfec_wire_ring).
 * operates in-place on a VRAM ring buffer ((K + P) * Z * stride bytes,
 * stride % 16 == 0) in wire arrival order: (esi * Z + block_z) * stride.
 */
static inline uint64_t vxfec_interleave_wire_seq(uint32_t Z, uint32_t block_z, uint32_t esi)
{
    return (uint64_t)esi * (uint64_t)Z + (uint64_t)block_z;
}

static inline void vxfec_interleave_unwire_seq(uint32_t Z, uint64_t wire_seq, uint32_t *out_block_z, uint32_t *out_esi)
{
    if (Z == 0) {
        *out_block_z = 0;
        *out_esi = 0;
        return;
    }
    *out_block_z = (uint32_t)(wire_seq % (uint64_t)Z);
    *out_esi = (uint32_t)(wire_seq / (uint64_t)Z);
}

/* explicit wire modes: RATELESS = RFC 6330, MDS = Cauchy RS8, RATELESS_GF2 = opt-in H=0 GF(2) */
enum {
    VXFEC_WIRE_RATELESS = 0,
    VXFEC_WIRE_MDS = 1,
    VXFEC_WIRE_RATELESS_GF2 = 2,
};

typedef struct vxfec_wire_ring vxfec_wire_ring;

typedef struct {
    uint64_t cpu_compile_ns;
    uint64_t gpu_exec_ns;
    uint64_t total_wall_ns;
    size_t tab_bytes;
    uint32_t blocks_repaired;
    uint32_t blocks_failed;
    uint32_t unique_masks;
    uint32_t gf2_blocks;
    uint32_t gf256_blocks;
    uint32_t symbols_repaired;
    uint32_t workers_used;
    bool pure_gf2;
} vxfec_wire_stats;

vxfec_wire_ring *vxfec_wire_ring_new(vxfec_vk_ctx *ctx, int codec, uint32_t Z, uint32_t K, uint32_t P, uint32_t stride,
                                     uint32_t num_threads);
vxfec_wire_ring *vxfec_wire_ring_new_ex(vxfec_vk_ctx *ctx, int codec, uint32_t Z, uint32_t K, uint32_t P, uint32_t stride,
                                        uint32_t num_threads, bool enable_tx, bool enable_rx);
void vxfec_wire_ring_free(vxfec_wire_ring *wring);

size_t vxfec_wire_ring_words_per_block(const vxfec_wire_ring *wring);
uint32_t vxfec_wire_ring_workers(const vxfec_wire_ring *wring);
bool vxfec_wire_ring_tx_is_pure_gf2(const vxfec_wire_ring *wring);

static inline bool vxfec_wire_mark_seq_checked(uint64_t *rx_block_bits, uint32_t Z, uint32_t K, uint32_t P, size_t words_per_blk,
                                               uint64_t wire_seq)
{
    if (!rx_block_bits || Z == 0 || words_per_blk == 0)
        return false;
    uint64_t total_slots = ((uint64_t)K + (uint64_t)P) * (uint64_t)Z;
    if (wire_seq >= total_slots)
        return false;
    uint32_t z = (uint32_t)(wire_seq % (uint64_t)Z);
    uint32_t esi = (uint32_t)(wire_seq / (uint64_t)Z);
    size_t word_idx = (esi >> 6u);
    if (word_idx >= words_per_blk)
        return false;
    rx_block_bits[(size_t)z * words_per_blk + word_idx] |= (1ULL << (esi & 63u));
    return true;
}

static inline void vxfec_wire_mark_seq(uint64_t *rx_block_bits, uint32_t Z, size_t words_per_blk, uint64_t wire_seq)
{
    if (!rx_block_bits || Z == 0 || words_per_blk == 0)
        return;
    uint32_t z = (uint32_t)(wire_seq % (uint64_t)Z);
    uint64_t esi = wire_seq / (uint64_t)Z;
    uint64_t word_idx = esi >> 6u;
    if (word_idx >= (uint64_t)words_per_blk)
        return;
    rx_block_bits[(size_t)z * words_per_blk + (size_t)word_idx] |= (1ULL << (esi & 63u));
}

bool vxfec_wire_encode_vram(vxfec_wire_ring *wring, vxfec_vk_buf *wire_vram, uint64_t *gpu_ns_out);
bool vxfec_wire_recover_vram(vxfec_wire_ring *wring, vxfec_vk_buf *wire_vram, const uint64_t *rx_block_bits,
                             vxfec_wire_stats *out_stats);

/* non-blocking timeline-semaphore submissions; returns ticket > 0 or 0 if busy/failed */
bool vxfec_wire_can_submit(vxfec_wire_ring *wring);
uint64_t vxfec_wire_encode_submit(vxfec_wire_ring *wring, vxfec_vk_buf *wire_vram);
uint64_t vxfec_wire_recover_submit(vxfec_wire_ring *wring, vxfec_vk_buf *wire_vram, const uint64_t *rx_block_bits,
                                   vxfec_wire_stats *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* VXFEC_INTERLEAVE_H */
