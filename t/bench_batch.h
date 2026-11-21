#ifndef VXFEC_BENCH_BATCH_H
#define VXFEC_BENCH_BATCH_H

#include <stddef.h>
#include <stdint.h>

static inline uint32_t vxfec_bench_blocks(size_t target_bytes, size_t max_range, size_t matrix_block_bytes,
                                          size_t output_block_bytes, uint32_t max_blocks)
{
    if (!matrix_block_bytes || !output_block_bytes || !max_blocks || matrix_block_bytes > max_range ||
        output_block_bytes > max_range)
        return 0;

    size_t blocks = target_bytes / matrix_block_bytes;
    if (blocks == 0)
        blocks = 1;
    if (blocks > max_blocks)
        blocks = max_blocks;
    if (blocks > max_range / matrix_block_bytes)
        blocks = max_range / matrix_block_bytes;
    if (blocks > max_range / output_block_bytes)
        blocks = max_range / output_block_bytes;
    return (uint32_t)blocks;
}

#endif /* VXFEC_BENCH_BATCH_H */
