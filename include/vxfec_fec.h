#ifndef VXFEC_FEC_H
#define VXFEC_FEC_H

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "nanorq_core.h"

/* regime-aware codec selection across Cauchy RS8 MDS and RaptorQ */
typedef enum {
    VXFEC_CODEC_AUTO = 0,
    VXFEC_CODEC_RS_CAUCHY8 = 1, /* GF(256) Cauchy MDS: K + M <= 256, ir_slack = 0 */
    VXFEC_CODEC_RQ_GF2 = 2,     /* H=0 pure-GF(2) RQ fountain (NANORQ_HAS_GF2_MODE) */
    VXFEC_CODEC_RQ_GF256 = 3,   /* RFC 6330 RQ with GF(256) HDPC, ir_slack = 2 */
} vxfec_codec_t;

static inline vxfec_codec_t vxfec_select_codec(uint32_t K, uint32_t max_repairs, bool rateless)
{
    if (K == 0u)
        return VXFEC_CODEC_RQ_GF256;
    if (!rateless && K <= 200u && max_repairs > 0u && max_repairs <= 256u - K)
        return VXFEC_CODEC_RS_CAUCHY8;
#ifdef NANORQ_HAS_GF2_MODE
    if (K >= 48u)
        return VXFEC_CODEC_RQ_GF2;
#else
    (void)rateless;
#endif
    return VXFEC_CODEC_RQ_GF256;
}

static inline uint32_t vxfec_codec_ir_slack_ex(vxfec_codec_t codec, uint32_t K, uint32_t max_repairs, bool rateless)
{
    if (codec == VXFEC_CODEC_AUTO)
        codec = vxfec_select_codec(K, max_repairs, rateless);
    switch (codec) {
    case VXFEC_CODEC_RS_CAUCHY8:
        return 0u;
    case VXFEC_CODEC_RQ_GF2:
        return 8u;
    case VXFEC_CODEC_RQ_GF256:
    default:
        return 2u;
    }
}

static inline uint32_t vxfec_codec_ir_slack(vxfec_codec_t codec, uint32_t K)
{
    return vxfec_codec_ir_slack_ex(codec, K, 56u, false);
}

#ifdef __cplusplus
}
#endif

#endif /* VXFEC_FEC_H */
