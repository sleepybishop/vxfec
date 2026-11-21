#ifndef VXFEC_NCCL_COLLNET_V8_H
#define VXFEC_NCCL_COLLNET_V8_H

#include "nccl_net_v8.h"

/*
 * self-contained C11 ABI definitions for NCCL / RCCL collNetPlugin_v8,
 * retained for the planned multicast collective adapter.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ncclInt8 = 0,
    ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclHalf = 6,
    ncclFloat32 = 7,
    ncclFloat = 7,
    ncclFloat64 = 8,
    ncclDouble = 8,
    ncclBfloat16 = 9,
    ncclNumTypes = 10
} ncclDataType_t;

typedef enum { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4, ncclNumOps = 5 } ncclRedOp_t;

typedef struct {
    void *sendbuff;
    void *recvbuff;
    void *sendMhandle;
    void *recvMhandle;
} ncclCollNetSharedRes_v8_t;

typedef struct {
    const char *name;
    ncclResult_t (*init)(ncclDebugLogger_t logFunction);
    ncclResult_t (*devices)(int *ndev);
    ncclResult_t (*getProperties)(int dev, ncclNetProperties_v8_t *props);
    ncclResult_t (*listen)(int dev, void *collNetHandle, void **collNetListenComm);
    ncclResult_t (*connect)(void *handles[], int nranks, int rank, void *collNetListenComm, void **collNetComm);
    ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int *supported);
    ncclResult_t (*regMr)(void *collNetComm, void *data, size_t size, int type, void **mhandle);
    ncclResult_t (*regMrDmaBuf)(void *collNetComm, void *data, size_t size, int type, uint64_t offset, int fd, void **mhandle);
    ncclResult_t (*deregMr)(void *collNetComm, void *mhandle);
    ncclResult_t (*iallreduce)(void *collNetComm, void *sendData, void *recvData, int count, ncclDataType_t dataType,
                               ncclRedOp_t redOp, void *sendMhandle, void *recvMhandle, void **request);
    ncclResult_t (*iallgather)(void *collNetComm, void *sendData, int nRecvParts, ncclCollNetSharedRes_v8_t *parts,
                               size_t bytesPerPart, size_t sendOffset, size_t recvOffset, void *sendMhandle, void **request);
    ncclResult_t (*ireducescatter)(void *collNetComm, int nSendParts, ncclCollNetSharedRes_v8_t *parts, void *recvData,
                                   size_t bytesPerPart, size_t sendOffset, size_t recvOffset, ncclDataType_t dataType,
                                   ncclRedOp_t redOp, void *recvMhandle, void **request);
    ncclResult_t (*iflush)(void *collNetComm, void *data, int size, void *mhandle, void **request);
    ncclResult_t (*test)(void *request, int *done, int *sizes);
    ncclResult_t (*closeColl)(void *collNetComm);
    ncclResult_t (*closeListen)(void *collNetListenComm);
} ncclCollNet_v8_t;

#ifdef __cplusplus
}
#endif

#endif /* VXFEC_NCCL_COLLNET_V8_H */
