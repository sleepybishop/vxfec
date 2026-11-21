#ifndef NANORQ_NCCL_NET_V8_H
#define NANORQ_NCCL_NET_V8_H

#include <stddef.h>
#include <stdint.h>

/*
 * Self-contained C11 ABI definitions compatible with the NCCL / RCCL External
 * Network Plugin v8 specification (`ncclNetPlugin_v8`).
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8
} ncclResult_t;

typedef enum {
    NCCL_LOG_NONE = 0,
    NCCL_LOG_VERSION = 1,
    NCCL_LOG_WARN = 2,
    NCCL_LOG_INFO = 3,
    NCCL_LOG_ABORT = 4,
    NCCL_LOG_TRACE = 5
} ncclDebugLogLevel;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4
#define NCCL_PTR_NVLS 0x8

#define NCCL_NET_HANDLE_MAXSIZE 128
#define NCCL_NET_MAX_REQUESTS 32

typedef enum { NCCL_NET_DEVICE_HOST = 0, NCCL_NET_DEVICE_UNPACK = 1 } ncclNetDeviceType;

typedef struct {
    ncclNetDeviceType netDeviceType;
    int netDeviceVersion;
    void *handle;
    size_t size;
    int needsProxyProgress;
} ncclNetDeviceHandle_v8_t;

typedef struct {
    char *name;
    char *pciPath;
    uint64_t guid;
    int ptrSupport;
    int regIsGlobal;
    int speed;
    int port;
    float latency;
    int maxComms;
    int maxRecvs;
    ncclNetDeviceType netDeviceType;
    int netDeviceVersion;
} ncclNetProperties_v8_t;

typedef struct {
    const char *name;
    ncclResult_t (*init)(ncclDebugLogger_t logFunction);
    ncclResult_t (*devices)(int *ndev);
    ncclResult_t (*getProperties)(int dev, ncclNetProperties_v8_t *props);
    ncclResult_t (*listen)(int dev, void *handle, void **listenComm);
    ncclResult_t (*connect)(int dev, void *handle, void **sendComm, ncclNetDeviceHandle_v8_t **sendDevComm);
    ncclResult_t (*accept)(void *listenComm, void **recvComm, ncclNetDeviceHandle_v8_t **recvDevComm);
    ncclResult_t (*regMr)(void *comm, void *data, size_t size, int type, void **mhandle);
    ncclResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type, uint64_t offset, int fd, void **mhandle);
    ncclResult_t (*deregMr)(void *comm, void *mhandle);
    ncclResult_t (*isend)(void *sendComm, void *data, int size, int tag, void *mhandle, void **request);
    ncclResult_t (*irecv)(void *recvComm, int n, void **data, int *sizes, int *tags, void **mhandles, void **request);
    ncclResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes, void **mhandles, void **request);
    ncclResult_t (*test)(void *request, int *done, int *sizes);
    ncclResult_t (*closeSend)(void *sendComm);
    ncclResult_t (*closeRecv)(void *recvComm);
    ncclResult_t (*closeListen)(void *listenComm);
    ncclResult_t (*getDeviceMr)(void *comm, void *mhandle, void **dptr_mhandle);
    ncclResult_t (*irecvConsumed)(void *recvComm, int n, void *request);
} ncclNet_v8_t;

#ifdef __cplusplus
}
#endif

#endif
