#define _GNU_SOURCE
#include "nccl_net_internal.h"

extern int __cxa_atexit(void (*func)(void *), void *arg, void *dso_handle);
extern void *__dso_handle __attribute__((__visibility__("hidden"), __weak__));

static atomic_bool g_plugin_shut_down = false;
static atomic_bool g_atexit_gpurt_reg = false;
static atomic_bool g_atexit_vk_reg = false;

static void vxf_plugin_shutdown_once(void *unused)
{
    (void)unused;
    if (atomic_exchange_explicit(&g_plugin_shut_down, true, memory_order_acq_rel))
        return;
    vxf_conn_shutdown();
    vxf_dev_shutdown();
}

/* run before HIP/CUDA/Vulkan C++ static destructors at exit(), and via __cxa_finalize on dlclose() */
void vxf_register_gpu_atexit(bool is_gpurt)
{
    atomic_bool *flag = is_gpurt ? &g_atexit_gpurt_reg : &g_atexit_vk_reg;
    if (!atomic_exchange_explicit(flag, true, memory_order_acq_rel))
        (void)__cxa_atexit(vxf_plugin_shutdown_once, NULL, &__dso_handle);
}

__attribute__((destructor)) static void vxf_plugin_fini(void)
{
    vxf_plugin_shutdown_once(NULL);
}

__attribute__((visibility("default"))) const ncclNet_v8_t ncclNetPlugin_v8 = {
    .name = "vxfec-ecmp-harq",
    .init = vxf_init,
    .devices = vxf_devices,
    .getProperties = vxf_get_properties,
    .listen = vxf_listen,
    .connect = vxf_connect,
    .accept = vxf_accept,
    .regMr = vxf_reg_mr,
    .regMrDmaBuf = vxf_reg_mr_dmabuf,
    .deregMr = vxf_dereg_mr,
    .isend = vxf_isend,
    .irecv = vxf_irecv,
    .iflush = vxf_iflush,
    .test = vxf_test,
    .closeSend = vxf_close_send,
    .closeRecv = vxf_close_recv,
    .closeListen = vxf_close_listen,
    .getDeviceMr = vxf_get_device_mr,
    .irecvConsumed = vxf_irecv_consumed,
};
