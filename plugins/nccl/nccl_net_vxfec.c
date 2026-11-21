#define _GNU_SOURCE
#include "nccl_net_internal.h"

__attribute__((destructor)) static void vxf_plugin_fini(void)
{
    vxf_conn_shutdown();
    vxf_dev_shutdown();
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
