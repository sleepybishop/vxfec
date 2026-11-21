#ifndef VXFEC_IBV_TRANSPORT_H
#define VXFEC_IBV_TRANSPORT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VXFEC_IBV_MAX_DEVS 16u
#define VXFEC_IBV_MAX_QPS 16u
#define VXFEC_IBV_TX_PER_QP 256u

typedef struct vxfec_ibv_ctx vxfec_ibv_ctx_t;
typedef struct vxfec_ibv_mr vxfec_ibv_mr_t;
typedef struct vxfec_ibv_ep vxfec_ibv_ep_t;

typedef struct {
    char dev_name[64];
    char pci_path[256];
    uint8_t port_num;
    uint8_t gid_index;
    uint8_t gid[16];
    uint16_t lid;
    uint32_t ipv4_be;
    uint32_t active_mtu;
    uint64_t speed_mbps;
    uint64_t guid;
    bool is_roce;
} vxfec_ibv_dev_info_t;

typedef struct {
    uint8_t gid[16];
    uint16_t lid;
    uint8_t port_num;
    uint8_t gid_index;
    uint8_t mtu_enum;
    uint8_t num_qps;
    uint8_t is_roce;
    uint8_t reserved;
    uint32_t qpn[VXFEC_IBV_MAX_QPS];
    uint32_t psn[VXFEC_IBV_MAX_QPS];
} vxfec_ibv_wire_info_t;

typedef struct {
    const uint8_t *data;
    uint32_t byte_len;
    uint32_t imm_data;
    uint32_t slot_id;
    bool has_imm;
    bool is_rdma_imm;
} vxfec_ibv_rx_wc_t;

static inline uint32_t vxfec_ibv_slot_qp(uint32_t slot_id)
{
    return (slot_id >> 16u) & 0xffffu;
}

/* probe dynamic libibverbs.so.1 availability and enumerate active RoCEv2/IB ports */
bool vxfec_ibv_available(void);
bool vxfec_ibv_has_dmabuf(void);
uint32_t vxfec_ibv_enumerate(vxfec_ibv_dev_info_t *out_devs, uint32_t max_devs);

/* open RDMA device context + Protection Domain */
vxfec_ibv_ctx_t *vxfec_ibv_ctx_open(uint32_t dev_idx);
void vxfec_ibv_ctx_close(vxfec_ibv_ctx_t *ctx);
const vxfec_ibv_dev_info_t *vxfec_ibv_ctx_info(const vxfec_ibv_ctx_t *ctx);

/* register host memory or GPU DMA-BUF (ibv_reg_dmabuf_mr) with local + remote write */
vxfec_ibv_mr_t *vxfec_ibv_mr_reg_host(vxfec_ibv_ctx_t *ctx, void *addr, size_t bytes, bool remote_write);
vxfec_ibv_mr_t *vxfec_ibv_mr_reg_dmabuf(vxfec_ibv_ctx_t *ctx, uint64_t offset, size_t bytes, uint64_t iova, int fd,
                                        bool remote_write);
void vxfec_ibv_mr_dereg(vxfec_ibv_mr_t *mr);
uint32_t vxfec_ibv_mr_lkey(const vxfec_ibv_mr_t *mr);
uint32_t vxfec_ibv_mr_rkey(const vxfec_ibv_mr_t *mr);
uint64_t vxfec_ibv_mr_iova(const vxfec_ibv_mr_t *mr);
bool vxfec_ibv_mr_is_dmabuf(const vxfec_ibv_mr_t *mr);

/* multi-QP Unreliable Connection (IBV_QPT_UC) endpoint for ECMP spraying + zero-bounce RDMA */
vxfec_ibv_ep_t *vxfec_ibv_ep_create(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes,
                                    vxfec_ibv_wire_info_t *out_local_info);
vxfec_ibv_ep_t *vxfec_ibv_ep_create_ex(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                       bool enable_rx, vxfec_ibv_wire_info_t *out_local_info);
vxfec_ibv_ep_t *vxfec_ibv_ep_create_opts(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                         bool enable_rx, bool header_only_tx, vxfec_ibv_wire_info_t *out_local_info);
bool vxfec_ibv_ep_connect(vxfec_ibv_ep_t *ep, const vxfec_ibv_wire_info_t *peer_info, uint8_t traffic_class);
bool vxfec_ibv_ep_destroy(vxfec_ibv_ep_t *ep);
bool vxfec_ibv_ep_has_error(const vxfec_ibv_ep_t *ep);
bool vxfec_ibv_ep_tx_is_hdr_only(const vxfec_ibv_ep_t *ep);
uint32_t vxfec_ibv_ep_path_mtu(const vxfec_ibv_ep_t *ep);

/* 2-SGE send (24B wire header + registered payload/VRAM buffer) or zero-bounce RDMA WRITE WITH IMM */
bool vxfec_ibv_ep_send_pkt(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *hdr, uint32_t hdr_len, const void *payload,
                           uint32_t pay_len, uint32_t pay_lkey);
bool vxfec_ibv_ep_send_pkt_ex(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *hdr, uint32_t hdr_len, const void *payload,
                              uint32_t pay_len, uint32_t pay_lkey, bool force_signal);
bool vxfec_ibv_ep_rdma_write_imm(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *src_addr, uint32_t byte_len, uint32_t lkey,
                                 uint64_t remote_addr, uint32_t rkey, uint32_t imm_host_order);

/* drain RX completions and re-post consumed RX ring slots */
uint32_t vxfec_ibv_ep_poll_rx(vxfec_ibv_ep_t *ep, vxfec_ibv_rx_wc_t *out_wcs, uint32_t max_wcs);
bool vxfec_ibv_ep_repost_rx(vxfec_ibv_ep_t *ep, uint32_t slot_id);
void vxfec_ibv_ep_drain_tx(vxfec_ibv_ep_t *ep);
void vxfec_ibv_ep_record_qp_tail(const vxfec_ibv_ep_t *ep, uint32_t qp_idx, uint64_t target_seq[VXFEC_IBV_MAX_QPS]);
bool vxfec_ibv_ep_tx_done(vxfec_ibv_ep_t *ep, const uint64_t target_seq[VXFEC_IBV_MAX_QPS]);

#ifdef __cplusplus
}
#endif

#endif /* VXFEC_IBV_TRANSPORT_H */
