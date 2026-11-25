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
#define VXFEC_IBV_TX_PER_QP 512u
#define VXFEC_IBV_BATCH_MAX 64u
#define VXFEC_IBV_POLL_MAX 128u
#define VXFEC_IBV_CTRL_MAX_BYTES 96u
#define VXFEC_IBV_WIRE_RX_KIB_MASK 0x7fu
#define VXFEC_IBV_WIRE_CTRL_RX 0x80u
#define VXFEC_IBV_CTRL_SLOT_FLAG 0x80000000u

typedef struct vxfec_ibv_ctx vxfec_ibv_ctx_t;
typedef struct vxfec_ibv_mr vxfec_ibv_mr_t;
typedef struct vxfec_ibv_ep vxfec_ibv_ep_t;

typedef struct {
    const void *src_addr;
    uint64_t remote_addr;
    uint32_t byte_len;
    uint32_t lkey;
    uint32_t rkey;
    uint32_t imm_host_order;
    bool force_signal;
} vxfec_ibv_rdma_wr_t;

typedef struct {
    const void *hdr;
    const void *payload;
    uint32_t hdr_len;
    uint32_t pay_len;
    uint32_t pay_lkey;
    bool force_signal;
} vxfec_ibv_send_wr_desc_t;

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
    return (slot_id >> 16u) & 0x7fffu;
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
bool vxfec_ibv_mr_read_to_host(vxfec_ibv_ctx_t *ctx, const vxfec_ibv_mr_t *src_mr, size_t src_off, void *dst, size_t bytes);
bool vxfec_ibv_mr_write_from_host(vxfec_ibv_ctx_t *ctx, vxfec_ibv_mr_t *dst_mr, size_t dst_off, const void *src, size_t bytes);

/* multi-QP Unreliable Connection (IBV_QPT_UC) endpoint for ECMP spraying + zero-bounce RDMA */
vxfec_ibv_ep_t *vxfec_ibv_ep_create_ex(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                       bool enable_rx, vxfec_ibv_wire_info_t *out_local_info);
vxfec_ibv_ep_t *vxfec_ibv_ep_create_opts(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                         bool enable_rx, bool header_only_tx, vxfec_ibv_wire_info_t *out_local_info);
vxfec_ibv_ep_t *vxfec_ibv_ep_create_duplex(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                           bool enable_rx, bool header_only_tx, bool ctrl_duplex,
                                           vxfec_ibv_wire_info_t *out_local_info);
bool vxfec_ibv_ep_connect(vxfec_ibv_ep_t *ep, const vxfec_ibv_wire_info_t *peer_info, uint8_t traffic_class);
bool vxfec_ibv_ep_destroy(vxfec_ibv_ep_t *ep);
bool vxfec_ibv_ep_has_error(const vxfec_ibv_ep_t *ep);
bool vxfec_ibv_ep_tx_is_hdr_only(const vxfec_ibv_ep_t *ep);
uint32_t vxfec_ibv_ep_path_mtu(const vxfec_ibv_ep_t *ep);
uint32_t vxfec_ibv_ep_num_qps(const vxfec_ibv_ep_t *ep);
uint64_t vxfec_ibv_ep_max_msg_bytes(const vxfec_ibv_ep_t *ep);
uint32_t vxfec_ibv_ep_peer_max_recv(const vxfec_ibv_ep_t *ep);
uint32_t vxfec_ibv_ep_tx_stage_cap(const vxfec_ibv_ep_t *ep);
bool vxfec_ibv_ep_peer_has_ctrl_rx(const vxfec_ibv_ep_t *ep);

/* 2-SGE send (wire header + registered payload/VRAM buffer) or zero-bounce RDMA WRITE WITH IMM (single or doorbell-batched) */
bool vxfec_ibv_ep_send_pkt(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *hdr, uint32_t hdr_len, const void *payload,
                           uint32_t pay_len, uint32_t pay_lkey);
uint32_t vxfec_ibv_ep_send_pkt_batch(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const vxfec_ibv_send_wr_desc_t *pkts, uint32_t count);
bool vxfec_ibv_ep_send_ctrl_inline(vxfec_ibv_ep_t *ep, const void *buf, uint32_t len);
bool vxfec_ibv_ep_rdma_write_imm(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *src_addr, uint32_t byte_len, uint32_t lkey,
                                 uint64_t remote_addr, uint32_t rkey, uint32_t imm_host_order);
uint32_t vxfec_ibv_ep_rdma_write_imm_batch(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const vxfec_ibv_rdma_wr_t *wrs, uint32_t count);

/* drain RX completions and re-post consumed RX ring slots (single or per-QP doorbell-batched) */
uint32_t vxfec_ibv_ep_poll_rx(vxfec_ibv_ep_t *ep, vxfec_ibv_rx_wc_t *out_wcs, uint32_t max_wcs);
bool vxfec_ibv_ep_repost_rx(vxfec_ibv_ep_t *ep, uint32_t slot_id);
bool vxfec_ibv_ep_repost_rx_batch(vxfec_ibv_ep_t *ep, const uint32_t *slot_ids, uint32_t count);
void vxfec_ibv_ep_drain_tx(vxfec_ibv_ep_t *ep);
void vxfec_ibv_ep_record_qp_tail(const vxfec_ibv_ep_t *ep, uint32_t qp_idx, uint64_t target_seq[VXFEC_IBV_MAX_QPS]);
bool vxfec_ibv_ep_tx_done(vxfec_ibv_ep_t *ep, const uint64_t target_seq[VXFEC_IBV_MAX_QPS]);

#ifdef __cplusplus
}
#endif

#endif /* VXFEC_IBV_TRANSPORT_H */
