#ifndef VXFEC_NCCL_NET_INTERNAL_H
#define VXFEC_NCCL_NET_INTERNAL_H

#include "nccl_net_vxfec.h"
#include "../../transport/harq.h"
#include "../../transport/ibv_transport.h"
#include "vxfec_fec.h"
#include "vxfec_rq.h"
#include "vxfec_rs_cauchy.h"
#include "vxfec_vk.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <limits.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#ifndef VXFEC_ENABLE_SIM
#define VXFEC_ENABLE_SIM 1
#endif

#ifndef IPTOS_ECN_MASK
#define IPTOS_ECN_MASK 0x03u
#endif
#ifndef IPTOS_ECN_ECT0
#define IPTOS_ECN_ECT0 0x02u
#endif
#ifndef IPTOS_ECN_CE
#define IPTOS_ECN_CE 0x03u
#endif

#define VXFEC_PORT_WEIGHT_MAX 16
#define VXFEC_CTRL_CMSG_BYTES 32u

#define VXFEC_NET_MAGIC 0x56584638u /* "VXF8" */
#define VXFEC_MAX_DEVS 16u
#define VXFEC_MAX_VK_DEVS 8u
#define VXFEC_MAX_PORTS 32u
#define VXFEC_DEF_PORTS 16u
#define VXFEC_DEF_SYM_T 1280u
#define VXFEC_MAX_SYM_T 8192u
#define VXFEC_SACK_MAX 64u
#define VXFEC_DONE_RING 4096u
#define VXFEC_BATCH_PKTS 512u
#define VXFEC_UDP_TX_BATCH 128u
#define VXFEC_RX_BATCH 64u
#define VXFEC_DELAY_RING 4096u
#define VXFEC_WIN_ACK_PKTS 16u
#define VXFEC_SLICE_WIN_ACK_PKTS 32u
#define VXFEC_TAG_INLINE_CAP 64u
#define VXFEC_MAX_LISTEN_BACKLOG 256u
#define VXFEC_RQ_GPU_MIN_DISCRETE_BYTES (512u * 1024u)
#define VXFEC_RQ_GPU_MIN_OTHER_BYTES (4u * 1024u * 1024u)
#define VXFEC_RQ_GPU_MAX_JOBS 32u
#define VXFEC_VK_ASYNC_POOL_SLOTS 8u
#define VXFEC_STAGE_POOL_SLOTS 8u
#define VXFEC_HELLO_RETRY_NS 1000000ULL
#define VXFEC_HELLO_GF2 0x01u
#define VXFEC_HELLO_IBV 0x02u
#define VXFEC_HELLO_SLICE 0x04u
#ifdef NANORQ_HAS_GF2_MODE
#define VXFEC_LOCAL_FEATURES (VXFEC_HELLO_GF2 | VXFEC_HELLO_SLICE)
#else
#define VXFEC_LOCAL_FEATURES VXFEC_HELLO_SLICE
#endif
#define VXFEC_CONN_TIMEOUT_NS (30ULL * 1000000000ULL)

/* sub-block generation slicing (K <= 128 Cauchy RS8 windows, ir_slack = 0) */
#define VXFEC_DEF_SLICE_MIN_BYTES (384u * 1024u)
#define VXFEC_RS8_WIN_K 128u
#define VXFEC_GEN_PIPE_DEPTH 24u
#define VXFEC_TX_WIRE_SLOTS 64u
#define VXFEC_TX_HASH_SLOTS 128u
#define VXFEC_RX_WIRE_SLOTS 128u
#define VXFEC_RX_HASH_SLOTS 256u
#define VXFEC_ABORT_RING 64u
#define VXFEC_LISTEN_TOMBSTONES 256u

enum {
    PKT_HELLO = 1,
    PKT_HELLO_ACK = 2,
    PKT_DATA = 3,
    PKT_TAIL = 4,
    PKT_SACK = 5,
    PKT_HARQ_IR = 6,
    PKT_DONE = 7,
    PKT_GBN_NACK = 8,
    PKT_ERR = 9,
    PKT_WIN_ACK = 10,
};

enum {
    WIRE_FLAG_RQ_GF256 = 0x00u,
    WIRE_FLAG_RQ_GF2 = 0x01u,
    WIRE_FLAG_RS_CAUCHY8 = 0x02u,
    WIRE_FLAG_NOFEC_GBN = 0x04u,
    WIRE_FLAG_SACK_ONLY = 0x08u,
    WIRE_CODEC_MASK = 0x0fu,
    WIRE_FLAG_GEN_EXT = 0x40u,
    WIRE_FLAG_NEED_ACK = 0x80u,
};

typedef enum {
    VXFEC_CODEC_OVR_AUTO = 0,
    VXFEC_CODEC_OVR_GBN = 1,
    VXFEC_CODEC_OVR_SACK = 2,
    VXFEC_CODEC_OVR_RS8 = 3,
    VXFEC_CODEC_OVR_RQ_GF256 = 4,
    VXFEC_CODEC_OVR_RQ_GF2 = 5,
} vxfec_codec_ovr_t;

typedef struct {
    uint32_t magic;
    uint32_t cookie;
    uint8_t pkt_type;
    uint8_t flags;
    uint16_t sym_len;
    uint32_t msg_seq;
    uint32_t tag_seq;
    uint32_t tag;
    uint32_t total_bytes;
    uint32_t K;
    uint16_t sym_T;
    uint16_t srtt_us;
    uint32_t esi;
} vxfec_wire_hdr_t;

typedef struct {
    uint32_t msg_total_bytes;
    uint16_t gen_idx;
    uint16_t num_gens;
} vxfec_gen_ext_t;

typedef struct {
    vxfec_wire_hdr_t base;
    vxfec_gen_ext_t ext;
} vxfec_wire_hdr_ext_t;

_Static_assert(sizeof(vxfec_wire_hdr_t) == 40u, "vxfec_wire_hdr_t must remain 40 bytes");
_Static_assert(sizeof(vxfec_wire_hdr_ext_t) == 48u, "vxfec_wire_hdr_ext_t must be 48 bytes");

typedef struct {
    uint32_t magic;
    uint32_t cookie;
    uint8_t pkt_type;
    uint8_t flags;
    uint16_t sack_cnt;
    uint32_t msg_seq;
    uint32_t tag;
    uint32_t K;
    uint32_t rx_pkts;
    uint32_t ecn_ce_pkts;
    uint32_t ecn_port_mask;
    uint32_t esi;
    uint32_t ir_count;
    uint32_t dwell_us;
} vxfec_ctrl_pkt_t;

typedef struct {
    uint32_t magic;
    uint32_t ip_be;
    uint16_t ctrl_port_be;
    uint16_t num_ports;
    uint32_t cookie;
    uint8_t dev_idx;
    uint8_t is_ibv;
    uint16_t pad_h;
    uint32_t reserved_h;
} vxfec_net_handle_t;

_Static_assert(sizeof(vxfec_net_handle_t) <= NCCL_NET_HANDLE_MAXSIZE, "vxfec_net_handle_t exceeds NCCL_NET_HANDLE_MAXSIZE");

typedef struct {
    uint32_t magic;
    uint8_t pkt_type;
    uint8_t has_ibv;
    uint16_t num_ports;
    uint32_t cookie;
    uint16_t tx_ctrl_port_be;
    uint8_t features;
    uint8_t pad0;
    uint64_t conn_id;
    uint16_t tx_ports_be[VXFEC_MAX_PORTS];
    vxfec_ibv_wire_info_t ibv_info;
} vxfec_hello_payload_t;

typedef struct {
    uint32_t magic;
    uint8_t pkt_type;
    uint8_t has_ibv;
    uint16_t num_ports;
    uint32_t cookie;
    uint16_t rx_ctrl_port_be;
    uint8_t features;
    uint8_t pad0;
    uint64_t conn_id;
    uint16_t rx_ports_be[VXFEC_MAX_PORTS];
    vxfec_ibv_wire_info_t ibv_info;
} vxfec_hello_ack_t;

#define VXFEC_HELLO_BASE_BYTES offsetof(vxfec_hello_payload_t, ibv_info)
#define VXFEC_HELLO_ACK_BASE_BYTES offsetof(vxfec_hello_ack_t, ibv_info)

_Static_assert(offsetof(vxfec_hello_payload_t, has_ibv) == 5u, "vxfec_hello_payload_t has_ibv offset mismatch");
_Static_assert(VXFEC_HELLO_BASE_BYTES == 88u, "vxfec_hello_payload_t UDP base size must remain 88 bytes");
_Static_assert(VXFEC_HELLO_ACK_BASE_BYTES == 88u, "vxfec_hello_ack_t UDP base size must remain 88 bytes");

/* refcounted RDMA context wrapper so vxf_init re-entry never frees an active PD/context */
typedef struct {
    atomic_uint refcnt;
    vxfec_ibv_ctx_t *ctx;
} vxfec_ibv_dev_ref_t;

typedef struct {
    atomic_uint refcnt;
    void *base_ptr;
    size_t size;
    uint64_t offset;
    int ptr_type;
    vxfec_vk_ctx *vk_ctx;
    vxfec_vk_buf *vk_buf;
    vxfec_ibv_mr_t *ibv_mrs[VXFEC_MAX_DEVS];
    vxfec_ibv_dev_ref_t *ibv_refs[VXFEC_MAX_DEVS];
} vxfec_mr_t;

typedef struct {
    char name[128];
    char ifname[32];
    char pci_path[PATH_MAX];
    uint32_t ip_be;
    uint32_t mtu;
    uint64_t speed_mbps;
    uint64_t guid;
    bool is_ibv;
    uint8_t ibv_port;
    uint32_t ibv_idx;
    vxfec_ibv_dev_ref_t *ibv_ref;
} vxfec_nic_dev_t;

typedef struct {
    uint64_t deliver_ns;
    uint32_t port_idx;
    uint16_t len;
    bool is_ecn_ce;
    uint8_t data[sizeof(vxfec_wire_hdr_ext_t) + VXFEC_MAX_SYM_T];
} vxfec_delay_data_t;

typedef struct {
    uint64_t deliver_ns;
    uint16_t len;
    uint8_t data[sizeof(vxfec_ctrl_pkt_t) + VXFEC_SACK_MAX * sizeof(uint32_t)];
} vxfec_delay_ctrl_t;

typedef struct {
    bool used;
    uint32_t tag;
    uint32_t next_seq;
} vxfec_tag_slot_t;

typedef struct {
    vxfec_tag_slot_t *slots;
    uint32_t cap;
    uint32_t count;
    vxfec_tag_slot_t inline_slots[VXFEC_TAG_INLINE_CAP];
} vxfec_tag_map_t;

typedef struct vxfec_rx_req vxfec_rx_req_t;
typedef struct vxfec_recv_comm vxfec_recv_comm_t;
typedef struct vxfec_send_comm vxfec_send_comm_t;
typedef struct vxfec_tx_user_req vxfec_tx_user_req_t;

typedef enum { VXFEC_REQ_SEND = 1, VXFEC_REQ_RECV = 2, VXFEC_REQ_FLUSH = 3 } vxfec_req_kind_t;

typedef struct vxfec_accept_node {
    atomic_uint refcnt;
    atomic_bool rc_closed;
    atomic_bool hs_confirmed;
    uint32_t peer_ip_be;
    uint16_t peer_port_be;
    size_t ack_len;
    vxfec_hello_payload_t hello;
    vxfec_hello_ack_t ack;
    vxfec_recv_comm_t *rc;
    struct vxfec_accept_node *next;
} vxfec_accept_node_t;

typedef struct vxfec_listen_comm {
    int ctrl_fd;
    int wake_fds[2];
    int dev_idx;
    uint32_t num_ports;
    uint32_t ip_be;
    uint16_t ctrl_port_be;
    uint32_t cookie;
    uint64_t retired_ns;
    pthread_mutex_t mtx;
    bool mtx_ready;
    pthread_t bg_thr;
    bool bg_started;
    atomic_bool stop;
    atomic_bool closed_by_user;
    uint32_t backlog_cnt;
    vxfec_accept_node_t *head;
    vxfec_accept_node_t *tail;
    struct vxfec_listen_comm *retired_next;
} vxfec_listen_comm_t;

typedef enum {
    VXFEC_GPU_JOB_RQ = 1,
    VXFEC_GPU_JOB_RS8 = 2,
} vxfec_gpu_job_kind_t;

typedef struct vxfec_gpu_job {
    vxfec_gpu_job_kind_t kind;
    int vk_idx;
    int pool_idx;
    vxfec_rq_sched *rq_sched;
    vxfec_vk_buf *in_buf;
    vxfec_vk_buf *out_buf;
    bool rs8_is_decode;
    uint32_t rs8_K;
    uint32_t rs8_M;
    uint32_t rs8_count;
    uint8_t *rs8_erased_k;
    uint8_t *rs8_parity_m;
    uint64_t ticket;
    struct vxfec_gpu_job *next;
} vxfec_gpu_job_t;

typedef struct {
    vxfec_wire_hdr_ext_t hdrs[VXFEC_UDP_TX_BATCH];
    struct iovec iovs[VXFEC_UDP_TX_BATCH][2];
    struct mmsghdr msgs[VXFEC_UDP_TX_BATCH];
} vxfec_tx_udp_batch_t;

typedef struct {
    uint32_t pkt_stride;
    uint8_t *pkt_bufs;
    _Alignas(struct cmsghdr) uint8_t ctrl_bufs[VXFEC_RX_BATCH][VXFEC_CTRL_CMSG_BYTES];
    struct iovec iovs[VXFEC_RX_BATCH];
    struct mmsghdr msgs[VXFEC_RX_BATCH];
} vxfec_rx_udp_batch_t;

struct vxfec_tx_user_req {
    vxfec_req_kind_t kind;
    bool in_use;
    bool failed;
    bool completion_ready;
    bool ibv_zc_pending;
    bool done;
    vxfec_send_comm_t *comm;
    uint32_t tag;
    uint32_t tag_seq;
    uint32_t msg_total_bytes;
    uint16_t sym_T;
    uint16_t num_gens;
    uint16_t next_gen_to_spawn;
    uint16_t gens_completed;
    uint16_t active_gens;
    ncclResult_t status;
    vxfec_mr_t *mr;
    const uint8_t *src_host;
    const uint8_t *src_ibv_ptr;
    uint32_t src_ibv_lkey;
    bool src_ibv_dmabuf;
    uint8_t *bounce_buf;
    uint64_t ibv_target_seq[VXFEC_IBV_MAX_QPS];
};

typedef struct vxfec_tx_req_impl {
    bool in_use;
    bool wire_active;
    bool done;
    uint8_t slot_idx;
    vxfec_send_comm_t *comm;
    vxfec_tx_user_req_t *parent;
    uint32_t msg_seq;
    uint32_t tag_seq;
    uint32_t tag;
    uint32_t msg_total_bytes;
    uint32_t total_bytes;
    uint32_t K;
    uint16_t sym_T;
    uint16_t gen_idx;
    uint16_t num_gens;
    vxfec_codec_t codec;
    uint8_t wire_flags;
    ncclResult_t status;

    vxfec_mr_t *mr;
    const uint8_t *src_host;
    const uint8_t *src_ibv_ptr;
    uint32_t src_ibv_lkey;
    bool src_ibv_dmabuf;
    uint8_t *bounce_buf;
    uint8_t *rs_sys_padded;

    bool rq_ready;
    bool enc_borrowed;
    nanorq_core *enc_ptr;
    nanorq_core *enc_owned;
    uint8_t *enc_prep;
    uint8_t *enc_work;
    schedule enc_sched;
    uint32_t enc_rows;
    uint32_t enc_stride;
    uint8_t *enc_d_cpu;
    vxfec_gpu_job_t *tx_gpu;
    uint32_t tx_gpu_esi;
    uint32_t tx_gpu_count;
    uint32_t tx_gpu_offset;
    uint32_t tx_gpu_stride;
    uint8_t tx_gpu_flags;
    bool tx_gpu_proactive;
    bool tx_gpu_ready;
    bool tx_gpu_failed;

    uint32_t tx_sys_cursor;
    uint32_t proactive_target;
    uint32_t tx_proactive_cursor;
    uint32_t next_rep_esi;
    uint32_t win_rx_pkts;
    uint32_t win_init_cleared;
    uint32_t win_ecn_ce;
    uint32_t win_ecn_port_mask;
    uint64_t win_acked_bytes;
    uint64_t queued_bytes;
    uint64_t first_sys_tx_ns;
    uint64_t ack_probe_tx_ns;
    uint64_t last_send_ns;
    uint64_t burst_end_tx_ns;
    uint64_t last_tail_ns;
    uint64_t last_sack_tx_ns;
    uint64_t last_ir_tx_ns;
    uint32_t last_ir_ctrl_esi;
    uint32_t last_ir_ctrl_cnt;
    uint8_t last_ir_ctrl_codec;
    bool initial_tail_sent;
    bool retx_or_ir_sent;

    uint32_t *sack_queue;
    uint32_t sack_head;
    uint32_t sack_cnt;
    uint32_t ir_start_esi;
    uint32_t ir_rem;
    uint32_t gbn_cursor;
    uint32_t gbn_rem;
    bool need_recovery_tail;
} vxfec_tx_slot_t;

typedef struct {
    uint32_t msg_seq;
    uint32_t tag_seq;
    uint32_t tag;
    uint64_t last_tx_ns;
    uint32_t tx_tries;
    bool acked;
} vxfec_abort_entry_t;

struct vxfec_send_comm {
    pthread_mutex_t mtx;
    bool mtx_ready;
    bool tx_bg_registered;
    vxfec_send_comm_t *tx_bg_next;
    uint32_t unacked_aborts;
    int dev_idx;
    uint32_t bind_ip_be;
    int ctrl_fd;
    int fds[VXFEC_MAX_PORTS];
    uint32_t num_ports;
    uint16_t path_sym_limit;
    uint8_t features;
    uint32_t slice_min_bytes;
    vxfec_ibv_dev_ref_t *ibv_ref;
    vxfec_ibv_ep_t *ibv_ep;
    bool ibv_failed;
    struct sockaddr_in peer_ctrl;
    struct sockaddr_in peer_rx_ctrl;
    struct sockaddr_in peer_data[VXFEC_MAX_PORTS];
    bool ctrl_connected;
    bool data_connected;
    int32_t swrr_cur[VXFEC_MAX_PORTS];
    int32_t port_weights[VXFEC_MAX_PORTS];
    bool has_nonuniform_weights;
    uint32_t cookie;
    vxfec_hello_payload_t hello;
    uint64_t last_hello_ns;
    uint32_t empty_hello_polls;
    bool connected;
    uint32_t next_msg_seq;
    uint32_t spawn_rr_cursor;
    vxfec_tag_map_t tag_map;
    uint32_t cfg_epoch;
    double loss_ewma;
    uint64_t inflight_bytes;
    harq_pacer_t pacer;
    uint64_t active_tx_mask;
    int16_t tx_seq_hash[VXFEC_TX_HASH_SLOTS];
    vxfec_tx_user_req_t user_tx_pool[NCCL_NET_MAX_REQUESTS];
    vxfec_tx_slot_t tx_pool[VXFEC_TX_WIRE_SLOTS];
    vxfec_abort_entry_t tx_aborts[VXFEC_ABORT_RING];
    uint32_t tx_abort_cursor;

    bool rq_cache_valid;
    uint32_t rq_cache_K;
    vxfec_codec_t rq_cache_codec;
    nanorq_core rq_cache_enc;
    uint8_t *rq_cache_prep;
    uint8_t *rq_cache_work;
    schedule rq_cache_sched;
    uint32_t rq_cache_rows;

#if VXFEC_ENABLE_SIM
    vxfec_delay_ctrl_t *ctrl_delay_q;
    uint32_t ctrl_delay_head;
    uint32_t ctrl_delay_cnt;
#endif

    vxfec_tx_udp_batch_t *tx_udp;
    _Alignas(struct cmsghdr) uint8_t tx_cmsg_ce[CMSG_SPACE(sizeof(uint8_t))];
    size_t tx_cmsg_ce_len;
};

typedef struct {
    uint32_t msg_seq;
    uint32_t tag_seq;
    uint32_t tag;
    uint32_t msg_total_bytes;
    uint32_t total_bytes;
    size_t gen_offset;
    uint32_t K;
    uint16_t sym_T;
    uint16_t gen_idx;
    uint16_t num_gens;
    uint8_t slot_idx;
    int8_t stage_pool_idx;
    uint8_t flags;
    bool active;
    bool tail_seen;
    bool stage_failed;
    bool fec_repaired;

    vxfec_rx_req_t *bound_req;

    uint32_t rx_sys_cnt;
    uint32_t initial_missing_sys;
    uint32_t sack_cursor;
    uint8_t *sys_bitmap;
    uint8_t *sys_stage;

    uint32_t rx_rep_cnt;
    uint32_t rep_cap;
    uint32_t max_rep_esi;
    uint32_t *rep_esis;
    uint8_t *rep_data;

    vxfec_gpu_job_t *rq_gpu_job;
    uint32_t *rq_gpu_missing;
    uint32_t rq_gpu_count;
    uint32_t rq_gpu_stride;
    bool rq_gpu_disabled;

    uint32_t rx_pkt_cnt;
    uint32_t last_acked_pkts;
    uint32_t max_init_esi_plus1;
    uint32_t rx_ecn_ce_cnt;
    uint32_t ecn_port_mask;

    uint32_t ir_slack;
    uint64_t probe_rx_ns;
    uint64_t tail_seen_ns;
    uint64_t last_pkt_ns;
    uint64_t last_fb_ns;

    vxfec_gpu_job_t *rs8_gpu_job;
    uint32_t vk_missing;
    uint32_t vk_stride;
    uint8_t *vk_compact;
} vxfec_rx_slot_t;

struct vxfec_rx_req {
    vxfec_req_kind_t kind;
    bool in_use;
    bool any_gen_used_fec;
    bool done;
    vxfec_recv_comm_t *comm;
    uint32_t expected_tag_seq;
    uint32_t tag;
    void *dst_ptr;
    int max_size;
    int actual_size;
    uint32_t msg_total_bytes;
    uint16_t expected_num_gens;
    uint16_t gens_completed;
    uint16_t bound_sym_T;
    ncclResult_t status;
    vxfec_mr_t *mr;
    uint64_t inline_gen_bits;
    uint64_t *heap_gen_bits;
};

typedef struct vxfec_done_range {
    uint64_t first;
    uint64_t last;
    struct vxfec_done_range *next;
} vxfec_done_range_t;

typedef struct {
    uint8_t *buf;
    size_t cap;
    bool in_use;
} vxfec_stage_slot_t;

struct vxfec_recv_comm {
    pthread_mutex_t mtx;
    bool mtx_ready;
    bool bg_registered;
    vxfec_recv_comm_t *bg_next;
    uint8_t features;
    int dev_idx;
    int ctrl_fd;
    uint16_t ctrl_port_be;
    bool ctrl_connected;
    struct sockaddr_in peer_ctrl;
    int fds[VXFEC_MAX_PORTS];
    uint16_t rx_ports_be[VXFEC_MAX_PORTS];
    uint32_t num_ports;
    vxfec_ibv_dev_ref_t *ibv_ref;
    vxfec_ibv_ep_t *ibv_ep;
    bool ibv_failed;
    struct sockaddr_in peer_data[VXFEC_MAX_PORTS];
    int32_t swrr_cur[VXFEC_MAX_PORTS];
    uint32_t cookie;
    uint64_t peer_srtt_ns;
    vxfec_accept_node_t *accept_node;
    vxfec_tag_map_t tag_map;
    vxfec_abort_entry_t peer_aborts[VXFEC_ABORT_RING];
    uint32_t peer_abort_cursor;

    uint64_t active_slot_mask[VXFEC_RX_WIRE_SLOTS / 64u];
    int16_t msg_seq_hash[VXFEC_RX_HASH_SLOTS];
    vxfec_rx_slot_t slots[VXFEC_RX_WIRE_SLOTS];
    vxfec_rx_req_t rx_pool[NCCL_NET_MAX_REQUESTS];

#if VXFEC_ENABLE_SIM
    vxfec_delay_data_t *delay_q;
    uint32_t delay_head;
    uint32_t delay_cnt;
    vxfec_delay_ctrl_t *header_delay_q;
    uint32_t header_delay_head;
    uint32_t header_delay_cnt;
#endif

    vxfec_stage_slot_t stage_pool[VXFEC_STAGE_POOL_SLOTS];

    uint64_t done_seqs[VXFEC_DONE_RING];
    uint32_t done_lost[VXFEC_DONE_RING];
    uint32_t done_k[VXFEC_DONE_RING];
    uint32_t done_ecn_ce[VXFEC_DONE_RING];
    uint32_t done_rx_pkts[VXFEC_DONE_RING];
    uint32_t done_ecn_mask[VXFEC_DONE_RING];
    bool done_err[VXFEC_DONE_RING];
    uint64_t completed_watermark_seq;
    uint64_t highest_done_seq;
    vxfec_done_range_t *pending_done;
    vxfec_done_range_t *done_errors;

    vxfec_rx_udp_batch_t *rx_udp;
};

/* shared globals across plugin compilation units (hidden symbol visibility) */
extern ncclDebugLogger_t vxf_logger;
extern vxfec_nic_dev_t vxf_devs[VXFEC_MAX_DEVS];
extern uint32_t vxf_num_devs;
extern atomic_uint vxf_leaked_ibv_eps;

extern atomic_uint vxf_cfg_ports;
extern atomic_uint vxf_cfg_mtu_sym;
extern atomic_uint vxf_cfg_bind_ip_be;
extern atomic_uint vxf_cfg_busy_poll_us;
extern atomic_uint vxf_cfg_slice_min_bytes;
extern atomic_uint vxf_granted_rcvbuf;
extern atomic_int vxf_env_codec_ovr;
extern atomic_int vxf_codec_override;

extern atomic_uint vxf_retired_lc_cnt;
extern atomic_bool vxf_retired_lc_reap_needed;

extern _Atomic uint64_t vxf_st_tx_messages;
extern _Atomic uint64_t vxf_st_tx_sys_pkts;
extern _Atomic uint64_t vxf_st_tx_proactive_rep_pkts;
extern _Atomic uint64_t vxf_st_tx_sack_retx_pkts;
extern _Atomic uint64_t vxf_st_tx_harq_ir_pkts;
extern _Atomic uint64_t vxf_st_rx_messages;
extern _Atomic uint64_t vxf_st_rx_fast_completions;
extern _Atomic uint64_t vxf_st_rx_fec_decodes;
extern _Atomic uint64_t vxf_st_rx_rs8_decodes;
extern _Atomic uint64_t vxf_st_rx_rq_decodes;
extern _Atomic uint64_t vxf_st_rx_ooo_pkts;
extern _Atomic uint64_t vxf_st_vk_offload_calls;
extern _Atomic uint64_t vxf_st_sim_dropped_pkts;
extern _Atomic uint64_t vxf_st_rx_ecn_ce_pkts;
extern _Atomic uint64_t vxf_st_tx_cnp_events;
extern _Atomic uint64_t vxf_st_tx_paced_slices;
extern _Atomic uint64_t vxf_st_tx_fec_clamped_pkts;
extern _Atomic uint64_t vxf_st_ibv_direct_pkts;
extern _Atomic uint64_t vxf_st_vk_async_polls;

#define VXF_LOG_WARN(...)                                                                                                          \
    do {                                                                                                                           \
        if (vxf_logger)                                                                                                            \
            vxf_logger(NCCL_LOG_WARN, ~0UL, __FILE__, __LINE__, __VA_ARGS__);                                                      \
    } while (0)

static inline bool seq_before(uint32_t a, uint32_t b)
{
    return (int32_t)(a - b) < 0;
}

static inline uint32_t next_non_zero_seq(uint32_t *seq)
{
    uint32_t s = (*seq)++;
    if (s == 0u)
        s = (*seq)++;
    if (*seq == 0u)
        *seq = 1u;
    return s;
}

static inline uint64_t mono_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* simulator APIs (nccl_net_sim.c) */
vxfec_codec_ovr_t vxf_parse_codec_mode(const char *mode);
void vxf_sim_init_from_env(void);

#if VXFEC_ENABLE_SIM
extern atomic_bool vxf_sim_active;
extern atomic_uint vxf_sim_cfg_epoch;
extern atomic_bool vxf_sim_reorder;
extern _Atomic uint64_t vxf_sim_max_rate_mbps;
extern _Atomic uint64_t vxf_sim_link_delay_ns;
extern atomic_uint vxf_sim_drop_hellos;
extern atomic_uint vxf_sim_drop_acks;
extern atomic_uint vxf_sim_drop_dones;
extern atomic_uint vxf_sim_drop_aborts;
extern atomic_uint vxf_sim_fail_gpu_rx_writes;

uint32_t vxf_fast_rand32(void);
bool vxf_sim_should_drop_data_slow(void);
bool vxf_sim_should_drop_counter_slow(atomic_uint *counter);
bool vxf_sim_should_mark_ecn_slow(uint32_t port_idx);

static inline bool vxf_sim_should_drop_data(void)
{
    if (!atomic_load_explicit(&vxf_sim_active, memory_order_relaxed))
        return false;
    return vxf_sim_should_drop_data_slow();
}

static inline bool vxf_sim_should_drop_hellos(void)
{
    if (!atomic_load_explicit(&vxf_sim_active, memory_order_relaxed))
        return false;
    return vxf_sim_should_drop_counter_slow(&vxf_sim_drop_hellos);
}

static inline bool vxf_sim_should_drop_acks(void)
{
    if (!atomic_load_explicit(&vxf_sim_active, memory_order_relaxed))
        return false;
    return vxf_sim_should_drop_counter_slow(&vxf_sim_drop_acks);
}

static inline bool vxf_sim_should_drop_dones(void)
{
    if (!atomic_load_explicit(&vxf_sim_active, memory_order_relaxed))
        return false;
    return vxf_sim_should_drop_counter_slow(&vxf_sim_drop_dones);
}

static inline bool vxf_sim_should_drop_aborts(void)
{
    if (!atomic_load_explicit(&vxf_sim_active, memory_order_relaxed))
        return false;
    return vxf_sim_should_drop_counter_slow(&vxf_sim_drop_aborts);
}

static inline bool vxf_sim_should_fail_gpu_rx(void)
{
    if (!atomic_load_explicit(&vxf_sim_active, memory_order_relaxed))
        return false;
    return vxf_sim_should_drop_counter_slow(&vxf_sim_fail_gpu_rx_writes);
}

static inline bool vxf_sim_should_mark_ecn(uint32_t port_idx)
{
    if (!atomic_load_explicit(&vxf_sim_active, memory_order_relaxed))
        return false;
    return vxf_sim_should_mark_ecn_slow(port_idx);
}

static inline bool vxf_sim_reorder_enabled(void)
{
    return atomic_load_explicit(&vxf_sim_active, memory_order_relaxed) &&
           atomic_load_explicit(&vxf_sim_reorder, memory_order_relaxed);
}

static inline uint64_t vxf_sim_delay_ns(void)
{
    if (!atomic_load_explicit(&vxf_sim_active, memory_order_relaxed))
        return 0u;
    return atomic_load_explicit(&vxf_sim_link_delay_ns, memory_order_relaxed);
}

static inline uint64_t vxf_sim_rate_cap_mbps(void)
{
    if (!atomic_load_explicit(&vxf_sim_active, memory_order_relaxed))
        return 0u;
    return atomic_load_explicit(&vxf_sim_max_rate_mbps, memory_order_relaxed);
}

static inline uint32_t vxf_sim_epoch(void)
{
    return atomic_load_explicit(&vxf_sim_cfg_epoch, memory_order_acquire);
}
#else
static inline uint32_t vxf_fast_rand32(void)
{
    return 0u;
}
static inline bool vxf_sim_should_drop_data(void)
{
    return false;
}
static inline bool vxf_sim_should_drop_hellos(void)
{
    return false;
}
static inline bool vxf_sim_should_drop_acks(void)
{
    return false;
}
static inline bool vxf_sim_should_drop_dones(void)
{
    return false;
}
static inline bool vxf_sim_should_drop_aborts(void)
{
    return false;
}
static inline bool vxf_sim_should_fail_gpu_rx(void)
{
    return false;
}
static inline bool vxf_sim_should_mark_ecn(uint32_t port_idx)
{
    (void)port_idx;
    return false;
}
static inline bool vxf_sim_reorder_enabled(void)
{
    return false;
}
static inline uint64_t vxf_sim_delay_ns(void)
{
    return 0u;
}
static inline uint64_t vxf_sim_rate_cap_mbps(void)
{
    return 0u;
}
static inline uint32_t vxf_sim_epoch(void)
{
    return 1u;
}
#endif

/* device, MR, and GPU offload APIs (nccl_net_dev.c) */
void vxf_ibv_ref_retain(vxfec_ibv_dev_ref_t *ref);
void vxf_ibv_ref_release(vxfec_ibv_dev_ref_t *ref);
void vxf_mr_retain(vxfec_mr_t *mr);
void vxf_mr_release(vxfec_mr_t *mr);
bool vxf_mr_vk_read(vxfec_mr_t *mr, size_t byte_offset, void *dst, size_t bytes);
bool vxf_mr_vk_write(vxfec_mr_t *mr, size_t byte_offset, const void *src, size_t bytes);
uint32_t vxf_get_dev_ip_be(int dev_idx);
uint32_t vxf_parse_slice_min_env(uint32_t fallback);

vxfec_gpu_job_t *vxf_rq_gpu_job_new(vxfec_vk_ctx *prefer_vk, nanorq_core *rq, const schedule *ops, size_t input_bytes,
                                    size_t output_bytes);
bool vxf_rq_gpu_job_submit(vxfec_gpu_job_t *job, nanorq_core *rq, uint32_t stride, const uint32_t *esis, uint32_t count);
vxfec_gpu_job_t *vxf_rs8_gpu_job_new(vxfec_vk_ctx *prefer_vk, bool is_decode, uint32_t K, uint32_t rs_m, const uint8_t *erased_k,
                                     const uint8_t *parity_m, uint32_t count, uint32_t sym_T);
bool vxf_rs8_gpu_job_submit(vxfec_gpu_job_t *job, uint32_t sym_T);
int vxf_gpu_job_poll(vxfec_gpu_job_t *job);
void vxf_gpu_job_free(vxfec_gpu_job_t *job);

ncclResult_t vxf_init(ncclDebugLogger_t logFunction);
ncclResult_t vxf_devices(int *ndev);
ncclResult_t vxf_get_properties(int dev, ncclNetProperties_v8_t *props);
ncclResult_t vxf_reg_mr(void *comm, void *data, size_t size, int type, void **mhandle);
ncclResult_t vxf_reg_mr_dmabuf(void *comm, void *data, size_t size, int type, uint64_t offset, int fd, void **mhandle);
ncclResult_t vxf_dereg_mr(void *comm, void *mhandle);
ncclResult_t vxf_get_device_mr(void *comm, void *mhandle, void **dptr_mhandle);
void vxf_dev_shutdown(void);

/* connection & background epoll APIs (nccl_net_conn.c) */
uint64_t vxf_next_unique_u64(void);
uint32_t vxf_next_unique_u32(void);
void vxf_accept_node_release(vxfec_accept_node_t *node);
void vxf_rx_bg_wake(void);
#if VXFEC_ENABLE_SIM
void vxf_rx_bg_schedule_delay(void);
uint64_t vxf_rx_deliver_due_headers(vxfec_recv_comm_t *comm, uint64_t now_ns);
#endif
void vxf_rx_bg_register(vxfec_recv_comm_t *rc);
void vxf_rx_bg_unregister(vxfec_recv_comm_t *rc);
void vxf_tx_bg_register(vxfec_send_comm_t *sc);
void vxf_tx_bg_unregister(vxfec_send_comm_t *sc);
void vxf_reap_retired_listeners(bool force_all);
void vxf_conn_shutdown(void);

ncclResult_t vxf_listen(int dev, void *opaque_handle, void **listenComm);
ncclResult_t vxf_connect(int dev, void *opaque_handle, void **sendComm, ncclNetDeviceHandle_v8_t **sendDevComm);
ncclResult_t vxf_accept(void *listenComm, void **recvComm, ncclNetDeviceHandle_v8_t **recvDevComm);
ncclResult_t vxf_close_send(void *sendComm);
ncclResult_t vxf_close_recv(void *recvComm);
ncclResult_t vxf_close_listen(void *listenComm);

/* protocol & data-plane APIs (nccl_net_proto.c) */
void vxf_tag_map_free(vxfec_tag_map_t *map);
uint32_t vxf_next_tag_seq(vxfec_tag_map_t *map, uint32_t tag);
uint32_t vxf_peek_tag_seq(const vxfec_tag_map_t *map, uint32_t tag);
uint16_t vxf_tx_path_sym_limit(const vxfec_send_comm_t *comm);
void vxf_init_send_comm_pacer(vxfec_send_comm_t *sc, uint64_t now_ns);
void vxf_free_send_comm_rq_cache(vxfec_send_comm_t *sc);
void vxf_free_tx_req_resources(vxfec_tx_slot_t *req);
void vxf_reset_rx_slot(vxfec_recv_comm_t *comm, vxfec_rx_slot_t *s);
void vxf_rx_done_range_discard(vxfec_done_range_t **head, uint64_t through);
void vxf_rx_drain_ctrl_wire(vxfec_recv_comm_t *comm, uint64_t now_ns);
void vxf_rx_bg_peek_data_ports(vxfec_recv_comm_t *comm);
void vxf_progress_send_comm(vxfec_send_comm_t *comm);
void vxf_progress_recv_comm(vxfec_recv_comm_t *comm);

ncclResult_t vxf_isend(void *sendComm, void *data, int size, int tag, void *mhandle, void **request);
ncclResult_t vxf_irecv(void *recvComm, int n, void **data, int *sizes, int *tags, void **mhandles, void **request);
ncclResult_t vxf_iflush(void *recvComm, int n, void **data, int *sizes, void **mhandles, void **request);
ncclResult_t vxf_test(void *request, int *done, int *sizes);
ncclResult_t vxf_irecv_consumed(void *recvComm, int n, void *request);

#endif /* VXFEC_NCCL_NET_INTERNAL_H */
