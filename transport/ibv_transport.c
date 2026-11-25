#define _GNU_SOURCE
#include "ibv_transport.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <time.h>
#include <unistd.h>

#ifdef VXFEC_HAVE_IBVERBS
#include <infiniband/verbs.h>

static inline int query_port_via_vctx(struct ibv_context *ctx, uint8_t port_num, struct ibv_port_attr *attr)
{
    struct verbs_context *vctx = verbs_get_ctx_op(ctx, query_port);
    if (!vctx || !vctx->query_port)
        return -ENOSYS;
    memset(attr, 0, sizeof(*attr));
    return vctx->query_port(ctx, port_num, attr, sizeof(*attr));
}

#undef ibv_get_device_list
#undef ibv_query_device
#undef ibv_query_port
#undef ibv_reg_mr
#undef ibv_reg_mr_iova

#define VXFEC_IBV_RX_PER_QP_DEF 512u
#define VXFEC_IBV_TX_HDR_CAP 96u
#define VXFEC_IBV_SIGNAL_MOD 32u
#define VXFEC_IBV_DEF_MAX_MSG_BYTES (1ULL << 30)
#define VXFEC_IBV_MAX_STAGE_STRIDE 8320u
#define VXFEC_IBV_CTRL_RX_DEPTH 128u
#define VXFEC_IBV_CTRL_RX_STRIDE 128u
#define VXFEC_IBV_CTRL_TX_DEPTH 128u

typedef struct {
    void *dl;
    bool ready;
    struct ibv_device **(*get_device_list)(int *num_devices);
    void (*free_device_list)(struct ibv_device **list);
    const char *(*get_device_name)(struct ibv_device *device);
    uint64_t (*get_device_guid)(struct ibv_device *device);
    struct ibv_context *(*open_device)(struct ibv_device *device);
    int (*close_device)(struct ibv_context *context);
    int (*query_device)(struct ibv_context *context, struct ibv_device_attr *device_attr);
    int (*query_gid)(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *context);
    int (*dealloc_pd)(struct ibv_pd *pd);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *pd, void *addr, size_t length, int access);
    struct ibv_mr *(*reg_mr_iova2)(struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, unsigned int access);
    struct ibv_mr *(*reg_dmabuf_mr)(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access);
    int (*dereg_mr)(struct ibv_mr *mr);
    struct ibv_cq *(*create_cq)(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel,
                                int comp_vector);
    int (*destroy_cq)(struct ibv_cq *cq);
    struct ibv_qp *(*create_qp)(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
    int (*modify_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
    int (*destroy_qp)(struct ibv_qp *qp);
} vxfec_ibv_syms_t;

static vxfec_ibv_syms_t g_ibv;
static pthread_once_t g_ibv_once = PTHREAD_ONCE_INIT;

static void *resolve_ibv_sym(void *dl, bool use_self, const char *name)
{
    if (use_self) {
        char mname[96];
        snprintf(mname, sizeof(mname), "vxfec_test_%s", name);
        return dlsym(RTLD_DEFAULT, mname);
    }
    void *p = dlvsym(dl, name, "IBVERBS_1.1");
    if (!p)
        p = dlsym(dl, name);
    return p;
}

static void init_ibv_once(void)
{
    memset(&g_ibv, 0, sizeof(g_ibv));
    const char *ibv_lib = getenv("VXFEC_IBV_LIB_PATH");
    bool use_self = (ibv_lib && strcmp(ibv_lib, "self") == 0);
    if (!use_self) {
        if (ibv_lib && *ibv_lib) {
            g_ibv.dl = dlopen(ibv_lib, RTLD_NOW | RTLD_LOCAL);
        } else {
            g_ibv.dl = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
            if (!g_ibv.dl)
                g_ibv.dl = dlopen("libibverbs.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (!g_ibv.dl)
            return;
    }

    g_ibv.get_device_list = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_get_device_list");
    g_ibv.free_device_list = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_free_device_list");
    g_ibv.get_device_name = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_get_device_name");
    g_ibv.get_device_guid = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_get_device_guid");
    g_ibv.open_device = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_open_device");
    g_ibv.close_device = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_close_device");
    g_ibv.query_device = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_query_device");
    g_ibv.query_gid = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_query_gid");
    g_ibv.alloc_pd = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_alloc_pd");
    g_ibv.dealloc_pd = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_dealloc_pd");
    g_ibv.reg_mr = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_reg_mr");
    g_ibv.reg_mr_iova2 = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_reg_mr_iova2");
    g_ibv.reg_dmabuf_mr = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_reg_dmabuf_mr");
    g_ibv.dereg_mr = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_dereg_mr");
    g_ibv.create_cq = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_create_cq");
    g_ibv.destroy_cq = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_destroy_cq");
    g_ibv.create_qp = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_create_qp");
    g_ibv.modify_qp = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_modify_qp");
    g_ibv.destroy_qp = resolve_ibv_sym(g_ibv.dl, use_self, "ibv_destroy_qp");

    if (g_ibv.get_device_list && g_ibv.free_device_list && g_ibv.get_device_name && g_ibv.open_device && g_ibv.close_device &&
        g_ibv.query_gid && g_ibv.alloc_pd && g_ibv.dealloc_pd && (g_ibv.reg_mr_iova2 || g_ibv.reg_mr) && g_ibv.dereg_mr &&
        g_ibv.create_cq && g_ibv.destroy_cq && g_ibv.create_qp && g_ibv.modify_qp && g_ibv.destroy_qp) {
        g_ibv.ready = true;
    }
}

struct vxfec_ibv_ctx {
    struct ibv_context *ibv;
    struct ibv_pd *pd;
    vxfec_ibv_dev_info_t info;
    uint64_t max_msg_bytes;
    uint32_t max_qp_wr;
    uint8_t mtu_enum;
    pthread_mutex_t dma_mtx;
    bool dma_mtx_ready;
    struct ibv_cq *dma_cq;
    struct ibv_qp *dma_qp;
    struct ibv_mr *dma_stage_mr;
    uint8_t *dma_stage_buf;
    size_t dma_stage_cap; /* Nonzero only while the loopback QP is ready for copies. */
    struct vxfec_ibv_ctx *next_closing;
};

struct vxfec_ibv_mr {
    struct ibv_mr *mr;
    uint64_t iova;
    size_t length;
    bool is_dmabuf;
    bool remote_write;
};

typedef struct {
    _Alignas(64) uint64_t sq_posted;
    uint64_t sq_last_sig;
    uint64_t sq_done;
    uint32_t tx_unsignaled;
    bool qp_err;
} vxfec_ibv_qp_tx_t;

struct vxfec_ibv_ep {
    vxfec_ibv_ctx_t *ctx;
    struct ibv_cq *tx_cq;
    struct ibv_cq *rx_cq;
    struct ibv_qp *qps[VXFEC_IBV_MAX_QPS];
    uint32_t local_psn[VXFEC_IBV_MAX_QPS];
    uint32_t num_qps;
    uint32_t tx_per_qp;
    uint32_t rx_per_qp;
    uint32_t ctrl_tx_depth;
    uint32_t ctrl_rx_depth;
    uint32_t rx_stride;
    uint32_t tx_stride;
    uint32_t max_pkt_bytes;
    uint32_t max_inline_data;
    uint32_t path_mtu_bytes;
    uint32_t peer_max_recv_bytes;
    uint64_t max_msg_bytes;
    bool enable_tx;
    bool enable_rx;
    bool rx_imm_only;
    bool ctrl_tx_enabled;
    bool ctrl_rx_enabled;
    bool peer_has_ctrl_rx;
    bool tx_hdr_only;
    bool tx_error;
    bool destroying;
    bool tx_cq_mtx_ready;
    pthread_mutex_t tx_cq_mtx;

    uint8_t *rx_slab;
    uint8_t *ctrl_tx_buf;
    size_t rx_slab_bytes;
    struct ibv_mr *rx_mr;

    uint8_t *tx_slab;
    uint8_t *ctrl_rx_buf;
    size_t tx_slab_bytes;
    struct ibv_mr *tx_mr;
    vxfec_ibv_qp_tx_t qp_tx[VXFEC_IBV_MAX_QPS];
};

static uint32_t mtu_enum_to_bytes(enum ibv_mtu m)
{
    switch (m) {
    case IBV_MTU_256:
        return 256u;
    case IBV_MTU_512:
        return 512u;
    case IBV_MTU_1024:
        return 1024u;
    case IBV_MTU_2048:
        return 2048u;
    case IBV_MTU_4096:
        return 4096u;
    default:
        return 1024u;
    }
}

static uint64_t port_speed_mbps(const struct ibv_port_attr *pa)
{
    uint64_t lanes = 1u;
    switch (pa->active_width) {
    case 1:
        lanes = 1u;
        break;
    case 2:
        lanes = 4u;
        break;
    case 4:
        lanes = 8u;
        break;
    case 8:
        lanes = 12u;
        break;
    case 16:
        lanes = 2u;
        break;
    default:
        lanes = 4u;
        break;
    }
    uint64_t per_lane_mbps = 25000ULL;
#ifdef VXFEC_HAVE_IBV_SPEED_EX
    uint32_t spd = pa->active_speed_ex ? pa->active_speed_ex : (uint32_t)pa->active_speed;
#else
    uint32_t spd = (uint32_t)pa->active_speed;
#endif
    if (spd & 256u)
        per_lane_mbps = 200000ULL; /* XDR: 200G/lane */
    else if (spd & 128u)
        per_lane_mbps = 100000ULL; /* NDR: 100G/lane */
    else if (spd & 64u)
        per_lane_mbps = 50000ULL; /* HDR: 50G/lane */
    else if (spd & 32u)
        per_lane_mbps = 25000ULL; /* EDR: 25G/lane */
    else if (spd & 16u)
        per_lane_mbps = 14000ULL; /* FDR: 14G/lane */
    else if (spd & (4u | 8u))
        per_lane_mbps = 10000ULL; /* QDR / FDR10: 10G/lane */
    else if (spd & 2u)
        per_lane_mbps = 5000ULL; /* DDR: 5G/lane */
    else if (spd & 1u)
        per_lane_mbps = 2500ULL; /* SDR: 2.5G/lane */
    return lanes * per_lane_mbps;
}

static int read_sysfs_gid_type(const char *dname, uint8_t port_num, int idx)
{
    const char *root = getenv("VXFEC_IBV_SYSFS_ROOT");
    if (!root || !*root)
        root = "/sys/class/infiniband";
    char path[PATH_MAX];
    int n = snprintf(path, sizeof(path), "%s/%s/ports/%u/gid_attrs/types/%d", root, dname, (unsigned)port_num, idx);
    if (n <= 0 || (size_t)n >= sizeof(path))
        return 0;
    FILE *fp = fopen(path, "r");
    if (!fp)
        return 0;
    char buf[64] = {0};
    bool ok = (fgets(buf, sizeof(buf), fp) != NULL);
    fclose(fp);
    if (!ok)
        return 0;
    if (strstr(buf, "RoCE v2") != NULL)
        return 2;
    if (strstr(buf, "RoCE v1") != NULL)
        return 1;
    return 0;
}

static bool extract_v4_mapped_gid(const union ibv_gid *g, uint32_t *out_ip_be)
{
    for (int b = 0; b < 10; b++) {
        if (g->raw[b] != 0)
            return false;
    }
    if (g->raw[10] != 0xff || g->raw[11] != 0xff)
        return false;
    uint32_t ip_be = 0;
    memcpy(&ip_be, &g->raw[12], 4);
    if (ip_be == 0u)
        return false;
    *out_ip_be = ip_be;
    return true;
}

static void pick_best_gid(const char *dname, struct ibv_context *ctx, uint8_t port_num, int tbl_len, bool is_roce, uint8_t *out_idx,
                          uint8_t out_gid[16], uint32_t *out_ipv4_be)
{
    *out_idx = 0u;
    *out_ipv4_be = 0u;
    memset(out_gid, 0, 16);

    int limit = (tbl_len > 64) ? 64 : tbl_len;
    if (limit <= 0)
        limit = 4;

    const char *forced_env = getenv("VXFEC_IBV_GID_INDEX");
    if (!forced_env || !*forced_env)
        forced_env = getenv("NCCL_IB_GID_INDEX");
    if (forced_env && *forced_env) {
        char *end = NULL;
        errno = 0;
        long fidx = strtol(forced_env, &end, 10);
        if (errno == 0 && end != forced_env && *end == '\0' && fidx >= 0 && fidx < 256) {
            union ibv_gid fg;
            memset(&fg, 0, sizeof(fg));
            if (g_ibv.query_gid(ctx, port_num, (int)fidx, &fg) == 0) {
                for (int b = 0; b < 16; b++) {
                    if (fg.raw[b] != 0) {
                        *out_idx = (uint8_t)fidx;
                        memcpy(out_gid, fg.raw, 16);
                        (void)extract_v4_mapped_gid(&fg, out_ipv4_be);
                        return;
                    }
                }
            }
        }
    }

    int best_score = INT_MIN;
    for (int idx = 0; idx < limit; idx++) {
        union ibv_gid g;
        memset(&g, 0, sizeof(g));
        if (g_ibv.query_gid(ctx, port_num, idx, &g) != 0)
            continue;
        bool non_zero = false;
        for (int b = 0; b < 16; b++) {
            if (g.raw[b] != 0) {
                non_zero = true;
                break;
            }
        }
        if (!non_zero)
            continue;

        int score = 0;
        if (is_roce && dname) {
            int gtype = read_sysfs_gid_type(dname, port_num, idx);
            if (gtype == 2)
                score += 100; /* RoCE v2 */
            else if (gtype == 1)
                score -= 50; /* RoCE v1 */
        }

        uint32_t ip_be = 0;
        if (extract_v4_mapped_gid(&g, &ip_be)) {
            bool link_local_v4 = ((ntohl(ip_be) & 0xffff0000u) == 0xa9fe0000u);
            score += link_local_v4 ? 10 : 40;
        } else if (g.raw[0] == 0xfeu && (g.raw[1] & 0xc0u) == 0x80u) {
            score += 5; /* IPv6 link-local fe80:: */
        } else {
            score += 20; /* global IPv6 / IB GID */
        }

        if (score > best_score) {
            best_score = score;
            *out_idx = (uint8_t)idx;
            *out_ipv4_be = ip_be;
            memcpy(out_gid, g.raw, 16);
        }
    }
}

static bool ibv_dev_matches_filter(const char *filter, const char *dname, uint8_t port_num)
{
    if (!filter || !*filter)
        return true;
    size_t dlen = strlen(dname);
    const char *p = filter;
    while (*p) {
        while (*p && (isspace((unsigned char)*p) || *p == ','))
            p++;
        if (!*p)
            break;
        const char *tok = p;
        while (*p && *p != ',' && !isspace((unsigned char)*p))
            p++;
        size_t tlen = (size_t)(p - tok);
        const char *colon = memchr(tok, ':', tlen);
        size_t name_len = colon ? (size_t)(colon - tok) : tlen;
        if (name_len == dlen && memcmp(tok, dname, dlen) == 0) {
            if (!colon || port_num == 0u)
                return true;
            char *end = NULL;
            long want_port = strtol(colon + 1, &end, 10);
            if (end == tok + tlen && want_port == (long)port_num)
                return true;
        }
    }
    return false;
}

bool vxfec_ibv_available(void)
{
    pthread_once(&g_ibv_once, init_ibv_once);
    if (!g_ibv.ready)
        return false;
    vxfec_ibv_dev_info_t tmp;
    return vxfec_ibv_enumerate(&tmp, 1u) > 0u;
}

bool vxfec_ibv_has_dmabuf(void)
{
    pthread_once(&g_ibv_once, init_ibv_once);
    return g_ibv.ready && g_ibv.reg_dmabuf_mr != NULL;
}

typedef struct {
    struct ibv_device *device;
    const char *name;
} named_ibv_device_t;

static int compare_ibv_devices(const void *a, const void *b)
{
    const named_ibv_device_t *left = a;
    const named_ibv_device_t *right = b;
    return strcmp(left->name, right->name);
}

uint32_t vxfec_ibv_enumerate(vxfec_ibv_dev_info_t *out_devs, uint32_t max_devs)
{
    pthread_once(&g_ibv_once, init_ibv_once);
    if (!g_ibv.ready || !out_devs || max_devs == 0u)
        return 0u;

    int num_raw = 0;
    struct ibv_device **list = g_ibv.get_device_list(&num_raw);
    if (!list || num_raw <= 0) {
        if (list)
            g_ibv.free_device_list(list);
        return 0u;
    }

    const char *filter = getenv("VXFEC_IBV_DEVS");
    named_ibv_device_t *devices = calloc((size_t)num_raw, sizeof(*devices));
    if (!devices) {
        g_ibv.free_device_list(list);
        return 0u;
    }
    size_t named_count = 0u;
    for (int i = 0; i < num_raw; i++) {
        const char *name = g_ibv.get_device_name(list[i]);
        if (name && *name && ibv_dev_matches_filter(filter, name, 0u))
            devices[named_count++] = (named_ibv_device_t){list[i], name};
    }
    qsort(devices, named_count, sizeof(*devices), compare_ibv_devices);

    uint32_t count = 0u;
    for (size_t i = 0; i < named_count && count < max_devs; i++) {
        const char *dname = devices[i].name;
        struct ibv_context *ctx = g_ibv.open_device(devices[i].device);
        if (!ctx)
            continue;

        uint8_t max_port = 2u;
        struct ibv_device_attr dattr;
        memset(&dattr, 0, sizeof(dattr));
        if (g_ibv.query_device && g_ibv.query_device(ctx, &dattr) == 0 && dattr.phys_port_cnt > 0u)
            max_port = dattr.phys_port_cnt;

        for (unsigned p = 1u; p <= max_port && count < max_devs; p++) {
            if (!ibv_dev_matches_filter(filter, dname, p))
                continue;
            struct ibv_port_attr pa;
            if (query_port_via_vctx(ctx, p, &pa) != 0 || pa.state != IBV_PORT_ACTIVE)
                continue;

            vxfec_ibv_dev_info_t *info = &out_devs[count];
            memset(info, 0, sizeof(*info));
            snprintf(info->dev_name, sizeof(info->dev_name), "%s", dname);
            info->port_num = p;
            info->lid = pa.lid;
            info->active_mtu = mtu_enum_to_bytes(pa.active_mtu);
            info->speed_mbps = port_speed_mbps(&pa);
            info->guid = g_ibv.get_device_guid ? ((uint64_t)g_ibv.get_device_guid(devices[i].device) ^ ((uint64_t)(p - 1u) << 56u))
                                               : (0x5658494256000000ULL | count);
            info->is_roce = (pa.link_layer == IBV_LINK_LAYER_ETHERNET);

            pick_best_gid(dname, ctx, p, pa.gid_tbl_len, info->is_roce, &info->gid_index, info->gid, &info->ipv4_be);

            char sys_link[PATH_MAX];
            char resolved[PATH_MAX];
            int n = snprintf(sys_link, sizeof(sys_link), "/sys/class/infiniband/%s/device", dname);
            if (n > 0 && (size_t)n < sizeof(sys_link) && realpath(sys_link, resolved))
                snprintf(info->pci_path, sizeof(info->pci_path), "%.*s", (int)(sizeof(info->pci_path) - 1u), resolved);

            count++;
        }
        g_ibv.close_device(ctx);
    }

    free(devices);
    g_ibv.free_device_list(list);
    return count;
}

static bool teardown_loopback_dma_locked(vxfec_ibv_ctx_t *ctx)
{
    ctx->dma_stage_cap = 0u;
    if (ctx->dma_qp) {
        if (g_ibv.destroy_qp(ctx->dma_qp) != 0)
            return false;
        ctx->dma_qp = NULL;
    }
    if (ctx->dma_cq) {
        if (g_ibv.destroy_cq(ctx->dma_cq) != 0)
            return false;
        ctx->dma_cq = NULL;
    }
    if (ctx->dma_stage_mr) {
        if (g_ibv.dereg_mr(ctx->dma_stage_mr) != 0)
            return false;
        ctx->dma_stage_mr = NULL;
    }
    free(ctx->dma_stage_buf);
    ctx->dma_stage_buf = NULL;
    return true;
}

/* A closed context has no caller left to retry a busy provider resource. */
static pthread_mutex_t closing_mtx = PTHREAD_MUTEX_INITIALIZER;
static vxfec_ibv_ctx_t *closing_contexts;

static void reap_closing_contexts(vxfec_ibv_ctx_t *newly_closed)
{
    pthread_mutex_lock(&closing_mtx);
    if (newly_closed) {
        newly_closed->next_closing = closing_contexts;
        closing_contexts = newly_closed;
    }
    vxfec_ibv_ctx_t **link = &closing_contexts;
    while (*link) {
        vxfec_ibv_ctx_t *ctx = *link;
        if (!teardown_loopback_dma_locked(ctx) || g_ibv.dealloc_pd(ctx->pd) != 0) {
            link = &ctx->next_closing;
            continue;
        }
        g_ibv.close_device(ctx->ibv);
        if (ctx->dma_mtx_ready)
            pthread_mutex_destroy(&ctx->dma_mtx);
        *link = ctx->next_closing;
        free(ctx);
    }
    pthread_mutex_unlock(&closing_mtx);
}

vxfec_ibv_ctx_t *vxfec_ibv_ctx_open(uint32_t dev_idx)
{
    reap_closing_contexts(NULL);
    vxfec_ibv_dev_info_t infos[VXFEC_IBV_MAX_DEVS];
    uint32_t n = vxfec_ibv_enumerate(infos, VXFEC_IBV_MAX_DEVS);
    if (dev_idx >= n)
        return NULL;

    int num_raw = 0;
    struct ibv_device **list = g_ibv.get_device_list(&num_raw);
    if (!list)
        return NULL;

    struct ibv_context *opened = NULL;
    for (int i = 0; i < num_raw; i++) {
        const char *dname = g_ibv.get_device_name(list[i]);
        if (dname && strcmp(dname, infos[dev_idx].dev_name) == 0) {
            opened = g_ibv.open_device(list[i]);
            break;
        }
    }
    g_ibv.free_device_list(list);
    if (!opened)
        return NULL;

    struct ibv_port_attr pa;
    if (query_port_via_vctx(opened, infos[dev_idx].port_num, &pa) != 0) {
        g_ibv.close_device(opened);
        return NULL;
    }

    struct ibv_pd *pd = g_ibv.alloc_pd(opened);
    if (!pd) {
        g_ibv.close_device(opened);
        return NULL;
    }

    vxfec_ibv_ctx_t *ctx = (vxfec_ibv_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        g_ibv.dealloc_pd(pd);
        g_ibv.close_device(opened);
        return NULL;
    }
    ctx->ibv = opened;
    ctx->pd = pd;
    ctx->info = infos[dev_idx];
    ctx->mtu_enum = (uint8_t)pa.active_mtu;
    ctx->max_msg_bytes = (pa.max_msg_sz > 0u) ? (uint64_t)pa.max_msg_sz : VXFEC_IBV_DEF_MAX_MSG_BYTES;
    if (pthread_mutex_init(&ctx->dma_mtx, NULL) == 0)
        ctx->dma_mtx_ready = true;
    struct ibv_device_attr dattr;
    memset(&dattr, 0, sizeof(dattr));
    if (g_ibv.query_device && g_ibv.query_device(opened, &dattr) == 0) {
        if (dattr.max_qp_wr > 0)
            ctx->max_qp_wr = (uint32_t)dattr.max_qp_wr;
    }
    return ctx;
}

void vxfec_ibv_ctx_close(vxfec_ibv_ctx_t *ctx)
{
    if (ctx)
        reap_closing_contexts(ctx);
}

const vxfec_ibv_dev_info_t *vxfec_ibv_ctx_info(const vxfec_ibv_ctx_t *ctx)
{
    return ctx ? &ctx->info : NULL;
}

static struct ibv_mr *reg_host_raw(struct ibv_pd *pd, void *addr, size_t bytes, int access)
{
    if (g_ibv.reg_mr_iova2)
        return g_ibv.reg_mr_iova2(pd, addr, bytes, (uint64_t)(uintptr_t)addr, (unsigned int)access);
    return g_ibv.reg_mr(pd, addr, bytes, access);
}

vxfec_ibv_mr_t *vxfec_ibv_mr_reg_host(vxfec_ibv_ctx_t *ctx, void *addr, size_t bytes, bool remote_write)
{
    if (!ctx || !ctx->pd || !addr || bytes == 0)
        return NULL;
    int access = IBV_ACCESS_LOCAL_WRITE | (remote_write ? IBV_ACCESS_REMOTE_WRITE : 0);
    struct ibv_mr *raw = reg_host_raw(ctx->pd, addr, bytes, access);
    if (!raw)
        return NULL;
    vxfec_ibv_mr_t *mr = (vxfec_ibv_mr_t *)calloc(1, sizeof(*mr));
    if (!mr) {
        g_ibv.dereg_mr(raw);
        return NULL;
    }
    mr->mr = raw;
    mr->iova = (uint64_t)(uintptr_t)addr;
    mr->length = bytes;
    mr->is_dmabuf = false;
    mr->remote_write = remote_write;
    return mr;
}

vxfec_ibv_mr_t *vxfec_ibv_mr_reg_dmabuf(vxfec_ibv_ctx_t *ctx, uint64_t offset, size_t bytes, uint64_t iova, int fd,
                                        bool remote_write)
{
    if (!ctx || !ctx->pd || fd < 0 || bytes == 0 || !g_ibv.reg_dmabuf_mr)
        return NULL;
    int access = IBV_ACCESS_LOCAL_WRITE | (remote_write ? IBV_ACCESS_REMOTE_WRITE : 0);
    struct ibv_mr *raw = g_ibv.reg_dmabuf_mr(ctx->pd, offset, bytes, iova, fd, access);
    if (!raw)
        return NULL;
    vxfec_ibv_mr_t *mr = (vxfec_ibv_mr_t *)calloc(1, sizeof(*mr));
    if (!mr) {
        g_ibv.dereg_mr(raw);
        return NULL;
    }
    mr->mr = raw;
    mr->iova = iova;
    mr->length = bytes;
    mr->is_dmabuf = true;
    mr->remote_write = remote_write;
    return mr;
}

void vxfec_ibv_mr_dereg(vxfec_ibv_mr_t *mr)
{
    if (!mr)
        return;
    if (mr->mr)
        g_ibv.dereg_mr(mr->mr);
    free(mr);
}

uint32_t vxfec_ibv_mr_lkey(const vxfec_ibv_mr_t *mr)
{
    return (mr && mr->mr) ? mr->mr->lkey : 0u;
}

uint32_t vxfec_ibv_mr_rkey(const vxfec_ibv_mr_t *mr)
{
    return (mr && mr->mr) ? mr->mr->rkey : 0u;
}

uint64_t vxfec_ibv_mr_iova(const vxfec_ibv_mr_t *mr)
{
    return mr ? mr->iova : 0ULL;
}

bool vxfec_ibv_mr_is_dmabuf(const vxfec_ibv_mr_t *mr)
{
    return mr ? mr->is_dmabuf : false;
}

static bool ensure_loopback_dma_locked(vxfec_ibv_ctx_t *ctx)
{
    if (ctx->dma_stage_cap)
        return true;
    if (ctx->dma_stage_buf && !teardown_loopback_dma_locked(ctx))
        return false;

    const size_t cap = 256u * 1024u;
    void *buf = NULL;
    if (posix_memalign(&buf, 4096u, cap) != 0 || !buf)
        return false;
    memset(buf, 0, cap);

    ctx->dma_stage_buf = buf;
    ctx->dma_stage_mr = reg_host_raw(ctx->pd, buf, cap, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!ctx->dma_stage_mr)
        goto fail;
    ctx->dma_cq = g_ibv.create_cq(ctx->ibv, 16, NULL, NULL, 0);
    if (!ctx->dma_cq)
        goto fail;
    struct ibv_qp_init_attr qia;
    memset(&qia, 0, sizeof(qia));
    qia.send_cq = ctx->dma_cq;
    qia.recv_cq = ctx->dma_cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr = 16u;
    qia.cap.max_recv_wr = 4u;
    qia.cap.max_send_sge = 1u;
    qia.cap.max_recv_sge = 1u;
    struct ibv_qp *qp = g_ibv.create_qp(ctx->pd, &qia);
    if (!qp)
        goto fail;
    ctx->dma_qp = qp;

    struct ibv_qp_attr qa;
    memset(&qa, 0, sizeof(qa));
    qa.qp_state = IBV_QPS_INIT;
    qa.pkey_index = 0;
    qa.port_num = ctx->info.port_num ? ctx->info.port_num : 1u;
    qa.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    if (g_ibv.modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0)
        goto fail;

    memset(&qa, 0, sizeof(qa));
    qa.qp_state = IBV_QPS_RTR;
    qa.path_mtu = ctx->mtu_enum ? (enum ibv_mtu)ctx->mtu_enum : IBV_MTU_4096;
    qa.dest_qp_num = qp->qp_num;
    qa.rq_psn = 0u;
    qa.ah_attr.port_num = ctx->info.port_num ? ctx->info.port_num : 1u;
    qa.ah_attr.dlid = ctx->info.lid;
    if (ctx->info.is_roce || ctx->info.lid == 0u) {
        qa.ah_attr.is_global = 1;
        memcpy(qa.ah_attr.grh.dgid.raw, ctx->info.gid, 16);
        qa.ah_attr.grh.sgid_index = ctx->info.gid_index;
        qa.ah_attr.grh.hop_limit = 64u;
    }
    if (g_ibv.modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0)
        goto fail;

    memset(&qa, 0, sizeof(qa));
    qa.qp_state = IBV_QPS_RTS;
    qa.sq_psn = 0u;
    if (g_ibv.modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0)
        goto fail;

    ctx->dma_stage_cap = cap;
    return true;

fail:
    teardown_loopback_dma_locked(ctx);
    return false;
}

static bool loopback_rdma_copy_locked(vxfec_ibv_ctx_t *ctx, uint64_t src_iova, uint32_t src_lkey, uint64_t dst_iova,
                                      uint32_t dst_rkey, uint32_t len)
{
    struct ibv_recv_wr rwr, *bad_rwr = NULL;
    memset(&rwr, 0, sizeof(rwr));
    rwr.wr_id = 2ULL;
    if (ibv_post_recv(ctx->dma_qp, &rwr, &bad_rwr) != 0) {
        teardown_loopback_dma_locked(ctx);
        return false;
    }

    struct ibv_sge sge = {
        .addr = src_iova,
        .length = len,
        .lkey = src_lkey,
    };
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 1ULL;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    /* RDMA_WRITE_WITH_IMM forces a responder receive CQE only after target DMA commits */
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = htonl(0x56584601u);
    wr.wr.rdma.remote_addr = dst_iova;
    wr.wr.rdma.rkey = dst_rkey;

    struct ibv_send_wr *bad_wr = NULL;
    if (ibv_post_send(ctx->dma_qp, &wr, &bad_wr) != 0) {
        teardown_loopback_dma_locked(ctx);
        return false;
    }

    bool got_send = false, got_recv = false, failed = false;
    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint32_t polls = 0u;
    for (;;) {
        struct ibv_wc wcs[2];
        int n = ibv_poll_cq(ctx->dma_cq, 2, wcs);
        if (n < 0)
            failed = true;
        for (int i = 0; i < n; i++) {
            if (wcs[i].status != IBV_WC_SUCCESS)
                failed = true;
            if (wcs[i].wr_id == 1ULL)
                got_send = true;
            else if (wcs[i].wr_id == 2ULL)
                got_recv = true;
        }
        if (!failed && got_send && got_recv)
            return true;
        if (!failed && (++polls & 255u) == 0u) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            uint64_t elapsed_ns = (uint64_t)(now.tv_sec - t0.tv_sec) * 1000000000ULL + (uint64_t)(now.tv_nsec - t0.tv_nsec);
            failed = elapsed_ns > 2000000000ULL;
        }
        if (!failed)
            continue;

        /* The synchronous caller can reuse its MR as soon as we return. */
        teardown_loopback_dma_locked(ctx);
        if (!ctx->dma_qp || (got_send && got_recv))
            return false;
        struct ibv_qp_attr reset = {.qp_state = IBV_QPS_RESET};
        for (uint32_t tries = 0; tries < 50u; tries++) {
            if (g_ibv.modify_qp(ctx->dma_qp, &reset, IBV_QP_STATE) == 0)
                return false;
            const struct timespec retry = {.tv_nsec = 1000000L};
            nanosleep(&retry, NULL);
        }
        return false;
    }
}

bool vxfec_ibv_mr_read_to_host(vxfec_ibv_ctx_t *ctx, const vxfec_ibv_mr_t *src_mr, size_t src_off, void *dst, size_t bytes)
{
    if (!ctx || !ctx->dma_mtx_ready || !src_mr || !src_mr->mr || (!dst && bytes > 0u))
        return false;
    if (bytes == 0u)
        return true;
    if (src_off > src_mr->length || bytes > src_mr->length - src_off)
        return false;

    pthread_mutex_lock(&ctx->dma_mtx);
    bool ok = ensure_loopback_dma_locked(ctx);
    size_t done = 0u;
    uint64_t max_chunk = ctx->dma_stage_cap;
    if (ctx->max_msg_bytes > 0u && ctx->max_msg_bytes < max_chunk)
        max_chunk = ctx->max_msg_bytes;
    while (ok && done < bytes) {
        size_t chunk = bytes - done;
        if (chunk > (size_t)max_chunk)
            chunk = (size_t)max_chunk;
        uint64_t st_iova = (uint64_t)(uintptr_t)ctx->dma_stage_buf;
        ok = loopback_rdma_copy_locked(ctx, src_mr->iova + src_off + done, src_mr->mr->lkey, st_iova, ctx->dma_stage_mr->rkey,
                                       (uint32_t)chunk);
        if (ok)
            memcpy((uint8_t *)dst + done, ctx->dma_stage_buf, chunk);
        done += chunk;
    }
    pthread_mutex_unlock(&ctx->dma_mtx);
    return ok;
}

bool vxfec_ibv_mr_write_from_host(vxfec_ibv_ctx_t *ctx, vxfec_ibv_mr_t *dst_mr, size_t dst_off, const void *src, size_t bytes)
{
    if (!ctx || !ctx->dma_mtx_ready || !dst_mr || !dst_mr->mr || !dst_mr->remote_write || dst_mr->mr->rkey == 0u ||
        (!src && bytes > 0u))
        return false;
    if (bytes == 0u)
        return true;
    if (dst_off > dst_mr->length || bytes > dst_mr->length - dst_off)
        return false;

    pthread_mutex_lock(&ctx->dma_mtx);
    bool ok = ensure_loopback_dma_locked(ctx);
    size_t done = 0u;
    uint64_t max_chunk = ctx->dma_stage_cap;
    if (ctx->max_msg_bytes > 0u && ctx->max_msg_bytes < max_chunk)
        max_chunk = ctx->max_msg_bytes;
    while (ok && done < bytes) {
        size_t chunk = bytes - done;
        if (chunk > (size_t)max_chunk)
            chunk = (size_t)max_chunk;
        memcpy(ctx->dma_stage_buf, (const uint8_t *)src + done, chunk);
        uint64_t st_iova = (uint64_t)(uintptr_t)ctx->dma_stage_buf;
        ok = loopback_rdma_copy_locked(ctx, st_iova, ctx->dma_stage_mr->lkey, dst_mr->iova + dst_off + done, dst_mr->mr->rkey,
                                       (uint32_t)chunk);
        done += chunk;
    }
    /* issue a non-posted PCIe BAR1 readback to flush prior posted writes into GPU VRAM before returning */
    if (ok && dst_mr->is_dmabuf && dst_mr->mr->lkey != 0u) {
        uint32_t fence_len = (bytes < 4u) ? (uint32_t)bytes : 4u;
        uint64_t st_iova = (uint64_t)(uintptr_t)ctx->dma_stage_buf;
        ok = loopback_rdma_copy_locked(ctx, dst_mr->iova + dst_off + bytes - fence_len, dst_mr->mr->lkey, st_iova,
                                       ctx->dma_stage_mr->rkey, fence_len);
    }
    pthread_mutex_unlock(&ctx->dma_mtx);
    return ok;
}

static uint32_t select_rx_per_qp(const vxfec_ibv_ctx_t *ctx)
{
    uint32_t depth = VXFEC_IBV_RX_PER_QP_DEF;
    const char *env = getenv("VXFEC_IBV_RX_DEPTH");
    if (env && *env) {
        char *end = NULL;
        errno = 0;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 64 && v <= 16384)
            depth = (uint32_t)v;
    }
    if (ctx->max_qp_wr >= 64u && depth > ctx->max_qp_wr)
        depth = ctx->max_qp_wr;
    return depth;
}

static uint32_t select_tx_per_qp(const vxfec_ibv_ctx_t *ctx)
{
    uint32_t depth = VXFEC_IBV_TX_PER_QP;
    const char *env = getenv("VXFEC_IBV_TX_DEPTH");
    if (env && *env) {
        char *end = NULL;
        errno = 0;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 64 && v <= 4096) {
            uint32_t p2 = 64u;
            while (p2 * 2u <= (uint32_t)v)
                p2 *= 2u;
            depth = p2;
        }
    }
    if (ctx->max_qp_wr >= 64u && depth > ctx->max_qp_wr) {
        uint32_t p2 = 64u;
        while (p2 * 2u <= ctx->max_qp_wr)
            p2 *= 2u;
        depth = p2;
    }
    return depth;
}

static void generate_qp_psns(uint32_t *out_psn, uint32_t num_qps)
{
    if (getrandom(out_psn, (size_t)num_qps * sizeof(uint32_t), GRND_NONBLOCK) != (ssize_t)((size_t)num_qps * sizeof(uint32_t))) {
        struct timespec ts = {0};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t seed = ((uint64_t)ts.tv_sec << 32u) ^ (uint64_t)ts.tv_nsec ^ ((uint64_t)getpid() * 0x9e3779b97f4a7c15ULL);
        for (uint32_t q = 0; q < num_qps; q++) {
            seed ^= seed >> 12u;
            seed ^= seed << 25u;
            seed ^= seed >> 27u;
            out_psn[q] = (uint32_t)(seed * 0x2545f4914f6cdd1dULL);
        }
    }
    for (uint32_t q = 0; q < num_qps; q++)
        out_psn[q] &= 0x00ffffffu;
}

vxfec_ibv_ep_t *vxfec_ibv_ep_create_duplex(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                           bool enable_rx, bool header_only_tx, bool ctrl_duplex,
                                           vxfec_ibv_wire_info_t *out_local_info)
{
    if (!ctx || !out_local_info || num_qps == 0u || (!enable_tx && !enable_rx) ||
        (max_pkt_bytes == 0u && enable_tx && !header_only_tx))
        return NULL;
    if (num_qps > VXFEC_IBV_MAX_QPS)
        num_qps = VXFEC_IBV_MAX_QPS;

    if (enable_rx && max_pkt_bytes > 0u && max_pkt_bytes < 1024u)
        max_pkt_bytes = 1024u;

    uint64_t max_msg = ctx->max_msg_bytes ? ctx->max_msg_bytes : VXFEC_IBV_DEF_MAX_MSG_BYTES;
    if (max_pkt_bytes > UINT32_MAX - 63u || (uint64_t)max_pkt_bytes > max_msg)
        return NULL;

    vxfec_ibv_ep_t *ep = NULL;
    if (posix_memalign((void **)&ep, 64u, sizeof(*ep)) != 0 || !ep)
        return NULL;
    memset(ep, 0, sizeof(*ep));
    uint32_t ctrl_tx_depth = VXFEC_IBV_CTRL_TX_DEPTH;
    uint32_t ctrl_rx_depth = VXFEC_IBV_CTRL_RX_DEPTH;
    if (ctx->max_qp_wr > 0u) {
        if (ctrl_tx_depth > ctx->max_qp_wr)
            ctrl_tx_depth = ctx->max_qp_wr;
        if (ctrl_rx_depth > ctx->max_qp_wr)
            ctrl_rx_depth = ctx->max_qp_wr;
    }
    if (pthread_mutex_init(&ep->tx_cq_mtx, NULL) != 0) {
        free(ep);
        return NULL;
    }
    ep->tx_cq_mtx_ready = true;
    ep->ctx = ctx;
    ep->num_qps = num_qps;
    ep->tx_per_qp = enable_tx ? select_tx_per_qp(ctx) : (ctrl_duplex ? ctrl_tx_depth : 1u);
    ep->rx_per_qp = enable_rx ? select_rx_per_qp(ctx) : 1u;
    ep->ctrl_tx_depth = ctrl_tx_depth;
    ep->ctrl_rx_depth = ctrl_rx_depth;
    ep->max_pkt_bytes = max_pkt_bytes;
    ep->path_mtu_bytes = mtu_enum_to_bytes((enum ibv_mtu)ctx->mtu_enum);
    ep->peer_max_recv_bytes = ep->path_mtu_bytes;
    ep->max_msg_bytes = max_msg;
    ep->rx_stride = max_pkt_bytes ? ((max_pkt_bytes + 63u) & ~63u) : 0u;
    ep->enable_tx = enable_tx;
    ep->enable_rx = enable_rx;
    ep->rx_imm_only = enable_rx && (max_pkt_bytes == 0u);
    ep->ctrl_tx_enabled = ctrl_duplex && enable_rx && !enable_tx;
    ep->ctrl_rx_enabled = ctrl_duplex && enable_tx && !enable_rx;

    int rx_cqe = enable_rx ? (int)(num_qps * ep->rx_per_qp) : (ep->ctrl_rx_enabled ? (int)ctrl_rx_depth : 16);
    int tx_cqe = enable_tx ? (int)(num_qps * ep->tx_per_qp) : (ep->ctrl_tx_enabled ? (int)ctrl_tx_depth : 16);
    ep->rx_cq = g_ibv.create_cq(ctx->ibv, rx_cqe, NULL, NULL, 0);
    ep->tx_cq = g_ibv.create_cq(ctx->ibv, tx_cqe, NULL, NULL, 0);
    if (!ep->rx_cq || !ep->tx_cq) {
        vxfec_ibv_ep_destroy(ep);
        return NULL;
    }

    if (enable_rx && (!ep->rx_imm_only || ep->ctrl_tx_enabled)) {
        size_t rx_data_bytes = ep->rx_imm_only ? 0u : ((size_t)num_qps * ep->rx_per_qp * ep->rx_stride);
        size_t ctrl_tx_bytes = ep->ctrl_tx_enabled ? ((size_t)ep->ctrl_tx_depth * VXFEC_IBV_TX_HDR_CAP) : 0u;
        ep->rx_slab_bytes = (rx_data_bytes + ctrl_tx_bytes + 4095u) & ~(size_t)4095u;
        if (posix_memalign((void **)&ep->rx_slab, 4096u, ep->rx_slab_bytes) != 0) {
            vxfec_ibv_ep_destroy(ep);
            return NULL;
        }
        memset(ep->rx_slab, 0, ep->rx_slab_bytes);
        if (ep->ctrl_tx_enabled)
            ep->ctrl_tx_buf = ep->rx_slab + rx_data_bytes;
        ep->rx_mr = reg_host_raw(ctx->pd, ep->rx_slab, ep->rx_slab_bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!ep->rx_mr) {
            vxfec_ibv_ep_destroy(ep);
            return NULL;
        }
    }

    if (enable_tx) {
        size_t ctrl_rx_bytes = ep->ctrl_rx_enabled ? ((size_t)ep->ctrl_rx_depth * VXFEC_IBV_CTRL_RX_STRIDE) : 0u;
        if (!header_only_tx) {
            ep->tx_stride = (ep->rx_stride < VXFEC_IBV_MAX_STAGE_STRIDE) ? ep->rx_stride : VXFEC_IBV_MAX_STAGE_STRIDE;
            size_t tx_data_bytes = (size_t)num_qps * ep->tx_per_qp * ep->tx_stride;
            ep->tx_slab_bytes = (tx_data_bytes + ctrl_rx_bytes + 4095u) & ~(size_t)4095u;
            if (posix_memalign((void **)&ep->tx_slab, 4096u, ep->tx_slab_bytes) == 0) {
                memset(ep->tx_slab, 0, ep->tx_slab_bytes);
                if (ep->ctrl_rx_enabled)
                    ep->ctrl_rx_buf = ep->tx_slab + tx_data_bytes;
                ep->tx_mr = reg_host_raw(ctx->pd, ep->tx_slab, ep->tx_slab_bytes, IBV_ACCESS_LOCAL_WRITE);
            }
        }
        if (!ep->tx_mr) {
            /* compact 64B/slot header-only MR for zero-copy 2-SGE sends (65x smaller pinned slab) */
            free(ep->tx_slab);
            ep->tx_slab = NULL;
            ep->ctrl_rx_buf = NULL;
            ep->tx_stride = VXFEC_IBV_TX_HDR_CAP;
            ep->tx_hdr_only = true;
            size_t tx_data_bytes = (size_t)num_qps * ep->tx_per_qp * ep->tx_stride;
            ep->tx_slab_bytes = (tx_data_bytes + ctrl_rx_bytes + 4095u) & ~(size_t)4095u;
            if (posix_memalign((void **)&ep->tx_slab, 4096u, ep->tx_slab_bytes) != 0) {
                vxfec_ibv_ep_destroy(ep);
                return NULL;
            }
            memset(ep->tx_slab, 0, ep->tx_slab_bytes);
            if (ep->ctrl_rx_enabled)
                ep->ctrl_rx_buf = ep->tx_slab + tx_data_bytes;
            ep->tx_mr = reg_host_raw(ctx->pd, ep->tx_slab, ep->tx_slab_bytes, IBV_ACCESS_LOCAL_WRITE);
            if (!ep->tx_mr) {
                vxfec_ibv_ep_destroy(ep);
                return NULL;
            }
        }
    }

    memset(out_local_info, 0, sizeof(*out_local_info));
    memcpy(out_local_info->gid, ctx->info.gid, 16);
    out_local_info->lid = ctx->info.lid;
    out_local_info->port_num = ctx->info.port_num;
    out_local_info->gid_index = ctx->info.gid_index;
    out_local_info->mtu_enum = ctx->mtu_enum;
    out_local_info->num_qps = (uint8_t)num_qps;
    out_local_info->is_roce = ctx->info.is_roce ? 1u : 0u;
    uint8_t wire_flags = 0u;
    if (enable_rx && max_pkt_bytes >= 1024u) {
        uint32_t rx_kib = max_pkt_bytes / 1024u;
        if (rx_kib > VXFEC_IBV_WIRE_RX_KIB_MASK)
            rx_kib = VXFEC_IBV_WIRE_RX_KIB_MASK;
        wire_flags |= (uint8_t)rx_kib;
    }
    if (ep->ctrl_rx_enabled)
        wire_flags |= VXFEC_IBV_WIRE_CTRL_RX;
    out_local_info->reserved = wire_flags;
    generate_qp_psns(ep->local_psn, num_qps);
    ep->max_inline_data = UINT32_MAX;

    for (uint32_t q = 0; q < num_qps; q++) {
        bool q_has_ctrl_tx = (q == 0u && ep->ctrl_tx_enabled);
        bool q_has_ctrl_rx = (q == 0u && ep->ctrl_rx_enabled);
        struct ibv_qp_init_attr qia;
        memset(&qia, 0, sizeof(qia));
        qia.send_cq = ep->tx_cq;
        qia.recv_cq = ep->rx_cq;
        qia.qp_type = IBV_QPT_UC;
        qia.sq_sig_all = 0;
        qia.cap.max_send_wr = enable_tx ? ep->tx_per_qp : (q_has_ctrl_tx ? ctrl_tx_depth : 1u);
        qia.cap.max_recv_wr = enable_rx ? ep->rx_per_qp : (q_has_ctrl_rx ? ctrl_rx_depth : 1u);
        qia.cap.max_send_sge = enable_tx ? 2u : 1u;
        qia.cap.max_recv_sge = 1u;
        qia.cap.max_inline_data = (enable_tx || q_has_ctrl_tx) ? VXFEC_IBV_TX_HDR_CAP : 0u;

        ep->qps[q] = g_ibv.create_qp(ctx->pd, &qia);
        if (!ep->qps[q] && qia.cap.max_inline_data > 0u) {
            qia.cap.max_inline_data = 0u;
            ep->qps[q] = g_ibv.create_qp(ctx->pd, &qia);
        }
        if (!ep->qps[q]) {
            vxfec_ibv_ep_destroy(ep);
            return NULL;
        }
        if ((enable_tx || q_has_ctrl_tx) && qia.cap.max_inline_data < ep->max_inline_data)
            ep->max_inline_data = qia.cap.max_inline_data;

        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = ctx->info.port_num;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        int init_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        if (g_ibv.modify_qp(ep->qps[q], &attr, init_mask) != 0) {
            vxfec_ibv_ep_destroy(ep);
            return NULL;
        }

        out_local_info->qpn[q] = ep->qps[q]->qp_num;
        out_local_info->psn[q] = ep->local_psn[q];

        if (enable_rx) {
            uint32_t slots[VXFEC_IBV_POLL_MAX];
            uint32_t s = 0u;
            while (s < ep->rx_per_qp) {
                uint32_t chunk = (ep->rx_per_qp - s > VXFEC_IBV_POLL_MAX) ? VXFEC_IBV_POLL_MAX : (ep->rx_per_qp - s);
                for (uint32_t i = 0; i < chunk; i++)
                    slots[i] = (q << 16u) | (s + i);
                if (!vxfec_ibv_ep_repost_rx_batch(ep, slots, chunk)) {
                    vxfec_ibv_ep_destroy(ep);
                    return NULL;
                }
                s += chunk;
            }
        } else if (q_has_ctrl_rx) {
            uint32_t cslots[VXFEC_IBV_CTRL_RX_DEPTH];
            for (uint32_t s = 0; s < ep->ctrl_rx_depth; s++)
                cslots[s] = VXFEC_IBV_CTRL_SLOT_FLAG | s;
            if (!vxfec_ibv_ep_repost_rx_batch(ep, cslots, ep->ctrl_rx_depth)) {
                vxfec_ibv_ep_destroy(ep);
                return NULL;
            }
        }
    }

    if (ep->max_inline_data == UINT32_MAX)
        ep->max_inline_data = 0u;
    return ep;
}

vxfec_ibv_ep_t *vxfec_ibv_ep_create_opts(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                         bool enable_rx, bool header_only_tx, vxfec_ibv_wire_info_t *out_local_info)
{
    return vxfec_ibv_ep_create_duplex(ctx, num_qps, max_pkt_bytes, enable_tx, enable_rx, header_only_tx, false, out_local_info);
}

vxfec_ibv_ep_t *vxfec_ibv_ep_create_ex(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                       bool enable_rx, vxfec_ibv_wire_info_t *out_local_info)
{
    return vxfec_ibv_ep_create_opts(ctx, num_qps, max_pkt_bytes, enable_tx, enable_rx, false, out_local_info);
}

bool vxfec_ibv_ep_connect(vxfec_ibv_ep_t *ep, const vxfec_ibv_wire_info_t *peer_info, uint8_t traffic_class)
{
    if (!ep || !peer_info || peer_info->num_qps != ep->num_qps)
        return false;

    uint8_t path_mtu = (peer_info->mtu_enum < ep->ctx->mtu_enum) ? peer_info->mtu_enum : ep->ctx->mtu_enum;
    if (path_mtu < IBV_MTU_256 || path_mtu > IBV_MTU_4096)
        path_mtu = IBV_MTU_1024;
    ep->path_mtu_bytes = mtu_enum_to_bytes((enum ibv_mtu)path_mtu);
    uint32_t peer_kib = (uint32_t)(peer_info->reserved & VXFEC_IBV_WIRE_RX_KIB_MASK);
    ep->peer_max_recv_bytes = (peer_kib > 0u) ? (peer_kib * 1024u) : ep->path_mtu_bytes;
    ep->peer_has_ctrl_rx = (peer_info->reserved & VXFEC_IBV_WIRE_CTRL_RX) != 0u;

    for (uint32_t q = 0; q < ep->num_qps; q++) {
        struct ibv_qp_attr rtr;
        memset(&rtr, 0, sizeof(rtr));
        rtr.qp_state = IBV_QPS_RTR;
        rtr.path_mtu = (enum ibv_mtu)path_mtu;
        rtr.dest_qp_num = peer_info->qpn[q];
        rtr.rq_psn = peer_info->psn[q] & 0x00ffffffu;
        rtr.ah_attr.dlid = peer_info->lid;
        rtr.ah_attr.sl = 0;
        rtr.ah_attr.src_path_bits = 0;
        rtr.ah_attr.port_num = ep->ctx->info.port_num;

        if (ep->ctx->info.is_roce || peer_info->is_roce) {
            rtr.ah_attr.is_global = 1;
            memcpy(rtr.ah_attr.grh.dgid.raw, peer_info->gid, 16);
            /* distinct flow_label per QP drives RoCEv2 UDP src-port ECMP hashing across spine switches */
            rtr.ah_attr.grh.flow_label = ((q + 1u) * 0x13579u) & 0x000fffffu;
            rtr.ah_attr.grh.sgid_index = ep->ctx->info.gid_index;
            rtr.ah_attr.grh.hop_limit = 64u;
            rtr.ah_attr.grh.traffic_class = traffic_class;
        }

        int rtr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
        if (g_ibv.modify_qp(ep->qps[q], &rtr, rtr_mask) != 0)
            return false;

        struct ibv_qp_attr rts;
        memset(&rts, 0, sizeof(rts));
        rts.qp_state = IBV_QPS_RTS;
        rts.sq_psn = ep->local_psn[q] & 0x00ffffffu;
        if (g_ibv.modify_qp(ep->qps[q], &rts, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0)
            return false;
    }

    return true;
}

bool vxfec_ibv_ep_destroy(vxfec_ibv_ep_t *ep)
{
    if (!ep)
        return true;

    __atomic_store_n(&ep->destroying, true, __ATOMIC_RELEASE);
    __atomic_store_n(&ep->tx_error, true, __ATOMIC_RELEASE);
    bool qp_busy = false;
    for (uint32_t q = 0; q < VXFEC_IBV_MAX_QPS; q++) {
        if (ep->qps[q]) {
            struct ibv_qp_attr err_attr;
            memset(&err_attr, 0, sizeof(err_attr));
            err_attr.qp_state = IBV_QPS_ERR;
            (void)g_ibv.modify_qp(ep->qps[q], &err_attr, IBV_QP_STATE);
            if (g_ibv.destroy_qp(ep->qps[q]) == 0)
                ep->qps[q] = NULL;
            else
                qp_busy = true;
        }
    }
    if (qp_busy) {
        __atomic_store_n(&ep->tx_error, true, __ATOMIC_RELEASE);
        return false;
    }

    bool mr_busy = false;
    if (ep->rx_mr) {
        if (g_ibv.dereg_mr(ep->rx_mr) == 0)
            ep->rx_mr = NULL;
        else
            mr_busy = true;
    }
    if (ep->tx_mr) {
        if (g_ibv.dereg_mr(ep->tx_mr) == 0)
            ep->tx_mr = NULL;
        else
            mr_busy = true;
    }
    if (!ep->rx_mr) {
        free(ep->rx_slab);
        ep->rx_slab = NULL;
        ep->ctrl_tx_buf = NULL;
    }
    if (!ep->tx_mr) {
        free(ep->tx_slab);
        ep->tx_slab = NULL;
        ep->ctrl_rx_buf = NULL;
    }
    if (ep->rx_cq && g_ibv.destroy_cq(ep->rx_cq) == 0)
        ep->rx_cq = NULL;
    if (ep->tx_cq && g_ibv.destroy_cq(ep->tx_cq) == 0)
        ep->tx_cq = NULL;
    if (mr_busy || ep->rx_cq || ep->tx_cq)
        return false;

    if (ep->tx_cq_mtx_ready)
        pthread_mutex_destroy(&ep->tx_cq_mtx);
    free(ep);
    return true;
}

bool vxfec_ibv_ep_has_error(const vxfec_ibv_ep_t *ep)
{
    return ep ? __atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE) : false;
}

bool vxfec_ibv_ep_tx_is_hdr_only(const vxfec_ibv_ep_t *ep)
{
    return ep ? ep->tx_hdr_only : false;
}

uint32_t vxfec_ibv_ep_path_mtu(const vxfec_ibv_ep_t *ep)
{
    return ep ? ep->path_mtu_bytes : 0u;
}

uint32_t vxfec_ibv_ep_num_qps(const vxfec_ibv_ep_t *ep)
{
    return ep ? ep->num_qps : 0u;
}

uint64_t vxfec_ibv_ep_max_msg_bytes(const vxfec_ibv_ep_t *ep)
{
    return ep ? ep->max_msg_bytes : 0ULL;
}

uint32_t vxfec_ibv_ep_peer_max_recv(const vxfec_ibv_ep_t *ep)
{
    return ep ? ep->peer_max_recv_bytes : 0u;
}

uint32_t vxfec_ibv_ep_tx_stage_cap(const vxfec_ibv_ep_t *ep)
{
    return (ep && ep->enable_tx && !ep->tx_hdr_only) ? ep->tx_stride : 0u;
}

bool vxfec_ibv_ep_peer_has_ctrl_rx(const vxfec_ibv_ep_t *ep)
{
    return ep ? ep->peer_has_ctrl_rx : false;
}

void vxfec_ibv_ep_drain_tx(vxfec_ibv_ep_t *ep)
{
    if (!ep || !ep->tx_cq || (!ep->enable_tx && !ep->ctrl_tx_enabled))
        return;
    if (ep->tx_cq_mtx_ready)
        pthread_mutex_lock(&ep->tx_cq_mtx);
    struct ibv_wc wcs[64];
    for (;;) {
        int n = ibv_poll_cq(ep->tx_cq, 64, wcs);
        if (n < 0) {
            __atomic_store_n(&ep->tx_error, true, __ATOMIC_RELEASE);
            break;
        }
        if (n == 0)
            break;
        for (int i = 0; i < n; i++) {
            uint32_t q = (uint32_t)(wcs[i].wr_id >> 56u);
            uint64_t seq = wcs[i].wr_id & 0x00ffffffffffffffULL;
            if (wcs[i].status != IBV_WC_SUCCESS) {
                __atomic_store_n(&ep->tx_error, true, __ATOMIC_RELEASE);
                if (q < ep->num_qps)
                    __atomic_store_n(&ep->qp_tx[q].qp_err, true, __ATOMIC_RELEASE);
            }
            if (q < ep->num_qps) {
                uint64_t cur = __atomic_load_n(&ep->qp_tx[q].sq_done, __ATOMIC_RELAXED);
                if (seq > cur)
                    __atomic_store_n(&ep->qp_tx[q].sq_done, seq, __ATOMIC_RELEASE);
            }
        }
    }
    if (ep->tx_cq_mtx_ready)
        pthread_mutex_unlock(&ep->tx_cq_mtx);
}

void vxfec_ibv_ep_record_qp_tail(const vxfec_ibv_ep_t *ep, uint32_t qp_idx, uint64_t target_seq[VXFEC_IBV_MAX_QPS])
{
    if (!ep || !target_seq || ep->num_qps == 0u)
        return;
    uint32_t q = qp_idx % ep->num_qps;
    if (ep->qp_tx[q].sq_posted > target_seq[q])
        target_seq[q] = ep->qp_tx[q].sq_posted;
}

bool vxfec_ibv_ep_tx_done(vxfec_ibv_ep_t *ep, const uint64_t target_seq[VXFEC_IBV_MAX_QPS])
{
    if (!ep || !target_seq || (!ep->enable_tx && !ep->ctrl_tx_enabled))
        return true;
    if (__atomic_load_n(&ep->destroying, __ATOMIC_ACQUIRE) || __atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE))
        return false;
    vxfec_ibv_ep_drain_tx(ep);
    if (__atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE))
        return false;

    bool posted_fence = false;
    for (uint32_t q = 0; q < ep->num_qps; q++) {
        vxfec_ibv_qp_tx_t *qt = &ep->qp_tx[q];
        uint64_t done_q = __atomic_load_n(&qt->sq_done, __ATOMIC_ACQUIRE);
        if (done_q >= target_seq[q])
            continue;
        if (!ep->qps[q])
            return false;
        if (!__atomic_load_n(&qt->qp_err, __ATOMIC_ACQUIRE) && qt->sq_last_sig < target_seq[q] &&
            (qt->sq_posted - done_q) < ep->tx_per_qp) {
            uint64_t next_seq = qt->sq_posted + 1ULL;
            struct ibv_send_wr wr, *bad = NULL;
            memset(&wr, 0, sizeof(wr));
            wr.wr_id = ((uint64_t)q << 56u) | (next_seq & 0x00ffffffffffffffULL);
            /* 0-byte RDMA_WRITE fences prior SQ WQEs without consuming any remote UC Receive WQE */
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.send_flags = IBV_SEND_SIGNALED;
            if (ibv_post_send(ep->qps[q], &wr, &bad) == 0) {
                qt->sq_posted = next_seq;
                qt->sq_last_sig = next_seq;
                qt->tx_unsignaled = 0u;
                posted_fence = true;
            } else {
                __atomic_store_n(&ep->tx_error, true, __ATOMIC_RELEASE);
                __atomic_store_n(&qt->qp_err, true, __ATOMIC_RELEASE);
            }
        }
    }
    if (posted_fence)
        vxfec_ibv_ep_drain_tx(ep);
    if (__atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE))
        return false;

    for (uint32_t q = 0; q < ep->num_qps; q++) {
        if (__atomic_load_n(&ep->qp_tx[q].sq_done, __ATOMIC_ACQUIRE) < target_seq[q])
            return false;
    }
    return true;
}

static uint32_t ep_sq_reserve_batch(vxfec_ibv_ep_t *ep, uint32_t q, uint32_t count)
{
    uint32_t n = (count > VXFEC_IBV_BATCH_MAX) ? VXFEC_IBV_BATCH_MAX : count;
    uint32_t tx_cap = ep->tx_per_qp;
    uint32_t usable_cap = (tx_cap > 1u) ? (tx_cap - 1u) : 0u;
    vxfec_ibv_qp_tx_t *qt = &ep->qp_tx[q];
    uint64_t inflight = qt->sq_posted - __atomic_load_n(&qt->sq_done, __ATOMIC_ACQUIRE);
    if (inflight + n > usable_cap || inflight >= tx_cap - VXFEC_IBV_SIGNAL_MOD) {
        vxfec_ibv_ep_drain_tx(ep);
        if (__atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE))
            return 0u;
        inflight = qt->sq_posted - __atomic_load_n(&qt->sq_done, __ATOMIC_ACQUIRE);
    }
    if (inflight >= usable_cap)
        return 0u;
    uint32_t avail = (uint32_t)(usable_cap - inflight);
    return (n > avail) ? avail : n;
}

static uint32_t ep_sq_commit_post(vxfec_ibv_ep_t *ep, uint32_t q, struct ibv_send_wr *wrs, uint32_t valid_n, uint64_t seq,
                                  const uint32_t *post_unsig, const uint64_t *post_last_sig)
{
    vxfec_ibv_qp_tx_t *qt = &ep->qp_tx[q];
    wrs[valid_n - 1u].next = NULL;
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(ep->qps[q], &wrs[0], &bad);
    if (rc != 0) {
        uint32_t ok_cnt = (bad >= &wrs[0] && bad < &wrs[valid_n]) ? (uint32_t)(bad - &wrs[0]) : 0u;
        if (ok_cnt > 0u) {
            qt->sq_posted += ok_cnt;
            qt->tx_unsignaled = post_unsig[ok_cnt - 1u];
            qt->sq_last_sig = post_last_sig[ok_cnt - 1u];
        }
        int err = (rc > 0) ? rc : errno;
        if (err != ENOMEM && err != EAGAIN) {
            __atomic_store_n(&ep->tx_error, true, __ATOMIC_RELEASE);
            __atomic_store_n(&qt->qp_err, true, __ATOMIC_RELEASE);
        }
        return ok_cnt;
    }

    qt->sq_posted = seq;
    qt->tx_unsignaled = post_unsig[valid_n - 1u];
    qt->sq_last_sig = post_last_sig[valid_n - 1u];
    return valid_n;
}

uint32_t vxfec_ibv_ep_send_pkt_batch(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const vxfec_ibv_send_wr_desc_t *pkts, uint32_t count)
{
    if (!ep || !ep->enable_tx || __atomic_load_n(&ep->destroying, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE) || !pkts || count == 0u || ep->num_qps == 0u)
        return 0u;

    uint32_t q = qp_idx % ep->num_qps;
    if (!ep->qps[q] || !ep->tx_mr)
        return 0u;

    uint32_t tx_cap = ep->tx_per_qp;
    uint32_t n = ep_sq_reserve_batch(ep, q, count);
    if (n == 0u)
        return 0u;

    uint32_t pkt_cap = ep->peer_max_recv_bytes ? ep->peer_max_recv_bytes : ep->path_mtu_bytes;
    if (ep->max_pkt_bytes > 0u && pkt_cap > ep->max_pkt_bytes)
        pkt_cap = ep->max_pkt_bytes;
    if ((uint64_t)pkt_cap > ep->max_msg_bytes)
        pkt_cap = (uint32_t)ep->max_msg_bytes;

    struct ibv_sge sges[VXFEC_IBV_BATCH_MAX][2];
    struct ibv_send_wr wrs[VXFEC_IBV_BATCH_MAX];
    uint32_t post_unsig[VXFEC_IBV_BATCH_MAX];
    uint64_t post_last_sig[VXFEC_IBV_BATCH_MAX];
    memset(wrs, 0, (size_t)n * sizeof(wrs[0]));

    vxfec_ibv_qp_tx_t *qt = &ep->qp_tx[q];
    uint64_t seq = qt->sq_posted;
    uint32_t unsig = qt->tx_unsignaled;
    uint32_t valid_n = 0u;

    for (uint32_t i = 0; i < n; i++) {
        const vxfec_ibv_send_wr_desc_t *p = &pkts[i];
        if (!p->hdr || p->hdr_len == 0u || (p->pay_len > 0u && !p->payload))
            break;
        bool has_pay = (p->pay_len > 0u);
        bool stage_pay = has_pay && (p->pay_lkey == 0u);
        if (p->hdr_len > pkt_cap || (has_pay && p->pay_len > pkt_cap - p->hdr_len))
            break;
        if (stage_pay) {
            if (ep->tx_hdr_only || p->hdr_len + p->pay_len > ep->tx_stride)
                break;
        } else if (p->hdr_len > VXFEC_IBV_TX_HDR_CAP) {
            break;
        }

        int num_sge = 1;
        bool send_inline = false;
        if (!has_pay && p->hdr_len <= ep->max_inline_data) {
            sges[i][0].addr = (uint64_t)(uintptr_t)p->hdr;
            sges[i][0].length = p->hdr_len;
            sges[i][0].lkey = 0u;
            send_inline = true;
        } else if (stage_pay && p->hdr_len + p->pay_len <= ep->max_inline_data) {
            sges[i][0].addr = (uint64_t)(uintptr_t)p->hdr;
            sges[i][0].length = p->hdr_len;
            sges[i][0].lkey = 0u;
            sges[i][1].addr = (uint64_t)(uintptr_t)p->payload;
            sges[i][1].length = p->pay_len;
            sges[i][1].lkey = 0u;
            num_sge = 2;
            send_inline = true;
        } else {
            uint32_t slot = (uint32_t)(seq & (tx_cap - 1u));
            uint8_t *tdst = ep->tx_slab + ((size_t)q * tx_cap + slot) * ep->tx_stride;
            memcpy(tdst, p->hdr, p->hdr_len);

            sges[i][0].addr = (uint64_t)(uintptr_t)tdst;
            sges[i][0].length = p->hdr_len;
            sges[i][0].lkey = ep->tx_mr->lkey;

            if (stage_pay) {
                memcpy(tdst + p->hdr_len, p->payload, p->pay_len);
                sges[i][0].length = p->hdr_len + p->pay_len;
            } else if (has_pay) {
                sges[i][1].addr = (uint64_t)(uintptr_t)p->payload;
                sges[i][1].length = p->pay_len;
                sges[i][1].lkey = p->pay_lkey;
                num_sge = 2;
            }
        }

        seq++;
        unsig++;
        bool sig = p->force_signal || (unsig >= VXFEC_IBV_SIGNAL_MOD);

        wrs[i].wr_id = ((uint64_t)q << 56u) | (seq & 0x00ffffffffffffffULL);
        wrs[i].sg_list = sges[i];
        wrs[i].num_sge = num_sge;
        wrs[i].opcode = IBV_WR_SEND;
        wrs[i].send_flags = (sig ? IBV_SEND_SIGNALED : 0) | (send_inline ? IBV_SEND_INLINE : 0);
        if (i > 0u)
            wrs[i - 1u].next = &wrs[i];

        if (sig)
            unsig = 0u;
        post_unsig[i] = unsig;
        post_last_sig[i] = sig ? seq : ((i > 0u) ? post_last_sig[i - 1u] : qt->sq_last_sig);
        valid_n++;
    }
    if (valid_n == 0u)
        return 0u;
    return ep_sq_commit_post(ep, q, wrs, valid_n, seq, post_unsig, post_last_sig);
}

bool vxfec_ibv_ep_send_pkt(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *hdr, uint32_t hdr_len, const void *payload,
                           uint32_t pay_len, uint32_t pay_lkey)
{
    vxfec_ibv_send_wr_desc_t d = {
        .hdr = hdr,
        .payload = payload,
        .hdr_len = hdr_len,
        .pay_len = pay_len,
        .pay_lkey = pay_lkey,
        .force_signal = false,
    };
    return vxfec_ibv_ep_send_pkt_batch(ep, qp_idx, &d, 1u) == 1u;
}

bool vxfec_ibv_ep_send_ctrl_inline(vxfec_ibv_ep_t *ep, const void *buf, uint32_t len)
{
    if (!ep || !ep->ctrl_tx_enabled || !ep->peer_has_ctrl_rx || __atomic_load_n(&ep->destroying, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE) || !buf || len == 0u || len > VXFEC_IBV_CTRL_MAX_BYTES || !ep->qps[0] ||
        !ep->ctrl_tx_buf || !ep->rx_mr)
        return false;

    vxfec_ibv_qp_tx_t *qt = &ep->qp_tx[0];
    uint32_t tx_cap = ep->ctrl_tx_depth ? ep->ctrl_tx_depth : 1u;
    uint32_t sig_mod = (tx_cap > VXFEC_IBV_SIGNAL_MOD) ? VXFEC_IBV_SIGNAL_MOD : 1u;
    if ((qt->sq_posted - __atomic_load_n(&qt->sq_done, __ATOMIC_ACQUIRE)) >= (tx_cap - sig_mod))
        vxfec_ibv_ep_drain_tx(ep);
    if (__atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE) ||
        (qt->sq_posted - __atomic_load_n(&qt->sq_done, __ATOMIC_ACQUIRE)) >= (tx_cap > 1u ? (tx_cap - 1u) : 1u))
        return false;

    bool send_inline = (len <= ep->max_inline_data);
    struct ibv_sge sge;
    if (send_inline) {
        sge.addr = (uint64_t)(uintptr_t)buf;
        sge.length = len;
        sge.lkey = 0u;
    } else {
        uint32_t slot = (uint32_t)(qt->sq_posted % tx_cap);
        uint8_t *tdst = ep->ctrl_tx_buf + (size_t)slot * VXFEC_IBV_TX_HDR_CAP;
        memcpy(tdst, buf, len);
        sge.addr = (uint64_t)(uintptr_t)tdst;
        sge.length = len;
        sge.lkey = ep->rx_mr->lkey;
    }

    uint64_t next_seq = qt->sq_posted + 1ULL;
    qt->tx_unsignaled++;
    bool sig = (qt->tx_unsignaled >= VXFEC_IBV_SIGNAL_MOD);

    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = next_seq & 0x00ffffffffffffffULL;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = (send_inline ? IBV_SEND_INLINE : 0) | (sig ? IBV_SEND_SIGNALED : 0);

    int rc = ibv_post_send(ep->qps[0], &wr, &bad);
    if (rc != 0) {
        qt->tx_unsignaled--;
        int err = (rc > 0) ? rc : errno;
        if (err != ENOMEM && err != EAGAIN) {
            __atomic_store_n(&ep->tx_error, true, __ATOMIC_RELEASE);
            __atomic_store_n(&qt->qp_err, true, __ATOMIC_RELEASE);
        }
        return false;
    }
    qt->sq_posted = next_seq;
    if (sig) {
        qt->sq_last_sig = next_seq;
        qt->tx_unsignaled = 0u;
    }
    return true;
}

uint32_t vxfec_ibv_ep_rdma_write_imm_batch(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const vxfec_ibv_rdma_wr_t *wrs, uint32_t count)
{
    if (!ep || !ep->enable_tx || __atomic_load_n(&ep->destroying, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE) || !wrs || count == 0u || ep->num_qps == 0u)
        return 0u;

    uint32_t q = qp_idx % ep->num_qps;
    if (!ep->qps[q])
        return 0u;

    uint32_t n = ep_sq_reserve_batch(ep, q, count);
    if (n == 0u)
        return 0u;

    struct ibv_sge sges[VXFEC_IBV_BATCH_MAX];
    struct ibv_send_wr ibv_wrs[VXFEC_IBV_BATCH_MAX];
    uint32_t post_unsig[VXFEC_IBV_BATCH_MAX];
    uint64_t post_last_sig[VXFEC_IBV_BATCH_MAX];
    memset(ibv_wrs, 0, (size_t)n * sizeof(ibv_wrs[0]));

    vxfec_ibv_qp_tx_t *qt = &ep->qp_tx[q];
    uint64_t seq = qt->sq_posted;
    uint32_t unsig = qt->tx_unsignaled;
    uint32_t valid_n = 0u;

    for (uint32_t i = 0; i < n; i++) {
        if ((wrs[i].byte_len > 0u && !wrs[i].src_addr) || (uint64_t)wrs[i].byte_len > ep->max_msg_bytes)
            break;

        sges[i].addr = (uint64_t)(uintptr_t)wrs[i].src_addr;
        sges[i].length = wrs[i].byte_len;
        sges[i].lkey = wrs[i].lkey;

        seq++;
        unsig++;
        bool sig = wrs[i].force_signal || (unsig >= VXFEC_IBV_SIGNAL_MOD);

        ibv_wrs[i].wr_id = ((uint64_t)q << 56u) | (seq & 0x00ffffffffffffffULL);
        ibv_wrs[i].sg_list = wrs[i].byte_len ? &sges[i] : NULL;
        ibv_wrs[i].num_sge = wrs[i].byte_len ? 1 : 0;
        ibv_wrs[i].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        ibv_wrs[i].send_flags = sig ? IBV_SEND_SIGNALED : 0;
        ibv_wrs[i].imm_data = htonl(wrs[i].imm_host_order);
        ibv_wrs[i].wr.rdma.remote_addr = wrs[i].remote_addr;
        ibv_wrs[i].wr.rdma.rkey = wrs[i].rkey;
        if (i > 0u)
            ibv_wrs[i - 1u].next = &ibv_wrs[i];

        if (sig)
            unsig = 0u;
        post_unsig[i] = unsig;
        post_last_sig[i] = sig ? seq : ((i > 0u) ? post_last_sig[i - 1u] : qt->sq_last_sig);
        valid_n++;
    }
    if (valid_n == 0u)
        return 0u;
    return ep_sq_commit_post(ep, q, ibv_wrs, valid_n, seq, post_unsig, post_last_sig);
}

bool vxfec_ibv_ep_rdma_write_imm(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *src_addr, uint32_t byte_len, uint32_t lkey,
                                 uint64_t remote_addr, uint32_t rkey, uint32_t imm_host_order)
{
    vxfec_ibv_rdma_wr_t wr = {
        .src_addr = src_addr,
        .remote_addr = remote_addr,
        .byte_len = byte_len,
        .lkey = lkey,
        .rkey = rkey,
        .imm_host_order = imm_host_order,
    };
    return vxfec_ibv_ep_rdma_write_imm_batch(ep, qp_idx, &wr, 1u) == 1u;
}

uint32_t vxfec_ibv_ep_poll_rx(vxfec_ibv_ep_t *ep, vxfec_ibv_rx_wc_t *out_wcs, uint32_t max_wcs)
{
    if (!ep || (!ep->enable_rx && !ep->ctrl_rx_enabled) || __atomic_load_n(&ep->destroying, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE) || !ep->rx_cq || !out_wcs || max_wcs == 0u)
        return 0u;
    struct ibv_wc wcs[VXFEC_IBV_POLL_MAX];
    int cap = (max_wcs > VXFEC_IBV_POLL_MAX) ? (int)VXFEC_IBV_POLL_MAX : (int)max_wcs;
    int n = ibv_poll_cq(ep->rx_cq, cap, wcs);
    if (n < 0) {
        __atomic_store_n(&ep->tx_error, true, __ATOMIC_RELEASE);
        return 0u;
    }
    if (n == 0)
        return 0u;

    uint32_t out_n = 0u;
    for (int i = 0; i < n; i++) {
        uint32_t slot_id = (uint32_t)wcs[i].wr_id;
        bool is_ctrl = (slot_id & VXFEC_IBV_CTRL_SLOT_FLAG) != 0u;
        uint32_t q = vxfec_ibv_slot_qp(slot_id);
        uint32_t s = slot_id & 0xffffu;
        bool slot_valid = is_ctrl
                              ? (ep->ctrl_rx_enabled && q == 0u && s < ep->ctrl_rx_depth && ep->ctrl_rx_buf != NULL)
                              : (ep->enable_rx && q < ep->num_qps && s < ep->rx_per_qp && (ep->rx_imm_only || ep->rx_slab != NULL));
        if (wcs[i].status != IBV_WC_SUCCESS || !slot_valid) {
            __atomic_store_n(&ep->tx_error, true, __ATOMIC_RELEASE);
            if (q < ep->num_qps)
                __atomic_store_n(&ep->qp_tx[q].qp_err, true, __ATOMIC_RELEASE);
            continue;
        }

        vxfec_ibv_rx_wc_t *dst = &out_wcs[out_n++];
        dst->data = is_ctrl ? (ep->ctrl_rx_buf + (size_t)s * VXFEC_IBV_CTRL_RX_STRIDE)
                            : (ep->rx_imm_only ? NULL : (ep->rx_slab + ((size_t)q * ep->rx_per_qp + s) * ep->rx_stride));
        dst->byte_len = wcs[i].byte_len;
        dst->slot_id = slot_id;
        dst->is_rdma_imm = (wcs[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM);
        dst->has_imm = dst->is_rdma_imm || ((wcs[i].wc_flags & IBV_WC_WITH_IMM) != 0);
        dst->imm_data = dst->has_imm ? ntohl(wcs[i].imm_data) : 0u;
    }
    return out_n;
}

bool vxfec_ibv_ep_repost_rx_batch(vxfec_ibv_ep_t *ep, const uint32_t *slot_ids, uint32_t count)
{
    if (!ep || (!ep->enable_rx && !ep->ctrl_rx_enabled) || __atomic_load_n(&ep->destroying, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&ep->tx_error, __ATOMIC_ACQUIRE) || !slot_ids)
        return false;
    if (count == 0u)
        return true;

    uint32_t off = 0u;
    while (off < count) {
        uint32_t chunk = (count - off > VXFEC_IBV_POLL_MAX) ? VXFEC_IBV_POLL_MAX : (count - off);
        struct ibv_sge sges[VXFEC_IBV_POLL_MAX];
        struct ibv_recv_wr wrs[VXFEC_IBV_POLL_MAX];
        struct ibv_recv_wr *q_head[VXFEC_IBV_MAX_QPS] = {0};
        struct ibv_recv_wr *q_tail[VXFEC_IBV_MAX_QPS] = {0};

        for (uint32_t i = 0; i < chunk; i++) {
            uint32_t slot_id = slot_ids[off + i];
            bool is_ctrl = (slot_id & VXFEC_IBV_CTRL_SLOT_FLAG) != 0u;
            uint32_t q = vxfec_ibv_slot_qp(slot_id);
            uint32_t s = slot_id & 0xffffu;
            if (q >= ep->num_qps || !ep->qps[q] || __atomic_load_n(&ep->qp_tx[q].qp_err, __ATOMIC_ACQUIRE))
                return false;
            if (is_ctrl) {
                if (!ep->ctrl_rx_enabled || q != 0u || s >= ep->ctrl_rx_depth || !ep->tx_mr || !ep->ctrl_rx_buf)
                    return false;
                sges[i].addr = (uint64_t)(uintptr_t)(ep->ctrl_rx_buf + (size_t)s * VXFEC_IBV_CTRL_RX_STRIDE);
                sges[i].length = VXFEC_IBV_CTRL_RX_STRIDE;
                sges[i].lkey = ep->tx_mr->lkey;
                wrs[i].sg_list = &sges[i];
                wrs[i].num_sge = 1;
            } else if (ep->rx_imm_only) {
                if (!ep->enable_rx || s >= ep->rx_per_qp)
                    return false;
                wrs[i].sg_list = NULL;
                wrs[i].num_sge = 0;
            } else {
                if (!ep->enable_rx || s >= ep->rx_per_qp || !ep->rx_mr || !ep->rx_slab)
                    return false;
                size_t boff = ((size_t)q * ep->rx_per_qp + s) * ep->rx_stride;
                sges[i].addr = (uint64_t)(uintptr_t)(ep->rx_slab + boff);
                sges[i].length = ep->max_pkt_bytes;
                sges[i].lkey = ep->rx_mr->lkey;
                wrs[i].sg_list = &sges[i];
                wrs[i].num_sge = 1;
            }
            wrs[i].wr_id = slot_id;
            wrs[i].next = NULL;

            if (q_tail[q])
                q_tail[q]->next = &wrs[i];
            else
                q_head[q] = &wrs[i];
            q_tail[q] = &wrs[i];
        }

        for (uint32_t q = 0; q < ep->num_qps; q++) {
            if (!q_head[q])
                continue;
            struct ibv_recv_wr *bad = NULL;
            if (ibv_post_recv(ep->qps[q], q_head[q], &bad) != 0) {
                __atomic_store_n(&ep->tx_error, true, __ATOMIC_RELEASE);
                __atomic_store_n(&ep->qp_tx[q].qp_err, true, __ATOMIC_RELEASE);
                return false;
            }
        }
        off += chunk;
    }
    return true;
}

bool vxfec_ibv_ep_repost_rx(vxfec_ibv_ep_t *ep, uint32_t slot_id)
{
    return vxfec_ibv_ep_repost_rx_batch(ep, &slot_id, 1u);
}

#else /* !VXFEC_HAVE_IBVERBS */

bool vxfec_ibv_available(void)
{
    return false;
}
uint32_t vxfec_ibv_enumerate(vxfec_ibv_dev_info_t *out_devs, uint32_t max_devs)
{
    (void)out_devs;
    (void)max_devs;
    return 0u;
}
bool vxfec_ibv_has_dmabuf(void)
{
    return false;
}
vxfec_ibv_ctx_t *vxfec_ibv_ctx_open(uint32_t dev_idx)
{
    (void)dev_idx;
    return NULL;
}
void vxfec_ibv_ctx_close(vxfec_ibv_ctx_t *ctx)
{
    (void)ctx;
}
const vxfec_ibv_dev_info_t *vxfec_ibv_ctx_info(const vxfec_ibv_ctx_t *ctx)
{
    (void)ctx;
    return NULL;
}
vxfec_ibv_mr_t *vxfec_ibv_mr_reg_host(vxfec_ibv_ctx_t *ctx, void *addr, size_t bytes, bool remote_write)
{
    (void)ctx;
    (void)addr;
    (void)bytes;
    (void)remote_write;
    return NULL;
}
vxfec_ibv_mr_t *vxfec_ibv_mr_reg_dmabuf(vxfec_ibv_ctx_t *ctx, uint64_t offset, size_t bytes, uint64_t iova, int fd,
                                        bool remote_write)
{
    (void)ctx;
    (void)offset;
    (void)bytes;
    (void)iova;
    (void)fd;
    (void)remote_write;
    return NULL;
}
void vxfec_ibv_mr_dereg(vxfec_ibv_mr_t *mr)
{
    (void)mr;
}
uint32_t vxfec_ibv_mr_lkey(const vxfec_ibv_mr_t *mr)
{
    (void)mr;
    return 0u;
}
uint32_t vxfec_ibv_mr_rkey(const vxfec_ibv_mr_t *mr)
{
    (void)mr;
    return 0u;
}
uint64_t vxfec_ibv_mr_iova(const vxfec_ibv_mr_t *mr)
{
    (void)mr;
    return 0ULL;
}
bool vxfec_ibv_mr_is_dmabuf(const vxfec_ibv_mr_t *mr)
{
    (void)mr;
    return false;
}
vxfec_ibv_ep_t *vxfec_ibv_ep_create_duplex(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                           bool enable_rx, bool header_only_tx, bool ctrl_duplex,
                                           vxfec_ibv_wire_info_t *out_local_info)
{
    (void)ctx;
    (void)num_qps;
    (void)max_pkt_bytes;
    (void)enable_tx;
    (void)enable_rx;
    (void)header_only_tx;
    (void)ctrl_duplex;
    (void)out_local_info;
    return NULL;
}
vxfec_ibv_ep_t *vxfec_ibv_ep_create_opts(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                         bool enable_rx, bool header_only_tx, vxfec_ibv_wire_info_t *out_local_info)
{
    return vxfec_ibv_ep_create_duplex(ctx, num_qps, max_pkt_bytes, enable_tx, enable_rx, header_only_tx, false, out_local_info);
}
vxfec_ibv_ep_t *vxfec_ibv_ep_create_ex(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                       bool enable_rx, vxfec_ibv_wire_info_t *out_local_info)
{
    return vxfec_ibv_ep_create_opts(ctx, num_qps, max_pkt_bytes, enable_tx, enable_rx, false, out_local_info);
}
bool vxfec_ibv_ep_connect(vxfec_ibv_ep_t *ep, const vxfec_ibv_wire_info_t *peer_info, uint8_t traffic_class)
{
    (void)ep;
    (void)peer_info;
    (void)traffic_class;
    return false;
}
bool vxfec_ibv_ep_destroy(vxfec_ibv_ep_t *ep)
{
    (void)ep;
    return true;
}
bool vxfec_ibv_ep_has_error(const vxfec_ibv_ep_t *ep)
{
    (void)ep;
    return false;
}
bool vxfec_ibv_ep_tx_is_hdr_only(const vxfec_ibv_ep_t *ep)
{
    (void)ep;
    return false;
}
uint32_t vxfec_ibv_ep_path_mtu(const vxfec_ibv_ep_t *ep)
{
    (void)ep;
    return 0u;
}
uint32_t vxfec_ibv_ep_num_qps(const vxfec_ibv_ep_t *ep)
{
    (void)ep;
    return 0u;
}
uint64_t vxfec_ibv_ep_max_msg_bytes(const vxfec_ibv_ep_t *ep)
{
    (void)ep;
    return 0ULL;
}
uint32_t vxfec_ibv_ep_peer_max_recv(const vxfec_ibv_ep_t *ep)
{
    (void)ep;
    return 0u;
}
uint32_t vxfec_ibv_ep_tx_stage_cap(const vxfec_ibv_ep_t *ep)
{
    (void)ep;
    return 0u;
}
bool vxfec_ibv_ep_peer_has_ctrl_rx(const vxfec_ibv_ep_t *ep)
{
    (void)ep;
    return false;
}
uint32_t vxfec_ibv_ep_send_pkt_batch(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const vxfec_ibv_send_wr_desc_t *pkts, uint32_t count)
{
    (void)ep;
    (void)qp_idx;
    (void)pkts;
    (void)count;
    return 0u;
}
bool vxfec_ibv_ep_send_pkt(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *hdr, uint32_t hdr_len, const void *payload,
                           uint32_t pay_len, uint32_t pay_lkey)
{
    (void)ep;
    (void)qp_idx;
    (void)hdr;
    (void)hdr_len;
    (void)payload;
    (void)pay_len;
    (void)pay_lkey;
    return false;
}
bool vxfec_ibv_ep_send_ctrl_inline(vxfec_ibv_ep_t *ep, const void *buf, uint32_t len)
{
    (void)ep;
    (void)buf;
    (void)len;
    return false;
}
uint32_t vxfec_ibv_ep_rdma_write_imm_batch(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const vxfec_ibv_rdma_wr_t *wrs, uint32_t count)
{
    (void)ep;
    (void)qp_idx;
    (void)wrs;
    (void)count;
    return 0u;
}
bool vxfec_ibv_ep_rdma_write_imm(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *src_addr, uint32_t byte_len, uint32_t lkey,
                                 uint64_t remote_addr, uint32_t rkey, uint32_t imm_host_order)
{
    (void)ep;
    (void)qp_idx;
    (void)src_addr;
    (void)byte_len;
    (void)lkey;
    (void)remote_addr;
    (void)rkey;
    (void)imm_host_order;
    return false;
}
uint32_t vxfec_ibv_ep_poll_rx(vxfec_ibv_ep_t *ep, vxfec_ibv_rx_wc_t *out_wcs, uint32_t max_wcs)
{
    (void)ep;
    (void)out_wcs;
    (void)max_wcs;
    return 0u;
}
bool vxfec_ibv_ep_repost_rx_batch(vxfec_ibv_ep_t *ep, const uint32_t *slot_ids, uint32_t count)
{
    (void)ep;
    (void)slot_ids;
    (void)count;
    return false;
}
bool vxfec_ibv_ep_repost_rx(vxfec_ibv_ep_t *ep, uint32_t slot_id)
{
    (void)ep;
    (void)slot_id;
    return false;
}
void vxfec_ibv_ep_drain_tx(vxfec_ibv_ep_t *ep)
{
    (void)ep;
}
void vxfec_ibv_ep_record_qp_tail(const vxfec_ibv_ep_t *ep, uint32_t qp_idx, uint64_t target_seq[VXFEC_IBV_MAX_QPS])
{
    (void)ep;
    (void)qp_idx;
    (void)target_seq;
}
bool vxfec_ibv_ep_tx_done(vxfec_ibv_ep_t *ep, const uint64_t target_seq[VXFEC_IBV_MAX_QPS])
{
    (void)ep;
    (void)target_seq;
    return true;
}
bool vxfec_ibv_mr_read_to_host(vxfec_ibv_ctx_t *ctx, const vxfec_ibv_mr_t *src_mr, size_t src_off, void *dst, size_t bytes)
{
    (void)ctx;
    (void)src_mr;
    (void)src_off;
    (void)dst;
    (void)bytes;
    return false;
}
bool vxfec_ibv_mr_write_from_host(vxfec_ibv_ctx_t *ctx, vxfec_ibv_mr_t *dst_mr, size_t dst_off, const void *src, size_t bytes)
{
    (void)ctx;
    (void)dst_mr;
    (void)dst_off;
    (void)src;
    (void)bytes;
    return false;
}

#endif /* VXFEC_HAVE_IBVERBS */
