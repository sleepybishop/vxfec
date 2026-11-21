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
#define VXFEC_IBV_TX_HDR_CAP 64u
#define VXFEC_IBV_SIGNAL_MOD 32u

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
    uint32_t max_qp_wr;
    uint8_t mtu_enum;
};

struct vxfec_ibv_mr {
    struct ibv_mr *mr;
    uint64_t iova;
    bool is_dmabuf;
};

struct vxfec_ibv_ep {
    vxfec_ibv_ctx_t *ctx;
    struct ibv_cq *tx_cq;
    struct ibv_cq *rx_cq;
    struct ibv_qp *qps[VXFEC_IBV_MAX_QPS];
    uint32_t local_psn[VXFEC_IBV_MAX_QPS];
    uint32_t num_qps;
    uint32_t rx_per_qp;
    uint32_t rx_stride;
    uint32_t tx_stride;
    uint32_t max_pkt_bytes;
    uint32_t path_mtu_bytes;
    bool enable_tx;
    bool enable_rx;
    bool tx_hdr_only;
    bool tx_error;
    bool destroying;
    bool qp_err[VXFEC_IBV_MAX_QPS];

    uint8_t *rx_slab;
    size_t rx_slab_bytes;
    struct ibv_mr *rx_mr;

    uint8_t *tx_slab;
    size_t tx_slab_bytes;
    struct ibv_mr *tx_mr;
    uint32_t tx_unsignaled[VXFEC_IBV_MAX_QPS];
    uint64_t sq_posted[VXFEC_IBV_MAX_QPS];
    uint64_t sq_last_sig[VXFEC_IBV_MAX_QPS];
    uint64_t sq_done[VXFEC_IBV_MAX_QPS];
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
            if (!colon)
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

uint32_t vxfec_ibv_enumerate(vxfec_ibv_dev_info_t *out_devs, uint32_t max_devs)
{
    pthread_once(&g_ibv_once, init_ibv_once);
    if (!g_ibv.ready || (!out_devs && max_devs > 0u))
        return 0u;

    int num_raw = 0;
    struct ibv_device **list = g_ibv.get_device_list(&num_raw);
    if (!list || num_raw <= 0) {
        if (list)
            g_ibv.free_device_list(list);
        return 0u;
    }

    const char *filter = getenv("VXFEC_IBV_DEVS");
    uint32_t count = 0u;

    for (int i = 0; i < num_raw && count < max_devs; i++) {
        const char *dname = g_ibv.get_device_name(list[i]);
        if (!dname || !*dname)
            continue;
        if (filter && *filter && !ibv_dev_matches_filter(filter, dname, 1u) && !ibv_dev_matches_filter(filter, dname, 2u) &&
            !ibv_dev_matches_filter(filter, dname, 3u) && !ibv_dev_matches_filter(filter, dname, 4u))
            continue;

        struct ibv_context *ctx = g_ibv.open_device(list[i]);
        if (!ctx)
            continue;

        uint8_t max_port = 2u;
        struct ibv_device_attr dattr;
        memset(&dattr, 0, sizeof(dattr));
        if (g_ibv.query_device && g_ibv.query_device(ctx, &dattr) == 0 && dattr.phys_port_cnt > 0u)
            max_port = dattr.phys_port_cnt;

        for (uint8_t p = 1u; p <= max_port && count < max_devs; p++) {
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
            info->guid = g_ibv.get_device_guid ? ((uint64_t)g_ibv.get_device_guid(list[i]) ^ ((uint64_t)(p - 1u) << 56u))
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

    g_ibv.free_device_list(list);
    return count;
}

vxfec_ibv_ctx_t *vxfec_ibv_ctx_open(uint32_t dev_idx)
{
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
    struct ibv_device_attr dattr;
    memset(&dattr, 0, sizeof(dattr));
    if (g_ibv.query_device && g_ibv.query_device(opened, &dattr) == 0 && dattr.max_qp_wr > 0)
        ctx->max_qp_wr = (uint32_t)dattr.max_qp_wr;
    return ctx;
}

void vxfec_ibv_ctx_close(vxfec_ibv_ctx_t *ctx)
{
    if (!ctx)
        return;
    if (ctx->pd)
        g_ibv.dealloc_pd(ctx->pd);
    if (ctx->ibv)
        g_ibv.close_device(ctx->ibv);
    free(ctx);
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
    mr->is_dmabuf = false;
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
    mr->is_dmabuf = true;
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

vxfec_ibv_ep_t *vxfec_ibv_ep_create_opts(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                         bool enable_rx, bool header_only_tx, vxfec_ibv_wire_info_t *out_local_info)
{
    if (!ctx || !out_local_info || num_qps == 0u || max_pkt_bytes == 0u || (!enable_tx && !enable_rx))
        return NULL;
    if (num_qps > VXFEC_IBV_MAX_QPS)
        num_qps = VXFEC_IBV_MAX_QPS;

    vxfec_ibv_ep_t *ep = (vxfec_ibv_ep_t *)calloc(1, sizeof(*ep));
    if (!ep)
        return NULL;
    ep->ctx = ctx;
    ep->num_qps = num_qps;
    ep->rx_per_qp = enable_rx ? select_rx_per_qp(ctx) : 1u;
    ep->max_pkt_bytes = max_pkt_bytes;
    ep->path_mtu_bytes = mtu_enum_to_bytes((enum ibv_mtu)ctx->mtu_enum);
    ep->rx_stride = (max_pkt_bytes + 63u) & ~63u;
    ep->enable_tx = enable_tx;
    ep->enable_rx = enable_rx;

    int rx_cqe = enable_rx ? (int)(num_qps * ep->rx_per_qp) : 16;
    int tx_cqe = enable_tx ? (int)(num_qps * VXFEC_IBV_TX_PER_QP) : 16;
    ep->rx_cq = g_ibv.create_cq(ctx->ibv, rx_cqe, NULL, NULL, 0);
    ep->tx_cq = g_ibv.create_cq(ctx->ibv, tx_cqe, NULL, NULL, 0);
    if (!ep->rx_cq || !ep->tx_cq) {
        vxfec_ibv_ep_destroy(ep);
        return NULL;
    }

    if (enable_rx) {
        ep->rx_slab_bytes = ((size_t)num_qps * ep->rx_per_qp * ep->rx_stride + 4095u) & ~(size_t)4095u;
        if (posix_memalign((void **)&ep->rx_slab, 4096u, ep->rx_slab_bytes) != 0) {
            vxfec_ibv_ep_destroy(ep);
            return NULL;
        }
        memset(ep->rx_slab, 0, ep->rx_slab_bytes);
        ep->rx_mr = reg_host_raw(ctx->pd, ep->rx_slab, ep->rx_slab_bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!ep->rx_mr) {
            vxfec_ibv_ep_destroy(ep);
            return NULL;
        }
    }

    if (enable_tx) {
        if (!header_only_tx) {
            ep->tx_stride = ep->rx_stride;
            ep->tx_slab_bytes = ((size_t)num_qps * VXFEC_IBV_TX_PER_QP * ep->tx_stride + 4095u) & ~(size_t)4095u;
            if (posix_memalign((void **)&ep->tx_slab, 4096u, ep->tx_slab_bytes) == 0) {
                memset(ep->tx_slab, 0, ep->tx_slab_bytes);
                ep->tx_mr = reg_host_raw(ctx->pd, ep->tx_slab, ep->tx_slab_bytes, IBV_ACCESS_LOCAL_WRITE);
            }
        }
        if (!ep->tx_mr) {
            /* compact 64B/slot header-only MR for zero-copy 2-SGE sends (65x smaller pinned slab) */
            free(ep->tx_slab);
            ep->tx_slab = NULL;
            ep->tx_stride = VXFEC_IBV_TX_HDR_CAP;
            ep->tx_hdr_only = true;
            ep->tx_slab_bytes = ((size_t)num_qps * VXFEC_IBV_TX_PER_QP * ep->tx_stride + 4095u) & ~(size_t)4095u;
            if (posix_memalign((void **)&ep->tx_slab, 4096u, ep->tx_slab_bytes) != 0) {
                vxfec_ibv_ep_destroy(ep);
                return NULL;
            }
            memset(ep->tx_slab, 0, ep->tx_slab_bytes);
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
    generate_qp_psns(ep->local_psn, num_qps);

    for (uint32_t q = 0; q < num_qps; q++) {
        struct ibv_qp_init_attr qia;
        memset(&qia, 0, sizeof(qia));
        qia.send_cq = ep->tx_cq;
        qia.recv_cq = ep->rx_cq;
        qia.qp_type = IBV_QPT_UC;
        qia.sq_sig_all = 0;
        qia.cap.max_send_wr = enable_tx ? VXFEC_IBV_TX_PER_QP : 1u;
        qia.cap.max_recv_wr = enable_rx ? ep->rx_per_qp : 1u;
        qia.cap.max_send_sge = enable_tx ? 2u : 1u;
        qia.cap.max_recv_sge = 1u;
        qia.cap.max_inline_data = enable_tx ? VXFEC_IBV_TX_HDR_CAP : 0u;

        ep->qps[q] = g_ibv.create_qp(ctx->pd, &qia);
        if (!ep->qps[q]) {
            vxfec_ibv_ep_destroy(ep);
            return NULL;
        }

        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = ctx->info.port_num;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        int init_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        if (g_ibv.modify_qp(ep->qps[q], &attr, init_mask) != 0) {
            vxfec_ibv_ep_destroy(ep);
            return NULL;
        }

        out_local_info->qpn[q] = ep->qps[q]->qp_num;
        out_local_info->psn[q] = ep->local_psn[q];

        if (enable_rx) {
            /* pre-post RX ring slots for this QP */
            for (uint32_t s = 0; s < ep->rx_per_qp; s++) {
                uint32_t slot_id = (q << 16u) | s;
                if (!vxfec_ibv_ep_repost_rx(ep, slot_id)) {
                    vxfec_ibv_ep_destroy(ep);
                    return NULL;
                }
            }
        }
    }

    return ep;
}

vxfec_ibv_ep_t *vxfec_ibv_ep_create_ex(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                       bool enable_rx, vxfec_ibv_wire_info_t *out_local_info)
{
    return vxfec_ibv_ep_create_opts(ctx, num_qps, max_pkt_bytes, enable_tx, enable_rx, false, out_local_info);
}

vxfec_ibv_ep_t *vxfec_ibv_ep_create(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes,
                                    vxfec_ibv_wire_info_t *out_local_info)
{
    return vxfec_ibv_ep_create_opts(ctx, num_qps, max_pkt_bytes, true, true, false, out_local_info);
}

bool vxfec_ibv_ep_connect(vxfec_ibv_ep_t *ep, const vxfec_ibv_wire_info_t *peer_info, uint8_t traffic_class)
{
    if (!ep || !peer_info || peer_info->num_qps != ep->num_qps)
        return false;

    uint8_t path_mtu = (peer_info->mtu_enum < ep->ctx->mtu_enum) ? peer_info->mtu_enum : ep->ctx->mtu_enum;
    if (path_mtu < IBV_MTU_256 || path_mtu > IBV_MTU_4096)
        path_mtu = IBV_MTU_1024;
    ep->path_mtu_bytes = mtu_enum_to_bytes((enum ibv_mtu)path_mtu);

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

    ep->destroying = true;
    ep->tx_error = true;
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
        ep->tx_error = true;
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
    }
    if (!ep->tx_mr) {
        free(ep->tx_slab);
        ep->tx_slab = NULL;
    }
    if (ep->rx_cq && g_ibv.destroy_cq(ep->rx_cq) == 0)
        ep->rx_cq = NULL;
    if (ep->tx_cq && g_ibv.destroy_cq(ep->tx_cq) == 0)
        ep->tx_cq = NULL;
    if (mr_busy || ep->rx_cq || ep->tx_cq)
        return false;

    free(ep);
    return true;
}

bool vxfec_ibv_ep_has_error(const vxfec_ibv_ep_t *ep)
{
    return ep ? ep->tx_error : false;
}

bool vxfec_ibv_ep_tx_is_hdr_only(const vxfec_ibv_ep_t *ep)
{
    return ep ? ep->tx_hdr_only : false;
}

uint32_t vxfec_ibv_ep_path_mtu(const vxfec_ibv_ep_t *ep)
{
    return ep ? ep->path_mtu_bytes : 0u;
}

void vxfec_ibv_ep_drain_tx(vxfec_ibv_ep_t *ep)
{
    if (!ep || !ep->tx_cq || !ep->enable_tx)
        return;
    struct ibv_wc wcs[32];
    for (;;) {
        int n = ibv_poll_cq(ep->tx_cq, 32, wcs);
        if (n < 0) {
            ep->tx_error = true;
            break;
        }
        if (n == 0)
            break;
        for (int i = 0; i < n; i++) {
            uint32_t q = (uint32_t)(wcs[i].wr_id >> 56u);
            uint64_t seq = wcs[i].wr_id & 0x00ffffffffffffffULL;
            if (wcs[i].status != IBV_WC_SUCCESS) {
                ep->tx_error = true;
                if (q < ep->num_qps)
                    ep->qp_err[q] = true;
            }
            if (q < ep->num_qps && seq > ep->sq_done[q])
                ep->sq_done[q] = seq;
        }
    }
}

void vxfec_ibv_ep_record_qp_tail(const vxfec_ibv_ep_t *ep, uint32_t qp_idx, uint64_t target_seq[VXFEC_IBV_MAX_QPS])
{
    if (!ep || !target_seq || ep->num_qps == 0u)
        return;
    uint32_t q = qp_idx % ep->num_qps;
    if (ep->sq_posted[q] > target_seq[q])
        target_seq[q] = ep->sq_posted[q];
}

bool vxfec_ibv_ep_tx_done(vxfec_ibv_ep_t *ep, const uint64_t target_seq[VXFEC_IBV_MAX_QPS])
{
    if (!ep || !target_seq || !ep->enable_tx)
        return true;
    if (ep->destroying || ep->tx_error)
        return false;
    vxfec_ibv_ep_drain_tx(ep);
    if (ep->tx_error)
        return false;

    bool posted_fence = false;
    for (uint32_t q = 0; q < ep->num_qps; q++) {
        if (ep->sq_done[q] >= target_seq[q])
            continue;
        if (!ep->qps[q])
            return false;
        if (!ep->qp_err[q] && ep->sq_last_sig[q] < target_seq[q] && (ep->sq_posted[q] - ep->sq_done[q]) < VXFEC_IBV_TX_PER_QP) {
            uint64_t next_seq = ep->sq_posted[q] + 1ULL;
            struct ibv_send_wr wr, *bad = NULL;
            memset(&wr, 0, sizeof(wr));
            wr.wr_id = ((uint64_t)q << 56u) | (next_seq & 0x00ffffffffffffffULL);
            /* 0-byte RDMA_WRITE fences prior SQ WQEs without consuming any remote UC Receive WQE */
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.send_flags = IBV_SEND_SIGNALED;
            if (ibv_post_send(ep->qps[q], &wr, &bad) == 0) {
                ep->sq_posted[q] = next_seq;
                ep->sq_last_sig[q] = next_seq;
                ep->tx_unsignaled[q] = 0u;
                posted_fence = true;
            } else {
                ep->tx_error = true;
                ep->qp_err[q] = true;
            }
        }
    }
    if (posted_fence)
        vxfec_ibv_ep_drain_tx(ep);
    if (ep->tx_error)
        return false;

    for (uint32_t q = 0; q < ep->num_qps; q++) {
        if (ep->sq_done[q] < target_seq[q])
            return false;
    }
    return true;
}

bool vxfec_ibv_ep_send_pkt_ex(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *hdr, uint32_t hdr_len, const void *payload,
                              uint32_t pay_len, uint32_t pay_lkey, bool force_signal)
{
    if (!ep || !ep->enable_tx || ep->destroying || ep->tx_error || !hdr || hdr_len == 0u || ep->num_qps == 0u)
        return false;
    const bool has_pay = (pay_len > 0u && payload != NULL);
    const bool stage_pay = has_pay && (pay_lkey == 0u);
    uint32_t pkt_cap = (ep->path_mtu_bytes && ep->path_mtu_bytes < ep->max_pkt_bytes) ? ep->path_mtu_bytes : ep->max_pkt_bytes;
    if (hdr_len > pkt_cap || (has_pay && pay_len > pkt_cap - hdr_len))
        return false;
    if (stage_pay) {
        if (ep->tx_hdr_only)
            return false;
    } else if (hdr_len > VXFEC_IBV_TX_HDR_CAP) {
        return false;
    }

    uint32_t q = qp_idx % ep->num_qps;
    if (!ep->qps[q] || !ep->tx_mr)
        return false;
    if ((ep->sq_posted[q] - ep->sq_done[q]) >= (VXFEC_IBV_TX_PER_QP - VXFEC_IBV_SIGNAL_MOD))
        vxfec_ibv_ep_drain_tx(ep);
    if (ep->tx_error)
        return false;
    /* reserve 1 SQ slot per QP so vxfec_ibv_ep_tx_done can always post a signaled fence */
    if ((ep->sq_posted[q] - ep->sq_done[q]) >= (VXFEC_IBV_TX_PER_QP - 1u))
        return false;

    uint32_t slot = (uint32_t)(ep->sq_posted[q] & (VXFEC_IBV_TX_PER_QP - 1u));
    uint8_t *tdst = ep->tx_slab + ((size_t)q * VXFEC_IBV_TX_PER_QP + slot) * ep->tx_stride;
    memcpy(tdst, hdr, hdr_len);

    struct ibv_sge sges[2];
    sges[0].addr = (uint64_t)(uintptr_t)tdst;
    sges[0].length = hdr_len;
    sges[0].lkey = ep->tx_mr->lkey;

    int num_sge = 1;
    if (stage_pay) {
        memcpy(tdst + hdr_len, payload, pay_len);
        sges[0].length = hdr_len + pay_len;
    } else if (has_pay) {
        sges[1].addr = (uint64_t)(uintptr_t)payload;
        sges[1].length = pay_len;
        sges[1].lkey = pay_lkey;
        num_sge = 2;
    }

    uint64_t next_seq = ep->sq_posted[q] + 1ULL;
    ep->tx_unsignaled[q]++;
    bool sig = force_signal || (ep->tx_unsignaled[q] >= VXFEC_IBV_SIGNAL_MOD);

    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = ((uint64_t)q << 56u) | (next_seq & 0x00ffffffffffffffULL);
    wr.sg_list = sges;
    wr.num_sge = num_sge;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = sig ? IBV_SEND_SIGNALED : 0;
    if (num_sge == 1 && sges[0].length <= VXFEC_IBV_TX_HDR_CAP)
        wr.send_flags |= IBV_SEND_INLINE;

    if (ibv_post_send(ep->qps[q], &wr, &bad) != 0) {
        ep->tx_unsignaled[q]--;
        ep->tx_error = true;
        ep->qp_err[q] = true;
        return false;
    }
    ep->sq_posted[q] = next_seq;
    if (sig) {
        ep->sq_last_sig[q] = next_seq;
        ep->tx_unsignaled[q] = 0u;
    }
    return true;
}

bool vxfec_ibv_ep_send_pkt(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *hdr, uint32_t hdr_len, const void *payload,
                           uint32_t pay_len, uint32_t pay_lkey)
{
    return vxfec_ibv_ep_send_pkt_ex(ep, qp_idx, hdr, hdr_len, payload, pay_len, pay_lkey, false);
}

bool vxfec_ibv_ep_rdma_write_imm(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *src_addr, uint32_t byte_len, uint32_t lkey,
                                 uint64_t remote_addr, uint32_t rkey, uint32_t imm_host_order)
{
    if (!ep || !ep->enable_tx || ep->destroying || ep->tx_error || ep->num_qps == 0u || (byte_len > 0u && !src_addr))
        return false;
    if (ep->path_mtu_bytes > 0u && byte_len > ep->path_mtu_bytes)
        return false;
    uint32_t q = qp_idx % ep->num_qps;
    if (!ep->qps[q])
        return false;
    if ((ep->sq_posted[q] - ep->sq_done[q]) >= (VXFEC_IBV_TX_PER_QP - VXFEC_IBV_SIGNAL_MOD))
        vxfec_ibv_ep_drain_tx(ep);
    if (ep->tx_error || (ep->sq_posted[q] - ep->sq_done[q]) >= (VXFEC_IBV_TX_PER_QP - 1u))
        return false;

    struct ibv_sge sge;
    sge.addr = (uint64_t)(uintptr_t)src_addr;
    sge.length = byte_len;
    sge.lkey = lkey;

    uint64_t next_seq = ep->sq_posted[q] + 1ULL;
    ep->tx_unsignaled[q]++;
    bool sig = (ep->tx_unsignaled[q] >= VXFEC_IBV_SIGNAL_MOD);

    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = ((uint64_t)q << 56u) | (next_seq & 0x00ffffffffffffffULL);
    wr.sg_list = byte_len ? &sge : NULL;
    wr.num_sge = byte_len ? 1 : 0;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = sig ? IBV_SEND_SIGNALED : 0;
    wr.imm_data = htonl(imm_host_order);
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;

    if (ibv_post_send(ep->qps[q], &wr, &bad) != 0) {
        ep->tx_unsignaled[q]--;
        ep->tx_error = true;
        ep->qp_err[q] = true;
        return false;
    }
    ep->sq_posted[q] = next_seq;
    if (sig) {
        ep->sq_last_sig[q] = next_seq;
        ep->tx_unsignaled[q] = 0u;
    }
    return true;
}

uint32_t vxfec_ibv_ep_poll_rx(vxfec_ibv_ep_t *ep, vxfec_ibv_rx_wc_t *out_wcs, uint32_t max_wcs)
{
    if (!ep || !ep->enable_rx || ep->destroying || ep->tx_error || !ep->rx_cq || !out_wcs || max_wcs == 0u)
        return 0u;
    struct ibv_wc wcs[64];
    int cap = (max_wcs > 64u) ? 64 : (int)max_wcs;
    int n = ibv_poll_cq(ep->rx_cq, cap, wcs);
    if (n < 0) {
        ep->tx_error = true;
        return 0u;
    }
    if (n == 0)
        return 0u;

    uint32_t out_n = 0u;
    for (int i = 0; i < n; i++) {
        uint32_t slot_id = (uint32_t)wcs[i].wr_id;
        uint32_t q = (slot_id >> 16u) & 0xffffu;
        uint32_t s = slot_id & 0xffffu;
        if (wcs[i].status != IBV_WC_SUCCESS || q >= ep->num_qps || s >= ep->rx_per_qp) {
            ep->tx_error = true;
            if (q < ep->num_qps)
                ep->qp_err[q] = true;
            continue;
        }
        size_t off = ((size_t)q * ep->rx_per_qp + s) * ep->rx_stride;

        vxfec_ibv_rx_wc_t *dst = &out_wcs[out_n++];
        dst->data = ep->rx_slab + off;
        dst->byte_len = wcs[i].byte_len;
        dst->slot_id = slot_id;
        dst->is_rdma_imm = (wcs[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM);
        dst->has_imm = dst->is_rdma_imm || ((wcs[i].wc_flags & IBV_WC_WITH_IMM) != 0);
        dst->imm_data = dst->has_imm ? ntohl(wcs[i].imm_data) : 0u;
    }
    return out_n;
}

bool vxfec_ibv_ep_repost_rx(vxfec_ibv_ep_t *ep, uint32_t slot_id)
{
    if (!ep || !ep->enable_rx || ep->destroying || ep->tx_error || !ep->rx_mr)
        return false;
    uint32_t q = (slot_id >> 16u) & 0xffffu;
    uint32_t s = slot_id & 0xffffu;
    if (q >= ep->num_qps || s >= ep->rx_per_qp || !ep->qps[q] || ep->qp_err[q])
        return false;

    size_t off = ((size_t)q * ep->rx_per_qp + s) * ep->rx_stride;
    struct ibv_sge sge;
    sge.addr = (uint64_t)(uintptr_t)(ep->rx_slab + off);
    sge.length = ep->max_pkt_bytes;
    sge.lkey = ep->rx_mr->lkey;

    struct ibv_recv_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = slot_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    if (ibv_post_recv(ep->qps[q], &wr, &bad) != 0) {
        ep->tx_error = true;
        ep->qp_err[q] = true;
        return false;
    }
    return true;
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
vxfec_ibv_ep_t *vxfec_ibv_ep_create_opts(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                         bool enable_rx, bool header_only_tx, vxfec_ibv_wire_info_t *out_local_info)
{
    (void)ctx;
    (void)num_qps;
    (void)max_pkt_bytes;
    (void)enable_tx;
    (void)enable_rx;
    (void)header_only_tx;
    (void)out_local_info;
    return NULL;
}
vxfec_ibv_ep_t *vxfec_ibv_ep_create_ex(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes, bool enable_tx,
                                       bool enable_rx, vxfec_ibv_wire_info_t *out_local_info)
{
    return vxfec_ibv_ep_create_opts(ctx, num_qps, max_pkt_bytes, enable_tx, enable_rx, false, out_local_info);
}
vxfec_ibv_ep_t *vxfec_ibv_ep_create(vxfec_ibv_ctx_t *ctx, uint32_t num_qps, uint32_t max_pkt_bytes,
                                    vxfec_ibv_wire_info_t *out_local_info)
{
    return vxfec_ibv_ep_create_opts(ctx, num_qps, max_pkt_bytes, true, true, false, out_local_info);
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
bool vxfec_ibv_ep_send_pkt_ex(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *hdr, uint32_t hdr_len, const void *payload,
                              uint32_t pay_len, uint32_t pay_lkey, bool force_signal)
{
    (void)ep;
    (void)qp_idx;
    (void)hdr;
    (void)hdr_len;
    (void)payload;
    (void)pay_len;
    (void)pay_lkey;
    (void)force_signal;
    return false;
}
bool vxfec_ibv_ep_send_pkt(vxfec_ibv_ep_t *ep, uint32_t qp_idx, const void *hdr, uint32_t hdr_len, const void *payload,
                           uint32_t pay_len, uint32_t pay_lkey)
{
    return vxfec_ibv_ep_send_pkt_ex(ep, qp_idx, hdr, hdr_len, payload, pay_len, pay_lkey, false);
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

#endif /* VXFEC_HAVE_IBVERBS */
