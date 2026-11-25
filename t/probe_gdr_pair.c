#include "ibv_transport.h"
#include "vxfec_vk.h"
#include "bench_common.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define MAX_RAILS VXFEC_IBV_MAX_DEVS

typedef struct {
    uint8_t uuid[VK_UUID_SIZE];
    uint32_t vk_index;
    vxfec_vk_ctx *vk;
    int cuda_device;
    char bdf[32];
    char pci_path[PATH_MAX];
} gpu_t;

typedef struct {
    uint32_t dev, gpu, hca, mtu;
    int depth;
    uint64_t speed;
} pair_t;

typedef struct {
    vxfec_ibv_dev_info_t devs[MAX_RAILS];
    gpu_t gpus[MAX_RAILS];
    uint32_t ndev, ngpu;
    pair_t pairs[MAX_RAILS * MAX_RAILS];
    size_t npairs;
    const pair_t *assigned[MAX_RAILS]; /* Indexed by physical HCA, even on multi-port devices. */
} topology_t;

static int shared_path_components(const char *a, const char *b)
{
    int comps = 0;
    size_t i = 0;
    while (a[i] && b[i] && a[i] == b[i]) {
        if (a[i] == '/')
            comps++;
        i++;
    }
    if ((a[i] == '\0' || a[i] == '/') && (b[i] == '\0' || b[i] == '/'))
        return comps + 1;
    return comps;
}

static bool map_vulkan_gpus(topology_t *t, bool require_all)
{
    const vxfec_bench_cu_syms_t *cu = vxfec_bench_cu_syms();
    for (uint32_t g = 0; g < t->ngpu; ++g) {
        gpu_t *gpu = &t->gpus[g];
        if (cu->dev_get_uuid(gpu->uuid, gpu->cuda_device) != 0) {
            fprintf(stderr, "probe: cannot read device UUID for CUDA GPU %u\n", g);
            if (require_all)
                return false;
            gpu->cuda_device = -1;
        }
    }

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    uint32_t nvk = 0;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS)
        return false;
    VkResult result = vkEnumeratePhysicalDevices(inst, &nvk, NULL);
    vkDestroyInstance(inst, NULL);
    if (result != VK_SUCCESS)
        return false;

    uint32_t mapped = 0;
    for (uint32_t v = 0; v < nvk && mapped < t->ngpu; ++v) {
        vxfec_vk_ctx *vk = vxfec_vk_ctx_new_at(v, NULL);
        uint8_t uuid[VK_UUID_SIZE];
        if (!vk)
            continue;
        char vk_bdf[sizeof(t->gpus[0].bdf)];
        if (vxfec_vk_device_uuid(vk, uuid) && vxfec_vk_device_pci_bus_info(vk, vk_bdf, sizeof(vk_bdf))) {
            for (uint32_t g = 0; g < t->ngpu; ++g) {
                gpu_t *gpu = &t->gpus[g];
                if (gpu->cuda_device < 0 || gpu->vk || strcmp(vk_bdf, gpu->bdf) || memcmp(uuid, gpu->uuid, sizeof(uuid)) != 0)
                    continue;
                for (uint32_t p = 0; p < t->ngpu; ++p) {
                    if (t->gpus[p].vk && strcmp(t->gpus[p].bdf, gpu->bdf) == 0) {
                        fprintf(stderr, "probe: CUDA GPUs %u and %u share PCI address %s\n", p, g, gpu->bdf);
                        vxfec_vk_ctx_free(vk);
                        return false;
                    }
                }
                gpu->vk_index = v;
                gpu->vk = vk;
                vk = NULL;
                mapped++;
                break;
            }
        }
        if (vk)
            vxfec_vk_ctx_free(vk);
    }
    if (mapped != t->ngpu)
        fprintf(stderr, "probe: mapped %u of %u CUDA-visible GPUs to Vulkan devices with PCI paths\n", mapped, t->ngpu);
    return require_all ? mapped == t->ngpu : mapped > 0;
}

static bool discover_cuda_gpus(topology_t *t, bool require_all)
{
    const vxfec_bench_cu_syms_t *cu = vxfec_bench_cu_syms();
    int (*get_pci)(char *, int, int) = cu ? dlsym(cu->dl, "cuDeviceGetPCIBusId") : NULL;
    int count = 0;
    if (!get_pci || cu->dev_get_count(&count) != 0)
        return false;
    for (int i = 0; i < count && t->ngpu < MAX_RAILS; ++i) {
        gpu_t *gpu = &t->gpus[t->ngpu];
        char raw_bdf[32], path[64];
        unsigned domain, bus, device, function;
        if (cu->dev_get(&gpu->cuda_device, i) != 0 || get_pci(raw_bdf, sizeof(raw_bdf), gpu->cuda_device) != 0 ||
            sscanf(raw_bdf, "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4 || domain > 0xffff || bus > 0xff ||
            device > 0x1f || function > 7)
            continue;
        snprintf(gpu->bdf, sizeof(gpu->bdf), "%04x:%02x:%02x.%x", domain, bus, device, function);
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s", gpu->bdf);
        if (realpath(path, gpu->pci_path))
            t->ngpu++;
    }
    return t->ngpu > 0 && (!require_all || (int)t->ngpu == count);
}

static vxfec_ibv_ctx_t *open_checked_hca(const topology_t *t, uint32_t d)
{
    vxfec_ibv_ctx_t *ib = vxfec_ibv_ctx_open(d);
    if (!ib)
        return NULL;
    const vxfec_ibv_dev_info_t *opened = vxfec_ibv_ctx_info(ib);
    if (!opened || strcmp(opened->dev_name, t->devs[d].dev_name) || opened->port_num != t->devs[d].port_num) {
        fprintf(stderr, "probe: IB device list changed while opening %s:%u\n", t->devs[d].dev_name, t->devs[d].port_num);
        vxfec_ibv_ctx_close(ib);
        return NULL;
    }
    return ib;
}

/* Exercise the CUDA device pointer -> DMA-BUF -> writable verbs MR path that NCCL uses. */
static bool register_cuda_pair(const topology_t *t, uint32_t d, uint32_t g)
{
    const vxfec_bench_cu_syms_t *cu = vxfec_bench_cu_syms();
    int (*alloc)(unsigned long long *, size_t) = cu ? dlsym(cu->dl, "cuMemAlloc_v2") : NULL;
    int (*release)(unsigned long long) = cu ? dlsym(cu->dl, "cuMemFree_v2") : NULL;
    long page = sysconf(_SC_PAGESIZE);
    if (!alloc || !release || page <= 0)
        return false;
    vxfec_ibv_ctx_t *ib = open_checked_hca(t, d);
    if (!ib)
        return false;
    const gpu_t *gpu = &t->gpus[g];
    const size_t size = 4u * 1024u * 1024u;
    void *ctx = NULL, *popped = NULL;
    unsigned long long base = 0;
    int fd = -1;
    vxfec_ibv_mr_t *mr = NULL;
    bool retained = cu->pctx_retain(&ctx, gpu->cuda_device) == 0;
    bool pushed = retained && cu->ctx_push(ctx) == 0;
    size_t align = ((size_t)page < 65536u) ? 65536u : (size_t)page;
    if (pushed && alloc(&base, size + align) == 0) {
        unsigned long long start = (base + align - 1u) & ~((unsigned long long)align - 1u);
        if (cu->mem_get_range_handle(&fd, start, size, 1 /* DMA_BUF_FD */, 0) == 0 && fd >= 0)
            mr = vxfec_ibv_mr_reg_dmabuf(ib, 0u, size, start, fd, true);
    }
    bool ok = mr != NULL;
    if (mr)
        vxfec_ibv_mr_dereg(mr);
    else
        fprintf(stderr, "probe: native CUDA DMA-BUF registration failed for %s:%u and %s\n", t->devs[d].dev_name,
                t->devs[d].port_num, gpu->bdf);
    if (fd >= 0)
        close(fd);
    if (base)
        (void)release(base);
    if (pushed)
        (void)cu->ctx_pop(&popped);
    if (retained)
        (void)cu->pctx_release(gpu->cuda_device);
    vxfec_ibv_ctx_close(ib);
    return ok;
}

static bool register_pair(const topology_t *t, uint32_t d, uint32_t g)
{
    const gpu_t *gpu = &t->gpus[g];
    const size_t size = 4u * 1024u * 1024u;
    vxfec_ibv_ctx_t *ib = open_checked_hca(t, d);
    if (!ib)
        return false;
    int fd = -1;
    vxfec_vk_buf *buf = vxfec_bench_alloc_gdr_buf(gpu->vk, size, &fd);
    long page = sysconf(_SC_PAGESIZE);
    uint64_t iova = (page > 0) ? (uint64_t)page : 65536ULL;
    vxfec_ibv_mr_t *mr = NULL;
    if (buf && fd >= 0)
        mr = vxfec_ibv_mr_reg_dmabuf(ib, 0, size, iova, fd, true);
    bool ok = mr != NULL;
    if (mr)
        vxfec_ibv_mr_dereg(mr);
    else
        fprintf(stderr, "probe: %s:%u vk=%u pci=%s %s failed (%s)\n", t->devs[d].dev_name, t->devs[d].port_num, gpu->vk_index,
                gpu->bdf, buf && fd >= 0 ? "DMA-BUF MR registration" : "buffer export", strerror(errno));
    if (fd >= 0)
        close(fd);
    if (buf)
        vxfec_vk_buf_free(gpu->vk, buf);
    vxfec_ibv_ctx_close(ib);
    return ok;
}

static int cmp_pair(const void *va, const void *vb)
{
    const pair_t *a = va, *b = vb;
    if (a->mtu != b->mtu)
        return (a->mtu < b->mtu) ? 1 : -1;
    if (a->depth != b->depth)
        return b->depth - a->depth;
    if (a->speed != b->speed)
        return (a->speed < b->speed) ? 1 : -1;
    if (a->dev != b->dev)
        return (a->dev < b->dev) ? -1 : 1;
    return (a->gpu > b->gpu) - (a->gpu < b->gpu);
}

static bool assign_gpu(topology_t *t, uint32_t gpu, uint32_t mtu, bool seen[MAX_RAILS])
{
    for (size_t i = 0; i < t->npairs; ++i) {
        const pair_t *p = &t->pairs[i];
        if (p->gpu != gpu || p->mtu != mtu || seen[p->hca])
            continue;
        seen[p->hca] = true;
        const pair_t *old = t->assigned[p->hca];
        if (!old || assign_gpu(t, old->gpu, mtu, seen)) {
            t->assigned[p->hca] = p;
            return true;
        }
    }
    return false;
}

static void rank_nearest_pairs(topology_t *t)
{
    t->npairs = 0;
    for (uint32_t d = 0; d < t->ndev; ++d) {
        const vxfec_ibv_dev_info_t *dev = &t->devs[d];
        if (dev->is_roce || !dev->pci_path[0] || !dev->active_mtu)
            continue;
        uint32_t hca = 0;
        while (hca < d && strcmp(t->devs[hca].dev_name, dev->dev_name))
            ++hca;
        int depth[MAX_RAILS], nearest = 0;
        for (uint32_t g = 0; g < t->ngpu; ++g) {
            depth[g] = t->gpus[g].pci_path[0] ? shared_path_components(dev->pci_path, t->gpus[g].pci_path) : -1;
            if (depth[g] > nearest)
                nearest = depth[g];
        }
        for (uint32_t g = 0; g < t->ngpu; ++g) {
            if (depth[g] == nearest)
                t->pairs[t->npairs++] = (pair_t){d, g, hca, dev->active_mtu, depth[g], dev->speed_mbps};
        }
    }
    qsort(t->pairs, t->npairs, sizeof(t->pairs[0]), cmp_pair);
}

typedef bool (*register_mr_fn)(const topology_t *, uint32_t, uint32_t);

static const pair_t *select_single_pair(topology_t *t, register_mr_fn cuda_mr, register_mr_fn wire_mr)
{
    rank_nearest_pairs(t);
    for (size_t i = 0; i < t->npairs; ++i) {
        const pair_t *p = &t->pairs[i];
        /* The one-way throughput experiment needs a 400G rail and both allocations on that rail. */
        if (wire_mr && (p->speed < 400000 || !t->gpus[p->gpu].vk))
            continue;
        if (cuda_mr(t, p->dev, p->gpu) && (!wire_mr || wire_mr(t, p->dev, p->gpu)))
            return p;
    }
    return NULL;
}

static bool select_rails(topology_t *t, bool (*register_mr)(const topology_t *, uint32_t, uint32_t))
{
    rank_nearest_pairs(t);
    size_t registered = 0;
    for (size_t i = 0; i < t->npairs; ++i) {
        pair_t p = t->pairs[i];
        if (register_mr(t, p.dev, p.gpu))
            t->pairs[registered++] = p;
    }
    t->npairs = registered;
    for (size_t i = 0; i < t->npairs; ++i) {
        uint32_t mtu = t->pairs[i].mtu;
        if (i && t->pairs[i - 1].mtu == mtu)
            continue;
        memset(t->assigned, 0, sizeof(t->assigned));
        uint32_t g;
        for (g = 0; g < t->ngpu; ++g) {
            bool seen[MAX_RAILS] = {false};
            if (!assign_gpu(t, g, mtu, seen))
                break;
        }
        if (g == t->ngpu)
            return true;
    }
    return false;
}

static void print_map(const topology_t *t, FILE *out)
{
    char rails[MAX_RAILS * (sizeof(t->devs[0].dev_name) + 5)] = "";
    char indices[MAX_RAILS * 12] = "", bdfs[MAX_RAILS * sizeof(t->gpus[0].bdf)] = "";
    size_t r = 0, v = 0, b = 0;
    const pair_t *best = NULL;
    /* Use the same device-name and port order as filtered plugin enumeration. */
    for (uint32_t d = 0; d < t->ndev; ++d) {
        for (uint32_t h = 0; h < t->ndev; ++h) {
            const pair_t *p = t->assigned[h];
            if (!p || p->dev != d)
                continue;
            const gpu_t *gpu = &t->gpus[p->gpu];
            const vxfec_ibv_dev_info_t *dev = &t->devs[d];
            const char *sep = best ? "," : "";
            r += (size_t)snprintf(rails + r, sizeof(rails) - r, "%s%s:%u", sep, dev->dev_name, dev->port_num);
            v += (size_t)snprintf(indices + v, sizeof(indices) - v, "%s%u", sep, gpu->vk_index);
            b += (size_t)snprintf(bdfs + b, sizeof(bdfs) - b, "%s%s", sep, gpu->bdf);
            if (!best || cmp_pair(p, best) < 0)
                best = p;
        }
    }
    const gpu_t *gpu = &t->gpus[best->gpu];
    const vxfec_ibv_dev_info_t *dev = &t->devs[best->dev];
    fprintf(out,
            "%s:%u|%u|%d|%s|%s|%s|%s|PAIR_OK ibv_dev=%s:%u rails=%u mtu=%u speed=%llu is_roce=%d vk_dev=%u "
            "gpu_bdf=%s vk_map=%s gpu_name=\"%s\" pci_depth=%d\n",
            dev->dev_name, dev->port_num, gpu->vk_index, best->depth, gpu->bdf, rails, indices, bdfs, dev->dev_name, dev->port_num,
            t->ngpu, dev->active_mtu, (unsigned long long)dev->speed_mbps, (int)dev->is_roce, gpu->vk_index, gpu->bdf, indices,
            vxfec_vk_device_name(gpu->vk), best->depth);
}

int main(int argc, char **argv)
{
    bool cuda_only = argc == 2 && strcmp(argv[1], "--single-cuda-pair") == 0;
    bool cuda_vulkan = argc == 2 && strcmp(argv[1], "--single-cuda-vulkan-pair") == 0;
    if (argc != 1 && !cuda_only && !cuda_vulkan) {
        fprintf(stderr, "Usage: %s [--single-cuda-pair|--single-cuda-vulkan-pair]\n", argv[0]);
        return 2;
    }
    topology_t t = {0};
    int status = 2;
    t.ndev = vxfec_ibv_enumerate(t.devs, MAX_RAILS);
    if (!t.ndev)
        fprintf(stderr, "probe: 0 active IBV devices\n");
    else if (!discover_cuda_gpus(&t, !cuda_only && !cuda_vulkan))
        fprintf(stderr, "probe: cannot enumerate CUDA GPUs with PCI paths\n");
    else if (cuda_only || cuda_vulkan) {
        bool found = cuda_only || map_vulkan_gpus(&t, false);
        const pair_t *p = found ? select_single_pair(&t, register_cuda_pair, cuda_vulkan ? register_pair : NULL) : NULL;
        if (p) {
            const vxfec_ibv_dev_info_t *dev = &t.devs[p->dev];
            const gpu_t *gpu = &t.gpus[p->gpu];
            printf("%s:%u|%s|", dev->dev_name, dev->port_num, gpu->bdf);
            if (cuda_vulkan)
                printf("%u|", gpu->vk_index);
            printf("PAIR_OK is_roce=0 pci_depth=%d mtu=%u speed=%llu%s\n", p->depth, dev->active_mtu,
                   (unsigned long long)dev->speed_mbps, cuda_vulkan ? " cuda_dmabuf=1 vk_dmabuf=1" : "");
            status = 0;
        } else {
            fprintf(stderr, "probe: no %s pair accepted writable DMA-BUF registration\n",
                    cuda_vulkan ? "CUDA/Vulkan GPU and 400G IB port" : "native CUDA GPU and IB port");
        }
    } else if (map_vulkan_gpus(&t, true)) {
        if (select_rails(&t, register_pair)) {
            print_map(&t, stdout);
            status = 0;
        } else {
            fprintf(stderr, "probe: no complete DMA-BUF registered IB rail assignment for all %u GPUs\n", t.ngpu);
        }
    }
    for (uint32_t g = 0; g < t.ngpu; ++g)
        if (t.gpus[g].vk)
            vxfec_vk_ctx_free(t.gpus[g].vk);
    return status;
}
