/* Offline topology fixtures. Build with the same flags and libraries as probe_gdr_pair.c. */
#define main probe_gdr_pair_main
#include "probe_gdr_pair.c"
#undef main
#include <assert.h>

static unsigned registrations[MAX_RAILS][MAX_RAILS];
static unsigned wire_registrations[MAX_RAILS][MAX_RAILS];
static int failed_rail = -1;
static int failed_gpu = -1;
static int failed_wire_rail = -1;

static bool mock_register(const topology_t *t, uint32_t d, uint32_t g)
{
    assert(d < t->ndev && g < t->ngpu);
    registrations[d][g]++;
    return (int)d != failed_rail || (failed_gpu >= 0 && (int)g != failed_gpu);
}

static bool mock_wire_register(const topology_t *t, uint32_t d, uint32_t g)
{
    assert(d < t->ndev && g < t->ngpu && registrations[d][g] > 0);
    wire_registrations[d][g]++;
    return (int)d != failed_wire_rail;
}

static topology_t fixture(void)
{
    topology_t t = {.ngpu = 2, .ndev = 2};
    t.devs[0] = (vxfec_ibv_dev_info_t){
        .dev_name = "mlx5_0", .port_num = 1, .active_mtu = 4096, .pci_path = "/sys/devices/pci0000:00/bridgeA/switchA/hca0"};
    t.devs[1] = (vxfec_ibv_dev_info_t){
        .dev_name = "mlx5_1", .port_num = 1, .active_mtu = 4096, .pci_path = "/sys/devices/pci0000:ff/switchB/hca1"};
    strcpy(t.gpus[0].pci_path, "/sys/devices/pci0000:00/bridgeA/switchA/gpu0");
    strcpy(t.gpus[1].pci_path, "/sys/devices/pci0000:ff/switchB/gpu1");
    memset(registrations, 0, sizeof(registrations));
    memset(wire_registrations, 0, sizeof(wire_registrations));
    failed_rail = -1;
    failed_gpu = -1;
    failed_wire_rail = -1;
    t.devs[0].speed_mbps = t.devs[1].speed_mbps = 400000;
    return t;
}

static topology_t dual_fixture(void)
{
    topology_t t = fixture();
    /* The registration mocks use the mapping's presence, never the Vulkan context itself. */
    t.gpus[0].vk = t.gpus[1].vk = (vxfec_vk_ctx *)(uintptr_t)1;
    return t;
}

int main(void)
{
    topology_t t = fixture();
    assert(select_rails(&t, mock_register));
    assert(t.assigned[0]->gpu == 0 && t.assigned[1]->gpu == 1);
    assert(t.assigned[0]->depth > t.assigned[1]->depth);
    assert(registrations[0][0] == 1 && registrations[1][1] == 1);
    assert(!registrations[0][1] && !registrations[1][0]);
    t.gpus[0].vk_index = 3;
    t.gpus[1].vk_index = 1;
    strcpy(t.gpus[0].bdf, "0000:20:00.0");
    strcpy(t.gpus[1].bdf, "0000:10:00.0");
    FILE *out = tmpfile();
    assert(out);
    print_map(&t, out);
    rewind(out);
    char line[1024];
    assert(fgets(line, sizeof(line), out));
    fclose(out);
    assert(strstr(line, "|0000:20:00.0|mlx5_0:1,mlx5_1:1|3,1|0000:20:00.0,0000:10:00.0|PAIR_OK "));

    t = fixture();
    failed_rail = 1;
    assert(!select_rails(&t, mock_register));
    assert(registrations[1][1] == 1);

    /* A local spare HCA may replace an unregistrable one. */
    t = fixture();
    failed_rail = 1;
    t.ndev = 3;
    t.devs[2] = t.devs[1];
    strcpy(t.devs[2].dev_name, "mlx5_2");
    assert(select_rails(&t, mock_register));
    assert(!t.assigned[1] && t.assigned[2]->gpu == 1 && registrations[2][1] == 1);

    /* The first HCA is equally near either GPU, but the second can serve only GPU zero. */
    t = fixture();
    strcpy(t.devs[0].pci_path, "/sys/devices/pci0000:00/switch/hca0");
    strcpy(t.devs[1].pci_path, "/sys/devices/pci0000:00/switch/child/hca1");
    strcpy(t.gpus[0].pci_path, "/sys/devices/pci0000:00/switch/child/gpu0");
    strcpy(t.gpus[1].pci_path, "/sys/devices/pci0000:00/switch/gpu1");
    assert(select_rails(&t, mock_register));
    assert(t.assigned[0]->gpu == 1 && t.assigned[1]->gpu == 0);

    /* Reassign an already matched HCA when the next GPU has just one registered choice. */
    t = fixture();
    strcpy(t.devs[1].pci_path, t.devs[0].pci_path);
    strcpy(t.gpus[1].pci_path, "/sys/devices/pci0000:00/bridgeA/switchA/gpu1");
    failed_rail = 1;
    failed_gpu = 1;
    assert(select_rails(&t, mock_register));
    assert(t.assigned[0]->gpu == 1 && t.assigned[1]->gpu == 0);

    t = fixture();
    strcpy(t.devs[1].dev_name, "mlx5_0");
    t.devs[1].port_num = 2;
    assert(!select_rails(&t, mock_register));

    t = fixture();
    t.devs[1].is_roce = true;
    assert(!select_rails(&t, mock_register));

    t = fixture();
    t.devs[1].active_mtu = 2048;
    assert(!select_rails(&t, mock_register));

    /* The native two-rank profile needs one usable pair, even on a partially usable host. */
    t = fixture();
    failed_rail = 1;
    const pair_t *single = select_single_pair(&t, mock_register, NULL);
    assert(single && single->gpu == 0 && single->dev == 0);
    assert(registrations[0][0] == 1 && !registrations[1][1]);
    t = fixture();
    failed_rail = 0;
    single = select_single_pair(&t, mock_register, NULL);
    assert(single && single->gpu == 1 && single->dev == 1);
    assert(registrations[0][0] == 1 && registrations[1][1] == 1 && !registrations[0][1] && !registrations[1][0]);
    t = fixture();
    failed_rail = 0;
    t.devs[1].is_roce = true;
    assert(!select_single_pair(&t, mock_register, NULL));
    assert(!registrations[1][1]);

    /* A wire+collective run must find both registrations on the very same port and GPU. */
    t = dual_fixture();
    failed_wire_rail = 0;
    single = select_single_pair(&t, mock_register, mock_wire_register);
    assert(single && single->dev == 1 && single->gpu == 1);
    assert(wire_registrations[0][0] == 1 && wire_registrations[1][1] == 1);
    t = dual_fixture();
    failed_wire_rail = 0;
    failed_rail = 1;
    assert(!select_single_pair(&t, mock_register, mock_wire_register));
    assert(wire_registrations[0][0] == 1 && !wire_registrations[1][1]);

    /* A slower rail with better locality and unmapped idle devices are ineligible. */
    t = dual_fixture();
    t.devs[0].speed_mbps = 200000;
    single = select_single_pair(&t, mock_register, mock_wire_register);
    assert(single && single->dev == 1 && !registrations[0][0]);
    t = dual_fixture();
    t.gpus[0].vk = NULL;
    t.devs[1].is_roce = true;
    assert(!select_single_pair(&t, mock_register, mock_wire_register));
    assert(!registrations[0][0] && !registrations[0][1] && !wire_registrations[1][1]);
    puts("GDR rail assignment fixtures passed");
    return 0;
}
