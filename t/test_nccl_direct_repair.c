#include "../plugins/nccl/nccl_net_proto.c"

static bool check_legacy_sliced_generation(uint16_t gen_idx)
{
    vxfec_recv_comm_t *comm = calloc(1, sizeof(*comm));
    if (!comm)
        return false;
    comm->features = VXFEC_HELLO_SLICE;
    vxfec_rx_req_t *req = &comm->rx_pool[0];
    req->in_use = true;
    req->rdma_cts_sent = true;
    req->cts_token = 1u;

    /* main's wire format has a 14-bit generation followed by a 7-bit ESI. */
    uint32_t imm = (1u << 31u) | (1u << 22u) | ((uint32_t)gen_idx << 7u);
    rx_process_rdma_imm(comm, 2u, imm | 126u, 1u);
    vxfec_rx_slot_t *slot = &comm->slots[0];
    *slot = (vxfec_rx_slot_t){.active = true,
                              .bound_req = req,
                              .sys_bitmap = slot->inline_bitmap,
                              .K = 128u,
                              .sym_T = 4u,
                              .total_bytes = 512u,
                              .gen_idx = gen_idx,
                              .num_gens = 16384u};
    comm->active_slot_mask[0] = 1u;
    rx_replay_early_rdma_syms(comm, slot, 2u);
    rx_process_rdma_imm(comm, 1u, imm | 127u, 3u);
    bool ok = req->early_rdma_cnt == 0 && slot->rx_sys_cnt == 2 && slot->sys_bitmap[15] == 0xc0u && slot->rx_qp_pkts[1] == 1 &&
              slot->rx_qp_pkts[2] == 1 && slot->rx_rep_cnt == 0;
    free(req->early_rdma_syms);
    free(comm);
    return ok;
}

static bool check_cts_repair_arena(uint8_t features)
{
    vxfec_send_comm_t *comm = calloc(1, sizeof(*comm));
    if (!comm)
        return false;
    comm->features = features;
    /* An unversioned listener can advertise a repair arena in its CTS. */
    vxfec_rdma_cts_pkt_t cts = {.hdr = {.K = 1u, .flags = 1u, .rx_pkts = 123u, .ecn_port_mask = 321u, .ir_count = 64u},
                                .remote_iova = 456u,
                                .remote_rkey = 654u,
                                .max_bytes = 4096u};
    tx_handle_rdma_cts(comm, (const uint8_t *)&cts, sizeof(cts));
    const vxfec_cts_entry_t *entry = &comm->cts_ring[0];
    bool repair = (features & VXFEC_HELLO_RDMA_REPAIR) != 0u;
    bool ok = entry->valid && entry->remote_iova == cts.remote_iova && entry->remote_rkey == cts.remote_rkey &&
              entry->rep_iova == (repair ? 123u : 0u) && entry->rep_rkey == (repair ? 321u : 0u) &&
              entry->rep_slots == (repair ? 64u : 0u);
    free(comm);
    return ok;
}

static bool check_direct_repairs(uint8_t initial_codec, uint8_t rq_codec)
{
    vxfec_rx_slot_t slot = {.flags = initial_codec, .K = 8u, .sym_T = 4u};
    const uint8_t parity[] = {17, 24, 39, 42};
    rx_apply_direct_rep_sym(NULL, &slot, 0u, 0u, parity, false, 1u);
    bool ok = slot.rx_rep_cnt == 1u && slot.rep_esis[0] == slot.K && memcmp(slot.rep_data, parity, sizeof(parity)) == 0;

    if (initial_codec == WIRE_FLAG_RS_CAUCHY8) {
        rx_promote_slot_to_rq(&slot, rq_codec);
        /* Another RS8 write completes after the switch, on a different rail. */
        rx_apply_direct_rep_sym(NULL, &slot, 1u, 1u, parity, false, 2u);
        ok = ok && slot.rx_rep_cnt == 0u && slot.max_rep_esi == 255u && slot.rx_pkt_cnt == 2u && slot.rx_qp_pkts[1] == 1u;
    }
    free(slot.rep_esis);
    free(slot.rep_data);
    return ok;
}

static bool check_rep_slot_quarantine(void)
{
    vxfec_send_comm_t *comm = calloc(1, sizeof(*comm));
    if (!comm)
        return false;
    vxfec_tx_slot_t *req = &comm->tx_pool[0];
    req->slot_idx = 0u;
    req->rdma_rep_slots = 1u;
    uint8_t s0 = 0xffu;
    uint64_t t0 = mono_ns();
    bool ok = tx_alloc_rx_rep_slot(comm, req, t0, &s0) && s0 == 0u;
    tx_deactivate_slot(comm, req, false);
    /* slot 0 must stay quarantined immediately after PKT_DONE */
    uint8_t s1 = 0xffu;
    ok = ok && !tx_alloc_rx_rep_slot(comm, req, t0, &s1);
    /* once quarantine expires, slot 0 is reusable */
    ok = ok && tx_alloc_rx_rep_slot(comm, req, comm->rx_rep_ready_ns[0], &s1) && s1 == 0u;
    free(comm);
    return ok;
}

int main(void)
{
    if (!check_legacy_sliced_generation(8191u) || !check_legacy_sliced_generation(8192u) ||
        !check_legacy_sliced_generation(16383u)) {
        fputs("FAIL: legacy sliced generation before or after the generation header\n", stderr);
        return 1;
    }
    puts("PASS: all 14 legacy sliced generation bits preserved for early and active RDMA writes");
    if (!check_cts_repair_arena(VXFEC_HELLO_SLICE) || !check_cts_repair_arena(VXFEC_VERBS_FEATURES)) {
        fputs("FAIL: repair arena did not match negotiated verbs features\n", stderr);
        return 1;
    }
    const uint8_t rq_codecs[] = {WIRE_FLAG_RQ_GF256, WIRE_FLAG_RQ_GF2};
    for (size_t i = 0; i < sizeof(rq_codecs); i++) {
        if (!check_direct_repairs(WIRE_FLAG_RS_CAUCHY8, rq_codecs[i]) || !check_direct_repairs(rq_codecs[i], rq_codecs[i])) {
            fprintf(stderr, "FAIL: direct repair handling across codec promotion (RQ flags %u)\n", rq_codecs[i]);
            return 1;
        }
    }
    puts("PASS: native codec direct repairs accepted and late RS8 repairs discarded after promotion");
    if (!check_rep_slot_quarantine()) {
        fputs("FAIL: released direct repair slot was reused before quarantine expired\n", stderr);
        return 1;
    }
    puts("PASS: released direct repair slots respect quarantine before reuse");
    return 0;
}
