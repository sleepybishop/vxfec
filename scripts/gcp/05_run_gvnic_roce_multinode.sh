#!/usr/bin/env bash
# scripts/gcp/05_run_gvnic_roce_multinode.sh
# Provision a 4-node physical GCP multi-NIC gVNIC (8896 B Jumbo MTU) + RoCEv2
# libibverbs benchmark cluster (Spot c2d-standard-16, 32 Gbps gVNIC/VM, <$0.35/run):
#   - vxfec-bench-nic-0 & vxfec-bench-nic-1 (us-central1-a, dual-NIC 8896B Jumbo)
#   - vxfec-bench-nic-xzone (us-central1-b, real cross-zone DCI RTT)
#   - vxfec-bench-nic-wan   (us-east4-a, real Iowa<->Virginia WAN RTT)
# Runs:
#   1) Dual-NIC gVNIC 8896B Jumbo superframe wire saturation (bench_wire_net)
#   2) Intra-zone dual-NIC 2-node collective AllReduce (bench_nccl --rank 0/1)
#   3) Real Linux kernel `tc netem` burst-loss experiments across Cross-Zone DCI
#      and Cross-Region WAN (GBN vs SACK vs vxfec MDS)
#   4) Real 2-node RoCEv2 `libibverbs` (IBV_QPT_UC + RDMA_WRITE_WITH_IMM +
#      VXFEC_NCCL_BACKEND=ibv) over Linux `rdma_rxe` across physical gVNIC
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
    [[ -x "$HOME/google-cloud-sdk/bin/gcloud" ]] && export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    [[ -x "/tmp/google-cloud-sdk/bin/gcloud" ]] && export PATH="/tmp/google-cloud-sdk/bin:$PATH"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
: "${GCP_PROJECT:?Set GCP_PROJECT=<project-id>}"

GCP_REGION="${GCP_REGION:-us-central1}"
GCP_ZONE_A="${GCP_ZONE_A:-${GCP_REGION}-a}"
GCP_ZONE_B="${GCP_ZONE_B:-${GCP_REGION}-c}"
GCP_REGION_WAN="${GCP_REGION_WAN:-us-east4}"
GCP_ZONE_WAN="${GCP_ZONE_WAN:-${GCP_REGION_WAN}-a}"
MACHINE_TX="${MACHINE_TX:-c2d-standard-32}"
MACHINE_RX="${MACHINE_RX:-c2d-standard-16}"
MACHINE_PEER="${MACHINE_PEER:-c2d-standard-8}"
KEEP_VMS="${KEEP_VMS:-0}"

VPC_NAME="${VPC_NAME:-vxfec-bench-vpc}"
SUBNET_NAME="${SUBNET_NAME:-vxfec-bench-subnet-${GCP_REGION}}"
if [[ "${VPC_NAME}" == "vxfec-bench-vpc" ]]; then
    SUBNET_WAN_NAME="${SUBNET_WAN_NAME:-vxfec-bench-subnet-${GCP_REGION_WAN}}"
else
    SUBNET_WAN_NAME="${SUBNET_WAN_NAME:-${VPC_NAME}-subnet-${GCP_REGION_WAN}}"
fi
BENCH_NET_TAG="${BENCH_NET_TAG:-vxfec-bench}"
export VPC_NAME SUBNET_NAME BENCH_NET_TAG
VPC1="${VPC_NAME}"
SUB1="${SUBNET_NAME}"
SUB1_WAN="${SUBNET_WAN_NAME}"

NODE0="${NODE0:-vxfec-bench-nic-0}"
NODE1="${NODE1:-vxfec-bench-nic-1}"
NODEX="${NODEX:-vxfec-bench-nic-xzone}"
NODEW="${NODEW:-vxfec-bench-nic-wan}"

OUT_DIR="${REPO_ROOT}/docs/cloud_runs/gcp_gvnic_multinode"
mkdir -p "${OUT_DIR}"

TARBALL=""
NODEX_ZONE_FILE="/tmp/vxfec_multinode_xz_$$.txt"
VXFEC_CLEANUP_TAG="05_run_gvnic_roce_multinode"
# shellcheck source=scripts/gcp/lib_cleanup.sh
source "${REPO_ROOT}/scripts/gcp/lib_cleanup.sh"
VXFEC_TEMP_FILES+=("${NODEX_ZONE_FILE}")
vxfec_setup_cleanup_traps

VPC2="vxfec-bench-${VXFEC_RUN_TOKEN}-vpc2"
SUB2="vxfec-bench-${VXFEC_RUN_TOKEN}-sub2-${GCP_REGION}"
FW2="${VPC2}-allow-internal"

"${REPO_ROOT}/scripts/gcp/01_setup_vpc_jumbo.sh"

# Shared WAN subnet on VPC1 stays persistent like VPC1/SUB1 (torn down by 99_teardown_all.sh);
# secondary VPC2/SUB2/FW2 use per-run names and are registered before creation.
if ! gcloud compute networks subnets describe "${SUB1_WAN}" --project="${GCP_PROJECT}" --region="${GCP_REGION_WAN}" >/dev/null 2>&1; then
    vxfec_run_create gcloud compute networks subnets create "${SUB1_WAN}" \
        --project="${GCP_PROJECT}" --network="${VPC1}" --region="${GCP_REGION_WAN}" --range="10.130.0.0/20"
fi

vxfec_add_unique VXFEC_CREATED_VPCS "${VPC2}"
vxfec_run_create gcloud compute networks create "${VPC2}" --project="${GCP_PROJECT}" --subnet-mode=custom --mtu=8896
vxfec_add_unique VXFEC_CREATED_SUBNETS "${SUB2}:${GCP_REGION}:${VPC2}"
vxfec_run_create gcloud compute networks subnets create "${SUB2}" \
    --project="${GCP_PROJECT}" --network="${VPC2}" --region="${GCP_REGION}" --range="10.129.0.0/20"
vxfec_add_unique VXFEC_CREATED_FWS "${FW2}:${VPC2}"
vxfec_run_create gcloud compute firewall-rules create "${FW2}" \
    --project="${GCP_PROJECT}" --network="${VPC2}" --allow=tcp:1-65535,udp:1-65535,icmp --source-ranges="10.128.0.0/12" --target-tags="${BENCH_NET_TAG}"

SPOT_COMMON=(
    --boot-disk-size=40GB
    --boot-disk-type=pd-ssd
    --image-family=ubuntu-2204-lts
    --image-project=ubuntu-os-cloud
    --provisioning-model=SPOT
    --instance-termination-action=DELETE
    --max-run-duration=1h
    --tags="${BENCH_NET_TAG}"
    --labels="vxfec-run=${VXFEC_RUN_TOKEN}"
)

for pair in "${NODE0}:${GCP_ZONE_A}" "${NODE1}:${GCP_ZONE_A}" "${NODEX}:${GCP_ZONE_B}" "${NODEX}:${GCP_REGION}-f" "${NODEW}:${GCP_ZONE_WAN}"; do
    if vxfec_vm_exists "${pair%%:*}" "${pair#*:}"; then
        echo "ERROR: Instance ${pair%%:*} already exists in ${pair#*:}; refusing to clobber." >&2
        exit 1
    fi
done

vxfec_add_unique VXFEC_CREATED_VMS "${NODE0}:${GCP_ZONE_A}"
vxfec_add_unique VXFEC_CREATED_VMS "${NODE1}:${GCP_ZONE_A}"
vxfec_add_unique VXFEC_CREATED_VMS "${NODEX}:${GCP_ZONE_B}"
vxfec_add_unique VXFEC_CREATED_VMS "${NODEX}:${GCP_REGION}-f"
vxfec_add_unique VXFEC_CREATED_VMS "${NODEW}:${GCP_ZONE_WAN}"

echo "[05_run_gvnic_roce_multinode] Launching 4 Spot VMs (TX=${MACHINE_TX} 32Gbps sustained, RX=${MACHINE_RX}, Peers=${MACHINE_PEER})..."
(
    gcloud compute instances create "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
        --machine-type="${MACHINE_TX}" \
        --network-interface="nic-type=GVNIC,network=${VPC1},subnet=${SUB1}" \
        --network-interface="nic-type=GVNIC,network=${VPC2},subnet=${SUB2},no-address" \
        "${SPOT_COMMON[@]}" && echo "vm:${NODE0}:${GCP_ZONE_A}" >> "${VXFEC_CREATED_LOG}"
) &
P0=$!
VXFEC_CREATE_PIDS+=("${P0}")

(
    gcloud compute instances create "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
        --machine-type="${MACHINE_RX}" \
        --network-interface="nic-type=GVNIC,network=${VPC1},subnet=${SUB1}" \
        --network-interface="nic-type=GVNIC,network=${VPC2},subnet=${SUB2},no-address" \
        "${SPOT_COMMON[@]}" && echo "vm:${NODE1}:${GCP_ZONE_A}" >> "${VXFEC_CREATED_LOG}"
) &
P1=$!
VXFEC_CREATE_PIDS+=("${P1}")

(
    if gcloud compute instances create "${NODEX}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_B}" \
        --machine-type="${MACHINE_PEER}" \
        --network-interface="nic-type=GVNIC,network=${VPC1},subnet=${SUB1}" \
        "${SPOT_COMMON[@]}"; then
        echo "${GCP_ZONE_B}" > "${NODEX_ZONE_FILE}"
        echo "vm:${NODEX}:${GCP_ZONE_B}" >> "${VXFEC_CREATED_LOG}"
    elif [[ ! -f "${VXFEC_CANCEL_FILE}" ]]; then
        gcloud compute instances create "${NODEX}" --project="${GCP_PROJECT}" --zone="${GCP_REGION}-f" \
            --machine-type="${MACHINE_PEER}" \
            --network-interface="nic-type=GVNIC,network=${VPC1},subnet=${SUB1}" \
            "${SPOT_COMMON[@]}"
        echo "${GCP_REGION}-f" > "${NODEX_ZONE_FILE}"
        echo "vm:${NODEX}:${GCP_REGION}-f" >> "${VXFEC_CREATED_LOG}"
    else
        exit 1
    fi
) &
PX=$!
VXFEC_CREATE_PIDS+=("${PX}")

(
    gcloud compute instances create "${NODEW}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_WAN}" \
        --machine-type="${MACHINE_PEER}" \
        --network-interface="nic-type=GVNIC,network=${VPC1},subnet=${SUB1_WAN}" \
        "${SPOT_COMMON[@]}" && echo "vm:${NODEW}:${GCP_ZONE_WAN}" >> "${VXFEC_CREATED_LOG}"
) &
PW=$!
VXFEC_CREATE_PIDS+=("${PW}")

VM_ERR=0
wait "${P0}" || VM_ERR=1
wait "${P1}" || VM_ERR=1
if wait "${PX}" && [[ -s "${NODEX_ZONE_FILE}" ]]; then
    GCP_ZONE_B="$(cat "${NODEX_ZONE_FILE}")"
    rm -f "${NODEX_ZONE_FILE}"
else
    VM_ERR=1
fi
wait "${PW}" || VM_ERR=1
VXFEC_CREATE_PIDS=()
vxfec_ingest_created_log
[[ "${VM_ERR}" == "0" ]] || exit 1

TARBALL="$(umask 077 && mktemp /tmp/vxfec_multinode_XXXXXX.tar.gz)"
vxfec_package_repo_tarball "${REPO_ROOT}" "${TARBALL}"

prep_vm() {
    local vm="$1"
    local zone="$2"
    for _ in $(seq 1 30); do
        gcloud compute ssh "${vm}" --project="${GCP_PROJECT}" --zone="${zone}" --command="true" >/dev/null 2>&1 && break
        sleep 4
    done
    gcloud compute scp "${TARBALL}" "${vm}:/tmp/vxfec_src.tar.gz" --project="${GCP_PROJECT}" --zone="${zone}"
    gcloud compute ssh "${vm}" --project="${GCP_PROJECT}" --zone="${zone}" --command="
        set -euo pipefail
        for _ in 1 2 3 4 5; do
            sudo apt-get -o DPkg::Lock::Timeout=120 update -qq && \
            sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=120 install -y -qq --no-install-recommends \
                build-essential ca-certificates pkg-config libvulkan-dev mesa-vulkan-drivers \
                rdma-core ibverbs-utils libibverbs-dev iproute2 && break
            sleep 5
        done
        sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=120 install -y -qq --no-install-recommends \
            linux-modules-extra-\$(uname -r) || true
        sudo sysctl -w net.core.rmem_max=67108864 net.core.wmem_max=67108864 \
                       net.core.rmem_default=16777216 net.core.wmem_default=16777216 \
                       net.core.netdev_max_backlog=65536 >/dev/null
        echo -e '* soft memlock unlimited\n* hard memlock unlimited' | sudo tee /etc/security/limits.d/99-rdma.conf >/dev/null
        sudo modprobe sch_netem cls_u32 || true
        rm -rf ~/vxfec && mkdir -p ~/vxfec
        tar -xzf /tmp/vxfec_src.tar.gz -C ~/vxfec
        cd ~/vxfec && make -j\$(nproc) all
    "
}

echo "[05_run_gvnic_roce_multinode] Provisioning packages and compiling vxfec on all 4 nodes in parallel..."
prep_vm "${NODE0}" "${GCP_ZONE_A}" & B0=$!
prep_vm "${NODE1}" "${GCP_ZONE_A}" & B1=$!
prep_vm "${NODEX}" "${GCP_ZONE_B}" & BX=$!
prep_vm "${NODEW}" "${GCP_ZONE_WAN}" & BW=$!
VXFEC_WORK_PIDS+=("${B0}" "${B1}" "${BX}" "${BW}")
wait "${B0}"
wait "${B1}"
wait "${BX}"
wait "${BW}"
rm -f "${TARBALL}"

NIC_CFG0="$(vxfec_configure_vm_nics "${NODE0}" "${GCP_ZONE_A}")"
NIC_CFG1="$(vxfec_configure_vm_nics "${NODE1}" "${GCP_ZONE_A}")"
IP0_LIST="${NIC_CFG0%%|*}"
IFACES0="${NIC_CFG0#*|}"
IP1_LIST="${NIC_CFG1%%|*}"
IFACES1="${NIC_CFG1#*|}"
IP0_A="$(echo "${IP0_LIST}" | cut -d',' -f1)"
IP0_B="$(echo "${IP0_LIST}" | cut -d',' -f2)"
IP1_A="$(echo "${IP1_LIST}" | cut -d',' -f1)"
IP1_B="$(echo "${IP1_LIST}" | cut -d',' -f2)"
IPX="$(gcloud compute instances describe "${NODEX}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_B}" --format='get(networkInterfaces[0].networkIP)')"
IPW="$(gcloud compute instances describe "${NODEW}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_WAN}" --format='get(networkInterfaces[0].networkIP)')"
NIC0_PRIMARY="$(echo "${IFACES0}" | cut -d',' -f1)"
NICX_PRIMARY="$(gcloud compute ssh "${NODEX}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_B}" \
    --command="ip -o -4 addr show | awk '\$2 != \"lo\" {print \$2; exit}'")"
NICW_PRIMARY="$(gcloud compute ssh "${NODEW}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_WAN}" \
    --command="ip -o -4 addr show | awk '\$2 != \"lo\" {print \$2; exit}'")"

echo "[05_run_gvnic_roce_multinode] Capturing hardware topology, gVNIC 8896B Jumbo MTU & RTT baselines..."
gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" --command="
    {
        echo '=== CPU & Kernel ==='
        lscpu | grep -E 'Model name|CPU\(s\):'
        uname -a
        echo '=== gVNIC Interfaces & MTU ==='
        ip link show
        echo '=== Intra-Zone RTT (${NODE0} -> ${NODE1} ${IP1_A}) ==='
        ping -c 10 -i 0.1 ${IP1_A}
        echo '=== Intra-Zone Jumbo 8800B DF Ping (${NODE0} -> ${NODE1}) ==='
        ping -c 4 -M do -s 8800 ${IP1_A}
        echo '=== Cross-Zone DCI RTT (${GCP_ZONE_A} -> ${GCP_ZONE_B} ${IPX}) ==='
        ping -c 10 -i 0.1 ${IPX}
        echo '=== Cross-Region WAN RTT (${GCP_ZONE_A} -> ${GCP_ZONE_WAN} ${IPW}) ==='
        ping -c 10 -i 0.1 ${IPW}
    } | tee ~/vxfec/gcp_topology_and_rtt.txt
"

# ---------------------------------------------------------------------------
# Suite 1: Dual-NIC gVNIC (2x NICs, 8896 B Jumbo) Superframe Wire Saturation
# ---------------------------------------------------------------------------
echo "[Suite 1] Dual-NIC gVNIC (${IP0_A},${IP0_B} -> ${IP1_A},${IP1_B}) 8192B Jumbo Superframe Saturation..."
rm -f "${OUT_DIR}/gcp_wire_dual_gvnic.log"
for spec in "rateless 0 28" "rateless 10 28" "mds 10 28" "rateless 15 28"; do
    read -r codec bpct rate <<<"${spec}"
    gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
        --command="cd ~/vxfec && ./bench_wire_net --mode rx --bind ${IP1_A},${IP1_B} --port 31500 --ports-per-ip 4" &
    RX_PID=$!
    VXFEC_WORK_PIDS+=("${RX_PID}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
        --command="cd ~/vxfec && ./bench_wire_net --mode tx --peers ${IP1_A},${IP1_B} --port 31500 --ports-per-ip 4 --codec ${codec} --z 16 --k 200 --p 48 --sym-t 8192 --iters 4 --blackout-pct ${bpct} --rate-gbps ${rate} --csv ~/vxfec/gcp_wire_dual_gvnic.csv" | tee -a "${OUT_DIR}/gcp_wire_dual_gvnic.log" || STEP_ERR=1
    wait "${RX_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1
done

# ---------------------------------------------------------------------------
# Suite 2: Intra-Zone Dual-NIC gVNIC (ndev=2, 8896B Jumbo) 2-Node AllReduce
# ---------------------------------------------------------------------------
echo "[Suite 2] Intra-Zone Dual-NIC gVNIC (ndev=2) 2-Node Collective AllReduce..."
gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
    --command="cd ~/vxfec && VXFEC_NCCL_IFNAMES=${IFACES0} ./bench_nccl --rank 0 --master ${IP0_A}:29600 --iters 30 --csv ~/vxfec/gcp_nccl_intrazone_dualnic.csv | tee ~/vxfec/gcp_nccl_intrazone_dualnic.txt" &
R0_PID=$!
VXFEC_WORK_PIDS+=("${R0_PID}")
sleep 1
STEP_ERR=0
gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
    --command="cd ~/vxfec && VXFEC_NCCL_IFNAMES=${IFACES1} ./bench_nccl --rank 1 --master ${IP0_A}:29600 --iters 30" || STEP_ERR=1
wait "${R0_PID}" || STEP_ERR=1
[[ "${STEP_ERR}" == "0" ]] || exit 1

# ---------------------------------------------------------------------------
# Suite 3: Real Linux Kernel `tc qdisc netem` Burst Loss over Physical
#          Cross-Zone DCI (us-central1-a <-> us-central1-b) & Cross-Region WAN
# ---------------------------------------------------------------------------
# Gilbert-Elliott 2-state Markov parameters: stationary loss pi_B = p / (p + r), mean burst = 1 / r.
# Default p=0.38%, r=25% gives pi_B = 0.0038 / (0.0038 + 0.25) = 1.50% stationary loss, mean burst = 4 pkts.
GEMODEL_P="${GEMODEL_P:-0.38%}"
GEMODEL_R="${GEMODEL_R:-25%}"

set_udp_netem() {
    local vm="$1"
    local zone="$2"
    local nic="$3"
    gcloud compute ssh "${vm}" --project="${GCP_PROJECT}" --zone="${zone}" --command="
        sudo tc qdisc del dev ${nic} root 2>/dev/null || true
        sudo tc qdisc add dev ${nic} root handle 1: prio bands 3
        sudo tc qdisc add dev ${nic} parent 1:3 handle 30: netem loss gemodel ${GEMODEL_P} ${GEMODEL_R}
        sudo tc filter add dev ${nic} protocol ip parent 1:0 prio 1 u32 match ip protocol 17 0xff flowid 1:3
    "
}

record_udp_netem_stats() {
    local vm="$1"
    local zone="$2"
    local nic="$3"
    local log_file="$4"
    gcloud compute ssh "${vm}" --project="${GCP_PROJECT}" --zone="${zone}" --command="
        echo '=== tc -s qdisc (${vm} ${nic}: gemodel p=${GEMODEL_P} r=${GEMODEL_R}) ==='
        sudo tc -s qdisc show dev ${nic}
    " | tee -a "${log_file}" || true
}

echo "[Suite 3A] Real Kernel tc-netem (gemodel p=${GEMODEL_P} r=${GEMODEL_R}, ~1.5% stationary burst loss) on Cross-Zone DCI (${GCP_ZONE_A} <-> ${GCP_ZONE_B})..."
set_udp_netem "${NODE0}" "${GCP_ZONE_A}" "${NIC0_PRIMARY}"
set_udp_netem "${NODEX}" "${GCP_ZONE_B}" "${NICX_PRIMARY}"

for spec in "gbn~512 KiB | XZone Kernel-Drop (GBN)~524288" \
            "sack~512 KiB | XZone Kernel-Drop (SACK)~524288" \
            "rs8~512 KiB | XZone Kernel-Drop (MDS)~524288" \
            "gbn~2.0 MiB | XZone Kernel-Drop (GBN)~2097152" \
            "sack~2.0 MiB | XZone Kernel-Drop (SACK)~2097152" \
            "rs8~2.0 MiB | XZone Kernel-Drop (MDS)~2097152"; do
    IFS='~' read -r pol lbl bsz <<<"${spec}"
    ( gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
        --command="cd ~/vxfec && VXFEC_NCCL_IFNAMES=${NIC0_PRIMARY} ./bench_nccl --rank 0 --master ${IP0_A}:29610 --single-codec ${pol} --label '${lbl}' --bytes ${bsz} --iters 35 --csv ~/vxfec/gcp_nccl_xzone_netem.csv" | tee -a "${OUT_DIR}/gcp_nccl_xzone_netem.log" ) &
    X0_PID=$!
    VXFEC_WORK_PIDS+=("${X0_PID}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODEX}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_B}" \
        --command="cd ~/vxfec && VXFEC_NCCL_IFNAMES=${NICX_PRIMARY} ./bench_nccl --rank 1 --master ${IP0_A}:29610 --single-codec ${pol} --label '${lbl}' --bytes ${bsz} --iters 35" || STEP_ERR=1
    wait "${X0_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1
done

record_udp_netem_stats "${NODE0}" "${GCP_ZONE_A}" "${NIC0_PRIMARY}" "${OUT_DIR}/gcp_nccl_xzone_netem.log"
record_udp_netem_stats "${NODEX}" "${GCP_ZONE_B}" "${NICX_PRIMARY}" "${OUT_DIR}/gcp_nccl_xzone_netem.log"

gcloud compute ssh "${NODEX}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_B}" \
    --command="sudo tc qdisc del dev ${NICX_PRIMARY} root || true"

echo "[Suite 3B] Real Kernel tc-netem (gemodel p=${GEMODEL_P} r=${GEMODEL_R}, ~1.5% stationary burst loss) on Cross-Region Iowa<->Virginia WAN (~24ms RTT)..."
set_udp_netem "${NODE0}" "${GCP_ZONE_A}" "${NIC0_PRIMARY}"
set_udp_netem "${NODEW}" "${GCP_ZONE_WAN}" "${NICW_PRIMARY}"

for spec in "gbn~512 KiB | WAN 24ms Kernel-Drop (GBN)~524288" \
            "sack~512 KiB | WAN 24ms Kernel-Drop (SACK)~524288" \
            "rs8~512 KiB | WAN 24ms Kernel-Drop (MDS)~524288"; do
    IFS='~' read -r pol lbl bsz <<<"${spec}"
    ( gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
        --command="cd ~/vxfec && VXFEC_NCCL_IFNAMES=${NIC0_PRIMARY} ./bench_nccl --rank 0 --master ${IP0_A}:29620 --single-codec ${pol} --label '${lbl}' --bytes ${bsz} --iters 20 --csv ~/vxfec/gcp_nccl_wan_netem.csv" | tee -a "${OUT_DIR}/gcp_nccl_wan_netem.log" ) &
    W0_PID=$!
    VXFEC_WORK_PIDS+=("${W0_PID}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODEW}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_WAN}" \
        --command="cd ~/vxfec && VXFEC_NCCL_IFNAMES=${NICW_PRIMARY} ./bench_nccl --rank 1 --master ${IP0_A}:29620 --single-codec ${pol} --label '${lbl}' --bytes ${bsz} --iters 20" || STEP_ERR=1
    wait "${W0_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1
done

record_udp_netem_stats "${NODE0}" "${GCP_ZONE_A}" "${NIC0_PRIMARY}" "${OUT_DIR}/gcp_nccl_wan_netem.log"
record_udp_netem_stats "${NODEW}" "${GCP_ZONE_WAN}" "${NICW_PRIMARY}" "${OUT_DIR}/gcp_nccl_wan_netem.log"

gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
    --command="sudo tc qdisc del dev ${NIC0_PRIMARY} root || true"
gcloud compute ssh "${NODEW}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_WAN}" \
    --command="sudo tc qdisc del dev ${NICW_PRIMARY} root || true"

# ---------------------------------------------------------------------------
# Suite 4: Real 2-Node RoCEv2 `libibverbs` (`rdma_rxe` over gVNIC)
#          Validates `IBV_QPT_UC` + `RDMA_WRITE_WITH_IMM` + `VXFEC_NCCL_BACKEND=ibv`
# ---------------------------------------------------------------------------
echo "[Suite 4] Enabling RoCEv2 (rdma_rxe) on ${NODE0} & ${NODE1} and validating libibverbs IBV_QPT_UC..."
RXE_OK=1
for vm in "${NODE0}" "${NODE1}"; do
    if ! gcloud compute ssh "${vm}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" --command="
        set -euo pipefail
        NIC=\$(ip -o -4 addr show | awk '\$2 != \"lo\" {print \$2; exit}')
        sudo modprobe rdma_rxe
        sudo rdma link add rxe0 type rxe netdev \"\${NIC}\" 2>/dev/null || true
        ibv_devinfo -d rxe0 | tee ~/vxfec/gcp_ibv_devinfo.txt
    "; then
        RXE_OK=0
    fi
done

if [[ "${RXE_OK}" == "1" ]]; then
    for spec in "rateless 0" "rateless 10" "mds 10"; do
        read -r codec bpct <<<"${spec}"
        gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
            --command="cd ~/vxfec && ./bench_wire_net --mode rx --bind ${IP1_A} --port 31700" &
        RX_PID=$!
        VXFEC_WORK_PIDS+=("${RX_PID}")
        sleep 1
        STEP_ERR=0
        gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
            --command="cd ~/vxfec && ./bench_wire_net --mode tx --peers ${IP1_A} --port 31700 --ibv --codec ${codec} --z 8 --k 128 --p 36 --sym-t 2048 --iters 5 --blackout-pct ${bpct} --csv ~/vxfec/gcp_wire_roce_ibv.csv" | tee -a "${OUT_DIR}/gcp_wire_roce_ibv.log" || STEP_ERR=1
        wait "${RX_PID}" || STEP_ERR=1
        [[ "${STEP_ERR}" == "0" ]] || exit 1
    done

    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
        --command="cd ~/vxfec && VXFEC_NCCL_BACKEND=ibv ./bench_nccl --rank 0 --master ${IP0_A}:29630 --iters 25 --csv ~/vxfec/gcp_nccl_roce_ibv.csv | tee ~/vxfec/gcp_nccl_roce_ibv.txt" &
    I0_PID=$!
    VXFEC_WORK_PIDS+=("${I0_PID}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}" \
        --command="cd ~/vxfec && VXFEC_NCCL_BACKEND=ibv ./bench_nccl --rank 1 --master ${IP0_A}:29630 --iters 25" || STEP_ERR=1
    wait "${I0_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1
else
    echo "[Suite 4] NOTE: rdma_rxe unavailable on running cloud kernel; skipping Soft-RoCE suite."
fi

echo "[05_run_gvnic_roce_multinode] Pulling all benchmark logs and CSVs to ${OUT_DIR}..."
gcloud compute scp "${NODE0}:~/vxfec/gcp_*" "${OUT_DIR}/" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_A}"
echo "[05_run_gvnic_roce_multinode] Complete! Results saved in ${OUT_DIR}."
