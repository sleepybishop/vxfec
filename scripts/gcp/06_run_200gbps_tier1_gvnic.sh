#!/usr/bin/env bash
# scripts/gcp/06_run_200gbps_tier1_gvnic.sh
# Provision a 2-node 4x-NIC (8896 B Jumbo MTU) GCP Titanium gVNIC 200 Gbps
# high-bandwidth benchmark pair with COMPACT collocated placement, run:
#   1) Unidirectional 64-rail 8192B Jumbo superframe saturation (bench_wire_net)
#   2) Simultaneous Full-Duplex Bidirectional 200+ Gbps superframe saturation
#   3) 4-NIC (ndev=4) 2-Node 32 MiB Collective AllReduce (bench_nccl)
# Pulls all CSVs/logs to docs/cloud_runs/gcp_tier1_200g/ and tears down this
# run's VMs and run-created network resources on EXIT (unless KEEP_VMS=1).
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
    [[ -x "$HOME/google-cloud-sdk/bin/gcloud" ]] && export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    [[ -x "/tmp/google-cloud-sdk/bin/gcloud" ]] && export PATH="/tmp/google-cloud-sdk/bin:$PATH"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
: "${GCP_PROJECT:?Set GCP_PROJECT=<project-id>}"
export GCP_PROJECT

GCP_REGION="${GCP_REGION:-us-central1}"
OUT_DIR="${REPO_ROOT}/docs/cloud_runs/gcp_tier1_200g"
mkdir -p "${OUT_DIR}"

NODE0="${NODE0:-vxfec-bench-t1-0}"
NODE1="${NODE1:-vxfec-bench-t1-1}"
KEEP_VMS="${KEEP_VMS:-0}"
if [[ -z "${AUTO_POWEROFF_MINS:-}" ]]; then
    AUTO_POWEROFF_MINS="$([[ "${KEEP_VMS}" == "1" ]] && echo 0 || echo 45)"
fi
ACTIVE_ZONE=""
SELECTED_SHAPE=""
TARBALL=""
VXFEC_CLEANUP_TAG="06_run_200gbps_tier1_gvnic"
# shellcheck source=scripts/gcp/lib_cleanup.sh
source "${REPO_ROOT}/scripts/gcp/lib_cleanup.sh"
vxfec_setup_cleanup_traps

# 1) Provision shared base Jumbo VPC + 3 run-scoped 8896B Jumbo VPCs (default + 4 = 5 VPCs)
echo "[06_run_200gbps_tier1_gvnic] Provisioning 4x 8896B Jumbo VPCs for 4-NIC Titanium gVNIC striping..."
VPC_NAME="${VPC_NAME:-vxfec-bench-vpc}"
SUBNET_NAME="${SUBNET_NAME:-vxfec-bench-subnet-${GCP_REGION}}"
POLICY_NAME="${POLICY_NAME:-vxfec-bench-compact-${GCP_REGION}}"
BENCH_NET_TAG="${BENCH_NET_TAG:-vxfec-bench}"
export VPC_NAME SUBNET_NAME POLICY_NAME BENCH_NET_TAG
"${REPO_ROOT}/scripts/gcp/01_setup_vpc_jumbo.sh"

VPC_CREATE_PIDS=()
for idx in 2 3 4; do
    VN="vxfec-bench-${VXFEC_RUN_TOKEN}-vpc${idx}"
    vxfec_add_unique VXFEC_CREATED_VPCS "${VN}"
    gcloud compute networks create "${VN}" --project="${GCP_PROJECT}" --subnet-mode=custom --mtu=8896 &
    pid=$!
    VPC_CREATE_PIDS+=("${pid}")
    VXFEC_CREATE_PIDS+=("${pid}")
done
CREATE_ERR=0
for pid in "${VPC_CREATE_PIDS[@]}"; do
    wait "${pid}" || CREATE_ERR=1
done
VXFEC_CREATE_PIDS=()
[[ "${CREATE_ERR}" == "0" ]] || exit 1

SUB_FW_PIDS=()
for idx in 2 3 4; do
    VN="vxfec-bench-${VXFEC_RUN_TOKEN}-vpc${idx}"
    SN="vxfec-bench-${VXFEC_RUN_TOKEN}-sub${idx}"
    FW="${VN}-internal"
    CIDR="10.$((127 + idx)).0.0/20"
    vxfec_add_unique VXFEC_CREATED_SUBNETS "${SN}:${GCP_REGION}:${VN}"
    gcloud compute networks subnets create "${SN}" \
        --project="${GCP_PROJECT}" --network="${VN}" --region="${GCP_REGION}" --range="${CIDR}" &
    pid=$!
    SUB_FW_PIDS+=("${pid}")
    VXFEC_CREATE_PIDS+=("${pid}")

    vxfec_add_unique VXFEC_CREATED_FWS "${FW}:${VN}"
    gcloud compute firewall-rules create "${FW}" \
        --project="${GCP_PROJECT}" --network="${VN}" --allow=tcp:1-65535,udp:1-65535,icmp --source-ranges="10.128.0.0/12" --target-tags="${BENCH_NET_TAG}" &
    pid=$!
    SUB_FW_PIDS+=("${pid}")
    VXFEC_CREATE_PIDS+=("${pid}")
done
for pid in "${SUB_FW_PIDS[@]}"; do
    wait "${pid}" || CREATE_ERR=1
done
VXFEC_CREATE_PIDS=()
[[ "${CREATE_ERR}" == "0" ]] || exit 1

NIC_ARGS=(
    --network-interface="nic-type=GVNIC,network=${VPC_NAME},subnet=${SUBNET_NAME}"
    --network-interface="nic-type=GVNIC,network=vxfec-bench-${VXFEC_RUN_TOKEN}-vpc2,subnet=vxfec-bench-${VXFEC_RUN_TOKEN}-sub2,no-address"
    --network-interface="nic-type=GVNIC,network=vxfec-bench-${VXFEC_RUN_TOKEN}-vpc3,subnet=vxfec-bench-${VXFEC_RUN_TOKEN}-sub3,no-address"
    --network-interface="nic-type=GVNIC,network=vxfec-bench-${VXFEC_RUN_TOKEN}-vpc4,subnet=vxfec-bench-${VXFEC_RUN_TOKEN}-sub4,no-address"
)

POLICY_ARG=()
if gcloud compute resource-policies describe "${POLICY_NAME}" --project="${GCP_PROJECT}" --region="${GCP_REGION}" >/dev/null 2>&1; then
    POLICY_ARG=(--resource-policies="${POLICY_NAME}")
fi

try_launch_pair() {
    local shape="$1"
    local disk_type="$2"
    local tier_flag="$3"
    local maint_flag="$4"

    for z_suffix in a b c f; do
        [[ -f "${VXFEC_CANCEL_FILE}" ]] && return 1
        local zone="${GCP_REGION}-${z_suffix}"
        if vxfec_vm_exists "${NODE0}" "${zone}" || vxfec_vm_exists "${NODE1}" "${zone}"; then
            echo "[06_run_200gbps_tier1_gvnic] Skipping ${zone}: ${NODE0} or ${NODE1} already exists."
            continue
        fi
        echo "[06_run_200gbps_tier1_gvnic] Trying 2x ${shape} (${tier_flag:-DEFAULT_200G}) in ${zone}..."
        local extra_flags=(--tags="${BENCH_NET_TAG}" --labels="vxfec-run=${VXFEC_RUN_TOKEN}")
        [[ -n "${tier_flag}" ]] && extra_flags+=(--network-performance-configs="total-egress-bandwidth-tier=${tier_flag}")
        [[ -n "${maint_flag}" ]] && extra_flags+=(--maintenance-policy="${maint_flag}")

        vxfec_add_unique VXFEC_CREATED_VMS "${NODE0}:${zone}"
        vxfec_add_unique VXFEC_CREATED_VMS "${NODE1}:${zone}"
        if vxfec_run_create gcloud compute instances create "${NODE0}" "${NODE1}" \
            --project="${GCP_PROJECT}" \
            --zone="${zone}" \
            --machine-type="${shape}" \
            --boot-disk-size=40GB \
            --boot-disk-type="${disk_type}" \
            --image-family=ubuntu-2204-lts \
            --image-project=ubuntu-os-cloud \
            "${NIC_ARGS[@]}" \
            "${POLICY_ARG[@]}" \
            "${extra_flags[@]}"; then
            ACTIVE_ZONE="${zone}"
            SELECTED_SHAPE="${shape}${tier_flag:+ (${tier_flag})}"
            return 0
        fi
        vxfec_cleanup_failed_attempt "${zone}" "${NODE0}" "${NODE1}"
        [[ -f "${VXFEC_CANCEL_FILE}" ]] && return 1
        if [[ ${#POLICY_ARG[@]} -gt 0 ]] && vxfec_run_create gcloud compute instances create "${NODE0}" "${NODE1}" \
            --project="${GCP_PROJECT}" \
            --zone="${zone}" \
            --machine-type="${shape}" \
            --boot-disk-size=40GB \
            --boot-disk-type="${disk_type}" \
            --image-family=ubuntu-2204-lts \
            --image-project=ubuntu-os-cloud \
            "${NIC_ARGS[@]}" \
            "${extra_flags[@]}"; then
            ACTIVE_ZONE="${zone}"
            SELECTED_SHAPE="${shape}${tier_flag:+ (${tier_flag})}"
            return 0
        fi
        vxfec_cleanup_failed_attempt "${zone}" "${NODE0}" "${NODE1}"
    done
    return 1
}

# Default to n2-standard-96 (TIER_1: 100 Gbps/VM = 200 Gbps full-duplex across 4x gVNICs);
# set TRY_GEN4_SHAPES=1 when CPUS_PER_VM_FAMILY > 24 is unlocked for H3 / C4 / C3D.
LAUNCHED=0
if [[ "${TRY_GEN4_SHAPES:-0}" == "1" ]]; then
    if try_launch_pair "h3-standard-88" "pd-balanced" "" "TERMINATE" || \
       try_launch_pair "c4-standard-96" "hyperdisk-balanced" "TIER_1" "" || \
       try_launch_pair "c3d-standard-90" "pd-balanced" "TIER_1" ""; then
        LAUNCHED=1
    fi
fi
if [[ "${LAUNCHED}" == "0" ]] && ! try_launch_pair "n2-standard-96" "pd-balanced" "TIER_1" "TERMINATE"; then
    echo "ERROR: Could not launch high-bandwidth TIER_1 pair in ${GCP_REGION}" >&2
    exit 1
fi

echo "[06_run_200gbps_tier1_gvnic] Launched 2x ${SELECTED_SHAPE} in ${ACTIVE_ZONE}!"

TARBALL="$(umask 077 && mktemp /tmp/vxfec_200g_XXXXXX.tar.gz)"
vxfec_package_repo_tarball "${REPO_ROOT}" "${TARBALL}"

prep_node() {
    local vm="$1"
    for _ in $(seq 1 36); do
        gcloud compute ssh "${vm}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" --command="true" >/dev/null 2>&1 && break
        sleep 4
    done
    gcloud compute scp "${TARBALL}" "${vm}:/tmp/vxfec_src.tar.gz" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}"
    gcloud compute ssh "${vm}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" --command="
        set -euo pipefail
        # Optional guest dead-man auto-poweroff safety net (disabled when KEEP_VMS=1 or AUTO_POWEROFF_MINS=0)
        if [[ '${AUTO_POWEROFF_MINS}' -gt 0 ]]; then
            sudo systemd-run --on-active='${AUTO_POWEROFF_MINS}m' /sbin/poweroff >/dev/null 2>&1 || true
        fi
        for _ in 1 2 3 4 5; do
            sudo apt-get -o DPkg::Lock::Timeout=120 update -qq && \
            sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=120 install -y -qq --no-install-recommends \
                build-essential ca-certificates pkg-config libvulkan-dev mesa-vulkan-drivers \
                libibverbs-dev iproute2 ethtool && break
            sleep 5
        done
        sudo sysctl -w net.core.rmem_max=268435456 net.core.wmem_max=268435456 \
                       net.core.rmem_default=33554432 net.core.wmem_default=33554432 \
                       net.core.netdev_max_backlog=250000 net.core.optmem_max=4194304 \
                       net.ipv4.conf.all.rp_filter=0 net.ipv4.conf.default.rp_filter=0 >/dev/null
        rm -rf ~/vxfec && mkdir -p ~/vxfec
        tar -xzf /tmp/vxfec_src.tar.gz -C ~/vxfec
        cd ~/vxfec && make -j\$(nproc) bench_wire_net bench_nccl
    "
}

prep_node "${NODE0}" & P0=$!
prep_node "${NODE1}" & P1=$!
VXFEC_WORK_PIDS+=("${P0}" "${P1}")
wait "${P0}"
wait "${P1}"
rm -f "${TARBALL}"

# Match all 4 Titanium gVNIC interfaces by hardware MAC address and bring up 8896B Jumbo MTU
NIC_CFG0="$(vxfec_configure_vm_nics "${NODE0}" "${ACTIVE_ZONE}")"
NIC_CFG1="$(vxfec_configure_vm_nics "${NODE1}" "${ACTIVE_ZONE}")"
IPS0="${NIC_CFG0%%|*}"
IFACES0="${NIC_CFG0#*|}"
IPS1="${NIC_CFG1%%|*}"
IFACES1="${NIC_CFG1#*|}"
IP0_PRIMARY="$(echo "${IPS0}" | cut -d',' -f1)"

gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" --command="
    {
        echo '=== Selected Shape: ${SELECTED_SHAPE} in ${ACTIVE_ZONE} ==='
        lscpu | grep -E 'Model name|CPU\(s\):|NUMA node'
        ip -br addr show
        for d in \$(ls /sys/class/net | grep -v '^lo\$' | sort); do
            echo \"--- ethtool -i \${d} ---\"
            ethtool -i \"\${d}\" || true
            cat /sys/class/net/\"\${d}\"/speed 2>/dev/null || true
        done
        IFS=',' read -r -a PIPS <<< '${IPS1}'
        for pip in \"\${PIPS[@]}\"; do
            ping -c 3 -i 0.05 -M do -s 8800 \"\${pip}\"
        done
    } | tee ~/vxfec/tier1_topology.txt
"

NCCL_ONLY="${NCCL_ONLY:-0}"
for arg in "$@"; do
    [[ "${arg}" == "--nccl-only" ]] && NCCL_ONLY=1
done
if [[ "${NCCL_ONLY}" == "1" ]]; then
    rm -f "${OUT_DIR}/tier1_topology.txt" "${OUT_DIR}"/tier1_nccl_*
else
    rm -f "${OUT_DIR}"/tier1_*.log "${OUT_DIR}"/tier1_*.csv "${OUT_DIR}"/tier1_*.txt
fi

if [[ "${NCCL_ONLY}" != "1" ]]; then
# ---------------------------------------------------------------------------
# 1) Unidirectional 4-NIC (64-Rail, 8192B Jumbo) Wire Saturation
# ---------------------------------------------------------------------------
echo "[1/3] Unidirectional 4-NIC Titanium gVNIC (${IPS0} -> ${IPS1}, 64 rails, 400 MiB superframe)..."
for spec in "rateless 0 --wire-only 256 0" \
            "rateless 0 --wire-only 256 98" \
            "rateless 10 --wire-only 256 98" \
            "mds 10 --wire-only 256 98" \
            "rateless 10 full 64 98"; do
    read -r codec bpct mode zval rate_g <<<"${spec}"
    WFLAG=""
    [[ "${mode}" == "--wire-only" ]] && WFLAG="--wire-only"
    gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" \
        --command="cd ~/vxfec && ./bench_wire_net --mode rx --bind ${IPS1} --port 31500 --ports-per-ip 16" &
    RX_PID=$!
    VXFEC_WORK_PIDS+=("${RX_PID}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" \
        --command="cd ~/vxfec && ./bench_wire_net --mode tx --peers ${IPS1} --port 31500 --ports-per-ip 16 --codec ${codec} --z ${zval} --k 200 --p 48 --sym-t 8192 --iters 5 --blackout-pct ${bpct} --rate-gbps ${rate_g} ${WFLAG} --csv ~/vxfec/tier1_unidir.csv" | tee -a "${OUT_DIR}/tier1_unidir.log" || STEP_ERR=1
    wait "${RX_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1
done

# ---------------------------------------------------------------------------
# 2) Full-Duplex Simultaneous Bidirectional 4-NIC Saturation (Node0 <-> Node1)
# ---------------------------------------------------------------------------
echo "[2/3] Simultaneous Full-Duplex Bidirectional 4-NIC Titanium gVNIC Saturation (${NODE0} <-> ${NODE1})..."
for bidir_spec in "98 10" "0 0"; do
    read -r brate bdrop <<<"${bidir_spec}"
    echo "--- Bidirectional pass: rate=${brate} Gbps/dir, blackout=${bdrop}% ---"
    gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" \
        --command="cd ~/vxfec && ./bench_wire_net --mode rx --bind ${IPS1} --port 31500 --ports-per-ip 12" &
    RX1=$!
    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" \
        --command="cd ~/vxfec && ./bench_wire_net --mode rx --bind ${IPS0} --port 32500 --ports-per-ip 12" &
    RX0=$!
    VXFEC_WORK_PIDS+=("${RX1}" "${RX0}")
    sleep 1

    ( gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" \
        --command="cd ~/vxfec && ./bench_wire_net --mode tx --peers ${IPS1} --port 31500 --ports-per-ip 12 --codec rateless --z 256 --k 200 --p 48 --sym-t 8192 --iters 5 --blackout-pct ${bdrop} --rate-gbps ${brate} --wire-only --bidir-sync ${IP0_PRIMARY}:31490 --bidir-leader --csv ~/vxfec/tier1_bidir_dirA.csv" | tee -a "${OUT_DIR}/tier1_bidir_dirA.log" ) &
    TX0=$!
    ( gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" \
        --command="cd ~/vxfec && ./bench_wire_net --mode tx --peers ${IPS0} --port 32500 --ports-per-ip 12 --codec rateless --z 256 --k 200 --p 48 --sym-t 8192 --iters 5 --blackout-pct ${bdrop} --rate-gbps ${brate} --wire-only --bidir-sync ${IP0_PRIMARY}:31490 --csv ~/vxfec/tier1_bidir_dirB.csv" | tee -a "${OUT_DIR}/tier1_bidir_dirB.log" ) &
    TX1=$!
    VXFEC_WORK_PIDS+=("${TX0}" "${TX1}")

    BIDIR_ERR=0
    wait "${TX0}" || BIDIR_ERR=1
    wait "${TX1}" || BIDIR_ERR=1
    wait "${RX1}" || BIDIR_ERR=1
    wait "${RX0}" || BIDIR_ERR=1
    [[ "${BIDIR_ERR}" == "0" ]] || exit 1
done
fi

# ---------------------------------------------------------------------------
# 3) 4-NIC (ndev=4, nchan=16) 2-Node Collective AllReduce (bench_nccl, up to 64 MiB)
# ---------------------------------------------------------------------------
echo "[3/3] 4-NIC (ndev=4, nchan=16) 2-Node Collective AllReduce..."
gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" \
    --command="cd ~/vxfec && VXFEC_NCCL_IFNAMES=${IFACES0} VXFEC_BENCH_CHANNELS=16 ./bench_nccl --rank 0 --master ${IP0_PRIMARY}:29600 --iters 30 --csv ~/vxfec/tier1_nccl_4nic.csv | tee ~/vxfec/tier1_nccl_4nic.txt" &
N0=$!
VXFEC_WORK_PIDS+=("${N0}")
sleep 1
STEP_ERR=0
gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" \
    --command="cd ~/vxfec && VXFEC_NCCL_IFNAMES=${IFACES1} VXFEC_BENCH_CHANNELS=16 ./bench_nccl --rank 1 --master ${IP0_PRIMARY}:29600 --iters 30" || STEP_ERR=1
wait "${N0}" || STEP_ERR=1
[[ "${STEP_ERR}" == "0" ]] || exit 1

# Also run 64 MiB multi-channel scaling (nchan=4, 16, 32 across 0% and 2% Burst=6) across 4x 100G Titanium gVNICs
for spec in "4~gbn~64 MiB | 4x gVNIC nchan=4 Lossless (GBN)~67108864~0~1" \
            "4~auto~64 MiB | 4x gVNIC nchan=4 Lossless (vxfec)~67108864~0~1" \
            "16~gbn~64 MiB | 4x gVNIC nchan=16 Lossless (GBN)~67108864~0~1" \
            "16~auto~64 MiB | 4x gVNIC nchan=16 Lossless (vxfec)~67108864~0~1" \
            "32~gbn~64 MiB | 4x gVNIC nchan=32 Lossless (GBN)~67108864~0~1" \
            "32~auto~64 MiB | 4x gVNIC nchan=32 Lossless (vxfec)~67108864~0~1" \
            "32~sack~64 MiB | 4x gVNIC nchan=32 2% Burst=6 (SACK)~67108864~20000~6" \
            "32~auto~64 MiB | 4x gVNIC nchan=32 2% Burst=6 (vxfec)~67108864~20000~6"; do
    IFS='~' read -r nch pol lbl bsz lppm brst <<<"${spec}"
    ( gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" \
        --command="cd ~/vxfec && VXFEC_NCCL_IFNAMES=${IFACES0} VXFEC_BENCH_CHANNELS=${nch} ./bench_nccl --rank 0 --master ${IP0_PRIMARY}:29605 --single-codec ${pol} --label '${lbl}' --bytes ${bsz} --loss-ppm ${lppm} --burst ${brst} --iters 16 --csv ~/vxfec/tier1_nccl_64mb.csv" | tee -a "${OUT_DIR}/tier1_nccl_64mb.log" ) &
    M0=$!
    VXFEC_WORK_PIDS+=("${M0}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}" \
        --command="cd ~/vxfec && VXFEC_NCCL_IFNAMES=${IFACES1} VXFEC_BENCH_CHANNELS=${nch} ./bench_nccl --rank 1 --master ${IP0_PRIMARY}:29605 --single-codec ${pol} --label '${lbl}' --bytes ${bsz} --loss-ppm ${lppm} --burst ${brst} --iters 16" || STEP_ERR=1
    wait "${M0}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1
done

gcloud compute scp "${NODE0}:~/vxfec/tier1_*" "${OUT_DIR}/" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}"
if [[ "${NCCL_ONLY}" != "1" ]]; then
    gcloud compute scp "${NODE1}:~/vxfec/tier1_bidir_dirB.csv" "${OUT_DIR}/" --project="${GCP_PROJECT}" --zone="${ACTIVE_ZONE}"
fi
echo "[06_run_200gbps_tier1_gvnic] Saved all 200G TIER_1 artifacts to ${OUT_DIR}."
