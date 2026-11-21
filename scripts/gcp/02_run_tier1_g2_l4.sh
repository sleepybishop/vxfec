#!/usr/bin/env bash
# scripts/gcp/02_run_tier1_g2_l4.sh
# Provision 2x GCP G2 (NVIDIA L4 + 32 Gbps gVNIC + 8896 B Jumbo VPC) Spot VMs,
# build vxfec, run single-node Vulkan + 2-node gVNIC superframe & CUDA/NCCL E2E
# benchmarks (intra-zone + optional cross-zone DCI), collect CSVs/logs, and tear down.
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
    [[ -x "$HOME/google-cloud-sdk/bin/gcloud" ]] && export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    [[ -x "/tmp/google-cloud-sdk/bin/gcloud" ]] && export PATH="/tmp/google-cloud-sdk/bin:$PATH"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
: "${GCP_PROJECT:?Set GCP_PROJECT=<project-id>}"
GCP_REGION="${GCP_REGION:-us-central1}"
GCP_ZONE="${GCP_ZONE:-${GCP_REGION}-a}"
GCP_ZONE_B="${GCP_ZONE_B:-${GCP_REGION}-b}"
G2_MACHINE_TYPE="${G2_MACHINE_TYPE:-g2-standard-32}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
VPC_NAME="${VPC_NAME:-vxfec-bench-vpc}"
SUBNET_NAME="${SUBNET_NAME:-vxfec-bench-subnet-${GCP_REGION}}"
POLICY_NAME="${POLICY_NAME:-vxfec-bench-compact-${GCP_REGION}}"
BENCH_NET_TAG="${BENCH_NET_TAG:-vxfec-bench}"
export VPC_NAME SUBNET_NAME POLICY_NAME BENCH_NET_TAG
CROSS_ZONE="${CROSS_ZONE:-1}"
KEEP_VMS="${KEEP_VMS:-0}"

OUT_DIR="${REPO_ROOT}/docs/cloud_runs/gcp_g2_l4"
mkdir -p "${OUT_DIR}"

NODE0="${NODE0:-vxfec-bench-g2-0}"
NODE1="${NODE1:-vxfec-bench-g2-1}"
NODE2="${NODE2:-vxfec-bench-g2-xzone}"
TARBALL=""
VXFEC_CLEANUP_TAG="02_run_tier1_g2_l4"
# shellcheck source=scripts/gcp/lib_cleanup.sh
source "${REPO_ROOT}/scripts/gcp/lib_cleanup.sh"
vxfec_setup_cleanup_traps

"${REPO_ROOT}/scripts/gcp/01_setup_vpc_jumbo.sh"

POLICY_FLAG=()
if [[ "${PROVISIONING_MODEL}" == "STANDARD" ]] && gcloud compute resource-policies describe "${POLICY_NAME}" \
    --project="${GCP_PROJECT}" --region="${GCP_REGION}" >/dev/null 2>&1; then
    POLICY_FLAG=(--resource-policies="${POLICY_NAME}")
fi

SPOT_FLAGS=()
if [[ "${PROVISIONING_MODEL}" == "SPOT" ]]; then
    SPOT_FLAGS=(
        --provisioning-model=SPOT
        --instance-termination-action=DELETE
        --max-run-duration=2h
    )
else
    SPOT_FLAGS=(--maintenance-policy=TERMINATE)
fi

IMAGE_ARGS=(
    --image-family=ubuntu-accelerator-2404-amd64-with-nvidia-580
    --image-project=ubuntu-os-accelerator-images
)

echo "[02_run_tier1_g2_l4] Launching ${NODE0} and ${NODE1} (${G2_MACHINE_TYPE}, ${PROVISIONING_MODEL}) in ${GCP_ZONE}..."
for vm in "${NODE0}" "${NODE1}"; do
    if vxfec_vm_exists "${vm}" "${GCP_ZONE}"; then
        echo "ERROR: Instance ${vm} already exists in ${GCP_ZONE}; refusing to clobber." >&2
        exit 1
    fi
    vxfec_add_unique VXFEC_CREATED_VMS "${vm}:${GCP_ZONE}"
    vxfec_run_create gcloud compute instances create "${vm}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --machine-type="${G2_MACHINE_TYPE}" \
        --tags="${BENCH_NET_TAG}" \
        --labels="vxfec-run=${VXFEC_RUN_TOKEN}" \
        --network-interface="nic-type=GVNIC,network=${VPC_NAME},subnet=${SUBNET_NAME}" \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-ssd \
        "${IMAGE_ARGS[@]}" \
        "${SPOT_FLAGS[@]}" \
        "${POLICY_FLAG[@]}"
done

if [[ "${CROSS_ZONE}" == "1" ]]; then
    if vxfec_vm_exists "${NODE2}" "${GCP_ZONE_B}"; then
        echo "[02_run_tier1_g2_l4] ${NODE2} already exists in ${GCP_ZONE_B}; skipping cross-zone launch."
        CROSS_ZONE=0
    else
        echo "[02_run_tier1_g2_l4] Launching cross-zone peer ${NODE2} in ${GCP_ZONE_B}..."
        vxfec_add_unique VXFEC_CREATED_VMS "${NODE2}:${GCP_ZONE_B}"
        if ! vxfec_run_create gcloud compute instances create "${NODE2}" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE_B}" \
            --machine-type="${G2_MACHINE_TYPE}" \
            --tags="${BENCH_NET_TAG}" \
            --labels="vxfec-run=${VXFEC_RUN_TOKEN}" \
            --network-interface="nic-type=GVNIC,network=${VPC_NAME},subnet=${SUBNET_NAME}" \
            --boot-disk-size=100GB \
            --boot-disk-type=pd-ssd \
            "${IMAGE_ARGS[@]}" \
            "${SPOT_FLAGS[@]}"; then
            vxfec_cleanup_failed_attempt "${GCP_ZONE_B}" "${NODE2}"
            CROSS_ZONE=0
        fi
    fi
fi

vxfec_wait_nvidia_ssh "${NODE0}" "${GCP_ZONE}" 36
vxfec_wait_nvidia_ssh "${NODE1}" "${GCP_ZONE}" 36
[[ "${CROSS_ZONE}" == "1" ]] && vxfec_wait_nvidia_ssh "${NODE2}" "${GCP_ZONE_B}" 36

TARBALL="$(umask 077 && mktemp /tmp/vxfec_src_XXXXXX.tar.gz)"
vxfec_package_repo_tarball "${REPO_ROOT}" "${TARBALL}"

vxfec_bootstrap_nvidia_vm "${NODE0}" "${GCP_ZONE}" "${TARBALL}" 67108864
vxfec_bootstrap_nvidia_vm "${NODE1}" "${GCP_ZONE}" "${TARBALL}" 67108864
[[ "${CROSS_ZONE}" == "1" ]] && vxfec_bootstrap_nvidia_vm "${NODE2}" "${GCP_ZONE_B}" "${TARBALL}" 67108864
rm -f "${TARBALL}"

IP0="$(gcloud compute instances describe "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" --format='get(networkInterfaces[0].networkIP)')"
IP1="$(gcloud compute instances describe "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" --format='get(networkInterfaces[0].networkIP)')"
NIC0="$(gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" --command="ip -o -4 route show to default | awk '{print \$5; exit}'")"
NIC1="$(gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" --command="ip -o -4 route show to default | awk '{print \$5; exit}'")"

echo "[02_run_tier1_g2_l4] Running single-node NVIDIA L4 Vulkan + PCIe Gen4 suite on ${NODE0}..."
gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" --command="
    set -euo pipefail
    cd ~/vxfec
    nvidia-smi | tee g2_nvidia_smi.txt
    vulkaninfo --summary | tee g2_vulkaninfo.txt
    ip link show ${NIC0} | tee g2_nic_mtu.txt
    make check | tee g2_make_check.txt
    ./bench_vulkan | tee g2_bench_vulkan.txt
    ./bench_pipeline | tee g2_bench_pipeline.txt
    ./bench_nccl | tee g2_bench_nccl_loopback.txt
"

echo "[02_run_tier1_g2_l4] Running 2-node intra-zone gVNIC (8896B Jumbo) superframe saturation (${IP0} -> ${IP1})..."
for spec in "rateless 0" "rateless 10" "mds 10" "rateless 20"; do
    read -r codec bpct <<<"${spec}"
    gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="cd ~/vxfec && ./bench_wire_net --mode rx --bind ${IP1} --port 31500 --ports-per-ip 8" &
    RX_PID=$!
    VXFEC_WORK_PIDS+=("${RX_PID}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="cd ~/vxfec && ./bench_wire_net --mode tx --peers ${IP1} --port 31500 --ports-per-ip 8 --codec ${codec} --z 64 --k 200 --p 48 --sym-t 8192 --iters 5 --blackout-pct ${bpct} --csv ~/vxfec/g2_wire_intrazone.csv" | tee -a "${OUT_DIR}/g2_wire_intrazone.log" || STEP_ERR=1
    wait "${RX_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1
done

echo "[02_run_tier1_g2_l4] Running 2-node physical CUDA + NCCL AllReduce E2E (${IP0} <-> ${IP1})..."
gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
    --command="cd ~/vxfec && ./demo_nccl_e2e --rank 0 --nranks 2 --master ${IP0}:29500 --ifname ${NIC0} --max-mb 16 --csv ~/vxfec/g2_nccl_intrazone.csv | tee ~/vxfec/g2_nccl_intrazone.txt" &
R0_PID=$!
VXFEC_WORK_PIDS+=("${R0_PID}")
sleep 1
STEP_ERR=0
gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
    --command="cd ~/vxfec && ./demo_nccl_e2e --rank 1 --nranks 2 --master ${IP0}:29500 --ifname ${NIC1} --max-mb 16" || STEP_ERR=1
wait "${R0_PID}" || STEP_ERR=1
[[ "${STEP_ERR}" == "0" ]] || exit 1

if [[ "${CROSS_ZONE}" == "1" ]]; then
    IP2="$(gcloud compute instances describe "${NODE2}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_B}" --format='get(networkInterfaces[0].networkIP)')"
    NIC2="$(gcloud compute ssh "${NODE2}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_B}" --command="ip -o -4 route show to default | awk '{print \$5; exit}'")"
    echo "[02_run_tier1_g2_l4] Measuring cross-zone DCI RTT (${GCP_ZONE} <-> ${GCP_ZONE_B}) and running AllReduce..."
    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="ping -c 10 -i 0.2 ${IP2} | tee ~/vxfec/g2_xzone_ping.txt" || true

    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="cd ~/vxfec && ./demo_nccl_e2e --rank 0 --nranks 2 --master ${IP0}:29510 --ifname ${NIC0} --max-mb 16 --csv ~/vxfec/g2_nccl_xzone.csv | tee ~/vxfec/g2_nccl_xzone.txt" &
    X0_PID=$!
    VXFEC_WORK_PIDS+=("${X0_PID}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODE2}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE_B}" \
        --command="cd ~/vxfec && ./demo_nccl_e2e --rank 1 --nranks 2 --master ${IP0}:29510 --ifname ${NIC2} --max-mb 16" || STEP_ERR=1
    wait "${X0_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1
fi

echo "[02_run_tier1_g2_l4] Pulling benchmark artifacts to ${OUT_DIR}..."
gcloud compute scp "${NODE0}:~/vxfec/g2_*" "${OUT_DIR}/" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}"
echo "[02_run_tier1_g2_l4] Complete! Results saved in ${OUT_DIR}."
