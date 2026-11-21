#!/usr/bin/env bash
# scripts/gcp/03_run_tier2_a3_h100.sh
# Provision GCP A3 (NVIDIA H100 80GB HBM3 + PCIe Gen5 + multi-NIC gVNIC) Spot or
# DWS Flex-Start VMs (--max-run-duration=1h), run HBM3/Gen5 + multi-NIC 200+ Gbps
# superframe & NCCL AllReduce benchmarks, pull CSVs/logs, and tear down.
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
A3_MACHINE_TYPE="${A3_MACHINE_TYPE:-a3-highgpu-2g}"
NUM_NODES="${NUM_NODES:-2}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
VPC_NAME="${VPC_NAME:-vxfec-bench-vpc}"
SUBNET_NAME="${SUBNET_NAME:-vxfec-bench-subnet-${GCP_REGION}}"
BENCH_NET_TAG="${BENCH_NET_TAG:-vxfec-bench}"
export VPC_NAME SUBNET_NAME BENCH_NET_TAG
KEEP_VMS="${KEEP_VMS:-0}"

OUT_DIR="${REPO_ROOT}/docs/cloud_runs/gcp_a3_h100"
mkdir -p "${OUT_DIR}"

NODE0="${NODE0:-vxfec-bench-a3-0}"
NODE1="${NODE1:-vxfec-bench-a3-1}"
TARBALL=""
VXFEC_CLEANUP_TAG="03_run_tier2_a3_h100"
# shellcheck source=scripts/gcp/lib_cleanup.sh
source "${REPO_ROOT}/scripts/gcp/lib_cleanup.sh"
vxfec_setup_cleanup_traps

"${REPO_ROOT}/scripts/gcp/01_setup_vpc_jumbo.sh"

# Determine required secondary GPU NICs for A3 shapes:
# a3-highgpu-1g: 1 GPU NIC, a3-highgpu-2g: 1 GPU NIC, a3-highgpu-4g: 2 GPU NICs, a3-highgpu-8g: 4 GPU NICs
GPU_NICS=1
case "${A3_MACHINE_TYPE}" in
    a3-highgpu-4g) GPU_NICS=2 ;;
    a3-highgpu-8g|a3-megagpu-8g) GPU_NICS=4 ;;
esac

NIC_FLAGS=(--network-interface="nic-type=GVNIC,network=${VPC_NAME},subnet=${SUBNET_NAME}")
for idx in $(seq 1 "${GPU_NICS}"); do
    GVPC="vxfec-bench-${VXFEC_RUN_TOKEN}-gpunic${idx}"
    GSUB="vxfec-bench-${VXFEC_RUN_TOKEN}-gpusub${idx}"
    GFW="${GVPC}-internal"
    GCIDR="10.$((128 + idx)).0.0/20"
    vxfec_add_unique VXFEC_CREATED_VPCS "${GVPC}"
    vxfec_run_create gcloud compute networks create "${GVPC}" --project="${GCP_PROJECT}" --subnet-mode=custom --mtu=8896
    vxfec_add_unique VXFEC_CREATED_SUBNETS "${GSUB}:${GCP_REGION}:${GVPC}"
    vxfec_run_create gcloud compute networks subnets create "${GSUB}" \
        --project="${GCP_PROJECT}" --network="${GVPC}" --region="${GCP_REGION}" --range="${GCIDR}"
    vxfec_add_unique VXFEC_CREATED_FWS "${GFW}:${GVPC}"
    vxfec_run_create gcloud compute firewall-rules create "${GFW}" --project="${GCP_PROJECT}" --network="${GVPC}" \
        --allow=tcp:1-65535,udp:1-65535,icmp --source-ranges="${GCIDR}" --target-tags="${BENCH_NET_TAG}"
    NIC_FLAGS+=(--network-interface="nic-type=GVNIC,network=${GVPC},subnet=${GSUB},no-address")
done

PROV_FLAGS=(--maintenance-policy=TERMINATE)
if [[ "${PROVISIONING_MODEL}" == "SPOT" ]]; then
    PROV_FLAGS=(
        --provisioning-model=SPOT
        --instance-termination-action=DELETE
        --max-run-duration=1h
        --maintenance-policy=TERMINATE
    )
elif [[ "${PROVISIONING_MODEL}" == "FLEX_START" ]]; then
    PROV_FLAGS=(
        --provisioning-model=FLEX_START
        --instance-termination-action=DELETE
        --max-run-duration=1h
        --maintenance-policy=TERMINATE
    )
fi

IMAGE_ARGS=(
    --image-family=ubuntu-accelerator-2404-amd64-with-nvidia-580
    --image-project=ubuntu-os-accelerator-images
)

NODES=("${NODE0}")
[[ "${NUM_NODES}" -ge 2 ]] && NODES+=("${NODE1}")

for vm in "${NODES[@]}"; do
    if vxfec_vm_exists "${vm}" "${GCP_ZONE}"; then
        echo "ERROR: Instance ${vm} already exists in ${GCP_ZONE}; refusing to clobber." >&2
        exit 1
    fi
    echo "[03_run_tier2_a3_h100] Creating ${vm} (${A3_MACHINE_TYPE}, ${PROVISIONING_MODEL}) in ${GCP_ZONE}..."
    vxfec_add_unique VXFEC_CREATED_VMS "${vm}:${GCP_ZONE}"
    vxfec_run_create gcloud compute instances create "${vm}" \
        --project="${GCP_PROJECT}" \
        --zone="${GCP_ZONE}" \
        --machine-type="${A3_MACHINE_TYPE}" \
        --tags="${BENCH_NET_TAG}" \
        --labels="vxfec-run=${VXFEC_RUN_TOKEN}" \
        "${NIC_FLAGS[@]}" \
        --boot-disk-size=150GB \
        --boot-disk-type=pd-ssd \
        "${IMAGE_ARGS[@]}" \
        "${PROV_FLAGS[@]}"
done

TARBALL="$(umask 077 && mktemp /tmp/vxfec_a3_src_XXXXXX.tar.gz)"
vxfec_package_repo_tarball "${REPO_ROOT}" "${TARBALL}"

for vm in "${NODES[@]}"; do
    vxfec_wait_nvidia_ssh "${vm}" "${GCP_ZONE}" 48
    vxfec_bootstrap_nvidia_vm "${vm}" "${GCP_ZONE}" "${TARBALL}" 134217728
done
rm -f "${TARBALL}"

echo "[03_run_tier2_a3_h100] Running H100 80GB HBM3 + PCIe Gen5 benchmarks on ${NODE0}..."
gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" --command="
    set -euo pipefail
    cd ~/vxfec
    nvidia-smi | tee a3_nvidia_smi.txt
    nvidia-smi topo -m | tee a3_nvidia_topo.txt
    vulkaninfo --summary | tee a3_vulkaninfo.txt
    make check | tee a3_make_check.txt
    ./bench_vulkan | tee a3_bench_vulkan.txt
    ./bench_pipeline | tee a3_bench_pipeline.txt
    ./demo_nccl_e2e --max-mb 32 --csv ~/vxfec/a3_nccl_intranode.csv | tee a3_nccl_intranode.txt
"

if [[ "${NUM_NODES}" -ge 2 ]]; then
    NIC_CFG0="$(vxfec_configure_vm_nics "${NODE0}" "${GCP_ZONE}")"
    NIC_CFG1="$(vxfec_configure_vm_nics "${NODE1}" "${GCP_ZONE}")"
    IP0_LIST="${NIC_CFG0%%|*}"
    IFACES0="${NIC_CFG0#*|}"
    IP1_LIST="${NIC_CFG1%%|*}"
    IFACES1="${NIC_CFG1#*|}"
    MASTER_IP0="$(echo "${IP0_LIST}" | cut -d',' -f1)"
    PEER_IP1="$(echo "${IP1_LIST}" | cut -d',' -f1)"
    NIC0="$(gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="ip -o -4 route get '${PEER_IP1}' | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\"){print \$(i+1); exit}}'")"
    NIC1="$(gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="ip -o -4 route get '${MASTER_IP0}' | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\"){print \$(i+1); exit}}'")"
    : "${NIC0:?Failed to resolve routable interface on ${NODE0} toward ${PEER_IP1}}"
    : "${NIC1:?Failed to resolve routable interface on ${NODE1} toward ${MASTER_IP0}}"
    : "${IFACES0:=${NIC0}}"
    : "${IFACES1:=${NIC1}}"

    echo "[03_run_tier2_a3_h100] Running 2-node multi-NIC gVNIC superframe saturation (${IP0_LIST} -> ${IP1_LIST})..."
    for spec in "rateless 0" "rateless 10" "mds 10"; do
        read -r codec bpct <<<"${spec}"
        gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
            --command="cd ~/vxfec && ./bench_wire_net --mode rx --bind ${IP1_LIST} --port 31500 --ports-per-ip 4" &
        RX_PID=$!
        VXFEC_WORK_PIDS+=("${RX_PID}")
        sleep 1
        STEP_ERR=0
        gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
            --command="cd ~/vxfec && ./bench_wire_net --mode tx --peers ${IP1_LIST} --port 31500 --ports-per-ip 4 --codec ${codec} --z 128 --k 200 --p 48 --sym-t 8192 --iters 5 --blackout-pct ${bpct} --csv ~/vxfec/a3_wire_multinic.csv" | tee -a "${OUT_DIR}/a3_wire_multinic.log" || STEP_ERR=1
        wait "${RX_PID}" || STEP_ERR=1
        [[ "${STEP_ERR}" == "0" ]] || exit 1
    done

    echo "[03_run_tier2_a3_h100] Running 2-node H100 multi-NIC NCCL AllReduce (${IFACES0} <-> ${IFACES1})..."
    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="cd ~/vxfec && NCCL_SOCKET_IFNAME='${NIC0}' ./demo_nccl_e2e --rank 0 --nranks 2 --master ${MASTER_IP0}:29500 --ifname '${IFACES0}' --max-mb 32 --csv ~/vxfec/a3_nccl_multinode.csv | tee ~/vxfec/a3_nccl_multinode.txt" &
    R0_PID=$!
    VXFEC_WORK_PIDS+=("${R0_PID}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="cd ~/vxfec && NCCL_SOCKET_IFNAME='${NIC1}' ./demo_nccl_e2e --rank 1 --nranks 2 --master ${MASTER_IP0}:29500 --ifname '${IFACES1}' --max-mb 32" || STEP_ERR=1
    wait "${R0_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1
fi

gcloud compute scp "${NODE0}:~/vxfec/a3_*" "${OUT_DIR}/" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}"
echo "[03_run_tier2_a3_h100] Complete! Results saved in ${OUT_DIR}."
