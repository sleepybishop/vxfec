#!/usr/bin/env bash
# scripts/gcp/04_run_tier3_rdma.sh
# Validate native libibverbs (IBV_QPT_UC + ibv_reg_dmabuf_mr zero-bounce GPUDirect RDMA)
# across either:
#   1) GCP A3-Ultra (--gcp-a3-ultra: 2x a3-ultragpu-8g[-nolssd], 16x H200 141GB HBM3e +
#      8x 400G ConnectX-7 MRDMA RoCEv2 VPC, with automatic EXIT/INT/TERM teardown), or
#   2) Generic SSH hosts with ConnectX-6/7 RoCEv2/IB (--hosts user@host0,user@host1 --ips ip0,ip1).
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
    [[ -x "$HOME/google-cloud-sdk/bin/gcloud" ]] && export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    [[ -x "/tmp/google-cloud-sdk/bin/gcloud" ]] && export PATH="/tmp/google-cloud-sdk/bin:$PATH"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/docs/cloud_runs/rdma_ibv"
mkdir -p "${OUT_DIR}"

HOSTS_CSV=""
IPS_CSV=""
GCP_A3_ULTRA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts) HOSTS_CSV="$2"; shift 2 ;;
        --ips) IPS_CSV="$2"; shift 2 ;;
        --gcp-a3-ultra) GCP_A3_ULTRA=1; shift ;;
        *) echo "Usage: $0 [--gcp-a3-ultra] | [--hosts user@h0,user@h1 --ips ip0,ip1]" >&2; exit 1 ;;
    esac
done

if [[ "${GCP_A3_ULTRA}" == "1" ]]; then
    GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
    : "${GCP_PROJECT:?Set GCP_PROJECT=<project-id>}"
    GCP_REGION="${GCP_REGION:-us-central1}"
    GCP_ZONE="${GCP_ZONE:-${GCP_REGION}-b}"
    A3_ULTRA_TYPE="${A3_ULTRA_TYPE:-a3-ultragpu-8g-nolssd}"
    PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
    A3_IMAGE_FAMILY="${A3_IMAGE_FAMILY:-ubuntu-accelerator-2404-amd64-with-nvidia-580}"
    MAX_RUN_DURATION="${MAX_RUN_DURATION:-45m}"
    VPC_NAME="${VPC_NAME:-vxfec-bench-vpc}"
    SUBNET_NAME="${SUBNET_NAME:-vxfec-bench-subnet-${GCP_REGION}}"
    BENCH_NET_TAG="${BENCH_NET_TAG:-vxfec-bench}"
    export VPC_NAME SUBNET_NAME BENCH_NET_TAG
    KEEP_VMS="${KEEP_VMS:-0}"

    OUT_DIR="${REPO_ROOT}/docs/cloud_runs/gcp_a3_ultra"
    mkdir -p "${OUT_DIR}"

    NODE0="${NODE0:-vxfec-bench-u0}"
    NODE1="${NODE1:-vxfec-bench-u1}"
    TARBALL=""
    VXFEC_CLEANUP_TAG="04_run_tier3_rdma_a3_ultra"
    # shellcheck source=scripts/gcp/lib_cleanup.sh
    source "${REPO_ROOT}/scripts/gcp/lib_cleanup.sh"
    vxfec_setup_cleanup_traps

    for vm in "${NODE0}" "${NODE1}"; do
        if vxfec_vm_exists "${vm}" "${GCP_ZONE}"; then
            echo "ERROR: Instance ${vm} already exists in ${GCP_ZONE}; refusing to clobber." >&2
            exit 1
        fi
    done

    # 1) Primary host gVNIC VPC (10.128.0.0/20, MTU 8896, SSH + internal TCP/UDP)
    "${REPO_ROOT}/scripts/gcp/01_setup_vpc_jumbo.sh"

    # 2) Secondary host gVNIC VPC (10.129.0.0/20, MTU 8896)
    GVNIC1_VPC="vxfec-bench-${VXFEC_RUN_TOKEN}-gvnic1"
    GVNIC1_SUB="vxfec-bench-${VXFEC_RUN_TOKEN}-gvsub1"
    GVNIC1_FW="${GVNIC1_VPC}-internal"
    vxfec_add_unique VXFEC_CREATED_VPCS "${GVNIC1_VPC}"
    vxfec_run_create gcloud compute networks create "${GVNIC1_VPC}" \
        --project="${GCP_PROJECT}" --subnet-mode=custom --mtu=8896
    vxfec_add_unique VXFEC_CREATED_SUBNETS "${GVNIC1_SUB}:${GCP_REGION}:${GVNIC1_VPC}"
    vxfec_run_create gcloud compute networks subnets create "${GVNIC1_SUB}" \
        --project="${GCP_PROJECT}" --network="${GVNIC1_VPC}" --region="${GCP_REGION}" --range="10.129.0.0/20"
    vxfec_add_unique VXFEC_CREATED_FWS "${GVNIC1_FW}:${GVNIC1_VPC}"
    vxfec_run_create gcloud compute firewall-rules create "${GVNIC1_FW}" \
        --project="${GCP_PROJECT}" --network="${GVNIC1_VPC}" \
        --allow=tcp:1-65535,udp:1-65535,icmp --source-ranges="10.129.0.0/20" --target-tags="${BENCH_NET_TAG}"

    # 3) Single zonal RoCEv2 MRDMA VPC (--network-profile=${GCP_ZONE}-vpc-roce, MTU 8896) + 8 CX-7 subnets
    #    Uses only 3 VPCs total (+ default = 4 <= 5 default NETWORKS quota). RoCE VPC profiles forbid firewall-rules.
    ROCE_VPC="vxfec-bench-${VXFEC_RUN_TOKEN}-roce"
    vxfec_add_unique VXFEC_CREATED_VPCS "${ROCE_VPC}"
    vxfec_run_create gcloud compute networks create "${ROCE_VPC}" \
        --project="${GCP_PROJECT}" \
        --subnet-mode=custom \
        --network-profile="projects/${GCP_PROJECT}/global/networkProfiles/${GCP_ZONE}-vpc-roce" \
        --mtu=8896

    NIC_FLAGS=(
        --network-interface="nic-type=GVNIC,network=${VPC_NAME},subnet=${SUBNET_NAME}"
        --network-interface="nic-type=GVNIC,network=${GVNIC1_VPC},subnet=${GVNIC1_SUB},no-address"
    )
    for idx in $(seq 0 7); do
        RSUB="vxfec-bench-${VXFEC_RUN_TOKEN}-rocesub${idx}"
        RCIDR="192.168.${idx}.0/24"
        vxfec_add_unique VXFEC_CREATED_SUBNETS "${RSUB}:${GCP_REGION}:${ROCE_VPC}"
        vxfec_run_create gcloud compute networks subnets create "${RSUB}" \
            --project="${GCP_PROJECT}" --network="${ROCE_VPC}" --region="${GCP_REGION}" --range="${RCIDR}"
        NIC_FLAGS+=(--network-interface="nic-type=MRDMA,network=${ROCE_VPC},subnet=${RSUB},no-address")
    done

    PROV_FLAGS=(--maintenance-policy=TERMINATE --no-restart-on-failure)
    if [[ "${PROVISIONING_MODEL}" == "SPOT" ]]; then
        PROV_FLAGS=(
            --provisioning-model=SPOT
            --instance-termination-action=DELETE
            --max-run-duration="${MAX_RUN_DURATION}"
            --maintenance-policy=TERMINATE
            --no-restart-on-failure
        )
    elif [[ "${PROVISIONING_MODEL}" == "FLEX_START" ]]; then
        PROV_FLAGS=(
            --provisioning-model=FLEX_START
            --request-valid-for-duration=2h
            --instance-termination-action=DELETE
            --max-run-duration="${MAX_RUN_DURATION}"
            --maintenance-policy=TERMINATE
            --no-restart-on-failure
        )
    fi

    for vm in "${NODE0}" "${NODE1}"; do
        echo "[04_run_tier3_rdma] Creating ${vm} (${A3_ULTRA_TYPE}, ${PROVISIONING_MODEL}) in ${GCP_ZONE}..."
        vxfec_add_unique VXFEC_CREATED_VMS "${vm}:${GCP_ZONE}"
        vxfec_run_create gcloud compute instances create "${vm}" \
            --project="${GCP_PROJECT}" \
            --zone="${GCP_ZONE}" \
            --machine-type="${A3_ULTRA_TYPE}" \
            --tags="${BENCH_NET_TAG}" \
            --labels="vxfec-run=${VXFEC_RUN_TOKEN}" \
            "${NIC_FLAGS[@]}" \
            --boot-disk-size=150GB \
            --boot-disk-type=hyperdisk-balanced \
            --image-family="${A3_IMAGE_FAMILY}" \
            --image-project=ubuntu-os-accelerator-images \
            "${PROV_FLAGS[@]}"
    done

    TARBALL="$(umask 077 && mktemp /tmp/vxfec_a3u_src_XXXXXX.tar.gz)"
    vxfec_package_repo_tarball "${REPO_ROOT}" "${TARBALL}"

    for vm in "${NODE0}" "${NODE1}"; do
        vxfec_wait_nvidia_ssh "${vm}" "${GCP_ZONE}" 60
        vxfec_bootstrap_nvidia_vm "${vm}" "${GCP_ZONE}" "${TARBALL}" 268435456
    done
    rm -f "${TARBALL}"
    TARBALL=""

    IP0="$(gcloud compute instances describe "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --format='value(networkInterfaces[0].networkIP)')"
    IP1="$(gcloud compute instances describe "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --format='value(networkInterfaces[0].networkIP)')"
    : "${IP0:?Failed to query primary VPC IP for ${NODE0}}"
    : "${IP1:?Failed to query primary VPC IP for ${NODE1}}"

    NIC0="$(gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="ip -o -4 route get '${IP1}' | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\"){print \$(i+1); exit}}'")"
    NIC1="$(gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="ip -o -4 route get '${IP0}' | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\"){print \$(i+1); exit}}'")"

    echo "[04_run_tier3_rdma] Running H200 141GB HBM3e + ConnectX-7 topology & resident-VRAM benchmarks on ${NODE0}..."
    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" --command="
        set -euo pipefail
        cd ~/vxfec
        nvidia-smi | tee rdma_nvidia_smi.txt
        nvidia-smi topo -m | tee rdma_nvidia_topo.txt
        ibv_devinfo | head -n 120 | tee rdma_ibv_devinfo.txt
        vulkaninfo --summary | tee rdma_vulkaninfo.txt
        make check | tee rdma_make_check.txt
        ./bench_vulkan | tee rdma_bench_vulkan.txt
        ./bench_pipeline | tee rdma_bench_pipeline.txt
    "

    echo "[04_run_tier3_rdma] Running 2-node ConnectX-7 400G RoCEv2 (IBV_QPT_UC + DMA-BUF VRAM) superframe saturation..."
    for spec in \
        "rateless-gf2 0 1 512 4096 8 20" \
        "rateless-gf2 10 1 512 4096 8 20" \
        "rateless 10 1 512 4096 8 20" \
        "mds 10 1 512 4096 8 20" \
        "rateless-gf2 10 0 256 4096 8 5"; do
        read -r codec bpct pipe zval symt qps iters <<<"${spec}"
        PIPE_ARG=()
        [[ "${pipe}" == "1" ]] && PIPE_ARG=(--pipeline)
        gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
            --command="cd ~/vxfec && VXFEC_IBV_RX_DEPTH=16384 ./bench_wire_net --mode rx --bind ${IP1} --port 31500 --ports-per-ip ${qps} --gdr ${PIPE_ARG[*]}" \
            | tee -a "${OUT_DIR}/rdma_wire_ibv_rx.log" &
        RX_PID=$!
        VXFEC_WORK_PIDS+=("${RX_PID}")
        sleep 1
        STEP_ERR=0
        gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
            --command="cd ~/vxfec && ./bench_wire_net --mode tx --peers ${IP1} --port 31500 --ports-per-ip ${qps} --gdr ${PIPE_ARG[*]} --codec ${codec} --z ${zval} --k 200 --p 48 --sym-t ${symt} --iters ${iters} --blackout-pct ${bpct} --csv ~/vxfec/rdma_wire_ibv.csv" | tee -a "${OUT_DIR}/rdma_wire_ibv.log" || STEP_ERR=1
        wait "${RX_PID}" || STEP_ERR=1
        [[ "${STEP_ERR}" == "0" ]] || exit 1
    done

    echo "[04_run_tier3_rdma] Running 2-node H200 GPUDirect RDMA (VXFEC_NCCL_BACKEND=ibv --gdr) CUDA/NCCL AllReduce..."
    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="cd ~/vxfec && NCCL_SOCKET_IFNAME='${NIC0}' ./demo_nccl_e2e --rank 0 --nranks 2 --master ${IP0}:29500 --ifname '${NIC0}' --backend ibv --gdr --max-mb 64 --csv ~/vxfec/rdma_nccl_ibv.csv | tee ~/vxfec/rdma_nccl_ibv.txt" &
    R0_PID=$!
    VXFEC_WORK_PIDS+=("${R0_PID}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="cd ~/vxfec && NCCL_SOCKET_IFNAME='${NIC1}' ./demo_nccl_e2e --rank 1 --nranks 2 --master ${IP0}:29500 --ifname '${NIC1}' --backend ibv --gdr --max-mb 64" || STEP_ERR=1
    wait "${R0_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1

    echo "[04_run_tier3_rdma] Running 2-node 8x ConnectX-7 multi-channel bench_nccl (64 MiB, 32 channels)..."
    gcloud compute ssh "${NODE0}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="cd ~/vxfec && VXFEC_NCCL_BACKEND=ibv VXFEC_NCCL_IFNAMES='${NIC0}' VXFEC_BENCH_MAX_MB=64 VXFEC_BENCH_CHANNELS=32 ./bench_nccl --rank 0 --master ${IP0}:29600 --csv ~/vxfec/rdma_bench_nccl.csv | tee ~/vxfec/rdma_bench_nccl.txt" &
    B0_PID=$!
    VXFEC_WORK_PIDS+=("${B0_PID}")
    sleep 1
    STEP_ERR=0
    gcloud compute ssh "${NODE1}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" \
        --command="cd ~/vxfec && VXFEC_NCCL_BACKEND=ibv VXFEC_NCCL_IFNAMES='${NIC1}' VXFEC_BENCH_MAX_MB=64 VXFEC_BENCH_CHANNELS=32 ./bench_nccl --rank 1 --master ${IP0}:29600" || STEP_ERR=1
    wait "${B0_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1

    gcloud compute scp "${NODE0}:~/vxfec/rdma_*" "${OUT_DIR}/" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}"
    echo "[04_run_tier3_rdma] Complete! Results saved in ${OUT_DIR}; tearing down GCP A3-Ultra resources..."
    exit 0
fi

if [[ -z "${HOSTS_CSV}" || -z "${IPS_CSV}" ]]; then
    echo "Usage: $0 [--gcp-a3-ultra] | [--hosts user@host0,user@host1 --ips rdma_ip0,rdma_ip1]" >&2
    exit 1
fi

HOST0="$(echo "${HOSTS_CSV}" | cut -d',' -f1)"
HOST1="$(echo "${HOSTS_CSV}" | cut -d',' -f2)"
IP0="$(echo "${IPS_CSV}" | cut -d',' -f1)"
IP1="$(echo "${IPS_CSV}" | cut -d',' -f2)"

KEEP_REMOTE_DIR="${KEEP_REMOTE_DIR:-0}"
REMOTE_DIR="/tmp/vxfec_rdma_run_$$_$(date +%s)"
TARBALL="/tmp/vxfec_rdma_src_$$.tar.gz"
STAGED_HOSTS=()
WORK_PIDS=()

CLEANUP_DONE=0
kill_tree() {
    local pid="$1"
    if [[ -z "${pid}" ]]; then
        return 0
    fi
    local child
    for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
        kill_tree "${child}"
    done
    kill -TERM "${pid}" 2>/dev/null || true
}

on_signal() {
    local code="$1"
    trap '' INT TERM
    for pid in "${WORK_PIDS[@]}"; do
        kill_tree "${pid}"
    done
    exit "${code}"
}

cleanup() {
    local exit_rc=$?
    set +e
    if [[ "${CLEANUP_DONE}" == "1" ]]; then
        return "${exit_rc}"
    fi
    CLEANUP_DONE=1
    trap '' INT TERM
    for pid in "${WORK_PIDS[@]}"; do
        kill_tree "${pid}"
    done
    for pid in "${WORK_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    rm -f "${TARBALL:-}"
    if [[ "${KEEP_REMOTE_DIR}" == "1" ]]; then
        exit "${exit_rc}"
    fi
    for h in "${STAGED_HOSTS[@]}"; do
        if [[ -z "${h}" ]]; then
            continue
        fi
        ssh "${h}" "rm -rf '${REMOTE_DIR}'" >/dev/null 2>&1 || true
    done
    exit "${exit_rc}"
}
trap cleanup EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

if [[ ! -f "${REPO_ROOT}/deps/nanorq/lib/nanorq_core.c" ]]; then
    echo "ERROR: Pinned submodule deps/nanorq is missing; run 'git submodule update --init'." >&2
    exit 1
fi
TARBALL="$(umask 077 && mktemp /tmp/vxfec_rdma_src_XXXXXX.tar.gz)"
(
    set -euo pipefail
    umask 077
    git -C "${REPO_ROOT}" ls-files --recurse-submodules -z \
        | while IFS= read -r -d '' rel_path; do
            [[ -e "${REPO_ROOT}/${rel_path}" ]] && printf '%s\0' "${rel_path}"
        done \
        | tar --no-recursion -C "${REPO_ROOT}" --null -T - -czf "${TARBALL}"
)

for h in "${HOST0}" "${HOST1}"; do
    echo "[04_run_tier3_rdma] Syncing and building vxfec in isolated ${REMOTE_DIR} on ${h}..."
    ssh "${h}" "mkdir -p '${REMOTE_DIR}'"
    STAGED_HOSTS+=("${h}")
    scp "${TARBALL}" "${h}:${REMOTE_DIR}/vxfec_src.tar.gz"
    ssh "${h}" "
        set -euo pipefail
        SUDO=\"\"; [[ \$(id -u) -ne 0 ]] && SUDO=\"sudo\"
        if ! command -v vulkaninfo >/dev/null 2>&1 || ! dpkg -s libvulkan-dev libibverbs-dev >/dev/null 2>&1; then
            \${SUDO} DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
            \${SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
                build-essential pkg-config libvulkan-dev vulkan-tools ibverbs-utils libibverbs-dev iproute2 libnccl2 libnccl-dev || \
            \${SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
                build-essential pkg-config libvulkan-dev vulkan-tools ibverbs-utils libibverbs-dev iproute2 || true
        fi
        if command -v nvidia-smi >/dev/null 2>&1 && ! vulkaninfo --summary 2>/dev/null | grep -qi nvidia; then
            \${SUDO} mkdir -p /etc/vulkan/icd.d
            printf '{\n  \"file_format_version\": \"1.0.0\",\n  \"ICD\": {\"library_path\": \"libGLX_nvidia.so.0\", \"api_version\": \"1.3.277\"}\n}\n' \
                | \${SUDO} tee /etc/vulkan/icd.d/nvidia_icd.json >/dev/null || true
        fi
        tar -xzf '${REMOTE_DIR}/vxfec_src.tar.gz' -C '${REMOTE_DIR}'
        rm -f '${REMOTE_DIR}/vxfec_src.tar.gz'
        cd '${REMOTE_DIR}' && make -j\$(nproc) all
        ibv_devinfo | head -n 40 | tee '${REMOTE_DIR}/rdma_ibv_devinfo.txt'
        vulkaninfo --summary 2>/dev/null | head -n 40 | tee '${REMOTE_DIR}/rdma_vulkaninfo.txt' || true
    "
done
rm -f "${TARBALL}"

NIC0="$(ssh "${HOST0}" "ip -o -4 route get ${IP1} | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\"){print \$(i+1); exit}}'")"
NIC1="$(ssh "${HOST1}" "ip -o -4 route get ${IP0} | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\"){print \$(i+1); exit}}'")"
: "${NIC0:?Failed to resolve routable interface on ${HOST0} toward ${IP1}}"
: "${NIC1:?Failed to resolve routable interface on ${HOST1} toward ${IP0}}"

echo "[04_run_tier3_rdma] Running 2-node libibverbs IBV_QPT_UC + DMA-BUF GPU-to-GPU superframe saturation..."
for spec in \
    "rateless-gf2 0 1 256 4096 8 20" \
    "rateless-gf2 10 1 256 4096 8 20" \
    "rateless 10 1 256 4096 8 20" \
    "mds 10 1 256 4096 8 20" \
    "rateless-gf2 10 0 128 4096 8 5"; do
    read -r codec bpct pipe zval symt qps iters <<<"${spec}"
    PIPE_ARG=()
    [[ "${pipe}" == "1" ]] && PIPE_ARG=(--pipeline)
    ssh "${HOST1}" "cd '${REMOTE_DIR}' && VXFEC_IBV_RX_DEPTH=8192 ./bench_wire_net --mode rx --bind ${IP1} --port 31500 --ports-per-ip ${qps} --gdr ${PIPE_ARG[*]}" \
        | tee -a "${OUT_DIR}/rdma_wire_ibv_rx.log" &
    RX_PID=$!
    WORK_PIDS+=("${RX_PID}")
    sleep 1
    STEP_ERR=0
    ssh "${HOST0}" "cd '${REMOTE_DIR}' && ./bench_wire_net --mode tx --peers ${IP1} --port 31500 --ports-per-ip ${qps} --gdr ${PIPE_ARG[*]} --codec ${codec} --z ${zval} --k 200 --p 48 --sym-t ${symt} --iters ${iters} --blackout-pct ${bpct} --csv '${REMOTE_DIR}/rdma_wire_ibv.csv'" | tee -a "${OUT_DIR}/rdma_wire_ibv.log" || STEP_ERR=1
    wait "${RX_PID}" || STEP_ERR=1
    [[ "${STEP_ERR}" == "0" ]] || exit 1
done

echo "[04_run_tier3_rdma] Running 2-node GPUDirect RDMA (VXFEC_NCCL_BACKEND=ibv --gdr) AllReduce..."
ssh "${HOST0}" "cd '${REMOTE_DIR}' && ./demo_nccl_e2e --rank 0 --nranks 2 --master ${IP0}:29500 --ifname '${NIC0}' --backend ibv --gdr --max-mb 32 --csv '${REMOTE_DIR}/rdma_nccl_ibv.csv' | tee '${REMOTE_DIR}/rdma_nccl_ibv.txt'" &
R0_PID=$!
WORK_PIDS+=("${R0_PID}")
sleep 1
STEP_ERR=0
ssh "${HOST1}" "cd '${REMOTE_DIR}' && ./demo_nccl_e2e --rank 1 --nranks 2 --master ${IP0}:29500 --ifname '${NIC1}' --backend ibv --gdr --max-mb 32" || STEP_ERR=1
wait "${R0_PID}" || STEP_ERR=1
[[ "${STEP_ERR}" == "0" ]] || exit 1

scp "${HOST0}:${REMOTE_DIR}/rdma_*" "${OUT_DIR}/"
echo "[04_run_tier3_rdma] Complete! Results saved in ${OUT_DIR}."
