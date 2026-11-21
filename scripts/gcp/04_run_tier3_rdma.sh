#!/usr/bin/env bash
# scripts/gcp/04_run_tier3_rdma.sh
# Validate native libibverbs (IBV_QPT_UC + ibv_reg_dmabuf_mr zero-bounce GPUDirect RDMA)
# across either:
#   1) Generic SSH hosts with ConnectX-6/7 RoCEv2/IB (--hosts user@host0,user@host1 --ips ip0,ip1), or
#   2) GCP A3-Ultra (a3-ultragpu-8g H200 + ConnectX-7 MRDMA) when --gcp-a3-ultra is passed.
set -euo pipefail

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
        *) echo "Usage: $0 [--hosts user@h0,user@h1 --ips ip0,ip1] | [--gcp-a3-ultra]" >&2; exit 1 ;;
    esac
done

if [[ "${GCP_A3_ULTRA}" == "1" ]]; then
    echo "[04_run_tier3_rdma] Delegating A3-Ultra MRDMA provisioning to GCP gcloud..."
    GCP_PROJECT="${GCP_PROJECT:?Set GCP_PROJECT=<project-id>}"
    echo "Checking NVIDIA_H200_GPUS quota in ${GCP_PROJECT}..."
    "${REPO_ROOT}/scripts/gcp/00_check_quotas.sh" | grep -E "H200|Region" || true
    echo "Ensure MRDMA VPC (--network-profile=*-vpc-roce) and H200 quota are active, then pass provisioned SSH hosts via --hosts."
    exit 0
fi

if [[ -z "${HOSTS_CSV}" || -z "${IPS_CSV}" ]]; then
    echo "Usage: $0 --hosts user@host0,user@host1 --ips rdma_ip0,rdma_ip1" >&2
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
        tar -xzf '${REMOTE_DIR}/vxfec_src.tar.gz' -C '${REMOTE_DIR}'
        rm -f '${REMOTE_DIR}/vxfec_src.tar.gz'
        cd '${REMOTE_DIR}' && make -j\$(nproc) all
        ibv_devinfo | head -n 30 | tee '${REMOTE_DIR}/rdma_ibv_devinfo.txt'
    "
done
rm -f "${TARBALL}"

NIC0="$(ssh "${HOST0}" "ip -o -4 route get ${IP1} | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\"){print \$(i+1); exit}}'")"
NIC1="$(ssh "${HOST1}" "ip -o -4 route get ${IP0} | awk '{for(i=1;i<=NF;i++) if(\$i==\"dev\"){print \$(i+1); exit}}'")"
: "${NIC0:?Failed to resolve routable interface on ${HOST0} toward ${IP1}}"
: "${NIC1:?Failed to resolve routable interface on ${HOST1} toward ${IP0}}"

echo "[04_run_tier3_rdma] Running 2-node libibverbs IBV_QPT_UC + DMA-BUF superframe saturation..."
for spec in "rateless 0 0" "rateless 10 0" "rateless-gf2 10 0" "mds 10 0" "rateless-gf2 10 1"; do
    read -r codec bpct pipe <<<"${spec}"
    PIPE_ARG=()
    [[ "${pipe}" == "1" ]] && PIPE_ARG=(--pipeline)
    ssh "${HOST1}" "cd '${REMOTE_DIR}' && ./bench_wire_net --mode rx --bind ${IP1} --port 31500" &
    RX_PID=$!
    WORK_PIDS+=("${RX_PID}")
    sleep 1
    STEP_ERR=0
    ssh "${HOST0}" "cd '${REMOTE_DIR}' && ./bench_wire_net --mode tx --peers ${IP1} --port 31500 --ibv --codec ${codec} --z 128 --k 200 --p 48 --sym-t 4096 --iters 5 --blackout-pct ${bpct} ${PIPE_ARG[*]} --csv '${REMOTE_DIR}/rdma_wire_ibv.csv'" | tee -a "${OUT_DIR}/rdma_wire_ibv.log" || STEP_ERR=1
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
