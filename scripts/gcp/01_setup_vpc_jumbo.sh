#!/usr/bin/env bash
# scripts/gcp/01_setup_vpc_jumbo.sh
# Idempotently provision an 8896-byte Jumbo-MTU VPC, regional subnet, internal
# firewall rules, and COMPACT collocated placement policy for vxfec benchmarks.
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
    [[ -x "$HOME/google-cloud-sdk/bin/gcloud" ]] && export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    [[ -x "/tmp/google-cloud-sdk/bin/gcloud" ]] && export PATH="/tmp/google-cloud-sdk/bin:$PATH"
fi

GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
: "${GCP_PROJECT:?Set GCP_PROJECT=<project-id>}"
GCP_REGION="${GCP_REGION:-us-central1}"
VPC_NAME="${VPC_NAME:-vxfec-bench-vpc}"
SUBNET_NAME="${SUBNET_NAME:-vxfec-bench-subnet-${GCP_REGION}}"
SUBNET_CIDR="${SUBNET_CIDR:-10.128.0.0/20}"
POLICY_NAME="${POLICY_NAME:-vxfec-bench-compact-${GCP_REGION}}"
BENCH_NET_TAG="${BENCH_NET_TAG:-vxfec-bench}"

resolve_ssh_source_ranges() {
    if [[ -n "${SSH_SOURCE_RANGES:-}" ]]; then
        if [[ ",${SSH_SOURCE_RANGES}," == *",0.0.0.0/0,"* && "${ALLOW_WORLD_SSH:-0}" != "1" ]]; then
            echo "ERROR: Refusing 0.0.0.0/0 in SSH_SOURCE_RANGES (set SSH_SOURCE_RANGES=<cidr> or ALLOW_WORLD_SSH=1)." >&2
            exit 1
        fi
        printf '%s\n' "${SSH_SOURCE_RANGES}"
        return 0
    fi
    local ip=""
    if command -v curl >/dev/null 2>&1; then
        ip="$(curl -4 -fsSL --connect-timeout 2 --max-time 3 https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]' || true)"
    fi
    if [[ "${ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        printf '%s/32,35.235.240.0/20\n' "${ip}"
    else
        printf '35.235.240.0/20\n'
    fi
}

SSH_RANGES="$(resolve_ssh_source_ranges)"

echo "[01_setup_vpc_jumbo] Project=${GCP_PROJECT} Region=${GCP_REGION} VPC=${VPC_NAME} (MTU=8896, tag=${BENCH_NET_TAG})"

if ! gcloud compute networks describe "${VPC_NAME}" --project="${GCP_PROJECT}" >/dev/null 2>&1; then
    gcloud compute networks create "${VPC_NAME}" \
        --project="${GCP_PROJECT}" \
        --subnet-mode=custom \
        --mtu=8896 \
        --bgp-routing-mode=global
else
    echo "  VPC ${VPC_NAME} already exists."
fi

if ! gcloud compute networks subnets describe "${SUBNET_NAME}" \
    --project="${GCP_PROJECT}" --region="${GCP_REGION}" >/dev/null 2>&1; then
    gcloud compute networks subnets create "${SUBNET_NAME}" \
        --project="${GCP_PROJECT}" \
        --network="${VPC_NAME}" \
        --region="${GCP_REGION}" \
        --range="${SUBNET_CIDR}"
else
    echo "  Subnet ${SUBNET_NAME} already exists."
fi

if ! gcloud compute firewall-rules describe "${VPC_NAME}-allow-internal" --project="${GCP_PROJECT}" >/dev/null 2>&1; then
    gcloud compute firewall-rules create "${VPC_NAME}-allow-internal" \
        --project="${GCP_PROJECT}" \
        --network="${VPC_NAME}" \
        --allow=tcp:1-65535,udp:1-65535,icmp \
        --source-ranges=10.128.0.0/12 \
        --target-tags="${BENCH_NET_TAG}"
else
    gcloud compute firewall-rules update "${VPC_NAME}-allow-internal" \
        --project="${GCP_PROJECT}" \
        --source-ranges=10.128.0.0/12 \
        --target-tags="${BENCH_NET_TAG}" >/dev/null
fi

if ! gcloud compute firewall-rules describe "${VPC_NAME}-allow-ssh" --project="${GCP_PROJECT}" >/dev/null 2>&1; then
    gcloud compute firewall-rules create "${VPC_NAME}-allow-ssh" \
        --project="${GCP_PROJECT}" \
        --network="${VPC_NAME}" \
        --allow=tcp:22 \
        --source-ranges="${SSH_RANGES}" \
        --target-tags="${BENCH_NET_TAG}"
else
    gcloud compute firewall-rules update "${VPC_NAME}-allow-ssh" \
        --project="${GCP_PROJECT}" \
        --source-ranges="${SSH_RANGES}" \
        --target-tags="${BENCH_NET_TAG}" >/dev/null
fi

if ! gcloud compute resource-policies describe "${POLICY_NAME}" \
    --project="${GCP_PROJECT}" --region="${GCP_REGION}" >/dev/null 2>&1; then
    gcloud compute resource-policies create group-placement "${POLICY_NAME}" \
        --project="${GCP_PROJECT}" \
        --region="${GCP_REGION}" \
        --collocation=COLLOCATED \
        --vm-count=2 || echo "  NOTE: COMPACT policy creation skipped/unsupported in ${GCP_REGION}; continuing without policy."
fi

echo "[01_setup_vpc_jumbo] Ready: ${VPC_NAME} (MTU 8896) + ${SUBNET_NAME}."
