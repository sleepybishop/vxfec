#!/usr/bin/env bash
# scripts/gcp/99_teardown_all.sh
# Unconditionally tear down all vxfec-bench-* instances, placement policies,
# firewall rules, subnets, and VPCs in $GCP_PROJECT, and verify zero active VMs.
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
    [[ -x "$HOME/google-cloud-sdk/bin/gcloud" ]] && export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    [[ -x "/tmp/google-cloud-sdk/bin/gcloud" ]] && export PATH="/tmp/google-cloud-sdk/bin:$PATH"
fi

GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
: "${GCP_PROJECT:?Set GCP_PROJECT=<project-id>}"
VM_REGEX='^vxfec-bench-(g2-[01]|g2-xzone|a3-[01]|u[01]|nic-[01]|nic-xzone|nic-wan|n[01]|n-xzone|n-wan|t1-[01])$'
POL_REGEX='^vxfec-bench-compact-[a-z0-9-]+$'
FW_REGEX='^vxfec-bench-(vpc-allow-(internal|ssh)|vpc2-allow-internal|(vpc[234]|vpc-gpunic[0-9]+|gvnic[0-9]+)-internal|r[0-9a-f]{16}-((gpunic|gvnic)[0-9]+-internal|vpc[234]-(allow-)?internal))$'
SUB_REGEX='^vxfec-bench-(subnet[234]?-[a-z0-9-]+(-gpunic[0-9]+)?|r[0-9a-f]{16}-((gpusub|gvsub|rocesub)[0-9]+|sub[234](-[a-z0-9-]+)?))$'
VPC_REGEX='^vxfec-bench-(vpc([234]|-gpunic[0-9]+)?|r[0-9a-f]{16}-((gpunic|gvnic)[0-9]+|roce|vpc[234]))$'
EXIT_RC=0

run_delete_ignore_not_found() {
    local err
    if ! err="$("$@" 2>&1)"; then
        if echo "${err}" | grep -Eiq "was not found|NOT_FOUND"; then
            return 0
        fi
        echo "ERROR: [99_teardown_all] $* failed: ${err}" >&2
        return 1
    fi
    return 0
}

echo "[99_teardown_all] Auditing and deleting all vxfec benchmark VMs in project ${GCP_PROJECT} (parallel)..."
VM_PIDS=()
while read -r name zone; do
    [[ -z "${name}" ]] && continue
    zone_short="${zone##*/}"
    echo "  Deleting VM ${name} in ${zone_short}..."
    run_delete_ignore_not_found gcloud compute instances delete "${name}" --project="${GCP_PROJECT}" --zone="${zone_short}" --quiet &
    VM_PIDS+=("$!")
done < <(gcloud compute instances list --project="${GCP_PROJECT}" \
    --filter="labels.vxfec-run:* OR name~'${VM_REGEX}'" --format="value(name,zone)")
for pid in "${VM_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && ! wait "${pid}"; then
        EXIT_RC=1
    fi
done

echo "[99_teardown_all] Deleting vxfec placement policies..."
while read -r name region; do
    [[ -z "${name}" ]] && continue
    reg_short="${region##*/}"
    if ! run_delete_ignore_not_found gcloud compute resource-policies delete "${name}" --project="${GCP_PROJECT}" --region="${reg_short}" --quiet; then
        EXIT_RC=1
    fi
done < <(gcloud compute resource-policies list --project="${GCP_PROJECT}" \
    --filter="name~'${POL_REGEX}'" --format="value(name,region)")

echo "[99_teardown_all] Deleting vxfec firewall rules (parallel)..."
FW_PIDS=()
while read -r fw; do
    [[ -z "${fw}" ]] && continue
    run_delete_ignore_not_found gcloud compute firewall-rules delete "${fw}" --project="${GCP_PROJECT}" --quiet &
    FW_PIDS+=("$!")
done < <(gcloud compute firewall-rules list --project="${GCP_PROJECT}" \
    --filter="name~'${FW_REGEX}'" --format="value(name)")
for pid in "${FW_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && ! wait "${pid}"; then
        EXIT_RC=1
    fi
done

echo "[99_teardown_all] Deleting vxfec subnets (parallel)..."
SUB_PIDS=()
while read -r sub region; do
    [[ -z "${sub}" ]] && continue
    reg_short="${region##*/}"
    run_delete_ignore_not_found gcloud compute networks subnets delete "${sub}" --project="${GCP_PROJECT}" --region="${reg_short}" --quiet &
    SUB_PIDS+=("$!")
done < <(gcloud compute networks subnets list --project="${GCP_PROJECT}" \
    --filter="name~'${SUB_REGEX}'" --format="value(name,region)")
for pid in "${SUB_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && ! wait "${pid}"; then
        EXIT_RC=1
    fi
done

echo "[99_teardown_all] Deleting vxfec VPC networks (parallel)..."
NET_PIDS=()
while read -r net; do
    [[ -z "${net}" ]] && continue
    run_delete_ignore_not_found gcloud compute networks delete "${net}" --project="${GCP_PROJECT}" --quiet &
    NET_PIDS+=("$!")
done < <(gcloud compute networks list --project="${GCP_PROJECT}" \
    --filter="name~'${VPC_REGEX}'" --format="value(name)")
for pid in "${NET_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && ! wait "${pid}"; then
        EXIT_RC=1
    fi
done

echo ""
echo "[99_teardown_all] Final verification — remaining vxfec benchmark VMs (must be empty):"
REMAINING_VMS="$(gcloud compute instances list --project="${GCP_PROJECT}" \
    --filter="labels.vxfec-run:* OR name~'${VM_REGEX}'" --format="value(name,zone)")"
if [[ -n "${REMAINING_VMS}" ]]; then
    echo "ERROR: [99_teardown_all] Matching vxfec benchmark VMs still exist:" >&2
    printf '%s\n' "${REMAINING_VMS}" >&2
    exit 1
fi
if [[ "${EXIT_RC}" -ne 0 ]]; then
    echo "ERROR: [99_teardown_all] Teardown encountered one or more deletion errors." >&2
    exit "${EXIT_RC}"
fi
echo "[99_teardown_all] Done."
