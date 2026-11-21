#!/usr/bin/env bash
# scripts/gcp/00_check_quotas.sh
# Inspect active gcloud authentication, GCP projects, GPU/gVNIC regional quotas,
# and estimated Spot $/hr costs before provisioning any paid cloud resources.
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
    if [[ -x "$HOME/google-cloud-sdk/bin/gcloud" ]]; then
        export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    elif [[ -x "/tmp/google-cloud-sdk/bin/gcloud" ]]; then
        export PATH="/tmp/google-cloud-sdk/bin:$PATH"
    else
        echo "ERROR: gcloud CLI not found. Install Google Cloud SDK first." >&2
        exit 1
    fi
fi

GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
REGIONS=(${GCP_REGIONS:-us-central1 us-east4 us-west1 us-west4 europe-west4})

echo "================================================================================"
echo "vxfec GCP Cloud Pre-Flight: Auth, Projects & GPU Quota Audit"
echo "================================================================================"
echo "Active gcloud account:"
gcloud auth list --filter="status:ACTIVE" --format="table(account,status)" || true
echo ""

echo "Accessible GCP projects:"
gcloud projects list --format="table(projectId,name,projectNumber)" || true
echo ""

if [[ -z "${GCP_PROJECT}" || "${GCP_PROJECT}" == "(unset)" ]]; then
    echo "NOTE: Set GCP_PROJECT=<your-project-id> to query regional GPU quotas."
    exit 0
fi

echo "Auditing regional GPU quotas for project: ${GCP_PROJECT}"
echo "--------------------------------------------------------------------------------"
QUOTA_METRICS="NVIDIA_L4_GPUS|PREEMPTIBLE_NVIDIA_L4_GPUS|NVIDIA_H100_GPUS|PREEMPTIBLE_NVIDIA_H100_GPUS|NVIDIA_H100_MEGA_GPUS|NVIDIA_H200_GPUS"

echo "Global GPU & CPU quotas:"
gcloud compute project-info describe --project="${GCP_PROJECT}" \
    --flatten="quotas[]" \
    --format="table(quotas.metric,quotas.limit,quotas.usage)" \
    | grep -E "METRIC|GPUS_ALL_REGIONS|CPUS_ALL_REGIONS" || true
echo ""

for reg in "${REGIONS[@]}"; do
    echo ">>> Region: ${reg}"
    gcloud compute regions describe "${reg}" --project="${GCP_PROJECT}" \
        --flatten="quotas[]" \
        --format="table(quotas.metric,quotas.limit,quotas.usage)" \
        | grep -E "METRIC|^CPUS |PREEMPTIBLE_CPUS|${QUOTA_METRICS}|NVIDIA_V100_GPUS|PREEMPTIBLE_NVIDIA_V100_GPUS" || echo "  (unable to query ${reg})"
done

echo ""
echo "================================================================================"
echo "Estimated Spot Compute Spend (Bounded by --max-run-duration Guardrails)"
echo "================================================================================"
echo "  Tier 1 (2x g2-standard-32 Spot, 2x L4 + 32 Gbps gVNIC) : ~\$1.30-\$1.80/hr (~30 min = <\$1.00)"
echo "  Tier 2 (2x a3-highgpu-2g Spot, H100 80GB HBM3 + Gen5)  : ~\$8.00-\$12.00/hr (~20 min = ~\$4.00)"
echo "  Tier 3 (a3-ultragpu-8g MRDMA or BYO Bare-Metal mlx5)   : ~\$28-\$35/hr Spot (~15 min = ~\$9.00)"
echo "================================================================================"
