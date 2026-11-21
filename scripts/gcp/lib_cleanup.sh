#!/usr/bin/env bash
# Shared run-scoped resource tracking, cancellation draining, MAC-matched NIC
# configuration, and overlap-safe GCP cleanup helpers sourced by scripts/gcp/02..06.

vxfec_gen_run_token() {
    local hex=""
    if [[ -r /proc/sys/kernel/random/uuid ]]; then
        hex="$(tr -d -- '-\n' < /proc/sys/kernel/random/uuid | head -c 16)"
    fi
    if [[ ${#hex} -lt 16 ]]; then
        hex="$(od -An -N8 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n' | head -c 16)"
    fi
    if [[ ${#hex} -lt 16 ]]; then
        hex="$(printf '%04x%04x%08x' "$$" "${RANDOM:-0}" "$(date +%s)")"
    fi
    printf 'r%s\n' "${hex}"
}

VXFEC_RUN_TOKEN="$(vxfec_gen_run_token)"
export VXFEC_RUN_TOKEN
VXFEC_CLEANUP_TAG="${VXFEC_CLEANUP_TAG:-gcp}"
VXFEC_CREATED_LOG="/tmp/vxfec_created_${VXFEC_RUN_TOKEN}.log"
VXFEC_CANCEL_FILE="/tmp/vxfec_cancel_${VXFEC_RUN_TOKEN}.flag"
: > "${VXFEC_CREATED_LOG}"
rm -f "${VXFEC_CANCEL_FILE}"

VXFEC_CLEANUP_DONE=0
VXFEC_CREATE_PIDS=()
VXFEC_WORK_PIDS=()
VXFEC_TEMP_FILES=()
VXFEC_CREATED_VMS=()
VXFEC_CREATED_VPCS=()
VXFEC_CREATED_SUBNETS=()
VXFEC_CREATED_FWS=()

# Package ONLY git-tracked superproject files and the pinned deps/nanorq submodule
# into a mode-0600 archive so untracked local files/secrets are never uploaded.
vxfec_package_repo_tarball() {
    local repo_root="$1"
    local out_tar="$2"
    if [[ ! -f "${repo_root}/deps/nanorq/lib/nanorq_core.c" ]]; then
        echo "ERROR: Pinned submodule deps/nanorq is missing; run 'git submodule update --init'." >&2
        return 1
    fi
    (
        set -euo pipefail
        umask 077
        rm -f "${out_tar}"
        git -C "${repo_root}" ls-files --recurse-submodules -z \
            | while IFS= read -r -d '' rel_path; do
                [[ -e "${repo_root}/${rel_path}" ]] && printf '%s\0' "${rel_path}"
            done \
            | tar --no-recursion -C "${repo_root}" --null -T - -czf "${out_tar}"
    )
}

# Wait until SSH and the NVIDIA kernel driver are responsive on an accelerator VM.
vxfec_wait_nvidia_ssh() {
    local vm="$1"
    local zone="$2"
    local max_tries="${3:-36}"
    local i
    for ((i = 1; i <= max_tries; i++)); do
        if gcloud compute ssh "${vm}" --project="${GCP_PROJECT}" --zone="${zone}" \
            --command="nvidia-smi -L" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Timed out waiting for SSH + NVIDIA driver on ${vm} in ${zone}" >&2
    return 1
}

# Upload packaged vxfec tarball, install build/RDMA/Vulkan/NCCL packages, register headless
# NVIDIA Vulkan ICD when missing, tune socket buffers, and compile all targets.
vxfec_bootstrap_nvidia_vm() {
    local vm="$1"
    local zone="$2"
    local tarball="$3"
    local sock_buf_bytes="${4:-67108864}"
    gcloud compute scp "${tarball}" "${vm}:/tmp/vxfec_src.tar.gz" --project="${GCP_PROJECT}" --zone="${zone}"
    gcloud compute ssh "${vm}" --project="${GCP_PROJECT}" --zone="${zone}" --command="
        set -euo pipefail
        for _ in 1 2 3 4 5; do
            sudo apt-get -o DPkg::Lock::Timeout=120 update -qq && break
            sleep 5
        done
        sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=120 install -y -qq \
            build-essential pkg-config libvulkan-dev vulkan-tools ibverbs-utils libibverbs-dev iproute2 libnccl2 libnccl-dev || \
        sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=120 install -y -qq \
            build-essential pkg-config libvulkan-dev vulkan-tools ibverbs-utils libibverbs-dev iproute2
        if ! vulkaninfo --summary 2>/dev/null | grep -qi nvidia; then
            sudo mkdir -p /etc/vulkan/icd.d
            cat <<'EOF' | sudo tee /etc/vulkan/icd.d/nvidia_icd.json >/dev/null
{
    \"file_format_version\" : \"1.0.0\",
    \"ICD\": {
        \"library_path\": \"libGLX_nvidia.so.0\",
        \"api_version\" : \"1.3.277\"
    }
}
EOF
        fi
        sudo sysctl -w net.core.rmem_max=${sock_buf_bytes} net.core.wmem_max=${sock_buf_bytes} >/dev/null
        rm -rf ~/vxfec && mkdir -p ~/vxfec
        tar -xzf /tmp/vxfec_src.tar.gz -C ~/vxfec
        rm -f /tmp/vxfec_src.tar.gz
        cd ~/vxfec && make -j\$(nproc) all
    "
}

vxfec_add_unique() {
    local -n arr_ref="$1"
    local val="$2"
    if [[ -z "${val}" ]]; then
        return 0
    fi
    local item
    for item in "${arr_ref[@]}"; do
        if [[ "${item}" == "${val}" ]]; then
            return 0
        fi
    done
    arr_ref+=("${val}")
}

vxfec_ingest_created_log() {
    if [[ ! -f "${VXFEC_CREATED_LOG}" ]]; then
        return 0
    fi
    local kind a b c
    while IFS=: read -r kind a b c; do
        case "${kind}" in
            vm)     vxfec_add_unique VXFEC_CREATED_VMS "${a}:${b}" ;;
            vpc)    vxfec_add_unique VXFEC_CREATED_VPCS "${a}" ;;
            subnet) vxfec_add_unique VXFEC_CREATED_SUBNETS "${a}:${b}:${c}" ;;
            fw)     vxfec_add_unique VXFEC_CREATED_FWS "${a}:${b}" ;;
        esac
    done < "${VXFEC_CREATED_LOG}"
    : > "${VXFEC_CREATED_LOG}"
}

# Run a gcloud create command in a tracked background job so INT/TERM interrupts wait
# immediately while vxfec_cleanup drains the in-flight RPC before teardown.
vxfec_run_create() {
    if [[ -f "${VXFEC_CANCEL_FILE}" ]]; then
        return 1
    fi
    "$@" &
    local pid=$!
    VXFEC_CREATE_PIDS+=("${pid}")
    wait "${pid}"
}

# Returns 0 if VM exists in zone, 1 if absent, exits script on API/auth failure.
vxfec_vm_exists() {
    local vm="$1"
    local zone="$2"
    local out
    if ! out="$(gcloud compute instances list --project="${GCP_PROJECT}" \
        --zones="${zone}" --filter="name=('${vm}')" --format="value(name)")"; then
        echo "ERROR: Failed to query GCP instances in ${zone} (check gcloud auth/project)." >&2
        exit 1
    fi
    [[ -n "${out}" ]]
}

# Delete only VMs carrying this run's vxfec-run label AND matching the attempted VM names.
vxfec_cleanup_failed_attempt() {
    local zone="$1"
    shift
    local targets=("$@")
    if [[ ${#targets[@]} -eq 0 ]]; then
        return 0
    fi
    local name z t match
    local del_pids=()
    while read -r name z; do
        if [[ -z "${name}" ]]; then
            continue
        fi
        match=0
        for t in "${targets[@]}"; do
            if [[ "${name}" == "${t}" ]]; then
                match=1
            fi
        done
        if [[ "${match}" == "1" ]]; then
            gcloud compute instances delete "${name}" --project="${GCP_PROJECT}" --zone="${zone}" --quiet &
            del_pids+=("$!")
        fi
    done < <(gcloud compute instances list --project="${GCP_PROJECT}" --zones="${zone}" \
        --filter="labels.vxfec-run=${VXFEC_RUN_TOKEN}" --format="value(name,zone)" 2>/dev/null || true)
    local pid
    for pid in "${del_pids[@]}"; do
        wait "${pid}" || true
    done
}

# Match cloud NICs to guest Linux interfaces by hardware MAC address, configure MTU 8896
# and VPC IPs, and echo "<ips_csv>|<ifaces_csv>".
vxfec_configure_vm_nics() {
    local vm="$1"
    local zone="$2"
    local nic_lines ips_csv ifaces_csv ssh_rc exp_cnt got_cnt
    if ! nic_lines="$(gcloud compute instances describe "${vm}" --project="${GCP_PROJECT}" --zone="${zone}" \
        --flatten='networkInterfaces[]' --format='csv[no-heading](networkInterfaces.mac,networkInterfaces.networkIP)' | tr -d '\r')"; then
        echo "ERROR: Could not describe networkInterfaces on ${vm} in ${zone}" >&2
        return 1
    fi
    ips_csv="$(printf '%s\n' "${nic_lines}" | awk -F',' 'NF>=2 && $2!="" {print $2}' | paste -sd',' -)"
    if [[ -z "${ips_csv}" ]]; then
        echo "ERROR: Could not query networkInterfaces on ${vm} in ${zone}" >&2
        return 1
    fi
    ifaces_csv="$(gcloud compute ssh "${vm}" --project="${GCP_PROJECT}" --zone="${zone}" --command="
        set -euo pipefail
        ALL_DEVS=(\$(ls /sys/class/net | grep -Ev '^(lo|docker.*|virbr.*|veth.*|br-.*|rxe.*)\$' | sort -V))
        CLAIMED=()
        ACTIVE=()
        is_claimed() {
            local cand=\"\$1\" c
            for c in \"\${CLAIMED[@]}\"; do
                if [[ \"\${c}\" == \"\${cand}\" ]]; then
                    return 0
                fi
            done
            return 1
        }
        idx=0
        while IFS=',' read -r raw_mac want_ip; do
            [[ -z \"\${want_ip}\" ]] && continue
            want_mac=\"\$(printf '%s' \"\${raw_mac}\" | tr 'A-F' 'a-f' | tr -d '[:space:]')\"
            if [[ -z \"\${want_mac}\" ]] && command -v curl >/dev/null 2>&1; then
                meta_mac=\"\$(curl -fsSL --connect-timeout 1 --max-time 2 -H 'Metadata-Flavor: Google' \
                    \"http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/\${idx}/mac\" 2>/dev/null || true)\"
                want_mac=\"\$(printf '%s' \"\${meta_mac}\" | tr 'A-F' 'a-f' | tr -d '[:space:]')\"
            fi
            if [[ -z \"\${want_mac}\" ]] && command -v wget >/dev/null 2>&1; then
                meta_mac=\"\$(wget -qO- --timeout=2 --header='Metadata-Flavor: Google' \
                    \"http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/\${idx}/mac\" 2>/dev/null || true)\"
                want_mac=\"\$(printf '%s' \"\${meta_mac}\" | tr 'A-F' 'a-f' | tr -d '[:space:]')\"
            fi
            matched=\"\"
            if [[ -n \"\${want_mac}\" ]]; then
                for d in \"\${ALL_DEVS[@]}\"; do
                    is_claimed \"\${d}\" && continue
                    if [[ -r \"/sys/class/net/\${d}/address\" ]]; then
                        dmac=\$(tr 'A-F' 'a-f' < \"/sys/class/net/\${d}/address\" | tr -d '[:space:]')
                        if [[ \"\${dmac}\" == \"\${want_mac}\" ]]; then
                            matched=\"\${d}\"
                            break
                        fi
                    fi
                done
            fi
            if [[ -z \"\${matched}\" && -z \"\${want_mac}\" ]]; then
                for d in \"\${ALL_DEVS[@]}\"; do
                    is_claimed \"\${d}\" && continue
                    if ip -o -4 addr show dev \"\${d}\" | awk '{print \$4}' | cut -d'/' -f1 | grep -Fxq \"\${want_ip}\"; then
                        matched=\"\${d}\"
                        break
                    fi
                done
            fi
            if [[ -z \"\${matched}\" && -z \"\${want_mac}\" ]]; then
                for d in \"\${ALL_DEVS[@]}\"; do
                    if ! is_claimed \"\${d}\"; then
                        matched=\"\${d}\"
                        break
                    fi
                done
            fi
            if [[ -z \"\${matched}\" ]]; then
                echo \"ERROR: Could not uniquely match NIC index \${idx} (mac=\${want_mac}, ip=\${want_ip})\" >&2
                exit 1
            fi
            CLAIMED+=(\"\${matched}\")
            sudo ip link set \"\${matched}\" mtu 8896 up
            sudo sysctl -w net.ipv4.conf.\"\${matched}\".rp_filter=0 >/dev/null 2>&1 || true
            sudo ethtool -G \"\${matched}\" rx 4096 tx 4096 2>/dev/null || true
            if ! ip -o -4 addr show dev \"\${matched}\" | awk '{print \$4}' | cut -d'/' -f1 | grep -Fxq \"\${want_ip}\"; then
                sudo ip addr add \"\${want_ip}/20\" dev \"\${matched}\" 2>/dev/null || true
            fi
            ACTIVE+=(\"\${matched}\")
            idx=\$((idx + 1))
        done <<< '${nic_lines}'
        IFS=','; echo \"\${ACTIVE[*]}\"
    ")" || {
        ssh_rc=$?
        echo "ERROR: SSH multi-NIC setup failed on ${vm} in ${zone} (exit ${ssh_rc})" >&2
        return "${ssh_rc}"
    }
    if [[ -z "${ifaces_csv}" ]]; then
        echo "ERROR: Multi-NIC setup returned empty interface list on ${vm} in ${zone}" >&2
        return 1
    fi
    exp_cnt="$(printf '%s' "${ips_csv}" | awk -F',' '{print NF}')"
    got_cnt="$(printf '%s' "${ifaces_csv}" | awk -F',' '{print NF}')"
    if [[ "${got_cnt}" -ne "${exp_cnt}" ]]; then
        echo "ERROR: Multi-NIC interface count mismatch on ${vm} (expected ${exp_cnt}, got ${got_cnt})" >&2
        return 1
    fi
    printf '%s|%s\n' "${ips_csv}" "${ifaces_csv}"
}

vxfec_kill_worker() {
    local pid="$1"
    if [[ -z "${pid}" ]]; then
        return 0
    fi
    local child
    for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
        vxfec_kill_worker "${child}"
    done
    kill -TERM "${pid}" 2>/dev/null || true
}

vxfec_vpc_is_unused() {
    local vn="$1"
    if [[ -z "${vn}" ]]; then
        return 1
    fi
    local out
    if ! out="$(gcloud compute instances list --project="${GCP_PROJECT}" \
        --filter="networkInterfaces.network~'/${vn}$'" --limit=1 --format="value(name)" 2>/dev/null)"; then
        return 1
    fi
    [[ -z "${out}" ]]
}

vxfec_subnet_is_unused() {
    local sn="$1"
    local reg="$2"
    if [[ -z "${sn}" ]]; then
        return 1
    fi
    local out
    if ! out="$(gcloud compute instances list --project="${GCP_PROJECT}" \
        --filter="networkInterfaces.subnetwork~'/regions/${reg}/subnetworks/${sn}$'" --limit=1 --format="value(name)" 2>/dev/null)"; then
        return 1
    fi
    [[ -z "${out}" ]]
}

vxfec_on_signal() {
    local code="$1"
    trap '' INT TERM
    : > "${VXFEC_CANCEL_FILE}"
    echo "[${VXFEC_CLEANUP_TAG}] Caught signal; stopping workers and draining pending GCP creates..." >&2
    local pid
    for pid in "${VXFEC_WORK_PIDS[@]}"; do
        vxfec_kill_worker "${pid}"
    done
    exit "${code}"
}

vxfec_cleanup() {
    local exit_rc=$?
    set +e
    if [[ "${VXFEC_CLEANUP_DONE}" == "1" ]]; then
        return "${exit_rc}"
    fi
    VXFEC_CLEANUP_DONE=1
    trap '' INT TERM
    : > "${VXFEC_CANCEL_FILE}"

    local pid
    for pid in "${VXFEC_WORK_PIDS[@]}"; do
        vxfec_kill_worker "${pid}"
    done
    for pid in "${VXFEC_WORK_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    # Drain any in-flight gcloud create RPCs so server-side resources settle before discovery/delete.
    for pid in "${VXFEC_CREATE_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    vxfec_ingest_created_log
    rm -f "${TARBALL:-}" "${VXFEC_CREATED_LOG:-}" "${VXFEC_CANCEL_FILE:-}" "${VXFEC_TEMP_FILES[@]}"

    if [[ "${KEEP_VMS:-0}" == "1" ]]; then
        echo "[${VXFEC_CLEANUP_TAG}] KEEP_VMS=1; leaving instances and networks up."
        exit "${exit_rc}"
    fi

    # Intersect recorded run-created VMs with live vxfec-run label ownership so neither
    # recycled Spot names nor retained KEEP_VMS=1 instances from another invocation can be deleted.
    local verified_vms=()
    local list_out name zone cand entry vm_lbl desc_err
    if list_out="$(gcloud compute instances list --project="${GCP_PROJECT}" \
        --filter="labels.vxfec-run=${VXFEC_RUN_TOKEN}" --format="value(name,zone)")"; then
        while read -r name zone; do
            if [[ -n "${name}" && -n "${zone}" ]]; then
                cand="${name}:${zone##*/}"
                for entry in "${VXFEC_CREATED_VMS[@]}"; do
                    if [[ "${entry}" == "${cand}" ]]; then
                        vxfec_add_unique verified_vms "${cand}"
                        break
                    fi
                done
            fi
        done <<< "${list_out}"
    else
        echo "WARN: [${VXFEC_CLEANUP_TAG}] instances list failed; verifying recorded VMs individually..." >&2
        for entry in "${VXFEC_CREATED_VMS[@]}"; do
            name="${entry%%:*}"
            zone="${entry#*:}"
            if vm_lbl="$(gcloud compute instances describe "${name}" \
                --project="${GCP_PROJECT}" --zone="${zone}" --format="value(labels.vxfec-run)" 2>/dev/null)"; then
                if [[ "${vm_lbl}" == "${VXFEC_RUN_TOKEN}" ]]; then
                    vxfec_add_unique verified_vms "${name}:${zone}"
                fi
            else
                desc_err="$(gcloud compute instances describe "${name}" \
                    --project="${GCP_PROJECT}" --zone="${zone}" --format="value(name)" 2>&1 >/dev/null || true)"
                if echo "${desc_err}" | grep -Eiq "was not found|NOT_FOUND"; then
                    continue
                fi
                echo "ERROR: [${VXFEC_CLEANUP_TAG}] Could not verify ownership/state of VM ${name} in ${zone}: ${desc_err}" >&2
                exit_rc=1
            fi
        done
    fi

    if [[ ${#verified_vms[@]} -gt 0 || ${#VXFEC_CREATED_VPCS[@]} -gt 0 || \
          ${#VXFEC_CREATED_SUBNETS[@]} -gt 0 || ${#VXFEC_CREATED_FWS[@]} -gt 0 ]]; then
        echo "[${VXFEC_CLEANUP_TAG}] Tearing down run-created GCP resources..."
    fi

    local del_pids=()
    local del_entries=()
    for entry in "${verified_vms[@]}"; do
        gcloud compute instances delete "${entry%%:*}" \
            --project="${GCP_PROJECT}" --zone="${entry#*:}" --quiet &
        del_pids+=("$!")
        del_entries+=("${entry}")
    done
    local i still_there
    for i in "${!del_pids[@]}"; do
        if ! wait "${del_pids[$i]}"; then
            entry="${del_entries[$i]}"
            name="${entry%%:*}"
            zone="${entry#*:}"
            if ! still_there="$(gcloud compute instances list --project="${GCP_PROJECT}" --zones="${zone}" \
                --filter="name=('${name}') AND labels.vxfec-run='${VXFEC_RUN_TOKEN}'" --format="value(name)")"; then
                echo "ERROR: [${VXFEC_CLEANUP_TAG}] Failed to verify deletion of VM ${name} in ${zone}!" >&2
                exit_rc=1
            elif [[ -n "${still_there}" ]]; then
                if ! gcloud compute instances delete "${name}" --project="${GCP_PROJECT}" --zone="${zone}" --quiet; then
                    echo "ERROR: [${VXFEC_CLEANUP_TAG}] Failed to delete VM ${name} in ${zone}!" >&2
                    exit_rc=1
                fi
            fi
        fi
    done

    local sub_pids=()
    local sub_name sub_rest sub_reg
    for entry in "${VXFEC_CREATED_SUBNETS[@]}"; do
        sub_name="${entry%%:*}"
        sub_rest="${entry#*:}"
        sub_reg="${sub_rest%%:*}"
        if ! gcloud compute networks subnets describe "${sub_name}" \
            --project="${GCP_PROJECT}" --region="${sub_reg}" >/dev/null 2>&1; then
            continue
        fi
        if vxfec_subnet_is_unused "${sub_name}" "${sub_reg}"; then
            gcloud compute networks subnets delete "${sub_name}" \
                --project="${GCP_PROJECT}" --region="${sub_reg}" --quiet 2>/dev/null &
            sub_pids+=("$!")
        else
            echo "[${VXFEC_CLEANUP_TAG}] Preserving subnet ${sub_name} (in use by active instances)."
        fi
    done
    for pid in "${sub_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    local checked_vpcs=()
    local unused_vpcs=()
    local fw_name fw_vpc vn is_checked is_unused v
    local fw_pids=()
    for entry in "${VXFEC_CREATED_FWS[@]}"; do
        fw_name="${entry%%:*}"
        fw_vpc="${entry#*:}"
        if ! gcloud compute firewall-rules describe "${fw_name}" --project="${GCP_PROJECT}" >/dev/null 2>&1; then
            continue
        fi
        is_checked=0
        is_unused=0
        for v in "${checked_vpcs[@]}"; do
            if [[ "${v}" == "${fw_vpc}" ]]; then
                is_checked=1
            fi
        done
        if [[ "${is_checked}" == "0" ]]; then
            checked_vpcs+=("${fw_vpc}")
            if vxfec_vpc_is_unused "${fw_vpc}"; then
                unused_vpcs+=("${fw_vpc}")
            fi
        fi
        for v in "${unused_vpcs[@]}"; do
            if [[ "${v}" == "${fw_vpc}" ]]; then
                is_unused=1
            fi
        done
        if [[ "${is_unused}" == "1" ]]; then
            gcloud compute firewall-rules delete "${fw_name}" --project="${GCP_PROJECT}" --quiet 2>/dev/null &
            fw_pids+=("$!")
        else
            echo "[${VXFEC_CLEANUP_TAG}] Preserving firewall rule ${fw_name} (VPC ${fw_vpc} still has active instances)."
        fi
    done
    for pid in "${fw_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    local vpc_pids=()
    for vn in "${VXFEC_CREATED_VPCS[@]}"; do
        if ! gcloud compute networks describe "${vn}" --project="${GCP_PROJECT}" >/dev/null 2>&1; then
            continue
        fi
        is_checked=0
        is_unused=0
        for v in "${checked_vpcs[@]}"; do
            if [[ "${v}" == "${vn}" ]]; then
                is_checked=1
            fi
        done
        if [[ "${is_checked}" == "0" ]]; then
            checked_vpcs+=("${vn}")
            if vxfec_vpc_is_unused "${vn}"; then
                unused_vpcs+=("${vn}")
            fi
        fi
        for v in "${unused_vpcs[@]}"; do
            if [[ "${v}" == "${vn}" ]]; then
                is_unused=1
            fi
        done
        if [[ "${is_unused}" == "1" ]]; then
            gcloud compute networks delete "${vn}" --project="${GCP_PROJECT}" --quiet 2>/dev/null &
            vpc_pids+=("$!")
        else
            echo "[${VXFEC_CLEANUP_TAG}] Preserving VPC ${vn} (still has active instances)."
        fi
    done
    for pid in "${vpc_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    exit "${exit_rc}"
}

vxfec_setup_cleanup_traps() {
    trap vxfec_cleanup EXIT
    trap 'vxfec_on_signal 130' INT
    trap 'vxfec_on_signal 143' TERM
}
