#!/usr/bin/env bash
#
# run_rocev2_clos.sh — Local Multi-Spine 2-Leaf / N-Spine L3 ECMP Clos + Soft-RoCEv2 (rdma_rxe) Benchmark
#
# Default Topology (8 Spines = 14 Namespaces, MTU 9000 Jumbo Frames, Linux fib_multipath_hash_policy=1 L4 ECMP):
#
#   [vxf_h0: 10.10.0.2] ──┐                    ┌── [vxf_spine0..7] (8x Parallel Clos Spines, E-core pinned) ──┐                    ┌── [vxf_h2: 10.10.2.2]
#                         ├── [vxf_leaf0 ECMP] ┤      * 1 Hot Spine (Elephant Collision)                      ├── [vxf_leaf1 ECMP] ┤
#   [vxf_h1: 10.10.1.2] ──┘                    └      * 7 Clean Spines (87.5% Open Bisection)               ──┘                    └── [vxf_h3: 10.10.3.2]
#
# Modes:
#   1. Unprivileged (default, $0/zero-sudo via `unshare -Urnm`):
#      Builds a production `-DVXFEC_ENABLE_SIM=0` `libnccl-vxfec.so` (zero simulator hooks!) and runs
#      live 4-rank ROCm/CUDA GPU `MoE AllToAll` + `DP AllReduce` + `SGEMM` across the 8-spine Clos.
#   2. Kernel Soft-RoCEv2 (`sudo ./scripts/rocev2_clos/run_rocev2_clos.sh --rxe`):
#      Loads `rdma_rxe`, sets `rdma system set netns exclusive`, creates `rxe0` (UDP port 4791, `is_roce=1`)
#      inside `vxf_h0..3`, and compares Stock RCCL/NCCL `NCCL_IB` (`IBV_QPT_RC`) head-to-head against
#      `vxfec` (`IBV_QPT_UC` 16 QPs + per-QP UC loss eviction) over the exact same Clos fabric.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

USE_RXE=0
INNER_RUN=0
NUM_SPINES=8
ITERS=20
HOST_NIC_MBPS=1200
SPINE_MBPS=1000
ELEPHANT_MBPS=1040
MMU_KB=160
ECN_KB=96
ONE_WAY_US=1500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rxe) USE_RXE=1; shift ;;
        --spines) NUM_SPINES="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --host-mbps) HOST_NIC_MBPS="$2"; shift 2 ;;
        --spine-mbps) SPINE_MBPS="$2"; shift 2 ;;
        --elephant-mbps) ELEPHANT_MBPS="$2"; shift 2 ;;
        --mmu-kb) MMU_KB="$2"; shift 2 ;;
        --ecn-kb) ECN_KB="$2"; shift 2 ;;
        --one-way-us) ONE_WAY_US="$2"; shift 2 ;;
        --_inner) INNER_RUN=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "$NUM_SPINES" -lt 2 || "$NUM_SPINES" -gt 8 ]]; then
    echo "ERROR: --spines must be between 2 and 8 (got $NUM_SPINES)" >&2
    exit 1
fi

if [[ "${USE_RXE}" -eq 1 && "$(id -u)" -ne 0 ]]; then
    echo "ERROR: --rxe requires root (run: sudo $0 --rxe)" >&2
    exit 1
fi

command -v ethtool >/dev/null || { echo "ERROR: ethtool is required for packet-level Clos forwarding" >&2; exit 1; }
TOTAL_CPUS="$(nproc)"

# Step 1: Compile clos_switch + production -DVXFEC_ENABLE_SIM=0 libnccl-vxfec.so in host workspace first
if [[ "$INNER_RUN" -eq 0 ]]; then
    mkdir -p build build-nosim
    cc -O3 -g -std=c11 -Wall -Wextra -Werror scripts/rocev2_clos/clos_switch.c -o build/clos_switch
    make -j"$TOTAL_CPUS" all
    make -j"$TOTAL_CPUS" BUILD_DIR=build-nosim OUT_PREFIX=build-nosim/ \
        CPPFLAGS="-DVXFEC_ENABLE_SIM=0" \
        build-nosim/libnccl-vxfec.so
fi

# Step 2: Re-exec inside unshare -Urnm when running unprivileged so `ip netns` works without sudo
if [[ "$(id -u)" -ne 0 ]]; then
    exec unshare -Urnm bash "$0" --_inner \
        --spines "$NUM_SPINES" \
        --iters "$ITERS" \
        --host-mbps "$HOST_NIC_MBPS" \
        --spine-mbps "$SPINE_MBPS" \
        --elephant-mbps "$ELEPHANT_MBPS" \
        --mmu-kb "$MMU_KB" \
        --ecn-kb "$ECN_KB" \
        --one-way-us "$ONE_WAY_US"
fi

WORK_TMP="$(mktemp -d /tmp/vxfec_clos.XXXXXX)"
ORIG_RDMA_MODE=""
BG_PIDS=()
SPINE_PIDS=()
ALL_NS=(vxf_h0 vxf_h1 vxf_h2 vxf_h3 vxf_leaf0 vxf_leaf1)
created_namespaces=()
for ((s = 0; s < NUM_SPINES; s++)); do
    ALL_NS+=("vxf_spine${s}")
done

cleanup() {
    set +e
    for p in "${BG_PIDS[@]:-}"; do
        [[ -n "$p" ]] && kill "$p" 2>/dev/null
    done
    wait 2>/dev/null || true
    for ns in "${created_namespaces[@]}"; do
        ip netns del "$ns" 2>/dev/null || true
    done
    if [[ -n "$ORIG_RDMA_MODE" ]]; then
        rdma system set netns "$ORIG_RDMA_MODE" 2>/dev/null || true
    fi
    rm -rf "$WORK_TMP"
}
trap cleanup EXIT

if [[ ! -w /run ]]; then
    export XDG_RUNTIME_DIR="$WORK_TMP/xdg"
    mkdir -p "$XDG_RUNTIME_DIR"
    mount -t tmpfs tmpfs /run
fi
mkdir -p /run/netns

enable_rxe() {
    local original_mode
    modprobe rdma_rxe
    original_mode="$(rdma system show | awk '{for (i=1; i<NF; i++) if ($i == "netns") {print $(i+1); exit}}')"
    case "$original_mode" in
        exclusive) return ;;
        shared)
            rdma system set netns exclusive
            ORIG_RDMA_MODE="$original_mode"
            ;;
        *) echo "Cannot determine original RDMA netns mode" >&2; exit 1 ;;
    esac
}

if [[ "$USE_RXE" -eq 1 ]]; then
    # RDMA namespace mode is host-wide. Hold the lock through its restoration.
    exec {rdma_lock_fd}>>/run/vxfec-clos-rxe.lock
    flock -n "$rdma_lock_fd" || { echo "Another Soft-RoCE Clos run is active" >&2; exit 1; }
    enable_rxe
fi

create_namespaces() {
    local ns
    for ns in "${ALL_NS[@]}"; do
        ip netns add "$ns"
        created_namespaces+=("$ns")
        ip -n "$ns" link set lo up
    done
}

create_namespaces

disable_rp_filter() {
    local ns="$1"
    ip netns exec "$ns" sysctl -qw net.ipv4.conf.all.rp_filter=0
    ip netns exec "$ns" sysctl -qw net.ipv4.conf.default.rp_filter=0
}

setup_router_ns() {
    local ns="$1"
    disable_rp_filter "$ns"
    ip netns exec "$ns" sysctl -qw net.ipv4.ip_forward=1
    ip netns exec "$ns" sysctl -qw net.ipv4.fib_multipath_hash_policy=1
    ip netns exec "$ns" sysctl -qw net.ipv4.tcp_ecn=1
}

for h in vxf_h0 vxf_h1 vxf_h2 vxf_h3; do
    disable_rp_filter "$h"
done
setup_router_ns vxf_leaf0
setup_router_ns vxf_leaf1

# Keep MMU/ECN/shaping units at one Ethernet frame per queue entry. Preserve
# checksum offload metadata with PACKET_VNET_HDR, but prohibit GSO/GRO batches.
disable_link_offloads() {
    local ns="$1" dev="$2" features
    ip netns exec "$ns" ethtool -K "$dev" tso off gso off gro off lro off \
        tx-udp-segmentation off rx-gro-hw off rx-udp-gro-forwarding off >/dev/null 2>&1 || true
    features="$(ip netns exec "$ns" ethtool -k "$dev")" || return 1
    if ! awk '
        BEGIN {
            split("tcp-segmentation-offload generic-segmentation-offload generic-receive-offload", core)
            for (i in core) {required[core[i]]=1; wanted[core[i]]=1}
            split("large-receive-offload tx-udp-segmentation rx-gro-hw rx-udp-gro-forwarding", optional)
            for (i in optional) wanted[optional[i]]=1
        }
        {key=$1; sub(/:$/, "", key); if (wanted[key]) {seen[key]=1; if ($2 != "off") bad=1}}
        END {for (key in required) if (!seen[key]) bad=1; exit bad ? 1 : 0}
    ' <<< "$features"; then
        echo "ERROR: segmentation/coalescing remains enabled or unverifiable on $ns/$dev" >&2
        return 1
    fi
}

# Wire Hosts <-> Leaves (MTU 9000 Jumbo frames so rdma_rxe activates IBV_MTU_4096)
wire_host() {
    local h_idx="$1" leaf_ns="$2" subnet="$3"
    local h_ns="vxf_h${h_idx}"
    local l_if="l_h${h_idx}"
    ip link add eth0 netns "$h_ns" type veth peer name "$l_if" netns "$leaf_ns"
    ip -n "$h_ns" link set eth0 mtu 9000 up
    ip -n "$leaf_ns" link set "$l_if" mtu 9000 up
    disable_link_offloads "$h_ns" eth0
    disable_link_offloads "$leaf_ns" "$l_if"
    ip -n "$h_ns" addr add "10.10.${subnet}.2/24" dev eth0
    ip -n "$leaf_ns" addr add "10.10.${subnet}.1/24" dev "$l_if"
    ip -n "$h_ns" route add default via "10.10.${subnet}.1" dev eth0
    ip netns exec "$h_ns" sysctl -qw net.ipv4.conf.eth0.rp_filter=0
    ip netns exec "$leaf_ns" sysctl -qw "net.ipv4.conf.${l_if}.rp_filter=0"
    if [[ "$USE_RXE" -eq 1 ]]; then
        ip netns exec "$h_ns" rdma link add rxe0 type rxe netdev eth0
    fi
}

wire_host 0 vxf_leaf0 0
wire_host 1 vxf_leaf0 1
wire_host 2 vxf_leaf1 2
wire_host 3 vxf_leaf1 3

# Wire Leaf0 <-> Spine[0..N-1] <-> Leaf1
NH_LEAF0=()
NH_LEAF1=()
for ((s = 0; s < NUM_SPINES; s++)); do
    sp_ns="vxf_spine${s}"
    ip link add "l0_s${s}" netns vxf_leaf0 type veth peer name "s${s}_l0" netns "$sp_ns"
    ip link add "l1_s${s}" netns vxf_leaf1 type veth peer name "s${s}_l1" netns "$sp_ns"

    ip -n vxf_leaf0 link set "l0_s${s}" mtu 9000 up
    ip -n vxf_leaf1 link set "l1_s${s}" mtu 9000 up
    ip -n "$sp_ns" link set "s${s}_l0" mtu 9000 up
    ip -n "$sp_ns" link set "s${s}_l1" mtu 9000 up
    disable_link_offloads vxf_leaf0 "l0_s${s}"
    disable_link_offloads vxf_leaf1 "l1_s${s}"
    disable_link_offloads "$sp_ns" "s${s}_l0"
    disable_link_offloads "$sp_ns" "s${s}_l1"

    ip -n vxf_leaf0 addr add "10.20.${s}.1/30" dev "l0_s${s}"
    ip -n vxf_leaf1 addr add "10.20.${s}.2/30" dev "l1_s${s}"
    ip netns exec vxf_leaf0 sysctl -qw "net.ipv4.conf.l0_s${s}.rp_filter=0"
    ip netns exec vxf_leaf1 sysctl -qw "net.ipv4.conf.l1_s${s}.rp_filter=0"

    NH_LEAF0+=(nexthop via "10.20.${s}.2" dev "l0_s${s}" weight 1)
    NH_LEAF1+=(nexthop via "10.20.${s}.1" dev "l1_s${s}" weight 1)
done

# Equal-Cost Multi-Path (ECMP) routes on Leaf0 & Leaf1 across all N Spines
for dst_sub in 2 3; do
    ip -n vxf_leaf0 route replace "10.10.${dst_sub}.0/24" "${NH_LEAF0[@]}"
done
for dst_sub in 0 1; do
    ip -n vxf_leaf1 route replace "10.10.${dst_sub}.0/24" "${NH_LEAF1[@]}"
done

# Launch Spine[0..N-1] MMU Switch ASIC emulators (1 dedicated E-core per spine on >= 20-core hosts)
for ((s = 0; s < NUM_SPINES; s++)); do
    if [[ "$TOTAL_CPUS" -ge 20 ]]; then
        sp_cpu=$(( 12 + (s % 8) ))
        taskset -c "$sp_cpu" ip netns exec "vxf_spine${s}" "$ROOT_DIR/build/clos_switch" switch "s${s}_l0" "s${s}_l1" \
            "$SPINE_MBPS" "$MMU_KB" "$ECN_KB" "$ONE_WAY_US" "$WORK_TMP/spine${s}.stats" &
    elif [[ "$TOTAL_CPUS" -ge 16 ]]; then
        sp_cpu=$(( 8 + (s % (TOTAL_CPUS - 8)) ))
        taskset -c "$sp_cpu" ip netns exec "vxf_spine${s}" "$ROOT_DIR/build/clos_switch" switch "s${s}_l0" "s${s}_l1" \
            "$SPINE_MBPS" "$MMU_KB" "$ECN_KB" "$ONE_WAY_US" "$WORK_TMP/spine${s}.stats" &
    else
        ip netns exec "vxf_spine${s}" "$ROOT_DIR/build/clos_switch" switch "s${s}_l0" "s${s}_l1" \
            "$SPINE_MBPS" "$MMU_KB" "$ECN_KB" "$ONE_WAY_US" "$WORK_TMP/spine${s}.stats" &
    fi
    sp_pid="$!"
    BG_PIDS+=("$sp_pid")
    SPINE_PIDS+=("$sp_pid")
done
sleep 0.2

# Warm up ARP across Leaf0 <-> Leaf1 <-> Hosts
ip netns exec vxf_h0 ping -c 1 -W 2 10.10.2.2 >/dev/null
ip netns exec vxf_h1 ping -c 1 -W 2 10.10.3.2 >/dev/null

flush_spine_stats() {
    local s before current task_deadline
    for ((s = 0; s < NUM_SPINES; s++)); do
        before="$(read_stat_val "$WORK_TMP/spine${s}.stats" stat_seq)"
        [[ "$before" =~ ^[0-9]+$ ]] || return 1
        kill -USR1 "${SPINE_PIDS[s]}" || return 1
        task_deadline=$((SECONDS + 5))
        while :; do
            check_spine_health || return 1
            current="$(read_stat_val "$WORK_TMP/spine${s}.stats" stat_seq)"
            if [[ "$current" =~ ^[0-9]+$ ]] && (( current > before )); then break; fi
            if (( SECONDS >= task_deadline )); then
                echo "ERROR: spine$s did not publish fresh forwarding statistics" >&2
                return 1
            fi
            sleep 0.01
        done
    done
}

read_stat_val() {
    local file="$1" key="$2"
    awk -v k="$key" '{for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]==k){print a[2]; exit}}}' "$file"
}

check_spine_health() {
    local s
    for ((s = 0; s < NUM_SPINES; s++)); do
        if ! kill -0 "${SPINE_PIDS[s]}" 2>/dev/null ||
           [[ "$(read_stat_val "$WORK_TMP/spine${s}.stats" valid)" != 1 ]]; then
            echo "ERROR: spine$s exited or reported invalid packet forwarding" >&2
            return 1
        fi
    done
}

monitor_case() {
    # Stop ranks promptly if the switch rejects a frame or records hidden loss.
    # Otherwise collectives could wait for a failed forwarding process forever.
    while check_spine_health; do sleep 0.02; done
    kill "$@" 2>/dev/null || true
    return 1
}

check_spine_health

# Start persistent cross-rack elephant flow (h0:50021 -> h2:52001) and read which symmetric spine it hashed onto
HOT_SPINE=0
if [[ "$ELEPHANT_MBPS" -gt 0 ]]; then
    ele_cpu=$(( TOTAL_CPUS >= 20 ? 10 : (TOTAL_CPUS > 1 ? TOTAL_CPUS - 1 : 0) ))
    ele_tx_cpu=$(( TOTAL_CPUS >= 20 ? 11 : ele_cpu ))
    taskset -c "$ele_cpu" ip netns exec vxf_h2 "$ROOT_DIR/build/clos_switch" elephant-rx 10.10.2.2 52001 &
    BG_PIDS+=("$!")
    sleep 0.05
    taskset -c "$ele_tx_cpu" ip netns exec vxf_h0 "$ROOT_DIR/build/clos_switch" elephant-tx 10.10.0.2 50021 10.10.2.2 52001 "$ELEPHANT_MBPS" 4096 &
    BG_PIDS+=("$!")
    sleep 0.20
    flush_spine_stats
    max_pkts=0
    for ((s = 0; s < NUM_SPINES; s++)); do
        pk="$(read_stat_val "$WORK_TMP/spine${s}.stats" tot_tx_pkts)"
        if [[ "$pk" -gt "$max_pkts" ]]; then
            max_pkts="$pk"
            HOT_SPINE="$s"
        fi
    done
fi

TOTAL_NS=$(( 6 + NUM_SPINES ))
SPRAY_PORTS=$(( USE_RXE == 1 ? 16 : 32 ))
echo "==============================================================================================================================="
echo "Local ${TOTAL_NS}-Namespace 2-Leaf / ${NUM_SPINES}-Spine L3 Clos Fabric (Production -DVXFEC_ENABLE_SIM=0, Zero Synthetic Drops)"
echo "  Topology    : [h0,h1] <-> [leaf0 L4-ECMP] <-> [${NUM_SPINES}x spines @ ${SPINE_MBPS} Mbps ($((NUM_SPINES * SPINE_MBPS)) Mbps bisection), MMU=${MMU_KB}KB, ECN=${ECN_KB}KB, 1-way=${ONE_WAY_US}us] <-> [leaf1] <-> [h2,h3]"
echo "  CPU Pinning : $( [[ "$TOTAL_CPUS" -ge 20 ]] && echo "Ranks 0..3 -> Shared P-core Pool 0..9 | Elephant -> 10..11 | Spines 0..$((NUM_SPINES-1)) -> Dedicated E-cores 12..$((11 + NUM_SPINES))" || echo "Ranks 0..3 -> P-cores 0..7 | Spines 0..$((NUM_SPINES-1)) -> E-cores 8..$((7 + NUM_SPINES))" )"
echo "  Collision   : Cross-rack elephant flow (${ELEPHANT_MBPS} Mbps, h0:50021 -> h2:52001) hashed by kernel onto spine${HOT_SPINE} ($((NUM_SPINES - 1))/${NUM_SPINES} spines clean)"
echo "  Transport   : $( [[ "$USE_RXE" -eq 1 ]] && echo "Kernel Soft-RoCEv2 (rdma_rxe0, UDP 4791, MTU 4096)" || echo "Routed Clos Ethernet + GPU GDR (run with 'sudo $0 --rxe' for rdma_rxe)" )"
echo "==============================================================================================================================="

run_clos_case() {
    local label="$1"
    local mode="$2"      # native-ib | vxfec-ibv | vxfec-udp
    local codec="$3"     # rc | gbn | sack | auto
    local nports="$4"    # 1..32
    local port_base="$5"

    flush_spine_stats
    local b0=() e0=() d0=()
    for ((s = 0; s < NUM_SPINES; s++)); do
        b0+=("$(read_stat_val "$WORK_TMP/spine${s}.stats" tot_app_tx_bytes)")
        e0+=("$(read_stat_val "$WORK_TMP/spine${s}.stats" tot_app_ecn_marks)")
        d0+=("$(read_stat_val "$WORK_TMP/spine${s}.stats" tot_app_tail_drops)")
    done

    local rcv_err0=0
    for r in 0 1 2 3; do
        local re
        re="$(ip netns exec "vxf_h${r}" awk '/^Udp:/{cnt++; if(cnt==2) print $6}' /proc/net/snmp)"
        rcv_err0=$(( rcv_err0 + ${re:-0} ))
    done

    local num_gpus=0
    local drm_cards=()
    shopt -s nullglob
    drm_cards=(/sys/class/drm/card[0-9]*)
    shopt -u nullglob
    num_gpus="${#drm_cards[@]}"
    local gpu_fec_be="auto"
    local vk_mode="1"
    if [[ "$num_gpus" -le 1 ]]; then
        # Avoid 4-process single-GPU KFD/RADV queue preemption against RCCL persistent Ring kernels
        gpu_fec_be="cpu"
        vk_mode="0"
    fi

    local rank_pids=()
    for r in 0 1 2 3; do
        local h_ns="vxf_h${r}"
        local extra_args=(--train-step --clos-fabric --master "10.10.0.2:${port_base}" --rank "$r" --nranks 4 --ifname eth0 --iters "$ITERS" --label-tag "$label")
        local env_vars=(
            VXFEC_NCCL_SO_PATH="$ROOT_DIR/build-nosim/libnccl-vxfec.so"
            NCCL_SOCKET_IFNAME=eth0
            NCCL_MAX_NCHANNELS=2
            VXFEC_NCCL_IFNAMES=eth0
            VXFEC_NCCL_PORTS="$nports"
            VXFEC_NCCL_MAX_RATE_MBPS="$HOST_NIC_MBPS"
            VXFEC_NCCL_GPU_BACKEND="$gpu_fec_be"
            VXFEC_NCCL_VULKAN="$vk_mode"
        )
        if [[ "$mode" == "native-ib" ]]; then
            extra_args+=(--native-ib --ibv-devs rxe0:1)
            env_vars+=(NCCL_IB_HCA="=rxe0:1" NCCL_IB_ROCE_VERSION_NUM=2)
        elif [[ "$mode" == "vxfec-ibv" ]]; then
            extra_args+=(--backend ibv --single-codec "$codec" --ibv-devs rxe0:1)
        else
            extra_args+=(--backend udp --single-codec "$codec")
        fi

        local r_cpus="0-$((TOTAL_CPUS - 1))"
        if [[ "$TOTAL_CPUS" -ge 20 ]]; then
            r_cpus="0-9"
        elif [[ "$TOTAL_CPUS" -ge 16 ]]; then
            r_cpus="0-7"
        fi

        if [[ "$r" -eq 0 ]]; then
            taskset -c "$r_cpus" ip netns exec "$h_ns" env "${env_vars[@]}" "$ROOT_DIR/demo_nccl_e2e" "${extra_args[@]}" >"$WORK_TMP/r0.out" 2>"$WORK_TMP/r0.err" &
        else
            taskset -c "$r_cpus" ip netns exec "$h_ns" env "${env_vars[@]}" "$ROOT_DIR/demo_nccl_e2e" "${extra_args[@]}" >/dev/null 2>"$WORK_TMP/r${r}.err" &
        fi
        rank_pids+=("$!")
        BG_PIDS+=("$!")
    done
    monitor_case "${rank_pids[@]}" &
    local monitor_pid=$!
    BG_PIDS+=("$monitor_pid")

    local rc=0
    for pid in "${rank_pids[@]}"; do
        if ! wait "$pid"; then
            rc=1
        fi
    done

    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    # Forget finished temporary children so cleanup cannot target reused PIDs.
    local -a keep_pids=()
    local bg temporary is_temporary
    for bg in "${BG_PIDS[@]}"; do
        is_temporary=0
        for temporary in "${rank_pids[@]}" "$monitor_pid"; do
            if [[ "$bg" == "$temporary" ]]; then is_temporary=1; break; fi
        done
        if (( ! is_temporary )); then keep_pids+=("$bg"); fi
    done
    BG_PIDS=("${keep_pids[@]}")
    if ! flush_spine_stats; then
        echo "  [FAILED] $label: invalid or unavailable Clos forwarding telemetry" >&2
        cat "$WORK_TMP/r0.err" >&2 || true
        return 1
    fi
    local hot_mb clean_mb=0 hot_e hot_d clean_e=0 clean_d=0
    hot_mb="$(awk -v a="${b0[$HOT_SPINE]}" -v b="$(read_stat_val "$WORK_TMP/spine${HOT_SPINE}.stats" tot_app_tx_bytes)" 'BEGIN{printf "%.1f", (b-a)/1048576.0}')"
    hot_e=$(( $(read_stat_val "$WORK_TMP/spine${HOT_SPINE}.stats" tot_app_ecn_marks) - e0[HOT_SPINE] ))
    hot_d=$(( $(read_stat_val "$WORK_TMP/spine${HOT_SPINE}.stats" tot_app_tail_drops) - d0[HOT_SPINE] ))

    local clean_bytes_sum=0
    local per_spine_str=""
    for ((s = 0; s < NUM_SPINES; s++)); do
        [[ "$s" -eq "$HOT_SPINE" ]] && continue
        local sb1 se1 sd1 db sm
        sb1="$(read_stat_val "$WORK_TMP/spine${s}.stats" tot_app_tx_bytes)"
        se1="$(read_stat_val "$WORK_TMP/spine${s}.stats" tot_app_ecn_marks)"
        sd1="$(read_stat_val "$WORK_TMP/spine${s}.stats" tot_app_tail_drops)"
        db=$(( sb1 - b0[s] ))
        clean_bytes_sum=$(( clean_bytes_sum + db ))
        clean_e=$(( clean_e + (se1 - e0[s]) ))
        clean_d=$(( clean_d + (sd1 - d0[s]) ))
        sm="$(awk -v x="$db" 'BEGIN{printf "%.0f", x/1048576.0}')"
        per_spine_str+=" s${s}=${sm}M"
    done
    clean_mb="$(awk -v x="$clean_bytes_sum" 'BEGIN{printf "%.1f", x/1048576.0}')"

    local rcv_err1=0
    for r in 0 1 2 3; do
        local re
        re="$(ip netns exec "vxf_h${r}" awk '/^Udp:/{cnt++; if(cnt==2) print $6}' /proc/net/snmp)"
        rcv_err1=$(( rcv_err1 + ${re:-0} ))
    done
    local rcv_drops=$(( rcv_err1 - rcv_err0 ))

    echo ""
    if [[ "$rc" -ne 0 ]]; then
        echo "  [FAILED] $label"
        cat "$WORK_TMP/r0.err" >&2 || true
        return 1
    fi
    grep -E "Solo GPU|Training Collective|---|MoE AllToAll|DP AllReduce" "$WORK_TMP/r0.out"
    echo "  [Clos Telemetry] spine${HOT_SPINE}(hot): ${hot_mb} MiB, ECN=${hot_e}, TailDrops=${hot_d} | $((NUM_SPINES-1)) clean spines: ${clean_mb} MiB (${per_spine_str# }), ECN=${clean_e}, TailDrops=${clean_d} | HostRcvDrops=${rcv_drops}"
}

if [[ "$USE_RXE" -eq 1 ]]; then
    run_clos_case "Stock NCCL_IB RC"         native-ib rc   1             29610
    run_clos_case "1-QP RoCE SACK"           vxfec-ibv sack 1             29620
    run_clos_case "${SPRAY_PORTS}-QP vxfec UC" vxfec-ibv auto "$SPRAY_PORTS" 29630
else
    run_clos_case "1-Flow GBN"               vxfec-udp gbn  1             29610
    run_clos_case "1-Flow SACK"              vxfec-udp sack 1             29620
    run_clos_case "${SPRAY_PORTS}-Flow SACK"  vxfec-udp sack "$SPRAY_PORTS" 29630
    run_clos_case "${SPRAY_PORTS}-Flow vxfec" vxfec-udp auto "$SPRAY_PORTS" 29640
fi

echo ""
echo "==============================================================================================================================="
echo "  PASS: All 4-Host ${NUM_SPINES}-Spine Clos ECMP collectives verified 100% bit-exact with -DVXFEC_ENABLE_SIM=0 (0 synthetic drops)."
echo "==============================================================================================================================="
