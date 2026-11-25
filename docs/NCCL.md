# NCCL / RCCL plugin reference

`libnccl-vxfec.so` implements the NCCL/RCCL Net v8 interface as `vxfec-ecmp-harq`. It supports UDP and `libibverbs` UC transport, multiple NICs, and GPU memory registration where the driver and NIC support DMA-BUF.

## Setup

```sh
export NCCL_NET_PLUGIN="$PWD/libnccl-vxfec.so"
export NCCL_NET="vxfec-ecmp-harq"
export VXFEC_NCCL_IFNAMES="eth0,eth1"
```

Choose interfaces that can reach your other hosts. `NCCL_SOCKET_IFNAME` is also supported. Without an interface or bind address, the plugin binds only to loopback (`127.0.0.1`). Interface filters accept NCCL's `^` exclusion and `=` exact-match prefixes.

## Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VXFEC_NCCL_IFNAMES` | `NCCL_SOCKET_IFNAME` (`lo` if unset) | Comma-separated interface filter (required for multi-host runs; supports NCCL `^` exclude and `=` exact prefixes). |
| `VXFEC_NCCL_BACKEND` | `udp` | Transport backend: `udp` or `ibv` (`libibverbs` `IBV_QPT_UC` with 1-SGE `RDMA_WRITE_WITH_IMM`, falls back to UDP if unavailable). |
| `VXFEC_NCCL_UDP_GSO` | enabled when supported | Batches eligible UDP datagrams using Linux UDP segmentation. Set to `0` to send individual datagrams. With `NCCL_DEBUG=INFO`, each send connection logs how many datagrams used segmentation when it closes. |
| `VXFEC_NCCL_GPU_BACKEND` | `auto` | GPU FEC backend: `auto` (selects native `cuda`/`hip` when NCCL/RCCL is resident, else `vulkan`), `cuda`, `hip`, `vulkan`, or `cpu`. |
| `VXFEC_NCCL_GREEN_SMS` | `8` | Requested SM count for the FEC stream on drivers with Green Context support. The driver may round the count up. A value of `0` selects an ordinary stream. |
| `VXFEC_NCCL_CU_COUNT` | `0` *(high-prio stream)* | Requested HIP compute-unit mask size. A value of `0` selects a priority stream. Only a verified mask is reported as a partition. |
| `VXFEC_NCCL_RAIL_WORKERS` | `auto` | Dedicated per-rail (`vxfec-rail*`) TX/RX progress threads: `auto` (enabled when `ndev > 1`), `1`/`2` (force on), or `0` (inline caller progress). |
| `VXFEC_NCCL_DEV_SLICE` | *(all NICs)* | MPI 1-process-per-NIC device partition `"p/P"` (`0 <= p < P`) so rank `p` binds exclusively to its local NIC slice. |
| `VXFEC_NCCL_PORTS` | `16` | Multi-path spray fanout per channel: `1–32` UDP ports; `ibv` supports `1–16` QPs (selects `8` when set to `17–32`). |
| `VXFEC_NCCL_CODEC` | `auto` | Recovery policy: `auto`, `rs8` (Cauchy MDS), `rq-gf256`, `rq-gf2` (pure-`GF(2)` fountain mode), `sack`, or `gbn` (cumulative-NACK Go-Back-N). |
| `VXFEC_NCCL_SLICE_KB` | `384` | Slice messages above this threshold into pipelined Cauchy RS8 sub-blocks (`0` disables). |
| `VXFEC_NCCL_STRICT_PIX_MR` | `0` | Require strict same-PCIe-switch (`PIX`) GPU `DMA-BUF` MR affinity per RDMA rail. |

## GPU and RDMA support

The plugin can compile its Cauchy RS8 and RaptorQ kernels through NVRTC or HIPRTC and operate on registered GPU buffers, using a stream and context per GPU. These native backends are optional; the standalone Vulkan libraries do not require CUDA or HIP.

CUDA Green Context support requires CUDA 12.4 or newer driver headers at build time. The build checks the compiler include path and `/usr/local/cuda/include`; set `CUDA_HOME` for a toolkit elsewhere. The ordinary CUDA stream path also builds without CUDA headers. HIP compute-unit masks are reported as partitions only when verified.

The RDMA path uses one-SGE `RDMA_WRITE_WITH_IMM` and zero-SGE receive WQEs for immediate-only notifications. Generation and symbol identifiers travel in the immediate field. Peers negotiate direct repair writes during connection setup. With a legacy peer, repairs use the packet path and the receiver accepts the original 14-bit sliced generation index. GPU DMA-BUF registration and direct receive into GPU memory depend on endpoint support and topology. Use the [hardware probes and benchmarks](BENCHMARKS.md) to verify the actual path.

Per-rail workers provide background progress across NICs. `VXFEC_NCCL_DEV_SLICE=p/P` assigns a process its NIC slice for one-process-per-NIC deployments. Worker and codec performance depend on the workload; compare settings using the [benchmark tools](BENCHMARKS.md).

For deployment builds, pass `CPPFLAGS="-DVXFEC_ENABLE_SIM=0"` to compile out the fault/congestion simulation hooks.
