# vxfec

**Forward error correction for GPUs, built on a portable Vulkan compute framework.**

`vxfec` adds repair data so missing pieces can be recovered without sending the original data again. It brings Reed–Solomon and RaptorQ coding to GPU memory, with a C11 API you can embed in your own application.

The core uses Vulkan 1.2 to work across GPU vendors without requiring CUDA or HIP. You can use it for erasure coding, streaming over unreliable links, or recovering bursts of missing data. The repository also includes an NCCL/RCCL transport plugin, with optional native CUDA and HIP backends for GPU collectives.

## Getting started

You'll need Linux, a C11 compiler, `make`, and Vulkan 1.2 headers and a compute driver. AMD, NVIDIA and Intel Vulkan drivers are supported; Mesa lavapipe provides a CPU fallback for development.

```sh
git submodule update --init --recursive
make -j$(nproc)
make check
```

For sanitizer checks, run `make asan`. RDMA support is detected when `libibverbs` headers are installed. Live collective examples additionally need CUDA with NCCL or ROCm with RCCL.

## Use it in your application

The build produces `libvxfec_vk.a` and `libvxfec.a` for embedding, plus `libnccl-vxfec.so` for NCCL/RCCL. Start with the header for the feature you need:

- [Vulkan buffers and execution](include/vxfec_vk.h) — GPU contexts, memory import/export and asynchronous work.
- [Reed–Solomon](include/vxfec_rs_cauchy.h) — Cauchy RS8 coding over GF(256) (`1 / ((M + k) ^ m)` and `1 / (α^(254-k) + α^m)`), compatible with [nanors](https://github.com/sleepybishop/nanors).
- [RaptorQ](include/vxfec_rq.h) — RFC 6330 coding and a GF(2) mode, compatible with [nanorq](https://github.com/sleepybishop/nanorq).
- [Burst recovery](include/vxfec_interleave.h) — interleaved GPU buffers that spread a burst of missing data across coding blocks.

## Use it with NCCL or RCCL

Select the plugin and the network interface that connects your hosts:

```sh
export NCCL_NET_PLUGIN="$PWD/libnccl-vxfec.so"
export NCCL_NET="vxfec-ecmp-harq"
export VXFEC_NCCL_IFNAMES="eth0"  # replace with your interface
```

The plugin supports UDP and RDMA transports, multiple NICs, and selectable recovery policies. See the [plugin reference](docs/NCCL.md) for configuration and GPU requirements.

## Benchmark tools

- `bench_vulkan` — GPU codec throughput and recovery.
- `bench_pipeline` — GPU superframe and interleaving pipeline.
- `bench_wire_net` — wire throughput and GPU-to-GPU delivery.
- `bench_nccl` — collective transport, rail scaling and link handoff.
- `demo_nccl_e2e` — live NCCL/RCCL collectives and training-step workloads.
- [`run_rocev2_clos.sh`](scripts/rocev2_clos/run_rocev2_clos.sh) — local ECMP Clos testbed, with optional Soft-RoCE.
- `t/probe_gdr_pair` — GPU/NIC topology and memory-registration checks.

See the [benchmark guide](docs/BENCHMARKS.md) for commands and validation.
