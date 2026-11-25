# Local hardware benchmarks

Build with `make -j$(nproc)` and run `make check` before benchmarking. `make asan` repeats the correctness suite with Clang ASan/UBSan. GPU-specific tests can skip when the required runtime/device is absent; a skip is not hardware validation. Record the revision, GPU, NIC, topology, message size, codec, loss policy and sample count with each result.

## GPU codecs and transport

| Entry point | Purpose |
| --- | --- |
| `./bench_vulkan` | Resident Vulkan codec throughput and bit-exact recovery |
| `./bench_pipeline` | GPU superframe/interleaving pipeline |
| `./bench_wire_net --self-test` | Wire loopback and repaired payload verification |
| `./bench_nccl` | Host-buffer collective transport and loss scenarios |
| `./bench_nccl --scale-bench --rails 8 --bytes 8388608 --loss-ppm 20000 --burst 6 --iters 100` | Rail scaling with an explicit synthetic loss policy |
| `./bench_nccl --leo-bench --iters 100` | Simulated link handoff and backup-path behavior |
| `./demo_nccl_e2e --ab-bench --gdr --iters 100` | Live GPU NCCL/RCCL AllReduce |
| `./demo_nccl_e2e --train-step --gdr --iters 100` | Overlapped collective/SGEMM training-step workload |

For live multi-host collectives, use `--rank <rank> --nranks <count> --master <rank0-ip:port>` on each host, or `--node-rank <node> --nnodes <count> --gpus-per-node <count>`. Select interfaces with `VXFEC_NCCL_IFNAMES`. The `--native-ib` mode supplies a stock NCCL IB control when working verbs hardware is available. Use the same workload and placement for each control.

The focused `--b200-four-case` and `--h200-four-case` demo profiles remain available for two-rank CUDA/InfiniBand validation, with `--backend ibv --gdr --seed <seed> --iters 100 --csv <path>` and the rank/master options (omit `--backend ibv` and pass `--native-ib` for the native control). These profiles verify every sum, paired first-transmission impairment and native CUDA repair use; they do not provision hardware. A failed correctness gate invalidates timings, while a measured performance regression is a valid result.

For one-way GPU-to-GPU wire tests, run `./bench_wire_net --mode rx --bind <receiver-ip> --gdr --pipeline` on the receiver and `./bench_wire_net --mode tx --peers <receiver-ip> --gdr --pipeline --tx-threads 4 --iters 100 --blackout-pct 10 --csv gdr.csv` on the sender. Both endpoints need compatible Vulkan DMA-BUF export and RDMA registration. Receiver goodput and ready-to-verified latency include GPU readback and byte comparison of every recovered payload.

## Hardware discovery and result checks

```sh
make t/probe_gdr_pair t/test_probe_gdr_pair
./t/probe_gdr_pair
./t/probe_gdr_pair --single-cuda-pair
./t/probe_gdr_pair --single-cuda-vulkan-pair
./t/test_probe_gdr_pair
python3 t/test_nccl_e2e.py
```

The probe checks InfiniBand device topology and actual GPU-memory registration rather than assuming NIC/GPU affinity from enumeration order (`--single-cuda-vulkan-pair` requires a 400 Gbps or faster InfiniBand rail). Its offline fixture checks matching and registration-failure behavior without renting hardware. `t/check_benchmarks.py` validates collective CSVs, per-iteration samples, multi-rail exchange output and topology. The demo fixture uses CPU stand-ins to test control flow and rejection of corrupt/mismatched evidence; it does not establish native GPU performance.

## Local Clos benchmark

Run the namespace-based two-leaf/eight-spine ECMP testbed, optionally using kernel Soft-RoCE:

```sh
./scripts/rocev2_clos/run_rocev2_clos.sh --spines 8 --iters 20
sudo ./scripts/rocev2_clos/run_rocev2_clos.sh --rxe --spines 8 --iters 20
```

The launcher requires `ethtool` and disables and verifies segmentation/coalescing offloads on every virtual-link endpoint so congestion and shaping counters describe individual Ethernet frames. Unexpected aggregates, truncation, socket drops or fatal forwarding errors invalidate the run. Intentional MMU tail drops remain valid benchmark events.

Run the forwarding regressions without GPUs:

```sh
python3 -m unittest discover -s scripts/rocev2_clos -p 'test_*.py'
cc -O2 -std=c11 -Wall -Wextra -Werror scripts/rocev2_clos/clos_switch.c -o /tmp/vxfec-clos-switch
sudo unshare --mount --net --fork python3 scripts/rocev2_clos/test_switch_netns.py --switch /tmp/vxfec-clos-switch
```

Kernel fixtures require root and isolated mount/network namespaces. Software namespaces do not establish physical ECMP path diversity or production-fabric performance.

