#!/usr/bin/env python3
# Run with: make demo_nccl_e2e && python3 t/test_nccl_e2e.py
"""Run the actual demo executable against a CPU-only GPU and NCCL fixture."""
import csv
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo_nccl_e2e"
CHECK_RUN = ROOT / "t/check_benchmarks.py"
GDR = ["--backend", "ibv", "--gdr"]
AB = ["--ab-bench", "--iters", "3"]
NATIVE = ["--native-ib", "--gdr", *AB]
B200 = ["--b200-four-case", "--iters", "3"]
B200_MARKER = "Verified B200 four-case timed profile"
FULL_NODE = [
    "--nnodes", "2", "--gpus-per-node", "2",
    "--ibv-devs", "mlx5_b:1,mlx5_a:2", "--vk-devs", "0,1",
]


def check(directory, env, name, options, expected, error=False, timed_copy=False, full_node=False, phases=None, remote_pair=False,
          peer_args=(), peer_profile=None):
    csv_path = directory / f"{name}.csv"
    if full_node or remote_pair:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        launch = [*FULL_NODE, "--node-rank"] if full_node else ["--rank"]
        command = [DEMO, *options, "--master", f"127.0.0.1:{port}", *launch]
        commands = [command + ["0", "--csv", csv_path], command + ["1", *peer_args]]
        if peer_profile:
            commands[1] = [peer_profile if str(arg).endswith("-four-case") else arg for arg in commands[1]]
    else:
        commands = [[DEMO, *options, "--csv", csv_path]]

    processes = [
        subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        for command in commands
    ]
    try:
        log = "".join(proc.communicate(timeout=30)[0] for proc in processes)
    finally:
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    statuses = [proc.returncode for proc in processes]
    assert statuses == [int(error)] * len(commands), (name, statuses, log)
    assert expected in log, (name, expected, log)
    assert csv_path.exists() != error, (name, csv_path, log)
    samples_path = Path(f"{csv_path}.samples.csv")
    if phases is not None:
        assert samples_path.exists() != error, (name, samples_path, log)
        profile = next(option[2:] for option in options if option.endswith("-four-case"))
        model = profile.split("-")[0].upper()
        assert (f"Verified {model} four-case timed profile" in log) != error, (name, log)
        if not error:
            with csv_path.open() as fp:
                rows = list(csv.DictReader(fp))
            with samples_path.open() as fp:
                samples = list(csv.DictReader(fp))
            seed = options[options.index("--seed") + 1]
            iters = int(options[options.index("--iters") + 1])
            assert [row["phase"] for row in rows] == phases, rows
            assert [(row["phase"], int(row["iteration"])) for row in samples] == [
                (phase, iteration) for phase in phases for iteration in range(1, iters + 1)
            ], samples
            for row in rows:
                phase = row["phase"]
                loss = phase.endswith("_loss")
                scope = "first-systematic" if loss else "none"
                assert row["profile"] == profile and row["seed"] == seed and row["impairment_scope"] == scope, row
                assert int(row["nranks"]) == 2 and float(row["tensor_mb"]) == 0.5 and int(row["iters"]) == iters, row
                assert int(row["loss_ppm"]) == (20_000 if loss else 0) and int(row["burst"]) == (6 if loss else 1), row
                assert int(row["drops"]) == (4 * iters if loss else 0), row
                assert int(row["systematic_trials"]) == (200 * iters if loss else 0), row
                assert int(row["cuda_rx_repairs"]) == (2 * iters if phase == "plugin_auto_loss" else 0), row
                assert int(row["fallback_rx_repairs"]) == 0, row
                latencies = []
                for sample in samples:
                    if sample["phase"] == phase:
                        assert sample["profile"] == row["profile"] and sample["seed"] == seed and sample["impairment_scope"] == scope
                        latencies.append(int(sample["worst_rank_ns"]))
                assert min(latencies) > 0, latencies
                sorted_ns = sorted(latencies)
                assert abs(float(row["p50_ms"]) - sorted_ns[iters // 2] / 1e6) <= 0.000501, row
                assert abs(float(row["p99_ms"]) - sorted_ns[(99 * iters + 99) // 100 - 1] / 1e6) <= 0.000501, row
            fingerprints = [int(row["drop_fingerprint"]) for row in rows if row["phase"].endswith("_loss")]
            assert not fingerprints or fingerprints == [iters * (2 * int(seed) + 3)] * 2, fingerprints
    if timed_copy:
        with csv_path.open() as fp:
            rows = list(csv.DictReader(fp))
        # The fixture waits 15 ms in every copy. Collective latency excludes both copies.
        assert rows and max(float(row["avg_ms"]) for row in rows) < 15, (name, rows)
    print(f"{name} passed")
    return log


def main():
    with tempfile.TemporaryDirectory(prefix="vxfec-nccl-e2e-") as work:
        directory = Path(work)
        standin = directory / "libnccl.so.2"
        subprocess.run(
            [*shlex.split(os.environ.get("CC", "cc")), "-shared", "-fPIC", f"-I{ROOT / 'plugins/nccl'}",
             ROOT / "t/nccl_e2e_standin.c", "-o", standin, "-ldl"],
            check=True,
        )
        for library in ("libcudart.so.13", "libamdhip64.so.7", "librccl.so.1"):
            (directory / library).symlink_to(standin.name)
        env = dict(os.environ, LD_LIBRARY_PATH=work, VXFEC_NCCL_SO_PATH=str(standin), NCCL_PROTO="LL128")
        for model in ("b200", "h200"):
            profile = f"--{model}-four-case"
            check(directory, env, f"{model} rejects training", [profile, "--train-step", "--nranks", "2", "--seed", "101", *GDR],
                  f"Select either --train-step or {profile}", error=True, phases=[])
        delayed = env | {"TEST_SIMPLE": "1", "TEST_COPY_DELAY": "1"}
        check(directory, env, "training without P2P", GDR + ["--train-step", "--nranks", "2", "--iters", "1"],
              "failed to initialize RTC SGEMM or NCCL/RCCL P2P AllToAll symbols", error=True)
        for name, options in [("A/B", GDR + AB), ("demo", GDR)]:
            check(directory, delayed, f"correct {name}".replace("/", "-"), options, "NCCL_PROTO=Simple", timed_copy=True)
            check(directory, env | {"TEST_STALE": "1"}, f"stale {name}".replace("/", "-"),
                  options, "phase 0 iteration 2 idx 0", error=True)
        check(directory, env | {"TEST_CORRUPT": "1"}, "corrupt rank", GDR + AB, "phase 0 iteration 1 idx 0", error=True)
        check(directory, env | {"TEST_SIMPLE": "1"}, "native Simple", NATIVE, "NCCL_PROTO=Simple")

        multi = env | {"TEST_SIMPLE": "1", "TEST_HCAS": "1"}
        check(directory, multi, "vxfec all rails", GDR + AB, "TX 0x0003, RX 0x0003", full_node=True)
        missing = "TX 0x0002, RX 0x0002, expected 0x0003"
        check(directory, multi | {"TEST_MISSING_RAIL": "1"}, "vxfec missing rail", GDR + AB, missing, error=True, full_node=True)
        check(directory, multi | {"TEST_WRONG_ORDER": "1"}, "vxfec unstable HCA ID", GDR + AB,
              "rank 1: vxfec device IDs do not match", error=True, full_node=True)
        native = multi | {"TEST_NATIVE_COUNTERS": "1", "LD_PRELOAD": str(standin)}
        check(directory, native, "native port activity", NATIVE, "timed sweep host-wide IB port activity", full_node=True)
        check(directory, native | {"TEST_MISSING_RAIL": "1"}, "native missing port", NATIVE, missing, error=True, full_node=True)

        focused = env | {"TEST_FOUR_CASE": "1", "TEST_SIMPLE": "1", "NCCL_MIN_CTAS": "4", "NCCL_MAX_CTAS": "16"}
        plugin_phases = ["plugin_clean", "plugin_sack_loss", "plugin_auto_loss"]
        check(directory, focused, "B200 built-in IB clean", [*B200, "--native-ib", "--gdr", "--seed", "101"],
              B200_MARKER, phases=["native_clean"])
        check(directory, focused, "B200 paired plugin odd seed", [*B200, *GDR, "--seed", "101"], B200_MARKER, phases=plugin_phases)
        check(directory, focused, "B200 paired plugin even seed", [*B200, *GDR, "--seed", "102"],
              B200_MARKER, phases=[plugin_phases[0], *reversed(plugin_phases[1:])])
        for fault in ("TEST_CPU_ONLY", "TEST_VULKAN_ONLY", "TEST_MIXED_FALLBACK", "TEST_WARM_CUDA_ONLY"):
            check(directory, focused | {fault: "1"}, f"B200 rejects {fault}", [*B200, *GDR, "--seed", "101"],
                  "requires timed native CUDA RX repair without fallback", error=True, phases=[])
        for fault in ("TEST_BAD_TRACE", "TEST_NO_DROPS", "TEST_DIRTY_CLEAN"):
            check(directory, focused | {fault: "1"}, f"B200 rejects {fault}", [*B200, *GDR, "--seed", "101"],
                  "missing or mismatched systematic impairment", error=True, phases=[])
        check(directory, focused | {"TEST_NO_PAIRED": "1"}, "B200 requires an enabled paired fault hook",
              [*B200, *GDR, "--seed", "101"], "lacks enabled paired fault hooks", error=True, phases=[])
        check(directory, focused | {"TEST_NO_SACK": "1"}, "B200 requires observed SACK", [*B200, *GDR, "--seed", "101"],
              "requires timed SACK retransmissions", error=True, phases=[])
        check(directory, focused, "B200 requires seed", [*B200, *GDR], "--b200-four-case requires two ranks", error=True, phases=[])
        check(directory, focused, "B200 requires two ranks", [*B200, *GDR, "--seed", "101", "--nranks", "3"],
              "--b200-four-case requires two ranks", error=True, phases=[])
        check(directory, focused, "B200 ranks must agree on the seed", [*B200, *GDR, "--seed", "101"],
              "ranks disagree on GPU profile", error=True, phases=[], remote_pair=True, peer_args=("--seed", "103"))

        check(directory, focused, "ranks must agree on GPU profile", [*B200, *GDR, "--seed", "101"],
              "ranks disagree on GPU profile", error=True, phases=[], remote_pair=True, peer_profile="--h200-four-case")

        for model in ("b200", "h200"):
            for seed in ("101", "102", "103"):
                lossy_phases = plugin_phases[1:] if int(seed) % 2 else plugin_phases[:0:-1]
                for mode, flags, phases in (
                    ("native", ["--native-ib", "--gdr"], ["native_clean"]),
                    ("plugin", GDR, [plugin_phases[0], *lossy_phases]),
                ):
                    stem = f"{model}_{seed}_{mode}"
                    log = check(directory, focused, stem, [f"--{model}-four-case", "--iters", "100", "--seed", seed, *flags],
                                f"Verified {model.upper()} four-case timed profile", phases=phases, remote_pair=True)
                    (directory / f"{stem}.txt").write_text(log)
            subprocess.run([sys.executable, CHECK_RUN, f"{model}_pairs", directory], check=True)
            with (directory / f"{model}_pairs.csv").open() as fp:
                assert [row["seed"] for row in csv.DictReader(fp)] == ["101", "102", "103"]
            print(f"{model.upper()} demo output passed paired benchmark validation")

        mixed = subprocess.run(
            [sys.executable, CHECK_RUN, "h200", "plugin", "101", directory / "h200_101_plugin.csv",
             directory / "b200_101_plugin.csv.samples.csv", directory / "h200_101_plugin.txt"],
            capture_output=True, text=True,
        )
        assert mixed.returncode and "invalid or out-of-sequence H200 timing sample" in mixed.stderr, mixed


if __name__ == "__main__":
    main()
