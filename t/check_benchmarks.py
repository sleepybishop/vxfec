"""Validate local hardware benchmark outputs and GPU/NIC topology."""

import csv
import math
import re
import sys
from functools import partial
from pathlib import Path


def bdf(value):
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])", value)
    if not match:
        raise ValueError(f"invalid GPU PCI address {value!r}")
    return f"{int(match[1], 16):04x}:{match[2].lower()}"


def topology(count, pair, inventory):
    count = int(count)
    dev, vk, depth, gpu, devs, vks, gpus, info = pair.split("|", 7)
    rails = devs.split(",")
    vk_ids = vks.split(",")
    pci_ids = [bdf(value) for value in gpus.split(",")]
    visible = [bdf(value.strip()) for value in inventory.splitlines()]
    if (int(depth) <= 0 or not info.startswith("PAIR_OK ") or " is_roce=0 " not in info
            or len({rail.partition(":")[0] for rail in rails}) != count
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+:[1-9][0-9]*", rail) for rail in rails)
            or any(not re.fullmatch(r"0|[1-9][0-9]*", index) for index in vk_ids)
            or any(len(values) != count or len(set(values)) != count for values in (rails, vk_ids, pci_ids, visible))
            or set(pci_ids) != set(visible) or (dev, vk, bdf(gpu)) not in zip(rails, vk_ids, pci_ids)):
        raise ValueError(f"expected {count} distinct IB rail/Vulkan/CUDA mappings for the visible GPUs: {pair}")


def check_cuda_pair(rail, gpu, info, inventory, expected_model):
    rows = [row for row in csv.reader(inventory.splitlines(), skipinitialspace=True) if len(row) == 2]
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+:[1-9][0-9]*", rail)
            or not info.startswith("PAIR_OK ") or "is_roce=0" not in info.split()
            or not re.search(r"\bpci_depth=[1-9][0-9]*\b", info)
            or not any(bdf(address) == bdf(gpu) and re.search(rf"\b{expected_model}\b", model) for address, model in rows)):
        raise ValueError(f"expected a registered {expected_model} CUDA/IB pair for {rail} and {gpu}")


def b200_pair(pair, inventory):
    rail, gpu, info = pair.split("|")
    check_cuda_pair(rail, gpu, info, inventory, "B200")


def h200_pair(pair, inventory):
    rail, gpu, vk, info = pair.split("|")
    check_cuda_pair(rail, gpu, info, inventory, "H200")
    speed = re.search(r"(?:^| )speed=([0-9]+)(?: |$)", info)
    if (not re.fullmatch(r"0|[1-9][0-9]*", vk) or not speed or int(speed[1]) < 400000
            or not {"cuda_dmabuf=1", "vk_dmabuf=1"}.issubset(info.split())):
        raise ValueError(f"expected CUDA and Vulkan registrations on one 400G H200/IB pair: {pair}")


def expected_phases(mode):
    if mode == "demo":
        names = ("Lossless Fabric (Baseline GBN)", "Lossless Fabric (vxfec)",
                 "0.4% OOO + Drops (Baseline GBN)", "0.4% OOO + Drops (vxfec)",
                 "3.0% Burst=6 (Baseline GBN)", "3.0% Burst=6 (SACK)", "3.0% Burst=6 (vxfec MDS)",
                 "3.0% Burst=6 64MB (Baseline GBN)", "3.0% Burst=6 64MB (SACK)", "3.0% Burst=6 64MB (vxfec)")
        codecs = ("gbn", "auto", "gbn", "auto", "gbn", "sack", "auto", "gbn", "sack", "auto")
        return {(f"{i}. {name}", codec, 1 if i <= 7 else 64)
                for i, (name, codec) in enumerate(zip(names, codecs), 1)}
    native = mode == "native"
    sizes = (("512 KiB", 0.5), ("8.0 MiB", 8), ("64 MiB", 64), ("256 MiB", 256))
    tag = "NCCL_IB RC" if native else "vxfec UC"
    phases = {(f"{label} | 0.0% Lossless ({tag})", "rc" if native else "auto", size) for label, size in sizes}
    if not native:
        for label, size in sizes:
            for codec, tag in (("sack", "SACK"), ("auto", "vxfec")):
                phases.add((f"{label} | 2.0% Burst=6 ({tag})", codec, size))
                if size == 0.5:
                    phases.add((f"{label} | 0.4% OOO Loss ({tag})", codec, size))
    return phases


def results(mode, ranks, path):
    expected = expected_phases(mode)
    csv_path = Path(path)
    log_path = csv_path.with_name(csv_path.name.removesuffix(".part")).with_suffix(".txt")
    marker = f"Verified 100% bit-exact GPU FP32 `ncclAllReduce` across all {len(expected)} collective phases."
    if marker not in log_path.read_text():
        raise ValueError("missing rank-zero verification summary")
    seen = set()
    with csv_path.open(newline="") as result:
        for row in csv.DictReader(result, strict=True):
            if None in row or None in row.values() or row["nranks"] != ranks:
                raise ValueError("incomplete row or unexpected rank count")
            metrics = ["avg_ms", "bus_gbps", "alg_gbps"]
            if mode != "demo":
                metrics += ["p50_ms", "p99_ms"]
                transport = "nccl-ib-rc" if mode == "native" else "vxfec-ibv-uc"
                if row["transport"] != transport or row["iters"] != "100":
                    raise ValueError("unexpected transport or iteration count")
            if any(not math.isfinite(float(row[k])) or float(row[k]) <= 0 for k in metrics):
                raise ValueError("missing or invalid performance measurement")
            phase = (" ".join(row["phase"].split()), row["codec"], float(row["tensor_mb"]))
            if phase not in expected or phase in seen:
                raise ValueError(f"duplicate or unexpected phase {phase}")
            seen.add(phase)
    if seen != expected:
        raise ValueError(f"missing {len(expected - seen)} requested phase(s)")


def four_case(model, mode, seed, path, samples, log):
    """Validate one native or plugin process pair, including unsorted timed samples."""
    if mode not in {"native", "plugin"}:
        raise ValueError(f"unknown {model.upper()} transport {mode}")
    lossy = ["plugin_sack_loss", "plugin_auto_loss"]
    if int(seed) % 2 == 0:
        lossy.reverse()
    phases = ["native_clean"] if mode == "native" else ["plugin_clean", *lossy]
    text = Path(log).read_text()
    marker = f"Verified 100% bit-exact GPU FP32 `ncclAllReduce` across all {len(phases)} collective phases."
    if marker not in text or f"Verified {model.upper()} four-case timed profile" not in text:
        raise ValueError("missing verification of the focused timed AllReduce")
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream, strict=True))
    if [row["phase"] for row in rows] != phases:
        raise ValueError("unexpected focused phase set or order")

    profile = f"{model}-four-case"
    times = {phase: [] for phase in phases}
    with Path(samples).open(newline="") as stream:
        for row in csv.DictReader(stream, strict=True):
            phase = row["phase"]
            ns = int(row["worst_rank_ns"])
            if (None in row or None in row.values() or phase not in times or row["profile"] != profile
                    or row["seed"] != seed or row["impairment_scope"] != ("first-systematic" if phase in lossy else "none")
                    or int(row["iteration"]) != len(times[phase]) + 1 or ns <= 0):
                raise ValueError(f"invalid or out-of-sequence {model.upper()} timing sample")
            times[phase].append(ns)

    traces = set()
    for row in rows:
        phase = row["phase"]
        impaired = phase in lossy
        expected = {"profile": profile, "seed": seed, "nranks": "2", "iters": "100",
                    "transport": "nccl-ib-rc" if mode == "native" else "vxfec-ibv-uc",
                    "codec": "rc" if mode == "native" else ("sack" if phase == "plugin_sack_loss" else "auto"),
                    "impairment_scope": "first-systematic" if impaired else "none",
                    "loss_ppm": "20000" if impaired else "0", "burst": "6" if impaired else "1"}
        if (None in row or None in row.values() or any(row[key] != value for key, value in expected.items())
                or float(row["tensor_mb"]) != 0.5 or len(times[phase]) != 100):
            raise ValueError(f"incorrect experiment parameters or sample count for {phase}")
        ns = sorted(times[phase])
        for key, value in (("p50_ms", ns[50]), ("p99_ms", ns[98]), ("avg_ms", sum(ns) / len(ns))):
            reported = float(row[key])
            if not math.isfinite(reported) or abs(reported - value / 1e6) > 0.000501:
                raise ValueError(f"{phase} {key} disagrees with raw worst-rank times")
        for key in ("alg_gbps", "bus_gbps"):
            if not math.isfinite(float(row[key])) or float(row[key]) <= 0:
                raise ValueError(f"invalid {key}")
        keys = ("drops", "systematic_trials", "drop_fingerprint", "cuda_rx_repairs", "fallback_rx_repairs",
                "ibv_pkts", "sack", "harq_ir", "rs8_dec", "rq_dec")
        counters = {key: int(row[key]) for key in keys}
        if any(n < 0 for n in counters.values()) or (mode == "plugin" and not counters["ibv_pkts"]):
            raise ValueError("invalid timed counters or missing IB payload")
        if phase == "plugin_auto_loss" and (not counters["cuda_rx_repairs"] or counters["fallback_rx_repairs"]):
            raise ValueError("timed FEC repairs did not exclusively complete on native CUDA")
        if phase == "plugin_sack_loss" and (not counters["sack"] or counters["cuda_rx_repairs"] or counters["fallback_rx_repairs"]):
            raise ValueError("missing retransmission-only SACK evidence")
        if impaired:
            trace = tuple(counters[k] for k in ("drops", "systematic_trials", "drop_fingerprint"))
            if not all(trace) or trace[0] > trace[1]:
                raise ValueError("missing timed loss evidence")
            traces.add(trace)
        elif any(counters[k] for k in ("drops", "systematic_trials", "drop_fingerprint")):
            raise ValueError("loss in a clean phase")
    if mode == "plugin" and len(traces) != 1:
        raise ValueError("SACK and auto saw different systematic packet loss traces")
    return {phase: sorted(values)[98] for phase, values in times.items()}


def four_case_pairs(model, directory):
    """Produce one interpretable comparison per seed using the unrounded samples."""
    root = Path(directory)
    output = []
    for seed in ("101", "102", "103"):
        times = {}
        for mode in ("native", "plugin"):
            path = root / f"{model}_{seed}_{mode}.csv"
            times.update(four_case(model, mode, seed, path, f"{path}.samples.csv", path.with_suffix(".txt")))
        clean = 100 * (times["plugin_clean"] / times["native_clean"] - 1)
        gain = 100 * (1 - times["plugin_auto_loss"] / times["plugin_sack_loss"])
        output.append([seed, times["native_clean"], times["plugin_clean"], times["plugin_sack_loss"],
                       times["plugin_auto_loss"], f"{clean:.3f}", f"{gain:.3f}"])
    path = root / f"{model}_pairs.csv"
    with path.with_suffix(".part").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "native_clean_p99_ns", "plugin_clean_p99_ns", "sack_loss_p99_ns",
                         "auto_loss_p99_ns", "clean_penalty_pct", "loss_p99_reduction_pct"])
        writer.writerows(output)
    path.with_suffix(".part").replace(path)


def wire(codec, blackout, pipeline, z, sym_t, qps, path, previous=None):
    """Each successful TX/RX pair appends one row to the already verified checkpoint."""
    with Path(path).open(newline="") as result:
        rows = list(csv.DictReader(result, strict=True))
    prior = []
    if previous and Path(previous).exists():
        with Path(previous).open(newline="") as checkpoint:
            prior = list(csv.DictReader(checkpoint, strict=True))
    if not rows or rows[:-1] != prior:
        raise ValueError("wire checkpoint did not append exactly one row")
    row = rows[-1]
    use_pipeline = bool(pipeline) and str(pipeline).strip().lower() not in ("0", "false", "no")
    requested = {"transport": "ibv-gdr-pipeline" if use_pipeline else "ibv-gdr", "codec": codec, "Z": z, "K": "200",
                 "P": "48", "T": sym_t, "rails": qps, "wire_only": "0", "gdr_dmabuf": "1"}
    if (None in row or None in row.values() or any(row[k] != v for k, v in requested.items())
            or float(row["blackout_pct"]) != float(blackout)):
        raise ValueError("wire result does not match the requested GPU DMA-BUF scenario")
    metrics = [float(row[k]) for k in ("wire_gbps", "e2e_gbps", "gpu_gbps")]
    if any(not math.isfinite(v) or v <= 0 for v in metrics) or not 0 < int(row["rx_frames"]) <= int(row["planned_frames"]):
        raise ValueError("wire result has invalid throughput or frame counts")
    return row


def exchange(rails, path):
    """Check the complete N=100 GPU-buffer exchange copied from stage eight."""
    tags = {"gbn": "GBN", "sack": "SACK", "auto": "vxfec", "rs8": "MDS", "rq-gf256": "RQ-GF256", "rq-gf2": "RQ-GF2"}
    expected = set()
    for name, size, lossless, burst in (
            ("512 KiB", 512 << 10, ("auto", "gbn"), ("sack", "gbn", "rq-gf256", "rq-gf2", "rs8")),
            ("8.0 MiB", 8 << 20, ("auto", "gbn"), ("sack", "gbn", "rq-gf256", "rq-gf2", "auto")),
            ("64 MiB", 64 << 20, ("auto",), ("sack", "auto")),
            ("256 MiB", 256 << 20, ("auto",), ("sack", "auto"))):
        scenarios = {"0.0% Lossless": lossless, "2.0% Burst=6": burst}
        if size == 512 << 10:
            scenarios["0.4% OOO Loss"] = ("gbn", "sack", "auto")
        for condition, codecs in scenarios.items():
            for codec in codecs:
                expected.add((f"GDR exchange | verified IBV | {name} | {condition} ({tags[codec]})", codec, size))

    with Path(path).open(newline="") as result:
        for row in csv.DictReader(result, strict=True):
            if (None in row or None in row.values() or row["ndev"] != rails or row["iters"] != "100"
                    or int(row["ibv_pkts"]) <= 0 or row["alg_gbps"] or row["bus_gbps"]):
                raise ValueError("missing measured IB exchange or unexpected rail/iteration count")
            phase = (" ".join(row["scenario"].split()), row["codec"], int(row["bytes"]))
            if phase not in expected:
                raise ValueError(f"duplicate or unexpected exchange phase {phase}")
            timings = [float(row[key]) for key in ("p50_ms", "p99_ms", "max_ms")]
            measurements = timings + [float(row["goodput_gbps"])]
            if any(not math.isfinite(v) or v <= 0 for v in measurements) or timings != sorted(timings):
                raise ValueError(f"invalid exchange timings or goodput for {phase}")
            expected.remove(phase)
    if expected:
        raise ValueError(f"missing {len(expected)} requested exchange phase(s)")


if __name__ == "__main__":
    try:
        checks = {"topology": topology, "results": results, "wire": wire, "exchange": exchange,
                  "b200_pair": b200_pair, "h200_pair": h200_pair}
        for model in ("b200", "h200"):
            checks[model] = partial(four_case, model)
            checks[f"{model}_pairs"] = partial(four_case_pairs, model)
        checks[sys.argv[1]](*sys.argv[2:])
    except (KeyError, ValueError, TypeError, OSError, csv.Error) as exc:
        sys.exit(f"Benchmark {sys.argv[1]} check failed: {exc}")
