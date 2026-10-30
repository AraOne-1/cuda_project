#!/usr/bin/env python3
"""Run measured CPU/GPU trials and create genuine plots (no estimated results)."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", default="build/knn_cuda.exe" if sys.platform == "win32" else "build/knn_cuda")
    p.add_argument("--data", default="data/covertype.bin")
    p.add_argument("--queries", type=int, nargs="+", default=[32, 128, 512, 2048])
    p.add_argument("--refs", type=int, default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gpu-baseline", action="store_true")
    p.add_argument("--cpu-naive-max-queries", type=int, default=32,
                   help="Run naive CPU only when query count <= this limit; 0 disables it")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    metric = output / "benchmark.csv"
    if metric.is_file():
        metric.unlink()
    for nq in args.queries:
        cmd = [args.exe, "--data", args.data, "--mode", "both", "--queries", str(nq),
               "--repeats", str(args.repeats), "--metrics", str(metric)]
        if args.refs is not None:
            cmd += ["--refs", str(args.refs)]
        if args.gpu_baseline:
            cmd += ["--gpu-baseline"]
        if 0 < nq <= args.cpu_naive_max_queries:
            cmd += ["--cpu-naive"]
        print("\nRUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    with metric.open(newline="") as f:
        rows = list(csv.DictReader(f))
    optimized = sorted((r for r in rows if r["version"] == "gpu_optimized"),
                       key=lambda r: int(r["query_rows"]))
    if not optimized:
        raise RuntimeError("No optimized measurements produced")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; CSV is complete; run pip install -r requirements.txt")
        return
    x = [int(r["query_rows"]) for r in optimized]
    plt.figure(figsize=(9, 5))
    for col, name in [("cpu_ms", "CPU sequential"),
                      ("gpu_kernel_ms", "GPU kernels only"),
                      ("gpu_e2e_ms", "GPU single-pass E2E")]:
        plt.plot(x, [float(r[col]) for r in optimized], "o-", label=name)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Number of query records (reference set fixed)")
    plt.ylabel("Elapsed time (milliseconds; log scale)")
    plt.title("Exact 1-NN on UCI Covertype: measured runtimes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "runtime.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for col, name in [("kernel_speedup", "Search-only speedup"),
                      ("e2e_speedup", "Single-pass E2E speedup")]:
        plt.plot(x, [float(r[col]) for r in optimized], "o-", label=name)
    plt.axhline(1.0, linestyle="--", linewidth=1, label="CPU/GPU parity")
    plt.xscale("log", base=2)
    plt.xlabel("Number of query records (reference set fixed)")
    plt.ylabel("Speedup (CPU / GPU)")
    plt.title("Measured acceleration vs. query count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "speedup.png", dpi=180)
    plt.close()
    naive = sorted((r for r in rows if r["version"] == "cpu_naive"),
                   key=lambda r: int(r["query_rows"]))
    if naive:
        print("Naive CPU measured at query count(s):",
              ", ".join(str(r["query_rows"]) for r in naive))
    print("Saved:", metric, output / "runtime.png", output / "speedup.png")


if __name__ == "__main__":
    main()
