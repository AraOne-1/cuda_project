# GPU-Accelerated Exact 1-Nearest-Neighbor Search (UCI Forest Covertype) - v2

**Course:** Parallel Computing with CUDA — individual term project (20%)  
**Problem:** Exact 1-nearest-neighbor (1-NN) classification over a large public geospatial-feature dataset.  
**Dataset:** UCI Forest Covertype, **581,012** rows, **54** features, **7** target classes.  
**Objective:** Compare naive and early-abandonment **sequential CPU** versus **parallel CUDA GPU** for the same exhaustive exact nearest-neighbor problem; measure latency, agreement, accuracy, and speedup.

> **Important:** This package contains executable source, a documented real-dataset loader, a deterministic test fixture, a benchmarking script, and a presentation. It does **not** claim any measured CUDA speedup without execution on an NVIDIA GPU. You must run the benchmark and replace the presentation's intentionally blank results slide.

## 1. Prerequisites

- Python 3.9+; `pip install -r requirements.txt` (NumPy and Matplotlib)
- C++17 compiler: GCC/G++ or equivalent (for CPU-only testing)
- NVIDIA CUDA-capable GPU, NVIDIA driver, and CUDA Toolkit providing `nvcc` (required for GPU measurements). Check with `nvidia-smi` and `nvcc --version`.
- RAM: at least ~1 GB recommended. Device memory: ~0.5 GB or more of free VRAM for the reference data and partial arrays (depends on query count).
- **Intel-only integrated graphics cannot run CUDA.** The CPU version is usable without NVIDIA. If your computer has no NVIDIA GPU, see `docs/COLAB_GPU_TH.md` for a Google Colab GPU workflow (subject to Colab availability).

## 2. Get the REAL publicly available dataset

```bash
python -m pip install -r requirements.txt
python scripts/prepare_data.py
```

This downloads the original `covtype.data.gz` from UCI (with an alternate Figshare mirror), checks its SHA-256 against the known archive digest, and creates `data/covertype.bin` (float32 features and labels) and `data/covertype.json` (provenance and scaling statistics). If automatic download fails, get `covtype.data.gz` manually from https://archive.ics.uci.edu/dataset/31/covertype and save it to `data/`, then rerun the command.

**Split is deterministic** (`seed=2026`): **572,820 reference records** and **8,192 held-out query records**. Scaling parameters (mean/std for first 10 continuous features) are computed from **reference data only** and applied to queries; 44 binary features stay 0/1. The label is not included as a distance feature.

## 3. Build

### Linux / WSL / Make (with NVIDIA CUDA accessible)

```bash
make cpu
make gpu
# OR: make all
```

### Manual Linux commands

```bash
mkdir -p build
# CPU executable (runs even without CUDA)
g++ -O3 -std=c++17 -ffp-contract=off src/cpu_only.cpp -o build/knn_cpu
# CUDA executable (CPU and GPU modes, requires CUDA Toolkit)
nvcc -O3 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off src/main.cu -o build/knn_cuda
```

### Windows PowerShell with CUDA Toolkit + MSVC

```powershell
New-Item -ItemType Directory -Force build
nvcc -O3 -std=c++17 --fmad=false src/main.cu -o build/knn_cuda.exe
# Optional CPU-only program if g++ is installed:
g++ -O3 -std=c++17 src/cpu_only.cpp -o build/knn_cpu.exe
```

Run the Windows `nvcc` command in a Developer PowerShell or command prompt where the supported Visual Studio C++ compiler is configured. On native Windows without `make`, build manually.

## 4. Verify correctness FIRST

```bash
python scripts/make_tiny_fixture.py
./build/knn_cpu --data data/tiny.bin --queries 3 --output results/tiny_predictions.csv
python scripts/test_correctness.py ./build/knn_cpu
./build/knn_cpu --data data/tiny.bin --queries 3 --compare-naive
# Then optionally test CPU against CUDA using the same tiny dataset:
./build/knn_cuda --data data/tiny.bin --queries 3 --gpu-baseline
```

The handmade fixture expects nearest reference IDs **[0, 3, 4]**. On a CUDA system, the GPU automatically checks its answers against CPU reference predictions. If any nearest-reference ID differs, the executable exits with code 2; ties are defined by the smallest reference row ID. Both CUDA variants must agree. The CPU oracle test compares to a separately implemented NumPy brute-force solution.

## 5. Run a complete benchmark

Start with a small real test:

```bash
./build/knn_cuda --data data/covertype.bin --queries 32 --gpu-baseline --repeats 3
```

Then run measured scalability experiments:

```bash
python scripts/benchmark.py --queries 32 128 512 2048 --gpu-baseline
# On Windows specify: --exe build/knn_cuda.exe
```

Outputs (generated **from real measurements**, no estimated values):

- `results/benchmark.csv`: CPU latency, GPU kernel latency, single-pass GPU end-to-end latency, two speedups, accuracy, correctness mismatches
- `results/runtime.png`: CPU vs GPU timing curve (log axes)
- `results/speedup.png`: speedup as query count increases

To run a large standalone benchmark:

```bash
./build/knn_cuda --queries 4096 --mode both --repeats 3 --metrics results/large.csv --output results/predictions.csv
```

Options: `--data`, `--refs`, `--queries`, `--mode cpu|gpu|both`, `--repeats`, `--gpu-baseline`, `--metrics`, `--output`.

### Timing and fair comparison

- **CPU time:** one sequential `cpu_search` pass, including exact pruning. For the smallest query size, `scripts/benchmark.py` additionally enables `--cpu-naive` (naive full-distance serial scan). It prints and logs a separate naive serial baseline; the main CPU vs GPU speedup is always measured against `cpu_search` on the exact same input. On some small or random workloads the naive loop is faster due to branch overhead; neither version is universally optimal. Excludes file loading and preprocessing.
- **GPU kernel time:** median of `--repeats` CUDA-event measurements after a warm-up, comprising both search and final reduction kernels. Excludes transfers and layout conversion.
- **GPU end-to-end:** one GPU search including CPU AoS→SoA transpose, allocation, H2D, kernel execution, and D2H. Excludes input-file loading and device deallocation; on first use, GPU runtime initialization may contribute. The implementation reports BOTH speedups to avoid hiding transfer overhead.
- To mitigate variability, repeat `benchmark.py` and report medians across runs, together with hardware/compiler versions, NVIDIA GPU name, and run flags.
- `--gpu-baseline` tests a naive AoS GPU kernel against the optimized SoA kernel; CPU and GPU perform the same exhaustive exact 1-NN search, not approximate kNN or a synthetic substitute.

**Do not infer acceleration from asymptotics.** Whether the GPU is faster is an empirical question depending on the machine and workload, and it is a course submission requirement to **demonstrate** a speedup on large data.

## 6. Algorithms & why the sequential solution is strong

Distance for a query `q` and a reference `x` is the **squared Euclidean distance**:

\[d^2(q,x)=\sum_{f=1}^{54}(q_f-x_f)^2.\]

No `sqrt` is required because square root is monotonic. Two CPU algorithms are provided: a naive serial exhaustive full-distance scan and an early-abandonment variant. The latter performs a contiguous row-major (Array-of-Structures, AoS) scan, maintains the best-so-far squared distance, and **abandons** any candidate as soon as its partial nonnegative squared sum exceeds the current best. This is exact: omitted nonnegative terms cannot make that candidate better. Equal distances deterministically prefer the smallest row ID. No multicore threads or OpenMP are used. To compare both serial variants independently, run `./build/knn_cpu --data data/covertype.bin --refs 10000 --queries 32 --compare-naive`. Both must return the same reference IDs. The naive version can outperform pruning when the branch cost outweighs skipped arithmetic.

Time: worst-case **O(Q × N × D)**, memory: **O(ND + QD)** for data plus **O(Q)** results. This is a strong **exhaustive exact** sequential baseline but **is not claimed to be the fastest conceivable nearest-neighbor method**; spatial indexing can help some distributions, and approximate search changes the problem.

CUDA uses a grid with Y-axis queries and X-axis reference tiles:

1. Each block checks a subset of reference rows and produces a local `(min_distance², reference_id)` pair.
2. A second CUDA kernel reduces these block winners to a global winner per query.
3. The baseline GPU stores **AoS** references. The optimized version transposes into **SoA (Structure-of-Arrays)** for coalesced warp loads, caches query features in shared memory, assigns **4 candidates per thread** to reduce block count, and uses **warp shuffle reductions** with deterministic tie-breaking.

Both GPU designs compute the same distance and produce the same result (subject to standard floating-point behavior). No `atomicMin` or inter-block synchronization is needed.

## 7. Complexity & WS analysis

Let `N`=reference records, `Q`=query records, `D`=features, `B`=threads per CUDA block (256), and `R`=references per thread (4 optimized).

| Approach | Total work | Critical path / span (idealized) | Notes |
|---|---:|---:|---|
| CPU naive / early-abandon scan | O(QND) worst case | O(QND) | Early abandon can reduce actual work, but adds branches |
| Naive CUDA | O(QND) | O(D + log B + ceil(N/B)/B + log B) per query | Two-stage block minima; resource limits apply |
| Optimized CUDA | O(QND) | O(RD + log B + ceil(N/(BR))/B + log B) per query | Coalescing / shared query reduce memory traffic |

These are idealized per-query spans assuming enough blocks can execute simultaneously. GPU occupancy, queueing, memory bandwidth, and PCIe transfer prevent treating theoretical span as actual wall time. Both GPU variants perform **work-optimal** exhaustive arithmetic up to constant-factor overhead relative to O(QND) exhaustive scan.

## 8. Course optimization concepts and APOD workflow

The CUDA course APOD stages map to this project as: **Assess** expensive exhaustive comparisons; **Parallelize** query-reference work across grid axes; **Optimize** SoA coalescing, query shared-memory reuse, multiple references/thread and warp reduction; **Deploy** validated executable, reproducible benchmarks and documentation.

To populate the presentation results only after a real run:

```bash
python scripts/fill_results_slide.py --csv results/benchmark.csv --pptx docs/CUDA_Covertype_Presentation.pptx
```

Creates `docs/CUDA_Covertype_Presentation_MEASURED.pptx`. It refuses incomplete or unmeasured rows. Do not hand in the placeholder slide as measured evidence.

## 9. Reproducibility / hardware disclosure

Copy the following to `results/hardware.txt` after executing on your actual NVIDIA machine:

```text
GPU model:
GPU driver version:
CUDA Toolkit version (nvcc --version):
CPU model:
RAM:
Operating system:
Compiler flags:
Dataset checksum:
Reference rows / query counts:
Number of repeats / trial medians:
```

The PDF project brief asks for **1) a publicly accessible problem/dataset distinct from labs, 2) a well-defended sequential solution, 3) an argued parallel design, 4) CPU implementation, 5) CPU-GPU implementation, 6) proven large-data speedup**. Required GitHub items are source folder, how-to-run document, and presentation. Deadline: **Wed Nov 4, 2026, 11:59 PM**; individual presentation: **Thu Nov 5, 2026, 1:00–3:00 PM**, ~15 minutes including Q&A.

## Attribution & data licensing

Blackard, J. (1998). *Covertype* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N. Dataset distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Source archive: https://archive.ics.uci.edu/dataset/31/covertype. See `docs/PROJECT_REPORT.md` for analysis and limitations.
