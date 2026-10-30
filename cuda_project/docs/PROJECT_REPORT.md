# Technical Report — CUDA Accelerated Exact 1-NN on Covertype

## Executive summary
This individual project investigates a real nearest-neighbor classification problem on 581,012 publicly available UCI Covertype records (54 features). It implements (i) naive and early-abandonment exact serial C++ references, (ii) an AoS CUDA baseline, and (iii) an improved CUDA implementation using SoA coalescing, shared-memory query caching, multiple candidates per thread, and a deterministic two-stage block reduction. The accompanying scripts prepare data, validate result indices, measure timings, and draw plots.

**Measurement status:** CUDA runtime execution was not possible in the authoring environment because no `nvcc` or NVIDIA GPU was available. The project does not include fabricated timings. A real NVIDIA GPU test and slide update are mandatory before submission.

## 1. Motivation and problem definition
For each query landscape cell, predict its forest-cover category as the category of the **single nearest** reference cell under standardized Euclidean feature distance. This is 1-NN classification, a canonical nontrivial exact-search workload that can be mapped to two levels of parallel reductions. It differs from a standalone histogram, scan, vector addition, reduction, or tiled matrix multiplication lab because it is a complete nearest-neighbor classifier with a genuine labeled public dataset and row-level classification evaluation. The GPU implementation nevertheless uses reduction as a subroutine.

The algorithm's answer is `(argmin_j sum_f (Q[i,f] - X[j,f])², label[argmin])` for each query `i`.

## 2. Data and partitioning
- Source: UCI Covertype, 581,012 instances, 54 features, 7 labels; 10 continuous cartographic features and 44 binary wilderness/soil features.
- Dataset credit: Blackard (1998), UCI ML Repository, DOI 10.24432/C50K5N, license CC BY 4.0.
- Default partition: deterministic permutation with seed 2026; 572,820 reference records and 8,192 query records. The subset of queries increases from 32 up to 2,048 or 8,192 for scaling experiments without changing references.
- Scaling: compute reference-only z-score for first 10 continuous values; keep 44 binary fields unchanged. Do not pass target class into distance calculation.
- Input: `data/covertype.bin` generated from public compressed CSV, not hardcoded fabricated observations. Full data not committed due to binary size.

## 3. Sequential design and justification
Use an AoS contiguous vector and one pass over reference rows per query. The running best distance allows exact candidate pruning: when partial distance > best, the remaining squared terms, all nonnegative, cannot reduce it. Compute squared distances only (avoid sqrt). Scan references in increasing row ID, allowing stable tie-breaking. No CPU threads are created; this is a strong serial implementation of exhaustive exact 1-NN. This **does not** imply that indexed nearest-neighbor approaches cannot be faster on some data. Worst-case work is O(QND), constant extra state per query.

## 4. Parallel algorithm
**Baseline CUDA (AoS):** map each query to grid Y, map 256 reference points per block to grid X; each thread computes squared distance; block computes `(distance, index)` min-pair with `__shfl_down_sync` warp operations and shared leaders. The second kernel reduces block winners for the query.

**Optimized CUDA (SoA):** transpose reference memory once so `ref[f*N + r]` makes accesses by 32 adjacent threads contiguous. Load 54 query coordinates cooperatively into block shared memory. Each thread handles four reference points (four local candidates in registers). Reduce to local block minimum and write one pair; reduce partial minima per query in a second kernel. All tie comparisons consider `(distance, index)` lexicographically.

This design eliminates the need for atomics, keeps input immutable, and reduces partial output from `ceil(N/256)` blocks/query to `ceil(N/1024)` blocks/query.

## 5. Work–Span and overheads
Brute-force all-pair distance arithmetic needs O(QND) work. For one candidate, squared-distance calculation spans O(D) in the feature loop; block-minimum reduction spans O(log B), and second-stage reduction processes O(N/(BRB)) pairs per thread followed by O(log B) span. At finite GPU resources, device-level throughput and block scheduling change the realized duration. Feature-wise SoA global reads are naturally coalesced across a warp, while the layout transposition and host-to-device copy are one-time overheads that the benchmark exposes in a separate end-to-end metric.

## 6. Correctness
- The CPU and GPU solve identical exact argmin with no approximation.
- Since `sqrt` is strictly increasing over nonnegative values, minimizing distance² returns the same nearest neighbor as Euclidean distance.
- A pruned CPU candidate cannot beat the incumbent once its partial sum exceeds incumbent.
- Group-wise minimum then a minimum across groups equals a global minimum. Tie break by smaller reference row index makes the intended answer independent of reduction order.
- Binary floats can differ by tiny amounts across instructions; the build disables fused multiply-add to reduce discrepancies. The program checks exact winning indices; if different, investigate and report floating-point effects rather than conceal differences.
- A 3-query handmade fixture and independent NumPy brute-force checks support CPU correctness. GPU agreement needs actual CUDA hardware.

## 7. Benchmark protocol and required measurements
1. Record GPU model and driver, CUDA toolkit, CPU, OS and compilation options.
2. Prepare one real training/query split.
3. Warm up GPU timing, measure both CUDA kernels with events, and record a separate first-pass total including layout conversion and host/device transfers.
4. Use `--cpu-naive` at small query counts to benchmark a second serial reference; assert it agrees with the pruned CPU implementation. For each query count `[32, 128, 512, 2048]`, keep the number of references fixed at 572,820. Report CPU milliseconds, optimized and baseline GPU kernel milliseconds, GPU single-pass end-to-end milliseconds, 1-NN accuracy, mismatches, kernel speedup, end-to-end speedup.
5. Repeat independent trials and summarize medians when feasible; describe variability.
6. Demonstrate CPU/GPU speedup > 1 on large data, as required. If it does not occur on available hardware, tune your kernel, increase query count if feasible, or explain limitations truthfully; do not invent numbers.

### Results table (fill only after actual measured execution)

Kernel timings are the **median** of repeated CUDA-event samples after warm-up, not an estimated value. CPU and GPU comparisons use identical reference/query rows; the CPU timing is one real serial pass. The naive CPU timing is a supplementary sanity baseline, not the denominator for the default speedup unless explicitly recomputed.
| Query count | CPU ms | Optimized CUDA kernel ms | CUDA single-pass E2E ms | Search speedup | E2E speedup | Exact index mismatches |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | not measured | not measured | not measured | not measured | not measured | not measured |
| 128 | not measured | not measured | not measured | not measured | not measured | not measured |
| 512 | not measured | not measured | not measured | not measured | not measured | not measured |
| 2048 | not measured | not measured | not measured | not measured | not measured | not measured |

## 8. Discussion & threats to validity
- Exact exhaustive brute force is not a proof of universal sequential optimality; an indexed data structure might win for some workloads and approximate search is another task.
- Different hardware may have different cross-over sizes; PCIe transfers and row-to-column-major transform make GPU end-to-end slower for small query sets.
- Covertype is an ML dataset with an imbalanced class distribution, and 1-NN accuracy should not be described as state-of-the-art predictive accuracy.
- Standardization assigns meaning to feature scales but is a design choice; class accuracy is shown for context, while speedup and exact agreement are principal performance results.
- Processing the complete 54-feature dataset with early abandonment is data dependent: actual serial operations may be fewer than worst-case O(QND).
- GPU memory pressure and warp divergence cause variations with queries and reference size.

## 9. Originality
The project is a real geospatial-feature search/classifier—not a standalone vector add, histogram, scan or general reduction problem. The performance study explicitly contrasts baseline CUDA (AoS) and multi-technique optimized CUDA (SoA coalescing + query shared cache + register tiling + warp reductions), with exact index-by-index validation rather than only evaluating elapsed time.

## 10. APOD software development workflow

**Assess:** profile the serial query/reference scan. **Parallelize:** map reference comparisons and query rows onto blocks and threads. **Optimize:** transpose AoS→SoA, cache queries in shared memory, reduce partial min-pairs using warps, amortize layout/copy over many queries. **Deploy:** correctness tests, reproducible scripts, actual GPU benchmark and documented run instructions.

## 11. Submission checklist
- [x] A non-lab problem defined and large public dataset identified
- [x] Serial C++ algorithm and standalone CPU executable
- [x] Two CUDA kernel variants, exact two-stage reduction, code delivered
- [x] Dataset download, deterministic preprocessing, independent CPU tests
- [x] Reproducible measured benchmark/plot scripts
- [ ] Actual NVIDIA GPU compile and execution **(required)**
- [ ] Document measured >1× speedup on sufficiently large data **(required)**
- [ ] Replace results placeholders in slides with measured chart/table
- [ ] Upload folder, guide, slides to own GitHub repository and get the URL
- [ ] Present ~15 minutes including questions

## References
1. Blackard, J. (1998), *Covertype*. UCI ML Repository. https://doi.org/10.24432/C50K5N (CC BY 4.0).
2. NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
3. CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
