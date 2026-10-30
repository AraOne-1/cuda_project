#include "common.hpp"
#include <cuda_runtime.h>
#include <climits>
#include <cfloat>
#include <sstream>
#include <math_constants.h>

#define CUDA_CHECK(expr) do { \
    cudaError_t rc = (expr); \
    if (rc != cudaSuccess) { \
        std::ostringstream error; error << #expr << " failed: " \
            << cudaGetErrorString(rc) << " (" << __FILE__ << ':' << __LINE__ << ')'; \
        throw std::runtime_error(error.str()); \
    } \
} while (0)

constexpr int BLOCK = 256;
constexpr int REFS_PER_THREAD = 4;

__device__ __forceinline__ bool better_pair(float test_d, int test_id,
                                            float best_d, int best_id) {
    return test_d < best_d || (test_d == best_d && test_id < best_id);
}

// First reduce within 8 warps using warp shuffles, then across warp leaders.
// The final result is valid in thread 0. Invalid candidates have ID INT_MAX.
__device__ __forceinline__ void block_minpair(float& value, int& id) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float dv = __shfl_down_sync(0xffffffff, value, offset);
        const int di = __shfl_down_sync(0xffffffff, id, offset);
        if (lane + offset < 32 && better_pair(dv, di, value, id)) {
            value = dv; id = di;
        }
    }
    __shared__ float warp_values[BLOCK / 32];
    __shared__ int warp_ids[BLOCK / 32];
    if (lane == 0) { warp_values[warp] = value; warp_ids[warp] = id; }
    __syncthreads();
    if (warp == 0) {
        value = (lane < BLOCK / 32) ? warp_values[lane] : CUDART_INF_F;
        id = (lane < BLOCK / 32) ? warp_ids[lane] : INT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float dv = __shfl_down_sync(0xffffffff, value, offset);
            const int di = __shfl_down_sync(0xffffffff, id, offset);
            if (lane + offset < 32 && better_pair(dv, di, value, id)) {
                value = dv; id = di;
            }
        }
    }
}

// Baseline GPU kernel: 1 thread per reference, row-major AoS global memory.
// This version intentionally has no shared-query cache or multi-item register tiling.
__global__ void scan_aos(const float* refs, const float* queries, int nref, int dim,
                         int tiles, float* partial_dist, int* partial_idx) {
    const int r = blockIdx.x * BLOCK + threadIdx.x;
    const int q = blockIdx.y;
    float dist = CUDART_INF_F;
    int best_id = INT_MAX;
    if (r < nref) {
        dist = 0.0f;
        const float* p = refs + size_t(r) * dim;
        const float* x = queries + size_t(q) * dim;
        for (int f = 0; f < dim; ++f) {
            const float delta = p[f] - x[f];
            dist += delta * delta;
        }
        best_id = r;
    }
    block_minpair(dist, best_id);
    if (threadIdx.x == 0) {
        const size_t ix = size_t(q) * tiles + blockIdx.x;
        partial_dist[ix] = dist;
        partial_idx[ix] = best_id;
    }
}

// Optimized: SoA layout for coalescing; query in shared memory; each thread
// processes 4 references so fewer blocks / partial outputs are needed.
__global__ void scan_soa_tiled(const float* refs_soa, const float* queries,
                               int nref, int dim, int tiles,
                               float* partial_dist, int* partial_idx) {
    const int q = blockIdx.y;
    __shared__ float q_shared[64];
    for (int f = threadIdx.x; f < dim; f += BLOCK)
        q_shared[f] = queries[size_t(q) * dim + f];
    __syncthreads();

    const int tile_start = blockIdx.x * BLOCK * REFS_PER_THREAD;
    float best = CUDART_INF_F;
    int best_id = INT_MAX;
    #pragma unroll
    for (int item = 0; item < REFS_PER_THREAD; ++item) {
        const int r = tile_start + item * BLOCK + threadIdx.x;
        if (r >= nref) continue;
        float dist = 0.0f;
        for (int f = 0; f < dim; ++f) {
            const float delta = refs_soa[size_t(f) * nref + r] - q_shared[f];
            dist += delta * delta;
        }
        if (better_pair(dist, r, best, best_id)) {
            best = dist; best_id = r;
        }
    }
    block_minpair(best, best_id);
    if (threadIdx.x == 0) {
        const size_t ix = size_t(q) * tiles + blockIdx.x;
        partial_dist[ix] = best;
        partial_idx[ix] = best_id;
    }
}

__global__ void reduce_partials(const float* dists, const int* ids, int tiles,
                                float* result_dist, int* result_id) {
    const int q = blockIdx.x;
    float best = CUDART_INF_F;
    int best_id = INT_MAX;
    for (int tile = threadIdx.x; tile < tiles; tile += BLOCK) {
        const size_t off = size_t(q) * tiles + tile;
        const float value = dists[off];
        const int candidate = ids[off];
        if (better_pair(value, candidate, best, best_id)) {
            best = value; best_id = candidate;
        }
    }
    block_minpair(best, best_id);
    if (threadIdx.x == 0) {
        result_dist[q] = best;
        result_id[q] = best_id;
    }
}

struct GpuResult {
    std::vector<Answer> answers;
    float kernel_ms = 0;
    double end_to_end_ms = 0;
};

// Includes optional SoA transpose, allocations, H2D copies, device execution,
// D2H copies and deallocations in end_to_end_ms. CPU preprocessing/load excluded
// for BOTH CPU and GPU search benchmarks.
GpuResult gpu_search(const Dataset& data, int nr, int nq, bool optimized, int repeats) {
    const auto whole_start = std::chrono::steady_clock::now();
    const int dim = int(data.dim);
    const int tiles = (nr + BLOCK * (optimized ? REFS_PER_THREAD : 1) - 1) /
                      (BLOCK * (optimized ? REFS_PER_THREAD : 1));
    std::vector<float> transposed;
    const float* host_refs = data.refs.data();
    if (optimized) {
        transposed.resize(size_t(nr) * dim);
        for (int f = 0; f < dim; ++f)
            for (int r = 0; r < nr; ++r)
                transposed[size_t(f) * nr + r] = data.refs[size_t(r) * dim + f];
        host_refs = transposed.data();
    }

    float *d_refs = nullptr, *d_queries = nullptr, *d_part_dist = nullptr, *d_out_dist = nullptr;
    int *d_part_id = nullptr, *d_out_id = nullptr;
    const size_t ref_bytes = size_t(nr) * dim * sizeof(float);
    const size_t query_bytes = size_t(nq) * dim * sizeof(float);
    const size_t partial_count = size_t(tiles) * nq;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_refs), ref_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_queries), query_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_part_dist), partial_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_part_id), partial_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out_dist), size_t(nq) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out_id), size_t(nq) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_refs, host_refs, ref_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queries, data.queries.data(), query_bytes, cudaMemcpyHostToDevice));

    const dim3 grid(tiles, nq);
    auto launch = [&]() {
        if (optimized)
            scan_soa_tiled<<<grid, BLOCK>>>(d_refs, d_queries, nr, dim, tiles,
                                           d_part_dist, d_part_id);
        else
            scan_aos<<<grid, BLOCK>>>(d_refs, d_queries, nr, dim, tiles,
                                     d_part_dist, d_part_id);
        CUDA_CHECK(cudaGetLastError());
        reduce_partials<<<nq, BLOCK>>>(d_part_dist, d_part_id, tiles,
                                      d_out_dist, d_out_id);
        CUDA_CHECK(cudaGetLastError());
    };

    // Measure a SINGLE end-to-end GPU pass (setup -> transpose/copy -> launch
    // -> result copy); this excludes dataset-file loading and late deallocation.
    // Separately run warm-up + repeated CUDA-event timings for stable kernel ms.
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> host_indices(nq);
    std::vector<float> host_distances(nq);
    CUDA_CHECK(cudaMemcpy(host_indices.data(), d_out_id, size_t(nq) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_distances.data(), d_out_dist, size_t(nq) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    const auto single_pass_end = std::chrono::steady_clock::now();
    const double single_pass_ms =
        std::chrono::duration<double, std::milli>(single_pass_end - whole_start).count();

    // Warm-up and timed compute-only passes. Time *both* GPU kernels.
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t event_start, event_stop;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_stop));
    std::vector<float> measurements;
    measurements.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventRecord(event_start));
        launch();
        CUDA_CHECK(cudaEventRecord(event_stop));
        CUDA_CHECK(cudaEventSynchronize(event_stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, event_start, event_stop));
        measurements.push_back(ms);
    }
    std::sort(measurements.begin(), measurements.end());
    const float kernel_median = (repeats % 2 == 1)
        ? measurements[repeats / 2]
        : 0.5f * (measurements[repeats / 2 - 1] + measurements[repeats / 2]);
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_stop));

    CUDA_CHECK(cudaFree(d_refs));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_part_dist));
    CUDA_CHECK(cudaFree(d_part_id));
    CUDA_CHECK(cudaFree(d_out_dist));
    CUDA_CHECK(cudaFree(d_out_id));

    GpuResult result;
    result.kernel_ms = kernel_median;
    result.end_to_end_ms = single_pass_ms;
    result.answers.resize(nq);
    for (int q = 0; q < nq; ++q) {
        if (host_indices[q] < 0 || host_indices[q] >= nr)
            throw std::runtime_error("GPU returned invalid nearest-neighbor index");
        result.answers[q] = {host_indices[q], host_distances[q]};
    }
    return result;
}

size_t count_mismatches(const std::vector<Answer>& a, const std::vector<Answer>& b) {
    size_t mismatches = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (a[i].index != b[i].index) ++mismatches;
    return mismatches;
}

void append_metric(const std::string& path, const std::string& version, int nr,
                   int nq, int dim, double cpu_ms, double kernel_ms,
                   double e2e_ms, double acc, int mismatches) {
    if (path.empty()) return;
    std::ifstream check(path);
    const bool exists = check.good() && check.peek() != std::ifstream::traits_type::eof();
    check.close();
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("Cannot write metrics: " + path);
    if (!exists)
        out << "version,reference_rows,query_rows,dimensions,cpu_ms,gpu_kernel_ms,"
               "gpu_e2e_ms,kernel_speedup,e2e_speedup,accuracy_percent,index_mismatches\n";
    out << std::setprecision(10) << version << ',' << nr << ',' << nq << ',' << dim << ','
        << cpu_ms << ',' << kernel_ms << ',' << e2e_ms << ','
        << (cpu_ms > 0 && kernel_ms > 0 ? cpu_ms / kernel_ms : 0.0) << ','
        << (cpu_ms > 0 && e2e_ms > 0 ? cpu_ms / e2e_ms : 0.0) << ','
        << acc << ',' << mismatches << '\n';
}

int main(int argc, char** argv) {
    try {
        if (has_flag(argc, argv, "--help")) {
            std::cout << "Usage: knn_cuda --data data/covertype.bin [--queries 64]"
                         " [--refs 572820] [--mode both|cpu|gpu] [--repeats 3]"
                         " [--gpu-baseline] [--cpu-naive] [--metrics results/benchmark.csv]"
                         " [--output results/predictions.csv]\n";
            return 0;
        }
        const auto data = read_dataset(get_arg(argc, argv, "--data", "data/covertype.bin"));
        const int nr = get_int_arg(argc, argv, "--refs", int(data.n_ref));
        const int nq = get_int_arg(argc, argv, "--queries", 64);
        const int repeats = get_int_arg(argc, argv, "--repeats", 3);
        const std::string mode = get_arg(argc, argv, "--mode", "both");
        const std::string metrics = get_arg(argc, argv, "--metrics", "");
        const std::string output = get_arg(argc, argv, "--output", "");
        check_limits(nr, nq, data);
        if (repeats < 1 || repeats > 100 || (mode != "both" && mode != "cpu" && mode != "gpu") ||
            (mode == "gpu" && has_flag(argc, argv, "--cpu-naive")))
            throw std::runtime_error("Invalid --repeats or --mode");
        std::cout << "Dataset: " << nr << " references, " << nq
                  << " queries, " << data.dim << " features\n";

        std::vector<Answer> cpu_answers;
        double cpu_ms = 0.0;
        if (mode != "gpu") {
            const auto t0 = std::chrono::steady_clock::now();
            cpu_answers = cpu_search(data, nr, nq);
            const auto t1 = std::chrono::steady_clock::now();
            cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << std::fixed << std::setprecision(3)
                      << "CPU exact sequential (ms): " << cpu_ms
                      << " | accuracy (%): " << accuracy(data, cpu_answers) << '\n';
        }
        // Optional naive serial comparison, distinct from the mandatory optimized
        // CPU baseline. Use a small query count because full exhaustive work is costly.
        if (has_flag(argc, argv, "--cpu-naive")) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto naive = cpu_search_naive(data, nr, nq);
            const auto t1 = std::chrono::steady_clock::now();
            const double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const auto& oracle = (mode != "gpu") ? cpu_answers : naive;
            int bad = 0;
            for (int i = 0; i < nq; ++i)
                if (naive[i].index != oracle[i].index) ++bad;
            std::cout << "CPU naive exhaustive (ms): " << naive_ms
                      << " | naive/optimized CPU speedup: "
                      << (cpu_ms > 0 ? naive_ms/cpu_ms : 0)
                      << " | index mismatches: " << bad << '\n';
            append_metric(metrics, "cpu_naive", nr, nq, int(data.dim), naive_ms,
                          0, 0, accuracy(data, naive), bad);
            if (bad) return 2;
        }
        if (mode == "cpu") {
            dump_csv(output, data, cpu_answers);
            append_metric(metrics, "cpu_only", nr, nq, int(data.dim), cpu_ms,
                          0, 0, accuracy(data, cpu_answers), 0);
            return 0;
        }
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices < 1) throw std::runtime_error("No NVIDIA CUDA-capable GPU found");
        cudaDeviceProp device;
        CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        std::cout << "CUDA GPU: " << device.name << '\n';
        if (mode == "gpu") {
            const int vq = std::min(nq, 16);
            std::cout << "GPU-only mode: checking first " << vq << " queries on CPU\n";
            cpu_answers = cpu_search(data, nr, vq);
        }

        const GpuResult tuned = gpu_search(data, nr, nq, true, repeats);
        const size_t mismatches = count_mismatches(cpu_answers, tuned.answers);
        const double acc = accuracy(data, tuned.answers);
        std::cout << "GPU optimized kernel (ms, median of " << repeats << "): "
                  << tuned.kernel_ms << '\n'
                  << "GPU single-pass end-to-end (ms): " << tuned.end_to_end_ms << '\n'
                  << "GPU accuracy (%): " << acc << '\n'
                  << "CPU/GPU nearest-reference index mismatches: " << mismatches
                  << " of " << cpu_answers.size() << " checked\n";
        if (cpu_ms > 0) {
            std::cout << "CPU / GPU kernel speedup: " << cpu_ms / tuned.kernel_ms << "x\n"
                      << "CPU / GPU end-to-end speedup*: " << cpu_ms / tuned.end_to_end_ms << "x\n";
        }
        std::cout << "*End-to-end includes SoA conversion, allocation, H2D, one search,"
                  << " and D2H (no file loading; deallocation excluded).\n";
        dump_csv(output, data, tuned.answers);
        append_metric(metrics, "gpu_optimized", nr, nq, int(data.dim), cpu_ms,
                      tuned.kernel_ms, tuned.end_to_end_ms, acc, int(mismatches));

        if (has_flag(argc, argv, "--gpu-baseline")) {
            const GpuResult baseline = gpu_search(data, nr, nq, false, repeats);
            const size_t bad = count_mismatches(tuned.answers, baseline.answers);
            std::cout << "GPU baseline AoS kernel (ms): " << baseline.kernel_ms << '\n'
                      << "Optimized / baseline improvement: "
                      << baseline.kernel_ms / tuned.kernel_ms << "x\n"
                      << "GPU baseline vs optimized result mismatches: " << bad << '\n';
            append_metric(metrics, "gpu_baseline", nr, nq, int(data.dim), cpu_ms,
                          baseline.kernel_ms, baseline.end_to_end_ms,
                          accuracy(data, baseline.answers), int(bad));
            if (bad != 0) return 2;
        }
        return mismatches == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
