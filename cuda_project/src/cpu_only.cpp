#include "common.hpp"

int main(int argc, char** argv) {
    try {
        if (has_flag(argc, argv, "--help")) {
            std::cout << "Usage: knn_cpu --data data/covertype.bin [--queries 64]"
                         " [--refs 50000] [--compare-naive] [--output results/cpu.csv]\n";
            return 0;
        }
        const auto data = read_dataset(get_arg(argc, argv, "--data", "data/covertype.bin"));
        const int nr = get_int_arg(argc, argv, "--refs", int(data.n_ref));
        const int nq = get_int_arg(argc, argv, "--queries", 64);
        check_limits(nr, nq, data);
        std::cout << "Dataset: refs=" << nr << " queries=" << nq
                  << " dimensions=" << data.dim << "\n";
        const auto t0 = std::chrono::steady_clock::now();
        const auto ans = cpu_search(data, nr, nq);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        std::cout << std::fixed << std::setprecision(3)
                  << "CPU time (ms): " << ms << "\n"
                  << "1-NN accuracy (%): " << accuracy(data, ans) << "\n"
                  << "Checksum (first NN index): " << ans[0].index << "\n";
        if (has_flag(argc, argv, "--compare-naive")) {
            const auto b0 = std::chrono::steady_clock::now();
            const auto naive = cpu_search_naive(data, nr, nq);
            const auto b1 = std::chrono::steady_clock::now();
            const double naive_ms = std::chrono::duration<double, std::milli>(b1-b0).count();
            int mismatches = 0;
            for (int i = 0; i < nq; ++i)
                if (naive[i].index != ans[i].index) ++mismatches;
            std::cout << "CPU naive exhaustive (ms): " << naive_ms << "\n"
                      << "CPU naive/optimized speedup: " << naive_ms / ms << "x\n"
                      << "CPU baseline index mismatches: " << mismatches << "\n";
            if (mismatches) return 2;
        }
        dump_csv(get_arg(argc, argv, "--output", ""), data, ans);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
