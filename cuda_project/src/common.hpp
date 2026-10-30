#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Dataset {
    uint32_t n_ref = 0, n_queries = 0, dim = 0;
    std::vector<float> refs, queries; // row-major: [sample][feature]
    std::vector<uint8_t> ref_labels, query_labels; // classes 1..7
};

inline void read_checked(std::ifstream& in, void* dst, size_t bytes) {
    if (!in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("Truncated or invalid data file");
    }
}

inline Dataset read_dataset(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + filename + "; run scripts/prepare_data.py first");
    char magic[4];
    read_checked(in, magic, 4);
    if (std::memcmp(magic, "CKNN", 4) != 0) throw std::runtime_error("Not a CKNN binary file");
    Dataset d;
    read_checked(in, &d.n_ref, 4);
    read_checked(in, &d.n_queries, 4);
    read_checked(in, &d.dim, 4);
    if (d.n_ref < 1 || d.n_queries < 1 || d.dim < 1 || d.dim > 64 ||
        d.n_ref > 10000000 || d.n_queries > 500000 ||
        uint64_t(d.n_ref + d.n_queries) * d.dim > 600000000ULL) {
        throw std::runtime_error("Invalid dataset dimensions");
    }
    d.refs.resize(size_t(d.n_ref) * d.dim);
    d.ref_labels.resize(d.n_ref);
    d.queries.resize(size_t(d.n_queries) * d.dim);
    d.query_labels.resize(d.n_queries);
    read_checked(in, d.refs.data(), d.refs.size() * sizeof(float));
    read_checked(in, d.ref_labels.data(), d.ref_labels.size());
    read_checked(in, d.queries.data(), d.queries.size() * sizeof(float));
    read_checked(in, d.query_labels.data(), d.query_labels.size());
    return d;
}

struct Answer { int index; float distance2; };

// Reference exhaustive serial 1-NN: no pruning or early stop.
// Same squared-Euclidean metric and stable smallest-row-ID tie breaking.
inline std::vector<Answer> cpu_search_naive(const Dataset& data, int nr, int nq) {
    std::vector<Answer> out(nq);
    const int d = int(data.dim);
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = data.queries.data() + size_t(qi) * d;
        float best = std::numeric_limits<float>::infinity();
        int best_index = -1;
        for (int r = 0; r < nr; ++r) {
            const float* x = data.refs.data() + size_t(r) * d;
            float dist = 0.0f;
            for (int f = 0; f < d; ++f) {
                const float delta = x[f] - q[f];
                dist += delta * delta;
            }
            if (dist < best) { best = dist; best_index = r; }
        }
        out[qi] = {best_index, best};
    }
    return out;
}

// Exact 1-nearest-neighbor scan. Early abandonment is valid because every
// remaining squared-distance term is nonnegative. Ties pick the smaller row ID.
// Runtime O(number_of_queries * number_of_references * dimensions) worst case.
inline std::vector<Answer> cpu_search(const Dataset& data, int nr, int nq) {
    std::vector<Answer> out(nq);
    const int d = int(data.dim);
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = data.queries.data() + size_t(qi) * d;
        float best = std::numeric_limits<float>::infinity();
        int best_index = -1;
        for (int r = 0; r < nr; ++r) {
            const float* p = data.refs.data() + size_t(r) * d;
            float sum = 0.0f;
            for (int f = 0; f < d; ++f) {
                const float delta = p[f] - q[f];
                sum += delta * delta;
                if (sum > best) break; // exact branch-and-bound pruning
            }
            if (sum < best) { best = sum; best_index = r; }
        }
        out[qi] = {best_index, best};
    }
    return out;
}

inline double accuracy(const Dataset& data, const std::vector<Answer>& answers) {
    int good = 0;
    for (size_t i = 0; i < answers.size(); ++i) {
        if (answers[i].index >= 0 &&
            data.ref_labels[answers[i].index] == data.query_labels[i]) ++good;
    }
    return answers.empty() ? 0 : (100.0 * good / answers.size());
}

inline void dump_csv(const std::string& path, const Dataset& data,
                     const std::vector<Answer>& answers) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write " + path);
    out << "query_id,nearest_reference_id,predicted_cover_type,true_cover_type,distance_squared\n";
    for (size_t i = 0; i < answers.size(); ++i) {
        const Answer a = answers[i];
        out << i << ',' << a.index << ',' << int(data.ref_labels[a.index]) << ','
            << int(data.query_labels[i]) << ',' << std::setprecision(9) << a.distance2 << '\n';
    }
}

inline std::string get_arg(int argc, char** argv, const std::string& opt,
                           const std::string& fallback) {
    for (int i = 1; i < argc; ++i)
        if (argv[i] == opt && i + 1 < argc) return argv[i + 1];
    return fallback;
}
inline bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) if (argv[i] == flag) return true;
    return false;
}
inline int get_int_arg(int argc, char** argv, const std::string& opt, int fallback) {
    return std::stoi(get_arg(argc, argv, opt, std::to_string(fallback)));
}
inline void check_limits(int nr, int nq, const Dataset& d) {
    if (nr < 1 || nr > int(d.n_ref) || nq < 1 || nq > int(d.n_queries))
        throw std::runtime_error("--refs/--queries must be within dataset bounds");
}
