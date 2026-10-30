#!/usr/bin/env python3
"""Prepare the real UCI Forest Covertype dataset for the C++/CUDA 1-NN program.

Requirements: Python 3.9+, numpy. Dataset: 581,012 rows, 54 features, one label.
Attribution: Blackard (1998), Covertype, UCI ML Repository, CC BY 4.0.
https://doi.org/10.24432/C50K5N
"""
import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
    "https://ndownloader.figshare.com/files/5976039",
]
KNOWN_SHA256 = "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771"
N_QUERY = 8192


def download(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        print("Using cached archive:", path)
        return
    for url in URLS:
        try:
            print("Downloading:", url)
            with urllib.request.urlopen(url, timeout=120) as response, open(path, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            return
        except Exception as exc:
            print("Download failed:", exc)
            path.unlink(missing_ok=True)
    raise RuntimeError("Dataset download failed; download covtype.data.gz manually from UCI into data/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/covtype.data.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/covertype.bin"))
    parser.add_argument("--query-count", type=int, default=N_QUERY)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    download(args.archive)
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    print("Archive SHA256:", digest)
    if digest != KNOWN_SHA256:
        print("WARNING: checksum differs from the scikit-learn mirror; verify source manually")

    # NumPy parses directly into a compact int32 matrix (avoid keeping ~32M
    # Python integers in RAM, which can otherwise consume > 1 GB).
    with gzip.open(args.archive, "rt") as f:
        raw = np.loadtxt(f, dtype=np.int32, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] != 55:
        raise ValueError(f"Expected 55 CSV columns, got shape {raw.shape}")
    if raw.shape != (581012, 55):
        print("NOTE: unexpected row count", raw.shape)
    if not 1 <= args.query_count < raw.shape[0]:
        raise ValueError("--query-count must lie in [1, n_rows-1]")

    rng = np.random.default_rng(args.seed)
    shuffled = raw[rng.permutation(len(raw))]
    n_ref = len(raw) - args.query_count
    reference = shuffled[:n_ref, :54].astype(np.float32)
    queries = shuffled[n_ref:, :54].astype(np.float32)
    ref_labels = shuffled[:n_ref, 54].astype(np.uint8)
    query_labels = shuffled[n_ref:, 54].astype(np.uint8)

    # Scale the ten quantitative variables with statistics ONLY from references.
    # The remaining 44 binary wilderness/soil features are left as 0/1.
    mean = reference[:, :10].mean(axis=0, dtype=np.float64)
    std = reference[:, :10].std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    reference[:, :10] = ((reference[:, :10] - mean) / std).astype(np.float32)
    queries[:, :10] = ((queries[:, :10] - mean) / std).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        out.write(struct.pack("<4sIII", b"CKNN", n_ref, args.query_count, 54))
        reference.tofile(out)    # float32 ref features, row-major
        ref_labels.tofile(out)  # uint8 ref labels
        queries.tofile(out)      # float32 query features, row-major
        query_labels.tofile(out)

    meta = {
        "dataset": "Forest Covertype (UCI ID 31)",
        "doi": "10.24432/C50K5N",
        "dataset_license": "CC BY 4.0",
        "dataset_sha256": digest,
        "seed": args.seed,
        "reference_rows": n_ref,
        "query_rows": args.query_count,
        "dimensions": 54,
        "numeric_features_scaled": 10,
        "binary_features_unchanged": 44,
        "scale_means": mean.tolist(),
        "scale_stds": std.tolist(),
        "binary_layout": "CKNN magic (4B), n_ref/query/dim (uint32 LE), reference float32 matrix, reference uint8 labels, query float32 matrix, query uint8 labels",
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Ready:", args.output, "bytes=", args.output.stat().st_size)
    print("Reference rows:", n_ref, "  Query rows:", args.query_count)


if __name__ == "__main__":
    main()
