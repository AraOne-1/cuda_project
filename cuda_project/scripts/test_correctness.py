#!/usr/bin/env python3
"""Test the sequential implementation against an independent NumPy oracle."""
import csv
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import numpy as np


def main():
    exe = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/knn_cpu")
    rng = np.random.default_rng(24)
    nr, nq, dim = 601, 9, 54
    ref = rng.standard_normal((nr, dim)).astype(np.float32)
    query = rng.standard_normal((nq, dim)).astype(np.float32)
    labels = (rng.integers(1, 8, size=nr)).astype(np.uint8)
    targets = (rng.integers(1, 8, size=nq)).astype(np.uint8)

    # Use a 32-bit accumulation oracle matching the C++/CUDA float arithmetic,
    # and intentionally duplicate a row to exercise tie-breaking.
    ref[200] = ref[25]
    with tempfile.TemporaryDirectory() as tmp:
        dataset = Path(tmp) / "test.bin"
        preds = Path(tmp) / "preds.csv"
        with dataset.open("wb") as f:
            f.write(struct.pack("<4sIII", b"CKNN", nr, nq, dim))
            ref.tofile(f); labels.tofile(f); query.tofile(f); targets.tofile(f)
        cmd = [str(exe.resolve()), "--data", str(dataset), "--queries", str(nq),
               "--output", str(preds), "--compare-naive"]
        print("Running", " ".join(cmd))
        subprocess.run(cmd, check=True)
        with preds.open(newline="") as f:
            actual = [int(row["nearest_reference_id"]) for row in csv.DictReader(f)]
        expected = [int(np.argmin(np.sum((ref - q) ** 2, axis=1))) for q in query]
        assert actual == expected, f"Mismatch actual={actual}, expected={expected}"
        print("PASS: all", nq, "CPU nearest neighbors agree with independent NumPy")


if __name__ == "__main__":
    main()
