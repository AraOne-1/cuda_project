"""Deterministic handmade sanity dataset (for tests, NOT research benchmarks)."""
from pathlib import Path
import struct
import numpy as np

ref = np.array([[0, 0], [1, 0], [0, 2], [3, 3], [5, 4]], dtype=np.float32)
ref_y = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
query = np.array([[0.1, 0.2], [2.9, 3.1], [4.9, 4.1]], dtype=np.float32)
query_y = np.array([1, 4, 5], dtype=np.uint8)
outfile = Path("data/tiny.bin")
outfile.parent.mkdir(exist_ok=True)
with outfile.open("wb") as f:
    f.write(struct.pack("<4sIII", b"CKNN", len(ref), len(query), 2))
    ref.tofile(f)
    ref_y.tofile(f)
    query.tofile(f)
    query_y.tofile(f)
print("Created", outfile, "Expected nearest neighbors: [0, 3, 4]")
