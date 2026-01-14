"""
SRM Memory-Wall / Scalability Experiment (1M + 3 lines)
-------------------------------------------------------
Compares memory scaling as N grows:

1) Dense baseline (theoretical):
     N * d * 4 bytes  (float32 embeddings)

2) SRM routing (Python / worst-case overhead):
     - ids: Python list of ints
     - buckets: list[K] of Python lists (ints)
     - codebook: K x d float32

3) SRM routing (Packed / production-style):
     - flat_ids: uint32[N] grouped by bucket
     - offsets:  uint32[K+1]
     - codebook: K x d float32

Config intent:
  SRMConfig(d=768, K=256, store_item_embeddings=False)

Notes:
  - This isolates MEMORY scaling. Bucket assignment is simulated (random).
  - We still use your srm.py to train/build the codebook (constant memory).

Run:
  python srm_scalability_memory_wall_1M_three_lines.py

Outputs:
  - srm_memory_wall_scaling_1M_three_lines.png
  - srm_memory_wall_scaling_1M_three_lines.csv
"""

import sys
import gc
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

# --- Import your SRM implementation ---
# Ensure srm.py is in the same directory or on PYTHONPATH.
import srm
SRMConfig = srm.SRMConfig
SemanticRoutingMemory = srm.SemanticRoutingMemory


def mb(x: int) -> float:
    return x / (1024 ** 2)


def total_size_dedup(obj) -> int:
    """
    Deduplicated deep size using sys.getsizeof + graph traversal with 'seen'.
    - Avoids double-counting shared objects.
    - Treats numpy arrays as leaf nodes (sys.getsizeof already includes buffer).
    """
    seen = set()
    q = deque([obj])
    size = 0

    while q:
        o = q.popleft()
        oid = id(o)
        if oid in seen:
            continue
        seen.add(oid)

        size += sys.getsizeof(o)

        # numpy arrays: leaf (buffer counted inside sys.getsizeof)
        if isinstance(o, np.ndarray):
            continue

        # expand containers
        if isinstance(o, dict):
            for k, v in o.items():
                q.append(k)
                q.append(v)
        elif isinstance(o, (list, tuple, set, frozenset)):
            for it in o:
                q.append(it)

    return size


def build_codebook(d=768, K=256, train_n=4000, seed=0) -> np.ndarray:
    """
    Train a codebook once via SRM. This is a constant memory term.
    We keep n_iter modest to keep the script fast; memory footprint is what matters.
    """
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((train_n, d), dtype=np.float32)

    cfg = SRMConfig(
        d=d,
        K=K,
        store_item_embeddings=False,  # important
        store_payloads=False,
        pre_normalize=True,
        seed=seed,
    )

    srm_inst = SemanticRoutingMemory(cfg)
    srm_inst.fit_codebook(X_train, n_iter=10, init="kmeans++", verbose=False)
    return srm_inst.codebook.copy()


def build_python_routing(N: int, K: int, seed: int = 0):
    """
    Python routing structures (worst-case overhead):
      - ids: list(range(N))        => N Python int objects
      - buckets: K lists of ints   => references to the same int objects in ids
      - assignments: uint16[N] (for building packed structure too)
    """
    rng = np.random.default_rng(seed)
    assignments = rng.integers(0, K, size=N, dtype=np.uint16)

    ids = list(range(N))
    buckets = [[] for _ in range(K)]
    for i, b in enumerate(assignments):
        buckets[int(b)].append(ids[i])  # share the exact int objects from ids list

    return ids, buckets, assignments


def build_packed_routing(assignments: np.ndarray, K: int):
    """
    Packed routing (production-style idea):
      - flat_ids: uint32[N] grouped by bucket
      - offsets:  uint32[K+1] prefix sums, offsets[b]..offsets[b+1] is slice for bucket b
    """
    counts = np.bincount(assignments.astype(np.int64), minlength=K)

    # stable sort by assignment so each bucket is contiguous in flat_ids
    order = np.argsort(assignments, kind="stable")
    flat_ids = order.astype(np.uint32, copy=True)

    offsets = np.empty(K + 1, dtype=np.uint32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts, dtype=np.uint64).astype(np.uint32)

    return flat_ids, offsets


def main():
    # Experiment settings
    d = 768
    K = 256
    Ns = [1000, 5000, 10000, 50000, 100000, 1_000_000]

    # Train codebook once (constant term)
    codebook = build_codebook(d=d, K=K, train_n=4000, seed=0)
    codebook_bytes = sys.getsizeof(codebook)

    rows = []
    for N in Ns:
        # ---- Python routing build ----
        t0 = time.time()
        ids, buckets, assignments = build_python_routing(N, K, seed=N)
        build_py_s = time.time() - t0

        # Measure dedup deep sizes
        ids_bytes = total_size_dedup(ids)
        buckets_bytes = total_size_dedup(buckets)

        # Combined routing object (dedup across ids/buckets/codebook)
        python_total_bytes = total_size_dedup(
            {"ids": ids, "buckets": buckets, "codebook": codebook}
        )

        # ---- Packed routing build ----
        t1 = time.time()
        flat_ids, offsets = build_packed_routing(assignments, K)
        build_packed_s = time.time() - t1

        packed_bytes = sys.getsizeof(flat_ids) + sys.getsizeof(offsets) + codebook_bytes

        # ---- Dense baseline (theoretical) ----
        dense_bytes = N * d * 4  # float32

        rows.append(
            {
                "N": N,
                "Dense_MB_theoretical": mb(dense_bytes),
                "SRM_python_ids_MB": mb(ids_bytes),
                "SRM_python_buckets_MB": mb(buckets_bytes),
                "SRM_python_total_MB": mb(python_total_bytes),
                "SRM_packed_total_MB": mb(packed_bytes),
                "Dense/SRM_python_total_ratio": dense_bytes / python_total_bytes,
                "Dense/SRM_packed_total_ratio": dense_bytes / packed_bytes,
                "build_time_python_s": build_py_s,
                "build_time_packed_s": build_packed_s,
            }
        )

        # cleanup
        del ids, buckets, assignments, flat_ids, offsets
        gc.collect()

    # ---- Save table ----
    out_csv = "srm_memory_wall_scaling_1M_three_lines.csv"
    if HAVE_PANDAS:
        df = pd.DataFrame(rows).sort_values("N")
        df.to_csv(out_csv, index=False)
        print(f"Saved CSV: {out_csv}")
        print(df.to_string(index=False))
    else:
        print("pandas not installed; printing rows:")
        for r in rows:
            print(r)

    # ---- Plot (log-log makes 1M contrast pop) ----
    Ns_sorted = [r["N"] for r in rows]
    dense = [r["Dense_MB_theoretical"] for r in rows]
    srm_py = [r["SRM_python_total_MB"] for r in rows]
    srm_pk = [r["SRM_packed_total_MB"] for r in rows]

    plt.figure(figsize=(9, 5.5))
    plt.plot(Ns_sorted, dense, marker="o", label="Dense Baseline (Linear Growth)")
    plt.plot(Ns_sorted, srm_py, marker="o", label="SRM (Python Routing Layer)")
    plt.plot(Ns_sorted, srm_pk, marker="o", label="SRM (Packed Routing - uint32)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N (Data Size)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Wall Scalability: Dense vs SRM Routing (d=768, K=256, no item embeddings)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    out_png = "srm_memory_wall_scaling_1M_three_lines.png"
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
