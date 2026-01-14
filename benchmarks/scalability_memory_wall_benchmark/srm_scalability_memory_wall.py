"""
SRM Memory-Wall / Scalability Experiment
---------------------------------------
Compares:
  - Dense baseline: N * d * 4 bytes (float32 embeddings)
  - SRM routing layer: buckets + ids + codebook
    with SRMConfig(d=768, K=256, store_item_embeddings=False)

Notes:
  * This script measures real Python object overhead for SRM's routing structures.
  * It intentionally does NOT allocate a dense index; dense is computed theoretically.

Requirements:
  pip install numpy matplotlib
  (optional) pip install pandas

Run:
  python srm_scalability_memory_wall.py
"""

import sys, gc, time
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

# If srm.py isn't importable, add current directory
if "srm" not in sys.modules:
    sys.path.insert(0, ".")

import srm
SRMConfig = srm.SRMConfig
SemanticRoutingMemory = srm.SemanticRoutingMemory

def mb(bytes_): 
    return bytes_ / (1024**2)

def deep_size_int_list(lst):
    # list container size (includes pointer array) + sizes of int objects
    return sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst)

def deep_size_buckets(buckets):
    total = sys.getsizeof(buckets)
    for inner in buckets:
        total += deep_size_int_list(inner)
    return total

def codebook_size(arr: np.ndarray):
    # For numpy arrays, sys.getsizeof includes the underlying data buffer.
    return sys.getsizeof(arr)

def build_codebook(d=768, K=256, train_n=5000, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((train_n, d), dtype=np.float32)

    cfg = SRMConfig(
        d=d,
        K=K,
        store_item_embeddings=False,   # <-- required
        store_payloads=False,          # keep routing-only measurement clean
        pre_normalize=True,
        seed=seed,
    )
    srm_inst = SemanticRoutingMemory(cfg)
    srm_inst.fit_codebook(X_train, n_iter=15, init="kmeans++", verbose=False)
    return srm_inst.codebook.copy()

def fill_srm(srm_inst, N, batch_size=5000, seed=0):
    rng = np.random.default_rng(seed)
    remaining = N
    while remaining > 0:
        bs = min(batch_size, remaining)
        X = rng.standard_normal((bs, srm_inst.cfg.d), dtype=np.float32)
        srm_inst.add_items(X)  # ids auto
        remaining -= bs

def main():
    Ns = [1000, 5000, 10000, 50000, 100000]
    d = 768
    K = 256

    # Train codebook once (constant cost)
    codebook = build_codebook(d=d, K=K, train_n=5000, seed=0)

    rows = []
    for N in Ns:
        cfg = SRMConfig(
            d=d,
            K=K,
            store_item_embeddings=False,  # <-- required
            store_payloads=False,
            pre_normalize=True,
            seed=0,
        )
        srm_inst = SemanticRoutingMemory(cfg)

        # Reuse same codebook for comparability
        srm_inst.codebook = codebook.copy()
        srm_inst._buckets = [[] for _ in range(cfg.K)]

        t0 = time.time()
        fill_srm(srm_inst, N, batch_size=5000, seed=N)
        build_time = time.time() - t0

        size_buckets = deep_size_buckets(srm_inst._buckets)
        size_ids = deep_size_int_list(srm_inst._ids)
        size_codebook = codebook_size(srm_inst.codebook)
        routing_size = size_buckets + size_ids + size_codebook

        dense_bytes = N * d * 4  # float32 baseline

        rows.append({
            "N": N,
            "Dense_MB_theoretical": mb(dense_bytes),
            "SRM_buckets_MB": mb(size_buckets),
            "SRM_ids_MB": mb(size_ids),
            "SRM_codebook_MB": mb(size_codebook),
            "SRM_routing_MB": mb(routing_size),
            "Dense/SRM_routing_ratio": dense_bytes / routing_size,
            "build_time_s": build_time
        })

        del srm_inst
        gc.collect()

    # Plot
    Ns_sorted = sorted([r["N"] for r in rows])
    dense = [next(r["Dense_MB_theoretical"] for r in rows if r["N"] == n) for n in Ns_sorted]
    srm_mem = [next(r["SRM_routing_MB"] for r in rows if r["N"] == n) for n in Ns_sorted]

    plt.figure(figsize=(8,5))
    plt.plot(Ns_sorted, dense, marker="o", label="Dense Baseline (Linear Growth)")
    plt.plot(Ns_sorted, srm_mem, marker="o", label="SRM (Routing Layer)")
    plt.xscale("log")
    plt.xlabel("N (Data Size)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("SRM vs Dense: Memory Scaling (d=768, K=256, store_item_embeddings=False)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("srm_memory_wall_scaling.png", dpi=200)
    print("Saved plot: srm_memory_wall_scaling.png")

    if HAVE_PANDAS:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values("N")
        df.to_csv("srm_memory_wall_scaling.csv", index=False)
        print("Saved table: srm_memory_wall_scaling.csv")
        print(df.to_string(index=False))
    else:
        print("pandas not installed; printing rows:")
        for r in sorted(rows, key=lambda x: x["N"]):
            print(r)

if __name__ == "__main__":
    main()
