
"""
K-Sweep Analysis for Semantic Routing Memory (SRM)

What it does (per requested spec):
- Creates a synthetic dataset with N=10_000, d=768 (gaussian blobs)
- Splits into queries (1_000) and database (9_000)
- Sweeps K over: [2,4,8,16,32,64,128,256,512,1024,2048]
- For each K:
  * Fits SRM codebook (K-means) using database vectors
  * Adds items to SRM using add_items
  * Measures:
      - Recall@10 vs brute-force ground truth (cosine)
      - Routing storage size of srm._buckets + srm._ids (bytes)
      - Theoretical compression ratio vs float32 dense vectors
      - Mean query latency (ms)
  * Prints a Pandas DataFrame and plots 3 graphs (matplotlib)

Usage:
  python k_sweep_analysis.py

Notes:
- This script assumes srm.py is in the same folder or importable in PYTHONPATH.
- Runtime can be heavy because it runs K-means for large K. If needed, reduce n_iter_kmeans
  or fit on a subset of database vectors (see FIT_SUBSAMPLE).
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from srm import SRMConfig, SemanticRoutingMemory


# -----------------------------
# Helpers: normalization & sizing
# -----------------------------
def l2_normalize(X: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def deep_getsizeof(obj: Any, seen: Optional[Set[int]] = None) -> int:
    """
    Recursively estimate the memory footprint (bytes) of a Python object using sys.getsizeof.
    This is an approximation of in-memory Python object overhead, not a perfect "RSS" measure.
    """
    import sys

    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_getsizeof(k, seen)
            size += deep_getsizeof(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += deep_getsizeof(item, seen)

    return size


def code_bytes_per_item(K: int) -> int:
    """
    Smallest integer bytes to encode [0..K-1] codes.
    E.g. K<=256 -> 1 byte; K<=65536 -> 2 bytes ...
    """
    bits = math.ceil(math.log2(K))
    return max(1, math.ceil(bits / 8))


# -----------------------------
# Ground-truth brute force (cosine)
# -----------------------------
def brute_force_topk_cosine(
    Q: np.ndarray,
    DB: np.ndarray,
    topk: int = 10,
    batch: int = 100,
) -> np.ndarray:
    """
    Computes exact TopK by brute-force cosine similarity in batches to control memory.

    Returns:
      gt: (n_queries, topk) indices into DB (0..n_db-1) sorted by descending similarity.
    """
    Qn = l2_normalize(Q)
    DBn = l2_normalize(DB)
    n_q = Qn.shape[0]
    gt = np.empty((n_q, topk), dtype=np.int64)

    DBt = DBn.T  # (d, n_db)
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        sims = Qn[start:end] @ DBt  # (b, n_db)

        # argpartition to get topk, then sort those topk
        part = np.argpartition(sims, -topk, axis=1)[:, -topk:]  # (b, topk)
        part_sims = np.take_along_axis(sims, part, axis=1)      # (b, topk)
        order = np.argsort(-part_sims, axis=1)                  # (b, topk)
        top = np.take_along_axis(part, order, axis=1)           # (b, topk)

        gt[start:end] = top

    return gt


# -----------------------------
# Synthetic dataset: gaussian blobs
# -----------------------------
def make_blobs(
    n: int,
    d: int,
    n_centers: int = 50,
    noise_std: float = 0.35,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_centers, d), dtype=np.float32)
    labels = rng.integers(0, n_centers, size=n, dtype=np.int64)
    noise = rng.standard_normal((n, d), dtype=np.float32) * noise_std
    X = centers[labels] + noise
    return X.astype(np.float32, copy=False)


# -----------------------------
# Main Experiment
# -----------------------------
def run_k_sweep(
    K_list: Sequence[int],
    n_total: int = 10_000,
    n_queries: int = 1_000,
    d: int = 768,
    topk: int = 10,
    m: int = 8,
    max_candidates: int = 5_000,
    seed: int = 0,
    n_iter_kmeans: int = 15,
    FIT_SUBSAMPLE: Optional[int] = None,  # set e.g. 6000 to speed up codebook fit
    gt_batch: int = 100,
) -> pd.DataFrame:
    # 1) Data
    X = make_blobs(n_total, d=d, n_centers=50, noise_std=0.35, seed=seed)
    Q = X[:n_queries]
    DB = X[n_queries:]
    n_db = DB.shape[0]

    # 2) Ground truth (exact)
    print("[GT] Computing brute-force ground truth Top-10 (cosine)...")
    gt = brute_force_topk_cosine(Q, DB, topk=topk, batch=gt_batch)
    gt_sets: List[Set[int]] = [set(row.tolist()) for row in gt]

    rows: List[Dict[str, Any]] = []

    # 3) K sweep
    for K in K_list:
        print(f"\n=== K = {K} ===")

        # SRM constraint: m (multi-probe count) cannot exceed K.
        # You requested m=8 fixed; for K < 8 we must clamp to m_eff=K to run the sweep.
        m_eff = int(min(m, K))
        if m_eff != int(m):
            print(f"[warn] K={K} < requested m={m}; using m_eff={m_eff} for this K only.")

        cfg = SRMConfig(
            d=d,
            K=int(K),
            m=int(m_eff),
            top_k=int(topk),
            max_candidates=int(max_candidates),
            rerank_metric="cosine",
            pre_normalize=True,
            store_item_embeddings=True,
            embeddings_dtype="float16",
            store_payloads=False,
            seed=int(seed),
        )

        srm = SemanticRoutingMemory(cfg)

        # Fit on DB (optionally subsampled for speed)
        X_fit = DB
        if FIT_SUBSAMPLE is not None and FIT_SUBSAMPLE < n_db:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_db, size=int(FIT_SUBSAMPLE), replace=False)
            X_fit = DB[idx]
            print(f"[fit] Using subsample for codebook fit: {X_fit.shape[0]} / {n_db}")

        t0 = time.perf_counter()
        srm.fit_codebook(X_fit, n_iter=n_iter_kmeans, init="kmeans++", verbose=False)
        t_fit = time.perf_counter() - t0
        print(f"[fit] codebook fit time: {t_fit:.3f} s")

        # Add items (ids = 0..n_db-1)
        t0 = time.perf_counter()
        srm.add_items(DB, ids=list(range(n_db)))
        t_add = time.perf_counter() - t0
        print(f"[add] add_items time: {t_add:.3f} s")

        # 3a) Routing storage size: _buckets + _ids
        storage_buckets = deep_getsizeof(srm._buckets)
        storage_ids = deep_getsizeof(srm._ids)
        routing_storage_bytes = storage_buckets + storage_ids

        # 3b) Theoretical compression ratio vs dense float32 vectors
        dense_bytes_total = n_db * d * 4  # float32
        code_bytes = code_bytes_per_item(K)
        theoretical_ratio = (d * 4) / code_bytes  # per item (routing code only)

        # 3c) Query & Recall
        # Warmup (optional small)
        for i in range(min(10, n_queries)):
            _ = srm.query(Q[i], top_k=topk, m=m_eff, max_candidates=max_candidates)

        t0 = time.perf_counter()
        hit_sum = 0.0
        cand_sum = 0.0

        for i in range(n_queries):
            res = srm.query(Q[i], top_k=topk, m=m_eff, max_candidates=max_candidates)
            retrieved = res["ids"]
            cand_sum += float(res.get("n_candidates", 0))
            # Recall@10 per query
            hit = len(set(retrieved) & gt_sets[i])
            hit_sum += hit / float(topk)

        elapsed = time.perf_counter() - t0
        mean_latency_ms = (elapsed / n_queries) * 1000.0
        recall_at_10 = hit_sum / n_queries
        mean_candidates = cand_sum / n_queries

        rows.append(
            {
                "K": int(K),
                "m_used": int(m_eff),
                "Recall@10": float(recall_at_10),
                "QueryLatency_ms": float(mean_latency_ms),
                "RoutingStorage_bytes": int(routing_storage_bytes),
                "Buckets_bytes": int(storage_buckets),
                "IDs_bytes": int(storage_ids),
                "CodeBytes_per_item": int(code_bytes),
                "TheoreticalCompression_x": float(theoretical_ratio),
                "MeanCandidates": float(mean_candidates),
                "FitTime_s": float(t_fit),
                "AddTime_s": float(t_add),
            }
        )

        print(
            f"[metrics] m_used={m_eff} | Recall@10={recall_at_10:.4f} | "
            f"Latency={mean_latency_ms:.3f} ms | "
            f"RoutingStorage={routing_storage_bytes:,} B | "
            f"TheoryCompression={theoretical_ratio:.1f}x | "
            f"MeanCandidates={mean_candidates:.1f}"
        )

    df = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)

    # Add an "EffectiveCompression" column (optional but useful):
    # compares total dense float32 bytes (DB only) vs measured routing bytes.
    dense_bytes_total = (n_total - n_queries) * d * 4
    df["EffectiveCompression_x"] = dense_bytes_total / df["RoutingStorage_bytes"].clip(lower=1)

    return df


def plot_results(df: pd.DataFrame) -> None:
    K = df["K"].to_numpy()

    # 1) Recall@10 vs K
    plt.figure()
    plt.plot(K, df["Recall@10"].to_numpy(), marker="o")
    plt.xscale("log")
    plt.xlabel("K (log scale)")
    plt.ylabel("Recall@10")
    plt.title("K-Sweep: Recall@10 vs K")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 2) Storage vs K
    plt.figure()
    plt.plot(K, df["RoutingStorage_bytes"].to_numpy(), marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("K (log scale)")
    plt.ylabel("Routing Storage Size (bytes)  [buckets + ids]")
    plt.title("K-Sweep: Routing Storage vs K")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 3) Compression ratio vs K (theoretical)
    plt.figure()
    plt.plot(K, df["TheoreticalCompression_x"].to_numpy(), marker="o")
    plt.xscale("log")
    plt.xlabel("K (log scale)")
    plt.ylabel("Compression Ratio (x)  [theoretical]")
    plt.title("K-Sweep: Theoretical Compression vs K")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()


if __name__ == "__main__":
    K_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    df = run_k_sweep(
        K_list=K_LIST,
        n_total=10_000,
        n_queries=1_000,
        d=768,
        topk=10,
        m=8,
        max_candidates=5_000,
        seed=0,
        n_iter_kmeans=15,
        FIT_SUBSAMPLE=None,  # set to e.g. 6000 if codebook fit is too slow
        gt_batch=100,
    )

    # Print results
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print("\n===== Results =====")
    print(df)

    # Plot
    plot_results(df)
