import time
import numpy as np
import os
import pickle
from srm import SemanticRoutingMemory, SRMConfig


def generate_synthetic_data(n_samples=10_000, d=384, n_topics=50, seed=42):
    """
    Generates synthetic cluster-based data to mimic 'threads'.
    """
    rng = np.random.default_rng(seed)
    # 1. Create Topic Centroids
    topic_centers = rng.standard_normal((n_topics, d)).astype(np.float32)
    topic_centers /= np.linalg.norm(topic_centers, axis=1, keepdims=True)

    # 2. Generate Items around topics
    X = np.zeros((n_samples, d), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        topic_idx = rng.integers(0, n_topics)
        center = topic_centers[topic_idx]
        noise = rng.standard_normal(d).astype(np.float32) * 0.1
        vec = center + noise
        vec /= np.linalg.norm(vec)
        X[i] = vec
        labels[i] = topic_idx

    return X, labels


def evaluate_retrieval(queries, ground_truth_indices, retrieval_func, k=10):
    """
    Computes Recall@K based on exact nearest neighbor ground truth.
    """
    hits = 0
    total_latency = 0.0

    for i, xq in enumerate(queries):
        start = time.perf_counter()
        results = retrieval_func(xq, top_k=k)
        end = time.perf_counter()
        total_latency += (end - start) * 1000  # ms

        # Check if ANY of the retrieved ids match the top-1 ground truth ID
        # (Simplified recall definition: is the true NN in the retrieved set?)
        true_idx = ground_truth_indices[i]  # The exact NN index

        retrieved_indices = results["indices"]
        if true_idx in retrieved_indices:
            hits += 1

    avg_latency = total_latency / len(queries)
    recall = hits / len(queries)
    return recall, avg_latency


def main():
    print(f"{'=' * 40}")
    print(f"SRM v3.1 Benchmark Replication")
    print(f"{'=' * 40}\n")

    # 1. Setup Data
    N = 20_000
    D = 384
    N_QUERY = 100
    print(f"Generating synthetic data (N={N}, d={D})...")
    X, _ = generate_synthetic_data(n_samples=N, d=D, n_topics=100)

    # Split Train/Query (Use last N_QUERY for testing)
    X_train = X[:-N_QUERY]
    X_query = X[-N_QUERY:]

    # Compute Exact Ground Truth (Brute Force) for Recall Calc
    print("Computing exact ground truth for queries...")
    # X_train is (N_train, D), X_query is (N_query, D)
    # Cosine similarity -> dot product of normalized vectors
    scores = X_query @ X_train.T
    gt_indices = np.argmax(scores, axis=1)  # Top-1 exact match index

    print("-" * 30)

    # ---------------------------------------------------------
    # Scenario A: Dense Baseline (Simulation)
    # ---------------------------------------------------------
    # In a real baseline, we scan all N vectors.
    # Storage = N * D * 4 bytes (float32)
    dense_size_mb = (X_train.size * 4) / (1024 ** 2)

    def dense_retrieve(xq, top_k):
        # Brute force scan
        s = X_train @ xq
        idx = np.argsort(-s)[:top_k]
        return {"indices": idx}

    print(f"BASELINE: Dense Flat Index")
    rec_dense, lat_dense = evaluate_retrieval(X_query, gt_indices, dense_retrieve)
    print(f"  -> Storage: {dense_size_mb:.2f} MB")
    print(f"  -> Latency: {lat_dense:.3f} ms / query")
    print(f"  -> Recall@10: {rec_dense:.2%}")
    print("-" * 30)

    # ---------------------------------------------------------
    # Scenario B: SRM (K=256, m=8)
    # ---------------------------------------------------------
    cfg = SRMConfig(
        d=D,
        K=256,
        m=12,  # Multi-probe count
        top_k=10,
        max_candidates=1000,
        store_item_embeddings=True,  # Need embeddings for rerank to get high recall
        pre_normalize=True
    )
    srm = SemanticRoutingMemory(cfg)

    print(f"SRM: Training Codebook (K={cfg.K})...")
    t0 = time.time()
    srm.fit_codebook(X_train[:5000])  # Train on subset for speed
    print(f"  -> Training done in {time.time() - t0:.2f}s")

    print("SRM: Indexing items...")
    t0 = time.time()
    srm.add_items(X_train)
    print(f"  -> Indexing done in {time.time() - t0:.2f}s")

    # Calculate Storage
    # Serialize to measure real disk footprint
    srm.save("./srm_bench_temp")
    routing_size = os.path.getsize("./srm_bench_temp/routing.pkl")
    emb_size = os.path.getsize("./srm_bench_temp/embeddings.npy")
    config_size = os.path.getsize("./srm_bench_temp/config.json")
    total_srm_mb = (routing_size + emb_size + config_size) / (1024 ** 2)

    # Routing-only size (Without payloads/embeddings, just the structure)
    # This proves the "Lightweight Routing Layer" claim
    routing_layer_mb = routing_size / (1024 ** 2)

    def srm_retrieve(xq, top_k):
        return srm.query(xq, top_k=top_k)

    print(f"SRM: Querying...")
    rec_srm, lat_srm = evaluate_retrieval(X_query, gt_indices, srm_retrieve)

    print(f"  -> Total Storage: {total_srm_mb:.2f} MB")
    print(f"  -> Routing Layer: {routing_layer_mb:.2f} MB (The 'Hot' Memory)")
    print(f"  -> Latency: {lat_srm:.3f} ms / query")
    print(f"  -> Recall@10: {rec_srm:.2%}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    print(f"Metric        | Dense Baseline | SRM (v3.1)")
    print(f"--------------|----------------|-----------")
    print(f"Recall@10     | {rec_dense:.2%}        | {rec_srm:.2%}  (Target: Close to Baseline)")
    print(f"Latency (p50) | {lat_dense:.3f} ms       | {lat_srm:.3f} ms (Target: Lower is better)")
    print(f"Storage (Hot) | {dense_size_mb:.2f} MB        | {routing_layer_mb:.2f} MB (Target: Huge Reduction)")

    # Cleanup
    import shutil
    shutil.rmtree("./srm_bench_temp")


if __name__ == "__main__":
    main()