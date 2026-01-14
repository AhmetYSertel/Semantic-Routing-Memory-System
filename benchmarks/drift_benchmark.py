import time
import numpy as np
from srm import SemanticRoutingMemory, SRMConfig


def generate_adversarial_data(n_samples=5000, d=384, seed=42):
    """
    Creates 'Clumped' data designed to cause bucket overflows.
    Instead of uniform topics, we create SUPER DENSE clusters.
    """
    rng = np.random.default_rng(seed)
    # Fewer topics, higher density -> Force collisions
    n_topics = 20
    centers = rng.standard_normal((n_topics, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    X = np.zeros((n_samples, d), dtype=np.float32)
    for i in range(n_samples):
        topic_idx = rng.integers(0, n_topics)
        c = centers[topic_idx]
        # Very small noise = Very dense clusters
        noise = rng.standard_normal(d).astype(np.float32) * 0.02
        vec = c + noise
        vec /= np.linalg.norm(vec)
        X[i] = vec
    return X


def analyze_srm(srm, name, test_data):
    # Quantization Error
    _, dists = srm._quantize_internal(test_data)
    mse = np.mean(dists)

    # Bucket Stats
    sizes = np.array([len(b) for b in srm._buckets])
    max_bucket = int(np.max(sizes))
    # Standard Deviation of bucket sizes (Lower is better/more balanced)
    std_bucket = np.std(sizes)

    # Recall (Very strict limit)
    hits = 0
    # Test on a subset
    queries = test_data[:1000]

    for xq in queries:
        # Use return_embeddings=False for speed, just check ID inclusion logic
        # Here we simulate: "Did we find the exact item?"
        # Since we just added them, they MUST be retrievable.
        res = srm.query(xq, top_k=10)

        # Check if the query vector itself is in the retrieved candidates?
        # Since exact vector match is float-sensitive, we trust the retrieval logic.
        # If the candidate set includes the item's ID (we cheat a bit here by knowing logic),
        # but simpler: Let's assume if score > 0.99 it's a hit.
        if res["scores"] and max(res["scores"]) > 0.98:
            hits += 1

    recall = hits / len(queries)

    print(f"--- {name} ---")
    print(f"  Recall (Strict):          {recall:.2%}")
    print(f"  Quantization Error (MSE): {mse:.4f}")
    print(f"  Bucket Max Size:          {max_bucket}")
    print(f"  Bucket Load StdDev:       {std_bucket:.2f} (Lower = Better Load Balancing)")
    return recall


def main():
    print(f"{'=' * 50}")
    print(f"SRM Stress Test v4: The Capacity Crunch")
    print(f"{'=' * 50}\n")

    D = 384
    # EXTREME CONSTRAINT: We can only afford to check 50 items per query.
    # This simulates a massive system where reading memory is expensive.
    MAX_CANDIDATES = 50
    N_TRAIN = 5000
    N_DRIFT = 5000

    # K=128. With 10,000 items, avg bucket size should be ~78.
    # But since MAX_CANDIDATES=50, if a bucket goes over 50, we start losing recall!
    K = 128

    print("1. Generating Initial World (Phase A)...")
    X_train = generate_adversarial_data(n_samples=N_TRAIN, d=D, seed=1)

    print("2. Generating New World (Phase B - Drift)...")
    # Different seed = Different cluster centers
    X_drift = generate_adversarial_data(n_samples=N_DRIFT, d=D, seed=99)

    # ---------------------------------------------------------
    # STATIC
    # ---------------------------------------------------------
    print("\n[Running STATIC Scenario]...")
    cfg_static = SRMConfig(d=D, K=K, max_candidates=MAX_CANDIDATES, adaptive_update=False)
    srm_static = SemanticRoutingMemory(cfg_static)
    srm_static.fit_codebook(X_train)
    srm_static.add_items(X_train)
    srm_static.add_items(X_drift)  # Add drift data to static bins

    r_static = analyze_srm(srm_static, "STATIC", X_drift)

    # ---------------------------------------------------------
    # ADAPTIVE
    # ---------------------------------------------------------
    print("\n[Running ADAPTIVE Scenario]...")
    # Low eta (0.05) prevents "Black Hole" clumping
    # High tau (0.75) ensures we only move empty/useless centroids, not the good ones
    cfg_adapt = SRMConfig(
        d=D, K=K, max_candidates=MAX_CANDIDATES,
        adaptive_update=True,
        tau=0.75,
        eta=0.05
    )
    srm_adapt = SemanticRoutingMemory(cfg_adapt)
    srm_adapt.fit_codebook(X_train)
    srm_adapt.add_items(X_train)
    srm_adapt.add_items(X_drift)  # Centroids should gently migrate to new dense areas

    r_adapt = analyze_srm(srm_adapt, "ADAPTIVE", X_drift)

    print("\n" + "=" * 30)
    print(f"Improvement: {r_adapt - r_static:+.2%}")
    print("=" * 30)

    if r_adapt > r_static:
        print("SUCCESS: Adaptive logic handled the load better.")
    else:
        print("INSIGHT: Static routing is surprisingly robust.")


if __name__ == "__main__":
    main()