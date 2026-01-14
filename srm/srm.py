# srm.py
# Semantic Routing Memory (SRM)

from __future__ import annotations

import os
import json
import pickle
import numpy as np
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# -----------------------------
# Utilities
# -----------------------------
def _as_float_array(x: ArrayLike, dim: int = 2, dtype=np.float32) -> np.ndarray:
    """Safe converter to float array with dimension check."""
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim != dim:
        raise ValueError(f"Expected {dim}D array, got shape={arr.shape}")
    return arr


def _l2_normalize_inplace(X: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """True in-place L2 normalization."""
    norms = np.linalg.norm(X, axis=axis, keepdims=True)
    np.maximum(norms, eps, out=norms)
    X /= norms
    return X


def _pairwise_l2_dist2(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared L2 distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b"""
    X_sq = np.sum(np.square(X), axis=1, keepdims=True)         # (N,1)
    C_sq = np.sum(np.square(C), axis=1, keepdims=True).T       # (1,K)
    dist2 = X_sq + C_sq - (2.0 * (X @ C.T))                    # (N,K)
    np.maximum(dist2, 0.0, out=dist2)
    return dist2


def kmeans(
    X: np.ndarray,
    K: int,
    n_iter: int = 25,
    seed: int = 0,
    init: Literal["kmeans++", "random"] = "kmeans++",
    verbose: bool = False,
) -> np.ndarray:
    """Dependency-free optimized K-Means (KMeans++ or random init)."""
    if K <= 1:
        raise ValueError("K must be >= 2")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    N, d = X.shape
    if K > N:
        raise ValueError(f"K ({K}) cannot exceed N ({N})")

    rng = np.random.default_rng(seed)

    if init == "kmeans++":
        centers = np.empty((K, d), dtype=X.dtype)
        idx0 = int(rng.integers(0, N))
        centers[0] = X[idx0]
        dist2 = np.sum((X - centers[0]) ** 2, axis=1)
        for k in range(1, K):
            probs = dist2 / (np.sum(dist2) + 1e-12)
            idx = int(rng.choice(N, p=probs))
            centers[k] = X[idx]
            new_dist2 = np.sum((X - centers[k]) ** 2, axis=1)
            np.minimum(dist2, new_dist2, out=dist2)
        C = centers
    elif init == "random":
        idx = rng.choice(N, size=K, replace=False)
        C = X[idx].copy()
    else:
        # No silent fallback in production
        raise ValueError("init must be 'kmeans++' or 'random'")

    newC = np.zeros_like(C)
    counts = np.zeros((K,), dtype=np.int64)

    for it in range(n_iter):
        dist2 = _pairwise_l2_dist2(X, C)
        codes = np.argmin(dist2, axis=1)

        newC.fill(0)
        counts.fill(0)

        np.add.at(newC, codes, X)
        np.add.at(counts, codes, 1)

        empty_mask = counts == 0
        n_empty = int(np.count_nonzero(empty_mask))
        if n_empty > 0:
            repl_idx = rng.choice(N, size=n_empty, replace=False)
            newC[empty_mask] = X[repl_idx]
            counts[empty_mask] = 1

        # Average only non-empty
        nonempty = ~empty_mask
        newC[nonempty] /= counts[nonempty, None]

        shift = float(np.mean(np.linalg.norm(newC - C, axis=1)))
        C[:] = newC

        if verbose:
            print(f"[kmeans] iter={it+1} shift={shift:.6f}")
        if shift < 1e-5:
            break

    return C


# -----------------------------
# SRM Config / State
# -----------------------------
@dataclass(slots=True)
class SRMConfig:
    d: int
    K: int = 256
    m: int = 8
    top_k: int = 10
    max_candidates: int = 5000

    quantize_metric: Literal["l2"] = "l2"

    # Reranking behavior:
    # "cosine" -> normalize query, cosine via dot
    # "dot"    -> raw query dot (magnitude matters)
    rerank_metric: Literal["cosine", "dot"] = "cosine"

    # Storage/Routing behavior:
    # If True, items are L2 normalized before quantization & storage
    pre_normalize: bool = True

    store_item_embeddings: bool = True
    embeddings_dtype: Literal["float16", "float32"] = "float16"
    store_payloads: bool = True

    adaptive_update: bool = False
    tau: float = 0.85
    eta: float = 0.1

    seed: int = 0
    initial_capacity: int = 1024


class SemanticRoutingMemory:
    __slots__ = (
        "cfg",
        "codebook",
        "_buckets",
        "_ids",
        "_id_to_index",
        "_emb",
        "_emb_capacity",
        "_emb_cursor",
        "_payloads",
        "_rng",
    )

    def __init__(self, config: SRMConfig):
        self.cfg = config

        # Strict validations
        if config.d <= 0:
            raise ValueError("d must be > 0")
        if config.K < 2:
            raise ValueError("K must be >= 2")
        if config.m <= 0:
            raise ValueError("m must be > 0")
        if config.m > config.K:
            raise ValueError(f"m ({config.m}) cannot exceed K ({config.K})")
        if config.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if config.max_candidates <= 0:
            raise ValueError("max_candidates must be > 0")
        if config.top_k > config.max_candidates:
            raise ValueError(
                f"top_k ({config.top_k}) cannot be larger than max_candidates ({config.max_candidates})"
            )
        if config.embeddings_dtype not in ("float16", "float32"):
            raise ValueError("embeddings_dtype must be 'float16' or 'float32'")
        if not (0.0 < config.eta <= 1.0):
            raise ValueError("eta must be in (0, 1]")
        if config.tau < 0:
            raise ValueError("tau must be >= 0")
        if config.initial_capacity <= 0:
            raise ValueError("initial_capacity must be > 0")

        self.codebook: Optional[np.ndarray] = None  # (K, d)
        self._buckets: List[List[int]] = [[] for _ in range(config.K)]
        self._ids: List[Any] = []
        self._id_to_index: Dict[Any, int] = {}

        # Embeddings dynamic array
        self._emb: Optional[np.ndarray] = None
        self._emb_capacity: int = 0
        self._emb_cursor: int = 0

        self._payloads: Optional[List[Any]] = [] if config.store_payloads else None
        self._rng = np.random.default_rng(config.seed)

    # -------------
    # Internal Storage Management
    # -------------
    def _ensure_capacity(self, required_add: int) -> None:
        """Doubles capacity if needed. Amortized O(1)."""
        if not self.cfg.store_item_embeddings:
            return

        needed = self._emb_cursor + required_add

        if self._emb is None:
            new_cap = max(self.cfg.initial_capacity, needed)
            dt = np.float16 if self.cfg.embeddings_dtype == "float16" else np.float32
            self._emb = np.zeros((new_cap, self.cfg.d), dtype=dt)
            self._emb_capacity = new_cap
            return

        if needed > self._emb_capacity:
            new_cap = max(self._emb_capacity * 2, needed)
            dt = self._emb.dtype
            new_emb = np.zeros((new_cap, self.cfg.d), dtype=dt)
            new_emb[: self._emb_cursor] = self._emb[: self._emb_cursor]
            self._emb = new_emb
            self._emb_capacity = new_cap

    def _check_embedding_consistency(self) -> None:
        """
        Ensures embeddings storage aligns with ids.
        If embeddings are enabled, we expect cursor == number of ids.
        """
        if not self.cfg.store_item_embeddings:
            return
        if self._emb is None:
            # Allowed only when empty
            if len(self._ids) != 0:
                raise RuntimeError("Embeddings enabled but _emb is None while ids exist.")
            return
        if self._emb_cursor != len(self._ids):
            raise RuntimeError(
                f"Embedding store inconsistent: _emb_cursor={self._emb_cursor} "
                f"but len(_ids)={len(self._ids)}"
            )

    # -------------
    # Codebook
    # -------------
    def fit_codebook(
        self,
        X_train: ArrayLike,
        n_iter: int = 25,
        init: Literal["kmeans++", "random"] = "kmeans++",
        verbose: bool = False,
    ) -> "SemanticRoutingMemory":
        X = _as_float_array(X_train, dim=2)
        if X.shape[1] != self.cfg.d:
            raise ValueError(f"Dim mismatch: {X.shape[1]} vs {self.cfg.d}")

        if self.cfg.pre_normalize:
            X = _l2_normalize_inplace(X.copy())

        self.codebook = kmeans(
            X, self.cfg.K, n_iter=n_iter, seed=self.cfg.seed, init=init, verbose=verbose
        )

        if self.cfg.pre_normalize:
            _l2_normalize_inplace(self.codebook)

        self._buckets = [[] for _ in range(self.cfg.K)]
        return self

    # -------------
    # Build / Add
    # -------------
    def add_items(
        self,
        X_items: ArrayLike,
        ids: Optional[Sequence[Any]] = None,
        payloads: Optional[Sequence[Any]] = None,
        update_codebook_adaptively: Optional[bool] = None,
    ) -> List[int]:
        if self.codebook is None:
            raise RuntimeError("Codebook not fitted.")

        X = _as_float_array(X_items, dim=2)
        n = int(X.shape[0])
        if X.shape[1] != self.cfg.d:
            raise ValueError(f"Dim mismatch: {X.shape[1]} vs {self.cfg.d}")

        start_idx = len(self._ids)  # corresponds to embedding cursor in normal use
        if ids is None:
            ids = list(range(start_idx, start_idx + n))
        else:
            if len(ids) != n:
                raise ValueError(f"IDs length mismatch: {len(ids)} vs {n}")
            for _id in ids:
                if _id in self._id_to_index:
                    raise ValueError(f"Duplicate ID: {_id}")

        if self.cfg.store_payloads:
            if payloads is None:
                batch_payloads = [None] * n
            else:
                if len(payloads) != n:
                    raise ValueError("Payloads length mismatch")
                batch_payloads = list(payloads)

        # Normalize items for routing/storage if configured
        if self.cfg.pre_normalize:
            X_route = _l2_normalize_inplace(X.copy())
        else:
            X_route = X

        codes, err = self._quantize_internal(X_route)

        new_indices = list(range(start_idx, start_idx + n))
        self._ids.extend(ids)

        for i, idx in enumerate(new_indices):
            self._id_to_index[ids[i]] = idx
            self._buckets[int(codes[i])].append(idx)

        if self.cfg.store_payloads and self._payloads is not None:
            self._payloads.extend(batch_payloads)

        if self.cfg.store_item_embeddings:
            self._ensure_capacity(n)
            dt = np.float16 if self.cfg.embeddings_dtype == "float16" else np.float32
            X_store = X_route.astype(dt, copy=False)
            self._emb[self._emb_cursor : self._emb_cursor + n] = X_store
            self._emb_cursor += n

        do_update = self.cfg.adaptive_update if update_codebook_adaptively is None else update_codebook_adaptively
        if do_update:
            self._adaptive_update_safe(X_route, codes, err)

        return new_indices

    # -------------
    # Quantize / Update Logic
    # -------------
    def _quantize_internal(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.cfg.quantize_metric != "l2":
            raise ValueError("Only quantize_metric='l2' is supported.")
        dist2 = _pairwise_l2_dist2(X, self.codebook)
        codes = np.argmin(dist2, axis=1)
        mins = dist2[np.arange(X.shape[0]), codes]
        return codes.astype(np.int64, copy=False), mins.astype(np.float32, copy=False)

    def _adaptive_update_safe(self, X: np.ndarray, codes: np.ndarray, err: np.ndarray) -> None:
        """Mean-based drift update to prevent overshoot."""
        tau = float(self.cfg.tau)
        eta = float(self.cfg.eta)

        mask = err > tau
        if not np.any(mask):
            return

        X_drift = X[mask]
        codes_drift = codes[mask]

        K, d = self.codebook.shape
        sum_x = np.zeros((K, d), dtype=np.float32)
        cnt = np.zeros((K,), dtype=np.int64)

        np.add.at(sum_x, codes_drift, X_drift)
        np.add.at(cnt, codes_drift, 1)

        nonzero = cnt > 0
        mean_drift = sum_x[nonzero] / cnt[nonzero, None]
        current = self.codebook[nonzero]
        self.codebook[nonzero] = current + eta * (mean_drift - current)

        if self.cfg.pre_normalize:
            _l2_normalize_inplace(self.codebook[nonzero])

    # -------------
    # Query
    # -------------
    def query(
        self,
        xq: ArrayLike,
        top_k: Optional[int] = None,
        m: Optional[int] = None,
        max_candidates: Optional[int] = None,
        return_payloads: bool = False,
        return_embeddings: bool = False,
    ) -> Dict[str, Any]:
        if self.codebook is None:
            return self._empty_result(return_payloads, return_embeddings)

        if top_k is None:
            top_k = self.cfg.top_k
        if m is None:
            m = self.cfg.m
        if max_candidates is None:
            max_candidates = self.cfg.max_candidates

        # Cache query vectors
        xq_raw = np.asarray(xq, dtype=np.float32)
        if xq_raw.ndim != 1:
            raise ValueError("Query must be 1D vector")
        if xq_raw.shape[0] != self.cfg.d:
            raise ValueError(f"Dim mismatch: {xq_raw.shape[0]} vs {self.cfg.d}")

        # One normalized version for cosine-style computations
        xq_unit = xq_raw / (np.linalg.norm(xq_raw) + 1e-12)

        # Routing vector depends on training/storage normalization
        xq_route = xq_unit if self.cfg.pre_normalize else xq_raw

        # 1) Multi-probe routing (deterministic)
        if self.cfg.pre_normalize:
            scores_c = self.codebook @ xq_route  # cosine proxy
            if m >= self.cfg.K:
                probed = np.arange(self.cfg.K)
                probed = probed[np.argsort(-scores_c)]
            else:
                unsorted = np.argpartition(scores_c, -m)[-m:]
                probed = unsorted[np.argsort(-scores_c[unsorted])]
        else:
            dist2 = _pairwise_l2_dist2(xq_route[None, :], self.codebook).ravel()
            if m >= self.cfg.K:
                probed = np.arange(self.cfg.K)
                probed = probed[np.argsort(dist2)]
            else:
                unsorted = np.argpartition(dist2, m - 1)[:m]
                probed = unsorted[np.argsort(dist2[unsorted])]

        # 2) Gather candidates from all probed buckets (no early-break)
        candidates: List[int] = []
        for c_idx in probed.tolist():
            candidates.extend(self._buckets[int(c_idx)])

        if not candidates:
            return self._empty_result(return_payloads, return_embeddings, probed)

        # Truncate after gathering (deterministic order)
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        # 3) Rerank
        if not self.cfg.store_item_embeddings or self._emb is None:
            final_idxs = candidates[:top_k]
            final_scores = [0.0] * len(final_idxs)
        else:
            # Ensure embeddings store is consistent
            self._check_embedding_consistency()

            cand_indices = np.asarray(candidates, dtype=np.int64)
            if int(cand_indices.max()) >= self._emb_cursor:
                raise RuntimeError(
                    "Candidate index beyond embedding cursor. "
                    f"max_idx={int(cand_indices.max())}, _emb_cursor={self._emb_cursor}"
                )

            X_cand = self._emb[cand_indices].astype(np.float32, copy=False)

            if self.cfg.rerank_metric == "cosine":
                xq_rerank = xq_unit
                if self.cfg.pre_normalize:
                    scores = X_cand @ xq_rerank
                else:
                    _l2_normalize_inplace(X_cand)
                    scores = X_cand @ xq_rerank
            else:  # "dot"
                scores = X_cand @ xq_raw

            k_safe = min(top_k, len(scores))
            if k_safe < len(scores):
                top_local = np.argpartition(scores, -k_safe)[-k_safe:]
                top_local = top_local[np.argsort(-scores[top_local])]
            else:
                top_local = np.argsort(-scores)

            final_idxs = cand_indices[top_local].tolist()
            final_scores = scores[top_local].tolist()

        out = {
            "ids": [self._ids[i] for i in final_idxs],
            "scores": final_scores,
            "indices": final_idxs,
            "probed_centroids": probed.tolist(),
            "n_candidates": len(candidates),
        }

        if return_payloads:
            out["payloads"] = (
                [self._payloads[i] for i in final_idxs] if (self._payloads is not None) else []
            )
        else:
            out["payloads"] = None

        if return_embeddings:
            if self._emb is None:
                out["embeddings"] = []
            else:
                out["embeddings"] = [self._emb[i] for i in final_idxs]
        else:
            out["embeddings"] = None

        return out

    def _empty_result(self, payloads: bool = False, embeddings: bool = False, probed=None) -> Dict[str, Any]:
        return {
            "ids": [],
            "scores": [],
            "indices": [],
            "probed_centroids": probed.tolist() if probed is not None else [],
            "n_candidates": 0,
            "payloads": [] if payloads else None,
            "embeddings": [] if embeddings else None,
        }

    # -------------
    # Persistence
    # -------------
    def save(self, path: str) -> None:
        """Saves SRM state. Trims unused embedding capacity to save disk space."""
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)

        state = {
            "codebook": self.codebook,
            "ids": self._ids,
            "buckets": self._buckets,
            "payloads": self._payloads,
            "emb_cursor": self._emb_cursor,
        }
        with open(os.path.join(path, "routing.pkl"), "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.cfg.store_item_embeddings and self._emb is not None:
            trimmed = self._emb[: self._emb_cursor]
            np.save(os.path.join(path, "embeddings.npy"), trimmed)

    @classmethod
    def load(cls, path: str) -> "SemanticRoutingMemory":
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            cfg = SRMConfig(**json.load(f))

        srm = cls(cfg)

        with open(os.path.join(path, "routing.pkl"), "rb") as f:
            state = pickle.load(f)

        srm.codebook = state["codebook"]
        srm._ids = state["ids"]
        srm._buckets = state["buckets"]
        srm._payloads = state["payloads"]

        srm._emb_cursor = int(state.get("emb_cursor", len(srm._ids)))
        srm._id_to_index = {uid: i for i, uid in enumerate(srm._ids)}

        emb_path = os.path.join(path, "embeddings.npy")
        if cfg.store_item_embeddings and os.path.exists(emb_path):
            loaded_emb = np.load(emb_path, allow_pickle=False)
            srm._emb = loaded_emb
            srm._emb_capacity = int(loaded_emb.shape[0])
            # Clamp cursor if needed
            if srm._emb_cursor > srm._emb_capacity:
                srm._emb_cursor = srm._emb_capacity

        # Final sanity check (if embeddings enabled and present)
        if cfg.store_item_embeddings and srm._emb is not None:
            if srm._emb_cursor != len(srm._ids):
                # Prefer to fail fast (production safety)
                raise RuntimeError(
                    f"Loaded SRM is inconsistent: emb_cursor={srm._emb_cursor} "
                    f"but len(ids)={len(srm._ids)}"
                )

        return srm