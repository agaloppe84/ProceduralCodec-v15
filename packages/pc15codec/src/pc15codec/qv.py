from __future__ import annotations
from typing import Tuple
import numpy as np


__all__ = [
    "train_qv_numpy",
    "quantize_params",
    "dequantize_params",
    "serialize_codebook",
    "deserialize_codebook",
]


def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Initialisation kmeans++ déterministe (seed du rng fourni)."""
    n, d = X.shape
    C = np.empty((k, d), dtype=X.dtype)
    # 1) premier centre au hasard
    i0 = rng.integers(0, n)
    C[0] = X[i0]
    # 2) suivants via D^2 sampling
    dists = np.full(n, np.inf, dtype=X.dtype)
    for c in range(1, k):
        # distance au centre le plus proche déjà choisi
        d2 = np.sum((X - C[c-1])**2, axis=1)
        dists = np.minimum(dists, d2)
        probs = dists / dists.sum() if dists.sum() > 0 else np.full(n, 1.0/n)
        idx = rng.choice(n, p=probs)
        C[c] = X[idx]
    return C


def train_qv_numpy(
    data: np.ndarray,
    k: int,
    seed: int = 1234,
    iters: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Entraîne un codebook QV (k x d) via kmeans (Lloyd) simple.
    - data: (N, d) float32/float64
    - k: nombre de vecteurs de code
    - seed: déterminisme complet
    """
    X = np.asarray(data, dtype=np.float32)
    assert X.ndim == 2 and X.shape[0] >= k, "data doit être (N,d) et N>=k"
    n, d = X.shape
    rng = np.random.default_rng(int(seed))
    C = _kmeans_pp_init(X, k, rng)

    prev_inertia = np.inf
    for _ in range(max(1, iters)):
        # Assign
        # (X - C[j])^2 = ||X||^2 + ||C||^2 - 2 X·C
        XX = np.sum(X * X, axis=1, keepdims=True)          # (N,1)
        CC = np.sum(C * C, axis=1, keepdims=True).T        # (1,k)
        d2 = XX + CC - 2.0 * (X @ C.T)                     # (N,k)
        labels = np.argmin(d2, axis=1)

        # Update
        C_new = np.zeros_like(C)
        counts = np.bincount(labels, minlength=k).astype(np.int64)
        for j in range(k):
            if counts[j] > 0:
                C_new[j] = X[labels == j].mean(axis=0)
            else:
                # ré-échantillonne un centre si cluster vide (rare)
                C_new[j] = X[rng.integers(0, n)]
        C = C_new

        # Inertie / convergence
        inertia = float(np.sum((X - C[labels])**2))
        if abs(prev_inertia - inertia) <= tol * max(1.0, inertia):
            break
        prev_inertia = inertia

    return C.astype(np.float32, copy=False)


def quantize_params(
    vec: np.ndarray,
    codebook: np.ndarray,
    scales: np.ndarray | float = 1.0,
) -> Tuple[int, list[int]]:
    """
    Choisit l’entrée de codebook la plus proche (L2) et code les offsets signés
    en entiers bornés [-128,127] avec un pas `scales` (par dimension).
      offsets = clip(round((vec - code) / scales), -128..127)
    """
    v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    C = np.asarray(codebook, dtype=np.float32)
    assert v.shape[1] == C.shape[1], "dimension mismatch"
    # NN (L2)
    j = int(np.argmin(np.sum((C - v)**2, axis=1)))
    base = C[j]
    s = np.asarray(scales, dtype=np.float32)
    if s.ndim == 0:
        s = np.full_like(base, float(s))
    assert s.shape == base.shape, "scales must be scalar or match vec dim"

    raw = (v[0] - base) / np.where(s != 0, s, 1.0)
    offs = np.clip(np.rint(raw), -128, 127).astype(np.int32).tolist()
    return j, offs


def dequantize_params(
    idx: int,
    offsets: list[int],
    codebook: np.ndarray,
    scales: np.ndarray | float = 1.0,
) -> np.ndarray:
    """
    Reconstitue une approx: codebook[idx] + offsets * scales (avec broadcast).
    """
    C = np.asarray(codebook, dtype=np.float32)
    assert 0 <= idx < C.shape[0], "idx out of range"
    base = C[int(idx)]
    s = np.asarray(scales, dtype=np.float32)
    if s.ndim == 0:
        s = np.full_like(base, float(s))
    assert len(offsets) == base.shape[0], "offsets dim mismatch"
    offs = np.asarray(offsets, dtype=np.float32)
    return (base + offs * s).astype(np.float32, copy=False)


def serialize_codebook(codebook: np.ndarray) -> bytes:
    """
    Sérialisation simple binaire: u32 k, u32 d, puis k*d float32 row-major.
    """
    C = np.asarray(codebook, dtype=np.float32)
    k, d = C.shape
    header = np.array([k, d], dtype=np.uint32).tobytes()
    body = C.tobytes(order="C")
    return header + body


def deserialize_codebook(data: bytes) -> np.ndarray:
    """Inverse de serialize_codebook."""
    import struct
    if len(data) < 8:
        raise ValueError("codebook payload too short")
    k, d = struct.unpack_from("<II", data, 0)
    expected = 8 + 4 * k * d
    if len(data) != expected:
        raise ValueError("codebook payload size mismatch")
    arr = np.frombuffer(data, dtype=np.float32, offset=8, count=k * d)
    return arr.reshape((k, d)).copy()
