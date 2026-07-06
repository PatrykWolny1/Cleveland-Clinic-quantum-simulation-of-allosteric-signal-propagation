"""
Operator spectral wprost z Matrix Data (not z cech).

INPUT = Laplacian structure apo L = D - A (matrix data).
Representation porownywalna between proteins roznej wielkosci = WIDMO L:
  udzial modalny P[i,b] = sum_{a: lambda_a in band b} |v_a(i)|^2
  (ile residue i "zyje" w band czestotliwosci b; lambda znormalizowana [0,1]).
Operator: P W = y -> W = P^+ y = spectral filter g(lambda) = PRAWO.
g(lambda) jest shared for proteins (these same band) -> generalizuje in 4. case.

Forma closed-form (ridge/pinv) -> quickly; one eig in protein (spectrum matrix data).
"""
from __future__ import annotations
import numpy as np
from . import graph, propagate
from .backend import to_numpy, get_backend

N_BINS = 20


def modal_participation(coords, n_bins=N_BINS, k=10, gpu=False):
    """P [N x n_bins] z spectrum Laplacianu apo (matrix data)."""
    A = graph.build_graph(coords, "knn", k=k)
    L = graph.Hamiltonian(A, "laplacian")
    w, V, _, _ = propagate.eig(L, gpu)
    w = to_numpy(w); V = to_numpy(V)
    ln = (w - w.min()) / (w.max() - w.min() + 1e-12) # znormalizowane band
    P = np.zeros((V.shape[0], n_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for b in range(n_bins):
        hi = ln <= edges[b + 1] if b == n_bins - 1 else ln < edges[b + 1]
        mask = (ln >= edges[b]) & hi
        if mask.any():
            P[:, b] = (V[:, mask] ** 2).sum(1)
    P = (P - P.mean(0)) / (P.std(0) + 1e-9)
    return P


def fit_operator(P, y, ridge=1.0):
    yc = y - y.mean()
    G = P.T @ P + ridge * np.eye(P.shape[1])
    return np.linalg.solve(G, P.T @ yc)


def apply_operator(P, W):
    return P @ W


def agreement(Ws):
    Ws = [w / (np.linalg.norm(w) + 1e-12) for w in Ws]
    n = len(Ws)
    return np.array([[float(Ws[i] @ Ws[j]) for j in range(n)] for i in range(n)])
