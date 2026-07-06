"""
Stage 1 - contact graph + Stage 3 - Hamiltonian + distance grafowa.

Variants graph: cutoff (threshold A) | knn (tau-threshold, stale k neighbors)
Variants H: laplacian | adjacency | tight_binding
Distance grafowa: scipy.csgraph (BFS w C) - quickly nawet for N~2500.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from .backend import njit


def contact_graph_cutoff(coords, cutoff=8.0):
    n = len(coords)
    A = np.zeros((n, n), float)
    for i, j in cKDTree(coords).query_pairs(cutoff):
        A[i, j] = A[j, i] = 1.0
    return A


@njit(cache=True)
def _fill_knn(A, idx):
    n = idx.shape[0]
    for i in range(n):
        for t in range(1, idx.shape[1]):
            j = idx[i, t]
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


def contact_graph_knn(coords, k=10):
    """tau-threshold: each node dostaje k najblizszych neighbors (symetryzacja).
    Wypelnienie matrix via kernel Numba."""
    n = len(coords)
    A = np.zeros((n, n), float)
    kk = min(k + 1, n)
    _, idx = cKDTree(coords).query(coords, k=kk)
    return _fill_knn(A, np.ascontiguousarray(idx))


def build_graph(coords, variant="cutoff", **kw):
    if variant == "cutoff":
        return contact_graph_cutoff(coords, kw.get("cutoff", 8.0))
    if variant == "knn":
        return contact_graph_knn(coords, kw.get("k", 10))
    raise ValueError(f"nieznany variant graph: {variant}")


def Hamiltonian(A, variant="laplacian", **kw):
    if variant == "laplacian":
        return np.diag(A.sum(1)) - A
    if variant == "adjacency":
        return A.copy()
    if variant == "tight_binding":
        w = kw.get("hop", 1.0)
        onsite = kw.get("onsite", A.sum(1))
        return np.diag(np.asarray(onsite, float)) - w * A
    if variant == "logf":
        # LST rown. 18: significance ~ exp(-(log f_i - log f_j)^2 / 2 sigma^2),
        # f_i = czestotliwosc node (proxy: lokalna gestosc kontaktow = degree).
        # Residues o podobnej "sztywnosci" rezonuja silniej.
        sigma = kw.get("sigma", 1.0)
        f = np.log(A.sum(1) + 1.0)
        D = np.abs(f[:, None] - f[None, :])
        W = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
        Aw = A * W
        return np.diag(Aw.sum(1)) - Aw
    raise ValueError(f"nieznany variant Hamiltonianu: {variant}")


def logf_weights(A, sigma=1.0):
    """Matrix weights sviaenia log-f (LST rown. 18) - to features resonance."""
    f = np.log(A.sum(1) + 1.0)
    D = np.abs(f[:, None] - f[None, :])
    return A * np.exp(-(D ** 2) / (2.0 * sigma ** 2))


def graph_distance(A):
    """Distance grafowa (number steps), inf for par nieosiagalnych."""
    D = shortest_path(csr_matrix(A > 0), method="D",
                      unweighted=True, directed=False)
    return D # float z inf for rozlacznych
