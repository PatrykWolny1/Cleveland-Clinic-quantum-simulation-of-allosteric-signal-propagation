"""
EML - equation discovery (symbolic regression, SINDy style) from matrix data.

Per-residue descriptors derived from the apo Laplacian SPECTRUM (intrinsic to
the matrix data, not hand-crafted features): participation in the low/mid/high
bands, Fiedler component, effective mass, degree. Nonlinear library (squares,
pairwise products, higher-order terms). Sparse regression (STLSQ) -> compact
formula y ~ sum c_j term_j.

Multi-task: a single shared formula fitted to ALL proteins at once (pooled)
-> a generalizing LAW (leave-one-protein-out test). Per-protein -> 3 parameter
sets for comparison (is it one form with different parameterizations).

Closed form (least squares + thresholding) -> fast; MP/GPU for the descriptors.
"""
from __future__ import annotations
import numpy as np
from . import graph, propagate
from .backend import to_numpy


def descriptors(coords, gpu=False):
    """D [N x d] - scalars from the matrix-data spectrum + names."""
    A = graph.build_graph(coords, "knn", k=10)
    L = graph.Hamiltonian(A, "laplacian")
    w, V, _, _ = propagate.eig(L, gpu)
    w = to_numpy(w); V = to_numpy(V); N = V.shape[0]
    nz = w > 1e-9
    idx = np.where(nz)[0]
    lo = idx[: max(1, len(idx) // 5)] # lowest 20% of modes
    mi = idx[len(idx) // 5: 3 * len(idx) // 5]
    hi = idx[3 * len(idx) // 5:]
    p_low = (V[:, lo] ** 2).sum(1)
    p_mid = (V[:, mi] ** 2).sum(1)
    p_high = (V[:, hi] ** 2).sum(1)
    fied = np.abs(V[:, idx[0]]) if len(idx) else np.zeros(N) # Fiedler
    emass = to_numpy(propagate.effective_mass(w, V, np, 20.0))
    deg = A.sum(1)
    V2 = V ** 2
    # multi-scale heat signature HKS_t(i) = sum_a e^{-lambda_a t} v_a(i)^2
    hks = {t: (V2 * np.exp(-w * t)).sum(1) for t in (1.0, 5.0, 20.0, 60.0)}
    # commute-time centrality = (L^+)_ii = sum_{lambda>0} v_a(i)^2 / lambda_a
    invw = np.where(w > 1e-9, 1.0 / np.where(w > 1e-9, w, 1.0), 0.0)
    commute = (V2 * invw).sum(1)
    ipr = (V2 ** 2).sum(1) # inverse participation ratio (localization)
    D = np.column_stack([p_low, p_mid, p_high, fied, emass, deg,
                         hks[1.0], hks[5.0], hks[20.0], hks[60.0], commute, ipr]).astype(float)
    names = ["p_low", "p_mid", "p_high", "fiedler", "eff_mass", "degree",
             "hks1", "hks5", "hks20", "hks60", "commute", "ipr"]
    D = (D - D.mean(0)) / (D.std(0) + 1e-9) # per-protein standardization
    return D, names


def build_library(D, names):
    """Nonlinear library: base terms + squares + pairwise products + higher-order terms."""
    cols, terms = [], []
    d = D.shape[1]
    for i in range(d):
        cols.append(D[:, i]); terms.append(names[i])
    for i in range(d):
        cols.append(D[:, i] ** 2); terms.append(f"{names[i]}^2")
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(D[:, i] * D[:, j]); terms.append(f"{names[i]}*{names[j]}")
    Phi = np.column_stack(cols)
    Phi = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-9)
    return Phi, terms


def stlsq(Phi, y, thresh=0.05, ridge=1.0, iters=10):
    """Sequential Thresholded Least Squares -> sparse coefficients."""
    yc = y - y.mean()
    G = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    c = np.linalg.solve(G, Phi.T @ yc)
    for _ in range(iters):
        small = np.abs(c) < thresh
        c[small] = 0.0
        big = ~small
        if big.sum() == 0:
            break
        Gb = Phi[:, big].T @ Phi[:, big] + ridge * np.eye(big.sum())
        c[big] = np.linalg.solve(Gb, Phi[:, big].T @ yc)
    return c


def formula_str(c, terms, top=6):
    idx = np.argsort(-np.abs(c))
    parts = [f"{c[i]:+.3f}*{terms[i]}" for i in idx if abs(c[i]) > 1e-6][:top]
    return " ".join(parts) if parts else "(empty)"
