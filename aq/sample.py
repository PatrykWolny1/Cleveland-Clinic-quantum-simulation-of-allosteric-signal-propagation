"""
Mechanism sampling (z coded model / calki over trajektoriach).

Stochastyczna estymacja connectivity M = f(H) BEZ eigendecomposition:
  - probes gaussowskie xi ~ N(0,1) (miara mu_code),
  - f(H) xi via rozwiniecie Czebyszewa (only mnozenia matrix-wektor),
  - M_ij ~ (1/R) sum_r (f(H)xi_r)_i (xi_r)_j [Hutchinson].
Cost: O(edges * K * R) instead of O(N^3) eig -> skaluje to largech N, GPU-friendly.
"""
from __future__ import annotations
import numpy as np


def _cheb_coeffs(fn, K, a, b, M=200):
    """Wspolczynniki Czebyszewa f in [a,b] (quadrature)."""
    k = np.arange(M)
    x = np.cos(np.pi * (k + 0.5) / M) # nodes
    lam = 0.5 * (b - a) * x + 0.5 * (a + b)
    fx = fn(lam)
    c = np.zeros(K)
    for j in range(K):
        c[j] = (2.0 / M) * np.sum(fx * np.cos(np.pi * j * (k + 0.5) / M))
    c[0] *= 0.5
    return c


def apply_spectral(A, Vp, fn, a, b, K=40, gpu=False):
    """Stosuje f(L) to kolumn Vp via Czebyszew (matvec rzadkie). L=D-A.
    gpu=True -> cupyx.scipy.sparse in GPU (macvec rzadki przyjazny karcie)."""
    deg = A.sum(1)
    Ldense = np.diag(deg) - A
    if gpu:
        try:
            import cupy as cp
            import cupyx.scipy.sparse as csp
            Ls = csp.csr_matrix(cp.asarray(Ldense)); Vp = cp.asarray(Vp); xp = cp
        except Exception:
            from scipy.sparse import csr_matrix
            Ls = csr_matrix(Ldense); xp = np
    else:
        from scipy.sparse import csr_matrix
        Ls = csr_matrix(Ldense); xp = np
    c = _cheb_coeffs(fn, K, a, b)
    def Ly(x):
        return (2.0 * (Ls @ x) - (a + b) * x) / (b - a)
    T0 = Vp; T1 = Ly(Vp)
    out = c[0] * T0 + c[1] * T1
    for j in range(2, K):
        T2 = 2.0 * Ly(T1) - T0
        out = out + c[j] * T2
        T0, T1 = T1, T2
    return out.get() if hasattr(out, "get") else out # cupy->numpy or numpy


def stochastic_connectivity(A, fn, R=None, K=40, seed=0, gpu=False):
    """Estymuje M = f(L) samplingm. R=None -> adaptacyjne (~sqrt(N))."""
    n = A.shape[0]
    if R is None:
        R = int(min(400, max(40, 6 * np.sqrt(n)))) # adaptacyjne under accuracy
    b = 2.0 * A.sum(1).max()
    rng = np.random.default_rng(seed)
    Xi = rng.standard_normal((n, R))
    Y = apply_spectral(A, Xi, fn, 0.0, b, K, gpu=gpu)
    return (Y @ Xi.T) / R


def stochastic_rowsum(A, fn, K=40):
    """Deterministyczny row-sum f(L)1 (one macvec) - exact, O(edges*K)."""
    n = A.shape[0]
    b = 2.0 * A.sum(1).max()
    ones = np.ones((n, 1))
    return apply_spectral(A, ones, fn, 0.0, b, K).ravel()
