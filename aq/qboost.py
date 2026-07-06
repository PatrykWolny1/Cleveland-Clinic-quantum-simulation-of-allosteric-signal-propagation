"""
QBoost - quantum tree boosting (Neven et al.). "A tree like LGBM, but quantum".

Weak classifiers = decision stumps (nodes: threshold on a feature, +-sign).
Instead of greedy addition (LGBM) - ensemble selection posed as a QUBO:
  min_w sum_n ( (1/K) sum_i w_i h_i(x_n) - y_n )^2 + lambda sum_i w_i
  w_i in {0,1} -> QUBO -> QPU (D-Wave anneal / QAOA on gate-based hardware).
I feed in ALL channels (amplitude + phase).

Optimization: stump/QUBO construction vectorized (numpy); the QUBO solve is the
quantum part (QPU). CPU simulation via simulated annealing for training.
"""
from __future__ import annotations
import numpy as np


def build_stumps(X, q=(0.35, 0.5, 0.65)):
    """Stump pool: for each feature, thresholds at quantiles x 2 signs."""
    stumps = []
    for j in range(X.shape[1]):
        for tq in q:
            t = np.quantile(X[:, j], tq)
            stumps.append((j, t, +1)); stumps.append((j, t, -1))
    return stumps


def stump_matrix(X, stumps):
    """H [N x K] in {-1,+1}: h = s * sign(x_j - t)."""
    H = np.empty((X.shape[0], len(stumps)))
    for k, (j, t, s) in enumerate(stumps):
        H[:, k] = s * np.where(X[:, j] - t >= 0, 1.0, -1.0)
    return H


def build_qubo(H, y, lam=0.04):
    """Q [K x K]: QBoost. y in {-1,+1}. (w_i^2=w_i -> on the diagonal.)"""
    N, K = H.shape
    HtH = H.T @ H; Hty = H.T @ y
    Q = (2.0 / K**2) * HtH # off-diag = 2*(H^tH)/K^2
    diag = (1.0 / K**2) * np.diag(HtH) - (2.0 / K) * Hty + lam
    np.fill_diagonal(Q, diag)
    return Q


try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _sa_core(Q, steps, restarts, rand_i, rand_u, w0):
    K = Q.shape[0]; best_e = 1e18; best_w = w0[0].copy()
    for r in range(restarts):
        w = w0[r].copy()
        Qw = Q @ w # O(K^2) once
        e = float(w @ Qw)
        for t in range(steps):
            T = max(1e-3, 1.0 - t / steps)
            i = rand_i[r, t]
            dw = 1.0 - 2.0 * w[i]
            de = dw * (Q[i, i] + 2.0 * (Qw[i] - Q[i, i] * w[i]))
            if de < 0 or rand_u[r, t] < np.exp(-de / T):
                w[i] += dw; Qw += dw * Q[:, i]; e += de
        if e < best_e:
            best_e = e; best_w = w.copy()
    return best_w, best_e

if _HAVE_NUMBA:
    _sa_core = njit(cache=True)(_sa_core)


def solve_sa(Q, steps=4000, restarts=8, seed=0):
    """Simulated annealing QUBO (delta O(K), Numba). On a QPU -> D-Wave/QAOA."""
    rng = np.random.default_rng(seed); K = Q.shape[0]
    ri = rng.integers(0, K, (restarts, steps))
    ru = rng.random((restarts, steps))
    w0 = rng.integers(0, 2, (restarts, K)).astype(np.float64)
    return _sa_core(np.ascontiguousarray(Q), steps, restarts, ri, ru, w0)


def predict(X, stumps, w):
    H = stump_matrix(X, stumps)
    return (H * w) .sum(1) / max(1.0, w.sum())
