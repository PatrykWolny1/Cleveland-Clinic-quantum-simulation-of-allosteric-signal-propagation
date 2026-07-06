"""
Noise resilience (cel drugorzedny + wiarygodnosc near-term hardware).

Three source noise:
  1) DEKOHERENCJA (dephasing) - fizycznie: tlumienie interference.
     Spectral filter interpoluje sinc(lT) [coherent] -> heat(lT) [classical]:
       f_p(l) = (1-p) sinc(lT) + p (1-e^{-lT})/(lT)
     p=0 full interference quantum, p=1 full dyfusion classical.
  2) SZUM STRUKTURALNY - perturbation coordinates (uncertainty input).
  3) SZUM POMIAROWY (shots) - estymacja |U|^2 skonczona number samples.
Miara: AUC detector w funkcji sily noise (gentle decline = resilience).
"""
from __future__ import annotations
import numpy as np
from . import graph, propagate
from .backend import to_numpy


def _band(M, gd, d_min=3):
    W = M.astype(float).copy(); np.fill_diagonal(W, 0.0)
    b = np.isfinite(gd) & (gd >= d_min)
    return (W * b).sum(1)


def _sinc(x):
    return np.where(np.abs(x) > 1e-9, np.sin(x) / np.where(np.abs(x) > 1e-9, x, 1.0), 1.0)


def _heat(x):
    return np.where(np.abs(x) > 1e-9, (1.0 - np.exp(-x)) / np.where(np.abs(x) > 1e-9, x, 1.0), 1.0)


def dephased_metric(w, V, T, p):
    """M(p) = V diag[(1-p) sinc(lT) + p heat(lT)] V^T (one eig)."""
    x = w * T
    f = (1.0 - p) * _sinc(x) + p * _heat(x)
    return (V * f) @ V.T


def dephasing_curve(coords, y, gd_of, sigma=1.0, T=20.0, ps=(0.0, 0.25, 0.5, 0.75, 1.0), gpu=False):
    from aq import score
    A = graph.build_graph(coords, "cutoff", cutoff=8.0)
    gd = gd_of(A)
    H = graph.Hamiltonian(A, "logf", sigma=sigma)
    w, V, _, _ = propagate.eig(H, gpu); w = to_numpy(w); V = to_numpy(V)
    out = []
    for p in ps:
        M = dephased_metric(w, V, T, p)
        sc = _band(M, gd)
        out.append(score.evaluate(sc, np.argsort(-sc), y, (5,)).get("roc_auc"))
    return list(ps), out


def coordinate_noise(coords, y, gd_of, sigmas=(0.5, 1.0, 1.5), n_real=10,
                     res_sigma=1.0, T=20.0, seed=0, gpu=False):
    from aq import score
    rng = np.random.default_rng(seed)
    res = {}
    for s in sigmas:
        aucs = []
        for _ in range(n_real):
            c = coords + rng.normal(0, s, coords.shape)
            A = graph.build_graph(c, "cutoff", cutoff=8.0); gd = gd_of(A)
            H = graph.Hamiltonian(A, "logf", sigma=res_sigma)
            w, V, _, _ = propagate.eig(H, gpu); w = to_numpy(w); V = to_numpy(V)
            sc = _band(dephased_metric(w, V, T, 0.0), gd)
            aucs.append(score.evaluate(sc, np.argsort(-sc), y, (5,)).get("roc_auc"))
        res[s] = (float(np.mean(aucs)), float(np.std(aucs)))
    return res
