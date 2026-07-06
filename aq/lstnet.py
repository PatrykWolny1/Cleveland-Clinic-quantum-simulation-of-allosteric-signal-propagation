"""
LST-QNN: siec neural zbudowana z mechanismow LST.

Warstwa features = four mechanismy LST naraz (fizyka, not generyczny ansatz):
  Threshold : graph tau (knn, stale k) - aksjomat 11
  REZONANS : significance log-f o szerokosci sigma - rown. 18
  Superposition: propagation e^{-iH(sigma)t} (quantum walk)
  INTERFERENCJA: odczyt that signiem M=V sinc(lambda T) V^T - rown. 10
Bank cech: interference w siatce (sigma, T) -> wieloskalowy sygnal LST.
Glowica: wypukla (quantum kernel / logistic) -> None barren plateau,
few fizycznie interpretowalnych parametrow.
"""
from __future__ import annotations
import numpy as np
from . import graph, propagate
from .backend import to_numpy, get_backend

SIGMAS = (0.5, 1.0, 2.0) # szerokosci resonance (log-f)
TIMES = (10.0, 40.0) # times interference
K_TAU = 10 # threshold tau: stala number neighbors


def _band(M, gd, d_min=3):
    W = M.astype(float).copy(); np.fill_diagonal(W, 0.0)
    b = np.isfinite(gd) & (gd >= d_min)
    return (W * b).sum(1)


def lst_features(coords, gpu=False, mode="combined"):
    """Returns (X[N,F], gd). mode: 'lst' (only interference) |
    'combined' (4 mechanismy LST + features structurelne transferowalne)."""
    A = graph.build_graph(coords, "knn", k=K_TAU) # Threshold tau
    gd = graph.graph_distance(A)
    xp, _ = get_backend(gpu)
    feats, names = [], []
    for sig in SIGMAS: # REZONANS (sigma)
        H = graph.Hamiltonian(A, "logf", sigma=sig)
        w, V, _, _ = propagate.eig(H, gpu) # Superposition
        for T in TIMES: # INTERFERENCJA (sign)
            feats.append(_band(to_numpy(propagate.q_interference(w, V, xp, T)), gd))
            names.append(f"interf_s{sig}_T{int(T)}")
        if mode == "combined": # Superposition: coherent walk
            feats.append(_band(to_numpy(propagate.q_coherent(w, V, xp, None)), gd))
            names.append(f"coh_s{sig}")
    if mode == "combined":
        # coded model (main_EN): propagator masywny (Yukawa) + masa efektywna
        Lp = graph.Hamiltonian(A, "laplacian")
        wL, VL, _, _ = propagate.eig(Lp, gpu)
        feats.append(_band(to_numpy(propagate.yukawa(wL, VL, xp, 0.5)), gd)); names.append("yukawa")
        feats.append(to_numpy(propagate.effective_mass(wL, VL, xp, 20.0))); names.append("eff_mass")
        # REZONANS: sila sviaenia log-f (row-sum weights)
        feats.append(graph.logf_weights(A, 1.0).sum(1)); names.append("resonance")
        # features structurelne (transferowalne between proteins)
        feats.append(A.sum(1)); names.append("degree")
        feats.append(np.array([np.mean(gd[i][np.isfinite(gd[i])]) for i in range(len(A))]))
        names.append("centrality")
    X = np.column_stack(feats).astype(float)
    X = np.tanh((X - X.mean(0)) / (X.std(0) + 1e-9))
    return X, gd, names


def feature_names():
    return [f"interf_s{s}_T{int(t)}" for s in SIGMAS for t in TIMES]
