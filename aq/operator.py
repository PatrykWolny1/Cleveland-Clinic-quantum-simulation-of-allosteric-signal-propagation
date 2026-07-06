"""
Operator transfer w space sygnalow (rama INPUT * W = OUTPUT).

For each protein: matrix sygnalow S [residues x channels] = spectrum LST residues
(interference w siatce sigma x T). Label y (pocket).
  model: S W = y
  operator: W = S^+ y (pseudoodwrotnosc = "{operator}^-1")
W this spectral filter above plaszczyzna sigma-T (PRAWO). Szukamy W wspolnego
for 3 par treningowych i stosujemy in 4. case (test/c-Myc).

Forma closed-form (pinv/ridge) -> quickly, without iteracji; xp -> CPU/GPU.
"""
from __future__ import annotations
import numpy as np
from . import graph, propagate
from .backend import to_numpy, get_backend

SIGMAS = (0.3, 0.6, 1.0, 1.6, 2.5) # szerokosci resonance (log-f)
TIMES = (5.0, 15.0, 30.0, 60.0) # times interference
CHANNELS = [(s, t) for s in SIGMAS for t in TIMES]


def _band(M, gd, d_min=3):
    W = M.astype(float).copy(); np.fill_diagonal(W, 0.0)
    b = np.isfinite(gd) & (gd >= d_min)
    return (W * b).sum(1)


YUKAWA_M2 = (0.1, 0.5, 2.0) # skale ekranowania (dlugosci 1/m)


def signal_matrix(coords, gpu=False, rich=True):
    """S [N x channels]. rich=False: only interference sigma-T.
    rich=True: + Yukawa (m2), + coherent (sigma), + masa efektywna.
    Channels ortogonalne pozwalaja operatorowi objac ABL i myosin one baza."""
    A = graph.build_graph(coords, "knn", k=10) # graph tau
    gd = graph.graph_distance(A)
    xp, _ = get_backend(gpu)
    cols, names = [], []
    for s in SIGMAS: # REZONANS
        H = graph.Hamiltonian(A, "logf", sigma=s)
        w, V, _, _ = propagate.eig(H, gpu)
        for t in TIMES: # INTERFERENCJA
            cols.append(_band(to_numpy(propagate.q_interference(w, V, xp, t)), gd))
            names.append(f"interf_s{s}_T{int(t)}")
        if rich: # Superposition
            cols.append(_band(to_numpy(propagate.q_coherent(w, V, xp, None)), gd))
            names.append(f"coh_s{s}")
    if rich:
        Lp = graph.Hamiltonian(A, "laplacian")
        wL, VL, _, _ = propagate.eig(Lp, gpu)
        for m2 in YUKAWA_M2: # coded: propagator ekranowany
            cols.append(_band(to_numpy(propagate.yukawa(wL, VL, xp, m2)), gd))
            names.append(f"yukawa_m{m2}")
        cols.append(to_numpy(propagate.effective_mass(wL, VL, xp, 20.0)))
        names.append("eff_mass")
    S = np.column_stack(cols).astype(float)
    S = (S - S.mean(0)) / (S.std(0) + 1e-9)
    signal_matrix.names = names
    return S, gd


def fit_operator(S, y, ridge=1.0):
    """W = (S^T S + ridge I)^-1 S^T y_c (forma closed-form)."""
    yc = y - y.mean()
    G = S.T @ S + ridge * np.eye(S.shape[1])
    return np.linalg.solve(G, S.T @ yc)


def apply_operator(S, W):
    return S @ W


def operator_agreement(Ws):
    """Matrix cos-podobienstwa operatorow (whether istnieje one PRAWO)."""
    Ws = [w / (np.linalg.norm(w) + 1e-12) for w in Ws]
    n = len(Ws)
    return np.array([[float(Ws[i] @ Ws[j]) for j in range(n)] for i in range(n)])
