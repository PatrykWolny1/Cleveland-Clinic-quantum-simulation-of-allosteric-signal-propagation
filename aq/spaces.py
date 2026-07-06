"""
Representation space + OOF: I search for a space in which the rule for
transfer between proteins becomes linear (simple).

Descriptors come from the spectrum of the data matrix (cutoff graph - where LST wins).
Per-protein transforms (crucial for transfer):
  zscore - standardization (baseline)
  rank_gauss - rank -> inverse normal CDF (equalizes distributions)
  whiten - PCA-whitening (equalizes covariances)
Evaluation: OOF (leave-one-protein-out) with a linear model (ridge) - high transfer
= the rule is linear in this space. Additionally an RBF kernel (nonlinear space).

Closed form -> fast; multiprocessing over proteins; GPU for eig.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtri
from . import graph, propagate, gauge
from .backend import to_numpy


def descriptors(coords, gpu=False):
    """Descriptors from the spectrum of L (cutoff graph). Returns (D[N x d], names)."""
    A = graph.build_graph(coords, "cutoff", cutoff=8.0)
    L = graph.Hamiltonian(A, "laplacian")
    w, V, _, _ = propagate.eig(L, gpu); w = to_numpy(w); V = to_numpy(V)
    idx = np.where(w > 1e-9)[0]
    lo = idx[:max(1, len(idx)//5)]; mi = idx[len(idx)//5:3*len(idx)//5]; hi = idx[3*len(idx)//5:]
    V2 = V**2
    p_low = V2[:, lo].sum(1); p_mid = V2[:, mi].sum(1); p_high = V2[:, hi].sum(1)
    fied = np.abs(V[:, idx[0]]) if len(idx) else np.zeros(V.shape[0])
    emass = to_numpy(propagate.effective_mass(w, V, np, 20.0))
    invw = np.where(w > 1e-9, 1.0/np.where(w > 1e-9, w, 1.0), 0.0)
    commute = (V2*invw).sum(1)
    hks20 = (V2*np.exp(-w*20.0)).sum(1)
    deg = A.sum(1)
    # --- phase channels (superposition + interference + resonance + phase) ---
    Hf = graph.Hamiltonian(A, "logf", sigma=1.0)
    wf, Vf, _, _ = propagate.eig(Hf, gpu); wf = to_numpy(wf); Vf = to_numpy(Vf)
    def band(M):
        W = M.copy(); np.fill_diagonal(W, 0.0)
        gd = graph.graph_distance(A); b = np.isfinite(gd) & (gd >= 3); return (W*b).sum(1)
    interf = band(to_numpy(propagate.q_interference(wf, Vf, np, 20.0))) # Re (cos)
    quad = band(to_numpy(propagate.q_quadrature(wf, Vf, np, 20.0))) # Im (sin) - new
    pcoh = to_numpy(propagate.phase_coherence(wf, Vf, np)) # phase-locking - new
    # --- Peierls-phase channels (chiral interference, genuine phase) ---
    gcoh, ginterf = gauge.gauge_features(A, coords, 0.15, 20.0, gpu=gpu)
    D = np.column_stack([p_low, p_mid, p_high, fied, emass, commute, hks20, deg,
                         interf, quad, pcoh, gcoh, ginterf]).astype(float)
    names = ["p_low","p_mid","p_high","fiedler","eff_mass","commute","hks20","degree",
             "interf","quadrature","phase_coh","gauge_coh","gauge_interf"]
    return D, names


# --- space transforms (per protein) ---
def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def rank_gauss(X):
    out = np.zeros_like(X, float)
    n = X.shape[0]
    for j in range(X.shape[1]):
        r = np.argsort(np.argsort(X[:, j]))
        u = (r + 0.5) / n
        out[:, j] = ndtri(np.clip(u, 1e-6, 1-1e-6))
    return out


def whiten(X):
    Xc = X - X.mean(0)
    C = np.cov(Xc, rowvar=False) + 1e-6*np.eye(X.shape[1])
    wv, U = np.linalg.eigh(C)
    Wm = U @ np.diag(1.0/np.sqrt(np.clip(wv, 1e-9, None))) @ U.T
    return Xc @ Wm


TRANSFORMS = {"zscore": zscore, "rank_gauss": rank_gauss, "whiten": whiten}


def ridge_fit(X, y, lam=1.0):
    yc = y - y.mean()
    return np.linalg.solve(X.T @ X + lam*np.eye(X.shape[1]), X.T @ yc)


def rbf_kernel(Xa, Xb, gamma):
    d = ((Xa[:, None, :] - Xb[None, :, :])**2).sum(2)
    return np.exp(-gamma*d)


def kernel_ridge_fit(K, y, lam=1.0):
    yc = y - y.mean()
    return np.linalg.solve(K + lam*np.eye(K.shape[0]), yc)


def all_features(coords, gpu=False):
    """Full set of channels (amplitude + phase + spectral + coded + gauge) from ~4 eig."""
    A = graph.build_graph(coords, "cutoff", cutoff=8.0)
    gd = graph.graph_distance(A); b = np.isfinite(gd) & (gd >= 3)
    def band(M):
        W = M.copy(); np.fill_diagonal(W, 0.0); return (W*b).sum(1)
    xp = np
    cols, names = [], []
    # --- L: spectral + coded ---
    L = graph.Hamiltonian(A, "laplacian"); wL, VL, _, _ = propagate.eig(L, gpu)
    wL = to_numpy(wL); VL = to_numpy(VL); V2 = VL**2
    idx = np.where(wL > 1e-9)[0]
    lo = idx[:max(1, len(idx)//5)]; mi = idx[len(idx)//5:3*len(idx)//5]; hi = idx[3*len(idx)//5:]
    for nm, v in [("p_low", V2[:, lo].sum(1)), ("p_mid", V2[:, mi].sum(1)),
                  ("p_high", V2[:, hi].sum(1)), ("fiedler", np.abs(VL[:, idx[0]])),
                  ("eff_mass", to_numpy(propagate.effective_mass(wL, VL, np, 20.0))),
                  ("commute", (V2*np.where(wL > 1e-9, 1/np.where(wL > 1e-9, wL, 1), 0)).sum(1)),
                  ("hks5", (V2*np.exp(-wL*5)).sum(1)), ("hks20", (V2*np.exp(-wL*20)).sum(1)),
                  ("hks60", (V2*np.exp(-wL*60)).sum(1)), ("ipr", (V2**2).sum(1)),
                  ("degree", A.sum(1)),
                  ("yukawa0.1", band(to_numpy(propagate.yukawa(wL, VL, np, 0.1)))),
                  ("yukawa0.5", band(to_numpy(propagate.yukawa(wL, VL, np, 0.5)))),
                  ("yukawa2.0", band(to_numpy(propagate.yukawa(wL, VL, np, 2.0))))]:
        cols.append(v); names.append(nm)
    # --- log-f: interference (amplitude+sign) + phase + superposition ---
    Hf = graph.Hamiltonian(A, "logf", sigma=1.0); wf, Vf, _, _ = propagate.eig(Hf, gpu)
    wf = to_numpy(wf); Vf = to_numpy(Vf)
    for T in (10.0, 20.0, 40.0):
        cols.append(band(to_numpy(propagate.q_interference(wf, Vf, np, T)))); names.append(f"interf_T{int(T)}")
    cols.append(band(to_numpy(propagate.q_quadrature(wf, Vf, np, 20.0)))); names.append("quadrature")
    cols.append(to_numpy(propagate.phase_coherence(wf, Vf, np))); names.append("phase_coh")
    cols.append(band(to_numpy(propagate.q_coherent(wf, Vf, np, None)))); names.append("coherent")
    # --- adjacency: communicability ---
    Aadj = graph.Hamiltonian(A, "adjacency"); wa, Va, _, _ = propagate.eig(Aadj, gpu)
    cols.append(band(to_numpy(propagate.q_communicability(to_numpy(wa), to_numpy(Va), np, 0.5)))); names.append("communic")
    # --- gauge: Peierls phase ---
    gcoh, ginterf = gauge.gauge_features(A, coords, 0.15, 20.0, gpu=gpu)
    cols.append(gcoh); names.append("gauge_coh"); cols.append(ginterf); names.append("gauge_interf")
    D = np.column_stack(cols).astype(float)
    return D, names
