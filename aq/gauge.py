"""
Peierls phase - a synthetic gauge field (genuinely complex phase).

So far the eigenvectors were real (phase = sign). Here H is complex-Hermitian
with a phase on the edges:
  theta_ij = B * (x_i y_j - x_j y_i) (antisymmetric -> H Hermitian)
  H_ij = -exp(i theta_ij) for a contact; diag = degree.
The quantum walk becomes CHIRAL -> a genuine interference phase.
I average the descriptors over +-B (removing the arbitrary field direction).

Optimization: complex eigh (numpy), reuse; MP; GPU (cupy complex eigh).
"""
from __future__ import annotations
import numpy as np
from . import graph
from .backend import get_backend, to_numpy


def Hamiltonian_gauge(A, coords, B=0.15):
    x = coords[:, 0]; y = coords[:, 1]
    theta = B * (np.outer(x, y) - np.outer(y, x)) # antisymmetric
    H = -(A > 0).astype(complex) * np.exp(1j * theta)
    np.fill_diagonal(H, A.sum(1))
    return H


def eig_complex(H, gpu=False):
    xp, name = get_backend(gpu)
    w, V = xp.linalg.eigh(xp.asarray(H, dtype=xp.complex128))
    return to_numpy(w), to_numpy(V)


def gauge_features(A, coords, B=0.15, T=20.0, d_min=3, gpu=False):
    """Both Peierls-phase descriptors from a single eig at each B (averaged over +-B):
       gauge_coh - phase-locking per residue (chiral propagator),
       gauge_interf - band-score chiral interference (|Im<U>|)."""
    gd = graph.graph_distance(A)
    b = np.isfinite(gd) & (gd >= d_min)
    coh = np.zeros(A.shape[0]); interf = np.zeros(A.shape[0])
    for Bv in (B, -B):
        H = Hamiltonian_gauge(A, coords, Bv)
        w, V = eig_complex(H, gpu)
        ws = np.sort(w); gap = np.median(np.diff(ws)); ts = 1.0 / max(abs(gap), 1e-6)
        U = (V * np.exp(-1j * w * ts)) @ V.conj().T
        coh += np.abs(U.sum(1)) / (np.abs(U).sum(1) + 1e-12)
        x = w * T
        f = np.where(np.abs(x) > 1e-9, (1 - np.exp(-1j * x)) / (1j * np.where(np.abs(x) > 1e-9, x, 1.0)), 1.0)
        M = (V * f) @ V.conj().T
        Wm = np.abs(M.imag); np.fill_diagonal(Wm, 0.0)
        interf += (Wm * b).sum(1)
    return 0.5 * coh, 0.5 * interf
