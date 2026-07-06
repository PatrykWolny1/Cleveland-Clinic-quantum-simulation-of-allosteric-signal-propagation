"""
Phase Peierlsa - syntetyczne pole cechowania (genuinnie zespolona phase).

Dotad eigenvectors byly rzeczywiste (phase = sign). Here H jest zespolony-
hermitowski z phase in edges:
  theta_ij = B * (x_i y_j - x_j y_i) (antysymetryczne -> H hermitowski)
  H_ij = -exp(i theta_ij) for kontaktu; diag = degree.
Quantum walk staje CHIRALNY -> genuinna interference phase.
Descriptory umeanamy over +-B (removeiecie arbitralnego kierunku pola).

Optimization: eigh zespolony (numpy), reuzycie; MP; GPU (cupy eigh complex).
"""
from __future__ import annotations
import numpy as np
from . import graph
from .backend import get_backend, to_numpy


def Hamiltonian_gauge(A, coords, B=0.15):
    x = coords[:, 0]; y = coords[:, 1]
    theta = B * (np.outer(x, y) - np.outer(y, x)) # antysymetryczne
    H = -(A > 0).astype(complex) * np.exp(1j * theta)
    np.fill_diagonal(H, A.sum(1))
    return H


def eig_complex(H, gpu=False):
    xp, name = get_backend(gpu)
    w, V = xp.linalg.eigh(xp.asarray(H, dtype=xp.complex128))
    return to_numpy(w), to_numpy(V)


def gauge_features(A, coords, B=0.15, T=20.0, d_min=3, gpu=False):
    """Oba descriptory phase Peierlsa z One eig in B (umeanone over +-B):
       gauge_coh - phase-locking per residue (chiralny propagator),
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
