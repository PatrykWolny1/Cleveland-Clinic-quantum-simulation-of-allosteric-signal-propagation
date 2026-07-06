"""
Plateau mechanism (n-bubble equilibria, Slow 2025) as a geometric descriptor.

At Plateau equilibrium the vectors from a residue to its neighbors satisfy cos(phi_n)=-1/(n-1).
A residue in a packed core -> small deviation; a residue lining a void
pocket -> neighbors are one-sided -> large angular deviation from equilibrium.
Score = mean of (cos_angle - (-1/(n-1)))^2 over pairs of neighbors.
Numba (CPU multiprocessing). Purely structural (from apo), complementary to spectral metrics.
"""
from __future__ import annotations
import numpy as np

try:
    from numba import njit, prange
    _HAVE = True
except Exception:
    _HAVE = False
    def njit(*a, **k):
        def d(f): return f
        return d if not a else d(a[0])
    prange = range


@njit(cache=True, parallel=True)
def _plateau(coords, indptr, indices):
    N = coords.shape[0]
    out = np.zeros(N)
    for i in prange(N):
        s, e = indptr[i], indptr[i+1]
        deg = e - s
        if deg < 2:
            continue
        target = -1.0 / (deg - 1)
        acc = 0.0; cnt = 0
        for a in range(s, e):
            ja = indices[a]
            dxa = coords[ja, 0]-coords[i, 0]; dya = coords[ja, 1]-coords[i, 1]; dza = coords[ja, 2]-coords[i, 2]
            na = (dxa*dxa+dya*dya+dza*dza)**0.5 + 1e-9
            for b in range(a+1, e):
                jb = indices[b]
                dxb = coords[jb, 0]-coords[i, 0]; dyb = coords[jb, 1]-coords[i, 1]; dzb = coords[jb, 2]-coords[i, 2]
                nb = (dxb*dxb+dyb*dyb+dzb*dzb)**0.5 + 1e-9
                cosv = (dxa*dxb+dya*dyb+dza*dzb)/(na*nb)
                d = cosv - target
                acc += d*d; cnt += 1
        if cnt > 0:
            out[i] = acc/cnt
    return out


def plateau_score(coords, A):
    """Geometric deviation of neighbors from Plateau equilibrium, per residue."""
    Acsr = (A > 0)
    indptr = np.zeros(len(coords)+1, dtype=np.int64)
    idx = []
    for i in range(len(coords)):
        nb = np.where(Acsr[i])[0]
        idx.extend(nb.tolist()); indptr[i+1] = indptr[i] + len(nb)
    indices = np.asarray(idx, dtype=np.int64)
    return _plateau(np.ascontiguousarray(coords, dtype=np.float64), indptr, indices)
