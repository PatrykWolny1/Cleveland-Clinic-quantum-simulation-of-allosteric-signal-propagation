"""
Warstwa optimization: one przełącznik CPU(numpy) <-> GPU(CuPy).

Cała ciężka algebra linear (eigh, matmul, einsum w propagation)
przechodzi via `xp` zwrócone via get_backend(). Kod stage'y jest
napisany once i działa in obu backendach.

- use_gpu=True i CuPy dostępne -> computation in GPU
- w przeciwnym razie -> numpy (LAPACK wielowatkowy)

Numba (optionally) accelerates pojedyncze pętle skalarne in CPU;
if niedostepna, uzywane are czyste variants numpy.
"""
from __future__ import annotations
import numpy as np

# --- GPU (CuPy) ----------------------------------------------------------
try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False

# --- Numba ---------------------------------------------------------------
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def njit(*args, **kwargs): # no-op dekorator, when none Numby
        def wrap(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return wrap

    def prange(*a): # fallback
        return range(*a)


def get_backend(use_gpu: bool):
    """Returns (xp, name). xp this numpy or cupy."""
    if use_gpu and _HAS_CUPY:
        return _cp, "cupy(gpu)"
    return np, "numpy(cpu)"


def to_numpy(a):
    """Sprowadza tablice (numpy or cupy) to numpy for zapisu/porownan."""
    if _HAS_CUPY and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)


def capabilities() -> dict:
    return {"cupy": _HAS_CUPY, "numba": _HAS_NUMBA}
