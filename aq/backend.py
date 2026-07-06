"""
Optimization layer: a single switch CPU(numpy) <-> GPU(CuPy).

All the heavy linear algebra (eigh, matmul, einsum in the propagation)
goes through `xp` returned by get_backend(). The stage code is written
once and runs on both backends.

- use_gpu=True and CuPy available -> computation on the GPU
- otherwise -> numpy (multithreaded LAPACK)

Numba (optionally) accelerates individual scalar loops on the CPU;
if it is unavailable, pure numpy variants are used.
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

    def njit(*args, **kwargs): # no-op decorator when Numba is missing
        def wrap(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return wrap

    def prange(*a): # fallback
        return range(*a)


def get_backend(use_gpu: bool):
    """Returns (xp, name). xp is either numpy or cupy."""
    if use_gpu and _HAS_CUPY:
        return _cp, "cupy(gpu)"
    return np, "numpy(cpu)"


def to_numpy(a):
    """Bring an array (numpy or cupy) back to numpy for saving/comparison."""
    if _HAS_CUPY and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)


def capabilities() -> dict:
    return {"cupy": _HAS_CUPY, "numba": _HAS_NUMBA}
