"""
Quantum siec neural Under QPU: data re-uploading VQC.

Architektura (Perez-Salinas): few qubits, WSZYSTKIE features wgrywane
wielokrotnie (re-uploading) -> uniwersalny aproksymator robust in barren
plateau. Warstwa: enkodowanie data (RY/RZ z features PCA) + warstwa trenowalna
(RY/RZ) + pierscien CNOT. Odczyt <Z_0>.
Trening: parameter-shift (gradient NATYWNY for QPU). Circuit = jawna sekwencja
gates -> export in AWS Braket / Classiq.

This jest kod Under QPU: symulujemy statevector to treningu, but circuit mapuje 
1:1 in hardware gate-based (without sztuczek nieexecutablech on QPU).
"""
from __future__ import annotations
import numpy as np


def _ry(state, ang, q, n, xp):
    B = state.shape[0]
    ang = xp.asarray(ang)
    c = xp.cos(ang/2); s = xp.sin(ang/2)
    if c.ndim == 0:
        c = xp.full(B, c); s = xp.full(B, s)
    bshape = (B,) + (1,)*(n-1)
    c = c.reshape(bshape); s = s.reshape(bshape)
    st = xp.moveaxis(state, q+1, -1)
    n0 = st[..., 0]*c - st[..., 1]*s
    n1 = st[..., 0]*s + st[..., 1]*c
    return xp.moveaxis(xp.stack([n0, n1], -1), -1, q+1)


def _rz(state, ang, q, n, xp):
    B = state.shape[0]
    ang = xp.asarray(ang)
    e0 = xp.exp(-1j*ang/2); e1 = xp.exp(1j*ang/2)
    if e0.ndim == 0:
        e0 = xp.full(B, e0); e1 = xp.full(B, e1)
    bshape = (B,) + (1,)*(n-1)
    e0 = e0.reshape(bshape); e1 = e1.reshape(bshape)
    st = xp.moveaxis(state, q+1, -1)
    st = xp.stack([st[..., 0]*e0, st[..., 1]*e1], -1)
    return xp.moveaxis(st, -1, q+1)


def _cnot(state, c, t, n, xp):
    sl = [slice(None)]*(n+1); sl[c+1] = 1
    tax = t+1 - (1 if t > c else 0)
    state[tuple(sl)] = xp.flip(state[tuple(sl)], axis=tax)
    return state


def forward(X, theta, n, L, xp):
    """X:(B, 2n) features PCA; theta:(L,n,2). Returns P(1) in kubicie 0."""
    B = X.shape[0]
    state = xp.zeros((B, 2**n), dtype=xp.complex128); state[:, 0] = 1.0
    state = state.reshape((B,)+(2,)*n)
    for l in range(L):
        for q in range(n): # enkodowanie data (re-uploading)
            state = _ry(state, np.pi*X[:, 2*q], q, n, xp)
            state = _rz(state, np.pi*X[:, 2*q+1], q, n, xp)
        for q in range(n): # warstwa trenowalna
            state = _ry(state, theta[l, q, 0], q, n, xp)
            state = _rz(state, theta[l, q, 1], q, n, xp)
        for q in range(n): # splatanie
            state = _cnot(state, q, (q+1) % n, n, xp)
    psi = state.reshape(B, 2, 2**(n-1))
    return (xp.abs(psi[:, 1, :])**2).sum(1)


def _loss(X, y, theta, n, L, xp, w):
    p = xp.clip(forward(X, theta, n, L, xp), 1e-6, 1-1e-6)
    return float(-(w*(y*xp.log(p)+(1-y)*xp.log(1-p))).mean())


def train(X, y, n, L=3, steps=120, lr=0.15, seed=0):
    """Parameter-shift + Adam (gradient natywny QPU)."""
    import numpy as xp
    rng = np.random.default_rng(seed)
    pos = max(1.0, y.sum()); neg = max(1.0, (1-y).sum())
    w = np.where(y > 0.5, neg/pos, 1.0)
    theta = rng.normal(0, 0.1, (L, n, 2))
    mt = np.zeros_like(theta); vt = np.zeros_like(theta); b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps+1):
        g = np.zeros_like(theta)
        for l in range(L): # parameter-shift: +-pi/2
            for q in range(n):
                for r in range(2):
                    tp = theta.copy(); tp[l, q, r] += np.pi/2
                    tm = theta.copy(); tm[l, q, r] -= np.pi/2
                    g[l, q, r] = 0.5*(_loss(X, y, tp, n, L, xp, w) - _loss(X, y, tm, n, L, xp, w))
        mt = b1*mt+(1-b1)*g; vt = b2*vt+(1-b2)*g*g
        mh = mt/(1-b1**t); vh = vt/(1-b2**t)
        theta -= lr*mh/(np.sqrt(vh)+eps)
    return theta


def predict(X, theta, n, L=3):
    import numpy as xp
    return forward(X, theta, n, L, xp)


def export_circuit(theta, n, L):
    """Jawna sekwencja gates (for Braket/Classiq). x_k = features PCA."""
    lines = [f"# data re-uploading VQC: {n} qubits, {L} warstw"]
    for l in range(L):
        for q in range(n):
            lines.append(f"RY(pi*x_{2*q}) q{q}; RZ(pi*x_{2*q+1}) q{q} # enkod L{l}")
        for q in range(n):
            lines.append(f"RY({theta[l,q,0]:+.3f}) q{q}; RZ({theta[l,q,1]:+.3f}) q{q} # tren L{l}")
        for q in range(n):
            lines.append(f"CNOT q{q},q{(q+1)%n}")
    lines.append("MEASURE q0 -> P(1)")
    return "\n".join(lines)
