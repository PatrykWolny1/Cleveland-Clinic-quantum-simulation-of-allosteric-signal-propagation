"""
Quantum neural network (variational quantum classifier, VQC).

A learned combiner of unsupervised quantum features -> allosteric score.
Statevector simulation vectorized over residues (numpy/CuPy). SPSA training
(2 forward passes per iteration, standard in QML). Leave-one-protein-out
evaluation (leakage-free: the model does not see the test protein's labels).

Ansatz (hardware-efficient, shallow -> depth-aware for the near term):
  data encoding: RY(pi * x_i) on qubit i
  L layers: RY(theta) on each qubit + CNOT ring
  readout: P(1) on qubit 0 = (1 - <Z_0>)/2
"""
from __future__ import annotations
import numpy as np
from .backend import get_backend, to_numpy


# ---------- vectorized statevector simulator (batch over residues) -------
def _ry(theta, xp):
    c = xp.cos(theta / 2); s = xp.sin(theta / 2)
    return xp.array([[c, -s], [s, c]], dtype=xp.complex128)


def _apply_1q(state, G, q, n, xp):
    """state: (B, 2,2,...,2) [n qubit axes]; G: 2x2; qubit q."""
    st = xp.moveaxis(state, q + 1, -1) # batch axis = 0
    st = st @ G.T
    return xp.moveaxis(st, -1, q + 1)


def _apply_cnot(state, c, t, n, xp):
    """CNOT: when control c=1, flip target t."""
    sl = [slice(None)] * (n + 1); sl[c + 1] = 1
    block = state[tuple(sl)] # (B, ... without axis c)
    tax = t + 1 - (1 if t > c else 0) # axis shift due to the slice
    state[tuple(sl)] = xp.flip(block, axis=tax)
    return state


def vqc_forward(X, theta, n, L, xp):
    """X: (B, n) features in [-1,1]-ish; theta: (L, n). Returns P(1) on qubit 0."""
    B = X.shape[0]
    shape = (B,) + (2,) * n
    state = xp.zeros((B, 2 ** n), dtype=xp.complex128); state[:, 0] = 1.0
    state = state.reshape(shape)
    # data encoding: RY(pi x_i)
    for q in range(n):
        ang = np.pi * X[:, q]
        bshape = (X.shape[0],) + (1,) * (n - 1) # broadcast over the remaining qubits
        c = xp.cos(ang / 2).reshape(bshape); s = xp.sin(ang / 2).reshape(bshape)
        st = xp.moveaxis(state, q + 1, -1)
        new0 = st[..., 0] * c - st[..., 1] * s
        new1 = st[..., 0] * s + st[..., 1] * c
        st = xp.stack([new0, new1], axis=-1)
        state = xp.moveaxis(st, -1, q + 1)
    # variational layers
    for l in range(L):
        for q in range(n):
            state = _apply_1q(state, _ry(theta[l, q], xp), q, n, xp)
        for q in range(n):
            state = _apply_cnot(state, q, (q + 1) % n, n, xp)
    # <Z_0>: probability of |1> on qubit 0
    psi = state.reshape(B, 2, 2 ** (n - 1))
    p1 = (xp.abs(psi[:, 1, :]) ** 2).sum(1)
    return p1 # in [0,1]


# ---------- SPSA training (leave-one-protein-out) --------------------------
def _loss(X, y, theta, n, L, xp, w):
    p = vqc_forward(X, theta, n, L, xp)
    p = xp.clip(p, 1e-6, 1 - 1e-6)
    ce = -(w * (y * xp.log(p) + (1 - y) * xp.log(1 - p))) # class-weighted
    return float(to_numpy(ce.mean()))


def train_spsa(X, y, n, L=2, iters=150, a=0.15, c=0.1, seed=0, use_gpu=False):
    xp, _ = get_backend(use_gpu)
    rng = np.random.default_rng(seed)
    Xx = xp.asarray(X); yy = xp.asarray(y.astype(float))
    # class weights (balancing): positives are rare
    pos = max(1.0, float(y.sum())); neg = max(1.0, float((1 - y).sum()))
    w = xp.where(yy > 0.5, neg / pos, 1.0)
    theta = xp.asarray(rng.normal(0, 0.1, size=(L, n)))
    for k in range(iters):
        ak = a / (k + 1 + 10) ** 0.602; ck = c / (k + 1) ** 0.101
        delta = xp.asarray(rng.choice([-1.0, 1.0], size=(L, n)))
        lp = _loss(Xx, yy, theta + ck * delta, n, L, xp, w)
        lm = _loss(Xx, yy, theta - ck * delta, n, L, xp, w)
        ghat = (lp - lm) / (2 * ck) * delta
        theta = theta - ak * ghat
    return theta


def predict(X, theta, n, L=2, use_gpu=False):
    xp, _ = get_backend(use_gpu)
    return to_numpy(vqc_forward(xp.asarray(X), theta, n, L, xp))


# ==================== quantum kernel (QSVM-style) ========================
# ZZ feature map (Havlicek): entangled -> a genuinely quantum kernel. Training
# is classical and convex (kernel logistic) -> no barren plateau.
def _hadamard_all(state, n, xp):
    H = xp.array([[1, 1], [1, -1]], dtype=xp.complex128) / xp.sqrt(xp.asarray(2.0))
    st = state.reshape((state.shape[0],) + (2,) * n)
    for q in range(n):
        st = _apply_1q(st, H, q, n, xp)
    return st.reshape(state.shape[0], 2 ** n)


def zz_feature_states(X, reps, xp):
    """Returns statevectors phi(x) (B, 2^n) for the ZZ feature map."""
    B, n = X.shape
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)[::-1]) & 1).astype(float)
    z = xp.asarray(1.0 - 2.0 * bits) # (2^n, n) Z eigenvalues
    Xx = xp.asarray(X)
    state = xp.ones((B, 2 ** n), dtype=xp.complex128) / xp.sqrt(xp.asarray(2.0 ** n))
    zz = {}
    for i in range(n):
        for j in range(i + 1, n):
            zz[(i, j)] = z[:, i] * z[:, j]
    for _ in range(reps):
        lin = Xx @ z.T # (B, 2^n)
        quad = xp.zeros((B, 2 ** n))
        for (i, j), zij in zz.items():
            aij = (np.pi - Xx[:, i]) * (np.pi - Xx[:, j])
            quad = quad + aij[:, None] * zij[None, :]
        state = state * xp.exp(1j * (lin + quad))
        state = _hadamard_all(state, n, xp)
    return state


def quantum_kernel(Pa, Pb, xp):
    """K_ij = |<phi_i|phi_j>|^2 - a single matrix multiplication (GPU-friendly)."""
    G = Pa @ Pb.conj().T
    return to_numpy(G.real ** 2 + G.imag ** 2)


def kernel_logistic(Ktr, ytr, Kte, iters=400, lr=0.5):
    """Convex kernel logistic regression with class weights. f = K alpha + b."""
    n = Ktr.shape[0]; alpha = np.zeros(n); b = 0.0
    pos = max(1.0, ytr.sum()); neg = max(1.0, (1 - ytr).sum())
    sw = np.where(ytr > 0.5, neg / pos, 1.0)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Ktr @ alpha + b)))
        g = sw * (p - ytr)
        alpha -= lr * (Ktr @ g) / n + lr * 1e-3 * alpha # + regularization
        b -= lr * g.mean()
    return 1 / (1 + np.exp(-(Kte @ alpha + b)))
