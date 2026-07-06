"""
Stage 6 - ranking -> hit-list.

Tryby:
  fluctuation : transfer to residues w band distance [d_min,d_max]
  active_site : connectivity to active site (orthosteric), optionally
                znormalizowana PER-POWLOKA distance grafowej (z-score):
                residue jest allosteryczna, when has anomalnie High transfer
                to active-site as in swoja distance -> ukryty channel.

z-score per-powloka jest proper poprawka in dystalnosc pocket
(surowa connectivity smallje z distancea i gubi dalekie pockety).
Statystyki powlok liczone kernelem Numba (quickly for largech N).
"""
from __future__ import annotations
import numpy as np
from .backend import njit


@njit(cache=True)
def _shell_stats(shell, raw, n_shell):
    """meana i odch.std raw w obrebie powlok (shell in 0..n_shell-1)."""
    s = np.zeros(n_shell); s2 = np.zeros(n_shell); cnt = np.zeros(n_shell)
    for i in range(raw.shape[0]):
        k = shell[i]
        if k < 0:
            continue
        s[k] += raw[i]; s2[k] += raw[i] * raw[i]; cnt[k] += 1.0
    mean = np.zeros(n_shell); std = np.zeros(n_shell)
    for k in range(n_shell):
        if cnt[k] > 0:
            mean[k] = s[k] / cnt[k]
            var = s2[k] / cnt[k] - mean[k] * mean[k]
            std[k] = np.sqrt(var) if var > 1e-12 else 0.0
    return mean, std, cnt


def _zscore_by_shell(raw, shell):
    n_shell = int(shell.max()) + 1 if shell.size and shell.max() >= 0 else 1
    mean, std, cnt = _shell_stats(shell.astype(np.int64),
                                  raw.astype(np.float64), n_shell)
    z = np.zeros_like(raw, dtype=float)
    for i in range(len(raw)):
        k = shell[i]
        if k >= 0 and std[k] > 0:
            z[i] = (raw[i] - mean[k]) / std[k]
    return z


def smooth_score(score, A, iters=1, beta=0.5):
    """Wygladza score over graph kontaktow (label propagation):
    residue w klastrze high results zostaje wzmocniona, izolowany hub
    rozmyty. Allosteric pocket jest spacenym KLASTREM, so this
    podnosi precyzje szczytu i konsoliduje przewidziane pockety."""
    deg = A.sum(1); deg[deg == 0] = 1.0
    s = score.astype(float).copy()
    for _ in range(int(iters)):
        s = (1.0 - beta) * s + beta * (A @ s) / deg
    return s


def burial_weight(A, gamma=1.0):
    """Weight powierzchniowa: residues o niskim degreesu kontaktow (powierzchnia)
    dostaja wieksza wage; zagrzebane w rdzeniu (high degree) - mniejsza.
    Druggable allosteric pockets are available for rozpuszczalnika."""
    deg = A.sum(1).astype(float)
    if deg.max() > deg.min():
        d = (deg - deg.min()) / (deg.max() - deg.min())
    else:
        d = np.zeros_like(deg)
    return (1.0 - d) ** gamma


def _finalize(score, exclude_idx):
    s = score.astype(float).copy()
    if exclude_idx:
        s[np.asarray(list(exclude_idx), int)] = -np.inf
    order = np.asarray([int(i) for i in np.argsort(-s) if np.isfinite(s[i])])
    return order, score


def rank_fluctuation(M, gdist=None, d_min=3, d_max=None, exclude_idx=None,
                     A=None, smooth=0, beta=0.5, burial=0):
    W = M.astype(float).copy()
    np.fill_diagonal(W, 0.0)
    if gdist is not None:
        band = np.isfinite(gdist) & (gdist >= d_min)
        if d_max is not None:
            band &= (gdist <= d_max)
        W = W * band
    score = W.sum(1)
    if A is not None and smooth:
        score = smooth_score(score, A, smooth, beta)
    if A is not None and burial:
        score = score * burial_weight(A, burial)
    return _finalize(score, exclude_idx)


def rank_active_site(M, active_idx, gdist=None, normalize="zscore",
                     exclude_idx=None, A=None, smooth=0, beta=0.5, burial=0):
    """score_i = connectivity to active-site; normalize: 'zscore'|'distance'|'none'."""
    a = np.asarray(active_idx, int)
    raw = M[:, a].sum(1)
    if normalize in ("zscore", "zabs") and gdist is not None:
        dmin = gdist[:, a]
        dmin = np.where(np.isfinite(dmin), dmin, -1).min(1) # min odl. to active
        shell = np.where(dmin >= 0, np.round(dmin).astype(int), -1)
        z = _zscore_by_shell(raw, shell)
        score = np.abs(z) if normalize == "zabs" else z
    elif normalize == "distance" and gdist is not None:
        d = gdist[:, a].astype(float); d[~np.isfinite(d)] = 0.0
        score = (M[:, a] * d).sum(1)
    else:
        score = raw
    if A is not None and smooth:
        score = smooth_score(score, A, smooth, beta)
    if A is not None and burial:
        score = score * burial_weight(A, burial)
    ex = set(int(x) for x in a) | set(int(x) for x in (exclude_idx or []))
    return _finalize(score, ex)


def hit_list(order, keys, resnames, score, top=5):
    out = []
    for rank, i in enumerate(order[:top], 1):
        ch, resseq, icode = keys[i]
        out.append({"rank": rank, "node_index": int(i), "chain": ch,
                    "resseq": int(resseq), "icode": icode,
                    "resname": resnames[i], "score": float(score[i])})
    return out
