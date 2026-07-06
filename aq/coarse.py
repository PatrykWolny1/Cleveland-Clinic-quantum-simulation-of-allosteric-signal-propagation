"""
Stage 2 - coarse-graining z proofem retention sygnalu topologicznego.

Cel drugorzedny challenge'u: zredukowac N residues to M super-nodes (budzet
qubitow / skalowalnosc) tak, by ZACHOWAC significant sygnal topologiczny.

Method: spectral grupowanie kontaktowego graph (embedding z pierwszych
r eigenvectors Laplacianu + k-means). Super-graph wazy edges number
kontaktow between clustermi. Result in super-graph rzutujemy z powrotem in
residues (broadcast oceny super-node).

Proof retention sygnalu:
  - spectral: pierwsze k niskich eigenvalues L_coarse sledza L_full
    (relatywny error), bo this low modes niosa dalekozasiegowa dynamike allosterii,
  - funkcjonalny: AUC(coarse) ~ AUC(full) for this samego detector.
"""
from __future__ import annotations
import numpy as np
from scipy.cluster.vq import kmeans2
from .backend import get_backend, to_numpy


def spectral_coarsen(A, n_clusters, r=None, seed=0):
    """Returns (labels[N], A_coarse[MxM], sizes[M]).
    labels[i] = indeks super-node for residues i."""
    N = A.shape[0]
    n_clusters = int(min(max(2, n_clusters), N))
    r = r or min(max(4, n_clusters), N - 1)
    deg = A.sum(1)
    L = np.diag(deg) - A
    w, V = np.linalg.eigh(L)
    emb = V[:, 1:r + 1] # pomijamy zero mode
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norm > 1e-12, norm, 1.0)
    _, labels = kmeans2(emb, n_clusters, minit="++", seed=seed)
    # scal empty clusters: przenumeruj to zwartego zakresu
    uniq = np.unique(labels)
    remap = {u: i for i, u in enumerate(uniq)}
    labels = np.array([remap[l] for l in labels], int)
    M = len(uniq)
    # super-graf: number kontaktow between clustermi
    Ac = np.zeros((M, M))
    ii, jj = np.nonzero(np.triu(A, 1))
    for i, j in zip(ii, jj):
        a, b = labels[i], labels[j]
        if a != b:
            Ac[a, b] += 1.0; Ac[b, a] += 1.0
    sizes = np.bincount(labels, minlength=M).astype(float)
    return labels, Ac, sizes


def project_up(score_coarse, labels):
    """Rzutuje ocene super-nodes in residues (broadcast)."""
    return np.asarray(score_coarse)[labels]


def spectral_preservation(A_full, A_coarse, k=6):
    """Relatywny error pierwszych k niskich (niezerowych) eigenvalues L.
    Blisko 0 => low modes (dalekozasiegowa dynamika) zachowane."""
    def low_eigs(A, k):
        L = np.diag(A.sum(1)) - A
        w = np.linalg.eigvalsh(L)
        w = w[w > 1e-9]
        w = w[:k]
        return w / w[0] if len(w) else w # normalizacja to lambda_1 (ksztalt spectrum)
    ef = low_eigs(A_full, k); ec = low_eigs(A_coarse, k)
    m = min(len(ef), len(ec))
    if m == 0:
        return float("nan")
    ef, ec = ef[:m], ec[:m]
    return float(np.mean(np.abs(ec - ef) / (ef + 1e-9)))
