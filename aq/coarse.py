"""
Stage 2 - coarse-graining with a proof of topological-signal retention.

Secondary goal of the challenge: reduce N residues to M super-nodes (qubit
budget / scalability) while PRESERVING the significant topological signal.

Method: spectral clustering of the contact graph (embedding from the first
r eigenvectors of the Laplacian + k-means). The super-graph weights edges by the
number of contacts between clusters. The result on the super-graph is projected
back onto residues (broadcast of the super-node score).

Proof of signal retention:
  - spectral: the first k low eigenvalues of L_coarse track L_full
    (relative error), because these low modes carry the long-range allosteric dynamics,
  - functional: AUC(coarse) ~ AUC(full) for the same detector.
"""
from __future__ import annotations
import numpy as np
from scipy.cluster.vq import kmeans2
from .backend import get_backend, to_numpy


def spectral_coarsen(A, n_clusters, r=None, seed=0):
    """Returns (labels[N], A_coarse[MxM], sizes[M]).
    labels[i] = super-node index for residue i."""
    N = A.shape[0]
    n_clusters = int(min(max(2, n_clusters), N))
    r = r or min(max(4, n_clusters), N - 1)
    deg = A.sum(1)
    L = np.diag(deg) - A
    w, V = np.linalg.eigh(L)
    emb = V[:, 1:r + 1] # skip the zero mode
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norm > 1e-12, norm, 1.0)
    _, labels = kmeans2(emb, n_clusters, minit="++", seed=seed)
    # merge empty clusters: renumber into a compact range
    uniq = np.unique(labels)
    remap = {u: i for i, u in enumerate(uniq)}
    labels = np.array([remap[l] for l in labels], int)
    M = len(uniq)
    # super-graph: number of contacts between clusters
    Ac = np.zeros((M, M))
    ii, jj = np.nonzero(np.triu(A, 1))
    for i, j in zip(ii, jj):
        a, b = labels[i], labels[j]
        if a != b:
            Ac[a, b] += 1.0; Ac[b, a] += 1.0
    sizes = np.bincount(labels, minlength=M).astype(float)
    return labels, Ac, sizes


def project_up(score_coarse, labels):
    """Projects the super-node score onto residues (broadcast)."""
    return np.asarray(score_coarse)[labels]


def spectral_preservation(A_full, A_coarse, k=6):
    """Relative error of the first k low (non-zero) eigenvalues of L.
    Close to 0 => the low modes (long-range dynamics) are preserved."""
    def low_eigs(A, k):
        L = np.diag(A.sum(1)) - A
        w = np.linalg.eigvalsh(L)
        w = w[w > 1e-9]
        w = w[:k]
        return w / w[0] if len(w) else w # normalize to lambda_1 (spectrum shape)
    ef = low_eigs(A_full, k); ec = low_eigs(A_coarse, k)
    m = min(len(ef), len(ec))
    if m == 0:
        return float("nan")
    ef, ec = ef[:m], ec[:m]
    return float(np.mean(np.abs(ec - ef) / (ef + 1e-9)))
