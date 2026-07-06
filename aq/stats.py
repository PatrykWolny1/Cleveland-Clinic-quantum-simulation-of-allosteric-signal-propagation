"""Statistical significance: p-value (Mann-Whitney) + bootstrap CI for AUC."""
from __future__ import annotations
import numpy as np
from scipy.stats import mannwhitneyu
from .score import roc_auc


def mannwhitney(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    try:
        _, p = mannwhitneyu(pos, neg, alternative="greater"); return float(p)
    except ValueError:
        return float("nan")


def bootstrap_auc_ci(scores, labels, n_boot=2000, seed=0, alpha=0.05):
    """95% CI for AUC via bootstrap over residues (stratyfikowany)."""
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]; neg = np.where(labels == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"), float("nan"))
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        pi = rng.choice(pos, len(pos), replace=True)
        ni = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([pi, ni])
        aucs[b] = roc_auc(scores[idx], labels[idx])
    lo, hi = np.percentile(aucs, [100*alpha/2, 100*(1-alpha/2)])
    return float(roc_auc(scores, labels)), float(lo), float(hi)


def significance(scores, labels, n_boot=2000, seed=0):
    auc, lo, hi = bootstrap_auc_ci(scores, labels, n_boot, seed)
    p = mannwhitney(scores, labels)
    return {"auc": auc, "ci_low": lo, "ci_high": hi, "p": p}
