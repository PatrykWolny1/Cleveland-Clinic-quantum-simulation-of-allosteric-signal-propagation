"""
Stage 7 - evaluation harness (shared across ALL configurations).

Goal of the main challenge: the algorithm must assign significantly higher
scores to the known regulatory residues (holo pocket) than to the background.

Metrics:
  roc_auc : whether the ranking separates pocket from background (0.5 = random)
  mwu_p : Mann-Whitney U test, p-value (pocket > background?)
  hit@k : whether any pocket residue landed in the top-k
  enrichment@k : (fraction of pocket in top-k) / (baseline fraction)
"""
from __future__ import annotations
import numpy as np
from scipy.stats import mannwhitneyu


def roc_auc(score, labels):
    """AUC = P(score_pos > score_neg). Equivalent to the U statistic."""
    labels = np.asarray(labels)
    pos = score[labels == 1]
    neg = score[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average the ranks over ties
    _tie_correct(score, ranks)
    r_pos = ranks[labels == 1].sum()
    n1, n0 = len(pos), len(neg)
    u = r_pos - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def _tie_correct(score, ranks):
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1


def evaluate(score, order, labels, ks=(5, 10, 20)):
    labels = np.asarray(labels)
    n = len(labels)
    n_pos = int(labels.sum())
    res = {"n_nodes": n, "n_pocket": n_pos}
    if n_pos == 0:
        res["note"] = "no ground-truth (e.g. c-Myc without holo) - hit-list only"
        return res

    res["roc_auc"] = round(roc_auc(np.asarray(score, float), labels), 4)
    pos = np.asarray(score)[labels == 1]
    neg = np.asarray(score)[labels == 0]
    try:
        _, p = mannwhitneyu(pos, neg, alternative="greater")
        res["mwu_p"] = float(f"{p:.3e}")
    except ValueError:
        res["mwu_p"] = float("nan")

    base = n_pos / n
    ranked_labels = labels[order]
    for k in ks:
        if k > n:
            continue
        hit = int(ranked_labels[:k].sum())
        res[f"hit@{k}"] = int(hit > 0)
        res[f"n_hits@{k}"] = hit
        res[f"enrichment@{k}"] = round((hit / k) / base, 3) if base > 0 else float("nan")
    return res
