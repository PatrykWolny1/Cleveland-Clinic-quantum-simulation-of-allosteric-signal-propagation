"""
Stage 7 - harness oceny (shared for WSZYSTKICH konfiguracji).

Cel main challenge'u: algorytm has przypisywac istotnie wyzsze results
znanym resztom regulatorowym (pocket holo) niz tlu.

Metryki:
  roc_auc : whether ranking odroznia pocket from residues (0.5 = losowo)
  mwu_p : test Manna-Whitneya, p-value (pocket > tlo ?)
  hit@k : whether jakas residue pocket trafila w top-k
  enrichment@k : (frakcja pocket w top-k) / (frakcja bazowa)
"""
from __future__ import annotations
import numpy as np
from scipy.stats import mannwhitneyu


def roc_auc(score, labels):
    """AUC = P(score_pos > score_neg). Rownowaznik statystyki U."""
    labels = np.asarray(labels)
    pos = score[labels == 1]
    neg = score[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # umeannie rang at remisach
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
        res["note"] = "none ground-truth (np. c-Myc without holo) - only hit-list"
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
