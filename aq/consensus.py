"""
Consensus of rankings (rank aggregation).

Different metrics detect different pocket types (proximal/distal/anomalous).
Aggregation modes:
  mean : mean rank (Borda) - stable, but blurs when a term is weak
  min : best rank (best-of) - a residue flagged by ANY strong
           detector rises to the top; suitable for heterogeneous
           pockets (each type is caught by a different detector)
  median : robust to a single anti-correlated term
"""
from __future__ import annotations
import numpy as np


def _rank_matrix(orders, n_nodes):
    R = np.full((len(orders), n_nodes), float(n_nodes))
    for r, order in enumerate(orders):
        for pos, idx in enumerate(order):
            R[r, int(idx)] = pos
    return R


def aggregate(orders, n_nodes, mode="min", rrf_k=60.0):
    """Returns (consensus_order, score). Higher score = better.
    mode: min(best-of) | mean(Borda) | median | rrf(reciprocal rank fusion).
    RRF: score_i = sum_m 1/(k + rank_m(i)) - a residue ranked high by ANY
    member rises, without one member dominating; robust to an anti-correlated term."""
    R = _rank_matrix(orders, n_nodes)
    if mode == "min":
        agg = R.min(0); return np.argsort(agg).astype(int), -agg
    if mode == "median":
        agg = np.median(R, 0); return np.argsort(agg).astype(int), -agg
    if mode == "rrf":
        fused = (1.0 / (rrf_k + R)).sum(0) # higher = better
        return np.argsort(-fused).astype(int), fused
    agg = R.mean(0); return np.argsort(agg).astype(int), -agg


# backward compatibility
def borda(orders, n_nodes):
    return aggregate(orders, n_nodes, "mean")
