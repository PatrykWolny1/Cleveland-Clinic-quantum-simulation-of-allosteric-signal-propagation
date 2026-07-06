"""
Consensus rankingow (rank aggregation).

Different metric wykrywaja different typy pocket (proximal/distal/anomalne).
Tryby agregacji:
  mean : meana ranga (Borda) - stable, but rozmywa if term weak
  min : best ranga (best-of) - residue wskazana via KTORYKOLWIEK
           silny detector wyplywa in gore; proper for heterogenicznych
           pocket (each typ lapie other detector)
  median : robust in single anty-skorelowany term
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
    """Returns (consensus_order, score). score wyzej = better.
    mode: min(best-of) | mean(Borda) | median | rrf(reciprocal rank fusion).
    RRF: score_i = sum_m 1/(k + rank_m(i)) - residue high u KTOREGOKOLWIEK
    czlonka wyplywa, but without dominacji one; robust in term anty-skorelowany."""
    R = _rank_matrix(orders, n_nodes)
    if mode == "min":
        agg = R.min(0); return np.argsort(agg).astype(int), -agg
    if mode == "median":
        agg = np.median(R, 0); return np.argsort(agg).astype(int), -agg
    if mode == "rrf":
        fused = (1.0 / (rrf_k + R)).sum(0) # wyzej = better
        return np.argsort(-fused).astype(int), fused
    agg = R.mean(0); return np.argsort(agg).astype(int), -agg


# agreement wstecz
def borda(orders, n_nodes):
    return aggregate(orders, n_nodes, "mean")
