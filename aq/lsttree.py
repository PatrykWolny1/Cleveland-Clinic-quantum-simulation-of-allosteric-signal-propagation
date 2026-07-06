"""
Drzewo LST: symbolic regression z prymitywami fizyki (instead of generycznych).

Zestaw funkcji = mechanismy LST:
  sin_log(x) = sin(log|x|) - REZONANS (log) + INTERFERENCJA (sin)
  thr(x) = sigmoid(k x) - Threshold amplitudes/phase
  interf(x,y)= cos(x - y) - INTERFERENCJA dwoch sygnalow (difference faz)
  reso(x,y) = exp(-(x-y)^2) - REZONANS (proximity -> significance)
  + add, sub, mul (Superposition)
Input: channels amplitude I phase (interf, quadrature, phase_coh, gauge).
Fit: wprost AUC (not MAE) -> drzewo learns rozrozniac, not zgadywac.
"""
from __future__ import annotations
import numpy as np
from .score import roc_auc


def _make():
    from gplearn.functions import make_function
    def sin_log(x):
        return np.sin(np.log(np.abs(x) + 1e-6))
    def thr(x):
        return 1.0 / (1.0 + np.exp(-np.clip(4.0 * x, -30, 30)))
    def interf(x, y):
        return np.cos(x - y)
    def reso(x, y):
        return np.exp(-np.clip((x - y) ** 2, 0, 30))
    return [make_function(function=sin_log, name="sinlog", arity=1),
            make_function(function=thr, name="thr", arity=1),
            make_function(function=interf, name="interf", arity=2),
            make_function(function=reso, name="reso", arity=2)]


def _auc_fitness():
    from gplearn.fitness import make_fitness
    def _a(y, y_pred, w):
        if np.std(y_pred) < 1e-12:
            return 0.5
        return roc_auc(np.asarray(y_pred, float), np.asarray(y, int))
    return make_fitness(function=_a, greater_is_better=True)


def fit_predict(Xtr, ytr, Xte, feature_names=None, gens=20, pop=1500, seed=0):
    """Returns (predykcja_te, wzor_str) or (None,None) when none gplearn."""
    try:
        from gplearn.genetic import SymbolicRegressor
    except Exception:
        return None, None
    fset = ["add", "sub", "mul"] + _make()
    est = SymbolicRegressor(
        population_size=pop, generations=gens, function_set=fset,
        metric=_auc_fitness(), parsimony_coefficient=0.0008,
        p_crossover=0.7, p_subtree_mutation=0.12, p_hoist_mutation=0.05,
        p_point_mutation=0.1, max_samples=0.9, random_state=seed,
        n_jobs=-1, feature_names=feature_names) # n_jobs=-1: MP (CPU)
    est.fit(Xtr, ytr)
    return est.predict(Xte), str(est._program)
