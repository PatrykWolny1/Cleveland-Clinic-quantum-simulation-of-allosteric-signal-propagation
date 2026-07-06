"""
LST tree: symbolic regression with physics primitives (instead of generic ones).

The function set = LST mechanisms:
  sin_log(x) = sin(log|x|) - RESONANCE (log) + INTERFERENCE (sin)
  thr(x) = sigmoid(k x) - threshold on amplitude/phase
  interf(x,y)= cos(x - y) - INTERFERENCE of two signals (phase difference)
  reso(x,y) = exp(-(x-y)^2) - RESONANCE (proximity -> significance)
  + add, sub, mul (superposition)
Input: amplitude and phase channels (interf, quadrature, phase_coh, gauge).
Fit: directly on AUC (not MAE) -> the tree learns to discriminate, not to guess.
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
    """Returns (te_prediction, formula_str) or (None, None) if gplearn is missing."""
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
        n_jobs=-1, feature_names=feature_names) # n_jobs=-1: multiprocessing (CPU)
    est.fit(Xtr, ytr)
    return est.predict(Xte), str(est._program)
