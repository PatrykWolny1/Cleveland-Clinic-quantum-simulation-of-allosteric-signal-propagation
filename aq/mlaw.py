"""
Law discovery via ML/AI methods (stronger than linear STLSQ).

  1) Gradient Boosting (sklearn) - a nonlinear model with feature importances.
  2) Genetic symbolic regression (gplearn) - EVOLUTIONARY search over
     expression trees -> an actual FORMULA (true "AI equation discovery").
rank_gauss space (distribution equalization), leave-one-protein-out evaluation.
Class weights (pockets are rare). ML runs on CPU (light: 3 proteins) - the
numeric core (eig/kernels) still runs under MP/GPU.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def _weights(y):
    pos = max(1.0, y.sum()); neg = max(1.0, (1 - y).sum())
    return np.where(y > 0.5, neg / pos, 1.0)


def gb_fit_predict(Xtr, ytr, Xte, seed=0):
    m = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                  subsample=0.8, random_state=seed)
    m.fit(Xtr, ytr, sample_weight=_weights(ytr))
    return m.predict(Xte), m.feature_importances_


def symbolic_fit_predict(Xtr, ytr, Xte, seed=0, gens=12, pop=800, feature_names=None):
    """gplearn SymbolicRegressor -> (prediction, formula_str). None if gplearn is missing."""
    try:
        from gplearn.genetic import SymbolicRegressor
    except Exception:
        return None, None
    est = SymbolicRegressor(
        population_size=pop, generations=gens,
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg"),
        metric="mean absolute error", parsimony_coefficient=0.01,
        p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
        p_point_mutation=0.1, max_samples=0.9, random_state=seed, n_jobs=1,
        feature_names=feature_names)
    est.fit(Xtr, ytr, sample_weight=_weights(ytr))
    return est.predict(Xte), str(est._program)
