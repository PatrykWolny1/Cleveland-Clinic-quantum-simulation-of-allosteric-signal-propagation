"""
Router pocket geometry: routes protein to proper predictor.

Unsupervised. Decisiveness detector = kurtosis distribution score
(sharp peak = confident detector). Candidates:
  - LST_logf_interf (distal) - always,
  - gnm_zabs (proximal) - when detected active site (cofactor).
Select most decisive -> naturally proximal protein (KRAS: GDP)
go to gnm_zabs, distal (ABL/myosin: no cofactor) to LST.
"""
from __future__ import annotations
import numpy as np


def decisiveness(score):
    s = np.asarray(score, float)
    s = s[np.isfinite(s)]
    if len(s) < 4 or s.std() < 1e-12:
        return 0.0
    z = (s - s.mean()) / s.std()
    return float((z ** 4).mean()) # kurtosis (peak sharpness)


def route(candidates):
    """candidates: dict name->score. Returns (name, score, regime)."""
    name = max(candidates, key=lambda k: decisiveness(candidates[k]))
    regime = "proximal" if name.startswith("gnm") or "active" in name else "distal"
    return name, candidates[name], regime
