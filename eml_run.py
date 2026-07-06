#!/usr/bin/env python3
"""
EML: for each protein a symbolic formula from the spectral data matrix, then
a law over the parameters (multi-task pooled) + leave-one-protein-out + c-Myc.
  python eml_run.py
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, eml, score

ROOT = os.path.dirname(os.path.abspath(__file__))
GPU = False
THRESH = 0.03


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")


def _worker(arg):
    t, clean = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    D, names = eml.descriptors(apo.coords, GPU)
    Phi, terms = eml.build_library(D, names)
    y = None
    if t.get("holo"):
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
    return t["name"], Phi, y, apo.keys, apo.resnames, terms


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    train = [t for t in T if t.get("holo")]; test = [t for t in T if not t.get("holo")]
    with ProcessPoolExecutor(max_workers=min(4, len(T))) as ex:
        res = list(ex.map(_worker, [(t, clean) for t in (train + test)]))
    D = {r[0]: (r[1], r[2], r[3], r[4]) for r in res}
    terms = res[0][5]
    names = [t["name"] for t in train]

    # LAW: multi-task (pooled) formula from 3 proteins
    Phi_all = np.vstack([D[n][0] for n in names])
    y_all = np.concatenate([D[n][1] for n in names]).astype(float)
    c_all = eml.stlsq(Phi_all, y_all, thresh=THRESH)
    print("LAW (pooled EML, shared formula):")
    print(" y ~ " + eml.formula_str(c_all, terms, top=8))

    # per-protein formulas (same form, different parameters?)
    print("\nPer-protein formulas (to compare parameters):")
    cs = {}
    for n in names:
        cs[n] = eml.stlsq(D[n][0], D[n][1].astype(float), thresh=THRESH)
        print(f" {n[:20]:20s}: {eml.formula_str(cs[n], terms, top=4)}")
    # which terms are shared across all (core of the law)
    active = [set(np.where(np.abs(cs[n]) > 1e-6)[0]) for n in names]
    common = set.intersection(*active) if active else set()
    print(" terms shared across ALL: " + (", ".join(terms[i] for i in common) if common else "none"))

    # leave-one-protein-out
    print("\nLeave-one-protein-out (formula from 2 proteins -> 3rd):")
    aucs = []
    for these in names:
        Phitr = np.vstack([D[n][0] for n in names if n != these])
        ytr = np.concatenate([D[n][1] for n in names if n != these]).astype(float)
        ct = eml.stlsq(Phitr, ytr, thresh=THRESH)
        a = auc(D[these][0] @ ct, D[these][1]); aucs.append(a)
        print(f" test={these:22s} AUC={a:.3f}")
    print(f" mean={np.mean(aucs):.3f}")

    # c-Myc via the shared law
    for t in test:
        Phi, _, keys, resn = D[t["name"]]
        p = Phi @ c_all; order = np.argsort(-p)
        top = ", ".join(f"{keys[i][0]}{keys[i][1]}{resn[i]}" for i in order[:5])
        print(f"\n EML law -> {t['name']} top5: {top}")


if __name__ == "__main__":
    main()
