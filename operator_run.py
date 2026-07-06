#!/usr/bin/env python3
"""
Operator transfer: 3 pary -> prawo -> 4. case.
  - operator per protein + ich agreement (whether one prawo),
  - leave-one-protein-out: operator z 2 proteins in 3.,
  - prawo (filtr sigma-T) i prediction c-Myc.
  python operator_run.py
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, operator, score

ROOT = os.path.dirname(os.path.abspath(__file__))
GPU = False


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")


def _sig(arg):
    t, clean = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    S, _ = operator.signal_matrix(apo.coords, GPU)
    names = operator.signal_matrix.names
    keys, resn = apo.keys, apo.resnames
    if t.get("holo"):
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
    else:
        y = None
    return t["name"], S, y, keys, resn, names


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    train = [t for t in T if t.get("holo")]; test = [t for t in T if not t.get("holo")]

    with ProcessPoolExecutor(max_workers=min(4, len(T))) as ex:
        res = list(ex.map(_sig, [(t, clean) for t in (train + test)]))
    D = {r[0]: r[1:5] for r in res}
    chan = res[0][5]
    names = [t["name"] for t in train]

    # operatory per protein + agreement
    Ws = [operator.fit_operator(D[n][0], D[n][1].astype(float), ridge=2.0) for n in names]
    Agr = operator.operator_agreement(Ws)
    print("Agreement operatorow (cos-podobienstwo, 1=identyczne prawo):")
    for i, n in enumerate(names):
        print(" " + n[:20].ljust(20) + " " + " ".join(f"{Agr[i,j]:+.2f}" for j in range(len(names))))

    # leave-one-protein-out: operator z 2 -> test in 3.
    print("\nLeave-one-protein-out (operator z 2 par in 3.):")
    aucs = []
    for these in names:
        Str = np.vstack([D[n][0] for n in names if n != these])
        ytr = np.concatenate([D[n][1] for n in names if n != these]).astype(float)
        W = operator.fit_operator(Str, ytr, ridge=2.0)
        p = operator.apply_operator(D[these][0], W)
        a = auc(p, D[these][1]); aucs.append(a)
        print(f" test={these:22s} AUC={a:.3f}")
    print(f" meana={np.mean(aucs):.3f}")

    # DIAGNOSTYKA: os dzielaca most anty-consistent pare
    n = len(names)
    ij = min([(i, j) for i in range(n) for j in range(i + 1, n)], key=lambda p: Agr[p])
    print(f"\nNajbardziej rozbiezna para: {names[ij[0]]} vs {names[ij[1]]} (cos={Agr[ij]:+.2f})")
    diff = np.abs(Ws[ij[0]] / (np.linalg.norm(Ws[ij[0]]) + 1e-9)
                  - Ws[ij[1]] / (np.linalg.norm(Ws[ij[1]]) + 1e-9))
    top = np.argsort(-diff)[:5]
    print(" channels najsilniej rozdzielajace: " + ", ".join(f"{chan[k]}({diff[k]:.2f})" for k in top))

    # PRAWO: shared operator z 3 par + top channels
    Wall = operator.fit_operator(np.vstack([D[n][0] for n in names]),
                                 np.concatenate([D[n][1] for n in names]).astype(float), ridge=2.0)
    print("\nPRAWO (shared operator, top-8 channels |weights|):")
    top = np.argsort(-np.abs(Wall))[:8]
    for k in top:
        print(f" {chan[k]:16s} W={Wall[k]:+.3f}")

    # prediction c-Myc wspolnym operatorem
    for t in test:
        S, _, keys, resn = D[t["name"]]
        p = operator.apply_operator(S, Wall); order = np.argsort(-p)
        top = ", ".join(f"{keys[i][0]}{keys[i][1]}{resn[i]}" for i in order[:5])
        print(f"\n operator -> {t['name']} top5: {top}")


if __name__ == "__main__":
    main()
