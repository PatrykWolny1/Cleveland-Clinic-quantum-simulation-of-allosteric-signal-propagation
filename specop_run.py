#!/usr/bin/env python3
"""
Spectral operator from the data matrix (not from features): 3 pairs -> law g(lambda) -> the 4th.
  python specop_run.py
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, specop, score

ROOT = os.path.dirname(os.path.abspath(__file__))
GPU = False


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")


def _worker(arg):
    t, clean = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    P = specop.modal_participation(apo.coords, gpu=GPU)
    y = None
    if t.get("holo"):
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
    return t["name"], P, y, apo.keys, apo.resnames


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    train = [t for t in T if t.get("holo")]; test = [t for t in T if not t.get("holo")]
    with ProcessPoolExecutor(max_workers=min(4, len(T))) as ex:
        res = list(ex.map(_worker, [(t, clean) for t in (train + test)]))
    D = {r[0]: r[1:] for r in res}
    names = [t["name"] for t in train]

    Ws = [specop.fit_operator(D[n][0], D[n][1].astype(float), ridge=1.0) for n in names]
    Agr = specop.agreement(Ws)
    print("Agreement of spectral operators (1=identical law):")
    for i, n in enumerate(names):
        print(" " + n[:20].ljust(20) + " " + " ".join(f"{Agr[i,j]:+.2f}" for j in range(len(names))))

    print("\nLeave-one-protein-out (operator from 2 pairs -> the 3rd):")
    aucs = []
    for these in names:
        Ptr = np.vstack([D[n][0] for n in names if n != these])
        ytr = np.concatenate([D[n][1] for n in names if n != these]).astype(float)
        W = specop.fit_operator(Ptr, ytr, ridge=1.0)
        a = auc(specop.apply_operator(D[these][0], W), D[these][1]); aucs.append(a)
        print(f" test={these:22s} AUC={a:.3f}")
    print(f" mean={np.mean(aucs):.3f}")

    Wall = specop.fit_operator(np.vstack([D[n][0] for n in names]),
                               np.concatenate([D[n][1] for n in names]).astype(float), ridge=1.0)
    print("\nLAW g(lambda) (weight per spectral band, band 0 = lowest frequencies / global modes):")
    for b in range(len(Wall)):
        bar = "#" * int(abs(Wall[b]) / (np.abs(Wall).max() + 1e-9) * 20)
        print(f" band {b:2d}: {Wall[b]:+.3f} {bar}")

    for t in test:
        P, _, keys, resn = D[t["name"]]
        p = specop.apply_operator(P, Wall); order = np.argsort(-p)
        top = ", ".join(f"{keys[i][0]}{keys[i][1]}{resn[i]}" for i in order[:5])
        print(f"\n operator -> {t['name']} top5: {top}")


if __name__ == "__main__":
    main()
