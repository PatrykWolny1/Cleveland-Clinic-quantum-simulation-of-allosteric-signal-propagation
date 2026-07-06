#!/usr/bin/env python3
"""
LST tree (symbolic, with sin_log/thr/interf/reso primitives, fitness=AUC)
over ALL features (amplitude + phase), leave-one-protein-out + c-Myc.
  python lsttree_run.py (requires: pip install gplearn)
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, spaces, lsttree, score

ROOT = os.path.dirname(os.path.abspath(__file__)); GPU = False


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")


def _worker(arg):
    t, clean = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    D, names = spaces.all_features(apo.coords, GPU)
    D = spaces.rank_gauss(D)
    y = None
    if t.get("holo"):
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
    return t["name"], D, y, apo.keys, apo.resnames, names


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    train = [t for t in T if t.get("holo")]; test = [t for t in T if not t.get("holo")]
    with ProcessPoolExecutor(max_workers=min(4, len(T))) as ex:
        res = list(ex.map(_worker, [(t, clean) for t in (train+test)]))
    D = {r[0]: (r[1], r[2], r[3], r[4]) for r in res}; names = res[0][5]
    tr = [t["name"] for t in train]
    print(f"LST tree leave-one-protein-out (sin_log/thr/interf/reso, fitness=AUC)\n")

    aucs = []
    for these in tr:
        Xtr = np.vstack([D[n][0] for n in tr if n != these])
        ytr = np.concatenate([D[n][1] for n in tr if n != these]).astype(float)
        p, form = lsttree.fit_predict(Xtr, ytr, D[these][0], feature_names=names, gens=20, pop=2000)
        if p is None:
            print(" gplearn none: pip install gplearn"); return
        a = auc(p, D[these][1]); aucs.append(a)
        print(f" test={these:22s} AUC={a:.3f}")
        print(f" tree: {form}")
    print(f"\n mean={np.mean(aucs):.3f}")

    # final tree from 3 proteins -> c-Myc
    Xall = np.vstack([D[n][0] for n in tr]); yall = np.concatenate([D[n][1] for n in tr]).astype(float)
    for t in test:
        p, form = lsttree.fit_predict(Xall, yall, D[t["name"]][0], feature_names=names, gens=20, pop=2000)
        order = np.argsort(-p); keys, resn = D[t["name"]][2], D[t["name"]][3]
        print(f"\n tree -> {t['name']} top5: " +
              ", ".join(f"{keys[i][0]}{keys[i][1]}{resn[i]}" for i in order[:5]))
        print(f" tree: {form}")


if __name__ == "__main__":
    main()
