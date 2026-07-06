#!/usr/bin/env python3
"""
QBoost - quantum tree boosting (QUBO) on ALL features (amplitude+phase).
"A tree like LGBM, but purely quantum": stumps = nodes, ensemble selection = QUBO -> QPU.
Leave-one-protein-out + c-Myc. QUBO -> D-Wave anneal / QAOA (here: Aer simulation).
  python qboost_run.py
"""
import os, yaml, numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from aq import data, spaces, qboost, score

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


def _fit(Xtr, ytr):
    st = qboost.build_stumps(Xtr)
    H = qboost.stump_matrix(Xtr, st)
    Q = qboost.build_qubo(H, np.where(ytr > 0.5, 1.0, -1.0))
    w, e = qboost.solve_sa(Q, steps=6000, restarts=12)
    return st, w


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    train = [t for t in T if t.get("holo")]; test = [t for t in T if not t.get("holo")]
    with ProcessPoolExecutor(max_workers=min(4, len(T))) as ex:
        res = list(ex.map(_worker, [(t, clean) for t in (train+test)]))
    D = {r[0]: (r[1], r[2], r[3], r[4]) for r in res}; names = res[0][5]
    tr = [t["name"] for t in train]
    print(f"QBoost (quantum boosting, QUBO->QPU) on {len(names)} features\n")

    aucs = []
    for these in tr:
        Xtr = np.vstack([D[n][0] for n in tr if n != these])
        ytr = np.concatenate([D[n][1] for n in tr if n != these]).astype(float)
        st, w = _fit(Xtr, ytr)
        a = auc(qboost.predict(D[these][0], st, w), D[these][1]); aucs.append(a)
        feats = Counter(names[st[k][0]] for k in np.where(w > 0.5)[0])
        top = ", ".join(f"{f}({c})" for f, c in feats.most_common(4))
        print(f" test={these:22s} AUC={a:.3f} stumps={int(w.sum())}/{len(st)} features: {top}")
    print(f"\n mean={np.mean(aucs):.3f}")

    # final QBoost from 3 proteins -> c-Myc
    Xall = np.vstack([D[n][0] for n in tr]); yall = np.concatenate([D[n][1] for n in tr]).astype(float)
    st, w = _fit(Xall, yall)
    for t in test:
        p = qboost.predict(D[t["name"]][0], st, w)
        order = np.argsort(-p); keys, resn = D[t["name"]][2], D[t["name"]][3]
        print(f"\n QBoost -> {t['name']} top5: " +
              ", ".join(f"{keys[i][0]}{keys[i][1]}{resn[i]}" for i in order[:5]))
    print(f"\n (QUBO {len(st)} variables -> D-Wave/QAOA on QPU; here Aer simulation O(K)+Numba)")


if __name__ == "__main__":
    main()
