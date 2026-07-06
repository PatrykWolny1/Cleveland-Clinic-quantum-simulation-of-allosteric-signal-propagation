#!/usr/bin/env python3
"""
ML/AI law discovery: Gradient Boosting + genetic symbolic regression (gplearn),
rank_gauss space, leave-one-protein-out + c-Myc.
  python mlaw_run.py (gplearn optional: pip install gplearn)
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, spaces, mlaw, score

ROOT = os.path.dirname(os.path.abspath(__file__)); GPU = False


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")


def _worker(arg):
    t, clean = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    D, names = spaces.descriptors(apo.coords, GPU)
    y = None
    if t.get("holo"):
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
    return t["name"], spaces.rank_gauss(D), y, apo.keys, apo.resnames, names


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    train = [t for t in T if t.get("holo")]; test = [t for t in T if not t.get("holo")]
    with ProcessPoolExecutor(max_workers=min(4, len(T))) as ex:
        res = list(ex.map(_worker, [(t, clean) for t in (train + test)]))
    D = {r[0]: (r[1], r[2], r[3], r[4]) for r in res}; names = res[0][5]
    tr = [t["name"] for t in train]

    print("ML/AI leave-one-protein-out (rank_gauss space)\n")
    gb, sy, imp_acc = [], [], np.zeros(len(names))
    for these in tr:
        Xtr = np.vstack([D[n][0] for n in tr if n != these])
        ytr = np.concatenate([D[n][1] for n in tr if n != these]).astype(float)
        pg, imp = mlaw.gb_fit_predict(Xtr, ytr, D[these][0]); imp_acc += imp
        ag = auc(pg, D[these][1]); gb.append(ag)
        ps, form = mlaw.symbolic_fit_predict(Xtr, ytr, D[these][0], gens=14, pop=1000,
                                             feature_names=names)
        if ps is not None:
            asy = auc(ps, D[these][1]); sy.append(asy)
            print(f" test={these:22s} GB={ag:.3f} symbolic={asy:.3f}")
            print(f" formula: {form}")
        else:
            print(f" test={these:22s} GB={ag:.3f} (gplearn not installed: pip install gplearn)")
    print(f"\n mean GB={np.mean(gb):.3f}" + (f" symbolic={np.mean(sy):.3f}" if sy else ""))
    print(" GB feature importance (top):", ", ".join(
        f"{names[i]}={imp_acc[i]/len(tr):.2f}" for i in np.argsort(-imp_acc)[:5]))

    # c-Myc: GB trained on 3 + symbolic formula
    for t in test:
        Xtr = np.vstack([D[n][0] for n in tr]); ytr = np.concatenate([D[n][1] for n in tr]).astype(float)
        pg, _ = mlaw.gb_fit_predict(Xtr, ytr, D[t["name"]][0])
        order = np.argsort(-pg); keys, resn = D[t["name"]][2], D[t["name"]][3]
        top = ", ".join(f"{keys[i][0]}{keys[i][1]}{resn[i]}" for i in order[:5])
        print(f"\n GB -> {t['name']} top5: {top}")


if __name__ == "__main__":
    main()
