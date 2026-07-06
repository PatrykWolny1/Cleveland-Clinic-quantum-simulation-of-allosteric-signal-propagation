#!/usr/bin/env python3
"""
Szukanie space, w ktorej regula transfer jest Linear + OOF stacking.
  python spaces_run.py
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, spaces, score

ROOT = os.path.dirname(os.path.abspath(__file__))
GPU = False


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")


def _worker(arg):
    t, clean = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    D, names = spaces.descriptors(apo.coords, GPU)
    y = None
    if t.get("holo"):
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
    return t["name"], D, y, apo.keys, apo.resnames


def _rank(v):
    r = np.argsort(np.argsort(v)); return r / (len(r) - 1)


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    train = [t for t in T if t.get("holo")]; test = [t for t in T if not t.get("holo")]
    with ProcessPoolExecutor(max_workers=min(4, len(T))) as ex:
        res = list(ex.map(_worker, [(t, clean) for t in (train + test)]))
    D = {r[0]: (r[1], r[2], r[3], r[4]) for r in res}
    names = [t["name"] for t in train]

    print("Transfer Linear (ridge) i JADRO RBF w roznych spaceach (leave-one-out):\n")
    print(f" {'space':12s} {'linear/protein':32s} lin_meana rbf_meana")
    oof = {} # oof[space][protein] = prediction leave-one-out
    for sp, tf in spaces.TRANSFORMS.items():
        Xs = {n: tf(D[n][0]) for n in names}
        lin, rbf, oof[sp] = [], [], {}
        for these in names:
            Xtr = np.vstack([Xs[n] for n in names if n != these])
            ytr = np.concatenate([D[n][1] for n in names if n != these]).astype(float)
            # linear
            w = spaces.ridge_fit(Xtr, ytr); pl = Xs[these] @ w
            lin.append(auc(pl, D[these][1])); oof[sp][these] = pl
            # RBF
            gamma = 1.0 / (Xtr.shape[1])
            Ktr = spaces.rbf_kernel(Xtr, Xtr, gamma); a = spaces.kernel_ridge_fit(Ktr, ytr)
            pk = spaces.rbf_kernel(Xs[these], Xtr, gamma) @ a
            rbf.append(auc(pk, D[these][1]))
        perb = " ".join(f"{n.split('_')[0]}={a:.2f}" for n, a in zip(names, lin))
        print(f" {sp:12s} {perb:32s} {np.mean(lin):.3f} {np.mean(rbf):.3f}")

    # OOF stacking: rank-meana prediction liniowych that WSZYSTKICH space
    print("\nOOF stacking (rank-meana space liniowych):")
    st = []
    for these in names:
        blended = np.mean([_rank(oof[sp][these]) for sp in spaces.TRANSFORMS], 0)
        st.append(auc(blended, D[these][1]))
        print(f" test={these:22s} AUC={st[-1]:.3f}")
    print(f" meana={np.mean(st):.3f}")

    # best space -> prawo in c-Myc
    for t in test:
        Dc, _, keys, resn = D[t["name"]]
        # shared operator linear w rank_gauss (wyrownanie distributions)
        Xtr = np.vstack([spaces.rank_gauss(D[n][0]) for n in names])
        ytr = np.concatenate([D[n][1] for n in names]).astype(float)
        w = spaces.ridge_fit(Xtr, ytr)
        p = spaces.rank_gauss(Dc) @ w; order = np.argsort(-p)
        top = ", ".join(f"{keys[i][0]}{keys[i][1]}{resn[i]}" for i in order[:5])
        print(f"\n rank_gauss operator -> {t['name']} top5: {top}")


if __name__ == "__main__":
    main()
