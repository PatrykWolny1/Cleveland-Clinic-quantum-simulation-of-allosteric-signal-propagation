#!/usr/bin/env python3
"""
LST-QNN leave-one-protein-out: bank features LST + glowica (quantum kernel / logistic).
Comparison z single best feature LST.
  python lstnet_run.py
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, lstnet, qnn, score
from aq.backend import get_backend

ROOT = os.path.dirname(os.path.abspath(__file__))
GPU = False


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")


def logistic(Xtr, ytr, Xte, iters=600, lr=0.3):
    w = np.zeros(Xtr.shape[1]); b = 0.0
    pos = max(1.0, ytr.sum()); neg = max(1.0, (1 - ytr).sum())
    sw = np.where(ytr > 0.5, neg / pos, 1.0)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = sw * (p - ytr)
        w -= lr * (Xtr.T @ g) / len(ytr); b -= lr * g.mean()
    return 1 / (1 + np.exp(-(Xte @ w + b)))


def _predict_cmyc(D, names_t, xp):
    import json, csv
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    out = os.path.join(ROOT, base["paths"]["outputs"]); os.makedirs(out, exist_ok=True)
    tg = [t for t in yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
          if not t.get("holo")]
    if not tg:
        return
    Xtr = np.vstack([D[n][0] for n in names_t])
    ytr = np.concatenate([D[n][1] for n in names_t]).astype(float)
    for t in tg:
        apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
        Xc, _, _ = lstnet.lst_features(apo.coords)
        pl = logistic(Xtr, ytr, Xc) # sieć LST wyuczona in 3 walidacyjnych
        order = np.argsort(-pl)
        hits = [{"rank": r + 1, "chain": apo.keys[i][0], "resseq": int(apo.keys[i][1]),
                 "resname": apo.resnames[i], "score": float(pl[i])} for r, i in enumerate(order[:5])]
        with open(os.path.join(out, f"{t['name']}_hitlist_lstnet.json"), "w") as f:
            json.dump(hits, f, indent=2)
        top = ", ".join(f"{h['chain']}{h['resseq']}{h['resname']}" for h in hits)
        print(f"\n LST-net -> {t['name']} top5: {top}")


def _feat_worker(arg):
    """Extraction features for one protein (poziom modulu -> picklowalne for MP)."""
    t, clean, gpu = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    X, _, names = lstnet.lst_features(apo.coords, gpu)
    pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
    return t["name"], X, data.label_nodes(apo, pocket), names


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    targets = [t for t in yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
               if t.get("holo")]
    D = {}; names = None
    args = [(t, clean, GPU) for t in targets]
    with ProcessPoolExecutor(max_workers=min(3, len(targets))) as ex:
        for nm, X, y, nms in ex.map(_feat_worker, args):
            D[nm] = (X, y); names = nms
    kidx = [i for i, n in enumerate(names) if n.startswith("interf")] # kernel: interference LST
    names_t = list(D); xp, _ = get_backend(GPU)
    F = D[names_t[0]][0].shape[1]
    print(f"LST-QNN leave-one-protein-out ({F} cech: 4 mechanismy LST + structurelne)\n")
    qk, lg, bl, bn = [], [], [], []
    for these in names_t:
        Xtr = np.vstack([D[n][0] for n in names_t if n != these])
        ytr = np.concatenate([D[n][1] for n in names_t if n != these]).astype(float)
        Xte, yte = D[these]
        Ptr = qnn.zz_feature_states(Xtr[:, kidx], 2, xp); Pte = qnn.zz_feature_states(Xte[:, kidx], 2, xp)
        Ktr = qnn.quantum_kernel(Ptr, Ptr, xp); Kte = qnn.quantum_kernel(Pte, Ptr, xp)
        pk = qnn.kernel_logistic(Ktr, ytr, Kte)
        pl = logistic(Xtr, ytr, Xte)
        # BLEND: umeanona ranga (logistic + kernel + best single feature treningowa)
        def _rankpct(v):
            r = np.argsort(np.argsort(v)); return r / (len(r) - 1)
        # wybierz best single feature in TRENINGU (leakage-free)
        jb = max(range(Xtr.shape[1]), key=lambda j: max(auc(Xtr[:, j], ytr), auc(-Xtr[:, j], ytr)))
        sgn = 1.0 if auc(Xtr[:, jb], ytr) >= auc(-Xtr[:, jb], ytr) else -1.0
        blend = (_rankpct(pl) + _rankpct(pk) + _rankpct(sgn * Xte[:, jb])) / 3
        blend_noqk = (_rankpct(pl) + _rankpct(sgn * Xte[:, jb])) / 2
        bf = max(max(auc(Xte[:, j], yte), auc(-Xte[:, j], yte)) for j in range(Xte.shape[1]))
        qk.append(auc(pk, yte)); lg.append(auc(pl, yte)); bl.append(auc(blend, yte)); bn.append(auc(blend_noqk, yte))
        print(f" test={these:22s} QKernel={qk[-1]:.3f} logistic={lg[-1]:.3f} "
              f"BLEND={bl[-1]:.3f} BLEND_noQK={bn[-1]:.3f} best_single={bf:.3f}")
    print(f"\n meana: QKernel={np.mean(qk):.3f} logistic={np.mean(lg):.3f} "
          f"BLEND={np.mean(bl):.3f} BLEND_noQK={np.mean(bn):.3f}")
    print(f" -> wklad quantumgo kernela to BLEND: {np.mean(bl)-np.mean(bn):+.3f}")
    _predict_cmyc(D, names_t, xp)


if __name__ == "__main__":
    main()
