#!/usr/bin/env python3
"""Audit: transferable power of each LST channel (leave-one-out), by group."""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, spaces, score

ROOT = os.path.dirname(os.path.abspath(__file__)); GPU = False
GROUP = { # channel type
 "p_low":"spectral","p_mid":"spectral","p_high":"spectral","fiedler":"spectral","eff_mass":"spectral",
 "commute":"spectral","hks5":"spectral","hks20":"spectral","hks60":"spectral","ipr":"spectral","degree":"spectral",
 "yukawa0.1":"coded","yukawa0.5":"coded","yukawa2.0":"coded",
 "interf_T10":"interf","interf_T20":"interf","interf_T40":"interf",
 "quadrature":"PHASE","phase_coh":"PHASE","gauge_coh":"PHASE","gauge_interf":"PHASE",
 "coherent":"superpos","communic":"communic"}


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")


def _worker(arg):
    t, clean = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    D, names = spaces.all_features(apo.coords, GPU); D = spaces.rank_gauss(D)
    pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
    return t["name"], D, data.label_nodes(apo, pocket), names


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    T = [t for t in yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"] if t.get("holo")]
    with ProcessPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(_worker, [(t, clean) for t in T]))
    D = {r[0]: (r[1], r[2]) for r in res}; names = res[0][3]; nm = list(D)

    print("Transferable channel power (leave-one-out, sign fixed on the training set):\n")
    rows = []
    for j, ch in enumerate(names):
        fold = []
        for these in nm:
            xtr = [(D[n][0][:, j], D[n][1]) for n in nm if n != these]
            # sign from training
            s_pos = np.mean([auc(x, y) for x, y in xtr]); s_neg = np.mean([auc(-x, y) for x, y in xtr])
            sgn = 1.0 if s_pos >= s_neg else -1.0
            fold.append(auc(sgn * D[these][0][:, j], D[these][1]))
        rows.append((ch, GROUP.get(ch, "?"), float(np.mean(fold)), fold))
    for ch, g, m, fold in sorted(rows, key=lambda r: -r[2]):
        print(f" {ch:14s} [{g:8s}] transfer={m:.3f} (KRAS={fold[0]:.2f} ABL={fold[1]:.2f} MYO={fold[2]:.2f})")
    print("\nMean per type:")
    from collections import defaultdict
    g2 = defaultdict(list)
    for ch, g, m, _ in rows: g2[g].append(m)
    for g, v in sorted(g2.items(), key=lambda x: -np.mean(x[1])):
        print(f" {g:10s}: {np.mean(v):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
