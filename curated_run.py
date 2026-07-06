#!/usr/bin/env python3
"""
Curation features (remove ballast) + marginalny wklad FAZY: GB amplitude vs +phase.
Rozstrzyga, ile phase channels dokladaja (zwlaszcza KRAS). Leave-one-out.
  python curated_run.py
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, spaces, mlaw, score

ROOT = os.path.dirname(os.path.abspath(__file__)); GPU = False
BALAST = {"coherent", "communic", "interf_T40"} # transfer < losowy (audyt)
PHASE = {"interf_T10", "interf_T20", "quadrature", "phase_coh", "gauge_coh", "gauge_interf"}


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")


def _worker(arg):
    t, clean = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    D, names = spaces.all_features(apo.coords, GPU); D = spaces.rank_gauss(D)
    y = None
    if t.get("holo"):
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
    return t["name"], D, y, apo.keys, apo.resnames, names


def _loo(D, tr, idx):
    aucs = {}
    for these in tr:
        Xtr = np.vstack([D[n][0][:, idx] for n in tr if n != these])
        ytr = np.concatenate([D[n][1] for n in tr if n != these]).astype(float)
        p, _ = mlaw.gb_fit_predict(Xtr, ytr, D[these][0][:, idx])
        aucs[these] = auc(p, D[these][1])
    return aucs


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    train = [t for t in T if t.get("holo")]; test = [t for t in T if not t.get("holo")]
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(_worker, [(t, clean) for t in (train+test)]))
    D = {r[0]: (r[1], r[2], r[3], r[4]) for r in res}; names = res[0][5]
    tr = [t["name"] for t in train]

    keep = [i for i, n in enumerate(names) if n not in BALAST] # curation
    amp = [i for i in keep if names[i] not in PHASE] # only amplitude/spectrum
    curated = keep # amplitude + phase (without ballastu)

    print("GB leave-one-protein-out (marginalny wklad FAZY)\n")
    a_amp = _loo(D, tr, amp)
    a_cur = _loo(D, tr, curated)
    print(f" {'protein':22s} amplitude +phase delta")
    for these in tr:
        d = a_cur[these] - a_amp[these]
        print(f" {these:22s} {a_amp[these]:.3f} {a_cur[these]:.3f} {d:+.3f}")
    has, mc = np.mean(list(a_amp.values())), np.mean(list(a_cur.values()))
    print(f" {'SREDNIA':22s} {has:.3f} {mc:.3f} {mc-has:+.3f}")
    print(f"\n ({len(amp)} channels amplitude/spectrum, +{len(curated)-len(amp)} phase; "
          f"removeieto ballast: {', '.join(sorted(BALAST))})")

    # final GB (curated) -> c-Myc
    Xall = np.vstack([D[n][0][:, curated] for n in tr]); yall = np.concatenate([D[n][1] for n in tr]).astype(float)
    for t in test:
        p, _ = mlaw.gb_fit_predict(Xall, yall, D[t["name"]][0][:, curated])
        order = np.argsort(-p); keys, resn = D[t["name"]][2], D[t["name"]][3]
        print(f"\n GB(curated) -> {t['name']} top5: " +
              ", ".join(f"{keys[i][0]}{keys[i][1]}{resn[i]}" for i in order[:5]))


if __name__ == "__main__":
    main()
