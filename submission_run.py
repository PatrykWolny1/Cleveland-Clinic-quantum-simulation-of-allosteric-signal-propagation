#!/usr/bin/env python3
"""
FINAL CONSOLIDATION: router (unsupervised) + transfer GB (learned) -> fusion,
significance (AUC+p+CI) for each, selection best per protein. c-Myc + matrices.
  python submission_run.py
Optimization: MP over proteins, analytically, GPU eig; GB light CPU (not QPU).
"""
import os, json, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, graph, propagate, rank, spaces, mlaw, stats, router, score
from aq.backend import to_numpy, get_backend

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
GPU = CFG.get("backend", {}).get("use_gpu", False)
BALAST = {"coherent", "communic", "interf_T40"}
USE_RESONANCE = False # resonance hurt KRAS (0.414) -> optional, by default OFF
LEAN_K = 20 # transfer GB: selekcja top-k features by AUC on training set (anty-overfitting)


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")
def _rank(v):
    v = np.where(np.isfinite(v), v, np.nanmin(v)); r = np.argsort(np.argsort(v)); return r/(len(r)-1)
def _raw(name, rd): p = os.path.join(rd, name.split("_")[0]+".pdb"); return p if os.path.exists(p) else None


def _worker(arg):
    """Returns features (curated), detector routera, etykiete, klucze."""
    t, clean, rawdir = arg
    xp, _ = get_backend(GPU)
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    # router: LST distal + gnm proximal (when cofactor)
    A = graph.build_graph(apo.coords, "cutoff", cutoff=8.0); gd = graph.graph_distance(A)
    Hf = graph.Hamiltonian(A, "logf", sigma=1.0); wf, Vf, _, _ = propagate.eig(Hf, GPU)
    M_lst = to_numpy(propagate.q_interference(wf, Vf, xp, 20.0))
    _, s_lst = rank.rank_fluctuation(M_lst, gd, d_min=3)
    cand = {"LST_logf_interf": s_lst}
    rp = _raw(t["name"], rawdir); active = []
    if rp: active, _ = data.detect_active_site(rp, apo)
    if len(active) > 0:
        L = graph.Hamiltonian(A, "laplacian"); wL, VL, _, _ = propagate.eig(L, GPU)
        _, s_gnm = rank.rank_active_site(to_numpy(propagate.gnm_cov(wL, VL, xp)), active, gd, normalize="zabs")
        cand["gnm_zabs"] = s_gnm
        if USE_RESONANCE: # resonance (QPU-native CTQW) - eksperymentalny; hurt KRAS
            res = to_numpy(propagate.resonance_overlap(to_numpy(wL), to_numpy(VL), xp, active,
                                                       times=(10.0, 20.0, 40.0), gpu=GPU))
            res[np.asarray(active, int)] = -np.inf
            cand["resonance_active"] = res
    det_name, det_score, regime = router.route(cand)
    # features GB (curated)
    D, names = spaces.all_features(apo.coords, GPU); D = spaces.rank_gauss(D)
    keep = [i for i, n in enumerate(names) if n not in BALAST]
    y = None
    if t.get("holo"):
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
    return t["name"], D[:, keep], y, apo.keys, apo.resnames, det_score, det_name, regime


def main():
    clean = os.path.join(ROOT, CFG["paths"]["pdb_clean"]); rawdir = os.path.join(ROOT, CFG["paths"]["pdb_raw"])
    out = os.path.join(ROOT, CFG["paths"]["outputs"]); os.makedirs(out, exist_ok=True)
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    extra = os.path.join(ROOT, "extra_targets.yaml")
    if os.path.exists(extra): T += yaml.safe_load(open(extra)).get("targets", [])
    train = [t for t in T if t.get("holo")]; test = [t for t in T if not t.get("holo")]
    with ProcessPoolExecutor(max_workers=CFG.get("backend", {}).get("n_workers", 3)) as ex:
        res = list(ex.map(_worker, [(t, clean, rawdir) for t in (train+test)]))
    R = {r[0]: r[1:] for r in res}; tr = [t["name"] for t in train]

    print("="*82); print("FINAL CONSOLIDATION: router + transfer GB -> fusion (significance)"); print("="*82)
    best = {}
    for these in tr:
        X, y, keys, resn, det, detn, regime = R[these]
        # transfer GB leave-one-out z LEAN feature selection (top-k by AUC in TRENINGU, leakage-free)
        tr_others = [n for n in tr if n != these]
        def _fauc(j):
            vs = []
            for n in tr_others:
                xj = R[n][0][:, j]; yj = R[n][1]
                vs.append(max(auc(xj, yj), auc(-xj, yj)))
            return np.mean(vs)
        F = R[these][0].shape[1]
        topk = np.argsort([-_fauc(j) for j in range(F)])[:LEAN_K]
        Xtr = np.vstack([R[n][0][:, topk] for n in tr_others]); ytr = np.concatenate([R[n][1] for n in tr_others]).astype(float)
        gb, _ = mlaw.gb_fit_predict(Xtr, ytr, X[:, topk])
        ens = _rank(det) + _rank(gb) # fusion
        cand = {"router("+detn+")": det, "transfer_GB": gb, "fusion": ens}
        print(f"\n{these} [{regime}]")
        best_auc, best_nm, best_sc = -1, None, None
        for nm, sc in cand.items():
            st = stats.significance(np.where(np.isfinite(sc), sc, np.nanmin(sc)), y, n_boot=1500)
            flag = ""
            if st["auc"] > best_auc: best_auc, best_nm, best_sc = st["auc"], nm, sc; 
            print(f" {nm:20s} AUC={st['auc']:.3f} CI[{st['ci_low']:.3f},{st['ci_high']:.3f}] p={st['p']:.2e}")
        print(f" -> best: {best_nm} (AUC={best_auc:.3f})")
        best[these] = (best_nm, best_sc, keys, resn)
        # save: top5 fusion + significance all
        st_all = {nm: stats.significance(np.where(np.isfinite(sc), sc, np.nanmin(sc)), y, n_boot=1500)
                  for nm, sc in cand.items()}
        order = np.argsort(-ens)
        hits = [{"rank": r+1, "chain": keys[i][0], "resseq": int(keys[i][1]), "resname": resn[i]}
                for r, i in enumerate(order[:5])]
        json.dump({"name": these, "regime": regime, "best": best_nm, "significance": st_all,
                   "fusion_top5": hits}, open(os.path.join(out, f"{these}_submission.json"), "w"), indent=2)

    # c-Myc: fusion router + GB(3 protein)
    Xall = np.vstack([R[n][0] for n in tr]); yall = np.concatenate([R[n][1] for n in tr]).astype(float)
    for t in test:
        X, _, keys, resn, det, detn, regime = R[t["name"]]
        gb, _ = mlaw.gb_fit_predict(Xall, yall, X)
        ens = _rank(det) + _rank(gb); order = np.argsort(-ens)
        top = ", ".join(f"{keys[i][0]}{keys[i][1]}{resn[i]}" for i in order[:5])
        print(f"\n{t['name']} [{regime}] fusion top5: {top}")
    print(f"\n-> artifacts in {out}")


if __name__ == "__main__":
    main()
