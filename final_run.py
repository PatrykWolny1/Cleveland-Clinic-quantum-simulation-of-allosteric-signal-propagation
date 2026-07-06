#!/usr/bin/env python3
"""
CONSOLIDATED submission. For each protein:
  - the router selects the predictor (proximal gnm_zabs / distal LST),
  - significance: AUC + p-value + 95% CI (bootstrap),
  - connectivity matrix (.npy) + top-5 hit-list.
c-Myc: routed top-5 (without scoring). Also pulls in the pairs from extra_targets.yaml (ASD).

  python final_run.py

Optimization: analytic metric, multiprocessing over proteins, GPU under use_gpu; sampling
(sample.py) for N beyond the budget.
"""
import os, json, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, graph, propagate, rank, score, stats, router
from aq.backend import to_numpy, get_backend

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
GPU = CFG.get("backend", {}).get("use_gpu", False)


def _raw_path(name, raw_dir):
    p = os.path.join(raw_dir, name.split("_")[0] + ".pdb")
    return p if os.path.exists(p) else None


def _detectors(coords, raw_path, apo):
    """Returns (candidates{name:score}, matrices{name:M}, gd)."""
    xp, _ = get_backend(GPU)
    A = graph.build_graph(coords, "cutoff", cutoff=8.0)
    gd = graph.graph_distance(A)
    # distal: LST_logf_interf
    Hf = graph.Hamiltonian(A, "logf", sigma=1.0)
    wf, Vf, _, _ = propagate.eig(Hf, GPU)
    M_lst = to_numpy(propagate.q_interference(wf, Vf, xp, 20.0))
    _, s_lst = rank.rank_fluctuation(M_lst, gd, d_min=3)
    cand = {"LST_logf_interf": s_lst}; mats = {"LST_logf_interf": M_lst}
    # proximal: gnm_zabs (when a cofactor is present)
    active = []
    if raw_path is not None:
        active, cof = data.detect_active_site(raw_path, apo)
    if len(active) > 0:
        L = graph.Hamiltonian(A, "laplacian")
        wL, VL, _, _ = propagate.eig(L, GPU)
        M_gnm = to_numpy(propagate.gnm_cov(wL, VL, xp))
        _, s_gnm = rank.rank_active_site(M_gnm, active, gd, normalize="zabs")
        cand["gnm_zabs"] = s_gnm; mats["gnm_zabs"] = M_gnm
    return cand, mats, gd, len(active)


def _worker(arg):
    t, clean, raw_dir = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    cand, mats, gd, n_active = _detectors(apo.coords, _raw_path(t["name"], raw_dir), apo)
    name, sc, regime = router.route(cand)
    order = np.argsort(-np.where(np.isfinite(sc), sc, -np.inf))
    hits = rank.hit_list(order, apo.keys, apo.resnames, sc, top=5)
    out = {"name": t["name"], "detector": name, "regime": regime,
           "n_active_site": n_active, "hits": hits}
    if t.get("holo"):
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
        fin = np.where(np.isfinite(sc), sc, np.nanmin(sc))
        out["stats"] = stats.significance(fin, y, n_boot=2000)
    return out, mats[name], name


def main():
    clean = os.path.join(ROOT, CFG["paths"]["pdb_clean"])
    raw_dir = os.path.join(ROOT, CFG["paths"]["pdb_raw"])
    out_dir = os.path.join(ROOT, CFG["paths"]["outputs"]); os.makedirs(out_dir, exist_ok=True)
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    extra_path = os.path.join(ROOT, "extra_targets.yaml") # ASD (optional)
    if os.path.exists(extra_path):
        T += yaml.safe_load(open(extra_path)).get("targets", [])

    args = [(t, clean, raw_dir) for t in T]
    results = []
    with ProcessPoolExecutor(max_workers=CFG.get("backend", {}).get("n_workers", 3)) as ex:
        for out, M, name in ex.map(_worker, args):
            np.save(os.path.join(out_dir, f"{out['name']}_connectivity_{name}.npy"), M)
            with open(os.path.join(out_dir, f"{out['name']}_hitlist_final.json"), "w") as f:
                json.dump(out, f, indent=2)
            results.append(out)

    print("=" * 78)
    print("CONSOLIDATED submission (router + significance)")
    print("=" * 78)
    for r in results:
        print(f"\n{r['name']} [{r['regime']}, detector={r['detector']}, "
              f"active_site={r['n_active_site']}]")
        if "stats" in r:
            st = r["stats"]
            print(f" AUC={st['auc']:.3f} 95% CI [{st['ci_low']:.3f}, {st['ci_high']:.3f}] "
                  f"p={st['p']:.2e}")
        top = ", ".join(f"{h['chain']}{h['resseq']}{h['resname']}" for h in r["hits"])
        print(f" top5: {top}")
    print(f"\n-> connectivity matrices (.npy) + hit-lists (.json) in {out_dir}")


if __name__ == "__main__":
    main()
