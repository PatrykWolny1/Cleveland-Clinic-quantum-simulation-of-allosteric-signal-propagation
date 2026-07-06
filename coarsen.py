#!/usr/bin/env python3
"""
Coarse-graining: proof of topological-signal retention + speedup.

For each target computes the detector AUC at FULL resolution and after
reduction to M super-nodes (at various reduction levels). Reports:
  AUC_full, AUC_coarse, retention = AUC_coarse/AUC_full,
  spectral_err (relative error of the low L modes), eig speedup (N^3/M^3).

  python coarsen.py
"""
import os, time, yaml, numpy as np
from aq import data, graph, propagate, rank, score, coarse

ROOT = os.path.dirname(os.path.abspath(__file__))


def detector_auc(A, gd, labels, ham="adjacency", metric="time_avg",
                 d_min=3, d_max=8, gpu=False):
    H = graph.Hamiltonian(A, ham)
    L = graph.Hamiltonian(A, "laplacian")
    w, V, xp, _ = propagate.eig(H, gpu)
    M = propagate.metric_from_cache({"H": (w, V), "L": propagate.eig(L, gpu)[:2]},
                                    {"metric": metric, "analytic": True}, xp)
    o, sc = rank.rank_fluctuation(M, gd, d_min, d_max)
    return score.evaluate(sc, o, labels, (5,)).get("roc_auc"), sc


def run_target(tgt, base, ratios=(2, 3, 5)):
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    apo = data.load_apo(os.path.join(clean, tgt["apo"]), tgt["name"], None, "largest")
    A = graph.build_graph(apo.coords, "cutoff", cutoff=8.0)
    gd = graph.graph_distance(A)
    pocket, _ = data.extract_pocket(os.path.join(clean, tgt["holo"]), None, 5.0)
    labels = data.label_nodes(apo, pocket)
    N = apo.n

    t = time.time(); auc_full, _ = detector_auc(A, gd, labels); t_full = time.time() - t
    print(f"\n{tgt['name']} N={N} AUC_full={auc_full:.3f} (eig full {t_full*1000:.0f}ms)")

    for rr in ratios:
        M = max(8, N // rr)
        lab, Ac, sizes = coarse.spectral_coarsen(A, M)
        Mc = Ac.shape[0]
        gdc = graph.graph_distance(Ac)
        # super-node labels: a cluster is positive if it contains a pocket residue
        clab = np.zeros(Mc, int)
        for i, l in enumerate(lab):
            if labels[i]:
                clab[l] = 1
        t = time.time()
        auc_c, sc_c = detector_auc(Ac, gdc, clab, d_min=1, d_max=None)
        t_c = time.time() - t
        # project down to residues and AUC at the residue level
        sc_res = coarse.project_up(sc_c, lab)
        from aq.rank import _finalize
        o_res, _ = _finalize(sc_res, None)
        auc_res = score.evaluate(sc_res, o_res, labels, (5,)).get("roc_auc")
        serr = coarse.spectral_preservation(A, Ac, k=6)
        print(f" ->1/{rr}: M={Mc:4d} AUC_super={auc_c:.3f} AUC_residues={auc_res:.3f} "
              f"retention={auc_res/max(auc_full,1e-9):.2f} spectral_err={serr:.3f} "
              f"eig x{(N/Mc)**3:.0f} cheaper ({t_c*1000:.0f}ms)")


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    targets = [t for t in yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
               if t.get("holo")]
    print("Coarse-graining: signal retention + speedup")
    for t in targets:
        run_target(t, base)


if __name__ == "__main__":
    main()
